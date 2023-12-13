"""Higher-level functions of automated time series modeling."""
import random
import copy
import json
import gc
import sys
import time
import traceback as tb
import numpy as np
import pandas as pd

from autots.tools.shaping import (
    long_to_wide,
    df_cleanup,
    subset_series,
    simple_train_test_split,
    NumericTransformer,
    clean_weights,
    infer_frequency,
    freq_to_timedelta,
)
from autots.tools.transform import GeneralTransformer, RandomTransform
from autots.evaluator.auto_model import (
    TemplateEvalObject,
    NewGeneticTemplate,
    RandomTemplate,
    TemplateWizard,
    unpack_ensemble_models,
    generate_score,
    generate_score_per_series,
    model_forecast,
    validation_aggregation,
    back_forecast,
    remove_leading_zeros,
    horizontal_template_to_model_list,
    create_model_id,
    ModelPrediction,
)
from autots.models.ensemble import (
    EnsembleTemplateGenerator,
    HorizontalTemplateGenerator,
    generate_mosaic_template,
    generate_crosshair_score,
    process_mosaic_arrays,
    parse_forecast_length,
    n_limited_horz,
    is_horizontal, is_mosaic,
)
from autots.models.model_list import model_lists, no_shared, update_fit
from autots.tools.cpu_count import set_n_jobs
from autots.evaluator.validation import (
    validate_num_validations,
    generate_validation_indices,
)


class AutoTS(object):
    """Automate time series modeling using a genetic algorithm.

    Args:
        forecast_length (int): number of periods over which to evaluate forecast. Can be overriden later in .predict().
            when you don't have much historical data, using a small forecast length for .fit and the full desired forecast lenght for .predict is usually the best possible approach given limitations.
        frequency (str): 'infer' or a specific pandas datetime offset. Can be used to force rollup of data (ie daily input, but frequency 'M' will rollup to monthly).
        prediction_interval (float): 0-1, uncertainty range for upper and lower forecasts. Adjust range, but rarely matches actual containment.
        max_generations (int): number of genetic algorithms generations to run.
            More runs = longer runtime, generally better accuracy.
            It's called `max` because someday there will be an auto early stopping option, but for now this is just the exact number of generations to run.
        no_negatives (bool): if True, all negative predictions are rounded up to 0.
        constraint (float): when not None, use this float value * data st dev above max or below min for constraining forecast values.
            now also instead accepts a dictionary containing the following key/values:
                constraint_method (str): one of
                    stdev_min - threshold is min and max of historic data +/- constraint * st dev of data
                    stdev - threshold is the mean of historic data +/- constraint * st dev of data
                    absolute - input is array of length series containing the threshold's final value for each
                    quantile - constraint is the quantile of historic data to use as threshold
                constraint_regularization (float): 0 to 1
                    where 0 means no constraint, 1 is hard threshold cutoff, and in between is penalty term
                upper_constraint (float): or array, depending on method, None if unused
                lower_constraint (float): or array, depending on method, None if unused
                bounds (bool): if True, apply to upper/lower forecast, otherwise False applies only to forecast
        ensemble (str): None or list or comma-separated string containing:
            'auto', 'simple', 'distance', 'horizontal', 'horizontal-min', 'horizontal-max', "mosaic", "subsample"
        initial_template (str): 'Random' - randomly generates starting template, 'General' uses template included in package, 'General+Random' - both of previous. Also can be overriden with self.import_template()
        random_seed (int): random seed allows (slightly) more consistent results.
        holiday_country (str): passed through to Holidays package for some models.
        subset (int): maximum number of series to evaluate at once. Useful to speed evaluation when many series are input.
            takes a new subset of columns on each validation, unless mosaic ensembling, in which case columns are the same in each validation
        aggfunc (str): if data is to be rolled up to a higher frequency (daily -> monthly) or duplicate timestamps are included. Default 'first' removes duplicates, for rollup try 'mean' or np.sum.
            Beware numeric aggregations like 'mean' will not work with non-numeric inputs.
            Numeric aggregations like 'sum' will also change nan values to 0
        na_tolerance (float): 0 to 1. Series are dropped if they have more than this percent NaN. 0.95 here would allow series containing up to 95% NaN values.
        metric_weighting (dict): weights to assign to metrics, effecting how the ranking score is generated.
        drop_most_recent (int): option to drop n most recent data points. Useful, say, for monthly sales data where the current (unfinished) month is included.
            occurs after any aggregration is applied, so will be whatever is specified by frequency, will drop n frequencies
        drop_data_older_than_periods (int): take only the n most recent timestamps
        model_list (list): str alias or list of names of model objects to use
            now can be a dictionary of {"model": prob} but only affects starting random templates. Genetic algorithim takes from there.
        transformer_list (list): list of transformers to use, or dict of transformer:probability. Note this does not apply to initial templates.
            can accept string aliases: "all", "fast", "superfast", 'scalable' (scalable is a subset of fast that should have fewer memory issues at scale)
        transformer_max_depth (int): maximum number of sequential transformers to generate for new Random Transformers. Fewer will be faster.
        models_mode (str): option to adjust parameter options for newly generated models. Only sporadically utilized. Currently includes:
            'default'/'random', 'deep' (searches more params, likely slower), and 'regressor' (forces 'User' regressor mode in regressor capable models),
            'gradient_boosting', 'neuralnets' (~Regression class models only)
        num_validations (int): number of cross validations to perform. 0 for just train/test on best split.
            Possible confusion: num_validations is the number of validations to perform *after* the first eval segment, so totally eval/validations will be this + 1.
            Also "auto" and "max" aliases available. Max maxes out at 50.
        models_to_validate (int): top n models to pass through to cross validation. Or float in 0 to 1 as % of tried.
            0.99 is forced to 100% validation. 1 evaluates just 1 model.
            If horizontal or mosaic ensemble, then additional min per_series models above the number here are added to validation.
        max_per_model_class (int): of the models_to_validate what is the maximum to pass from any one model class/family.
        validation_method (str): 'even', 'backwards', or 'seasonal n' where n is an integer of seasonal
            'backwards' is better for recency and for shorter training sets
            'even' splits the data into equally-sized slices best for more consistent data, a poetic but less effective strategy than others here
            'seasonal' most similar indexes
            'seasonal n' for example 'seasonal 364' would test all data on each previous year of the forecast_length that would immediately follow the training data.
            'similarity' automatically finds the data sections most similar to the most recent data that will be used for prediction
            'custom' - if used, .fit() needs validation_indexes passed - a list of pd.DatetimeIndex's, tail of each is used as test
        min_allowed_train_percent (float): percent of forecast length to allow as min training, else raises error.
            0.5 with a forecast length of 10 would mean 5 training points are mandated, for a total of 15 points.
            Useful in (unrecommended) cases where forecast_length > training length.
        remove_leading_zeroes (bool): replace leading zeroes with NaN. Useful in data where initial zeroes mean data collection hasn't started yet.
        prefill_na (str): value to input to fill all NaNs with. Leaving as None and allowing model interpolation is recommended.
            None, 0, 'mean', or 'median'. 0 may be useful in for examples sales cases where all NaN can be assumed equal to zero.
        introduce_na (bool): whether to force last values in one training validation to be NaN. Helps make more robust models.
            defaults to None, which introduces NaN in last rows of validations if any NaN in tail of training data. Will not introduce NaN to all series if subset is used.
            if True, will also randomly change 20% of all rows to NaN in the validations
        preclean (dict): if not None, a dictionary of Transformer params to be applied to input data
            {"fillna": "median", "transformations": {}, "transformation_params": {}}
            This will change data used in model inputs for fit and predict, and for accuracy evaluation in cross validation!
        model_interrupt (bool): if False, KeyboardInterrupts quit entire program.
            if True, KeyboardInterrupts attempt to only quit current model.
            if True, recommend use in conjunction with `verbose` > 0 and `result_file` in the event of accidental complete termination.
            if "end_generation", as True and also ends entire generation of run. Note skipped models will not be tried again.
        generation_timeout (int): if not None, this is the number of minutes from start at which the generational search ends, then proceeding to validation
            This is only checked after the end of each generation, so only offers an 'approximate' timeout for searching
        current_model_file (str): file path to write to disk of current model params (for debugging if computer crashes). .json is appended
        force_gc (bool): if True, run gc.collect() after each model run. Probably won't make much difference.
        verbose (int): setting to 0 or lower should reduce most output. Higher numbers give more output.
        n_jobs (int): Number of cores available to pass to parallel processing. A joblib context manager can be used instead (pass None in this case). Also 'auto'.

    Attributes:
        best_model (pd.DataFrame): DataFrame containing template for the best ranked model
        best_model_name (str): model name
        best_model_params (dict): model params
        best_model_transformation_params (dict): transformation parameters
        best_model_ensemble (int): Ensemble type int id
        regression_check (bool): If True, the best_model uses an input 'User' future_regressor
        df_wide_numeric (pd.DataFrame): dataframe containing shaped final data, will include preclean
        initial_results.model_results (object): contains a collection of result metrics
        score_per_series (pd.DataFrame): generated score of metrics given per input series, if horizontal ensembles

    Methods:
        fit, predict
        export_template, import_template, import_results
        results, failure_rate
        horizontal_to_df, mosaic_to_df
        plot_horizontal, plot_horizontal_transformers, plot_generation_loss, plot_backforecast
    """

    def __init__(
        self,
        forecast_length: int = 14,
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        max_generations: int = 10,
        no_negatives: bool = False,
        constraint: float = None,
        ensemble: str = None,  # 'auto',
        initial_template: str = 'General+Random',
        random_seed: int = 2022,
        holiday_country: str = 'US',
        subset: int = None,
        aggfunc: str = 'first',
        na_tolerance: float = 1,
        metric_weighting: dict = {
            'smape_weighting': 5,
            'mae_weighting': 2,
            'rmse_weighting': 2,
            'made_weighting': 0.5,
            'mage_weighting': 0,
            'mle_weighting': 0,
            'imle_weighting': 0,
            'spl_weighting': 3,
            'containment_weighting': 0,
            'contour_weighting': 1,
            'runtime_weighting': 0.05,
            'oda_weighting': 0.001,
        },
        drop_most_recent: int = 0,
        drop_data_older_than_periods: int = None,
        model_list: str = 'default',
        transformer_list: dict = "auto",
        transformer_max_depth: int = 6,
        models_mode: str = "random",
        num_validations: int = "auto",
        models_to_validate: float = 0.15,
        max_per_model_class: int = None,
        validation_method: str = 'backwards',
        min_allowed_train_percent: float = 0.5,
        remove_leading_zeroes: bool = False,
        prefill_na: str = None,
        introduce_na: bool = None,
        preclean: dict = None,
        model_interrupt: bool = True,
        generation_timeout: int = None,
        current_model_file: str = None,
        force_gc: bool = False,
        verbose: int = 1,
        n_jobs: int = -2,
    ):
        assert forecast_length > 0, "forecast_length must be greater than 0"
        # assert transformer_max_depth > 0, "transformer_max_depth must be greater than 0"
        self.forecast_length = int(abs(forecast_length))
        self.frequency = frequency
        self.aggfunc = aggfunc
        self.prediction_interval = prediction_interval
        self.no_negatives = no_negatives
        self.constraint = constraint
        self.random_seed = random_seed
        self.holiday_country = holiday_country
        self.subset = subset
        self.na_tolerance = na_tolerance
        self.metric_weighting = metric_weighting
        self.drop_most_recent = drop_most_recent
        self.drop_data_older_than_periods = drop_data_older_than_periods
        self.model_list = model_list
        self.transformer_list = transformer_list
        self.transformer_max_depth = transformer_max_depth
        self.num_validations = num_validations
        self.models_to_validate = models_to_validate
        self.max_per_model_class = max_per_model_class
        self.validation_method = str(validation_method).lower()
        self.min_allowed_train_percent = min_allowed_train_percent
        self.max_generations = max_generations
        self.generation_timeout = generation_timeout
        self.remove_leading_zeroes = remove_leading_zeroes
        self.prefill_na = prefill_na
        self.introduce_na = introduce_na
        self.preclean = preclean
        self.model_interrupt = model_interrupt
        self.verbose = int(verbose)
        self.n_jobs = n_jobs
        self.models_mode = models_mode
        self.current_model_file = current_model_file
        self.force_gc = force_gc
        self.validate_import = None
        random.seed(self.random_seed)
        if self.max_generations is None and self.generation_timeout is not None:
            self.max_generations = 99999
        if self.generation_timeout is None:
            self.generation_timeout = 9e6  # 20 years
        if holiday_country == "RU":
            self.holiday_country = "UA"
        elif holiday_country == 'CN':
            self.holiday_country = 'TW'
        if isinstance(ensemble, str):
            ensemble = str(ensemble).lower()
        if ensemble == 'all':
            ensemble = [
                'simple',
                "distance",
                "horizontal",
                "horizontal-max",
                "mosaic",
                'mosaic-window',
                "subsample",
                'mlensemble',
            ]
        elif ensemble == 'auto':
            if model_list in ['superfast']:
                ensemble = ['horizontal-max']
            elif any([x for x in model_list if x in ['PytorchForecasting', 'GluonTS']]):
                ensemble = None
            else:
                ensemble = ['simple', "distance", "horizontal-max"]
        if isinstance(ensemble, str):
            self.ensemble = ensemble.split(",")
        elif isinstance(ensemble, list):
            self.ensemble = ensemble
        elif ensemble is None or not ensemble:
            self.ensemble = []
        else:
            raise ValueError(
                f"ensemble arg: {ensemble} not a recognized string or list"
            )

        # check metric weights are valid
        metric_weighting_values = self.metric_weighting.values()
        if sum(metric_weighting_values) < -10:
            raise ValueError(
                f"Metric weightings should generally be >= 0. Current weightings: {self.metric_weighting}"
            )
        if (
            'seasonal' in self.validation_method
            and self.validation_method != "seasonal"
        ):
            val_list = [x for x in str(self.validation_method) if x.isdigit()]
            self.seasonal_val_periods = int(''.join(val_list))

        self.n_jobs = set_n_jobs(self.n_jobs, verbose=self.verbose)

        # convert shortcuts of model lists to actual lists of models
        if model_list in list(model_lists.keys()):
            self.model_list = model_lists[model_list]
        # prepare for a common Typo
        elif 'Prophet' in model_list:
            self.model_list = ["FBProphet" if x == "Prophet" else x for x in model_list]

        # generate template to begin with
        initial_template = str(initial_template).lower()
        if initial_template == 'random':
            self.initial_template = RandomTemplate(
                len(self.model_list) * 12,
                model_list=self.model_list,
                transformer_list=self.transformer_list,
                transformer_max_depth=self.transformer_max_depth,
                models_mode=self.models_mode,
            )
        elif initial_template == 'general':
            from autots.templates.general import general_template

            self.initial_template = general_template
        elif initial_template == 'general+random':
            from autots.templates.general import general_template

            random_template = RandomTemplate(
                len(self.model_list) * 5,
                model_list=self.model_list,
                transformer_list=self.transformer_list,
                transformer_max_depth=self.transformer_max_depth,
                models_mode=self.models_mode,
            )
            self.initial_template = (
                pd.concat([general_template, random_template], axis=0)
                .drop_duplicates()
                .reset_index(drop=True)
            )
        elif isinstance(initial_template, pd.DataFrame):
            self.initial_template = initial_template
        else:
            print("Input initial_template unrecognized. Using Random.")
            self.initial_template = RandomTemplate(
                50,
                model_list=self.model_list,
                transformer_list=self.transformer_list,
                transformer_max_depth=self.transformer_max_depth,
                models_mode=self.models_mode,
            )

        # remove models not in given model list
        self.initial_template = self.initial_template[
            self.initial_template['Model'].isin(self.model_list)
        ]
        if self.initial_template.shape[0] == 0:
            raise ValueError(
                "No models in template! Adjust initial_template or model_list"
            )
        # remove transformers not in transformer_list and max_depth
        # yes it is awkward, but I cannot think of a better way at this time
        if self.transformer_max_depth < 6 or self.transformer_list not in [
            "all",
            "fast",
            "superfast",
        ]:
            from autots.tools.transform import transformer_list_to_dict

            transformer_lst, prb = transformer_list_to_dict(self.transformer_list)
            for index, row in self.initial_template.iterrows():
                full_params = json.loads(row['TransformationParameters'])
                try:
                    transformations = full_params['transformations']
                    transformation_params = full_params['transformation_params']
                except KeyError:
                    raise ValueError(
                        "initial_template is missing transformation parameters for one or more models"
                    )
                # remove those not in transformer_list
                bad_keys = [
                    i
                    for i, x in json.loads(row['TransformationParameters'])[
                        'transformations'
                    ].items()
                    if x not in transformer_lst
                ]
                [transformations.pop(key) for key in bad_keys]
                [transformation_params.pop(key) for key in bad_keys]

                # shorten any remaining if beyond length
                transformations = dict(
                    list(transformations.items())[: self.transformer_max_depth]
                )
                transformation_params = dict(
                    list(transformation_params.items())[: self.transformer_max_depth]
                )

                full_params['transformations'] = transformations
                full_params['transformation_params'] = transformation_params
                self.initial_template.loc[
                    index, 'TransformationParameters'
                ] = json.dumps(full_params)

        self.regressor_used = False
        # do not add 'ID' to the below unless you want to refactor things.
        self.template_cols = [
            'Model',
            'ModelParameters',
            'TransformationParameters',
            'Ensemble',
        ]
        self.template_cols_id = (
            self.template_cols
            if "ID" in self.template_cols
            else ['ID'] + self.template_cols
        )
        self.grouping_ids = None
        self.initial_results = TemplateEvalObject()
        self.best_model = pd.DataFrame()
        self.best_model_id = ""
        self.best_model_name = ""
        self.best_model_params = {}
        self.best_model_transformation_params = ""
        self.best_model_ensemble = -1
        self.traceback = True if verbose > 1 else False
        self.future_regressor_train = None
        self.validation_train_indexes = []
        self.validation_test_indexes = []
        self.preclean_transformer = None
        self.score_per_series = None
        self.best_model_non_horizontal = None
        self.validation_forecasts_template = None
        self.validation_forecasts = {}
        self.validation_results = None
        # this is temporary until proper validation param passing is sorted out
        stride_size = round(self.forecast_length / 2)
        stride_size = stride_size if stride_size > 0 else 1
        self.similarity_validation_params = {
            "stride_size": stride_size,
            "distance_metric": "canberra",
            "include_differenced": True,
            "window_size": 30,
        }
        self.seasonal_validation_params = {
            'window_size': 10,
            'distance_metric': 'mae',
            'datepart_method': 'common_fourier_rw',
        }
        self.model_count = 0
        self.model = None  # intended for fit_data update models only

        if verbose > 2:
            msg = '"Hello. Would you like to destroy some evil today?" - Sanderson'
            # unicode may not be supported on all platforms
            try:
                print("\N{dagger} " + msg)
            except Exception:
                print(msg)

    @staticmethod
    def get_new_params(method='random'):
        """Randomly generate new parameters for the class."""
        if method != 'full':
            ensemble_choice = random.choices(
                [
                    None,
                    ['simple'],
                    ['simple', 'horizontal-max'],
                    [
                        'simple',
                        "distance",
                        "horizontal",
                        "horizontal-max",
                    ],
                ],
                [0.3, 0.1, 0.2, 0.2],
            )[0]
            max_generations = random.choices([5, 15, 25, 50], [0.2, 0.5, 0.1, 0.4])[0]
        else:
            max_generations = random.choices([15, 25, 50, 200], [0.2, 0.5, 0.2, 0.1])[0]
            ensemble_choice = random.choices(
                [
                    None,
                    ['simple'],
                    ['simple', 'horizontal-max'],
                    [
                        'simple',
                        "distance",
                        "horizontal",
                        "horizontal-max",
                        "mosaic",
                        'mosaic-window',
                        'mosaic-crosshair',
                        "subsample",
                        "mlensemble",
                    ],
                ],
                [0.3, 0.1, 0.2, 0.2],
            )[0]
        if method in ["full", "fast", "superfast"]:
            metric_weighting = {
                'smape_weighting': random.choices([0, 1, 5, 10], [0.3, 0.2, 0.3, 0.1])[
                    0
                ],
                'mae_weighting': random.choices([0, 1, 3, 5], [0.1, 0.3, 0.3, 0.3])[0],
                'rmse_weighting': random.choices([0, 1, 3, 5], [0.1, 0.3, 0.3, 0.3])[0],
                'made_weighting': random.choices([0, 1, 3, 5], [0.7, 0.3, 0.1, 0.05])[
                    0
                ],
                'mage_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'mle_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'imle_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'spl_weighting': random.choices([0, 1, 3, 5], [0.1, 0.3, 0.3, 0.3])[0],
                'oda_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'mqae_weighting': random.choices([0, 1, 3, 5], [0.4, 0.2, 0.1, 0.0])[0],
                'dwae_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'maxe_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'containment_weighting': random.choices(
                    [0, 1, 3, 5], [0.9, 0.1, 0.05, 0.0]
                )[0],
                'contour_weighting': random.choices(
                    [0, 1, 3, 5], [0.7, 0.2, 0.05, 0.05]
                )[0],
                'runtime_weighting': random.choices(
                    [0, 0.05, 0.3, 1], [0.1, 0.6, 0.2, 0.1]
                )[0],
                'uwmse_weighting': random.choices(
                    [0, 0.05, 0.3, 1, 5], [0.1, 0.6, 0.2, 0.1, 0.1]
                )[0],
                'smoothness_weighting': random.choices(
                    [0, 0.05, 3, 1, -0.5, -3], [0.4, 0.1, 0.1, 0.1, 0.2, 0.1]
                )[0],
                'ewmae_weighting': random.choices(
                    [0, 0.05, 0.3, 1, 5], [0.1, 0.6, 0.2, 0.1, 0.1]
                )[0],
                'mate_weighting': random.choices(
                    [0, 0.05, 0.3, 1, 5], [0.1, 0.6, 0.2, 0.1, 0.1]
                )[0],
                'wasserstein_weighting': random.choices(
                    [0, 0.05, 0.3, 1, 5], [0.1, 0.6, 0.2, 0.1, 0.1]
                )[0],
                'dwd_weighting': random.choices(
                    [0, 0.05, 0.3, 1, 5], [0.1, 0.6, 0.2, 0.1, 0.1]
                )[0],
            }
            validation_method = random.choices(
                ['backwards', 'even', 'similarity', 'seasonal 364', 'seasonal'],
                [0.4, 0.1, 0.3, 0.3, 0.2],
            )[0]
        else:
            metric_weighting = {
                'smape_weighting': random.choices([0, 1, 5, 10], [0.3, 0.2, 0.3, 0.1])[
                    0
                ],
                'mae_weighting': random.choices([0, 1, 3, 5], [0.1, 0.3, 0.3, 0.3])[0],
                'rmse_weighting': random.choices([0, 1, 3, 5], [0.1, 0.3, 0.3, 0.3])[0],
                'made_weighting': random.choices([0, 1, 3, 5], [0.7, 0.3, 0.1, 0.05])[
                    0
                ],
                'mage_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'mle_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'imle_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'spl_weighting': random.choices([0, 1, 3, 5], [0.1, 0.3, 0.3, 0.3])[0],
                'oda_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'mqae_weighting': random.choices([0, 1, 3, 5], [0.4, 0.2, 0.1, 0.0])[0],
                'maxe_weighting': random.choices([0, 1, 3, 5], [0.8, 0.1, 0.1, 0.0])[0],
                'containment_weighting': random.choices(
                    [0, 1, 3, 5], [0.9, 0.1, 0.05, 0.0]
                )[0],
                'contour_weighting': random.choices(
                    [0, 1, 3, 5], [0.7, 0.2, 0.05, 0.05]
                )[0],
                'runtime_weighting': random.choices(
                    [0, 0.05, 0.3, 1], [0.1, 0.6, 0.2, 0.1]
                )[0],
            }
            validation_method = random.choices(
                ['backwards', 'even', 'similarity', 'seasonal 364'],
                [0.4, 0.1, 0.3, 0.3],
            )[0]
        preclean_choice = random.choices(
            [
                None,
                {
                    "fillna": "ffill",
                    "transformations": {0: "EWMAFilter"},
                    "transformation_params": {
                        0: {"span": 3},
                    },
                },
                {
                    "fillna": "mean",
                    "transformations": {0: "EWMAFilter"},
                    "transformation_params": {
                        0: {"span": 7},
                    },
                },
                {
                    "fillna": None,
                    "transformations": {0: "StandardScaler"},
                    "transformation_params": {0: {}},
                },
                {
                    "fillna": None,
                    "transformations": {0: "QuantileTransformer"},
                    "transformation_params": {0: {}},
                },
                {
                    "fillna": None,
                    "transformations": {0: "AnomalyRemoval"},
                    "transformation_params": {
                        0: {
                            "method": "IQR",
                            "transform_dict": {},
                            "method_params": {
                                "iqr_threshold": 2.0,
                                "iqr_quantiles": [0.4, 0.6],
                            },
                            "fillna": 'ffill',
                        }
                    },
                },
                {
                    'fillna': None,
                    'transformations': {
                        '0': 'ClipOutliers',
                        '1': 'RegressionFilter',
                        '2': 'ClipOutliers',
                    },
                    'transformation_params': {
                        '0': {
                            'method': 'remove',
                            'std_threshold': 2.5,
                            'fillna': None,
                        },  # "SeasonalityMotifImputerLinMix"
                        '1': {
                            "sigma": 2,
                            "rolling_window": 90,
                            "run_order": "season_first",
                            "regression_params": {
                                "regression_model": {
                                    "model": "ElasticNet",
                                    "model_params": {},
                                },
                                "datepart_method": ['common_fourier'],
                                "polynomial_degree": None,
                                "transform_dict": None,
                                "holiday_countries_used": False,
                            },
                            "holiday_params": None,
                        },
                        '2': {
                            'method': 'remove',
                            'std_threshold': 3.0,
                            'fillna': "SeasonalityMotifImputerLinMix",
                        },
                    },
                },
                {
                    'fillna': None,
                    'transformations': {
                        '0': 'ClipOutliers',
                        '1': "LevelShiftMagic",
                        '2': 'RegressionFilter',
                        '3': 'ClipOutliers',
                    },
                    'transformation_params': {
                        '0': {
                            'method': 'remove',
                            'std_threshold': 2.5,
                            'fillna': None,
                        },  # "SeasonalityMotifImputerLinMix"
                        '1': {
                            'window_size': 90,
                            'alpha': 2.5,
                            'grouping_forward_limit': 3,
                            'max_level_shifts': 5,
                            'alignment': 'average',
                        },
                        '2': {
                            "sigma": 2,
                            "rolling_window": 90,
                            "run_order": "season_first",
                            "regression_params": {
                                "regression_model": {
                                    "model": "ElasticNet",
                                    "model_params": {},
                                },
                                "datepart_method": ['common_fourier'],
                                "polynomial_degree": None,
                                "transform_dict": None,
                                "holiday_countries_used": False,
                            },
                            "holiday_params": None,
                        },
                        '3': {
                            'method': 'remove',
                            'std_threshold': 3.0,
                            'fillna': "SeasonalityMotifImputerLinMix",
                        },
                    },
                },
                {
                    "fillna": None,
                    "transformations": {"0": "LocalLinearTrend"},
                    "transformation_params": {
                        "0": {
                            'rolling_window': 30,
                            'n_tails': 0.1,
                            'n_future': 0.2,
                            'method': 'mean',
                            'macro_micro': True,
                        },
                    },
                },
                {
                    "fillna": None,
                    "transformations": {"0": "CenterSplit"},
                    "transformation_params": {
                        "0": {
                            'fillna': 'ffill',
                            'center': 'zero',
                        },
                    },
                },
                'random',
            ],
            [0.9, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.15, 0.05, 0.1],
        )[0]
        if preclean_choice == "random":
            preclean_choice = RandomTransform(
                transformer_list="fast", transformer_max_depth=2
            )
        if method == 'full':
            model_list = random.choices(
                [
                    'fast',
                    'superfast',
                    'default',
                    'fast_parallel_no_arima',
                    'all',
                    'motifs',
                    'no_shared_fast',
                    'multivariate',
                    'univariate',
                    'all_result_path',
                    'regressions',
                    'best',
                    'regressor',
                    'probabilistic',
                    'no_shared',
                ],
                [
                    0.2,
                    0.4,
                    0.1,
                    0.2,
                    0.01,
                    0.1,
                    0.1,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                ],
            )[0]
        elif method == 'fast':
            model_list = random.choices(
                [
                    'fast',
                    'superfast',
                    'motifs',
                    'no_shared_fast',
                    'fast_parallel_no_arima',
                ],
                [
                    0.2,
                    0.3,
                    0.2,
                    0.2,
                    0.05,
                ],
            )[0]
        elif method == "superfast":
            model_list = 'superfast'
        else:
            model_list = random.choices(
                [
                    'fast',
                    'superfast',
                    'default',
                    'fast_parallel',
                    'motifs',
                    'no_shared_fast',
                ],
                [0.2, 0.3, 0.2, 0.2, 0.05, 0.1],
            )[0]

        return {
            'max_generations': max_generations,
            'model_list': model_list,
            'transformer_list': random.choices(
                ['all', 'fast', 'superfast'],
                [0.2, 0.5, 0.3],
            )[0],
            'transformer_max_depth': random.choices(
                [1, 2, 4, 6, 8, 10],
                [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
            )[0],
            'num_validations': random.choices(
                [0, 1, 2, 3, 4, 6], [0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
            )[0],
            'validation_method': validation_method,
            'models_to_validate': random.choices(
                [0.15, 0.10, 0.25, 0.35, 0.45], [0.3, 0.1, 0.3, 0.3, 0.1]
            )[0],
            'ensemble': ensemble_choice,
            'initial_template': random.choices(
                ['random', 'general+random'], [0.8, 0.2]
            )[0],
            'subset': random.choices([None, 10, 100], [0.9, 0.05, 0.05])[0],
            'models_mode': random.choices(['random', 'regressor'], [0.95, 0.05])[0],
            # 'drop_most_recent': random.choices([0, 1, 2], [0.8, 0.1, 0.1])[0],
            'introduce_na': random.choice([None, True, False]),
            'prefill_na': None,
            'remove_leading_zeroes': False,
            'constraint': random.choices(
                [
                    None,
                    {
                        "constraint_method": "stdev_min",
                        "constraint_regularization": 0.7,
                        "upper_constraint": 1,
                        "lower_constraint": 1,
                        "bounds": True,
                    },
                    {
                        "constraint_method": "stdev",
                        "constraint_regularization": 1,
                        "upper_constraint": 2,
                        "lower_constraint": 2,
                        "bounds": False,
                    },
                    {
                        "constraint_method": "quantile",
                        "constraint_regularization": 0.9,
                        "upper_constraint": 0.99,
                        "lower_constraint": 0.01,
                        "bounds": True,
                    },
                    {
                        "constraint_method": "quantile",
                        "constraint_regularization": 0.4,
                        "upper_constraint": 0.9,
                        "lower_constraint": 0.1,
                        "bounds": False,
                    },
                ],
                [0.9, 0.1, 0.1, 0.1, 0.1],
            )[0],
            'preclean': preclean_choice,
            'metric_weighting': metric_weighting,
        }

    def __repr__(self):
        """Print."""
        if self.best_model.empty:
            return "Uninitiated AutoTS object"
        else:
            try:
                base_res = self.initial_results.model_results[
                    self.initial_results.model_results['ID'] == self.best_model_id
                ]
                res = ", ".join(base_res['smape'].astype(str).tolist())
                res2 = ", ".join(base_res['mae'].astype(str).tolist())
                res3 = ", ".join(base_res['spl'].astype(str).tolist())
                len_list = list(range(base_res.shape[0]))
                res_len = ", ".join([str(x) for x in len_list])
                return f"Initiated AutoTS object with best model: \n{self.best_model_name}\n{self.best_model_transformation_params}\n{self.best_model_params}\nValidation: {res_len}\nSMAPE: {res}\nMAE: {res2}\nSPL: {res3}"
            except Exception:
                return "Initiated AutoTS object"

    def fit_data(
        self,
        df,
        date_col=None,
        value_col=None,
        id_col=None,
        future_regressor=None,
        weights={},
    ):
        """Part of the setup that involves fitting the initial data but not running any models."""
        self.date_col = date_col
        self.value_col = value_col
        self.id_col = id_col

        # convert data to wide format
        if date_col is None and value_col is None:
            df_wide = pd.DataFrame(df).copy()
            assert (
                type(df_wide.index) is pd.DatetimeIndex
            ), "df index is not pd.DatetimeIndex"
            df_wide = df_wide.sort_index(ascending=True)
        else:
            df_wide = long_to_wide(
                df,
                date_col=self.date_col,
                value_col=self.value_col,
                id_col=self.id_col,
                aggfunc=self.aggfunc,
            )

        # infer frequency
        inferred_freq = infer_frequency(df_wide)
        if self.frequency == 'infer' or self.frequency is None:
            self.used_frequency = inferred_freq
        else:
            self.used_frequency = self.frequency
        if self.verbose > 0:
            print(
                f"Data frequency is: {inferred_freq}, used frequency is: {self.used_frequency}"
            )
        if (self.used_frequency is None) and (self.verbose >= 0):
            print("Frequency is 'None'! Data frequency not recognized.")

        df_wide = df_cleanup(
            df_wide,
            frequency=self.used_frequency,
            prefill_na=self.prefill_na,
            na_tolerance=self.na_tolerance,
            drop_data_older_than_periods=self.drop_data_older_than_periods,
            aggfunc=self.aggfunc,
            drop_most_recent=self.drop_most_recent,
            verbose=self.verbose,
        )

        # handle categorical data if present
        self.categorical_transformer = NumericTransformer(verbose=self.verbose)
        df_wide_numeric = self.categorical_transformer.fit_transform(df_wide)
        del df_wide

        # check that column names are unique:
        if not df_wide_numeric.columns.is_unique:
            # maybe should make this an actual error in the future
            print(
                "Warning: column/series names are not unique. Unique column names are required for some features!"
            )
            time.sleep(3)  # give the message a chance to be seen

        if self.transformer_list == "auto":
            if df_wide_numeric.shape[1] <= 8:
                self.transformer_list = "all"
            elif df_wide_numeric.shape[1] <= 500:
                self.transformer_list = "fast"
            else:
                self.transformer_list = "superfast"

        # remove other ensembling types if univariate
        if df_wide_numeric.shape[1] == 1:
            self.ensemble = [x for x in self.ensemble if "horizontal" not in x]

        # because horizontal cannot handle non-string columns/series_ids
        self.h_ens_used = is_horizontal(self.ensemble)
        self.mosaic_used = is_mosaic(self.ensemble)
        if self.h_ens_used or self.mosaic_used:
            df_wide_numeric.columns = [str(xc) for xc in df_wide_numeric.columns]

        # flag if weights are given
        if bool(weights):
            self.weighted = True
        else:
            self.weighted = False

        # use "mean" to assign weight as mean
        if self.weighted:
            if weights == 'mean':
                weights = df_wide_numeric.mean(axis=0).to_dict()
            elif weights == 'median':
                weights = df_wide_numeric.median(axis=0).to_dict()
            elif weights == 'min':
                weights = df_wide_numeric.min(axis=0).to_dict()
            elif weights == 'max':
                weights = df_wide_numeric.max(axis=0).to_dict()
            elif weights == "inverse_mean":
                weights = (1 / df_wide_numeric.mean(axis=0)).to_dict()
        # clean up series weighting input
        weights = clean_weights(weights, df_wide_numeric.columns, self.verbose)
        self.weights = weights

        # replace any zeroes that occur prior to all non-zero values
        if self.remove_leading_zeroes:
            df_wide_numeric = remove_leading_zeros(df_wide_numeric)

        # check if NaN in last row
        self._nan_tail = df_wide_numeric.tail(2).isna().sum(axis=1).sum() > 0

        # preclean data
        if self.preclean is not None:
            self.preclean_transformer = GeneralTransformer(
                **self.preclean,
                n_jobs=self.n_jobs,
                holiday_country=self.holiday_country,
            )
            df_wide_numeric = self.preclean_transformer.fit_transform(df_wide_numeric)

        self.df_wide_numeric = df_wide_numeric
        self.startTimeStamps = df_wide_numeric.notna().idxmax()

        if future_regressor is not None:
            if not isinstance(future_regressor, pd.DataFrame):
                future_regressor = pd.DataFrame(future_regressor)
            if future_regressor.empty:
                raise ValueError(
                    "future_regressor empty, pass None if intending not to use"
                )
            if not isinstance(future_regressor.index, pd.DatetimeIndex):
                # should be same length as history as this is not yet the predict step
                future_regressor.index = df_wide_numeric.index
            # test shape
            if future_regressor.shape[0] != self.df_wide_numeric.shape[0]:
                print(
                    "future_regressor row count does not match length of training data"
                )
                time.sleep(2)

            # handle any non-numeric data, crudely
            self.regr_num_trans = NumericTransformer(verbose=self.verbose)
            self.future_regressor_train = self.regr_num_trans.fit_transform(
                future_regressor
            )

        # check how many validations are possible given the length of the data.
        self.num_validations = validate_num_validations(
            self.validation_method,
            self.num_validations,
            self.df_wide_numeric,
            self.forecast_length,
            self.min_allowed_train_percent,
            self.verbose,
        )

        # generate validation indices (so it can fail now, not after all the generations)
        self.validation_indexes = generate_validation_indices(
            self.validation_method,
            self.forecast_length,
            self.num_validations,
            self.df_wide_numeric,
            validation_params=self.similarity_validation_params
            if self.validation_method == "similarity"
            else self.seasonal_validation_params,
            preclean=None,
            verbose=0,
        )

    def fit(
        self,
        df,
        date_col: str = None,
        value_col: str = None,
        id_col: str = None,
        future_regressor=None,
        weights: dict = {},
        result_file: str = None,
        grouping_ids=None,
        validation_indexes: list = None,
    ):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed dataframe of series, or dataframe of three columns as below.
            date_col (str): name of datetime column
            value_col (str): name of column containing the data of series.
            id_col (str): name of column identifying different series.
            future_regressor (numpy.Array): single external regressor matching train.index
            weights (dict): {'colname1': 2, 'colname2': 5} - increase importance of a series in metric evaluation. Any left blank assumed to have weight of 1.
                pass the alias 'mean' as a str ie `weights='mean'` to automatically use the mean value of a series as its weight
                available aliases: mean, median, min, max
            result_file (str): results saved on each new generation. Does not include validation rounds.
                ".csv" save model results table.
                ".pickle" saves full object, including ensemble information.
            grouping_ids (dict): currently a one-level dict containing series_id:group_id mapping.
                used in 0.2.x but not 0.3.x+ versions. retained for potential future use
        """
        self.model = None
        self.grouping_ids = grouping_ids
        self.fitStart = pd.Timestamp.now()

        # convert class variables to local variables (makes testing easier)
        if self.validation_method == "custom":
            self.validation_indexes = validation_indexes
            assert (
                validation_indexes is not None
            ), "validation_indexes needs to be filled with 'custom' validation"
            # if auto num_validation, use as many as provided in custom
            if self.num_validations in ["auto", 'max']:
                self.num_validations == len(validation_indexes) - 1
            else:
                assert len(validation_indexes) >= (
                    self.num_validations + 1
                ), "validation_indexes needs to be >= num_validations + 1 with 'custom' validation"
        else:
            self.validation_indexes = []

        random_seed = self.random_seed
        metric_weighting = self.metric_weighting
        verbose = self.verbose
        template_cols = self.template_cols

        # shut off warnings if running silently
        if verbose <= 0:
            import warnings

            warnings.filterwarnings("ignore")

        # clean up result_file input, if given.
        if result_file is not None:
            formats = ['.csv', '.pickle']
            if not any(x in result_file for x in formats):
                print("result_file must be a valid str with .csv or .pickle")
                result_file = None

        # set random seeds for environment
        random_seed = abs(int(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.fit_data(
            df=df,
            date_col=date_col,
            value_col=value_col,
            id_col=id_col,
            future_regressor=future_regressor,
            weights=weights,
        )

        # record if subset or not
        if self.subset is not None:
            self.subset = abs(int(self.subset))
            if self.subset >= self.df_wide_numeric.shape[1]:
                self.subset_flag = False
            else:
                self.subset_flag = True
        else:
            self.subset_flag = False

        #
        # take a subset of the data if working with a large number of series
        if self.subset_flag:
            df_subset = subset_series(
                self.df_wide_numeric,
                list((self.weights.get(i)) for i in self.df_wide_numeric.columns),
                n=self.subset,
                random_state=random_seed,
            )
            if self.verbose > 1:
                print(f'First subset is of: {df_subset.columns}')
        else:
            df_subset = self.df_wide_numeric.copy()
        # go to first index
        first_idx = self.validation_indexes[0]
        if max(first_idx) > max(df_subset.index):
            raise ValueError("provided validation index exceeds historical data period")
        df_subset = df_subset.reindex(first_idx)

        # subset the weighting information as well
        if not self.weighted:
            current_weights = {x: 1 for x in df_subset.columns}
        else:
            current_weights = {x: self.weights[x] for x in df_subset.columns}

        # split train and test portions, and split regressor if present
        df_train, df_test = simple_train_test_split(
            df_subset,
            forecast_length=self.forecast_length,
            min_allowed_train_percent=self.min_allowed_train_percent,
            verbose=self.verbose,
        )
        self.validation_train_indexes.append(df_train.index)
        self.validation_test_indexes.append(df_test.index)
        if future_regressor is not None:
            future_regressor_train = self.future_regressor_train.reindex(
                index=df_train.index
            )
            future_regressor_test = self.future_regressor_train.reindex(
                index=df_test.index
            )
        else:
            future_regressor_train = None
            future_regressor_test = None

        self.start_time = pd.Timestamp.now()

        # unpack ensemble models so sub models appear at highest level
        self.initial_template = unpack_ensemble_models(
            self.initial_template,
            self.template_cols,
            keep_ensemble=True,
            recursive=True,
        )
        # remove horizontal ensembles from initial_template
        if 'Ensemble' in self.initial_template['Model'].tolist():
            self.initial_template = self.initial_template[
                self.initial_template['Ensemble'] <= 1
            ]
        # run the initial template
        submitted_parameters = self.initial_template.copy()
        self._run_template(
            self.initial_template,
            df_train,
            df_test,
            future_regressor_train=future_regressor_train,
            future_regressor_test=future_regressor_test,
            current_weights=current_weights,
            validation_round=0,
            max_generations=self.max_generations,
            current_generation=0,
            result_file=result_file,
        )

        # now run new generations, trying more models based on past successes.
        current_generation = 0
        num_mod_types = len(self.model_list)
        max_per_model_class_g = 5
        passedTime = (pd.Timestamp.now() - self.start_time).total_seconds() / 60

        while (
            current_generation < self.max_generations
            and passedTime < self.generation_timeout
        ):
            current_generation += 1
            if verbose > 0:
                print(
                    "New Generation: {} of {}".format(
                        current_generation, self.max_generations
                    )
                )
            # affirmative action to have more models represented, then less
            if current_generation < 5:
                cutoff_multiple = max_per_model_class_g
            elif current_generation < 10:
                cutoff_multiple = max_per_model_class_g - 1
            elif current_generation < 20:
                cutoff_multiple = max_per_model_class_g - 2
            else:
                cutoff_multiple = max_per_model_class_g - 3
            cutoff_multiple = 1 if cutoff_multiple < 1 else cutoff_multiple
            top_n = (
                num_mod_types * cutoff_multiple
                if num_mod_types > 2
                else num_mod_types * max_per_model_class_g
            )
            if df_train.shape[1] > 1:
                self.score_per_series = generate_score_per_series(
                    self.initial_results, self.metric_weighting, 1
                )
            new_template = NewGeneticTemplate(
                self.initial_results.model_results,
                submitted_parameters=submitted_parameters,
                sort_column="Score",
                sort_ascending=True,
                max_results=top_n,
                max_per_model_class=max_per_model_class_g,
                top_n=top_n,
                template_cols=template_cols,
                transformer_list=self.transformer_list,
                transformer_max_depth=self.transformer_max_depth,
                models_mode=self.models_mode,
                score_per_series=self.score_per_series,
                model_list=self.model_list,
            )
            submitted_parameters = pd.concat(
                [submitted_parameters, new_template],
                axis=0,
                ignore_index=True,
                sort=False,
            ).reset_index(drop=True)

            self._run_template(
                new_template,
                df_train,
                df_test,
                future_regressor_train=future_regressor_train,
                future_regressor_test=future_regressor_test,
                current_weights=current_weights,
                validation_round=0,
                max_generations=self.max_generations,
                current_generation=current_generation,
                result_file=result_file,
            )

            passedTime = (pd.Timestamp.now() - self.start_time).total_seconds() / 60

        # try ensembling
        if self.ensemble:
            try:
                self.score_per_series = generate_score_per_series(
                    self.initial_results, self.metric_weighting, 1
                )
                ensemble_templates = EnsembleTemplateGenerator(
                    self.initial_results,
                    forecast_length=self.forecast_length,
                    ensemble=self.ensemble,
                    score_per_series=self.score_per_series,
                )
                self._run_template(
                    ensemble_templates,
                    df_train,
                    df_test,
                    future_regressor_train=future_regressor_train,
                    future_regressor_test=future_regressor_test,
                    current_weights=current_weights,
                    validation_round=0,
                    max_generations="Ensembles",
                    current_generation=(current_generation + 1),
                    result_file=result_file,
                )
            except Exception as e:
                print(
                    f"Ensembling Error: {repr(e)}: {''.join(tb.format_exception(None, e, e.__traceback__))}"
                )

        # drop any duplicates in results
        self.initial_results.model_results = (
            self.initial_results.model_results.drop_duplicates(
                subset=(['ID'] + self.template_cols)
            )
        )

        # validation model count if float
        if (self.models_to_validate < 1) and (self.models_to_validate > 0):
            val_frac = self.models_to_validate
            val_frac = 1 if val_frac >= 0.99 else val_frac
            temp_len = self.initial_results.model_results.shape[0]
            self.models_to_validate = val_frac * temp_len
            self.models_to_validate = int(np.ceil(self.models_to_validate))
        if self.max_per_model_class is None:
            temp_len = len(self.model_list)
            self.max_per_model_class = (self.models_to_validate / temp_len) + 1
            self.max_per_model_class = int(np.ceil(self.max_per_model_class))

        # construct validation template
        validation_template = self.initial_results.model_results[
            self.initial_results.model_results['Exceptions'].isna()
        ]
        validation_template = validation_template[validation_template['Ensemble'] <= 1]
        validation_template = validation_template.drop_duplicates(
            subset=template_cols, keep='first'
        )
        validation_template = validation_template.sort_values(
            by="Score", ascending=True, na_position='last'
        )
        if str(self.max_per_model_class).isdigit():
            validation_template = (
                validation_template.sort_values(
                    'Score', ascending=True, na_position='last'
                )
                .groupby('Model')
                .head(self.max_per_model_class)
                .reset_index(drop=True)
            )
        validation_template = validation_template.sort_values(
            'Score', ascending=True, na_position='last'
        ).head(self.models_to_validate)
        # add on best per_series models (which may not be in the top scoring)
        if self.h_ens_used or self.mosaic_used:
            model_results = self.initial_results.model_results
            if self.models_to_validate < 50:
                n_per_series = 1
            elif self.models_to_validate > 500:
                n_per_series = 5
            else:
                n_per_series = 3
            self.score_per_series = generate_score_per_series(
                self.initial_results, self.metric_weighting, 1
            )
            mods = self.score_per_series.index[
                np.argsort(-self.score_per_series.values, axis=0)[
                    -1 : -1 - n_per_series : -1
                ].flatten()
            ]
            per_series_val = model_results[
                model_results['ID'].isin(mods.unique().tolist())
            ]
            validation_template = pd.concat(
                [validation_template, per_series_val], axis=0
            )
            validation_template = validation_template.drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )
        self.validation_template = validation_template[self.template_cols]
        if self.validate_import is not None:
            self.validation_template = pd.concat(
                [self.validation_template, self.validate_import]
            ).drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )

        # run validations
        if self.num_validations > 0:
            self._run_validations(
                df_wide_numeric=self.df_wide_numeric,
                num_validations=self.num_validations,
                validation_template=self.validation_template,
                future_regressor=self.future_regressor_train,
            )
            # ensembles built on validation results
            if self.ensemble:
                try:
                    ens_copy = copy.copy(self.validation_results)
                    run_count = (
                        self.initial_results.model_results[
                            self.initial_results.model_results.Exceptions.isna()
                        ][['Model', 'ID']]
                        .groupby("ID")
                        .count()
                    )
                    models_to_use = run_count[
                        run_count['Model'] >= (self.num_validations + 1)
                    ].index.tolist()
                    ens_copy.model_results = ens_copy.model_results[
                        ens_copy.model_results.ID.isin(models_to_use)
                    ]
                    self.ens_copy = ens_copy
                    self.score_per_series = generate_score_per_series(
                        self.initial_results,
                        self.metric_weighting,
                        total_validations=(self.num_validations + 1),
                    )
                    ensemble_templates = EnsembleTemplateGenerator(
                        ens_copy,
                        forecast_length=self.forecast_length,
                        ensemble=self.ensemble,
                        score_per_series=self.score_per_series,
                    )
                    self.ensemble_templates2 = ensemble_templates
                    self._run_template(
                        ensemble_templates,
                        df_train,
                        df_test,
                        future_regressor_train=future_regressor_train,
                        future_regressor_test=future_regressor_test,
                        current_weights=current_weights,
                        validation_round=0,
                        max_generations="Ensembles",
                        current_generation=(current_generation + 2),
                        result_file=result_file,
                    )
                    self._run_validations(
                        df_wide_numeric=self.df_wide_numeric,
                        num_validations=self.num_validations,
                        validation_template=ensemble_templates,
                        future_regressor=self.future_regressor_train,
                        first_validation=False,
                    )
                except Exception as e:
                    print(
                        f"Post-Validation Ensembling Error: {repr(e)}: {''.join(tb.format_exception(None, e, e.__traceback__))}"
                    )
                    time.sleep(5)

        # run validation_results aggregation
        self = self.validation_agg()

        # Construct horizontal style ensembles
        models_to_use = None
        if self.h_ens_used or self.mosaic_used:
            ensemble_templates = pd.DataFrame()
            try:
                self.score_per_series = generate_score_per_series(
                    self.initial_results,
                    metric_weighting=metric_weighting,
                    total_validations=(self.num_validations + 1),
                )
                ens_templates = HorizontalTemplateGenerator(
                    self.score_per_series,
                    model_results=self.initial_results.model_results,
                    forecast_length=self.forecast_length,
                    ensemble=self.ensemble,
                    subset_flag=self.subset_flag,
                )
                ensemble_templates = pd.concat(
                    [ensemble_templates, ens_templates], axis=0
                )
                models_to_use = horizontal_template_to_model_list(ens_templates)
            except Exception as e:
                if self.verbose >= 0:
                    print(f"Horizontal Ensemble Generation Error: {repr(e)}")
                    time.sleep(5)
            try:
                # eventually plan to allow window size to be controlled by params
                if self.mosaic_used:
                    weight_per_value = (
                        self.initial_results.full_mae_errors
                        * metric_weighting.get('mae_weighting', 0)
                        + self.initial_results.full_pl_errors
                        * metric_weighting.get('spl_weighting', 0)
                        + self.initial_results.squared_errors
                        * metric_weighting.get('rmse_weighting', 0)
                    )
                if "n_crosshair" in self.ensemble:
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=self.initial_results.full_mae_errors,
                        models_to_use=models_to_use,
                        smoothing_window=7,
                        metric_name="H-MAE",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    try:
                        # find a way of parsing it down to n models to use
                        total_vals = self.num_validations + 1
                        local_results = self.initial_results.model_results.copy()
                        full_mae_errors=[
                            generate_crosshair_score(x)
                            for x in self.initial_results.full_mae_errors
                        ]
                        id_array, errors_array = process_mosaic_arrays(
                            local_results, full_mae_ids=self.initial_results.full_mae_ids,
                            full_mae_errors=full_mae_errors,
                            total_vals=total_vals,
                            models_to_use=models_to_use, smoothing_window=None
                        )
                        # so it's summarized by progressively longer chunks
                        chunks = parse_forecast_length(self.forecast_length)
                        all_pieces = []
                        for piece in chunks:
                            all_pieces.append(pd.DataFrame(errors_array[:, piece, :].mean(axis=1)))
                        n_pieces = pd.concat(all_pieces, axis=1)
                        n_pieces.index = id_array
                        # can modify K later
                        chosen_model_n = n_limited_horz(n_pieces, K=50, safety_model=False)
                        ens_templates = generate_mosaic_template(
                            initial_results=self.initial_results.model_results,
                            full_mae_ids=self.initial_results.full_mae_ids,
                            num_validations=self.num_validations,
                            col_names=df_subset.columns,
                            full_mae_errors=full_mae_errors,
                            smoothing_window=None,
                            metric_name="n_crosshair",
                            models_to_use=chosen_model_n,
                        )
                        ensemble_templates = pd.concat(
                            [ensemble_templates, ens_templates], axis=0
                        )
                    except Exception as e:
                        print(f"N_CROSSHAIR FAILED WITH ERROR: {repr(e)}")
                if "mosaic_crosshair" in self.ensemble or "mosaic-crosshair" in self.ensemble:
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=[
                            generate_crosshair_score(x)
                            for x in self.initial_results.full_mae_errors
                        ],
                        smoothing_window=None,
                        metric_name="mae-crosshair",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=[
                            generate_crosshair_score(x)
                            for x in self.initial_results.squared_errors
                        ],
                        smoothing_window=None,
                        metric_name="se-crosshair",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=[
                            generate_crosshair_score(x)
                            for x in self.initial_results.full_pl_errors
                        ],
                        smoothing_window=3,
                        metric_name="spl-crosshair",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=[
                            generate_crosshair_score(x) for x in weight_per_value
                        ],
                        smoothing_window=None,
                        metric_name="weighted-crosshair",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                if "mosaic_window" in self.ensemble or "mosaic-window" in self.ensemble:
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=self.initial_results.full_mae_errors,
                        smoothing_window=14,
                        metric_name="MAE",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=self.initial_results.full_pl_errors,
                        smoothing_window=10,
                        metric_name="SPL",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=self.initial_results.full_mae_errors,
                        smoothing_window=7,
                        metric_name="MAE",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=self.initial_results.full_mae_errors,
                        models_to_use=models_to_use,
                        smoothing_window=7,
                        metric_name="H-MAE",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=self.initial_results.full_mae_errors,
                        smoothing_window=3,
                        metric_name="MAE",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=weight_per_value,
                        smoothing_window=3,
                        metric_name="Weighted",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=weight_per_value,
                        smoothing_window=10,
                        metric_name="Weighted",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                if 'mosaic' in self.ensemble:
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=self.initial_results.squared_errors,
                        smoothing_window=None,
                        metric_name="SE",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=self.initial_results.full_mae_errors,
                        smoothing_window=None,
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    ens_templates = generate_mosaic_template(
                        initial_results=self.initial_results.model_results,
                        full_mae_ids=self.initial_results.full_mae_ids,
                        num_validations=self.num_validations,
                        col_names=df_subset.columns,
                        full_mae_errors=weight_per_value,
                        smoothing_window=None,
                        metric_name="Weighted",
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                    if models_to_use is not None:
                        ens_templates = generate_mosaic_template(
                            initial_results=self.initial_results.model_results,
                            full_mae_ids=self.initial_results.full_mae_ids,
                            num_validations=self.num_validations,
                            col_names=df_subset.columns,
                            full_mae_errors=weight_per_value,
                            smoothing_window=None,
                            models_to_use=models_to_use,
                            metric_name="Horiz-Weighted",
                        )
                        ensemble_templates = pd.concat(
                            [ensemble_templates, ens_templates], axis=0
                        )
            except Exception as e:
                if self.verbose >= 0:
                    print(f"Mosaic Ensemble Generation Error: {repr(e)}")
            try:
                # test on initial test split to make sure they work
                self._run_template(
                    ensemble_templates,
                    df_train,
                    df_test,
                    future_regressor_train=future_regressor_train,
                    future_regressor_test=future_regressor_test,
                    current_weights=current_weights,
                    validation_round=0,
                    max_generations="Horizontal Ensembles",
                    model_count=0,
                    current_generation=0,
                    result_file=result_file,
                )
            except Exception as e:
                if self.verbose >= 0:
                    print(
                        f"Horizontal/Mosaic Ensembling Error: {repr(e)}: {''.join(tb.format_exception(None, e, e.__traceback__))}"
                    )
                # hens_model_results = TemplateEvalObject().model_results.copy()

            # rerun validation_results aggregation with new models added
            self = self.validation_agg()

        self._set_best_model()

        # clean up any remaining print statements
        sys.stdout.flush()
        self.fitRuntime = pd.Timestamp.now() - self.fitStart
        return self

    def validation_agg(self):
        self.validation_results = copy.copy(self.initial_results)
        self.validation_results = validation_aggregation(
            self.validation_results, df_train=self.df_wide_numeric
        )
        return self

    def _set_best_model(self, metric_weighting=None, allow_horizontal=True):
        if metric_weighting is None:
            metric_weighting = self.metric_weighting
        hens_model_results = self.initial_results.model_results[
            self.initial_results.model_results['Ensemble'] == 2
        ].copy()
        # remove failures
        hens_model_results = hens_model_results[
            hens_model_results['Exceptions'].isnull()
        ]
        requested_H_ens = (
            (self.h_ens_used or self.mosaic_used) and allow_horizontal
        )
        # here I'm assuming that if some horizontal models ran, we are going to use those
        # horizontal ensembles can't be compared directly to others because they don't get run through all validations
        # they are built themselves from cross validation so a full rerun of validations is unnecessary
        self.best_model_non_horizontal = self._best_non_horizontal(
            metric_weighting=metric_weighting
        )
        if not hens_model_results.empty and requested_H_ens:
            hens_model_results.loc['Score'] = generate_score(
                hens_model_results,
                metric_weighting=metric_weighting,
                prediction_interval=self.prediction_interval,
            )
            self.best_model = hens_model_results.sort_values(
                by="Score", ascending=True, na_position='last'
            ).head(1)[self.template_cols_id]
            self.ensemble_check = 1
        # print a warning if requested but unable to produce a horz ensemble
        elif requested_H_ens:
            if self.verbose >= 0:
                print("Horizontal ensemble failed. Using best non-horizontal.")
                time.sleep(3)  # give it a chance to be seen
            self.best_model = self.best_model_non_horizontal
        elif not hens_model_results.empty:
            if self.verbose >= 0:
                print(
                    f"Horizontal ensemble available but not requested in model selection: {self.ensemble} and allow_horizontal: {allow_horizontal}."
                )
                time.sleep(3)  # give it a chance to be seen
            self.best_model = self.best_model_non_horizontal
        else:
            self.best_model = self.best_model_non_horizontal
        # give a more convenient dict option
        self.parse_best_model()
        return self

    def _best_non_horizontal(self, metric_weighting=None, series=None):
        if self.validation_results is None:
            if not self.initial_results.model_results.empty:
                self = self.validation_agg()
            else:
                raise ValueError(
                    "validation results are None, cannot choose best model without fit"
                )
        if metric_weighting is None:
            metric_weighting = self.metric_weighting
        # choose best model, when no horizontal ensembling is done
        eligible_models = self.validation_results.model_results[
            self.validation_results.model_results['Runs'] >= (self.num_validations + 1)
        ].copy()
        if eligible_models.empty:
            # this may occur if there is enough data for full validations
            # but a lot of that data is bad leading to complete validation round failures
            print(
                "your validation results are questionable, perhaps bad data and too many num_validations"
            )
            time.sleep(3)  # give it a chance to be seen
            max_vals = self.validation_results.model_results['Runs'].max()
            eligible_models = self.validation_results.model_results[
                self.validation_results.model_results['Runs'] >= max_vals
            ]
        if series is not None:
            # return a result based on the performance of only one series
            score_per_series = generate_score_per_series(
                self.initial_results,
                metric_weighting=metric_weighting,
                total_validations=(self.num_validations + 1),
            )
            best_model_id = score_per_series[series].idxmin()
            best_model = self.initial_results.model_results[
                self.initial_results.model_results['ID'] == best_model_id
            ].iloc[0:1][self.template_cols_id]
        else:
            # previously I was relying on the mean of Scores calculated for each individual validation
            eligible_models['Score'] = generate_score(
                eligible_models,
                metric_weighting=metric_weighting,
                prediction_interval=self.prediction_interval,
            )
            try:
                best_model = (
                    eligible_models.sort_values(
                        by="Score", ascending=True, na_position='last'
                    )
                    .drop_duplicates(subset=self.template_cols)
                    .head(1)[self.template_cols_id]
                )
            except IndexError:
                raise ValueError(
                    """No models available from validation.
    Try increasing models_to_validate, max_per_model_class
    or otherwise increase models available."""
                )
        return best_model

    def parse_best_model(self):
        if self.best_model.empty:
            raise ValueError(
                "no best model present. Run .fit() of the AutoTS class first."
            )
        self.best_model_name = self.best_model['Model'].iloc[0]
        self.best_model_id = self.best_model['ID'].iloc[0]
        self.best_model_params = json.loads(self.best_model['ModelParameters'].iloc[0])
        self.best_model_transformation_params = json.loads(
            self.best_model['TransformationParameters'].iloc[0]
        )
        self.best_model_ensemble = self.best_model['Ensemble'].iloc[0]
        # flag if is any type of ensemble
        self.ensemble_check = int(self.best_model_ensemble > 0)
        # set flags to check if regressors or ensemble used in final model.
        self.used_regressor_check = self._regr_param_check(
            self.best_model_params.copy()
        )
        self.regressor_used = self.used_regressor_check

    def _regr_param_check(self, param_dict):
        """Help to search for if a regressor was used in model."""
        out = False
        # load string if not dictionary
        if isinstance(param_dict, dict):
            cur_dict = param_dict.copy()
        else:
            cur_dict = json.loads(param_dict)
        current_keys = cur_dict.keys()
        # always look in ModelParameters if present
        if 'ModelParameters' in current_keys:
            return self._regr_param_check(cur_dict["ModelParameters"])
        # then dig in and see if regression type is a key
        if "regression_type" in current_keys:
            reg_param = cur_dict['regression_type']
            if str(reg_param).lower() == 'user':
                return True
        # now check if it's an Ensemble
        if "models" in current_keys and 'regression_model' not in current_keys:
            for key in cur_dict['models'].keys():
                # stop as soon as any finds a regressor
                if self._regr_param_check(cur_dict['models'][key]):
                    return True
        return out

    def _run_template(
        self,
        template,
        df_train,
        df_test,
        future_regressor_train,
        future_regressor_test,
        current_weights,
        validation_round=0,
        max_generations="0",
        model_count=None,
        current_generation=0,
        result_file=None,
    ):
        """Get results for one batch of models."""
        model_count = self.model_count if model_count is None else model_count
        template_result = TemplateWizard(
            template,
            df_train=df_train,
            df_test=df_test,
            weights=current_weights,
            model_count=model_count,
            forecast_length=self.forecast_length,
            frequency=self.used_frequency,
            prediction_interval=self.prediction_interval,
            no_negatives=self.no_negatives,
            constraint=self.constraint,
            ensemble=self.ensemble,
            future_regressor_train=future_regressor_train,
            future_regressor_forecast=future_regressor_test,
            holiday_country=self.holiday_country,
            startTimeStamps=self.startTimeStamps,
            template_cols=self.template_cols,
            model_interrupt=self.model_interrupt,
            grouping_ids=self.grouping_ids,
            random_seed=self.random_seed,
            verbose=self.verbose,
            max_generations=max_generations,
            n_jobs=self.n_jobs,
            validation_round=validation_round,
            traceback=self.traceback,
            current_model_file=self.current_model_file,
            current_generation=current_generation,
            mosaic_used=self.mosaic_used,
            force_gc=self.force_gc,
        )
        if model_count == 0:
            self.model_count += template_result.model_count
        else:
            self.model_count = template_result.model_count
        # capture results from lower-level template run
        if "TotalRuntime" in template_result.model_results.columns:
            template_result.model_results['TotalRuntime'].fillna(
                pd.Timedelta(seconds=60), inplace=True
            )
        else:
            # trying to catch a rare and sneaky bug (perhaps some variety of beetle?)
            print(f"TotalRuntime missing in {current_generation}!")
            self.template_result_error = template_result.model_results.copy()
            self.template_error = template.copy()
        # gather results of template run
        self.initial_results = self.initial_results.concat(template_result)
        self.initial_results.model_results['Score'] = generate_score(
            self.initial_results.model_results,
            metric_weighting=self.metric_weighting,
            prediction_interval=self.prediction_interval,
        )
        if result_file is not None:
            self.initial_results.save(result_file)

    def _run_validations(
        self,
        df_wide_numeric,
        num_validations,
        validation_template,
        future_regressor,
        first_validation=True,
        skip_first_index=True,
    ):
        """Loop through a template for n validation segments."""
        for y in range(num_validations):
            cslc = y + 1 if skip_first_index else y
            if self.verbose > 0:
                print("Validation Round: {}".format(str(cslc)))
            # slice the validation data into current validation slice
            current_slice = df_wide_numeric.reindex(self.validation_indexes[cslc])

            # subset series (if used) and take a new train/test split
            if self.subset_flag:
                # mosaic can't handle different cols in each validation
                if self.mosaic_used:
                    rand_st = self.random_seed
                else:
                    rand_st = self.random_seed + y + 1
                df_subset = subset_series(
                    current_slice,
                    list((self.weights.get(i)) for i in current_slice.columns),
                    n=self.subset,
                    random_state=rand_st,
                )
                if self.verbose > 1:
                    print(f'Val {cslc} subset is of: {df_subset.columns}')
            else:
                df_subset = current_slice
            # subset weighting info
            if not self.weighted:
                current_weights = {x: 1 for x in df_subset.columns}
            else:
                current_weights = {x: self.weights[x] for x in df_subset.columns}

            val_df_train, val_df_test = simple_train_test_split(
                df_subset,
                forecast_length=self.forecast_length,
                min_allowed_train_percent=self.min_allowed_train_percent,
                verbose=self.verbose,
            )
            if first_validation:
                self.validation_train_indexes.append(val_df_train.index)
                self.validation_test_indexes.append(val_df_test.index)
            if self.verbose >= 2:
                print(f'Validation train index is {val_df_train.index}')

            # slice regressor into current validation slices
            if future_regressor is not None:
                val_future_regressor_train = future_regressor.reindex(
                    index=val_df_train.index
                )
                val_future_regressor_test = future_regressor.reindex(
                    index=val_df_test.index
                )
            else:
                val_future_regressor_train = None
                val_future_regressor_test = None

            # force NaN for robustness
            if self.introduce_na or (self.introduce_na is None and self._nan_tail):
                if self.introduce_na:
                    idx = val_df_train.index
                    # make 20% of rows NaN at random
                    val_df_train = val_df_train.sample(
                        frac=0.8, random_state=self.random_seed
                    ).reindex(idx)
                nan_frac = val_df_train.shape[1] / num_validations
                val_df_train.iloc[
                    -2:, int(nan_frac * y) : int(nan_frac * (y + 1))
                ] = np.nan

            # run validation template on current slice
            self._run_template(
                validation_template,
                val_df_train,
                val_df_test,
                future_regressor_train=val_future_regressor_train,
                future_regressor_test=val_future_regressor_test,
                current_weights=current_weights,
                validation_round=(y + 1),
                max_generations="0",
                model_count=0,
                result_file=None,
            )

        self = self.validation_agg()

    def _predict(
        self,
        forecast_length: int = "self",
        prediction_interval: float = 'self',
        future_regressor=None,
        fail_on_forecast_nan: bool = True,
        verbose: int = 'self',
        model_name=None,
        model_params=None,
        model_transformation_params=None,
        df_wide_numeric=None,
        future_regressor_train=None,
        refit=False,  # refit model
        bypass_save=False,  # don't even try saving model
    ):
        use_model = self.best_model_name if model_name is None else model_name
        use_params = (
            self.best_model_params.copy() if model_params is None else model_params
        )
        use_trans = (
            self.best_model_transformation_params
            if model_transformation_params is None
            else model_transformation_params
        )
        use_data = self.df_wide_numeric if df_wide_numeric is None else df_wide_numeric
        use_regr_train = (
            self.future_regressor_train
            if future_regressor_train is None
            else future_regressor_train
        )
        self.update_fit_check = (
            use_model in update_fit
        )  # True means model can be updated without retraining
        if self.update_fit_check and not bypass_save:
            if self.model is None or refit:
                self.model = ModelPrediction(
                    transformation_dict=use_trans,
                    model_str=use_model,
                    parameter_dict=use_params,
                    forecast_length=forecast_length,
                    frequency=self.used_frequency,
                    prediction_interval=prediction_interval,
                    no_negatives=self.no_negatives,
                    constraint=self.constraint,
                    holiday_country=self.holiday_country,
                    startTimeStamps=self.startTimeStamps,
                    grouping_ids=self.grouping_ids,
                    fail_on_forecast_nan=fail_on_forecast_nan,
                    random_seed=self.random_seed,
                    verbose=verbose,
                    n_jobs=self.n_jobs,
                    current_model_file=self.current_model_file,
                    return_model=True,
                )
                self.model = self.model.fit(use_data, future_regressor=use_regr_train)
            else:
                self.model.fit_data(use_data, future_regressor=use_regr_train)
            df_forecast = self.model.predict(
                forecast_length, future_regressor=future_regressor
            )
        else:
            df_forecast = model_forecast(
                model_name=use_model,
                model_param_dict=use_params,
                model_transform_dict=use_trans,
                df_train=use_data,
                forecast_length=forecast_length,
                frequency=self.used_frequency,
                prediction_interval=prediction_interval,
                no_negatives=self.no_negatives,
                constraint=self.constraint,
                future_regressor_train=use_regr_train,
                future_regressor_forecast=future_regressor,
                holiday_country=self.holiday_country,
                startTimeStamps=self.startTimeStamps,
                grouping_ids=self.grouping_ids,
                fail_on_forecast_nan=fail_on_forecast_nan,
                random_seed=self.random_seed,
                verbose=verbose,
                n_jobs=self.n_jobs,
                template_cols=self.template_cols,
                current_model_file=self.current_model_file,
                return_model=True,
                force_gc=self.force_gc,
            )
        # convert categorical back to numeric
        trans = self.categorical_transformer
        df_forecast.forecast = trans.inverse_transform(df_forecast.forecast)
        df_forecast.lower_forecast = trans.inverse_transform(df_forecast.lower_forecast)
        df_forecast.upper_forecast = trans.inverse_transform(df_forecast.upper_forecast)
        # undo preclean transformations if necessary
        if self.preclean is not None:
            # self.raw_forecast = copy.copy(df_forecast)
            df_forecast.forecast = self.preclean_transformer.inverse_transform(
                df_forecast.forecast
            )
            df_forecast.lower_forecast = self.preclean_transformer.inverse_transform(
                df_forecast.lower_forecast
            )
            df_forecast.upper_forecast = self.preclean_transformer.inverse_transform(
                df_forecast.upper_forecast
            )
        sys.stdout.flush()
        if self.force_gc:
            gc.collect()
        return df_forecast

    def predict(
        self,
        forecast_length: int = "self",
        prediction_interval: float = 'self',
        future_regressor=None,
        hierarchy=None,
        just_point_forecast: bool = False,
        fail_on_forecast_nan: bool = True,
        verbose: int = 'self',
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        If using a model from update_fit list, with no ensembling, underlying model will not be retrained when used as below, with a single prediction interval:
        This designed for high speed forecasting. Full retraining is best when there is sufficient time.
        ```python
        model = AutoTS(model_list='update_fit')
        model.fit(df)
        model.predict()
        # for new data without retraining
        model.fit_data(df)
        model.predict()
        # to force retrain of best model (but not full model search)
        model.model = None
        model.fit_data(df)
        model.predict()
        ```

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            prediction_interval (float): interval of upper/lower forecasts.
                defaults to 'self' ie the interval specified in __init__()
                if prediction_interval is a list, then returns a dict of forecast objects.
                    {str(interval): prediction_object}
            future_regressor (numpy.Array): additional regressor
            hierarchy: Not yet implemented
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts
            fail_on_forecast_nan (bool): if False, return forecasts even if NaN present, if True, raises error if any nan in forecast

        Return:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        verbose = self.verbose if verbose == 'self' else verbose
        if forecast_length == 'self':
            forecast_length = self.forecast_length
        if prediction_interval == 'self':
            prediction_interval = self.prediction_interval

        # checkup regressor
        if future_regressor is not None:
            if not isinstance(future_regressor, pd.DataFrame):
                future_regressor = pd.DataFrame(future_regressor)
            if self.future_regressor_train is None:
                raise ValueError(
                    "regressor passed to .predict but no regressor was passed to .fit"
                )
            # handle any non-numeric data, crudely
            future_regressor = self.regr_num_trans.transform(future_regressor)
            # make sure training regressor fits training data index
            self.future_regressor_train = self.future_regressor_train.reindex(
                index=self.df_wide_numeric.index
            )

        # allow multiple prediction intervals
        if isinstance(prediction_interval, list):
            forecast_objects = {}
            for interval in prediction_interval:
                df_forecast = self._predict(
                    forecast_length=forecast_length,
                    prediction_interval=interval,
                    future_regressor=future_regressor,
                    fail_on_forecast_nan=fail_on_forecast_nan,
                    verbose=verbose,
                    refit=True,  # need to audit all models to make sure they don't require update train with new prediction_interval
                )
                forecast_objects[str(interval)] = df_forecast
            return forecast_objects
        else:
            df_forecast = self._predict(
                forecast_length=forecast_length,
                prediction_interval=prediction_interval,
                future_regressor=future_regressor,
                fail_on_forecast_nan=fail_on_forecast_nan,
                verbose=verbose,
            )
            if just_point_forecast:
                return df_forecast.forecast
            else:
                return df_forecast

    def results(self, result_set: str = 'initial'):
        """Convenience function to return tested models table.

        Args:
            result_set (str): 'validation' or 'initial'
        """
        if result_set == 'validation':
            return self.validation_results.model_results
        else:
            return self.initial_results.model_results

    def failure_rate(self, result_set: str = 'initial'):
        """Return fraction of models passing with exceptions.

        Args:
            result_set (str, optional): 'validation' or 'initial'. Defaults to 'initial'.

        Returns:
            float.

        """
        initial_results = self.results(result_set=result_set)
        n = initial_results.shape[0]
        x = (n - initial_results['Exceptions'].isna().sum()) / n
        return x

    def export_template(
        self,
        filename=None,
        models: str = 'best',
        n: int = 20,
        max_per_model_class: int = None,
        include_results: bool = False,
    ):
        """Export top results as a reusable template.

        Args:
            filename (str): 'csv' or 'json' (in filename).
                `None` to return a dataframe and not write a file.
            models (str): 'best' or 'all', and 'slowest' for diagnostics
            n (int): if models = 'best', how many n-best to export
            max_per_model_class (int): if models = 'best',
                the max number of each model class to include in template
            include_results (bool): whether to include performance metrics
        """
        if models == 'all':
            export_template = self.initial_results.model_results[self.template_cols_id]
            export_template = export_template.drop_duplicates()
        elif models == 'best':
            # skip to the answer if just n==1
            if n == 1 and not include_results:
                export_template = self.best_model
            else:
                export_template = self.validation_results.model_results
                # all validated models + horizontal ensembles
                export_template = export_template[
                    (export_template['Runs'] >= (self.num_validations + 1))
                    | (export_template['Ensemble'] >= 2)
                ]
                if str(max_per_model_class).isdigit():
                    export_template = (
                        export_template.sort_values('Score', ascending=True)
                        .groupby('Model')
                        .head(max_per_model_class)
                        .reset_index()
                    )
                export_template = export_template.nsmallest(n, columns=['Score'])
                if self.best_model_id not in export_template['ID']:
                    export_template = pd.concat(
                        [
                            self.validation_results.model_results[
                                self.validation_results.model_results['ID']
                                == self.best_model_id
                            ],
                            export_template,
                        ]
                    ).drop_duplicates()
                if not include_results:
                    export_template = export_template[self.template_cols_id]
        elif models == "slowest":
            return self.save_template(
                filename,
                self.initial_results.model_results.nlargest(
                    n, columns=['TotalRuntime']
                ),
            )
        else:
            raise ValueError("`models` must be 'all' or 'best'")
        return self.save_template(filename, export_template)

    def save_template(self, filename, export_template, **kwargs):
        """Helper function for the save part of export_template."""
        try:
            if filename is None:
                return export_template
            elif '.csv' in filename:
                return export_template.to_csv(
                    filename, index=False, **kwargs
                )  # lineterminator='\r\n'
            elif '.json' in filename:
                return export_template.to_json(filename, orient='columns', **kwargs)
            else:
                raise ValueError("file must be .csv or .json")
        except PermissionError as e:
            raise PermissionError(
                "Permission Error: directory or existing file is locked for editing."
            ) from e

    def load_template(self, filename):
        """Helper funciton for just loading the file part of import_template."""
        if isinstance(filename, pd.DataFrame):
            import_template = filename.copy()
        elif '.csv' in filename:
            import_template = pd.read_csv(filename)
        elif '.json' in filename:
            import_template = pd.read_json(filename, orient='columns')
        else:
            raise ValueError("file must be .csv or .json")

        try:
            import_template = import_template[self.template_cols_id]
        except Exception:
            try:
                import_template = import_template[self.template_cols]
            except Exception:
                print(
                    "Column names {} were not recognized as matching template columns: {}".format(
                        str(import_template.columns), str(self.template_cols_id)
                    )
                )
        return import_template

    def _enforce_model_list(
        self,
        template,
        model_list=None,
        include_ensemble=False,
        addon_flag=False,
        include_horizontal=True,
    ):
        """remove models not in given model list."""
        if model_list is None:
            model_list = self.model_list
        if isinstance(model_list, dict):
            model_list = list(model_list.keys())
        if include_ensemble:
            mod_list = model_list + ['Ensemble']
        else:
            mod_list = model_list
        present = template['Model'].unique().tolist()
        template = template[template['Model'].isin(mod_list)]
        # double method of removing Ensemble
        if not include_ensemble and "Ensemble" in template.columns:
            template = template[template["Ensemble"] == 0]
        elif not include_horizontal and "Ensemble" in template.columns:
            # remove only horizontal ensembles
            template = template[template["Ensemble"] <= 1]
        if template.shape[0] == 0:
            error_msg = f"Len 0. Model_list {model_list} does not match models in imported template {present}, template import failed."
            if addon_flag:
                # if template is addon, then this is fine as just a warning
                print(error_msg)
            else:
                raise ValueError(error_msg)
        return template

    def import_template(
        self,
        filename: str,
        method: str = "add_on",
        enforce_model_list: bool = True,
        include_ensemble: bool = False,
        include_horizontal: bool = False,
        force_validation: bool = False,
    ):
        """Import a previously exported template of model parameters.
        Must be done before the AutoTS object is .fit().

        Args:
            filename (str): file location (or a pd.DataFrame already loaded)
            method (str): 'add_on' or 'only' - "add_on" keeps `initial_template` generated in init. "only" uses only this template.
            enforce_model_list (bool): if True, remove model types not in model_list
            include_ensemble (bool): if enforce_model_list is True, this specifies whether to allow ensembles anyway (otherwise they are unpacked and parts kept)
            include_horizontal (bool): if enforce_model_list is True, this specifies whether to allow ensembles except horizontal (overridden by keep_ensemble)
            force_validation (bool): if True, all models imported here will automatically get sent to full cross validation (regardless of first eval performance)
        """
        if method.lower() in ['add on', 'addon', 'add_on']:
            addon_flag = True
        else:
            addon_flag = False

        import_template = self.load_template(filename)

        import_template = unpack_ensemble_models(
            import_template, self.template_cols, keep_ensemble=True, recursive=True
        )

        # _enforce_model_list can handle this but when false this is needed
        if not include_ensemble:
            import_template = unpack_ensemble_models(
                import_template, self.template_cols, keep_ensemble=False, recursive=True
            )

        if enforce_model_list:
            import_template = self._enforce_model_list(
                template=import_template,
                model_list=None,
                include_ensemble=include_ensemble,
                addon_flag=addon_flag,
                include_horizontal=False,
            )

        if addon_flag:
            self.initial_template = self.initial_template.merge(
                import_template,
                how='outer',
                on=self.initial_template.columns.intersection(
                    import_template.columns
                ).to_list(),
            )
            self.initial_template = self.initial_template.drop_duplicates(
                subset=self.template_cols
            )
        elif method.lower() in ['only', 'user only', 'user_only', 'import_only']:
            self.initial_template = import_template
        else:
            return ValueError("method must be 'addon' or 'only'")

        if force_validation:
            if self.validate_import is None:
                self.validate_import = import_template
            else:
                self.validate_import = pd.concat(
                    [self.validate_import, import_template]
                )

        return self

    def export_best_model(self, filename, **kwargs):
        """Basically the same as export_template but only ever the one best model."""
        return self.save_template(filename, self.best_model.copy(), **kwargs)

    def import_best_model(
        self,
        import_target,
        enforce_model_list: bool = True,
        include_ensemble: bool = True,
    ):
        """Load a best model, overriding any existing setting.

        Args:
            import_target: pd.DataFrame or file path
        """
        if isinstance(import_target, pd.DataFrame):
            template = import_target.copy()
        else:
            template = self.load_template(import_target)
        if not include_ensemble:
            template = unpack_ensemble_models(
                template, self.template_cols, keep_ensemble=False, recursive=True
            )
        if enforce_model_list:
            template = self._enforce_model_list(
                template=template,
                model_list=None,
                include_ensemble=include_ensemble,
                addon_flag=False,
            )

        self.best_model = template.iloc[0:1]

        self.parse_best_model()
        return self

    def import_results(self, filename):
        """Add results from another run on the same data.

        Input can be filename with .csv or .pickle.
        or can be a DataFrame of model results or a full TemplateEvalObject
        """
        csv_flag = False
        if isinstance(filename, str):
            if ".csv" in filename:
                csv_flag = True
        if isinstance(filename, pd.DataFrame) or csv_flag:
            if ".csv" not in filename:
                past_results = filename.copy()
            else:
                past_results = pd.read_csv(filename)
            # remove those that succeeded (ie had no Exception)
            past_results = past_results[pd.isnull(past_results['Exceptions'])]
            # remove validation results
            past_results = past_results[(past_results['ValidationRound']) == 0]
            past_results['TotalRuntime'] = pd.to_timedelta(past_results['TotalRuntime'])
            # combine with any existing results
            self.initial_results.model_results = pd.concat(
                [past_results, self.initial_results.model_results],
                axis=0,
                ignore_index=True,
                sort=False,
            ).reset_index(drop=True)
            self.initial_results.model_results.drop_duplicates(
                subset=self.template_cols, keep='first', inplace=True
            )
        else:
            if isinstance(filename, TemplateEvalObject):
                new_obj = filename
            elif '.pickle' in filename:
                import pickle

                new_obj = pickle.load(open(filename, "rb"))
            else:
                raise ValueError("import type not recognized.")
            self.initial_results = self.initial_results.concat(new_obj)
        return self

    def horizontal_per_generation(self):
        df_train = self.df_wide_numeric.reindex(self.validation_train_indexes[0])
        df_test = self.df_wide_numeric.reindex(self.validation_test_indexes[0])
        if not self.weighted:
            current_weights = {x: 1 for x in df_train.columns}
        else:
            current_weights = {x: self.weights[x] for x in df_train.columns}
        # ensemble_templates = pd.DataFrame()
        result = TemplateEvalObject()
        max_gens = self.initial_results.model_results['Generation'].max()
        for gen in range(max_gens + 1):
            mods = (
                self.initial_results.model_results[
                    (self.initial_results.model_results['Generation'] <= gen)
                    & (self.initial_results.model_results['ValidationRound'] == 0)
                    & (self.initial_results.model_results['Ensemble'] == 0)
                ]['ID']
                .unique()
                .tolist()
            )
            # note this is using validation results, but filtered by models from that gen
            score_per_series = generate_score_per_series(
                self.initial_results,
                metric_weighting=self.metric_weighting,
                total_validations=(self.num_validations + 1),
                models_to_use=mods,
            )
            ens_templates = HorizontalTemplateGenerator(
                score_per_series,
                model_results=self.initial_results.model_results,
                forecast_length=self.forecast_length,
                ensemble=self.ensemble,
                subset_flag=self.subset_flag,
                only_specified=True,
            )
            reg_tr = (
                self.future_regressor_train.reindex(index=df_train.index)
                if self.future_regressor_train is not None
                else None
            )
            reg_fc = (
                self.future_regressor_train.reindex(index=df_test.index)
                if self.future_regressor_train is not None
                else None
            )
            result.concat(
                TemplateWizard(
                    ens_templates,
                    df_train,
                    df_test,
                    weights=current_weights,
                    model_count=0,
                    current_generation=gen,
                    forecast_length=self.forecast_length,
                    frequency=self.used_frequency,
                    prediction_interval=self.prediction_interval,
                    ensemble=self.ensemble,
                    no_negatives=self.no_negatives,
                    constraint=self.constraint,
                    future_regressor_train=reg_tr,
                    future_regressor_forecast=reg_fc,
                    holiday_country=self.holiday_country,
                    startTimeStamps=self.startTimeStamps,
                    template_cols=self.template_cols,
                    model_interrupt=self.model_interrupt,
                    grouping_ids=self.grouping_ids,
                    max_generations="Horizontal Ensembles",
                    random_seed=self.random_seed,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    traceback=self.traceback,
                    current_model_file=self.current_model_file,
                    force_gc=self.force_gc,
                )
            )
        # this handles missing runtime information, which really shouldn't be missing
        if 'TotalRuntime' not in result.model_results.columns:
            result.model_results = pd.Timedelta(seconds=1)
        result.model_results['Score'] = generate_score(
            result.model_results,
            metric_weighting=self.metric_weighting,
            prediction_interval=self.prediction_interval,
        )
        return result

    def plot_horizontal_per_generation(
        self,
        title="Horizontal Ensemble Accuracy Gain (first eval sample only)",
        **kwargs,
    ):
        """Plot how well the horizontal ensembles would do after each new generation. Slow."""
        if (
            self.best_model_ensemble == 2
            and str(self.best_model_params.get('model_name', "Mosaic")).lower()
            != "mosaic"
        ):
            self.horizontal_per_generation().model_results['Score'].plot(
                ylabel="Lowest Score", xlabel="Generation", title=title, **kwargs
            )
        else:
            print("not a valid horizontal model for plot_horizontal_per_generation")

    def back_forecast(
        self, series=None, n_splits: int = "auto", tail: int = "auto", verbose: int = 0
    ):
        """Create forecasts for the historical training data, ie. backcast or back forecast. OUT OF SAMPLE

        This actually forecasts on historical data, these are not fit model values as are often returned by other packages.
        As such, this will be slower, but more representative of real world model performance.
        There may be jumps in data between chunks.

        Args are same as for model_forecast except...
        n_splits(int): how many pieces to split data into. Pass 2 for fastest, or "auto" for best accuracy
        series (str): if to run on only one column, pass column name. Faster than full.
        tail (int): df.tail() of the dataset, back_forecast is only run on n most recent observations.
            which points at eval_periods of lower-level back_forecast function

        Returns a standard prediction object (access .forecast, .lower_forecast, .upper_forecast)
        """
        if self.best_model.empty:
            raise ValueError("No best_model. AutoTS .fit() needs to be run.")
        if series is not None and (
            self.best_model_name in no_shared or self.best_model_ensemble == 2
        ):
            input_df = pd.DataFrame(self.df_wide_numeric[series])
        else:
            input_df = self.df_wide_numeric
        eval_periods = None
        if tail is not None:
            if tail == "auto":
                eval_periods = self.forecast_length * (self.num_validations + 1)
            else:
                eval_periods = tail
        result = back_forecast(
            df=input_df,
            model_name=self.best_model_name,
            model_param_dict=self.best_model_params.copy(),
            model_transform_dict=self.best_model_transformation_params,
            future_regressor_train=self.future_regressor_train,
            n_splits=n_splits,
            forecast_length=self.forecast_length,
            frequency=self.used_frequency,
            prediction_interval=self.prediction_interval,
            no_negatives=self.no_negatives,
            constraint=self.constraint,
            holiday_country=self.holiday_country,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            verbose=verbose,
            eval_periods=eval_periods,
        )
        return result

    def horizontal_to_df(self):
        """helper function for plotting."""
        if self.best_model.empty:
            raise ValueError("No best_model. AutoTS .fit() needs to be run.")
        if self.best_model['Ensemble'].iloc[0] != 2:
            raise ValueError("Only works on horizontal ensemble type models.")
        ModelParameters = self.best_model_params.copy()
        series = ModelParameters['series']
        series = pd.DataFrame.from_dict(series, orient="index").reset_index(drop=False)
        if series.shape[1] > 2:
            # for mosaic style ensembles, choose the mode model id
            series.set_index(series.columns[0], inplace=True)
            series = series.mode(axis=1)[0].to_frame().reset_index(drop=False)
        series.columns = ['Series', 'ID']
        results = pd.Series(
            {
                x: self.best_model_params['models'][x]['Model']
                for x in self.best_model_params['models'].keys()
            }
        )
        results.name = "Model"
        series = series.merge(results, left_on="ID", right_index=True)
        # series = series.merge(self.results()[['ID', "Model"]].drop_duplicates(), on="ID")  # old
        series = series.merge(
            self.df_wide_numeric.std().to_frame(), right_index=True, left_on="Series"
        )
        series = series.merge(
            self.df_wide_numeric.mean().to_frame(), right_index=True, left_on="Series"
        )
        series.columns = ["Series", "ID", 'Model', "Volatility", "Mean"]
        series['Transformers'] = series['ID'].copy()
        series['FillNA'] = series['ID'].copy()
        lookup = {}
        na_lookup = {}
        for k, v in ModelParameters['models'].items():
            try:
                trans_params = json.loads(v.get('TransformationParameters', '{}'))
                lookup[k] = ",".join(trans_params.get('transformations', {}).values())
                na_lookup[k] = trans_params.get('fillna', '')
            except Exception:
                lookup[k] = "None"
                na_lookup[k] = "None"
        series['Transformers'] = (
            series['Transformers'].replace(lookup).replace("", "None")
        )
        series['FillNA'] = series['FillNA'].replace(na_lookup).replace("", "None")
        return series

    def mosaic_to_df(self):
        """Helper function to create a readable df of models in mosaic."""
        if self.best_model.empty:
            raise ValueError("No best_model. AutoTS .fit() needs to be run.")
        if self.best_model_ensemble != 2:
            raise ValueError("Only works on horizontal ensemble type models.")
        ModelParameters = self.best_model_params.copy()
        if str(ModelParameters['model_name']).lower() != 'mosaic':
            raise ValueError("Only works on mosaic ensembles.")
        series = pd.DataFrame.from_dict(ModelParameters['series'])
        lookup = {k: v['Model'] for k, v in ModelParameters['models'].items()}
        return series.replace(lookup)

    def plot_horizontal(
        self, max_series: int = 20, title="Model Types Chosen by Series", **kwargs
    ):
        """Simple plot to visualize assigned series: models.

        Note that for 'mosaic' ensembles, it only plots the type of the most common model_id for that series, or the first if all are mode.

        Args:
            max_series (int): max number of points to plot
            **kwargs passed to pandas.plot()
        """
        series = self.horizontal_to_df().copy()
        # remove some data to prevent overcrowding the graph, if necessary
        max_series = series.shape[0] if series.shape[0] < max_series else max_series
        series = series.sample(max_series, replace=False)
        # sklearn.preprocessing.normalizer also might work
        series[['log(Volatility)', 'log(Mean)']] = np.log1p(
            np.abs(series[['Volatility', 'Mean']])
        )
        sx = (
            series.set_index(['Model', 'log(Mean)'], append=True)
            .unstack('Model')['log(Volatility)']
            .reset_index(drop=True)
        )
        # plot
        return sx.plot(style='o', title=title, **kwargs)

    def plot_horizontal_transformers(
        self, method="transformers", color_list=None, **kwargs
    ):
        """Simple plot to visualize transformers used.
        Note this doesn't capture transformers nested in simple ensembles.

        Args:
            method (str): 'fillna' or 'transformers' - which to plot
            color_list = list of colors to *sample* for bar colors. Can be names or hex.
            **kwargs passed to pandas.plot()
        """
        series = self.horizontal_to_df()
        if str(method).lower() == "fillna":
            transformers = series['FillNA'].value_counts()
            title = "Most Frequently Chosen FillNA Method"
        else:
            transformers = pd.Series(
                ",".join(series['Transformers']).split(",")
            ).value_counts()
            title = "Most Frequently Chosen Preprocessing"
        if color_list is None:
            color_list = colors_list
        colors = random.sample(color_list, transformers.shape[0])
        # plot
        transformers.plot(kind='bar', color=colors, title=title, **kwargs)

    def plot_generation_loss(
        self, title="Single Model Accuracy Gain Over Generations", **kwargs
    ):
        """Plot improvement in accuracy over generations.
        Note: this is only "one size fits all" accuracy and
        doesn't account for the benefits seen for ensembling.

        Args:
            **kwargs passed to pd.DataFrame.plot()
        """
        for_gens = self.initial_results.model_results[
            (self.initial_results.model_results['ValidationRound'] == 0)
            & (self.initial_results.model_results['Ensemble'] < 1)
        ]
        for_gens.groupby("Generation")['Score'].min().cummin().plot(
            ylabel="Lowest Score", title=title, **kwargs
        )

    def plot_backforecast(
        self,
        series=None,
        n_splits: int = "auto",
        start_date="auto",
        title=None,
        alpha=0.25,
        facecolor="black",
        loc="upper left",
        **kwargs,
    ):
        """Plot the historical data and fit forecast on historic. Out of sample in chunks = forecast_length by default.

        Args:
            series (str or list): column names of time series
            n_splits (int or str): "auto", number > 2, higher more accurate but slower
            start_date (datetime.datetime): or "auto"
            title (str)
            **kwargs passed to pd.DataFrame.plot()
        """
        if series is None:
            series = random.choice(self.df_wide_numeric.columns)
        if title is None:
            title = f"Out of Sample Back Forecasts for {str(series)[0:40]}"
        tail = None
        if start_date is not None:
            if start_date == "auto":
                tail = self.forecast_length * (self.num_validations + 1)
                start_date = self.df_wide_numeric.index[-tail]
            else:
                tail = len(
                    self.df_wide_numeric.index[self.df_wide_numeric.index >= start_date]
                )
                if tail == len(self.df_wide_numeric.index):
                    tail = None
        bd = self.back_forecast(series=series, n_splits=n_splits, verbose=0, tail=tail)
        b_df = pd.DataFrame(bd.forecast[series]).rename(
            columns=lambda x: str(x) + "_forecast"
        )
        b_df_up = pd.DataFrame(bd.upper_forecast[series]).rename(
            columns=lambda x: str(x) + "_upper_forecast"
        )
        b_df_low = pd.DataFrame(bd.lower_forecast[series]).rename(
            columns=lambda x: str(x) + "_lower_forecast"
        )
        plot_df = pd.concat(
            [pd.DataFrame(self.df_wide_numeric[series]), b_df, b_df_up, b_df_low],
            axis=1,
        )
        if start_date is not None:
            plot_df = plot_df[plot_df.index >= start_date]
        plot_df = remove_leading_zeros(plot_df)
        try:
            import matplotlib.pyplot as plt

            ax = plt.subplot()
            ax.set_title(title)
            ax.fill_between(
                plot_df.index,
                plot_df.iloc[:, 3],
                plot_df.iloc[:, 2],
                facecolor=facecolor,
                alpha=alpha,
                interpolate=True,
                label=f"{self.prediction_interval * 100}% upper/lower forecast",
            )
            ax.plot(plot_df.index, plot_df.iloc[:, 1], label="forecast", **kwargs)
            ax.plot(plot_df.index, plot_df.iloc[:, 0], label="actuals")
            ax.legend(loc=loc)
            for label in ax.get_xticklabels():
                label.set_ha("right")
                label.set_rotation(45)
            return ax
        except Exception:
            plot_df.plot(title=title, **kwargs)

    def plot_back_forecast(self, **kwargs):
        return self.plot_backforecast(**kwargs)

    def _validation_forecasts(self, models=None, compare_horizontal=False):
        if models is None:
            if self.best_model_non_horizontal is not None and compare_horizontal:
                validation_template = pd.concat(
                    [self.best_model, self.best_model_non_horizontal], axis=0
                )
            else:
                validation_template = self.best_model
        elif isinstance(models, str):
            val_results = self.results()
            validation_template = val_results[val_results['ID'].isin([models])][
                self.template_cols
            ].drop_duplicates()
        elif isinstance(models, list):
            validation_template = val_results[val_results['ID'].isin(models)][
                self.template_cols
            ].drop_duplicates()
        elif isinstance(models, pd.DataFrame):
            validation_template = models
        # the duplicated check is only for exact match, and could be improved
        duplicated = False
        if self.validation_forecasts_template is not None:
            if self.validation_forecasts_template.equals(validation_template):
                duplicated = True
        if not duplicated:
            self.validation_forecast_cuts = []
            self.validation_forecast_cuts_ends = []
            # self.validation_forecasts = {}
            for val in range(len(self.validation_indexes)):
                val_df_train, val_df_test = simple_train_test_split(
                    self.df_wide_numeric.reindex(self.validation_indexes[val]),
                    forecast_length=self.forecast_length,
                    min_allowed_train_percent=self.min_allowed_train_percent,
                    verbose=self.verbose,
                )
                sec_idx = val_df_test.index
                self.validation_forecast_cuts.append(sec_idx[0])
                self.validation_forecast_cuts_ends.append(sec_idx[-1])
                try:
                    train_reg = self.future_regressor_train.reindex(val_df_train.index)
                    fut_reg = self.future_regressor_train.reindex(sec_idx)
                except Exception:
                    train_reg = None
                    fut_reg = None
                for index, row in validation_template.iterrows():
                    df_forecast = self._predict(
                        forecast_length=self.forecast_length,
                        prediction_interval=self.prediction_interval,
                        future_regressor=fut_reg,
                        fail_on_forecast_nan=False,
                        verbose=self.verbose,
                        model_name=row["Model"],
                        model_params=row["ModelParameters"],
                        model_transformation_params=row["TransformationParameters"],
                        df_wide_numeric=val_df_train,
                        future_regressor_train=train_reg,
                        bypass_save=True,
                    )
                    idz = create_model_id(
                        row["Model"],
                        row["ModelParameters"],
                        row["TransformationParameters"],
                    )
                    if idz == self.best_model_id:
                        idz = "chosen_model"
                    self.validation_forecasts[str(val) + "_" + str(idz)] = df_forecast
        else:
            if self.verbose > 0:
                print("using stored results for plot_validations")
        self.validation_forecasts_template = validation_template

    def retrieve_validation_forecasts(
        self,
        models=None,
        compare_horizontal=False,
        id_name="SeriesID",
        value_name="Value",
        interval_name='PredictionInterval',
    ):
        self._validation_forecasts(models=models, compare_horizontal=compare_horizontal)
        needed_mods = self.validation_forecasts_template['ID'].tolist()
        df_list = []
        for x in self.validation_forecasts.keys():
            mname = x.split("_")[1]
            if mname == "chosen" or mname in needed_mods:
                new_df = self.validation_forecasts[x].long_form_results(
                    id_name=id_name,
                    value_name=value_name,
                    interval_name=interval_name,
                )
                new_df['ValidationRound'] = x.split("_")[0]
                df_list.append(new_df)
        return pd.concat(df_list, sort=True, axis=0)

    def plot_validations(
        self,
        df_wide=None,
        models=None,
        series=None,
        title=None,
        start_date="auto",
        end_date="auto",
        subset=None,
        compare_horizontal=False,
        colors=None,
        include_bounds=True,
        alpha=0.35,
        start_color="darkred",
        end_color="#A2AD9C",
        **kwargs,
    ):
        """Similar to plot_backforecast but using the model's validation segments specifically. Must reforecast.
        Saves results to self.validation_forecasts and caches. Set that to None to force rerun otherwise it uses stored (when models is the same).
        'chosen' refers to best_model_id, the model chosen to run for predict
        Validation sections may overlap (depending on method) which can confuse graph readers.

        Args:
            models (list): list, str, df or None, models to compare (IDs unless df of model params)
            series (str): time series to graph
            title (str): graph title
            start_date (str): 'auto' or datetime, place to begin graph, None for full
            end_date (str): 'auto' or datetime, end of graph x axis
            subset (str): overrides series, shows either 'best' or 'worst'
            compare_horizontal (bool): if True, plot horizontal ensemble versus best non-horizontal model, when available
            include_bounds (bool): if True (default) include the upper/lower forecast bounds
            start_color (str): color of vline for val start marker, None to remove vline
            end_color (str): color of vline for val end marker, None to remove vline
        """
        if df_wide is None:
            df_wide = self.df_wide_numeric
        # choose which series to plot
        if series is None:
            if subset is None:
                series = random.choice(df_wide.columns)
            else:
                scores = self.best_model_per_series_mape().index.tolist()
                scores = [x for x in scores if "_lltmicro" not in x]
                mapes = self.best_model_per_series_score().index.tolist()
                mapes = [x for x in mapes if "_lltmicro" not in x]
                if str(subset).lower() == "best":
                    series = mapes[-1]
                elif str(subset).lower() == "best score":
                    series = scores[-1]
                elif str(subset).lower() == "worst":
                    series = mapes[0]
                elif str(subset).lower() == "worst score":
                    series = scores[0]
                else:
                    raise ValueError(
                        "plot_validations arg subset must be None, 'best' or 'worst'"
                    )
        # run the forecasts on the past validations
        self._validation_forecasts(models=models, compare_horizontal=compare_horizontal)
        if not compare_horizontal and colors is None:
            colors = {
                'actuals': '#AFDBF5',
                'chosen': '#4D4DFF',
                'chosen_lower': '#A7AFB2',
                'chosen_upper': '#A7AFB2',
            }
        needed_mods = self.validation_forecasts_template['ID'].tolist()
        df_list = []
        for x in self.validation_forecasts.keys():
            mname = x.split("_")[1]
            if mname == "chosen" or mname in needed_mods:
                new_df = pd.DataFrame(index=df_wide.index)
                new_df[mname] = self.validation_forecasts[x].forecast[series]
                new_df[mname + "_" + "upper"] = self.validation_forecasts[
                    x
                ].upper_forecast[series]
                new_df[mname + "_" + "lower"] = self.validation_forecasts[
                    x
                ].lower_forecast[series]
                df_list.append(new_df)
        plot_df = pd.concat(df_list, sort=True, axis=0)
        # self.val_plot_df = plot_df.copy()
        plot_df = plot_df.groupby(level=0).last()
        plot_df = (
            df_wide[series]
            .rename("actuals")
            .to_frame()
            .merge(plot_df, left_index=True, right_index=True, how="left")
        )
        if not include_bounds:
            colb = [
                x for x in plot_df.columns if "_lower" not in x and "_upper" not in x
            ]
            plot_df = plot_df[colb]
        if start_date == "auto":
            frequency_numeric = [x for x in self.used_frequency if x.isdigit()]
            if not frequency_numeric:
                used_freq = "1" + self.used_frequency
            else:
                used_freq = self.used_frequency
            start_date = plot_df[plot_df.columns.difference(['actuals'])].dropna(
                how='all', axis=0
            ).index.min() - (
                freq_to_timedelta(used_freq) * int(self.forecast_length * 3)
            )
        if end_date == "auto":
            end_date = plot_df[plot_df.columns.difference(['actuals'])].dropna(
                how='all', axis=0
            ).index.max() + pd.Timedelta(days=7)
        if start_date is not None:
            plot_df = plot_df[plot_df.index >= start_date]
        if end_date is not None:
            plot_df = plot_df[plot_df.index <= end_date]
        # try to make visible (if not quite right) on a short forecast
        if self.forecast_length == 1:
            if plot_df.shape[0] > 3:
                plot_df.loc[:, 'chosen'] = plot_df['chosen'].fillna(
                    method='bfill', limit=1
                )
        # set Title
        if title is None:
            if subset is not None:
                if "score" in str(subset).lower():
                    title = f"Validation Forecasts for {subset} Tested Series {series}"
                else:
                    title = (
                        f"Validation Forecasts for {subset} Tested MAPE Series {series}"
                    )
            else:
                title = f"Validation Forecasts for {series}"
        # actual plotting section
        if colors is not None:
            # this will need to change is users are allowed to input colors
            ax = plot_df[['actuals', 'chosen']].plot(
                title=title, color=colors, **kwargs
            )
            if include_bounds:
                ax.fill_between(
                    plot_df.index,
                    plot_df['chosen_upper'],
                    plot_df['chosen_lower'],
                    alpha=alpha,
                    color="#A5ADAF",
                )
        else:
            ax = plot_df.plot(title=title, **kwargs)
        if end_color is not None:
            ax.vlines(
                x=self.validation_forecast_cuts_ends,
                ls='-.',
                lw=1,
                colors='#D3D3D3',
                ymin=plot_df.min().min(),
                ymax=plot_df.max().max(),
            )
        if start_color is not None:
            ax.vlines(
                x=self.validation_forecast_cuts,
                ls='--',
                lw=1,
                colors='darkred',
                ymin=plot_df.min().min(),
                ymax=plot_df.max().max(),
            )
        return ax

    def list_failed_model_types(self):
        """Return a list of model types (ie ETS, LastValueNaive) that failed.
        If all had at least one success, then return an empty list.
        """
        if self.best_model.empty:
            raise ValueError("No best_model. AutoTS .fit() needs to be run.")
        temp = self.initial_results.model_results[['Model', 'Exceptions']].copy()
        temp['Exceptions'] = temp['Exceptions'].isnull().astype(int)
        temp = temp.groupby("Model")['Exceptions'].sum()
        return temp[temp <= 0].index.to_list()

    def best_model_per_series_mape(self):
        best_model_per_series_mae = self.initial_results.per_series_mae[
            self.initial_results.per_series_mae.index == self.best_model_id
        ].mean(axis=0)
        # obsess over avoiding division by zero
        scaler = self.df_wide_numeric.mean(axis=0)
        scaler[scaler == 0] == np.nan
        scaler = scaler.fillna(self.df_wide_numeric.max(axis=0))
        scaler[scaler == 0] == 1
        temp = (
            ((best_model_per_series_mae / scaler) * 100)
            .round(2)
            .sort_values(ascending=False)
        )
        temp.name = 'MAPE'
        temp.index.name = 'Series'
        return temp

    def plot_per_series_mape(
        self,
        title: str = None,
        max_series: int = 10,
        max_name_chars: int = 25,
        color: str = "#ff9912",
        figsize=(12, 4),
        kind: str = "bar",
        **kwargs,
    ):
        """Plot which series are contributing most to SMAPE of final model. Avg of validations for best_model

        Args:
            title (str): plot title
            max_series (int): max number of series to show on plot (sorted)
            max_name_chars (str): if horizontal ensemble, will chop series names to this
            color (str): hex or name of color of plot
            figsize (tuple): passed through to plot axis
            kind (str): bar or pie
            **kwargs passed to pandas.plot()
        """
        if self.best_model.empty:
            raise ValueError("No best_model. AutoTS .fit() needs to be run.")
        if title is None:
            title = f"Top {max_series} Series Contributing MAPE Error"

        temp = self.best_model_per_series_mape().reset_index().head(max_series)

        if self.best_model_ensemble == 2:
            series = self.horizontal_to_df()
            temp = temp.merge(series, on='Series')
            temp['Series'] = (
                temp['Series'].str.slice(0, max_name_chars) + " (" + temp["Model"] + ")"
            )

        if kind == "pie":
            return temp.set_index("Series").plot(
                y="MAPE",
                kind="pie",
                title=title,
                figsize=figsize,
                legend=False,
                **kwargs,
            )
        else:
            return temp.plot(
                x="Series",
                y="MAPE",
                kind=kind,
                title=title,
                color=color,
                figsize=figsize,
                **kwargs,
            )

    def plot_per_series_smape(
        self,
        title: str = None,
        max_series: int = 10,
        max_name_chars: int = 25,
        color: str = "#ff9912",
        figsize=(12, 4),
        kind: str = "bar",
        **kwargs,
    ):
        """To be backwards compatible, not necessarily maintained, plot_per_series_mape is to be preferred."""
        print("please switch to plot_per_series_mape")
        return self.plot_per_series_mape(
            title=title,
            max_series=max_series,
            max_name_chars=max_name_chars,
            color=color,
            figsize=figsize,
            kind=kind,
            **kwargs,
        )

    def best_model_per_series_score(self):
        return (
            generate_score_per_series(
                self.initial_results,
                metric_weighting=self.metric_weighting,
                total_validations=(self.num_validations + 1),
                models_to_use=[self.best_model_id],
            )
            .mean(axis=0)
            .sort_values(ascending=False)
            .round(3)
        )

    def plot_per_series_error(
        self,
        title: str = "Top Series Contributing Score Error",
        max_series: int = 10,
        max_name_chars: int = 25,
        color: str = "#ff9912",
        figsize=(12, 4),
        kind: str = "bar",
        upper_clip: float = 1000,
        **kwargs,
    ):
        """Plot which series are contributing most to error (Score) of final model. Avg of validations for best_model

        Args:
            title (str): plot title
            max_series (int): max number of series to show on plot (sorted)
            max_name_chars (str): if horizontal ensemble, will chop series names to this
            color (str): hex or name of color of plot
            figsize (tuple): passed through to plot axis
            kind (str): bar or pie
            upper_clip (float): set max error show to this value, to prevent unnecessary distortion
            **kwargs passed to pandas.plot()
        """
        if self.best_model.empty:
            raise ValueError("No best_model. AutoTS .fit() needs to be run.")
        # best_model_per = self.initial_results.per_series_mae[self.initial_results.per_series_mae.index == self.best_model_id]
        best_model_per = self.best_model_per_series_score().head(max_series)
        temp = best_model_per.reset_index()
        temp.columns = ["Series", "Error"]
        temp["Error"] = temp["Error"].clip(upper=upper_clip, lower=0)
        if self.best_model["Ensemble"].iloc[0] == 2:
            series = self.horizontal_to_df()
            temp = temp.merge(series, on='Series')
            temp['Series'] = (
                temp['Series'].str.slice(0, max_name_chars) + " (" + temp["Model"] + ")"
            )

        if kind == "pie":
            temp.set_index("Series").plot(
                y="Error", kind="pie", title=title, figsize=figsize, **kwargs
            )
        else:
            temp.plot(
                x="Series",
                y="Error",
                kind=kind,
                title=title,
                color=color,
                figsize=figsize,
                **kwargs,
            )

    def plot_horizontal_model_count(
        self,
        color_list=None,
        top_n: int = 20,
        title="Most Frequently Chosen Models",
        **kwargs,
    ):
        """Plots most common models. Does not factor in nested in non-horizontal Ensembles."""
        if self.best_model.empty:
            raise ValueError("AutoTS not yet fit.")
        elif self.best_model_ensemble != 2:
            raise ValueError("this plot only works on horizontal-style ensembles.")

        if str(self.best_model_params.get('model_name', None)).lower() == "mosaic":
            series = self.mosaic_to_df()
            transformers = series.stack().value_counts()
        else:
            series = self.horizontal_to_df()
            transformers = series['Model'].value_counts().iloc[0:top_n]

        if color_list is None:
            color_list = colors_list
        colors = random.sample(color_list, transformers.shape[0])
        # plot
        transformers.plot(kind='bar', color=colors, title=title, **kwargs)

    def get_metric_corr(self, percent_best=0.1):
        """Returns a dataframe of correlation among evaluation metrics across evaluations.

        Args:
            percent_best (float): percent (ie 0.1 for 10%) of models to use, best by score first
        """
        res = self.initial_results.model_results
        res = res[res['Exceptions'].isnull()]
        # correlation is much more interesting among the top models than among the full trials
        res = res.loc[
            res.sort_values("Score").index[0 : int(res.shape[0] * percent_best) + 1]
        ]
        metrics = res.select_dtypes("number")
        metrics = metrics[[x for x in metrics.columns if "weighted" not in x]]
        metrics = metrics.drop(
            columns=[
                'Ensemble',
                'Runs',
                'TransformationRuntime',
                'FitRuntime',
                'PredictRuntime',
                'Generation',
                'ValidationRound',
                'PostMemoryPercent',
                'TotalRuntimeSeconds',
            ],
            errors='ignore',
        )
        metrics = (metrics) / metrics.std()

        return metrics.corr()

    def plot_metric_corr(self, cols=None, percent_best=0.1):
        """Plot correlation in results among metrics.
        The metrics that are highly correlated are those that mostly the unscaled ones

        Args:
            cols (list): strings of columns to show, 'all' for all
            percent_best (float): percent (ie 0.1 for 10%) of models to use, best by score first
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        self.metric_corr = self.get_metric_corr(percent_best=percent_best)

        if cols is None:
            mostly_one = (self.metric_corr.abs() == 1).sum() == (
                self.metric_corr.abs() == 1
            ).sum().max()
            cols = self.metric_corr[~mostly_one].abs().sum().nlargest(15).index.tolist()
            if len(cols) < 15:
                cols.extend(
                    self.metric_corr[mostly_one].index[0 : 15 - len(cols)].tolist()
                )
        elif cols == 'all':
            cols = self.metric_corr.columns

        if len(cols) <= 2:
            correlation_matrix = self.metric_corr.loc[cols]
        else:
            correlation_matrix = self.metric_corr[cols].loc[cols]
        # Create a mask for the upper triangle to hide redundant information
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Set up the matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(16, 12))

        # Generate a diverging colormap
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        # Create the correlogram using a heatmap
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7},
        )
        sns.set_style("whitegrid")  # Add a grid for clarity

        # Add a title
        plt.title("Correlogram of Metric Correlations from Optimized Forecasts")
        return ax

    def diagnose_params(self, target='runtime', waterfall_plots=True):
        """Attempt to explain params causing measured outcomes using shap and linear regression coefficients.

        Args:
            target (str): runtime, smape, mae, oda, or exception, the measured outcome to correlate parameters with
            waterfall_plots (bool): whether to show waterfall SHAP plots
        """

        from autots.tools.transform import transformer_dict
        from sklearn.linear_model import Lasso, ElasticNet
        from sklearn.preprocessing import StandardScaler

        initial_results = self.results()
        all_trans = list(transformer_dict.keys())
        all_trans = [x for x in all_trans if x is not None]
        master_df = []
        res = []
        mas_trans = []
        for x in initial_results['Model'].unique():
            if x not in ["Ensemble"]:
                set_mod = initial_results[initial_results['Model'] == x]
                # pull all transformations
                for idx, row in set_mod.iterrows():
                    t = pd.Categorical(
                        list(
                            set(
                                json.loads(row['TransformationParameters'])[
                                    'transformations'
                                ].values()
                            )
                        ),
                        categories=all_trans,
                    )
                    mas_trans.append(
                        pd.get_dummies(t).max(axis=0).to_frame().transpose()
                    )
                    y = pd.json_normalize(json.loads(row["ModelParameters"]))
                    y.index = [row['ID']]
                    y[
                        'Model'
                    ] = x  # might need to remove this and do analysis independently for each
                    res.append(
                        pd.DataFrame(
                            {
                                'runtime': row['TotalRuntimeSeconds'],
                                'smape': row['smape'],
                                'mae': row['mae'],
                                'exception': int(not pd.isnull(row['Exceptions'])),
                                'oda': row['oda'],
                            },
                            index=[row['ID']],
                        )
                    )
                    master_df.append(y)
        master_df = pd.concat(master_df, axis=0)
        mas_trans = pd.concat(mas_trans, axis=0)
        mas_trans.index = master_df.index
        res = pd.concat(res, axis=0)
        self.lasso_X = pd.concat(
            [
                pd.get_dummies(master_df.astype(str)),
                mas_trans.rename(columns=lambda x: "Transformer_" + x),
            ],
            axis=1,
        )
        self.lasso_X['intercept'] = 1

        # target = 'runtime'
        y = res[target]
        # y = y.dropna(how='any')
        y_reset = y.reset_index()
        y_drop = y_reset.index[~y_reset.reset_index().isnull().any(axis=1)]
        y = y.iloc[y_drop]
        # print(f"y shape {y.shape} and index {y.index[0:10]}")
        # Standardize the features
        scaler = StandardScaler()
        X_train = self.lasso_X.fillna(0)
        # print(f"x shape {X_train.shape} and index {X_train.index[0:10]}")
        # X_train = X_train[X_train.index.isin(y.index)]
        X_train = X_train.iloc[y_drop]
        X_train_scaled = scaler.fit_transform(X_train)
        feature_names = self.lasso_X.columns.tolist()

        preprocess = True
        try:
            from flaml import AutoML

            # Initialize FLAML AutoML instance
            automl = AutoML()

            # Specify the task as regression and the estimator as xgboost
            if target in ['exception']:
                settings = {
                    "time_budget": 60,  # in seconds
                    "metric": 'accuracy',
                    "task": 'classification',
                    "estimator_list": ['xgboost'],
                    "verbose": 1,
                }
            else:
                settings = {
                    "time_budget": 60,  # in seconds
                    "metric": 'mae',
                    "task": 'regression',
                    "estimator_list": [
                        'xgboost'
                    ],  # specify xgboost as the estimator, extra_tree
                    "verbose": 1,
                    # "preprocess": preprocess,
                }
            # Train with FLAML
            automl.fit(X_train, y, **settings)
            print("##########################################################")
            print(
                f"FLAML loss: {automl.best_loss:.3f} vs avg runtime {res['runtime'].mean():.3f}"
            )
            shap_X = automl._preprocess(X_train)
            feature_names = shap_X.columns
            bst = automl.model.estimator
        except Exception as e:
            print(repr(e))
            from sklearn.ensemble import ExtraTreesRegressor

            # replace with most likely present scikit-learn
            # bst = xgb.XGBRegressor(objective ='reg:linear', n_estimators=10, seed=123)
            bst = ExtraTreesRegressor(
                max_features=0.63857,
                max_leaf_nodes=81,
                n_estimators=5,
                n_jobs=-1,
                random_state=12032022,
            )
            bst = bst.fit(X_train, y)
            shap_X = X_train

        try:
            import shap
            import matplotlib.pyplot as plt

            # Compute SHAP values
            explainer = shap.Explainer(bst, feature_names=feature_names)
            shap_values = explainer(shap_X)
            # Plot summary plot
            shap.summary_plot(
                shap_values,
                shap_X,
                feature_names=feature_names,
                title=f"SHAP Summary for {target}",
            )
            # Plot SHAP values for a single prediction
            try:
                if waterfall_plots:
                    # show impact on biggest or best, as relevant
                    lvalues = res.reset_index()[target].nlargest(n=2).index
                    svalues = res.reset_index()[target].nsmallest(n=2).index
                    if target in ['runtime', 'oda', 'exception']:
                        val1 = lvalues[0]
                        val2 = lvalues[1]
                        desc = "largest"
                        val3 = svalues[0]
                    elif target is not None:
                        val1 = svalues[0]
                        val2 = svalues[1]
                        desc = "smallest"
                        val3 = lvalues[0]
                    else:
                        val1 = 1
                        val2 = 2
                        val3 = -4
                        desc = "fixed"
                    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
                    plt.title(f"SHAP Waterfall for {target} and row 1")
                    plt.show()
                    print(f"val1 {val1} and val2 {val2}")
                    shap.plots.waterfall(shap_values[val1], max_display=15, show=False)
                    plt.title(f"SHAP Waterfall for {target} and row {val1} ({desc})")
                    plt.show()
                    shap.plots.waterfall(shap_values[val2], max_display=15, show=False)
                    plt.title(f"SHAP Waterfall for {target} and row {val2} ({desc})")
                    plt.show()
                    shap.plots.waterfall(shap_values[-1], max_display=10, show=False)
                    plt.title(f"SHAP Waterfall for {target} and row -1")
                    plt.show()
                    shap.plots.waterfall(shap_values[val3], max_display=10, show=False)
                    plt.title(f"SHAP Waterfall for {target} and row {val3}")
                    plt.show()
                    shap.plots.waterfall(shap_values[-8], max_display=10, show=False)
                    plt.title(f"SHAP Waterfall for {target} and row -8")
                    plt.show()
            except Exception as e:
                print(repr(e))

            # Compute the mean absolute SHAP value for each feature
            mean_shap_values = np.abs(shap_values.values).mean(axis=0)
            # Pair the feature names with their mean absolute SHAP values in a list of tuples
            # feature_shap_pairs = list(zip(feature_names, mean_shap_values))
            # Sort the list of tuples by the SHAP values from smallest to largest
            # sorted_feature_shap_pairs = sorted(feature_shap_pairs, key=lambda x: x[1])
            # Print the sorted pairs
            # print("Sorted Mean Absolute SHAP Values for Feature Importance:")
            # for feature, mean_shap in sorted_feature_shap_pairs[-10:]:
            #     print(f"{feature}: {mean_shap} impact (not direction)")

            # Compute the mean SHAP value for each feature
            mean_shap_values = shap_values.values.mean(axis=0)
            # Pair the feature names with their mean SHAP values in a list of tuples
            # feature_shap_pairs = list(zip(feature_names, mean_shap_values))
            # Sort the list of tuples by the SHAP values from smallest to largest
            # sorted_feature_shap_pairs = sorted(feature_shap_pairs, key=lambda x: x[1])
            # Print the sorted pairs
            # print("Sorted Mean SHAP Values for Feature Importance:")
            # for feature, mean_shap in sorted_feature_shap_pairs[-10:]:
            #     print(f"{feature}: {mean_shap}")
        except Exception as e:
            print(repr(e))
            mean_shap_values = 0

        # IF the outcome is a 0/1 flag
        if target in ['exception']:
            from sklearn.linear_model import LogisticRegression

            self.lasso = LogisticRegression(
                penalty='l1', solver='saga', C=10
            )  # C=10 is the inverse of alpha\
        # elif target == "smape":
        #     self.lasso = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=0)
        else:
            # Fit a Lasso regression model
            self.lasso = Lasso(alpha=0.01)
        self.lasso.fit(shap_X if preprocess else X_train_scaled, y)

        lasso_coef = self.lasso.coef_.flatten()
        # Pair the feature names with their coefficients in a list of tuples
        feature_coef_pairs = list(zip(feature_names, lasso_coef))
        # Sort the list of tuples by the coefficient values from smallest to largest
        sorted_feature_coef_pairs = sorted(feature_coef_pairs, key=lambda x: x[1])
        # Print the sorted pairs
        print("Sorted Lasso Coefficients for Feature Importance:")
        for feature, coef in sorted_feature_coef_pairs[-10:]:
            print(f"{feature}: {coef:.4f}")

        param_impact = pd.DataFrame(
            {"shap_value": mean_shap_values, "lasso_value": lasso_coef},
            index=feature_names,
        )
        # give two different approaches for runtime
        if target not in ['exception']:
            lasso2 = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=0)
            lasso2.fit(shap_X if preprocess else X_train_scaled, y)
            param_impact['elastic_value'] = lasso2.coef_.flatten()
        return param_impact.copy().rename(columns=lambda x: f"{target}_" + str(x))


colors_list = [
    '#FF00FF',
    '#7FFFD4',
    '#00FFFF',
    '#F5DEB3',
    '#FF6347',
    '#8B008B',
    '#696969',
    '#FFC0CB',
    '#C71585',
    '#008080',
    '#663399',
    '#32CD32',
    '#66CDAA',
    '#A9A9A9',
    '#2F4F4F',
    '#FFDEAD',
    '#800000',
    '#FDF5E6',
    '#F5F5F5',
    '#F0FFF0',
    '#87CEEB',
    '#A52A2A',
    '#90EE90',
    '#7FFF00',
    '#E9967A',
    '#1E90FF',
    '#FFF0F5',
    '#ADD8E6',
    '#008B8B',
    '#FFF5EE',
    '#00FA9A',
    '#9370DB',
    '#4682B4',
    '#006400',
    '#AFEEEE',
    '#CD853F',
    '#9400D3',
    '#EE82EE',
    '#00008B',
    '#4B0082',
    '#0403A7',
    "#000000",
]


def fake_regressor(
    df,
    forecast_length: int = 14,
    date_col: str = None,
    value_col: str = None,
    id_col: str = None,
    frequency: str = 'infer',
    aggfunc: str = 'first',
    drop_most_recent: int = 0,
    na_tolerance: float = 0.95,
    drop_data_older_than_periods: int = 100000,
    dimensions: int = 1,
    verbose: int = 0,
):
    """Create a fake regressor of random numbers for testing purposes."""

    if date_col is None and value_col is None:
        df_wide = pd.DataFrame(df)
        assert (
            type(df_wide.index) is pd.DatetimeIndex
        ), "df index is not pd.DatetimeIndex"
    else:
        df_wide = long_to_wide(
            df,
            date_col=date_col,
            value_col=value_col,
            id_col=id_col,
            aggfunc=aggfunc,
        )

    df_wide = df_cleanup(
        df_wide,
        frequency=frequency,
        na_tolerance=na_tolerance,
        drop_data_older_than_periods=drop_data_older_than_periods,
        aggfunc=aggfunc,
        drop_most_recent=drop_most_recent,
        verbose=verbose,
    )
    if frequency == 'infer':
        frequency = infer_frequency(df_wide)

    forecast_index = pd.date_range(
        freq=frequency, start=df_wide.index[-1], periods=forecast_length + 1
    )[1:]

    if dimensions <= 1:
        future_regressor_train = pd.Series(
            np.random.randint(0, 100, size=len(df_wide.index)), index=df_wide.index
        )
        future_regressor_forecast = pd.Series(
            np.random.randint(0, 100, size=(forecast_length)), index=forecast_index
        )
    else:
        future_regressor_train = pd.DataFrame(
            np.random.randint(0, 100, size=(len(df_wide.index), dimensions)),
            index=df_wide.index,
        )
        future_regressor_forecast = pd.DataFrame(
            np.random.randint(0, 100, size=(forecast_length, dimensions)),
            index=forecast_index,
        )
    return future_regressor_train, future_regressor_forecast


def error_correlations(all_result, result: str = 'corr'):
    """
    Onehot encode AutoTS result df and return df or correlation with errors.

    Args:
        all_results (pandas.DataFrame): AutoTS model_results df
        result (str): whether to return 'df', 'corr', 'poly corr' with errors
    """
    import json
    from sklearn.preprocessing import OneHotEncoder

    all_results = all_result.copy()
    all_results = all_results.drop_duplicates()
    all_results['ExceptionFlag'] = (~all_results['Exceptions'].isna()).astype(int)
    all_results = all_results[all_results['ExceptionFlag'] > 0]
    all_results = all_results.reset_index(drop=True)

    trans_df = all_results['TransformationParameters'].apply(json.loads)
    try:
        trans_df = pd.json_normalize(trans_df)  # .fillna(value='NaN')
    except Exception:
        trans_df = pd.io.json.json_normalize(trans_df)
    trans_cols1 = trans_df.columns
    trans_df = trans_df.astype(str).replace('nan', 'NaNZ')
    trans_transformer = OneHotEncoder(sparse=False).fit(trans_df)
    trans_df = pd.DataFrame(trans_transformer.transform(trans_df))
    trans_cols = np.array(
        [x1 + x2 for x1, x2 in zip(trans_cols1, trans_transformer.categories_)]
    )
    trans_cols = [item for sublist in trans_cols for item in sublist]
    trans_df.columns = trans_cols

    model_df = all_results['ModelParameters'].apply(json.loads)
    try:
        model_df = pd.json_normalize(model_df)  # .fillna(value='NaN')
    except Exception:
        model_df = pd.io.json.json_normalize(model_df)
    model_cols1 = model_df.columns
    model_df = model_df.astype(str).replace('nan', 'NaNZ')
    model_transformer = OneHotEncoder(sparse=False).fit(model_df)
    model_df = pd.DataFrame(model_transformer.transform(model_df))
    model_cols = np.array(
        [x1 + x2 for x1, x2 in zip(model_cols1, model_transformer.categories_)]
    )
    model_cols = [item for sublist in model_cols for item in sublist]
    model_df.columns = model_cols

    modelstr_df = all_results['Model']
    modelstr_transformer = OneHotEncoder(sparse=False).fit(
        modelstr_df.values.reshape(-1, 1)
    )
    modelstr_df = pd.DataFrame(
        modelstr_transformer.transform(modelstr_df.values.reshape(-1, 1))
    )
    modelstr_df.columns = modelstr_transformer.categories_[0]

    except_df = all_results['Exceptions'].copy()
    except_df = except_df.where(except_df.duplicated(), 'UniqueError')
    except_transformer = OneHotEncoder(sparse=False).fit(
        except_df.values.reshape(-1, 1)
    )
    except_df = pd.DataFrame(
        except_transformer.transform(except_df.values.reshape(-1, 1))
    )
    except_df.columns = except_transformer.categories_[0]

    test = pd.concat(
        [except_df, all_results[['ExceptionFlag']], modelstr_df, model_df, trans_df],
        axis=1,
    )

    if result == 'corr':
        test_corr = test.corr()[except_df.columns]
        return test_corr
    if result == 'poly corr':
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(interaction_only=True, include_bias=False)
        poly = poly.fit(test)
        col_names = poly.get_feature_names(input_features=test.columns)
        test = pd.DataFrame(poly.transform(test), columns=col_names)
        test_corr = test.corr()[except_df.columns]
        return test_corr
    elif result == 'df':
        return test
    else:
        raise ValueError("arg 'result' not recognized")
