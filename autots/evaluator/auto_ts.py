"""Higher-level backbone of auto time series modeling."""
import numpy as np
import pandas as pd
import random
import copy
import json
import sys

from autots.tools.shaping import (
    long_to_wide,
    df_cleanup,
    subset_series,
    simple_train_test_split,
    NumericTransformer,
)
from autots.evaluator.auto_model import (
    TemplateEvalObject,
    NewGeneticTemplate,
    RandomTemplate,
    TemplateWizard,
    unpack_ensemble_models,
    generate_score,
    PredictWitch,
    validation_aggregation,
)
from autots.models.ensemble import (
    EnsembleTemplateGenerator,
    HorizontalTemplateGenerator,
)
from autots.models.model_list import model_lists


class AutoTS(object):
    """Automate time series modeling using a genetic algorithm.

    Args:
        forecast_length (int): number of periods over which to evaluate forecast. Can be overriden later in .predict().
        frequency (str): 'infer' or a specific pandas datetime offset. Can be used to force rollup of data (ie daily input, but frequency 'M' will rollup to monthly).
        prediction_interval (float): 0-1, uncertainty range for upper and lower forecasts. Adjust range, but rarely matches actual containment.
        max_generations (int): number of genetic algorithms generations to run.
            More runs = longer runtime, generally better accuracy.
        no_negatives (bool): if True, all negative predictions are rounded up to 0.
        constraint (float): when not None, use this value * data st dev above max or below min for constraining forecast values. Applied to point forecast only, not upper/lower forecasts.
        ensemble (str): None or list or comma-separated string containing:
            'auto', 'simple', 'distance', 'horizontal-max', 'probabilistic-max', "hdist"
        initial_template (str): 'Random' - randomly generates starting template, 'General' uses template included in package, 'General+Random' - both of previous. Also can be overriden with self.import_template()
        random_seed (int): random seed allows (slightly) more consistent results.
        holiday_country (str): passed through to Holidays package for some models.
        subset (int): maximum number of series to evaluate at once. Useful to speed evaluation when many series are input.
        aggfunc (str): if data is to be rolled up to a higher frequency (daily -> monthly) or duplicate timestamps are included. Default 'first' removes duplicates, for rollup try 'mean' or np.sum.
            Beware numeric aggregations like 'mean' will not work with non-numeric inputs.
        na_tolerance (float): 0 to 1. Series are dropped if they have more than this percent NaN. 0.95 here would allow series containing up to 95% NaN values.
        metric_weighting (dict): weights to assign to metrics, effecting how the ranking score is generated.
        drop_most_recent (int): option to drop n most recent data points. Useful, say, for monthly sales data where the current (unfinished) month is included.
            occurs after any aggregration is applied, so will be whatever is specified by frequency, will drop n frequencies
        drop_data_older_than_periods (int): take only the n most recent timestamps
        model_list (list): str alias or list of names of model objects to use
        transformer_list (list): list of transformers to use, or dict of transformer:probability. Note this does not apply to initial templates.
        transformer_max_depth (int): maximum number of sequential transformers to generate for new Random Transformers. Fewer will be faster.
        num_validations (int): number of cross validations to perform. 0 for just train/test on final split.
        models_to_validate (int): top n models to pass through to cross validation. Or float in 0 to 1 as % of tried.
            0.99 is forced to 100% validation. 1 evaluates just 1 model.
            If horizontal or probabilistic ensemble, then additional min per_series models above the number here may be added to validation.
        max_per_model_class (int): of the models_to_validate what is the maximum to pass from any one model class/family.
        validation_method (str): 'even', 'backwards', or 'seasonal n' where n is an integer of seasonal
            'backwards' is better for recency and for shorter training sets
            'even' splits the data into equally-sized slices best for more consistent data
            'seasonal n' for example 'seasonal 364' would test all data on each previous year of the forecast_length that would immediately follow the training data.
        min_allowed_train_percent (float): percent of forecast length to allow as min training, else raises error.
            0.5 with a forecast length of 10 would mean 5 training points are mandated, for a total of 15 points.
            Useful in (unrecommended) cases where forecast_length > training length.
        remove_leading_zeroes (bool): replace leading zeroes with NaN. Useful in data where initial zeroes mean data collection hasn't started yet.
        model_interrupt (bool): if False, KeyboardInterrupts quit entire program.
            if True, KeyboardInterrupts attempt to only quit current model.
            if True, recommend use in conjunction with `verbose` > 0 and `result_file` in the event of accidental complete termination.
        verbose (int): setting to 0 or lower should reduce most output. Higher numbers give more output.
        n_jobs (int): Number of cores available to pass to parallel processing. A joblib context manager can be used instead (pass None in this case). Also 'auto'.

    Attributes:
        best_model (pandas.DataFrame): DataFrame containing template for the best ranked model
        regression_check (bool): If True, the best_model uses an input 'User' future_regressor
    """

    def __init__(
        self,
        forecast_length: int = 14,
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        max_generations: int = 20,
        no_negatives: bool = False,
        constraint: float = None,
        ensemble: str = 'auto',
        initial_template: str = 'General+Random',
        random_seed: int = 2020,
        holiday_country: str = 'US',
        subset: int = None,
        aggfunc: str = 'first',
        na_tolerance: float = 1,
        metric_weighting: dict = {
            'smape_weighting': 10,
            'mae_weighting': 2,
            'rmse_weighting': 2,
            'containment_weighting': 0,
            'runtime_weighting': 0,
            'spl_weighting': 1,
            'contour_weighting': 0,
        },
        drop_most_recent: int = 0,
        drop_data_older_than_periods: int = 100000,
        model_list: str = 'default',
        transformer_list: dict = "fast",
        transformer_max_depth: int = 6,
        num_validations: int = 2,
        models_to_validate: float = 0.15,
        max_per_model_class: int = None,
        validation_method: str = 'backwards',
        min_allowed_train_percent: float = 0.5,
        remove_leading_zeroes: bool = False,
        model_interrupt: bool = False,
        verbose: int = 1,
        n_jobs: int = None,
    ):
        assert forecast_length > 0, "forecast_length must be greater than 0"
        assert transformer_max_depth > 0, "transformer_max_depth must be greater than 0"
        self.forecast_length = int(abs(forecast_length))
        self.frequency = frequency
        self.aggfunc = aggfunc
        self.prediction_interval = prediction_interval
        self.no_negatives = no_negatives
        self.constraint = constraint
        self.random_seed = random_seed
        self.holiday_country = holiday_country
        if isinstance(ensemble, list):
            ensemble = ",".join(ensemble)
        self.ensemble = str(ensemble).lower()
        self.subset = subset
        self.na_tolerance = na_tolerance
        self.metric_weighting = metric_weighting
        self.drop_most_recent = drop_most_recent
        self.drop_data_older_than_periods = drop_data_older_than_periods
        self.model_list = model_list
        self.transformer_list = transformer_list
        self.transformer_max_depth = transformer_max_depth
        self.num_validations = abs(int(num_validations))
        self.models_to_validate = models_to_validate
        self.max_per_model_class = max_per_model_class
        self.validation_method = str(validation_method).lower()
        self.min_allowed_train_percent = min_allowed_train_percent
        self.max_generations = max_generations
        self.remove_leading_zeroes = remove_leading_zeroes
        self.model_interrupt = model_interrupt
        self.verbose = int(verbose)
        self.n_jobs = n_jobs
        if self.ensemble == 'all':
            self.ensemble = 'simple,distance,horizontal-max,probabilistic'
        elif self.ensemble == 'auto':
            if model_list in ['fast', 'default', 'all', 'multivariate']:
                self.ensemble = 'simple,distance,horizontal-max'
            else:
                self.ensemble = 'simple'

        if self.forecast_length == 1:
            if metric_weighting['contour_weighting'] > 0:
                print("Contour metric does not work with forecast_length == 1")

        if 'seasonal' in self.validation_method:
            val_list = [x for x in str(self.validation_method) if x.isdigit()]
            self.seasonal_val_periods = int(''.join(val_list))

        if self.n_jobs == 'auto':
            from autots.tools import cpu_count

            self.n_jobs = cpu_count()

        # convert shortcuts of model lists to actual lists of models
        if model_list in list(model_lists.keys()):
            self.model_list = model_lists[model_list]
        # prepare for a common Typo
        elif 'Prophet' in model_list:
            model_list = ["FBProphet" if x=="Prophet" else x for x in model_list]

        # generate template to begin with
        initial_template = str(initial_template).lower()
        if initial_template == 'random':
            self.initial_template = RandomTemplate(
                50,
                model_list=self.model_list,
                transformer_list=self.transformer_list,
                transformer_max_depth=self.transformer_max_depth,
            )
        elif initial_template == 'general':
            from autots.templates.general import general_template

            self.initial_template = general_template
        elif initial_template == 'general+random':
            from autots.templates.general import general_template

            random_template = RandomTemplate(
                40,
                model_list=self.model_list,
                transformer_list=self.transformer_list,
                transformer_max_depth=self.transformer_max_depth,
            )
            self.initial_template = pd.concat(
                [general_template, random_template], axis=0
            ).drop_duplicates()
        elif isinstance(initial_template, pd.DataFrame):
            self.initial_template = initial_template
        else:
            print("Input initial_template unrecognized. Using Random.")
            self.initial_template = RandomTemplate(
                50,
                model_list=self.model_list,
                transformer_list=self.transformer_list,
                transformer_max_depth=self.transformer_max_depth,
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
        ]:
            from autots.tools.transform import transformer_list_to_dict

            transformer_lst, prb = transformer_list_to_dict(self.transformer_list)
            for index, row in self.initial_template.iterrows():
                full_params = json.loads(row['TransformationParameters'])
                transformations = full_params['transformations']
                transformation_params = full_params['transformation_params']
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

        self.best_model = pd.DataFrame()
        self.regressor_used = False
        # do not add 'ID' to the below unless you want to refactor things.
        self.template_cols = [
            'Model',
            'ModelParameters',
            'TransformationParameters',
            'Ensemble',
        ]
        self.initial_results = TemplateEvalObject()

        if verbose > 2:
            print('"Hello. Would you like to destroy some evil today?" - Sanderson')

    def __repr__(self):
        """Print."""
        if self.best_model.empty:
            return "Uninitiated AutoTS object"
        else:
            try:
                return f"Initiated AutoTS object with best model: \n{self.best_model['Model'].iloc[0]}\n{self.best_model['TransformationParameters'].iloc[0]}\n{self.best_model['ModelParameters'].iloc[0]}"
            except Exception:
                return "Initiated AutoTS object"

    def fit(
        self,
        df,
        date_col: str = None,
        value_col: str = None,
        id_col: str = None,
        future_regressor=[],
        weights: dict = {},
        result_file: str = None,
        grouping_ids=None,
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
        self.weights = weights
        self.date_col = date_col
        self.value_col = value_col
        self.id_col = id_col
        self.grouping_ids = grouping_ids

        # import mkl
        # so this actually works it seems, on all sub process models
        # mkl.set_num_threads_local(8)

        # convert class variables to local variables (makes testing easier)
        forecast_length = self.forecast_length
        # flag if weights are given
        if bool(weights):
            weighted = True
        else:
            weighted = False
        self.weighted = weighted
        frequency = self.frequency
        prediction_interval = self.prediction_interval
        no_negatives = self.no_negatives
        random_seed = self.random_seed
        holiday_country = self.holiday_country
        ensemble = self.ensemble
        metric_weighting = self.metric_weighting
        num_validations = self.num_validations
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

        # convert data to wide format
        if date_col is None and value_col is None:
            df_wide = pd.DataFrame(df)
            assert (
                type(df_wide.index) is pd.DatetimeIndex
            ), "df index is not pd.DatetimeIndex"
        else:
            df_wide = long_to_wide(
                df,
                date_col=self.date_col,
                value_col=self.value_col,
                id_col=self.id_col,
                aggfunc=self.aggfunc,
            )

        df_wide = df_cleanup(
            df_wide,
            frequency=self.frequency,
            na_tolerance=self.na_tolerance,
            drop_data_older_than_periods=self.drop_data_older_than_periods,
            aggfunc=self.aggfunc,
            drop_most_recent=self.drop_most_recent,
            verbose=self.verbose,
        )

        # handle categorical data if present
        self.categorical_transformer = NumericTransformer(verbose=self.verbose)
        df_wide_numeric = self.categorical_transformer.fit_transform(df_wide)


        # use "mean" to assign weight as mean
        if weights == 'mean':
            weights = df_wide_numeric.mean(axis=0).to_dict()
        elif weights == 'median':
            weights = df_wide_numeric.median(axis=0).to_dict()
        elif weights == 'min':
            weights = df_wide_numeric.min(axis=0).to_dict()
        elif weights == 'max':
            weights = df_wide_numeric.max(axis=0).to_dict()
        # clean up series weighting input
        if not weighted:
            weights = {x: 1 for x in df_wide_numeric.columns}
        else:
            # handle not all weights being provided
            if self.verbose > 1:
                key_count = 0
                for col in df_wide_numeric.columns:
                    if col in weights:
                        key_count += 1
                key_count = df_wide_numeric.shape[1] - key_count
                if key_count > 0:
                    print(f"{key_count} series_id not in weights. Inferring 1.")
                else:
                    print("All series_id present in weighting.")
            weights = {
                col: (weights[col] if col in weights else 1) for col in df_wide_numeric.columns
            }
            # handle non-numeric inputs
            weights = {
                key: (abs(float(weights[key])) if str(weights[key]).isdigit() else 1)
                for key in weights
            }

        # replace any zeroes that occur prior to all non-zero values
        if self.remove_leading_zeroes:
            # keep the last row unaltered to keep metrics happier if all zeroes
            temp = df_wide_numeric.head(df_wide_numeric.shape[0] - 1)
            temp = temp.abs().cumsum(axis=0).replace(0, np.nan)
            temp = df_wide_numeric[~temp.isna()]
            temp = temp.head(df_wide_numeric.shape[0] - 1)
            df_wide_numeric = pd.concat([temp, df_wide_numeric.tail(1)], axis=0)

        self.df_wide_numeric = df_wide_numeric
        self.startTimeStamps = df_wide_numeric.notna().idxmax()

        # record if subset or not
        if self.subset is not None:
            self.subset = abs(int(self.subset))
            if self.subset >= self.df_wide_numeric.shape[1]:
                self.subset_flag = False
            else:
                self.subset_flag = True
        else:
            self.subset_flag = False

        # take a subset of the data if working with a large number of series
        if self.subset_flag:
            df_subset = subset_series(
                df_wide_numeric,
                list((weights.get(i)) for i in df_wide_numeric.columns),
                n=self.subset,
                random_state=random_seed,
            )
            if self.verbose > 1:
                print(f'First subset is of: {df_subset.columns}')
        else:
            df_subset = df_wide_numeric.copy()

        # subset the weighting information as well
        if not weighted:
            current_weights = {x: 1 for x in df_subset.columns}
        else:
            current_weights = {x: weights[x] for x in df_subset.columns}

        # split train and test portions, and split regressor if present
        df_train, df_test = simple_train_test_split(
            df_subset,
            forecast_length=forecast_length,
            min_allowed_train_percent=self.min_allowed_train_percent,
            verbose=self.verbose,
        )
        try:
            if not isinstance(future_regressor, pd.DataFrame):
                future_regressor = pd.DataFrame(future_regressor)
            if not isinstance(future_regressor.index, pd.DatetimeIndex):
                future_regressor.index = df_subset.index
            # handle any non-numeric data, crudely
            self.regr_num_trans = NumericTransformer(verbose=self.verbose)
            future_regressor = self.regr_num_trans.fit_transform(
                future_regressor
            )
            self.future_regressor_train = future_regressor
            future_regressor_train = future_regressor.reindex(index=df_train.index)
            future_regressor_test = future_regressor.reindex(index=df_test.index)
        except Exception:
            future_regressor_train = []
            future_regressor_test = []

        model_count = 0

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
        template_result = TemplateWizard(
            self.initial_template,
            df_train,
            df_test,
            weights=current_weights,
            model_count=model_count,
            ensemble=ensemble,
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            no_negatives=no_negatives,
            constraint=self.constraint,
            future_regressor_train=future_regressor_train,
            future_regressor_forecast=future_regressor_test,
            holiday_country=holiday_country,
            startTimeStamps=self.startTimeStamps,
            template_cols=template_cols,
            random_seed=random_seed,
            model_interrupt=self.model_interrupt,
            grouping_ids=self.grouping_ids,
            verbose=verbose,
            n_jobs=self.n_jobs,
            max_generations=self.max_generations,
        )
        model_count = template_result.model_count

        # capture the data from the lower level results
        self.initial_results = self.initial_results.concat(template_result)
        self.initial_results.model_results['Score'] = generate_score(
            self.initial_results.model_results,
            metric_weighting=metric_weighting,
            prediction_interval=prediction_interval,
        )
        if result_file is not None:
            self.initial_results.save(result_file)

        # now run new generations, trying more models based on past successes.
        current_generation = 0
        while current_generation < self.max_generations:
            current_generation += 1
            if verbose > 0:
                print(
                    "New Generation: {} of {}".format(
                        current_generation, self.max_generations
                    )
                )
            cutoff_multiple = 5 if current_generation < 10 else 3
            top_n = len(self.model_list) * cutoff_multiple
            new_template = NewGeneticTemplate(
                self.initial_results.model_results,
                submitted_parameters=submitted_parameters,
                sort_column="Score",
                sort_ascending=True,
                max_results=top_n,
                max_per_model_class=5,
                top_n=top_n,
                template_cols=template_cols,
                transformer_list=self.transformer_list,
                transformer_max_depth=self.transformer_max_depth,
            )
            submitted_parameters = pd.concat(
                [submitted_parameters, new_template],
                axis=0,
                ignore_index=True,
                sort=False,
            ).reset_index(drop=True)

            template_result = TemplateWizard(
                new_template,
                df_train,
                df_test,
                weights=current_weights,
                model_count=model_count,
                ensemble=ensemble,
                forecast_length=forecast_length,
                frequency=frequency,
                prediction_interval=prediction_interval,
                no_negatives=no_negatives,
                constraint=self.constraint,
                future_regressor_train=future_regressor_train,
                future_regressor_forecast=future_regressor_test,
                holiday_country=holiday_country,
                startTimeStamps=self.startTimeStamps,
                template_cols=template_cols,
                model_interrupt=self.model_interrupt,
                grouping_ids=self.grouping_ids,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=self.n_jobs,
                current_generation=current_generation,
                max_generations=self.max_generations,
            )
            model_count = template_result.model_count

            # capture results from lower-level template run
            self.initial_results = self.initial_results.concat(template_result)
            self.initial_results.model_results['Score'] = generate_score(
                self.initial_results.model_results,
                metric_weighting=metric_weighting,
                prediction_interval=prediction_interval,
            )
            if result_file is not None:
                self.initial_results.save(result_file)

        # try ensembling
        if ensemble not in [None, 'none']:
            try:
                ensemble_templates = EnsembleTemplateGenerator(
                    self.initial_results,
                    forecast_length=forecast_length,
                    ensemble=ensemble,
                )
                template_result = TemplateWizard(
                    ensemble_templates,
                    df_train,
                    df_test,
                    weights=current_weights,
                    model_count=model_count,
                    forecast_length=forecast_length,
                    frequency=frequency,
                    prediction_interval=prediction_interval,
                    no_negatives=no_negatives,
                    constraint=self.constraint,
                    ensemble=ensemble,
                    future_regressor_train=future_regressor_train,
                    future_regressor_forecast=future_regressor_test,
                    holiday_country=holiday_country,
                    startTimeStamps=self.startTimeStamps,
                    template_cols=template_cols,
                    model_interrupt=self.model_interrupt,
                    grouping_ids=self.grouping_ids,
                    random_seed=random_seed,
                    verbose=verbose,
                    n_jobs=self.n_jobs,
                )
                model_count = template_result.model_count
                # capture results from lower-level template run
                self.initial_results = self.initial_results.concat(template_result)
                self.initial_results.model_results['Score'] = generate_score(
                    self.initial_results.model_results,
                    metric_weighting=metric_weighting,
                    prediction_interval=prediction_interval,
                )
                if result_file is not None:
                    self.initial_results.save(result_file)
            except Exception as e:
                print(f"Ensembling Error: {e}")

        # drop any duplicates in results
        self.initial_results.model_results = (
            self.initial_results.model_results.drop_duplicates(
                subset=(['ID'] + self.template_cols)
            )
        )

        # validations if float
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

        # check how many validations are possible given the length of the data.
        if 'seasonal' in self.validation_method:
            temp = df_wide_numeric.shape[0] + self.forecast_length
            max_possible = temp / self.seasonal_val_periods
        else:
            max_possible = (df_wide_numeric.shape[0]) / forecast_length
        if (max_possible - np.floor(max_possible)) > self.min_allowed_train_percent:
            max_possible = int(max_possible)
        else:
            max_possible = int(max_possible) - 1
        if max_possible < (num_validations + 1):
            num_validations = max_possible - 1
            if num_validations < 0:
                num_validations = 0
            print(
                "Too many training validations for length of data provided, decreasing num_validations to {}".format(
                    num_validations
                )
            )
        self.num_validations = num_validations

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
        ensy = ['horizontal', 'probabilistic']
        if (
            any(x in ensemble for x in ensy)
            and not self.subset_flag
            and self.models_to_validate > 30
        ):
            model_results = self.initial_results.model_results
            mods = pd.DataFrame()
            if 'horizontal' in ensemble:
                mods = pd.concat([mods, self.initial_results.per_series_mae.idxmin()])
            if 'probabilistic' in ensemble:
                mods = pd.concat([mods, self.initial_results.per_series_spl.idxmin()])
            per_series_val = model_results[
                model_results['ID'].isin(mods.iloc[:, 0].unique().tolist())
            ]
            validation_template = pd.concat(
                [validation_template, per_series_val], axis=0
            )
            validation_template = validation_template.drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )
        validation_template = validation_template[self.template_cols]

        # run validations
        if num_validations > 0:
            model_count = 0
            for y in range(num_validations):
                if verbose > 0:
                    print("Validation Round: {}".format(str(y + 1)))
                # slice the validation data into current slice
                val_list = ['backwards', 'back', 'backward']
                if self.validation_method in val_list:
                    # gradually remove the end
                    current_slice = df_wide_numeric.head(
                        df_wide_numeric.shape[0] - (y + 1) * forecast_length
                    )
                elif self.validation_method == 'even':
                    # /num_validations biases it towards the last segment
                    validation_size = len(df_wide_numeric.index) - forecast_length
                    validation_size = validation_size / (num_validations + 1)
                    validation_size = int(np.floor(validation_size))
                    current_slice = df_wide_numeric.head(
                        validation_size * (y + 1) + forecast_length
                    )
                elif 'seasonal' in self.validation_method:
                    val_per = (y + 1) * self.seasonal_val_periods
                    if self.seasonal_val_periods < forecast_length:
                        pass
                    else:
                        val_per = val_per - forecast_length
                    val_per = df_wide_numeric.shape[0] - val_per
                    current_slice = df_wide_numeric.head(val_per)
                else:
                    raise ValueError(
                        "Validation Method not recognized try 'even', 'backwards'"
                    )

                # subset series (if used) and take a new train/test split
                if self.subset_flag:
                    df_subset = subset_series(
                        current_slice,
                        list((weights.get(i)) for i in current_slice.columns),
                        n=self.subset,
                        random_state=(random_seed + y + 1),
                    )
                    if self.verbose > 1:
                        print(f'{y + 1} subset is of: {df_subset.columns}')
                else:
                    df_subset = current_slice
                if not weighted:
                    current_weights = {x: 1 for x in df_subset.columns}
                else:
                    current_weights = {x: weights[x] for x in df_subset.columns}

                val_df_train, val_df_test = simple_train_test_split(
                    df_subset,
                    forecast_length=forecast_length,
                    min_allowed_train_percent=self.min_allowed_train_percent,
                    verbose=self.verbose,
                )
                if self.verbose >= 2:
                    print(f'Validation index is {val_df_train.index}')

                # slice regressor into current validation slices
                try:
                    val_future_regressor_train = future_regressor.reindex(
                        index=val_df_train.index
                    )
                    val_future_regressor_test = future_regressor.reindex(
                        index=val_df_test.index
                    )
                except Exception:
                    val_future_regressor_train = []
                    val_future_regressor_test = []

                # run validation template on current slice
                template_result = TemplateWizard(
                    validation_template,
                    df_train=val_df_train,
                    df_test=val_df_test,
                    weights=current_weights,
                    # model_count=model_count,
                    forecast_length=forecast_length,
                    frequency=frequency,
                    prediction_interval=prediction_interval,
                    no_negatives=no_negatives,
                    constraint=self.constraint,
                    ensemble=ensemble,
                    future_regressor_train=val_future_regressor_train,
                    future_regressor_forecast=val_future_regressor_test,
                    holiday_country=holiday_country,
                    startTimeStamps=self.startTimeStamps,
                    template_cols=self.template_cols,
                    model_interrupt=self.model_interrupt,
                    grouping_ids=self.grouping_ids,
                    random_seed=random_seed,
                    verbose=verbose,
                    n_jobs=self.n_jobs,
                    validation_round=(y + 1),
                )
                model_count = template_result.model_count
                # gather results of template run
                self.initial_results = self.initial_results.concat(template_result)
                self.initial_results.model_results['Score'] = generate_score(
                    self.initial_results.model_results,
                    metric_weighting=metric_weighting,
                    prediction_interval=prediction_interval,
                )

        self.validation_results = copy.copy(self.initial_results)
        # aggregate validation results
        self.validation_results = validation_aggregation(self.validation_results)
        error_msg_template = """No models available from validation.
Try increasing models_to_validate, max_per_model_class
or otherwise increase models available."""

        # Construct horizontal style ensembles
        ens_list = ['horizontal', 'probabilistic', 'hdist']
        if any(x in ensemble for x in ens_list):
            ensemble_templates = pd.DataFrame()
            try:
                if 'horizontal' in ensemble:
                    per_series = self.initial_results.per_series_mae.copy()
                    # select only those models which were validated
                    temp = per_series.mean(axis=1).groupby(level=0).count()
                    temp = temp[temp >= (num_validations + 1)]
                    per_series = per_series[per_series.index.isin(temp.index)]
                    # this .mean() should assure all series get a value
                    # as long as they worked in at least one validation
                    per_series = per_series.groupby(level=0).mean()
                    ens_templates = HorizontalTemplateGenerator(
                        per_series,
                        model_results=self.initial_results.model_results,
                        forecast_length=forecast_length,
                        ensemble=ensemble.replace('probabilistic', ' ').replace(
                            'hdist', ' '
                        ),
                        subset_flag=self.subset_flag,
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
                if 'hdist' in ensemble:
                    per_series = self.initial_results.per_series_rmse1.copy()
                    temp = per_series.mean(axis=1).groupby(level=0).count()
                    temp = temp[temp >= (num_validations + 1)]
                    per_series = per_series[per_series.index.isin(temp.index)]
                    per_series = per_series.groupby(level=0).mean()
                    per_series2 = self.initial_results.per_series_rmse2.copy()
                    temp = per_series2.mean(axis=1).groupby(level=0).count()
                    temp = temp[temp >= (num_validations + 1)]
                    per_series2 = per_series2[per_series2.index.isin(temp.index)]
                    per_series2 = per_series2.groupby(level=0).mean()
                    ens_templates = HorizontalTemplateGenerator(
                        per_series,
                        model_results=self.initial_results.model_results,
                        forecast_length=forecast_length,
                        ensemble=ensemble.replace('horizontal', ' ').replace(
                            'probabilistic', ' '
                        ),
                        subset_flag=self.subset_flag,
                        per_series2=per_series2,
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
            except Exception as e:
                if self.verbose >= 0:
                    print(f"Ensembling Error: {e}")
            try:
                if 'probabilistic' in ensemble:
                    per_series = self.initial_results.per_series_spl.copy()
                    temp = per_series.mean(axis=1).groupby(level=0).count()
                    temp = temp[temp >= (num_validations + 1)]
                    per_series = per_series[per_series.index.isin(temp.index)]
                    per_series = per_series.groupby(level=0).mean()
                    ens_templates = HorizontalTemplateGenerator(
                        per_series,
                        model_results=self.initial_results.model_results,
                        forecast_length=forecast_length,
                        ensemble=ensemble.replace('horizontal', ' ').replace(
                            'hdist', ' '
                        ),
                        subset_flag=self.subset_flag,
                    )
                    ensemble_templates = pd.concat(
                        [ensemble_templates, ens_templates], axis=0
                    )
            except Exception as e:
                if self.verbose >= 0:
                    print(f"Ensembling Error: {e}")
            try:
                # test on initial test split to make sure they work
                template_result = TemplateWizard(
                    ensemble_templates,
                    df_train,
                    df_test,
                    weights=current_weights,
                    model_count=0,
                    forecast_length=forecast_length,
                    frequency=frequency,
                    prediction_interval=prediction_interval,
                    no_negatives=no_negatives,
                    constraint=self.constraint,
                    future_regressor_train=future_regressor_train,
                    future_regressor_forecast=future_regressor_test,
                    holiday_country=holiday_country,
                    startTimeStamps=self.startTimeStamps,
                    template_cols=template_cols,
                    model_interrupt=self.model_interrupt,
                    grouping_ids=self.grouping_ids,
                    random_seed=random_seed,
                    verbose=verbose,
                    n_jobs=self.n_jobs,
                )
                # capture results from lower-level template run
                template_result.model_results['TotalRuntime'].fillna(
                    pd.Timedelta(seconds=60), inplace=True
                )
                self.initial_results.model_results = pd.concat(
                    [self.initial_results.model_results, template_result.model_results],
                    axis=0,
                    ignore_index=True,
                    sort=False,
                ).reset_index(drop=True)
                self.initial_results.model_results['Score'] = generate_score(
                    self.initial_results.model_results,
                    metric_weighting=metric_weighting,
                    prediction_interval=prediction_interval,
                )
                if result_file is not None:
                    self.initial_results.save(result_file)
            except Exception as e:
                if self.verbose >= 0:
                    print(f"Ensembling Error: {e}")
                template_result = TemplateEvalObject()
            try:
                template_result.model_results['smape']
            except KeyError:
                template_result.model_results['smape'] = 0
            # use the best of these if any ran successfully
            if template_result.model_results['smape'].sum(min_count=0) > 0:
                template_result.model_results['Score'] = generate_score(
                    template_result.model_results,
                    metric_weighting=metric_weighting,
                    prediction_interval=prediction_interval,
                )
                self.best_model = template_result.model_results.sort_values(
                    by="Score", ascending=True, na_position='last'
                ).head(1)[template_cols]
                self.ensemble_check = 1
            # else use the best of the previous
            else:
                if self.verbose >= 0:
                    print("Horizontal ensemble failed. Using best non-horizontal.")
                eligible_models = self.validation_results.model_results[
                    self.validation_results.model_results['Runs']
                    >= (num_validations + 1)
                ]
                try:
                    self.best_model = (
                        eligible_models.sort_values(
                            by="Score", ascending=True, na_position='last'
                        )
                        .drop_duplicates(subset=self.template_cols)
                        .head(1)[template_cols]
                    )
                    self.ensemble_check = int((self.best_model['Ensemble'].iloc[0]) > 0)
                except IndexError:
                    raise ValueError(error_msg_template)
        else:
            # choose best model
            eligible_models = self.validation_results.model_results[
                self.validation_results.model_results['Runs'] >= (num_validations + 1)
            ]
            try:
                self.best_model = (
                    eligible_models.sort_values(
                        by="Score", ascending=True, na_position='last'
                    )
                    .drop_duplicates(subset=self.template_cols)
                    .head(1)[template_cols]
                )
                self.ensemble_check = int((self.best_model['Ensemble'].iloc[0]) > 0)
            except IndexError:
                raise ValueError(error_msg_template)

        # set flags to check if regressors or ensemble used in final model.
        param_dict = json.loads(self.best_model.iloc[0]['ModelParameters'])
        if self.ensemble_check == 1:
            self.used_regressor_check = self._regr_param_check(param_dict)

        if self.ensemble_check == 0:
            self.used_regressor_check = False
            try:
                reg_param = param_dict['regression_type']
                if reg_param == 'User':
                    self.used_regressor_check = True
            except KeyError:
                pass
        # clean up any remaining print statements
        sys.stdout.flush()
        return self

    def _regr_param_check(self, param_dict):
        """Help to search for if a regressor was used in model."""
        out = False
        for key in param_dict['models']:
            cur_dict = json.loads(param_dict['models'][key]['ModelParameters'])
            try:
                reg_param = cur_dict['regression_type']
                if reg_param == 'User':
                    return True
            except KeyError:
                pass
            if param_dict['models'][key]['Model'] == 'Ensemble':
                out = self._regr_param_check(cur_dict)
                if out:
                    return out
        return out

    def predict(
        self,
        forecast_length: int = "self",
        prediction_interval: float = 'self',
        future_regressor=[],
        hierarchy=None,
        just_point_forecast: bool = False,
        verbose: int = 'self',
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            prediction_interval (float): interval of upper/lower forecasts.
                defaults to 'self' ie the interval specified in __init__()
                if prediction_interval is a list, then returns a dict of forecast objects.
            future_regressor (numpy.Array): additional regressor
            hierarchy: Not yet implemented
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Return:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        verbose = self.verbose if verbose == 'self' else verbose
        if forecast_length == 'self':
            forecast_length = self.forecast_length
        if prediction_interval == 'self':
            prediction_interval = self.prediction_interval

        # if the models don't need the regressor, ignore it...
        if not self.used_regressor_check:
            future_regressor = []
            self.future_regressor_train = []
        else:
            if not isinstance(future_regressor, pd.DataFrame):
                future_regressor = pd.DataFrame(future_regressor)
            # handle any non-numeric data, crudely
            future_regressor = self.regr_num_trans.transform(
                future_regressor
            )
            # make sure training regressor fits training data index
            self.future_regressor_train = self.future_regressor_train.reindex(
                index=self.df_wide_numeric.index
            )

        # allow multiple prediction intervals
        if isinstance(prediction_interval, list):
            forecast_objects = {}
            for interval in prediction_interval:
                df_forecast = PredictWitch(
                    self.best_model,
                    df_train=self.df_wide_numeric,
                    forecast_length=forecast_length,
                    frequency=self.frequency,
                    prediction_interval=interval,
                    no_negatives=self.no_negatives,
                    constraint=self.constraint,
                    future_regressor_train=self.future_regressor_train,
                    future_regressor_forecast=future_regressor,
                    holiday_country=self.holiday_country,
                    startTimeStamps=self.startTimeStamps,
                    grouping_ids=self.grouping_ids,
                    random_seed=self.random_seed,
                    verbose=verbose,
                    template_cols=self.template_cols,
                )
                # convert categorical back to numeric
                trans = self.categorical_transformer
                df_forecast.forecast = trans.inverse_transform(df_forecast.forecast)
                df_forecast.lower_forecast = trans.inverse_transform(
                    df_forecast.lower_forecast
                )
                df_forecast.upper_forecast = trans.inverse_transform(
                    df_forecast.upper_forecast
                )
                forecast_objects[interval] = df_forecast
            return forecast_objects
        else:
            df_forecast = PredictWitch(
                self.best_model,
                df_train=self.df_wide_numeric,
                forecast_length=forecast_length,
                frequency=self.frequency,
                prediction_interval=prediction_interval,
                no_negatives=self.no_negatives,
                constraint=self.constraint,
                future_regressor_train=self.future_regressor_train,
                future_regressor_forecast=future_regressor,
                holiday_country=self.holiday_country,
                startTimeStamps=self.startTimeStamps,
                grouping_ids=self.grouping_ids,
                random_seed=self.random_seed,
                verbose=verbose,
                template_cols=self.template_cols,
            )
            # convert categorical back to numeric
            trans = self.categorical_transformer
            df_forecast.forecast = trans.inverse_transform(df_forecast.forecast)
            df_forecast.lower_forecast = trans.inverse_transform(
                df_forecast.lower_forecast
            )
            df_forecast.upper_forecast = trans.inverse_transform(
                df_forecast.upper_forecast
            )
            sys.stdout.flush()
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
        n: int = 5,
        max_per_model_class: int = None,
        include_results: bool = False,
    ):
        """Export top results as a reusable template.

        Args:
            filename (str): 'csv' or 'json' (in filename).
                `None` to return a dataframe and not write a file.
            models (str): 'best' or 'all'
            n (int): if models = 'best', how many n-best to export
            max_per_model_class (int): if models = 'best',
                the max number of each model class to include in template
            include_results (bool): whether to include performance metrics
        """
        if models == 'all':
            export_template = self.initial_results.model_results[self.template_cols]
            export_template = export_template.drop_duplicates()
        elif models == 'best':
            # skip to the answer if just n==1
            if n == 1 and not include_results:
                export_template = self.best_model
            else:
                export_template = self.validation_results.model_results
                export_template = export_template[
                    export_template['Runs'] >= (self.num_validations + 1)
                ]
                ens_list = ['horizontal', 'probabilistic', 'hdist']
                if any(x in self.ensemble for x in ens_list):
                    temp = self.initial_results.model_results
                    temp = temp[temp['Ensemble'] >= 2]
                    temp = temp[temp['Exceptions'].isna()]
                    export_template = export_template.merge(
                        temp,
                        how='outer',
                        on=export_template.columns.intersection(temp.columns).to_list(),
                    )
                    export_template['Score'] = generate_score(
                        export_template,
                        metric_weighting=self.metric_weighting,
                        prediction_interval=self.prediction_interval,
                    )
                if str(max_per_model_class).isdigit():
                    export_template = (
                        export_template.sort_values('Score', ascending=True)
                        .groupby('Model')
                        .head(max_per_model_class)
                        .reset_index()
                    )
                export_template = export_template.nsmallest(n, columns=['Score'])
                if not include_results:
                    export_template = export_template[self.template_cols]
        else:
            raise ValueError("`models` must be 'all' or 'best'")
        try:
            if filename is None:
                return export_template
            elif '.csv' in filename:
                return export_template.to_csv(filename, index=False)
            elif '.json' in filename:
                return export_template.to_json(filename, orient='columns')
            else:
                raise ValueError("file must be .csv or .json")
        except PermissionError:
            raise PermissionError(
                "Permission Error: directory or existing file is locked for editing."
            )

    def import_template(
        self, filename: str, method: str = "add_on", enforce_model_list: bool = True
    ):
        """Import a previously exported template of model parameters.
        Must be done before the AutoTS object is .fit().

        Args:
            filename (str): file location (or a pd.DataFrame already loaded)
            method (str): 'add_on' or 'only' - "add_on" keeps `initial_template` generated in init. "only" uses only this template.
            enforce_model_list (bool): if True, remove model types not in model_list
        """
        if isinstance(filename, pd.DataFrame):
            import_template = filename.copy()
        elif '.csv' in filename:
            import_template = pd.read_csv(filename)
        elif '.json' in filename:
            import_template = pd.read_json(filename, orient='columns')
        else:
            raise ValueError("file must be .csv or .json")

        try:
            import_template = import_template[self.template_cols]
        except Exception:
            print(
                "Column names {} were not recognized as matching template columns: {}".format(
                    str(import_template.columns), str(self.template_cols)
                )
            )

        import_template = unpack_ensemble_models(
            import_template, self.template_cols, keep_ensemble=True, recursive=True
        )

        if enforce_model_list:
            # remove models not in given model list
            mod_list = self.model_list + ['Ensemble']
            import_template = import_template[import_template['Model'].isin(mod_list)]
            if import_template.shape[0] == 0:
                raise ValueError(
                    "Len 0. Model_list does not match models in template! Try enforce_model_list=False."
                )

        if method.lower() in ['add on', 'addon', 'add_on']:
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
            return ValueError("method must be 'add_on' or 'only'")

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


class AutoTSIntervals(object):
    """Autots looped to test multiple prediction intervals. Experimental.

    Runs max_generations on first prediction interval, then validates on remainder.
    Most args are passed through to AutoTS().

    Args:
        interval_models_to_validate (int): number of models to validate on each prediction interval.
        import_results (str): results from run on same data to load, `filename.pickle`.
            Currently result_file and import only save/load initial run, no validations.
    """

    def fit(
        self,
        prediction_intervals,
        forecast_length,
        df_long,
        max_generations,
        num_validations,
        validation_method,
        models_to_validate,
        interval_models_to_validate,
        date_col,
        value_col,
        id_col=None,
        import_template=None,
        import_method='only',
        import_results=None,
        result_file=None,
        model_list='all',
        metric_weighting: dict = {
            'smape_weighting': 1,
            'mae_weighting': 0,
            'rmse_weighting': 1,
            'containment_weighting': 0,
            'runtime_weighting': 0,
            'spl_weighting': 10,
            'contour_weighting': 0,
        },
        weights: dict = {},
        grouping_ids=None,
        future_regressor=[],
        model_interrupt: bool = False,
        constraint=2,
        no_negatives=False,
        remove_leading_zeroes=False,
        random_seed=2020,
    ):
        """Train and find best."""
        overall_results = TemplateEvalObject()
        per_series_spl = pd.DataFrame()
        runs = 0
        for interval in prediction_intervals:
            if runs != 0:
                max_generations = 0
                models_to_validate = 0.99
            print(f"Current interval is {interval}")
            current_model = AutoTS(
                forecast_length=forecast_length,
                prediction_interval=interval,
                ensemble="probabilistic-max",
                max_generations=max_generations,
                model_list=model_list,
                constraint=constraint,
                no_negatives=no_negatives,
                remove_leading_zeroes=remove_leading_zeroes,
                metric_weighting=metric_weighting,
                subset=None,
                random_seed=random_seed,
                num_validations=num_validations,
                validation_method=validation_method,
                model_interrupt=model_interrupt,
                models_to_validate=models_to_validate,
            )
            if import_template is not None:
                current_model = current_model.import_template(
                    import_template, method=import_method
                )
            if import_results is not None:
                current_model = current_model.import_results(import_results)
            current_model = current_model.fit(
                df_long,
                future_regressor=future_regressor,
                weights=weights,
                grouping_ids=grouping_ids,
                result_file=result_file,
                date_col=date_col,
                value_col=value_col,
                id_col=id_col,
            )
            current_model.initial_results.model_results['interval'] = interval
            temp = current_model.initial_results
            overall_results = overall_results.concat(temp)
            temp = current_model.initial_results.per_series_spl
            per_series_spl = pd.concat([per_series_spl, temp], axis=0)
            if runs == 0:
                result_file = None
                import_results = None
                import_template = current_model.export_template(
                    None, models='best', n=interval_models_to_validate
                )
            runs += 1
        self.validation_results = validation_aggregation(overall_results)
        self.results = overall_results.model_results
        # remove models not validated
        temp = per_series_spl.mean(axis=1).groupby(level=0).count()
        temp = temp[temp >= ((runs) * (num_validations + 1))]
        per_series_spl = per_series_spl[per_series_spl.index.isin(temp.index)]
        per_series_spl = per_series_spl.groupby(level=0).mean()
        # from autots.models.ensemble import HorizontalTemplateGenerator
        ens_templates = HorizontalTemplateGenerator(
            per_series_spl,
            model_results=overall_results.model_results,
            forecast_length=forecast_length,
            ensemble='probabilistic-max',
            subset_flag=False,
        )
        self.per_series_spl = per_series_spl
        self.ens_templates = ens_templates
        self.prediction_intervals = prediction_intervals

        self.future_regressor_train = future_regressor
        self.forecast_length = forecast_length
        self.df_wide_numeric = current_model.df_wide_numeric
        self.frequency = current_model.frequency
        self.no_negatives = current_model.no_negatives
        self.constraint = current_model.constraint
        self.holiday_country = current_model.holiday_country
        self.startTimeStamps = current_model.startTimeStamps
        self.random_seed = current_model.random_seed
        self.verbose = current_model.verbose
        self.template_cols = current_model.template_cols
        self.categorical_transformer = current_model.categorical_transformer
        return self

    def predict(self, future_regressor=[], verbose: int = 'self') -> dict:
        """Generate forecasts after training complete."""
        if len(future_regressor) > 0:
            future_regressor = pd.DataFrame(future_regressor)
            self.future_regressor_train = self.future_regressor_train.reindex(
                index=self.df_wide_numeric.index
            )
        forecast_objects = {}
        verbose = self.verbose if verbose == 'self' else verbose

        for interval in self.prediction_intervals:
            df_forecast = PredictWitch(
                self.ens_templates,
                df_train=self.df_wide_numeric,
                forecast_length=self.forecast_length,
                frequency=self.frequency,
                prediction_interval=interval,
                no_negatives=self.no_negatives,
                constraint=self.constraint,
                future_regressor_train=self.future_regressor_train,
                future_regressor_forecast=future_regressor,
                holiday_country=self.holiday_country,
                startTimeStamps=self.startTimeStamps,
                grouping_ids=self.grouping_ids,
                random_seed=self.random_seed,
                verbose=verbose,
                template_cols=self.template_cols,
            )

            trans = self.categorical_transformer
            df_forecast.forecast = trans.inverse_transform(df_forecast.forecast)
            df_forecast.lower_forecast = trans.inverse_transform(
                df_forecast.lower_forecast
            )
            df_forecast.upper_forecast = trans.inverse_transform(
                df_forecast.upper_forecast
            )
            forecast_objects[interval] = df_forecast
        return forecast_objects


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
        frequency = pd.infer_freq(df_wide.index, warn=True)

    forecast_index = pd.date_range(
        freq=frequency, start=df_wide.index[-1], periods=forecast_length + 1
    )
    forecast_index = forecast_index[1:]

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
    # test_cols = [column for column in test.columns if 'NaNZ' not in column]
    # test = test[test_cols]
    """
    try:
        from mlxtend.frequent_patterns import association_rules
        from mlxtend.frequent_patterns import apriori
        import re
        freq_itemsets = apriori(test.drop('ExceptionFlag', axis=1),
                                min_support=0.3, use_colnames=True)
        rules = association_rules(freq_itemsets)
        err_rules = pd.DataFrame()
        for err in except_df.columns:
            err = re.sub('[^a-zA-Z0-9\s]', '', err)
            edf = rules[
                rules['consequents'].astype(
                    str).str.replace('[^a-zA-Z0-9\s]', '').str.contains(err)]
            err_rules = pd.concat([err_rules, edf],
                                     axis=0, ignore_index=True)
        err_rules = err_rules.drop_duplicates()
    except Exception as e:
        print(repr(e))
    """
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
