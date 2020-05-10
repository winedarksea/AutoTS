"""Higher-level backbone of auto time series modeling."""
import numpy as np
import pandas as pd
import copy
import json

from autots.tools.shaping import long_to_wide
import random
from autots.tools.shaping import values_to_numeric
from autots.tools.profile import data_profile
from autots.tools.shaping import subset_series
from autots.tools.shaping import simple_train_test_split
from autots.evaluator.auto_model import TemplateEvalObject
from autots.evaluator.auto_model import NewGeneticTemplate
from autots.evaluator.auto_model import RandomTemplate
from autots.evaluator.auto_model import TemplateWizard
from autots.evaluator.auto_model import unpack_ensemble_models
from autots.evaluator.auto_model import generate_score
from autots.models.ensemble import EnsembleTemplateGenerator
from autots.evaluator.auto_model import PredictWitch
from autots.tools.shaping import categorical_inverse
from autots.evaluator.auto_model import validation_aggregation


class AutoTS(object):
    """Automate time series modeling using a genetic algorithm.

    Args:
        forecast_length (int): number of periods over which to evaluate forecast. Can be overriden later in .predict().
        frequency (str): 'infer' or a specific pandas datetime offset. Can be used to force rollup of data (ie daily input, but frequency 'M' will rollup to monthly).
        aggfunc (str): if data is to be rolled up to a higher frequency (daily -> monthly) or duplicates are included. Default 'first' removes duplicates, for rollup try 'mean' or 'sum'. Beware numeric aggregations like 'mean' will *drop* categorical features as cat->num occurs later.
        prediction_interval (float): 0-1, uncertainty range for upper and lower forecasts. Adjust range, but rarely matches actual containment.
        no_negatives (bool): if True, all negative predictions are rounded up to 0.
        ensemble (str): None, 'simple', 'distance'
        initial_template (str): 'Random' - randomly generates starting template, 'General' uses template included in package, 'General+Random' - both of previous. Also can be overriden with self.import_template()
        figures (bool): Not yet implemented
        random_seed (int): random seed allows (slightly) more consistent results.
        holiday_country (str): passed through to Holidays package for some models.
        subset (int): maximum number of series to evaluate at once. Useful to speed evaluation when many series are input.
        na_tolerance (float): 0 to 1. Series are dropped if they have more than this percent NaN. 0.95 here would allow data containing upto 95% NaN values.
        metric_weighting (dict): weights to assign to metrics, effecting how the ranking score is generated.
        drop_most_recent (int): option to drop n most recent data points. Useful, say, for monthly sales data where the current (unfinished) month is included.
        drop_data_older_than_periods (int): take only the n most recent timestamps
        model_list (list): list of names of model objects to use
        num_validations (int): number of cross validations to perform. 0 for just train/test on final split.
        models_to_validate (int): top n models to pass through to cross validation. Or float in 0 to 1 as % of tried.
        max_per_model_class (int): of the models_to_validate what is the maximum to pass from any one model class/family.
        validation_method (str): 'even' or 'backwards' where backwards is better for shorter training sets
        min_allowed_train_percent (float): useful in (unrecommended) cases where forecast_length > training length. Percent of forecast length to allow as min training, else raises error.
        max_generations (int): number of genetic algorithms generations to run. More runs = better chance of better accuracy.
        verbose (int): setting to 0 or lower should reduce most output. Higher numbers give slightly more output.
        
    Attributes:
        best_model (pandas.DataFrame): DataFrame containing template for the best ranked model
        regression_check (bool): If True, the best_model uses an input 'User' preord_regressor
    """

    def __init__(self,
                 forecast_length: int = 14,
                 frequency: str = 'infer',
                 aggfunc: str = 'first',
                 prediction_interval: float = 0.9,
                 no_negatives: bool = False,
                 ensemble: str = None,
                 initial_template: str = 'General+Random',
                 figures: bool = False,
                 random_seed: int = 2020,
                 holiday_country: str = 'US',
                 subset: int = None,
                 na_tolerance: float = 0.99,
                 metric_weighting: dict = {'smape_weighting': 10,
                                           'mae_weighting': 2,
                                           'rmse_weighting': 2,
                                           'containment_weighting': 0,
                                           'runtime_weighting': 0,
                                           'spl_weighting': 1,
                                           'contour_weighting': 0
                                           },
                 drop_most_recent: int = 0,
                 drop_data_older_than_periods: int = 100000,
                 model_list: str = 'default',
                 num_validations: int = 2,
                 models_to_validate: float = 0.05,
                 max_per_model_class: int = None,
                 validation_method: str = 'even',
                 min_allowed_train_percent: float = 0.5,
                 max_generations: int = 5,
                 verbose: int = 1
                 ):
        self.forecast_length = int(abs(forecast_length))
        self.frequency = frequency
        self.aggfunc = aggfunc
        self.prediction_interval = prediction_interval
        self.no_negatives = no_negatives
        self.random_seed = random_seed
        self.holiday_country = holiday_country
        self.ensemble = ensemble
        self.subset = subset
        self.na_tolerance = na_tolerance
        self.metric_weighting = metric_weighting
        self.drop_most_recent = drop_most_recent
        self.drop_data_older_than_periods = drop_data_older_than_periods
        self.model_list = model_list
        self.num_validations = num_validations
        self.models_to_validate = models_to_validate
        self.max_per_model_class = max_per_model_class
        self.validation_method = validation_method
        self.min_allowed_train_percent = min_allowed_train_percent
        self.max_generations = max_generations
        self.verbose = int(verbose)
        if self.ensemble is not None:
            self.ensemble = str(self.ensemble).lower()
            if self.ensemble == 'all':
                self.ensemble = 'simple,distance'

        if self.forecast_length == 1:
            if metric_weighting['contour_weighting'] > 0:
                print("Contour metric does not work with forecast_length == 1")

        # convert shortcuts of model lists to actual lists of models
        if model_list == 'default':
            self.model_list = ['ZeroesNaive', 'LastValueNaive',
                               'AverageValueNaive', 'GLS', 'SeasonalNaive',
                               'GLM', 'ETS', 'ARIMA', 'FBProphet',
                               'RollingRegression', 'GluonTS',
                               'UnobservedComponents', 'VARMAX',
                               'VECM', 'DynamicFactor', 'MotifSimulation',
                               'WindowRegression']
        if model_list == 'superfast':
            self.model_list = ['ZeroesNaive', 'LastValueNaive',
                               'AverageValueNaive', 'GLS', 'SeasonalNaive']
        if model_list == 'fast':
            self.model_list = ['ZeroesNaive', 'LastValueNaive',
                               'AverageValueNaive', 'GLS', 'GLM', 'ETS',
                               'RollingRegression', 'WindowRegression',
                               'GluonTS', 'VAR',
                               'SeasonalNaive', 'UnobservedComponents',
                               'VECM']
        if model_list == 'probabilistic':
            self.model_list = ['ARIMA', 'GluonTS', 'FBProphet',
                               'AverageValueNaive', 'MotifSimulation',
                               'VARMAX', 'DynamicFactor', 'VAR']
        if model_list == 'multivariate':
            self.model_list = ['VECM', 'DynamicFactor', 'GluonTS', 'VARMAX',
                               'RollingRegression', 'WindowRegression','VAR']
        if model_list == 'all':
            self.model_list = ['ZeroesNaive', 'LastValueNaive',
                               'AverageValueNaive', 'GLS', 'GLM', 'ETS',
                               'ARIMA', 'FBProphet', 'RollingRegression',
                               'GluonTS', 'SeasonalNaive',
                               'UnobservedComponents', 'VARMAX', 'VECM',
                               'DynamicFactor', 'TSFreshRegressor',
                               'MotifSimulation', 'WindowRegression', 'VAR',
                               'TensorflowSTS', 'TFPRegression']

        # generate template to begin with
        if initial_template.lower() == 'random':
            self.initial_template = RandomTemplate(50,
                                                   model_list=self.model_list)
        elif initial_template.lower() == 'general':
            from autots.templates.general import general_template
            self.initial_template = general_template
        elif initial_template.lower() == 'general+random':
            from autots.templates.general import general_template
            random_template = RandomTemplate(40, model_list=self.model_list)
            self.initial_template = pd.concat([general_template,
                                               random_template],
                                              axis=0).drop_duplicates()
        else:
            print("Input initial_template either unrecognized or not yet implemented. Using Random.")
            self.initial_template = RandomTemplate(50)

        # remove models not in given model list
        self.initial_template = self.initial_template[self.initial_template['Model'].isin(self.model_list)]
        if len(self.initial_template.index) == 0:
            raise ValueError("No models in template! Adjust initial_template or model_list")

        self.best_model = pd.DataFrame()
        self.regressor_used = False
        self.template_cols = ['Model', 'ModelParameters',
                              'TransformationParameters', 'Ensemble']
        self.initial_results = TemplateEvalObject()

    def __repr__(self):
        """Print."""
        if self.best_model.empty:
            return "Uninitiated AutoTS object"
        else:
            try:
                return f"Initiated AutoTS object with best model: \n{self.best_model['Model'].iloc[0]}\n{self.best_model['TransformationParameters'].iloc[0]}\n{self.best_model['ModelParameters'].iloc[0]}"
            except Exception:
                return "Initiated AutoTS object"

    def fit(self, df,
            date_col: str = 'datetime', value_col: str = 'value',
            id_col: str = None, preord_regressor=[],
            weights: dict = {}, result_file: str = None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            date_col (str): name of datetime column
            value_col (str): name of column containing the data of series.
            id_col (str): name of column identifying different series.
            preord_regressor (numpy.Array): single external regressor matching train.index
            weights (dict): {'colname1': 2, 'colname2': 5} - increase importance of a series in metric evaluation. Any left blank assumed to have weight of 1.
            result_file (str): Location of template/results.csv to be saved at intermediate/final time.
        """
        self.weights = weights
        self.date_col = date_col
        self.value_col = value_col
        self.id_col = id_col

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
            try:
                if ".csv" not in str(result_file):
                    print("Result filename must be a valid 'filename.csv'")
                    result_file = None
            except Exception:
                print("Result filename must be a valid 'filename.csv'")
                result_file = None

        # set random seeds for environment
        random_seed = abs(int(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)

        # convert data to wide format
        df_wide = long_to_wide(
            df, date_col=self.date_col,
            value_col=self.value_col,
            id_col=self.id_col,
            frequency=self.frequency,
            na_tolerance=self.na_tolerance,
            drop_data_older_than_periods=self.drop_data_older_than_periods,
            aggfunc=self.aggfunc,
            drop_most_recent=self.drop_most_recent,
            verbose=self.verbose
            )

        # clean up series weighting input
        if not weighted:
            weights = {x: 1 for x in df_wide.columns}
        else:
            # handle not all weights being provided
            if self.verbose > 1:
                key_count = 0
                for col in df_wide.columns:
                    if col in weights:
                        key_count += 1
                key_count = df_wide.shape[1] - key_count
                if key_count > 0:
                    print(f"{key_count} series_id not in weights. Inferring 1.")
                else:
                    print("All series_id present in weighting.")
            weights = {col: (weights[col] if col in weights else 1) for col in df_wide.columns}
            # handle non-numeric inputs
            weights = {key: (abs(float(weights[key])) if str(weights[key]).isdigit() else 1) for key in weights}

        # handle categorical (not numeric) data if present
        categorical_transformer = values_to_numeric(df_wide)
        self.categorical_transformer = categorical_transformer
        df_wide_numeric = categorical_transformer.dataframe
        self.df_wide_numeric = df_wide_numeric

        # capture some misc information
        profile_df = data_profile(df_wide_numeric)
        self.startTimeStamps = profile_df.loc['FirstDate']

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
            df_subset = subset_series(df_wide_numeric, list((weights.get(i)) for i in df_wide_numeric.columns), n=self.subset, random_state=random_seed)
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
            verbose=self.verbose)
        try:
            preord_regressor = pd.DataFrame(preord_regressor)
            if not isinstance(preord_regressor.index, pd.DatetimeIndex):
                preord_regressor.index = df_subset.index
            self.preord_regressor_train = preord_regressor
            preord_regressor_train = preord_regressor.reindex(index=df_train.index)
            preord_regressor_test = preord_regressor.reindex(index=df_test.index)
        except Exception:
            preord_regressor_train = []
            preord_regressor_test = []

        model_count = 0

        # run the initial template
        self.initial_template = unpack_ensemble_models(
            self.initial_template, template_cols, keep_ensemble=True)
        submitted_parameters = self.initial_template.copy()
        template_result = TemplateWizard(
            self.initial_template, df_train,
            df_test, weights=current_weights,
            model_count=model_count,
            ensemble=ensemble,
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            no_negatives=no_negatives,
            preord_regressor_train=preord_regressor_train,
            preord_regressor_forecast=preord_regressor_test,
            holiday_country=holiday_country,
            startTimeStamps=self.startTimeStamps,
            template_cols=template_cols,
            random_seed=random_seed,
            verbose=verbose)
        model_count = template_result.model_count

        # capture the data from the lower level results
        self.initial_results.model_results = pd.concat(
            [self.initial_results.model_results,
             template_result.model_results],
            axis=0, ignore_index=True, sort=False).reset_index(drop=True)
        self.initial_results.per_series_mae = pd.concat(
            [self.initial_results.per_series_mae,
             template_result.per_series_mae],
            axis=0, sort=False)
        self.initial_results.per_timestamp_smape = pd.concat(
                [self.initial_results.per_timestamp_smape,
                 template_result.per_timestamp_smape],
                axis=0, sort=False)
        self.initial_results.model_results['Score'] = generate_score(self.initial_results.model_results, metric_weighting=metric_weighting,prediction_interval=prediction_interval)
        if result_file is not None:
            self.initial_results.model_results.to_csv(result_file, index=False)

        # now run new generations, trying more models based on past successes.
        current_generation = 0
        while current_generation < self.max_generations:
            current_generation += 1
            if verbose > 0:
                print("New Generation: {}".format(current_generation))
            cutoff_multiple = 5 if current_generation < 10 else 3
            top_n = len(self.model_list) * cutoff_multiple
            new_template = NewGeneticTemplate(
                self.initial_results.model_results,
                submitted_parameters=submitted_parameters,
                sort_column="Score", sort_ascending=True,
                max_results=top_n, max_per_model_class=5,
                top_n=top_n, template_cols=template_cols
                )
            submitted_parameters = pd.concat(
                [submitted_parameters, new_template],
                axis=0, ignore_index=True, sort=False).reset_index(drop=True)

            template_result = TemplateWizard(
                new_template, df_train, df_test,
                weights=current_weights,
                model_count=model_count,
                ensemble=ensemble,
                forecast_length=forecast_length,
                frequency=frequency,
                prediction_interval=prediction_interval,
                no_negatives=no_negatives,
                preord_regressor_train=preord_regressor_train,
                preord_regressor_forecast=preord_regressor_test,
                holiday_country=holiday_country,
                startTimeStamps=profile_df.loc['FirstDate'],
                template_cols=template_cols,
                random_seed=random_seed, verbose=verbose
                )
            model_count = template_result.model_count

            # capture results from lower-level template run
            self.initial_results.model_results = pd.concat(
                [self.initial_results.model_results,
                 template_result.model_results],
                axis=0, ignore_index=True, sort=False).reset_index(drop=True)
            self.initial_results.per_series_mae = pd.concat(
                [self.initial_results.per_series_mae,
                 template_result.per_series_mae],
                axis=0, sort=False)
            self.initial_results.per_timestamp_smape = pd.concat(
                [self.initial_results.per_timestamp_smape,
                 template_result.per_timestamp_smape],
                axis=0, sort=False)
            self.initial_results.model_results['Score'] = generate_score(self.initial_results.model_results, metric_weighting=metric_weighting, prediction_interval=prediction_interval)
            if result_file is not None:
                self.initial_results.model_results.to_csv(result_file,
                                                          index=False)

        # try ensembling
        if ensemble is not None:
            try:
                ensemble_templates = EnsembleTemplateGenerator(
                    self.initial_results, forecast_length=forecast_length,
                    ensemble=ensemble.replace('horizontal', ''),
                    subset_flag=self.subset_flag)

                template_result = TemplateWizard(
                    ensemble_templates, df_train, df_test,
                    weights=current_weights,
                    model_count=model_count,
                    forecast_length=forecast_length,
                    frequency=frequency,
                    prediction_interval=prediction_interval,
                    no_negatives=no_negatives,
                    ensemble=ensemble,
                    preord_regressor_train=preord_regressor_train,
                    preord_regressor_forecast=preord_regressor_test,
                    holiday_country=holiday_country,
                    startTimeStamps=profile_df.loc['FirstDate'],
                    template_cols=template_cols,
                    random_seed=random_seed, verbose=verbose)
                model_count = template_result.model_count
                # capture results from lower-level template run
                self.initial_results.model_results = pd.concat(
                    [self.initial_results.model_results,
                     template_result.model_results],
                    axis=0, ignore_index=True, sort=False
                    ).reset_index(drop=True)
                self.initial_results.per_series_mae = pd.concat(
                    [self.initial_results.per_series_mae,
                     template_result.per_series_mae],
                    axis=0, sort=False)
                self.initial_results.model_results['Score'] = generate_score(self.initial_results.model_results, metric_weighting=metric_weighting, prediction_interval=prediction_interval)
                if result_file is not None:
                    self.initial_results.model_results.to_csv(result_file,
                                                              index=False)
                if 'horizontal' in ensemble:
                    ensemble_templates = EnsembleTemplateGenerator(
                        self.initial_results, forecast_length=forecast_length,
                        ensemble=ensemble.replace('simple', '').replace('distance', ''),
                        subset_flag=self.subset_flag)
                    template_result = TemplateWizard(ensemble_templates,
                                                     df_train,
                                                     df_test,
                                                     weights=current_weights,
                                                     model_count=model_count,
                                                     forecast_length=forecast_length,
                                                     frequency=frequency,
                                                     prediction_interval=prediction_interval,
                                                     no_negatives=no_negatives,
                                                     preord_regressor_train=preord_regressor_train,
                                                     preord_regressor_forecast=preord_regressor_test,
                                                     holiday_country=holiday_country,
                                                     startTimeStamps=profile_df.loc['FirstDate'],
                                                     template_cols=template_cols,
                                                     random_seed=random_seed,
                                                     verbose=verbose)
                    model_count = template_result.model_count
                    # capture results from lower-level template run
                    self.initial_results.model_results = pd.concat(
                        [self.initial_results.model_results,
                         template_result.model_results],
                        axis=0, ignore_index=True, sort=False).reset_index(drop=True)
                    self.initial_results.model_results['Score'] = generate_score(self.initial_results.model_results, metric_weighting=metric_weighting, prediction_interval=prediction_interval)
                    if result_file is not None:
                        self.initial_results.model_results.to_csv(result_file,
                                                                  index=False)
            except Exception as e:
                print(f"Ensembling Error: {e}")

        # drop any duplicates in results
        self.initial_results.model_results = self.initial_results.model_results.drop_duplicates(subset = (['ID'] + self.template_cols))

        # validations if float
        if (self.models_to_validate < 1) and (self.models_to_validate > 0):
            temp_len = self.initial_results.model_results.shape[0]
            self.models_to_validate = self.models_to_validate * temp_len
            self.models_to_validate = int(np.ceil(self.models_to_validate))
        if (self.max_per_model_class is None):
            temp_len = len(self.model_list)
            self.max_per_model_class = (self.models_to_validate / temp_len) + 1
            self.max_per_model_class = int(np.ceil(self.max_per_model_class))

        # check how many validations are possible given the length of the data.
        num_validations = abs(int(num_validations))
        max_possible = len(df_wide_numeric.index)/forecast_length
        if (max_possible - np.floor(max_possible)) > self.min_allowed_train_percent:
            max_possible = int(max_possible)
        else:
            max_possible = int(max_possible) - 1
        if max_possible < (num_validations + 1):
            num_validations = max_possible - 1
            if num_validations < 0:
                num_validations = 0
            print("Too many training validations for length of data provided, decreasing num_validations to {}".format(num_validations))
        
        # construct validation template
        validation_template = self.initial_results.model_results[self.initial_results.model_results['Exceptions'].isna()]
        validation_template = validation_template.drop_duplicates(subset=template_cols, keep='first')
        validation_template = validation_template.sort_values(
            by="Score", ascending=True, na_position='last')
        if str(self.max_per_model_class).isdigit():
            validation_template = validation_template.sort_values('Score', ascending = True, na_position = 'last').groupby('Model').head(self.max_per_model_class).reset_index(drop=True)
        validation_template = validation_template.sort_values('Score', ascending = True, na_position = 'last').head(self.models_to_validate)
        validation_template = validation_template[self.template_cols]
        if not ensemble:
            validation_template = validation_template[validation_template['Ensemble'] == 0]

        # run validations
        """
        'Even' cuts the data into equal slices of the pie
        'Backwards' cuts the data backwards starting from the most recent data

        Both will look nearly identical on small datasets.
        Backwards is more recency focused on data with lots of history.
        """
        if num_validations > 0:
            model_count = 0
            for y in range(num_validations):
                if verbose > 0:
                    print("Validation Round: {}".format(str(y + 1)))
                # slice the validation data into current slice
                if self.validation_method == 'even':
                    # /num_validations biases it towards the last segment
                    validation_size = (len(df_wide_numeric.index) - forecast_length)
                    validation_size = validation_size/(num_validations + 1)
                    validation_size = int(np.floor(validation_size))
                    current_slice = df_wide_numeric.head(validation_size * (y+1) + forecast_length)
                elif str(self.validation_method).lower() in ['backwards', 'back', 'backward']:
                    # gradually remove the end
                    current_slice = df_wide_numeric.head(len(df_wide_numeric.index) - (y+1) * forecast_length)
                else:
                    raise ValueError("Validation Method not recognized try 'even', 'backwards'")

                # subset series (if used) and take a new train/test split
                if self.subset_flag:
                    df_subset = subset_series(current_slice, list((weights.get(i)) for i in current_slice.columns), n=self.subset, random_state=random_seed)
                    if self.verbose > 1:
                        print(f'{y + 1} subset is of: {df_subset.columns}')
                else:
                    df_subset = current_slice
                if not weighted:
                    current_weights = {x: 1 for x in df_subset.columns}
                else:
                    current_weights = {x: weights[x] for x in df_subset.columns}
                df_train, df_test = simple_train_test_split(
                    df_subset, forecast_length=forecast_length,
                    min_allowed_train_percent=self.min_allowed_train_percent,
                    verbose=self.verbose)
                if self.verbose > 2:
                    print(f'Validation index is {df_train.index}')

                # slice regressor into current validation slices
                try:
                    preord_regressor_train = preord_regressor.reindex(
                        index=df_train.index)
                    preord_regressor_test = preord_regressor.reindex(
                        index=df_test.index)
                except Exception:
                    preord_regressor_train = []
                    preord_regressor_test = []

                # run validation template on current slice
                template_result = TemplateWizard(
                    validation_template, df_train, df_test,
                    weights=current_weights,
                    model_count=model_count,
                    forecast_length=forecast_length,
                    frequency=frequency,
                    prediction_interval=prediction_interval,
                    no_negatives=no_negatives,
                    preord_regressor_forecast=preord_regressor_test,
                    holiday_country=holiday_country,
                    startTimeStamps=profile_df.loc['FirstDate'],
                    template_cols=template_cols,
                    random_seed=random_seed, verbose=verbose,
                    validation_round=(y + 1))
                model_count = template_result.model_count
                # gather results of template run
                self.initial_results.model_results = pd.concat(
                    [self.initial_results.model_results,
                     template_result.model_results],
                    axis=0, ignore_index=True,
                    sort=False).reset_index(drop=True)
                self.initial_results.model_results['Score'] = generate_score(self.initial_results.model_results, metric_weighting=metric_weighting, prediction_interval=prediction_interval)

        self.validation_results = copy.copy(self.initial_results)
        # aggregate validation results
        self.validation_results = validation_aggregation(
            self.validation_results)

        # store errors in separate dataframe
        val_errors = self.initial_results.model_results[
            ~self.initial_results.model_results['Exceptions'].isna()]
        self.error_templates = val_errors[template_cols + ['Exceptions']]
        
        # choose best model
        eligible_models = self.validation_results.model_results[self.validation_results.model_results['Runs'] >= (num_validations + 1)]
        try:
            self.best_model = eligible_models.sort_values(by = "Score", ascending = True, na_position = 'last').drop_duplicates(subset = self.template_cols).head(1)[template_cols]
            self.ensemble_check = (self.best_model['Ensemble'].iloc[0])
        except IndexError:
            raise ValueError("""No models available from validation.
 Try increasing models_to_validate, max_per_model_class
 or otherwise increase models available.""")

        # set flags to check if regressors or ensemble used in final model.
        param_dict = json.loads(self.best_model['ModelParameters'].iloc[0])
        if self.ensemble_check == 1:
            self.used_regressor_check = False
            for key in param_dict['models']:
                try:
                    reg_param = json.loads(param_dict['models'][key]['ModelParameters'])['regression_type']
                    if reg_param == 'User':
                        self.used_regressor_check = True
                except Exception:
                    pass
        if self.ensemble_check == 0:
            self.used_regressor_check = False
            try:
                reg_param = param_dict['ModelParameters']['regression_type']
                if reg_param == 'User':
                    self.used_regressor_check = True
            except Exception:
                pass
        return self
  
    def predict(self, forecast_length: int = "self",
                preord_regressor = [], hierarchy = None,
                just_point_forecast: bool = False):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            preord_regressor (numpy.Array): additional regressor, not used
            hierarchy: Not yet implemented
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Return:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        if forecast_length == 'self':
            forecast_length = self.forecast_length

        # if the models don't need the regressor, ignore it...
        if not self.used_regressor_check:
            preord_regressor = []
            self.preord_regressor_train = []
        else:
            preord_regressor = pd.DataFrame(preord_regressor)
            self.preord_regressor_train = self.preord_regressor_train.reindex(index=self.df_wide_numeric.index)

        df_forecast = PredictWitch(self.best_model,
                                   df_train=self.df_wide_numeric,
                                   forecast_length=forecast_length,
                                   frequency=self.frequency,
                                   prediction_interval=self.prediction_interval,
                                   no_negatives=self.no_negatives,
                                   preord_regressor_train=self.preord_regressor_train,
                                   preord_regressor_forecast=preord_regressor,
                                   holiday_country=self.holiday_country,
                                   startTimeStamps=self.startTimeStamps,
                                   random_seed=self.random_seed,
                                   verbose=self.verbose,
                                   template_cols=self.template_cols)

        df_forecast.forecast = categorical_inverse(self.categorical_transformer,
                                                   df_forecast.forecast)
        # df_forecast.lower_forecast = categorical_inverse(self.categorical_transformer, df_forecast.lower_forecast)
        # df_forecast.upper_forecast = categorical_inverse(self.categorical_transformer, df_forecast.upper_forecast)

        if just_point_forecast:
            return df_forecast.forecast
        else:
            return df_forecast

    def export_template(self, filename, models: str = 'best', n: int = 1,
                        max_per_model_class: int = None):
        """Export top results as a reusable template.

        Args:
            filename (str): 'csv' or 'json' (in filename)
            models (str): 'best' or 'all'
            n (int): if models = 'best', how many n-best to export
            max_per_model_class (int): if models = 'best', the max number of each model class to include in template
        """
        if models == 'all':
            export_template = self.initial_results[self.template_cols]
        if models == 'best':
            export_template = self.validation_results.model_results
            export_template = export_template[export_template['Runs'] >= (self.num_validations + 1)]
            if str(max_per_model_class).isdigit():
                export_template = export_template.sort_values('Score', ascending=True).groupby('Model').head(max_per_model_class).reset_index()
            export_template = export_template.nsmallest(n, columns = ['Score'])[self.template_cols]
        try:
            if '.csv' in filename:
                return export_template.to_csv(filename, index=False)
            if '.json' in filename:
                return export_template.to_json(filename, orient='columns')
        except PermissionError:
            raise PermissionError("Permission Error: directory or existing file is locked for editing.")
    def import_template(self, filename: str, method: str = "Add On"):
        """"
        "Hello. Would you like to destroy some evil today?" - Sanderson
        
        Args:
            filename (str): file location
            method (str): 'Add On' or 'Only'
        """
        if '.csv' in filename:
            import_template = pd.read_csv(filename)
        if '.json' in filename:
            import_template = pd.read_json(filename, orient='columns')

        try:
            import_template = import_template[self.template_cols]
        except Exception:
            print("Column names {} were not recognized as matching template columns: {}".format(str(import_template.columns), str(self.template_cols)))

        if method.lower() in ['add on', 'addon']:
            self.initial_template = self.initial_template.merge(import_template, on=self.initial_template.columns.intersection(import_template.columns).to_list())
            self.initial_template = self.initial_template.drop_duplicates(subset=self.template_cols)
        if method.lower() in ['only', 'user only']:
            self.initial_template = import_template
        return self

    def import_results(self, filename):
        """Add results from another run on the same data."""
        past_results = pd.read_csv(filename)
        past_results = past_results[pd.isnull(past_results['Exceptions'])]
        past_results['TotalRuntime'] = pd.to_timedelta(past_results['TotalRuntime'])
        self.initial_results.model_results = pd.concat(
            [past_results, self.initial_results.model_results],
            axis=0, ignore_index=True, sort=False).reset_index(drop=True)
        self.initial_results.model_results = self.initial_results.model_results.drop_duplicates(subset=self.template_cols, keep='first')
        return self

    def get_params(self):
        pass


def fake_regressor(df_long, forecast_length: int = 14,
                   date_col: str = 'datetime', value_col: str = 'value',
                   id_col: str = 'series_id',
                   frequency: str = 'infer', aggfunc: str = 'first',
                   drop_most_recent: int = 0, na_tolerance: float = 0.95,
                   drop_data_older_than_periods: int = 10000,
                   dimensions: int = 1):
    """Creates a fake regressor of random numbers for testing purposes."""

    from autots.tools.shaping import long_to_wide
    df_wide = long_to_wide(df_long, date_col = date_col, value_col = value_col,
                       id_col = id_col, frequency = frequency, na_tolerance = na_tolerance,
                       drop_data_older_than_periods = drop_data_older_than_periods, aggfunc = aggfunc,
                       drop_most_recent = drop_most_recent)
    if frequency == 'infer':
        frequency = pd.infer_freq(df_wide.index, warn=True)
    
        
    forecast_index = pd.date_range(freq = frequency, start = df_wide.index[-1], periods = forecast_length + 1)
    forecast_index = forecast_index[1:]
    
    if dimensions <= 1:
        preord_regressor_train = pd.Series(np.random.randint(0, 100, size = len(df_wide.index)), index = df_wide.index)
        preord_regressor_forecast = pd.Series(np.random.randint(0, 100, size = (forecast_length)), index = forecast_index)
    else:
        preord_regressor_train = pd.DataFrame(np.random.randint(0, 100, size = (len(df_wide.index), dimensions)), index = df_wide.index)
        preord_regressor_forecast = pd.DataFrame(np.random.randint(0, 100, size = (forecast_length, dimensions)), index = forecast_index)
    return preord_regressor_train, preord_regressor_forecast


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
    trans_df = pd.io.json.json_normalize(trans_df)  # .fillna(value='NaN')
    trans_cols1 = trans_df.columns
    trans_df = trans_df.astype(str).replace('nan', 'NaNZ')
    trans_transformer = OneHotEncoder(sparse=False).fit(trans_df)
    trans_df = pd.DataFrame(trans_transformer.transform(trans_df))
    trans_cols = np.array([x1 + x2 for x1, x2 in zip(
        trans_cols1, trans_transformer.categories_)])
    trans_cols = [item for sublist in trans_cols for item in sublist]
    trans_df.columns = trans_cols

    model_df = all_results['ModelParameters'].apply(json.loads)
    model_df = pd.io.json.json_normalize(model_df)  # .fillna(value='NaN')
    model_cols1 = model_df.columns
    model_df = model_df.astype(str).replace('nan', 'NaNZ')
    model_transformer = OneHotEncoder(sparse=False).fit(model_df)
    model_df = pd.DataFrame(model_transformer.transform(model_df))
    model_cols = np.array([x1 + x2 for x1, x2 in zip(
        model_cols1, model_transformer.categories_)])
    model_cols = [item for sublist in model_cols for item in sublist]
    model_df.columns = model_cols

    modelstr_df = all_results['Model']
    modelstr_transformer = OneHotEncoder(sparse=False).fit(
        modelstr_df.values.reshape(-1, 1))
    modelstr_df = pd.DataFrame(modelstr_transformer.transform(
        modelstr_df.values.reshape(-1, 1)))
    modelstr_df.columns = modelstr_transformer.categories_[0]

    except_df = all_results['Exceptions'].copy()
    except_df = except_df.where(except_df.duplicated(), 'UniqueError')
    except_transformer = OneHotEncoder(sparse=False).fit(
        except_df.values.reshape(-1, 1))
    except_df = pd.DataFrame(except_transformer.transform(
        except_df.values.reshape(-1, 1)))
    except_df.columns = except_transformer.categories_[0]

    test = pd.concat([except_df, all_results[['ExceptionFlag']],
                      modelstr_df, model_df, trans_df], axis=1)
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
