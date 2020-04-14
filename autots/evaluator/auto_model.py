"""Mid-level helper functions for AutoTS."""
import numpy as np
import pandas as pd
import datetime
import json
from hashlib import md5
from autots.evaluator.metrics import PredictionEval


def seasonal_int(include_one: bool = False):
    """Generate a random integer of typical seasonalities."""
    if include_one:
        lag = np.random.choice(
            a=['random_int', 1, 2, 4, 7, 10, 12, 24, 28,
               60, 96, 364, 1440, 420, 52, 84],
            size=1,
            p=[0.10, 0.05, 0.05, 0.05, 0.15, 0.01, 0.1, 0.1,
               0.1, 0.1, 0.05, 0.1, 0.01, 0.01, 0.01, 0.01]).item()
    else:
        lag = np.random.choice(
            a=['random_int', 2, 4, 7, 10, 12, 24, 28,
               60, 96, 364, 1440, 420, 52, 84],
            size=1, p=[0.15, 0.05, 0.05, 0.15, 0.01, 0.1, 0.1, 0.1,
                       0.1, 0.05, 0.1, 0.01, 0.01, 0.01, 0.01]).item()
    if lag == 'random_int':
        lag = np.random.randint(2, 100, size=1).item()
    return int(lag)


def create_model_id(model_str: str, parameter_dict: dict = {},
                    transformation_dict: dict = {}):
    """Create a hash ID which should be unique to the model parameters."""
    str_repr = str(model_str) + json.dumps(parameter_dict) + json.dumps(transformation_dict)
    str_repr = ''.join(str_repr.split())
    hashed = md5(str_repr.encode('utf-8')).hexdigest()
    return hashed


class ModelObject(object):
    """Generic class for holding forecasting models.
    
    Models should all have methods:
        .fit(df) (taking a DataFrame with DatetimeIndex and n columns of n timeseries)
        .predict(forecast_length = int, regressor)
        .get_new_params(method)
    
    Args:
        name (str): Model Name
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
    """

    def __init__(self, name: str = "Uninitiated Model Name",
                 frequency: str = 'infer',
                 prediction_interval: float = 0.9, regression_type: str = None,
                 fit_runtime=datetime.timedelta(0), holiday_country: str ='US',
                 random_seed: int = 2020, verbose: int = 0):
        self.name = name
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.regression_type = regression_type
        self.fit_runtime = fit_runtime
        self.holiday_country = holiday_country
        self.random_seed = random_seed
        self.verbose = verbose
        self.verbose_bool = True if self.verbose > 1 else False
    
    def __repr__(self):
        """Print."""
        return 'ModelObject of ' + self.name + ' uses standard .fit/.predict'
    
    def basic_profile(self, df):
        """Capture basic training details."""
        self.startTime = datetime.datetime.now()
        self.train_shape = df.shape
        self.column_names = df.columns
        self.train_last_date = df.index[-1]
        if self.frequency == 'infer':
            self.frequency = pd.infer_freq(df.index, warn=False)

        return df

    def create_forecast_index(self, forecast_length: int):
        """Generate a pd.DatetimeIndex appropriate for a new forecast.

        Warnings:
            Requires ModelObject.basic_profile() being called as part of .fit()
        """
        forecast_index = pd.date_range(freq=self.frequency,
                                       start=self.train_last_date,
                                       periods=forecast_length + 1)
        forecast_index = forecast_index[1:]
        self.forecast_index = forecast_index
        return forecast_index
    
    def get_params(self):
        """Return dict of current parameters."""
        return {}
    
    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        return {}


class PredictionObject(object):
    """Generic class for holding forecast information."""

    def __init__(self, model_name: str = 'Uninitiated',
                 forecast_length: int = 0,
                 forecast_index=np.nan, forecast_columns=np.nan,
                 lower_forecast=np.nan, forecast=np.nan, upper_forecast=np.nan,
                 prediction_interval: float = 0.9,
                 predict_runtime=datetime.timedelta(0),
                 fit_runtime=datetime.timedelta(0),
                 model_parameters={}, transformation_parameters={},
                 transformation_runtime=datetime.timedelta(0)):
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.transformation_parameters = transformation_parameters
        self.forecast_length = forecast_length
        self.forecast_index = forecast_index
        self.forecast_columns = forecast_columns
        self.lower_forecast = lower_forecast
        self.forecast = forecast
        self.upper_forecast = upper_forecast
        self.prediction_interval = prediction_interval
        self.predict_runtime = predict_runtime
        self.fit_runtime = fit_runtime
        self.transformation_runtime = transformation_runtime

    def __repr__(self):
        """Print."""
        if self.forecast != np.nan:
            return "Prediction object: \nReturn .forecast, \n .upper_forecast, \n .lower_forecast \n .model_parameters \n .transformation_parameters"
        else:
            return "Empty prediction object."

    def total_runtime(self):
        """Combine runtimes."""
        return self.fit_runtime + self.predict_runtime + self.transformation_runtime


def ModelMonster(model: str, parameters: dict = {}, frequency: str = 'infer',
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 startTimeStamps = None, forecast_length: int = 14,
                 random_seed: int = 2020, verbose: int = 0):
    """Directs strings and parameters to appropriate model objects.

    Args:
        model (str): Name of Model Function
        parameters (dict): Dictionary of parameters to pass through to model
    """
    model = str(model)

    if model == 'ZeroesNaive':
        from autots.models.basics import ZeroesNaive
        return ZeroesNaive(frequency=frequency,
                           prediction_interval=prediction_interval)

    if model == 'LastValueNaive':
        from autots.models.basics import LastValueNaive
        return LastValueNaive(frequency=frequency,
                              prediction_interval=prediction_interval)

    if model == 'AverageValueNaive':
        from autots.models.basics import AverageValueNaive
        if parameters == {}:
            return AverageValueNaive(frequency=frequency,
                                     prediction_interval=prediction_interval)
        else:
            return AverageValueNaive(frequency=frequency,
                                     prediction_interval=prediction_interval,
                                     method=parameters['method'])
    if model == 'SeasonalNaive':
        from autots.models.basics import SeasonalNaive
        if parameters == {}:
            return SeasonalNaive(frequency=frequency,
                                 prediction_interval=prediction_interval)
        else:
            return SeasonalNaive(frequency=frequency,
                                 prediction_interval=prediction_interval,
                                 method=parameters['method'],
                                 lag_1=parameters['lag_1'],
                                 lag_2=parameters['lag_2'])
    
    if model == 'GLS':
        from autots.models.statsmodels import GLS
        return GLS(frequency=frequency,
                   prediction_interval=prediction_interval)
    
    if model == 'GLM':
        from autots.models.statsmodels import GLM
        if parameters == {}:
            model = GLM(frequency=frequency,
                        prediction_interval=prediction_interval,
                        holiday_country=holiday_country,
                        random_seed=random_seed, verbose=verbose)
        else:
            model = GLM(frequency=frequency,
                        prediction_interval=prediction_interval,
                        holiday_country=holiday_country,
                        random_seed=random_seed, verbose=verbose,
                        family=parameters['family'],
                        constant=parameters['constant'],
                        regression_type=parameters['regression_type'])
        return model
    
    if model == 'ETS':
        from autots.models.statsmodels import ETS
        if parameters == {}:
            model = ETS(frequency=frequency,
                        prediction_interval=prediction_interval,
                        random_seed=random_seed, verbose=verbose)
        else:
            model = ETS(frequency=frequency,
                        prediction_interval=prediction_interval,
                        damped=parameters['damped'],
                        trend=parameters['trend'],
                        seasonal=parameters['seasonal'],
                        seasonal_periods=parameters['seasonal_periods'],
                        random_seed=random_seed, verbose=verbose)
        return model
    
    if model == 'ARIMA':
        from autots.models.statsmodels import ARIMA
        if parameters == {}:
            model = ARIMA(frequency=frequency,
                          prediction_interval=prediction_interval,
                          holiday_country=holiday_country,
                          random_seed=random_seed, verbose=verbose)
        else:
            model = ARIMA(frequency=frequency,
                          prediction_interval=prediction_interval,
                          holiday_country=holiday_country, p=parameters['p'],
                          d=parameters['d'], q=parameters['q'],
                          regression_type=parameters['regression_type'],
                          random_seed=random_seed, verbose=verbose)
        return model
    
    if model == 'FBProphet':
        from autots.models.prophet import FBProphet
        if parameters == {}:
            model = FBProphet(frequency=frequency,
                              prediction_interval=prediction_interval,
                              holiday_country=holiday_country,
                              random_seed=random_seed, verbose=verbose)
        else:
            model = FBProphet(frequency=frequency,
                              prediction_interval=prediction_interval,
                              holiday_country=holiday_country,
                              holiday=parameters['holiday'],
                              regression_type=parameters['regression_type'],
                              random_seed=random_seed, verbose=verbose)
        return model

    if model == 'RollingRegression':
        from autots.models.sklearn import RollingRegression
        if parameters == {}:
            model = RollingRegression(frequency=frequency,
                                      prediction_interval=prediction_interval,
                                      holiday_country=holiday_country,
                                      random_seed=random_seed, verbose=verbose)
        else:
            model = RollingRegression(
                frequency=frequency,
                prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                holiday=parameters['holiday'],
                regression_type=parameters['regression_type'],
                random_seed=random_seed, verbose=verbose,
                regression_model=parameters['regression_model'],
                mean_rolling_periods=parameters['mean_rolling_periods'],
                std_rolling_periods=parameters['std_rolling_periods'],
                macd_periods=parameters['macd_periods'],
                max_rolling_periods=parameters['max_rolling_periods'],
                min_rolling_periods=parameters['min_rolling_periods'],
                ewm_alpha=parameters['ewm_alpha'],
                additional_lag_periods=parameters['additional_lag_periods'],
                x_transform=parameters['x_transform'],
                rolling_autocorr_periods=parameters['rolling_autocorr_periods'],
                abs_energy=parameters['abs_energy'],
                add_date_part=parameters['add_date_part'],
                polynomial_degree=parameters['polynomial_degree']
                )
        return model

    if model == 'UnobservedComponents':
        from autots.models.statsmodels import UnobservedComponents
        if parameters == {}:
            model = UnobservedComponents(
                frequency=frequency,
                prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                random_seed=random_seed,
                verbose=verbose)
        else:
            model = UnobservedComponents(
                frequency=frequency,
                prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                regression_type=parameters['regression_type'],
                random_seed=random_seed,
                verbose=verbose,
                level=parameters['level'],
                trend=parameters['trend'],
                cycle=parameters['cycle'],
                damped_cycle=parameters['damped_cycle'],
                irregular=parameters['irregular'],
                stochastic_trend=parameters['stochastic_trend'],
                stochastic_level=parameters['stochastic_level'],
                stochastic_cycle=parameters['stochastic_cycle'])
        return model

    if model == 'DynamicFactor':
        from autots.models.statsmodels import DynamicFactor
        if parameters == {}:
            model = DynamicFactor(frequency=frequency,
                                  prediction_interval=prediction_interval,
                                  holiday_country=holiday_country,
                                  random_seed=random_seed, verbose=verbose)
        else:
            model = DynamicFactor(frequency=frequency,
                                  prediction_interval=prediction_interval,
                                  holiday_country=holiday_country,
                                  regression_type=parameters['regression_type'],
                                  random_seed=random_seed, verbose=verbose,
                                  k_factors=parameters['k_factors'],
                                  factor_order=parameters['factor_order'])
        return model
    
    if model == 'VECM':
        from autots.models.statsmodels import VECM
        if parameters == {}:
            model = VECM(frequency=frequency,
                         prediction_interval=prediction_interval,
                         holiday_country=holiday_country,
                         random_seed=random_seed, verbose=verbose)
        else:
            model = VECM(frequency=frequency,
                         prediction_interval=prediction_interval,
                         holiday_country=holiday_country,
                         regression_type=parameters['regression_type'],
                         random_seed=random_seed, verbose=verbose,
                         deterministic=parameters['deterministic'],
                         k_ar_diff=parameters['k_ar_diff'])
        return model
    
    if model == 'VARMAX':
        from autots.models.statsmodels import VARMAX
        if parameters == {}:
            model = VARMAX(frequency=frequency,
                           prediction_interval=prediction_interval,
                           holiday_country=holiday_country,
                           random_seed=random_seed, verbose=verbose)
        else:
            model = VARMAX(frequency=frequency,
                           prediction_interval=prediction_interval,
                           holiday_country=holiday_country,
                           random_seed=random_seed, verbose=verbose,
                           order=parameters['order'],
                           trend=parameters['trend'])
        return model
    
    if model == 'GluonTS':
        from autots.models.gluonts import GluonTS
        if parameters == {}:
            model = GluonTS(frequency=frequency,
                            prediction_interval=prediction_interval,
                            holiday_country=holiday_country,
                            random_seed=random_seed, verbose=verbose,
                            forecast_length=forecast_length)
        else:
            model = GluonTS(frequency=frequency,
                            prediction_interval=prediction_interval,
                            holiday_country=holiday_country,
                            random_seed=random_seed, verbose=verbose,
                            gluon_model=parameters['gluon_model'],
                            epochs=parameters['epochs'],
                            learning_rate=parameters['learning_rate'],
                            forecast_length=forecast_length)
        return model
    
    if model == 'TSFreshRegressor':
        from autots.models.tsfresh import TSFreshRegressor
        if parameters == {}:
            model = TSFreshRegressor(frequency=frequency,
                                     prediction_interval=prediction_interval,
                                     holiday_country=holiday_country,
                                     random_seed=random_seed, verbose=verbose)
        else:
            model = TSFreshRegressor(frequency=frequency,
                                     prediction_interval=prediction_interval,
                                     holiday_country=holiday_country,
                                     random_seed=random_seed, verbose=verbose,
                                     max_timeshift=parameters['max_timeshift'],
                                     regression_model=parameters['regression_model'],
                                     feature_selection=parameters['feature_selection'])
        return model
    
    if model == 'MotifSimulation':
        from autots.models.basics import MotifSimulation
        if parameters == {}:
            model = MotifSimulation(frequency=frequency,
                                    prediction_interval=prediction_interval,
                                    holiday_country=holiday_country,
                                    random_seed=random_seed, verbose=verbose)
        else:
            model = MotifSimulation(frequency=frequency,
                                    prediction_interval=prediction_interval,
                                    holiday_country=holiday_country,
                                    random_seed=random_seed, verbose=verbose,
                                    phrase_len=parameters['phrase_len'],
                                    comparison=parameters['comparison'],
                                    shared=parameters['shared'],
                                    distance_metric=parameters['distance_metric'],
                                    max_motifs=parameters['max_motifs'],
                                    recency_weighting=parameters['recency_weighting'],
                                    cutoff_threshold=parameters['cutoff_threshold'],
                                    cutoff_minimum=parameters['cutoff_minimum'],
                                    point_method=parameters['point_method'])
        return model
    else:
        raise AttributeError(("Model String '{}' not a recognized model type").format(model))

def ModelPrediction(df_train, forecast_length: int, transformation_dict: dict, 
                    model_str: str, parameter_dict: dict,
                    frequency: str = 'infer',
                    prediction_interval: float = 0.9,
                    no_negatives: bool = False,
                    preord_regressor_train = [],
                    preord_regressor_forecast = [],
                    holiday_country: str = 'US', startTimeStamps = None,
                    random_seed: int = 2020, verbose: int = 0):
    """Feed parameters into modeling pipeline
    
    Args:
        df_train (pandas.DataFrame): numeric training dataset of DatetimeIndex and series as cols
        forecast_length (int): number of periods to forecast
        transformation_dict (dict): a dictionary of outlier, fillNA, and transformation methods to be used
        model_str (str): a string to be direct to the appropriate model, used in ModelMonster
        frequency (str): str representing frequency alias of time series
        prediction_interval (float): width of errors (note: rarely do the intervals accurately match the % asked for...)
        no_negatives (bool): whether to force all forecasts to be > 0
        preord_regressor_train (pd.Series): with datetime index, of known in advance data, section matching train data
        preord_regressor_forecast (pd.Series): with datetime index, of known in advance data, section matching test data
        holiday_country (str): passed through to holiday package, used by a few models as 0/1 regressor.            
        startTimeStamps (pd.Series): index (series_ids), columns (Datetime of First start of series)
        
    Returns:
        PredictionObject (autots.PredictionObject): Prediction from AutoTS model object
    """
    transformationStartTime = datetime.datetime.now()
    from autots.tools.transform import GeneralTransformer
    transformer_object = GeneralTransformer(
        outlier_method=transformation_dict['outlier_method'],
        outlier_threshold=transformation_dict['outlier_threshold'],
        outlier_position=transformation_dict['outlier_position'],
        fillna=transformation_dict['fillna'],
        transformation=transformation_dict['transformation'],
        detrend=transformation_dict['detrend'],
        second_transformation=transformation_dict['second_transformation'],
        transformation_param=transformation_dict['transformation_param'],
        third_transformation=transformation_dict['third_transformation'],
        discretization=transformation_dict['discretization'],
        n_bins=transformation_dict['n_bins']
                                            ).fit(df_train)
    df_train_transformed = transformer_object.transform(df_train)
    
    # slice the context, ie shorten the amount of data available.
    if transformation_dict['context_slicer'] not in [None, 'None']:
        from autots.tools.transform import simple_context_slicer
        df_train_transformed = simple_context_slicer(df_train_transformed, method = transformation_dict['context_slicer'], forecast_length = forecast_length)
    
    # make sure regressor has same length. This could be a problem if wrong size regressor is passed.
    if len(preord_regressor_train) > 0:
        preord_regressor_train = preord_regressor_train.tail(df_train_transformed.shape[0])
    
    transformation_runtime = datetime.datetime.now() - transformationStartTime
    # from autots.evaluator.auto_model import ModelMonster
    model = ModelMonster(model_str, parameters=parameter_dict,
                         frequency = frequency,
                         prediction_interval=prediction_interval,
                         holiday_country=holiday_country,
                         random_seed=random_seed, verbose=verbose,
                         forecast_length = forecast_length)
    model = model.fit(df_train_transformed,
                      preord_regressor = preord_regressor_train)
    df_forecast = model.predict(forecast_length = forecast_length,
                                preord_regressor = preord_regressor_forecast)
    
    if df_forecast.forecast.isnull().all(axis = 0).astype(int).sum() > 0:
        raise ValueError("Model {} returned NaN for one or more series".format(model_str))
    
    transformationStartTime = datetime.datetime.now()
    # Inverse the transformations
    df_forecast.forecast = pd.DataFrame(transformer_object.inverse_transform(df_forecast.forecast))#, index = df_forecast.forecast_index, columns = df_forecast.forecast_columns)
    df_forecast.lower_forecast = pd.DataFrame(transformer_object.inverse_transform(df_forecast.lower_forecast))# , index = df_forecast.forecast_index, columns = df_forecast.forecast_columns)
    df_forecast.upper_forecast = pd.DataFrame(transformer_object.inverse_transform(df_forecast.upper_forecast)) #, index = df_forecast.forecast_index, columns = df_forecast.forecast_columns)
    
    df_forecast.transformation_parameters = transformation_dict
    # Remove negatives if desired
    # There's df.where(df_forecast.forecast > 0, 0) or  df.clip(lower = 0), not sure which faster
    if no_negatives:
        df_forecast.lower_forecast = df_forecast.lower_forecast.clip(lower = 0)
        df_forecast.forecast = df_forecast.forecast.clip(lower = 0)
        df_forecast.upper_forecast = df_forecast.upper_forecast.clip(lower = 0)
    transformation_runtime = transformation_runtime + (datetime.datetime.now() - transformationStartTime)
    df_forecast.transformation_runtime = transformation_runtime
    
    return df_forecast

class TemplateEvalObject(object):
    """Object to contain all your failures!
    """
    def __init__(self, model_results = pd.DataFrame(), model_results_per_timestamp_smape=pd.DataFrame(),
                 model_results_per_timestamp_mae=pd.DataFrame(), model_results_per_series_mae =pd.DataFrame(),
                 model_results_per_series_smape = pd.DataFrame(), forecasts= [],
                 upper_forecasts = [], lower_forecasts=[], forecasts_list = [],
                 forecasts_runtime = [], model_count: int = 0
                 ):
        self.model_results = model_results
        self.model_results_per_timestamp_smape = model_results_per_timestamp_smape
        self.model_results_per_timestamp_mae = model_results_per_timestamp_mae
        self.model_results_per_series_mae = model_results_per_series_mae
        self.model_results_per_series_smape = model_results_per_series_smape
        self.forecasts = forecasts
        self.upper_forecasts = upper_forecasts
        self.lower_forecasts = lower_forecasts
        self.forecasts_list = forecasts_list
        self.forecasts_runtime = forecasts_runtime
        self.model_count = model_count

def unpack_ensemble_models(template, 
                           template_cols: list = ['Model','ModelParameters','TransformationParameters','Ensemble'],
                           keep_ensemble: bool = True):
    """
    Takes ensemble models from a template and returns template + ensemble models
    """
    ensemble_template = pd.DataFrame()
    for index, value in template[template['Ensemble'] == 1]['ModelParameters'].iteritems():
        model_dict = json.loads(value)['models']
        model_df = pd.DataFrame.from_dict(model_dict, orient='index')
        model_df = model_df.rename_axis('ID').reset_index(drop = False)
        model_df['Ensemble'] = 0
        ensemble_template = pd.concat([ensemble_template, model_df], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
    
    template = pd.concat([template, ensemble_template], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
    template = template.drop_duplicates(subset = template_cols)
    if not keep_ensemble:
        template = template[template['Ensemble'] == 0]
    return template

def PredictWitch(template, df_train,forecast_length: int,
                    frequency: str = 'infer', 
                    prediction_interval: float = 0.9, no_negatives: bool = False,
                    preord_regressor_train = [], preord_regressor_forecast = [], 
                    holiday_country: str = 'US', startTimeStamps = None,
                    random_seed: int = 2020, verbose: int = 0,
                    template_cols: list = ['Model','ModelParameters','TransformationParameters','Ensemble']):
    """
    Takes numeric data, returns numeric forecasts.
    Only one model (albeit potentially an ensemble)!
    
    Well, she turned me into a newt.
    A newt?
    I got better. -Python

    Args:
        df_train (pandas.DataFrame): numeric training dataset of DatetimeIndex and series as cols
        forecast_length (int): number of periods to forecast
        transformation_dict (dict): a dictionary of outlier, fillNA, and transformation methods to be used
        model_str (str): a string to be direct to the appropriate model, used in ModelMonster
        frequency (str): str representing frequency alias of time series
        prediction_interval (float): width of errors (note: rarely do the intervals accurately match the % asked for...)
        no_negatives (bool): whether to force all forecasts to be > 0
        preord_regressor_train (pd.Series): with datetime index, of known in advance data, section matching train data
        preord_regressor_forecast (pd.Series): with datetime index, of known in advance data, section matching test data
        holiday_country (str): passed through to holiday package, used by a few models as 0/1 regressor.            
        startTimeStamps (pd.Series): index (series_ids), columns (Datetime of First start of series)
        template_cols (list): column names of columns used as model template
        
    Returns:
        PredictionObject (autots.PredictionObject): Prediction from AutoTS model object):
    """
    if isinstance(template, pd.Series):
        template = pd.DataFrame(template).transpose()
    template = template.head(1) 
    for index_upper, row_upper in template.iterrows():
        # if an ensemble
        if row_upper['Ensemble'] == 1:
            from autots.models.ensemble import EnsembleForecast
            forecasts_list = []
            forecasts_runtime = []
            forecasts = []
            upper_forecasts = []
            lower_forecasts = []
            ens_model_str = row_upper['Model']
            ens_params = json.loads(row_upper['ModelParameters'])
            ens_template = unpack_ensemble_models(template, template_cols, keep_ensemble = False)
            for index, row in ens_template.iterrows():
                model_str = row['Model']
                parameter_dict = json.loads(row['ModelParameters'])
                transformation_dict = json.loads(row['TransformationParameters'])

                df_forecast = ModelPrediction(df_train, forecast_length,transformation_dict, 
                                              model_str, parameter_dict, frequency=frequency, 
                                              prediction_interval=prediction_interval, 
                                              no_negatives=no_negatives,
                                              preord_regressor_train = preord_regressor_train,
                                              preord_regressor_forecast = preord_regressor_forecast, 
                                              holiday_country=holiday_country,
                                              startTimeStamps = startTimeStamps,
                                              random_seed=random_seed, verbose=verbose)
                model_id = create_model_id(df_forecast.model_name, df_forecast.model_parameters, df_forecast.transformation_parameters)
                total_runtime = df_forecast.fit_runtime + df_forecast.predict_runtime + df_forecast.transformation_runtime

                forecasts_list.extend([model_id])
                forecasts_runtime.extend([total_runtime])
                forecasts.extend([df_forecast.forecast])
                upper_forecasts.extend([df_forecast.upper_forecast])
                lower_forecasts.extend([df_forecast.lower_forecast])
            ens_forecast = EnsembleForecast(ens_model_str, ens_params, forecasts_list=forecasts_list, forecasts=forecasts, 
                                            lower_forecasts=lower_forecasts, upper_forecasts=upper_forecasts, forecasts_runtime=forecasts_runtime, prediction_interval=prediction_interval)
            return ens_forecast
        # if not an ensemble
        else:
            model_str = row_upper['Model']
            parameter_dict = json.loads(row_upper['ModelParameters'])
            transformation_dict = json.loads(row_upper['TransformationParameters'])
            
            df_forecast = ModelPrediction(df_train, forecast_length,transformation_dict, 
                                          model_str, parameter_dict, frequency=frequency, 
                                          prediction_interval=prediction_interval, 
                                          no_negatives=no_negatives,
                                          preord_regressor_train = preord_regressor_train,
                                          preord_regressor_forecast = preord_regressor_forecast, 
                                          holiday_country=holiday_country,
                                          random_seed=random_seed, verbose=verbose,
                                          startTimeStamps = startTimeStamps)
    
            return df_forecast
   
def TemplateWizard(template, df_train, df_test, weights,
                   model_count: int = 0, ensemble: bool = True,
                   forecast_length: int = 14, frequency: str = 'infer', 
                    prediction_interval: float = 0.9, no_negatives: bool = False,
                    preord_regressor_train = [], preord_regressor_forecast = [], 
                    holiday_country: str = 'US', startTimeStamps = None,
                    random_seed: int = 2020, verbose: int = 0,
                    per_timestamp_errors: bool = False,
                    per_series_errors: bool = True,
                    template_cols: list = ['Model','ModelParameters','TransformationParameters','Ensemble']):

    """
    takes Template, returns Results
    
    There are some who call me... Tim. - Python
    
    Args:
        template (pandas.DataFrame): containing model str, and json of transformations and hyperparamters
        df_train (pandas.DataFrame): numeric training dataset of DatetimeIndex and series as cols
        df_test (pandas.DataFrame): dataframe of actual values of (forecast length * n series)
        weights (dict): key = column/series_id, value = weight
        
        forecast_length (int): number of periods to forecast
        transformation_dict (dict): a dictionary of outlier, fillNA, and transformation methods to be used
        model_str (str): a string to be direct to the appropriate model, used in ModelMonster
        frequency (str): str representing frequency alias of time series
        prediction_interval (float): width of errors (note: rarely do the intervals accurately match the % asked for...)
        no_negatives (bool): whether to force all forecasts to be > 0
        preord_regressor_train (pd.Series): with datetime index, of known in advance data, section matching train data
        preord_regressor_forecast (pd.Series): with datetime index, of known in advance data, section matching test data
        holiday_country (str): passed through to holiday package, used by a few models as 0/1 regressor.            
        startTimeStamps (pd.Series): index (series_ids), columns (Datetime of First start of series)
        template_cols (list): column names of columns used as model template
    
    Returns:
        TemplateEvalObject
    """
    template_result = TemplateEvalObject()
    template_result.model_count = model_count
    
    # template = unpack_ensemble_models(template, template_cols, keep_ensemble = False)
    
    for index in template.index:
        try:
            current_template = template.loc[index]
            model_str = current_template['Model']
            parameter_dict = json.loads(current_template['ModelParameters'])
            transformation_dict = json.loads(current_template['TransformationParameters'])
            ensemble_input = current_template['Ensemble']
            current_template = pd.DataFrame(current_template).transpose()
            template_result.model_count += 1
            if verbose > 0:
                if verbose > 1:
                    print("Model Number: {} with model {} with params {} and transformations {}".format(str(template_result.model_count), model_str, json.dumps(parameter_dict),json.dumps(transformation_dict)))
                else:
                    print("Model Number: {} with model {}".format(str(template_result.model_count), model_str))
            df_forecast = PredictWitch(current_template, df_train = df_train, forecast_length=forecast_length,frequency=frequency, 
                                          prediction_interval=prediction_interval, 
                                          no_negatives=no_negatives,
                                          preord_regressor_train = preord_regressor_train,
                                          preord_regressor_forecast = preord_regressor_forecast, 
                                          holiday_country=holiday_country,
                                          startTimeStamps = startTimeStamps,
                                          random_seed=random_seed, verbose=verbose,
                                       template_cols = template_cols)
            
            model_error = PredictionEval(df_forecast, df_test, series_weights = weights, per_timestamp_errors = per_timestamp_errors)
            model_id = create_model_id(df_forecast.model_name, df_forecast.model_parameters, df_forecast.transformation_parameters)
            total_runtime = df_forecast.fit_runtime + df_forecast.predict_runtime + df_forecast.transformation_runtime
            result = pd.DataFrame({
                    'ID': model_id,
                    'Model': df_forecast.model_name,
                    'ModelParameters': json.dumps(df_forecast.model_parameters),
                    'TransformationParameters': json.dumps(df_forecast.transformation_parameters),
                    'TransformationRuntime': df_forecast.transformation_runtime,
                    'FitRuntime': df_forecast.fit_runtime,
                    'PredictRuntime': df_forecast.predict_runtime,
                    'TotalRuntime': total_runtime,
                    'Ensemble': ensemble_input,
                    'Exceptions': np.nan,
                    'Runs': 1
                    }, index = [0])
            a = pd.DataFrame(model_error.avg_metrics_weighted.rename(lambda x: x + '_weighted')).transpose()
            result = pd.concat([result, pd.DataFrame(model_error.avg_metrics).transpose(), a], axis = 1)
            
            template_result.model_results = pd.concat([template_result.model_results, result], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
            if per_timestamp_errors:
                temp = pd.DataFrame(model_error.per_timestamp_metrics.loc['smape']).transpose()
                temp.index = result['ID'] 
                template_result.model_results_per_timestamp_smape = template_result.model_results_per_timestamp_smape.append(temp)
                temp = pd.DataFrame(model_error.per_timestamp_metrics.loc['mae']).transpose()
                temp.index = result['ID']  
                template_result.model_results_per_timestamp_mae = template_result.model_results_per_timestamp_mae.append(temp)
            if per_series_errors:    
                temp = pd.DataFrame(model_error.per_series_metrics.loc['smape']).transpose()
                temp.index = result['ID']            
                template_result.model_results_per_series_smape = template_result.model_results_per_series_smape.append(temp)
                temp = pd.DataFrame(model_error.per_series_metrics.loc['mae']).transpose()
                temp.index = result['ID']
                template_result.model_results_per_series_mae = template_result.model_results_per_series_mae.append(temp)
                
            if ensemble:
                template_result.forecasts_list.extend([model_id])
                template_result.forecasts_runtime.extend([total_runtime])
                template_result.forecasts.extend([df_forecast.forecast])
                template_result.upper_forecasts.extend([df_forecast.upper_forecast])
                template_result.lower_forecasts.extend([df_forecast.lower_forecast])
        
        except Exception as e:
            print('Template Eval Error: {} in model {}'.format(str(repr(e)), model_str))
            result = pd.DataFrame({
                'ID': create_model_id(model_str, parameter_dict, transformation_dict),
                'Model': model_str,
                'ModelParameters': json.dumps(parameter_dict),
                'TransformationParameters': json.dumps(transformation_dict),
                'Ensemble': ensemble_input,
                'TransformationRuntime': datetime.timedelta(0),
                'FitRuntime': datetime.timedelta(0),
                'PredictRuntime': datetime.timedelta(0),
                'TotalRuntime': datetime.timedelta(0),
                'Exceptions': str(e),
                'Runs': 1
                }, index = [0])
            template_result.model_results = pd.concat([template_result.model_results, result], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)

    return template_result 


from autots.tools.transform import RandomTransform
def RandomTemplate(n: int = 10, model_list: list = ['ZeroesNaive', 'LastValueNaive', 'AverageValueNaive', 'GLS',
              'GLM', 'ETS', 'ARIMA', 'FBProphet', 'RollingRegression', 'GluonTS',
              'UnobservedComponents', 'VARMAX', 'VECM', 'DynamicFactor']):
    """"
    Returns a template dataframe of randomly generated transformations, models, and hyperparameters
    
    Args:
        n (int): number of random models to return
    """
    n = abs(int(n))
    template = pd.DataFrame()
    counter = 0
    while (len(template.index) < n):
        model_str = np.random.choice(model_list)
        param_dict = ModelMonster(model_str).get_new_params()
        trans_dict = RandomTransform()
        row = pd.DataFrame({
                'Model': model_str,
                'ModelParameters': json.dumps(param_dict),
                'TransformationParameters': json.dumps(trans_dict),
                'Ensemble': 0
                }, index = [0])
        template = pd.concat([template, row], axis = 0, ignore_index = True)
        template.drop_duplicates(inplace = True)
        counter += 1
        if counter > (n * 3):
            break
    return template

def UniqueTemplates(existing_templates, new_possibilities, selection_cols: list = ['Model','ModelParameters','TransformationParameters','Ensemble']):
    """
    Returns unique dataframe rows from new_possiblities not in existing_templates
    
    Args:
        selection_cols (list): list of column namess to use to judge uniqueness/match on
    """
    keys = list(new_possibilities[selection_cols].columns.values)
    idx1 = existing_templates.copy().set_index(keys).index
    idx2 = new_possibilities.set_index(keys).index
    new_template = new_possibilities[~idx2.isin(idx1)]
    return new_template

def NewGeneticTemplate(model_results, submitted_parameters, sort_column: str = "smape_weighted", 
                       sort_ascending: bool = True, max_results: int = 40,
                       max_per_model_class: int = 5,
                       top_n: int = 15, template_cols: list = ['Model','ModelParameters','TransformationParameters','Ensemble']):
    """
    Returns new template given old template with model accuracies
    
    Args:
        model_results (pandas.DataFrame): models that have actually been run
        submitted_paramters (pandas.DataFrame): models tried (may have returned different parameters to results)
    
    """
    new_template = pd.DataFrame()
    
    # filter existing templates
    sorted_results =  model_results[model_results['Ensemble'] == 0].copy().sort_values(by = sort_column, ascending = sort_ascending, na_position = 'last').drop_duplicates(subset = template_cols, keep = 'first')
    if str(max_per_model_class).isdigit():
        sorted_results = sorted_results.sort_values(sort_column, ascending=sort_ascending).groupby('Model').head(max_per_model_class).reset_index()
    sorted_results = sorted_results.sort_values(by = sort_column, ascending = sort_ascending, na_position = 'last').head(top_n)
    
    # sorted_results = model_results[model_results['Ensemble'] == 0].copy().sort_values(by = sort_column, ascending = sort_ascending, na_position = 'last')
    # mutation
    for index, row in sorted_results.iterrows():
        param_dict = ModelMonster(row['Model']).get_new_params()
        trans_dict = RandomTransform()
        new_row = pd.DataFrame({
                'Model': row['Model'],
                'ModelParameters': json.dumps(param_dict),
                'TransformationParameters': row['TransformationParameters'],
                'Ensemble': 0
                }, index = [0])
        new_template = pd.concat([new_template, new_row], axis = 0, ignore_index = True, sort = False)
        new_row = pd.DataFrame({
                'Model': row['Model'],
                'ModelParameters': row['ModelParameters'],
                'TransformationParameters': json.dumps(trans_dict),
                'Ensemble': 0
                }, index = [0])
        new_template = pd.concat([new_template, new_row], axis = 0, ignore_index = True, sort = False)

    # recombination of transforms across models by shifting
    recombination = sorted_results.tail(len(sorted_results.index) - 1).copy()
    recombination['TransformationParameters'] = sorted_results['TransformationParameters'].shift(1).tail(len(sorted_results.index) - 1)
    new_template = pd.concat([new_template, recombination.head(top_n)[template_cols]], axis = 0, ignore_index = True, sort = False)
    
    # internal recombination of model parameters, not implemented because some options are mutually exclusive.
    # Recombine best two of each model, if two or more present
    
    # remove generated models which have already been tried
    sorted_results = pd.concat([submitted_parameters, sorted_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
    new_template = UniqueTemplates(sorted_results, new_template, selection_cols = template_cols).head(max_results)
    return new_template


def validation_aggregation(validation_results,
                           per_timestamp_errors: bool = False,
                           per_series_errors: bool = False):
    """Aggregate a TemplateEvalObject."""
    groupby_cols = ['ID', 'Model', 'ModelParameters',
                    'TransformationParameters', 'Ensemble']
    col_aggs = {'Runs': 'sum',
                # 'TransformationRuntime': 'mean',
                # 'FitRuntime': 'mean',
                # 'PredictRuntime': 'mean',
                'smape': 'mean',
                'mae': 'mean',
                'rmse': 'mean',
                'containment': 'mean',
                'lower_mae': 'mean',
                'upper_mae': 'mean',
                'contour': 'mean',
                'smape_weighted': 'mean',
                'mae_weighted': 'mean',
                'rmse_weighted': 'mean',
                'containment_weighted': 'mean',
                'TotalRuntimeSeconds': 'mean',
                'Score': 'mean'
                }
    validation_results.model_results['TotalRuntimeSeconds'] = validation_results.model_results['TotalRuntime'].dt.seconds + 1
    validation_results.model_results = validation_results.model_results[pd.isnull(validation_results.model_results['Exceptions'])]
    validation_results.model_results = validation_results.model_results.replace([np.inf, -np.inf], np.nan)
    validation_results.model_results = validation_results.model_results.groupby(groupby_cols).agg(col_aggs)
    validation_results.model_results = validation_results.model_results.reset_index(drop = False)
    if per_timestamp_errors:
        try:
            validation_results.model_results_per_timestamp_smape = validation_results.model_results_per_timestamp_smape.groupby('ID').mean()
            validation_results.model_results_per_timestamp_mae = validation_results.model_results_per_timestamp_mae.groupby('ID').mean()
        except KeyError:
            raise KeyError("Per_timestamp data not available. Make sure per_timestamp_errors = True")
    if per_series_errors:
        try:
            validation_results.model_results_per_series_smape = validation_results.model_results_per_series_smape.groupby('ID').mean()
            validation_results.model_results_per_series_mae = validation_results.model_results_per_series_mae.groupby('ID').mean()
        except KeyError:
            raise KeyError("Per_timestamp data not available. Make sure per_timestamp_errors = True")
    return validation_results


def generate_score(model_results, metric_weighting: dict = {},
                   prediction_interval: float = 0.9):
    """Generate score based on relative accuracies."""
    try:
        smape_weighting = metric_weighting['smape_weighting']
    except KeyError:
        smape_weighting = 1
    try:
        mae_weighting = metric_weighting['mae_weighting']
    except KeyError:
        mae_weighting = 0
    try:
        rmse_weighting = metric_weighting['rmse_weighting']
    except KeyError:
        rmse_weighting = 0
    try:
        containment_weighting = metric_weighting['containment_weighting']
    except KeyError:
        containment_weighting = 0
    try:
        runtime_weighting = metric_weighting['runtime_weighting'] * 0.1
    except KeyError:
        runtime_weighting = 0
    try:
        lower_mae_weighting = metric_weighting['lower_mae_weighting']
    except KeyError:
        lower_mae_weighting = 0
    try:
        upper_mae_weighting = metric_weighting['upper_mae_weighting']
    except KeyError:
        upper_mae_weighting = 0
    try:
        contour_weighting = metric_weighting['contour_weighting']
    except KeyError:
        contour_weighting = 0
    try:
        model_results = model_results.replace([np.inf, -np.inf], np.nan)
        # model_results = model_results.fillna(value = model_results.max(axis = 0))
        smape_score = model_results['smape_weighted']/(model_results['smape_weighted'].min(skipna=True) + 1) # smaller better
        rmse_score = model_results['rmse_weighted']/(model_results['rmse_weighted'].min(skipna=True) + 1) # smaller better
        mae_score = model_results['mae_weighted']/(model_results['mae_weighted'].min(skipna=True) + 1) # smaller better
        containment_score = (abs(prediction_interval - model_results['containment'])) # from 0 to 1, smaller better
        runtime_score = model_results['TotalRuntime']/(model_results['TotalRuntime'].min(skipna=True) + datetime.timedelta(minutes = 1)) # smaller better
        lower_mae_score = model_results['lower_mae_weighted']/(model_results['lower_mae_weighted'].min(skipna=True) +1) # smaller better
        upper_mae_score = model_results['upper_mae_weighted']/(model_results['upper_mae_weighted'].min(skipna=True) +1) # smaller better
        contour_score =  (1/(model_results['contour_weighted'])).replace([np.inf, -np.inf, np.nan], 10).clip(upper = 10)
    except KeyError:
        raise KeyError("Inconceivable! Evaluation Metrics are missing and all models have failed, by an error in TemplateWizard or metrics. A new template may help, or an adjusted model_list.")
        
    return (smape_score * smape_weighting) + (mae_score * mae_weighting) + (rmse_score * rmse_weighting) + (containment_score * containment_weighting) + (runtime_score * runtime_weighting) + (lower_mae_score * lower_mae_weighting) + (upper_mae_score * upper_mae_weighting) + (contour_score * contour_weighting)

