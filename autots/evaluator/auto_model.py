"""Mid-level helper functions for AutoTS."""
import numpy as np
import pandas as pd
import datetime
import json
from hashlib import md5
from autots.evaluator.metrics import PredictionEval
from autots.tools.transform import RandomTransform


def seasonal_int(include_one: bool = False):
    """Generate a random integer of typical seasonalities."""
    if include_one:
        lag = np.random.choice(
            a=['random_int', 1, 2, 4, 7, 10, 12, 24, 28,
               60, 96, 168, 364, 1440, 420, 52, 84],
            size=1,
            p=[0.10, 0.05, 0.05, 0.05, 0.15, 0.01, 0.1, 0.1,
               0.1, 0.1, 0.04, 0.01, 0.1, 0.01, 0.01, 0.01, 0.01]).item()
    else:
        lag = np.random.choice(
            a=['random_int', 2, 4, 7, 10, 12, 24, 28,
               60, 96, 168, 364, 1440, 420, 52, 84],
            size=1, p=[0.15, 0.05, 0.05, 0.15, 0.01, 0.1, 0.1, 0.1,
                       0.1, 0.04, 0.01, 0.1, 0.01, 0.01, 0.01, 0.01]).item()
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
        if isinstance(self.forecast, pd.DataFrame):
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

    if model == 'VAR':
        from autots.models.statsmodels import VAR
        if parameters == {}:
            model = VAR(frequency=frequency,
                        prediction_interval=prediction_interval,
                        holiday_country=holiday_country,
                        random_seed=random_seed, verbose=verbose)
        else:
            model = VAR(frequency=frequency,
                        prediction_interval=prediction_interval,
                        holiday_country=holiday_country,
                        regression_type=parameters['regression_type'],
                        maxlags=parameters['maxlags'],
                        ic=parameters['ic'],
                        random_seed=random_seed, verbose=verbose
                        )
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
    if model == 'WindowRegression':
        from autots.models.sklearn import WindowRegression
        if parameters == {}:
            model = WindowRegression(frequency=frequency,
                                     prediction_interval=prediction_interval,
                                     holiday_country=holiday_country,
                                     random_seed=random_seed, verbose=verbose,
                                     forecast_length=forecast_length)
        else:
            model = WindowRegression(
                frequency=frequency, prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                random_seed=random_seed, verbose=verbose,
                window_size=parameters['window_size'],
                regression_model=parameters['regression_model'],
                input_dim=parameters['input_dim'],
                output_dim=parameters['output_dim'],
                normalize_window=parameters['normalize_window'],
                shuffle=parameters['shuffle'],
                max_windows=parameters['max_windows'],
                forecast_length=forecast_length)
        return model
    if model == 'TensorflowSTS':
        from autots.models.tfp import TensorflowSTS
        if parameters == {}:
            model = TensorflowSTS(frequency=frequency,
                                  prediction_interval=prediction_interval,
                                  holiday_country=holiday_country,
                                  random_seed=random_seed, verbose=verbose)
        else:
            model = TensorflowSTS(frequency=frequency,
                                  prediction_interval=prediction_interval,
                                  holiday_country=holiday_country,
                                  random_seed=random_seed, verbose=verbose,
                                  seasonal_periods=parameters['seasonal_periods'],
                                  ar_order=parameters['ar_order'],
                                  trend=parameters['trend'],
                                  fit_method=parameters['fit_method'],
                                  num_steps=parameters['num_steps']
                                  )
        return model
    if model == 'TFPRegression':
        from autots.models.tfp import TFPRegression
        if parameters == {}:
            model = TFPRegression(frequency=frequency,
                                  prediction_interval=prediction_interval,
                                  holiday_country=holiday_country,
                                  random_seed=random_seed, verbose=verbose)
        else:
            model = TFPRegression(frequency=frequency,
                                  prediction_interval=prediction_interval,
                                  holiday_country=holiday_country,
                                  random_seed=random_seed, verbose=verbose,
                                  kernel_initializer=parameters['kernel_initializer'],
                                  epochs=parameters['epochs'],
                                  batch_size=parameters['batch_size'],
                                  optimizer=parameters['optimizer'],
                                  loss=parameters['loss'],
                                  dist=parameters['dist'],
                                  regression_type=parameters['regression_type']
                                  )
        return model
    if model == 'ComponentAnalysis':
        from autots.models.sklearn import ComponentAnalysis
        if parameters == {}:
            model = ComponentAnalysis(
                frequency=frequency,
                prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                random_seed=random_seed, verbose=verbose,
                forecast_length=forecast_length)
        else:
            model = ComponentAnalysis(
                frequency=frequency, prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                random_seed=random_seed, verbose=verbose,
                model=parameters['model'],
                model_parameters=parameters['model_parameters'],
                decomposition=parameters['decomposition'],
                n_components=parameters['n_components'],
                forecast_length=forecast_length)
        return model
    else:
        raise AttributeError(("Model String '{}' not a recognized model type").format(model))


def ModelPrediction(df_train, forecast_length: int, transformation_dict: dict, 
                    model_str: str, parameter_dict: dict,
                    frequency: str = 'infer',
                    prediction_interval: float = 0.9,
                    no_negatives: bool = False,
                    constraint: float = None,
                    future_regressor_train = [],
                    future_regressor_forecast = [],
                    holiday_country: str = 'US', startTimeStamps = None,
                    grouping_ids=None,
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
        constraint (float): when not None, use this value * data st dev above max or below min for constraining forecast values.
        future_regressor_train (pd.Series): with datetime index, of known in advance data, section matching train data
        future_regressor_forecast (pd.Series): with datetime index, of known in advance data, section matching test data
        holiday_country (str): passed through to holiday package, used by a few models as 0/1 regressor.            
        startTimeStamps (pd.Series): index (series_ids), columns (Datetime of First start of series)

    Returns:
        PredictionObject (autots.PredictionObject): Prediction from AutoTS model object
    """
    transformationStartTime = datetime.datetime.now()
    from autots.tools.transform import GeneralTransformer
    try:
        coerce_integer = transformation_dict['coerce_integer']
        grouping = transformation_dict['grouping']
        if grouping == 'user' and grouping_ids is None:
            grouping = 'kmeans5'
            transformation_dict['grouping'] = 'kmeans5'
        reconciliation = transformation_dict['reconciliation']
    except Exception:
        coerce_integer = False
        grouping = None
        grouping_ids = None
        reconciliation = None
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
        transformation_param2=transformation_dict['transformation_param2'],
        fourth_transformation=transformation_dict['fourth_transformation'],
        discretization=transformation_dict['discretization'],
        n_bins=transformation_dict['n_bins'],
        grouping=grouping,
        grouping_ids=grouping_ids,
        reconciliation=reconciliation,
        coerce_integer=coerce_integer
                                            ).fit(df_train)
    df_train_transformed = transformer_object.transform(df_train)

    # slice the context, ie shorten the amount of data available.
    if transformation_dict['context_slicer'] not in [None, 'None']:
        from autots.tools.transform import simple_context_slicer
        df_train_transformed = simple_context_slicer(df_train_transformed, method = transformation_dict['context_slicer'], forecast_length = forecast_length)

    # make sure regressor has same length. This could be a problem if wrong size regressor is passed.
    if len(future_regressor_train) > 0:
        future_regressor_train = future_regressor_train.tail(df_train_transformed.shape[0])

    transformation_runtime = datetime.datetime.now() - transformationStartTime
    # from autots.evaluator.auto_model import ModelMonster
    model = ModelMonster(model_str, parameters=parameter_dict,
                         frequency=frequency,
                         prediction_interval=prediction_interval,
                         holiday_country=holiday_country,
                         random_seed=random_seed, verbose=verbose,
                         forecast_length=forecast_length)
    model = model.fit(df_train_transformed,
                      future_regressor=future_regressor_train)
    df_forecast = model.predict(forecast_length=forecast_length,
                                future_regressor=future_regressor_forecast)

    if df_forecast.forecast.isnull().all(axis=0).astype(int).sum() > 0:
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
    if constraint is not None:
        if verbose > 2:
            print("Using constraint.")
        constraint = float(constraint)
        train_std = df_train.std(axis=0)
        train_min = df_train.min(axis=0) - (constraint * train_std)
        train_max = df_train.max(axis=0) + (constraint * train_std)
        df_forecast.forecast = df_forecast.forecast.clip(lower = train_min,
                                                         axis=1)
        df_forecast.forecast = df_forecast.forecast.clip(upper = train_max,
                                                         axis=1)
    transformation_runtime = transformation_runtime + (datetime.datetime.now() - transformationStartTime)
    df_forecast.transformation_runtime = transformation_runtime

    return df_forecast


class TemplateEvalObject(object):
    """Object to contain all your failures!."""

    def __init__(self, model_results=pd.DataFrame(),
                 per_timestamp_smape=pd.DataFrame(),
                 per_series_mae=pd.DataFrame(),
                 per_series_spl=pd.DataFrame(),
                 per_series_rmse1=pd.DataFrame(),
                 per_series_rmse2=pd.DataFrame(),
                 model_count: int = 0
                 ):
        self.model_results = model_results
        self.model_count = model_count
        self.per_series_mae = per_series_mae
        self.per_series_spl = per_series_spl
        self.per_series_rmse1 = per_series_rmse1
        self.per_series_rmse2 = per_series_rmse2
        self.per_timestamp_smape = per_timestamp_smape

    def __repr__(self):
        """Print."""
        return 'Results objects, result table at self.model_results (pd.df)'
    def concat(self, another_eval):
        """Merge another TemplateEvalObject onto this one."""
        self.model_results = pd.concat(
            [self.model_results,
             another_eval.model_results],
            axis=0, ignore_index=True, sort=False).reset_index(drop=True)
        self.per_series_mae = pd.concat(
            [self.per_series_mae,
             another_eval.per_series_mae],
            axis=0, sort=False)
        self.per_series_spl = pd.concat(
            [self.per_series_spl,
             another_eval.per_series_spl],
            axis=0, sort=False)
        self.per_series_rmse1 = pd.concat(
            [self.per_series_rmse1,
             another_eval.per_series_rmse1],
            axis=0, sort=False)
        self.per_series_rmse2 = pd.concat(
            [self.per_series_rmse2,
             another_eval.per_series_rmse2],
            axis=0, sort=False)
        self.per_timestamp_smape = pd.concat(
                [self.per_timestamp_smape,
                 another_eval.per_timestamp_smape],
                axis=0, sort=False)
        self.model_count = self.model_count + another_eval.model_count
        return self
    def save(self, filename):
        """Save results to a file."""
        if '.csv' in filename:
            self.model_results.to_csv(filename, index=False)
        elif '.pickle' in filename:
            import pickle
            with open(filename, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError("filename not .csv or .pickle")


def unpack_ensemble_models(template,
                           template_cols: list = ['Model', 'ModelParameters',
                                                  'TransformationParameters',
                                                  'Ensemble'],
                           keep_ensemble: bool = True,
                           recursive: bool = False):
    """Take ensemble models from template and add as new rows."""
    ensemble_template = pd.DataFrame()
    template['Ensemble'] = np.where(((template['Model'] == 'Ensemble') & (template['Ensemble'] < 1)),
                                    1, template['Ensemble'])
    for index, value in template[template['Ensemble'] != 0]['ModelParameters'].iteritems():
        model_dict = json.loads(value)['models']
        model_df = pd.DataFrame.from_dict(model_dict, orient='index')
        model_df = model_df.rename_axis('ID').reset_index(drop=False)
        model_df['Ensemble'] = 0
        # unpack nested ensembles, if recursive specified
        if recursive and 'Ensemble' in model_df['Model'].tolist():
            model_df = pd.concat([unpack_ensemble_models(
                model_df, recursive=True, template_cols=template_cols),
                model_df], axis=0, ignore_index=True,
                sort=False).reset_index(drop=True)
        ensemble_template = pd.concat([ensemble_template, model_df], axis=0,
                                      ignore_index=True,
                                      sort=False).reset_index(drop=True)
    if not keep_ensemble:
        template = template[template['Ensemble'] == 0]
    template = pd.concat([template, ensemble_template], axis=0,
                         ignore_index=True,
                         sort=False).reset_index(drop=True)
    template = template.drop_duplicates(subset=template_cols)
    return template


def PredictWitch(template, df_train, forecast_length: int,
                 frequency: str = 'infer',
                 prediction_interval: float = 0.9,
                 no_negatives: bool = False,
                 constraint: float = None,
                 future_regressor_train=[],
                 future_regressor_forecast=[],
                 holiday_country: str = 'US', startTimeStamps=None,
                 grouping_ids=None,
                 random_seed: int = 2020, verbose: int = 0,
                 template_cols: list = ['Model', 'ModelParameters',
                                        'TransformationParameters',
                                        'Ensemble']):
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
        constraint (float): when not None, use this value * data st dev above max or below min for constraining forecast values.
        future_regressor_train (pd.Series): with datetime index, of known in advance data, section matching train data
        future_regressor_forecast (pd.Series): with datetime index, of known in advance data, section matching test data
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
        if row_upper['Model'] == 'Ensemble':
            from autots.models.ensemble import EnsembleForecast
            forecasts_list = []
            forecasts_runtime = []
            forecasts = []
            upper_forecasts = []
            lower_forecasts = []
            ens_model_str = row_upper['Model']
            ens_params = json.loads(row_upper['ModelParameters'])
            ens_template = unpack_ensemble_models(template, template_cols,
                                                  keep_ensemble=False)
            for index, row in ens_template.iterrows():
                # recursive recursion!
                df_forecast = PredictWitch(
                    row, df_train=df_train,
                    forecast_length=forecast_length, frequency=frequency,
                    prediction_interval=prediction_interval,
                    no_negatives=no_negatives,
                    constraint=constraint,
                    future_regressor_train=future_regressor_train,
                    future_regressor_forecast=future_regressor_forecast,
                    holiday_country=holiday_country,
                    startTimeStamps=startTimeStamps,
                    grouping_ids=grouping_ids,
                    random_seed=random_seed, verbose=verbose,
                    template_cols=template_cols)
                model_id = create_model_id(df_forecast.model_name, df_forecast.model_parameters, df_forecast.transformation_parameters)
                total_runtime = df_forecast.fit_runtime + df_forecast.predict_runtime + df_forecast.transformation_runtime

                forecasts_list.extend([model_id])
                forecasts_runtime.extend([total_runtime])
                forecasts.extend([df_forecast.forecast])
                upper_forecasts.extend([df_forecast.upper_forecast])
                lower_forecasts.extend([df_forecast.lower_forecast])
            ens_forecast = EnsembleForecast(ens_model_str, ens_params,
                                            forecasts_list=forecasts_list,
                                            forecasts=forecasts,
                                            lower_forecasts=lower_forecasts,
                                            upper_forecasts=upper_forecasts,
                                            forecasts_runtime=forecasts_runtime,
                                            prediction_interval=prediction_interval)
            return ens_forecast
        # if not an ensemble
        else:
            model_str = row_upper['Model']
            parameter_dict = json.loads(row_upper['ModelParameters'])
            transformation_dict = json.loads(row_upper['TransformationParameters'])

            df_forecast = ModelPrediction(
                df_train, forecast_length, transformation_dict,
                model_str, parameter_dict, frequency=frequency,
                prediction_interval=prediction_interval,
                no_negatives=no_negatives,
                constraint=constraint,
                future_regressor_train=future_regressor_train,
                future_regressor_forecast=future_regressor_forecast,
                grouping_ids=grouping_ids,
                holiday_country=holiday_country, random_seed=random_seed,
                verbose=verbose, startTimeStamps=startTimeStamps)

            return df_forecast


def TemplateWizard(template, df_train, df_test, weights,
                   model_count: int = 0, ensemble: str = True,
                   forecast_length: int = 14, frequency: str = 'infer',
                   prediction_interval: float = 0.9,
                   no_negatives: bool = False,
                   constraint: float = None,
                   future_regressor_train=[],
                   future_regressor_forecast=[],
                   holiday_country: str = 'US', startTimeStamps=None,
                   random_seed: int = 2020, verbose: int = 0,
                   validation_round: int = 0,
                   model_interrupt: bool = False,
                   grouping_ids=None,
                   template_cols: list = ['Model', 'ModelParameters',
                                          'TransformationParameters',
                                          'Ensemble']):
    """
    Take Template, returns Results.

    There are some who call me... Tim. - Python

    Args:
        template (pandas.DataFrame): containing model str, and json of transformations and hyperparamters
        df_train (pandas.DataFrame): numeric training dataset of DatetimeIndex and series as cols
        df_test (pandas.DataFrame): dataframe of actual values of (forecast length * n series)
        weights (dict): key = column/series_id, value = weight
        ensemble (str): desc of ensemble types to prepare metric collection
        forecast_length (int): number of periods to forecast
        transformation_dict (dict): a dictionary of outlier, fillNA, and transformation methods to be used
        model_str (str): a string to be direct to the appropriate model, used in ModelMonster
        frequency (str): str representing frequency alias of time series
        prediction_interval (float): width of errors (note: rarely do the intervals accurately match the % asked for...)
        no_negatives (bool): whether to force all forecasts to be > 0
        constraint (float): when not None, use this value * data st dev above max or below min for constraining forecast values.
        future_regressor_train (pd.Series): with datetime index, of known in advance data, section matching train data
        future_regressor_forecast (pd.Series): with datetime index, of known in advance data, section matching test data
        holiday_country (str): passed through to holiday package, used by a few models as 0/1 regressor.
        startTimeStamps (pd.Series): index (series_ids), columns (Datetime of First start of series)
        validation_round (int): int passed to record current validation.
        model_interrupt (bool): if True, keyboard interrupts are caught and only break current model eval.
        template_cols (list): column names of columns used as model template

    Returns:
        TemplateEvalObject
    """
    ensemble = str(ensemble)
    template_result = TemplateEvalObject()
    template_result.model_count = model_count
    if isinstance(template, pd.Series):
        template = pd.DataFrame(template).transpose()
    # template = unpack_ensemble_models(template, template_cols, keep_ensemble = False)

    for index, row in template.iterrows():
        try:
            model_str = row['Model']
            parameter_dict = json.loads(row['ModelParameters'])
            transformation_dict = json.loads(row['TransformationParameters'])
            ensemble_input = row['Ensemble']
            current_template = pd.DataFrame(row).transpose()
            template_result.model_count += 1
            if verbose > 0:
                if verbose > 1:
                    print("Model Number: {} with model {} in Validation {} with params {} and transformations {}".format(str(template_result.model_count), model_str, str(validation_round), json.dumps(parameter_dict),json.dumps(transformation_dict)))
                else:
                    print("Model Number: {} with model {} in Validation {} ".format(str(template_result.model_count), model_str, str(validation_round)))
            df_forecast = PredictWitch(
                current_template, df_train=df_train,
                forecast_length=forecast_length, frequency=frequency,
                prediction_interval=prediction_interval,
                no_negatives=no_negatives,
                constraint=constraint,
                future_regressor_train=future_regressor_train,
                future_regressor_forecast=future_regressor_forecast,
                holiday_country=holiday_country,
                startTimeStamps=startTimeStamps,
                grouping_ids=grouping_ids,
                random_seed=random_seed, verbose=verbose,
                template_cols=template_cols)

            per_ts = True if 'distance' in ensemble else False
            if 'hdist' in ensemble:
                dist_n = int(np.ceil(0.3 * forecast_length))
            else:
                dist_n = None
            model_error = PredictionEval(df_forecast, df_test,
                                         series_weights=weights,
                                         df_train=df_train,
                                         per_timestamp_errors=per_ts,
                                         dist_n=dist_n)
            model_id = create_model_id(df_forecast.model_name,
                                       df_forecast.model_parameters,
                                       df_forecast.transformation_parameters)
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
                    'Runs': 1,
                    'ValidationRound': validation_round
                    }, index = [0])
            a = pd.DataFrame(model_error.avg_metrics_weighted.rename(lambda x: x + '_weighted')).transpose()
            result = pd.concat([result, pd.DataFrame(model_error.avg_metrics).transpose(), a], axis=1)
            template_result.model_results = pd.concat([template_result.model_results, result], axis=0, ignore_index=True, sort=False).reset_index(drop=True)
            if 'horizontal' in ensemble:
                cur_mae = model_error.per_series_metrics.loc['mae']
                cur_mae = pd.DataFrame(cur_mae).transpose()
                cur_mae.index = [model_id]
                template_result.per_series_mae = pd.concat(
                    [template_result.per_series_mae, cur_mae],
                    axis=0
                    )
            if 'probabilistic' in ensemble:
                cur_spl = model_error.per_series_metrics.loc['spl']
                cur_spl = pd.DataFrame(cur_spl).transpose()
                cur_spl.index = [model_id]
                template_result.per_series_spl = pd.concat(
                    [template_result.per_series_spl, cur_spl],
                    axis=0)
            if 'distance' in ensemble:
                cur_smape = model_error.per_timestamp.loc['weighted_smape']
                cur_smape = pd.DataFrame(cur_smape).transpose()
                cur_smape.index = [model_id]
                template_result.per_timestamp_smape = pd.concat(
                    [template_result.per_timestamp_smape, cur_smape],
                    axis=0)
            if 'hdist' in ensemble:
                cur_rmse1 = model_error.per_series_metrics.loc['rmse1']
                cur_rmse2 = model_error.per_series_metrics.loc['rmse2']
                cur_rmse1 = pd.DataFrame(cur_rmse1).transpose()
                cur_rmse2 = pd.DataFrame(cur_rmse2).transpose()
                cur_rmse1.index = [model_id]
                cur_rmse2.index = [model_id]
                template_result.per_series_rmse1 = pd.concat(
                    [template_result.per_series_rmse1, cur_rmse1],
                    axis=0)
                template_result.per_series_rmse2 = pd.concat(
                    [template_result.per_series_rmse2, cur_rmse2],
                    axis=0)
        except KeyboardInterrupt:
            if model_interrupt:
                result = pd.DataFrame({
                    'ID': create_model_id(model_str,
                                          parameter_dict,
                                          transformation_dict),
                    'Model': model_str,
                    'ModelParameters': json.dumps(parameter_dict),
                    'TransformationParameters': json.dumps(transformation_dict),
                    'Ensemble': ensemble_input,
                    'TransformationRuntime': datetime.timedelta(0),
                    'FitRuntime': datetime.timedelta(0),
                    'PredictRuntime': datetime.timedelta(0),
                    'TotalRuntime': datetime.timedelta(0),
                    'Exceptions': "KeyboardInterrupt by user",
                    'Runs': 1,
                    'ValidationRound': validation_round
                    }, index=[0])
                template_result.model_results = pd.concat([template_result.model_results, result], axis=0, ignore_index=True, sort=False).reset_index(drop=True)
            else:
                raise KeyboardInterrupt
        except Exception as e:
            if verbose >= 0:
                print('Template Eval Error: {} in model {}: {}'.format((repr(e)), template_result.model_count, model_str))
            result = pd.DataFrame({
                'ID': create_model_id(model_str,
                                      parameter_dict,
                                      transformation_dict),
                'Model': model_str,
                'ModelParameters': json.dumps(parameter_dict),
                'TransformationParameters': json.dumps(transformation_dict),
                'Ensemble': ensemble_input,
                'TransformationRuntime': datetime.timedelta(0),
                'FitRuntime': datetime.timedelta(0),
                'PredictRuntime': datetime.timedelta(0),
                'TotalRuntime': datetime.timedelta(0),
                'Exceptions': repr(e),
                'Runs': 1,
                'ValidationRound': validation_round
                }, index=[0])
            template_result.model_results = pd.concat([template_result.model_results, result], axis=0, ignore_index=True, sort=False).reset_index(drop=True)

    return template_result


def RandomTemplate(n: int = 10,
                   model_list: list = ['ZeroesNaive', 'LastValueNaive', 
                                       'AverageValueNaive', 'GLS', 'GLM',
                                       'ETS', 'ARIMA', 'FBProphet',
                                       'RollingRegression', 'GluonTS',
                                       'UnobservedComponents', 'VARMAX',
                                       'VECM', 'DynamicFactor']):
    """
    Returns a template dataframe of randomly generated transformations, models, and hyperparameters.

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
                }, index=[0])
        template = pd.concat([template, row], axis=0, ignore_index=True)
        template.drop_duplicates(inplace=True)
        counter += 1
        if counter > (n * 3):
            break
    return template


def UniqueTemplates(existing_templates, new_possibilities,
                    selection_cols: list = ['Model', 'ModelParameters',
                                            'TransformationParameters',
                                            'Ensemble']):
    """
    Returns unique dataframe rows from new_possiblities not in existing_templates.

    Args:
        selection_cols (list): list of column namess to use to judge uniqueness/match on
    """
    keys = list(new_possibilities[selection_cols].columns.values)
    idx1 = existing_templates.copy().set_index(keys).index
    idx2 = new_possibilities.set_index(keys).index
    new_template = new_possibilities[~idx2.isin(idx1)]
    return new_template


def dict_recombination(a: dict, b: dict):
    """Recombine two dictionaries with identical keys. Return new dict."""
    b_keys = [*b]
    key_size = int(len(b_keys)/2) if len(b_keys) > 1 else 1
    bs_keys = np.random.choice(b_keys, size=key_size)
    b_prime = {k: b[k] for k in bs_keys}
    c = {**a, **b_prime}  # overwrites with B
    return c


def trans_dict_recomb(dict_array):
    """Recombine two transformation param dictionaries from array of dicts."""
    r_sel = np.random.choice(dict_array, size=2, replace=False)
    a = r_sel[0]
    b = r_sel[1]
    c = dict_recombination(a, b)

    out_keys = ['outlier_method', 'outlier_threshold', 'outlier_position']
    current_dict = np.random.choice([a, b], size=1).item()
    c = {**c, **{k: current_dict[k] for k in out_keys}}

    mid_trans_keys = ['second_transformation', 'transformation_param']
    current_dict = np.random.choice([a, b], size=1).item()
    c = {**c, **{k: current_dict[k] for k in mid_trans_keys}}
    
    mid_trans_keys = ['third_transformation', 'transformation_param2']
    current_dict = np.random.choice([a, b], size=1).item()
    c = {**c, **{k: current_dict[k] for k in mid_trans_keys}}

    disc_keys = ['discretization', 'n_bins']
    current_dict = np.random.choice([a, b], size=1).item()
    c = {**c, **{k: current_dict[k] for k in disc_keys}}
    
    disc_keys = ['grouping', 'reconciliation']
    current_dict = np.random.choice([a, b], size=1).item()
    if all([x in current_dict.keys() for x in disc_keys]):
        c = {**c, **{k: current_dict[k] for k in disc_keys}}
    return c


def _trans_dicts(current_ops, best = None, n: int = 5):
    fir = json.loads(current_ops.iloc[0, :]['TransformationParameters'])
    cur_len = current_ops.shape[0]
    if cur_len > 1:
        # select randomly from best of data, doesn't handle lengths < 2
        top_r = np.floor((cur_len / 5) + 2)
        r_id = np.random.randint(1, top_r)
        sec = json.loads(current_ops.iloc[r_id, :]['TransformationParameters'])
    else:
        sec = RandomTransform()
    r = RandomTransform()
    if best is None:
        best = RandomTransform()
    arr = [fir, sec, best, r]
    trans_dicts = [json.dumps(trans_dict_recomb(arr)) for _ in range(n)]
    return trans_dicts


def NewGeneticTemplate(model_results, submitted_parameters,
                       sort_column: str = "smape_weighted",
                       sort_ascending: bool = True, max_results: int = 50,
                       max_per_model_class: int = 5,
                       top_n: int = 50,
                       template_cols: list = ['Model', 'ModelParameters',
                                              'TransformationParameters',
                                              'Ensemble']):
    """
    Return new template given old template with model accuracies.

    Args:
        model_results (pandas.DataFrame): models that have actually been run
        submitted_paramters (pandas.DataFrame): models tried (may have returned different parameters to results)

    """
    new_template = pd.DataFrame()

    # filter existing templates
    sorted_results = model_results[model_results['Ensemble'] == 0].copy()
    sorted_results = sorted_results.sort_values(by=sort_column,
                                                ascending=sort_ascending,
                                                na_position='last')
    sorted_results = sorted_results.drop_duplicates(subset=template_cols,
                                                    keep='first')
    if str(max_per_model_class).isdigit():
        sorted_results = sorted_results.sort_values(
            sort_column, ascending=sort_ascending).groupby('Model').head(
                max_per_model_class).reset_index()
    sorted_results = sorted_results.sort_values(by=sort_column,
                                                ascending=sort_ascending,
                                                na_position='last').head(top_n)

    no_params = ['ZeroesNaive', 'LastValueNaive', 'GLS']
    recombination_approved = ['SeasonalNaive', 'MotifSimulation', "ETS",
                              'DynamicFactor', 'VECM', 'VARMAX', 'GLM',
                              'ARIMA', 'FBProphet', 'GluonTS',
                              'RollingRegression', 'VAR', 'WindowRegression',
                              'TensorflowSTS', 'TFPRegression']
    borrow = ['ComponentAnalysis']
    best = json.loads(sorted_results.iloc[0, :]['TransformationParameters'])

    for model_type in sorted_results['Model'].unique():
        if model_type in no_params:
            current_ops = sorted_results[sorted_results['Model'] == model_type]
            n = 3
            trans_dicts = _trans_dicts(current_ops, best=best, n=n)
            model_param = current_ops.iloc[0, :]['ModelParameters']
            new_row = pd.DataFrame({
                'Model': model_type,
                'ModelParameters': model_param,
                'TransformationParameters': trans_dicts,
                'Ensemble': 0
                }, index=list(range(n)))
        elif model_type in recombination_approved:
            current_ops = sorted_results[sorted_results['Model'] == model_type]
            n = 4
            trans_dicts = _trans_dicts(current_ops, best=best, n=n)
            # select the best model of this type
            fir = json.loads(current_ops.iloc[0, :]['ModelParameters'])
            cur_len = current_ops.shape[0]
            if cur_len > 1:
                # select randomly from best of data, doesn't handle lengths < 2
                top_r = np.floor((cur_len / 5) + 2)
                r_id = np.random.randint(1, top_r)
                sec = json.loads(current_ops.iloc[r_id, :]['ModelParameters'])
            else:
                sec = ModelMonster(model_type).get_new_params()
            # generate new random parameters ('mutations')
            r = ModelMonster(model_type).get_new_params()
            r2 = ModelMonster(model_type).get_new_params()
            arr = [fir, sec, r2, r]
            model_dicts = list()
            # recombine best and random to create new generation
            for _ in range(n):
                r_sel = np.random.choice(arr, size=2, replace=False)
                a = r_sel[0]
                b = r_sel[1]
                c = dict_recombination(a, b)
                model_dicts.append(json.dumps(c))
            new_row = pd.DataFrame({
                'Model': model_type,
                'ModelParameters': model_dicts,
                'TransformationParameters': trans_dicts,
                'Ensemble': 0
                }, index=list(range(n)))
        else:
            current_ops = sorted_results[sorted_results['Model'] == model_type]
            n = 3
            trans_dicts = _trans_dicts(current_ops, best=best, n=n)
            model_dicts = list()
            for _ in range(n):
                c = ModelMonster(model_type).get_new_params()
                model_dicts.append(json.dumps(c))
            new_row = pd.DataFrame({
                'Model': model_type,
                'ModelParameters': model_dicts,
                'TransformationParameters': trans_dicts,
                'Ensemble': 0
                }, index=list(range(n)))
        new_template = pd.concat([new_template, new_row],
                                 axis=0, ignore_index=True, sort=False)
    """
    # recombination of transforms across models by shifting transforms
    recombination = sorted_results.tail(len(sorted_results.index) - 1).copy()
    recombination['TransformationParameters'] = sorted_results['TransformationParameters'].shift(1).tail(len(sorted_results.index) - 1)
    new_template = pd.concat([new_template,
                              recombination.head(top_n)[template_cols]],
                             axis=0, ignore_index=True, sort=False)
    """
    # remove generated models which have already been tried
    sorted_results = pd.concat([submitted_parameters, sorted_results], axis=0,
                               ignore_index=True, sort=False
                               ).reset_index(drop=True)
    new_template = UniqueTemplates(sorted_results, new_template,
                                   selection_cols=template_cols
                                   ).head(max_results)
    return new_template


def validation_aggregation(validation_results):
    """Aggregate a TemplateEvalObject."""
    groupby_cols = ['ID', 'Model', 'ModelParameters',
                    'TransformationParameters', 'Ensemble']
    col_aggs = {'Runs': 'sum',
                'smape': 'mean',
                'mae': 'mean',
                'rmse': 'mean',
                'containment': 'mean',
                'spl': 'mean',
                'contour': 'mean',
                'smape_weighted': 'mean',
                'mae_weighted': 'mean',
                'rmse_weighted': 'mean',
                'containment_weighted': 'mean',
                'contour_weighted': 'mean',
                'spl_weighted': 'mean',
                'containment_weighted': 'mean',
                'TotalRuntimeSeconds': 'mean',
                'Score': 'mean'
                }
    validation_results.model_results['TotalRuntimeSeconds'] = validation_results.model_results['TotalRuntime'].dt.seconds + 1
    validation_results.model_results = validation_results.model_results[pd.isnull(validation_results.model_results['Exceptions'])]
    validation_results.model_results = validation_results.model_results.replace([np.inf, -np.inf], np.nan)
    validation_results.model_results = validation_results.model_results.groupby(groupby_cols).agg(col_aggs)
    validation_results.model_results = validation_results.model_results.reset_index(drop = False)
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
        spl_weighting = metric_weighting['spl_weighting']
    except KeyError:
        spl_weighting = 0
    try:
        contour_weighting = metric_weighting['contour_weighting']
    except KeyError:
        contour_weighting = 0
    # handle various runtime information records
    if 'TotalRuntimeSeconds' in model_results.columns:
        if 'TotalRuntime' in model_results.columns:
            model_results['TotalRuntimeSeconds'] = np.where(
                model_results['TotalRuntimeSeconds'].isna(), 
                model_results['TotalRuntime'].dt.seconds,
                model_results['TotalRuntimeSeconds'])
        else:
            model_results['TotalRuntimeSeconds'] = np.where(
                model_results['TotalRuntimeSeconds'].isna(), 
                model_results['TotalRuntimeSeconds'].max(),
                model_results['TotalRuntimeSeconds'])
    else:
        model_results['TotalRuntimeSeconds'] = model_results['TotalRuntime'].dt.seconds
    # generate minimizing scores, where smaller = better accuracy
    try:
        model_results = model_results.replace([np.inf, -np.inf], np.nan)
        # model_results = model_results.fillna(value=model_results.max(axis=0))
        smape_score = model_results['smape_weighted']/(model_results['smape_weighted'].min(skipna=True) + 1)  # smaller better
        rmse_scaler = (model_results['rmse_weighted'].median(skipna=True))
        rmse_scaler = 1 if rmse_scaler == 0 else rmse_scaler
        rmse_score = model_results['rmse_weighted']/rmse_scaler
        mae_scaler = (model_results['mae_weighted'].median(skipna=True))
        mae_scaler = 1 if mae_scaler == 0 else mae_scaler
        mae_score = model_results['mae_weighted']/mae_scaler
        containment_score = (abs(prediction_interval - model_results['containment'])) + 1  # from 1 to 2, smaller better
        runtime = model_results['TotalRuntimeSeconds'] + 120
        runtime_score = runtime/(runtime.min(skipna=True))  # smaller better
        spl_score = model_results['spl_weighted']/(model_results['spl_weighted'].min(skipna=True) + 1)  # smaller better
        contour_score = (1/(model_results['contour_weighted'])).replace(
            [np.inf, -np.inf, np.nan], 10).clip(upper=10)
    except KeyError:
        raise KeyError("Inconceivable! Evaluation Metrics are missing and all models have failed, by an error in TemplateWizard or metrics. A new template may help, or an adjusted model_list.")
    return (smape_score * smape_weighting) + (mae_score * mae_weighting) + (rmse_score * rmse_weighting) + (containment_score * containment_weighting) + (runtime_score * runtime_weighting) + (spl_score * spl_weighting) + (contour_score * contour_weighting)
