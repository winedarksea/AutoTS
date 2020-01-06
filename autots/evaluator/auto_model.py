import numpy as np
import pandas as pd
import datetime
import hashlib
import json

def create_model_id(model_str: str, parameter_dict: dict = {}, transformation_dict: dict = {}):
    """
    Create a hash model ID which should be unique to the model parameters
    """
    str_repr = str(model_str) + json.dumps(parameter_dict) + json.dumps(transformation_dict)
    str_repr = ''.join(str_repr.split())
    hashed = hashlib.md5(str_repr.encode('utf-8')).hexdigest()
    return hashed

class ModelObject(object):
    """
    Models should all have methods:
        .fit(df) (taking a DataFrame with DatetimeIndex and n columns of n timeseries)
        .predict(forecast_length = int, regressor)
        .get_new_params(method)
    
    Args:
        name (str): Model Name
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
    """
    def __init__(self, name: str = "Uninitiated Model Name", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, regression_type: str = None, 
                 fit_runtime=datetime.timedelta(0), holiday_country: str = 'US',
                 random_seed: int = 2020):
        self.name = name
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.regression_type = regression_type
        self.fit_runtime = fit_runtime
        self.holiday_country = holiday_country
        self.random_seed = random_seed
    
    def __repr__(self):
        return 'ModelObject of ' + self.name
    
    def basic_profile(self, df):
        """Capture basic training details
        """
        self.startTime = datetime.datetime.now()
        self.train_shape = df.shape
        self.column_names = df.columns
        self.train_last_date = df.index[-1]
        if self.frequency == 'infer':
            self.frequency = pd.infer_freq(df.index, warn = False)
        
        return df
    
    def create_forecast_index(self, forecast_length: int):
        """
        Requires ModelObject.basic_profile() being called as part of .fit()
        """
        forecast_index = pd.date_range(freq = self.frequency, start = self.train_last_date, periods = forecast_length + 1)
        forecast_index = forecast_index[1:]
        self.forecast_index = forecast_index
        return forecast_index
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {}
    
    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}

class PredictionObject(object):
    def __init__(self, model_name: str = 'Uninitiated',
                 forecast_length: int = 0, 
                 forecast_index = np.nan, forecast_columns = np.nan,
                 lower_forecast = np.nan, forecast = np.nan, upper_forecast = np.nan, 
                 prediction_interval: float = 0.9, predict_runtime=datetime.timedelta(0),
                 fit_runtime = datetime.timedelta(0),
                 model_parameters = {}, transformation_parameters = {},
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
        


def ModelPrediction(df_train, forecast_length: int, transformation_dict: dict, 
                    model_str: str, parameter_dict: dict, frequency: str = 'infer', 
                    prediction_interval: float = 0.9, no_negatives: bool = False,
                    preord_regressor_train = [], preord_regressor_forecast = [], 
                    holiday_country: str = 'US', startTimeStamps = None):
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
    transformer_object = GeneralTransformer(outlier=transformation_dict['outlier'],
                                            fillNA = transformation_dict['fillNA'], 
                                            transformation = transformation_dict['transformation']).fit(df_train)
    df_train_transformed = transformer_object.transform(df_train)
    
    if transformation_dict['context_slicer'] in ['2ForecastLength','HalfMax']:
        from autots.tools.transform import simple_context_slicer
        df_train_transformed = simple_context_slicer(df_train_transformed, method = transformation_dict['context_slicer'], forecast_length = forecast_length)
    
    transformation_runtime = datetime.datetime.now() - transformationStartTime
    from autots.evaluator.auto_model import ModelMonster
    model = ModelMonster(model_str, parameters=parameter_dict, frequency = frequency, 
                         prediction_interval = prediction_interval, holiday_country = holiday_country)
    model = model.fit(df_train_transformed, preord_regressor = preord_regressor_train)
    df_forecast = model.predict(forecast_length = forecast_length, preord_regressor = preord_regressor_forecast)
    
    transformationStartTime = datetime.datetime.now()
    # Inverse the transformations
    df_forecast.forecast = pd.DataFrame(transformer_object.inverse_transform(df_forecast.forecast), index = df_forecast.forecast_index, columns = df_forecast.forecast_columns)
    df_forecast.lower_forecast = pd.DataFrame(transformer_object.inverse_transform(df_forecast.lower_forecast), index = df_forecast.forecast_index, columns = df_forecast.forecast_columns)
    df_forecast.upper_forecast = pd.DataFrame(transformer_object.inverse_transform(df_forecast.upper_forecast), index = df_forecast.forecast_index, columns = df_forecast.forecast_columns)
    
    if df_forecast.forecast.isnull().all(axis = 0).astype(int).sum() > 0:
        raise ValueError("Model {} failed for one or more series".format(model_str))
        
    df_forecast.transformation_parameters = transformation_dict
    # Remove negatives if desired
    if no_negatives:
        df_forecast.lower_forecast = df_forecast.lower_forecast.where(df_forecast.lower_forecast > 0, 0)
        df_forecast.forecast = df_forecast.forecast.where(df_forecast.forecast > 0, 0)
        df_forecast.upper_forecast = df_forecast.upper_forecast.where(df_forecast.upper_forecast > 0, 0)
    transformation_runtime = transformation_runtime + (datetime.datetime.now() - transformationStartTime)
    df_forecast.transformation_runtime = transformation_runtime
    
    return df_forecast

ModelNames = ['ZeroesNaive', 'LastValueNaive', 'MedValueNaive',
              'GLM', 'ETS', 'ARIMA', 'FBProphet']

def ModelMonster(model: str, parameters: dict = {}, frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US', 
                 startTimeStamps = None):
    """Directs strings and parameters to appropriate model objects.
    
    Args:
        model (str): Name of Model Function
        parameters (dict): Dictionary of parameters to pass through to model
    """
    if model == 'ZeroesNaive':
        from autots.models.basics import ZeroesNaive
        return ZeroesNaive(frequency = frequency, prediction_interval = prediction_interval)
    
    if model == 'LastValueNaive':
        from autots.models.basics import LastValueNaive
        return LastValueNaive(frequency = frequency, prediction_interval = prediction_interval)
    
    if model == 'MedValueNaive':
        from autots.models.basics import MedValueNaive
        return MedValueNaive(frequency = frequency, prediction_interval = prediction_interval)
    
    if model == 'GLM':
        from autots.models.statsmodels import GLM
        return GLM(frequency = frequency, prediction_interval = prediction_interval)
    
    if model == 'ETS':
        from autots.models.statsmodels import ETS
        if parameters == {}:
            model = ETS(frequency = frequency, prediction_interval = prediction_interval)
        else:
            model = ETS(frequency = frequency, prediction_interval = prediction_interval, damped=parameters['damped'], trend=parameters['trend'], seasonal=parameters['seasonal'], seasonal_periods=parameters['seasonal_periods'])
        return model
    
    if model == 'ARIMA':
        from autots.models.statsmodels import ARIMA
        if parameters == {}:
            model = ARIMA(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country)
        else:
            model = ARIMA(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, p =parameters['p'], d=parameters['d'], q=parameters['q'], regression_type=parameters['regression_type'])
        return model
    
    if model == 'FBProphet':
        from autots.models.prophet import FBProphet
        if parameters == {}:
            model = FBProphet(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country)
        else:
            model = FBProphet(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, holiday =parameters['holiday'], regression_type=parameters['regression_type'])
        return model
    
    else:
        raise AttributeError("Model String not found in ModelMonster")


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
    
# from autots.evaluator.auto_model import create_model_id    
def TemplateWizard(template, df_train, df_test, weights,
                   model_count: int = 0, ensemble: bool = True,
                   forecast_length: int = 14, frequency: str = 'infer', 
                    prediction_interval: float = 0.9, no_negatives: bool = False,
                    preord_regressor_train = [], preord_regressor_forecast = [], 
                    holiday_country: str = 'US', startTimeStamps = None):
    """
    takes Template, returns Results
    
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
        
    """
    template_result = TemplateEvalObject()
    template_result.model_count = model_count
    
    
    for index, row in template.iterrows():
        model_str = row['Model']
        parameter_dict = json.loads(row['ModelParameters'])
        transformation_dict = json.loads(row['TransformationParameters'])
        template_result.model_count += 1
        print("Model Number: {} with model {}".format(str(template_result.model_count), model_str))
        try:
            # from autots.evaluator.auto_model import ModelPrediction
            df_forecast = ModelPrediction(df_train, forecast_length,transformation_dict, 
                                          model_str, parameter_dict, frequency=frequency, 
                                          prediction_interval=prediction_interval, 
                                          no_negatives=no_negatives,
                                          preord_regressor_train = preord_regressor_train,
                                          preord_regressor_forecast = preord_regressor_forecast, 
                                          holiday_country = holiday_country,
                                          startTimeStamps = startTimeStamps)
            
            from autots.evaluator.metrics import PredictionEval
            model_error = PredictionEval(df_forecast, df_test, series_weights = weights)
            model_id = create_model_id(model_str, df_forecast.model_parameters, df_forecast.transformation_parameters)
            total_runtime = df_forecast.fit_runtime + df_forecast.predict_runtime + df_forecast.transformation_runtime
            result = pd.DataFrame({
                    'ID': model_id,
                    'Model': model_str,
                    'ModelParameters': json.dumps(df_forecast.model_parameters),
                    'TransformationParameters': json.dumps(df_forecast.transformation_parameters),
                    'TransformationRuntime': df_forecast.transformation_runtime,
                    'FitRuntime': df_forecast.fit_runtime,
                    'PredictRuntime': df_forecast.predict_runtime,
                    'TotalRuntime': total_runtime,
                    'Ensemble': 0,
                    'Exceptions': np.nan,
                    'Runs': 1
                    }, index = [0])
            a = pd.DataFrame(model_error.avg_metrics_weighted.rename(lambda x: x + '_weighted')).transpose()
            result = pd.concat([result, pd.DataFrame(model_error.avg_metrics).transpose(), a], axis = 1)
            
            template_result.model_results = pd.concat([template_result.model_results, result], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
            temp = pd.DataFrame(model_error.per_timestamp_metrics.loc['smape']).transpose()
            temp.index = result['ID'] 
            template_result.model_results_per_timestamp_smape = template_result.model_results_per_timestamp_smape.append(temp)
            temp = pd.DataFrame(model_error.per_timestamp_metrics.loc['mae']).transpose()
            temp.index = result['ID']  
            template_result.model_results_per_timestamp_mae = template_result.model_results_per_timestamp_mae.append(temp)
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
            result = pd.DataFrame({
                'ID': create_model_id(model_str, parameter_dict, transformation_dict),
                'Model': model_str,
                'ModelParameters': json.dumps(parameter_dict),
                'TransformationParameters': json.dumps(transformation_dict),
                'Ensemble': 0,
                'TransformationRuntime': datetime.timedelta(0),
                'FitRuntime': datetime.timedelta(0),
                'PredictRuntime': datetime.timedelta(0),
                'TotalRuntime': datetime.timedelta(0),
                'Exceptions': str(e),
                'Runs': 1
                }, index = [0])
            template_result.model_results = pd.concat([template_result.model_results, result], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)

    return template_result 
