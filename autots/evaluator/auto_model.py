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
                 random_seed: int = 2020, verbose: int = 0):
        self.name = name
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.regression_type = regression_type
        self.fit_runtime = fit_runtime
        self.holiday_country = holiday_country
        self.random_seed = random_seed
        self.verbose = verbose
    
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
    
    def total_runtime(self):
        return self.fit_runtime + self.predict_runtime + self.transformation_runtime



def ModelPrediction(df_train, forecast_length: int, transformation_dict: dict, 
                    model_str: str, parameter_dict: dict, frequency: str = 'infer', 
                    prediction_interval: float = 0.9, no_negatives: bool = False,
                    preord_regressor_train = [], preord_regressor_forecast = [], 
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
                         prediction_interval = prediction_interval, holiday_country = holiday_country,
                         random_seed = random_seed, verbose = verbose)
    model = model.fit(df_train_transformed, preord_regressor = preord_regressor_train)
    df_forecast = model.predict(forecast_length = forecast_length, preord_regressor = preord_regressor_forecast)
    
    transformationStartTime = datetime.datetime.now()
    # Inverse the transformations
    df_forecast.forecast = pd.DataFrame(transformer_object.inverse_transform(df_forecast.forecast), index = df_forecast.forecast_index, columns = df_forecast.forecast_columns)
    df_forecast.lower_forecast = pd.DataFrame(transformer_object.inverse_transform(df_forecast.lower_forecast), index = df_forecast.forecast_index, columns = df_forecast.forecast_columns)
    df_forecast.upper_forecast = pd.DataFrame(transformer_object.inverse_transform(df_forecast.upper_forecast), index = df_forecast.forecast_index, columns = df_forecast.forecast_columns)
    
    if df_forecast.forecast.isnull().all(axis = 0).astype(int).sum() > 0:
        raise ValueError("Model {} returned NaN for one or more series".format(model_str))
        
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
              'GLM', 'ETS', 'ARIMA', 'FBProphet', 'RandomForestRolling',
              'UnobservedComponents', 'VARMAX', 'VECM', 'DynamicFactor']

def ModelMonster(model: str, parameters: dict = {}, frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US', 
                 startTimeStamps = None,
                 random_seed: int = 2020, verbose: int = 0):
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
            model = ETS(frequency = frequency, prediction_interval = prediction_interval, random_seed = random_seed, verbose = verbose)
        else:
            model = ETS(frequency = frequency, prediction_interval = prediction_interval, damped=parameters['damped'], trend=parameters['trend'], seasonal=parameters['seasonal'], seasonal_periods=parameters['seasonal_periods'], random_seed = random_seed, verbose = verbose)
        return model
    
    if model == 'ARIMA':
        from autots.models.statsmodels import ARIMA
        if parameters == {}:
            model = ARIMA(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, random_seed = random_seed, verbose = verbose)
        else:
            model = ARIMA(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, p =parameters['p'], d=parameters['d'], q=parameters['q'], regression_type=parameters['regression_type'], random_seed = random_seed, verbose = verbose)
        return model
    
    if model == 'FBProphet':
        from autots.models.prophet import FBProphet
        if parameters == {}:
            model = FBProphet(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, random_seed = random_seed, verbose = verbose)
        else:
            model = FBProphet(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, holiday =parameters['holiday'], regression_type=parameters['regression_type'], random_seed = random_seed, verbose = verbose)
        return model
    
    if model == 'RandomForestRolling':
        from autots.models.sklearn import RandomForestRolling
        if parameters == {}:
            model = RandomForestRolling(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, random_seed = random_seed, verbose = verbose)
        else:
            model = RandomForestRolling(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, holiday =parameters['holiday'], regression_type=parameters['regression_type'], random_seed = random_seed, verbose = verbose,
                 n_estimators =parameters['n_estimators'], min_samples_split =parameters['min_samples_split'], max_depth =parameters['max_depth'], mean_rolling_periods =parameters['mean_rolling_periods'], std_rolling_periods =parameters['std_rolling_periods'])
        return model
    
    if model == 'UnobservedComponents':
        from autots.models.statsmodels import UnobservedComponents
        if parameters == {}:
            model = UnobservedComponents(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, random_seed = random_seed, verbose = verbose)
        else:
            model = UnobservedComponents(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country,
                                         regression_type=parameters['regression_type'], random_seed = random_seed, verbose = verbose,
                                         level = parameters['level'], trend=parameters['trend'], cycle = parameters['cycle'],
                                         damped_cycle = parameters['damped_cycle'], irregular = parameters['irregular'],
                                         stochastic_trend=parameters['stochastic_trend'], stochastic_level=parameters['stochastic_level'],
                                         stochastic_cycle=parameters['stochastic_cycle'])
        return model
    
    if model == 'DynamicFactor':
        from autots.models.statsmodels import DynamicFactor
        if parameters == {}:
            model = DynamicFactor(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, random_seed = random_seed, verbose = verbose)
        else:
            model = DynamicFactor(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country,
                                         regression_type=parameters['regression_type'], random_seed = random_seed, verbose = verbose,
                                         k_factors = parameters['k_factors'], factor_order = parameters['factor_order'])
        return model
    
    if model == 'VECM':
        from autots.models.statsmodels import VECM
        if parameters == {}:
            model = VECM(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, random_seed = random_seed, verbose = verbose)
        else:
            model = VECM(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country,
                                         regression_type=parameters['regression_type'], random_seed = random_seed, verbose = verbose,
                                         deterministic = parameters['deterministic'], k_ar_diff = parameters['k_ar_diff'])
        return model
    
    if model == 'VARMAX':
        from autots.models.statsmodels import VARMAX
        if parameters == {}:
            model = VARMAX(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, random_seed = random_seed, verbose = verbose)
        else:
            model = VARMAX(frequency = frequency, prediction_interval = prediction_interval, holiday_country = holiday_country, random_seed = random_seed, verbose = verbose)
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

from autots.models.ensemble import EnsembleForecast
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
                                              holiday_country = holiday_country,
                                              startTimeStamps = startTimeStamps,
                                              random_seed = random_seed, verbose = verbose)
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
            
            # from autots.evaluator.auto_model import ModelPrediction
            df_forecast = ModelPrediction(df_train, forecast_length,transformation_dict, 
                                          model_str, parameter_dict, frequency=frequency, 
                                          prediction_interval=prediction_interval, 
                                          no_negatives=no_negatives,
                                          preord_regressor_train = preord_regressor_train,
                                          preord_regressor_forecast = preord_regressor_forecast, 
                                          holiday_country = holiday_country,
                                          random_seed = random_seed, verbose = verbose,
                                          startTimeStamps = startTimeStamps)
    
            return df_forecast

# from autots.evaluator.auto_model import create_model_id    
def TemplateWizard(template, df_train, df_test, weights,
                   model_count: int = 0, ensemble: bool = True,
                   forecast_length: int = 14, frequency: str = 'infer', 
                    prediction_interval: float = 0.9, no_negatives: bool = False,
                    preord_regressor_train = [], preord_regressor_forecast = [], 
                    holiday_country: str = 'US', startTimeStamps = None,
                    random_seed: int = 2020, verbose: int = 0,
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
        current_template = template.loc[index]
        model_str = current_template['Model']
        parameter_dict = json.loads(current_template['ModelParameters'])
        transformation_dict = json.loads(current_template['TransformationParameters'])
        ensemble_input = current_template['Ensemble']
        current_template = pd.DataFrame(current_template).transpose()
        template_result.model_count += 1
        try:
            df_forecast = PredictWitch(current_template, df_train = df_train, forecast_length=forecast_length,frequency=frequency, 
                                          prediction_interval=prediction_interval, 
                                          no_negatives=no_negatives,
                                          preord_regressor_train = preord_regressor_train,
                                          preord_regressor_forecast = preord_regressor_forecast, 
                                          holiday_country = holiday_country,
                                          startTimeStamps = startTimeStamps,
                                          random_seed = random_seed, verbose = verbose,
                                       template_cols = template_cols)
            if verbose > 0:
                print("Model Number: {} with model {}".format(str(template_result.model_count), df_forecast.model_name))
            
            from autots.evaluator.metrics import PredictionEval
            model_error = PredictionEval(df_forecast, df_test, series_weights = weights)
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
def RandomTemplate(n: int = 10):
    """"
    Returns a template dataframe of randomly generated transformations, models, and hyperparameters
    
    Args:
        n (int): number of random models to return
    """
    n = abs(int(n))
    template = pd.DataFrame()
    counter = 0
    while (len(template.index) < n):
        model_str = np.random.choice(ModelNames)
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
                       top_n: int = 15, template_cols: list = ['Model','ModelParameters','TransformationParameters','Ensemble']):
    """
    Returns new template given old template with model accuracies
    
    Args:
        model_results (pandas.DataFrame): models that have actually been run
        submitted_paramters (pandas.DataFrame): models tried (may have returned different parameters to results)
    
    """
    new_template = pd.DataFrame()
    
    sorted_results = model_results[model_results['Ensemble'] == 0].copy().sort_values(by = sort_column, ascending = sort_ascending, na_position = 'last')
    # mutation
    for index, row in sorted_results.drop_duplicates(subset = "Model", keep = 'first').head(top_n).iterrows():
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

    # recombination of transforms across models
    recombination = sorted_results.tail(len(sorted_results.index) - 1).copy()
    recombination['TransformationParameters'] = sorted_results['TransformationParameters'].shift(1).tail(len(sorted_results.index) - 1)
    new_template = pd.concat([new_template, recombination.head(top_n)[template_cols]], axis = 0, ignore_index = True, sort = False)
    
    # internal recombination of model parameters, not implemented because some options are mutually exclusive.
    # Recombine best two of each model, if two or more present
    
    # remove generated models which have already been tried
    sorted_results = pd.concat([submitted_parameters, sorted_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
    new_template = UniqueTemplates(sorted_results, new_template, selection_cols = template_cols).head(max_results)
    return new_template

def validation_aggregation(validation_results):
    """
    Aggregates a TemplateEvalObject
    """
    groupby_cols = ['ID', 'Model', 'ModelParameters', 'TransformationParameters', 'Ensemble'] # , 'Exceptions'
    col_aggs = {'Runs': 'sum',
                # 'TransformationRuntime': 'mean', 
                # 'FitRuntime': 'mean', 
                # 'PredictRuntime': 'mean',
                # 'TotalRuntime': 'mean',
                'smape': 'mean',
                'mae': 'mean',
                'rmse': 'mean',
                'containment': 'mean',
                'smape_weighted': 'mean',
                'mae_weighted': 'mean',
                'rmse_weighted': 'mean',
                'containment_weighted': 'mean',
                'Score': 'mean'
                }
    validation_results.model_results = validation_results.model_results[pd.isnull(validation_results.model_results['Exceptions'])]
    validation_results.model_results = validation_results.model_results.replace([np.inf, -np.inf], np.nan)
    validation_results.model_results = validation_results.model_results.groupby(groupby_cols).agg(col_aggs)
    validation_results.model_results = validation_results.model_results.reset_index(drop = False)

    validation_results.model_results_per_timestamp_smape = validation_results.model_results_per_timestamp_smape.groupby('ID').mean()
    validation_results.model_results_per_timestamp_mae = validation_results.model_results_per_timestamp_mae.groupby('ID').mean()
    validation_results.model_results_per_series_smape = validation_results.model_results_per_series_smape.groupby('ID').mean()
    validation_results.model_results_per_series_mae = validation_results.model_results_per_series_mae.groupby('ID').mean()
    return validation_results

def generate_score(model_results, metric_weighting: dict = {}, prediction_interval: float = 0.9):
    """
    Generates score based on relative accuracies
    """
    try:
        smape_weighting = metric_weighting['smape_weighting']
    except:
        smape_weighting = 9
    try:
        mae_weighting = metric_weighting['mae_weighting']
    except:
        mae_weighting = 1
    try:
        rmse_weighting = metric_weighting['rmse_weighting']
    except:
        rmse_weighting = 5
    try:
        containment_weighting = metric_weighting['containment_weighting']
    except:
        containment_weighting = 1
    try:
        runtime_weighting = metric_weighting['runtime_weighting'] * 0.1
    except:
        runtime_weighting = 0.5
    smape_score = model_results['smape_weighted']/model_results['smape_weighted'].min(skipna=True) # smaller better
    rmse_score = model_results['rmse_weighted']/model_results['rmse_weighted'].min(skipna=True) # smaller better
    mae_score = model_results['mae_weighted']/model_results['mae_weighted'].min(skipna=True) # smaller better
    containment_score = (abs(prediction_interval - model_results['containment'])) # from 0 to 1, smaller better
    runtime_score = model_results['TotalRuntime']/(model_results['TotalRuntime'].min(skipna=True) + datetime.timedelta(minutes = 1)) # smaller better
    return (smape_score * smape_weighting) + (mae_score * mae_weighting) + (rmse_score * rmse_weighting) + (containment_score * containment_weighting) + (runtime_score * runtime_weighting)

