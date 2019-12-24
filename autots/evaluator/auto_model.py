import numpy as np
import pandas as pd
import datetime

class ModelObject(object):
    """
    Models should all have methods:
        .fit(df) (taking a DataFrame with DatetimeIndex and n columns of n timeseries)
        .predict(forecast_length = int)
    
    Args:
        name (str): Model Name
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
    """
    def __init__(self, name: str = "Uniniated Model Name", frequency: str = 'infer', prediction_interval: float = 0.9, fit_runtime=datetime.timedelta(0)):
        self.name = name
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.fit_runtime = fit_runtime
    
    def __repr__(self):
        return 'ModelObject of ' + self.name
    
    def basic_profile(self, df):
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
    
    def get_new_params(method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}

class PredictionObject(object):
    def __init__(self, forecast_length: int = 0, lower_forecast = np.nan, forecast = np.nan, upper_forecast = np.nan, prediction_interval: float = 0.9, predict_runtime=datetime.timedelta(0)):
        self.forecast_length = forecast_length
        self.lower_forecast = lower_forecast
        self.forecast = forecast
        self.upper_forecast = upper_forecast
        self.prediction_interval = prediction_interval
        self.predict_runtime = predict_runtime


def ModelMonster(model: str, parameters: dict):
    """Directs strings and parameters to appropriate model objects.
    
    Args:
        model (str): Name of Model Function
        parameters (dict): Dictionary of parameters to pass through to model
    """
    if model = 'Zeroes':
        from autots.models.basics import Zeroes
        return Zeroes()