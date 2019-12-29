"""
Naives and Others Requiring No Additional Packages Beyond Numpy and Pandas
"""
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject
from autots.evaluator.auto_model import PredictionObject
from autots.tools.probabilistic import Point_to_Probability


class ZeroesNaive(ModelObject):
    """Naive forecasting predicting a dataframe of zeroes (0's)
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "ZeroesNaive", frequency: str = 'infer', prediction_interval: float = 0.9, holiday_country: str = 'US'):
        ModelObject.__init__(self, name, frequency, prediction_interval, holiday_country = holiday_country)
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, preord_regressor = [], just_point_forecast = False):
        """Generates forecast data immediately following dates of index supplied to .fit()
        
        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts
            
        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        df = pd.DataFrame(np.zeros((forecast_length,(self.train_shape[1]))), columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length) )
        if just_point_forecast:
            return df
        else:
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = df.index,
                                          forecast_columns = df.columns,
                                          lower_forecast=df,
                                          forecast=df, upper_forecast=df,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          model_parameters = self.get_params()))
            
            return prediction
        
    def get_params(self):
        """Return dict of current parameters
        """
        return {}
    
    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}

class LastValueNaive(ModelObject):
    """Naive forecasting predicting a dataframe of the last series value
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "LastValueNaive", frequency: str = 'infer', prediction_interval: float = 0.9, holiday_country: str = 'US'):
        ModelObject.__init__(self, name, frequency, prediction_interval, holiday_country = holiday_country)
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        self.last_values = df.tail(1).values
        self.df_train = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, preord_regressor = [], just_point_forecast = False):
        """Generates forecast data immediately following dates of index supplied to .fit()
        
        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts
            
        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        df = pd.DataFrame(np.tile(self.last_values, (forecast_length,1)), columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length))
        if just_point_forecast:
            return df
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, df, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = df.index,
                                          forecast_columns = df.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=df, upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {}
    
class MedValueNaive(ModelObject):
    """Naive forecasting predicting a dataframe of the series' median values
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "MedValueNaive", frequency: str = 'infer', prediction_interval: float = 0.9, holiday_country: str = 'US'):
        ModelObject.__init__(self, name, frequency, prediction_interval, holiday_country = holiday_country)
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        self.median_values = df.median(axis = 0).values
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, preord_regressor = [], just_point_forecast = False):
        """Generates forecast data immediately following dates of index supplied to .fit()
        
        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts
            
        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        df = pd.DataFrame(np.tile(self.median_values, (forecast_length,1)), columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length))
        if just_point_forecast:
            return df
        else:
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = df.index,
                                          forecast_columns = df.columns,
                                          lower_forecast=df,
                                          forecast=df, upper_forecast=df,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {}
