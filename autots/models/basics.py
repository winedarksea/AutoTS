"""
Naives and Others Requiring No Additional Packages Beyond Numpy and Pandas
"""
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject
from autots.evaluator.auto_model import PredictionObject


class Zeroes(ModelObject):
    """Naive forecasting predicting a dataframe of zeroes (0's)
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "Zeroes", frequency: str = 'infer', prediction_interval: float = 0.9):
        ModelObject.__init__(self, name, frequency, prediction_interval)
    def fit(self, df):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, regressor = [], just_point_forecast = False):
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
            prediction = PredictionObject(forecast_length=forecast_length,lower_forecast=df,
                                          forecast=df, upper_forecast=df,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime)
            
            return prediction
    def get_new_params(method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}

class LastValue(ModelObject):
    """Naive forecasting predicting a dataframe of the last series value
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "LastValue", frequency: str = 'infer', prediction_interval: float = 0.9):
        ModelObject.__init__(self, name, frequency, prediction_interval)
    def fit(self, df):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        self.last_values = df.tail(1).values
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, regressor = [], just_point_forecast = False):
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
        df = pd.DataFrame(np.tile(self.last_values, (forecast_length,1)))
        df = pd.DataFrame(np.zeros((forecast_length,(self.train_shape[1]))), columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length) )
        if just_point_forecast:
            return df
        else:
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(forecast_length=forecast_length,lower_forecast=df,
                                          forecast=df, upper_forecast=df,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime)
            
            return prediction
    def get_new_params(method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}


MedValue = ModelResult("Median Naive")
try:
    startTime = datetime.datetime.now()
    MedValue.forecast = np.tile(train.median(axis = 0).values, (forecast_length,1))
    MedValue.runtime = datetime.datetime.now() - startTime
    
    MedValue.mae = pd.DataFrame(mae(test.values, MedValue.forecast)).mean(axis=0, skipna = True)
    MedValue.overall_mae = np.nanmean(MedValue.mae)
    MedValue.smape = smape(test.values, MedValue.forecast)
    MedValue.overall_smape = np.nanmean(MedValue.smape)
except Exception as e:
    print(e)
    error_list.extend([traceback.format_exc()])

currentResult = pd.DataFrame({
        'method': MedValue.name, 
        'runtime': MedValue.runtime, 
        'overall_smape': MedValue.overall_smape, 
        'overall_mae': MedValue.overall_mae,
        'object_name': 'MedValue'
        }, index = [0])
model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
