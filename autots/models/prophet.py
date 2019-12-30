"""
Facebook's Prophet
"""
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject
from autots.evaluator.auto_model import PredictionObject

from fbprophet import Prophet

class FBProphet(ModelObject):
    """Facebook's Prophet
    
    'thou shall count to 3, no more, no less, 3 shall be the number thou shall count, and the number of the counting
    shall be 3. 4 thou shall not count, neither count thou 2, excepting that thou then preceed to 3.' -Python
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holidays
        regression_type (str): type of regression (None, 'User')

    """
    def __init__(self, name: str = "FBProphet", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, 
                 holiday: bool = False, 
                 regression_type: str = None, holiday_country: str = 'US',
                 random_seed: int = 2020):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type, 
                             holiday_country = holiday_country, random_seed = random_seed)
        self.holiday = holiday
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        if self.regression_type != None:
            if (len(preord_regressor) != len(df)):
                self.regression_type = None
        random_two = "n9032380gflljWfu8233koWQop3"
        random_one = "nJIOVxgQ0vZGC7nx"
        self.regressor_name = random_one if random_one not in df.columns else random_two
        self.regressor_train = preord_regressor.copy()
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
        #if self.regression_type != None:
         #   assert len(preord_regressor) == forecast_length, "regressor not equal to forecast length"
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        forecast = pd.DataFrame()
        lower_forecast = pd.DataFrame()
        upper_forecast = pd.DataFrame()

        for series in self.df_train.columns:
            current_series = self.df_train.copy()
            current_series['y'] = current_series[series]
            current_series['ds'] = current_series.index
            current_series[self.regressor_name] = self.regressor_train
            
            m = Prophet(interval_width = self.prediction_interval)
            if self.holiday == True:
                m.add_country_holidays(country_name= self.holiday_country)
            if self.regression_type == 'User':
                m.add_regressor(self.regressor_name)
            m = m.fit(current_series)
            future = m.make_future_dataframe(periods=forecast_length)
            if self.regression_type == 'User':
                a = np.append(self.regressor_train.values, preord_regressor.values)
                future[self.regressor_name] = a
            fcst = m.predict(future)  
            fcst = fcst.tail(forecast_length) # remove the backcast
            forecast = pd.concat([forecast, fcst['yhat']], axis = 1)
            lower_forecast = pd.concat([lower_forecast, fcst['yhat_lower']], axis = 1)
            upper_forecast = pd.concat([upper_forecast, fcst['yhat_upper']], axis = 1)
        forecast.columns = self.column_names
        forecast.index = test_index
        lower_forecast.columns = self.column_names
        lower_forecast.index = test_index
        upper_forecast.columns = self.column_names
        upper_forecast.index = test_index
        
        if just_point_forecast:
            return forecast
        else:           
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = forecast.index,
                                          forecast_columns = forecast.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=forecast, 
                                          upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        holiday_choice = np.random.choice(a = [True, False], size = 1, p = [0.5, 0.5]).item()
        regression_list = [None, 'User']
        regression_probability = [0.5, 0.5]
        regression_choice = np.random.choice(a = regression_list, size = 1, p = regression_probability).item()

        parameter_dict = {
                        'holiday' : holiday_choice,
                        'regression_type': regression_choice
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'holiday' : self.holiday,
                        'regression_type': self.regression_type
                        }
        return parameter_dict