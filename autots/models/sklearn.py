"""
Sklearn dependent models

Elastic Net, Random Forest, KNN
"""
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject
from autots.evaluator.auto_model import PredictionObject
from autots.tools.probabilistic import Point_to_Probability

def rolling_x_regressor(df, mean_rolling_periods: int = 30, std_rolling_periods: int = 7, holiday: bool = False, holiday_country: str = 'US', polynomial_degree = None):
    """
    Generate more features from initial time series
    """
    X = pd.concat([df, df.rolling(mean_rolling_periods,min_periods = 1).mean(), df.rolling(7,min_periods = 1).std()], axis = 1)
    X.columns = [x for x in range(len(X.columns))]
    X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    if holiday:
        from autots.tools.holiday import holiday_flag
        X['holiday_flag_'] = holiday_flag(X.index, country = holiday_country).values
    
    if str(polynomial_degree).isdigit():
        polynomial_degree = abs(int(polynomial_degree))
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(polynomial_degree)
        X = poly.fit_transform(X)
    
    return X
    

class RandomForestRolling(ModelObject):
    """Simple regression-framed approach to forecasting using sklearn
    
    Who are you who are so wise in the ways of science?
    I am Arthur, King of the Britons. -Python

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holiday flags
        regression_type (str): type of regression (None, 'User')

    """
    def __init__(self, name: str = "RandomForestRolling", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, regression_type: str = None, holiday_country: str = 'US',
                 verbose: int = 0, random_seed: int = 2020,
                 n_estimators: int = 100, min_samples_split: float = 2, max_depth: int = None,
                 holiday: bool = False, mean_rolling_periods: int = 30, std_rolling_periods: int = 7,
                 polynomial_degree: int = None):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type, 
                             holiday_country = holiday_country, 
                             random_seed = random_seed, verbose = verbose)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.holiday = holiday
        self.mean_rolling_periods = mean_rolling_periods
        self.std_rolling_periods = std_rolling_periods
        self.polynomial_degree = polynomial_degree
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """       
        df = self.basic_profile(df)

        self.df_train = df
        self.regressor_train = preord_regressor
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self
    
    def predict(self, forecast_length: int, preord_regressor = [], just_point_forecast: bool = False):
        """Generates forecast data immediately following dates of index supplied to .fit()
        
        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts
            
        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """        
        predictStartTime = datetime.datetime.now()
        index = self.create_forecast_index(forecast_length=forecast_length)
        if len(preord_regressor) == 0:
            self.regression_type = 'None'
        
        from sklearn.ensemble import RandomForestRegressor
        sktraindata = self.df_train.dropna(how = 'all', axis = 0).fillna(method='ffill').fillna(method='bfill')
        Y = sktraindata.drop(sktraindata.head(2).index) 
        Y.columns = [x for x in range(len(Y.columns))]
        
        X = rolling_x_regressor(sktraindata, mean_rolling_periods=self.mean_rolling_periods, std_rolling_periods=self.std_rolling_periods,holiday=self.holiday, holiday_country=self.holiday_country, polynomial_degree=self.polynomial_degree)
        if self.regression_type == 'User':
            X = pd.concat([X, self.regressor_train], axis = 1)
            complete_regressor = pd.concat([self.regressor_train, preord_regressor], axis = 0)
            
        X = X.drop(X.tail(1).index).drop(X.head(1).index)
        
        regr = RandomForestRegressor(random_state= self.random_seed, n_estimators=self.n_estimators, verbose = self.verbose)
        regr.fit(X, Y)
        
        combined_index = (self.df_train.index.append(index))
        forecast = pd.DataFrame()
        sktraindata.columns = [x for x in range(len(sktraindata.columns))]
        
        for x in range(forecast_length):
            x_dat = rolling_x_regressor(sktraindata, mean_rolling_periods=self.mean_rolling_periods, std_rolling_periods=self.std_rolling_periods,holiday=self.holiday, holiday_country=self.holiday_country, polynomial_degree=self.polynomial_degree)
            if self.regression_type == 'User':
                x_dat = pd.concat([x_dat, complete_regressor.head(len(x_dat.index))], axis = 1)
            rfPred =  pd.DataFrame(regr.predict(x_dat.tail(1).values))
        
            forecast = pd.concat([forecast, rfPred], axis = 0, ignore_index = True)
            sktraindata = pd.concat([sktraindata, rfPred], axis = 0, ignore_index = True)
            sktraindata.index = combined_index[:len(sktraindata.index)]

        forecast.columns = self.column_names
        forecast.index = index
        
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, forecast, prediction_interval = self.prediction_interval)
            
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
        
        large p,d,q can be very slow (a p of 30 can take hours, whereas 5 takes seconds)
        """
        n_estimators_choice = np.random.choice(a = [100, 1000], size = 1, p = [0.2, 0.8]).item()
        max_depth_choice = np.random.choice(a = [None, 5, 10], size = 1, p = [0.8, 0.1, 0.1]).item()
        mean_rolling_periods_choice = np.random.choice(a = [2, 5, 7, 10, 30], size = 1, p = [0.2, 0.2, 0.2, 0.2, 0.2]).item()
        std_rolling_periods_choice = np.random.choice(a = [2, 5, 7, 10, 30], size = 1, p = [0.2, 0.2, 0.2, 0.2, 0.2]).item()
        holiday_choice = np.random.choice(a=[True,False], size = 1, p = [0.5, 0.5]).item()
        polynomial_degree_choice = np.random.choice(a=[None,2], size = 1, p = [0.8, 0.2]).item()
        regression_choice = np.random.choice(a=['None','User'], size = 1, p = [0.7, 0.3]).item()

        parameter_dict = {
                        'n_estimators': n_estimators_choice,
                        'max_depth': max_depth_choice,
                        'min_samples_split': 2,
                        'holiday': holiday_choice,
                        'mean_rolling_periods': mean_rolling_periods_choice,
                        'std_rolling_periods': std_rolling_periods_choice,
                        'polynomial_degree': polynomial_degree_choice,
                        'regression_type': regression_choice
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'n_estimators': self.n_estimators,
                        'max_depth': self.max_depth,
                        'min_samples_split': self.min_samples_split,
                        'holiday': self.holiday,
                        'mean_rolling_periods': self.mean_rolling_periods,
                        'std_rolling_periods': self.std_rolling_periods,
                        'polynomial_degree': self.polynomial_degree,
                        'regression_type': self.regression_type
                        }
        return parameter_dict




"""
model = RandomForestRolling(regression_type = 'User')
model = model.fit(df_wide.fillna(method='ffill').fillna(method='bfill'), preord_regressor = preord_regressor_train)
prediction = model.predict(forecast_length = 3, preord_regressor = preord_regressor_forecast)
prediction.forecast
"""