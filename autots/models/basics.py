"""
Naives and Others Requiring No Additional Packages Beyond Numpy and Pandas
"""
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject, PredictionObject, seasonal_int
from autots.tools.probabilistic import Point_to_Probability, historic_quantile


class ZeroesNaive(ModelObject):
    """Naive forecasting predicting a dataframe of zeroes (0's)
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "ZeroesNaive", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country = holiday_country,
                             random_seed = random_seed, verbose = verbose)
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
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self,method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {}

class LastValueNaive(ModelObject):
    """Naive forecasting predicting a dataframe of the last series value
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "LastValueNaive", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country = holiday_country, random_seed = random_seed)
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        self.last_values = df.tail(1).values
        # self.df_train = df
        self.lower, self.upper = historic_quantile(df, prediction_interval = self.prediction_interval)
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
            # upper_forecast, lower_forecast = Point_to_Probability(self.df_train, df, prediction_interval = self.prediction_interval, method = 'historic_quantile')
            upper_forecast = df.astype(float) + self.upper
            lower_forecast = df.astype(float) - self.lower
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = df.index,
                                          forecast_columns = df.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=df, upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self,method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {}
    
class AverageValueNaive(ModelObject):
    """Naive forecasting predicting a dataframe of the series' median values
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "AverageValueNaive", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0,
                 method: str = 'Median'):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country = holiday_country, random_seed = random_seed,
                             verbose = verbose)
        self.method = method
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        if str(self.method).lower() == 'median':
            self.average_values = df.median(axis = 0).values
        if str(self.method).lower() == 'mean':
            self.average_values = df.mean(axis = 0).values
        if str(self.method).lower() == 'mode':
            self.average_values = df.mode(axis = 0).iloc[0].fillna(df.median(axis=0)).values
        self.fit_runtime = datetime.datetime.now() - self.startTime
        self.lower, self.upper = historic_quantile(df, prediction_interval = self.prediction_interval)
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
        df = pd.DataFrame(np.tile(self.average_values, (forecast_length,1)), columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length))
        if just_point_forecast:
            return df
        else:
            # upper_forecast, lower_forecast = Point_to_Probability(self.df_train, df, prediction_interval = self.prediction_interval, method = 'historic_quantile')
            upper_forecast = df.astype(float) + self.upper
            lower_forecast = df.astype(float) - self.lower
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = df.index,
                                          forecast_columns = df.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=df, upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self,method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        method_choice = np.random.choice(a=['Median', 'Mean', 'Mode'], size = 1, p = [0.3, 0.6, 0.1]).item()
        return {
                'method': method_choice
                }
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {
                'method' : self.method
                }


class SeasonalNaive(ModelObject):
    """Naive forecasting predicting a dataframe with seasonal (lag) forecasts.
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        method (str): Either 'LastValue' (use last value of lag n) or 'Mean' (avg of all lag n)
        lag_1 (int): The lag of the seasonality, should int > 1.
        lag_2 (int): Optional second lag of seasonality which is averaged with first lag to produce forecast.

    """
    def __init__(self, name: str = "SeasonalNaive", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020,
                 lag_1: int = 7, lag_2: int = None, method: str = 'LastValue'):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country = holiday_country, random_seed = random_seed)
        self.lag_1 = abs(int(lag_1))
        self.lag_2 = lag_2
        if str(self.lag_2).isdigit():
            self.lag_2 = abs(int(self.lag_2))
        self.method = method
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        if self.lag_1 == self.lag_2:
            raise ValueError("Lag 2 cannot equal Lag 1")
        df = self.basic_profile(df)
        self.df_train = df
        
        df_length = (self.train_shape[0])
        self.tile_values_lag_2 = None
        if self.method == 'Mean':
            tile_index = np.tile(np.arange(self.lag_1), int(np.ceil(df_length/self.lag_1)))
            tile_index = tile_index[len(tile_index)-(df_length):]
            df.index = tile_index
            self.tile_values_lag_1 = df.groupby(level = 0, axis = 0).mean()
            if str(self.lag_2).isdigit():
                if self.lag_2 == 1:
                    self.tile_values_lag_2 = df.tail(self.lag_2)
                else:
                    tile_index = np.tile(np.arange(self.lag_2), int(np.ceil(df_length/self.lag_2)))
                    tile_index = tile_index[len(tile_index)-(df_length):]
                    df.index = tile_index
                    self.tile_values_lag_2 = df.groupby(level = 0, axis = 0).mean()
        else:
            self.method == 'LastValue'
            self.tile_values_lag_1 = df.tail(self.lag_1)
            if str(self.lag_2).isdigit():
                self.tile_values_lag_2 = df.tail(self.lag_2)
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
        tile_len = len(self.tile_values_lag_1.index)
        df = pd.DataFrame(np.tile(self.tile_values_lag_1, (int(np.ceil(forecast_length/tile_len)),1))[0:forecast_length], columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length))
        if str(self.lag_2).isdigit():
            y = pd.DataFrame(np.tile(self.tile_values_lag_2, (int(np.ceil(forecast_length/len(self.tile_values_lag_2.index))),1))[0:forecast_length], columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length))
            df = (df + y) / 2
        # df = df.apply(pd.to_numeric, errors='coerce')
        df = df.astype(float)
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
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self,method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        lag_1_choice = seasonal_int()
        lag_2_choice = np.random.choice(a=['None', seasonal_int(include_one = True)], size = 1, p = [0.3, 0.7]).item()
        if str(lag_2_choice) == str(lag_1_choice):
            lag_2_choice = 1
        method_choice = np.random.choice(a=['Mean', 'LastValue'], size = 1, p = [0.5, 0.5]).item()
        return {
                'method' : method_choice,
                'lag_1' : lag_1_choice,
                'lag_2' : lag_2_choice
                }
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {
                'method' : self.method,
                'lag_1' : self.lag_1,
                'lag_2' : self.lag_2
                }
        

class ContouredMofitSimulation(ModelObject):
    """More dark magic created by the evil mastermind of this project.
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        

    """
    def __init__(self, name: str = "ContouredMofitSimulation", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US', random_seed: int = 2020,
                 phrase_len: str = '20thN',
                 comparison: str = 'pct_change_sign', #'pct_change_sign', 'magnitude', 'pct_change_magnitude','5bin'
                 shared: bool = False,
                 max_motifs: int = 500 # x or %n
                 ):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country = holiday_country, random_seed = random_seed)
        self.phrase_len = phrase_len
        self.comparison = comparison
        self.shared = shared
        self.max_motifs = max_motifs
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        if str(self.phrase_len).isdigit():
            self.phrase_n = self.phrase_len
        elif self.phrase_len == '10thN':
            self.phrase_n = int(np.floor((df.shape[0])/10))
        elif self.phrase_len == '100thN':
            self.phrase_n = int(np.floor((df.shape[0])/100))
        elif self.phrase_len == '1000thN':
            self.phrase_n = int(np.floor((df.shape[0])/1000))
        else:
            self.phrase_len = '20thN'
            self.phrase_n = int(np.floor((df.shape[0])/20))
        if (self.phrase_n > df.shape[0]) or (self.phrase_n <= 0):
            raise ValueError("phrase_len is inappropriate for the length of training data provided")
        
        df = self.basic_profile(df)
        if self.comparison == 'pct_change_sign':
            df = df.pct_change(periods=1, fill_method='ffill').tail(df.shape[0] - 1).fillna(0)
            # df = (df > 0).astype(int)
            # np.where(df > 0, 1, np.where(df < 0, -1, 0))
            df = df.where(df >= 0, -1).where(df <= 0, 1)
            df = df.replace({1 : 'a', 0: 'b', -1: 'c'})
            
            max_motif = 100
            phrase_n = 5
            last_motif = df.tail(phrase_n)
            numbers = np.random.choice((df.shape[0] - phrase_n), size=max_motif, replace=False)
            
            rand_row = np.random.randint(low = 0, high = (df.shape[0] - phrase_n), size = 1).item()
            rand_slice = df.iloc[rand_row:(rand_row + phrase_n),]
            comparative = pd.concat([last_motif.sum(), rand_slice.sum()], axis = 1)
            
            from sklearn.metrics.pairwise import pairwise_distances
            temp = pairwise_distances(last_motif.transpose(), rand_slice.transpose(), metric = "hamming")
            
            test = rand_slice.reset_index(drop = True).transpose().set_index(np.repeat(rand_row, (df.shape[1],)), append = True)
            test.index.get_level_values(1)
        elif self.comparison == 'pct_change_magnitude':
            df = df.pct_change(periods=1, fill_method='ffill').tail(df.shape[0] - 1).fillna(0)
        
        
        
        if self.shared:
            df
        else:
            df
        """
        In fit phase, only select motifs.
            table: start index, weight, column it applies to, and count of rows that follow motif
            slice into possible motifs
            compare motifs (efficiently)
        Select motifs that do not include last motif if possible
        
        Select slices (all columns at same) into new array, reshuffle that
        Similarity:
            string distance
                sign
            unordered pairwise similarity
                pct_change
            rmse
                pct_change
                magnitude
        
        KNN
            
        """
        
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
        tile_len = len(self.tile_values_lag_1.index)
        df = pd.DataFrame(np.tile(self.tile_values_lag_1, (int(np.ceil(forecast_length/tile_len)),1))[0:forecast_length], columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length))
        if str(self.lag_2).isdigit():
            y = pd.DataFrame(np.tile(self.tile_values_lag_2, (int(np.ceil(forecast_length/len(self.tile_values_lag_2.index))),1))[0:forecast_length], columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length))
            df = (df + y) / 2
        # df = df.apply(pd.to_numeric, errors='coerce')
        df = df.astype(float)
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
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self,method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        lag_1_choice = seasonal_int()
        lag_2_choice = np.random.choice(a=['None', seasonal_int(include_one = True)], size = 1, p = [0.3, 0.7]).item()
        if str(lag_2_choice) == str(lag_1_choice):
            lag_2_choice = 1
        method_choice = np.random.choice(a=['Mean', 'LastValue'], size = 1, p = [0.5, 0.5]).item()
        return {
                'method' : method_choice,
                'lag_1' : lag_1_choice,
                'lag_2' : lag_2_choice
                }
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {
                'method' : self.method,
                'lag_1' : self.lag_1,
                'lag_2' : self.lag_2
                }


"""
model = SeasonalNaive()
model = model.fit(df_wide.fillna(0)[df_wide.columns[0:5]].astype(float))
prediction = model.predict(forecast_length = 14)
prediction.forecast
"""