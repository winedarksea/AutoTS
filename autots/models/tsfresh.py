"""
tsfresh - automated feature extraction

n_jobs>1 causes Windows issues, sometimes
"""

import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject
from autots.evaluator.auto_model import PredictionObject
from autots.tools.probabilistic import Point_to_Probability

try:
    from tsfresh.feature_extraction import extract_features, EfficientFCParameters, MinimalFCParameters
except Exception: # except ImportError
    _has_tsfresh = False
else:
    _has_tsfresh = True


class RandomForestRolling(ModelObject):
    """Simple regression-framed approach to forecasting using sklearn

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
                 num_subsamples: int = 10):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type, 
                             holiday_country = holiday_country, 
                             random_seed = random_seed, verbose = verbose)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_subsamples = num_subsamples
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        if not _has_tsfresh:
            raise ImportError("Package tsfresh is required")
        
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
        if not _has_tsfresh:
            raise ImportError("Package tsfresh is required")
        # num_subsamples = 10
        predictStartTime = datetime.datetime.now()
        
        date_col_name = df_train.index.name
        row_count = int(np.floor((len(df_train.index)-1)/num_subsamples))
        X = pd.DataFrame()
        Y = pd.DataFrame()
        for start in range(0, df_train.shape[0]-row_count, row_count):
            df_subset = df_train.iloc[start:start + row_count]            
            tsfresh_long = df_subset.reset_index().melt(id_vars = date_col_name, var_name = 'series_id', value_name = 'value')
            tsfresh_long = tsfresh_long.fillna(method = 'ffill').fillna(method = 'bfill')
            if verbose > 0:
                extracted_features = extract_features(tsfresh_long, column_id="series_id", column_sort=date_col_name, column_value = 'value', default_fc_parameters=MinimalFCParameters(), n_jobs=1) # MinimalFCParameters, EfficientFCParameters
            else:
                extracted_features = extract_features(tsfresh_long, column_id="series_id", column_sort=date_col_name, column_value = 'value', disable_progressbar = True,  default_fc_parameters=MinimalFCParameters(), n_jobs=1)        
            a = extracted_features.to_numpy().flatten()
            X = pd.concat([X, pd.DataFrame(a.reshape(-1, len(a)))],axis = 0)
            Y = pd.concat([Y, pd.DataFrame(df_train.iloc[start+row_count]).transpose()], axis = 0)
        endTime = datetime.datetime.now()
        
        X =  X.dropna(how='any', axis = 1)
        Y = Y.fillna(method = 'ffill').fillna(method = 'bfill')
        
        index = self.create_forecast_index(forecast_length=forecast_length)
        if len(preord_regressor) == 0:
            self.regression_type = 'None'
        
        from sklearn.ensemble import RandomForestRegressor
        
        if self.regression_type == 'User':
            X = pd.concat([X, self.regressor_train], axis = 1)
            # complete_regressor = pd.concat([self.regressor_train, preord_regressor], axis = 0)
        
        regr = RandomForestRegressor(random_state= self.random_seed, n_estimators=self.n_estimators, verbose = self.verbose)
        regr.fit(X, Y)
        
        combined_index = (self.df_train.index.append(index))
        forecast = pd.DataFrame()
        sktraindata = df_train.copy()
        for x in range(forecast_length):
            
            tsfresh_long = sktraindata.tail(row_count).reset_index().melt(id_vars = date_col_name, var_name = 'series_id', value_name = 'value')
            tsfresh_long = tsfresh_long.fillna(method = 'ffill').fillna(method = 'bfill')
            if verbose > 0:
                extracted_features = extract_features(tsfresh_long, column_id="series_id", column_sort=date_col_name, column_value = 'value', default_fc_parameters=MinimalFCParameters(), n_jobs=1) # MinimalFCParameters EfficientFCParameters
            else:
                extracted_features = extract_features(tsfresh_long, column_id="series_id", column_sort=date_col_name, column_value = 'value', disable_progressbar = True,  default_fc_parameters=MinimalFCParameters(), n_jobs=1)        
            a = extracted_features.to_numpy().flatten()
            df = pd.DataFrame(a.reshape(-1, len(a))).fillna(0)
            if self.regression_type == 'User':
                df = pd.concat([df, preord_regressor.iloc[x]], axis = 1)
            df = df[X.columns]
            rfPred =  pd.DataFrame(regr.predict(df.tail(1).values))
        
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
        regression_choice = np.random.choice(a=['None','User'], size = 1, p = [0.7, 0.3]).item()

        parameter_dict = {
                        'n_estimators': 1000,
                        'max_depth': None,
                        'min_samples_split': 2,
                        'num_subsamples': 10,
                        'regression_type': 'None'
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'n_estimators': self.n_estimators,
                        'max_depth': self.max_depth,
                        'min_samples_split': self.min_samples_split,
                        'num_subsamples': self.num_subsamples,
                        'regression_type': self.regression_type
                        }
        return parameter_dict



