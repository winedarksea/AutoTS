"""
Statsmodels based Models
"""
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject
from autots.evaluator.auto_model import PredictionObject
from autots.tools.probabilistic import Point_to_Probability


class GLS(ModelObject):
    """Simple linear regression from statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "GLS", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country = holiday_country, random_seed = random_seed)
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        from statsmodels.regression.linear_model import GLS
        df = self.basic_profile(df)
        self.df_train = df
        self.model = GLS(df.values, (df.index.astype( int ).values), missing = 'drop').fit()
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
        index = self.create_forecast_index(forecast_length=forecast_length)
        forecast = self.model.predict(index.astype( int ).values)
        df = pd.DataFrame(forecast, columns = self.column_names, index = index)
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
        return {}
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {}

class GLM(ModelObject):
    """Simple linear regression from statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): type of regression (None, 'User')

    """
    def __init__(self, name: str = "GLM", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020, 
                 regression_type: str = None,
                 family = 'Gaussian', constant: bool = False, verbose: int = 1):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type,
                             holiday_country = holiday_country, random_seed = random_seed,
                             verbose = verbose)
        self.family = family
        self.constant = constant
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        
        df = self.basic_profile(df)
        self.df_train = df
        if self.verbose > 1:
            self.verbose = True
        else:
            self.verbose = False
        if self.regression_type == 'User':
            if (len(preord_regressor) != len(df.index)):
                self.regression_type = None
            self.preord_regressor_train = preord_regressor
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
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        from statsmodels.api import GLM
        X = (pd.to_numeric(self.df_train.index, errors = 'coerce',downcast='integer').values)
        if self.constant in [True, 'True', 'true']:
            from statsmodels.tools import add_constant
            X = add_constant(X, has_constant='add')
        if self.regression_type == 'User':
            if self.preord_regressor_train.ndim == 1:
                self.preord_regressor_train = np.array(self.preord_regressor_train).reshape(-1, 1)
            X = np.concatenate((X.reshape(-1, 1), self.preord_regressor_train), axis = 1)
        forecast = pd.DataFrame()
        self.df_train = self.df_train.replace(0, np.nan)
        fill_vals = self.df_train.abs().min(axis = 0, skipna = True)
        self.df_train = self.df_train.fillna(fill_vals).fillna(0.1)
        for y in self.df_train.columns:
            current_series = self.df_train[y]
            if str(self.family).lower() == 'poisson':
                from statsmodels.genmod.families.family import Poisson
                model = GLM(current_series.values, X, family= Poisson(), missing = 'drop').fit(disp = self.verbose)
            elif str(self.family).lower() == 'binomial':
                from statsmodels.genmod.families.family import Binomial
                model = GLM(current_series.values, X, family= Binomial(), missing = 'drop').fit(disp = self.verbose)
            elif str(self.family).lower() == 'negativebinomial':
                from statsmodels.genmod.families.family import NegativeBinomial
                model = GLM(current_series.values, X, family= NegativeBinomial(), missing = 'drop').fit(disp = self.verbose)
            elif str(self.family).lower() == 'tweedie':
                from statsmodels.genmod.families.family import Tweedie
                model = GLM(current_series.values, X, family= Tweedie(), missing = 'drop').fit(disp = self.verbose)
            elif str(self.family).lower() == 'gamma':
                from statsmodels.genmod.families.family import Gamma
                model = GLM(current_series.values, X, family= Gamma(), missing = 'drop').fit(disp = self.verbose)
            else:
                self.family = 'Gaussian'
                model = GLM(current_series.values, X, missing = 'drop').fit()
            Xf = pd.to_numeric(test_index, errors = 'coerce',downcast='integer').values
            if self.constant or self.constant == 'True':
                Xf = add_constant(Xf, has_constant='add')
            if self.regression_type == 'User':
                if preord_regressor.ndim == 1:
                    preord_regressor = np.array(preord_regressor).reshape(-1, 1)
                Xf = np.concatenate((Xf.reshape(-1, 1), preord_regressor), axis = 1)   
            current_forecast = model.predict((Xf))
            forecast = pd.concat([forecast, pd.Series(current_forecast)], axis = 1)
        df_forecast = pd.DataFrame(forecast)
        df_forecast.columns = self.column_names
        df_forecast.index = test_index
        if just_point_forecast:
            return df_forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, df_forecast, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = df_forecast.index,
                                          forecast_columns = df_forecast.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=df_forecast, upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self,method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        family_choice = np.random.choice(a = ['Gaussian', 'Poisson','Binomial','NegativeBinomial','Tweedie', 'Gamma'], size = 1, p = [0.1, 0.3, 0.1, 0.3, 0.1, 0.1]).item()
        constant_choice = np.random.choice(a = [False, True], size = 1, p = [0.95, 0.05]).item()
        regression_type_choice = np.random.choice(a = [None, 'User'], size = 1, p = [0.8, 0.2]).item()
        return {'family':family_choice,
                'constant':constant_choice,
                'regression_type': regression_type_choice
                }
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {
                'family':self.family,
                'constant':self.constant,
                'regression_type': self.regression_type
                }

class ETS(ModelObject):
    """Exponential Smoothing from Statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        damped (bool): passed through to statsmodel ETS
        trend (str): passed through to statsmodel ETS
        seasonal (bool): passed through to statsmodel ETS
        seasonal_periods (int): passed through to statsmodel ETS

    """
    def __init__(self, name: str = "ETS", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, damped: bool = False, 
                 trend: str = None, seasonal: str=None, seasonal_periods:int=None, 
                 holiday_country: str = 'US',random_seed: int = 2020, verbose: int = 0):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country = holiday_country, random_seed = random_seed, verbose = verbose)
        self.damped = damped
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        
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
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        forecast = pd.DataFrame()
        for series in self.df_train.columns:
            current_series = self.df_train[series].copy()
            esModel = ExponentialSmoothing(current_series, damped = self.damped, trend = self.trend, seasonal=self.seasonal,seasonal_periods=self.seasonal_periods, freq = self.frequency).fit()
            esPred = esModel.predict(start=test_index[0], end=test_index[-1])
            forecast = pd.concat([forecast, esPred], axis = 1)
        forecast.columns = self.column_names
        if forecast.isnull().all(axis = 0).astype(int).sum() > 0:
            print("One or more series have failed to optimize with ETS model")
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, forecast, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = test_index,
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
        trend_list = ["additive", "multiplicative", None]
        trend_probability = [0.2, 0.2, 0.6]
        trend_choice = np.random.choice(a = trend_list, size = 1, p = trend_probability).item()
        if trend_choice in ["additive", "multiplicative"]:
            damped_choice = np.random.choice([True, False], size = 1).item()
        else:
            damped_choice = False
        seasonal_list = ["additive", "multiplicative", None]
        seasonal_probability = [0.2, 0.2, 0.6]
        seasonal_choice = np.random.choice(a = seasonal_list, size = 1, p = seasonal_probability).item()
        if seasonal_choice in ["additive", "multiplicative"]:
            seasonal_period_choice = np.random.choice(a=['random_int', 2, 7, 12, 24, 28, 60, 364], size = 1, p = [0.15, 0.05, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]).item()
            if seasonal_period_choice == 'random_int':
                seasonal_period_choice = np.random.randint(2, 100, size = 1).item()
        else:
            seasonal_period_choice = None
        parameter_dict = {
                        'damped' : damped_choice,
                        'trend': trend_choice,
                        'seasonal': seasonal_choice,
                        'seasonal_periods': seasonal_period_choice
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'damped' : self.damped,
                        'trend': self.trend,
                        'seasonal': self.seasonal,
                        'seasonal_periods': self.seasonal_periods
                        }
        return parameter_dict

        
class ARIMA(ModelObject):
    """ARIMA from Statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        p (int): is the number of autoregressive steps,
        d (int): is the number of differences needed for stationarity
        q (int): is the number of lagged forecast errors in the prediction . 
        regression_type (str): type of regression (None, 'User', or 'Holiday')

    """
    def __init__(self, name: str = "ARIMA", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, p:int = 0, d:int = 1,
                 q:int = 0, regression_type: str = None, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type, 
                             holiday_country = holiday_country, random_seed = random_seed,
                             verbose = verbose)
        self.p = p
        self.d = d
        self.q = q
        self.order = (p, d, q)
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag
            self.regressor_train = holiday_flag(df.index, country = self.holiday_country).values
        else:
            if self.regression_type != None:
                if (len(preord_regressor) != len(df.index)):
                    self.regression_type = None
            self.regressor_train = preord_regressor
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
        from statsmodels.tsa.arima_model import ARIMA
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag
            preord_regressor = holiday_flag(test_index, country = self.holiday_country).values
        if self.regression_type != None:
            assert len(preord_regressor) == forecast_length, "regressor not equal to forecast length"
        forecast = pd.DataFrame()
        upper_forecast = pd.DataFrame()
        lower_forecast = pd.DataFrame()
        for series in self.df_train.columns:
            current_series = self.df_train[series].copy()
            try:
                if (self.regression_type == "User") or (self.regression_type == "Holiday"):
                    maModel = ARIMA(current_series, order = self.order, freq = self.frequency, exog = self.regressor_train).fit(maxiter = 600)
                    # maPred = maModel.predict(start=test_index[0], end=test_index[-1], exog = preord_regressor)
                    maPred, stderr, conf = maModel.forecast(steps = self.forecast_length, alpha = (1 - self.prediction_interval), exog = preord_regressor)
                else:
                    maModel = ARIMA(current_series, order = self.order, freq = self.frequency).fit(maxiter = 400, disp = self.verbose)
                    # maPred = maModel.predict(start=test_index[0], end=test_index[-1])
                    maPred, stderr, conf = maModel.forecast(steps = self.forecast_length, alpha = (1 - self.prediction_interval))
            except Exception:
                # maPred = pd.Series((np.zeros((forecast_length,))), index = test_index)
                maPred = np.zeros((forecast_length,))
                conf = np.zeros((forecast_length, 2))
            forecast = pd.concat([forecast, pd.Series(maPred, index = test_index)], axis = 1)
            conf = pd.DataFrame(conf, index = test_index)
            lower_forecast = pd.concat([lower_forecast, conf[0]], axis = 1)
            upper_forecast = pd.concat([upper_forecast, conf[1]], axis = 1)
        forecast.columns = self.column_names
        
        if just_point_forecast:
            return forecast
        else:
            # upper_forecast, lower_forecast = Point_to_Probability(self.df_train, forecast, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = test_index,
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
        p_choice = np.random.choice(a = [0,1,2,3,4,5,7,12], size = 1, p = [0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1]).item()
        d_choice = np.random.choice(a = [0,1,2,3], size = 1, p = [0.4, 0.3, 0.2, 0.1]).item()
        q_choice = np.random.choice(a = [0,1,2,3,4,5,7,12], size = 1, p = [0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1]).item()
        regression_list = [None, 'User', 'Holiday']
        regression_probability = [0.4, 0.4, 0.2]
        regression_choice = np.random.choice(a = regression_list, size = 1, p = regression_probability).item()

        parameter_dict = {
                        'p' : p_choice,
                        'd': d_choice,
                        'q': q_choice,
                        'regression_type': regression_choice
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'p' : self.p,
                        'd': self.d,
                        'q': self.q,
                        'regression_type': self.regression_type
                        }
        return parameter_dict


class UnobservedComponents(ModelObject):
    """UnobservedComponents from Statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        regression_type (str): type of regression (None, 'User', or 'Holiday')

    """
    def __init__(self, name: str = "UnobservedComponents", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, 
                 regression_type: str = None, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0,
                 level: bool = False, trend: bool = False, 
                 cycle: bool = False, damped_cycle: bool = False,
                 irregular: bool = False, stochastic_cycle: bool = False,
                 stochastic_trend: bool = False, stochastic_level: bool = False):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type, 
                             holiday_country = holiday_country, random_seed = random_seed,
                             verbose = verbose)
        self.level = level
        self.trend = trend
        self.cycle = cycle
        self.damped_cycle = damped_cycle
        self.irregular = irregular
        self.stochastic_level = stochastic_level
        self.stochastic_cycle = stochastic_cycle
        self.stochastic_trend = stochastic_trend
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        self.df_train = df
        
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag
            self.regressor_train = holiday_flag(df.index, country = self.holiday_country).values
        else:
            if self.regression_type != None:
                if (len(preord_regressor) != len(df.index)):
                    self.regression_type = None
            self.regressor_train = preord_regressor
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
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag
            preord_regressor = holiday_flag(test_index, country = self.holiday_country).values
        if self.regression_type != None:
            assert len(preord_regressor) == forecast_length, "regressor not equal to forecast length"
        forecast = pd.DataFrame()
        for series in self.df_train.columns:
            current_series = self.df_train[series].copy()
            try:
                if (self.regression_type == "User") or (self.regression_type == "Holiday"):
                    maModel = UnobservedComponents(current_series, freq = self.frequency, exog = self.regressor_train, 
                                                   level = self.level, trend = self.trend,cycle=self.cycle, 
                                                   damped_cycle=self.damped_cycle,irregular=self.irregular,
                                                   stochastic_cycle=self.stochastic_cycle, stochastic_level=self.stochastic_level,
                                                   stochastic_trend = self.stochastic_trend).fit(disp = self.verbose_bool)
                    maPred = maModel.predict(start=test_index[0], end=test_index[-1], exog = preord_regressor)
                else:
                    maModel = UnobservedComponents(current_series, freq = self.frequency, 
                                                   level = self.level, trend = self.trend,cycle=self.cycle, 
                                                   damped_cycle=self.damped_cycle,irregular=self.irregular,
                                                   stochastic_cycle=self.stochastic_cycle, stochastic_level=self.stochastic_level,
                                                   stochastic_trend = self.stochastic_trend).fit(disp = self.verbose_bool)
                    maPred = maModel.predict(start=test_index[0], end=test_index[-1])
            except Exception:
                maPred = pd.Series((np.zeros((forecast_length,))), index = test_index)
            forecast = pd.concat([forecast, maPred], axis = 1)
        forecast.columns = self.column_names
        
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, forecast, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = test_index,
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
        level_choice = np.random.choice(a = [True, False], size = 1, p = [0.5, 0.5]).item()
        if level_choice:
            trend_choice = np.random.choice(a = [True, False], size = 1, p = [0.5, 0.5]).item()
        else:
            trend_choice = False
        cycle_choice = np.random.choice(a = [True, False], size = 1, p = [0.5, 0.5]).item()
        if cycle_choice:
            damped_cycle_choice = np.random.choice(a = [True, False], size = 1, p = [0.5, 0.5]).item()
        else:
            damped_cycle_choice = False
        irregular_choice = np.random.choice(a = [True, False], size = 1, p = [0.5, 0.5]).item()
        stochastic_trend_choice = np.random.choice(a = [True, False], size = 1, p = [0.5, 0.5]).item()
        stochastic_level_choice = np.random.choice(a = [True, False], size = 1, p = [0.5, 0.5]).item()
        stochastic_cycle_choice = np.random.choice(a = [True, False], size = 1, p = [0.5, 0.5]).item()
        
        regression_list = [None, 'User', 'Holiday']
        regression_probability = [0.6, 0.2, 0.2]
        regression_choice = np.random.choice(a = regression_list, size = 1, p = regression_probability).item()

        parameter_dict = {
                        'level' : level_choice,
                        'trend': trend_choice,
                        'cycle': cycle_choice,
                        'damped_cycle': damped_cycle_choice,
                        'irregular': irregular_choice,
                        'stochastic_trend': stochastic_trend_choice,
                        'stochastic_level': stochastic_level_choice,
                        'stochastic_cycle': stochastic_cycle_choice,
                        'regression_type': regression_choice
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'level' : self.level,
                        'trend': self.trend,
                        'cycle': self.cycle,
                        'damped_cycle': self.damped_cycle,
                        'irregular':self.irregular,
                        'stochastic_trend': self.stochastic_trend,
                        'stochastic_level': self.stochastic_level,
                        'stochastic_cycle': self.stochastic_cycle,
                        'regression_type': self.regression_type
                        }
        return parameter_dict
    
    
class DynamicFactor(ModelObject):
    """DynamicFactor from Statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        regression_type (str): type of regression (None, 'User', or 'Holiday')

    """
    def __init__(self, name: str = "DynamicFactor", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, 
                 regression_type: str = None, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0,
                 k_factors: int = 1, factor_order: int = 0):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type, 
                             holiday_country = holiday_country, random_seed = random_seed,
                             verbose = verbose)
        self.k_factors = k_factors
        self.factor_order = factor_order
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        if self.verbose > 1:
            self.verbose = True
        else:
            self.verbose = False
        self.df_train = df
        
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag
            self.regressor_train = holiday_flag(df.index, country = self.holiday_country).values
        else:
            if self.regression_type != None:
                if (len(preord_regressor) != len(df.index)):
                    self.regression_type = None
            self.regressor_train = preord_regressor
        
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
            
         maModel = DynamicFactor(df_train, freq = 'MS', k_factors = 2, factor_order=2).fit()
         maPred = maModel.predict()
        """        
        predictStartTime = datetime.datetime.now()
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag
            preord_regressor = holiday_flag(test_index, country = self.holiday_country).values
        if self.regression_type != None:
            assert len(preord_regressor) == forecast_length, "regressor not equal to forecast length"     
        
        if (self.regression_type == "User") or (self.regression_type == "Holiday"):
            maModel = DynamicFactor(self.df_train, freq = self.frequency, exog = self.regressor_train, 
                                           k_factors = self.k_factors, factor_order=self.factor_order).fit(disp = self.verbose, maxiter=100)
            forecast = maModel.predict(start=test_index[0], end=test_index[-1], exog = preord_regressor.values.reshape(-1,1))
        else:
            maModel = DynamicFactor(self.df_train, freq = self.frequency, 
                                           k_factors = self.k_factors, factor_order=self.factor_order).fit(disp = self.verbose, maxiter=100)
            forecast = maModel.predict(start=test_index[0], end=test_index[-1])
        
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, forecast, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = test_index,
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
        k_factors_choice = np.random.choice(a = [0,1,2,3,10], size = 1, p = [0.1, 0.4, 0.2, 0.2, 0.1]).item()
        factor_order_choice = np.random.choice(a = [0,1,2,3], size = 1, p = [0.4, 0.3, 0.2, 0.1]).item()
        
        regression_list = [None, 'User', 'Holiday']
        regression_probability = [0.6, 0.2, 0.2]
        regression_choice = np.random.choice(a = regression_list, size = 1, p = regression_probability).item()

        parameter_dict = {
                        'k_factors' : k_factors_choice, 
                        'factor_order' : factor_order_choice,
                        'regression_type': regression_choice
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'k_factors' : self.k_factors, 
                        'factor_order' : self.factor_order,
                        'regression_type': self.regression_type
                        }
        return parameter_dict
    
   
class VECM(ModelObject):
    """VECM from Statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        regression_type (str): type of regression (None, 'User', or 'Holiday')

    """
    def __init__(self, name: str = "VECM", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, 
                 regression_type: str = None, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0,
                 deterministic: str = 'nc', k_ar_diff: int = 1):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type, 
                             holiday_country = holiday_country, random_seed = random_seed,
                             verbose = verbose)
        self.deterministic = deterministic
        self.k_ar_diff = k_ar_diff
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        self.df_train = df
        
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag
            self.regressor_train = holiday_flag(df.index, country = self.holiday_country).values
        else:
            if self.regression_type != None:
                if (len(preord_regressor) != len(df.index)):
                    self.regression_type = None
            self.regressor_train = preord_regressor
        
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
        from statsmodels.tsa.vector_ar.vecm import VECM
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag
            preord_regressor = holiday_flag(test_index, country = self.holiday_country).values
        if self.regression_type != None:
            assert len(preord_regressor) == forecast_length, "regressor not equal to forecast length"     
        
        if (self.regression_type == "User") or (self.regression_type == "Holiday"):
            maModel = VECM(self.df_train, freq = self.frequency, exog = self.regressor_train, 
                                           deterministic=self.deterministic, k_ar_diff=self.k_ar_diff).fit()
            # forecast = maModel.predict(start=test_index[0], end=test_index[-1], exog = preord_regressor)
            forecast = maModel.predict(steps = len(test_index), exog = preord_regressor)
        else:
            maModel = VECM(self.df_train, freq = self.frequency, 
                                           deterministic=self.deterministic,k_ar_diff=self.k_ar_diff).fit()
            # forecast = maModel.predict(start=test_index[0], end=test_index[-1])
            forecast = maModel.predict(steps = len(test_index))
        forecast = pd.DataFrame(forecast, index = test_index, columns = self.column_names)
        
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, forecast, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = test_index,
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
        deterministic_choice = np.random.choice(a = ["nc", "co", "ci", "lo", "li","cili","colo"], size = 1, p = [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).item()
        k_ar_diff_choice = np.random.choice(a = [0,1,2,3], size = 1, p = [0.1, 0.5, 0.2, 0.2]).item()
        
        regression_list = [None, 'User', 'Holiday']
        regression_probability = [0.6, 0.2, 0.2]
        regression_choice = np.random.choice(a = regression_list, size = 1, p = regression_probability).item()

        parameter_dict = {
                        'deterministic' : deterministic_choice, 
                        'k_ar_diff' : k_ar_diff_choice,
                        'regression_type': regression_choice
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'deterministic' : self.deterministic, 
                        'k_ar_diff' : self.k_ar_diff,
                        'regression_type': self.regression_type
                        }
        return parameter_dict


class VARMAX(ModelObject):
    """VARMAX from Statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        regression_type (str): type of regression (None, 'User', or 'Holiday')
    """
    def __init__(self, name: str = "VARMAX", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, 
                 regression_type: str = None, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0,
                 order: tuple = (1, 0), trend: str = 'c'):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type, 
                             holiday_country = holiday_country, random_seed = random_seed,
                             verbose = verbose)
        self.order = order
        self.trend = trend
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        
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
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        from statsmodels.tsa.statespace.varmax import VARMAX

        maModel = VARMAX(self.df_train, freq = self.frequency, order = self.order, trend = self.trend).fit(disp = self.verbose_bool)
        forecast = maModel.predict(start=test_index[0], end=test_index[-1])
        
        if just_point_forecast:
            return forecast
        else:
            # point, lower_forecast, upper_forecast = maModel.forecast_interval(steps = self.forecast_length, alpha = 1 - self.prediction_interval)            
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, forecast, prediction_interval = self.prediction_interval)

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = test_index,
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
        ar_choice = np.random.choice(a = [0,1,2,7], size = 1, p = [0.1, 0.5, 0.2, 0.2]).item()
        ma_choice = np.random.choice(a = [0,1,2], size = 1, p = [0.5, 0.3, 0.2]).item()
        trend_choice = np.random.choice(a = ['n','c','t','ct', 'poly'], size = 1, p = [0.1, 0.5, 0.1, 0.2, 0.1]).item()
        if trend_choice == 'poly':
            trend_choice = [np.random.randint(0,2,1).item(), np.random.randint(0,2,1).item(),np.random.randint(0,2,1).item(),np.random.randint(0,2,1).item()]
        return {
                'order':(ar_choice, ma_choice),
                'trend':trend_choice
                }
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                'order':self.order,
                'trend':self.trend
                }
        return parameter_dict
    