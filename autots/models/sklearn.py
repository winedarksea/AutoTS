"""
Sklearn dependent models

Decision Tree, Elastic Net,  Random Forest, MLPRegressor, KNN, Adaboost 
"""
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject
from autots.evaluator.auto_model import PredictionObject
from autots.tools.probabilistic import Point_to_Probability

def rolling_x_regressor(df, mean_rolling_periods: int = 30, std_rolling_periods: int = 7, 
                        max_rolling_periods: int = None, min_rolling_periods: int = None,
                        ewm_alpha: float = 0.5, additional_lag_periods: int = 7,
                        holiday: bool = False, holiday_country: str = 'US', polynomial_degree = None,
                        abs_energy: bool = False):
    """
    Generate more features from initial time series
    
    Returns a dataframe of statistical features. Will need to be shifted by 1 or more to match Y for forecast.
    """
    X = df.copy()
    if str(mean_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(int(mean_rolling_periods), min_periods = 1).median()], axis = 1)
    if str(std_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(std_rolling_periods, min_periods = 1).std()], axis = 1)
    if str(max_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(max_rolling_periods, min_periods = 1).max()], axis = 1)
    if str(min_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(min_rolling_periods, min_periods = 1).min()], axis = 1)
    if str(ewm_alpha).replace('.', '').isdigit():
        X = pd.concat([X, df.ewm(alpha = ewm_alpha, min_periods = 1).mean()], axis = 1)
    if str(additional_lag_periods).isdigit():
        X = pd.concat([X, df.shift(additional_lag_periods)], axis = 1)
    if abs_energy:
        X = pd.concat([X, df.pow(other = ([2] * len(df.columns))).cumsum()], axis = 1)
    if holiday:
        from autots.tools.holiday import holiday_flag
        X['holiday_flag_'] = holiday_flag(X.index, country = holiday_country).values
        X['holiday_flag_future_'] = holiday_flag(X.index + pd.Timedelta('1D'), country = holiday_country).values
    if str(polynomial_degree).isdigit():
        polynomial_degree = abs(int(polynomial_degree))
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(polynomial_degree)
        X = poly.fit_transform(X)
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    
    X.columns = [x for x in range(len(X.columns))]
    
    return X
    

class RollingRegression(ModelObject):
    """General regression-framed approach to forecasting using sklearn
    
    Who are you who are so wise in the ways of science?
    I am Arthur, King of the Britons. -Python

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holiday flags
        regression_type (str): type of regression (None, 'User')

    """
    def __init__(self, name: str = "RollingRegression", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, regression_type: str = None, holiday_country: str = 'US',
                 verbose: int = 0, random_seed: int = 2020,
                 regression_model: str = 'RandomForest',
                 holiday: bool = False, mean_rolling_periods: int = 30, std_rolling_periods: int = 7,
                 max_rolling_periods: int = 7, min_rolling_periods: int = 7,
                 ewm_alpha: float = 0.5, additional_lag_periods: int = 7,
                 polynomial_degree: int = None):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type, 
                             holiday_country = holiday_country, 
                             random_seed = random_seed, verbose = verbose)
        self.regression_model = regression_model
        self.holiday = holiday
        self.mean_rolling_periods = mean_rolling_periods
        self.std_rolling_periods = std_rolling_periods
        self.max_rolling_periods = max_rolling_periods
        self.min_rolling_periods = min_rolling_periods
        self.ewm_alpha = ewm_alpha
        self.additional_lag_periods = additional_lag_periods
        self.polynomial_degree = polynomial_degree
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """       
        df = self.basic_profile(df)

        self.df_train = df
        if self.regression_type is not None:
            if ((np.array(preord_regressor).shape[0]) != (df.shape[0])):
                self.regression_type = None
            else:
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
        
        sktraindata = self.df_train.dropna(how = 'all', axis = 0).fillna(method='ffill').fillna(method='bfill')
        Y = sktraindata.drop(sktraindata.head(2).index)
        Y.columns = [x for x in range(len(Y.columns))]
        
        X = rolling_x_regressor(sktraindata, mean_rolling_periods=self.mean_rolling_periods, std_rolling_periods=self.std_rolling_periods,holiday=self.holiday, holiday_country=self.holiday_country, polynomial_degree=self.polynomial_degree)
        if self.regression_type == 'User':
            X = pd.concat([X, self.regressor_train], axis = 1)
            complete_regressor = pd.concat([self.regressor_train, preord_regressor], axis = 0)
        # 1 is dropped to shift data, and the first one is dropped because it will least accurately represnt rolling values
        X = X.drop(X.tail(1).index).drop(X.head(1).index)
        
        if self.regression_model == 'ElasticNet':
            from sklearn.linear_model import MultiTaskElasticNet
            regr = MultiTaskElasticNet(alpha = 1.0, random_state= self.random_seed)
        elif self.regression_model == 'DecisionTree':
            from sklearn.tree import DecisionTreeRegressor
            regr = DecisionTreeRegressor(random_state= self.random_seed)
        elif self.regression_model == 'MLP':
            from sklearn.neural_network import MLPRegressor
            regr = MLPRegressor(hidden_layer_sizes=(25, 15, 25),verbose = self.verbose_bool, max_iter = 250,
                  activation='tanh', solver='lbfgs', random_state= self.random_seed)
        elif self.regression_model == 'KNN':
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.neighbors import KNeighborsRegressor
            regr = MultiOutputRegressor(KNeighborsRegressor())
        elif self.regression_model == 'Adaboost':
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.ensemble import AdaBoostRegressor
            regr = MultiOutputRegressor(AdaBoostRegressor(n_estimators = 200, random_state=self.random_seed))
        elif self.regression_model == 'SVM':
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.svm import SVR
            regr = MultiOutputRegressor(SVR(kernel='rbf', verbose = self.verbose_bool))
        elif self.regression_model == 'ComplementNB':
            from sklearn.multioutput import MultiOutputClassifier
            from sklearn.naive_bayes import ComplementNB
            regr = MultiOutputClassifier(ComplementNB())
        else:
            self.regression_model = 'RandomForest'
            from sklearn.ensemble import RandomForestRegressor
            regr = RandomForestRegressor(random_state= self.random_seed, n_estimators=1000, verbose = self.verbose)
        
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
        """
        model_choice = np.random.choice(a = ['RandomForest','ElasticNet', 'MLP', 'DecisionTree', 'KNN', 'Adaboost', 'SVM', 'ComplementNB'], size = 1, p = [0.2, 0.1, 0.02, 0.225, 0.02, 0.4, 0.025, 0.01]).item()
        mean_rolling_periods_choice = np.random.choice(a = [None, 2, 5, 7, 10, 30], size = 1, p = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]).item()
        std_rolling_periods_choice = np.random.choice(a = [None, 2, 5, 7, 10, 30], size = 1, p = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]).item()
        max_rolling_periods_choice = np.random.choice(a = [None, 2, 5, 7, 10, 30], size = 1, p = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]).item()
        min_rolling_periods_choice = np.random.choice(a = [None, 2, 5, 7, 10, 30], size = 1, p = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]).item()
        lag_periods_choice = np.random.choice(a = [None, 2, 6, 11, 30], size = 1, p = [0.2, 0.2, 0.2, 0.2, 0.2]).item()
        ewm_choice = np.random.choice(a=[None, 0.2, 0.5, 0.8], size = 1, p = [0.25, 0.25, 0.25, 0.25]).item()
        holiday_choice = np.random.choice(a=[True,False], size = 1, p = [0.3, 0.7]).item()
        polynomial_degree_choice = np.random.choice(a=[None,2], size = 1, p = [0.8, 0.2]).item()
        regression_choice = np.random.choice(a=[None,'User'], size = 1, p = [0.7, 0.3]).item()
        #lag_1_choice = np.random.choice(a=['random_int', 2, 7, 12, 24, 28, 60, 364], size = 1, p = [0.15, 0.05, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]).item()
        # if lag_1_choice == 'random_int':
        #    lag_1_choice = np.random.randint(2, 100, size = 1).item()
        parameter_dict = {
                        'regression_model': model_choice,
                        'holiday': holiday_choice,
                        'mean_rolling_periods': mean_rolling_periods_choice,
                        'std_rolling_periods': std_rolling_periods_choice,
                        'max_rolling_periods': max_rolling_periods_choice,
                        'min_rolling_periods': min_rolling_periods_choice,
                        'ewm_alpha': ewm_choice,
                        'additional_lag_periods': lag_periods_choice,
                        'polynomial_degree': polynomial_degree_choice,
                        'regression_type': regression_choice
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'regression_model': self.regression_model,
                        'holiday': self.holiday,
                        'mean_rolling_periods': self.mean_rolling_periods,
                        'std_rolling_periods': self.std_rolling_periods,
                        'max_rolling_periods': self.max_rolling_periods,
                        'min_rolling_periods': self.min_rolling_periods,
                        'ewm_alpha': self.ewm_alpha,
                        'additional_lag_periods': self.additional_lag_periods,
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