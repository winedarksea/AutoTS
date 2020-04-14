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


def date_part(DTindex, method: str = 'simple'):
    """Create date part columns from pd.DatetimeIndex.
    
    Args:
        DTindex (pd.DatetimeIndex): datetime index to provide dates
        method (str): expanded' or 'simple' providing more or less columns
        
    Returns:
        pd.Dataframe with DTindex
    """
    date_part_df = pd.DataFrame({
        'year': DTindex.year,
        'month': DTindex.month,
        'day': DTindex.day,
        'weekday': DTindex.weekday
    })
    if method == 'expanded':
        date_part_df2 = pd.DataFrame({
            'hour': DTindex.hour,
            'week': DTindex.week,
            'quarter': DTindex.quarter,
            'dayofyear': DTindex.dayofyear,
            'midyear': ((DTindex.dayofyear > 74) &
                        (DTindex.dayofyear < 258)).astype(int),  # 2 season
            'weekend': (DTindex.weekday > 4).astype(int),  # weekend/weekday
            'month_end': (DTindex.is_month_end).astype(int),
            'month_start': (DTindex.is_month_start).astype(int),
            "quareter_end": (DTindex.is_quarter_end).astype(int),
            'year_end': (DTindex.is_year_end).astype(int),
            'daysinmonth': DTindex.daysinmonth,
            'epoch': DTindex.astype(int)
        })
        date_part_df = pd.concat([date_part_df, date_part_df2], axis=1)

    return date_part_df

def rolling_x_regressor(df, mean_rolling_periods: int = 30,
                        macd_periods: int = None,
                        std_rolling_periods: int = 7,
                        max_rolling_periods: int = None,
                        min_rolling_periods: int = None,
                        ewm_alpha: float = 0.5,
                        additional_lag_periods: int = 7,
                        abs_energy: bool = False,
                        rolling_autocorr_periods: int = None,
                        add_date_part: str = None,
                        holiday: bool = False, holiday_country: str = 'US',
                        polynomial_degree: int = None,
                        ):
    """
    Generate more features from initial time series.
    
    macd_periods ignored if mean_rolling is None.
    
    Returns a dataframe of statistical features. Will need to be shifted by 1 or more to match Y for forecast.
    """
    X = df.copy()
    if str(mean_rolling_periods).isdigit():
        temp = df.rolling(int(mean_rolling_periods), min_periods = 1).median()
        X = pd.concat([X, temp], axis = 1)
        if str(macd_periods).isdigit():
            temp = df.rolling(int(macd_periods), min_periods = 1).median() - temp
            X = pd.concat([X, temp], axis = 1)
    if str(std_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(std_rolling_periods,
                                     min_periods = 1).std()], axis = 1)
    if str(max_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(max_rolling_periods,
                                     min_periods = 1).max()], axis = 1)
    if str(min_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(min_rolling_periods,
                                     min_periods = 1).min()], axis = 1)
    if str(ewm_alpha).replace('.', '').isdigit():
        X = pd.concat([X, df.ewm(alpha = ewm_alpha, ignore_na=True,
                                 min_periods=1).mean()], axis=1)
    if str(additional_lag_periods).isdigit():
        X = pd.concat([X, df.shift(additional_lag_periods)],
                      axis=1).fillna(method='bfill')
    if abs_energy:
        X = pd.concat([X, df.pow(other = ([2] * len(df.columns))).cumsum()],
                      axis=1)
    if str(rolling_autocorr_periods).isdigit():
        temp = df.rolling(rolling_autocorr_periods).apply(lambda x: x.autocorr(), raw=False)
        X = pd.concat([X, temp], axis = 1).fillna(method='bfill')
    if (add_date_part) in ['simple', 'expanded']:
        X = pd.concat([X, date_part(df.index, method = add_date_part)],
                      axis=1)
    if holiday:
        from autots.tools.holiday import holiday_flag
        X['holiday_flag_'] = holiday_flag(X.index, country=holiday_country)
        X['holiday_flag_future_'] = holiday_flag(X.index.shift(1,freq=pd.infer_freq(X.index)),
                                                 country=holiday_country)
    if str(polynomial_degree).isdigit():
        polynomial_degree = abs(int(polynomial_degree))
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(polynomial_degree)
        X = pd.DataFrame(poly.fit_transform(X))

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(method='ffill').fillna(method='bfill')

    X.columns = [x for x in range(len(X.columns))]

    return X


def retrieve_regressor(regression_model: dict =
                       {"model": 'Adaboost',
                        "model_params":
                            {'n_estimators': 50,
                             'base_estimator': 'DecisionTree',
                             'loss': 'linear',
                             'learning_rate': 1.0}},
                       verbose: int = 0, verbose_bool: bool = False,
                       random_seed: int = 2020):
    """Convert a model param dict to model object for regression frameworks."""
    if regression_model['model'] == 'ElasticNet':
        from sklearn.linear_model import MultiTaskElasticNet
        regr = MultiTaskElasticNet(alpha=1.0,
                                   random_state=random_seed)
        return regr
    elif regression_model['model'] == 'DecisionTree':
        from sklearn.tree import DecisionTreeRegressor
        regr = DecisionTreeRegressor(
            max_depth=regression_model["model_params"]['max_depth'],
            min_samples_split=regression_model["model_params"]['min_samples_split'],
            random_state=random_seed)
        return regr
    elif regression_model['model'] == 'MLP':
        from sklearn.neural_network import MLPRegressor
        regr = MLPRegressor(hidden_layer_sizes=(25, 15, 25),
                            verbose=verbose_bool, max_iter=250,
                            activation='tanh', solver='lbfgs',
                            random_state=random_seed)
        return regr
    elif regression_model['model'] == 'KNN':
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.neighbors import KNeighborsRegressor
        regr = MultiOutputRegressor(KNeighborsRegressor())
        return regr
    elif regression_model['model'] == 'Adaboost':
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import AdaBoostRegressor
        if regression_model["model_params"]['base_estimator'] == 'SVR':
            from sklearn.svm import LinearSVR
            svc = LinearSVR(verbose = verbose, random_state = random_seed)
            regr = MultiOutputRegressor(AdaBoostRegressor(
                base_estimator=svc,
                n_estimators=regression_model["model_params"]['n_estimators'],
                loss=regression_model["model_params"]['loss'],
                learning_rate=regression_model["model_params"]['learning_rate'],
                random_state=random_seed))
            return regr
        elif regression_model["model_params"]['base_estimator'] == 'LinReg':
            from sklearn.linear_model import LinearRegression
            linreg = LinearRegression()
            regr = MultiOutputRegressor(AdaBoostRegressor(
                base_estimator=linreg,
                n_estimators=regression_model["model_params"]['n_estimators'],
                loss=regression_model["model_params"]['loss'],
                learning_rate=regression_model["model_params"]['learning_rate'],
                random_state=random_seed))
            return regr
        else:
            regr = MultiOutputRegressor(AdaBoostRegressor(
                n_estimators=regression_model["model_params"]['n_estimators'],
                loss=regression_model["model_params"]['loss'],
                learning_rate=regression_model["model_params"]['learning_rate'],
                random_state=random_seed))
            return regr
    elif regression_model['model'] == 'xgboost':
        import xgboost as xgb
        from sklearn.multioutput import MultiOutputRegressor
        regr = MultiOutputRegressor(
            xgb.XGBRegressor(
                objective=regression_model["model_params"]['objective'],
                eta=regression_model["model_params"]['eta'],
                min_child_weight=regression_model["model_params"]['min_child_weight'],
                max_depth=regression_model["model_params"]['max_depth'],
                subsample=regression_model["model_params"]['subsample'],
                verbosity=verbose))
        return regr
    elif regression_model['model'] == 'SVM':
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import SVR
        regr = MultiOutputRegressor(SVR(kernel='rbf', gamma='scale',
                                        verbose=verbose_bool))
        return regr
    elif regression_model['model'] == 'BayesianRidge':
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.linear_model import BayesianRidge
        regr = MultiOutputRegressor(BayesianRidge())
        return regr
    else:
        regression_model['model'] = 'RandomForest'
        from sklearn.ensemble import RandomForestRegressor
        regr = RandomForestRegressor(random_state=random_seed,
                                     n_estimators=1000, verbose=verbose)
        return regr


def generate_regressor_params(models: list = ['RandomForest','ElasticNet',
                                              'MLP', 'DecisionTree', 'KNN',
                                              'Adaboost', 'SVM', 'BayesianRidge',
                                              'xgboost'],
                              model_probs: list = [0.1, 0.1,
                                                  0.02, 0.2, 0.02,
                                                  0.3, 0.025, 0.035,
                                                  0.2]):
    """Generate new parameters for input to regressor."""
    model = np.random.choice(a=models, size=1, p=model_probs).item()
    if model in ['xgboost', 'Adaboost', 'DecisionTree']:
        if model == 'Adaboost':
            param_dict = {"model": 'Adaboost',
                    "model_params": {
                        "n_estimators": np.random.choice([50, 100, 500],
                                                         p=[0.7, 0.2, 0.1],
                                                         size=1).item(),
                        "loss": np.random.choice(['linear', 'square', 'exponential'],
                                                 p=[0.8, 0.1, 0.1],
                                                 size=1).item(),
                        "base_estimator": np.random.choice(['DecisionTree',
                                                            'LinReg', 'SVR'],
                                                      p=[0.8, 0.1, 0.1],
                                                      size=1).item(),
                        "learning_rate": np.random.choice([1, 0.5],
                                                      p=[0.9, 0.1],
                                                      size=1).item()
                        }}
        elif model == 'xgboost':
            param_dict = {"model": 'xgboost',
                    "model_params": {
                        "objective": np.random.choice(['count:poisson',
                                                       'reg:squarederror',
                                                       'reg:gamma'],
                                                 p=[0.4, 0.5, 0.1],
                                                 size=1).item(),
                        "eta": np.random.choice([0.3],
                                                      p=[1.0],
                                                      size=1).item(),
                        "min_child_weight": np.random.choice([1, 2, 5],
                                                      p=[0.8, 0.1 , 0.1],
                                                      size=1).item(),
                        "max_depth": np.random.choice([3, 6, 9],
                                                      p=[0.1, 0.8, 0.1],
                                                      size=1).item(),
                        "subsample": np.random.choice([1, 0.7, 0.5],
                                                      p=[0.9, 0.05, 0.05],
                                                      size=1).item()
                        }}
        else:
            min_samples = np.random.choice([1, 2, 0.05],
                                           p=[0.5, 0.3, 0.2],
                                           size=1).item()
            min_samples = int(min_samples) if min_samples in [2] else min_samples
            param_dict = {"model": 'DecisionTree',
                    "model_params": {
                        "max_depth": np.random.choice([None, 3, 9],
                                                      p=[0.5, 0.3, 0.2],
                                                      size=1).item(),
                        "min_samples_split": min_samples
                        }}
    else:
        param_dict = {"model": model,
                    "model_params": {}}
    return param_dict


class RollingRegression(ModelObject):
    """General regression-framed approach to forecasting using sklearn.
    
    Who are you who are so wise in the ways of science?
    I am Arthur, King of the Britons. -Python

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holiday flags
        regression_type (str): type of regression (None, 'User')

    """

    def __init__(self, name: str = "RollingRegression",
                 frequency: str = 'infer',
                 prediction_interval: float = 0.9,
                 regression_type: str = None,
                 holiday_country: str = 'US',
                 verbose: int = 0, random_seed: int = 2020,
                 regression_model: dict =
                    {"model": 'Adaboost',
                     "model_params":
                         {'n_estimators': 50,
                          'base_estimator': 'DecisionTree',
                          'loss': 'linear',
                          'learning_rate': 1.0}},
                 holiday: bool = False, mean_rolling_periods: int = 30,
                 macd_periods: int = None,
                 std_rolling_periods: int = 7,
                 max_rolling_periods: int = 7, min_rolling_periods: int = 7,
                 ewm_alpha: float = 0.5, additional_lag_periods: int = 7,
                 abs_energy: bool = False,
                 rolling_autocorr_periods: int = None,
                 add_date_part: str = None,
                 polynomial_degree: int = None,
                 x_transform: str = None):
        ModelObject.__init__(self, name, frequency, prediction_interval,
                             regression_type=regression_type,
                             holiday_country=holiday_country,
                             random_seed=random_seed, verbose=verbose)
        self.regression_model = regression_model
        self.holiday = holiday
        self.mean_rolling_periods = mean_rolling_periods
        self.macd_periods = macd_periods,
        self.std_rolling_periods = std_rolling_periods
        self.max_rolling_periods = max_rolling_periods
        self.min_rolling_periods = min_rolling_periods
        self.ewm_alpha = ewm_alpha
        self.additional_lag_periods = additional_lag_periods
        self.abs_energy = abs_energy
        self.rolling_autocorr_periods = rolling_autocorr_periods
        self.add_date_part = add_date_part,
        self.polynomial_degree = polynomial_degree
        self.x_transform = x_transform

    def _x_transformer(self):
        if self.x_transform == 'FastICA':
            from sklearn.decomposition import FastICA
            x_transformer = FastICA(n_components=None,
                                    random_state=2020,
                                    whiten=True)
        elif self.x_transform == 'Nystroem':
            from sklearn.kernel_approximation import Nystroem
            half_size = int(self.sktraindata.shape[0] / 2) + 1
            max_comp = 200
            n_comp = max_comp if half_size > max_comp else half_size
            x_transformer = Nystroem(kernel='rbf', gamma=0.2,
                                     random_state=2020,
                                     n_components=n_comp)
        else:
            # self.x_transform = 'RmZeroVariance'
            from sklearn.feature_selection import VarianceThreshold
            x_transformer = VarianceThreshold(threshold=0.0)
        return x_transformer

    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            preord_regressor (pandas.DataFrame or Series): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        # if external regressor, do some check up
        if self.regression_type is not None:
            if ((np.array(preord_regressor).shape[0]) != (df.shape[0])):
                self.regression_type = None
            else:
                self.regressor_train = preord_regressor

        # define X and Y
        self.sktraindata = self.df_train.dropna(how='all', axis=0)
        self.sktraindata = self.sktraindata.fillna(method='ffill').fillna(method='bfill')
        Y = self.sktraindata.drop(self.sktraindata.head(2).index)
        Y.columns = [x for x in range(len(Y.columns))]
        X = rolling_x_regressor(self.sktraindata,
                                mean_rolling_periods=self.mean_rolling_periods,
                                macd_periods=self.macd_periods,
                                std_rolling_periods=self.std_rolling_periods,
                                additional_lag_periods=self.additional_lag_periods,
                                ewm_alpha=self.ewm_alpha,
                                abs_energy=self.abs_energy,
                                rolling_autocorr_periods=self.rolling_autocorr_periods,
                                add_date_part=self.add_date_part,
                                holiday=self.holiday,
                                holiday_country=self.holiday_country,
                                polynomial_degree=self.polynomial_degree)
        if self.regression_type == 'User':
            X = pd.concat([X, self.regressor_train], axis=1)

        if self.x_transform in ['FastICA', 'Nystroem', 'RmZeroVariance']:
            self.x_transformer = self._x_transformer()
            self.x_transformer = self.x_transformer.fit(X)
            X = pd.DataFrame(self.x_transformer.transform(X))
            X = X.replace([np.inf, -np.inf], 0).fillna(0)
        """
        Tail(1) is dropped to shift data to become forecast 1 ahead
        and the first one is dropped because it will least accurately represent
        rolling values
        """
        X = X.drop(X.tail(1).index).drop(X.head(1).index)
        
        # retrieve model object to train
        self.regr = retrieve_regressor(regression_model=self.regression_model,
                                       verbose=self.verbose,
                                       verbose_bool=self.verbose_bool,
                                       random_seed=self.random_seed)
        self.regr = self.regr.fit(X, Y)
        
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self
        
    def predict(self, forecast_length: int, preord_regressor = [], just_point_forecast: bool = False):
        """Generate forecast data immediately following dates of index supplied to .fit().
        
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
        if self.regression_type == 'User':
            complete_regressor = pd.concat([self.regressor_train, preord_regressor], axis = 0)
        
        combined_index = (self.df_train.index.append(index))
        forecast = pd.DataFrame()
        self.sktraindata.columns = [x for x in range(len(self.sktraindata.columns))]
        
        # forecast, 1 step ahead, then another, and so on
        for x in range(forecast_length):
            x_dat = rolling_x_regressor(self.sktraindata,
                                        mean_rolling_periods=self.mean_rolling_periods,
                                        macd_periods=self.macd_periods,
                                        std_rolling_periods=self.std_rolling_periods,
                                        additional_lag_periods=self.additional_lag_periods,
                                        ewm_alpha=self.ewm_alpha,
                                        abs_energy=self.abs_energy,
                                        rolling_autocorr_periods=self.rolling_autocorr_periods,
                                        add_date_part=self.add_date_part,
                                        holiday=self.holiday,
                                        holiday_country=self.holiday_country,
                                        polynomial_degree=self.polynomial_degree)
            if self.regression_type == 'User':
                x_dat = pd.concat([x_dat,
                                   complete_regressor.head(len(x_dat.index))],
                                  axis=1)
            if self.x_transform in ['FastICA', 'Nystroem', 'RmZeroVariance']:
                x_dat = pd.DataFrame(self.x_transformer.transform(x_dat))
                x_dat = x_dat.replace([np.inf, -np.inf], 0).fillna(0)

            rfPred = pd.DataFrame(self.regr.predict(x_dat.tail(1).values))
        
            forecast = pd.concat([forecast, rfPred], axis = 0, ignore_index = True)
            self.sktraindata = pd.concat([self.sktraindata, rfPred], axis = 0, ignore_index = True)
            self.sktraindata.index = combined_index[:len(self.sktraindata.index)]

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
        """Return dict of new parameters for parameter tuning."""
        model_choice = generate_regressor_params()
        mean_rolling_periods_choice = np.random.choice(a=[None, 2, 5, 7,
                                                          10, 30],
                                                       size=1,
                                                       p=[0.1, 0.1, 0.2, 0.2,
                                                          0.2, 0.2]).item()
        if mean_rolling_periods_choice is not None:
            macd_periods_choice = np.random.choice(a=[None, 5, 7, 10, 30],
                                                   size=1,
                                                   p=[0.8, 0.05, 0.05, 0.05, 0.05]).item()
            if macd_periods_choice == mean_rolling_periods_choice:
                macd_periods_choice = mean_rolling_periods_choice + 10
        else:
            macd_periods_choice = None
        std_rolling_periods_choice = np.random.choice(a = [None, 2, 5, 7, 10, 30], size = 1, p = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]).item()
        max_rolling_periods_choice = np.random.choice(a = [None, 2, 5, 7, 10, 30], size = 1, p = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]).item()
        min_rolling_periods_choice = np.random.choice(a = [None, 2, 5, 7, 10, 30], size = 1, p = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]).item()
        lag_periods_choice = np.random.choice(a = [None, 2, 6, 11, 30], size = 1, p = [0.2, 0.2, 0.2, 0.2, 0.2]).item()
        ewm_choice = np.random.choice(a=[None, 0.2, 0.5, 0.8], size = 1, p = [0.25, 0.25, 0.25, 0.25]).item()
        abs_energy_choice = np.random.choice(a=[True,False], size = 1, p = [0.3, 0.7]).item()
        rolling_autocorr_periods_choice = np.random.choice(a = [None, 2, 6, 11, 30], size = 1, p = [0.8, 0.05, 0.05, 0.05, 0.05]).item()
        add_date_part_choice = np.random.choice(a=[None, "simple", "expanded"],
                                                size=1,
                                                p=[0.4, 0.2, 0.4]).item()
        holiday_choice = np.random.choice(a=[True,False], size = 1, p = [0.3, 0.7]).item()
        polynomial_degree_choice = np.random.choice(a=[None,2], size = 1, p = [0.8, 0.2]).item()
        x_transform_choice = np.random.choice(a=[None, 'FastICA',
                                                 'Nystroem', 'RmZeroVariance'],
                                              size=1, p=[0.7, 0.05,
                                                         0.05, 0.2]).item()
        regression_choice = np.random.choice(a=[None,'User'], size = 1, p = [0.7, 0.3]).item()
        parameter_dict = {
                        'regression_model': model_choice,
                        'holiday': holiday_choice,
                        'mean_rolling_periods': mean_rolling_periods_choice,
                        'macd_periods': macd_periods_choice,
                        'std_rolling_periods': std_rolling_periods_choice,
                        'max_rolling_periods': max_rolling_periods_choice,
                        'min_rolling_periods': min_rolling_periods_choice,
                        'ewm_alpha': ewm_choice,
                        'additional_lag_periods': lag_periods_choice,
                        'abs_energy': abs_energy_choice,
                        'rolling_autocorr_periods': rolling_autocorr_periods_choice,
                        'add_date_part': add_date_part_choice,
                        'polynomial_degree': polynomial_degree_choice,
                        'x_transform': x_transform_choice,
                        'regression_type': regression_choice
                        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
                        'regression_model': self.regression_model,
                        'holiday': self.holiday,
                        'mean_rolling_periods': self.mean_rolling_periods,
                        'macd_periods': self.macd_periods,
                        'std_rolling_periods': self.std_rolling_periods,
                        'max_rolling_periods': self.max_rolling_periods,
                        'min_rolling_periods': self.min_rolling_periods,
                        'ewm_alpha': self.ewm_alpha,
                        'additional_lag_periods': self.additional_lag_periods,
                        'abs_energy': self.abs_energy,
                        'rolling_autocorr_periods': self.rolling_autocorr_periods,
                        'add_date_part': self.add_date_part,
                        'polynomial_degree': self.polynomial_degree,
                        'x_transform': self.x_transform,
                        'regression_type': self.regression_type,
                        }
        return parameter_dict




"""
model = RandomForestRolling(regression_type = 'User')
model = model.fit(df_wide.fillna(method='ffill').fillna(method='bfill'), preord_regressor = preord_regressor_train)
prediction = model.predict(forecast_length = 3, preord_regressor = preord_regressor_forecast)
prediction.forecast
"""