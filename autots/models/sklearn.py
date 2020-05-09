"""
Sklearn dependent models

Decision Tree, Elastic Net,  Random Forest, MLPRegressor, KNN, Adaboost
"""
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject, PredictionObject, seasonal_int
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
        temp = df.rolling(int(mean_rolling_periods), min_periods=1).median()
        X = pd.concat([X, temp], axis=1)
        if str(macd_periods).isdigit():
            temp = df.rolling(int(macd_periods), min_periods=1).median() - temp
            X = pd.concat([X, temp], axis=1)
    if str(std_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(std_rolling_periods,
                                     min_periods=1).std()], axis=1)
    if str(max_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(max_rolling_periods,
                                     min_periods=1).max()], axis=1)
    if str(min_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(min_rolling_periods,
                                     min_periods=1).min()], axis=1)
    if str(ewm_alpha).replace('.', '').isdigit():
        X = pd.concat([X, df.ewm(alpha=ewm_alpha, ignore_na=True,
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
    
    if add_date_part in ['simple', 'expanded']:
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
        regr = MLPRegressor(
            hidden_layer_sizes=regression_model["model_params"]['hidden_layer_sizes'],
            max_iter=regression_model["model_params"]['max_iter'],
            activation=regression_model["model_params"]['activation'],
            solver=regression_model["model_params"]['solver'],
            early_stopping=regression_model["model_params"]['early_stopping'],
            learning_rate_init=regression_model["model_params"]['learning_rate_init'],
            random_state=random_seed, verbose=verbose_bool)
        return regr
    elif regression_model['model'] == 'KerasRNN':
        from autots.models.dnn import KerasRNN
        regr = KerasRNN(
            verbose=verbose, random_seed=random_seed,
            kernel_initializer=regression_model["model_params"]['kernel_initializer'],
            epochs=regression_model["model_params"]['epochs'],
            batch_size=regression_model["model_params"]['batch_size'],
            optimizer=regression_model["model_params"]['optimizer'],
            loss=regression_model["model_params"]['loss'],
            hidden_layer_sizes=regression_model["model_params"]['hidden_layer_sizes'],
            rnn_type=regression_model["model_params"]['rnn_type']
                        )
        return regr
    elif regression_model['model'] == 'KNN':
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.neighbors import KNeighborsRegressor
        regr = MultiOutputRegressor(KNeighborsRegressor(
            n_neighbors=regression_model["model_params"]['n_neighbors'],
            weights=regression_model["model_params"]['weights']
            ))
        return regr
    elif regression_model['model'] == 'Adaboost':
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import AdaBoostRegressor
        if regression_model["model_params"]['base_estimator'] == 'SVR':
            from sklearn.svm import LinearSVR
            svc = LinearSVR(verbose=verbose, random_state=random_seed,
                            max_iter=1500)
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
                                              'xgboost', 'KerasRNN'],
                              model_probs: list = [0.05, 0.05,
                                                  0.12, 0.2, 0.12,
                                                  0.2, 0.025, 0.035,
                                                  0.1, 0.1]):
    """Generate new parameters for input to regressor."""
    model = np.random.choice(a=models, size=1, p=model_probs).item()
    # model = 'KerasRNN'
    if model in ['xgboost', 'Adaboost', 'DecisionTree',
                 'MLP', 'KNN', 'KerasRNN']:
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
        elif model == 'MLP':
            solver = np.random.choice(['lbfgs', 'sgd', 'adam'],
                                      p=[0.5, 0.1, 0.4], size=1).item()
            if solver in ['sgd', 'adam']:
                early_stopping = np.random.choice([True, False], size=1).item()
                learning_rate_init = np.random.choice([0.01, 0.001, 0.0001, 0.00001],
                                                      p=[0.1, 0.7, 0.1, 0.1],
                                                      size=1).item()
            else:
                early_stopping = False
                learning_rate_init = 0.001
            param_dict = {"model": 'MLP',
                    "model_params": {
                        "hidden_layer_sizes": np.random.choice(
                            [(100,), (25, 15, 25), (72, 36, 72), (25, 50, 25),
                             (32, 64, 32), (32, 32, 32)],
                            p=[0.1, 0.3, 0.3, 0.1, 0.1, 0.1],
                            size=1).item(),
                        "max_iter": np.random.choice([250, 500, 1000],
                                                     p=[0.89, 0.1, 0.01],
                                                     size=1).item(),
                        "activation": np.random.choice(['identity', 'logistic',
                                                        'tanh', 'relu'],
                                                       p=[0.05, 0.05,
                                                          0.6, 0.3],
                                                       size=1).item(),
                        "solver": solver,
                        "early_stopping": early_stopping,
                        "learning_rate_init": learning_rate_init
                        }}
        elif model == 'KNN':
            param_dict = {"model": 'KNN',
                    "model_params": {
                        "n_neighbors": np.random.choice([3, 5, 10],
                                                         p=[0.2, 0.7, 0.1],
                                                         size=1).item(),
                        "weights": np.random.choice(['uniform', 'distance'],
                                                         p=[0.7, 0.3],
                                                         size=1).item()
                        }}
        elif model == 'KerasRNN':
            init_list = ['glorot_uniform', 'lecun_uniform',
                         'glorot_normal', 'RandomUniform', 'he_normal']
            param_dict = {"model": 'KerasRNN',
                    "model_params": {
                        "kernel_initializer": np.random.choice(init_list,
                                                               size=1).item(),
                        "epochs": np.random.choice([50, 100, 500],
                                                   p=[0.7, 0.2, 0.1],
                                                   size=1).item(),
                        "batch_size": np.random.choice([8, 16, 32, 72],
                                                       p=[0.2, 0.2, 0.5, 0.1],
                                                       size=1).item(),
                        "optimizer": np.random.choice(['adam', 'rmsprop',
                                                       'adagrad'],
                                                      p=[0.4, 0.5, 0.1],
                                                      size=1).item(),
                        "loss": np.random.choice(['mae', 'Huber',
                                                  'poisson', 'mse', 'mape'],
                                                 p=[0.2, 0.3,
                                                    0.1, 0.2, 0.2],
                                                 size=1).item(),
                        "hidden_layer_sizes": np.random.choice(
                            [(100,), (32,), (72, 36, 72), (25, 50, 25),
                             (32, 64, 32), (32, 32, 32)],
                            p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
                            size=1).item(),
                        "rnn_type": np.random.choice(['LSTM', 'GRU'],
                                                     p=[0.7, 0.3],
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
        if mean_rolling_periods is None:
            self.macd_periods = None
        else:
            self.macd_periods = macd_periods
        self.std_rolling_periods = std_rolling_periods
        self.max_rolling_periods = max_rolling_periods
        self.min_rolling_periods = min_rolling_periods
        self.ewm_alpha = ewm_alpha
        self.additional_lag_periods = additional_lag_periods
        self.abs_energy = abs_energy
        self.rolling_autocorr_periods = rolling_autocorr_periods
        self.add_date_part = add_date_part
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
                                   complete_regressor.head(x_dat.shape[0])],
                                  axis=1).fillna(0)
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
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train, forecast, method='inferred_normal',
                prediction_interval=self.prediction_interval)

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name=self.name,
                                          forecast_length=forecast_length,
                                          forecast_index=forecast.index,
                                          forecast_columns=forecast.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=forecast,
                                          upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime=self.fit_runtime,
                                          model_parameters=self.get_params())
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        model_choice = generate_regressor_params()
        mean_rolling_periods_choice = np.random.choice(
            a=[None, 5, 7, 12, 30], size=1,
            p=[0.2, 0.2, 0.2, 0.2, 0.2]).item()
        if mean_rolling_periods_choice is not None:
            macd_periods_choice = seasonal_int()
            if macd_periods_choice == mean_rolling_periods_choice:
                macd_periods_choice = mean_rolling_periods_choice + 10
        else:
            macd_periods_choice = None
        std_rolling_periods_choice = np.random.choice(
            a=[None, 5, 7, 10, 30], size=1,
            p=[0.2, 0.2, 0.2, 0.2, 0.2]).item()
        max_rolling_periods_choice = np.random.choice(a=[None, seasonal_int()],
                                                      size=1,
                                                      p=[0.2, 0.8]).item()
        min_rolling_periods_choice = np.random.choice(a=[None, seasonal_int()],
                                                      size=1,
                                                      p=[0.2, 0.8]).item()
        lag_periods_choice = seasonal_int() - 1
        lag_periods_choice = 2 if lag_periods_choice < 2 else lag_periods_choice
        ewm_choice = np.random.choice(a=[None, 0.2, 0.5, 0.8], size=1,
                                      p=[0.5, 0.15, 0.15, 0.2]).item()
        abs_energy_choice = np.random.choice(a=[True, False], size=1,
                                             p=[0.3, 0.7]).item()
        rolling_autocorr_periods_choice = np.random.choice(
            a=[None, 2, 7, 12, 30], size=1,
            p=[0.8, 0.05, 0.05, 0.05, 0.05]).item()
        add_date_part_choice = np.random.choice(a=[None, 'simple', 'expanded'],
                                                size=1,
                                                p=[0.6, 0.2, 0.2]).item()
        holiday_choice = np.random.choice(a=[True, False], size=1,
                                          p=[0.2, 0.8]).item()
        polynomial_degree_choice = np.random.choice(a=[None, 2], size=1,
                                                    p=[0.95, 0.05]).item()
        x_transform_choice = np.random.choice(
            a=[None, 'FastICA', 'Nystroem', 'RmZeroVariance'], size=1,
            p=[0.7, 0.05, 0.05, 0.2]).item()
        regression_choice = np.random.choice(a=[None, 'User'], size=1,
                                             p=[0.7, 0.3]).item()
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


def window_maker(df, window_size: int = 10,
                 input_dim: str = 'multivariate',
                 normalize_window: bool = False,
                 shuffle: bool = True,
                 output_dim: str = 'forecast_length',
                 forecast_length: int = 1,
                 max_windows: int = 5000
                 ):
    """Convert a dataset into slices with history and y forecast."""
    if output_dim == '1step':
        forecast_length = 1
    phrase_n = (forecast_length + window_size)
    max_pos_wind = df.shape[0] - phrase_n + 1
    max_pos_wind = max_windows if max_pos_wind > max_windows else max_pos_wind
    if max_pos_wind == max_windows:
        numbers = np.random.choice((df.shape[0] - phrase_n),
                                   size=max_pos_wind,
                                   replace=False)
        if not shuffle:
            numbers = np.sort(numbers)
    else:
        numbers = np.array(range(max_pos_wind))
        if shuffle:
            np.random.shuffle(numbers)

    X = pd.DataFrame()
    Y = pd.DataFrame()
    for z in numbers:
        if input_dim == 'univariate':
            rand_slice = df.iloc[z:(z + phrase_n), ]
            rand_slice = rand_slice.reset_index(drop=True).transpose().set_index(np.repeat(z, (df.shape[1], )), append=True)
            cX = rand_slice.iloc[:, 0:(window_size)]
            cY = rand_slice.iloc[:, window_size:]
        else:
            cX = df.iloc[z:(z + window_size), ]
            cX = pd.DataFrame(cX.stack().reset_index(drop=True)).transpose()
            cY = df.iloc[(z + window_size):(z + phrase_n), ]
            cY = pd.DataFrame(cY.stack().reset_index(drop=True)).transpose()
        X = pd.concat([X, cX], axis=0)
        Y = pd.concat([Y, cY], axis=0)
    if normalize_window:
        X  = X.div(X.sum(axis=1), axis=0)
    return X, Y


def last_window(df, window_size: int = 10,
                input_dim: str = 'multivariate',
                normalize_window: bool = False):
    z = df.shape[0] - window_size
    if input_dim == 'univariate':
        cX = df.iloc[z:(z + window_size), ]
        cX = cX.reset_index(drop=True).transpose().set_index(np.repeat(z, (df.shape[1], )), append=True)
    else:
        cX = df.iloc[z:(z + window_size), ]
        cX = pd.DataFrame(cX.stack().reset_index(drop=True)).transpose()
    if normalize_window:
        cX  = cX.div(cX.sum(axis=1), axis=0)
    return cX


class WindowRegression(ModelObject):
    """Regression use the last n values as the basis of training data.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        # transfer_learning: str = None,
        # transfer_learning_transformation: dict = None,
        # regression_type: str = None,
    """

    def __init__(self, name: str = "WindowRegression",
                 frequency: str = 'infer',
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0,
                 window_size: int = 10,
                 regression_model: dict =
                 {"model": 'Adaboost',
                  "model_params":
                      {'n_estimators': 50,
                       'base_estimator': 'DecisionTree',
                       'loss': 'linear',
                       'learning_rate': 1.0}},
                 input_dim: str = 'multivariate',
                 output_dim: str = '1step',
                 normalize_window: bool = False,
                 shuffle: bool = True,
                 forecast_length: int = 1,
                 max_windows: int = 5000
                 ):
        ModelObject.__init__(self, name, frequency, prediction_interval,
                             holiday_country=holiday_country,
                             random_seed=random_seed, verbose=verbose)
        self.window_size = abs(int(window_size))
        self.regression_model = regression_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_window = normalize_window
        self.shuffle = shuffle
        self.forecast_length = forecast_length
        self.max_windows = abs(int(max_windows))

    def fit(self, df, preord_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df
        X, Y = window_maker(df, window_size=self.window_size,
                            input_dim=self.input_dim,
                            normalize_window=self.normalize_window,
                            shuffle=self.shuffle,
                            output_dim=self.output_dim,
                            forecast_length=self.forecast_length,
                            max_windows=self.max_windows)
        self.regr = retrieve_regressor(regression_model=self.regression_model,
                                       verbose=self.verbose,
                                       verbose_bool=self.verbose_bool,
                                       random_seed=self.random_seed)
        self.regr = self.regr.fit(X, Y)
        self.last_window = df.tail(self.window_size)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int,
                preord_regressor = [], just_point_forecast: bool = False):
        """Generate forecast data immediately following dates of .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        """
        A VALUE IS BEING DROPPED FROM Y!
        for forecast:
        output_dim = 1
        don't forget to normalize if used
        collapse an output_dim into a forecastdf
        
        if univariate and 1, transpose
        """
        if int(forecast_length) > int(self.forecast_length):
            print("GluonTS must be refit to change forecast length!")
        predictStartTime = datetime.datetime.now()
        index = self.create_forecast_index(forecast_length=forecast_length)

        if self.output_dim == '1step':
            # combined_index = (self.df_train.index.append(index))
            forecast = pd.DataFrame()
            # forecast, 1 step ahead, then another, and so on
            for x in range(forecast_length):
                pred = last_window(self.last_window,
                                   window_size=self.window_size,
                                   input_dim=self.input_dim,
                                   normalize_window=self.normalize_window)
                rfPred = pd.DataFrame(self.regr.predict(pred))
                if self.input_dim == 'univariate':
                    rfPred = pd.DataFrame(rfPred).transpose()
                forecast = pd.concat([forecast, rfPred], axis=0,
                                     ignore_index=True)
                self.last_window = pd.concat([self.last_window, rfPred],
                                             axis=0, ignore_index=True)
                # self.sktraindata.index = combined_index[:len(self.sktraindata.index)]
            df = forecast

        else:
            pred = last_window(self.last_window, window_size=self.window_size,
                               input_dim=self.input_dim,
                               normalize_window=self.normalize_window)
            cY = pd.DataFrame(self.regr.predict(pred))
            if self.input_dim == 'multivariate':
                # cY = Y.tail(1)
                cY.index = ['values']
                cY.columns = np.tile(self.column_names,
                                     reps=self.forecast_length)
                cY = cY.transpose().reset_index()
                cY['timestep'] = np.repeat(range(forecast_length),
                                           repeats=len(self.column_names))
                cY = pd.pivot_table(cY, index='timestep', columns='index')
            else:
                # cY = Y.tail(df.shape[1])
                cY = cY.transpose()
            df = cY

        df.columns = self.column_names
        df.index = index
        if just_point_forecast:
            return df
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train, df,
                prediction_interval=self.prediction_interval,
                method='historic_quantile')

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name=self.name,
                                          forecast_length=forecast_length,
                                          forecast_index=df.index,
                                          forecast_columns=df.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=df,
                                          upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime=self.fit_runtime,
                                          model_parameters=self.get_params())
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        window_size_choice = np.random.choice([5, 10, 20, seasonal_int()],
                                              size=1).item()
        model_choice = generate_regressor_params()
        input_dim_choice = np.random.choice(['multivariate', 'univariate'],
                                            p=[0.3, 0.7],
                                            size=1).item()
        output_dim_choice = np.random.choice(['forecast_length', '1step'],
                                             size=1).item()
        normalize_window_choice = np.random.choice(a=[True, False],
                                                   size=1,
                                                   p=[0.05, 0.95]).item()
        shuffle_choice = np.random.choice(a=[True, False], size=1).item()
        max_windows_choice = np.random.choice(a=[5000, 1000, 50000],
                                              size=1,
                                              p=[0.95, 0.04, 0.01]).item()
        return {
                'window_size': window_size_choice,
                'regression_model': model_choice,
                'input_dim': input_dim_choice,
                'output_dim': output_dim_choice,
                'normalize_window': normalize_window_choice,
                'shuffle': shuffle_choice,
                'max_windows': max_windows_choice
                }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'window_size': self.window_size,
            'regression_model': self.regression_model,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'normalize_window': self.normalize_window,
            'shuffle': self.shuffle,
            'max_windows': self.max_windows
                }

"""
window_size: int = 10,
input_dim: str = 'multivariate',
output_dim: str = forecast_len, '1step'
normalize_window: bool = False, -rowwise, that is
regression_type: str = None

max number of windows to make...
forecast_length is passed into init
shuffle or not

df = df_wide_numeric.fillna(0).astype(float)
window_size = 10
input_dim = 'univariate'
input_dim = 'multivariate'
output_dim = 'forecast_length'
max_windows = 5000

for forecast:
    just last window
    output_dim = 1
    don't forget to normalize if used
    collapse an output_dim into a forecast df
"""



