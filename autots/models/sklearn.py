"""
Sklearn dependent models

Decision Tree, Elastic Net,  Random Forest, MLPRegressor, KNN, Adaboost
"""
import datetime
import random
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.tools.seasonal import date_part, seasonal_int


def rolling_x_regressor(
    df,
    mean_rolling_periods: int = 30,
    macd_periods: int = None,
    std_rolling_periods: int = 7,
    max_rolling_periods: int = None,
    min_rolling_periods: int = None,
    ewm_alpha: float = 0.5,
    additional_lag_periods: int = 7,
    abs_energy: bool = False,
    rolling_autocorr_periods: int = None,
    add_date_part: str = None,
    holiday: bool = False,
    holiday_country: str = 'US',
    polynomial_degree: int = None,
    window: int = None,
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
        X = pd.concat([X, df.rolling(std_rolling_periods, min_periods=1).std()], axis=1)
    if str(max_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(max_rolling_periods, min_periods=1).max()], axis=1)
    if str(min_rolling_periods).isdigit():
        X = pd.concat([X, df.rolling(min_rolling_periods, min_periods=1).min()], axis=1)
    if str(ewm_alpha).replace('.', '').isdigit():
        X = pd.concat(
            [X, df.ewm(alpha=ewm_alpha, ignore_na=True, min_periods=1).mean()], axis=1
        )
    if str(additional_lag_periods).isdigit():
        X = pd.concat([X, df.shift(additional_lag_periods)], axis=1).fillna(
            method='bfill'
        )
    if abs_energy:
        X = pd.concat([X, df.pow(other=([2] * len(df.columns))).cumsum()], axis=1)
    if str(rolling_autocorr_periods).isdigit():
        temp = df.rolling(rolling_autocorr_periods).apply(
            lambda x: x.autocorr(), raw=False
        )
        X = pd.concat([X, temp], axis=1).fillna(method='bfill')

    if add_date_part in ['simple', 'expanded', 'recurring']:
        date_part_df = date_part(df.index, method=add_date_part)
        date_part_df.index = df.index
        X = pd.concat(
            [
                X,
            ],
            axis=1,
        )
    if holiday:
        from autots.tools.holiday import holiday_flag

        X['holiday_flag_'] = holiday_flag(X.index, country=holiday_country)
        X['holiday_flag_future_'] = holiday_flag(
            X.index.shift(1, freq=pd.infer_freq(X.index)), country=holiday_country
        )
    if str(polynomial_degree).isdigit():
        polynomial_degree = abs(int(polynomial_degree))
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(polynomial_degree)
        X = pd.DataFrame(poly.fit_transform(X))
    # unlike the others, this pulls the entire window, not just one lag
    if str(window).isdigit():
        # we already have lag 1 using this
        for curr_shift in range(1, window):
            X = pd.concat([X, df.shift(curr_shift)], axis=1).fillna(method='bfill')

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(method='ffill').fillna(method='bfill')

    X.columns = [str(x) for x in range(len(X.columns))]

    return X


def retrieve_regressor(
    regression_model: dict = {
        "model": 'Adaboost',
        "model_params": {
            'n_estimators': 50,
            'base_estimator': 'DecisionTree',
            'loss': 'linear',
            'learning_rate': 1.0,
        },
    },
    verbose: int = 0,
    verbose_bool: bool = False,
    random_seed: int = 2020,
    n_jobs: int = 1,
    multioutput: bool = True,
):
    """Convert a model param dict to model object for regression frameworks."""
    model_class = regression_model['model']
    model_param_dict = regression_model.get("model_params", {})
    if model_class == 'ElasticNet':
        if multioutput:
            from sklearn.linear_model import MultiTaskElasticNet

            regr = MultiTaskElasticNet(
                alpha=1.0, random_state=random_seed, **model_param_dict
            )
        else:
            from sklearn.linear_model import ElasticNet

            regr = ElasticNet(alpha=1.0, random_state=random_seed, **model_param_dict)
        return regr
    elif model_class == 'DecisionTree':
        from sklearn.tree import DecisionTreeRegressor

        regr = DecisionTreeRegressor(random_state=random_seed, **model_param_dict)
        return regr
    elif model_class == 'MLP':
        from sklearn.neural_network import MLPRegressor

        regr = MLPRegressor(
            random_state=random_seed, verbose=verbose_bool, **model_param_dict
        )
        return regr
    elif model_class == 'KerasRNN':
        from autots.models.dnn import KerasRNN

        regr = KerasRNN(
            verbose=verbose,
            random_seed=random_seed,
            **model_param_dict
        )
        return regr
    elif model_class == 'Transformer':
        from autots.models.dnn import Transformer

        regr = Transformer(
            verbose=verbose,
            random_seed=random_seed,
            **model_param_dict
        )
        return regr
    elif model_class == 'KNN':
        from sklearn.neighbors import KNeighborsRegressor

        if multioutput:
            from sklearn.multioutput import MultiOutputRegressor

            regr = MultiOutputRegressor(
                KNeighborsRegressor(**model_param_dict),
                n_jobs=n_jobs,
            )
        else:
            regr = KNeighborsRegressor(**model_param_dict, n_jobs=n_jobs)
        return regr
    elif model_class == 'HistGradientBoost':
        try:
            from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        except Exception:
            pass
        from sklearn.ensemble import HistGradientBoostingRegressor

        if multioutput:
            from sklearn.multioutput import MultiOutputRegressor

            regr = MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_iter=200,
                    verbose=int(verbose_bool),
                    random_state=random_seed,
                    **model_param_dict,
                )
            )
        else:
            regr = HistGradientBoostingRegressor(
                max_iter=200,
                verbose=int(verbose_bool),
                random_state=random_seed,
                **model_param_dict,
            )
        return regr
    elif model_class == 'LightGBM':
        from lightgbm import LGBMRegressor

        regr = LGBMRegressor(
            verbose=int(verbose_bool),
            random_state=random_seed,
            n_jobs=n_jobs,
            **model_param_dict,
        )
        if multioutput:
            from sklearn.multioutput import RegressorChain

            return RegressorChain(regr)
        else:
            return regr
    elif model_class == 'Adaboost':
        from sklearn.ensemble import AdaBoostRegressor

        if regression_model["model_params"]['base_estimator'] == 'SVR':
            from sklearn.svm import LinearSVR

            svc = LinearSVR(verbose=verbose, random_state=random_seed, max_iter=1500)
            regr = AdaBoostRegressor(
                base_estimator=svc,
                n_estimators=regression_model["model_params"]['n_estimators'],
                loss=regression_model["model_params"]['loss'],
                learning_rate=regression_model["model_params"]['learning_rate'],
                random_state=random_seed,
            )
        elif regression_model["model_params"]['base_estimator'] == 'LinReg':
            from sklearn.linear_model import LinearRegression

            linreg = LinearRegression()
            regr = AdaBoostRegressor(
                base_estimator=linreg,
                n_estimators=regression_model["model_params"]['n_estimators'],
                loss=regression_model["model_params"]['loss'],
                learning_rate=regression_model["model_params"]['learning_rate'],
                random_state=random_seed,
            )
        else:
            regr = AdaBoostRegressor(random_state=random_seed, **model_param_dict)
        if multioutput:
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(regr, n_jobs=n_jobs)
        else:
            return regr
    elif model_class == 'xgboost':
        import xgboost as xgb

        if multioutput:
            from sklearn.multioutput import MultiOutputRegressor

            regr = MultiOutputRegressor(
                xgb.XGBRegressor(verbosity=verbose, **model_param_dict),
                n_jobs=n_jobs,
            )
        else:
            regr = xgb.XGBRegressor(
                verbosity=verbose, **model_param_dict, n_jobs=n_jobs
            )
        return regr
    elif model_class == 'SVM':
        from sklearn.svm import LinearSVR

        if multioutput:
            from sklearn.multioutput import MultiOutputRegressor

            regr = MultiOutputRegressor(
                LinearSVR(verbose=verbose_bool, **model_param_dict),
                n_jobs=n_jobs,
            )
        else:
            regr = LinearSVR(verbose=verbose_bool, **model_param_dict)
        return regr
    elif model_class == 'BayesianRidge':
        from sklearn.linear_model import BayesianRidge

        regr = BayesianRidge(**model_param_dict)
        if multioutput:
            from sklearn.multioutput import RegressorChain

            return RegressorChain(regr)
        else:
            return regr
    elif model_class == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_jobs=n_jobs, random_state=random_seed, **model_param_dict
        )
    elif model_class == "RadiusNeighbors":
        from sklearn.neighbors import RadiusNeighborsRegressor

        regr = RadiusNeighborsRegressor(n_jobs=n_jobs, **model_param_dict)
        return regr
    else:
        regression_model['model'] = 'RandomForest'
        from sklearn.ensemble import RandomForestRegressor

        regr = RandomForestRegressor(
            random_state=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **model_param_dict,
        )
        return regr


sklearn_model_dict: dict = {
    'RandomForest': 0.05,
    'ElasticNet': 0.05,
    'MLP': 0.05,
    'DecisionTree': 0.05,
    'KNN': 0.05,
    'Adaboost': 0.05,
    'SVM': 0.05,  # was slow, LinearSVR seems much faster
    'BayesianRidge': 0.05,
    'xgboost': 0.05,
    'KerasRNN': 0.05,
    'Transformer': 0.05,
    'HistGradientBoost': 0.05,
    'LightGBM': 0.05,
    'ExtraTrees': 0.05,
    'RadiusNeighbors': 0.05,
}
univariate_model_dict = sklearn_model_dict.copy()
del univariate_model_dict['KerasRNN']
del univariate_model_dict['Transformer']
# models where we can be sure the model isn't sharing information across multiple Y's...
no_shared_model_dict = {
    'KNN': 0.1,
    'Adaboost': 0.1,
    'SVM': 0.1,
    'xgboost': 0.1,
    'HistGradientBoost': 0.1,
}
datepart_model_dict: dict = {
    'RandomForest': 0.05,
    'ElasticNet': 0.05,
    'MLP': 0.05,
    'DecisionTree': 0.05,
    'Adaboost': 0.05,
    'SVM': 0.05,
    'KerasRNN': 0.05,
    'Transformer': 0.05,
    'ExtraTrees': 0.05,
    'RadiusNeighbors': 0.05,
}


def generate_regressor_params(
    model_dict=None,
):
    if model_dict is None:
        model_dict = sklearn_model_dict
    """Generate new parameters for input to regressor."""
    model = random.choices(list(model_dict.keys()), list(model_dict.values()), k=1)[0]
    if model in [
        'xgboost',
        'Adaboost',
        'DecisionTree',
        'LightGBM',
        'MLP',
        'KNN',
        'KerasRNN',
        'Transformer',
        'HistGradientBoost',
        'RandomForest',
        'ExtraTrees',
    ]:
        if model == 'Adaboost':
            param_dict = {
                "model": 'Adaboost',
                "model_params": {
                    "n_estimators": np.random.choice(
                        [50, 100, 500], p=[0.7, 0.2, 0.1], size=1
                    ).item(),
                    "loss": np.random.choice(
                        ['linear', 'square', 'exponential'], p=[0.8, 0.1, 0.1], size=1
                    ).item(),
                    "base_estimator": np.random.choice(
                        [None, 'LinReg', 'SVR'], p=[0.8, 0.1, 0.1], size=1
                    ).item(),
                    "learning_rate": np.random.choice(
                        [1, 0.5], p=[0.9, 0.1], size=1
                    ).item(),
                },
            }
        elif model == 'xgboost':
            param_dict = {
                "model": 'xgboost',
                "model_params": {
                    "objective": np.random.choice(
                        ['count:poisson', 'reg:squarederror', 'reg:gamma'],
                        p=[0.4, 0.5, 0.1],
                        size=1,
                    ).item(),
                    "eta": np.random.choice([0.3], p=[1.0], size=1).item(),
                    "min_child_weight": np.random.choice(
                        [1, 2, 5], p=[0.8, 0.1, 0.1], size=1
                    ).item(),
                    "max_depth": np.random.choice(
                        [3, 6, 9], p=[0.1, 0.8, 0.1], size=1
                    ).item(),
                    "subsample": np.random.choice(
                        [1, 0.7, 0.5], p=[0.9, 0.05, 0.05], size=1
                    ).item(),
                },
            }
        elif model == 'MLP':
            solver = np.random.choice(
                ['lbfgs', 'sgd', 'adam'], p=[0.5, 0.1, 0.4], size=1
            ).item()
            if solver in ['sgd', 'adam']:
                early_stopping = np.random.choice([True, False], size=1).item()
                learning_rate_init = np.random.choice(
                    [0.01, 0.001, 0.0001, 0.00001], p=[0.1, 0.7, 0.1, 0.1], size=1
                ).item()
            else:
                early_stopping = False
                learning_rate_init = 0.001
            param_dict = {
                "model": 'MLP',
                "model_params": {
                    "hidden_layer_sizes": random.choices(
                        [
                            (100,),
                            (25, 15, 25),
                            (72, 36, 72),
                            (25, 50, 25),
                            (32, 64, 32),
                            (32, 32, 32),
                        ],
                        [0.1, 0.3, 0.3, 0.1, 0.1, 0.1],
                    )[0],
                    "max_iter": np.random.choice(
                        [250, 500, 1000], p=[0.8, 0.1, 0.1], size=1
                    ).item(),
                    "activation": np.random.choice(
                        ['identity', 'logistic', 'tanh', 'relu'],
                        p=[0.05, 0.05, 0.6, 0.3],
                        size=1,
                    ).item(),
                    "solver": solver,
                    "early_stopping": early_stopping,
                    "learning_rate_init": learning_rate_init,
                },
            }
        elif model == 'KNN':
            param_dict = {
                "model": 'KNN',
                "model_params": {
                    "n_neighbors": np.random.choice(
                        [3, 5, 10], p=[0.2, 0.7, 0.1], size=1
                    ).item(),
                    "weights": np.random.choice(
                        ['uniform', 'distance'], p=[0.7, 0.3], size=1
                    ).item(),
                },
            }
        elif model == 'RandomForest':
            param_dict = {
                "model": 'RandomForest',
                "model_params": {
                    "n_estimators": random.choices(
                        [300, 100, 1000, 5000], [0.2, 0.2, 0.8, 0.05]
                    )[0],
                    "min_samples_leaf": random.choices(
                        [2, 4, 1], [0.2, 0.2, 0.8]
                    )[0],
                    "bootstrap": random.choices(
                        [True, False], [0.9, 0.1]
                    )[0],
                    "criterion": random.choices(
                        ["squared_error", "poisson", "absolute_error"], [0.8, 0.2, 0.2]
                    )[0],
                },
            }
        elif model == 'ExtraTrees':
            param_dict = {
                "model": 'ExtraTrees',
                "model_params": {
                    "n_estimators": random.choices(
                        [50, 100, 500], [0.1, 0.8, 0.1]
                    )[0],
                    "min_samples_leaf": random.choices(
                        [2, 4, 1], [0.1, 0.1, 0.8]
                    )[0],
                    "max_depth": random.choices(
                        [None, 5, 10], [0.9, 0.1, 0.1]
                    )[0],
                    "criterion": random.choices(
                        ["squared_error", "absolute_error"], [0.8, 0.2]
                    )[0],
                },
            }
        elif model == 'KerasRNN':
            init_list = [
                'glorot_uniform',
                'lecun_uniform',
                'glorot_normal',
                'RandomUniform',
                'he_normal',
                'zeros',
            ]
            param_dict = {
                "model": 'KerasRNN',
                "model_params": {
                    "kernel_initializer": random.choices(init_list)[0],
                    "epochs": random.choices(
                        [50, 100, 500], [0.7, 0.2, 0.1]
                    )[0],
                    "batch_size": random.choices(
                        [8, 16, 32, 72], [0.2, 0.2, 0.5, 0.1]
                    )[0],
                    "optimizer": random.choices(
                        ['adam', 'rmsprop', 'adagrad'], [0.4, 0.5, 0.1]
                    )[0],
                    "loss": random.choices(
                        ['mae', 'Huber', 'poisson', 'mse', 'mape'],
                        [0.2, 0.3, 0.1, 0.2, 0.2],
                    )[0],
                    "hidden_layer_sizes": random.choices(
                        [
                            (100,),
                            (32,),
                            (72, 36, 72),
                            (25, 50, 25),
                            (32, 64, 32),
                            (32, 32, 32),
                        ],
                        [0.1, 0.3, 0.3, 0.1, 0.1, 0.1],
                    )[0],
                    "rnn_type": random.choices(
                        ['LSTM', 'GRU', "E2D2", "CNN"], [0.5, 0.3, 0.15, 0.01]
                    )[0],
                    "shape": random.choice([1, 2]),
                },
            }
        elif model == 'Transformer':
            param_dict = {
                "model": 'Transformer',
                "model_params": {
                    "epochs": random.choices(
                        [50, 100, 500, 750], [0.7, 0.2, 0.1, 0.05]
                    )[0],
                    "batch_size": random.choices(
                        [8, 16, 32, 72], [0.2, 0.2, 0.5, 0.1]
                    )[0],
                    "optimizer": random.choices(
                        ['adam', 'rmsprop', 'adagrad'], [0.4, 0.5, 0.1]
                    )[0],
                    "loss": random.choices(
                        ['mae', 'Huber', 'poisson', 'mse', 'mape'],
                        [0.2, 0.3, 0.1, 0.2, 0.2],
                    )[0],
                    "head_size": random.choices(
                        [32, 64, 128, 256, 384], [0.1, 0.1, 0.3, 0.5, 0.05]
                    )[0],
                    "num_heads": random.choices(
                        [2, 4], [0.2, 0.2]
                    )[0],
                    "ff_dim": random.choices(
                        [2, 3, 4, 32, 64], [0.1, 0.1, 0.8, 0.05, 0.05]
                    )[0],
                    "num_transformer_blocks": random.choices(
                        [1, 2, 4, 6],
                        [0.2, 0.2, 0.6, 0.05],
                    )[0],
                    "mlp_units": random.choices(
                        [32, 64, 128, 256],
                        [0.2, 0.3, 0.8, 0.2],
                    ),
                    "mlp_dropout": random.choices(
                        [0.05, 0.2, 0.4],
                        [0.2, 0.8, 0.2],
                    )[0],
                    "dropout": random.choices(
                        [0.05, 0.2, 0.4],
                        [0.2, 0.8, 0.2],
                    )[0],
                },
            }
        elif model == 'HistGradientBoost':
            param_dict = {
                "model": 'HistGradientBoost',
                "model_params": {
                    "loss": np.random.choice(
                        a=['least_squares', 'poisson', 'least_absolute_deviation'],
                        p=[0.4, 0.3, 0.3],
                        size=1,
                    ).item(),
                    "learning_rate": np.random.choice(
                        a=[1, 0.1, 0.01], p=[0.3, 0.4, 0.3], size=1
                    ).item(),
                },
            }
        elif model == 'LightGBM':
            param_dict = {
                "model": 'LightGBM',
                "model_params": {
                    "objective": np.random.choice(
                        a=['regression', 'gamma', 'huber', 'regression_l1'],
                        p=[0.4, 0.3, 0.1, 0.2],
                        size=1,
                    ).item(),
                    "learning_rate": np.random.choice(
                        a=[0.001, 0.1, 0.01], p=[0.1, 0.6, 0.3], size=1
                    ).item(),
                    "num_leaves": np.random.choice(
                        a=[31, 127, 70], p=[0.6, 0.1, 0.3], size=1
                    ).item(),
                    "max_depth": np.random.choice(
                        a=[-1, 5, 10], p=[0.6, 0.1, 0.3], size=1
                    ).item(),
                    "boosting_type": np.random.choice(
                        a=['gbdt', 'rf', 'dart', 'goss'], p=[0.6, 0, 0.2, 0.2], size=1
                    ).item(),
                    "n_estimators": np.random.choice(
                        a=[100, 250, 50, 500], p=[0.6, 0.099, 0.3, 0.0010], size=1
                    ).item(),
                },
            }
        else:
            min_samples = np.random.choice(
                [1, 2, 0.05], p=[0.5, 0.3, 0.2], size=1
            ).item()
            min_samples = int(min_samples) if min_samples in [2] else min_samples
            param_dict = {
                "model": 'DecisionTree',
                "model_params": {
                    "max_depth": np.random.choice(
                        [None, 3, 9], p=[0.5, 0.3, 0.2], size=1
                    ).item(),
                    "min_samples_split": min_samples,
                },
            }
    else:
        param_dict = {"model": model, "model_params": {}}
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

    def __init__(
        self,
        name: str = "RollingRegression",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        verbose: int = 0,
        random_seed: int = 2020,
        regression_model: dict = {
            "model": 'Adaboost',
            "model_params": {
                'n_estimators': 50,
                'base_estimator': 'DecisionTree',
                'loss': 'linear',
                'learning_rate': 1.0,
            },
        },
        holiday: bool = False,
        mean_rolling_periods: int = 30,
        macd_periods: int = None,
        std_rolling_periods: int = 7,
        max_rolling_periods: int = 7,
        min_rolling_periods: int = 7,
        ewm_alpha: float = 0.5,
        additional_lag_periods: int = 7,
        abs_energy: bool = False,
        rolling_autocorr_periods: int = None,
        add_date_part: str = None,
        polynomial_degree: int = None,
        x_transform: str = None,
        window: int = None,
        n_jobs: int = -1,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            regression_type=regression_type,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
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
        self.window = window

    def _x_transformer(self):
        if self.x_transform == 'FastICA':
            from sklearn.decomposition import FastICA

            x_transformer = FastICA(n_components=None, random_state=2020, whiten=True)
        elif self.x_transform == 'Nystroem':
            from sklearn.kernel_approximation import Nystroem

            half_size = int(self.sktraindata.shape[0] / 2) + 1
            max_comp = 200
            n_comp = max_comp if half_size > max_comp else half_size
            x_transformer = Nystroem(
                kernel='rbf', gamma=0.2, random_state=2020, n_components=n_comp
            )
        else:
            # self.x_transform = 'RmZeroVariance'
            from sklearn.feature_selection import VarianceThreshold

            x_transformer = VarianceThreshold(threshold=0.0)
        return x_transformer

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            future_regressor (pandas.DataFrame or Series): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        # if external regressor, do some check up
        if self.regression_type is not None:
            if (np.array(future_regressor).shape[0]) != (df.shape[0]):
                self.regression_type = None
            else:
                self.regressor_train = future_regressor

        # define X and Y
        self.sktraindata = self.df_train.dropna(how='all', axis=0)
        self.sktraindata = self.sktraindata.fillna(method='ffill').fillna(
            method='bfill'
        )
        Y = self.sktraindata.drop(self.sktraindata.head(2).index)
        Y.columns = [x for x in range(len(Y.columns))]
        X = rolling_x_regressor(
            self.sktraindata,
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
            polynomial_degree=self.polynomial_degree,
            window=self.window,
        )
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
        if isinstance(X, pd.DataFrame):
            X.columns = [str(xc) for xc in X.columns]

        multioutput = True
        if Y.ndim < 2:
            multioutput = False
        elif Y.shape[1] < 2:
            multioutput = False
        # retrieve model object to train
        self.regr = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=self.verbose,
            verbose_bool=self.verbose_bool,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            multioutput=multioutput,
        )
        self.regr = self.regr.fit(X, Y)

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=[],
        just_point_forecast: bool = False,
    ):
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
            complete_regressor = pd.concat(
                [self.regressor_train, future_regressor], axis=0
            )

        combined_index = self.df_train.index.append(index)
        forecast = pd.DataFrame()
        self.sktraindata.columns = [x for x in range(len(self.sktraindata.columns))]

        # forecast, 1 step ahead, then another, and so on
        for x in range(forecast_length):
            x_dat = rolling_x_regressor(
                self.sktraindata,
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
                polynomial_degree=self.polynomial_degree,
            )
            if self.regression_type == 'User':
                x_dat = pd.concat(
                    [x_dat, complete_regressor.head(x_dat.shape[0])], axis=1
                ).fillna(0)
            if self.x_transform in ['FastICA', 'Nystroem', 'RmZeroVariance']:
                x_dat = pd.DataFrame(self.x_transformer.transform(x_dat))
                x_dat = x_dat.replace([np.inf, -np.inf], 0).fillna(0)
            if isinstance(x_dat, pd.DataFrame):
                x_dat.columns = [str(xc) for xc in x_dat.columns]

            rfPred = pd.DataFrame(self.regr.predict(x_dat.tail(1).values))

            forecast = pd.concat([forecast, rfPred], axis=0, ignore_index=True)
            self.sktraindata = pd.concat(
                [self.sktraindata, rfPred], axis=0, ignore_index=True
            )
            self.sktraindata.index = combined_index[: len(self.sktraindata.index)]

        forecast.columns = self.column_names
        forecast.index = index

        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=forecast.index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        model_choice = generate_regressor_params()
        mean_rolling_periods_choice = random.choices(
            [None, 5, 7, 12, 30], [0.2, 0.2, 0.2, 0.2, 0.2]
        )[0]
        if mean_rolling_periods_choice is not None:
            macd_periods_choice = seasonal_int()
            if macd_periods_choice == mean_rolling_periods_choice:
                macd_periods_choice = mean_rolling_periods_choice + 10
        else:
            macd_periods_choice = None
        std_rolling_periods_choice = random.choices(
            [None, 5, 7, 10, 30], [0.6, 0.1, 0.1, 0.1, 0.1]
        )[0]
        max_rolling_periods_choice = random.choices([None, seasonal_int()], [0.5, 0.5])[
            0
        ]
        min_rolling_periods_choice = random.choices([None, seasonal_int()], [0.5, 0.5])[
            0
        ]
        lag_periods_choice = seasonal_int() - 1
        lag_periods_choice = 2 if lag_periods_choice < 2 else lag_periods_choice
        ewm_choice = random.choices([None, 0.2, 0.5, 0.8], [0.7, 0.1, 0.1, 0.1])[0]
        abs_energy_choice = random.choices([True, False], [0.3, 0.7])[0]
        rolling_autocorr_periods_choice = random.choices(
            [None, 2, 7, 12, 30], [0.8, 0.05, 0.05, 0.05, 0.05]
        )[0]
        add_date_part_choice = random.choices(
            [None, 'simple', 'expanded', 'recurring'], [0.7, 0.1, 0.1, 0.1]
        )[0]
        holiday_choice = random.choices([True, False], [0.2, 0.8])[0]
        polynomial_degree_choice = random.choices([None, 2], [0.99, 0.01])[0]
        x_transform_choice = random.choices(
            [None, 'FastICA', 'Nystroem', 'RmZeroVariance'],
            [0.85, 0.05, 0.05, 0.05],
        )[0]
        regression_choice = random.choices([None, 'User'], [0.7, 0.3])[0]
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
            'regression_type': regression_choice,
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


def window_maker(
    df,
    window_size: int = 10,
    input_dim: str = 'univariate',
    normalize_window: bool = False,
    shuffle: bool = False,
    output_dim: str = 'forecast_length',
    forecast_length: int = 1,
    max_windows: int = 5000,
    regression_type: str = None,
    future_regressor=None,
    random_seed: int = 1234,
):
    """Convert a dataset into slices with history and y forecast.

    Args:
        df (pd.DataFrame): `wide` format df with sorted index
        window_size (int): length of history to use for X window
        input_dim (str): univariate or multivariate. If multivariate, all series in single X row
        shuffle (bool): (deprecated)
        output_dim (str): 'forecast_length' or '1step' where 1 step is basically forecast_length=1
        forecast_length (int): number of periods ahead that will be forecast
        max_windows (int): a cap on total number of windows to generate. If exceeded, random of this int are selected.
        regression_type (str): None or "user" if to try to concat regressor to windows
        future_regressor (pd.DataFrame): values of regressor if used
        random_seed (int): a consistent random

    Returns:
        X, Y
    """
    if output_dim == '1step':
        forecast_length = 1
    phrase_n = forecast_length + window_size
    try:
        if input_dim == "multivariate":
            raise ValueError("input_dim=`multivariate` not supported this way.")
        x = np.lib.stride_tricks.sliding_window_view(df.to_numpy(), phrase_n, axis=0)
        x = x.reshape(-1, x.shape[-1])
        Y = x[:, window_size:]
        if Y.ndim > 1:
            if Y.shape[1] == 1:
                Y = Y.ravel()
        X = x[:, :window_size]
        r_arr = None
        if max_windows is not None:
            X_size = x.shape[0]
            if max_windows < X_size:
                r_arr = np.random.default_rng(random_seed).integers(
                    0, X_size, size=max_windows
                )
                Y = Y[r_arr]
                X = X[r_arr]
        if normalize_window:
            div_sum = np.nansum(X, axis=1).reshape(-1, 1)
            X = X / np.where(div_sum == 0, 1, div_sum)
        # regressors
        if str(regression_type).lower() == "user":
            shape_1 = df.shape[1] if df.ndim > 1 else 1
            if isinstance(future_regressor, pd.DataFrame):
                regr_arr = np.repeat(
                    future_regressor.reindex(df.index).to_numpy()[(phrase_n - 1) :],
                    shape_1,
                    axis=0,
                )
                if r_arr is not None:
                    regr_arr = regr_arr[r_arr]
                X = np.concatenate([X, regr_arr], axis=1)

    except Exception:
        if str(regression_type).lower() == "user":
            if input_dim == "multivariate":
                raise ValueError(
                    "input_dim=`multivariate` and regression_type=`user` cannot be combined."
                )
            else:
                raise ValueError(
                    "WindowRegression regression_type='user' requires numpy >= 1.20"
                )
        max_pos_wind = df.shape[0] - phrase_n + 1
        max_pos_wind = max_windows if max_pos_wind > max_windows else max_pos_wind
        if max_pos_wind == max_windows:
            numbers = np.random.default_rng(random_seed).choice(
                (df.shape[0] - phrase_n), size=max_pos_wind, replace=False
            )
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
                rand_slice = df.iloc[
                    z : (z + phrase_n),
                ]
                rand_slice = (
                    rand_slice.reset_index(drop=True)
                    .transpose()
                    .set_index(np.repeat(z, (df.shape[1],)), append=True)
                )
                cX = rand_slice.iloc[:, 0:(window_size)]
                cY = rand_slice.iloc[:, window_size:]
            else:
                cX = df.iloc[
                    z : (z + window_size),
                ]
                cX = pd.DataFrame(cX.stack().reset_index(drop=True)).transpose()
                cY = df.iloc[
                    (z + window_size) : (z + phrase_n),
                ]
                cY = pd.DataFrame(cY.stack().reset_index(drop=True)).transpose()
            X = pd.concat([X, cX], axis=0)
            Y = pd.concat([Y, cY], axis=0)
        if normalize_window:
            X = X.div(X.sum(axis=1), axis=0)

    return X, Y


def last_window(
    df,
    window_size: int = 10,
    input_dim: str = 'univariate',
    normalize_window: bool = False,
):
    z = df.shape[0] - window_size
    shape_1 = df.shape[1] if df.ndim > 1 else 1
    if input_dim == 'univariate':
        cX = df.iloc[
            z : (z + window_size),
        ]
        cX = (
            cX.reset_index(drop=True)
            .transpose()
            .set_index(np.repeat(z, (shape_1,)), append=True)
        )
    else:
        cX = df.iloc[
            z : (z + window_size),
        ]
        cX = pd.DataFrame(cX.stack().reset_index(drop=True)).transpose()
    if normalize_window:
        cX = cX.div(cX.sum(axis=1), axis=0)
    return cX


class WindowRegression(ModelObject):
    """Regression use the last n values as the basis of training data.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        # regression_type: str = None,
    """

    def __init__(
        self,
        name: str = "WindowRegression",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2022,
        verbose: int = 0,
        window_size: int = 10,
        regression_model: dict = {
            "model": 'Adaboost',
            "model_params": {
                'n_estimators': 50,
                'base_estimator': 'DecisionTree',
                'loss': 'linear',
                'learning_rate': 1.0,
            },
        },
        input_dim: str = 'univariate',
        output_dim: str = 'forecast_length',
        normalize_window: bool = False,
        shuffle: bool = False,
        forecast_length: int = 1,
        max_windows: int = 5000,
        regression_type: str = None,
        n_jobs: int = -1,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            regression_type=regression_type,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.window_size = abs(int(window_size))
        self.regression_model = regression_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_window = normalize_window
        self.shuffle = shuffle
        self.forecast_length = forecast_length
        self.max_windows = abs(int(max_windows))

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        if (
            df.shape[1] * self.forecast_length
        ) > 200 and self.input_dim == "multivariate":
            raise ValueError(
                "Scale exceeds recommendation for input_dim == `multivariate`"
            )
        df = self.basic_profile(df)
        self.df_train = df
        X, Y = window_maker(
            df,
            window_size=self.window_size,
            input_dim=self.input_dim,
            normalize_window=self.normalize_window,
            shuffle=self.shuffle,
            output_dim=self.output_dim,
            forecast_length=self.forecast_length,
            max_windows=self.max_windows,
            regression_type=self.regression_type,
            future_regressor=future_regressor,
            random_seed=self.random_seed,
        )
        multioutput = True
        if Y.ndim < 2:
            multioutput = False
        elif Y.shape[1] < 2:
            multioutput = False
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.regr = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=self.verbose,
            verbose_bool=self.verbose_bool,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            multioutput=multioutput,
        )
        self.regr = self.regr.fit(X, Y)
        self.last_window = df.tail(self.window_size)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=[],
        just_point_forecast: bool = False,
    ):
        """Generate forecast data immediately following dates of .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        if int(forecast_length) > int(self.forecast_length):
            print("Regression must be refit to change forecast length!")
        predictStartTime = datetime.datetime.now()
        index = self.create_forecast_index(forecast_length=forecast_length)

        if self.output_dim == '1step':
            # combined_index = (self.df_train.index.append(index))
            forecast = pd.DataFrame()
            # forecast, 1 step ahead, then another, and so on
            for x in range(forecast_length):
                pred = last_window(
                    self.last_window,
                    window_size=self.window_size,
                    input_dim=self.input_dim,
                    normalize_window=self.normalize_window,
                )
                if str(self.regression_type).lower() == "user":
                    blasted_thing = future_regressor.iloc[x].to_frame().transpose()
                    tmerg = pd.concat([blasted_thing] * pred.shape[0], axis=0)
                    tmerg.index = pred.index
                    pred = pd.concat([pred, tmerg], axis=1, ignore_index=True)
                rfPred = pd.DataFrame(self.regr.predict(pred))
                if self.input_dim == 'univariate':
                    rfPred = rfPred.transpose()
                    rfPred.columns = self.last_window.columns
                forecast = pd.concat([forecast, rfPred], axis=0, ignore_index=True)
                self.last_window = pd.concat(
                    [self.last_window, rfPred], axis=0, ignore_index=True
                )
            df = forecast

        else:
            pred = last_window(
                self.last_window,
                window_size=self.window_size,
                input_dim=self.input_dim,
                normalize_window=self.normalize_window,
            )
            if str(self.regression_type).lower() == "user":
                tmerg = future_regressor.tail(1).loc[
                    future_regressor.tail(1).index.repeat(pred.shape[0])
                ]
                tmerg.index = pred.index
                pred = pd.concat([pred, tmerg], axis=1)
            cY = pd.DataFrame(self.regr.predict(pred))
            if self.input_dim == 'multivariate':
                cY.index = ['values']
                cY.columns = np.tile(self.column_names, reps=self.forecast_length)
                cY = cY.transpose().reset_index()
                cY['timestep'] = np.repeat(
                    range(forecast_length), repeats=len(self.column_names)
                )
                cY = pd.pivot_table(cY, index='timestep', columns='index')
            else:
                cY = cY.transpose()
            df = cY

        df.columns = self.column_names
        df.index = index
        if just_point_forecast:
            return df
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                df,
                prediction_interval=self.prediction_interval,
                method='historic_quantile',
            )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=df.index,
                forecast_columns=df.columns,
                lower_forecast=lower_forecast,
                forecast=df,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        window_size_choice = random.choice([5, 10, 20, seasonal_int()])
        model_choice = generate_regressor_params()
        input_dim_choice = random.choices(['multivariate', 'univariate'], [0.01, 0.99])[
            0
        ]
        if input_dim_choice == "multivariate":
            output_dim_choice = "1step"
            regression_type_choice = None
        else:
            output_dim_choice = random.choice(
                ['forecast_length', '1step'],
            )
            regression_type_choice = random.choices([None, "User"], weights=[0.8, 0.2])[
                0
            ]
        normalize_window_choice = random.choices([True, False], [0.05, 0.95])[0]
        max_windows_choice = random.choices([5000, 1000, 50000], [0.85, 0.05, 0.1])[0]
        return {
            'window_size': window_size_choice,
            'input_dim': input_dim_choice,
            'output_dim': output_dim_choice,
            'normalize_window': normalize_window_choice,
            'max_windows': max_windows_choice,
            'regression_type': regression_type_choice,
            'regression_model': model_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'window_size': self.window_size,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'normalize_window': self.normalize_window,
            'max_windows': self.max_windows,
            'regression_type': self.regression_type,
            'regression_model': self.regression_model,
        }


class ComponentAnalysis(ModelObject):
    """Forecasting on principle components.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        model (str): An AutoTS model str
        model_parameters (dict): parameters to pass to AutoTS model
        n_components (int): int or 'NthN' number of components to use
        decomposition (str): decomposition method to use from scikit-learn

    """

    def __init__(
        self,
        name: str = "ComponentAnalysis",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_components: int = 10,
        forecast_length: int = 14,
        model: str = 'GLS',
        model_parameters: dict = {},
        decomposition: str = 'PCA',
        n_jobs: int = -1,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.model = model
        self.model_parameters = model_parameters
        self.decomposition = decomposition
        self.n_components = n_components
        self.forecast_length = forecast_length

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df
        if 'thN' in str(self.n_components):
            n_int = int(''.join([x for x in str(self.n_components) if x.isdigit()]))
            n_int = int(np.floor(df.shape[1] / n_int))
            n_int = n_int if n_int >= 2 else 2
        else:
            n_int = int(''.join([x for x in str(self.n_components) if x.isdigit()]))
        self.n_int = n_int

        if self.decomposition == 'TruncatedSVD':
            from sklearn.decomposition import TruncatedSVD

            transformer = TruncatedSVD(
                n_components=self.n_int, random_state=self.random_seed
            )
        elif self.decomposition == 'WhitenedPCA':
            from sklearn.decomposition import PCA

            transformer = PCA(
                n_components=self.n_int, whiten=True, random_state=self.random_seed
            )
        elif self.decomposition == 'PCA':
            from sklearn.decomposition import PCA

            transformer = PCA(
                n_components=self.n_int, whiten=False, random_state=self.random_seed
            )
        elif self.decomposition == 'KernelPCA':
            from sklearn.decomposition import KernelPCA

            transformer = KernelPCA(
                n_components=self.n_int,
                kernel='rbf',
                random_state=self.random_seed,
                fit_inverse_transform=True,
            )
        elif self.decomposition == 'FastICA':
            from sklearn.decomposition import FastICA

            transformer = FastICA(
                n_components=self.n_int,
                whiten=True,
                random_state=self.random_seed,
                max_iter=500,
            )
        try:
            self.transformer = transformer.fit(df)
        except ValueError:
            raise ValueError(
                "n_components and decomposition not suitable for this dataset."
            )
        X = self.transformer.transform(df)
        X = pd.DataFrame(X)
        X.index = df.index
        from autots.evaluator.auto_model import ModelMonster

        try:
            self.modelobj = ModelMonster(
                self.model,
                parameters=self.model_parameters,
                frequency=self.frequency,
                prediction_interval=self.prediction_interval,
                holiday_country=self.holiday_country,
                random_seed=self.random_seed,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                forecast_length=self.forecast_length,
            ).fit(X, future_regressor=future_regressor)
        except Exception as e:
            raise ValueError(f"Model {str(self.model)} with error: {repr(e)}")
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=[],
        just_point_forecast: bool = False,
    ):
        """Generate forecast data immediately following dates of .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()

        XA = self.modelobj.predict(
            forecast_length=forecast_length, future_regressor=future_regressor
        )
        Xf = self.transformer.inverse_transform(np.array(XA.forecast))
        if not isinstance(Xf, pd.DataFrame):
            Xf = pd.DataFrame(Xf)
        Xf.columns = self.column_names
        Xf.index = self.create_forecast_index(forecast_length=forecast_length)
        Xf = Xf.astype(float)
        if just_point_forecast:
            return Xf
        else:
            """
            upper_forecast = self.transformer.inverse_transform(np.array(XA.upper_forecast))
            if not isinstance(upper_forecast, pd.DataFrame):
                upper_forecast = pd.DataFrame(upper_forecast)
            upper_forecast.columns = self.column_names
            upper_forecast.index = self.create_forecast_index(forecast_length=forecast_length)

            lower_forecast = self.transformer.inverse_transform(np.array(XA.lower_forecast))
            if not isinstance(lower_forecast, pd.DataFrame):
                lower_forecast = pd.DataFrame(lower_forecast)
            lower_forecast.columns = self.column_names
            lower_forecast.index = self.create_forecast_index(forecast_length=forecast_length)
            """
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                Xf,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=Xf.index,
                forecast_columns=Xf.columns,
                lower_forecast=lower_forecast,
                forecast=Xf,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        n_components_choice = np.random.choice(
            a=[10, '10thN'], size=1, p=[0.6, 0.4]
        ).item()
        decomposition_choice = np.random.choice(
            a=['TruncatedSVD', 'WhitenedPCA', 'PCA', 'KernelPCA', 'FastICA'],
            size=1,
            p=[0.05, 0.05, 0.5, 0.2, 0.2],
        ).item()
        model_list = [
            'LastValueNaive',
            'GLS',
            'TensorflowSTS',
            'GLM',
            'ETS',
            'FBProphet',
            'MotifSimulation',
            'RollingRegression',
            'WindowRegression',
            'UnobservedComponents',
            'VECM',
        ]
        model_str = np.random.choice(
            model_list,
            size=1,
            p=[0.01, 0.01, 0.01, 0.01, 0.01, 0.7, 0.01, 0.02, 0.1, 0.1, 0.02],
        ).item()
        model_str = np.random.choice(model_list)
        from autots.evaluator.auto_model import ModelMonster

        param_dict = ModelMonster(model_str).get_new_params()
        return {
            'model': model_str,
            'model_parameters': param_dict,
            'decomposition': decomposition_choice,
            'n_components': n_components_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'model': self.model,
            'model_parameters': self.model_parameters,
            'decomposition': self.decomposition,
            'n_components': self.n_components,
        }


class DatepartRegression(ModelObject):
    """Regression not on series but datetime

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
    """

    def __init__(
        self,
        name: str = "DatepartRegression",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        forecast_length: int = 1,
        n_jobs: int = None,
        regression_model: dict = {
            "model": 'DecisionTree',
            "model_params": {"max_depth": 5, "min_samples_split": 2},
        },
        datepart_method: str = 'expanded',
        regression_type: str = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            regression_type=regression_type,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.regression_model = regression_model
        self.datepart_method = datepart_method

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        # if external regressor, do some check up
        if self.regression_type is not None:
            if (np.array(future_regressor).shape[0]) != (df.shape[0]):
                self.regression_type = None

        y = df.values

        X = date_part(df.index, method=self.datepart_method)
        if self.regression_type == 'User':
            X = pd.concat([X, future_regressor], axis=0)
        X.columns = [str(xc) for xc in X.columns]

        multioutput = True
        if y.ndim < 2:
            multioutput = False
        elif y.shape[1] < 2:
            multioutput = False
            y = y.ravel()
        self.model = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=self.verbose,
            verbose_bool=self.verbose_bool,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            multioutput=multioutput,
        )
        self.df_train = df
        self.model = self.model.fit(X, y)
        self.shape = df.shape
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=[],
        just_point_forecast: bool = False,
    ):
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
        X = date_part(index, method=self.datepart_method)
        if self.regression_type == 'User':
            X = pd.concat([X, future_regressor], axis=0)
        X.columns = [str(xc) for xc in X.columns]

        forecast = pd.DataFrame(self.model.predict(X))

        forecast.columns = self.column_names
        forecast.index = index

        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=forecast.index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        model_choice = generate_regressor_params(model_dict=datepart_model_dict)
        datepart_choice = np.random.choice(
            a=["recurring", "simple", "expanded"], size=1, p=[0.4, 0.3, 0.3]
        ).item()
        regression_choice = np.random.choice(
            a=[None, 'User'], size=1, p=[0.7, 0.3]
        ).item()
        parameter_dict = {
            'regression_model': model_choice,
            'datepart_method': datepart_choice,
            'regression_type': regression_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'regression_model': self.regression_model,
            'datepart_method': self.datepart_method,
            'regression_type': self.regression_type,
        }
        return parameter_dict


class UnivariateRegression(ModelObject):
    """Regression-framed approach to forecasting using sklearn.
    A univariate version of rolling regression: ie each series is modeled independently

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holiday flags
        regression_type (str): type of regression (None, 'User')

    """

    def __init__(
        self,
        name: str = "UnivariateRegression",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        verbose: int = 0,
        random_seed: int = 2020,
        forecast_length: int = 7,
        regression_model: dict = {
            "model": 'Adaboost',
            "model_params": {
                'n_estimators': 50,
                'base_estimator': 'DecisionTree',
                'loss': 'linear',
                'learning_rate': 1.0,
            },
        },
        holiday: bool = False,
        mean_rolling_periods: int = 30,
        macd_periods: int = None,
        std_rolling_periods: int = 7,
        max_rolling_periods: int = 7,
        min_rolling_periods: int = 7,
        ewm_alpha: float = 0.5,
        additional_lag_periods: int = 7,
        abs_energy: bool = False,
        rolling_autocorr_periods: int = None,
        add_date_part: str = None,
        polynomial_degree: int = None,
        x_transform: str = None,
        window: int = None,
        n_jobs: int = -1,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            regression_type=regression_type,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.forecast_length = forecast_length
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
        self.window = window

    def _x_transformer(self):
        if self.x_transform == 'FastICA':
            from sklearn.decomposition import FastICA

            x_transformer = FastICA(n_components=None, random_state=2020, whiten=True)
        elif self.x_transform == 'Nystroem':
            from sklearn.kernel_approximation import Nystroem

            half_size = int(self.sktraindata.shape[0] / 2) + 1
            max_comp = 200
            n_comp = max_comp if half_size > max_comp else half_size
            x_transformer = Nystroem(
                kernel='rbf', gamma=0.2, random_state=2020, n_components=n_comp
            )
        else:
            # self.x_transform = 'RmZeroVariance'
            from sklearn.feature_selection import VarianceThreshold

            x_transformer = VarianceThreshold(threshold=0.0)
        return x_transformer

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            future_regressor (pandas.DataFrame or Series): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        # if external regressor, do some check up
        if self.regression_type is not None:
            if (np.array(future_regressor).shape[0]) != (df.shape[0]):
                self.regression_type = None
            else:
                self.regressor_train = future_regressor

        # define X and Y
        self.sktraindata = self.df_train.dropna(how='all', axis=0)
        self.sktraindata = self.sktraindata.fillna(method='ffill').fillna(
            method='bfill'
        )
        cols = self.sktraindata.columns

        def forecast_by_column(self, args, parallel, n_jobs, col):
            """Run one series of ETS and return prediction."""
            base = pd.DataFrame(self.sktraindata[col])
            Y = base.copy()
            for curr_shift in range(1, self.forecast_length):
                Y = pd.concat([Y, base.shift(-curr_shift)], axis=1)
            # drop incomplete data
            Y = Y.drop(index=Y.tail(self.forecast_length - 1).index)
            # drop the most recent because there's no future for it
            Y = Y.drop(index=Y.index[0])
            Y.columns = [x for x in range(len(Y.columns))]

            X = rolling_x_regressor(
                base,
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
                polynomial_degree=self.polynomial_degree,
                window=self.window,
            )
            if self.regression_type == 'User':
                X = pd.concat([X, self.regressor_train], axis=1)

            if self.x_transform in ['FastICA', 'Nystroem', 'RmZeroVariance']:
                self.x_transformer = self._x_transformer()
                self.x_transformer = self.x_transformer.fit(X)
                X = pd.DataFrame(self.x_transformer.transform(X))
                X = X.replace([np.inf, -np.inf], 0).fillna(0)

            X = X.drop(index=X.tail(self.forecast_length).index)
            Y.index = X.index  # and just hope I got the adjustments right

            # retrieve model object to train
            if not parallel and n_jobs > 1:
                n_jobs_passed = n_jobs
            else:
                n_jobs_passed = 1
            multioutput = True
            if Y.ndim < 2:
                multioutput = False
            elif Y.shape[1] < 2:
                multioutput = False
            dah_model = retrieve_regressor(
                regression_model=self.regression_model,
                verbose=self.verbose,
                verbose_bool=self.verbose_bool,
                random_seed=self.random_seed,
                n_jobs=n_jobs_passed,
                multioutput=multioutput,
            )
            dah_model.fit(X, Y)
            return {col: dah_model}

        self.parallel = True
        if self.n_jobs in [0, 1] or len(cols) < 3:
            self.parallel = False
        else:
            try:
                from joblib import Parallel, delayed
            except Exception:
                self.parallel = False
        args = {}
        # joblib multiprocessing to loop through series
        if self.parallel:
            df_list = Parallel(n_jobs=(self.n_jobs - 1))(
                delayed(forecast_by_column)(self, args, self.parallel, self.n_jobs, col)
                for (col) in cols
            )
            self.models = {k: v for d in df_list for k, v in d.items()}
        else:
            df_list = []
            for col in cols:
                df_list.append(
                    forecast_by_column(self, args, self.parallel, self.n_jobs, col)
                )
            self.models = {k: v for d in df_list for k, v in d.items()}
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int = None,
        just_point_forecast: bool = False,
        future_regressor=[],
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
                ignored here for this model, must be set in __init__ before .fit()
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        index = self.create_forecast_index(forecast_length=self.forecast_length)
        forecast = pd.DataFrame()
        for x_col in self.sktraindata.columns:
            base = pd.DataFrame(self.sktraindata[x_col])

            x_dat = rolling_x_regressor(
                base,
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
                polynomial_degree=self.polynomial_degree,
                window=self.window,
            )
            if self.regression_type == 'User':
                x_dat = pd.concat([x_dat, self.regressor_train], axis=1).fillna(0)
            if self.x_transform in ['FastICA', 'Nystroem', 'RmZeroVariance']:
                x_dat = pd.DataFrame(self.x_transformer.transform(x_dat))
                x_dat = x_dat.replace([np.inf, -np.inf], 0).fillna(0)
            rfPred = self.models[x_col].predict(x_dat.tail(1).values)
            # rfPred = pd.DataFrame(rfPred).transpose()
            # rfPred.columns = [x_col]
            rfPred = pd.Series(rfPred.flatten())
            rfPred.name = x_col
            forecast = pd.concat([forecast, rfPred], axis=1)

        forecast = forecast[self.column_names]
        forecast.index = index

        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=self.forecast_length,
                forecast_index=forecast.index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        model_choice = generate_regressor_params(model_dict=univariate_model_dict)
        mean_rolling_periods_choice = random.choices(
            [None, 5, 7, 12, 30], [0.6, 0.1, 0.1, 0.1, 0.1]
        )[0]
        if mean_rolling_periods_choice is not None:
            macd_periods_choice = seasonal_int()
            if macd_periods_choice == mean_rolling_periods_choice:
                macd_periods_choice = mean_rolling_periods_choice + 10
        else:
            macd_periods_choice = None
        std_rolling_periods_choice = random.choices(
            [None, 5, 7, 10, 30], [0.6, 0.1, 0.1, 0.1, 0.1]
        )[0]
        max_rolling_periods_choice = random.choices([None, seasonal_int()], [0.5, 0.5])[
            0
        ]
        min_rolling_periods_choice = random.choices([None, seasonal_int()], [0.5, 0.5])[
            0
        ]
        lag_periods_choice = seasonal_int() - 1
        lag_periods_choice = 2 if lag_periods_choice < 2 else lag_periods_choice
        ewm_choice = random.choices([None, 0.2, 0.5, 0.8], [0.75, 0.1, 0.1, 0.05])[0]
        abs_energy_choice = random.choices([True, False], [0.1, 0.9])[0]
        rolling_autocorr_periods_choice = random.choices(
            [None, 2, 7, 12, 30], [0.86, 0.01, 0.01, 0.01, 0.01]
        )[0]
        add_date_part_choice = random.choices(
            [None, 'simple', 'expanded', 'recurring'], [0.7, 0.1, 0.1, 0.1]
        )[0]
        holiday_choice = random.choices([True, False], [0.2, 0.8])[0]
        polynomial_degree_choice = None
        x_transform_choice = random.choices(
            [None, 'FastICA', 'Nystroem', 'RmZeroVariance'],
            [1.0, 0.0, 0.0, 0.0],
        )[0]
        regression_choice = random.choices([None, 'User'], [0.7, 0.3])[0]
        window_choice = random.choices([None, 3, 7, 10], [0.7, 0.2, 0.05, 0.05])[0]
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
            'regression_type': regression_choice,
            'window': window_choice,
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
            'window': self.window,
        }
        return parameter_dict
