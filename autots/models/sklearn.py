"""
Sklearn dependent models

Decision Tree, Elastic Net,  Random Forest, MLPRegressor, KNN, Adaboost
"""
import datetime
import random
import numpy as np
import pandas as pd

try:  # needs to go first
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.tools.seasonal import date_part, seasonal_int
from autots.tools.window_functions import window_maker, last_window


def rolling_x_regressor(
    df,
    mean_rolling_periods: int = 30,
    macd_periods: int = None,
    std_rolling_periods: int = 7,
    max_rolling_periods: int = None,
    min_rolling_periods: int = None,
    quantile90_rolling_periods: int = None,
    quantile10_rolling_periods: int = None,
    ewm_alpha: float = 0.5,
    ewm_var_alpha: float = None,
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
    # making this all or partially Numpy (if possible) would probably be faster
    X = [df.copy()]
    if str(mean_rolling_periods).isdigit():
        temp = df.rolling(int(mean_rolling_periods), min_periods=1).median()
        X.append(temp)
        if str(macd_periods).isdigit():
            temp = df.rolling(int(macd_periods), min_periods=1).median() - temp
            X.append(temp)
    if str(std_rolling_periods).isdigit():
        X.append(df.rolling(std_rolling_periods, min_periods=1).std())
    if str(max_rolling_periods).isdigit():
        X.append(df.rolling(max_rolling_periods, min_periods=1).max())
    if str(min_rolling_periods).isdigit():
        X.append(df.rolling(min_rolling_periods, min_periods=1).min())
    if str(quantile90_rolling_periods).isdigit():
        X.append(df.rolling(quantile90_rolling_periods, min_periods=1).quantile(0.9))
    if str(quantile10_rolling_periods).isdigit():
        X.append(df.rolling(quantile10_rolling_periods, min_periods=1).quantile(0.1))
    if str(ewm_alpha).replace('.', '').isdigit():
        X.append(df.ewm(alpha=ewm_alpha, ignore_na=True, min_periods=1).mean())
    if str(ewm_var_alpha).replace('.', '').isdigit():
        X.append(df.ewm(alpha=ewm_var_alpha, ignore_na=True, min_periods=1).var())
    if str(additional_lag_periods).isdigit():
        X.append(df.shift(additional_lag_periods))
    if abs_energy:
        X.append(df.pow(other=([2] * len(df.columns))).cumsum())
    if str(rolling_autocorr_periods).isdigit():
        temp = df.rolling(rolling_autocorr_periods).apply(
            lambda x: x.autocorr(), raw=False
        )
        X.append(temp)
    if add_date_part in ['simple', 'expanded', 'recurring', "simple_2"]:
        date_part_df = date_part(df.index, method=add_date_part)
        date_part_df.index = df.index
        X.append(date_part_df)
    # unlike the others, this pulls the entire window, not just one lag
    if str(window).isdigit():
        # we already have lag 1 using this
        for curr_shift in range(1, window):
            X.append(df.shift(curr_shift))
    X = pd.concat(X, axis=1)

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

    # X = X.replace([np.inf, -np.inf], np.nan)
    X.fillna(method='bfill', inplace=True)

    X.columns = [str(x) for x in range(len(X.columns))]

    return X


def rolling_x_regressor_regressor(
    df,
    mean_rolling_periods: int = 30,
    macd_periods: int = None,
    std_rolling_periods: int = 7,
    max_rolling_periods: int = None,
    min_rolling_periods: int = None,
    quantile90_rolling_periods: int = None,
    quantile10_rolling_periods: int = None,
    ewm_alpha: float = 0.5,
    ewm_var_alpha: float = None,
    additional_lag_periods: int = 7,
    abs_energy: bool = False,
    rolling_autocorr_periods: int = None,
    add_date_part: str = None,
    holiday: bool = False,
    holiday_country: str = 'US',
    polynomial_degree: int = None,
    window: int = None,
    future_regressor=None,
):
    """Adds in the future_regressor."""
    X = rolling_x_regressor(
        df,
        mean_rolling_periods=mean_rolling_periods,
        macd_periods=macd_periods,
        std_rolling_periods=std_rolling_periods,
        max_rolling_periods=max_rolling_periods,
        min_rolling_periods=min_rolling_periods,
        ewm_var_alpha=ewm_var_alpha,
        quantile90_rolling_periods=quantile90_rolling_periods,
        quantile10_rolling_periods=quantile10_rolling_periods,
        additional_lag_periods=additional_lag_periods,
        ewm_alpha=ewm_alpha,
        abs_energy=abs_energy,
        rolling_autocorr_periods=rolling_autocorr_periods,
        add_date_part=add_date_part,
        holiday=holiday,
        holiday_country=holiday_country,
        polynomial_degree=polynomial_degree,
        window=window,
    )
    if future_regressor is not None:
        X = pd.concat([X, future_regressor], axis=1)
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

        regr = KerasRNN(verbose=verbose, random_seed=random_seed, **model_param_dict)
        return regr
    elif model_class == 'Transformer':
        from autots.models.dnn import Transformer

        regr = Transformer(verbose=verbose, random_seed=random_seed, **model_param_dict)
        return regr
    elif model_class == 'KNN':
        from sklearn.neighbors import KNeighborsRegressor

        if multioutput:
            from sklearn.multioutput import MultiOutputRegressor

            regr = MultiOutputRegressor(
                KNeighborsRegressor(**model_param_dict, n_jobs=1),
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
                    verbose=int(verbose_bool),
                    random_state=random_seed,
                    **model_param_dict,
                )
            )
        else:
            regr = HistGradientBoostingRegressor(
                verbose=int(verbose_bool),
                random_state=random_seed,
                **model_param_dict,
            )
        return regr
    elif model_class == 'LightGBM':
        from lightgbm import LGBMRegressor

        if multioutput:
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(
                LGBMRegressor(
                    verbose=int(verbose_bool),
                    random_state=random_seed,
                    n_jobs=1,
                    **model_param_dict,
                ),
                n_jobs=n_jobs,
            )
        else:
            return LGBMRegressor(
                verbose=int(verbose_bool),
                random_state=random_seed,
                n_jobs=n_jobs,
                **model_param_dict,
            )
    elif model_class == "LightGBMRegressorChain":
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
                xgb.XGBRegressor(verbosity=verbose, **model_param_dict, n_jobs=1),
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
    elif model_class == 'Ridge':
        from sklearn.linear_model import Ridge

        return Ridge(random_state=random_seed, **model_param_dict)
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
    elif model_class == "PoissonRegresssion":
        from sklearn.linear_model import PoissonRegressor

        if multioutput:
            from sklearn.multioutput import MultiOutputRegressor

            regr = MultiOutputRegressor(
                PoissonRegressor(fit_intercept=True, max_iter=200, **model_param_dict),
                n_jobs=n_jobs,
            )
        else:
            regr = PoissonRegressor(**model_param_dict)
        return regr
    elif model_class == 'RANSAC':
        from sklearn.linear_model import RANSACRegressor

        return RANSACRegressor(random_state=random_seed, **model_param_dict)
    elif model_class == "GaussianProcessRegressor":
        from sklearn.gaussian_process import GaussianProcessRegressor

        kernel = model_param_dict.pop("kernel", None)
        if kernel is not None:
            if kernel == "DotWhite":
                from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

                kernel = DotProduct() + WhiteKernel()
            elif kernel == "White":
                from sklearn.gaussian_process.kernels import WhiteKernel

                kernel = WhiteKernel()
            elif kernel == "ExpSineSquared":
                from sklearn.gaussian_process.kernels import ExpSineSquared

                kernel = ExpSineSquared()
            else:
                from sklearn.gaussian_process.kernels import RBF

                kernel = RBF()
        return GaussianProcessRegressor(
            kernel=kernel, random_state=random_seed, **model_param_dict
        )
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


# models that can more quickly handle many X/Y obs, with modest number of features
sklearn_model_dict = {
    'RandomForest': 0.02,
    'ElasticNet': 0.05,
    'MLP': 0.05,
    'DecisionTree': 0.05,
    'KNN': 0.05,
    'Adaboost': 0.03,
    'SVM': 0.05,  # was slow, LinearSVR seems much faster
    'BayesianRidge': 0.05,
    'xgboost': 0.01,
    'KerasRNN': 0.02,
    'Transformer': 0.02,
    'HistGradientBoost': 0.03,
    'LightGBM': 0.03,
    'LightGBMRegressorChain': 0.03,
    'ExtraTrees': 0.05,
    'RadiusNeighbors': 0.02,
    'PoissonRegresssion': 0.03,
    'RANSAC': 0.05,
    'Ridge': 0.02,
    'GaussianProcessRegressor': 0.02,
}
multivariate_model_dict = {
    'RandomForest': 0.02,
    # 'ElasticNet': 0.05,
    'MLP': 0.03,
    'DecisionTree': 0.05,
    'KNN': 0.05,
    'Adaboost': 0.03,
    'SVM': 0.05,
    # 'BayesianRidge': 0.05,
    'xgboost': 0.01,
    'KerasRNN': 0.01,
    'HistGradientBoost': 0.03,
    'LightGBM': 0.03,
    'LightGBMRegressorChain': 0.03,
    'ExtraTrees': 0.05,
    'RadiusNeighbors': 0.02,
    'PoissonRegresssion': 0.03,
    'RANSAC': 0.05,
    'Ridge': 0.02,
}
# these should train quickly with low dimensional X/Y, and not mind being run multiple in parallel
univariate_model_dict = {
    'ElasticNet': 0.05,
    'MLP': 0.05,
    'DecisionTree': 0.05,
    'KNN': 0.03,
    'Adaboost': 0.05,
    'SVM': 0.05,
    'BayesianRidge': 0.03,
    'HistGradientBoost': 0.02,
    'LightGBM': 0.01,
    'LightGBMRegressorChain': 0.01,
    'ExtraTrees': 0.05,
    'RadiusNeighbors': 0.05,
    'RANSAC': 0.02,
}
# for high dimensionality, many-feature X, many-feature Y, but with moderate obs count
rolling_regression_dict = {
    'RandomForest': 0.02,
    'ElasticNet': 0.05,
    'MLP': 0.05,
    'DecisionTree': 0.05,
    'KNN': 0.05,
    'Adaboost': 0.03,
    'SVM': 0.05,
    'KerasRNN': 0.02,
    'LightGBM': 0.03,
    'LightGBMRegressorChain': 0.03,
    'ExtraTrees': 0.05,
    'RadiusNeighbors': 0.01,
    'PoissonRegresssion': 0.03,
    'RANSAC': 0.05,
    'Ridge': 0.02,
}
# models where we can be sure the model isn't sharing information across multiple Y's...
no_shared_model_dict = {
    'KNN': 0.1,
    'Adaboost': 0.1,
    'SVM': 0.1,
    'xgboost': 0.1,
    'LightGBM': 0.1,
    'HistGradientBoost': 0.1,
    'PoissonRegresssion': 0.05,
}
# these are models that are relatively fast with large multioutput Y, small n obs
datepart_model_dict: dict = {
    'RandomForest': 0.05,
    'ElasticNet': 0.05,
    'MLP': 0.05,
    'DecisionTree': 0.05,
    'Adaboost': 0.05,
    'SVM': 0.05,
    'KerasRNN': 0.05,
    'Transformer': 0.05,
    'ExtraTrees': 0.07,
    'RadiusNeighbors': 0.05,
}


def generate_regressor_params(
    model_dict=None,
    method="default",
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
        'LightGBMRegressorChain',
        'MLP',
        'KNN',
        'KerasRNN',
        'Transformer',
        'HistGradientBoost',
        'RandomForest',
        'ExtraTrees',
        'Ridge',
        'GaussianProcessRegressor',
    ]:
        if model == 'Adaboost':
            param_dict = {
                "model": 'Adaboost',
                "model_params": {
                    "n_estimators": random.choices([50, 100, 500], [0.7, 0.2, 0.1])[0],
                    "loss": random.choices(
                        ['linear', 'square', 'exponential'], [0.8, 0.01, 0.1]
                    )[0],
                    "base_estimator": random.choices(
                        [None, 'LinReg', 'SVR'], [0.8, 0.1, 0.1]
                    )[0],
                    "learning_rate": random.choices([1, 0.5], [0.9, 0.1])[0],
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
                    "n_neighbors": random.choices([3, 5, 10, 14], [0.2, 0.7, 0.1, 0.1])[
                        0
                    ],
                    "weights": random.choices(['uniform', 'distance'], [0.7, 0.3])[0],
                    'p': random.choices([2, 1, 1.5], [0.7, 0.1, 0.1])[0],
                    'leaf_size': random.choices([30, 10, 50], [0.8, 0.1, 0.1])[0],
                },
            }
        elif model == 'RandomForest':
            param_dict = {
                "model": 'RandomForest',
                "model_params": {
                    "n_estimators": random.choices(
                        [300, 100, 1000, 5000], [0.4, 0.4, 0.2, 0.01]
                    )[0],
                    "min_samples_leaf": random.choices([2, 4, 1], [0.2, 0.2, 0.8])[0],
                    "bootstrap": random.choices([True, False], [0.9, 0.1])[0],
                    # absolute_error is noticeably slower
                    # "criterion": random.choices(
                    #     ["squared_error", "poisson"], [0.99, 0.001]
                    # )[0],
                },
            }
        elif model == 'ExtraTrees':
            max_depth_choice = random.choices([None, 5, 10, 20], [0.2, 0.1, 0.5, 0.3])[
                0
            ]
            estimators_choice = random.choices([50, 100, 500], [0.05, 0.9, 0.05])[0]
            param_dict = {
                "model": 'ExtraTrees',
                "model_params": {
                    "n_estimators": estimators_choice,
                    "min_samples_leaf": random.choices([2, 4, 1], [0.1, 0.1, 0.8])[0],
                    "max_depth": max_depth_choice,
                    # "criterion": "squared_error",
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
                        [50, 100, 200, 500, 750], [0.75, 0.2, 0.05, 0.01, 0.001]
                    )[0],
                    "batch_size": random.choices([8, 16, 32, 72], [0.2, 0.2, 0.5, 0.1])[
                        0
                    ],
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
                        [50, 100, 200, 500, 750], [0.75, 0.2, 0.05, 0.01, 0.001]
                    )[0],
                    "batch_size": random.choices(
                        [8, 16, 32, 64, 72], [0.01, 0.2, 0.5, 0.1, 0.1]
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
                    "num_heads": random.choices([2, 4], [0.2, 0.2])[0],
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
                    "loss": random.choices(
                        ['squared_error', 'poisson', 'absolute_error'], [0.8, 0.1, 0.1]
                    )[0],
                    "learning_rate": random.choices([1, 0.1, 0.01], [0.3, 0.4, 0.3])[0],
                    "max_depth": random.choices(
                        [None, 5, 10, 20], [0.7, 0.1, 0.1, 0.1]
                    )[0],
                    "min_samples_leaf": random.choices(
                        [20, 5, 10, 30], [0.9, 0.1, 0.1, 0.1]
                    )[0],
                    "max_iter": random.choices(
                        [100, 250, 50, 500], [0.9, 0.1, 0.1, 0.001]
                    )[0],
                    "l2_regularization": random.choices(
                        [0, 0.01, 0.02, 0.4], [0.9, 0.1, 0.1, 0.1]
                    )[0],
                },
            }
        elif model in ['LightGBM', "LightGBMRegressorChain"]:
            param_dict = {
                "model": 'LightGBM',
                "model_params": {
                    "objective": random.choices(
                        [
                            'regression',
                            'gamma',
                            'huber',
                            'regression_l1',
                            'tweedie',
                            'poisson',
                            'quantile',
                        ],
                        [0.4, 0.2, 0.2, 0.2, 0.2, 0.05, 0.01],
                    )[0],
                    "learning_rate": random.choices(
                        [0.001, 0.1, 0.01],
                        [0.1, 0.6, 0.3],
                    )[0],
                    "num_leaves": random.choices(
                        [31, 127, 70],
                        [0.6, 0.1, 0.3],
                    )[0],
                    "max_depth": random.choices(
                        [-1, 5, 10],
                        [0.6, 0.1, 0.3],
                    )[0],
                    "boosting_type": random.choices(
                        ['gbdt', 'rf', 'dart', 'goss'],
                        [0.6, 0, 0.2, 0.2],
                    )[0],
                    "n_estimators": random.choices(
                        [100, 250, 50, 500],
                        [0.6, 0.1, 0.3, 0.0010],
                    )[0],
                },
            }
        elif model == 'Ridge':
            param_dict = {
                "model": 'Ridge',
                "model_params": {
                    'alpha': random.choice([1.0, 10.0, 0.1, 0.00001]),
                },
            }
        elif model == 'GaussianProcessRegressor':
            param_dict = {
                "model": 'GaussianProcessRegressor',
                "model_params": {
                    'alpha': random.choice([0.0000000001, 0.00001]),
                    'kernel': random.choices(
                        [None, "DotWhite", "White", "RBF", "ExpSineSquared"],
                        [0.4, 0.1, 0.1, 0.1, 0.1],
                    )[0],
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
            "model": 'ExtraTrees',
            "model_params": {},
        },
        holiday: bool = False,
        mean_rolling_periods: int = 30,
        macd_periods: int = None,
        std_rolling_periods: int = 7,
        max_rolling_periods: int = 7,
        min_rolling_periods: int = 7,
        ewm_var_alpha: int = None,
        quantile90_rolling_periods: int = None,
        quantile10_rolling_periods: int = None,
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
        self.ewm_var_alpha = ewm_var_alpha
        self.quantile90_rolling_periods = quantile90_rolling_periods
        self.quantile10_rolling_periods = quantile10_rolling_periods
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

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            future_regressor (pandas.DataFrame or Series): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        # if external regressor, do some check up
        if self.regression_type is not None:
            if future_regressor is None:
                raise ValueError(
                    "future_regressor not supplied, necessary for regression_type"
                )
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
            max_rolling_periods=self.max_rolling_periods,
            min_rolling_periods=self.min_rolling_periods,
            ewm_var_alpha=self.ewm_var_alpha,
            quantile90_rolling_periods=self.quantile90_rolling_periods,
            quantile10_rolling_periods=self.quantile10_rolling_periods,
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
        """
        use_device = False
        try:
            device = dpctl.SyclDevice("gpu,cpu")
            if self.verbose > 0:
                print(f"{'GPU' if device.is_gpu else 'CPU'} targeted: ", device)
            try:
                x_device = from_numpy(X, usm_type='device', queue=dpctl.SyclQueue(device))
                y_device = from_numpy(Y, usm_type='device', queue=dpctl.SyclQueue(device))
            except Exception:
                x_device = from_numpy(X, usm_type='device', device=device, sycl_queue=dpctl.SyclQueue(device))
                y_device = from_numpy(Y, usm_type='device', device=device, sycl_queue=dpctl.SyclQueue(device))
            use_device = True
        except Exception:
            x_device = X
            y_device = Y
        """
        self.regr = self.regr.fit(X, Y)

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=None,
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
        if self.regression_type in ['User', 'user']:
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
                max_rolling_periods=self.max_rolling_periods,
                min_rolling_periods=self.min_rolling_periods,
                ewm_var_alpha=self.ewm_var_alpha,
                quantile90_rolling_periods=self.quantile90_rolling_periods,
                quantile10_rolling_periods=self.quantile10_rolling_periods,
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

            rfPred = pd.DataFrame(self.regr.predict(x_dat.tail(1).to_numpy()))

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
        rolling_model_dict = sklearn_model_dict.copy()
        del rolling_model_dict['KNN']
        model_choice = generate_regressor_params(model_dict=rolling_model_dict)
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
        ewm_choice = random.choices(
            [None, 0.05, 0.1, 0.2, 0.5, 0.8], [0.4, 0.01, 0.05, 0.1, 0.1, 0.05]
        )[0]
        abs_energy_choice = random.choices([True, False], [0.1, 0.9])[0]
        rolling_autocorr_periods_choice = random.choices(
            [None, 2, 7, 12, 30], [0.8, 0.05, 0.05, 0.05, 0.05]
        )[0]
        add_date_part_choice = random.choices(
            [None, 'simple', 'expanded', 'recurring', "simple_2"],
            [0.7, 0.05, 0.1, 0.1, 0.05],
        )[0]
        holiday_choice = random.choices([True, False], [0.2, 0.8])[0]
        polynomial_degree_choice = random.choices([None, 2], [0.99, 0.01])[0]
        x_transform_choice = random.choices(
            [None, 'FastICA', 'Nystroem', 'RmZeroVariance'],
            [0.85, 0.05, 0.05, 0.05],
        )[0]
        if "regressor" in method:
            regression_choice = "User"
        else:
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
            "ewm_var_alpha": self.ewm_var_alpha,
            "quantile90_rolling_periods": self.quantile90_rolling_periods,
            "quantile10_rolling_periods": self.quantile10_rolling_periods,
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
            "model": 'RandomForest',
            "model_params": {},
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

    def fit(self, df, future_regressor=None):
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
        if self.regression_type in ["User", "user"]:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but no future_regressor passed"
                )
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
        future_regressor=None,
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
                if self.regression_type in ["User", "user"]:
                    blasted_thing = future_regressor.iloc[x].to_frame().transpose()
                    tmerg = pd.concat([blasted_thing] * pred.shape[0], axis=0)
                    tmerg.index = pred.index
                    pred = pd.concat([pred, tmerg], axis=1, ignore_index=True)
                if isinstance(pred, pd.DataFrame):
                    pred = pred.to_numpy()
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
            if self.regression_type in ["User", "user"]:
                tmerg = future_regressor.tail(1).loc[
                    future_regressor.tail(1).index.repeat(pred.shape[0])
                ]
                tmerg.index = pred.index
                pred = pd.concat([pred, tmerg], axis=1)
            if isinstance(pred, pd.DataFrame):
                pred = pred.to_numpy()
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
        wnd_sz_choice = random.choice([5, 10, 20, seasonal_int()])
        if method != "deep":
            wnd_sz_choice = wnd_sz_choice if wnd_sz_choice < 91 else 90
        model_choice = generate_regressor_params(
            model_dict=sklearn_model_dict, method=method
        )
        if "regressor" in method:
            regression_type_choice = "User"
            input_dim_choice = 'univariate'
            output_dim_choice = random.choice(
                ['forecast_length', '1step'],
            )
        else:
            input_dim_choice = random.choices(
                ['multivariate', 'univariate'], [0.01, 0.99]
            )[0]
            if input_dim_choice == "multivariate":
                output_dim_choice = "1step"
                regression_type_choice = None
            else:
                output_dim_choice = random.choice(
                    ['forecast_length', '1step'],
                )
                regression_type_choice = random.choices(
                    [None, "User"], weights=[0.8, 0.2]
                )[0]
        normalize_window_choice = random.choices([True, False], [0.05, 0.95])[0]
        max_windows_choice = random.choices([5000, 1000, 50000], [0.85, 0.05, 0.1])[0]
        return {
            'window_size': wnd_sz_choice,
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

    def fit(self, df, future_regressor=None):
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
        future_regressor=None,
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
        polynomial_degree: int = None,
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
        self.polynomial_degree = polynomial_degree

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        # if external regressor, do some check up
        if self.regression_type is not None:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but no future_regressor passed"
                )

        y = df.to_numpy()

        X = date_part(
            df.index,
            method=self.datepart_method,
            polynomial_degree=self.polynomial_degree,
        )
        if self.regression_type in ['User', 'user']:
            # regr = future_regressor.copy()
            # regr.index = X.index
            X = pd.concat([X, future_regressor], axis=1)
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
        future_regressor=None,
        just_point_forecast: bool = False,
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            future_regressor (pandas.DataFrame or Series): Datetime Indexed
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        index = self.create_forecast_index(forecast_length=forecast_length)
        X = date_part(
            index, method=self.datepart_method, polynomial_degree=self.polynomial_degree
        )
        if self.regression_type in ['User', 'user']:
            X = pd.concat([X, future_regressor], axis=1)
            if X.shape[0] > index.shape[0]:
                raise ValueError("future_regressor and X index failed to align")
        X.columns = [str(xc) for xc in X.columns]

        forecast = pd.DataFrame(
            self.model.predict(X), index=index, columns=self.column_names
        )

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
        datepart_choice = random.choices(
            ["recurring", "simple", "expanded", "simple_2"], [0.4, 0.3, 0.3, 0.3]
        )[0]
        if datepart_choice in ["simple", "simple_2", "recurring"]:
            polynomial_choice = random.choices([None, 2, 3], [0.5, 0.2, 0.01])[0]
        else:
            polynomial_choice = None
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices([None, 'User'], [0.7, 0.3])[0]
        parameter_dict = {
            'regression_model': model_choice,
            'datepart_method': datepart_choice,
            'polynomial_degree': polynomial_choice,
            'regression_type': regression_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'regression_model': self.regression_model,
            'datepart_method': self.datepart_method,
            'polynomial_degree': self.polynomial_degree,
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
            "model": 'ExtraTrees',
            "model_params": {},
        },
        holiday: bool = False,
        mean_rolling_periods: int = 30,
        macd_periods: int = None,
        std_rolling_periods: int = 7,
        max_rolling_periods: int = 7,
        min_rolling_periods: int = 7,
        ewm_var_alpha: float = None,
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
        self.ewm_var_alpha = ewm_var_alpha
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

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            future_regressor (pandas.DataFrame or Series): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.sktraindata = df

        # if external regressor, do some check up
        if self.regression_type is not None:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but not future_regressor supplied."
                )
            elif future_regressor.shape[0] != df.shape[0]:
                raise ValueError(
                    "future_regressor shape does not match training data shape."
                )
            else:
                self.regressor_train = future_regressor

        cols = self.sktraindata.columns

        def forecast_by_column(self, args, parallel, n_jobs, col):
            """Run one series and return prediction."""
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
                max_rolling_periods=self.max_rolling_periods,
                min_rolling_periods=self.min_rolling_periods,
                ewm_var_alpha=self.ewm_var_alpha,
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
            # because the training messages get annoying
            inner_verbose = self.verbose - 1 if self.verbose > 0 else self.verbose
            dah_model = retrieve_regressor(
                regression_model=self.regression_model,
                verbose=inner_verbose,
                verbose_bool=False,
                random_seed=self.random_seed,
                n_jobs=n_jobs_passed,
                multioutput=multioutput,
            )
            dah_model.fit(X.to_numpy(), Y)
            return {col: dah_model}

        self.parallel = True
        self.not_parallel_models = [
            'LightGBM',
            'RandomForest',
            "BayesianRidge",
            'Transformer',
            "KerasRNN",
            "HistGradientBoost",
        ]
        out_n_jobs = int(self.n_jobs - 1)
        out_n_jobs = 1 if out_n_jobs < 1 else out_n_jobs
        if out_n_jobs in [0, 1] or len(cols) < 3:
            self.parallel = False
        elif (
            self.regression_model.get("model", "ElasticNet") in self.not_parallel_models
        ):
            self.parallel = False
        else:
            try:
                from joblib import Parallel, delayed
            except Exception:
                self.parallel = False
        args = {}
        # joblib multiprocessing to loop through series
        if self.parallel:
            df_list = Parallel(n_jobs=out_n_jobs)(
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
        future_regressor=None,
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
                ignored here for this model, must be set in __init__ before .fit()
            future_regressor (pd.DataFrame): additional regressor
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
                max_rolling_periods=self.max_rolling_periods,
                min_rolling_periods=self.min_rolling_periods,
                ewm_var_alpha=self.ewm_var_alpha,
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
            rfPred = self.models[x_col].predict(x_dat.tail(1).to_numpy())
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
                self.sktraindata,
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
        if method == "deep":
            x_transform_choice = random.choices(
                [None, 'FastICA', 'Nystroem', 'RmZeroVariance'],
                [0.9, 0.03, 0.03, 0.04],
            )[0]
            window_choice = random.choices(
                [None, 3, 7, 10, 24], [0.7, 0.2, 0.05, 0.05, 0.05]
            )[0]
        else:
            x_transform_choice = random.choices(
                [None, 'FastICA', 'Nystroem', 'RmZeroVariance'],
                [1.0, 0.0, 0.0, 0.0],
            )[0]
            window_choice = random.choices([None, 3, 7, 10], [0.7, 0.2, 0.05, 0.05])[0]
        model_choice = generate_regressor_params(
            model_dict=univariate_model_dict, method=method
        )
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
        ewm_choice = random.choices(
            [None, 0.1, 0.2, 0.5, 0.8], [0.75, 0.05, 0.1, 0.1, 0.05]
        )[0]
        ewm_var_alpha = random.choices(
            [None, 0.05, 0.1, 0.2, 0.5, 0.8], [0.7, 0.01, 0.05, 0.1, 0.1, 0.05]
        )[0]
        abs_energy_choice = random.choices([True, False], [0.05, 0.95])[0]
        rolling_autocorr_periods_choice = random.choices(
            [None, 2, 7, 12, 30], [0.86, 0.01, 0.01, 0.01, 0.01]
        )[0]
        add_date_part_choice = random.choices(
            [None, 'simple', 'expanded', 'recurring', "simple_2", "simple_2_poly"],
            [0.8, 0.05, 0.1, 0.1, 0.05, 0.05],
        )[0]
        holiday_choice = random.choices([True, False], [0.2, 0.8])[0]
        polynomial_degree_choice = None

        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices([None, 'User'], [0.7, 0.3])[0]

        parameter_dict = {
            'regression_model': model_choice,
            'holiday': holiday_choice,
            'mean_rolling_periods': mean_rolling_periods_choice,
            'macd_periods': macd_periods_choice,
            'std_rolling_periods': std_rolling_periods_choice,
            'max_rolling_periods': max_rolling_periods_choice,
            'min_rolling_periods': min_rolling_periods_choice,
            "ewm_var_alpha": ewm_var_alpha,
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
            "ewm_var_alpha": self.ewm_var_alpha,
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


class MultivariateRegression(ModelObject):
    """Regression-framed approach to forecasting using sklearn.
    A multiariate version of rolling regression: ie each series is lagged independently but modeled together

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holiday flags
        regression_type (str): type of regression (None, 'User')

    """

    def __init__(
        self,
        name: str = "MultivariateRegression",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        verbose: int = 0,
        random_seed: int = 2020,
        forecast_length: int = 7,
        regression_model: dict = {
            "model": 'RandomForest',
            "model_params": {},
        },
        holiday: bool = False,
        mean_rolling_periods: int = 30,
        macd_periods: int = None,
        std_rolling_periods: int = 7,
        max_rolling_periods: int = 7,
        min_rolling_periods: int = 7,
        ewm_var_alpha: float = None,
        quantile90_rolling_periods: int = None,
        quantile10_rolling_periods: int = None,
        ewm_alpha: float = 0.5,
        additional_lag_periods: int = None,
        abs_energy: bool = False,
        rolling_autocorr_periods: int = None,
        datepart_method: str = None,
        polynomial_degree: int = None,
        window: int = 5,
        probabilistic: bool = False,
        quantile_params: dict = {
            'learning_rate': 0.1,
            'max_depth': 20,
            'min_samples_leaf': 4,
            'min_samples_split': 5,
            'n_estimators': 250,
        },
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
        self.ewm_var_alpha = ewm_var_alpha
        self.quantile90_rolling_periods = quantile90_rolling_periods
        self.quantile10_rolling_periods = quantile10_rolling_periods
        self.ewm_alpha = ewm_alpha
        self.additional_lag_periods = additional_lag_periods
        self.abs_energy = abs_energy
        self.rolling_autocorr_periods = rolling_autocorr_periods
        self.datepart_method = datepart_method
        self.polynomial_degree = polynomial_degree
        self.window = window
        self.quantile_params = quantile_params
        self.regressor_train = None
        self.probabilistic = probabilistic

        # detect just the max needed for cutoff (makes faster)
        starting_min = 90  # based on what effects ewm alphas, too
        list_o_vals = [
            mean_rolling_periods,
            macd_periods,
            std_rolling_periods,
            max_rolling_periods,
            min_rolling_periods,
            quantile90_rolling_periods,
            quantile10_rolling_periods,
            additional_lag_periods,
            rolling_autocorr_periods,
            window,
            starting_min,
        ]
        self.min_threshold = max([x for x in list_o_vals if str(x).isdigit()])

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            future_regressor (pandas.DataFrame or Series): Datetime Indexed
        """
        df = self.basic_profile(df)
        from sklearn.ensemble import GradientBoostingRegressor

        # if external regressor, do some check up
        if self.regression_type is not None:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but not future_regressor supplied."
                )
            elif future_regressor.shape[0] != df.shape[0]:
                raise ValueError(
                    "future_regressor shape does not match training data shape."
                )
            else:
                self.regressor_train = future_regressor

        # define X and Y
        Y = df[1:].to_numpy().ravel(order="F")
        # drop look ahead data
        base = df[:-1]
        if self.regression_type is not None:
            cut_regr = self.regressor_train[1:]
            cut_regr.index = base.index
        else:
            cut_regr = None
        # open to suggestions on making this faster
        X = pd.concat(
            [
                rolling_x_regressor_regressor(
                    base[x_col].to_frame(),
                    mean_rolling_periods=self.mean_rolling_periods,
                    macd_periods=self.macd_periods,
                    std_rolling_periods=self.std_rolling_periods,
                    max_rolling_periods=self.max_rolling_periods,
                    min_rolling_periods=self.min_rolling_periods,
                    ewm_var_alpha=self.ewm_var_alpha,
                    quantile90_rolling_periods=self.quantile90_rolling_periods,
                    quantile10_rolling_periods=self.quantile10_rolling_periods,
                    additional_lag_periods=self.additional_lag_periods,
                    ewm_alpha=self.ewm_alpha,
                    abs_energy=self.abs_energy,
                    rolling_autocorr_periods=self.rolling_autocorr_periods,
                    add_date_part=self.datepart_method,
                    holiday=self.holiday,
                    holiday_country=self.holiday_country,
                    polynomial_degree=self.polynomial_degree,
                    window=self.window,
                    future_regressor=cut_regr,
                )
                for x_col in base.columns
            ]
        ).to_numpy()
        del base
        if self.probabilistic:
            alpha_base = (1 - self.prediction_interval) / 2
            self.model_upper = GradientBoostingRegressor(
                loss='quantile',
                alpha=(1 - alpha_base),
                random_state=self.random_seed,
                **self.quantile_params,
            )
            self.model_lower = GradientBoostingRegressor(
                loss='quantile',
                alpha=alpha_base,
                random_state=self.random_seed,
                **self.quantile_params,
            )

        multioutput = True
        if Y.ndim < 2:
            multioutput = False
        elif Y.shape[1] < 2:
            multioutput = False
        self.model = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=self.verbose,
            verbose_bool=self.verbose_bool,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            multioutput=multioutput,
        )
        self.model.fit(X, Y)

        if self.probabilistic:
            self.model_upper.fit(X, Y)
            self.model_lower.fit(X, Y)
        # we only need the N most recent points for predict
        self.sktraindata = df.tail(self.min_threshold)

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int = None,
        just_point_forecast: bool = False,
        future_regressor=None,
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
                ignored here for this model, must be set in __init__ before .fit()
            future_regressor (pd.DataFrame): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        index = self.create_forecast_index(forecast_length=forecast_length)
        forecast = pd.DataFrame()
        upper_forecast = pd.DataFrame()
        lower_forecast = pd.DataFrame()
        if self.regressor_train is not None:
            base_regr = pd.concat([self.regressor_train, future_regressor])
            # move index back one to align with training dates on merge
            regr_idx = base_regr.index[:-1]
            base_regr = base_regr[1:]
            base_regr.index = regr_idx
        # need to copy else multiple predictions move every on...
        current_x = self.sktraindata.copy()

        # and this is ridiculously slow, nested loop
        for fcst_step in range(forecast_length):
            cur_regr = None
            if self.regression_type is not None:
                cur_regr = base_regr.reindex(current_x.index)
            x_dat = pd.concat(
                [
                    rolling_x_regressor_regressor(
                        current_x[x_col].to_frame(),
                        mean_rolling_periods=self.mean_rolling_periods,
                        macd_periods=self.macd_periods,
                        std_rolling_periods=self.std_rolling_periods,
                        max_rolling_periods=self.max_rolling_periods,
                        min_rolling_periods=self.min_rolling_periods,
                        ewm_var_alpha=self.ewm_var_alpha,
                        quantile90_rolling_periods=self.quantile90_rolling_periods,
                        quantile10_rolling_periods=self.quantile10_rolling_periods,
                        additional_lag_periods=self.additional_lag_periods,
                        ewm_alpha=self.ewm_alpha,
                        abs_energy=self.abs_energy,
                        rolling_autocorr_periods=self.rolling_autocorr_periods,
                        add_date_part=self.datepart_method,
                        holiday=self.holiday,
                        holiday_country=self.holiday_country,
                        polynomial_degree=self.polynomial_degree,
                        window=self.window,
                        future_regressor=cur_regr,
                    ).tail(1)
                    for x_col in current_x.columns
                ]
            ).to_numpy()
            rfPred = self.model.predict(x_dat)
            pred_clean = pd.DataFrame(
                rfPred, index=current_x.columns, columns=[index[fcst_step]]
            ).transpose()
            forecast = pd.concat([forecast, pred_clean])
            if self.probabilistic:
                rfPred_upper = self.model_upper.predict(x_dat)
                pred_upper = pd.DataFrame(
                    rfPred_upper, index=current_x.columns, columns=[index[fcst_step]]
                ).transpose()
                rfPred_lower = self.model_lower.predict(x_dat)
                pred_lower = pd.DataFrame(
                    rfPred_lower, index=current_x.columns, columns=[index[fcst_step]]
                ).transpose()
                upper_forecast = pd.concat([upper_forecast, pred_upper])
                lower_forecast = pd.concat([lower_forecast, pred_lower])
            current_x = pd.concat(
                [
                    current_x,
                    pred_clean,
                ]
            )

        forecast = forecast[self.column_names]
        if not self.probabilistic:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.sktraindata,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )
        upper_forecast = upper_forecast[self.column_names]
        lower_forecast = lower_forecast[self.column_names]

        if just_point_forecast:
            return forecast
        else:
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
        if method == "deep":
            model_choice = generate_regressor_params(
                model_dict=sklearn_model_dict, method=method
            )
            window_choice = random.choices(
                [None, 3, 7, 10, 14, 28], [0.2, 0.2, 0.05, 0.05, 0.05, 0.05]
            )[0]
            probabilistic = random.choices([True, False], [0.2, 0.8])[0]
        else:
            model_choice = generate_regressor_params(
                model_dict=multivariate_model_dict, method=method
            )
            window_choice = random.choices([None, 3, 7, 10], [0.2, 0.2, 0.05, 0.05])[0]
            probabilistic = False
        mean_rolling_periods_choice = random.choices(
            [None, 5, 7, 12, 30, 90], [0.3, 0.1, 0.1, 0.1, 0.1, 0.05]
        )[0]
        if mean_rolling_periods_choice is not None:
            macd_periods_choice = seasonal_int(small=True)
            if macd_periods_choice == mean_rolling_periods_choice:
                macd_periods_choice = mean_rolling_periods_choice + 10
        else:
            macd_periods_choice = None
        std_rolling_periods_choice = random.choices(
            [None, 5, 7, 10, 30, 90], [0.3, 0.1, 0.1, 0.1, 0.1, 0.05]
        )[0]
        ewm_var_alpha = random.choices([None, 0.2, 0.5, 0.8], [0.4, 0.1, 0.1, 0.05])[0]
        quantile90_rolling_periods = random.choices(
            [None, 5, 7, 10, 30, 90], [0.3, 0.1, 0.1, 0.1, 0.1, 0.05]
        )[0]
        quantile10_rolling_periods = random.choices(
            [None, 5, 7, 10, 30, 90], [0.3, 0.1, 0.1, 0.1, 0.1, 0.05]
        )[0]
        max_rolling_periods_choice = random.choices(
            [None, seasonal_int(small=True)], [0.2, 0.5]
        )[0]
        min_rolling_periods_choice = random.choices(
            [None, seasonal_int(small=True)], [0.2, 0.5]
        )[0]
        lag_periods_choice = None
        ewm_choice = random.choices(
            [None, 0.1, 0.2, 0.5, 0.8], [0.4, 0.01, 0.1, 0.1, 0.05]
        )[0]
        abs_energy_choice = False
        rolling_autocorr_periods_choice = random.choices(
            [None, 2, 7, 12, 30], [0.4, 0.01, 0.01, 0.01, 0.01]
        )[0]
        add_date_part_choice = random.choices(
            [None, 'simple', 'expanded', 'recurring', "simple_2", "simple_2_poly"],
            [0.5, 0.05, 0.1, 0.1, 0.05, 0.1],
        )[0]
        holiday_choice = random.choices([True, False], [0.1, 0.9])[0]
        polynomial_degree_choice = random.choices([None, 2], [0.995, 0.005])[0]
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices([None, 'User'], [0.7, 0.3])[0]
        parameter_dict = {
            'regression_model': model_choice,
            'mean_rolling_periods': mean_rolling_periods_choice,
            'macd_periods': macd_periods_choice,
            'std_rolling_periods': std_rolling_periods_choice,
            'max_rolling_periods': max_rolling_periods_choice,
            'min_rolling_periods': min_rolling_periods_choice,
            "quantile90_rolling_periods": quantile90_rolling_periods,
            "quantile10_rolling_periods": quantile10_rolling_periods,
            'ewm_alpha': ewm_choice,
            "ewm_var_alpha": ewm_var_alpha,
            'additional_lag_periods': lag_periods_choice,
            'abs_energy': abs_energy_choice,
            'rolling_autocorr_periods': rolling_autocorr_periods_choice,
            'datepart_method': add_date_part_choice,
            'polynomial_degree': polynomial_degree_choice,
            'regression_type': regression_choice,
            'window': window_choice,
            'holiday': holiday_choice,
            "probabilistic": probabilistic,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'regression_model': self.regression_model,
            'mean_rolling_periods': self.mean_rolling_periods,
            'macd_periods': self.macd_periods,
            'std_rolling_periods': self.std_rolling_periods,
            'max_rolling_periods': self.max_rolling_periods,
            'min_rolling_periods': self.min_rolling_periods,
            "quantile90_rolling_periods": self.quantile90_rolling_periods,
            "quantile10_rolling_periods": self.quantile10_rolling_periods,
            'ewm_alpha': self.ewm_alpha,
            "ewm_var_alpha": self.ewm_var_alpha,
            'additional_lag_periods': self.additional_lag_periods,
            'abs_energy': self.abs_energy,
            'rolling_autocorr_periods': self.rolling_autocorr_periods,
            'datepart_method': self.datepart_method,
            'polynomial_degree': self.polynomial_degree,
            'regression_type': self.regression_type,
            'window': self.window,
            'holiday': self.holiday,
            'probabilistic': self.probabilistic,
        }
        return parameter_dict
