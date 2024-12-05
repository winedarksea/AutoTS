"""
Sklearn dependent models

Decision Tree, Elastic Net,  Random Forest, MLPRegressor, KNN, Adaboost
"""

import hashlib
import datetime
import random
import numpy as np
import pandas as pd

# because this attempts to make sklearn optional for overall usage
try:
    from sklearn import config_context
    from sklearn.multioutput import MultiOutputRegressor, RegressorChain
    from sklearn.linear_model import (
        ElasticNet,
        MultiTaskElasticNet,
        LinearRegression,
        Ridge,
    )
    from sklearn.tree import (
        DecisionTreeRegressor,
        DecisionTreeClassifier,
        ExtraTreeRegressor,
    )
except Exception:
    pass
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.tools.seasonal import date_part, seasonal_int, random_datepart
from autots.tools.window_functions import window_maker, last_window, sliding_window_view
from autots.tools.cointegration import coint_johansen, btcd_decompose
from autots.tools.holiday import holiday_flag
from autots.tools.shaping import infer_frequency

# scipy is technically optional but most likely is present
try:
    from scipy.stats import norm
except Exception:

    class norm(object):
        @staticmethod
        def ppf(x):
            return 1.95996398454

        # norm.ppf((1 + 0.95) / 2)


# for numba engine more is required, optional
try:
    import numba  # noqa

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Check if the pandas version is 1.3 or greater and if numba is installed
if pd.__version__ >= '1.3' and NUMBA_AVAILABLE:
    engine = 'numba'
else:
    engine = None

try:
    from joblib import Parallel, delayed

    joblib_present = True
except Exception:
    joblib_present = False


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
    nonzero_last_n: int = None,
    add_date_part: str = None,
    holiday: bool = False,
    holiday_country: str = 'US',
    polynomial_degree: int = None,
    window: int = None,
    cointegration: str = None,
    cointegration_lag: int = 1,
):
    """
    Generate more features from initial time series.

    macd_periods ignored if mean_rolling is None.

    Returns a dataframe of statistical features. Will need to be shifted by 1 or more to match Y for forecast.
    so for the index date of the output here, this represents the time of the prediction being made, NOT the target datetime.
    the datepart components should then represent the NEXT period ahead, which ARE the target datetime
    """
    # making this all or partially Numpy (if possible) would probably be faster
    local_df = df.copy()
    inferred_freq = infer_frequency(local_df.index)
    local_df.columns = [str(x) for x in range(len(df.columns))]
    X = [local_df.rename(columns=lambda x: "lastvalue_" + x)]
    # unlike the others, this pulls the entire window, not just one lag
    if str(window).isdigit():
        # we already have lag 1 using this
        for curr_shift in range(1, window):
            X.append(
                local_df.shift(curr_shift).rename(
                    columns=lambda x: "window_" + str(curr_shift) + "_" + x
                )
            )  # backfill should fill last values safely
    if str(mean_rolling_periods).isdigit():
        temp = local_df.rolling(int(mean_rolling_periods), min_periods=1).median(
            engine=engine
        )
        X.append(temp)
        if str(macd_periods).isdigit():
            # says mean, but median because it's been that way for ages
            temp = (
                local_df.rolling(int(macd_periods), min_periods=1).median(engine=engine)
                - temp
            )
            temp.columns = ['macd' for col in temp.columns]
            X.append(temp)
    if isinstance(mean_rolling_periods, list):
        for mrp in mean_rolling_periods:
            if isinstance(mrp, (tuple, list)):
                lag = mrp[0]
                mean_roll = mrp[1]
                temp = (
                    local_df.shift(lag)
                    .rolling(int(mean_roll), min_periods=1)
                    .mean(engine=engine)
                    .bfill()
                )
                temp.columns = [
                    f'rollingmean_{lag}_{mean_roll}_' + str(col) for col in temp.columns
                ]
            else:
                temp = local_df.rolling(int(mrp), min_periods=1).mean(engine=engine)
                temp.columns = ['rollingmean_' + str(col) for col in temp.columns]
            X.append(temp)
            if str(macd_periods).isdigit():
                temp = (
                    local_df.rolling(int(macd_periods), min_periods=1).mean(
                        engine=engine
                    )
                    - temp
                )
                temp.columns = ['macd' for col in temp.columns]
                X.append(temp)
    if str(std_rolling_periods).isdigit():
        X.append(local_df.rolling(std_rolling_periods, min_periods=1).std())
    if str(max_rolling_periods).isdigit():
        X.append(local_df.rolling(max_rolling_periods, min_periods=1).max())
    if str(min_rolling_periods).isdigit():
        X.append(local_df.rolling(min_rolling_periods, min_periods=1).min())
    if str(quantile90_rolling_periods).isdigit():
        X.append(
            local_df.rolling(quantile90_rolling_periods, min_periods=1).quantile(0.9)
        )
    if str(quantile10_rolling_periods).isdigit():
        X.append(
            local_df.rolling(quantile10_rolling_periods, min_periods=1).quantile(0.1)
        )
    if str(ewm_alpha).replace('.', '').isdigit():
        ewm_df = local_df.ewm(alpha=ewm_alpha, ignore_na=True, min_periods=1).mean()
        ewm_df.columns = ["ewm_alpha" for col in local_df.columns]
        X.append(ewm_df)
    if str(ewm_var_alpha).replace('.', '').isdigit():
        X.append(local_df.ewm(alpha=ewm_var_alpha, ignore_na=True, min_periods=1).var())
    if str(additional_lag_periods).isdigit():
        X.append(local_df.shift(additional_lag_periods))
    if nonzero_last_n is not None:
        full_index = local_df.index.union(
            local_df.index.shift(-nonzero_last_n, freq=inferred_freq)
        )
        X.append(
            (local_df.reindex(full_index).bfill() != 0)
            .rolling(nonzero_last_n, min_periods=1)
            .sum()
            .reindex(local_df.index)
        )
    if cointegration is not None:
        if str(cointegration).lower() == "btcd":
            X.append(
                pd.DataFrame(
                    np.matmul(
                        btcd_decompose(
                            local_df.values,
                            retrieve_regressor(
                                regression_model={
                                    "model": 'LinearRegression',
                                    "model_params": {},
                                },
                                verbose=0,
                                verbose_bool=False,
                                random_seed=2020,
                                multioutput=False,
                            ),
                            max_lag=cointegration_lag,
                        ),
                        (local_df.values).T,
                    ).T,
                    index=local_df.index,
                )
            )
        else:
            X.append(
                pd.DataFrame(
                    np.matmul(
                        coint_johansen(local_df.values, k_ar_diff=cointegration_lag),
                        (local_df.values).T,
                    ).T,
                    index=local_df.index,
                )
            )
    if abs_energy:
        X.append(local_df.pow(other=([2] * len(local_df.columns))).cumsum())
    if str(rolling_autocorr_periods).isdigit():
        temp = local_df.rolling(rolling_autocorr_periods).apply(
            lambda x: x.autocorr(), raw=False
        )
        temp.columns = ['rollautocorr' for col in temp.columns]
        X.append(temp)
    if add_date_part not in [None, "None", "none"]:
        ahead_index = local_df.index.shift(1, freq=inferred_freq)
        date_part_df = date_part(ahead_index, method=add_date_part)
        date_part_df.index = local_df.index
        X.append(date_part_df)
    X = pd.concat(X, axis=1)

    if holiday:
        ahead_index = local_df.index.shift(1, freq=inferred_freq)
        ahead_2_index = local_df.index.shift(2, freq=inferred_freq)
        full_index = ahead_index.union(ahead_2_index)
        hldflag = holiday_flag(full_index, country=holiday_country)
        X['holiday_flag_'] = hldflag.reindex(ahead_index).to_numpy()
        X['holiday_flag_future_'] = hldflag.reindex(ahead_2_index).to_numpy()

    # X = X.replace([np.inf, -np.inf], np.nan)
    X = X.bfill()

    if str(polynomial_degree).isdigit():
        polynomial_degree = abs(int(polynomial_degree))
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(polynomial_degree)
        X = pd.DataFrame(poly.fit_transform(X))

    # rename to remove duplicates but still keep names if present
    X.columns = [
        m + "_" + str(n)
        for m, n in zip([str(x) for x in range(len(X.columns))], X.columns.tolist())
    ]

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
    nonzero_last_n: int = None,
    add_date_part: str = None,
    holiday: bool = False,
    holiday_country: str = 'US',
    polynomial_degree: int = None,
    window: int = None,
    future_regressor=None,
    regressor_per_series=None,
    static_regressor=None,
    cointegration: str = None,
    cointegration_lag: int = 1,
    series_id=None,
    slice_index=None,
    series_id_to_multiindex=None,
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
        nonzero_last_n=nonzero_last_n,
        add_date_part=add_date_part,
        holiday=holiday,
        holiday_country=holiday_country,
        polynomial_degree=polynomial_degree,
        window=window,
        cointegration=cointegration,
        cointegration_lag=cointegration_lag,
    )
    if future_regressor is not None:
        X = pd.concat([X, future_regressor], axis=1)
    if regressor_per_series is not None:
        # this is actually wrong, merging on an index value that is off by one
        X = X.merge(
            regressor_per_series, left_index=True, right_index=True, how='left'
        ).bfill()
    if static_regressor is not None:
        X['series_id'] = df.columns[0]
        X = X.merge(static_regressor, left_on="series_id", right_index=True, how='left')
        X = X.drop(columns=['series_id'])
    if slice_index is not None:
        X = X[X.index.isin(slice_index)]
    if series_id_to_multiindex is not None:
        X["series_id"] = str(series_id_to_multiindex)
        X = X.set_index("series_id", append=True)
    if series_id is not None:
        hashed = (
            int(hashlib.sha256(str(series_id).encode('utf-8')).hexdigest(), 16) % 10**16
        )
        X['series_id'] = hashed
    return X


def retrieve_regressor(
    regression_model: dict = {
        "model": 'RandomForest',
        "model_params": {
            'n_estimators': 300,
            'min_samples_leaf': 1,
            'bootstrap': False,
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
            regr = MultiTaskElasticNet(
                alpha=1.0, random_state=random_seed, **model_param_dict
            )
        else:
            regr = ElasticNet(alpha=1.0, random_state=random_seed, **model_param_dict)
        return regr
    elif model_class == 'DecisionTree':
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
            return MultiOutputRegressor(
                LGBMRegressor(
                    verbose=-1,
                    random_state=random_seed,
                    n_jobs=1,
                    **model_param_dict,
                ),
                n_jobs=n_jobs,
            )
        else:
            return LGBMRegressor(
                verbose=-1,
                random_state=random_seed,
                n_jobs=n_jobs,
                **model_param_dict,
            )
    elif model_class == "LightGBMRegressorChain":
        from lightgbm import LGBMRegressor

        regr = LGBMRegressor(
            verbose=-1,
            random_state=random_seed,
            n_jobs=n_jobs,
            **model_param_dict,
        )
        if multioutput:
            return RegressorChain(regr)
        else:
            return regr
    elif model_class == 'Adaboost':
        from sklearn.ensemble import AdaBoostRegressor

        if regression_model["model_params"]['estimator'] == 'SVR':
            from sklearn.svm import LinearSVR

            svc = LinearSVR(verbose=0, random_state=random_seed, max_iter=1500)
            regr = AdaBoostRegressor(
                estimator=svc,
                n_estimators=regression_model["model_params"]['n_estimators'],
                loss=regression_model["model_params"]['loss'],
                learning_rate=regression_model["model_params"]['learning_rate'],
                random_state=random_seed,
            )
        elif regression_model["model_params"]['estimator'] == 'LinReg':
            linreg = LinearRegression()
            regr = AdaBoostRegressor(
                estimator=linreg,
                n_estimators=regression_model["model_params"]['n_estimators'],
                loss=regression_model["model_params"]['loss'],
                learning_rate=regression_model["model_params"]['learning_rate'],
                random_state=random_seed,
            )
        elif regression_model["model_params"]['estimator'] == 'ElasticNet':
            linreg = ElasticNet()
            regr = AdaBoostRegressor(
                estimator=linreg,
                n_estimators=regression_model["model_params"]['n_estimators'],
                loss=regression_model["model_params"]['loss'],
                learning_rate=regression_model["model_params"]['learning_rate'],
                random_state=random_seed,
            )
        elif regression_model["model_params"]['estimator'] == 'ExtraTree':
            linreg = ExtraTreeRegressor(
                max_depth=regression_model["model_params"].get("max_depth", 3)
            )
            regr = AdaBoostRegressor(
                estimator=linreg,
                n_estimators=regression_model["model_params"]['n_estimators'],
                loss=regression_model["model_params"]['loss'],
                learning_rate=regression_model["model_params"]['learning_rate'],
                random_state=random_seed,
            )
        elif regression_model["model_params"].get("max_depth", None) is not None:
            linreg = DecisionTreeRegressor(
                max_depth=regression_model["model_params"].get("max_depth")
            )
            regr = AdaBoostRegressor(
                estimator=linreg,
                n_estimators=regression_model["model_params"]['n_estimators'],
                loss=regression_model["model_params"]['loss'],
                learning_rate=regression_model["model_params"]['learning_rate'],
                random_state=random_seed,
            )
        else:
            regr = AdaBoostRegressor(random_state=random_seed, **model_param_dict)
        if multioutput:
            return MultiOutputRegressor(regr, n_jobs=n_jobs)
        else:
            return regr
    elif model_class in ['xgboost', 'XGBRegressor']:
        import xgboost as xgb

        smaller_n_jobs = int(n_jobs / 2) if n_jobs > 3 else n_jobs

        if False:  # this is no longer necessary in 1.6 and beyond
            regr = MultiOutputRegressor(
                xgb.XGBRegressor(verbosity=0, **model_param_dict, n_jobs=1),
                n_jobs=smaller_n_jobs,
            )
        else:
            regr = xgb.XGBRegressor(
                verbosity=0, **model_param_dict, n_jobs=smaller_n_jobs
            )
        return regr
    elif model_class in ['SVM', "LinearSVR"]:
        from sklearn.svm import LinearSVR

        if multioutput:
            regr = MultiOutputRegressor(
                LinearSVR(verbose=verbose_bool, **model_param_dict),
                n_jobs=n_jobs,
            )
        else:
            regr = LinearSVR(verbose=verbose_bool, **model_param_dict)
        return regr
    elif model_class == 'Ridge':
        return Ridge(random_state=random_seed, **model_param_dict)
    elif model_class == "FastRidge":
        return Ridge(alpha=1e-9, solver="cholesky", fit_intercept=False, copy_X=False)
    elif model_class == 'BayesianRidge':
        from sklearn.linear_model import BayesianRidge

        regr = BayesianRidge(**model_param_dict)
        if multioutput:
            return RegressorChain(regr)
        else:
            return regr
    elif model_class == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_jobs=n_jobs, random_state=random_seed, **model_param_dict
        )
    elif model_class in [
        "RadiusNeighbors",
        "RadiusNeighbors",
        "RadiusRegressor",
        "RadiusNeighborsRegressor",
    ]:
        from sklearn.neighbors import RadiusNeighborsRegressor

        regr = RadiusNeighborsRegressor(n_jobs=n_jobs, **model_param_dict)
        return regr
    elif model_class == "PoissonRegresssion":
        from sklearn.linear_model import PoissonRegressor

        if multioutput:
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
    elif model_class == "LinearRegression":
        return LinearRegression(**model_param_dict)
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
    elif model_class in ["MultioutputGPR", "VectorizedMultiOutputGPR"]:
        return VectorizedMultiOutputGPR(**model_param_dict)
    elif model_class in ['RandomForest', 'random_forest', 'randomforest']:
        regression_model['model'] = 'RandomForest'
        from sklearn.ensemble import RandomForestRegressor

        regr = RandomForestRegressor(
            random_state=random_seed,
            verbose=verbose_bool,
            n_jobs=n_jobs,
            **model_param_dict,
        )
        return regr
    elif model_class in ["ElasticNetwork"]:
        from autots.models.dnn import ElasticNetwork

        return ElasticNetwork(
            random_seed=random_seed, verbose=verbose, **model_param_dict
        )
    else:
        raise ValueError(f"model_class {model_class} regressor not recognized")


def retrieve_classifier(
    regression_model: dict = {
        "model": 'RandomForest',
        "model_params": {
            'n_estimators': 300,
            'min_samples_leaf': 1,
            'bootstrap': False,
        },
    },
    verbose: int = 0,
    verbose_bool: bool = False,
    random_seed: int = 2020,
    n_jobs: int = 1,
    multioutput: bool = True,
):
    """Convert a model param dict to model object for regression frameworks."""
    model_class = regression_model.get('model', 'RandomForest')
    model_param_dict = regression_model.get("model_params", {})
    if model_class == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_jobs=n_jobs, random_state=random_seed, **model_param_dict
        )
    elif model_class == 'DecisionTree':
        return DecisionTreeClassifier(random_state=random_seed, **model_param_dict)
    elif model_class in ['xgboost', 'XGBClassifier']:
        import xgboost as xgb

        return xgb.XGBClassifier(verbosity=verbose, **model_param_dict, n_jobs=n_jobs)
    elif model_class == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            random_state=random_seed,
            verbose=verbose_bool,
            n_jobs=n_jobs,
            **model_param_dict,
        )
    elif model_class == "KNN":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(
            n_jobs=n_jobs,
            **model_param_dict,
        )
    elif model_class == "SGD":
        from sklearn.linear_model import SGDClassifier
        from sklearn.multioutput import MultiOutputClassifier

        if multioutput:
            return MultiOutputClassifier(
                SGDClassifier(
                    random_state=random_seed,
                    verbose=verbose_bool,
                    n_jobs=n_jobs,
                    **model_param_dict,
                )
            )
        else:
            return SGDClassifier(
                random_state=random_seed,
                verbose=verbose_bool,
                n_jobs=n_jobs,
                **model_param_dict,
            )
    elif model_class == "GaussianNB":
        from sklearn.naive_bayes import GaussianNB

        if multioutput:
            return MultiOutputClassifier(GaussianNB(**model_param_dict))
        else:
            return GaussianNB(**model_param_dict)
    else:
        raise ValueError(f"classifier {model_class} not a recognized option.")


# models that can more quickly handle many X/Y obs, with modest number of features
sklearn_model_dict = {
    # 'RandomForest': 0.02,  # crashes sometimes at scale for unclear reasons
    'ElasticNet': 0.05,
    'MLP': 0.05,
    'DecisionTree': 0.02,
    'KNN': 0.05,
    'Adaboost': 0.01,
    'SVM': 0.02,  # was slow, LinearSVR seems much faster
    'BayesianRidge': 0.05,
    'xgboost': 0.05,
    'KerasRNN': 0.001,  # slow at scale
    'Transformer': 0.001,
    'HistGradientBoost': 0.03,
    'LightGBM': 0.1,
    'LightGBMRegressorChain': 0.03,
    'ExtraTrees': 0.01,
    'RadiusNeighbors': 0.02,
    'PoissonRegresssion': 0.03,
    'RANSAC': 0.05,
    'Ridge': 0.02,
    'GaussianProcessRegressor': 0.000000001,  # slow
    "ElasticNetwork": 0.01,
    # 'MultioutputGPR': 0.0000001,  # memory intensive kernel killing
}
multivariate_model_dict = {
    'RandomForest': 0.02,
    # 'ElasticNet': 0.05,
    'MLP': 0.03,
    'DecisionTree': 0.05,
    'KNN': 0.05,
    'Adaboost': 0.03,
    'SVM': 0.01,
    # 'BayesianRidge': 0.05,
    'xgboost': 0.09,
    # 'KerasRNN': 0.01,  # too slow on big data
    'HistGradientBoost': 0.03,
    'LightGBM': 0.09,
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
    'SVM': 0.02,
    'BayesianRidge': 0.03,
    'HistGradientBoost': 0.02,
    'LightGBM': 0.03,
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
    'SVM': 0.02,
    'KerasRNN': 0.02,
    'LightGBM': 0.09,
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
    'SVM': 0.01,
    'xgboost': 0.1,
    'LightGBM': 0.1,
    'HistGradientBoost': 0.1,
    'PoissonRegresssion': 0.05,
}
# these are models that are relatively fast with large multioutput Y, small n obs
datepart_model_dict: dict = {
    'ElasticNet': 0.1,
    'MLP': 0.05,
    'DecisionTree': 0.03,
    'Adaboost': 0.05,
    'SVM': 0.0001,
    'KerasRNN': 0.01,
    # 'Transformer': 0.02,  # slow, kernel failed
    'RadiusNeighbors': 0.03,  # vulnerable on short data to returning NaN but effective sometimes
    "ElasticNetwork": 0.05,
}
datepart_model_dict_deep = {
    'RandomForest': 0.05,  # crashes sometimes at scale for unclear reasons
    'ElasticNet': 0.1,
    'xgboost': 0.05,
    'MLP': 0.05,
    'DecisionTree': 0.02,
    'Adaboost': 0.05,
    'SVM': 0.01,
    'KerasRNN': 0.02,
    'Transformer': 0.02,  # slow
    'ExtraTrees': 0.01,  # some params cause RAM crash?
    'RadiusNeighbors': 0.1,
    'MultioutputGPR': 0.001,
    "ElasticNetwork": 0.05,
}
gpu = ['Transformer', 'KerasRNN', 'MLP', "ElasticNetwork"]  # or more accurately, no dnn
gradient_boosting = {
    'xgboost': 0.09,
    'HistGradientBoost': 0.03,
    'LightGBM': 0.09,
    'LightGBMRegressorChain': 0.03,
}
# all tree based models
tree_dict = {
    'DecisionTree': 1,
    'RandomForest': 1,
    'ExtraTrees': 1,
    'LightGBM': 1,
    'HistGradientBoost': 1,
    'xgboost': 1,
    'XGBRegressor': 1,
}
# pre-optimized model templates
xgparam3 = {
    "model": 'xgboost',
    "model_params": {
        "booster": 'gbtree',
        "colsample_bylevel": 0.54,
        "learning_rate": 0.0125,
        "max_depth": 11,
        "min_child_weight": 0.0127203,
        "n_estimators": 319,
    },
}
xgparam1 = {
    "model": 'xgboost',
    "model_params": {
        'n_estimators': 7,
        'max_leaves': 4,
        'min_child_weight': 2.5,
        'learning_rate': 0.35,
        'subsample': 0.95,
        'colsample_bylevel': 0.56,
        'colsample_bytree': 0.46,
        'reg_alpha': 0.0016,
        'reg_lambda': 5.3,
    },
}
xgparam2 = {
    "model": 'xgboost',
    "model_params": {
        "base_score": 0.5,
        "booster": 'gbtree',
        "colsample_bylevel": 0.692,
        "learning_rate": 0.022,
        "max_bin": 256,
        "max_depth": 14,
        "max_leaves": 0,
        "min_child_weight": 0.024,
        "n_estimators": 162,
    },
}
lightgbmp1 = {
    "model": 'LightGBM',
    "model_params": {
        "colsample_bytree": 0.1645,
        "learning_rate": 0.0203,
        "max_bin": 1023,
        "min_child_samples": 16,
        "n_estimators": 1794,
        "num_leaves": 15,
        "reg_alpha": 0.00098,
        "reg_lambda": 0.686,
    },
}
lightgbmp2 = {
    "model": 'LightGBM',
    "model_params": {
        "colsample_bytree": 0.947,
        "learning_rate": 0.7024,
        "max_bin": 255,
        "min_child_samples": 15,
        "n_estimators": 5,
        "num_leaves": 35,
        "reg_alpha": 0.00308,
        "reg_lambda": 5.182,
    },
}


def generate_classifier_params(
    model_dict=None,
    method="default",
):
    if model_dict is None:
        if method == "fast":
            model_dict = {
                'xgboost': 0.5,  # also crashes sometimes
                # 'ExtraTrees': 0.2,  # crashes sometimes
                # 'RandomForest': 0.1,
                'KNN': 1,
                'SGD': 0.1,
            }
        else:
            model_dict = {
                'xgboost': 0.5,
                'ExtraTrees': 0.2,
                'RandomForest': 0.1,
                'KNN': 1,
                'SGD': 0.1,
            }
    regr_params = generate_regressor_params(
        model_dict=model_dict,
        method=method,
    )
    if regr_params["model"] == 'xgboost':
        if "objective" in regr_params['model_params'].keys():
            regr_params['model_params'].pop('objective', None)
    elif regr_params["model"] == 'ExtraTrees':
        regr_params['model_params']['criterion'] = 'gini'
    return regr_params


def generate_regressor_params(
    model_dict=None,
    method="default",
):
    """Generate new parameters for input to regressor."""
    # force neural networks for testing purposes
    if method in ["default", 'random', 'fast']:
        pass
    elif method == "neuralnets":
        model_dict = {
            'KerasRNN': 0.05,
            'Transformer': 0.05,
            'MLP': 0.05,
            "ElasticNetwork": 0.05,
        }
        method = "deep"
    elif method == "gradient_boosting":
        model_dict = gradient_boosting
        method = "default"
    elif method == "trees":
        model_dict = tree_dict
        method = "default"
    elif method in sklearn_model_dict.keys():
        model_dict = {method: sklearn_model_dict[method]}
    elif model_dict is None:
        model_dict = sklearn_model_dict
    # used in Cassandra to remove slowest models
    if method == "no_gpu":
        model_dict = {x: y for (x, y) in model_dict.items() if x not in gpu}
    model_list = list(model_dict.keys())
    model = random.choices(model_list, list(model_dict.values()), k=1)[0]
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
        'MultioutputGPR',
        'SVM',
        "ElasticNetwork",
        "ElasticNet",
        "RadiusNeighbors",
    ]:
        if model == 'Adaboost':
            param_dict = {
                "model": 'Adaboost',
                "model_params": {
                    "n_estimators": random.choices(
                        [50, 100, 200, 500], [0.7, 0.2, 0.15, 0.1]
                    )[0],
                    "loss": random.choices(
                        ['linear', 'square', 'exponential'], [0.8, 0.01, 0.1]
                    )[0],
                    "estimator": random.choices(
                        [None, 'LinReg', 'SVR', 'ElasticNet', 'ExtraTree'],
                        [0.8, 0.1, 0.0, 0.1, 0.1],  # SVR slow and crash prone
                    )[0],
                    "learning_rate": random.choices(
                        [1, 0.5, 0.25, 0.8, 100], [0.9, 0.1, 0.1, 0.1, 0.025]
                    )[0],
                },
            }
            if param_dict["model_params"]["estimator"] in [None, "ExtraTree"]:
                param_dict["model_params"]["max_depth"] = random.choices(
                    [2, 3, 4, 5],
                    [0.2, 0.8, 0.1, 0.01],
                )[0]
        elif model == 'ElasticNet':
            param_dict = {
                "model": 'ElasticNet',
                "model_params": {
                    "l1_ratio": random.choices([0.5, 0.1, 0.9], [0.7, 0.2, 0.1])[0],
                    "fit_intercept": random.choices([True, False], [0.9, 0.1])[0],
                    "selection": random.choices(["cyclic", "random"], [0.8, 0.1])[0],
                    "max_iter": random.choices([1000, 2000, 5000], [0.8, 0.2, 0.01])[0],
                },
            }
        elif model == 'xgboost':
            branch = random.choices(['p1', 'p2', 'p3', 'random'], [0.1, 0.1, 0.1, 0.7])[
                0
            ]
            if branch == 'p1':
                param_dict = xgparam1
            elif branch == 'p2':
                param_dict = xgparam2
            elif branch == 'p3':
                param_dict = xgparam3
            else:
                objective = random.choices(
                    [
                        'count:poisson',
                        'reg:squarederror',
                        'reg:gamma',
                        'reg:pseudohubererror',
                        'reg:quantileerror',
                    ],
                    [0.1, 0.6, 0.1, 0.1, 0.1],
                )[0]
                param_dict = {
                    "model": 'xgboost',
                    "model_params": {
                        "booster": random.choices(['gbtree', 'gblinear'], [0.7, 0.3])[
                            0
                        ],
                        "objective": objective,
                        "max_depth": random.choices(
                            [6, 3, 2, 8], [0.6, 0.4, 0.2, 0.01]
                        )[0],
                        "eta": random.choices(
                            [1.0, 0.3, 0.01, 0.03, 0.05, 0.003],
                            [0.05, 0.1, 0.1, 0.1, 0.1, 0.1],
                        )[
                            0
                        ],  # aka learning_rate
                        "min_child_weight": random.choices(
                            [0.05, 0.5, 1, 2, 5, 10], [0.01, 0.05, 0.8, 0.1, 0.1, 0.1]
                        )[0],
                        "subsample": random.choices(
                            [1, 0.9, 0.7, 0.5], [0.9, 0.05, 0.05, 0.05]
                        )[0],
                        "colsample_bylevel": random.choices(
                            [1, 0.9, 0.7, 0.5], [0.4, 0.1, 0.1, 0.1]
                        )[0],
                        "reg_alpha": random.choices(
                            [0, 0.001, 0.05, 100], [0.9, 0.1, 0.05, 0.05]
                        )[0],
                        "reg_lambda": random.choices(
                            [1, 0.03, 0.11, 0.2, 5], [0.9, 0.05, 0.05, 0.05, 0.05]
                        )[0],
                    },
                }
                if random.choices([True, False], [0.4, 0.6])[0]:
                    param_dict["model_params"]["max_depth"] = random.choices(
                        [3, 6, 9], [0.1, 0.8, 0.1]
                    )[0]
                if random.choices([True, False], [0.5, 0.5])[0]:
                    param_dict["model_params"]["n_estimators"] = random.choices(
                        [4, 7, 10, 20, 100, 1000],
                        [0.2, 0.2, 0.2, 0.2, 0.5, 0.2],
                    )[0]
                if random.choices([True, False], [0.2, 0.8])[0]:
                    param_dict["model_params"]["grow_policy"] = "lossguide"
                if objective == "reg:quantileerror":
                    param_dict['model_params']["quantile_alpha"] = 0.5
                    param_dict['model_params']["tree_method"] = "hist"
                elif random.choices([True, False], [0.2, 0.8])[0]:
                    # new in 2.0 vector trees
                    param_dict['model_params']["multi_strategy"] = "multi_output_tree"
                    param_dict['model_params']["tree_method"] = "hist"
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
                            (2560,),
                            (25, 15, 25),
                            (72, 36, 72),
                            (25, 50, 25),
                            (32, 64, 32),
                            (32, 32, 32),
                        ],
                        [0.1, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1],
                    )[0],
                    "max_iter": random.choices(
                        [250, 500, 1000],
                        [0.8, 0.1, 0.1],
                    )[0],
                    "activation": random.choices(
                        ['identity', 'logistic', 'tanh', 'relu'],
                        [0.05, 0.05, 0.6, 0.3],
                    )[0],
                    "solver": solver,
                    "early_stopping": early_stopping,
                    "learning_rate_init": learning_rate_init,
                    "alpha": random.choices(
                        [None, 0.0001, 0.1, 0.0], [0.5, 0.2, 0.2, 0.2]
                    )[0],
                },
            }
        elif model == 'KNN':
            param_dict = {
                "model": 'KNN',
                "model_params": {
                    "n_neighbors": random.choices([3, 5, 10, 14], [0.2, 0.7, 0.1, 0.1])[
                        0
                    ],
                    "weights": random.choices(['uniform', 'distance'], [0.999, 0.001])[
                        0
                    ],
                    'p': random.choices([2, 1, 1.5], [0.7, 0.1, 0.1])[0],
                    'leaf_size': random.choices([30, 10, 50], [0.8, 0.1, 0.1])[0],
                },
            }
        elif model == 'RandomForest':
            if method == "fast":
                n_estimators = random.choices([4, 300, 100], [0.2, 0.4, 0.4])[0]
                min_samples_leaf = random.choices([2, 4, 1], [0.2, 0.2, 0.2])[0]
            else:
                n_estimators = random.choices(
                    [4, 300, 100, 1000, 5000], [0.1, 0.4, 0.4, 0.2, 0.01]
                )[0]
                min_samples_leaf = random.choices([2, 4, 1], [0.2, 0.2, 0.8])[0]
            param_dict = {
                "model": 'RandomForest',
                "model_params": {
                    "n_estimators": n_estimators,
                    "min_samples_leaf": min_samples_leaf,
                    "bootstrap": random.choices([True, False], [0.9, 0.1])[0],
                },
            }
        elif model == 'ExtraTrees':
            max_depth_choice = random.choices(
                [None, 5, 10, 20, 30], [0.4, 0.1, 0.3, 0.4, 0.1]
            )[0]
            estimators_choice = random.choices(
                [4, 50, 100, 500], [0.05, 0.05, 0.9, 0.05]
            )[0]
            param_dict = {
                "model": 'ExtraTrees',
                "model_params": {
                    "n_estimators": estimators_choice,
                    "min_samples_leaf": random.choices([2, 4, 1], [0.1, 0.1, 0.8])[0],
                    "min_samples_split": random.choices([2, 4, 1.0], [0.8, 0.1, 0.1])[
                        0
                    ],
                    "max_depth": max_depth_choice,
                    "criterion": random.choices(
                        ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                        [0.25, 0.0, 0.25, 0.1],  # abs error very slow
                    )[0],
                    "max_features": random.choices([1, 0.6, 0.3], [0.8, 0.1, 0.1])[0],
                },
            }
        elif model in ['KerasRNN']:
            init_list = [
                'glorot_uniform',
                'lecun_uniform',
                # 'glorot_normal',  # evidence it is slow sometimes
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
                        [
                            'mae',
                            'Huber',
                            'poisson',
                            'mse',
                            'mape',
                            "mean_squared_logarithmic_error",
                        ],
                        [0.2, 0.3, 0.1, 0.2, 0.2, 0.1],
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
        elif model in ["ElasticNetwork"]:
            param_dict = {
                "model": 'ElasticNetwork',
                "model_params": {
                    "size": random.choices(
                        [
                            32,
                            64,
                            128,
                            256,
                            2560,
                        ],
                        [0.1, 0.3, 0.3, 0.1, 0.3],
                    )[0],
                    "l1": random.choices(
                        [0.0, 0.0001, 0.01, 0.02, 0.2], [0.5, 0.3, 0.15, 0.1, 0.1]
                    )[0],
                    "l2": random.choices(
                        [0.0, 0.0001, 0.01, 0.02, 0.2], [0.5, 0.3, 0.15, 0.1, 0.1]
                    )[0],
                    "epochs": random.choices(
                        [10, 20, 50, 100], [0.75, 0.2, 0.05, 0.001]
                    )[0],
                    "batch_size": random.choices([8, 16, 32, 72], [0.2, 0.2, 0.5, 0.1])[
                        0
                    ],
                    "optimizer": random.choices(
                        ['adam', 'rmsprop', 'adagrad'], [0.8, 0.5, 0.1]
                    )[0],
                    "loss": random.choices(
                        [
                            'mae',
                            'Huber',
                            'poisson',
                            'mse',
                            'mape',
                            "mean_squared_logarithmic_error",
                        ],
                        [0.2, 0.3, 0.1, 0.2, 0.2, 0.1],
                    )[0],
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
            branch = random.choices(['p1', 'p2', 'random'], [0.1, 0.1, 0.8])[0]
            if branch == 'p1':
                param_dict = lightgbmp1
            elif branch == 'p2':
                param_dict = lightgbmp2
            else:
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
                                "fair",
                                'mape',
                                ['regression', 'mape'],
                            ],
                            [0.4, 0.2, 0.2, 0.2, 0.2, 0.05, 0.01, 0.05, 0.05, 0.1],
                        )[0],
                        "learning_rate": random.choices(
                            [0.001, 0.1, 0.01, 0.7],
                            [0.1, 0.6, 0.3, 0.2],
                        )[0],
                        "num_leaves": random.choices(
                            [31, 127, 70, 1000, 15, 2048],
                            [0.6, 0.1, 0.3, 0.1, 0.2, 0.1],
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
                        "linear_tree": random.choice([True, False]),
                        "lambda_l1": random.choices(
                            [0.0, 0.1, 1, 10], [0.5, 0.1, 0.1, 0.1]
                        )[0],
                        "lambda_l2": random.choices(
                            [0.0, 0.1, 1, 10], [0.5, 0.1, 0.1, 0.1]
                        )[0],
                        "min_data_in_leaf": random.choices(
                            [5, 15, 20, 30], [0.1, 0.2, 0.6, 0.1]
                        )[0],
                        "feature_fraction": random.choices(
                            [1.0, 0.1, 0.5, 0.8], [0.5, 0.1, 0.1, 0.1]
                        )[0],
                        "max_bin": random.choices(
                            [1500, 1000, 255, 50], [0.1, 0.2, 0.6, 0.1]
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
                    'alpha': random.choice([0.0000000001, 0.00001, 1]),
                    'kernel': random.choices(
                        [None, "DotWhite", "White", "RBF", "ExpSineSquared"],
                        [0.4, 0.1, 0.1, 0.4, 0.1],
                    )[0],
                },
            }
        elif model == "MultioutputGPR":
            kernel = random.choices(
                [
                    'linear',
                    "exponential",
                    "rbf",
                    "polynomial",
                    'periodic',
                    "locally_periodic",
                ],
                [0.1, 0.1, 0.4, 0.2, 0.01, 0.01],
            )[0]
            param_dict = {
                "model": 'MultioutputGPR',
                "model_params": {
                    'noise_var': random.choice([1e-7, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]),
                    'kernel': kernel,
                },
            }
            if kernel in ["exponential", "locally_periodic", "rbf", "periodic"]:
                param_dict['gamma'] = random.choices(
                    [0.1, 1, 10, 100],
                    [0.2, 0.05, 0.2, 0.2],
                )[0]
            if kernel in ["locally_periodic", "periodic"]:
                param_dict['p'] = random.choices(
                    [7, 12, 365.25, 52, 28], [0.8, 0.15, 0.15, 0.05, 0.05]
                )[0]
            if kernel == "locally_periodic":
                param_dict['lambda_prime'] = random.choice([0.1, 1, 10])
        elif model == "SVM":
            # LinearSVR
            param_dict = {
                "model": 'SVM',
                "model_params": {
                    'C': random.choices([1.0, 0.5, 2.0, 0.25], [0.6, 0.1, 0.1, 0.1])[0],
                    'tol': random.choices([1e-4, 1e-3, 1e-5], [0.6, 0.1, 0.1])[0],
                    "loss": random.choice(
                        ['epsilon_insensitive', 'squared_epsilon_insensitive']
                    ),
                    "max_iter": random.choice([500, 1000]),
                },
            }
        elif model == 'RadiusNeighbors':
            radius_choice = random.choices(
                [0.5, 1.0, 2.0, 5.0, 10.0], [0.1, 0.4, 0.4, 0.15, 0.05]
            )[0]
            weights_choice = random.choices(["uniform", "distance"], [0.6, 0.4])[0]
            algorithm_choice = random.choices(
                ["auto", "ball_tree", "kd_tree"], [0.7, 0.1, 0.1]
            )[0]
            leaf_size_choice = random.choices([10, 20, 30, 50], [0.5, 0.3, 0.15, 0.05])[
                0
            ]
            param_dict = {
                "model": 'RadiusNeighbors',
                "model_params": {
                    "radius": radius_choice,
                    "weights": weights_choice,
                    "algorithm": algorithm_choice,
                    "leaf_size": leaf_size_choice,
                    "p": random.choices([1, 1.5, 2], [0.5, 0.1, 0.5])[
                        0
                    ],  # Manhattan (p=1) or Euclidean (p=2)
                    "metric": random.choices(
                        ["minkowski", "manhattan", "euclidean"], [0.7, 0.2, 0.1]
                    )[0],
                },
            }
            return param_dict
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
        nonzero_last_n: int = None,
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
        self.nonzero_last_n = nonzero_last_n
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
        self.sktraindata = self.sktraindata.ffill().bfill()
        self.Y = self.sktraindata.drop(self.sktraindata.head(2).index)
        self.Y.columns = [x for x in range(len(self.Y.columns))]
        self.X = rolling_x_regressor(
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
            nonzero_last_n=self.nonzero_last_n,
            add_date_part=self.add_date_part,
            holiday=self.holiday,
            holiday_country=self.holiday_country,
            polynomial_degree=self.polynomial_degree,
            window=self.window,
        )
        if self.regression_type == 'User':
            self.X = pd.concat([self.X, self.regressor_train], axis=1)

        if self.x_transform in ['FastICA', 'Nystroem', 'RmZeroVariance']:
            self.x_transformer = self._x_transformer()
            self.x_transformer = self.x_transformer.fit(self.X)
            self.X = pd.DataFrame(self.x_transformer.transform(self.X))
            self.X = self.X.replace([np.inf, -np.inf], 0).fillna(0)
        """
        Tail(1) is dropped to shift data to become forecast 1 ahead
        and the first one is dropped because it will least accurately represent
        rolling values
        """
        self.X = self.X.drop(self.X.tail(1).index).drop(self.X.head(1).index)
        if isinstance(self.X, pd.DataFrame):
            self.X.columns = [str(xc) for xc in self.X.columns]

        multioutput = True
        if self.Y.ndim < 2:
            multioutput = False
        elif self.Y.shape[1] < 2:
            multioutput = False
        # retrieve model object to train
        self.model = retrieve_regressor(
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
        self.model = self.model.fit(self.X, self.Y)

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
                nonzero_last_n=self.nonzero_last_n,
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

            rfPred = pd.DataFrame(self.model.predict(x_dat.tail(1).to_numpy()))

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
            [
                None,
                'simple',
                'expanded',
                'recurring',
                "simple_2",
                "simple_binarized",
                "expanded_binarized",
            ],
            [0.7, 0.05, 0.05, 0.05, 0.05, 0.1, 0.01],
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
            'nonzero_last_n': self.nonzero_last_n,
            'add_date_part': self.add_date_part,
            'polynomial_degree': self.polynomial_degree,
            'x_transform': self.x_transform,
            'regression_type': self.regression_type,
        }
        return parameter_dict


class RandomFourierEncoding(object):
    def __init__(self, n_components=100, sigma=1.0, random_state=None):
        self.n_components = n_components
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X, y=None):
        # np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.W = np.random.normal(
            loc=0, scale=1 / self.sigma, size=(n_features, self.n_components)
        )
        self.b = np.random.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        projection = np.dot(X, self.W) + self.b
        X_new = np.sqrt(2 / self.n_components) * np.concatenate(
            [np.sin(projection), np.cos(projection)], axis=1
        )
        return X_new


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
        fourier_encoding_components: float = None,
        scale: bool = False,
        datepart_method: str = None,
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
        self.max_windows = max_windows
        self.fourier_encoding_components = fourier_encoding_components
        self.scale = scale
        self.datepart_method = datepart_method
        self.static_regressor = None

    def fit(self, df, future_regressor=None, static_regressor=None):
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
        regression_type = self.regression_type
        if self.regression_type in ["User", "user"]:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but no future_regressor passed"
                )
            if isinstance(future_regressor, pd.Series):
                future_regressor = future_regressor.to_frame()
            self.static_regressor = static_regressor
            if self.datepart_method is not None:
                future_regressor = pd.concat(
                    [
                        future_regressor,
                        date_part(
                            df.index,
                            method=self.datepart_method,
                            holiday_country=self.holiday_country,
                        ),
                    ],
                    axis=1,
                )
        elif self.datepart_method is not None:
            regression_type = "User"  # to convince the window maker to use it
            future_regressor = date_part(
                df.index,
                method=self.datepart_method,
                holiday_country=self.holiday_country,
            )
        self.df_train = df

        self.X, self.Y = window_maker(
            df,
            window_size=self.window_size,
            input_dim=self.input_dim,
            normalize_window=self.normalize_window,
            shuffle=self.shuffle,
            output_dim=self.output_dim,
            forecast_length=self.forecast_length,
            max_windows=self.max_windows,
            regression_type=regression_type,
            future_regressor=future_regressor,
            random_seed=self.random_seed,
        )
        multioutput = True
        if self.Y.ndim < 2:
            multioutput = False
        elif self.Y.shape[1] < 2:
            multioutput = False
        if self.scale:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            if isinstance(self.X, pd.DataFrame):
                self.X.columns = self.X.columns.astype(str)
            self.X = self.scaler.fit_transform(self.X)
        if self.fourier_encoding_components is not None:
            self.fourier_encoder = RandomFourierEncoding(
                n_components=int(self.X.shape[1] * self.fourier_encoding_components),
                sigma=1.0,
            ).fit(self.X)
            self.X = self.fourier_encoder.transform(self.X.copy())
        if isinstance(self.X, pd.DataFrame):
            self.X = self.X.to_numpy()
        self.model = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=self.verbose,
            verbose_bool=self.verbose_bool,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            multioutput=multioutput,
        )
        self.model = self.model.fit(self.X, self.Y)
        self.last_window = df.tail(self.window_size)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def fit_data(self, df, future_regressor=None):
        df = self.basic_profile(df)
        self.last_window = df.tail(self.window_size)
        return self

    def predict(
        self,
        forecast_length: int = None,
        future_regressor=None,
        just_point_forecast: bool = False,
        df=None,
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
        if df is not None:
            self.fit_data(df)
        if forecast_length is None:
            forecast_length = self.forecast_length
        if int(forecast_length) > int(self.forecast_length):
            print("Regression must be refit to change forecast length!")
        index = self.create_forecast_index(forecast_length=forecast_length)
        if isinstance(future_regressor, pd.Series):
            future_regressor = future_regressor.to_frame()
        if self.regression_type in ["User", "user"]:
            if self.datepart_method is not None:
                future_regressor = pd.concat(
                    [
                        future_regressor,
                        date_part(
                            index,
                            method=self.datepart_method,
                            holiday_country=self.holiday_country,
                        ),
                    ],
                    axis=1,
                )
        elif self.datepart_method is not None:
            future_regressor = date_part(
                index,
                method=self.datepart_method,
                holiday_country=self.holiday_country,
            )

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
                if (
                    self.regression_type in ["User", "user"]
                    or self.datepart_method is not None
                ):
                    blasted_thing = (
                        future_regressor.reindex(index).iloc[x].to_frame().transpose()
                    )
                    tmerg = pd.concat([blasted_thing] * pred.shape[0], axis=0)
                    tmerg.index = pred.index
                    pred = pd.concat([pred, tmerg], axis=1, ignore_index=True)
                if self.scale:
                    if isinstance(pred, pd.DataFrame):
                        pred.columns = pred.columns.astype(str)
                    pred = self.scaler.transform(pred)
                if self.fourier_encoding_components is not None:
                    pred = self.fourier_encoder.transform(pred)
                if isinstance(pred, pd.DataFrame):
                    pred = pred.to_numpy()
                rfPred = pd.DataFrame(self.model.predict(pred))
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
            if (
                self.regression_type in ["User", "user"]
                or self.datepart_method is not None
            ):
                tmerg = future_regressor.tail(1).loc[
                    future_regressor.tail(1).index.repeat(pred.shape[0])
                ]
                tmerg.index = pred.index
                pred = pd.concat([pred, tmerg], axis=1)
            if self.scale:
                if isinstance(pred, pd.DataFrame):
                    pred.columns = pred.columns.astype(str)
                pred = self.scaler.transform(pred)
            if self.fourier_encoding_components is not None:
                pred = self.fourier_encoder.transform(pred)
            if isinstance(pred, pd.DataFrame):
                pred = pred.to_numpy(dtype=np.float32)
            cY = pd.DataFrame(self.model.predict(pred.astype(float)))
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
        if self.regression_model.get("model", "None") == "RadiusNeighbors":
            print('interpolating')
            df = df.interpolate("pchip").bfill()
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
                # so it's producing float32 but pandas is better with float64
                lower_forecast=lower_forecast.astype(float),
                forecast=df.astype(float),
                upper_forecast=upper_forecast.astype(float),
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
        if method == "deep":
            max_windows_choice = random.choices(
                [5000, 50000, 5000000, None], [0.4, 0.2, 0.9, 0.05]
            )[0]
        elif method == "fast":
            max_windows_choice = random.choices([10000, 100000], [0.2, 0.2])[0]
        else:
            max_windows_choice = random.choices(
                [5000, 50000, 5000000], [0.2, 0.2, 0.9]
            )[0]
        datepart_method = random.choices([None, "something"], [0.9, 0.1])[0]
        if datepart_method == "something":
            datepart_method = random_datepart()
        return {
            'window_size': wnd_sz_choice,
            'input_dim': input_dim_choice,
            'output_dim': output_dim_choice,
            'normalize_window': normalize_window_choice,
            'max_windows': max_windows_choice,
            'fourier_encoding_components': random.choices(
                [None, 2, 5, 10], [0.8, 0.1, 0.1, 0.01]
            )[0],
            'scale': random.choices([True, False], [0.7, 0.3])[0],
            'datepart_method': datepart_method,
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
            'fourier_encoding_components': self.fourier_encoding_components,
            'scale': self.scale,
            'datepart_method': self.datepart_method,
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
            ).fit(X.astype(float), future_regressor=future_regressor)
        except Exception as e:
            raise ValueError(f"Model {str(self.model)} with error: {repr(e)}")
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int = None,
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

        if forecast_length is None:
            forecast_length = self.forecast_length
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
        holiday_countries_used: bool = False,
        lags: int = None,
        forward_lags: int = None,
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
        self.forecast_length = forecast_length
        self.lags = lags
        self.forward_lags = forward_lags
        self.holiday_countries_used = holiday_countries_used

    def fit(
        self,
        df,
        future_regressor=None,
        static_regressor=None,
        regressor_per_series=None,
    ):
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

        self.X = date_part(
            df.index,
            method=self.datepart_method,
            polynomial_degree=self.polynomial_degree,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
            lags=self.lags,
            forward_lags=self.forward_lags,
        )
        if self.regression_type in ['User', 'user']:
            # regr = future_regressor.copy()
            # regr.index = X.index
            self.X = pd.concat([self.X, future_regressor], axis=1)
        self.X.columns = [str(xc) for xc in self.X.columns]

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
        self.model = self.model.fit(self.X.astype(np.float32), y.astype(np.float32))
        self.shape = df.shape
        return self

    def fit_data(self, df, future_regressor=None):
        self.basic_profile(df)
        return self

    def predict(
        self,
        forecast_length: int = None,
        future_regressor=None,
        just_point_forecast: bool = False,
        df=None,
        regressor_per_series=None,
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
        if forecast_length is None:
            forecast_length = self.forecast_length
        if df is not None:
            self.fit_data(df)
        predictStartTime = datetime.datetime.now()
        index = self.create_forecast_index(forecast_length=forecast_length)
        self.X_pred = date_part(
            index,
            method=self.datepart_method,
            polynomial_degree=self.polynomial_degree,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
            lags=self.lags,
            forward_lags=self.forward_lags,
        )
        if self.regression_type in ['User', 'user']:
            self.X_pred = pd.concat(
                [self.X_pred, future_regressor.reindex(index)], axis=1
            )
            if self.X_pred.shape[0] > index.shape[0]:
                raise ValueError(
                    f"future_regressor {future_regressor.index} and X {self.X_pred.index} index failed to align"
                )
        self.X_pred.columns = [str(xc) for xc in self.X_pred.columns]

        try:
            forecast = pd.DataFrame(
                self.model.predict(self.X_pred.astype(float)),
                index=index,
                columns=self.column_names,
            )
        except Exception as e:
            raise ValueError(
                f"Datepart prediction with params {self.get_params()} failed. This is often due to an improperly indexed future_regressor (with drop_most_recent especially)"
            ) from e
        # RadiusNeighbors works decently well but starts throwing out NaN when it can't find neighbors
        if self.regression_model.get("model", "None") == "RadiusNeighbors":
            print('interpolating')
            forecast = forecast.interpolate("pchip").bfill()

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
        if method == 'deep':
            model_choice = generate_regressor_params(
                model_dict=datepart_model_dict_deep
            )
        else:
            model_choice = generate_regressor_params(model_dict=datepart_model_dict)
        datepart_choice = random_datepart()
        if datepart_choice in ["simple", "simple_2", "recurring", "simple_binarized"]:
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
            "holiday_countries_used": random.choices([True, False], [0.2, 0.8])[0],
            'lags': random.choices([None, 1, 2, 3, 4], [0.9, 0.1, 0.1, 0.05, 0.05])[0],
            'forward_lags': random.choices(
                [None, 1, 2, 3, 4], [0.9, 0.1, 0.1, 0.05, 0.05]
            )[0],
            'regression_type': regression_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'regression_model': self.regression_model,
            'datepart_method': self.datepart_method,
            'polynomial_degree': self.polynomial_degree,
            'holiday_countries_used': self.holiday_countries_used,
            'lags': self.lags,
            'forward_lags': self.forward_lags,
            'regression_type': self.regression_type,
        }
        return parameter_dict


class UnivariateRegression(ModelObject):
    """Regression-framed approach to forecasting using sklearn.
    A univariate version of rolling regression: ie each series is modeled independently

    "You've got to think for your selves!. You're ALL individuals"
    "Yes! We're all individuals!" - Python

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
            model_choice = generate_regressor_params(
                model_dict=univariate_model_dict, method=method
            )
        else:
            x_transform_choice = random.choices(
                [None, 'FastICA', 'Nystroem', 'RmZeroVariance'],
                [1.0, 0.0, 0.0, 0.0],
            )[0]
            window_choice = random.choices([None, 3, 7, 10], [0.7, 0.2, 0.05, 0.05])[0]
            model_choice = generate_regressor_params(
                model_dict={
                    'ElasticNet': 0.2,
                    'DecisionTree': 0.4,
                    'FastRidge': 0.2,
                    'LinearRegression': 0.2,
                    # 'ExtraTrees': 0.1,
                },
                method=method,
            )
        if method == 'neuralnets':
            print('`neuralnets` model_mode does not apply to UnivariateRegression')
            method = 'deep'

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
            [
                None,
                'simple',
                'expanded',
                'recurring',
                "simple_2",
                "simple_2_poly",
                "simple_binarized",
            ],
            [0.8, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05],
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
        forecast_length: int = 28,
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
        nonzero_last_n: int = None,
        datepart_method: str = None,
        polynomial_degree: int = None,
        window: int = 5,
        probabilistic: bool = False,
        scale_full_X: bool = False,
        quantile_params: dict = {
            'learning_rate': 0.1,
            'max_depth': 20,
            'min_samples_leaf': 4,
            'min_samples_split': 5,
            'n_estimators': 250,
        },
        cointegration: str = None,
        cointegration_lag: int = 1,
        series_hash: bool = False,
        frac_slice: float = None,
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
        self.nonzero_last_n = nonzero_last_n
        self.datepart_method = datepart_method
        self.polynomial_degree = polynomial_degree
        self.window = window
        self.quantile_params = quantile_params
        self.regressor_train = None
        self.regressor_per_series_train = None
        self.static_regressor = None
        self.probabilistic = probabilistic
        self.scale_full_X = scale_full_X
        self.cointegration = cointegration
        self.cointegration_lag = cointegration_lag
        self.series_hash = series_hash
        self.frac_slice = frac_slice

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
            nonzero_last_n,
            window,
            starting_min,
        ]
        self.min_threshold = max([x for x in list_o_vals if str(x).isdigit()])
        self.scaler_mean = None

    def base_scaler(self, df):
        self.scaler_mean = np.mean(df, axis=0)
        self.scaler_std = np.std(df, axis=0).replace(0.0, 1.0)
        return (df - self.scaler_mean) / self.scaler_std

    def scale_data(self, df):
        if self.scaler_mean is None:
            return self.base_scaler(df)
        else:
            return (df - self.scaler_mean) / self.scaler_std

    def to_origin_space(
        self, df, trans_method='forecast', components=False, bounds=False
    ):
        """Take transformed outputs back to original feature space."""
        return df * self.scaler_std + self.scaler_mean

    def fit(
        self,
        df,
        future_regressor=None,
        static_regressor=None,
        regressor_per_series=None,
    ):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            future_regressor (pandas.DataFrame or Series): Datetime Indexed
        """
        df = self.basic_profile(df)
        # assume memory and CPU count are correlated
        with config_context(assume_finite=True, working_memory=int(self.n_jobs * 512)):
            # if external regressor, do some check up
            if self.regression_type is not None:
                if future_regressor is None:
                    raise ValueError(
                        "regression_type='User' but not future_regressor supplied."
                    )
                else:
                    self.regressor_train = future_regressor.reindex(df.index)
                if regressor_per_series is not None:
                    self.regressor_per_series_train = regressor_per_series
                if static_regressor is not None:
                    self.static_regressor = static_regressor
            # define X and Y
            if self.frac_slice is not None:
                slice_size = int(df.shape[0] * self.frac_slice)
                self.slice_index = df.index[slice_size - 1 : -1]
                self.Y = df.iloc[slice_size:].to_numpy().ravel(order="F")
            else:
                self.slice_index = None
                self.Y = df[1:].to_numpy().ravel(order="F")
            # drop look ahead data
            base = df[:-1]
            if self.regression_type is not None:
                cut_regr = self.regressor_train[1:]
                cut_regr.index = base.index
            else:
                cut_regr = None
            # Create X, parallel and non-parallel versions
            parallel = True
            if self.n_jobs in [0, 1] or df.shape[1] < 20:
                parallel = False
            elif not joblib_present:
                parallel = False
            # joblib multiprocessing to loop through series
            # this might be causing issues, TBD Key Error from Resource Tracker
            if parallel:
                self.X = Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose, timeout=3600
                )(
                    delayed(rolling_x_regressor_regressor)(
                        base[x_col].to_frame().astype(float),
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
                        nonzero_last_n=self.nonzero_last_n,
                        add_date_part=self.datepart_method,
                        holiday=self.holiday,
                        holiday_country=self.holiday_country,
                        polynomial_degree=self.polynomial_degree,
                        window=self.window,
                        future_regressor=cut_regr,
                        # these rely the if part not being run if None
                        regressor_per_series=(
                            self.regressor_per_series_train[x_col]
                            if self.regressor_per_series_train is not None
                            else None
                        ),
                        static_regressor=(
                            static_regressor.loc[x_col].to_frame().T
                            if self.static_regressor is not None
                            else None
                        ),
                        cointegration=self.cointegration,
                        cointegration_lag=self.cointegration_lag,
                        series_id=x_col if self.series_hash else None,
                        slice_index=self.slice_index,
                    )
                    for x_col in base.columns
                )
                self.X = pd.concat(self.X)
            else:
                self.X = pd.concat(
                    [
                        rolling_x_regressor_regressor(
                            base[x_col].to_frame().astype(float),
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
                            nonzero_last_n=self.nonzero_last_n,
                            add_date_part=self.datepart_method,
                            holiday=self.holiday,
                            holiday_country=self.holiday_country,
                            polynomial_degree=self.polynomial_degree,
                            window=self.window,
                            future_regressor=cut_regr,
                            # these rely the if part not being run if None
                            regressor_per_series=(
                                self.regressor_per_series_train[x_col]
                                if self.regressor_per_series_train is not None
                                else None
                            ),
                            static_regressor=(
                                static_regressor.loc[x_col].to_frame().T
                                if self.static_regressor is not None
                                else None
                            ),
                            cointegration=self.cointegration,
                            cointegration_lag=self.cointegration_lag,
                            series_id=x_col if self.series_hash else None,
                            slice_index=self.slice_index,
                        )
                        for x_col in base.columns
                    ]
                )

            del base
            if self.probabilistic:
                from sklearn.ensemble import GradientBoostingRegressor

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
            if self.Y.ndim < 2:
                multioutput = False
            elif self.Y.shape[1] < 2:
                multioutput = False
            self.model = retrieve_regressor(
                regression_model=self.regression_model,
                verbose=self.verbose,
                verbose_bool=self.verbose_bool,
                random_seed=self.random_seed,
                n_jobs=self.n_jobs,
                multioutput=multioutput,
            )
            self.multioutputgpr = self.regression_model['model'] == "MultioutputGPR"
            if self.scale_full_X:
                self.X = self.scale_data(self.X)

            # Remember the X datetime is for the previous day to the Y datetime here
            assert self.X.index[-1] == df.index[-2]
            self.model.fit(self.X.to_numpy(), self.Y)

            if self.probabilistic and not self.multioutputgpr:
                self.model_upper.fit(self.X.to_numpy(), self.Y)
                self.model_lower.fit(self.X.to_numpy(), self.Y)
            # we only need the N most recent points for predict
            # self.sktraindata = df.tail(self.min_threshold)
            self.fit_data(df)

            self.fit_runtime = datetime.datetime.now() - self.startTime
            return self

    def fit_data(
        self,
        df,
        future_regressor=None,
        static_regressor=None,
        regressor_per_series=None,
    ):
        df = self.basic_profile(df)
        self.sktraindata = df.tail(self.min_threshold)
        if self.regression_type is not None:
            if future_regressor is not None:
                self.regressor_train = future_regressor.reindex(df)
            if regressor_per_series is not None:
                self.regressor_per_series_train = regressor_per_series
            if static_regressor is not None:
                self.static_regressor = static_regressor
        return self

    def predict(
        self,
        forecast_length: int = None,
        just_point_forecast: bool = False,
        future_regressor=None,
        df=None,
        regressor_per_series=None,
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
        if df is not None:
            self.fit_data(df)  # no new regressor support
        if forecast_length is None:
            forecast_length = self.forecast_length
        predictStartTime = datetime.datetime.now()
        index = self.create_forecast_index(forecast_length=forecast_length)
        forecast = []
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
            self.X_pred = pd.concat(
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
                        nonzero_last_n=self.nonzero_last_n,
                        add_date_part=self.datepart_method,
                        holiday=self.holiday,
                        holiday_country=self.holiday_country,
                        polynomial_degree=self.polynomial_degree,
                        window=self.window,
                        future_regressor=cur_regr,
                        # these rely the if part not being run if None
                        regressor_per_series=(
                            regressor_per_series[x_col]
                            if self.regressor_per_series_train is not None
                            else None
                        ),
                        static_regressor=(
                            self.static_regressor.loc[x_col].to_frame().T
                            if self.static_regressor is not None
                            else None
                        ),
                        cointegration=self.cointegration,
                        cointegration_lag=self.cointegration_lag,
                        series_id=x_col if self.series_hash else None,
                    ).tail(1)
                    for x_col in current_x.columns
                ]
            )
            if self.scale_full_X:
                c_x_pred = self.scale_data(self.X_pred).to_numpy()
                rfPred = self.model.predict(c_x_pred)
            else:
                c_x_pred = self.X_pred.to_numpy()
                rfPred = self.model.predict(c_x_pred)
            pred_clean = pd.DataFrame(
                rfPred, index=current_x.columns, columns=[index[fcst_step]]
            ).transpose()
            forecast.append(pred_clean)
            # a lot slower
            if self.probabilistic:
                if self.multioutputgpr:
                    med, var = self.model.predict_proba(c_x_pred)
                    stdev = np.sqrt(var)[..., np.newaxis].T * norm.ppf(
                        (1 + self.prediction_interval) / 2
                    )
                    pred_upper = pd.DataFrame(
                        med + stdev, index=[index[fcst_step]], columns=current_x.columns
                    )
                    pred_lower = pd.DataFrame(
                        med - stdev, index=[index[fcst_step]], columns=current_x.columns
                    )
                    upper_forecast = pd.concat([upper_forecast, pred_upper])
                    lower_forecast = pd.concat([lower_forecast, pred_lower])
                else:
                    rfPred_upper = self.model_upper.predict(c_x_pred)
                    pred_upper = pd.DataFrame(
                        rfPred_upper,
                        index=current_x.columns,
                        columns=[index[fcst_step]],
                    ).transpose()
                    rfPred_lower = self.model_lower.predict(c_x_pred)
                    pred_lower = pd.DataFrame(
                        rfPred_lower,
                        index=current_x.columns,
                        columns=[index[fcst_step]],
                    ).transpose()
                    upper_forecast = pd.concat([upper_forecast, pred_upper])
                    lower_forecast = pd.concat([lower_forecast, pred_lower])
            current_x = pd.concat(
                [
                    current_x,
                    pred_clean,
                ]
            )

        forecast = pd.concat(forecast)
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
            window_choice = random.choices([None, 3, 7, 10], [0.3, 0.3, 0.1, 0.05])[0]
            probabilistic = False
        mean_rolling_periods_choice = random.choices(
            [None, 5, 7, 12, 30, 90, [2, 4, 6, 8, 12, (52, 2)], [7, 28, 364, (362, 4)]],
            [0.3, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
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
            [None, 2, 7, 12, 30], [0.99, 0.01, 0.01, 0.01, 0.01]
        )[0]
        nonzero_last_n = random.choices(
            [None, 2, 7, 14, 30], [0.6, 0.01, 0.1, 0.1, 0.01]
        )[0]
        add_date_part_choice = random.choices(
            [
                None,
                'simple',
                'expanded',
                'recurring',
                "simple_2",
                "simple_2_poly",
                "simple_binarized",
                "common_fourier",
                "expanded_binarized",
                "common_fourier_rw",
                ["dayofweek", 365.25],
                "simple_binarized2_poly",
            ],
            [0.2, 0.1, 0.025, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05, 0.025, 0.05, 0.05],
        )[0]
        holiday_choice = random.choices([True, False], [0.1, 0.9])[0]
        polynomial_degree_choice = random.choices([None, 2], [0.995, 0.005])[0]
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices([None, 'User'], [0.7, 0.3])[0]
        coint_lag = 1
        if "deep" in method:
            coint_choice = random.choices([None, "BTCD", "Johansen"], [0.8, 0.1, 0.1])[
                0
            ]
        else:
            coint_choice = None
        if coint_choice is not None:
            coint_lag = random.choice([1, 2, 7])
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
            'nonzero_last_n': nonzero_last_n,
            'datepart_method': add_date_part_choice,
            'polynomial_degree': polynomial_degree_choice,
            'regression_type': regression_choice,
            'window': window_choice,
            'holiday': holiday_choice,
            "probabilistic": probabilistic,
            'scale_full_X': random.choices([True, False], [0.2, 0.8])[0],
            "cointegration": coint_choice,
            "cointegration_lag": coint_lag,
            "series_hash": random.choices([True, False], [0.5, 0.5])[0],
            "frac_slice": random.choices(
                [None, 0.8, 0.5, 0.2, 0.1], [0.6, 0.1, 0.1, 0.1, 0.1]
            )[0],
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
            'nonzero_last_n': self.nonzero_last_n,
            'datepart_method': self.datepart_method,
            'polynomial_degree': self.polynomial_degree,
            'regression_type': self.regression_type,
            'window': self.window,
            'holiday': self.holiday,
            'probabilistic': self.probabilistic,
            'scale_full_X': self.scale_full_X,
            "cointegration": self.cointegration,
            "cointegration_lag": self.cointegration_lag,
            "series_hash": self.series_hash,
            "frac_slice": self.frac_slice,
        }
        return parameter_dict


class VectorizedMultiOutputGPR:
    """Gaussian Process Regressor.

    Args:
        kernel (str): linear, polynomial, rbf, periodic, locally_periodic, exponential
        noise_var (float): noise variance, effectively regularization. Close to zero little regularization, larger values create more model flexiblity and noise tolerance.
        gamma: For the RBF, Exponential, and Locally Periodic kernels, γ is essentially an inverse length scale. [0.1,1,10,100].
        lambda_: For the Periodic and Locally Periodic kernels, lambda_ determines the smoothness of the periodic function. A reasonable range might be [0.1,1,10,100].
        lambda_prime: Specifically for the Locally Periodic kernel, this determines the smoothness of the periodic component. Same range as lambda_.
        p: The period parameter for the Periodic and Locally Periodic kernels such as 7 or 365.25 for daily data.
    """

    def __init__(self, kernel='rbf', noise_var=10, gamma=0.1, lambda_prime=0.1, p=7):
        self.kernel = kernel
        self.noise_var = noise_var
        self.gamma = gamma
        self.lambda_prime = lambda_prime
        self.p = p

    def _linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def _polynomial_kernel(self, x1, x2, p=2):
        return (1 + np.dot(x1, x2.T)) ** p

    def _rbf_kernel(self, x1, x2, gamma):
        # from scipy.spatial.distance import cdist
        # return np.exp(-gamma * cdist(x1, x2, 'sqeuclidean'))
        if gamma is None:
            gamma = 1.0 / x1.shape[1]
        distance = (
            np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        )
        return np.exp(-gamma * distance)

    def _old_exponential_kernel(self, x1, x2, gamma):
        # memory hungry
        diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        return np.exp(-np.abs(diff) / gamma)
        # return np.exp(-np.abs(x1 - x2.T) / gamma)

    def _exponential_kernel(self, x1, x2, gamma):
        # less memory hungry
        result = np.empty((x1.shape[0], x2.shape[0]))
        for i, xi in enumerate(x1):
            diff = np.abs(xi - x2)
            result[i, :] = np.exp(-diff.sum(axis=1) / gamma)
        return result

    def _old_periodic_kernel(self, x1, x2, gamma, p):
        sin_sq = np.sin(np.pi * np.abs(x1 - x2.T) / p) ** 2
        return np.exp(-2 * sin_sq / gamma**2)

    def _vec_periodic_kernel(self, x1, x2, gamma, p):
        diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        sin_sq = np.sin(np.pi * np.abs(diff) / p) ** 2
        return np.exp(-2 * sin_sq / gamma**2)

    def _periodic_kernel(self, x1, x2, gamma, p):
        result = np.empty((x1.shape[0], x2.shape[0]))
        for i, xi in enumerate(x1):
            diff = xi - x2
            sin_sq = (np.sin(np.pi * np.abs(diff) / p) ** 2).sum(axis=1)
            result[i, :] = np.exp(-2 * sin_sq / gamma**2)
        return result

    def _old_locally_periodic_kernel(self, x1, x2, gamma, lambda_prime, p):
        rbf_part = np.exp(
            -((x1 - x2) ** 2) / (2 * gamma**2)
        )  #  old: np.exp(-((x1 - x2.T) ** 2) / (2 * gamma**2))
        periodic_part = self._periodic_kernel(x1, x2, lambda_prime, p)
        return rbf_part * periodic_part

    def _locally_periodic_kernel(self, x1, x2, gamma, lambda_prime, p):
        result = np.empty((x1.shape[0], x2.shape[0]))
        for i, xi in enumerate(x1):
            diff = xi - x2
            rbf_part = np.exp(-np.sum(diff**2, axis=1) / (2 * gamma**2))
            sin_sq = (np.sin(np.pi * np.abs(diff) / p) ** 2).sum(axis=1)
            periodic_part = np.exp(-2 * sin_sq / gamma**2)
            result[i, :] = rbf_part * periodic_part
        return result

    def fit(self, X, Y):
        self.X_train = np.asarray(X)

        if self.kernel == 'linear':
            K = self._linear_kernel(self.X_train, self.X_train)
        elif self.kernel == 'polynomial':
            K = self._polynomial_kernel(self.X_train, self.X_train)
        elif self.kernel == 'rbf':
            K = self._rbf_kernel(self.X_train, self.X_train, self.gamma)
        elif self.kernel == 'exponential':
            K = self._exponential_kernel(self.X_train, self.X_train, self.gamma)
        elif self.kernel == 'periodic':
            K = self._periodic_kernel(self.X_train, self.X_train, self.gamma, self.p)
        elif self.kernel == 'locally_periodic':
            K = self._locally_periodic_kernel(
                self.X_train, self.X_train, self.gamma, self.lambda_prime, self.p
            )
        else:
            raise ValueError("Invalid Kernel")

        # Regularized Kernel
        # K_reg = K + self.noise_var * np.eye(K.shape[0])
        np.fill_diagonal(K, np.diag(K) + self.noise_var)

        # Cholesky decomposition and solve for alpha in a vectorized way
        if False:
            from scipy.sparse.linalg import cg

            self.alpha, _ = cg(K, np.asarray(Y))  # _ captures info about convergence
        else:
            self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(
                self.L.T, np.linalg.solve(self.L, np.asarray(Y))
            )
        del K
        # Regularized Kernel
        return self

    def predict(self, X):
        x_pred = np.asarray(X)
        if self.kernel == 'linear':
            k = self._linear_kernel(x_pred, self.X_train)
        elif self.kernel == 'polynomial':
            k = self._polynomial_kernel(x_pred, self.X_train)
        elif self.kernel == 'rbf':
            k = self._rbf_kernel(x_pred, self.X_train, self.gamma)
        elif self.kernel == 'exponential':
            k = self._exponential_kernel(x_pred, self.X_train, self.gamma)
        elif self.kernel == 'periodic':
            k = self._periodic_kernel(x_pred, self.X_train, self.gamma, self.p)
        elif self.kernel == 'locally_periodic':
            k = self._locally_periodic_kernel(
                x_pred, self.X_train, self.gamma, self.lambda_prime, self.p
            )
        else:
            raise ValueError("Invalid Kernel")

        # Making predictions
        Y_pred = k @ self.alpha

        return Y_pred

    def predict_proba(self, X):
        x_pred = np.asarray(X)
        if self.kernel == 'linear':
            k_star = self._linear_kernel(x_pred, self.X_train)
            k_star_star = np.diag(self._linear_kernel(x_pred, self.X_train))
        elif self.kernel == 'polynomial':
            k_star = self._polynomial_kernel(x_pred, self.X_train)
            k_star_star = np.diag(self._polynomial_kernel(x_pred, self.X_train))
        elif self.kernel == 'rbf':
            k_star = self._rbf_kernel(x_pred, self.X_train, self.gamma)
            k_star_star = np.diag(self._rbf_kernel(x_pred, self.X_train, self.gamma))

        elif self.kernel == 'exponential':
            k_star = self._exponential_kernel(X, self.X_train, self.gamma)
            k_star_star = np.diag(
                self._exponential_kernel(x_pred, self.X_train, self.gamma)
            )
        elif self.kernel == 'periodic':
            k_star = self._periodic_kernel(X, self.X_train, self.gamma, self.p)
            k_star_star = np.diag(
                self._periodic_kernel(x_pred, self.X_train, self.gamma, self.lambda_)
            )
        elif self.kernel == 'locally_periodic':
            k_star = self._locally_periodic_kernel(
                x_pred, self.X_train, self.gamma, self.lambda_prime, self.p
            )
            k_star_star = np.diag(
                self._locally_periodic_kernel(
                    x_pred, self.X_train, self.gamma, self.lambda_prime, self.p
                )
            )
        else:
            raise ValueError("Invalid Kernel")

        # Making predictions
        Y_pred = k_star @ self.alpha

        # Computing the predictive variance for each test point and each output
        v = np.linalg.solve(self.L, k_star.T)
        Y_var = k_star_star - np.sum(v**2, axis=0)

        return Y_pred, Y_var


class PreprocessingRegression(ModelObject):
    """Regression use the last n values as the basis of training data.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        # regression_type: str = None,
    """

    def __init__(
        self,
        name: str = "PreprocessingRegression",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2023,
        verbose: int = 0,
        window_size: int = 10,
        regression_model: dict = {
            "model": 'RandomForest',
            "model_params": {},
        },
        transformation_dict=None,
        max_history: int = None,
        one_step: bool = False,
        processed_y: bool = False,
        normalize_window: bool = False,
        datepart_method: str = "common_fourier",
        forecast_length: int = 28,
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
        self.regression_model = regression_model
        self.window_size = abs(int(window_size))
        self.max_history = max_history
        self.one_step = one_step
        self.processed_y = processed_y
        self.transformation_dict = transformation_dict
        self.datepart_method = datepart_method
        self.normalize_window = normalize_window
        self.forecast_length = forecast_length

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        self.fit_data(df)
        if self.regression_type in ["User", "user"]:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but no future_regressor passed"
                )
        from autots.tools.transform import GeneralTransformer  # avoid circular imports

        self.transformer_object = GeneralTransformer(
            n_jobs=self.n_jobs,
            holiday_country=self.holiday_country,
            verbose=self.verbose,
            random_seed=self.random_seed,
            forecast_length=self.forecast_length,
            **self.transformation_dict,
        )
        forecast_length = self.forecast_length
        one_step = self.one_step
        processed_y = self.processed_y
        df = self.transformer_object._first_fit(df)

        df_list = [df]
        new_df = df
        trans_keys = sorted(list(self.transformer_object.transformations.keys()))
        for i in trans_keys:
            trans_name = self.transformer_object.transformations[i]
            if trans_name not in ['Slice']:
                new_df = self.transformer_object._fit_one(new_df, i)
                df_list.append(new_df)

        max_history = 0 if self.max_history is None else abs(int(self.max_history))
        window_list = []
        for cdf in df_list:
            window_list.append(
                sliding_window_view(
                    np.asarray(cdf)[-max_history:],
                    window_shape=(self.window_size,),
                    axis=0,
                )
            )
        full = np.concatenate(window_list, axis=2)

        extras = self._construct_extras(
            df_index=df.index, future_regressor=future_regressor
        )

        full_end = df.shape[0] - 1
        window_end = full.shape[0] - 1
        max_hist_rev = df.shape[0] - max_history if max_history != 0 else 0

        if one_step:
            self.X = full[:-1].reshape(-1, full.shape[-1])
            if extras is not None:
                self.X = np.concatenate(
                    [
                        self.X,
                        extras[self.window_size + max_hist_rev :].reshape(
                            -1, extras.shape[-1]
                        ),
                    ],
                    axis=1,
                )
            if processed_y:
                self.Y = np.asarray(df_list[-1])[
                    self.window_size + max_hist_rev :
                ].reshape(-1)
            else:
                self.Y = np.asarray(df)[self.window_size + max_hist_rev :].reshape(-1)
        else:
            windows = []
            targets = []
            extra_sel = []
            forecast_steps = []
            # concat and drop window of end of dataset
            for n in range(forecast_length):
                # don't include last point for windows as it is the training point
                end_point = window_end - n - 1
                if end_point > 0:
                    window_idx = np.arange(0, end_point)
                    windows.append(full[window_idx])
                    # for y and target related vars, don't include first point
                    target_idx = np.arange(
                        self.window_size + n + max_hist_rev, full_end
                    )
                    if processed_y:
                        targets.append(np.asarray(df_list[-1])[target_idx])
                    else:
                        targets.append(np.asarray(df)[target_idx])
                    forecast_steps.append((np.ones_like(df) * n)[target_idx])
                    if extras is not None:
                        extra_sel.append(extras[target_idx])
            windows = np.concatenate(windows, axis=0)
            if extras is not None:
                extra_sel = np.concatenate(extra_sel, axis=0)
            forecast_steps = np.concatenate(forecast_steps, axis=0)
            if extras is not None:
                self.X = np.concatenate(
                    [
                        windows.reshape(-1, windows.shape[-1]),
                        extra_sel.reshape(-1, extras.shape[-1]),
                        forecast_steps.reshape(-1, 1),
                    ],
                    axis=1,
                )
            else:
                self.X = np.concatenate(
                    [
                        windows.reshape(-1, windows.shape[-1]),
                        forecast_steps.reshape(-1, 1),
                    ],
                    axis=1,
                )
            self.Y = np.concatenate(targets, axis=0).reshape(-1)

        # DROP values which contain numpy, not having filled initial nan values
        nan_rows = np.argwhere(np.isnan(np.sum(self.X, axis=1)))
        if nan_rows.size != 0:
            self.X = np.delete(self.X, nan_rows, axis=0)
            self.Y = np.delete(self.Y, nan_rows, axis=0)

        multioutput = True
        if self.Y.ndim < 2:
            multioutput = False
        elif self.Y.shape[1] < 2:
            multioutput = False
        self.model = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=self.verbose,
            verbose_bool=self.verbose_bool,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            multioutput=multioutput,
        )
        if self.normalize_window:
            self.X = self._base_scaler(self.X)
        self.model = self.model.fit(self.X.astype(float), self.Y.astype(float))
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def _construct_extras(self, df_index, future_regressor):
        extras = True
        if self.datepart_method is not None:
            date_part_df = date_part(df_index, method=self.datepart_method)
            if future_regressor is not None:
                date_part_df = date_part_df.merge(
                    future_regressor, left_index=True, right_index=True, how='left'
                )
        elif future_regressor is not None and self.regression_type in ["User", "user"]:
            date_part_df = future_regressor
        else:
            extras = None
        if extras is not None:
            N = self.train_shape[1]
            # memory efficient repeat, hopefully
            extras = np.moveaxis(
                np.asarray(date_part_df)[:, :, np.newaxis] * np.ones((1, 1, N)), 2, 1
            )
        return extras

    def _construct_full(self, df):
        # used in predict only
        df = self.transformer_object._first_fit(df)
        df_list = [df]
        new_df = df
        trans_keys = sorted(list(self.transformer_object.transformations.keys()))
        for i in trans_keys:
            trans_name = self.transformer_object.transformations[i]
            if trans_name not in ['Slice']:
                new_df = self.transformer_object._transform_one(new_df, i)
                df_list.append(new_df)

        # max_history = 0 if self.max_history is None else self.max_history
        window_list = []
        for cdf in df_list:
            window_list.append(
                sliding_window_view(
                    np.asarray(cdf),
                    window_shape=(self.window_size,),
                    axis=0,
                )
            )  # [-max_history:]
        return np.concatenate(window_list, axis=2)

    def _base_scaler(self, X):
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0)
        return (X - self.scaler_mean) / self.scaler_std

    def _scale(self, X):
        return (X - self.scaler_mean) / self.scaler_std

    def fit_data(self, df, future_regressor=None):
        df = self.basic_profile(df)
        self.last_window = df.tail(self.window_size)
        return self

    def predict(
        self,
        forecast_length: int = None,
        future_regressor=None,
        just_point_forecast: bool = False,
        df=None,
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
        if df is not None:
            self.fit_data(df)
        if forecast_length is None:
            forecast_length = self.forecast_length
        if int(forecast_length) > int(self.forecast_length) and not self.one_step:
            print("Regression must be refit to change forecast length!")
        index = self.create_forecast_index(forecast_length=forecast_length)

        if self.one_step:
            # combined_index = (self.df_train.index.append(index))
            forecast = pd.DataFrame()
            # forecast, 1 step ahead, then another, and so on
            lwindow = self.last_window
            for x in range(forecast_length):
                cindex = index[x : x + 1]
                full = self._construct_full(lwindow)
                c_reg = (
                    future_regressor.reindex(cindex)
                    if future_regressor is not None
                    else None
                )
                extras = self._construct_extras(cindex, c_reg)
                if extras is not None:
                    pred = np.concatenate(
                        [
                            full[-1:].reshape(-1, full.shape[-1]),
                            extras.reshape(-1, extras.shape[-1]),
                        ],
                        axis=1,
                    )
                else:
                    pred = full[-1:].reshape(-1, full.shape[-1])
                if self.normalize_window:
                    pred = self._scale(pred)
                rfPred = pd.DataFrame(self.model.predict(pred)).transpose()
                rfPred.columns = self.last_window.columns
                rfPred.index = cindex
                if self.processed_y:
                    # won't work with some preprocessing like diff
                    rfPred = self.transformer_object.inverse_transform(rfPred)
                forecast = pd.concat([forecast, rfPred], axis=0)
                lwindow = pd.concat([lwindow, rfPred], axis=0, ignore_index=False).tail(
                    self.window_size
                )
            df = forecast
        else:
            full = self._construct_full(self.last_window)[-1:]
            full = full.reshape(-1, full.shape[-1])
            N = self.train_shape[1]
            forecast_steps = np.repeat(np.arange(1, forecast_length + 1), N).reshape(
                -1, 1
            )
            c_reg = (
                future_regressor.reindex(index)
                if future_regressor is not None
                else None
            )
            extras = self._construct_extras(index, c_reg)
            if extras is None:
                self.pred = np.concatenate([full, forecast_steps], axis=1)
            else:
                self.pred = np.concatenate(
                    [
                        np.tile(full, (forecast_length, 1)),
                        extras.reshape(-1, extras.shape[-1]),
                        forecast_steps,
                    ],
                    axis=1,
                )
            if self.normalize_window:
                self.pred = self._scale(self.pred)
            cY = self.model.predict(self.pred.astype(float))
            cY = pd.DataFrame(cY.reshape(forecast_length, self.train_shape[1]))
            cY.columns = self.column_names
            cY.index = index
            if self.processed_y:
                # won't work with some preprocessing like diff
                cY = self.transformer_object.inverse_transform(cY)
            df = cY

        if just_point_forecast:
            return df
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.last_window,
                df,
                prediction_interval=self.prediction_interval,
                method='inferred_normal',
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
        else:
            regression_type_choice = random.choices([None, "User"], weights=[0.8, 0.2])[
                0
            ]
        normalize_window_choice = random.choices([True, False], [0.1, 0.9])[0]
        datepart_choice = random.choices(
            [
                "recurring",
                "simple",
                "expanded",
                "simple_2",
                "simple_binarized",
                "expanded_binarized",
                'common_fourier',
            ],
            [0.4, 0.3, 0.3, 0.3, 0.4, 0.05, 0.05],
        )[0]
        return {
            'window_size': wnd_sz_choice,
            'max_history': random.choices([None, 1000], [0.5, 0.5])[0],
            'one_step': random.choice([True, False]),
            'normalize_window': normalize_window_choice,
            'processed_y': random.choice([True, False]),
            'transformation_dict': None,  # assume this passed via AutoTS transformer dict
            'datepart_method': datepart_choice,
            'regression_type': regression_type_choice,
            'regression_model': model_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'window_size': self.window_size,
            'max_history': self.max_history,
            'one_step': self.one_step,
            'normalize_window': self.normalize_window,
            'processed_y': self.processed_y,
            'transformation_dict': self.transformation_dict,
            'datepart_method': self.datepart_method,
            'regression_type': self.regression_type,
            'regression_model': self.regression_model,
        }
