# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:41:21 2022

@author: Colin

point, contextual, and collective. Point anomalies are single values
that fall within low-density regions of values, collective anomalies
indicate that a sequence of values is anomalous rather than any
single value by itself, and contextual anomalies are single values
that do not fall within low-density regions yet are anomalous with
regard to local values - https://arxiv.org/pdf/1802.04431.pdf
"""
import random
import numpy as np
import pandas as pd
from autots.tools.percentile import nan_quantile
from autots.tools.thresholding import NonparametricThreshold, nonparametric
from autots.tools.calendar import (
    gregorian_to_chinese,
    gregorian_to_islamic,
    gregorian_to_hebrew,
    gregorian_to_hindu,
)


try:
    from joblib import Parallel, delayed

    joblib_present = True
except Exception:
    joblib_present = False
# because this attempts to make sklearn optional for overall usage
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    from sklearn.svm import OneClassSVM
    from sklearn.mixture import GaussianMixture
    from scipy.stats import chi2, norm, gamma, uniform, laplace, cauchy, beta
except Exception:
    pass


def sk_outliers(df, method, method_params={}):
    """scikit-learn outlier methods wrapper."""
    if method == "IsolationForest":
        model = IsolationForest(n_jobs=1, **method_params)  # n_estimators=200
        res = model.fit_predict(df)
        scores = model.decision_function(df)
    elif method == "LOF":
        model = LocalOutlierFactor(n_jobs=1, **method_params)  # n_neighbors=5
        res = model.fit_predict(df)
        scores = model.negative_outlier_factor_ + 1.45
    elif method == "OneClassSVM":
        model = OneClassSVM(**method_params)  # n_neighbors=5
        res = model.fit_predict(df)
        scores = model.decision_function(df)
    elif method == "EE":
        if method_params['contamination'] == "auto":
            method_params['contamination'] = 0.1
        model = EllipticEnvelope(**method_params)
        res = model.fit_predict(df)
        scores = model.decision_function(df)
    elif method == "GaussianMixture":
        model = GaussianMixture(**method_params)
        model.fit(df)
        scores = -model.score_samples(df)
        responsibilities = model.predict_proba(df)
        max_responsibilities = responsibilities.max(axis=1)
        threshold = 0.05
        res = np.where(max_responsibilities < threshold, -1, 1)
        # res = np.where(scores > np.percentile(scores, 95), -1, 1)
    return pd.DataFrame({"anomaly": res}, index=df.index), pd.DataFrame(
        {"anomaly_score": scores}, index=df.index
    )


def loop_sk_outliers(df, method, method_params={}, n_jobs=1):
    """Multiprocessing on each series for multivariate outliers with sklearn."""
    parallel = True
    if n_jobs in [0, 1] or df.shape[1] < 5:
        parallel = False
    else:
        if not joblib_present:
            parallel = False

    # joblib multiprocessing to loop through series
    if parallel:
        df_list = Parallel(n_jobs=(n_jobs - 1))(
            delayed(sk_outliers)(
                df=df.iloc[:, i : i + 1],
                method=method,
                method_params=method_params,
            )
            for i in range(df.shape[1])
        )
    else:
        df_list = []
        for i in range(df.shape[1]):
            df_list.append(
                sk_outliers(
                    df=df.iloc[:, i : i + 1],
                    method=method,
                    method_params=method_params,
                )
            )
    complete = list(map(list, zip(*df_list)))
    res = pd.concat(complete[0], axis=1)
    res.index = df.index
    res.columns = df.columns
    scores = pd.concat(complete[1], axis=1)
    scores.index = df.index
    scores.columns = df.columns
    return res, scores


def zscore_survival_function(
    df,
    output="multivariate",
    method="zscore",
    distribution="norm",
    rolling_periods: int = 200,
    center: bool = True,
):
    """Take a dataframe and generate zscores and then generating survival probabilities (smaller = more outliery).

    Args:
        df (pd.DataFramme): wide style time series data (datetimeindex, series)
        output (str): univariate (1 series from all) or multivariate (all series input returned unique)
        method (str): zscore, rolling_zscore, mad (median abs dev)
        distribution (str): distribution to sample sf/outliers from
        rolling_period (int): >1, used for rolling_zscore period
        center (bool): passed to pd.rolling for rolliing_zscore, True for holiday detection, False for anomaly detection generally
    Returns:
        pd.Dataframe of p-values
    """
    if method == "zscore":
        residual_score = np.abs((df - df.mean(axis=0))) / df.std(axis=0)
    elif method == "rolling_zscore":
        df_rolling = df.rolling(rolling_periods, min_periods=1, center=center)
        residual_score = np.abs((df - df_rolling.mean())) / df_rolling.std()
    elif method == "mad":
        median_diff = np.abs((df - df.median(axis=0)))
        residual_score = median_diff / median_diff.mean(axis=0)
    elif method == "med_diff":
        median_diff = df.diff().median()
        residual_score = (df.diff().fillna(0) / median_diff).abs()
    elif method == "max_diff":
        max_diff = df.diff().max()
        residual_score = (df.diff().fillna(0) / max_diff).abs()
    else:
        raise ValueError("zscore method not recognized")

    if output == "univariate":
        dof = df.shape[1]
        residual_score = residual_score.sum(axis=1)
        columns = ["p_values"]
    elif output == "multivariate":
        dof = 1
        columns = df.columns
    else:
        raise ValueError("zscore sf `output` arg not recognized")

    # chi2, nbinom, erlang, gamma, poisson, maxwell, [laplace, cosine, norm, arcsine, uniform]
    if distribution == "norm":
        return pd.DataFrame(
            norm.sf(residual_score, dof), index=df.index, columns=columns
        )
    elif distribution == "gamma":
        return pd.DataFrame(
            gamma.sf(residual_score, dof), index=df.index, columns=columns
        )
    elif distribution == "beta":
        return pd.DataFrame(
            beta.sf(residual_score, 0.5, 2, scale=dof), index=df.index, columns=columns
        )
    elif distribution == "chi2":
        return pd.DataFrame(
            chi2.sf(residual_score, dof), index=df.index, columns=columns
        )
    elif distribution == "cauchy":
        return pd.DataFrame(
            cauchy.sf(residual_score, dof), index=df.index, columns=columns
        )
    elif distribution == "laplace":
        return pd.DataFrame(laplace.sf(residual_score), index=df.index, columns=columns)
    elif distribution == "uniform":
        return pd.DataFrame(
            uniform.sf(residual_score, dof), index=df.index, columns=columns
        )
    else:
        raise ValueError("zscore sf `distribution` arg not recognized")


def limits_to_anomalies(
    df,
    output,
    upper_limit,
    lower_limit,
    method_params=None,
):
    scores = zscore_survival_function(
        np.minimum(abs(df - upper_limit), abs(df - lower_limit)),
        output=output,
        method=method_params.get("threshold_method", "zscore"),
        distribution=method_params.get("distribution", "gamma"),
        center=method_params.get('center', False),
    )
    if output == "univariate":
        alpha = method_params.get("alpha", 0.05)
        res = pd.DataFrame(
            np.where(scores < alpha, -1, 1),
            index=df.index,
            columns=["p_values"],
        )
    else:
        res = pd.DataFrame(
            np.where((df >= upper_limit) | (df <= lower_limit), -1, 1),
            index=df.index,
            columns=df.columns,
        )
    return res, scores


def values_to_anomalies(df, output, threshold_method, method_params, n_jobs=1):
    cols = ["anomaly_score"] if output == "univariate" else df.columns
    if threshold_method == "nonparametric":
        return nonparametric_multivariate(
            df=df, output=output, method_params=method_params, n_jobs=n_jobs
        )
    elif threshold_method in ["minmax"]:
        alpha = method_params.get("alpha", 0.05)
        df_abs = df.abs()
        if output == "univariate":
            df_abs = df_abs.sum(axis=1).to_frame()
            scores = 1 - (
                (df_abs - df_abs.min(axis=0))
                / (df_abs.max(axis=0) - df_abs.min(axis=0)).replace(0, 1)
            )
            res = pd.DataFrame(
                np.where(scores < alpha, -1, 1),
                index=df.index,
                columns=cols,
            )
        elif output == "multivariate":
            scores = 1 - (
                (df_abs - df_abs.min(axis=0))
                / (df_abs.max(axis=0) - df_abs.min(axis=0)).replace(0, 1)
            )
            res = pd.DataFrame(
                np.where(scores < alpha, -1, 1),
                index=df.index,
                columns=cols,
            )
        return res, scores
    elif threshold_method in [
        "zscore",
        "rolling_zscore",
        "mad",
        "med_diff",
        "max_diff",
    ]:
        alpha = method_params.get("alpha", 0.05)
        distribution = method_params.get("distribution", "norm")
        rolling_periods = method_params.get("rolling_periods", 200)
        center = method_params.get("center", True)

        # p_values
        scores = zscore_survival_function(
            df,
            output,
            method=threshold_method,
            distribution=distribution,
            rolling_periods=rolling_periods,
            center=center,
        )
        res = pd.DataFrame(
            np.where(scores < alpha, -1, 1),
            index=df.index,
            columns=cols,
        )

        # distance between points and the mean
        # mean_array = residual_score.mean().to_frame().to_numpy().T
        # distance = cdist(residual_score.to_numpy(), mean_array, "mahalanobis")
        # distance = cdist(df.fillna(1).to_numpy(), df.rolling(100, min_periods=1, center=True).mean(), "mahalanobis")  # multivariate

        # calculate p-values
        # residual_score = np.abs((df - df.mean(axis=0))) / df.std(axis=0)
        # p_values2 = gamma.sf(residual_score, 1)
        # p_values5 = uniform.sf(residual_score.sum(axis=1), df.shape[1])
        # df.index[p_values2 < alpha]
        # df.index[p_values5 < alpha]
        return res, scores
    else:
        raise ValueError(f"outlier method {threshold_method} not recognized.")


def nonparametric_multivariate(df, output, method_params, n_jobs=1):
    if output == "univariate":
        df_abs = df.abs()
        scores = 1 - (
            (df_abs - df_abs.min(axis=0))
            / (df.abs().max(axis=0) - df_abs.min(axis=0)).replace(0, 1)
        )
        scores = np.abs((df - df.mean(axis=0))) / df.std(axis=0)
        mod = NonparametricThreshold(scores.to_numpy().flatten(), **method_params)
        mod.find_epsilon()
        mod.prune_anoms()
        i_anom = mod.i_anom
        if method_params.get("inverse", False):
            mod.find_epsilon(inverse=True)
            mod.prune_anoms(inverse=True)
            i_anom = np.unique(np.concatenate([i_anom, mod.i_anom_inv]))
        if df_abs.shape[1] < 2:
            scores = pd.DataFrame(
                mod.score_anomalies(),
                index=df.index,
                columns=["anomaly_score"],
            )
            res = pd.DataFrame(
                np.where(df.index.isin(df.index[i_anom]), -1, 1),
                index=df.index,
                columns=["anomaly"],
            )
        else:
            # this particular take of univariate is a bit awkward
            if mod.i_anom is not None:
                if mod.i_anom.size != 0:
                    cnts = np.unique(
                        np.tile(np.arange(df.shape[0]), df.shape[1])[mod.i_anom],
                        return_counts=True,
                    )
                else:
                    cnts = [[], []]
            else:
                cnts = [[], []]
            scores = (
                1
                - pd.Series(
                    cnts[1], index=df.index[cnts[0]], name="anomaly_score"
                ).reindex(df.index, fill_value=0)
                / df.shape[1]
            )
            res = pd.DataFrame(
                np.where(scores <= 0.9, -1, 1),
                index=df.index,
                columns=["anomaly"],
            )
        return res, scores
    else:
        parallel = True
        if n_jobs in [0, 1] or df.shape[1] < 1000:
            parallel = False
        else:
            if not joblib_present:
                parallel = False

        # joblib multiprocessing to loop through series
        if parallel:
            df_list = Parallel(n_jobs=(n_jobs - 1))(
                delayed(nonparametric)(
                    series=df.iloc[:, i],
                    method_params=method_params,
                )
                for i in range(df.shape[1])
            )
        else:
            df_list = []
            for i in range(df.shape[1]):
                df_list.append(
                    nonparametric(
                        series=df.iloc[:, i],
                        method_params=method_params,
                    )
                )
        complete = list(map(list, zip(*df_list)))
        res = pd.concat(complete[0], axis=1)
        res.index = df.index
        res.columns = df.columns
        scores = pd.concat(complete[1], axis=1)
        scores.index = df.index
        scores.columns = df.columns
        return res, scores


def detect_anomalies(
    df,
    output,
    method,
    transform_dict=None,
    method_params={},
    eval_period=None,
    n_jobs=1,
):
    """All will return -1 for anomalies.

    Args:
        output (str): 'multivariate' (each series unique outliers), or 'univariate' (all series together for one outlier flag per timestamp)
        transform_dict (dict): option but helpful, often datepart, differencing, or other standard AutoTS transformer params
        forecast_params (dict): used to backcast and identify 'unforecastable' values, required only for predict_interval method
        eval_periods (int): only use this length tail of data, currently only implemented for forecast_params forecasting if used
        n_jobs (int): multiprocessing jobs, used by some methods

    Returns:
        pd.DataFrame (classifications, -1 = outlier, 1 = not outlier), pd.DataFrame s(scores)
    """
    df_anomaly = df  # .copy()
    """
    if transform_dict is not None:
        model = GeneralTransformer(
            **transform_dict
        )  # DATEPART, LOG, SMOOTHING, DIFF, CLIP OUTLIERS with high z score
        df_anomaly = model.fit_transform(df_anomaly)
    """

    if method in ["IsolationForest", "LOF", "EE", "OneClassSVM", "GaussianMixture"]:
        if output == "univariate":
            res, scores = sk_outliers(df_anomaly, method, method_params)
        else:
            res, scores = loop_sk_outliers(df_anomaly, method, method_params, n_jobs)
    elif method in [
        "zscore",
        "rolling_zscore",
        "mad",
        "minmax",
        "med_diff",
        "max_diff",
    ]:
        res, scores = values_to_anomalies(df_anomaly, output, method, method_params)
    elif method == "GaussianMixtureBase":
        res, scores = gaussian_mixture(df, **method_params)
    elif method in ["IQR"]:
        iqr_thresh = method_params.get("iqr_threshold", 2.0)
        iqr_quantiles = method_params.get("iqr_quantiles", [0.25, 0.75])
        resid_q_0 = nan_quantile(df_anomaly, iqr_quantiles[0])
        resid_q_1 = nan_quantile(df_anomaly, iqr_quantiles[1])
        iqr = resid_q_1 - resid_q_0
        limit_0 = resid_q_0 - (iqr_thresh * iqr)
        limit_1 = resid_q_1 + (iqr_thresh * iqr)
        res, scores = limits_to_anomalies(
            df_anomaly,
            output=output,
            method_params=method_params,
            upper_limit=limit_1,
            lower_limit=limit_0,
        )
    elif method in ["nonparametric"]:
        res, scores = values_to_anomalies(
            df_anomaly,
            output,
            threshold_method=method,
            method_params=method_params,
            n_jobs=n_jobs,
        )
    elif method in ["prediction_interval"]:
        raise ValueError(
            "prediction_interval method only handled by AnomalyDetector class."
        )
    else:
        raise ValueError(f"{method} outlier method not recognized")

    return res, scores


available_methods = [
    "IsolationForest",  # (sklearn)
    "LOF",  # Local Outlier Factor (sklearn)
    "EE",  # Elliptical Envelope (sklearn)
    "OneClassSVM",
    "GaussianMixture",
    "zscore",
    "rolling_zscore",
    "mad",
    "minmax",
    "prediction_interval",  # ridiculously slow
    "IQR",
    "nonparametric",
    "med_diff",
    "max_diff",
    # "GaussianMixtureBase",
]
fast_methods = [
    "zscore",
    "rolling_zscore",
    "mad",
    "minmax",
    "IQR",
    "nonparametric",
    "med_diff",
    "max_diff",
]


def anomaly_new_params(method='random'):
    if method == "deep":
        method_choice = random.choices(
            available_methods,
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1],
        )[0]
    elif method == "fast":
        method_choice = random.choices(
            fast_methods, [0.4, 0.3, 0.1, 0.1, 0.4, 0.05, 0.1, 0.1]
        )[0]
    elif method in available_methods:
        method_choice = method
    else:
        method_choice = random.choices(
            [
                "LOF",
                "EE",
                "zscore",
                "rolling_zscore",  # Matt likes this one best
                "mad",
                "minmax",
                "IQR",
                "nonparametric",
                "IsolationForest",
                # "OneClassSVM",  # seems too slow at times
                "GaussianMixture",
                # "GaussianMixtureBase",
            ],  # Isolation Forest is good but slower (parallelized also)
            [0.05, 0.1, 0.25, 0.3, 0.1, 0.1, 0.2, 0.1, 0.05, 0.05],
        )[0]

    if method_choice == "IsolationForest":
        method_params = {
            'contamination': random.choices(['auto', 0.1, 0.05], [0.8, 0.1, 0.1])[0],
            'n_estimators': random.choices([20, 50, 100, 200], [0.3, 0.4, 0.2, 0.01])[
                0
            ],
            'max_features': random.choices([1.0, 0.5], [0.9, 0.1])[0],
            'bootstrap': random.choices([False, True], [0.9, 0.1])[0],
        }
    elif method_choice == "LOF":
        method_params = {
            'contamination': random.choices(['auto', 0.1, 0.05], [0.8, 0.1, 0.1])[0],
            'n_neighbors': random.choices([3, 5, 10, 20], [0.3, 0.4, 0.3, 0.1])[0],
            'metric': random.choices(['minkowski', 'canberra'], [0.9, 0.1])[0],
        }
    elif method_choice == "OneClassSVM":
        method_params = {
            'kernel': random.choices(
                ['linear', "poly", "rbf", "sigmoid"], [0.1, 0.1, 0.4, 0.1]
            )[0],
            'degree': random.choices([3, 5, 10, 20], [0.3, 0.4, 0.3, 0.1])[0],
            'gamma': random.choices(['scale', 'auto'], [0.5, 0.5])[0],
            'shrinking': random.choices([True, False], [0.5, 0.5])[0],
            'nu': random.choices([0.3, 0.5, 0.7, 0.9, 0.1], [0.3, 0.5, 0.3, 0.1, 0.1])[
                0
            ],
        }
    elif method_choice == "EE":
        method_params = {
            'contamination': random.choices(
                [0.02, 0.1, 0.05, 0.15], [0.1, 0.8, 0.05, 0.05]
            )[0],
            'assume_centered': random.choices([False, True], [0.9, 0.1])[0],
            'support_fraction': random.choices([None, 0.2, 0.8], [0.9, 0.1, 0.1])[0],
        }
    elif method_choice == "GaussianMixture":
        method_params = {
            'n_components': random.choices([2, 3, 4, 5], [0.2, 0.3, 0.3, 0.2])[0],
            'covariance_type': random.choices(
                ['full', 'tied', 'diag', 'spherical'], [0.4, 0.3, 0.2, 0.1]
            )[0],
            'tol': random.choices([1e-3, 1e-4, 1e-5], [0.5, 0.3, 0.2])[0],
            'reg_covar': random.choices([1e-6, 1e-5, 1e-4], [0.3, 0.4, 0.3])[0],
            'max_iter': random.choices([100, 200, 300], [0.3, 0.4, 0.3])[0],
        }
    elif method_choice == "GaussianMixtureBase":
        method_params = {
            'n_components': random.choices([2, 3, 4, 5], [0.2, 0.3, 0.3, 0.2])[0],
            'tol': random.choices([1e-3, 1e-4, 1e-5], [0.5, 0.3, 0.2])[0],
            'max_iter': random.choices([50, 100, 200], [0.3, 0.5, 0.2])[0],
            "responsibility_threshold": random.choices(
                [0.05, 0.01, 0.1], [0.3, 0.2, 0.2]
            )[0],
        }
    elif method_choice == "zscore":
        method_params = {
            'distribution': random.choices(
                ['norm', 'gamma', 'chi2', 'uniform', "laplace", "cauchy"],
                [0.4, 0.2, 0.3, 0.2, 0.05, 0.05],
            )[0],
            'alpha': random.choices(
                [0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.14],
                [0.005, 0.1, 0.6, 0.1, 0.2, 0.05, 0.05],
            )[0],
        }
    elif method_choice == "rolling_zscore":
        method_params = {
            'distribution': random.choices(
                ['norm', 'gamma', 'chi2', 'uniform', "laplace", "cauchy"],
                [0.4, 0.2, 0.2, 0.2, 0.1, 0.1],
            )[0],
            'alpha': random.choices(
                [0.01, 0.03, 0.05, 0.1, 0.2, 0.4], [0.1, 0.1, 0.8, 0.1, 0.1, 0.01]
            )[0],
            'rolling_periods': random.choice([28, 90, 200, 300]),
            'center': random.choice([True, False]),
        }
    elif method_choice in ["med_diff", "max_diff"]:
        method_params = {
            'distribution': random.choices(
                ['norm', 'gamma', 'chi2', 'uniform', "laplace", "cauchy", "beta"],
                [0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
            )[0],
            'alpha': random.choices(
                [0.01, 0.03, 0.05, 0.1, 0.2, 0.6], [0.1, 0.1, 0.8, 0.1, 0.1, 0.05]
            )[0],
        }
    elif method_choice == "mad":
        method_params = {
            'distribution': random.choices(
                ['norm', 'gamma', 'chi2', 'uniform'], [0.4, 0.2, 0.2, 0.2]
            )[0],
            'alpha': random.choices([0.03, 0.05, 0.1], [0.1, 0.8, 0.1])[0],
        }
    elif method_choice == "minmax":
        method_params = {
            'alpha': random.choices([0.03, 0.05, 0.1], [0.3, 0.6, 0.1])[0],
        }
    elif method_choice == "prediction_interval":
        method_params = {"prediction_interval": random.choice([0.9, 0.99])}
    elif method_choice == "IQR":
        iqr_threshold = random.choices(
            [1.5, 2.0, 2.5, 3.0, 'other'], [0.1, 0.2, 0.2, 0.2, 0.1]
        )[0]
        if iqr_threshold == "other":
            iqr_threshold = random.randint(2, 5) + (random.randint(0, 10) / 10)
        method_params = {
            'iqr_threshold': iqr_threshold,
            'iqr_quantiles': random.choices([[0.25, 0.75], [0.4, 0.6]], [0.8, 0.2])[0],
        }
    elif method_choice == "nonparametric":
        method_params = {
            'p': random.choices(
                [None, 0.1, 0.05, 0.02, 0.25], [0.8, 0.1, 0.1, 0.01, 0.1]
            )[0],
            'z_init': random.choices([1.5, 2.0, 2.5], [0.3, 0.4, 0.3])[0],
            'z_limit': random.choices([10, 12], [0.5, 0.5])[0],
            'z_step': random.choice([0.5, 0.25]),
            'inverse': random.choices([False, True], [0.9, 0.1])[0],
            'max_contamination': random.choices([0.25, 0.1, 0.05], [0.9, 0.05, 0.05])[
                0
            ],
            'mean_weight': random.choices([1, 25, 100, 200], [0.1, 0.7, 0.1, 0.1])[0],
            'sd_weight': random.choices([1, 25, 100, 200], [0.1, 0.7, 0.1, 0.1])[0],
            'anomaly_count_weight': random.choices([0.5, 1.0, 2.0], [0.05, 0.9, 0.05])[
                0
            ],
        }
    transform_dict = random.choices(
        [
            None,
            'random',
            {
                "transformations": {0: "DatepartRegression"},
                "transformation_params": {
                    0: {
                        "datepart_method": "simple_3",
                        "regression_model": {
                            "model": "ElasticNet",
                            "model_params": {},
                        },
                    }
                },
            },
            {
                "transformations": {0: "DatepartRegression"},
                "transformation_params": {
                    0: {
                        "datepart_method": "simple_3",
                        "regression_model": {
                            "model": "FastRidge",
                            "model_params": {},
                        },
                    }
                },
            },
            {
                "transformations": {0: "DatepartRegression"},
                "transformation_params": {
                    0: {
                        "datepart_method": "simple_3",
                        "regression_model": {
                            "model": "DecisionTree",
                            "model_params": {
                                "max_depth": None,
                                "min_samples_split": 0.1,
                            },
                        },
                    }
                },
            },
            {
                "transformations": {0: "DifferencedTransformer"},
                "transformation_params": {0: {}},
            },
            {
                "fillna": None,
                "transformations": {"0": "EWMAFilter"},
                "transformation_params": {
                    "0": {"span": 7},
                },
            },
            {
                "fillna": None,
                "transformations": {"0": "ClipOutliers"},
                "transformation_params": {
                    "0": {"method": "clip", "std_threshold": 6},
                },
            },
        ],
        [0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    )[0]
    return method_choice, method_params, transform_dict


def create_dates_df(dates):
    """Take a pd.DatetimeIndex and create simple date parts."""
    dates_df = pd.DataFrame(
        {"month": dates.month, "day": dates.day, "dayofweek": dates.dayofweek},
        index=dates,
    ).rename_axis(index="date")
    dates_df["weekofmonth"] = (dates_df["day"] - 1) // 7 + 1
    dates_df['weekfromend'] = (dates_df['day'] - dates.daysinmonth) // -7
    return dates_df


def anomaly_df_to_holidays(
    anomaly_df,
    actuals=None,
    anomaly_scores=None,
    threshold=0.8,
    min_occurrences=2,
    splash_threshold=0.65,
    use_dayofmonth_holidays=True,
    use_wkdom_holidays=True,
    use_wkdeom_holidays=False,
    use_lunar_holidays=False,
    use_lunar_weekday=False,
    use_islamic_holidays=False,
    use_hebrew_holidays=False,
    use_hindu_holidays=False,
):
    if isinstance(anomaly_df, pd.Series):
        stacked = anomaly_df.copy()  # [anomaly_df == -1]
        stacked.index.name = 'date'
        stacked = pd.concat(
            [stacked], keys=["all"], names=["series"]
        )  # .reorder_levels([1, 0])
        # anomalies = anomaly_df.index[(anomaly_df == -1).iloc[:, 0]]
    else:
        anomaly_df.columns.name = "series"
        stacked = anomaly_df.stack()
        # stacked = stacked[stacked == -1]
        # stacked = stacked.reset_index()
        # anomalies = stacked.index[(stacked == -1)].get_level_values('date').unique()
    stacked.name = "count"
    # all anomaly values MUST be == -1
    stacked = (
        stacked.where(stacked == -1, 0).abs().rename_axis(index=['date', 'series'])
    )
    agg_dict = {"count": 'sum', "occurrence_rate": 'mean'}
    if isinstance(stacked, pd.Series):
        stacked = stacked.to_frame()
    if actuals is not None:
        mod_act = (
            actuals.stack().rename("avg_value").rename_axis(index=['date', 'series'])
        )
        # stacked = stacked.merge(mod_act, left_index=True, right_index=True)
        # I am not sure if these indexes will always be aligned, could be the source of trouble
        stacked = pd.concat([stacked, mod_act], axis=1)
        agg_dict['avg_value'] = 'mean'
    if anomaly_scores is not None:
        mod_anom = (
            anomaly_scores.stack()
            .rename("avg_anomaly_score")
            .rename_axis(index=['date', 'series'])
        )
        # print(f"mod_anom shape {mod_anom.shape}, stacked shape {stacked.shape} and {mod_anom.index} and {stacked.index}")
        # stacked = stacked.merge(mod_anom, left_index=True, right_index=True)
        stacked = pd.concat([stacked, mod_anom], axis=1)
        agg_dict['avg_anomaly_score'] = 'mean'

    dates = stacked.index.get_level_values('date').unique()
    try:
        year_range = dates.year.max() - dates.year.min() + 1
    except Exception as e:
        raise Exception(f"unrecognized dates: {dates}") from e

    if year_range <= 1:
        raise ValueError("more than 1 year of data is required for holiday detection.")

    dates_df = create_dates_df(dates)
    # dates_df['count'] = 1
    dates_df = dates_df.merge(stacked, left_index=True, right_index=True, how="outer")
    dates_df['occurrence_rate'] = dates_df['count']

    if use_dayofmonth_holidays:
        day_holidays = dates_df.groupby(["series", "month", "day"]).agg(agg_dict)
        if splash_threshold is not None:
            day_holidays = day_holidays.loc[
                lambda df: (
                    (df["occurrence_rate"] >= threshold)
                    | (
                        df['occurrence_rate']
                        .rolling(3, min_periods=1, center=True)
                        .mean()
                        > splash_threshold
                    )
                )
                & (df["count"] >= min_occurrences),
            ].reset_index(drop=False)
        else:
            day_holidays = day_holidays.loc[
                lambda df: (df["occurrence_rate"] >= threshold)
                & (df["count"] >= min_occurrences),
            ].reset_index(drop=False)
        day_holidays['holiday_name'] = (
            'dom_'
            + day_holidays['month'].astype(str).str.pad(2, side='left', fillchar="0")
            + "_"
            + day_holidays['day'].astype(str).str.pad(2, side='left', fillchar="0")
        )
        # replace a few major names
        day_holidays['holiday_name'] = day_holidays['holiday_name'].replace(
            {
                'dom_12_25': "Christmas",
                'dom_12_26': "BoxingDay",
                'dom_12_24': "ChristmasEve",
                'dom_07_01': "CanadaDay",
                'dom_07_04': "July4th",
                'dom_07_14': "BastilleDay",
                'dom_01_26': "RDIndia_AustraliaDay",
                'dom_01_01': "NewYearsDay",
                'dom_12_31': "NewYearsEve",
                'dom_02_14': 'ValentinesDay',
                'dom_10-31': 'Halloween',
                'dom_11-11': 'ArmisticeDay',
                "dom_04_22": "EarthDay",
            }
        )
    else:
        day_holidays = None
    if use_wkdom_holidays:
        wkdom_holidays = dates_df.groupby(
            ["series", "month", "weekofmonth", "dayofweek"]
        ).agg(agg_dict)
        if splash_threshold is not None:
            wkdom_holidays = wkdom_holidays.loc[
                lambda df: (
                    (df["occurrence_rate"] >= threshold)
                    | (
                        df['occurrence_rate']
                        .rolling(3, min_periods=1, center=True)
                        .mean()
                        > splash_threshold
                    )
                )
                & (df["count"] >= min_occurrences),
            ].reset_index(drop=False)
        else:
            wkdom_holidays = wkdom_holidays.loc[
                lambda df: (df["occurrence_rate"] >= threshold)
                & (df["count"] >= min_occurrences),
            ].reset_index(drop=False)
        wkdom_holidays['holiday_name'] = (
            'wkdom_'
            + wkdom_holidays['month'].astype(str).str.pad(2, side='left', fillchar="0")
            + "_"
            + wkdom_holidays['weekofmonth'].astype(str)
            + "_"
            + wkdom_holidays['dayofweek'].astype(str)
        )
        wkdom_holidays['holiday_name'] = wkdom_holidays['holiday_name'].replace(
            {
                'wkdom_11_4_4': "BlackFriday",
                'wkdom_11_4_3': "Thanksgiving",
                'wkdom_05_2_6': "MothersDay",
                'wkdom_06_3_6': "FathersDay",
                'wkdom_09_1_0': "LaborDay",
            }
        )
    else:
        wkdom_holidays = None
    if use_wkdeom_holidays:
        wkdeom_holidays = (
            dates_df.groupby(["series", "month", "weekfromend", "dayofweek"])
            .agg(agg_dict)
            .loc[
                lambda df: (df["occurrence_rate"] >= threshold)
                & (df["count"] >= min_occurrences),
            ]
        ).reset_index(drop=False)
        wkdeom_holidays['holiday_name'] = (
            'wkdeom_'
            + wkdeom_holidays['month'].astype(str).str.pad(2, side='left', fillchar="0")
            + "_"
            + wkdeom_holidays['weekfromend'].astype(str)
            + "_"
            + wkdeom_holidays['dayofweek'].astype(str)
        )
        wkdeom_holidays['holiday_name'] = wkdeom_holidays['holiday_name'].replace(
            {
                'wkdeom_05_0_0': "MemorialDay",
            }
        )
    else:
        wkdeom_holidays = None
    lunar_weekday = None
    if use_lunar_holidays:
        lunar_df = gregorian_to_chinese(dates)
        lunar_df["weekofmonth"] = (lunar_df["lunar_day"] - 1) // 7 + 1
        lunar_df['dayofweek'] = lunar_df.index.dayofweek
        # lunar_df['count'] = 1
        lunar_df = lunar_df.merge(
            stacked, left_index=True, right_index=True, how="outer"
        )
        lunar_df['occurrence_rate'] = lunar_df['count']
        lunar_holidays = (
            lunar_df.groupby(["series", "lunar_month", "lunar_day"])
            .agg(agg_dict)
            .loc[
                lambda df: (df["occurrence_rate"] >= threshold)
                & (df["count"] >= min_occurrences),
            ]
        ).reset_index(drop=False)
        lunar_holidays['holiday_name'] = (
            'lunar_'
            + lunar_holidays['lunar_month']
            .astype(str)
            .str.pad(2, side='left', fillchar="0")
            + "_"
            + lunar_holidays['lunar_day']
            .astype(str)
            .str.pad(2, side='left', fillchar="0")
        )
        lunar_holidays['holiday_name'] = lunar_holidays['holiday_name'].replace(
            {
                'lunar_01_01': "LunarNewYear",
            }
        )
        if use_lunar_weekday:
            lunar_weekday = (
                lunar_df.groupby(["series", "lunar_month", "weekofmonth", 'dayofweek'])
                .agg(agg_dict)
                .loc[
                    lambda df: (df["occurrence_rate"] >= threshold)
                    & (df["count"] >= min_occurrences),
                ]
            ).reset_index(drop=False)
            lunar_weekday['holiday_name'] = (
                'lunarwkd_'
                + lunar_weekday['lunar_month']
                .astype(str)
                .str.pad(2, side='left', fillchar="0")
                + "_"
                + lunar_weekday['weekofmonth'].astype(str)
                + "_"
                + lunar_weekday['dayofweek'].astype(str)
            )
    else:
        lunar_holidays = None
    if use_islamic_holidays:
        islamic_df = gregorian_to_islamic(dates)
        # islamic_df['count'] = 1
        islamic_df = islamic_df.merge(
            stacked, left_index=True, right_index=True, how="outer"
        )
        islamic_df['occurrence_rate'] = islamic_df['count']
        islamic_holidays = (
            islamic_df.groupby(["series", "month", "day"])
            .agg(agg_dict)
            .loc[
                lambda df: (df["occurrence_rate"] >= threshold)
                & (df["count"] >= min_occurrences),
            ]
        ).reset_index(drop=False)
        islamic_holidays['holiday_name'] = (
            'islamic_'
            + islamic_holidays['month']
            .astype(str)
            .str.pad(2, side='left', fillchar="0")
            + "_"
            + islamic_holidays['day'].astype(str).str.pad(2, side='left', fillchar="0")
        )
    else:
        islamic_holidays = None
    if use_hebrew_holidays:
        hebrew_df = gregorian_to_hebrew(dates)
        hebrew_df.index.name = "date"
        # hebrew_df['count'] = 1
        hebrew_df = hebrew_df.merge(
            stacked, left_index=True, right_index=True, how="outer"
        )
        hebrew_df['occurrence_rate'] = hebrew_df['count']
        hebrew_holidays = (
            hebrew_df.groupby(["series", "month", "day"])
            .agg(agg_dict)
            .loc[
                lambda df: (df["occurrence_rate"] >= threshold)
                & (df["count"] >= min_occurrences),
            ]
        ).reset_index(drop=False)
        hebrew_holidays['holiday_name'] = (
            'hebrew_'
            + hebrew_holidays['month'].astype(str).str.pad(2, side='left', fillchar="0")
            + "_"
            + hebrew_holidays['day'].astype(str).str.pad(2, side='left', fillchar="0")
        )
    else:
        hebrew_holidays = None
    if use_hindu_holidays:
        hindu_df = gregorian_to_hindu(dates)
        hindu_df.index.name = "date"
        hindu_df = hindu_df.merge(
            stacked, left_index=True, right_index=True, how="outer"
        )
        hindu_df['occurrence_rate'] = hindu_df['count']

        # Group by Hindu calendar components to find significant dates
        hindu_holidays = (
            hindu_df.groupby(["series", "hindu_month_number", "lunar_day"])
            .agg(agg_dict)
            .loc[
                lambda df: (df["occurrence_rate"] >= threshold)
                & (df["count"] >= min_occurrences),
            ]
        ).reset_index(drop=False)

        hindu_holidays['holiday_name'] = (
            'hindu_'
            + hindu_holidays['hindu_month_number']
            .astype(str)
            .str.pad(2, side='left', fillchar="0")
            + "_"
            + hindu_holidays['lunar_day']
            .astype(str)
            .str.pad(2, side='left', fillchar="0")
        )
    else:
        hindu_holidays = None
    return (
        day_holidays,
        wkdom_holidays,
        wkdeom_holidays,
        lunar_holidays,
        lunar_weekday,
        islamic_holidays,
        hebrew_holidays,
        hindu_holidays,
    )


def dates_to_holidays(
    dates,
    df_cols,
    style="long",
    holiday_impacts="value",
    day_holidays=None,
    wkdom_holidays=None,
    wkdeom_holidays=None,
    lunar_holidays=None,
    lunar_weekday=None,
    islamic_holidays=None,
    hebrew_holidays=None,
    hindu_holidays=None,
    max_features: int = None,
):
    """Populate date information for a given pd.DatetimeIndex.

    Args:
        dates (pd.DatetimeIndex): list of dates
        day_holidays (pd.DataFrame): list of month/day holidays. Pass None if not available
        style (str): option for how to return information
            "long" - return date, name, series for all holidays in a long style dataframe
            "impact" - returns dates, series with values of sum of impacts (if given) or joined string of holiday names
            'flag' - return dates, holidays flag, (is not 0-1 but rather sum of input series impacted for that holiday and day)
            'prophet' - return format required for prophet. Will need to be filtered on `series` for multivariate case
            'series_flag' - dates, series 0/1 for if holiday occurred in any calendar
        holiday_impacts (dict): a dict passed to .replace contaning values for holiday_names, or str 'value' or 'anomaly_score'
    """
    # need index in column for merge
    dates_df = create_dates_df(dates).reset_index(drop=False)

    if style in ['long', 'flag', 'prophet']:
        result = []
    elif style in ["impact", 'series_flag']:
        result = pd.DataFrame(0, columns=df_cols, index=dates)
    else:
        raise ValueError("`style` arg not recognized in dates_to_holidays")
    for holiday_df in [
        day_holidays,
        wkdom_holidays,
        wkdeom_holidays,
        lunar_holidays,
        lunar_weekday,
        islamic_holidays,
        hebrew_holidays,
        hindu_holidays,
    ]:
        if holiday_df is not None:
            if not holiday_df.empty:
                # various memory usage reduction is done by drop_duplicates, drop(columns)
                if style == 'flag':
                    # this might overwrite upstream holiday. If using multiple styles, problem
                    holiday_df = holiday_df.drop_duplicates(
                        subset=holiday_df.columns.difference(
                            [
                                'series',
                                'count',
                                'occurrence_rate',
                                'avg_value',
                                'avg_anomaly_score',
                            ]
                        )
                    )
                    drop_colz = [
                        'count',
                        'occurrence_rate',
                        'avg_value',
                        'avg_anomaly_score',
                    ]
                else:
                    drop_colz = ['count', 'occurrence_rate']
                # handle the different holiday calendars
                if "lunar_month" in holiday_df.columns:
                    lunar_dates = gregorian_to_chinese(dates)
                    if "weekofmonth" in holiday_df.columns:
                        on = ["lunar_month", "weekofmonth", 'dayofweek']
                        lunar_dates["weekofmonth"] = (
                            lunar_dates["lunar_day"] - 1
                        ) // 7 + 1
                        lunar_dates['dayofweek'] = lunar_dates.index.dayofweek
                    else:
                        on = ["lunar_month", "lunar_day"]
                    populated_holidays = (
                        lunar_dates.reset_index(drop=False)
                        .drop(columns=lunar_dates.columns.difference(on + ['date']))
                        .merge(
                            holiday_df.drop(columns=drop_colz, errors='ignore'),
                            on=on,
                            how="left",
                        )
                    )
                elif "hindu_month_number" in holiday_df.columns:
                    lunar_dates = gregorian_to_hindu(dates)
                    if "weekofmonth" in holiday_df.columns:
                        on = ["hindu_month_number", "lunar_day", 'weekofmonth']
                        lunar_dates["weekofmonth"] = (
                            lunar_dates["lunar_day"] - 1
                        ) // 7 + 1
                        lunar_dates['dayofweek'] = lunar_dates.index.dayofweek
                    else:
                        on = ["hindu_month_number", "lunar_day"]
                    populated_holidays = (
                        lunar_dates.reset_index(drop=False)
                        .drop(columns=lunar_dates.columns.difference(on + ['date']))
                        .merge(
                            holiday_df.drop(columns=drop_colz, errors='ignore'),
                            on=on,
                            how="left",
                        )
                    )
                else:
                    on = ['month', 'day']
                    if "weekofmonth" in holiday_df.columns:
                        on = ["month", "weekofmonth", "dayofweek"]
                    elif "weekfromend" in holiday_df.columns:
                        on = ["month", "weekfromend", "dayofweek"]
                    sample = holiday_df['holiday_name'].iloc[0]
                    if "hebrew" in sample:
                        hdates = gregorian_to_hebrew(dates)
                        populated_holidays = (
                            hdates.drop(
                                columns=hdates.columns.difference(on + ['date']),
                                errors='ignore',
                            )
                            .reset_index(drop=False)
                            .merge(
                                holiday_df.drop(columns=drop_colz, errors='ignore'),
                                on=on,
                                how="left",
                            )
                        )
                    elif "islamic" in sample:
                        idates = gregorian_to_islamic(dates)
                        populated_holidays = (
                            idates.drop(
                                columns=idates.columns.difference(on + ['date']),
                                errors='ignore',
                            )
                            .reset_index(drop=False)
                            .merge(
                                holiday_df.drop(columns=drop_colz, errors='ignore'),
                                on=on,
                                how="left",
                            )
                        )
                    elif "hindu" in sample:
                        idates = gregorian_to_hindu(dates)
                        populated_holidays = (
                            idates.drop(
                                columns=idates.columns.difference(on + ['date']),
                                errors='ignore',
                            )
                            .reset_index(drop=False)
                            .merge(
                                holiday_df.drop(columns=drop_colz, errors='ignore'),
                                on=on,
                                how="left",
                            )
                        )
                    else:
                        populated_holidays = dates_df.drop(
                            columns=dates_df.columns.difference(on + ['date'])
                        ).merge(
                            holiday_df.drop(columns=drop_colz, errors='ignore'),
                            on=on,
                            how="left",
                        )
                # reorg results depending on style
                if style == "flag":
                    populated_holidays = populated_holidays.drop_duplicates(
                        subset=['date', 'holiday_name']
                    )
                    populated_holidays['holiday_name'] = pd.Categorical(
                        populated_holidays['holiday_name'],
                        categories=holiday_df['holiday_name'].unique(),
                        ordered=True,
                    )
                    if max_features is not None:
                        # rarest first
                        frequent_categories = (
                            populated_holidays['holiday_name']
                            .value_counts(ascending=True)
                            .index[:max_features]
                        )
                        populated_holidays['holiday_name'] = populated_holidays[
                            'holiday_name'
                        ].apply(lambda x: x if x in frequent_categories else 'Other')
                    result_per_holiday = pd.get_dummies(
                        populated_holidays['holiday_name'],
                        dtype=float,
                    )
                    """
                    lb = LabelBinarizer()
                    result_per_holiday = pd.DataFrame(
                        lb.fit_transform(populated_holidays['holiday_name']),
                        # columns=lb.classes_, 
                        index=populated_holidays.index
                    )
                    """
                    # result_per_holiday = populated_holidays['holiday_name'].astype('category').cat.codes
                    result_per_holiday.index = populated_holidays['date']
                    result.append(result_per_holiday.groupby(level=0).sum())
                elif style in ["impact", 'series_flag']:
                    temp = populated_holidays.pivot(
                        index='date', columns='series', values='holiday_name'
                    ).reindex(columns=df_cols)
                    if style == "series_flag":
                        result = result + temp.where(temp.isnull(), 1).fillna(0.0)
                    else:
                        if isinstance(holiday_impacts, dict):
                            result = result + temp.replace(holiday_impacts).astype(
                                float
                            )
                        # if multiple holidays overlap, take the GREATER of the them, but don't ADD
                        # this also rather assumes most holidays will have a positive impact on values
                        elif holiday_impacts == "anomaly_score":
                            temp2 = holiday_df.pivot(
                                index='holiday_name',
                                columns='series',
                                values='avg_anomaly_score',
                            ).reindex(columns=df_cols)
                            replace_dict = temp2.to_dict()
                            result = np.maximum(
                                result,
                                temp.replace(replace_dict).astype(float).fillna(0),
                            )
                        elif holiday_impacts == "value":
                            temp2 = holiday_df.pivot(
                                index='holiday_name',
                                columns='series',
                                values='avg_value',
                            ).reindex(columns=df_cols)
                            replace_dict = temp2.to_dict()
                            result = np.maximum(
                                result,
                                temp.replace(replace_dict).astype(float).fillna(0),
                            )
                        else:
                            result = result.replace(0, "") + (
                                temp.astype(str) + ","
                            ).replace("nan,", "")
                else:
                    result.append(populated_holidays)
    if isinstance(result, list):
        if not result:
            if style == "flag":
                return pd.DataFrame(index=dates)
            else:
                return pd.DataFrame(
                    columns=[
                        'ds',
                        'date',
                        'holiday',
                        'holiday_name',
                        'series',
                        'lower_window',
                        'upper_window',
                    ]
                )
    if style in ['long', 'prophet']:
        result = pd.concat(result, axis=0)
    elif style == "flag":
        return pd.concat(result, axis=1)
    elif style == "series_flag":
        return result.clip(upper=1.0).astype(int)
    if style == "prophet":
        prophet_holidays = pd.DataFrame(
            {
                'ds': result['date'],
                'holiday': result['holiday_name'],
                'lower_window': 0,
                'upper_window': 0,
                'series': result['series'],
            }
        )  # needs to cover future, and at the time of object creation
        return prophet_holidays[~pd.isnull(prophet_holidays['holiday'])]
    else:
        return result


def holiday_new_params(method='random'):
    return {
        'threshold': random.choices([1.0, 0.9, 0.8, 0.7], [0.1, 0.4, 0.4, 0.1])[0],
        'splash_threshold': random.choices(
            [None, 0.85, 0.65, 0.4], [0.95, 0.05, 0.05, 0.05]
        )[0],
        'use_dayofmonth_holidays': random.choices([True, False], [0.95, 0.05])[0],
        'use_wkdom_holidays': random.choices([True, False], [0.9, 0.1])[0],
        'use_wkdeom_holidays': random.choices([True, False], [0.05, 0.95])[0],
        'use_lunar_holidays': random.choices([True, False], [0.1, 0.9])[0],
        'use_lunar_weekday': random.choices([True, False], [0.05, 0.95])[0],
        'use_islamic_holidays': random.choices([True, False], [0.1, 0.9])[0],
        'use_hebrew_holidays': random.choices([True, False], [0.1, 0.9])[0],
        'use_hindu_holidays': random.choices([True, False], [0.1, 0.9])[0],
    }


def gaussian_mixture(
    df, n_components=2, tol=1e-3, max_iter=100, responsibility_threshold=0.05
):
    from scipy.stats import multivariate_normal

    n, d = df.shape
    data = df.fillna(0).to_numpy()

    # Normalize the data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    np.random.seed(42)
    means = np.random.rand(n_components, d)
    covariances = np.array([np.eye(d)] * n_components)
    weights = np.ones(n_components) / n_components

    log_likelihood = -np.inf

    for iteration in range(max_iter):
        # E-step: compute responsibilities
        responsibilities = np.zeros((n, n_components))

        for i in range(n_components):
            responsibilities[:, i] = weights[i] * multivariate_normal.pdf(
                data, means[i], covariances[i]
            )

        sum_responsibilities = responsibilities.sum(axis=1)[:, np.newaxis]
        responsibilities /= sum_responsibilities

        # M-step: update parameters
        Nk = responsibilities.sum(axis=0)

        for i in range(n_components):
            means[i] = (responsibilities[:, i][:, np.newaxis] * data).sum(axis=0) / Nk[
                i
            ]
            diff = data - means[i]
            covariances[i] = (
                (responsibilities[:, i][:, np.newaxis] * diff).T @ diff / Nk[i]
            )
            covariances[i] += (tol * 10) * np.eye(d)  # Increased regularization

            # Check for invalid values in covariance matrix
            if np.any(np.isnan(covariances[i])) or np.any(np.isinf(covariances[i])):
                raise ValueError(
                    f"Covariance matrix for component {i} contains NaNs or infs"
                )

        weights = Nk / n

        # Compute log-likelihood and check convergence
        new_log_likelihood = np.sum(np.log(responsibilities.sum(axis=1)))

        if np.abs(new_log_likelihood - log_likelihood) < tol:
            break

        log_likelihood = new_log_likelihood

    # Calculate anomaly scores using responsibility threshold
    max_responsibilities = responsibilities.max(axis=1)

    # Identify anomalies: low responsibility indicates a higher likelihood of being an anomaly
    anomalies = pd.DataFrame(
        np.where(max_responsibilities < responsibility_threshold, -1, 1),
        index=df.index,
        columns=['anomaly'],
    )

    # Score: can still calculate log-likelihood-based scores per data point if needed
    scores = np.zeros((n, d))
    for i in range(n_components):
        comp_pdf = multivariate_normal.pdf(data, means[i], covariances[i])
        comp_scores = -np.log(comp_pdf).reshape(-1, 1)
        scores += responsibilities[:, i][:, np.newaxis] * comp_scores

    scores = pd.DataFrame(scores, index=df.index, columns=df.columns)

    return anomalies, scores
