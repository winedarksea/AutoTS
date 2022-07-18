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
    from scipy.stats import chi2, norm, gamma, uniform
except Exception:
    pass

# need to remove actual outliers first, or maybe scale data

# Anomaly Methods
# sklearn methods (isolation forest, LOF, EE)
# AutoTS Backforecast with Naive (anomaly above prediction interval)
# AutoTS Backforecast with Naive (anomaly by threshold detection on error)
# https://github.com/datamllab/tods/blob/d0a5f9d87f6b3cf57b849d8fb8481905b5930bd4/tods/detection_algorithm/core/utils/errors.py#L334
# _calc_anomaly_scores from https://facebookresearch.github.io/Kats/api/_modules/kats/detectors/outlier.html#MultivariateAnomalyDetector
# Kats Outlier Detections models
# abnormal diff

# update back_forecast to do forecast_length 1 except for reverse on start
# allow eval_period if detection only on a fraction of history

# Univariate models (one in, one out)
# Multivariate in but Univariate Return
# Multvariate in with Multivariate Out
# df of anomaly scores, pandas Series for univariate
# df of anomaly class, pandas Series for univariate
# for univariate case, flatten after taking log or normalizing
# remove in advance values with extremely high score
# runtime, number of holidays, prediction forecast gain


def sk_outliers(df, method, method_params={}):
    """scikit-learn outlier methods wrapper."""
    if method == "IsolationForest":
        model = IsolationForest(
            n_jobs=1, **method_params
        )  # n_estimators=200
        res = model.fit_predict(df)
        scores = model.decision_function(df)
    elif method == "LOF":
        model = LocalOutlierFactor(
            n_jobs=1, **method_params
        )  # n_neighbors=5
        res = model.fit_predict(df)
        scores = model.negative_outlier_factor_ + 1.45
    elif method == "EE":
        if method_params['contamination'] == "auto":
            method_params['contamination'] = 0.1
        model = EllipticEnvelope(**method_params)
        res = model.fit_predict(df)
        scores = model.decision_function(df)
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
    elif distribution == "chi2":
        return pd.DataFrame(
            chi2.sf(residual_score, dof), index=df.index, columns=columns
        )
    elif distribution == "uniform":
        return pd.DataFrame(
            uniform.sf(residual_score, dof), index=df.index, columns=columns
        )
    else:
        raise ValueError("zscore sf `distribution` arg not recognized")


def limits_to_anomalies(
    df, output, upper_limit, lower_limit, method_params=None,
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
            index=df.index, columns=df.columns
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
        scores = 1 - (
            (df_abs - df_abs.min(axis=0))
            / (df.abs().max(axis=0) - df_abs.min(axis=0)).replace(0, 1)
        )
        res = pd.DataFrame(
            np.where(scores < alpha, -1, 1),
            index=df.index,
            columns=cols,
        )
        return res, scores
    elif threshold_method in ["zscore", "rolling_zscore", "mad"]:
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
            cnts = np.unique(
                np.tile(np.arange(df.shape[0]), df.shape[1])[mod.i_anom],
                return_counts=True,
            )
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

    if method in ["IsolationForest", "LOF", "EE"]:
        if output == "univariate":
            res, scores = sk_outliers(df_anomaly, method, method_params)
        else:
            res, scores = loop_sk_outliers(
                df_anomaly, method, method_params, n_jobs
            )
    elif method in ["zscore", "rolling_zscore", "mad", "minmax"]:
        res, scores = values_to_anomalies(df_anomaly, output, method, method_params)
    elif method in ["IQR"]:
        iqr_thresh = method_params.get("iqr_threshold", 2.0)
        iqr_quantiles = method_params.get("iqr_quantiles", [0.25, 0.75])
        resid_q = nan_quantile(df_anomaly, iqr_quantiles)
        iqr = resid_q[1] - resid_q[0]
        limits = resid_q + (iqr_thresh * iqr)
        res, scores = limits_to_anomalies(
            df_anomaly,
            output=output,
            method_params=method_params,
            upper_limit=limits[1],
            lower_limit=limits[0],
        )
    elif method in "nonparametric":
        res, scores = values_to_anomalies(
            df_anomaly,
            output,
            threshold_method=method,
            method_params=method_params,
            n_jobs=n_jobs,
        )
    elif method in ["prediction_interval"]:
        raise ValueError("prediction_interval method only handled by AnomalyDetector class.")
    else:
        raise ValueError(f"{method} outlier method not recognized")

    return res, scores


available_methods = [
    "IsolationForest",
    "LOF",
    "EE",
    "zscore",
    "rolling_zscore",
    "mad",
    "minmax",
    "prediction_interval",
    "IQR",
    "nonparametric",
]


def anomaly_new_params(method='random'):
    if method == "fast":
        method_choice = random.choices(
            [
                "LOF", "EE", "zscore", "rolling_zscore", "mad",
                "minmax", "IQR", "nonparametric",
            ],
            [0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.2, 0.1]
        )[0]
    else:
        method_choice = random.choices(
            available_methods,
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1, 0.15],
        )[0]
    if method_choice == "IsolationForest":
        method_params = {
            'contamination': random.choices(['auto', 0.1, 0.05], [0.8, 0.1, 0.1])[0],
            'n_estimators': random.choices([50, 100, 200], [0.3, 0.4, 0.3])[0],
            'max_features': random.choices([1.0, 0.5], [0.9, 0.1])[0],
            'bootstrap': random.choices([False, True], [0.9, 0.1])[0],
        }
    elif method_choice == "LOF":
        method_params = {
            'contamination': random.choices(['auto', 0.1, 0.05], [0.8, 0.1, 0.1])[0],
            'n_neighbors': random.choices([3, 5, 10, 20], [0.3, 0.4, 0.3, 0.1])[0],
            'metric': random.choices(['minkowski', 'canberra'], [0.9, 0.1])[0],
        }
    elif method_choice == "EE":
        method_params = {
            'contamination': random.choices([0.02, 0.1, 0.05, 0.15], [0.1, 0.8, 0.05, 0.05])[0],
            'assume_centered': random.choices([False, True], [0.9, 0.1])[0],
            'support_fraction': random.choices([None, 0.2, 0.8], [0.9, 0.1, 0.1])[0]
        }
    elif method_choice == "zscore":
        method_params = {
            'distribution': random.choices(
                ['norm', 'gamma', 'chi2', 'uniform'], [0.4, 0.2, 0.2, 0.2]
            )[0],
            'alpha': random.choices([0.03, 0.05, 0.1], [0.1, 0.8, 0.1])[0],
        }
    elif method_choice == "rolling_zscore":
        method_params = {
            'distribution': random.choices(
                ['norm', 'gamma', 'chi2', 'uniform'], [0.4, 0.2, 0.2, 0.2]
            )[0],
            'alpha': random.choices([0.03, 0.05, 0.1], [0.1, 0.8, 0.1])[0],
            'rolling_periods': random.choice([28, 90, 200, 300]),
            'center': random.choice([True, False])
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
        method_params = {
            'iqr_threshold': random.choices(
                [1.5, 2.0, 2.5, 3.0], [0.1, 0.4, 0.2, 0.1]
            )[0],
            'iqr_quantiles': random.choices([[0.25, 0.75], [0.4, 0.6]], [0.8, 0.2])[0],
        }
    elif method_choice == "nonparametric":
        method_params = {
            'p': random.choices([None, 0.1, 0.05, 0.02, 0.25], [0.8, 0.1, 0.1, 0.01, 0.1])[0],
            'z_init': random.choices([1.5, 2.0, 2.5], [0.3, 0.4, 0.3])[0],
            'z_limit': random.choices([10, 12], [0.5, 0.5])[0],
            'z_step': random.choice([0.5, 0.25]),
            'inverse': random.choices([False, True], [0.9, 0.1])[0],
            'max_contamination': random.choices([0.25, 0.1, 0.05], [0.9, 0.05, 0.05])[0],
            'mean_weight': random.choices([1, 25, 100, 200], [0.1, 0.7, 0.1, 0.1])[0],
            'sd_weight': random.choices([1, 25, 100, 200], [0.1, 0.7, 0.1, 0.1])[0],
            'anomaly_count_weight': random.choices([0.5, 1.0, 2.0], [0.05, 0.9, 0.05])[0],
        }
    transform_dict = random.choices(
        [
            None, 'random',
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
                            "model": "DecisionTree",
                            "model_params": {"max_depth": None, "min_samples_split": 0.05},
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
        ], [0.5, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1],
    )[0]
    return method_choice, method_params, transform_dict


def anomaly_list_to_holidays(dates):
    # RETURN INSTEAD full table, filter threshold later
    # filter by % of available seasons (ie keep 1 if only 1 full season)
    dates_df = pd.DataFrame(
        {"month": dates.month, "day": dates.day, "dayofweek": dates.dayofweek},
        index=dates,
    )
    dates_df["weekofmonth"] = (dates_df["day"] - 1) // 7 + 1
    threshold = (dates.year.max() - dates.year.min() + 1) * 0.75
    threshold = threshold if threshold > 2 else 2
    day_holidays = (
        dates_df.groupby(["month", "day"])
        .count()
        .loc[
            lambda df: df["dayofweek"] > threshold,
        ]
    )
    wkdom_holidays = (
        dates_df.groupby(["month", "weekofmonth", "dayofweek"])
        .count()
        .loc[
            lambda df: df["day"] > threshold,
        ]
    )
    return day_holidays, wkdom_holidays
