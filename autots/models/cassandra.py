# -*- coding: utf-8 -*-
"""Cassandra Model.
Created on Tue Sep 13 19:45:57 2022

@author: Colin
"""
import random
import numpy as np
import pandas as pd
from autots.tools.transform import GeneralTransformer, RandomTransform, scalers, filters, HolidayTransformer, AnomalyRemoval
from autots.tools import cpu_count
from autots.models.base import ModelObject


class Cassandra(ModelObject):
    """Explainable decomposition-based forecasting with advanced trend modeling and preprocessing.

    Tunc etiam fatis aperit Cassandra futuris
    ora, dei iussu non umquam credita Teucris.
    Nos delubra deum miseri, quibus ultimus esset
    ille dies, festa velamus fronde per urbem.
    -Aeneid 2.246-2.249

    Args:
        pass
    """

    def __init__(
        self,
        preprocessing_transformation: dict = None,  # filters by default only
        scaling: str = "BaseScaler",  # pulled out from transformation as a scaler is not optional, maybe allow a list
        past_impacts_intervention: bool = False,  # 'remove', 'plot_only', 'regressor'
        seasonalities: dict = {},  # interactions added if fourier and order matches
        ar_lags: list = None,
        ar_interaction_seasonality: dict = None,  # equal or less than number of ar lags
        anomaly_detector_params: dict = None,  # apply before any preprocessing (as has own)
        anomaly_intervention: str = None,  # remove, create feature, model
        holiday_detector_params: dict = None,
        holiday_intervention: str = None,  # remove (median value) or create regression feature
        holiday_countries: dict = None,  # list or dict
        holiday_countries_used: bool = None,
        multivariate_feature: str = None,  # by group, if given
        multivariate_transformation: str = None,  # applied before creating feature
        regressor_transformation: dict = None,  # applied to future_regressor and regressor_per_series
        regressors_used: bool = None,  # on/off of additional user inputs
        linear_model: dict = None,  # lstsq WEIGHTED Ridge, bayesian, bayesian hierarchial, l1 normed or other loss (numba),
        randomwalk_n: int = None,  # use stats of source df
        trend_window: int = 30,
        trend_standin: str = None,  # rolling fit, intercept-only, random.normal features
        trend_anomaly_detector_params: dict = None,  # difference first, run on slope only, use Window n/2 diff to rule out return to
        trend_anomaly_intervention: str = None,
        trend_transformation: dict = {},
        trend_model: dict = {},  # have one or two in built, then redirect to any AutoTS model for other choices
        trend_phi: float = None,
        constraints: dict = None,
        # not modeling related:
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        random_seed: int = 2022,
        verbose: int = 0,
        n_jobs: int = "auto",
    ):
        self.preprocessing_transformation = preprocessing_transformation
        self.scaling = scaling
        self.past_impacts_intervention = past_impacts_intervention
        self.seasonalities = seasonalities
        self.ar_lags = ar_lags
        self.ar_interaction_seasonality = ar_interaction_seasonality
        self.anomaly_detector_params = anomaly_detector_params
        self.anomaly_intervention = anomaly_intervention
        self.holiday_detector_params = holiday_detector_params
        self.holiday_intervention = holiday_intervention
        self.holiday_countries = holiday_countries
        self.holiday_countries_used = holiday_countries_used
        self.multivariate_feature = multivariate_feature
        self.multivariate_transformation = multivariate_transformation
        self.regressor_transformation = regressor_transformation
        self.regressors_used = regressors_used
        self.linear_model = linear_model
        self.randomwalk_n = randomwalk_n
        self.trend_window = trend_window
        self.trend_standin = trend_standin
        self.trend_anomaly_detector_params = trend_anomaly_detector_params
        self.trend_anomaly_intervention = trend_anomaly_intervention
        self.trend_transformation = trend_transformation
        self.trend_model = trend_model
        self.trend_phi = trend_phi
        self.constraints = constraints
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.random_seed = self.random_seed
        self.verbose = verbose
        self.n_jobs = n_jobs
        if self.n_jobs == 'auto':
            self.n_jobs = cpu_count(modifier=0.75)
            if verbose > 0:
                print(f"Using {self.n_jobs} cpus for n_jobs.")
        self.starting_params = self.get_params()

    @staticmethod
    def base_scaler(df):
        return (df - np.mean(df, axis=0)) / np.std(df, axis=0)

    def fit(self, df, future_regressor, regressor_per_series, flag_regressors, categorical_groups, past_impacts):
        # flag regressors bypass preprocessing
        # ideally allow both pd.DataFrame and np.array inputs (index for array?
        self.df = df.copy()
        self.basic_profile(self.df)

        # what features will require separate models to be fit as X will not be consistent for all
        self.loop_required = (
            (self.ar_lags is not None) or (
            ) or (
                isinstance(self.holiday_countries, dict) and self.holiday_countries_used
            ) or (
                (regressor_per_series is not None) and self.regressors_used
            ) or (
                (anom is not None) and self.regressors_used
            ) or (
                (hold is not None) and self.regressors_used
            )
        )
        if self.anomaly_detector_params is not None:
            self.anomaly_detector = AnomalyRemoval(**self.anomaly_detector_params).fit(self.df)
            self.anomaly_intervention  # remove, create feature, model
        if self.holiday_detector_params is not None:
            self.holiday_detector = HolidayTransformer(**self.holiday_detector_params)
            # do I make one holiday feature for all detected holidays??
            OUT = self.holiday_detector.fit_transform(self.df)
            self.holiday_intervention  # remove, create feature
        if self.preprocessing_transformation is not None:
            self.preprocesser = GeneralTransformer(**self.preprocessing_transformation)
            self.preprocesser.fit_transform(self.df)
        if self.scaling is not None:
            if self.scaling == "BaseScaler":
                self.base_scaler(self.df)
            else:
                self.scaler = GeneralTransformer(**self.scaling)

        return NotImplemented

    def predict(self, forecast_length, future_regressor, regressor_per_series, flag_regressors, future_impacts, new_df=None):
        # if future regressors are None (& USED), but were provided for history, instead use forecasts of these features (warn)
        return NotImplemented

    def auto_fit(self, df, validation_method):  # also add regressor input
        # run cross validation to choose
        # return metrics of best model, set base params
        self.starting_params = self.get_params()
        return NotImplemented

    def next_fit(self):
        # option to print a graph of each while going (run again until you see one you like)
        # append to an internal record, print new params
        # plot: 3 stacked, in sample, then each validation
        # maybe use genetic optimization if this used after auto_fit
        return NotImplemented

    def cross_validate(self, df, validation_method):
        # run one model and return scores
        return NotImplemented

    def treatment_causal_impact(self, df, intervention_dates):  # also add regressor input
        # regressors important, estimation by linear flag (return stats) or by % vs forecast from cutoff
        # start and end dates, multiple treatments, vary by series
        # or maybe instead, just return the stats for all input regressors
        # still working on this, possibly just https://math.stackexchange.com/questions/2112553/statistical-significance-of-linear-least-squares
        # and other features ala Statsmodels OLS, Var(β^)=σ2(X′X)−1 where σ2>0 is the common variance of each element of the error vector
        return NotImplemented

    def predict_new_product(self):
        # borrow the linear components from a related product (or avg/mode if not given)
        # have the trend start at zero then up, like reverse phi
        # possibly borrow warmup from any series that have a logistic-like trend
        # cluster and choose the biggest cluster (as avg value naive method, maybe round or filter first)
        return NotImplemented

    @staticmethod
    def get_new_params(method='fast'):
        # have fast option that avoids any of the loop approaches
        scaling = random.choice(['BaseScaler', 'other'])
        if scaling == "other":
            scaling = RandomTransform(
                transformer_list=scalers, transformer_max_depth=1,
                allow_none=False, no_nan_fill=True
            )
        holiday_intervention = random.choice(['create_feature', 'use_impact'])
        holiday_params = HolidayTransformer.get_new_params(method=method)
        return {
            "preprocessing_transformation": RandomTransform(
                transformer_list=filters, transformer_max_depth=2, allow_none=True
            ),
            "scaling": scaling,
            "past_impacts_intervention": self.past_impacts_intervention,
            "seasonalities": self.seasonalities,
            "ar_lags": self.ar_lags,
            "ar_interaction_seasonality": self.ar_interaction_seasonality,
            "anomaly_detector_params": AnomalyRemoval.get_new_params(method=method),
            "anomaly_intervention": self.anomaly_intervention,
            "holiday_detector_params": holiday_params,
            "holiday_intervention": holiday_intervention,
            "holiday_countries": self.holiday_countries,
            "holiday_countries_used": self.holiday_countries_used,
            "multivariate_feature": self.multivariate_feature,
            "multivariate_transformation": self.multivariate_transformation,
            "regressor_transformation": self.regressor_transformation,
            "regressors_used": self.regressors_used,
            "linear_model": self.linear_model,
            "randomwalk_n": self.randomwalk_n,
            "trend_window": self.trend_window,
            "trend_standin": self.trend_standin,
            "trend_anomaly_detector_params": self.trend_anomaly_detector_params,
            "trend_anomaly_intervention": self.trend_anomaly_intervention,
            "trend_transformation": self.trend_transformation,
            "trend_model": self.trend_model,
            "trend_phi": self.trend_phi,
            "constraints": self.constraints,
        }

    def get_params(self):
        return {
            "preprocessing_transformation": self.preprocessing_transformation,
            "scaling": self.scaling,
            "past_impacts_intervention": self.past_impacts_intervention,
            "seasonalities": self.seasonalities,
            "ar_lags": self.ar_lags,
            "ar_interaction_seasonality": self.ar_interaction_seasonality,
            "anomaly_detector_params": self.anomaly_detector_params,
            "anomaly_intervention": self.anomaly_intervention,
            "holiday_detector_params": self.holiday_detector_params,
            "holiday_intervention": self.holiday_intervention,
            "holiday_countries": self.holiday_countries,
            "holiday_countries_used": self.holiday_countries_used,
            "multivariate_feature": self.multivariate_feature,
            "multivariate_transformation": self.multivariate_transformation,
            "regressor_transformation": self.regressor_transformation,
            "regressors_used": self.regressors_used,
            "linear_model": self.linear_model,
            "randomwalk_n": self.randomwalk_n,
            "trend_window": self.trend_window,
            "trend_standin": self.trend_standin,
            "trend_anomaly_detector_params": self.trend_anomaly_detector_params,
            "trend_anomaly_intervention": self.trend_anomaly_intervention,
            "trend_transformation": self.trend_transformation,
            "trend_model": self.trend_model,
            "trend_phi": self.trend_phi,
            "constraints": self.constraints,
        }

    def plot_things():  # placeholder for later plotting functions
        # plot components
        # plot % contribution of components
        # plot eval (show actuals, alongside full and components to diagnose)
        # plot residual distribution/PACF
        # plot inflection points (filtering or smoothing first)
        return NotImplemented


# Seasonalities
    # maybe fixed 3 seasonalities
    # interaction effect only on first two seasonalities if order matches
# Holiday countries (mapped countries per series)
    # make it holiday 1, holiday 2 so per country?
# Regressor (preprocessing)
# Flag Regressors
# Multivariate Summaries (pre and post processing)
# AR Lags
    # could this vary per series?
# Categorical Features
    # (multivariate summaries by group)

# initial preprocessing (only filters + Quantile Transformer)
# scaler
# holiday detection
# anomaly detection
    # anomaly risk model
# whether to pre-include rolling fit, intercept-only, random_walk, or random.normal features
# anomaly on rolling (difference first)
# constraints (copied)
# past impacts (option to not enforce, only show)
# phi on Trend model (to flat)

# What triggers loop:
    # AR Lags
    # Holiday countries as dict (not list)
    # regressor_per_series (only if regressor used)
    # Multivariate Holiday Detector
        # only if create feature, not removal (plot in this case)
    # Multivariate Anomaly Detector
        # only if create feature, not removal

# search space:
# fast on lstsqs model
# full search of predefined options
# scipy minimize for continuous params, or maybe int(param) for discrete too
    # minimize working for rolling window size but not for seasonality (actually probably window is overfit)
# do it stepwise:
    # first fit seasonality (with flag regressors but not others)
    # secondly scalers and preprocessing AND anomaly detection AND holiday detection
    # possibly add 3rd seasonality after preprocessing
    # thirdly rolling window size (include rolling window in initial X after this)
    # then test user regressor and preprocessing
    # then test AR lags (don't use if % gain is < x %)
        # or maybe use a runtime weighting
    # then multivariate summaries
    # FINALLY covariate lags (feature selection definitely needed)
