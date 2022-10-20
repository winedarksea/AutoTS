# -*- coding: utf-8 -*-
"""Cassandra Model.
Created on Tue Sep 13 19:45:57 2022

@author: Colin
"""
from operator import itemgetter
from itertools import groupby
import random
import numpy as np
import pandas as pd
# using transformer version of Anomaly/Holiday to use a lower level import than evaluator
from autots.tools.transform import GeneralTransformer, RandomTransform, scalers, filters, decompositions, HolidayTransformer, AnomalyRemoval, EmptyTransformer
from autots.tools import cpu_count
from autots.models.base import ModelObject, PredictionObject
from autots.templates.general import general_template
from autots.tools.holiday import holiday_flag
from autots.tools.window_functions import window_lin_reg_mean  # sliding_window_view
from autots.evaluator.auto_model import ModelMonster, model_forecast
# scipy is technically optional but most likely is present
try:
    from scipy.optimize import minimize
    from scipy.stats import norm
except Exception:
    class norm(object):
        @staticmethod
        def ppf(x):
            return 1.6448536269514722


class Cassandra(ModelObject):
    """Explainable decomposition-based forecasting with advanced trend modeling and preprocessing.

    Tunc etiam fatis aperit Cassandra futuris
    ora, dei iussu non umquam credita Teucris.
    Nos delubra deum miseri, quibus ultimus esset
    ille dies, festa velamus fronde per urbem.
    -Aeneid 2.246-2.249

    Warn about remove_excess_anomalies from holiday detector if relying on anomaly prediction
    Linear components are always model elements, but trend is actuals (history) and model (future)
    Running predict updates some internal attributes used in plotting and other figures, generally expect to use functions to latest predict
    Seasonalities are hard-coded to be as days so 7 will always = weekly even if data isn't daily
    For slope analysis and zero crossings, a slope of 0 evaluates as a positive sign (=>0). Exactly 0 slope is rare real world data
    Does not currently follow the regression_type='User' and fails if no regressor pattern of other models
    For component decomposition, scale will be inaccurate unless 'BaseScaler' is used, but regardless this won't affect final forecast

    Args:
        pass

    Methods:
         holiday_detector.dates_to_holidays

    Attributes:
        .anomaly_detector.anomalies
        .anomaly_detector.scores
        .holiday_count
        .holidays (series flags, holiday detector only)
        .params
        .keep_cols, .keep_cols_idx
        .x_array
        .predict_x_array
        .trend_train
    """

    def __init__(
        self,
        preprocessing_transformation: dict = None,  # filters by default only
        scaling: str = "BaseScaler",  # pulled out from transformation as a scaler is not optional, maybe allow a list
        past_impacts_intervention: str = None,  # 'remove', 'plot_only', 'regressor'
        seasonalities: dict = None,  # interactions added if fourier and order matches
        ar_lags: list = None,
        ar_interaction_seasonality: dict = None,  # equal or less than number of ar lags
        anomaly_detector_params: dict = None,  # apply before any preprocessing (as has own)
        anomaly_intervention: str = None,  # remove, create feature, model
        holiday_detector_params: dict = None,
        holiday_countries: dict = None,  # list or dict
        holiday_countries_used: bool = True,
        multivariate_feature: str = None,  # by group, if given
        multivariate_transformation: str = None,  # applied before creating feature
        regressor_transformation: dict = None,  # applied to future_regressor and regressor_per_series
        regressors_used: bool = True,  # on/off of additional user inputs
        linear_model: dict = None,  # lstsq WEIGHTED Ridge, bayesian, bayesian hierarchial, l1 normed or other loss (numba),
        randomwalk_n: int = None,  # use stats of source df
        trend_window: int = 30,  # set to None to use raw residuals
        trend_standin: str = None,  # rolling fit, intercept-only, random.normal features
        trend_anomaly_detector_params: dict = None,  # difference first, run on slope only, use Window n/2 diff to rule out return to
        # trend_anomaly_intervention: str = None,
        trend_transformation: dict = {},
        trend_model: dict = {'Model': 'MetricMotif', 'ModelParameters': {}},  # have one or two in built, then redirect to any AutoTS model for other choices
        trend_phi: float = None,
        constraint: dict = None,
        # not modeling related:
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        random_seed: int = 2022,
        verbose: int = 0,
        n_jobs: int = "auto",
    ):
        if preprocessing_transformation is None:
            preprocessing_transformation = {}
        self.preprocessing_transformation = preprocessing_transformation
        self.scaling = scaling
        self.past_impacts_intervention = past_impacts_intervention
        self.seasonalities = seasonalities
        self.ar_lags = ar_lags
        self.ar_interaction_seasonality = ar_interaction_seasonality
        self.anomaly_detector_params = anomaly_detector_params
        self.anomaly_intervention = anomaly_intervention
        self.holiday_detector_params = holiday_detector_params
        self.holiday_countries = holiday_countries
        if isinstance(self.holiday_countries, str):
            self.holiday_countries = self.holiday_countries.split(",")
        self.holiday_countries_used = holiday_countries_used
        self.multivariate_feature = multivariate_feature
        self.multivariate_transformation = multivariate_transformation
        self.regressor_transformation = regressor_transformation
        self.regressors_used = regressors_used
        self.linear_model = linear_model if linear_model is not None else {}
        self.randomwalk_n = randomwalk_n
        self.trend_window = trend_window
        self.trend_standin = trend_standin
        self.trend_anomaly_detector_params = trend_anomaly_detector_params
        # self.trend_anomaly_intervention = trend_anomaly_intervention
        self.trend_transformation = trend_transformation
        self.trend_model = trend_model
        self.trend_phi = trend_phi
        self.constraint = constraint
        # other parameters
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.int_std_dev = norm.ppf(0.5 + 0.5 * self.prediction_interval)  # 2 to 1 sided interval
        self.random_seed = random_seed
        self.verbose = verbose
        self.n_jobs = n_jobs
        if self.n_jobs == 'auto':
            self.n_jobs = cpu_count(modifier=0.75)
            if verbose > 0:
                print(f"Using {self.n_jobs} cpus for n_jobs.")
        self.starting_params = self.get_params()
        self.scaler = EmptyTransformer()
        self.preprocesser = EmptyTransformer()
        self.ds_min = pd.Timestamp("2000-01-01")
        self.ds_max = pd.Timestamp("2025-01-01")
        self.name = "Cassandra"
        self.future_regressor_train = None
        self.flag_regressor_train = None
        self.regr_per_series_tr = None
        self.trend_train = None
        self.components = None
        self.anomaly_detector = None
        self.holiday_detector = None
        if self.trend_anomaly_detector_params is not None:
            self.trend_anomaly_detector = AnomalyRemoval(**self.trend_anomaly_detector_params)
        else:
            self.trend_anomaly_detector = None

    def base_scaler(self, df):
        self.scaler_mean = np.mean(df, axis=0)
        self.scaler_std = np.std(df, axis=0)
        return (df - self.scaler_mean) / self.scaler_std

    def scale_data(self, df):
        if self.preprocessing_transformation is not None:
            df = self.preprocesser.transform(df.copy())
        if self.scaling is not None:
            if self.scaling == "BaseScaler":
                df = self.base_scaler(df)
            else:
                df = self.scaler.transform(df)
        return df

    def to_origin_space(self, df, trans_method='forecast', components=False, bounds=False):
        """Take transformed outputs back to original feature space."""
        if self.scaling == "BaseScaler":
            # return self.preprocesser.inverse_transform(df, trans_method=trans_method) * self.scaler_std + self.scaler_mean
            if components:
                return self.preprocesser.inverse_transform(df * self.scaler_std, trans_method=trans_method, bounds=bounds)
            else:
                return self.preprocesser.inverse_transform(df * self.scaler_std + self.scaler_mean, trans_method=trans_method, bounds=bounds)
        else:
            # return self.scaler.inverse_transform(self.preprocesser.inverse_transform(df, trans_method=trans_method), trans_method=trans_method)
            return self.preprocesser.inverse_transform(
                self.scaler.inverse_transform(df, trans_method=trans_method, bounds=bounds),
                trans_method=trans_method, bounds=bounds
            )

    def create_t(self, DTindex):
        return (DTindex - self.ds_min) / (self.ds_max - self.ds_min)

    def fit(self, df, future_regressor=None, regressor_per_series=None, flag_regressors=None, categorical_groups=None, past_impacts=None):
        # flag regressors bypass preprocessing
        # ideally allow both pd.DataFrame and np.array inputs (index for array?
        if self.constraint is not None:
            self.df_original = df
        self.df = df.copy()
        self.basic_profile(self.df)
        # standardize groupings (extra/missing dealt with)
        if categorical_groups is None:
            categorical_groups = {}
        self.categorical_groups = {col: (categorical_groups[col] if col in categorical_groups.keys() else "other") for col in df.columns}
        self.past_impacts = past_impacts
        # if past impacts given, assume removal unless otherwise specified
        if self.past_impacts_intervention is None and past_impacts is not None:
            self.past_impacts_intervention = "remove"
        elif past_impacts is None:
            self.past_impacts_intervention = None

        # what features will require separate models to be fit as X will not be consistent for all
        if isinstance(self.anomaly_detector_params, dict):
            self.anomaly_not_uni = self.anomaly_detector_params.get('output', None) != 'univariate'
        self.anomaly_not_uni = False
        self.loop_required = (
            (self.ar_lags is not None) or (
            ) or (
                isinstance(self.holiday_countries, dict) and self.holiday_countries_used
            ) or (
                (regressor_per_series is not None) and self.regressors_used
            ) or (
                isinstance(self.anomaly_intervention, dict) and self.anomaly_not_uni
            ) or (
                self.past_impacts_intervention == "regressor" and past_impacts is not None
            )
        )
        # check if rolling prediction is required
        self.predict_loop_req = (self.ar_lags is not None) or (self.multivariate_feature is not None)

        # remove past impacts to find "organic"
        if self.past_impacts_intervention == "remove":
            self.df = self.df / (1 + past_impacts)  # MINUS OR PLUS HERE???
        # holiday detection first, don't want any anomalies removed yet, and has own preprocessing
        if self.holiday_detector_params is not None:
            self.holiday_detector = HolidayTransformer(**self.holiday_detector_params)
            self.holiday_detector.fit(self.df)
            self.holidays = self.holiday_detector.dates_to_holidays(df.index, style='series_flag')
            self.holiday_count = np.count_nonzero(self.holidays)
            if self.holiday_detector_params["remove_excess_anomalies"]:
                self.df = self.df[~((self.holiday_detector.anomaly_model.anomalies == -1) & (self.holidays != 1))]
        # find anomalies, and either remove or setup for modeling the anomaly scores
        if self.anomaly_detector_params is not None:
            self.anomaly_detector = AnomalyRemoval(**self.anomaly_detector_params).fit(self.df)
            # REMOVE HOLIDAYS from anomalies, as they may look like anomalies but are dealt with by holidays
            # this, however, does nothing to preserve country holidays
            if self.holiday_detector_params is not None:
                hol_filt = (self.holidays == 1) & (self.anomaly_detector.anomalies == -1)
                self.anomaly_detector.anomalies[hol_filt] = 1
                # assume most are not anomalies so median is not anom
                self.anomaly_detector.scores[hol_filt] = np.median(self.anomaly_detector.scores)
            if self.anomaly_intervention == "remove":
                self.df = self.anomaly_detector.transform(self.df)
            elif isinstance(self.anomaly_intervention, dict):
                self.anomaly_detector.fit_anomaly_classifier()
                return NotImplemented
            # detect_only = pass
        # now do standard preprocessing
        if self.preprocessing_transformation is not None:
            self.preprocesser = GeneralTransformer(**self.preprocessing_transformation)
            self.df = self.preprocesser.fit_transform(self.df)
        if self.scaling is not None:
            if self.scaling == "BaseScaler":
                self.df = self.base_scaler(self.df)
            else:
                if self.scaling is None:
                    raise ValueError("scaling must not be None. Try 'BaseScaler'")
                self.scaler = GeneralTransformer(**self.scaling)
                self.df = self.scaler.fit_transform(self.df)
        # additional transforms before multivariate feature creation
        if self.multivariate_transformation is not None:
            self.multivariate_transformer = GeneralTransformer(**self.multivariate_transformation)
        # needs to come after preprocessing because of 'slice' transformer
        self.ds_min = self.df.index.min()
        self.ds_max = self.df.index.max()
        self.t_train = self.create_t(self.df.index)
        self.history_days = (self.ds_max - self.ds_min).days

        # BEGIN CONSTRUCTION OF X ARRAY
        x_list = []
        if isinstance(self.anomaly_intervention, dict):
            # need to model these for prediction
            x_list.append(self.anomaly_detector.scores)
        # all of the following are 1 day past lagged
        if self.multivariate_feature is not None:
            # includes backfill
            lag_1_indx = np.concatenate([[0], np.arange(len(self.df))])[0:len(self.df)]
            trs_df = self.multivariate_transformer.fit_transform(self.df)
            if trs_df.shape != self.df.shape:
                raise ValueError("Multivariate Transformer not usable for this role.")
            if self.multivariate_feature == "feature_agglomeration":
                from sklearn.cluster import FeatureAgglomeration

                self.agglom_n_clusters = 5
                self.agglomerator = FeatureAgglomeration(n_clusters=self.agglom_n_clusters)
                x_list.append(pd.DataFrame(
                    self.agglomerator.fit_transform(trs_df)[lag_1_indx],
                    index=self.df.index,
                    columns=["multivar_" + str(x) for x in range(self.agglom_n_clusters)]
                ))
            elif self.multivariate_feature == "group_average":
                multivar_df = trs_df.groupby(self.categorical_groups, axis=1).mean().iloc[lag_1_indx]
                multivar_df.index = self.df.index
                x_list.append(multivar_df.rename(columns=lambda x: "multivar_" + str(x)))
            elif self.multivariate_feature == "oscillator":
                return NotImplemented
                np.count_nonzero((df - df.shift(1)).clip(upper=0))[:-1]
        if self.seasonalities is not None:
            s_list = []
            for seasonality in self.seasonalities:
                s_list.append(create_seasonality_feature(self.df.index, self.t_train, seasonality, history_days=self.history_days))
                # INTERACTIONS NOT IMPLEMENTED
                # ORDER SPECIFICATION NOT IMPLEMENTED
            s_df = pd.concat(s_list, axis=1)
            s_df.index = self.df.index
            x_list.append(s_df)
        # These features are to prevent overfitting and standin for unobserved components here
        if self.randomwalk_n is not None:
            x_list.append(pd.DataFrame(
                np.random.normal(size=(len(self.df), self.randomwalk_n)).cumsum(axis=0),
                columns=["randomwalk_" + str(x) for x in range(self.randomwalk_n)],
                index=self.df.index,
            ))
        if self.trend_standin is not None:
            if self.trend_standin == "random_normal":
                num_standin = 4
                x_list.append(pd.DataFrame(
                    np.random.normal(size=(len(self.df), num_standin)),
                    columns=["randnorm_" + str(x) for x in range(num_standin)],
                    index=self.df.index,
                ))
            elif self.trend_standin == "rolling_trend":
                return NotImplemented
        if future_regressor is not None and self.regressors_used:
            if self.regressor_transformation is not None:
                self.regressor_transformer = GeneralTransformer(**self.regressor_transformation)
                self.future_regressor_train = self.regressor_transformer.fit_transform(clean_regressor(future_regressor))
            else:
                self.regressor_transformer = GeneralTransformer(**{})
            x_list.append(self.future_regressor_train.reindex(self.df.index))
        if flag_regressors is not None and self.regressors_used:
            self.flag_regressor_train = clean_regressor(flag_regressors, prefix="regrflags_")
            x_list.append(self.flag_regressor_train.reindex(self.df.index))
        if self.holiday_countries is not None and not isinstance(self.holiday_countries, dict) and self.holiday_countries_used:
            for holiday_country in self.holiday_countries:
                x_list.append(
                    holiday_flag(
                        self.df.index,
                        country=holiday_country,
                        encode_holiday_type=True,
                    ).rename(columns=lambda x: "holiday_" + str(x)),
                )
        # put this to the end as it takes up lots of feature space sometimes
        if self.holiday_detector_params is not None:
            x_list.append(self.holiday_detector.dates_to_holidays(self.df.index, style="flag").clip(upper=1).rename(columns=lambda x: "holiday_" + str(x)))

        # FINAL FEATURE PROCESSING
        x_array = pd.concat(x_list, axis=1)
        self.x_array = x_array  # can remove this later, it is for debugging
        if np.any(np.isnan(x_array.astype(float))):  # remove later, for debugging
            raise ValueError("nan values in x_array")
        # remove zero variance (corr is nan)
        corr = np.corrcoef(x_array, rowvar=0)
        nearz = x_array.columns[np.isnan(corr).all(axis=1)]
        if len(nearz) > 0:
            print(f"Dropping zero variance feature columns {nearz}")
            x_array = x_array.drop(columns=nearz)
        # remove colinear features
        # NOTE THESE REMOVALS REMOVE THE FIRST OF PAIR COLUMN FIRST
        corr = np.corrcoef(x_array, rowvar=0)  # second one
        w, vec = np.linalg.eig(corr)
        np.fill_diagonal(corr, 0)
        corel = x_array.columns[np.min(corr * np.tri(corr.shape[0]), axis=0) > 0.998]
        colin = x_array.columns[w < 0.005]
        if len(corel) > 0:
            print(f"Dropping colinear feature columns {corel}")
            x_array = x_array.drop(columns=corel)
        if len(colin) > 0:
            print(f"Dropping multi-colinear feature columns {colin}")
            x_array = x_array.drop(columns=colin)
        # things we want modeled but want to discard from evaluation (standins)
        remove_patterns = ["randnorm_", "rolling_trend_", "randomwalk_"]  # "intercept" added after, so not included

        # RUN LINEAR MODEL
        # add x features that don't apply to all, and need to be looped
        if self.loop_required:
            self.params = {}
            self.keep_cols = {}
            self.x_array = {}
            self.keep_cols_idx = {}
            self.col_groupings = {}
            trend_residuals = []
            for col in self.df.columns:
                c_x = x_array.copy()
                # implement per_series holiday countries flag
                if isinstance(self.holiday_countries, dict) and self.holiday_countries_used:
                    hc = self.holiday_countries.get(col, None)
                    if hc is not None:
                        c_hc = holiday_flag(
                            self.df.index,
                            country=hc,
                            encode_holiday_type=True,
                        ).rename(columns=lambda x: "holiday_" + str(x))
                    else:
                        c_hc = pd.DataFrame(0, index=self.df.index, columns=["holiday_0"])
                    c_x = pd.concat([
                        c_x,
                        c_hc
                    ], axis=1)
                # implement regressors per series
                if isinstance(regressor_per_series, dict) and self.regressors_used:
                    hc = regressor_per_series.get(col, None)
                    if hc is not None:
                        c_hc = pd.DataFrame(hc).rename(columns=lambda x: "regrperseries_" + str(x)).reindex(self.df.index)
                    else:
                        c_hc = pd.DataFrame(0, index=self.df.index, columns=["regrperseries_0"])
                    c_x = pd.concat([
                        c_x,
                        c_hc
                    ], axis=1)
                # implement past_impacts as regressor
                if isinstance(past_impacts, pd.DataFrame) and self.past_impacts_intervention == "regressor":
                    c_x = pd.concat([
                        c_x,
                        past_impacts[col].to_frame().rename(columns=lambda x: "impacts_" + str(x))
                    ], axis=1)
                # add AR features
                if self.ar_lags is not None:
                    for lag in self.ar_lags:
                        lag_idx = np.concatenate([np.repeat([0], lag), np.arange(len(self.df))])[0:len(self.df)]
                        lag_s = self.df[col].iloc[lag_idx].rename(f"lag{lag}_")
                        lag_s.index = self.df.index
                        c_x = pd.concat([
                            c_x,
                            lag_s
                        ], axis=1)
                # NOTE THERE IS NO REMOVING OF COLINEAR FEATURES ADDED HERE
                self.keep_cols[col] = c_x.columns[~c_x.columns.str.contains("|".join(remove_patterns))]
                self.keep_cols_idx[col] = c_x.columns.get_indexer_for(self.keep_cols[col])
                self.col_groupings[col] = self.keep_cols[col].str.partition("_").get_level_values(0)
                c_x['intercept'] = 1
                self.x_array[col] = c_x
                # ADDING RECENCY WEIGHTING AND RIDGE PARAMS
                self.params[col] = linear_model(c_x, self.df[col].to_frame(), params=self.linear_model)
                trend_residuals.append(
                    self.df[col] - pd.Series(np.dot(c_x[self.keep_cols[col]], self.params[col][self.keep_cols_idx[col]]).flatten(), name=col, index=self.df.index)
                )
            trend_residuals = pd.concat(trend_residuals, axis=1)
        else:
            # RUN LINEAR MODEL, WHEN NO LOOPED FEATURES
            self.keep_cols = x_array.columns[~x_array.columns.str.contains("|".join(remove_patterns))]
            self.keep_cols_idx = x_array.columns.get_indexer_for(self.keep_cols)
            self.col_groupings = self.keep_cols.str.partition("_").get_level_values(0)
            x_array['intercept'] = 1
            # run model
            self.params = linear_model(x_array, self.df, params=self.linear_model)
            trend_residuals = self.df - np.dot(x_array[self.keep_cols], self.params[self.keep_cols_idx])
            self.x_array = x_array

        # option to run trend model on full residuals or on rolling trend
        if self.trend_anomaly_detector_params is not None or self.trend_window is not None:
            # rolling trend
            trend_posterior, slope, intercept = self.rolling_trend(trend_residuals, np.array(self.t_train))
            trend_posterior = pd.DataFrame(trend_posterior, index=self.df.index, columns=self.df.columns)
            self.residual_uncertainty = (trend_residuals - trend_posterior).std()
        else:
            self.residual_uncertainty = pd.Series(0, index=self.df.columns)
        if self.trend_window is not None:
            self.trend_train = trend_posterior
        else:
            self.trend_train = pd.DataFrame(trend_residuals, index=self.df.index, columns=self.df.columns)

        self.zero_crossings, self.changepoints, self.slope_sign, self.accel = self.analyze_trend(slope, index=self.trend_train.index)
        if False:
            self.trend_anomaly_detector = AnomalyRemoval(**self.trend_anomaly_detector_params)
            # DIFF the length of W (or w-1?)
            shft_idx = np.concatenate([[0], np.arange(len(slope))])[0:len(slope)]
            slope_diff = slope - slope[shft_idx]
            shft_idx = np.concatenate([np.repeat([0], self.trend_window), np.arange(len(slope))])[0:len(slope)]
            slope_diff = slope_diff + slope_diff[shft_idx]
            # np.cumsum(slope, axis=0)
            # pd.DataFrame(slope).rolling(90, center=True, min_periods=2).mean()
            # pd.DataFrame(slope, index=df.index).rolling(365, center=True, min_periods=2).mean()[0].plot()
            self.trend_anomaly_detector.fit(
                pd.DataFrame((slope - slope[shft_idx]), index=self.df.index)
            )

        self.fit_runtime = self.time() - self.startTime
        return self

    def analyze_trend(self, slope, index):
        # desired behavior is staying >=0 or staying <= 0, only getting beyond 0 count as turning point
        false_row = np.zeros((1, slope.shape[1])).astype(bool)
        # where the slope crosses 0 to -1 or reverse
        slope_sign = np.signbit(slope)
        zero_crossings = np.vstack((np.diff(slope_sign, axis=0), false_row))  # Use on earliest, fill end
        # where the rate of change of the rate of change crosses 0 to -1 or reverse
        accel = np.diff(slope, axis=0)
        changepoints = np.vstack((false_row, np.diff(np.signbit(accel), axis=0), false_row))
        if self.trend_anomaly_detector_params is not None:
            self.trend_anomaly_detector.fit(pd.DataFrame(
                np.where(changepoints, slope, 0),
                index=index, columns=self.column_names  # Need different source for index
            ))
        # Notes on other approaches
        # np.nonzero(np.diff(np.sign(slope), axis=0)) (this favors previous point, add 1 to be after)
        # replacing 0 with -1 above might work
        # https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
        # np.where(np.diff(np.signbit(a)))[0] (finds zero at beginning and not at end)
        # np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
        return zero_crossings, changepoints, slope_sign, accel

    def rolling_trend(self, trend_residuals, t):
        dates_2d = np.repeat(
            t[..., None],
            # df_holiday_scaled.index.to_julian_date().to_numpy()[..., None],
            trend_residuals.shape[1], axis=1
        )
        wind = 30 if self.trend_window is None else self.trend_window
        w_1 = wind - 1
        steps_ahd = int(w_1 / 2)
        y0 = np.repeat(np.array(trend_residuals[0:1]), steps_ahd, axis=0)
        d0 = -1 * dates_2d[1:y0.shape[0] + 1][::-1]
        shape2 = (w_1 - steps_ahd, y0.shape[1])
        y2 = np.concatenate(
            [
                y0,
                trend_residuals,
                np.full(shape2, np.nan),
            ]
        )
        d = np.concatenate(
            [
                d0,
                dates_2d,
                np.full(shape2, np.nan),
            ]
        )
        slope, intercept = window_lin_reg_mean(d, y2, wind)
        trend_posterior = slope * t[..., None] + intercept
        return trend_posterior, slope, intercept

    def _predict_linear(self, dates, history_df, future_regressor, regressor_per_series, flag_regressors, impacts, return_components=False):
        # accepts any date in history (or lag beyond) as long as regressors include those dates in index as well

        self.t_predict = self.create_t(dates)
        x_list = []
        if isinstance(self.anomaly_intervention, dict):
            # need to model these for prediction
            x_list.append(self.anomaly_detector.scores)
            return NotImplemented
        # all of the following are 1 day past lagged
        if self.multivariate_feature is not None:
            # includes backfill
            full_idx = history_df.index.union(self.create_forecast_index(forecast_length=1, last_date=history_df.index[-1]))
            lag_1_indx = np.concatenate([[0], np.arange(len(history_df))])
            trs_df = self.multivariate_transformer.transform(history_df)
            if self.multivariate_feature == "feature_agglomeration":

                x_list.append(pd.DataFrame(
                    self.agglomerator.transform(trs_df)[lag_1_indx],
                    index=full_idx,
                    columns=["multivar_" + str(x) for x in range(self.agglom_n_clusters)]
                ).reindex(dates))
            elif self.multivariate_feature == "group_average":
                multivar_df = trs_df.groupby(self.categorical_groups, axis=1).mean().iloc[lag_1_indx]
                multivar_df.index = full_idx
                x_list.append(multivar_df.reindex(dates).rename(columns=lambda x: "multivar_" + str(x)))
            elif self.multivariate_feature == "oscillator":
                return NotImplemented
        if self.seasonalities is not None:
            s_list = []
            for seasonality in self.seasonalities:
                s_list.append(create_seasonality_feature(dates, self.t_predict, seasonality, history_days=self.history_days))
                # INTERACTIONS NOT IMPLEMENTED
                # ORDER SPECIFICATION NOT IMPLEMENTED
            s_df = pd.concat(s_list, axis=1)
            s_df.index = dates
            x_list.append(s_df)
        if future_regressor is not None and self.regressors_used:
            x_list.append(future_regressor.reindex(dates))
        if flag_regressors is not None and self.regressors_used:
            x_list.append(flag_regressors.reindex(dates))  # doesn't check for missing data
        if self.holiday_countries is not None and not isinstance(self.holiday_countries, dict) and self.holiday_countries_used:
            for holiday_country in self.holiday_countries:
                x_list.append(
                    holiday_flag(
                        dates,
                        country=holiday_country,
                        encode_holiday_type=True,
                    ).rename(columns=lambda x: "holiday_" + str(x)),
                )
        # put this to the end as it takes up lots of feature space sometimes
        if self.holiday_detector_params is not None:
            x_list.append(self.holiday_detector.dates_to_holidays(dates, style="flag").clip(upper=1).rename(columns=lambda x: "holiday_" + str(x)))

        # FINAL FEATURE PROCESSING
        x_array = pd.concat(x_list, axis=1)
        self.predict_x_array = x_array  # can remove this later, it is for debugging
        if np.any(np.isnan(x_array.astype(float))):  # remove later, for debugging
            raise ValueError("nan values in predict_x_array")

        # RUN LINEAR MODEL
        # add x features that don't apply to all, and need to be looped
        if self.loop_required:
            self.predict_x_array = {}
            self.components = []
            predicts = []
            for col in self.df.columns:
                c_x = x_array.copy()
                # implement per_series holiday countries flag
                if isinstance(self.holiday_countries, dict) and self.holiday_countries_used:
                    hc = self.holiday_countries.get(col, None)
                    if hc is not None:
                        c_hc = holiday_flag(
                            dates,
                            country=hc,
                            encode_holiday_type=True,
                        ).rename(columns=lambda x: "holiday_" + str(x))
                    else:
                        c_hc = pd.DataFrame(0, index=dates, columns=["holiday_0"])
                    c_x = pd.concat([
                        c_x,
                        c_hc
                    ], axis=1)
                # implement regressors per series
                if isinstance(regressor_per_series, dict) and self.regressors_used:
                    hc = regressor_per_series.get(col, None)
                    if hc is not None:
                        c_hc = pd.DataFrame(hc).rename(columns=lambda x: "regrperseries_" + str(x)).reindex(dates)
                    else:
                        c_hc = pd.DataFrame(0, index=dates, columns=["regrperseries_0"])
                    c_x = pd.concat([
                        c_x,
                        c_hc
                    ], axis=1)
                # implement past_impacts as regressor
                if isinstance(impacts, pd.DataFrame) and self.past_impacts_intervention == "regressor":
                    c_x = pd.concat([
                        c_x,
                        impacts.reindex(dates)[col].to_frame().fillna(0).rename(columns=lambda x: "impacts_" + str(x))
                    ], axis=1)
                # add AR features
                if self.ar_lags is not None:
                    # somewhat inefficient to create full df of dates, but simplest this way for 'any date'
                    for lag in self.ar_lags:
                        full_idx = history_df.index.union(self.create_forecast_index(forecast_length=lag, last_date=history_df.index[-1]))
                        lag_idx = np.concatenate([np.repeat([0], lag), np.arange(len(history_df))])  # [0:len(history_df)]
                        lag_s = history_df[col].iloc[lag_idx].rename(f"lag{lag}_")
                        lag_s.index = full_idx
                        c_x = pd.concat([
                            c_x,
                            lag_s.reindex(dates)
                        ], axis=1)

                # ADDING RECENCY WEIGHTING AND RIDGE PARAMS
                if np.any(np.isnan(c_x.astype(float))):  # remove later, for debugging
                    raise ValueError("nan values in predict c_x_array")
                predicts.append(
                    pd.Series(np.dot(c_x[self.keep_cols[col]], self.params[col][self.keep_cols_idx[col]]).flatten(), name=col, index=dates)
                )
                self.predict_x_array[col] = c_x
                if return_components:
                    indices = [tuple(group)[-1][0] for key, group in groupby(enumerate(self.col_groupings[col]), key=itemgetter(1))][:-1]
                    new_indx = [0] + [x + 1 for x in indices]
                    temp = c_x[self.keep_cols[col]] * self.params[col][self.keep_cols_idx[col]].flatten()
                    self.components.append(np.add.reduceat(np.asarray(temp), sorted(new_indx), axis=1))
            if return_components:
                self.components = np.moveaxis(np.array(self.components), 0, 2)
                self.component_indx = new_indx  # hopefully no mismatching for diff series
            return pd.concat(predicts, axis=1)
        else:
            # run model
            res = np.dot(x_array[self.keep_cols], self.params[self.keep_cols_idx])
            if return_components:
                arr = x_array[self.keep_cols].to_numpy()
                temp = (np.moveaxis(np.broadcast_to(arr, [self.params.shape[1], arr.shape[0], arr.shape[1]]), 0, 2) * self.params[self.keep_cols_idx])
                indices = [tuple(group)[-1][0] for key, group in groupby(enumerate(self.col_groupings), key=itemgetter(1))][:-1]
                new_indx = [0] + [x + 1 for x in indices]
                self.component_indx = new_indx
                self.components = np.add.reduceat(temp, sorted(new_indx), axis=1)  # (dates, comps, series)
                # np.allclose(np.add.reduceat(temp, [0], axis=1)[:, 0, :], np.dot(mod.predict_x_array[mod.keep_cols], mod.params[mod.keep_cols_idx]))
            return res

    def process_components(self, to_origin_space=True):
        """Scale and standardize component outputs."""
        if self.components is None:
            raise ValueError("Model has not yet had a prediction generated.")

        comp_list = []
        t_indx = next(iter(self.predict_x_array.values())).index if isinstance(self.predict_x_array, dict) else self.predict_x_array.index
        col_group = next(iter(self.col_groupings.values())) if isinstance(self.col_groupings, dict) else self.col_groupings
        # unfortunately, no choice but to loop that I see to apply inverse trans
        for comp in range(self.components.shape[1]):
            if to_origin_space:
                # will have issues on inverse with some transformers
                comp_df = self.to_origin_space(pd.DataFrame(
                    self.components[:, comp, :],
                    index=t_indx, columns=self.column_names,
                ), components=True)
            else:
                comp_df = (pd.DataFrame(
                    self.components[:, comp, :],
                    index=t_indx, columns=self.column_names,
                ))
            # column name needs to be unlikely to occur in an actual dataset
            comp_df['csmod_component'] = col_group[self.component_indx[comp]]
            comp_list.append(comp_df)
        return pd.pivot(pd.concat(comp_list, axis=0), columns='csmod_component')

    def _predict_step(self, dates, trend_component, history_df, future_regressor, flag_regressors, impacts, regressor_per_series):
        # Note this is scaled and doesn't account for impacts
        linear_pred = self._predict_linear(
            dates, history_df=history_df, future_regressor=future_regressor,
            flag_regressors=flag_regressors, impacts=impacts, regressor_per_series=regressor_per_series,
            return_components=True
        )
        # ADD PREPROCESSING BEFORE TREND (FIT X, REVERSE on PREDICT, THEN TREND)

        upper = trend_component.upper_forecast.reindex(dates) + linear_pred + self.residual_uncertainty * self.int_std_dev
        lower = trend_component.lower_forecast.reindex(dates) + linear_pred - self.residual_uncertainty * self.int_std_dev
        df_forecast = PredictionObject(
            model_name=self.name,
            forecast_length=len(dates),
            forecast_index=dates,
            forecast_columns=trend_component.forecast.columns,
            lower_forecast=lower,
            forecast=trend_component.forecast.reindex(dates) + linear_pred,
            upper_forecast=upper,
            prediction_interval=self.prediction_interval,
            fit_runtime=self.fit_runtime,
            model_parameters=self.get_params(),
        )
        return df_forecast

    def predict(
            self, forecast_length, include_history=False, future_regressor=None,
            regressor_per_series=None, flag_regressors=None, future_impacts=None, new_df=None,
            regressor_forecast_model=None, regressor_forecast_model_params=None, regressor_forecast_transformations=None,
    ):
        """Generate a forecast."""
        predictStartTime = self.time()
        if self.trend_train is None:
            raise ValueError("Cassandra must first be .fit() successfully.")

        # scale new_df if given
        if new_df is not None:
            df = self.scale_data(new_df)
        else:
            df = self.df.copy()
        # if future regressors are None (& USED), but were provided for history, instead use forecasts of these features (warn)
        full_regr = None
        if future_regressor is None and self.future_regressor_train is not None and forecast_length is not None:
            print("future_regressor not provided, using forecasts of historical")
            future_regressor = model_forecast(
                model_name=self.trend_model['Model'] if regressor_forecast_model is None else regressor_forecast_model,
                model_param_dict=self.trend_model['ModelParameters'] if regressor_forecast_model_params is None else regressor_forecast_model_params,
                model_transform_dict=self.preprocessing_transformation if regressor_forecast_transformations is None else regressor_forecast_transformations,
                df_train=self.future_regressor_train,
                forecast_length=forecast_length,
                frequency=self.frequency,
                prediction_interval=self.prediction_interval,
                fail_on_forecast_nan=False,
                random_seed=self.random_seed,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            ).forecast
            full_regr = pd.concat([self.future_regressor_train, future_regressor])
        if future_regressor is not None and self.regressors_used:
            full_regr = pd.concat([self.future_regressor_train, self.regressor_transformer.fit_transform(clean_regressor(future_regressor))])
        if flag_regressors is not None and forecast_length is not None and self.regressors_used:
            all_flags = pd.concat([self.flag_regressor_train, clean_regressor(flag_regressors, prefix="regrflags_")])
        else:
            if self.flag_regressor_train is not None and forecast_length is not None and self.regressors_used:
                raise ValueError("flag_regressors supplied in training but not predict")
            all_flags = self.flag_regressor_train
        if future_impacts is not None and forecast_length is not None:
            impacts = pd.concat([self.past_impacts, future_impacts])
        else:
            impacts = self.past_impacts
        # I don't think there is a more efficient way to combine these dicts of dataframes
        if regressor_per_series is not None and self.regressors_used:
            if not isinstance(regressor_per_series, dict):
                raise ValueError("regressor_per_series must be dict")
            regr_ps_fore = {}
            for key, value in self.regr_per_series_tr.items():
                regr_ps_fore[key] = pd.concat([self.regr_per_series_tr[key], regressor_per_series[key]])
        else:
            regr_ps_fore = self.regr_per_series_tr

        # generate trend
        # MAY WANT TO PASS future_regressor HERE
        if forecast_length is not None:
            # combine regressor types depending on what is given
            if self.future_regressor_train is None and self.flag_regressor_train is not None:
                comp_regr_train = self.flag_regressor_train
                comp_regr = flag_regressors
            elif self.future_regressor_train is not None and self.flag_regressor_train is None:
                comp_regr_train = self.future_regressor_train
                comp_regr = future_regressor
            elif self.future_regressor_train is not None and self.flag_regressor_train is not None:
                comp_regr_train = pd.concat([self.future_regressor_train, self.flag_regressor_train], axis=1)
                comp_regr = pd.concat([future_regressor, flag_regressors], axis=1)
            else:
                comp_regr_train = None
                comp_regr = None
            resid = None
            # create new rolling residual if new data provided
            if new_df is not None:
                resid = df - self._predict_linear(
                    dates=df.index, history_df=df,
                    future_regressor=full_regr, flag_regressors=all_flags,
                    impacts=impacts, regressor_per_series=regr_ps_fore,
                )
                if self.rolling_window is not None:
                    resid = self.rolling_trend(resid, np.array(self.create_t(df.index)))
            trend_forecast = model_forecast(
                model_name=self.trend_model['Model'],
                model_param_dict=self.trend_model['ModelParameters'],
                model_transform_dict=self.trend_transformation,
                df_train=self.trend_train if resid is None else resid,
                forecast_length=forecast_length,
                frequency=self.frequency,
                prediction_interval=self.prediction_interval,
                # no_negatives=no_negatives,
                # constraint=constraint,
                future_regressor_train=comp_regr_train,
                future_regressor_forecast=comp_regr,
                # holiday_country=holiday_country,
                fail_on_forecast_nan=True,
                random_seed=self.random_seed,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            # phi is on future predict step only
            if self.trend_phi is not None and self.trend_phi != 1:
                temp = trend_forecast.forecast.mul(
                    pd.Series([self.trend_phi] * trend_forecast.forecast.shape[0], index=trend_forecast.forecast.index).pow(
                        range(trend_forecast.forecast.shape[0])
                    ),
                    axis=0,
                )
                # overwrite bounds if using phi
                trend_forecast.forecast = trend_forecast.forecast + temp
                trend_forecast.upper_forecast = trend_forecast.forecast
                trend_forecast.lower_forecast = trend_forecast.forecast
            if include_history:
                trend_forecast.forecast = pd.concat([
                    self.trend_train, trend_forecast.forecast
                ])
                trend_forecast.lower_forecast = pd.concat([
                    self.trend_train, trend_forecast.lower_forecast
                ])
                trend_forecast.upper_forecast = pd.concat([
                    self.trend_train, trend_forecast.upper_forecast
                ])
        else:
            trend_forecast = PredictionObject(
                forecast=self.trend_train, lower_forecast=self.trend_train, upper_forecast=self.trend_train
            )

        # ar_lags, multivariate features require 1 step loop
        if forecast_length is None:
            df_forecast = self._predict_step(
                dates=df.index, trend_component=trend_forecast, history_df=df,
                future_regressor=full_regr, flag_regressors=all_flags,
                impacts=impacts, regressor_per_series=regr_ps_fore,
            )
        elif self.predict_loop_req:
            for step in range(forecast_length):
                forecast_index = df.index.union(self.create_forecast_index(1, last_date=df.index[-1]))
                df_forecast = self._predict_step(
                    dates=forecast_index, trend_component=trend_forecast, history_df=df,
                    future_regressor=full_regr, flag_regressors=all_flags,
                    impacts=impacts, regressor_per_series=regr_ps_fore,
                )
                df = pd.concat([df, df_forecast.forecast.iloc[-1:]])
            if not include_history:
                df_forecast.forecast = df_forecast.forecast.tail(forecast_length)
                df_forecast.lower_forecast = df_forecast.lower_forecast.tail(forecast_length)
                df_forecast.upper_forecast = df_forecast.upper_forecast.tail(forecast_length)
        else:
            forecast_index = self.create_forecast_index(forecast_length)
            if include_history:
                forecast_index = df.index.union(forecast_index)
            df_forecast = self._predict_step(
                dates=forecast_index, trend_component=trend_forecast, history_df=df,
                future_regressor=full_regr, flag_regressors=all_flags,
                impacts=impacts, regressor_per_series=regr_ps_fore,
            )

        # undo preprocessing and scaling
        # account for some transformers requiring different methods on original data and forecast
        if forecast_length is None:
            df_forecast.forecast = self.to_origin_space(df_forecast.forecast, trans_method='original')
            df_forecast.lower_forecast = self.to_origin_space(df_forecast.lower_forecast, trans_method='original')
            df_forecast.upper_forecast = self.to_origin_space(df_forecast.upper_forecast, trans_method='original')
            self.predicted_trend = self.to_origin_space(trend_forecast.forecast, trans_method='original')
        elif not include_history:
            df_forecast.forecast = self.to_origin_space(df_forecast.forecast, trans_method='forecast')
            df_forecast.lower_forecast = self.to_origin_space(df_forecast.lower_forecast, trans_method='forecast')
            df_forecast.upper_forecast = self.to_origin_space(df_forecast.upper_forecast, trans_method='forecast')
            self.predicted_trend = self.to_origin_space(trend_forecast.forecast, trans_method='forecast')
        else:
            hdn = len(df_forecast.forecast) - forecast_length
            df_forecast.forecast = pd.concat([
                self.to_origin_space(df_forecast.forecast.head(hdn), trans_method='original'),
                self.to_origin_space(df_forecast.forecast.tail(forecast_length), trans_method='forecast'),
            ])
            df_forecast.lower_forecast = pd.concat([
                self.to_origin_space(df_forecast.lower_forecast.head(hdn), trans_method='original', bounds=True),
                self.to_origin_space(df_forecast.lower_forecast.tail(forecast_length), trans_method='forecast', bounds=True),
            ])
            df_forecast.upper_forecast = pd.concat([
                self.to_origin_space(df_forecast.upper_forecast.head(hdn), trans_method='original', bounds=True),
                self.to_origin_space(df_forecast.upper_forecast.tail(forecast_length), trans_method='forecast', bounds=True),
            ])
            self.predicted_trend = pd.concat([
                self.to_origin_space(trend_forecast.forecast.head(hdn), trans_method='original'),
                self.to_origin_space(trend_forecast.forecast.tail(forecast_length), trans_method='forecast'),
            ])

        # update trend analysis based on trend forecast as well
        if forecast_length is not None and include_history:
            trend_posterior, self.slope, intercept = self.rolling_trend(self.predicted_trend, np.array(self.t_predict))
            self.zero_crossings, self.changepoints, self.slope_sign, self.accel = self.analyze_trend(self.slope, index=self.predicted_trend.index)

        # don't forget to add in past_impacts (use future impacts again?) AFTER unscaling
        if future_impacts is not None and self.past_impacts is None:
            past_impacts = pd.DataFrame(0, index=self.df.index, columns=self.df.columns)
        else:
            past_impacts = self.past_impacts
        # roll forward tail of past impacts, assuming it continues
        if self.past_impacts is not None and forecast_length is not None:
            future_impts = pd.DataFrame(
                np.repeat(self.past_impacts.iloc[-1:].to_numpy(), forecast_length, axis=0),
                index=forecast_index, columns=self.df.columns
            )
            if future_impacts is not None:
                future_impts = future_impts + future_impacts
        else:
            future_impts = pd.DataFrame()
        if self.past_impacts is not None or future_impacts is not None:
            impts = 1 + pd.concat([past_impacts, future_impts], axis=1)
            df_forecast.forecast = df_forecast.forecast * impts
            df_forecast.lower_forecast = df_forecast.lower_forecast * impts
            df_forecast.upper_forecast = df_forecast.upper_forecast * impts

        if self.constraint is not None:
            if isinstance(self.constraint, dict):
                constraint_method = self.constraint.get("constraint_method", "quantile")
                constraint_regularization = self.constraint.get("constraint_regularization", 1)
                lower_constraint = self.constraint.get("lower_constraint", 0)
                upper_constraint = self.constraint.get("upper_constraint", 1)
                bounds = self.constraint.get("bounds", False)
            else:
                constraint_method = "stdev_min"
                lower_constraint = float(self.constraint)
                upper_constraint = float(self.constraint)
                constraint_regularization = 1
                bounds = False
            if self.verbose >= 3:
                print(
                    f"Using constraint with method: {constraint_method}, {constraint_regularization}, {lower_constraint}, {upper_constraint}, {bounds}"
                )

            df_forecast = df_forecast.apply_constraints(
                constraint_method,
                constraint_regularization,
                upper_constraint,
                lower_constraint,
                bounds,
                self.df_original,
            )
        # RETURN COMPONENTS (long style) option
        df_forecast.predict_runtime = self.time() - predictStartTime
        return df_forecast

    def auto_fit(self, df, validation_method):  # also add regressor input
        # option to completely skip some things (anomalies, holiday detector, ar lags)
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

    def feature_importance(self):
        # rank coefficients by importance
        return NotImplemented

    def compare_actual_components(self):
        return NotImplemented

    @staticmethod
    def get_new_params(method='fast'):
        # have fast option that avoids any of the loop approaches
        scaling = random.choices(['BaseScaler', 'other'], [0.8, 0.2])[0]
        if scaling == "other":
            scaling = RandomTransform(
                transformer_list=scalers, transformer_max_depth=1,
                allow_none=False, no_nan_fill=True
            )
        holiday_params = random.choices([None, 'any'], [0.5, 0.5])[0]
        # holiday_intervention = None
        if holiday_params is not None:
            # holiday_intervention = random.choice(['create_feature', 'use_impact'])
            holiday_params = HolidayTransformer.get_new_params(method=method)
            holiday_params['impact'] = None
            holiday_params['regression_params'] = None
            holiday_params['remove_excess_anomalies'] = random.choices(
                [True, False], [0.05, 0.95]
            )[0]
            holiday_params['output'] = random.choices(['multivariate', 'univariate'], [0.9, 0.1])[0]
        anomaly_intervention = random.choices([None, 'remove', 'detect_only', 'model'], [0.9, 0.3, 0.05, 0.1])[0]
        if anomaly_intervention is not None:
            anomaly_detector_params = AnomalyRemoval.get_new_params(method=method)
            if anomaly_intervention == "model":
                anomaly_intervention = general_template.sample(1).to_dict("records")[0]  # placeholder, probably
        else:
            anomaly_detector_params = None
        model_str = random.choices(['AverageValueNaive', 'MetricMotif', "LastValueNaive"], [0.2, 0.7, 0.1], k=1)[0]
        trend_model = {'Model': model_str}
        trend_model['ModelParameters'] = ModelMonster(model_str).get_new_params(method=method)

        trend_anomaly_intervention = random.choices([None, 'detect_only'], [0.5, 0.5])[0]
        if trend_anomaly_intervention is not None:
            trend_anomaly_detector_params = AnomalyRemoval.get_new_params(method=method)
        else:
            trend_anomaly_detector_params = None
        linear_model = random.choices(['lstsq', 'linalg_solve', 'l1_norm'], [0.6, 0.2, 0.1])[0]
        recency_weighting = random.choices([None, 0.05, 0.1, 0.25], [0.7, 0.1, 0.1, 0.1])[0]
        if linear_model in ['lstsq']:
            linear_model = {
                'model': linear_model,
                'lambda': random.choices([None, 0.1, 1, 10], [0.7, 0.1, 0.1, 0.1])[0],
                'recency_weighting': recency_weighting,
            }
        if linear_model in ['linalg_solve']:
            linear_model = {
                'model': linear_model,
                'lambda': random.choices([0, 0.1, 1, 10], [0.4, 0.2, 0.2, 0.2])[0],
                'recency_weighting': recency_weighting,
            }
        elif linear_model == 'l1_norm':
            linear_model = {
                'model': linear_model,
                'recency_weighting': recency_weighting,
                'maxiter': random.choices([250, 15000, 25000], [0.2, 0.6, 0.2])[0],
            }
        if method == "regressor":
            regressors_used = True
        else:
            regressors_used = random.choices([True, False], [0.5, 0.5])[0]
        return {
            "preprocessing_transformation": RandomTransform(
                transformer_list=filters, transformer_max_depth=2, allow_none=True
            ),
            "scaling": scaling,
            # "past_impacts_intervention": self.past_impacts_intervention,
            "seasonalities": random.choices(
                [[7, 365.25], ["dayofweek", 365.25], ["month", "dayofweek", "weekdayofmonth"]],
                [0.1, 0.1, 0.1],
            )[0],
            "ar_lags": random.choices(
                [None, [1], [1, 7], [7]],
                [0.9, 0.025, 0.025, 0.05],
            )[0],
            "ar_interaction_seasonality": None,
            "anomaly_detector_params": anomaly_detector_params,
            "anomaly_intervention": anomaly_intervention,
            "holiday_detector_params": holiday_params,
            # "holiday_countries": self.holiday_countries,
            "holiday_countries_used": random.choices([True, False], [0.5, 0.5])[0],
            "multivariate_feature": random.choices(
                [None, "feature_agglomeration", 'group_average', 'oscillator'],
                [0.9, 0.1, 0.1, 0.01]
            )[0],
            "multivariate_transformation": RandomTransform(
                transformer_list="fast", transformer_max_depth=3  # probably want some more usable defaults first as many random are senseless
            ),
            "regressor_transformation": RandomTransform(
                transformer_list={**scalers, **decompositions}, transformer_max_depth=1,
                allow_none=False, no_nan_fill=False  # probably want some more usable defaults first as many random are senseless
            ),
            "regressors_used": regressors_used,
            "linear_model": linear_model,
            "randomwalk_n": random.choices([None, 10], [0.5, 0.5])[0],
            "trend_window": random.choices([3, 15, 90, 365], [0.2, 0.2, 0.2, 0.2])[0],
            "trend_standin": random.choices(
                [None, 'random_normal', 'rolling_trend'],
                [0.5, 0.4, 0.01],
            )[0],
            "trend_anomaly_detector_params": trend_anomaly_detector_params,
            # "trend_anomaly_intervention": trend_anomaly_intervention,
            "trend_transformation": RandomTransform(
                transformer_list="fast", transformer_max_depth=3  # probably want some more usable defaults first as many random are senseless
            ),
            "trend_model": trend_model,
            "trend_phi": random.choices([None, 0.98], [0.9, 0.1])[0],
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
            # "holiday_intervention": self.holiday_intervention,
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
            # "trend_anomaly_intervention": self.trend_anomaly_intervention,
            "trend_transformation": self.trend_transformation,
            "trend_model": self.trend_model,
            "trend_phi": self.trend_phi,
            # "constraint": self.constraint,
        }

    def plot_components(self, prediction=None, series=None, figsize=(16, 9), to_origin_space=True, title=None):
        if series is None:
            series = random.choice(self.column_names)
        if title is None:
            title = f"Model Components for {series}"
        plot_list = []
        if prediction is not None:
            plot_list.append(prediction.forecast[series].rename("forecast"))
            plot_list.append(self.predicted_trend[series].rename("trend"))
        plot_list.append(self.process_components(to_origin_space=to_origin_space)[series])
        return pd.concat(plot_list, axis=1).plot(subplots=True, figsize=figsize, title=title)

    def plot_trend(
            self, series=None, vline=None,
            colors=["#d4f74f", "#82ab5a", "#c12600", "#ff6c05"],
            title=None, start_date=None, **kwargs
    ):
        # YMAX from PLOT ONLY
        if series is None:
            series = random.choice(self.column_names)
        if title is None:
            title = f"Trend Breakdown for {series}"
        p_indx = self.column_names.get_loc(series)
        cur_trend = self.predicted_trend[series].copy()
        plot_df = pd.DataFrame({
            'decline_accelerating': cur_trend[np.hstack((np.signbit(self.accel[:, p_indx]), False)) & self.slope_sign[:, p_indx]],
            'decline_decelerating': cur_trend[(~ np.hstack((np.signbit(self.accel[:, p_indx]), False))) & self.slope_sign[:, p_indx]],
            'growth_decelerating': cur_trend[np.hstack((np.signbit(self.accel[:, p_indx]), False)) & (~ self.slope_sign[:, p_indx])],
            'growth_accelerating': cur_trend[(~ np.hstack((np.signbit(self.accel[:, p_indx]), False))) & (~ self.slope_sign[:, p_indx])],
        }, index=cur_trend.index)
        if start_date is not None:
            plot_df = plot_df[plot_df.index >= start_date]
        ax = plot_df.plot(title=title, color=colors, **kwargs)
        # ax.scatter(cur_trend.index[self.changepoints[:, p_indx]], cur_trend[self.changepoints[:, p_indx]], c='#fdcc09', s=4.0)
        # ax.scatter(cur_trend.index[self.zero_crossings[:, p_indx]], cur_trend[self.zero_crossings[:, p_indx]], c='#512f74', s=4.0)
        if mod.trend_anomaly_detector is not None:
            if mod.trend_anomaly_detector.output == "univariate":
                i_anom = mod.trend_anomaly_detector.anomalies.index[mod.anomaly_detector.anomalies.iloc[:, 0] == -1]
            else:
                series_anom = mod.trend_anomaly_detector.anomalies[series]
                i_anom = series_anom[series_anom == -1].index
            if start_date is not None:
                i_anom = i_anom[i_anom >= start_date]
            # only plot if some anomalies, and not way too many anomalies
            if len(i_anom) > 0 and len(i_anom) < len(plot_df) * 0.5:
                ax.scatter(i_anom.tolist(), cur_trend.loc[i_anom], c="red", s=16.0)
        if vline is not None:
            ax.vlines(x=vline, ls='--', lw=1, colors='darkred', ymin=cur_trend[cur_trend.index >= start_date].min(), ymax=cur_trend[cur_trend.index >= start_date].max())
        return ax

    def plot_things():  # placeholder for later plotting functions
        # plot components
        # plot transformed df if preprocess or anomaly removal
        # plot past impacts
        # plot % contribution of components
        # plot eval (show actuals, alongside full and components to diagnose)
        # plot residual distribution/PACF
        # plot inflection points (filtering or smoothing first)
        # plot highest error series, plot highest/lowest growth
        # trend: one color for growth, another for decline (darker as accelerating, ligher as slowing)
        return NotImplemented


def clean_regressor(in_d, prefix="regr_"):
    if not isinstance(in_d, pd.DataFrame):
        df = pd.DataFrame(in_d)
    else:
        df = in_d.copy()
    df.columns = [prefix + col for col in df.columns]
    return df


def create_t(ds):
    return (ds - ds.min()) / (ds.max() - ds.min())


def fourier_series(t, p=365.25, n=10):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x


def create_seasonality_feature(DTindex, t, seasonality, history_days=None):
    # for consistency, all must have a range index, not date index
    # fourier orders
    if isinstance(seasonality, (int, float)):
        if history_days is None:
            history_days = (DTindex.max() - DTindex.min()).days
        return pd.DataFrame(fourier_series(np.asarray(t), seasonality / history_days, n=10)).rename(columns=lambda x: f"seasonality{seasonality}_" + str(x))
    # dateparts
    elif seasonality == "dayofweek":
        return pd.get_dummies(pd.Categorical(
            DTindex.weekday, categories=list(range(7)), ordered=True
        )).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "month":
        return pd.get_dummies(pd.Categorical(
            DTindex.month, categories=list(range(1, 13)), ordered=True
        )).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "weekend":
        return pd.DataFrame((DTindex.weekday > 4).astype(int), columns=["weekend"])
    elif seasonality == "weekdayofmonth":
        return pd.get_dummies(pd.Categorical(
            (DTindex.day - 1) // 7 + 1,
            categories=list(range(1, 6)), ordered=True,
        )).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "hour":
        return pd.get_dummies(pd.Categorical(
            DTindex.hour, categories=list(range(1, 25)), ordered=True
        )).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "daysinmonth":
        return pd.DataFrame({'daysinmonth': DTindex.daysinmonth})
    elif seasonality == "quarter":
        return pd.get_dummies(pd.Categorical(
            DTindex.quarter, categories=list(range(1, 5)), ordered=True
        )).rename(columns=lambda x: f"{seasonality}_" + str(x))
    else:
        return ValueError(f"Seasonality `{seasonality}` not recognized")


#####################################
# JUST LEAST SQUARES UNIVARIATE
# https://stackoverflow.com/questions/17679140/multiple-linear-regression-with-python

def lstsq_solve(X, y, lamb=1, identity_matrix=None):
    if identity_matrix is None:
        identity_matrix = np.zeros((X.shape[1], X.shape[1]))
        np.fill_diagonal(identity_matrix, 1)
        identity_matrix[0, 0] = 0
    XtX_lamb = X.T.dot(X) + lamb * identity_matrix
    XtY = X.T.dot(y)
    return np.linalg.solve(XtX_lamb, XtY)


def cost_function_l1(params, X, y):
    return np.sum(np.abs(y - np.dot(X, params.reshape(X.shape[1], y.shape[1]))))


def cost_function_quantile(params, X, y, q=0.9):
    cut = int(y.shape[0] * q)
    return np.sum(np.partition(np.abs(y - np.dot(X, params.reshape(X.shape[1], y.shape[1]))), cut, axis=0)[0:cut])


def cost_function_l2(params, X, y):
    return np.linalg.norm(y - np.dot(X, params.reshape(X.shape[1], y.shape[1])))


# could do partial pooling by minimizing a function that mixes shared and unshared coefficients (multiplicative)
def lstsq_minimize(X, y, maxiter=15000):
    """Any cost function version of lin reg."""
    # start with lstsq fit as estimated point
    x0 = lstsq_solve(X, y).flatten()
    # assuming scaled, these should be reasonable bounds
    bounds = [(-10, 10) for x in x0]
    return minimize(
        cost_function_l1, x0, args=(X, y), bounds=bounds,
        options={'maxiter': maxiter}
    ).x.reshape(X.shape[1], y.shape[1])


def linear_model(x, y, params):
    model_type = params.get("model", "lstsq")
    lambd = params.get("lambda", None)
    rec = params.get("recency_weighting", None)
    if lambd is not None:
        id_mat = np.zeros((x.shape[1], x.shape[1]))
        np.fill_diagonal(id_mat, 1)
        id_mat[0, 0] = 0
    if rec is not None:
        weights = ((np.arange(len(x)) + 1) ** rec)  # 0.05 - 0.25
        x = x * weights[..., None]
        y = np.asarray(y) * weights[..., None]
    if model_type == "lstsq":
        if lambd is not None:
            return np.linalg.lstsq(x.T.dot(x) + lambd * id_mat, x.T.dot(y), rcond=None)[0]
        else:
            return np.linalg.lstsq(x, y, rcond=None)[0]
    elif model_type == "linalg_solve":
        return lstsq_solve(x, y, lamb=lambd, identity_matrix=id_mat)
    elif model_type == "l1_norm":
        return lstsq_minimize(x, y, maxiter=params.get("maxiter", 15000))
    else:
        raise ValueError("linear model not recognized")

# Seasonalities
    # maybe fixed 3 seasonalities
    # interaction effect only on first two seasonalities if order matches
# Holiday countries (mapped countries per series)
    # make it holiday 1, holiday 2 so per country?
# Regressor (preprocessing)
    # PREPEND REGR_ to name of regressor features
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

# what is still needed:
    # more linear model options
    # anomaly modeling
    # seasonality interactions
    # ar seasonality interaction (or remove)
    # test and bug fix everything
    # PLOT IMPACTS
    # l1_norm isn't working

# TEST
    # new_df
    # impacts
    # regressor types

# could do partial pooling by minimizing a function that mixes shared and unshared coefficients (multiplicative)

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

categorical_groups = {
    "wiki_United_States": 'country',
    "wiki_Germany": 'country',
    "wiki_Jesus": 'holiday',
    "wiki_Michael_Jackson": 'person',
    "wiki_Easter": 'holiday',
    "wiki_Christmas": 'holiday',
    "wiki_Chinese_New_Year": 'holiday',
    "wiki_Thanksgiving": 'holiday',
    "wiki_Elizabeth_II": 'person',
    "wiki_William_Shakespeare": 'person',
    "wiki_George_Washington": 'person',
    "wiki_Cleopatra": 'person',
}
holiday_countries = {
    'wiki_Elizabeth_II': 'uk',
    'wiki_United_States': 'us',
    'wiki_Germany': 'de',
}

if False:
    # test holiday countries, regressors, impacts
    from autots import load_daily
    import matplotlib.pyplot as plt
    df_daily = load_daily(long=False)
    forecast_length = 180
    df_train = df_daily[:-forecast_length]
    df_test = df_daily[-forecast_length:]
    constraint = {
        'constraint_method': 'quantile',
        'lower_constraint': 0,
        'upper_constraint': None,
        "bounds": True
    }

    c_params = Cassandra.get_new_params()
    c_params

    mod = Cassandra(n_jobs=1, **c_params, constraint=constraint)
    mod.fit(df_train, categorical_groups=categorical_groups)
    include_history = True
    pred = mod.predict(forecast_length=forecast_length, include_history=include_history)
    result = pred.forecast
    with plt.style.context("seaborn-white"):
        series = random.choice(mod.column_names)
        start_date = "2019-07-01"
        ax = pred.plot(df_daily if include_history else df_test, series=series, vline=df_test.index[0], start_date=start_date)
        if mod.anomaly_detector:
            if mod.anomaly_detector.output == "univariate":
                i_anom = mod.anomaly_detector.anomalies.index[mod.anomaly_detector.anomalies.iloc[:, 0] == -1]
            else:
                series_anom = mod.anomaly_detector.anomalies[series]
                i_anom = series_anom[series_anom == -1].index
                i_anom = i_anom[i_anom >= start_date]
            if len(i_anom) > 0 and len(i_anom) < len(df_daily) * 0.5:
                ax.scatter(i_anom.tolist(), df_daily.loc[i_anom, :][series], c="red", s=12.0)
        if mod.holiday_detector:
            i_anom = mod.holiday_detector.dates_to_holidays(mod.df.index, style="series_flag")[series]
            i_anom = i_anom.index[i_anom == 1]
            if len(i_anom) > 0 and len(i_anom) < len(df_daily) * 0.5:
                ax.scatter(i_anom.tolist(), df_daily.loc[i_anom, :][series], c="darkgreen", s=12.0)
        if mod.trend_anomaly_detector is not None:
            if mod.trend_anomaly_detector.output == "univariate":
                i_anom = mod.trend_anomaly_detector.anomalies.index[mod.anomaly_detector.anomalies.iloc[:, 0] == -1]
            else:
                series_anom = mod.trend_anomaly_detector.anomalies[series]
                i_anom = series_anom[series_anom == -1].index
            if start_date is not None:
                i_anom = i_anom[i_anom >= start_date]
            if len(i_anom) > 0 and len(i_anom) < len(df_daily) * 0.5:
                ax.scatter(i_anom.tolist(), df_daily.loc[i_anom][series], c="fuchsia", s=12.0)
        plt.show()
        # plt.savefig("forecast.png", dpi=300)
        # mod.plot_components(pred, series=series, to_origin_space=False)
        # plt.show()
        mod.plot_components(pred, series=series, to_origin_space=True)
        # plt.savefig("components.png", dpi=300)
        plt.show()
        mod.plot_trend(series=series, vline=df_test.index[0], start_date=start_date)
        # plt.savefig("trend.png", dpi=300)
    pred.evaluate(df_daily.reindex(result.index) if include_history else df_test)
    print(pred.avg_metrics.round(1))
    print(c_params['trend_model'])

# MULTIPLICATIVE SEASONALITY AND HOLIDAYS

# low memory option to delete stored dfs and arrays as soon as possible

# Make sure x features are all scaled
# Remove seasonality from regressors

# transfer learning
# graphics (AMFM watermark)
     # compare across past forecasts
# stability
# make it more modular (separate usable functions)

# Automation
# allow some config inputs, or automated fit
# output to table
# compare coefficients change over time, accuracy over time
# comparing different sources? and/or allowing one forecast to be used as a regressor for another
# would allow for earlier detection of broken connections
