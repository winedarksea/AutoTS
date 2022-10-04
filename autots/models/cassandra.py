# -*- coding: utf-8 -*-
"""Cassandra Model.
Created on Tue Sep 13 19:45:57 2022

@author: Colin
"""
import random
import numpy as np
import pandas as pd
# using transformer version of Anomaly/Holiday to use a lower level import than evaluator
from autots.tools.transform import GeneralTransformer, RandomTransform, scalers, filters, HolidayTransformer, AnomalyRemoval, EmptyTransformer
from autots.tools import cpu_count
from autots.models.base import ModelObject, PredictionObject
from autots.templates.general import general_template
from autots.tools.holiday import holiday_flag
from autots.tools.window_functions import sliding_window_view, window_lin_reg_mean
from autots.evaluator.auto_model import ModelMonster, model_forecast
# scipy is technically optional but most likely is present
try:
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

    Args:
        pass
    """

    def __init__(
        self,
        preprocessing_transformation: dict = None,  # filters by default only
        scaling: str = "BaseScaler",  # pulled out from transformation as a scaler is not optional, maybe allow a list
        past_impacts_intervention: str = None,  # 'remove', 'plot_only', 'regressor'
        seasonalities: dict = {},  # interactions added if fourier and order matches
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
        trend_model: dict = {},  # have one or two in built, then redirect to any AutoTS model for other choices
        trend_phi: float = None,
        constraint: dict = None,
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
        self.holiday_countries = holiday_countries
        if isinstance(self.holiday_countries, str):
            self.holiday_countries = self.holiday_countries.split(",")
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

    def to_origin_space(self, df, trans_method='forecast'):
        """Take transformed outputs back to original feature space."""
        if self.scaling == "BaseScaler":
            return self.preprocesser.inverse_transform(df, trans_method=trans_method) * self.scaler_std + self.scaler_mean
        else:
            return self.scaler.inverse_transform(
                self.preprocesser.inverse_transform(df, trans_method=trans_method),
                trans_method=trans_method,
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
            anomaly_not_uni = self.anomaly_detector_params.get('output', None) != 'univariate'
        self.anomaly_not_uni = False
        self.loop_required = (
            (self.ar_lags is not None) or (
            ) or (
                isinstance(self.holiday_countries, dict) and self.holiday_countries_used
            ) or (
                (regressor_per_series is not None) and self.regressors_used
            ) or (
                isinstance(self.anomaly_intervention, dict) and anomaly_not_uni
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
                NotImplemented
            # detect_only = pass
        # now do standard preprocessing
        if self.preprocessing_transformation is not None:
            self.preprocesser = GeneralTransformer(**self.preprocessing_transformation)
            self.df = self.preprocesser.fit_transform(self.df)
        if self.scaling is not None:
            if self.scaling == "BaseScaler":
                self.df = self.base_scaler(self.df)
            else:
                self.scaler = GeneralTransformer(**self.scaling)
                self.df = self.scaler.fit_transform(self.df)
        # additional transforms before multivariate feature creation
        if self.multivariate_transformation is not None:
            self.multivariate_transformer = GeneralTransformer(**self.multivariate_transformation)
        # needs to come after preprocessing because of 'slice' transformer
        self.ds_min = self.df.index.min()
        self.ds_max = self.df.index.max()
        self.t_train = self.create_t(self.df.index)

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
                x_list.append(multivar_df)
            elif self.multivariate_feature == "oscillator":
                return NotImplemented
                np.count_nonzero((df - df.shift(1)).clip(upper=0))[:-1]
        if self.seasonalities is not None:
            s_list = []
            for seasonality in self.seasonalities:
                s_list.append(create_seasonality_feature(self.df.index, self.t_train, seasonality))
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
                NotImplemented
        self.future_regressor_train = None
        if future_regressor is not None and self.regressors_used:
            if self.regressor_transformation is not None:
                self.regressor_transformer = GeneralTransformer(**self.regressor_transformation)
                self.future_regressor_train = self.regressor_transformer.fit_transform(clean_regressor(future_regressor))
            x_list.append(self.future_regressor_train)
        self.flag_regressor_train = None
        if flag_regressors is not None and self.regressors_used:
            self.flag_regressor_train = clean_regressor(flag_regressors, prefix="regrflags_")
            x_list.append(self.flag_regressor_train)
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
        self.regr_per_series_tr = None
        if self.loop_required:
            self.params = {}
            self.keep_cols = {}
            self.x_array = {}
            self.keep_cols_idx = {}
            trend_residuals = []
            for col in self.df.columns:
                c_x = x_array.copy()
                # implement per_series holiday countries flag
                if isinstance(self.holiday_countries, dict) and self.holiday_countries_used:
                    hc = self.holiday_countries.get(col, None)
                    if hc is not None:
                        c_x = pd.concat([
                            c_x,
                            holiday_flag(
                                self.df.index,
                                country=hc,
                                encode_holiday_type=True,
                            ).rename(columns=lambda x: "holiday_" + str(x))
                        ], axis=1)
                # implement regressors per series
                if isinstance(regressor_per_series, dict) and self.regressors_used:
                    self.regr_per_series_tr = regressor_per_series
                    hc = regressor_per_series.get(col, None)
                    if hc is not None:
                        c_x = pd.concat([
                            c_x,
                            pd.DataFrame(hc).rename(columns=lambda x: "regrperseries_" + str(x))
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
                c_x['intercept'] = 1
                # ADDING RECENCY WEIGHTING AND RIDGE PARAMS
                self.params[col] = linear_model(c_x, self.df[col])
                trend_residuals.append(
                    self.df[col] - pd.Series(np.dot(c_x[self.keep_cols[col]], self.params[col][self.keep_cols_idx[col]]), name=col, index=self.df.index)
                )
                self.x_array[col] = c_x
            trend_residuals = pd.concat(trend_residuals, axis=1)
        else:
            # RUN LINEAR MODEL, WHEN NO LOOPED FEATURES
            self.keep_cols = x_array.columns[~x_array.columns.str.contains("|".join(remove_patterns))]
            self.keep_cols_idx = x_array.columns.get_indexer_for(self.keep_cols)
            x_array['intercept'] = 1
            # run model
            self.params = linear_model(x_array, self.df)
            if self.linear_model == 'something_else':
                # ADDING RECENCY WEIGHTING AND RIDGE PARAMS
                NotImplemented
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
        # INFLECTION POINTS (cross zero), CHANGEPOINTS (trend of trend changes, on bigger window) (DIFF then find 0)
        zero_crossings = np.diff(np.signbit(slope), axis=0)  # Use on earliest, fill end
        accel = np.diff(slope, axis=0)
        changepoints = np.diff(np.signbit(accel), axis=0)
        # desired behavior is staying >=0 or staying <= 0, only getting beyond 0 count as turning point
        # np.nonzero(np.diff(np.sign(slope), axis=0)) (this favors previous point, add 1 to be after)
        # replacing 0 with -1 above might work
        # https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
        # np.where(np.diff(np.signbit(a)))[0] (finds zero at beginning and not at end)
        # np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
        # LEVEL SHIFT not yet accounted for (anomaly on SLOPE @ CHNG PTS)
        if self.trend_anomaly_detector_params is not None:
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

    def _predict_linear(self, dates, history_df, future_regressor, regressor_per_series, flag_regressors, impacts):
        # accepts any date in history (or lag beyond) as long as regressors include those dates in index as well
        # BY COMPONENT!!!

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
                x_list.append(multivar_df.reindex(dates))
            elif self.multivariate_feature == "oscillator":
                return NotImplemented
        if self.seasonalities is not None:
            s_list = []
            for seasonality in self.seasonalities:
                s_list.append(create_seasonality_feature(dates, self.t_predict, seasonality))
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
            predicts = []
            for col in self.df.columns:
                c_x = x_array.copy()
                # implement per_series holiday countries flag
                if isinstance(self.holiday_countries, dict) and self.holiday_countries_used:
                    hc = self.holiday_countries.get(col, None)
                    if hc is not None:
                        c_x = pd.concat([
                            c_x,
                            holiday_flag(
                                dates,
                                country=hc,
                                encode_holiday_type=True,
                            ).rename(columns=lambda x: "holiday_" + str(x))
                        ], axis=1)
                # implement regressors per series
                if isinstance(regressor_per_series, dict) and self.regressors_used:
                    hc = regressor_per_series.get(col, None)
                    if hc is not None:
                        c_x = pd.concat([
                            c_x,
                            pd.DataFrame(hc).rename(columns=lambda x: "regrperseries_" + str(x)).reindex(dates)
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
                        lag_idx = np.concatenate([np.repeat([0], lag), np.arange(len(history_df))])[0:len(history_df)]
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
                    pd.Series(np.dot(c_x[self.keep_cols[col]], self.params[col][self.keep_cols_idx[col]]), name=col, index=dates)
                )
                self.predict_x_array[col] = c_x
            return pd.concat(predicts, axis=1)
        else:
            # run model
            if self.linear_model == 'something_else':
                # ADDING RECENCY WEIGHTING AND RIDGE PARAMS
                NotImplemented
            return np.dot(x_array[self.keep_cols], self.params[self.keep_cols_idx])

    def _predict_step(self, dates, trend_component, history_df, future_regressor, flag_regressors, impacts, regressor_per_series):
        # Note this is scaled and doesn't account for impacts
        linear_pred = self._predict_linear(
            dates, history_df=history_df, future_regressor=future_regressor,
            flag_regressors=flag_regressors, impacts=impacts, regressor_per_series=regressor_per_series
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

    def predict(self, forecast_length, include_history=False, future_regressor=None, regressor_per_series=None, flag_regressors=None, future_impacts=None, new_df=None):
        predictStartTime = self.time()

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
                model_name=self.trend_model['Model'],
                model_param_dict=self.trend_model['ModelParameters'],
                model_transform_dict=self.preprocessing_transformation,
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
        if future_regressor is not None:
            full_regr = pd.concat([self.future_regressor_train, self.regressor_transformer.fit_transform(clean_regressor(future_regressor))])
        if flag_regressors is not None and forecast_length is not None:
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
                future_regressor_train=self.future_regressor_train,
                future_regressor_forecast=future_regressor,
                # holiday_country=holiday_country,
                fail_on_forecast_nan=True,
                random_seed=self.random_seed,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            # phi is on future predict step only
            if self.trend_phi is not None and self.trend_phi != 1:
                temp = trend_forecast.forecast.mul(
                    pd.Series([self.phi] * trend_forecast.forecast.shape[0], index=trend_forecast.forecast.index).pow(
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
                    self.trend_residuals, trend_forecast.forecast
                ])
                trend_forecast.lower_forecast = pd.concat([
                    self.trend_residuals, trend_forecast.lower_forecast
                ])
                trend_forecast.upper_forecast = pd.concat([
                    self.trend_residuals, trend_forecast.upper_forecast
                ])
        else:
            trend_forecast = PredictionObject(
                forecast=self.trend_residuals, lower_forecast=self.trend_residuals, upper_forecast=self.trend_residuals
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
        elif not include_history:
            df_forecast.forecast = self.to_origin_space(df_forecast.forecast, trans_method='forecast')
            df_forecast.lower_forecast = self.to_origin_space(df_forecast.lower_forecast, trans_method='forecast')
            df_forecast.upper_forecast = self.to_origin_space(df_forecast.upper_forecast, trans_method='forecast')
        else:
            df_forecast.forecast = pd.concat([
                self.to_origin_space(df_forecast.forecast.head(len(df)), trans_method='original'),
                self.to_origin_space(df_forecast.forecast.tail(forecast_length), trans_method='forecast'),
            ])
            df_forecast.lower_forecast = pd.concat([
                self.to_origin_space(df_forecast.lower_forecast.head(len(df)), trans_method='original'),
                self.to_origin_space(df_forecast.lower_forecast.tail(forecast_length), trans_method='forecast'),
            ])
            df_forecast.upper_forecast = pd.concat([
                self.to_origin_space(df_forecast.upper_forecast.head(len(df)), trans_method='original'),
                self.to_origin_space(df_forecast.upper_forecast.tail(forecast_length), trans_method='forecast'),
            ])

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

    def evaluate(self, fcst, actual):
        # return metrics for comparison, match on datetime index ('ds', 'datetime', 'date')
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
        anomaly_intervention = random.choices([None, 'remove', 'detect_only', 'model'], [0.9, 0.3, 0.1, 0.1])[0]
        if anomaly_intervention is not None:
            anomaly_detector_params = AnomalyRemoval.get_new_params(method=method)
            if anomaly_intervention == "model":
                anomaly_intervention = general_template.sample(1).to_dict("records")[0]  # placeholder, probably
        else:
            anomaly_detector_params = None
        model_str = random.choices(['AverageValueNaive', 'UnivariateMotif'], [0.5, 0.4], k=1)[0]
        trend_model = {'Model': model_str}
        trend_model['ModelParameters'] = ModelMonster(model_str).get_new_params(method=method)

        trend_anomaly_intervention = random.choices([None, 'detect_only'], [0.5, 0.5])[0]
        if trend_anomaly_intervention is not None:
            trend_anomaly_detector_params = AnomalyRemoval.get_new_params(method=method)
        else:
            trend_anomaly_detector_params = None
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
                [0.9, 0.05, 0.05, 0.05],
            )[0],
            "ar_interaction_seasonality": NotImplemented,
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
                transformer_list=scalers, transformer_max_depth=1,
                allow_none=False, no_nan_fill=False  # probably want some more usable defaults first as many random are senseless
            ),
            "regressors_used": random.choices([True, False], [0.5, 0.5])[0],
            "linear_model": 'lstsq',
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


def multivariate_feature(df, categorical_groups):
    # 1 day lag
    # Feature Agglomeration
    # Advancing:Declining (growth/decline over previous day)
    # Above/Below moving average %
    # McClellan Oscillator
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


def create_seasonality_feature(DTindex, t, seasonality):
    # fourier orders
    if isinstance(seasonality, (int, float)):
        return pd.DataFrame(fourier_series(t, seasonality, n=10)).rename(columns=lambda x: f"seasonality_{seasonality}_" + str(x))
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


def window_sum(x, w, axis=0):
    return sliding_window_view(x, w, axis=axis).sum(axis=-1)


def window_sum_nan(x, w, axis=0):
    return np.nansum(sliding_window_view(x, w, axis=axis), axis=-1)


def window_lin_reg(x, y, w):
    '''From https://stackoverflow.com/questions/70296498/efficient-computation-of-moving-linear-regression-with-numpy-numba/70304475#70304475'''
    sx = window_sum(x, w)
    sy = window_sum_nan(y, w)
    sx2 = window_sum(x**2, w)
    sxy = window_sum_nan(x * y, w)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx**2)
    intercept = (sy - slope * sx) / w
    return slope, intercept

def linear_model(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]

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

if False:
    # test holiday countries, regressors, impacts
    from autots import load_daily
    df_holiday = load_daily(long=False)

    params = Cassandra.get_new_params()
    mod = Cassandra(**params)
    mod.fit(df_holiday, categorical_groups=categorical_groups)
    mod.predict(forecast_length=10).forecast

# MULTIPLICATIVE SEASONALITY AND HOLIDAYS

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
