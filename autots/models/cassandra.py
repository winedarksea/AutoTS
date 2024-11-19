# -*- coding: utf-8 -*-
"""Cassandra Model.
Created on Tue Sep 13 19:45:57 2022

@author: Colin
with assistance from @crgillespie22
"""
import json
from operator import itemgetter
from itertools import groupby
import random
import numpy as np
import pandas as pd

# using transformer version of Anomaly/Holiday to use a lower level import than evaluator
from autots.tools.seasonal import (
    create_seasonality_feature,
    seasonal_int,
    datepart_components,
    date_part_methods,
    create_changepoint_features,
)
from autots.tools.fft import FFT
from autots.tools.transform import (
    GeneralTransformer,
    RandomTransform,
    scalers,
    superfast_transformer_dict,
    HolidayTransformer,
    AnomalyRemoval,
    EmptyTransformer,
    StandardScaler,
)
from autots.tools import cpu_count
from autots.models.base import ModelObject, PredictionObject
from autots.templates.general import general_template
from autots.tools.holiday import holiday_flag
from autots.tools.window_functions import window_lin_reg_mean_no_nan, np_2d_arange
from autots.evaluator.auto_model import ModelMonster, model_forecast
from autots.models.model_list import model_list_to_dict

# scipy is technically optional but most likely is present
try:
    from scipy.optimize import minimize
    from scipy.stats import norm
except Exception:

    class norm(object):
        @staticmethod
        def ppf(x):
            return 1.6448536269514722

        # norm.ppf((1 + 0.95) / 2)


class Cassandra(ModelObject):
    """Explainable decomposition-based forecasting with advanced trend modeling and preprocessing.

    Tunc etiam fatis aperit Cassandra futuris
    ora, dei iussu non umquam credita Teucris.
    Nos delubra deum miseri, quibus ultimus esset
    ille dies, festa velamus fronde per urbem.
    -Aeneid 2.246-2.249

    In general, all time series data inputs (df, regressors, impacts) should be wide style data in a pd.DataFrame
        an index that is a pd.DatetimeIndex
        one column per time series, with a uniquely identifiable column name

    Impacts get confusing.
    A past impact of 0.05 would mean an outside, unforecastable force caused/added 5% of the value at this time.
    Accordingly, that 5% will be removed before forecasting, then added back on after.
    Impacts can also be negative values.
    A future impact of 5% would mean an outside force adds 5% above the original forecast.
    Future impacts can be used to model product goals or temporary anomalies which can't or should't be modeled by forecasting and whose relative effect is known
    Compare this with regressors, which are essentially the model estimating the relative impact given the raw size or presence of an outside effect

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
         fit
         predict
         holiday_detector.dates_to_holidays
         create_forecast_index: after .fit, can be used to create index of prediction
         plot_forecast
         plot_components
         plot_trend
         get_new_params
         return_components

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
        .predicted_trend
    """

    def __init__(
        self,
        preprocessing_transformation: dict = None,
        scaling: str = "BaseScaler",  # pulled out from transformation as a scaler is not optional, maybe allow a list
        past_impacts_intervention: str = None,  # 'remove', 'plot_only', 'regressor'
        seasonalities: dict = [
            'common_fourier'
        ],  # interactions added if fourier and order matches
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
        trend_standin: str = None,  # rolling fit, intercept-only, random.normal features, rolling_trend can be memory intensive
        trend_anomaly_detector_params: dict = None,  # difference first, run on slope only, use Window n/2 diff to rule out return to
        # trend_anomaly_intervention: str = None,
        trend_transformation: dict = {},
        trend_model: dict = {
            'Model': 'LastValueNaive',
            'ModelParameters': {},
        },  # have one or two in built, then redirect to any AutoTS model for other choices
        trend_phi: float = None,
        constraint: dict = None,
        x_scaler: bool = False,
        max_colinearity: float = 0.998,
        max_multicolinearity: float = 0.001,
        # not modeling related:
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        random_seed: int = 2022,
        verbose: int = 0,
        n_jobs: int = "auto",
        forecast_length: int = 30,  # currently only used for an single use in the GeneralTransformer
        **kwargs,
    ):
        if preprocessing_transformation is None:
            preprocessing_transformation = {}
        self.preprocessing_transformation = preprocessing_transformation
        self.scaling = scaling
        self.past_impacts_intervention = past_impacts_intervention
        if not seasonalities:
            self.seasonalities = None
        else:
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
        self.x_scaler = x_scaler
        self.max_colinearity = max_colinearity
        self.max_multicolinearity = max_multicolinearity
        # other parameters
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.int_std_dev = norm.ppf(
            0.5 + 0.5 * self.prediction_interval
        )  # 2 to 1 sided interval
        self.random_seed = random_seed
        self.verbose = verbose
        self.n_jobs = n_jobs
        if self.n_jobs == 'auto':
            self.n_jobs = cpu_count(modifier=0.75)
            if verbose > 0:
                print(f"Using {self.n_jobs} cpus for n_jobs.")
        self.forecast_length = forecast_length
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
        self.impacts = None
        if self.trend_anomaly_detector_params is not None:
            self.trend_anomaly_detector = AnomalyRemoval(
                **self.trend_anomaly_detector_params
            )
        else:
            self.trend_anomaly_detector = None

    def base_scaler(self, df):
        self.scaler_mean = np.mean(df, axis=0)
        self.scaler_std = np.std(df, axis=0).replace(0, 1)
        return (df - self.scaler_mean) / self.scaler_std

    def scale_data(self, df):
        if self.preprocessing_transformation is not None:
            df = self.preprocesser.transform(df.copy())
        if self.scaling is not None:
            if self.scaling == "BaseScaler":
                df = self.base_scaler(df)
            else:
                df = self.scaler.transform(df)
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame(df, columns=df.columns, index=df.index)
        else:
            return df

    def to_origin_space(
        self, df, trans_method='forecast', components=False, bounds=False
    ):
        """Take transformed outputs back to original feature space."""
        if self.scaling == "BaseScaler":
            # return self.preprocesser.inverse_transform(df, trans_method=trans_method) * self.scaler_std + self.scaler_mean
            if components:
                return self.preprocesser.inverse_transform(
                    df * self.scaler_std, trans_method=trans_method, bounds=bounds
                )
            else:
                return self.preprocesser.inverse_transform(
                    df * self.scaler_std + self.scaler_mean,
                    trans_method=trans_method,
                    bounds=bounds,
                )
        else:
            # return self.scaler.inverse_transform(self.preprocesser.inverse_transform(df, trans_method=trans_method), trans_method=trans_method)
            return self.preprocesser.inverse_transform(
                self.scaler.inverse_transform(
                    df, trans_method=trans_method, bounds=bounds
                ),
                trans_method=trans_method,
                bounds=bounds,
            )

    def create_t(self, DTindex):
        return (DTindex - self.ds_min) / (self.ds_max - self.ds_min)

    def fit(
        self,
        df,
        future_regressor=None,
        regressor_per_series=None,
        flag_regressors=None,
        categorical_groups=None,
        past_impacts=None,
    ):
        # flag regressors bypass preprocessing
        # ideally allow both pd.DataFrame and np.array inputs (index for array?
        if self.constraint is not None:
            self.df_original = df
        self.df = df.copy()
        self.basic_profile(self.df)
        # standardize groupings (extra/missing dealt with)
        if categorical_groups is None:
            categorical_groups = {}
        self.categorical_groups = {
            col: (
                categorical_groups[col] if col in categorical_groups.keys() else "other"
            )
            for col in df.columns
        }
        self.past_impacts = past_impacts
        # if past impacts given, assume removal unless otherwise specified
        if self.past_impacts_intervention is None and past_impacts is not None:
            self.past_impacts_intervention = "remove"
        elif past_impacts is None:
            self.past_impacts_intervention = None
        if regressor_per_series is not None:
            self.regr_per_series_tr = regressor_per_series

        # what features will require separate models to be fit as X will not be consistent for all
        # What triggers loop:
        # AR Lags
        # Holiday countries as dict (not list)
        # regressor_per_series (only if regressor used)
        # Multivariate Holiday Detector, only if create feature, not removal (plot in this case)
        # Multivariate Anomaly Detector, only if create feature, not removal
        if isinstance(self.anomaly_detector_params, dict):
            self.anomaly_not_uni = (
                self.anomaly_detector_params.get('output', None) != 'univariate'
            )
        self.anomaly_not_uni = False
        self.loop_required = (
            (self.ar_lags is not None)
            or ()
            or (
                isinstance(self.holiday_countries, dict) and self.holiday_countries_used
            )
            or ((regressor_per_series is not None) and self.regressors_used)
            or (isinstance(self.anomaly_intervention, dict) and self.anomaly_not_uni)
            or (
                self.past_impacts_intervention == "regressor"
                and past_impacts is not None
            )
        )
        # check if rolling prediction is required
        self.predict_loop_req = (self.ar_lags is not None) or (
            self.multivariate_feature is not None and self.multivariate_feature != "fft"
        )
        # check if component processing must loop
        # self.component_loop_req = (isinstance(regressor_per_series, dict) and self.regressors_used) or (isinstance(self.holiday_countries, dict) and self.holiday_countries_used)

        # REMOVE NaN but only this so far, for holiday and anomaly detection
        if self.preprocessing_transformation is not None:
            self.preprocesser = GeneralTransformer(
                n_jobs=self.n_jobs,
                holiday_country=self.holiday_countries,
                verbose=self.verbose,
                random_seed=self.random_seed,
                forecast_length=self.forecast_length,
                **{'fillna': self.preprocessing_transformation.get('fillna', "ffill")},
            )
            self.df = self.preprocesser.fit_transform(self.df)

        # remove past impacts to find "organic"
        if self.past_impacts_intervention == "remove":
            self.df = self.df / (1 + past_impacts)  # would MINUS be better?
        # holiday detection first, don't want any anomalies removed yet, and has own preprocessing
        if self.holiday_detector_params is not None:
            self.holiday_detector = HolidayTransformer(**self.holiday_detector_params)
            self.holiday_detector.fit(self.df)
            self.holidays = self.holiday_detector.dates_to_holidays(
                self.df.index, style='series_flag'
            )
            self.holiday_count = np.count_nonzero(self.holidays)
            if self.holiday_detector_params["remove_excess_anomalies"]:
                self.df = self.df[
                    ~(
                        (self.holiday_detector.anomaly_model.anomalies == -1)
                        & (self.holidays != 1)
                    )
                ]
        # find anomalies, and either remove or setup for modeling the anomaly scores
        if self.anomaly_detector_params is not None:
            self.anomaly_detector = AnomalyRemoval(**self.anomaly_detector_params).fit(
                self.df
            )
            # REMOVE HOLIDAYS from anomalies, as they may look like anomalies but are dealt with by holidays
            # this, however, does nothing to preserve country holidays
            if self.holiday_detector_params is not None:
                hol_filt = (self.holidays == 1) & (
                    self.anomaly_detector.anomalies == -1
                )
                self.anomaly_detector.anomalies[hol_filt] = 1
                # assume most are not anomalies so median is not anom
                self.anomaly_detector.scores[hol_filt] = np.median(
                    self.anomaly_detector.scores
                )
            if self.anomaly_intervention == "remove":
                self.df = self.anomaly_detector.transform(self.df)
            elif isinstance(self.anomaly_intervention, dict):
                self.anomaly_detector.fit_anomaly_classifier()
            # detect_only = pass
        # now do standard preprocessing
        if self.preprocessing_transformation is not None:
            self.preprocesser = GeneralTransformer(
                n_jobs=self.n_jobs,
                holiday_country=self.holiday_countries,
                verbose=self.verbose,
                random_seed=self.random_seed,
                forecast_length=self.forecast_length,
                **self.preprocessing_transformation,
            )
            self.df = self.preprocesser.fit_transform(self.df)
        if self.scaling is not None:
            if self.scaling == "BaseScaler":
                self.df = self.base_scaler(self.df)
            else:
                if self.scaling is None:
                    raise ValueError("scaling must not be None. Try 'BaseScaler'")
                self.scaler = GeneralTransformer(
                    n_jobs=self.n_jobs,
                    holiday_country=self.holiday_countries,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    forecast_length=self.forecast_length,
                    **self.scaling,
                )
                self.df = self.scaler.fit_transform(self.df)
        # additional transforms before multivariate feature creation
        if self.multivariate_transformation is not None:
            self.multivariate_transformer = GeneralTransformer(
                n_jobs=self.n_jobs,
                holiday_country=self.holiday_countries,
                verbose=self.verbose,
                random_seed=self.random_seed,
                forecast_length=self.forecast_length,
                **self.multivariate_transformation,
            )
        # needs to come after preprocessing because of 'slice' transformer
        self.ds_min = self.df.index.min()
        self.ds_max = self.df.index.max()
        self.t_train = self.create_t(self.df.index)
        self.history_days = (self.ds_max - self.ds_min).days

        # BEGIN CONSTRUCTION OF X ARRAY
        x_list = []
        if isinstance(self.anomaly_intervention, dict):
            # need to model these for prediction
            x_list.append(
                self.anomaly_detector.scores.rename(
                    columns=lambda x: "anomalyscores_" + str(x)
                )
            )
        if self.multivariate_feature == "fft":
            self.fft = FFT(n_harm=4, detrend='linear')
            self.fft.fit(self.df.bfill().to_numpy())

            x_list.append(
                pd.DataFrame(
                    self.fft.generate_harmonics_dataframe(0), index=self.df.index
                ).rename(columns=lambda x: "mrktfft_" + str(x))
            )
        # all of the following are 1 day past lagged
        elif self.multivariate_feature is not None:
            # includes backfill
            lag_1_indx = np.concatenate([[0], np.arange(len(self.df))])[
                0 : len(self.df)
            ]
            if self.multivariate_transformation is not None:
                trs_df = self.multivariate_transformer.fit_transform(self.df)
                if trs_df.shape != self.df.shape:
                    raise ValueError(
                        "Multivariate Transformer not usable for this role."
                    )
            else:
                trs_df = self.df.copy()
            if self.multivariate_feature == "feature_agglomeration":
                from sklearn.cluster import FeatureAgglomeration

                self.agglom_n_clusters = 5
                self.agglomerator = FeatureAgglomeration(
                    n_clusters=self.agglom_n_clusters
                )
                x_list.append(
                    pd.DataFrame(
                        self.agglomerator.fit_transform(trs_df)[lag_1_indx],
                        index=self.df.index,
                        columns=[
                            "multivar_" + str(x) for x in range(self.agglom_n_clusters)
                        ],
                    )
                )
            elif self.multivariate_feature == "group_average":
                multivar_df = (
                    trs_df.T.groupby(self.categorical_groups)  # axis=1
                    .mean()
                    .iloc[lag_1_indx]
                )
                multivar_df.index = self.df.index
                x_list.append(
                    multivar_df.rename(columns=lambda x: "multivar_" + str(x))
                )
            elif self.multivariate_feature == "oscillator":
                return NotImplemented
                np.count_nonzero((df - df.shift(1)).clip(upper=0))[:-1]
        if self.seasonalities is not None:
            s_list = []
            try:
                for seasonality in self.seasonalities:
                    s_list.append(
                        create_seasonality_feature(
                            self.df.index,
                            self.t_train,
                            seasonality,
                            history_days=self.history_days,
                        )
                    )
                    # INTERACTIONS NOT IMPLEMENTED
                    # ORDER SPECIFICATION NOT IMPLEMENTED
                s_df = pd.concat(s_list, axis=1)
            except Exception as e:
                raise ValueError(f"seasonality {seasonality} creation error") from e
            s_df.index = self.df.index
            x_list.append(s_df)
        # These features are to prevent overfitting and standin for unobserved components here
        if self.randomwalk_n is not None:
            x_list.append(
                pd.DataFrame(
                    np.random.normal(size=(len(self.df), self.randomwalk_n)).cumsum(
                        axis=0
                    ),
                    columns=["randomwalk_" + str(x) for x in range(self.randomwalk_n)],
                    index=self.df.index,
                )
            )
        if self.trend_standin is not None:
            if self.trend_standin == "random_normal":
                num_standin = 4
                x_list.append(
                    pd.DataFrame(
                        np.random.normal(size=(len(self.df), num_standin)),
                        columns=["randnorm_" + str(x) for x in range(num_standin)],
                        index=self.df.index,
                    )
                )
            elif self.trend_standin == "rolling_trend":
                resid, slope, intercept = self.rolling_trend(
                    np.asarray(df), np.array(self.create_t(df.index))
                )
                resid = pd.DataFrame(
                    slope * np.asarray(self.t_train)[..., None] + intercept,
                    index=df.index,
                    columns=df.columns,
                )
                x_list.append(resid.rename(columns=lambda x: "rolling_trend_" + str(x)))
            elif self.trend_standin == "changepoints":
                x_t = create_changepoint_features(
                    self.df.index,
                    changepoint_spacing=60,
                    changepoint_distance_end=120,
                )
                x_list.append(x_t)
            else:
                raise ValueError(
                    f"trend_standin arg `{self.trend_standin}` not recognized"
                )
        if future_regressor is not None and self.regressors_used:
            if self.regressor_transformation is not None:
                self.regressor_transformer = GeneralTransformer(
                    n_jobs=self.n_jobs,
                    holiday_country=self.holiday_countries,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    forecast_length=self.forecast_length,
                    **self.regressor_transformation,
                )
            else:
                self.regressor_transformer = GeneralTransformer(**{'fillna': 'ffill'})
            self.future_regressor_train = self.regressor_transformer.fit_transform(
                clean_regressor(future_regressor)
            ).fillna(0)
            x_list.append(
                self.future_regressor_train.reindex(self.df.index, fill_value=0)
            )
        if flag_regressors is not None and self.regressors_used:
            self.flag_regressor_train = clean_regressor(
                flag_regressors, prefix="regrflags_"
            )
            x_list.append(self.flag_regressor_train.reindex(self.df.index))
        if (
            self.holiday_countries is not None
            and not isinstance(self.holiday_countries, dict)
            and self.holiday_countries_used
        ):
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
            x_list.append(
                self.holiday_detector.dates_to_holidays(self.df.index, style="flag")
                .clip(upper=1)
                .rename(columns=lambda x: "holiday_" + str(x))
            )

        # FINAL FEATURE PROCESSING
        x_array = pd.concat(x_list, axis=1)
        # drop duplicates (holiday flag can create these for multiple countries)
        x_array = x_array.loc[:, ~x_array.columns.duplicated()]
        self.x_array = x_array  # can remove this later, it is for debugging
        if np.any(np.isnan(x_array.astype(float))):  # remove later, for debugging
            nulz = x_array.isnull().sum()
            if self.verbose > 2:
                print(
                    f"the following columns contain nan values: {nulz[nulz > 0].index.tolist()}"
                )
            raise ValueError("nan values in x_array")
        if np.all(self.df == 0):
            raise ValueError("transformed data is all zeroes")
        if np.any((self.df.isnull().all())):
            raise ValueError("transformed df has NaN filled series")
        # remove zero variance (corr is nan)
        corr = np.corrcoef(x_array, rowvar=0)
        nearz = x_array.columns[np.isnan(corr).all(axis=1)]
        self.drop_colz = []
        if len(nearz) > 0:
            if self.verbose > 2:
                print(f"Dropping zero variance feature columns {nearz}")
            self.drop_colz.extend(nearz.tolist())
            # x_array = x_array.drop(columns=nearz)
        # remove colinear features
        # NOTE THESE REMOVALS REMOVE THE FIRST OF PAIR COLUMN FIRST
        corr = np.corrcoef(x_array, rowvar=0)  # second one
        w, vec = np.linalg.eig(np.nan_to_num(corr))
        np.fill_diagonal(corr, 0)
        if self.max_colinearity is not None:
            corel = x_array.columns[
                np.min(corr * np.tri(corr.shape[0]), axis=0) > self.max_colinearity
            ]
            if len(corel) > 0:
                if self.verbose > 2:
                    print(f"Dropping colinear feature columns {corel}")
                # x_array = x_array.drop(columns=corel)
                self.drop_colz.extend(corel.tolist())
        if self.max_multicolinearity is not None:
            colin = x_array.columns[w < self.max_multicolinearity]
            if len(colin) > 0:
                if self.verbose > 2:
                    print(f"Dropping multi-colinear feature columns {colin}")
                # x_array = x_array.drop(columns=colin)
                self.drop_colz.extend(colin.tolist())
        x_array = x_array.drop(columns=self.drop_colz)

        # things we want modeled but want to discard from evaluation (standins)
        remove_patterns = [
            "randnorm_",
            "rolling_trend_",
            "randomwalk_",
        ]  # "intercept" added after, so not included

        # RUN LINEAR MODEL
        # add x features that don't apply to all, and need to be looped
        if self.loop_required:
            self.params = {}
            self.x_scaler_obj = {}
            self.keep_cols = {}
            self.x_array = {}
            self.keep_cols_idx = {}
            self.col_groupings = {}
            trend_residuals = []
            for col in self.df.columns:
                c_x = x_array.copy()
                # implement per_series holiday countries flag
                if (
                    isinstance(self.holiday_countries, dict)
                    and self.holiday_countries_used
                ):
                    hc = self.holiday_countries.get(col, None)
                    if hc is not None:
                        c_hc = holiday_flag(
                            self.df.index,
                            country=hc,
                            encode_holiday_type=True,
                        ).rename(columns=lambda x: "holiday_" + str(x))
                    else:
                        c_hc = pd.DataFrame(
                            0, index=self.df.index, columns=["holiday_0"]
                        )
                    c_x = pd.concat([c_x, c_hc], axis=1)
                # implement regressors per series
                if isinstance(regressor_per_series, dict) and self.regressors_used:
                    hc = regressor_per_series.get(col, None)
                    if hc is not None:
                        c_hc = (
                            pd.DataFrame(hc)
                            .rename(columns=lambda x: "regrperseries_" + str(x))
                            .reindex(self.df.index)
                        )
                    else:
                        c_hc = pd.DataFrame(
                            0, index=self.df.index, columns=["regrperseries_0"]
                        )
                    c_x = pd.concat([c_x, c_hc], axis=1)
                # implement past_impacts as regressor
                if (
                    isinstance(past_impacts, pd.DataFrame)
                    and self.past_impacts_intervention == "regressor"
                ):
                    c_x = pd.concat(
                        [
                            c_x,
                            past_impacts[col]
                            .to_frame()
                            .rename(columns=lambda x: "impacts_" + str(x)),
                        ],
                        axis=1,
                    )
                # add AR features
                if self.ar_lags is not None:
                    for lag in self.ar_lags:
                        lag_idx = np.concatenate(
                            [np.repeat([0], lag), np.arange(len(self.df))]
                        )[0 : len(self.df)]
                        lag_s = self.df[col].iloc[lag_idx].rename(f"lag{lag}_")
                        lag_s.index = self.df.index
                        if self.ar_interaction_seasonality is not None:
                            s_feat = create_seasonality_feature(
                                self.df.index,
                                self.t_train,
                                self.ar_interaction_seasonality,
                                history_days=self.history_days,
                            )
                            s_feat.index = lag_s.index
                            lag_s = s_feat.mul(lag_s, axis=0).rename(
                                columns=lambda x: f"lag{lag}_" + str(x)
                            )
                        c_x = pd.concat([c_x, lag_s], axis=1)
                # NOTE THERE IS NO REMOVING OF COLINEAR FEATURES ADDED HERE
                self.keep_cols[col] = c_x.columns[
                    ~c_x.columns.str.contains("|".join(remove_patterns))
                ]
                self.keep_cols_idx[col] = c_x.columns.get_indexer_for(
                    self.keep_cols[col]
                )
                self.col_groupings[col] = (
                    self.keep_cols[col].str.partition("_").get_level_values(0)
                )
                if self.x_scaler:
                    self.x_scaler_obj[col] = StandardScaler()
                    c_x = self.x_scaler_obj[col].fit_transform(c_x)
                else:
                    self.x_scaler_obj[col] = EmptyTransformer()
                c_x['intercept'] = 1
                self.x_array[col] = c_x
                # ADDING RECENCY WEIGHTING AND RIDGE PARAMS
                self.params[col] = fit_linear_model(
                    c_x, self.df[col].to_frame(), params=self.linear_model
                )
                trend_residuals.append(
                    self.df[col]
                    - pd.Series(
                        np.dot(
                            c_x[self.keep_cols[col]],
                            self.params[col][self.keep_cols_idx[col]],
                        ).flatten(),
                        name=col,
                        index=self.df.index,
                    )
                )
            trend_residuals = pd.concat(trend_residuals, axis=1)
        else:
            # RUN LINEAR MODEL, WHEN NO LOOPED FEATURES
            self.keep_cols = x_array.columns[
                ~x_array.columns.str.contains("|".join(remove_patterns))
            ]
            self.keep_cols_idx = x_array.columns.get_indexer_for(self.keep_cols)
            self.col_groupings = self.keep_cols.str.partition("_").get_level_values(0)
            if self.x_scaler:
                self.x_scaler_obj = StandardScaler()
                x = self.x_scaler_obj.fit_transform(x_array)
            else:
                self.x_scaler_obj = EmptyTransformer()
                x = x_array
            x_array['intercept'] = 1
            # run model
            self.params = fit_linear_model(x, self.df, params=self.linear_model)
            trend_residuals = self.df - np.dot(
                x[self.keep_cols], self.params[self.keep_cols_idx]
            )
            self.x_array = x

        # option to run trend model on full residuals or on rolling trend
        if (
            self.trend_anomaly_detector_params is not None
            or self.trend_window is not None
        ):
            # rolling trend
            trend_posterior, slope, intercept = self.rolling_trend(
                trend_residuals, np.array(self.t_train)
            )
            trend_posterior = pd.DataFrame(
                trend_posterior, index=self.df.index, columns=self.df.columns
            )
            res_dif = trend_residuals - trend_posterior
            res_upper = res_dif[res_dif >= 0]
            res_lower = res_dif[res_dif <= 0]
            self.residual_uncertainty_upper = res_upper.mean()
            self.residual_uncertainty_lower = res_lower.mean().abs()
            self.residual_uncertainty_upper_std = res_upper.std()
            self.residual_uncertainty_lower_std = res_lower.std()
        else:
            self.residual_uncertainty_upper = pd.Series(0, index=self.df.columns)
            self.residual_uncertainty_lower = pd.Series(0, index=self.df.columns)
            self.residual_uncertainty_upper_std = pd.Series(0, index=self.df.columns)
            self.residual_uncertainty_lower_std = pd.Series(0, index=self.df.columns)
        if self.trend_window is not None:
            self.trend_train = trend_posterior
        else:
            self.trend_train = pd.DataFrame(
                trend_residuals, index=self.df.index, columns=self.df.columns
            )

        (
            self.zero_crossings,
            self.changepoints,
            self.slope_sign,
            self.accel,
        ) = self.analyze_trend(slope, index=self.trend_train.index)
        if False:
            self.trend_anomaly_detector = AnomalyRemoval(
                **self.trend_anomaly_detector_params
            )
            # DIFF the length of W (or w-1?)
            shft_idx = np.concatenate([[0], np.arange(len(slope))])[0 : len(slope)]
            slope_diff = slope - slope[shft_idx]
            shft_idx = np.concatenate(
                [np.repeat([0], self.trend_window), np.arange(len(slope))]
            )[0 : len(slope)]
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
        zero_crossings = np.vstack(
            (np.diff(slope_sign, axis=0), false_row)
        )  # Use on earliest, fill end
        # where the rate of change of the rate of change crosses 0 to -1 or reverse
        accel = np.diff(slope, axis=0)
        changepoints = np.vstack(
            (false_row, np.diff(np.signbit(accel), axis=0), false_row)
        )
        if self.trend_anomaly_detector_params is not None:
            self.trend_anomaly_detector.fit(
                pd.DataFrame(
                    np.where(changepoints, slope, 0),
                    index=index,
                    columns=self.column_names,  # Need different source for index
                )
            )
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
            trend_residuals.shape[1],
            axis=1,
        )
        wind = 30 if self.trend_window is None else self.trend_window
        # the uneven fraction of the window goes at the ened
        # and minus one is because there will always be at least one real point
        w_1 = wind - 1
        steps_ahd = int(w_1 / 2)
        y0 = np.repeat(np.array(trend_residuals[0:1]), steps_ahd, axis=0)
        # d0 = -1 * dates_2d[1 : y0.shape[0] + 1][::-1]
        start_pt = dates_2d[0, 0]
        step = dates_2d[1, 0] - start_pt
        extra_step = y0.shape[0] + 1
        # there's some weird float thing that can happen here I still don't understand
        # when it produces one more step than expected
        d0 = np_2d_arange(
            start_pt,
            stop=start_pt - (extra_step * step),
            step=-step,
            num_columns=dates_2d.shape[1],
        )[1:extra_step][::-1]
        shape2 = (w_1 - steps_ahd, y0.shape[1])
        # these combine a fake first half and fake last half window with real data in between
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
        # switch back to window_lin_reg_mean for nan tolerance, but it is much more memory intensive
        slope, intercept = window_lin_reg_mean_no_nan(d, y2, wind)
        trend_posterior = slope * t[..., None] + intercept
        return trend_posterior, slope, intercept

    def _predict_linear(
        self,
        dates,
        history_df,
        future_regressor,
        regressor_per_series,
        flag_regressors,
        impacts,
        return_components=False,
    ):
        # accepts any date in history (or lag beyond) as long as regressors include those dates in index as well

        self.t_predict = self.create_t(dates)
        x_list = []
        if isinstance(self.anomaly_intervention, dict):
            # forecast anomaly scores as time series
            len_inter = len(dates.intersection(self.anomaly_detector.scores.index))
            if len_inter < len(dates):
                amodel_params = self.anomaly_intervention['ModelParameters']
                # no regressors passed here
                if "regression_type" in amodel_params:
                    amodel_params = json.loads(amodel_params)
                    amodel_params['regression_type'] = None
                new_scores = model_forecast(
                    model_name=self.anomaly_intervention['Model'],
                    model_param_dict=amodel_params,
                    model_transform_dict=self.anomaly_intervention[
                        'TransformationParameters'
                    ],
                    df_train=self.anomaly_detector.scores,
                    forecast_length=len(dates) - len_inter,
                    frequency=self.frequency,
                    prediction_interval=self.prediction_interval,
                    fail_on_forecast_nan=False,
                    random_seed=self.random_seed,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                ).forecast
            # need to model these for prediction
            x_list.append(
                pd.concat([self.anomaly_detector.scores, new_scores], axis=0)
                .reindex(dates)
                .rename(columns=lambda x: "anomalyscores_" + str(x))
            )
        if self.multivariate_feature == 'fft':
            # need to translate the 'dates' into integer time steps ahead
            full_dates = self.df.index.union(dates)
            full_dates = pd.date_range(
                full_dates.min(), full_dates.max(), freq=self.frequency
            )
            req_len = len(full_dates) - self.df.index.shape[0]
            req_len = 0 if req_len < 0 else req_len
            x_list.append(
                pd.DataFrame(
                    self.fft.generate_harmonics_dataframe(req_len), index=full_dates
                )
                .rename(columns=lambda x: "mrktfft_" + str(x))
                .reindex(dates)
            )
        # all of the following are 1 day past lagged
        elif self.multivariate_feature is not None:
            # includes backfill
            full_idx = history_df.index.union(
                self.create_forecast_index(
                    forecast_length=1, last_date=history_df.index[-1]
                )
            )
            lag_1_indx = np.concatenate([[0], np.arange(len(history_df))])
            if self.multivariate_transformation is not None:
                trs_df = self.multivariate_transformer.transform(history_df)
            else:
                trs_df = history_df.copy()
            if self.multivariate_feature == "feature_agglomeration":
                x_list.append(
                    pd.DataFrame(
                        self.agglomerator.transform(trs_df)[lag_1_indx],
                        index=full_idx,
                        columns=[
                            "multivar_" + str(x) for x in range(self.agglom_n_clusters)
                        ],
                    ).reindex(dates)
                )
            elif self.multivariate_feature == "group_average":
                multivar_df = (
                    trs_df.T.groupby(self.categorical_groups)  # axis=1
                    .mean()
                    .iloc[lag_1_indx]
                )
                multivar_df.index = full_idx
                x_list.append(
                    multivar_df.reindex(dates).rename(
                        columns=lambda x: "multivar_" + str(x)
                    )
                )
            elif self.multivariate_feature == "oscillator":
                return NotImplemented
        if self.seasonalities is not None:
            s_list = []
            for seasonality in self.seasonalities:
                s_list.append(
                    create_seasonality_feature(
                        dates,
                        self.t_predict,
                        seasonality,
                        history_days=self.history_days,
                    )
                )
                # INTERACTIONS NOT IMPLEMENTED
                # ORDER SPECIFICATION NOT IMPLEMENTED
            s_df = pd.concat(s_list, axis=1)
            s_df.index = dates
            x_list.append(s_df)
        if future_regressor is not None and self.regressors_used:
            x_list.append(future_regressor.reindex(dates))
        if flag_regressors is not None and self.regressors_used:
            x_list.append(
                flag_regressors.reindex(dates)
            )  # doesn't check for missing data
        if (
            self.holiday_countries is not None
            and not isinstance(self.holiday_countries, dict)
            and self.holiday_countries_used
        ):
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
            x_list.append(
                self.holiday_detector.dates_to_holidays(dates, style="flag")
                .clip(upper=1)
                .rename(columns=lambda x: "holiday_" + str(x))
            )

        # FINAL FEATURE PROCESSING
        x_array = pd.concat(x_list, axis=1)
        # drop duplicates (holiday flag can create these for multiple countries)
        x_array = x_array.loc[:, ~x_array.columns.duplicated()]
        x_array = x_array.drop(columns=self.drop_colz, errors="ignore")
        self.predict_x_array = x_array  # can remove this later, it is for debugging
        if np.any(np.isnan(x_array.astype(float))):  # remove later, for debugging
            nulz = x_array.isnull().sum()
            if self.verbose > 2:
                print(
                    f"the following columns contain nan values: {nulz[nulz > 0].index.tolist()}"
                )
            raise ValueError(
                f"nan values in predict_x_array in columns {nulz[nulz > 0].index.tolist()[0:5]}"
            )

        # RUN LINEAR MODEL
        # add x features that don't apply to all, and need to be looped
        if self.loop_required:
            self.predict_x_array = {}
            self.components = []
            predicts = []
            for col in self.df.columns:
                c_x = x_array.copy()
                # implement per_series holiday countries flag
                if (
                    isinstance(self.holiday_countries, dict)
                    and self.holiday_countries_used
                ):
                    hc = self.holiday_countries.get(col, None)
                    if hc is not None:
                        c_hc = holiday_flag(
                            dates,
                            country=hc,
                            encode_holiday_type=True,
                        ).rename(columns=lambda x: "holiday_" + str(x))
                    else:
                        c_hc = pd.DataFrame(0, index=dates, columns=["holiday_0"])
                    c_x = pd.concat([c_x, c_hc], axis=1)
                # implement regressors per series
                if isinstance(regressor_per_series, dict) and self.regressors_used:
                    hc = regressor_per_series.get(col, None)
                    if hc is not None:
                        c_hc = (
                            pd.DataFrame(hc)
                            .rename(columns=lambda x: "regrperseries_" + str(x))
                            .reindex(dates)
                        )
                    else:
                        c_hc = pd.DataFrame(0, index=dates, columns=["regrperseries_0"])
                    c_x = pd.concat([c_x, c_hc], axis=1)
                # implement past_impacts as regressor
                if (
                    isinstance(impacts, pd.DataFrame)
                    and self.past_impacts_intervention == "regressor"
                ):
                    c_x = pd.concat(
                        [
                            c_x,
                            impacts.reindex(dates)[col]
                            .to_frame()
                            .fillna(0)
                            .rename(columns=lambda x: "impacts_" + str(x)),
                        ],
                        axis=1,
                    )
                # add AR features
                if self.ar_lags is not None:
                    # somewhat inefficient to create full df of dates, but simplest this way for 'any date'
                    for lag in self.ar_lags:
                        full_idx = history_df.index.union(
                            self.create_forecast_index(
                                forecast_length=lag, last_date=history_df.index[-1]
                            )
                        )
                        lag_idx = np.concatenate(
                            [np.repeat([0], lag), np.arange(len(history_df))]
                        )  # [0:len(history_df)]
                        lag_s = history_df[col].iloc[lag_idx].rename(f"lag{lag}_")
                        lag_s.index = full_idx
                        lag_s = lag_s.reindex(dates)
                        if self.ar_interaction_seasonality is not None:
                            s_feat = create_seasonality_feature(
                                dates,
                                self.t_predict,
                                self.ar_interaction_seasonality,
                                history_days=self.history_days,
                            )
                            s_feat.index = lag_s.index
                            lag_s = s_feat.mul(lag_s, axis=0).rename(
                                columns=lambda x: f"lag{lag}_" + str(x)
                            )
                        c_x = pd.concat([c_x, lag_s], axis=1)

                # ADDING RECENCY WEIGHTING AND RIDGE PARAMS
                if np.any(np.isnan(c_x.astype(float))):  # remove later, for debugging
                    raise ValueError(
                        f"nan values in predict c_x_array. Rows with NaN: {c_x.isna().any(axis=1).sum()}. Most nan columns: {c_x.isna().sum().sort_values(ascending=False).head(5)}"
                    )
                predicts.append(
                    pd.Series(
                        np.dot(
                            self.x_scaler_obj[col].transform(c_x)[self.keep_cols[col]],
                            self.params[col][self.keep_cols_idx[col]],
                        ).flatten(),
                        name=col,
                        index=dates,
                    )
                )
                self.predict_x_array[col] = c_x
                if return_components:
                    indices = [
                        tuple(group)[-1][0]
                        for key, group in groupby(
                            enumerate(self.col_groupings[col]), key=itemgetter(1)
                        )
                    ][:-1]
                    new_indx = [0] + [x + 1 for x in indices]
                    temp = (
                        self.x_scaler_obj[col].transform(c_x)[self.keep_cols[col]]
                        * self.params[col][self.keep_cols_idx[col]].flatten()
                    )
                    self.components.append(
                        np.add.reduceat(np.asarray(temp), sorted(new_indx), axis=1)
                    )
            if return_components:
                self.components = np.moveaxis(np.array(self.components), 0, 2)
                self.last_used = col
                self.component_indx = (
                    new_indx  # hopefully no mismatching for diff series
                )
            return pd.concat(predicts, axis=1)
        else:
            # run model
            if self.x_scaler:
                x = self.x_scaler_obj.transform(x_array)
                x = x[self.keep_cols]
            else:
                x = x_array[self.keep_cols]
            res = np.dot(x, self.params[self.keep_cols_idx])
            if return_components:
                arr = x.to_numpy()
                temp = (
                    np.moveaxis(
                        np.broadcast_to(
                            arr, [self.params.shape[1], arr.shape[0], arr.shape[1]]
                        ),
                        0,
                        2,
                    )
                    * self.params[self.keep_cols_idx]
                )
                indices = [
                    tuple(group)[-1][0]
                    for key, group in groupby(
                        enumerate(self.col_groupings), key=itemgetter(1)
                    )
                ][:-1]
                new_indx = [0] + [x + 1 for x in indices]
                self.component_indx = new_indx
                self.components = np.add.reduceat(
                    temp, sorted(new_indx), axis=1
                )  # (dates, comps, series)
                # np.allclose(np.add.reduceat(temp, [0], axis=1)[:, 0, :], np.dot(mod.predict_x_array[mod.keep_cols], mod.params[mod.keep_cols_idx]))
            return res

    def process_components(self, to_origin_space=True):
        """Scale and standardize component outputs."""
        if self.components is None:
            raise ValueError("Model has not yet had a prediction generated.")

        t_indx = (
            next(iter(self.predict_x_array.values())).index
            if isinstance(self.predict_x_array, dict)
            else self.predict_x_array.index
        )
        comp_list = []
        col_group = (
            self.col_groupings[self.last_used]
            if isinstance(self.col_groupings, dict)
            else self.col_groupings
        )
        # unfortunately, no choice but to loop that I see to apply inverse trans
        for comp in range(self.components.shape[1]):
            if to_origin_space:
                # will have issues on inverse with some transformers
                comp_df = self.to_origin_space(
                    pd.DataFrame(
                        self.components[:, comp, :],
                        index=t_indx,
                        columns=self.df.columns,
                    ),
                    components=True,
                    bounds=True,
                )  # bounds=True to align better
            else:
                comp_df = pd.DataFrame(
                    self.components[:, comp, :],
                    index=t_indx,
                    columns=self.column_names,
                )
            # column name needs to be unlikely to occur in an actual dataset
            comp_df['csmod_component'] = col_group[self.component_indx[comp]]
            comp_list.append(comp_df)
        return pd.pivot(pd.concat(comp_list, axis=0), columns='csmod_component')

    def _predict_step(
        self,
        dates,
        trend_component,
        history_df,
        future_regressor,
        flag_regressors,
        impacts,
        regressor_per_series,
    ):
        # Note this is scaled and doesn't account for impacts
        linear_pred = self._predict_linear(
            dates,
            history_df=history_df,
            future_regressor=future_regressor,
            flag_regressors=flag_regressors,
            impacts=impacts,
            regressor_per_series=regressor_per_series,
            return_components=True,
        )
        # ADD PREPROCESSING BEFORE TREND (FIT X, REVERSE on PREDICT, THEN TREND)
        zeros_df = pd.DataFrame(
            0.0,
            index=trend_component.forecast.index,
            columns=trend_component.forecast.columns,
        )
        upper_adjust = (zeros_df + self.residual_uncertainty_upper) + (
            self.residual_uncertainty_upper_std * self.int_std_dev
        )
        lower_adjust = (zeros_df + self.residual_uncertainty_lower) + (
            self.residual_uncertainty_lower_std * self.int_std_dev
        )
        # add a gradual increase to full uncertainty
        if linear_pred.shape[0] > 4:
            first_adjust = zeros_df + 1
            first_adjust.iloc[0] = 0.5
            first_adjust.iloc[1] = 0.7
            upper_adjust = upper_adjust * first_adjust
            lower_adjust = lower_adjust * first_adjust
        upper = (
            trend_component.upper_forecast.reindex(dates) + linear_pred + upper_adjust
        )
        lower = (
            trend_component.lower_forecast.reindex(dates) + linear_pred - lower_adjust
        )

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

    def fit_data(
        self,
        df,
        forecast_length=None,
        future_regressor=None,
        regressor_per_series=None,
        flag_regressors=None,
        future_impacts=None,
        regressor_forecast_model=None,
        regressor_forecast_model_params=None,
        regressor_forecast_transformations=None,
        include_history=False,
        past_impacts=None,
    ):
        self.df = self.preprocesser.transform(self.df)
        if self.past_impacts_intervention == "remove":
            try:
                self.df = self.df / (1 + past_impacts)
            except TypeError:
                raise ValueError(
                    "if using past impact with df updates, must pass past_impacts to .fit_data or .predict"
                )
        self.df = self.scale_data(df)
        (
            self.regr_ps_fore,
            self.use_impacts,
            self.full_regr,
            self.all_flags,
        ) = self._process_regressors(
            df=self.df,
            forecast_length=forecast_length,
            future_regressor=future_regressor,
            regressor_per_series=regressor_per_series,
            flag_regressors=flag_regressors,
            future_impacts=future_impacts,
            regressor_forecast_model=regressor_forecast_model,
            regressor_forecast_model_params=regressor_forecast_model_params,
            regressor_forecast_transformations=regressor_forecast_transformations,
            include_history=include_history,
        )
        self.trend_train = self.df - self._predict_linear(
            dates=df.index,
            history_df=self.df,
            future_regressor=self.full_regr,
            flag_regressors=self.all_flags,
            impacts=self.use_impacts,
            regressor_per_series=self.regr_ps_fore,
        )
        if self.trend_window is not None:
            self.trend_train, slope, intercept = self.rolling_trend(
                self.trend_train, np.array(self.create_t(df.index))
            )
            self.trend_train = pd.DataFrame(
                self.trend_train, index=df.index, columns=df.columns
            )

    def _process_regressors(
        self,
        df=None,
        forecast_length=None,
        future_regressor=None,
        regressor_per_series=None,
        flag_regressors=None,
        future_impacts=None,
        regressor_forecast_model=None,
        regressor_forecast_model_params=None,
        regressor_forecast_transformations=None,
        include_history=True,
    ):
        # if future regressors are None (& USED), but were provided for history, instead use forecasts of these features (warn)
        full_regr = None
        if (
            future_regressor is None
            and self.future_regressor_train is not None
            and forecast_length is not None
        ):
            if self.verbose > 0:
                print(
                    "future_regressor not provided to Cassandra, using forecasts of historical"
                )
            future_regressor = model_forecast(
                model_name=(
                    self.trend_model['Model']
                    if regressor_forecast_model is None
                    else regressor_forecast_model
                ),
                model_param_dict=(
                    self.trend_model['ModelParameters']
                    if regressor_forecast_model_params is None
                    else regressor_forecast_model_params
                ),
                model_transform_dict=(
                    self.preprocessing_transformation
                    if regressor_forecast_transformations is None
                    else regressor_forecast_transformations
                ),
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
        if forecast_length is None:
            expected_fore_len = len(df)
        else:
            expected_fore_len = (
                forecast_length + len(df) if include_history else forecast_length
            )
        if future_regressor is not None and self.regressors_used:
            if len(future_regressor) == expected_fore_len:
                full_regr = self.regressor_transformer.transform(
                    clean_regressor(future_regressor)
                )
            else:
                full_regr = pd.concat(
                    [
                        self.future_regressor_train,
                        self.regressor_transformer.transform(
                            clean_regressor(
                                future_regressor.loc[
                                    future_regressor.index.difference(
                                        self.future_regressor_train.index
                                    )
                                ]
                            )
                        ),
                    ]
                )
        if (
            flag_regressors is not None
            and forecast_length is not None
            and self.regressors_used
        ):
            if len(flag_regressors) == expected_fore_len:
                all_flags = clean_regressor(flag_regressors, prefix="regrflags_")
            else:
                all_flags = pd.concat(
                    [
                        self.flag_regressor_train,
                        clean_regressor(
                            flag_regressors.loc[
                                flag_regressors.index.difference(
                                    self.flag_regressor_train.index
                                )
                            ],
                            prefix="regrflags_",
                        ),
                    ]
                )
        else:
            if (
                self.flag_regressor_train is not None
                and forecast_length is not None
                and self.regressors_used
            ):
                raise ValueError("flag_regressors supplied in training but not predict")
            all_flags = self.flag_regressor_train
        if future_impacts is not None and forecast_length is not None:
            if len(future_impacts) == expected_fore_len:
                impacts = future_impacts
            else:
                impacts = pd.concat([self.past_impacts, future_impacts])
        else:
            impacts = self.past_impacts
        # I don't think there is a more efficient way to combine these dicts of dataframes
        if regressor_per_series is not None and self.regressors_used:
            if not isinstance(regressor_per_series, dict):
                raise ValueError("regressor_per_series must be dict")
            regr_ps_fore = {}
            for key, value in self.regr_per_series_tr.items():
                if len(regressor_per_series[key]) == expected_fore_len:
                    regr_ps_fore[key] = regressor_per_series[key]
                else:
                    regr_ps_fore[key] = pd.concat(
                        [self.regr_per_series_tr[key], regressor_per_series[key]]
                    )
                    regr_ps_fore[key] = regr_ps_fore[key].iloc[
                        ~regr_ps_fore[key].index.duplicated()
                    ]
                self.regr_ps = regr_ps_fore
        else:
            regr_ps_fore = self.regr_per_series_tr
        return regr_ps_fore, impacts, full_regr, all_flags

    def predict(
        self,
        forecast_length=None,
        include_history=False,
        future_regressor=None,
        regressor_per_series=None,
        flag_regressors=None,
        future_impacts=None,
        new_df=None,
        regressor_forecast_model=None,
        regressor_forecast_model_params=None,
        regressor_forecast_transformations=None,
        include_organic=False,
        df=None,  # to be compatiable with others, identical to new_df
        past_impacts=None,  # only if new_df provided
    ):
        """Generate a forecast.

        future_regressor and regressor_per_series should only include new future values, history is already stored
        they should match on forecast_length and index of forecasts

        Args:
            forecast_length (int): steps ahead to predict, or None
            include_history (bool): include past predictions if True
            all the same regressor args as .fit, but future forecast versions here
            future_impacts (pd.DataFrame): like past impacts but for the forecast ahead
            new_df (pd.DataFrame): or df, equivalent to fit_data update
        """
        self.forecast_length = forecast_length
        predictStartTime = self.time()
        if self.trend_train is None:
            raise ValueError("Cassandra must first be .fit() successfully.")

        # scale new_df if given
        if df is not None:
            new_df = df
        if new_df is not None:
            self.fit_data(
                df=new_df,
                forecast_length=forecast_length,
                future_regressor=future_regressor,
                regressor_per_series=regressor_per_series,
                flag_regressors=flag_regressors,
                future_impacts=future_impacts,
                regressor_forecast_model=regressor_forecast_model,
                regressor_forecast_model_params=regressor_forecast_model_params,
                regressor_forecast_transformations=regressor_forecast_transformations,
                include_history=include_history,
                past_impacts=past_impacts,
            )
        else:
            (
                self.regr_ps_fore,
                self.use_impacts,
                self.full_regr,
                self.all_flags,
            ) = self._process_regressors(
                df=self.df,
                forecast_length=forecast_length,
                future_regressor=future_regressor,
                regressor_per_series=regressor_per_series,
                flag_regressors=flag_regressors,
                future_impacts=future_impacts,
                regressor_forecast_model=regressor_forecast_model,
                regressor_forecast_model_params=regressor_forecast_model_params,
                regressor_forecast_transformations=regressor_forecast_transformations,
                include_history=include_history,
            )
        df = self.df.copy()

        # generate trend
        # MAY WANT TO PASS future_regressor HERE
        resid = None
        if forecast_length is not None:
            # create new rolling residual if new data provided
            df_train = self.trend_train
            # combine regressor types depending on what is given
            if (
                self.future_regressor_train is None
                and self.flag_regressor_train is not None
                and self.regressors_used
            ):
                comp_regr_train = self.flag_regressor_train.reindex(
                    index=df_train.index
                )
                comp_regr = self.all_flags.tail(forecast_length)
            elif (
                self.future_regressor_train is not None
                and self.flag_regressor_train is None
                and self.regressors_used
            ):
                comp_regr_train = self.future_regressor_train.reindex(
                    index=df_train.index
                )
                comp_regr = self.full_regr.tail(forecast_length)
            elif (
                self.future_regressor_train is not None
                and self.flag_regressor_train is not None
                and self.regressors_used
            ):
                comp_regr_train = pd.concat(
                    [self.future_regressor_train, self.flag_regressor_train], axis=1
                ).reindex(index=df_train.index)
                comp_regr = pd.concat([self.full_regr, self.all_flags], axis=1).tail(
                    forecast_length
                )
            else:
                comp_regr_train = None
                comp_regr = None

            trend_forecast = model_forecast(
                model_name=self.trend_model['Model'],
                model_param_dict=self.trend_model['ModelParameters'],
                model_transform_dict=self.trend_transformation,
                df_train=df_train,
                forecast_length=forecast_length,
                frequency=self.frequency,
                prediction_interval=self.prediction_interval,
                future_regressor_train=comp_regr_train,
                future_regressor_forecast=comp_regr,
                fail_on_forecast_nan=True,
                random_seed=self.random_seed,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            # phi is on future predict step only
            if (
                self.trend_phi is not None
                and self.trend_phi != 1
                and trend_forecast.forecast.shape[0] > 2
            ):
                req_len = trend_forecast.forecast.shape[0] - 1
                phi_series = pd.Series(
                    [self.trend_phi] * req_len,
                    index=trend_forecast.forecast.index[1:],
                ).pow(range(req_len))

                # adjust all by same margin
                trend_forecast.forecast = pd.concat(
                    [
                        trend_forecast.forecast.iloc[0:1],
                        trend_forecast.forecast.diff().iloc[1:].mul(phi_series, axis=0),
                    ]
                ).cumsum()
                trend_forecast.upper_forecast = pd.concat(
                    [
                        trend_forecast.upper_forecast.iloc[0:1],
                        trend_forecast.upper_forecast.diff()
                        .iloc[1:]
                        .mul(phi_series, axis=0),
                    ]
                ).cumsum()
                trend_forecast.lower_forecast = pd.concat(
                    [
                        trend_forecast.lower_forecast.iloc[0:1],
                        trend_forecast.lower_forecast.diff()
                        .iloc[1:]
                        .mul(phi_series, axis=0),
                    ]
                ).cumsum()
            if include_history:
                trend_forecast.forecast = pd.concat(
                    [
                        self.trend_train if resid is None else resid,
                        trend_forecast.forecast,
                    ]
                )
                trend_forecast.lower_forecast = pd.concat(
                    [
                        self.trend_train if resid is None else resid,
                        trend_forecast.lower_forecast,
                    ]
                )
                trend_forecast.upper_forecast = pd.concat(
                    [
                        self.trend_train if resid is None else resid,
                        trend_forecast.upper_forecast,
                    ]
                )
        else:
            trend_forecast = PredictionObject(
                forecast=self.trend_train,
                lower_forecast=self.trend_train,
                upper_forecast=self.trend_train,
            )

        # ar_lags, multivariate features require 1 step loop
        if forecast_length is None:
            df_forecast = self._predict_step(
                dates=df.index,
                trend_component=trend_forecast,
                history_df=df,
                future_regressor=self.full_regr,
                flag_regressors=self.all_flags,
                impacts=self.use_impacts,
                regressor_per_series=self.regr_ps_fore,
            )
        elif self.predict_loop_req:
            for step in range(forecast_length):
                forecast_index = df.index.union(
                    self.create_forecast_index(1, last_date=df.index[-1])
                )
                df_forecast = self._predict_step(
                    dates=forecast_index,
                    trend_component=trend_forecast,
                    history_df=df,
                    future_regressor=self.full_regr,
                    flag_regressors=self.all_flags,
                    impacts=self.use_impacts,
                    regressor_per_series=self.regr_ps_fore,
                )
                df = pd.concat([df, df_forecast.forecast.iloc[-1:]])
            if not include_history:
                df_forecast.forecast = df_forecast.forecast.tail(forecast_length)
                df_forecast.lower_forecast = df_forecast.lower_forecast.tail(
                    forecast_length
                )
                df_forecast.upper_forecast = df_forecast.upper_forecast.tail(
                    forecast_length
                )
        else:
            forecast_index = self.create_forecast_index(
                forecast_length, last_date=df.index[-1]
            )
            if include_history:
                forecast_index = df.index.union(forecast_index)
            df_forecast = self._predict_step(
                dates=forecast_index,
                trend_component=trend_forecast,
                history_df=df,
                future_regressor=self.full_regr,
                flag_regressors=self.all_flags,
                impacts=self.use_impacts,
                regressor_per_series=self.regr_ps_fore,
            )

        # save future index before include_history is added
        if future_impacts is not None and forecast_length is not None:
            future_impacts = future_impacts.reindex(
                columns=df_forecast.forecast.columns,
                index=forecast_index,
                fill_value=0,
            )
        # undo preprocessing and scaling
        # account for some transformers requiring different methods on original data and forecast
        self.predicted_trend = trend_forecast.forecast.copy()
        if forecast_length is None:
            df_forecast.forecast = self.to_origin_space(
                df_forecast.forecast, trans_method='original'
            )
            df_forecast.lower_forecast = self.to_origin_space(
                df_forecast.lower_forecast, trans_method='original'
            )
            df_forecast.upper_forecast = self.to_origin_space(
                df_forecast.upper_forecast, trans_method='original'
            )
        elif not include_history:
            df_forecast.forecast = self.to_origin_space(
                df_forecast.forecast, trans_method='forecast'
            )
            df_forecast.lower_forecast = self.to_origin_space(
                df_forecast.lower_forecast, trans_method='forecast'
            )
            df_forecast.upper_forecast = self.to_origin_space(
                df_forecast.upper_forecast, trans_method='forecast'
            )
        else:
            hdn = len(df_forecast.forecast) - forecast_length
            df_forecast.forecast = pd.concat(
                [
                    self.to_origin_space(
                        df_forecast.forecast.head(hdn), trans_method='original'
                    ),
                    self.to_origin_space(
                        df_forecast.forecast.tail(forecast_length),
                        trans_method='forecast',
                    ),
                ]
            )
            df_forecast.lower_forecast = pd.concat(
                [
                    self.to_origin_space(
                        df_forecast.lower_forecast.head(hdn),
                        trans_method='original',
                        bounds=True,
                    ),
                    self.to_origin_space(
                        df_forecast.lower_forecast.tail(forecast_length),
                        trans_method='forecast',
                        bounds=True,
                    ),
                ]
            )
            df_forecast.upper_forecast = pd.concat(
                [
                    self.to_origin_space(
                        df_forecast.upper_forecast.head(hdn),
                        trans_method='original',
                        bounds=True,
                    ),
                    self.to_origin_space(
                        df_forecast.upper_forecast.tail(forecast_length),
                        trans_method='forecast',
                        bounds=True,
                    ),
                ]
            )

        # don't forget to add in past_impacts (use future impacts again?) AFTER unscaling
        if self.past_impacts_intervention != "regressor":
            if future_impacts is not None and self.past_impacts is None:
                past_impacts = pd.DataFrame(
                    0, index=self.df.index, columns=self.df.columns
                )
            else:
                past_impacts = self.past_impacts
            # roll forward tail of past impacts, assuming it continues
            if self.past_impacts is not None and forecast_length is not None:
                future_impts = pd.DataFrame(
                    np.repeat(
                        self.past_impacts.iloc[-1:].to_numpy(), forecast_length, axis=0
                    ),
                    index=df_forecast.forecast.index[-forecast_length:],
                    columns=self.past_impacts.columns,
                )
                if future_impacts is not None:
                    future_impts = ((1 + future_impts) * (1 + future_impacts)) - 1
            else:
                future_impts = pd.DataFrame()
            if self.past_impacts is not None or future_impacts is not None:
                impts = 1 + (
                    pd.concat(
                        [
                            past_impacts,
                            future_impts[~future_impts.index.isin(past_impacts.index)],
                        ],
                        axis=0,
                    ).reindex(
                        index=df_forecast.forecast.index,
                        columns=df_forecast.forecast.columns,
                        fill_value=0,
                    )
                )  # minus or plus
                self.impacts = impts
                if include_organic:
                    df_forecast.organic_forecast = df_forecast.forecast.copy()
                df_forecast.forecast = df_forecast.forecast * impts
                df_forecast.lower_forecast = df_forecast.lower_forecast * impts
                df_forecast.upper_forecast = df_forecast.upper_forecast * impts

        if self.constraint is not None:
            # print(f"constraint is {self.constraint}")
            # this might work out weirdly since self.df is scaled
            df_forecast = df_forecast.apply_constraints(
                **self.constraint,
                df_train=self.to_origin_space(self.df, trans_method="original"),
            )
        # RETURN COMPONENTS (long style) option
        df_forecast.predict_runtime = self.time() - predictStartTime
        return df_forecast

    def trend_analysis(self):
        trend_posterior, self.slope, intercept = self.rolling_trend(
            self.predicted_trend, np.array(self.t_predict)
        )
        (
            self.zero_crossings,
            self.changepoints,
            self.slope_sign,
            self.accel,
        ) = self.analyze_trend(self.slope, index=self.predicted_trend.index)
        return self

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

    def treatment_causal_impact(
        self, df, intervention_dates
    ):  # also add regressor input
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

    def get_new_params(self, method='fast'):
        # have fast option that avoids any of the loop approaches
        scaling = random.choices(['BaseScaler', 'other'], [0.8, 0.2])[0]
        if scaling == "other":
            scaling = RandomTransform(
                transformer_list=scalers,
                transformer_max_depth=1,
                allow_none=False,
                no_nan_fill=True,
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
            holiday_params['output'] = random.choices(
                ['multivariate', 'univariate'], [0.9, 0.1]
            )[0]
        anomaly_intervention = random.choices(
            [None, 'remove', 'detect_only', 'model'], [0.9, 0.3, 0.02, 0.05]
        )[0]
        if anomaly_intervention is not None:
            anomaly_detector_params = AnomalyRemoval.get_new_params(method=method)
            if anomaly_intervention == "model":
                model_list, model_prob = model_list_to_dict("scalable")
                anomaly_intervention = (
                    general_template[general_template['Model'].isin(model_list)]
                    .sample(1)
                    .to_dict("records")[0]
                )  # placeholder, probably
        else:
            anomaly_detector_params = None

        # random or pretested defaults
        if method in ['deep', 'all']:
            trend_base = 'deep'
            trend_standin = random.choices(
                [None, 'random_normal', 'rolling_trend', "changepoints"],
                [0.7, 0.3, 0.1, 0.2],
            )[0]
        else:
            trend_base = random.choices(
                ['pb1', 'pb2', 'pb3', 'random'], [0.1, 0.1, 0.0, 0.8]
            )[0]
            trend_standin = random.choices(
                [None, 'random_normal', 'changepoints'],
                [0.7, 0.2, 0.2],
            )[0]
        if trend_base == "random":
            model_str = random.choices(
                [
                    'AverageValueNaive',
                    'MetricMotif',
                    "LastValueNaive",
                    'SeasonalityMotif',
                    'WindowRegression',
                    'SectionalMotif',
                    'ARDL',
                    'VAR',
                    'UnivariateMotif',
                    # 'UnobservedComponents',
                    # "KalmanStateSpace",
                    'RRVAR',
                    "NeuralForecast",
                ],
                [0.05, 0.05, 0.2, 0.05, 0.05, 0.05, 0.15, 0.05, 0.05, 0.05, 0.01],
                k=1,
            )[0]
            trend_model = {'Model': model_str}
            if model_str in ['WindowRegression', 'MultivariateRegression']:
                method = "no_gpu"
            trend_model['ModelParameters'] = ModelMonster(model_str).get_new_params(
                method=method
            )
            trend_transformation = RandomTransform(
                transformer_list="scalable",
                transformer_max_depth=3,  # probably want some more usable defaults first as many random are senseless
            )
        elif trend_base == 'pb1':
            trend_model = {'Model': 'ARDL'}
            trend_model['ModelParameters'] = {
                "lags": 1,
                "trend": "n",
                "order": 0,
                "causal": False,
                "regression_type": "simple",
            }
            trend_transformation = {
                "fillna": "nearest",
                "transformations": {"0": "StandardScaler", "1": "AnomalyRemoval"},
                "transformation_params": {
                    "0": {},
                    "1": {
                        "method": "IQR",
                        "transform_dict": {
                            "fillna": None,
                            "transformations": {"0": "ClipOutliers"},
                            "transformation_params": {
                                "0": {"method": "clip", "std_threshold": 6}
                            },
                        },
                        "method_params": {
                            "iqr_threshold": 2.5,
                            "iqr_quantiles": [0.25, 0.75],
                        },
                        "fillna": "ffill",
                    },
                },
            }
        elif trend_base == 'pb2':
            # REPLACE THIS WITHOUT EXTRA TREES
            trend_model = {'Model': 'WindowRegression'}
            trend_model['ModelParameters'] = {
                "window_size": 12,
                "input_dim": "univariate",
                "output_dim": "1step",
                "normalize_window": False,
                "max_windows": 8000,
                "regression_type": None,
                "regression_model": {
                    "model": "ExtraTrees",
                    "model_params": {
                        "n_estimators": 100,
                        "min_samples_leaf": 1,
                        "max_depth": 20,
                    },
                },
            }
            trend_transformation = {
                "fillna": "ffill",
                "transformations": {"0": "AnomalyRemoval", "1": "RobustScaler"},
                "transformation_params": {
                    "0": {
                        "method": "IQR",
                        "transform_dict": {
                            "fillna": None,
                            "transformations": {"0": "ClipOutliers"},
                            "transformation_params": {
                                "0": {"method": "clip", "std_threshold": 6}
                            },
                        },
                        "method_params": {
                            "iqr_threshold": 2.5,
                            "iqr_quantiles": [0.25, 0.75],
                        },
                        "fillna": "ffill",
                    },
                    "1": {},
                },
            }
        elif trend_base == "deep":
            model_list = "all_pragmatic"
            model_list, model_prob = model_list_to_dict(model_list)
            model_str = random.choices(model_list, model_prob, k=1)[0]
            trend_model = {'Model': model_str}
            trend_model['ModelParameters'] = ModelMonster(model_str).get_new_params(
                method=method
            )
            trend_transformation = RandomTransform(
                transformer_list="all",
                transformer_max_depth=3,  # probably want some more usable defaults first as many random are senseless
            )

        trend_anomaly_intervention = random.choices([None, 'detect_only'], [0.5, 0.5])[
            0
        ]
        if trend_anomaly_intervention is not None:
            trend_anomaly_detector_params = AnomalyRemoval.get_new_params(method=method)
        else:
            trend_anomaly_detector_params = None
        if method in ['deep', 'all_linear']:
            linear_model = random.choices(
                [
                    'lstsq',
                    'linalg_solve',
                    'l1_norm',
                    'dwae_norm',
                    'quantile_norm',
                    'l1_positive',
                    'bayesian_linear',
                ],  # the minimize based norms get slow and memory hungry at scale
                [0.9, 0.2, 0.01, 0.01, 0.005, 0.01, 0.05],
            )[0]
        else:
            linear_model = random.choices(
                [
                    'lstsq',
                    'linalg_solve',
                    'bayesian_linear',
                    'l1_positive',
                ],
                [0.8, 0.15, 0.05, 0.01],
            )[0]
        recency_weighting = random.choices(
            [None, 0.05, 0.1, 0.25, 0.5], [0.7, 0.1, 0.1, 0.1, 0.05]
        )[0]
        if linear_model in ['lstsq']:
            linear_model = {
                'model': linear_model,
                'lambda': random.choices(
                    [None, 0.1, 1, 10, 100], [0.7, 0.1, 0.1, 0.1, 0.05]
                )[0],
                'recency_weighting': recency_weighting,
            }
        if linear_model in ['linalg_solve']:
            linear_model = {
                'model': linear_model,
                'lambda': random.choices([0, 0.1, 1, 10], [0.4, 0.2, 0.2, 0.2])[0],
                'recency_weighting': recency_weighting,
            }
        elif linear_model in ['l1_norm', 'dwae_norm', 'quantile_norm', 'l1_positive']:
            linear_model = {
                'model': linear_model,
                'recency_weighting': recency_weighting,
                'maxiter': random.choices(
                    [250, 5000, 15000, 25000], [0.2, 0.6, 0.1, 0.1]
                )[0],
                'method': random.choices(
                    [None, 'L-BFGS-B', 'Nelder-Mead', 'TNC', 'Powell'],
                    [0.95, 0.02, 0.02, 0.02, 0.02],
                )[0],
            }
        elif linear_model == "bayesian_linear":
            linear_model = {
                'model': "bayesian_linear",
                'alpha': random.choices([1.0, 0.1, 10], [0.8, 0.1, 0.1])[0],
                'gaussian_prior_mean': random.choices([0, 0.1, 1], [0.8, 0.08, 0.02])[
                    0
                ],
                'wishart_prior_scale': random.choices([1.0, 0.1, 10], [0.8, 0.1, 0.1])[
                    0
                ],
                'wishart_dof_excess': random.choices([0, 1, 5], [0.9, 0.05, 0.05])[0],
            }
        if method == "regressor":
            regressors_used = True
        else:
            regressors_used = random.choices([True, False], [0.5, 0.5])[0]
        ar_lags = random.choices(
            [None, [1], [1, 7], [7], [seasonal_int(small=True)]],
            [0.9, 0.025, 0.025, 0.05, 0.05],
        )[0]
        ar_interaction_seasonality = None
        if ar_lags is not None:
            ar_interaction_seasonality = random.choices(
                [None, 7, 'dayofweek', 'common_fourier'], [0.4, 0.2, 0.2, 0.2]
            )[0]
        seasonalities = random.choices(
            [
                [7, 365.25],
                ["dayofweek", 365.25],
                ["quarterlydayofweek", 365.25],
                ["month", "dayofweek", "weekdayofmonth"],
                ['weekdayofmonth', 'common_fourier'],
                ["simple_binarized"],
                ['hourlydayofweek', 8766.0],  # for hourly data
                [12],  # monthly data
                None,
                "other",
            ],
            [0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.04, 0.04, 0.01, 0.1],
        )[0]
        if seasonalities == "other":
            predefined = random.choices([True, False], [0.5, 0.5])[0]
            if predefined:
                seasonalities = [random.choice(date_part_methods)]
            else:
                comp_opts = datepart_components + [7, 365.25, 12]
                seasonalities = random.choices(comp_opts, k=2)
        return {
            "preprocessing_transformation": RandomTransform(
                transformer_list=superfast_transformer_dict,
                transformer_max_depth=2,
                allow_none=True,
            ),
            "scaling": scaling,
            # "past_impacts_intervention": self.past_impacts_intervention,
            "seasonalities": seasonalities,
            "ar_lags": ar_lags,
            "ar_interaction_seasonality": ar_interaction_seasonality,
            "anomaly_detector_params": anomaly_detector_params,
            "anomaly_intervention": anomaly_intervention,
            "holiday_detector_params": holiday_params,
            # "holiday_countries": self.holiday_countries,
            "holiday_countries_used": random.choices([True, False], [0.5, 0.5])[0],
            "multivariate_feature": random.choices(
                [None, "feature_agglomeration", 'group_average', 'oscillator', 'fft'],
                [0.9, 0.02, 0.02, 0.0, 0.1],
            )[0],
            "multivariate_transformation": RandomTransform(
                transformer_list="scalable",
                transformer_max_depth=3,  # probably want some more usable defaults first as many random are senseless
            ),
            "regressor_transformation": RandomTransform(
                transformer_list="scalable",  # {**scalers, **decompositions}
                transformer_max_depth=1,
                allow_none=False,
                no_nan_fill=False,  # probably want some more usable defaults first as many random are senseless
            ),
            "regressors_used": regressors_used,
            "linear_model": linear_model,
            "randomwalk_n": random.choices([None, 10], [0.5, 0.5])[0],
            "trend_window": random.choices(
                [None, 3, 15, 90, 364], [0.2, 0.2, 0.2, 0.2, 0.2]
            )[0],
            "trend_standin": trend_standin,
            "trend_anomaly_detector_params": trend_anomaly_detector_params,
            # "trend_anomaly_intervention": trend_anomaly_intervention,
            "trend_transformation": trend_transformation,
            "trend_model": trend_model,
            "trend_phi": random.choices([None, 0.995, 0.98], [0.9, 0.05, 0.1])[0],
            "x_scaler": random.choices([True, False], [0.2, 0.8])[0],
        }

    def get_params(self):
        return {
            "preprocessing_transformation": self.preprocessing_transformation,
            "scaling": self.scaling,
            "past_impacts_intervention": self.past_impacts_intervention,  # not in new
            "seasonalities": self.seasonalities,
            "ar_lags": self.ar_lags,
            "ar_interaction_seasonality": (
                self.ar_interaction_seasonality if self.ar_lags is not None else None
            ),
            "anomaly_detector_params": self.anomaly_detector_params,
            "anomaly_intervention": self.anomaly_intervention,
            "holiday_detector_params": self.holiday_detector_params,
            # "holiday_intervention": self.holiday_intervention,
            # "holiday_countries": self.holiday_countries,
            "holiday_countries_used": self.holiday_countries_used,
            "multivariate_feature": self.multivariate_feature,
            "multivariate_transformation": (
                self.multivariate_transformation
                if self.multivariate_feature is not None
                else None
            ),
            "regressor_transformation": (
                self.regressor_transformation if self.regressors_used else None
            ),
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
            "x_scaler": self.x_scaler,
            # "constraint": self.constraint,
        }

    def plot_components(
        self,
        prediction=None,
        series=None,
        figsize=(16, 9),
        to_origin_space=True,
        title=None,
        start_date=None,
    ):
        """Plot breakdown of linear model components.

        Args:
            prediction: the forecast object
            series (str): name of series to plot, if desired
            figsize (tuple): figure size
            to_origin_space (bool): setting to False can make the graph look right due to preprocessing transformers, but to the wrong scale
                especially useful if "AlignLastValue" and other transformers present
            title (str): title
            start_date (str): slice point for start date, can make some high frequency components easier to see with a shorter window
        """
        if series is None:
            series = random.choice(self.column_names)
        if title is None:
            title = f"Model Components for {series}"
        plot_list = []
        if prediction is not None:
            plot_list.append(prediction.forecast[series].rename("forecast"))
            plt_idx = prediction.forecast.index
        else:
            plt_idx = None
        if to_origin_space:
            trend = self._trend_to_origin()
        plot_list.append(trend[series].rename("trend"))
        if self.impacts is not None:
            plot_list.append((self.impacts[series].rename("impact %") - 1.0) * 100)
        if plt_idx is None:
            plot_list.append(
                self.process_components(to_origin_space=to_origin_space)[series]
            )
        else:
            plot_list.append(
                self.process_components(to_origin_space=to_origin_space)[series].loc[
                    plt_idx
                ]
            )
        plot_df = pd.concat(plot_list, axis=1)
        if start_date is not None:
            if start_date == "auto":
                plot_df = plot_df.iloc[-(prediction.forecast_length * 3) :]
            else:
                plot_df = plot_df[plot_df.index >= start_date]
        return plot_df.plot(subplots=True, figsize=figsize, title=title)

    def _trend_to_origin(self):
        if self.predicted_trend.shape[0] == self.forecast_length:
            trend = self.to_origin_space(self.predicted_trend, trans_method='forecast')
        elif self.predicted_trend.shape[0] == self.trend_train.shape[0]:
            trend = self.to_origin_space(self.predicted_trend, trans_method='original')
        else:
            hdn = self.predicted_trend.shape[0] - self.forecast_length
            hdn = hdn if hdn > 0 else 0
            trend = pd.concat(
                [
                    self.to_origin_space(
                        self.predicted_trend.head(hdn), trans_method='original'
                    ),
                    self.to_origin_space(
                        self.predicted_trend.tail(self.forecast_length),
                        trans_method='forecast',
                    ),
                ]
            )
        return trend

    def return_components(self, to_origin_space=True, include_impacts=False):
        """Return additive elements of forecast, linear and trend. If impacts included, it is a multiplicative term.

        Args:
            to_origin_space (bool) if False, will not reverse transform linear components
            include_impacts (bool) if True, impacts are included in the returned dataframe
        """
        plot_list = []
        plot_list.append(self.process_components(to_origin_space=to_origin_space))
        if to_origin_space:
            trend = self._trend_to_origin()
        trend.columns = pd.MultiIndex.from_arrays(
            [trend.columns, ['trend'] * len(trend.columns)]
        )
        plot_list.append(trend)
        if self.impacts is not None and include_impacts:
            impacts = self.impacts.copy()
            impacts.columns = pd.MultiIndex.from_arrays(
                [impacts.columns, ['impacts'] * len(impacts.columns)]
            )
            plot_list.append(impacts)
        return pd.concat(plot_list, axis=1)

    def plot_trend(
        self,
        series=None,
        vline=None,
        colors=["#d4f74f", "#82ab5a", "#ff6c05", "#c12600"],
        title=None,
        start_date=None,
        **kwargs,
    ):
        # rerun the analysis with latest
        self.trend_analysis()
        """Trend plots have a bug if AlignLastValue or AlignLastDiff are present. Underlying data is still ok."""
        # YMAX from PLOT ONLY
        if series is None:
            series = random.choice(self.column_names)
        if title is None:
            title = f"Trend Breakdown for {series}"
        p_indx = self.column_names.get_loc(series)
        cur_trend = self.predicted_trend[series].copy()
        tls = len(cur_trend)
        plot_df = pd.DataFrame(
            {
                'decline_accelerating': cur_trend[
                    (
                        np.hstack((np.signbit(self.accel[:, p_indx]), False))
                        & self.slope_sign[:, p_indx]
                    )[-tls:]
                ],
                'decline_decelerating': cur_trend[
                    (
                        (~np.hstack((np.signbit(self.accel[:, p_indx]), False)))
                        & self.slope_sign[:, p_indx]
                    )[-tls:]
                ],
                'growth_decelerating': cur_trend[
                    (
                        np.hstack((np.signbit(self.accel[:, p_indx]), False))
                        & (~self.slope_sign[:, p_indx])
                    )[-tls:]
                ],
                'growth_accelerating': cur_trend[
                    (
                        (~np.hstack((np.signbit(self.accel[:, p_indx]), False)))
                        & (~self.slope_sign[:, p_indx])
                    )[-tls:]
                ],
            },
            index=cur_trend.index,
        )
        if start_date == "auto":
            slx = self.forecast_length * 3
            if slx > len(plot_df.index):
                slx = 0
            start_date = plot_df.index[-slx]
        if start_date is not None:
            plot_df = plot_df[plot_df.index >= start_date]
        ax = plot_df.plot(title=title, color=colors, **kwargs)
        handles, labels = ax.get_legend_handles_labels()
        # ax.scatter(cur_trend.index[self.changepoints[:, p_indx]], cur_trend[self.changepoints[:, p_indx]], c='#fdcc09', s=4.0)
        # ax.scatter(cur_trend.index[self.zero_crossings[:, p_indx]], cur_trend[self.zero_crossings[:, p_indx]], c='#512f74', s=4.0)
        if self.trend_anomaly_detector is not None:
            if self.trend_anomaly_detector.output == "univariate":
                i_anom = self.trend_anomaly_detector.anomalies.index[
                    self.anomaly_detector.anomalies.iloc[:, 0] == -1
                ]
            else:
                series_anom = self.trend_anomaly_detector.anomalies[series]
                i_anom = series_anom[series_anom == -1].index
            if start_date is not None:
                i_anom = i_anom[i_anom >= start_date]
            i_anom = i_anom[i_anom >= cur_trend.index[0]]
            # only plot if some anomalies, and not way too many anomalies
            if len(i_anom) > 0 and len(i_anom) < len(plot_df) * 0.5:
                scat1 = ax.scatter(
                    i_anom.tolist(), cur_trend.loc[i_anom], c="red", s=16.0
                )
                handles += [scat1]
                labels += ['trend anomalies']
        if vline is not None:
            ax.vlines(
                x=vline,
                ls='--',
                lw=1,
                colors='darkred',
                ymin=cur_trend[cur_trend.index >= start_date].min(),
                ymax=cur_trend[cur_trend.index >= start_date].max(),
            )
        ax.legend(handles, labels)
        return ax

    def plot_forecast(
        self,
        prediction,
        actuals=None,
        series=None,
        start_date=None,
        anomaly_color="darkslateblue",
        holiday_color="darkgreen",
        trend_anomaly_color='slategray',
        point_size=12.0,
    ):
        """Plot a forecast time series.

        Args:
            prediction (model prediction object, required)
            actuals (pd.DataFrame): wide style df, of know data if available
            series (str): name of time series column to plot
            start_date (str or Timestamp): point at which to begin X axis
            anomaly_color (str): name of anomaly point color
            holiday_color (str): name of holiday point color
            trend_anomaly_color (str): name of trend anomaly point color
            point_size (str): point size for all anomalies
        """
        if series is None:
            series = random.choice(self.column_names)
        if actuals is None or not isinstance(
            actuals, (pd.DataFrame, np.array, pd.Series)
        ):
            actuals_used = prediction.forecast
            actuals_flag = False
        else:
            actuals_flag = True
            actuals_used = actuals.reindex(prediction.forecast.index)
        vline = (
            None
            if self.forecast_length is None
            else prediction.forecast.index[-self.forecast_length]
        )
        if start_date == "auto":
            if actuals is not None:
                slx = -self.forecast_length * 3
                if abs(slx) > actuals.shape[0]:
                    slx = 0
                start_date = actuals.index[slx]
            else:
                start_date = prediction.forecast.index[0]
        ax = prediction.plot(
            (
                actuals_used.loc[prediction.forecast.index]
                if actuals_flag is not None
                else None
            ),
            series=series,
            vline=vline,
            start_date=start_date,
        )
        handles, labels = ax.get_legend_handles_labels()
        if self.anomaly_detector:
            if self.anomaly_detector.output == "univariate":
                i_anom = self.anomaly_detector.anomalies.index[
                    self.anomaly_detector.anomalies.iloc[:, 0] == -1
                ]
            else:
                series_anom = self.anomaly_detector.anomalies[series]
                i_anom = series_anom[series_anom == -1].index
            if start_date is not None:
                i_anom = i_anom[i_anom >= start_date]
            if actuals_flag:
                i_anom = i_anom[i_anom >= actuals.index[0]]
            if len(i_anom) > 0 and (
                not actuals_flag or len(i_anom) < len(actuals) * 0.5
            ):
                scat1 = ax.scatter(
                    i_anom.tolist(),
                    actuals_used.reindex(i_anom)[series],
                    c=anomaly_color,
                    s=point_size,
                )
                handles += [scat1]
                labels += ["anomalies"]
        if self.holiday_detector:
            i_anom = self.holiday_detector.dates_to_holidays(
                self.df.index, style="series_flag"
            )[series]
            i_anom = i_anom.index[i_anom == 1]
            if actuals_flag:
                i_anom = i_anom[i_anom >= actuals.index[0]]
            if len(i_anom) > 0 and (
                not actuals_flag or len(i_anom) < len(actuals) * 0.5
            ):
                scat2 = ax.scatter(
                    i_anom.tolist(),
                    actuals_used.reindex(i_anom)[series],
                    c=holiday_color,
                    s=point_size,
                )
                handles += [scat2]
                labels += ["detected holidays"]
        if self.trend_anomaly_detector is not None:
            if self.trend_anomaly_detector.output == "univariate":
                i_anom = self.trend_anomaly_detector.anomalies.index[
                    self.anomaly_detector.anomalies.iloc[:, 0] == -1
                ]
            else:
                series_anom = self.trend_anomaly_detector.anomalies[series]
                i_anom = series_anom[series_anom == -1].index
            if start_date is not None:
                i_anom = i_anom[i_anom >= start_date]
            if actuals_flag:
                i_anom = i_anom[i_anom >= actuals.index[0]]
            if len(i_anom) > 0 and (
                not actuals_flag or len(i_anom) < len(actuals) * 0.5
            ):
                scat3 = ax.scatter(
                    i_anom.tolist(),
                    actuals_used.reindex(i_anom)[series],
                    c=trend_anomaly_color,
                    s=point_size,
                )
                handles += [scat3]
                labels += ["trend anomalies"]
        ax.legend(handles, labels)
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
    df.columns = [str(prefix) + str(col) for col in df.columns]
    return df


def create_t(ds):
    return (ds - ds.min()) / (ds.max() - ds.min())


#####################################
# JUST LEAST SQUARES UNIVARIATE
# https://stackoverflow.com/questions/17679140/multiple-linear-regression-with-python


def lstsq_solve(X, y, lamb=1, identity_matrix=None):
    if identity_matrix is None:
        identity_matrix = np.zeros((X.shape[1], X.shape[1]))
        np.fill_diagonal(identity_matrix, 1)
        identity_matrix[0, 0] = 0
    if lamb is None:
        lamb = 1.0
    XtX_lamb = X.T.dot(X) + lamb * identity_matrix
    XtY = X.T.dot(y)
    return np.linalg.solve(XtX_lamb, XtY)


def cost_function_l1(params, X, y):
    return np.sum(np.abs(y - np.dot(X, params.reshape(X.shape[1], y.shape[1]))))


def cost_function_l1_positive(params, X, y):
    return np.sum(
        np.abs(
            y
            - np.dot(X, np.where(params < 0, 0, params).reshape(X.shape[1], y.shape[1]))
        )
    )


# actually this is more like MADE
def cost_function_dwae(params, X, y):
    A = y
    F = np.dot(X, params.reshape(X.shape[1], y.shape[1]))
    last_of_array = y[[0] + list(range(len(y) - 1))]
    return np.sum(
        np.nanmean(
            np.where(
                np.sign(F - last_of_array) == np.sign(A - last_of_array),
                np.abs(A - F),
                (np.abs(A - F) + 1) ** 2,
            ),
            axis=0,
        )
    )


def cost_function_quantile(params, X, y, q=0.9):
    cut = int(y.shape[0] * q)
    return np.sum(
        np.partition(
            np.abs(y - np.dot(X, params.reshape(X.shape[1], y.shape[1]))), cut, axis=0
        )[0:cut]
    )


def cost_function_l2(params, X, y):
    return np.linalg.norm(y - np.dot(X, params.reshape(X.shape[1], y.shape[1])))


# could do partial pooling by minimizing a function that mixes shared and unshared coefficients (multiplicative)
def lstsq_minimize(X, y, maxiter=15000, cost_function="l1", method=None):
    """Any cost function version of lin reg."""
    # start with lstsq fit as estimated point
    x0 = lstsq_solve(X, y).flatten()
    # assuming scaled, these should be reasonable bounds
    bounds = [(-10, 10) for x in x0]
    if cost_function == "dwae":
        bounds = [(-0.5, 10) for x in x0]
        cost_func = cost_function_dwae
    elif cost_function == "quantile":
        cost_func = cost_function_quantile
    elif cost_function == "l1_positive":
        max_bound = 14
        bounds = [(0, max_bound) for x in x0]
        cost_func = cost_function_l1
        x0[x0 <= 0] = 0.01
        x0[x0 > max_bound] = max_bound - 0.0001
    else:
        cost_func = cost_function_l1
    return minimize(
        cost_func,
        x0,
        args=(X, y),
        bounds=bounds,
        method=method,
        options={'maxiter': maxiter},
    ).x.reshape(X.shape[1], y.shape[1])


def fit_linear_model(x, y, params=None):
    if params is None:
        params = {}
    model_type = params.get("model", "lstsq")
    lambd = params.get("lambda", None)
    rec = params.get("recency_weighting", None)
    if lambd is not None:
        id_mat = np.zeros((x.shape[1], x.shape[1]))
        np.fill_diagonal(id_mat, 1)
        id_mat[0, 0] = 0
    else:
        id_mat = None
    if rec is not None:
        weights = (np.arange(len(x)) + 1) ** rec  # 0.05 - 0.25
        x = x * weights[..., None]
        y = np.asarray(y) * weights[..., None]
    if model_type == "lstsq":
        if lambd is not None:
            return np.linalg.lstsq(x.T.dot(x) + lambd * id_mat, x.T.dot(y), rcond=None)[
                0
            ]
        else:
            return np.linalg.lstsq(x, y, rcond=None)[0]
    elif model_type == "linalg_solve":
        return lstsq_solve(x, y, lamb=lambd, identity_matrix=id_mat)
    elif model_type == "l1_norm":
        return lstsq_minimize(
            np.asarray(x),
            np.asarray(y),
            maxiter=params.get("maxiter", 15000),
            method=params.get("method", None),
            cost_function="l1",
        )
    elif model_type == "quantile_norm":
        return lstsq_minimize(
            np.asarray(x),
            np.asarray(y),
            maxiter=params.get("maxiter", 15000),
            method=params.get("method", None),
            cost_function="quantile",
        )
    elif model_type == "dwae_norm":
        return lstsq_minimize(
            np.asarray(x),
            np.asarray(y),
            maxiter=params.get("maxiter", 15000),
            method=params.get("method", None),
            cost_function="dwae",
        )
    elif model_type == "l1_positive":
        return lstsq_minimize(
            np.asarray(x),
            np.asarray(y),
            maxiter=params.get("maxiter", 15000),
            method=params.get("method", None),
            cost_function="l1_positive",
        )
    elif model_type == "bayesian_linear":
        # this could support better probabilistic bounds but that is not yet done
        model = BayesianMultiOutputRegression(
            alpha=params.get("alpha", 1),
            gaussian_prior_mean=params.get("gaussian_prior_mean", 0),
            wishart_prior_scale=params.get("wishart_prior_scale", 1),
            wishart_dof_excess=params.get("wishart_dof_excess", 0),
        )
        model.fit(X=np.asarray(x), Y=np.asarray(y))
        return model.params
    else:
        raise ValueError("linear model not recognized")


class BayesianMultiOutputRegression:
    """Bayesian Linear Regression, conjugate prior update.

    Args:
        gaussian_prior_mean (float): mean of prior, a small positive value can encourage positive coefs which make better component plots
        alpha (float): prior scale of gaussian covariance, effectively a regularization term
        wishart_dof_excess (int): Larger values make the prior more peaked around the scale matrix.
        wishart_prior_scale (float): A larger value means a smaller prior variance on the noise covariance, while a smaller value means more prior uncertainty about it.
    """

    def __init__(
        self,
        gaussian_prior_mean=0,
        alpha=1.0,
        wishart_prior_scale=1.0,
        wishart_dof_excess=0,
    ):
        self.gaussian_prior_mean = gaussian_prior_mean
        self.alpha = alpha
        self.wishart_prior_scale = wishart_prior_scale
        self.wishart_dof_excess = wishart_dof_excess

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        n_outputs = Y.shape[1]

        # Prior for the regression coefficients: Gaussian
        # For Ridge regularization: Set the diagonal elements to alpha
        self.m_0 = (
            np.zeros((n_features, n_outputs)) + self.gaussian_prior_mean
        )  # Prior mean
        self.S_0 = self.alpha * np.eye(n_features)  # Prior covariance

        # Prior for the precision matrix (inverse covariance): Wishart
        self.nu_0 = n_features + self.wishart_dof_excess  # Degrees of freedom
        self.W_0_inv = self.wishart_prior_scale * np.eye(
            n_outputs
        )  # Scale matrix (inverse)

        # Posterior for the regression coefficients
        S_0_inv = np.linalg.inv(self.S_0)
        S_n_inv = S_0_inv + X.T @ X
        S_n = np.linalg.inv(S_n_inv)
        m_n = S_n @ (S_0_inv @ self.m_0 + X.T @ Y)

        # Posterior for the precision matrix
        nu_n = self.nu_0 + n_samples
        W_n_inv = (
            self.W_0_inv
            + Y.T @ Y
            + self.m_0.T @ S_0_inv @ self.m_0
            - m_n.T @ S_n_inv @ m_n
        )

        self.m_n = self.params = m_n
        self.S_n = S_n
        self.nu_n = nu_n
        self.W_n_inv = W_n_inv

    def predict(self, X, return_std=False):
        Y_pred = X @ self.m_n
        if return_std:
            # Average predictive variance for each output dimension
            Y_var = (
                np.einsum('ij,jk,ik->i', X, self.S_n, X)
                + np.trace(np.linalg.inv(self.nu_n * self.W_n_inv))
                / self.W_n_inv.shape[0]
            )
            return Y_pred, np.sqrt(Y_var)
        return Y_pred

    def sample_posterior(self, n_samples=1):
        # from scipy.stats import wishart
        # Sample from the posterior distribution of the coefficients
        # beta_samples = np.random.multivariate_normal(self.m_n.ravel(), self.S_n, size=n_samples)
        # Sample from the posterior distribution of the precision matrix
        # precision_samples = wishart(df=self.nu_n, scale=np.linalg.inv(self.W_n_inv)).rvs(n_samples)
        # return beta_samples, precision_samples

        sampled_weights = np.zeros((n_samples, self.m_n.shape[0], self.m_n.shape[1]))
        for i in range(self.m_n.shape[1]):
            sampled_weights[:, :, i] = np.random.multivariate_normal(
                self.m_n[:, i], self.S_n, n_samples
            )
        return sampled_weights


# Seasonalities
# interaction effect only on first two seasonalities if order matches

# Categorical Features
# (multivariate summaries by group)

# past impacts (option to not enforce, only show)

# what is still needed:
# bayesian linear model options
# more multivariate summaries
# add anomaly classifier to forecast
# refine transformations and models generated by get_new_params
# reweight so loop is not required so often
# test and bug fix everything
# l1_norm isn't working
# unittests

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


if False:
    # test holiday countries, regressors, impacts
    from autots import load_daily
    import matplotlib.pyplot as plt

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
    df_daily = load_daily(long=False)
    # add nan
    df_daily.iloc[100, :] = np.nan
    forecast_length = 240
    include_history = True
    df_train = df_daily[:-forecast_length].iloc[:, 1:]
    df_test = df_daily[-forecast_length:].iloc[:, 1:]
    fake_regr = df_daily[:-forecast_length].iloc[:, 0:1]
    fake_regr_fcst = (
        df_daily.iloc[:, 0:1]
        if include_history
        else df_daily[-forecast_length:].iloc[:, 0:1]
    )
    flag_regressor = pd.DataFrame(1, index=fake_regr.index, columns=["flag_test"])
    flag_regressor_fcst = pd.DataFrame(
        1, index=fake_regr_fcst.index, columns=["flag_test"]
    )
    regr_per_series = {
        str(df_train.columns[0]): pd.DataFrame(
            np.random.normal(size=(len(df_train), 1)), index=df_train.index
        )
    }
    regr_per_series_fcst = {
        str(df_train.columns[0]): pd.DataFrame(
            np.random.normal(size=(forecast_length, 1)), index=df_test.index
        )
    }
    constraint = None
    past_impacts = pd.DataFrame(
        0, index=df_train.index, columns=df_train.columns
    ).astype(float)
    past_impacts.iloc[-10:, 0] = np.geomspace(1, 10)[0:10] / 100
    past_impacts.iloc[-30:, -1] = np.linspace(1, 10)[0:30] / -100
    past_impacts_full = pd.DataFrame(0, index=df_daily.index, columns=df_daily.columns)
    future_impacts = pd.DataFrame(
        0, index=df_test.index, columns=df_test.columns
    ).astype(float)
    future_impacts.iloc[0:10, 0] = (np.linspace(1, 10)[0:10] + 10) / 100

    c_params = Cassandra().get_new_params()
    c_params = {
        "preprocessing_transformation": {
            "fillna": "ffill",
            "transformations": {"0": "ClipOutliers", "1": "AlignLastDiff"},
            "transformation_params": {
                "0": {"method": "clip", "std_threshold": 4.5, "fillna": None},
                "1": {
                    "rows": 7,
                    "displacement_rows": 1,
                    "quantile": 1.0,
                    "decay_span": 90,
                },
            },
        },
        "scaling": {
            "fillna": None,
            "transformations": {"0": "MaxAbsScaler"},
            "transformation_params": {"0": {}},
        },
        "past_impacts_intervention": None,
        "seasonalities": ["quarterlydayofweek", 365.25],
        "ar_lags": [1],
        "ar_interaction_seasonality": "common_fourier",
        "anomaly_detector_params": {
            "method": "IQR",
            "method_params": {"iqr_threshold": 3.0, "iqr_quantiles": [0.25, 0.75]},
            "fillna": "fake_date",
            "transform_dict": {
                "fillna": "ffill",
                "transformations": {"0": "StandardScaler"},
                "transformation_params": {"0": {}},
            },
            "isolated_only": False,
        },
        "anomaly_intervention": None,
        "holiday_detector_params": None,
        "holiday_countries_used": False,
        "multivariate_feature": "fft",
        "multivariate_transformation": None,
        "regressor_transformation": None,
        "regressors_used": False,
        "linear_model": {"model": "lstsq", "lambda": None, "recency_weighting": 0.05},
        "randomwalk_n": 10,
        "trend_window": None,
        "trend_standin": None,
        "trend_anomaly_detector_params": {
            "method": "nonparametric",
            "method_params": {
                "p": None,
                "z_init": 2.0,
                "z_limit": 12,
                "z_step": 0.25,
                "inverse": False,
                "max_contamination": 0.25,
                "mean_weight": 200,
                "sd_weight": 100,
                "anomaly_count_weight": 1.0,
            },
            "fillna": "fake_date",
            "transform_dict": {
                "fillna": "zero",
                "transformations": {"0": "AlignLastValue"},
                "transformation_params": {
                    "0": {
                        "rows": 1,
                        "lag": 7,
                        "method": "additive",
                        "strength": 0.2,
                        "first_value_only": False,
                        "threshold": 1,
                    }
                },
            },
            "isolated_only": False,
        },
        "trend_transformation": {
            "fillna": "mean",
            "transformations": {"0": "DiffSmoother", "1": "RegressionFilter"},
            "transformation_params": {
                "0": {
                    "method": "zscore",
                    "method_params": {"distribution": "uniform", "alpha": 0.05},
                    "transform_dict": None,
                    "reverse_alignment": False,
                    "isolated_only": True,
                    "fillna": "ffill",
                },
                "1": {
                    "sigma": 1,
                    "rolling_window": 90,
                    "run_order": "season_first",
                    "regression_params": {
                        "regression_model": {
                            "model": "ElasticNet",
                            "model_params": {
                                "l1_ratio": 0.9,
                                "fit_intercept": True,
                                "selection": "cyclic",
                            },
                        },
                        "datepart_method": ["db2_365.25_12_0.5", "morlet_7_7_1"],
                        "polynomial_degree": None,
                        "transform_dict": None,
                        "holiday_countries_used": False,
                    },
                    "holiday_params": None,
                    "trend_method": "rolling_mean",
                },
            },
        },
        "trend_model": {
            "Model": "ARDL",
            "ModelParameters": {
                "lags": 1,
                "trend": "n",
                "order": 0,
                "causal": False,
                "regression_type": "simple",
            },
        },
        "trend_phi": None,
    }
    c_params['regressors_used'] = False
    # c_params['trend_phi'] = 0.9

    mod = Cassandra(
        n_jobs=1,
        **c_params,
        constraint=constraint,
        holiday_countries=holiday_countries,
        max_multicolinearity=0.0001,
    )
    mod.fit(
        df_train,
        categorical_groups=categorical_groups,
        future_regressor=fake_regr,
        regressor_per_series=regr_per_series,
        # past_impacts=past_impacts,
        flag_regressors=flag_regressor,
    )
    pred = mod.predict(
        forecast_length=forecast_length,
        include_history=include_history,
        future_regressor=fake_regr_fcst,
        regressor_per_series=regr_per_series_fcst,
        future_impacts=future_impacts,
        flag_regressors=flag_regressor_fcst,
        include_organic=True,
    )
    result = pred.forecast
    series = random.choice(mod.column_names)
    # series = "wiki_Periodic_table"
    series = 'wiki_all'
    mod.regressors_used
    mod.holiday_countries_used
    with plt.style.context("ggplot"):
        start_date = "auto"
        mod.plot_forecast(
            pred,
            actuals=df_daily if include_history else df_test,
            series=series,
            start_date=start_date,
        )

        # plt.show()
        # plt.savefig("Cassandra_forecast.png", dpi=300)
        # mod.plot_components(pred, series=series, to_origin_space=False)
        # plt.show()
        mod.plot_components(
            pred, series=series, to_origin_space=False, start_date=start_date
        )
        # plt.savefig("Cassandra_components3.png", dpi=300, bbox_inches="tight")
        plt.show()
        mod.plot_trend(
            series=series, vline=result.index[-forecast_length], start_date=start_date
        )
        # plt.savefig("Cassandra_trend.png", dpi=300, bbox_inches="tight")
    pred.evaluate(
        df_daily.reindex(result.index)[df_train.columns]
        if include_history
        else df_test[df_train.columns]
    )
    print(pred.avg_metrics.round(1))

    ################################
    # if not mod.regressors_used:
    dates = df_daily.index.union(
        mod.create_forecast_index(forecast_length, last_date=df_daily.index[-1])
    )
    regr_ps = {
        'wiki_Germany': regr_per_series_fcst['wiki_Germany'].reindex(
            dates, fill_value=0
        )
    }
    # NOT PASSING PROPER REGRESSORS HERE (zero fill) SO MAY LOOK OFF IF REGRESSORS_USED
    pred2 = mod.predict(
        forecast_length=forecast_length,
        include_history=True,
        new_df=df_daily[df_train.columns],
        flag_regressors=flag_regressor_fcst.reindex(dates, fill_value=0),
        future_regressor=fake_regr_fcst.reindex(dates, fill_value=0),
        regressor_per_series=regr_ps,
        past_impacts=past_impacts_full[df_train.columns],
    )
    mod.plot_forecast(pred2, actuals=df_daily, series=series, start_date=start_date)
    mod.return_components()
    # and try retraining back with shorter dataset and see if it matches prior
    mod.fit_data(df_train, past_impacts=past_impacts[df_train.columns])
    pred3 = mod.predict(
        forecast_length=forecast_length,
        include_history=True,
        flag_regressors=flag_regressor_fcst.reindex(df_daily.index, fill_value=0),
        future_regressor=fake_regr_fcst.reindex(df_daily.index, fill_value=0),
        regressor_per_series=regr_ps,
        future_impacts=future_impacts,
    )
    mod.plot_forecast(
        pred3,
        actuals=df_daily if include_history else df_test,
        series=series,
        start_date=start_date,
    )

    print(c_params['trend_model'])

# Known and Possible ISSUES:
# Multivariate feature working or not?
# multiplicative seasonality


# Automation
# allow some config inputs, or automated fit
# output to table
# compare coefficients change over time, accuracy over time
# comparing different sources? and/or allowing one forecast to be used as a regressor for another
# would allow for earlier detection of broken connections

"""
ax = pred.plot(df_daily if include_history else df_test, series=series, vline=df_test.index[0], start_date=start_date)
handles, labels = ax.get_legend_handles_labels()
if mod.anomaly_detector:
    if mod.anomaly_detector.output == "univariate":
        i_anom = mod.anomaly_detector.anomalies.index[mod.anomaly_detector.anomalies.iloc[:, 0] == -1]
    else:
        series_anom = mod.anomaly_detector.anomalies[series]
        i_anom = series_anom[series_anom == -1].index
        i_anom = i_anom[i_anom >= start_date]
    if len(i_anom) > 0 and len(i_anom) < len(df_daily) * 0.5:
        scat1 = ax.scatter(i_anom.tolist(), df_daily.loc[i_anom, :][series], c="darkslateblue", s=12.0)
        handles += [scat1]
        labels += ["anomalies"]
if mod.holiday_detector:
    i_anom = mod.holiday_detector.dates_to_holidays(mod.df.index, style="series_flag")[series]
    i_anom = i_anom.index[i_anom == 1]
    if len(i_anom) > 0 and len(i_anom) < len(df_daily) * 0.5:
        scat2 = ax.scatter(i_anom.tolist(), df_daily.loc[i_anom, :][series], c="darkgreen", s=12.0)
        handles += [scat2]
        labels += ["detected holidays"]
if mod.trend_anomaly_detector is not None:
    if mod.trend_anomaly_detector.output == "univariate":
        i_anom = mod.trend_anomaly_detector.anomalies.index[mod.anomaly_detector.anomalies.iloc[:, 0] == -1]
    else:
        series_anom = mod.trend_anomaly_detector.anomalies[series]
        i_anom = series_anom[series_anom == -1].index
    if start_date is not None:
        i_anom = i_anom[i_anom >= start_date]
    if len(i_anom) > 0 and len(i_anom) < len(df_daily) * 0.5:
        scat3 = ax.scatter(i_anom.tolist(), df_daily.loc[i_anom][series], c="slategray", s=12.0)
        handles += [scat3]
        labels += ["trend anomalies"]
ax.legend(handles, labels)
"""
