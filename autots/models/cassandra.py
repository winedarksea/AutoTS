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
from autots.models.base import ModelObject
from autots.templates.general import general_template
from autots.tools.holiday import holiday_flag
from autots.tools.window_functions import sliding_window_view
from autots.evaluator.auto_model import ModelMonster, model_forecast


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
        holiday_countries_used: bool = None,
        multivariate_feature: str = None,  # by group, if given
        multivariate_transformation: str = None,  # applied before creating feature
        regressor_transformation: dict = None,  # applied to future_regressor and regressor_per_series
        regressors_used: bool = None,  # on/off of additional user inputs
        linear_model: dict = None,  # lstsq WEIGHTED Ridge, bayesian, bayesian hierarchial, l1 normed or other loss (numba),
        randomwalk_n: int = None,  # use stats of source df
        trend_window: int = 30,  # set to None to use raw residuals
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
        self.trend_anomaly_intervention = trend_anomaly_intervention
        self.trend_transformation = trend_transformation
        self.trend_model = trend_model
        self.trend_phi = trend_phi
        self.constraints = constraints
        self.frequency = frequency
        self.prediction_interval = prediction_interval
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

    def base_scaler(self, df):
        self.scaler_mean = np.mean(df, axis=0)
        self.scaler_std = np.std(df, axis=0)
        return (df - self.scaler_mean) / self.scaler_std

    def to_origin_space(self, df):
        """Take transformed outputs back to original feature space."""
        if self.scaling == "BaseScaler":
            return self.preprocesser.inverse_transform(df) * self.scaler_std + self.scaler_mean
        else:
            return self.scaler.inverse_transform(self.preprocesser.inverse_transform(df))

    def create_t(self, DTindex):
        return (DTindex - self.ds_min) / (self.ds_max - self.ds_min)

    def fit(self, df, future_regressor=None, regressor_per_series=None, flag_regressors=None, categorical_groups=None, past_impacts=None):
        # flag regressors bypass preprocessing
        # ideally allow both pd.DataFrame and np.array inputs (index for array?
        self.df = df.copy()
        self.basic_profile(self.df)
        # standardize groupings (extra/missing dealt with)
        if categorical_groups is None:
            categorical_groups = {}
        self.categorical_groups = {col: (categorical_groups[col] if col in categorical_groups.keys() else "other") for col in df.columns}
        self.past_impacts = past_impacts
        self.ds_min = df.index.min()
        self.ds_max = df.index.max()
        self.t_train = self.create_t(df.index)

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
            )
        )

        # if past impacts given, assume removal unless otherwise specified
        if self.past_impacts_intervention is None and past_impacts is not None:
            self.past_impacts_intervention = "remove"
        elif past_impacts is None:
            self.past_impacts_intervention = None
        # remove past impacts to find "organic"
        if self.past_impacts_intervention == "remove":
            self.df = self.df * (1 + past_impacts)
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
            self.multivariate_transformer = GeneralTransformer(**self.multivariate_transformation).fit(self.df.copy())

        # BEGIN CONSTRUCTION OF X ARRAY
        x_list = []
        if self.holiday_detector_params is not None:
            x_list.append(self.holiday_detector.dates_to_holidays(self.df.index, style="flag").clip(upper=1))
        if isinstance(self.anomaly_intervention, dict):
            # need to model these for prediction
            x_list.append(self.anomaly_detector.scores)
        # all of the following are 1 day past lagged
        if self.multivariate_feature is not None:
            if self.multivariate_feature == "feature_agglomeration":
                from sklearn.cluster import FeatureAgglomeration

                self.agglomerator = FeatureAgglomeration(n_clusters=10)
                self.agglomerator.fit_transform(self.df)[:-1]
            elif self.multivariate_feature == "group_average":
                df.groupby(self.categorical_groups, axis=1).mean()[:-1]
            elif self.multivariate_feature == "oscillator":
                return NotImplemented
                np.count_nonzero((df - df.shift(1)).clip(upper=0))[:-1]
        if self.seasonalities is not None:
            for seasonality in self.seasonalities:
                x_list.append(create_seasonality_feature(df.index, self.t_train, seasonality))
                # INTERACTIONS NOT IMPLEMENTED
                # ORDER SPECIFICATION NOT IMPLEMENTED
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
        if flag_regressors is not None and self.regressors_used:
            x_list.append(clean_regressor(flag_regressors, prefix="regrflags_"))
        if self.holiday_countries is not None and not isinstance(self.holiday_countries, dict) and self.holiday_countries_used:
            for holiday_country in self.holiday_countries:
                x_list.append(
                    holiday_flag(
                        self.df.index,
                        country=holiday_country,
                        encode_holiday_type=True,
                    ),  # may want to rename DF columns here
                )
        # regressor_per_series, AR lags, and holiday_countries (dict)
        if self.loop_required:
            return NotImplemented
        if self.past_impacts_intervention == "regressor":  # select only column
            return NotImplemented
        # RUN LINEAR MODEL
        x_array = pd.concat(x_list, axis=1)
        self.x_array = x_array
        # remove zero variance
        corr = np.corrcoef(x_array, rowvar=0)
        # remove colinear features
        w, vec = np.linalg.eig(corr)
        np.fill_diagonal(corr, 0)
        corel = x_array.columns[np.min(corr * np.tri(corr.shape[0]), axis=0) > 0.98]
        colin = x_array.columns[w < 0.005]
        if len(corel) > 0:
            print(f"Dropping colinear feature columns {corel}")
            x_array = x_array.drop(columns=corel)
        if len(colin) > 0:
            print(f"Dropping multi-colinear feature columns {colin}")
            x_array = x_array.drop(columns=colin)
        # things we want modeled but want to discard from evaluation (standins)
        remove_patterns = ["randnorm_", "rolling_trend_", "randomwalk_"]
        keep_cols = [col for col in x_array.columns if not any(remove_patterns in col)]
        self.keep_cols_idx = x_array.columns.get_indexer_for(keep_cols)
        x_array['intercept'] = 1
        # run model
        self.params = np.linalg.lstsq(x_array, df, rcond=None)[0]
        trend_residuals = df - np.dot(x_array[keep_cols], self.params[self.keep_cols_idx])
        self.trend_train = trend_residuals
        self.x_array = x_array
        if self.trend_anomaly_detector_params is not None or self.trend_window is not None:
            # rolling trend
            dates_2d = np.repeat(
                self.t_train[..., None],
                # df_holiday_scaled.index.to_julian_date().to_numpy()[..., None],
                df.shape[1], axis=1
            )
            wind = 30 if self.trend_window is None else self.trend_window
            w_1 = wind - 1
            steps_ahd = int(w_1 / 2)
            y0 = np.repeat(trend_residuals[0:1], steps_ahd, axis=0)
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
            if self.linear_model == 'something_else':
                # ADDING RECENCY WEIGHTING AND RIDGE PARAMS
                NotImplemented
            slope, intercept = window_lin_reg_mean(d, y2, wind)
            trend_posterior = slope * self.t_train[..., None] + intercept
            if self.trend_window is not None:
                self.trend_train = trend_posterior
        # INFLECTION POINTS (cross zero), CHANGEPOINTS (trend of trend changes, on bigger window)
        if self.trend_anomaly_detector_params is not None:
            self.trend_anomaly_detector = AnomalyRemoval(**self.trend_anomaly_detector_params)
            # DIFF the length of W (or w-1?)
            shft_idx = np.concatenate([[0], np.arange(len(slope))])[0:len(slope)]
            slope_diff = slope - slope[shft_idx]
            shft_idx = np.concatenate([np.repeat([0], wind), np.arange(len(slope))])[0:len(slope)]
            slope_diff = slope_diff + slope_diff[shft_idx]

            # np.cumsum(slope, axis=0)
            # pd.DataFrame(slope).rolling(90, center=True, min_periods=2).mean()
            # pd.DataFrame(slope, index=df.index).rolling(365, center=True, min_periods=2).mean()[0].plot()
            self.trend_anomaly_detector.fit(
                pd.DataFrame((slope - slope[shft_idx]), index=self.df.index)
            )
        # option to run trend model on full residuals or on rolling trend

        return self

    def predict(self, forecast_length, future_regressor, regressor_per_series, flag_regressors, future_impacts, new_df=None):
        # if future regressors are None (& USED), but were provided for history, instead use forecasts of these features (warn)
        if forecast_length is None:
            # use historic data
            # don't forget to add in past_impacts (use future impacts again?)
            NotImplemented
        # forecast trend
        df_forecast = model_forecast(
            model_name=self.trend_model['Model'],
            model_param_dict=self.trend_model['ModelParameters'],
            model_transform_dict=self.trend_transformation,
            df_train=self.trend_residuals,
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
            current_model_file=None,
        )
        # phi is on future predict step only
        if self.trend_phi is not None and self.trend_phi != 1:
            xxxx = NotImplemented
            temp = xxxx.mul(
                pd.Series([self.phi] * xxxx.shape[0], index=xxxx.index).pow(
                    range(xxxx.shape[0])
                ),
                axis=0,
            )
            xxxx = xxxx + temp
        return NotImplemented

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
        anomaly_intervention = random.choices([None, 'remove', 'detect_only', 'model'], [0.9, 0.3, 0.1, 0.3])[0]
        if anomaly_intervention is not None:
            anomaly_detector_params = AnomalyRemoval.get_new_params(method=method)
            if anomaly_intervention == "model":
                anomaly_intervention = general_template.sample(1).to_dict("records")[0]  # placeholder, probably
        else:
            anomaly_detector_params = None
        model_str = random.choices(['AverageValueNaive', 'UnivariateMotif'], [0.5, 0.4], k=1)[0]
        trend_model = {'Model': model_str}
        trend_model['ModelParameters'] = ModelMonster(model_str).get_new_params(method=method)

        trend_anomaly_intervention = random.choices([None, 'remove', 'detect_only', 'model'], [0.9, 0.3, 0.1, 0.3])[0]
        if trend_anomaly_intervention is not None:
            trend_anomaly_detector_params = AnomalyRemoval.get_new_params(method=method)
            if trend_anomaly_intervention == "model":  # Probably will NOT allow modeling for trend anomalies
                trend_anomaly_intervention = general_template.sample(1).to_dict("records")[0]  # placeholder, probably
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
            "ar_lags": None,  # NotImplemented
            "ar_interaction_seasonality": NotImplemented,
            "anomaly_detector_params": anomaly_detector_params,
            "anomaly_intervention": anomaly_intervention,
            "holiday_detector_params": holiday_params,
            # "holiday_countries": self.holiday_countries,
            "holiday_countries_used": random.choices([True, False], [0.5, 0.5])[0],
            "multivariate_feature": random.choices(
                [None, "feature_agglomeration", 'group_average', 'oscillator'],
                [0.7, 0.1, 0.1, 0.1]
            )[0],
            "multivariate_transformation": RandomTransform(
                transformer_list="fast", transformer_max_depth=3  # probably want some more usable defaults first as many random are senseless
            ),
            "regressor_transformation": RandomTransform(
                transformer_list="fast", transformer_max_depth=3  # probably want some more usable defaults first as many random are senseless
            ),
            "regressors_used": random.choices([True, False], [0.5, 0.5])[0],
            "linear_model": 'lstsq',
            "randomwalk_n": random.choices([None, 10], [0.5, 0.5])[0],
            "trend_window": random.choices([3, 15, 90, 365], [0.2, 0.2, 0.2, 0.2])[0],
            "trend_standin": random.choices(
                [None, 'random_normal', 'rolling_trend'],
                [0.5, 0.4, 0.1],
            )[0],
            "trend_anomaly_detector_params": trend_anomaly_detector_params,
            "trend_anomaly_intervention": trend_anomaly_intervention,
            "trend_transformation": RandomTransform(
                transformer_list="fast", transformer_max_depth=3  # probably want some more usable defaults first as many random are senseless
            ),
            "trend_model": trend_model,
            "trend_phi": random.choices([None, 0.98], [0.9, 0.1])[0],
            # "constraints": self.constraints,
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
            "trend_anomaly_intervention": self.trend_anomaly_intervention,
            "trend_transformation": self.trend_transformation,
            "trend_model": self.trend_model,
            "trend_phi": self.trend_phi,
            "constraints": self.constraints,
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
        fourier_series(t, seasonality, n=10)
    # dateparts
    elif seasonality == "dayofweek":
        return pd.get_dummies(pd.Categorical(
            DTindex.weekday, categories=list(range(7)), ordered=True
        )).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "month":
        return pd.get_dummies(pd.Categorical(
            DTindex.month, categories=list(range(12)), ordered=True
        )).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "weekend":
        return pd.DataFrame((DTindex.weekday > 4).astype(int), columns=["weekend"])
    elif seasonality == "weekdayofmonth":
        return pd.get_dummies(pd.Categorical(
            (DTindex.day - 1) // 7 + 1,
            categories=list(range(5)), ordered=True,
        )).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "hour":
        return pd.get_dummies(pd.Categorical(
            DTindex.hour, categories=list(range(24)), ordered=True
        )).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "daysinmonth":
        return pd.DataFrame({'daysinmonth': DTindex.daysinmonth})
    elif seasonality == "quarter":
        return pd.get_dummies(pd.Categorical(
            DTindex.quarter, categories=list(range(4)), ordered=True
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


def window_sum_nan_mean(x, w, axis=0):
    return np.nanmean(sliding_window_view(x, w, axis=axis), axis=-1)


def window_lin_reg_mean(x, y, w):
    '''From https://stackoverflow.com/questions/70296498/efficient-computation-of-moving-linear-regression-with-numpy-numba/70304475#70304475'''
    sx = window_sum_nan_mean(x, w)
    sy = window_sum_nan_mean(y, w)
    sx2 = window_sum_nan_mean(x**2, w)
    sxy = window_sum_nan_mean(x * y, w)
    slope = (sxy - sx * sy) / (sx2 - sx**2)
    intercept = (sy - slope * sx)
    return slope, intercept


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


mod = Cassandra(**Cassandra.get_new_params())
mod.fit(df_holiday)

# Automation
# allow some config inputs, or automated fit
# output to table
# compare coefficients change over time, accuracy over time
# comparing different sources? and/or allowing one forecast to be used as a regressor for another
# would allow for earlier detection of broken connections
