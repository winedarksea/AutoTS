"""
Naives and Others Requiring No Additional Packages Beyond Numpy and Pandas
"""

from math import ceil
import warnings
import random
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.seasonal import (
    seasonal_int,
    seasonal_window_match,
    seasonal_independent_match,
    date_part,
    base_seasonalities,
    create_changepoint_features,
    changepoint_fcst_from_last_row,
    random_datepart,
    half_yr_spacing,
)
from autots.tools.probabilistic import Point_to_Probability, historic_quantile
from autots.tools.window_functions import (
    window_id_maker,
    sliding_window_view,
    chunk_reshape,
)
from autots.tools.percentile import nan_quantile, trimmed_mean
from autots.tools.fast_kalman import KalmanFilter, new_kalman_params
from autots.tools.transform import (
    GeneralTransformer,
    RandomTransform,
    superfast_transformer_dict,
    StandardScaler,
    EmptyTransformer,
)
from autots.tools.fft import fourier_extrapolation
from autots.tools.impute import FillNA
from autots.evaluator.metrics import wasserstein
from autots.models.sklearn import rolling_x_regressor_regressor


# these are all optional packages
try:
    from scipy.spatial.distance import cdist
    from scipy.stats import norm

    # from scipy.stats import t as studentt
except Exception:
    pass
try:
    from joblib import Parallel, delayed

    joblib_present = True
except Exception:
    joblib_present = False
try:
    from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
except Exception:
    pass


class ConstantNaive(ModelObject):
    """Naive forecasting predicting a dataframe of zeroes (0's)

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        constant (float): value to fill with
    """

    def __init__(
        self,
        name: str = "ConstantNaive",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        constant: float = 0,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
        )
        self.constant = constant

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        df = pd.DataFrame(
            np.zeros((forecast_length, (self.train_shape[1]))) + self.constant,
            columns=self.column_names,
            index=self.create_forecast_index(forecast_length=forecast_length),
        )
        if just_point_forecast:
            return df
        else:
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=df.index,
                forecast_columns=df.columns,
                lower_forecast=df,
                forecast=df,
                upper_forecast=df,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning"""
        return {
            'constant': random.choices(
                [0, 1, -1, 0.1],
                [0.6, 0.25, 0.1, 0.05],
            )[0]
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {'constant': self.constant}


ZeroesNaive = ConstantNaive


class LastValueNaive(ModelObject):
    """Naive forecasting predicting a dataframe of the last series value

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """

    def __init__(
        self,
        name: str = "LastValueNaive",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
        )

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.last_values = df.tail(1).to_numpy()
        # self.df_train = df
        self.lower, self.upper = historic_quantile(
            df.iloc[-100:], prediction_interval=self.prediction_interval
        )
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        df = pd.DataFrame(
            np.tile(self.last_values, (forecast_length, 1)),
            columns=self.column_names,
            index=self.create_forecast_index(forecast_length=forecast_length),
        )
        if just_point_forecast:
            return df
        else:
            upper_forecast = df.astype(float) + (self.upper * 0.9)
            lower_forecast = df.astype(float) - (self.lower * 0.9)
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
        """Returns dict of new parameters for parameter tuning"""
        return {}

    def get_params(self):
        """Return dict of current parameters"""
        return {}


class AverageValueNaive(ModelObject):
    """Naive forecasting predicting a dataframe of the series' median values

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """

    def __init__(
        self,
        name: str = "AverageValueNaive",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        method: str = 'median',
        window: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
        )
        self.method = method
        self.window = window

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        method = str(self.method).lower()
        if self.window is not None:
            df_used = df[-self.window :]
        else:
            df_used = df

        if method == 'median':
            self.average_values = df_used.median(axis=0).to_numpy()
        elif method == 'mean':
            self.average_values = df_used.mean(axis=0).to_numpy()
        elif method == 'mode':
            self.average_values = (
                df_used.mode(axis=0).iloc[0].fillna(df_used.median(axis=0)).to_numpy()
            )
        elif method == "midhinge":
            results = df_used.to_numpy()
            q1 = nan_quantile(results, q=0.25, axis=0)
            q2 = nan_quantile(results, q=0.75, axis=0)
            self.average_values = (q1 + q2) / 2
        elif method in ["weighted_mean", "exp_weighted_mean"]:
            weights = pd.to_numeric(df_used.index)
            weights = weights - weights.min()
            if method == "exp_weighted_mean":
                weights = (weights / weights[weights != 0].min()) ** 2
            self.average_values = np.average(
                df_used.to_numpy(), axis=0, weights=weights
            )
        elif method == "trimmed_mean_20":
            self.average_values = trimmed_mean(df_used, percent=0.2, axis=0)
        elif method == "trimmed_mean_40":
            self.average_values = trimmed_mean(df_used, percent=0.4, axis=0)
        else:
            raise ValueError(f"method {method} not recognized")
        self.fit_runtime = datetime.datetime.now() - self.startTime
        self.lower, self.upper = historic_quantile(
            df_used, prediction_interval=self.prediction_interval
        )
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        df = pd.DataFrame(
            np.tile(self.average_values, (forecast_length, 1)),
            columns=self.column_names,
            index=self.create_forecast_index(forecast_length=forecast_length),
        )
        if just_point_forecast:
            return df
        else:
            upper_forecast = df.astype(float) + self.upper
            lower_forecast = df.astype(float) - self.lower
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
        """Returns dict of new parameters for parameter tuning"""
        method_choice = random.choices(
            [
                "Mean",
                "Median",
                "Mode",
                "Midhinge",
                "Weighted_Mean",
                "Exp_Weighted_Mean",
                "trimmed_mean_20",
                "trimmed_mean_40",
            ],
            [0.3, 0.3, 0.01, 0.1, 0.4, 0.1, 0.05, 0.05],
        )[0]

        return {
            'method': method_choice,
            'window': random.choices([None, seasonal_int()], [0.8, 0.2])[0],
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'method': self.method,
            'window': self.window,
        }


class SeasonalNaive(ModelObject):
    """Naive forecasting predicting a dataframe with seasonal (lag) forecasts.

    Concerto No. 2 in G minor, Op. 8, RV 315

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        method (str): Either 'LastValue' (use last value of lag n) or 'Mean' (avg of all lag n)
        lag_1 (int): The lag of the seasonality, should int > 1.
        lag_2 (int): Optional second lag of seasonality which is averaged with first lag to produce forecast.

    """

    def __init__(
        self,
        name: str = "SeasonalNaive",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        lag_1: int = 7,
        lag_2: int = None,
        method: str = 'lastvalue',
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
        )
        self.lag_1 = abs(int(lag_1))
        self.lag_2 = lag_2
        if str(self.lag_2).isdigit():
            self.lag_2 = abs(int(self.lag_2))
            if str(self.lag_2) == str(self.lag_1):
                self.lag_2 = 1
        self.method = str(method).lower()

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        df_length = self.train_shape[0]
        self.tile_values_lag_2 = None
        if self.method in ['mean', 'median']:
            tile_index = np.tile(
                np.arange(self.lag_1), int(np.ceil(df_length / self.lag_1))
            )
            tile_index = tile_index[len(tile_index) - (df_length) :]
            df.index = tile_index
            if self.method == "median":
                self.tile_values_lag_1 = df.groupby(level=0).median()
            else:
                self.tile_values_lag_1 = df.groupby(level=0).mean()
            if str(self.lag_2).isdigit():
                if self.lag_2 == 1:
                    self.tile_values_lag_2 = df.tail(self.lag_2)
                else:
                    tile_index = np.tile(
                        np.arange(self.lag_2), int(np.ceil(df_length / self.lag_2))
                    )
                    tile_index = tile_index[len(tile_index) - (df_length) :]
                    df.index = tile_index
                    if self.method == "median":
                        self.tile_values_lag_2 = df.groupby(level=0).median()
                    else:
                        self.tile_values_lag_2 = df.groupby(level=0).mean()
        else:
            self.method == 'lastvalue'
            self.tile_values_lag_1 = df.tail(self.lag_1)
            if str(self.lag_2).isdigit():
                self.tile_values_lag_2 = df.tail(self.lag_2)
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
        tile_len = len(self.tile_values_lag_1.index)
        df = pd.DataFrame(
            np.tile(
                self.tile_values_lag_1, (int(np.ceil(forecast_length / tile_len)), 1)
            )[0:forecast_length],
            columns=self.column_names,
            index=self.create_forecast_index(forecast_length=forecast_length),
        )
        if str(self.lag_2).isdigit():
            y = pd.DataFrame(
                np.tile(
                    self.tile_values_lag_2,
                    (
                        int(
                            np.ceil(forecast_length / len(self.tile_values_lag_2.index))
                        ),
                        1,
                    ),
                )[0:forecast_length],
                columns=self.column_names,
                index=self.create_forecast_index(forecast_length=forecast_length),
            )
            df = (df + y) / 2
        # df = df.apply(pd.to_numeric, errors='coerce')
        df = df.astype(float)
        if just_point_forecast:
            return df
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                df,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
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
        lag_1_choice = seasonal_int()
        lag_2_choice = random.choices(
            [None, seasonal_int(include_one=True)], [0.3, 0.7]
        )[0]
        if str(lag_2_choice) == str(lag_1_choice):
            lag_2_choice = 1
        method_choice = random.choices(
            ['mean', 'median', 'lastvalue'], [0.4, 0.2, 0.4]
        )[0]
        return {'method': method_choice, 'lag_1': lag_1_choice, 'lag_2': lag_2_choice}

    def get_params(self):
        """Return dict of current parameters."""
        return {'method': self.method, 'lag_1': self.lag_1, 'lag_2': self.lag_2}


class MotifSimulation(ModelObject):
    """More dark magic created by the evil mastermind of this project.
    Basically a highly-customized KNN

    Warning: if you are forecasting many steps (large forecast_length), and interested in probabilistic upper/lower forecasts, then set recency_weighting <= 0, and have a larger cutoff_minimum

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        phrase_len (int): length of motif vectors to compare as samples
        comparison (str): method to process data before comparison, 'magnitude' is original data
        shared (bool): whether to compare motifs across all series together, or separately
        distance_metric (str): passed through to sklearn pairwise_distances
        max_motifs (float): number of motifs to compare per series. If less 1, used as % of length training data
        recency_weighting (float): amount to the value of more recent data.
        cutoff_threshold (float): lowest value of distance metric to allow into forecast
        cutoff_minimum (int): minimum number of motif vectors to include in forecast.
        point_method (str): summarization method to choose forecast on, 'sample', 'mean', 'sign_biased_mean', 'median'
    """

    def __init__(
        self,
        name: str = "MotifSimulation",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        phrase_len: str = '5',
        comparison: str = 'magnitude_pct_change_sign',
        shared: bool = False,
        distance_metric: str = 'l2',
        max_motifs: float = 50,
        recency_weighting: float = 0.1,
        cutoff_threshold: float = 0.9,
        cutoff_minimum: int = 20,
        point_method: str = 'median',
        n_jobs: int = -1,
        verbose: int = 1,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            n_jobs=n_jobs,
        )
        self.phrase_len = phrase_len
        self.comparison = comparison
        self.shared = shared
        self.distance_metric = distance_metric
        self.max_motifs = max_motifs
        self.recency_weighting = recency_weighting
        self.cutoff_threshold = cutoff_threshold
        self.cutoff_minimum = cutoff_minimum
        self.point_method = point_method

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        if abs(float(self.max_motifs)) > 1:
            self.max_motifs_n = abs(int(self.max_motifs))
        elif float(self.max_motifs) == 1:
            self.max_motifs_n = 2
        elif float(self.max_motifs) < 1:
            self.max_motifs_n = int(abs(np.floor(self.max_motifs * df.shape[0])))
        else:
            self.max_motifs = 10
            self.max_motifs_n = 10

        self.phrase_n = int(self.phrase_len)

        # df = df_wide[df_wide.columns[0:3]].fillna(0).astype(float)
        df = self.basic_profile(df)
        """
        comparison = 'magnitude' # pct_change, pct_change_sign, magnitude_pct_change_sign, magnitude, magnitude_pct_change
        distance_metric = 'cityblock'
        max_motifs_n = 100
        phrase_n = 5
        shared = False
        recency_weighting = 0.1
        # cutoff_threshold = 0.8
        cutoff_minimum = 20
        prediction_interval = 0.9
        na_threshold = 0.1
        point_method = 'mean'
        """
        phrase_n = abs(int(self.phrase_n))
        max_motifs_n = abs(int(self.max_motifs_n))
        comparison = self.comparison
        distance_metric = self.distance_metric
        shared = self.shared
        recency_weighting = float(self.recency_weighting)
        # cutoff_threshold = float(self.cutoff_threshold)
        cutoff_minimum = abs(int(self.cutoff_minimum))
        prediction_interval = float(self.prediction_interval)
        na_threshold = 0.1
        point_method = self.point_method

        parallel = True
        if self.n_jobs in [0, 1] or df.shape[1] < 3:
            parallel = False
        else:
            if not joblib_present:
                parallel = False

        # start_time_1st = timeit.default_timer()
        # transform the data into different views (contour = percent_change)
        original_df = None
        if 'pct_change' in comparison:
            if comparison in ['magnitude_pct_change', 'magnitude_pct_change_sign']:
                original_df = df.copy()
            df = df.replace([0], np.nan)
            df = df.fillna(abs(df[df != 0]).min()).fillna(0.1)
            last_row = df.tail(1)
            df = df.ffill().pct_change(periods=1).tail(df.shape[0] - 1).fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
        # else:
        # self.comparison = 'magnitude'

        if 'pct_change_sign' in comparison:
            last_motif = df.where(df >= 0, -1).where(df <= 0, 1).tail(phrase_n)
        else:
            last_motif = df.tail(phrase_n)

        max_samps = df.shape[0] - phrase_n
        numbers = np.random.choice(
            max_samps,
            size=max_motifs_n if max_motifs_n < max_samps else max_samps,
            replace=False,
        )

        # make this faster
        motif_vecs_list = []
        # takes random slices of the time series and rearranges as phrase_n length vectors
        for z in numbers:
            rand_slice = df.iloc[z : (z + phrase_n),]
            rand_slice = (
                rand_slice.reset_index(drop=True)
                .transpose()
                .set_index(np.repeat(z, (df.shape[1],)), append=True)
            )
            # motif_vecs = pd.concat([motif_vecs, rand_slice], axis=0)
            motif_vecs_list.append(rand_slice)
        motif_vecs = pd.concat(motif_vecs_list, axis=0)

        if 'pct_change_sign' in comparison:
            motif_vecs = motif_vecs.where(motif_vecs >= 0, -1).where(motif_vecs <= 0, 1)
        # elapsed_1st = timeit.default_timer() - start_time_1st
        # start_time_2nd = timeit.default_timer()
        # compare the motif vectors to the most recent vector of the series
        args = {
            "cutoff_minimum": cutoff_minimum,
            "comparison": comparison,
            "point_method": point_method,
            "prediction_interval": prediction_interval,
            "phrase_n": phrase_n,
            "distance_metric": distance_metric,
            "shared": shared,
            "na_threshold": na_threshold,
            "original_df": original_df,
            "df": df,
        }

        if shared:
            comparative = pd.DataFrame(
                pairwise_distances(
                    motif_vecs.to_numpy(),
                    last_motif.transpose().to_numpy(),
                    metric=distance_metric,
                )
            )
            comparative.index = motif_vecs.index
            comparative.columns = last_motif.columns
        if not shared:

            def create_comparative(motif_vecs, last_motif, args, col):
                distance_metric = args["distance_metric"]
                x = motif_vecs[motif_vecs.index.get_level_values(0) == col]
                y = last_motif[col].to_numpy().reshape(1, -1)
                current_comparative = pd.DataFrame(
                    pairwise_distances(x.to_numpy(), y, metric=distance_metric)
                )
                current_comparative.index = x.index
                current_comparative.columns = [col]
                return current_comparative

            if parallel:
                verbs = 0 if self.verbose < 1 else self.verbose - 1
                df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                    delayed(create_comparative)(
                        motif_vecs=motif_vecs, last_motif=last_motif, args=args, col=col
                    )
                    for col in last_motif.columns
                )
            else:
                df_list = []
                for col in last_motif.columns:
                    df_list.append(
                        create_comparative(motif_vecs, last_motif, args, col)
                    )
            comparative = pd.concat(df_list, axis=0)
            comparative = comparative.groupby(level=[0, 1]).sum(min_count=0)

            # comparative comes out of this looking kinda funny, but get_level_values works with that later
            # it might be possible to reshape it to a more memory efficient design

        # comparative is a df of motifs (in index) with their value to each series (per column)
        if recency_weighting != 0:
            rec_weights = np.repeat(
                ((comparative.index.get_level_values(1)) / df.shape[0])
                .to_numpy()
                .reshape(-1, 1)
                * recency_weighting,
                len(comparative.columns),
                axis=1,
            )
            comparative = comparative.add(rec_weights, fill_value=0)

        def seek_the_oracle(comparative, args, col):
            # comparative.idxmax()
            cutoff_minimum = args["cutoff_minimum"]
            comparison = args["comparison"]
            point_method = args["point_method"]
            prediction_interval = args["prediction_interval"]
            phrase_n = args["phrase_n"]
            shared = args["shared"]
            na_threshold = args["na_threshold"]
            original_df = args["original_df"]
            df = args["df"]

            vals = comparative[col].sort_values(ascending=False)
            if not shared:
                vals = vals[vals.index.get_level_values(0) == col]
            # vals = vals[vals > cutoff_threshold]
            # if vals.shape[0] < cutoff_minimum:
            vals = comparative[col].sort_values(ascending=False)
            if not shared:
                vals = vals[vals.index.get_level_values(0) == col]
            vals = vals.head(cutoff_minimum)

            pos_forecasts = pd.DataFrame()
            for val_index, val_value in vals.items():
                sec_start = val_index[1] + phrase_n
                if comparison in ['magnitude_pct_change', 'magnitude_pct_change_sign']:
                    current_pos = original_df[val_index[0]].iloc[sec_start + 1 :]
                else:
                    current_pos = df[val_index[0]].iloc[sec_start:]
                pos_forecasts = pd.concat(
                    [pos_forecasts, current_pos.reset_index(drop=True)],
                    axis=1,
                    sort=False,
                )

            thresh = int(np.ceil(pos_forecasts.shape[1] * na_threshold))
            if point_method == 'mean':
                current_forecast = pos_forecasts.mean(axis=1)
            elif point_method == 'sign_biased_mean':
                axis_means = pos_forecasts.mean(axis=0)
                if axis_means.mean() > 0:
                    pos_forecasts = pos_forecasts[
                        pos_forecasts.columns[~(axis_means < 0)]
                    ]
                else:
                    pos_forecasts = pos_forecasts[
                        pos_forecasts.columns[~(axis_means > 0)]
                    ]
                current_forecast = pos_forecasts.mean(axis=1)
            else:
                point_method = 'median'
                current_forecast = pos_forecasts.median(axis=1)
            # current_forecast.columns = [col]
            forecast = current_forecast.copy()
            forecast.name = col

            current_forecast = (
                pos_forecasts.dropna(thresh=thresh, axis=0)
                .quantile(q=[(1 - prediction_interval), prediction_interval], axis=1)
                .transpose()
            )

            lower_forecast = pd.Series(current_forecast.iloc[:, 0], name=col)
            upper_forecast = pd.Series(current_forecast.iloc[:, 1], name=col)
            return (forecast, lower_forecast, upper_forecast)

        # seek_the_oracle(comparative, args, comparative.columns[0])
        if parallel:
            verbs = 0 if self.verbose < 1 else self.verbose - 1
            df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                delayed(seek_the_oracle)(comparative=comparative, args=args, col=col)
                for col in comparative.columns
            )
            complete = list(map(list, zip(*df_list)))
        else:
            df_list = []
            for col in comparative.columns:
                df_list.append(seek_the_oracle(comparative, args, col))
            complete = list(map(list, zip(*df_list)))
        forecasts = pd.concat(
            complete[0], axis=1
        )  # .reindex(self.column_names, axis=1)
        lower_forecasts = pd.concat(complete[1], axis=1)
        upper_forecasts = pd.concat(complete[2], axis=1)

        if comparison in ['pct_change', 'pct_change_sign']:
            forecasts = (forecasts + 1).replace([0], np.nan)
            forecasts = forecasts.fillna(abs(df[df != 0]).min()).fillna(0.1)
            forecasts = pd.concat(
                [last_row.reset_index(drop=True), (forecasts)], axis=0, sort=False
            ).cumprod()
            upper_forecasts = (upper_forecasts + 1).replace([0], np.nan)
            upper_forecasts = upper_forecasts.fillna(abs(df[df != 0]).min()).fillna(0.1)
            upper_forecasts = pd.concat(
                [last_row.reset_index(drop=True), (upper_forecasts)], axis=0, sort=False
            ).cumprod()
            lower_forecasts = (lower_forecasts + 1).replace([0], np.nan)
            lower_forecasts = lower_forecasts.fillna(abs(df[df != 0]).min()).fillna(0.1)
            lower_forecasts = pd.concat(
                [last_row.reset_index(drop=True), (lower_forecasts)], axis=0, sort=False
            ).cumprod()

        # reindex might be unnecessary but I assume the cost worth the safety
        self.forecasts = forecasts
        self.lower_forecasts = lower_forecasts
        self.upper_forecasts = upper_forecasts

        # elapsed_3rd = timeit.default_timer() - start_time_3rd
        # print(f"1st {elapsed_1st}\n2nd {elapsed_2nd}\n3rd {elapsed_3rd}")
        """
        In fit phase, only select motifs.
            table: start index, weight, column it applies to, and count of rows that follow motif
            slice into possible motifs
            compare motifs (efficiently)
            choose the motifs to use for each series
                if not shared, can drop column part of index ref
            combine the following values into forecasts
                consider the weights
                magnitude and percentage change
            account for forecasts not running the full length of forecast_length
                if longer than comparative, append na df then ffill

        Profile speed and which code to improve first
            Remove for loops
            Quantile not be calculated until after pos_forecasts narrowed down to only forecast length
        """
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()

        forecasts = self.forecasts.head(forecast_length)
        if forecasts.shape[0] < forecast_length:
            extra_len = forecast_length - forecasts.shape[0]
            empty_frame = pd.DataFrame(
                index=np.arange(extra_len), columns=forecasts.columns
            )
            forecasts = pd.concat([forecasts, empty_frame], axis=0, sort=False).ffill()
        forecasts.columns = self.column_names
        forecasts.index = self.create_forecast_index(forecast_length=forecast_length)

        if just_point_forecast:
            return forecasts
        else:
            lower_forecasts = self.lower_forecasts.head(forecast_length)
            upper_forecasts = self.upper_forecasts.head(forecast_length)
            if lower_forecasts.shape[0] < forecast_length:
                extra_len = forecast_length - lower_forecasts.shape[0]
                empty_frame = pd.DataFrame(
                    index=np.arange(extra_len), columns=lower_forecasts.columns
                )
                lower_forecasts = pd.concat(
                    [lower_forecasts, empty_frame], axis=0, sort=False
                ).ffill()
            lower_forecasts.columns = self.column_names
            lower_forecasts.index = self.create_forecast_index(
                forecast_length=forecast_length
            )

            if upper_forecasts.shape[0] < forecast_length:
                extra_len = forecast_length - upper_forecasts.shape[0]
                empty_frame = pd.DataFrame(
                    index=np.arange(extra_len), columns=upper_forecasts.columns
                )
                upper_forecasts = pd.concat(
                    [upper_forecasts, empty_frame], axis=0, sort=False
                ).ffill()
            upper_forecasts.columns = self.column_names
            upper_forecasts.index = self.create_forecast_index(
                forecast_length=forecast_length
            )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=forecasts.index,
                forecast_columns=forecasts.columns,
                lower_forecast=lower_forecasts,
                forecast=forecasts,
                upper_forecast=upper_forecasts,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        comparison_choice = random.choices(
            [
                'pct_change',
                'pct_change_sign',
                'magnitude_pct_change_sign',
                'magnitude',
                'magnitude_pct_change',
            ],
            [0.2, 0.1, 0.4, 0.2, 0.1],
        )[0]
        phrase_len_choice = random.choices(
            [5, 10, 15, 20, 30, 90, 360], [0.2, 0.2, 0.1, 0.25, 0.1, 0.1, 0.05]
        )[0]
        shared_choice = random.choices([True, False], [0.05, 0.95])[0]
        distance_metric_choice = random.choices(
            [
                'other',
                'hamming',
                'cityblock',
                'cosine',
                'euclidean',
                'l1',
                'l2',
                'manhattan',
            ],
            [0.44, 0.05, 0.1, 0.1, 0.1, 0.2, 0.0, 0.01],
        )[0]
        if distance_metric_choice == 'other':
            distance_metric_choice = random.choices(
                [
                    'braycurtis',
                    'canberra',
                    'chebyshev',
                    'correlation',
                    'dice',
                    'hamming',
                    'jaccard',
                    'kulczynski1',
                    'mahalanobis',
                    'minkowski',
                    'rogerstanimoto',
                    'russellrao',
                    # 'seuclidean',
                    'sokalmichener',
                    'sokalsneath',
                    'sqeuclidean',
                    'yule',
                ],
            )[0]
        max_motifs_choice = float(
            random.choices(
                [20, 50, 100, 200, 0.05, 0.2, 0.5],
                [0.4, 0.1, 0.2, 0.09, 0.1, 0.1, 0.01],
            )[0]
        )
        recency_weighting_choice = random.choices(
            [0, 0.5, 0.1, 0.01, -0.01, 0.001],
            [0.5, 0.02, 0.05, 0.35, 0.05, 0.03],
        )[0]
        # cutoff_threshold_choice = np.random.choice(
        #     a=[0.7, 0.9, 0.99, 1.5], size=1, p=[0.1, 0.1, 0.4, 0.4]
        # ).item()
        cutoff_minimum_choice = random.choices(
            [5, 10, 20, 50, 100, 200, 500], [0, 0, 0.2, 0.2, 0.4, 0.1, 0.1]
        )[0]
        point_method_choice = random.choices(
            ['median', 'mean', 'sign_biased_mean'],
            [0.59, 0.3, 0.1],
        )[0]

        return {
            'phrase_len': phrase_len_choice,
            'comparison': comparison_choice,
            'shared': shared_choice,
            'distance_metric': distance_metric_choice,
            'max_motifs': max_motifs_choice,
            'recency_weighting': recency_weighting_choice,
            'cutoff_minimum': cutoff_minimum_choice,
            'point_method': point_method_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'phrase_len': self.phrase_len,
            'comparison': self.comparison,
            'shared': self.shared,
            'distance_metric': self.distance_metric,
            'max_motifs': self.max_motifs,
            'recency_weighting': self.recency_weighting,
            # 'cutoff_threshold': self.cutoff_threshold,
            'cutoff_minimum': self.cutoff_minimum,
            'point_method': self.point_method,
        }


def looped_motif(
    Xa,
    Xb,
    name,
    r_arr=None,
    window=10,
    distance_metric="minkowski",
    k=10,
    point_method="mean",
    prediction_interval=0.9,
    return_result_windows=False,
):
    """inner function for Motif model."""
    if r_arr is None:
        y = Xa[:, window:]
        Xa = Xa[:, :window]
    else:
        y = Xa[r_arr, window:]
        Xa = Xa[r_arr, :window]

    # model = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='minkowski', n_jobs=1)
    # model.fit(Xa)
    # model.kneighbors(Xb)

    if distance_metric == "kdtree":
        from scipy.spatial import KDTree

        # Build a KDTree for Xb
        tree = KDTree(Xa, leafsize=14)
        # Query the KDTree to find k nearest neighbors for each point in Xa
        A, idx = tree.query(Xb, k=k)
        idx = idx.flatten()
    else:
        A = cdist(Xa, Xb, metric=distance_metric)
        # lowest values
        if point_method == "closest":
            idx = np.argsort(A, axis=0)[:k].flatten()
        else:
            if k > A.shape[0]:
                print("k too large for size of data in motif")
                k = A.shape[0]
            idx = np.argpartition(A, k, axis=0)[:k].flatten()
    # distances for weighted mean
    results = y[idx]
    if point_method == "weighted_mean":
        weights = A[idx].flatten()
        if weights.sum() == 0:
            weights = None
        forecast = np.average(results, axis=0, weights=weights)
    elif point_method == "mean":
        forecast = np.nanmean(results, axis=0)
    elif point_method == "median":
        forecast = np.nanmedian(results, axis=0)
    elif point_method == "midhinge":
        q1 = nan_quantile(results, q=0.25, axis=0)
        q2 = nan_quantile(results, q=0.75, axis=0)
        forecast = (q1 + q2) / 2
    elif point_method == "closest":
        forecast = results[0]
    else:
        raise ValueError(f"distance_metric {distance_metric} not recognized")

    pred_int = (1 - prediction_interval) / 2
    upper_forecast = nan_quantile(results, q=(1 - pred_int), axis=0)
    lower_forecast = nan_quantile(results, q=pred_int, axis=0)
    forecast = pd.Series(forecast, name=name)
    upper_forecast = pd.Series(upper_forecast, name=name)
    lower_forecast = pd.Series(lower_forecast, name=name)
    if return_result_windows:
        return (forecast, upper_forecast, lower_forecast, results)
    else:
        return (forecast, upper_forecast, lower_forecast)


class Motif(ModelObject):
    """Forecasts using a nearest neighbors type model adapted for probabilistic time series.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        n_jobs (int): how many parallel processes to run
        random_seed (int): used in selecting windows if max_windows is less than total available
        window (int): length of forecast history to match on
        point_method (int): how to summarize the nearest neighbors to generate the point forecast
            "weighted_mean", "mean", "median", "midhinge"
        distance_metric (str): all valid values for scipy cdist
        k (int): number of closest neighbors to consider
        max_windows (int): max number of windows to consider (a speed/accuracy tradeoff)
        multivariate (bool): if True, utilizes matches from all provided series for each series forecast. Else just own history of series.
        return_result_windows (bool): if True, result windows (all motifs gathered for forecast) will be saved in dict to result_windows attribute
    """

    def __init__(
        self,
        name: str = "Motif",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = 1,
        window: int = 5,
        point_method: str = "weighted_mean",
        distance_metric: str = "minkowski",
        k: int = 10,
        max_windows: int = 5000,
        multivariate: bool = False,
        return_result_windows: bool = False,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            "MultivariateMotif" if multivariate else "UnivariateMotif",
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.window = window
        self.point_method = point_method
        self.distance_metric = distance_metric
        self.k = k
        self.max_windows = max_windows
        self.multivariate = multivariate
        self.return_result_windows = return_result_windows

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        # keep this at top so it breaks quickly if missing version
        x = sliding_window_view(
            self.df.to_numpy(), self.window + forecast_length, axis=0
        )
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        # subsample windows if needed
        r_arr = None
        if self.max_windows is not None:
            if self.multivariate:
                X_size = x.shape[0] * x.shape[1]
            else:
                X_size = x.shape[0]
            if self.max_windows < X_size:
                r_arr = np.random.default_rng(self.random_seed).integers(
                    0, X_size, size=self.max_windows
                )

        self.parallel = True
        if self.n_jobs in [0, 1] or self.df.shape[1] < 5:
            self.parallel = False
        else:
            if not joblib_present:
                self.parallel = False

        # joblib multiprocessing to loop through series
        if self.parallel:
            df_list = Parallel(n_jobs=(self.n_jobs - 1))(
                delayed(looped_motif)(
                    Xa=x.reshape(-1, x.shape[-1]) if self.multivariate else x[:, i],
                    Xb=self.df.iloc[-self.window :, i].to_numpy().reshape(1, -1),
                    name=self.df.columns[i],
                    r_arr=r_arr,
                    window=self.window,
                    distance_metric=self.distance_metric,
                    k=self.k,
                    point_method=self.point_method,
                    prediction_interval=self.prediction_interval,
                    return_result_windows=self.return_result_windows,
                )
                for i in range(self.df.shape[1])
            )
        else:
            df_list = []
            for i in range(self.df.shape[1]):
                df_list.append(
                    looped_motif(
                        Xa=x.reshape(-1, x.shape[-1]) if self.multivariate else x[:, i],
                        Xb=self.df.iloc[-self.window :, i].to_numpy().reshape(1, -1),
                        name=self.df.columns[i],
                        r_arr=r_arr,
                        window=self.window,
                        distance_metric=self.distance_metric,
                        k=self.k,
                        point_method=self.point_method,
                        prediction_interval=self.prediction_interval,
                        return_result_windows=self.return_result_windows,
                    )
                )
        complete = list(map(list, zip(*df_list)))
        forecast = pd.concat(complete[0], axis=1)
        forecast.index = test_index
        # these were swapped in an earlier version, huge mistake!
        lower_forecast = pd.concat(complete[2], axis=1)
        lower_forecast.index = test_index
        upper_forecast = pd.concat(complete[1], axis=1)
        upper_forecast.index = test_index
        if self.return_result_windows:
            self.result_windows = dict(zip(forecast.columns, complete[3]))
        if just_point_forecast:
            return forecast
        else:
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
        """Returns dict of new parameters for parameter tuning"""
        metric_list = [
            'braycurtis',
            'canberra',
            'chebyshev',
            'cityblock',
            'correlation',
            'cosine',
            'dice',
            'euclidean',
            'hamming',
            'jaccard',
            'jensenshannon',
            'mahalanobis',
            'matching',
            'minkowski',
            'rogerstanimoto',
            'russellrao',
            # 'seuclidean',
            'sokalmichener',
            'sokalsneath',
            'sqeuclidean',
            'yule',
            'kdtree',
        ]
        if method == "event_risk":
            k_choice = random.choices(
                [10, 15, 20, 50, 100], [0.3, 0.1, 0.1, 0.05, 0.05]
            )[0]
        else:
            k_choice = random.choices(
                [1, 3, 5, 10, 15, 20, 100], [0.02, 0.2, 0.2, 0.5, 0.1, 0.1, 0.1]
            )[0]
        return {
            "window": random.choices(
                [2, 3, 5, 7, 10, 14, 28, 60],
                [0.01, 0.01, 0.01, 0.1, 0.5, 0.1, 0.1, 0.01],
            )[0],
            "point_method": random.choices(
                ["weighted_mean", "mean", "median", "midhinge", "closest"],
                [0.4, 0.2, 0.2, 0.2, 0.2],
            )[0],
            "distance_metric": random.choice(metric_list),
            "k": k_choice,
            "max_windows": random.choices([None, 1000, 10000], [0.01, 0.1, 0.8])[0],
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {
            "window": self.window,
            "point_method": self.point_method,
            "distance_metric": self.distance_metric,
            "k": self.k,
            "max_windows": self.max_windows,
        }


def predict_reservoir(
    df,
    forecast_length,
    prediction_interval=None,
    warmup_pts=1,
    k=2,
    ridge_param=2.5e-6,
    seed_pts: int = 1,
    seed_weighted: str = None,
):
    """Nonlinear Variable Autoregression or 'Next-Generation Reservoir Computing'

    based on https://github.com/quantinfo/ng-rc-paper-code/
    Gauthier, D.J., Bollt, E., Griffith, A. et al. Next generation reservoir computing. Nat Commun 12, 5564 (2021).
    https://doi.org/10.1038/s41467-021-25801-2
    with adjustments to make it probabilistic

    This is very slow and memory hungry when n series/dimensions gets big (ie > 50).
    Already effectively parallelized by linpack it seems.
    It's very sensitive to error in most recent data point!
    The seed_pts and seed_weighted can help address that.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        k (int): the AR order (keep this small, larger is slow and usually pointless)
        ridge_param (float): standard lambda for ridge regression
        warmup_pts (int): in reality, passing 1 here (no warmup) is fine
        seed_pts (int): number of back steps to use to simulate future
            if > 10, also increases search space of probabilistic forecast
        seed_weighted (str): how to summarize most recent points if seed_pts > 1
    """
    assert k > 0, "nvar `k` must be > 0"
    assert warmup_pts > 0, "nvar `warmup_pts` must be > 0"
    assert df.shape[1] > k, "nvar input data must contain at least k+1 records"

    n_pts = df.shape[1]
    # handle short data edge case
    min_train_pts = 12
    max_warmup_pts = n_pts - min_train_pts
    if warmup_pts >= max_warmup_pts:
        warmup_pts = max_warmup_pts if max_warmup_pts > 0 else 1

    traintime_pts = n_pts - warmup_pts  # round(traintime / dt)
    warmtrain_pts = warmup_pts + traintime_pts
    testtime_pts = forecast_length + 1  # round(testtime / dt)
    maxtime_pts = n_pts  # round(maxtime / dt)

    # input dimension
    d = df.shape[0]
    # size of the linear part of the feature vector
    dlin = k * d
    # size of nonlinear part of feature vector
    dnonlin = int(dlin * (dlin + 1) / 2)
    # total size of feature vector: constant + linear + nonlinear
    dtot = 1 + dlin + dnonlin

    # create an array to hold the linear part of the feature vector
    x = np.zeros((dlin, maxtime_pts))

    # fill in the linear part of the feature vector for all times
    for delay in range(k):
        for j in range(delay, maxtime_pts):
            x[d * delay : d * (delay + 1), j] = df[:, j - delay]

    # create an array to hold the full feature vector for training time
    # (use ones so the constant term is already 1)
    out_train = np.ones((dtot, traintime_pts))

    # copy over the linear part (shift over by one to account for constant)
    out_train[1 : dlin + 1, :] = x[:, warmup_pts - 1 : warmtrain_pts - 1]

    # fill in the non-linear part
    cnt = 0
    for row in range(dlin):
        for column in range(row, dlin):
            # shift by one for constant
            out_train[dlin + 1 + cnt] = (
                x[row, warmup_pts - 1 : warmtrain_pts - 1]
                * x[column, warmup_pts - 1 : warmtrain_pts - 1]
            )
            cnt += 1

    # ridge regression: train W_out to map out_train to Lorenz[t] - Lorenz[t - 1]
    W_out = (
        (x[0:d, warmup_pts:warmtrain_pts] - x[0:d, warmup_pts - 1 : warmtrain_pts - 1])
        @ out_train[:, :].T
        @ np.linalg.pinv(
            out_train[:, :] @ out_train[:, :].T + ridge_param * np.identity(dtot)
        )
    )

    # create a place to store feature vectors for prediction
    out_test = np.ones(dtot)  # full feature vector
    x_test = np.zeros((dlin, testtime_pts))  # linear part

    # copy over initial linear feature vector
    x_test[:, 0] = x[:, warmtrain_pts - 1]

    # do prediction
    for j in range(testtime_pts - 1):
        # copy linear part into whole feature vector
        out_test[1 : dlin + 1] = x_test[:, j]  # shift by one for constant
        # fill in the non-linear part
        cnt = 0
        for row in range(dlin):
            for column in range(row, dlin):
                # shift by one for constant
                out_test[dlin + 1 + cnt] = x_test[row, j] * x_test[column, j]
                cnt += 1
        # fill in the delay taps of the next state
        x_test[d:dlin, j + 1] = x_test[0 : (dlin - d), j]
        # do a prediction
        x_test[0:d, j + 1] = x_test[0:d, j] + W_out @ out_test[:]
    pred = x_test[0:d, 1:]

    if prediction_interval is not None or seed_pts > 1:
        # this works by using n most recent points as different starting "seeds"
        # this has the advantage of generating perfect bounds on perfect functions
        # on real world data, well, things will be less perfect...
        n_samples = 10 if seed_pts <= 10 else seed_pts
        interval_list = []
        for ns in range(n_samples):
            out_test = np.ones(dtot)  # full feature vector
            x_int = np.zeros((dlin, testtime_pts + n_samples))  # linear part
            # copy over initial linear feature vector
            x_int[:, 0] = x[:, warmtrain_pts - 2 - ns]
            # do prediction
            for j in range(testtime_pts - 1 + n_samples):
                # copy linear part into whole feature vector
                out_test[1 : dlin + 1] = x_int[:, j]  # shift by one for constant
                # fill in the non-linear part
                cnt = 0
                for row in range(dlin):
                    for column in range(row, dlin):
                        # shift by one for constant
                        out_test[dlin + 1 + cnt] = x_int[row, j] * x_int[column, j]
                        cnt += 1
                # fill in the delay taps of the next state
                x_int[d:dlin, j + 1] = x_int[0 : (dlin - d), j]
                # do a prediction
                x_int[0:d, j + 1] = x_int[0:d, j] + W_out @ out_test[:]
            start_slice = ns + 2
            end_slice = start_slice + testtime_pts - 1
            interval_list.append(x_int[:, start_slice:end_slice])

        interval_list = np.array(interval_list)
        if seed_pts > 1:
            pred_int = np.concatenate(
                [np.expand_dims(x_test[:, 1:], axis=0), interval_list]
            )
            # assuming interval_list has more recent first
            if seed_weighted == "linear":
                pred = np.average(
                    pred_int, axis=0, weights=range(pred_int.shape[0], 0, -1)
                )[0:d]
            elif seed_weighted == "exponential":
                pred = np.average(
                    pred_int, axis=0, weights=np.geomspace(100, 1, pred_int.shape[0])
                )[0:d]
            else:
                pred = np.quantile(pred_int, q=0.5, axis=0)[0:d]
        pred_upper = nan_quantile(interval_list, q=prediction_interval, axis=0)[0:d]
        pred_upper = np.where(pred_upper < pred, pred, pred_upper)
        pred_lower = nan_quantile(interval_list, q=(1 - prediction_interval), axis=0)[
            0:d
        ]
        pred_lower = np.where(pred_lower > pred, pred, pred_lower)
        return pred, pred_upper, pred_lower
    else:
        return pred


class NVAR(ModelObject):
    """Nonlinear Variable Autoregression or 'Next-Generation Reservoir Computing'

    based on https://github.com/quantinfo/ng-rc-paper-code/
    Gauthier, D.J., Bollt, E., Griffith, A. et al. Next generation reservoir computing. Nat Commun 12, 5564 (2021).
    https://doi.org/10.1038/s41467-021-25801-2
    with adjustments to make it probabilistic and to scale better

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        k (int): the AR order (keep this small, larger is slow and usually pointless)
        ridge_param (float): standard lambda for ridge regression
        warmup_pts (int): in reality, passing 1 here (no warmup) is fine
        batch_size (int): nvar scales exponentially, to scale linearly, series are split into batches of size n
        batch_method (str): method for collecting series to make batches
    """

    def __init__(
        self,
        name: str = "NVAR",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        k: int = 1,
        ridge_param: float = 2.5e-6,
        warmup_pts: int = 1,
        seed_pts: int = 1,
        seed_weighted: str = None,
        batch_size: int = 5,
        batch_method: str = "input_order",
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
        )
        self.k = k
        self.ridge_param = ridge_param
        self.warmup_pts = warmup_pts
        self.seed_pts = seed_pts
        self.seed_weighted = seed_weighted
        self.batch_size = batch_size
        self.batch_method = batch_method

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        self.basic_profile(df)
        if self.batch_method == "med_sorted":
            df = df.loc[:, df.median().sort_values(ascending=False).index]
        elif self.batch_method == "std_sorted":
            df = df.loc[:, df.std().sort_values(ascending=False).index]
        self.new_col_names = df.columns
        self.batch_steps = ceil(df.shape[1] / self.batch_size)
        self.df_train = df.to_numpy().T
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        the_index = self.create_forecast_index(forecast_length=forecast_length)
        df_list, df_list_up, df_list_low = [], [], []
        # since already uses 100% cpu, no point parallelizing
        for stp in range(self.batch_steps):
            srt = stp * self.batch_size
            stop = srt + self.batch_size
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fore, up, low = predict_reservoir(
                    self.df_train[srt:stop],
                    forecast_length=forecast_length,
                    warmup_pts=self.warmup_pts,
                    seed_pts=self.seed_pts,
                    seed_weighted=self.seed_weighted,
                    k=self.k,
                    ridge_param=self.ridge_param,
                    prediction_interval=self.prediction_interval,
                )
            df_list_up.append(
                pd.DataFrame(
                    up.T,
                    columns=self.new_col_names[srt:stop],
                    index=the_index,
                )
            )
            df_list.append(
                pd.DataFrame(
                    fore.T,
                    columns=self.new_col_names[srt:stop],
                    index=the_index,
                )
            )
            df_list_low.append(
                pd.DataFrame(
                    low.T,
                    columns=self.new_col_names[srt:stop],
                    index=the_index,
                )
            )
        forecast = pd.concat(df_list, axis=1)[self.column_names]
        if just_point_forecast:
            return forecast
        else:
            upper_forecast = pd.concat(df_list_up, axis=1)[self.column_names]
            lower_forecast = pd.concat(df_list_low, axis=1)[self.column_names]
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
        """Returns dict of new parameters for parameter tuning"""
        k_choice = random.choices([1, 2, 3, 4, 5], [0.5, 0.2, 0.1, 0.001, 0.001])[0]
        ridge_choice = random.choices(
            [0, -1, -2, -3, -4, -5, -6, -7, -8],
            [0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1],
        )[0]
        ridge_choice = 2 * 10**ridge_choice
        warmup_pts_choice = random.choices([1, 50, 250], [0.9, 0.1, 0.1])[0]
        seed_pts_choice = random.choices([1, 10, 100], [0.8, 0.2, 0.01])[0]
        if seed_pts_choice > 1:
            seed_weighted_choice = random.choices(
                [None, "linear", "exponential"], [0.3, 0.3, 0.3]
            )[0]
        else:
            seed_weighted_choice = None
        batch_size_choice = random.choices([5, 10, 20, 30], [0.5, 0.2, 0.01, 0.001])[0]
        batch_method_choice = random.choices(
            ["input_order", "std_sorted", "max_sorted"], [0.5, 0.1, 0.1]
        )[0]
        return {
            'k': k_choice,
            'ridge_param': ridge_choice,
            'warmup_pts': warmup_pts_choice,
            'seed_pts': seed_pts_choice,
            'seed_weighted': seed_weighted_choice,
            'batch_size': batch_size_choice,
            'batch_method': batch_method_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'k': self.k,
            'ridge_param': self.ridge_param,
            'warmup_pts': self.warmup_pts,
            'seed_pts': self.seed_pts,
            'seed_weighted': self.seed_weighted,
            'batch_size': self.batch_size,
            'batch_method': self.batch_method,
        }


class SectionalMotif(ModelObject):
    """Forecasts using a nearest neighbors type model adapted for probabilistic time series.
    This version takes the distance metric average for all series at once.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): "User" or None. If used, will be as covariate. The ratio of num_series:num_regressor_series will largely determine the impact
        window (int): length of forecast history to match on
        point_method (int): how to summarize the nearest neighbors to generate the point forecast
            "weighted_mean", "mean", "median", "midhinge"
        distance_metric (str): all valid values for scipy cdist + "nan_euclidean" from sklearn
        include_differenced (bool): True to have the distance metric result be an average of the distance on absolute values as well as differenced values
        k (int): number of closest neighbors to consider
        stride_size (int): how many obs to skip between each new window. Higher numbers will reduce the number of matching windows and make the model faster.
    """

    def __init__(
        self,
        name: str = "SectionalMotif",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        regression_type: str = None,
        window: int = 5,
        point_method: str = "weighted_mean",
        distance_metric: str = "nan_euclidean",
        include_differenced: bool = False,
        k: int = 10,
        stride_size: int = 1,
        fillna: str = "SimpleSeasonalityMotifImputer",  # excessive forecast length only
        comparison_transformation: dict = None,
        combination_transformation: dict = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            regression_type=regression_type,
        )
        self.window = window
        self.point_method = point_method
        self.distance_metric = distance_metric
        self.include_differenced = include_differenced
        self.k = k
        self.stride_size = stride_size
        self.fillna = fillna
        self.comparison_transformation = comparison_transformation
        self.combination_transformation = combination_transformation

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            regressor (numpy.Array): additional regressor
        """
        df = self.basic_profile(df)
        self.df = df
        if str(self.regression_type).lower() == "user":
            if future_regressor is None:
                raise ValueError(
                    "regression_type=='User' but no future_regressor supplied"
                )
            self.future_regressor = future_regressor
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        window_size = self.window
        point_method = self.point_method
        distance_metric = self.distance_metric
        regression_type = str(self.regression_type).lower()

        if self.comparison_transformation is not None:
            self.comparison_transformer = GeneralTransformer(
                **self.comparison_transformation
            )
            use_df = self.comparison_transformer.fit_transform(self.df)
        else:
            use_df = self.df

        # the regressor can be tacked on to provide (minor) influence to the distance metric
        if regression_type == "user":
            # here unlagging the regressor to match with history only
            full_regr = pd.concat([self.future_regressor, future_regressor], axis=0)
            full_regr = full_regr.tail(use_df.shape[0])
            full_regr.index = use_df.index
            array = pd.concat([use_df, full_regr], axis=1).to_numpy()
        elif regression_type in base_seasonalities or isinstance(
            self.regression_type, list
        ):
            X = date_part(use_df.index, method=self.regression_type)
            array = pd.concat([use_df, X], axis=1).to_numpy()
        else:
            array = use_df.to_numpy()
        tlt_len = array.shape[0]
        self.combined_window_size = window_size + forecast_length
        self.excessive_size_flag = False
        self.available_indexes = True
        if self.combined_window_size > (array.shape[0]):
            self.combined_window_size = int(array.shape[0] / 2)
            self.excessive_size_flag = True
            self.available_indexes = self.combined_window_size - self.window
        max_steps = array.shape[0] - self.combined_window_size
        window_idxs = window_id_maker(
            window_size=self.combined_window_size,
            start_index=0,
            max_steps=max_steps,
            stride_size=self.stride_size,
            skip_size=1,
        )
        # calculate distance between all points and last window of history
        if distance_metric == "nan_euclidean":
            res = np.array(
                [
                    nan_euclidean_distances(
                        array[:, a][window_idxs[:, :window_size]],
                        array[(tlt_len - window_size) : tlt_len, a].reshape(1, -1),
                    )
                    for a in range(array.shape[1])
                ]
            )
            if self.include_differenced:
                array_diff = np.diff(array, n=1, axis=0)
                array_diff = np.concatenate([array_diff[0:1], array_diff])
                res_diff = np.array(
                    [
                        nan_euclidean_distances(
                            array_diff[:, a][window_idxs[:, :window_size]],
                            array_diff[(tlt_len - window_size) : tlt_len, a].reshape(
                                1, -1
                            ),
                        )
                        for a in range(array_diff.shape[1])
                    ]
                )
                res = np.mean([res, res_diff], axis=0)
        elif distance_metric == "wasserstein":
            res = np.array(
                [
                    np.array(
                        [
                            wasserstein(
                                array[:, a][window_idxs[i, :window_size]],
                                array[(tlt_len - window_size) : tlt_len, a],
                            )
                            for i in range(window_idxs.shape[0])
                        ]
                    )
                    for a in range(array.shape[1])
                ]
            )
            if self.include_differenced:
                array_diff = np.diff(array, n=1, axis=0)
                array_diff = np.concatenate([array_diff[0:1], array_diff])
                res_diff = np.array(
                    [
                        np.array(
                            [
                                wasserstein(
                                    array_diff[:, a][window_idxs[i, :window_size]],
                                    array_diff[(tlt_len - window_size) : tlt_len, a],
                                )
                                for i in range(window_idxs.shape[0])
                            ]
                        )
                        for a in range(array_diff.shape[1])
                    ]
                )
                res = np.mean([res, res_diff], axis=0)
        else:
            res = np.array(
                [
                    cdist(
                        array[:, a][window_idxs[:, :window_size]],
                        array[(tlt_len - window_size) : tlt_len, a].reshape(1, -1),
                        metric=distance_metric,
                    )
                    for a in range(array.shape[1])
                ]
            )
            if self.include_differenced:
                array_diff = np.diff(array, n=1, axis=0)
                array_diff = np.concatenate([array_diff[0:1], array_diff])
                res_diff = np.array(
                    [
                        cdist(
                            array_diff[:, a][window_idxs[:, :window_size]],
                            array_diff[(tlt_len - window_size) : tlt_len, a].reshape(
                                1, -1
                            ),
                            metric=distance_metric,
                        )
                        for a in range(array_diff.shape[1])
                    ]
                )
                res = np.mean([res, res_diff], axis=0)
        # find the lowest distance historical windows
        res_sum = np.nansum(res, axis=0)
        num_top = self.k
        res_idx = np.argpartition(res_sum, num_top, axis=0)[0:num_top]
        self.windows = window_idxs[res_idx, window_size:]
        # handle window being too big for data, too close to end
        if self.windows.size == 0:
            count = 1
            while self.windows.size == 0:
                count += 1
                res_idx = np.argpartition(res_sum, num_top, axis=0)[0 : num_top * count]
                self.windows = window_idxs[res_idx, window_size:]
                # prevent overflow
                if count > 5:
                    if self.verbose >= 1:
                        print("SectionalMotif using fallout")
                    self.windows = window_idxs[res_idx, -forecast_length:]
                    self.available_indexes = False

        if self.combination_transformation is not None:
            self.combination_transformer = GeneralTransformer(
                **self.combination_transformation
            )
            array = self.combination_transformer.fit_transform(self.df).to_numpy()
        else:
            array = self.df.to_numpy()
        results = array[self.windows]
        # reshape results to (num_windows, forecast_length, num_series)
        if results.ndim == 4:
            res_shape = results.shape
            results = results.reshape((res_shape[0], res_shape[2], res_shape[3]))
        if (
            regression_type == "user"
            or regression_type in base_seasonalities
            or isinstance(self.regression_type, list)
        ):
            results = results[:, :, : use_df.shape[1]]
        # now aggregate results into point and bound forecasts
        if point_method == "weighted_mean":
            weights = res_sum[res_idx].flatten()
            if weights.sum() == 0:
                weights = None
            forecast = np.average(results, axis=0, weights=weights)
        elif point_method == "mean":
            forecast = np.nanmean(results, axis=0)
        elif point_method == "median":
            forecast = np.nanmedian(results, axis=0)
        elif point_method == "midhinge":
            q1 = nan_quantile(results, q=0.25, axis=0)
            q2 = nan_quantile(results, q=0.75, axis=0)
            forecast = (q1 + q2) / 2

        pred_int = round((1 - self.prediction_interval) / 2, 5)
        upper_forecast = nan_quantile(results, q=(1 - pred_int), axis=0)
        lower_forecast = nan_quantile(results, q=pred_int, axis=0)

        # more handling short data stuff
        if not self.available_indexes:
            self.available_indexes = forecast.shape[0]
        if self.excessive_size_flag:
            local_index = test_index[0 : self.available_indexes]
        else:
            local_index = test_index
        # convert to df from np
        forecast = pd.DataFrame(forecast, index=local_index, columns=self.column_names)
        lower_forecast = pd.DataFrame(
            lower_forecast, index=local_index, columns=self.column_names
        )
        upper_forecast = pd.DataFrame(
            upper_forecast, index=local_index, columns=self.column_names
        )
        if self.excessive_size_flag:
            forecast = FillNA(forecast.reindex(test_index), method=self.fillna)
            lower_forecast = FillNA(
                lower_forecast.reindex(test_index), method=self.fillna
            )
            upper_forecast = FillNA(
                upper_forecast.reindex(test_index), method=self.fillna
            )
        if self.combination_transformation is not None:
            forecast = self.combination_transformer.inverse_transform(forecast)
            lower_forecast = self.combination_transformer.inverse_transform(
                lower_forecast
            )
            upper_forecast = self.combination_transformer.inverse_transform(
                upper_forecast
            )
        self.result_windows = results
        if just_point_forecast:
            return forecast
        else:
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
        """Returns dict of new parameters for parameter tuning"""
        metric_list = [
            'braycurtis',
            'canberra',
            'chebyshev',
            'cityblock',
            'correlation',
            'cosine',
            # 'dice',
            'euclidean',
            'hamming',
            'jaccard',
            # 'jensenshannon',
            'kulczynski1',
            'mahalanobis',
            'matching',
            'minkowski',
            'rogerstanimoto',
            'russellrao',
            # 'seuclidean',
            'sokalmichener',
            'sokalsneath',
            'sqeuclidean',
            'yule',
            "nan_euclidean",
            "wasserstein",
        ]
        # note this custom override
        trans_dict = superfast_transformer_dict.copy()
        trans_dict["FFTFilter"] = 0.1
        trans_dict["FFTDecomposition"] = 0.1
        comparison_transformation = random.choices([None, True], [0.6, 0.4])[0]
        if comparison_transformation is not None:
            comparison_transformation = RandomTransform(
                transformer_list=trans_dict,
                transformer_max_depth=2,
                allow_none=True,
            )
        combination_transformation = random.choices([None, True], [0.6, 0.4])[0]
        if combination_transformation is not None:
            combination_transformation = RandomTransform(
                transformer_list=trans_dict,
                transformer_max_depth=2,
                allow_none=True,
            )

        if method == "event_risk":
            k_choice = random.choices(
                [10, 15, 20, 50, 100], [0.3, 0.1, 0.1, 0.05, 0.05]
            )[0]
        else:
            k_choice = random.choices(
                [1, 3, 5, 10, 15, 20, 100], [0.2, 0.2, 0.2, 0.5, 0.1, 0.1, 0.1]
            )[0]
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices(
                [None, 'datepart', 'User'], [0.9, 0.1, 0.2]
            )[0]
            if regression_choice == "datepart":
                regression_choice = random.choice(base_seasonalities)
        return {
            "window": random.choices(
                [3, 5, 7, 10, 15, 30, 50], [0.01, 0.2, 0.1, 0.5, 0.1, 0.1, 0.1]
            )[0],
            "point_method": random.choices(
                ["weighted_mean", "mean", "median", "midhinge"], [0.4, 0.2, 0.2, 0.2]
            )[0],
            "distance_metric": random.choice(metric_list),
            "include_differenced": random.choices([True, False], [0.9, 0.1])[0],
            "k": k_choice,
            "stride_size": random.choices([1, 2, 5, 10], [0.6, 0.1, 0.1, 0.1])[0],
            'regression_type': regression_choice,
            "comparison_transformation": comparison_transformation,
            "combination_transformation": combination_transformation,
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {
            "window": self.window,
            "point_method": self.point_method,
            "distance_metric": self.distance_metric,
            "include_differenced": self.include_differenced,
            "k": self.k,
            "stride_size": self.stride_size,
            'regression_type': self.regression_type,
            'comparison_transformation': self.comparison_transformation,
            'combination_transformation': self.combination_transformation,
        }


class KalmanStateSpace(ModelObject):
    """Forecast using a state space model solved by a Kalman Filter.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        subset (int): if not None, forecasts in chunks of this size. Reduces memory at the expense of compute time.
    """

    def __init__(
        self,
        name: str = "KalmanStateSpace",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        state_transition=[[1, 1], [0, 1]],
        process_noise=[[0.1, 0.0], [0.0, 0.01]],
        observation_model=[[1, 0]],
        observation_noise: float = 1.0,
        em_iter: int = 10,
        model_name: str = "undefined",
        forecast_length: int = None,
        subset=None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
        )
        self.state_transition = state_transition
        self.process_noise = process_noise
        self.observation_model = observation_model
        self.observation_noise = observation_noise
        self.em_iter = em_iter
        self.model_name = model_name
        self.forecast_length = forecast_length
        self.subset = subset

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        self.fit_data(df)

        if self.subset is None:
            self.kf = self._fit(df, future_regressor=None)
        elif isinstance(self.subset, (float, int)):
            if self.subset < 1 and self.subset > 0:
                self.subset = self.subset * df.shape[1]
            chunks = df.shape[1] // self.subset
            if chunks > 1:
                self.kf = {}
                self.subset_columns = {}
                if (df.shape[1] % self.subset) != 0:
                    chunks += 1
                for x in range(chunks):
                    subset = df.iloc[:, self.subset * x : (self.subset * (x + 1))]
                    self.subset_columns[str(x)] = subset.columns.tolist()
                    self.kf[str(x)] = self._fit(subset, future_regressor=None)
            else:
                self.kf = self._fit(df, future_regressor=None)
        else:
            raise ValueError(f"subset arg {self.subset} not recognized")

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def _fit(self, df, future_regressor=None):
        if self.observation_noise == "auto":
            self.fit_noise = self.tune_observational_noise(df)[0]
        else:
            self.fit_noise = self.observation_noise
        kf = KalmanFilter(
            state_transition=self.state_transition,  # matrix A
            process_noise=self.process_noise,  # Q
            observation_model=self.observation_model,  # H
            observation_noise=self.fit_noise,  # R
        )
        if self.em_iter is not None:
            kf = kf.em(df.to_numpy().T, n_iter=self.em_iter)

        return kf

    def fit_data(self, df, future_regressor=None):
        df = self.basic_profile(df)
        self.df_train = df  # df.to_numpy().T
        self.train_index = df.index
        return self

    def cost_function(self, param, df):
        try:
            # evaluating on a single, most recent holdout only, for simplicity
            local = df.to_numpy().T
            kf = KalmanFilter(
                state_transition=self.state_transition,  # matrix A
                process_noise=self.process_noise,  # Q
                observation_model=self.observation_model,  # H
                observation_noise=param[0],  # R
                # covariances=False,
            )
            if self.em_iter is not None:
                kf = kf.em(local[:, : -self.forecast_length], n_iter=self.em_iter)
            result = kf.predict(local[:, : -self.forecast_length], self.forecast_length)
            df_smooth = pd.DataFrame(
                result.observations.mean.T,
                index=df.index[-self.forecast_length :],
                columns=df.columns,
            )
            df_stdev = np.sqrt(result.observations.cov).T
            bound = df_stdev * norm.ppf(self.prediction_interval)
            upper_forecast = df_smooth + bound
            lower_forecast = df_smooth - bound

            # evaluate the prediction
            A = np.asarray(df.iloc[-self.forecast_length :, :])
            inv_prediction_interval = 1 - self.prediction_interval
            upper_diff = A - upper_forecast
            upper_pl = np.where(
                A >= upper_forecast,
                self.prediction_interval * upper_diff,
                inv_prediction_interval * -1 * upper_diff,
            )
            # note that the quantile here is the lower quantile
            low_diff = A - lower_forecast
            lower_pl = np.where(
                A >= lower_forecast,
                inv_prediction_interval * low_diff,
                self.prediction_interval * -1 * low_diff,
            )
            scaler = np.nanmean(np.abs(np.diff(np.asarray(df)[-100:], axis=0)), axis=0)
            fill_val = np.nanmax(scaler)
            fill_val = fill_val if fill_val > 0 else 1
            scaler[scaler == 0] = fill_val
            scaler[np.isnan(scaler)] = fill_val
            result_score = np.nansum(np.nanmean(upper_pl + lower_pl, axis=0) / scaler)
            # print(f"param is {param} with score {result_score}")
            return result_score
        except Exception as e:
            print(f"param {param} failed with {repr(e)}")
            return 1e9

    def tune_observational_noise(self, df):
        from scipy.optimize import minimize

        if self.forecast_length is None:
            raise ValueError("for tuning, forecast_length must be passed to init")
        x0 = [0.1]
        bounds = [(0.00001, 100) for x in x0]
        return minimize(
            self.cost_function,
            x0,
            args=(df),
            bounds=bounds,
            method=None,
            options={'maxiter': 50},
        ).x

    def predict(
        self,
        forecast_length: int = None,
        future_regressor=None,
        just_point_forecast=False,
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        if forecast_length is None:
            forecast_length = self.forecast_length
            if self.forecast_length is None:
                raise ValueError(
                    "must provide forecast_length to KalmanStateSpace predict"
                )
        predictStartTime = datetime.datetime.now()
        future_index = self.create_forecast_index(forecast_length)
        if isinstance(self.kf, dict):
            forecasts = []
            uppers = []
            lowers = []
            for x in self.subset_columns:
                current_cols = self.subset_columns[x]
                result = self.kf[x].predict(
                    self.df_train.reindex(columns=current_cols).to_numpy().T,
                    forecast_length,
                )
                df = pd.DataFrame(
                    result.observations.mean.T,
                    index=future_index,
                    columns=current_cols,
                )
                forecasts.append(df)
                df_stdev = np.sqrt(result.observations.cov).T
                bound = df_stdev * norm.ppf(self.prediction_interval)
                uppers.append(df + bound)
                lowers.append(df - bound)
            df = pd.concat(forecasts, axis=1).reindex(columns=self.column_names)
            upper_forecast = pd.concat(uppers, axis=1).reindex(
                columns=self.column_names
            )
            lower_forecast = pd.concat(lowers, axis=1).reindex(
                columns=self.column_names
            )
        else:
            result = self.kf.predict(self.df_train.to_numpy().T, forecast_length)
            df = pd.DataFrame(
                result.observations.mean.T,
                index=future_index,
                columns=self.column_names,
            )
            df_stdev = np.sqrt(result.observations.cov).T
            bound = df_stdev * norm.ppf(self.prediction_interval)
            upper_forecast = df + bound
            lower_forecast = df - bound
        if just_point_forecast:
            return df
        else:
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

    def get_new_params(self, method: str = "random"):
        # predefined, or random
        new_params = new_kalman_params(method=method)
        if method in ['deep']:
            new_params['subset'] = random.choices([None, 200, 300], [0.3, 0.3, 0.3])[0]
        else:
            new_params['subset'] = random.choices([100, 200, 300], [0.3, 0.3, 0.3])[
                0
            ]  # probably no difference
        return new_params

    def get_params(self):
        """Return dict of current parameters."""
        return {
            "model_name": self.model_name,
            "state_transition": self.state_transition,
            "process_noise": self.process_noise,
            "observation_model": self.observation_model,
            "observation_noise": self.observation_noise,
            "subset": self.subset,
        }


class MetricMotif(ModelObject):
    """Forecasts using a nearest neighbors type model adapted for probabilistic time series.
    This version is fully vectorized, using basic metrics for distance comparison.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        window (int): length of forecast history to match on
        point_method (int): how to summarize the nearest neighbors to generate the point forecast
            "weighted_mean", "mean", "median", "midhinge"
        distance_metric (str): mae, mqae, mse
        k (int): number of closest neighbors to consider
    """

    def __init__(
        self,
        name: str = "MetricMotif",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        regression_type: str = None,
        comparison_transformation: dict = None,
        combination_transformation: dict = None,
        window: int = 5,
        point_method: str = "mean",
        distance_metric: str = "mae",
        k: int = 10,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            regression_type=regression_type,
        )
        self.comparison_transformation = comparison_transformation
        self.combination_transformation = combination_transformation
        assert window >= 1, f"window {window} must be >= 1"
        self.window = int(window)
        self.point_method = point_method
        self.distance_metric = str(distance_metric).lower()
        self.k = k

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            regressor (numpy.Array): additional regressor
        """
        df = self.basic_profile(df)
        self.df = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        window_size = self.window
        point_method = self.point_method
        distance_metric = self.distance_metric
        k = self.k

        # fit transform only, no need for inverse as this is only for finding windows
        if self.comparison_transformation is not None:
            self.comparison_transformer = GeneralTransformer(
                **self.comparison_transformation
            )
            compare_df = self.comparison_transformer.fit_transform(self.df)
        else:
            compare_df = self.df
        # applied once, then inversed after windows combined as forecast
        if self.combination_transformation is not None:
            self.combination_transformer = GeneralTransformer(
                **self.combination_transformation
            )
            wind_arr = self.combination_transformer.fit_transform(self.df).to_numpy()
        else:
            wind_arr = self.df.to_numpy()

        array = compare_df.to_numpy()

        nan_flag = np.isnan(np.min(array))

        # when k is larger, can be more aggressive on allowing a longer portion into view
        min_k = 5
        if k > min_k:
            n_tail = min(window_size, forecast_length)
        else:
            n_tail = forecast_length
        # finding sliding windows to compare
        temp = sliding_window_view(array[:-n_tail, :], window_size, axis=0)
        # compare windows by metrics
        last_window = array[-window_size:, :]
        if distance_metric == "mae":
            ae = temp - last_window.T
            np.absolute(ae, out=ae)
            if nan_flag:
                scores = np.nanmean(ae, axis=2)
            else:
                scores = np.mean(ae, axis=2)
        elif distance_metric == "canberra":
            divisor = np.abs(temp) + np.abs(last_window.T)
            divisor[divisor == 0] = 1
            ae = temp - last_window.T
            np.absolute(ae, out=ae)
            if nan_flag:
                scores = np.nanmean(ae / divisor, axis=2)
            else:
                scores = np.mean(ae / divisor, axis=2)
        elif distance_metric == "minkowski":
            p = 2
            ae = temp - last_window.T
            np.absolute(ae, out=ae)
            scores = np.sum(ae**p, axis=2) ** (1 / p)
        elif distance_metric == "cosine":
            scores = 1 - np.sum(temp * last_window.T, axis=2) / (
                np.linalg.norm(temp, axis=2) * np.linalg.norm(last_window.T, axis=2)
            )
        elif distance_metric == "euclidean":
            scores = np.sqrt(np.sum((temp - last_window.T) ** 2, axis=2))
        elif distance_metric == "chebyshev":
            scores = np.max(np.abs(temp - last_window.T), axis=2)
        elif distance_metric == "wasserstein":
            scores = np.mean(
                np.abs(np.cumsum(temp, axis=-1) - np.cumsum(last_window, axis=0).T),
                axis=2,
            )
        elif distance_metric == "mqae":
            q = 0.85
            ae = temp - last_window.T
            np.absolute(ae, out=ae)
            if ae.shape[2] <= 1:
                vals = ae
            else:
                qi = int(ae.shape[2] * q)
                qi = qi if qi > 1 else 1
                vals = np.partition(ae, qi, axis=2)[..., :qi]
            if nan_flag:
                scores = np.nanmean(vals, axis=2)
            else:
                scores = np.mean(vals, axis=2)
        elif distance_metric == "mse":
            scores = np.nanmean((temp - last_window.T) ** 2, axis=2)
        else:
            raise ValueError(f"distance_metric: {distance_metric} not recognized")

        # select smallest windows
        if point_method == "closest":
            min_idx = np.argsort(scores, axis=0)[:k]
        else:
            min_idx = np.argpartition(scores, k - 1, axis=0)[:k]
        # take the period starting AFTER the window
        test = (
            np.moveaxis(
                np.broadcast_to(
                    np.arange(0, forecast_length)[:, None],
                    (1, forecast_length, min_idx.shape[1]),
                ),
                1,
                0,
            )
            + min_idx
            + window_size
        )
        # for data over the end, fill last value
        if k > min_k:
            test = np.where(test >= array.shape[0], -1, test)
        # if you can't tell already, keeping matching shapes is the biggest hassle
        results = np.moveaxis(
            np.take_along_axis(
                wind_arr[..., None],
                test.reshape(forecast_length, min_idx.shape[1], k),
                axis=0,
            ),
            2,
            0,
        )

        # now aggregate results into point and bound forecasts
        if point_method == "weighted_mean":
            weights = scores[min_idx].sum(axis=1)
            if weights.sum() == 0:
                weights = None
            forecast = np.average(
                results,
                axis=0,
                weights=np.repeat(weights[:, None, :], forecast_length, axis=1),
            )
        elif point_method == "mean":
            forecast = np.nanmean(results, axis=0)
        elif point_method == "median":
            forecast = np.nanmedian(results, axis=0)
        elif point_method == "midhinge":
            q1 = nan_quantile(results, q=0.25, axis=0)
            q2 = nan_quantile(results, q=0.75, axis=0)
            forecast = (q1 + q2) / 2
        elif point_method == "closest":
            forecast = results[0]
        else:
            raise ValueError(f"point_method {point_method} not recognized")

        del temp, scores, test, min_idx

        pred_int = round((1 - self.prediction_interval) / 2, 5)
        upper_forecast = nan_quantile(results, q=(1 - pred_int), axis=0)
        lower_forecast = nan_quantile(results, q=pred_int, axis=0)

        forecast = pd.DataFrame(forecast, index=test_index, columns=self.column_names)
        lower_forecast = pd.DataFrame(
            lower_forecast, index=test_index, columns=self.column_names
        )
        upper_forecast = pd.DataFrame(
            upper_forecast, index=test_index, columns=self.column_names
        )
        if self.combination_transformation is not None:
            forecast = self.combination_transformer.inverse_transform(forecast)
            lower_forecast = self.combination_transformer.inverse_transform(
                lower_forecast
            )
            upper_forecast = self.combination_transformer.inverse_transform(
                upper_forecast
            )

        self.result_windows = results
        if just_point_forecast:
            return forecast
        else:
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
        """Returns dict of new parameters for parameter tuning"""
        metric_list = [
            'mae',
            'mqae',
            'mse',
            "canberra",
            "minkowski",
            "euclidean",
            "chebyshev",
            "wasserstein",
        ]
        # comparisons = filters.copy()
        # comparisons['CenterLastValue'] = 0.05
        # comparisons['StandardScaler'] = 0.05
        # combinations = filters.copy()
        # combinations['AlignLastValue'] = 0.1
        if method == "event_risk":
            k_choice = random.choices(
                [10, 15, 20, 50, 100], [0.3, 0.1, 0.1, 0.05, 0.05]
            )[0]
        else:
            k_choice = random.choices(
                [1, 3, 5, 10, 15, 20, 100], [0.2, 0.2, 0.2, 0.5, 0.1, 0.1, 0.1]
            )[0]
        return {
            "window": random.choices(
                [3, 5, 7, 10, 15, 30, 50], [0.01, 0.2, 0.1, 0.5, 0.1, 0.1, 0.1]
            )[0],
            # weighted mean is effective but higher memory usage (weights= call)
            "point_method": random.choices(
                ["weighted_mean", "mean", "median", "midhinge", "closest"],
                [0.1, 0.3, 0.2, 0.2, 0.1],
            )[0],
            "distance_metric": random.choice(metric_list),
            "k": k_choice,
            "comparison_transformation": RandomTransform(
                transformer_list=superfast_transformer_dict,
                transformer_max_depth=1,
                allow_none=True,
            ),
            "combination_transformation": RandomTransform(
                transformer_list=superfast_transformer_dict,
                transformer_max_depth=1,
                allow_none=True,
            ),
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {
            "window": self.window,
            "point_method": self.point_method,
            "distance_metric": self.distance_metric,
            "k": self.k,
            "comparison_transformation": self.comparison_transformation,
            "combination_transformation": self.combination_transformation,
        }


class SeasonalityMotif(ModelObject):
    """Forecasts using a nearest neighbors type model adapted for probabilistic time series.
    This version is fully vectorized, using basic metrics for distance comparison.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        window (int): length of forecast history to match on
        point_method (int): how to summarize the nearest neighbors to generate the point forecast
            "weighted_mean", "mean", "median", "midhinge"
        distance_metric (str): mae, mqae, mse
        k (int): number of closest neighbors to consider
        independent (bool): if True, each time step is separate. This is the one motif that can then handle large forecast_length to short historical data.
    """

    def __init__(
        self,
        name: str = "SeasonalityMotif",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        regression_type: str = None,
        window: int = 5,
        point_method: str = "mean",
        distance_metric: str = "mae",
        k: int = 10,
        datepart_method: str = "common_fourier",
        independent: bool = False,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            regression_type=regression_type,
        )
        assert window >= 1, f"window {window} must be >= 1"
        self.window = int(window)
        self.point_method = point_method
        self.distance_metric = str(distance_metric).lower()
        self.k = k
        self.datepart_method = datepart_method
        self.independent = independent

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            regressor (numpy.Array): additional regressor
        """
        df = self.basic_profile(df)
        self.df = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        window_size = self.window
        point_method = self.point_method
        distance_metric = self.distance_metric
        datepart_method = self.datepart_method
        k = self.k
        full_sort = self.point_method == "closest"

        if forecast_length >= self.df.shape[0]:
            self.independent = True
            if self.verbose > 0:
                print(
                    "prediction too long for indepedent=False, falling back on indepedent=True"
                )
        if self.independent:
            # each timestep is considered individually and not as a series
            test, scores = seasonal_independent_match(
                DTindex=self.df.index,
                DTindex_future=test_index,
                k=k,
                datepart_method=datepart_method,
                distance_metric=distance_metric,
                full_sort=full_sort,
            )
        else:
            # original method and perhaps smoother
            test, scores = seasonal_window_match(
                DTindex=self.df.index,
                k=k,
                window_size=window_size,
                forecast_length=forecast_length,
                datepart_method=datepart_method,
                distance_metric=distance_metric,
                full_sort=full_sort,
            )
        # (num_windows, forecast_length, num_series)
        results = np.moveaxis(np.take(self.df.to_numpy(), test, axis=0), 1, 0)

        # now aggregate results into point and bound forecasts
        if point_method == "weighted_mean":
            weights = scores[test[0] - window_size].sum(axis=1)
            if weights.sum() == 0:
                weights = None
            forecast = np.average(results, axis=0, weights=weights)
        elif point_method == "mean":
            forecast = np.nanmean(results, axis=0)
        elif point_method == "median":
            forecast = np.nanmedian(results, axis=0)
        elif point_method == "midhinge":
            q1 = nan_quantile(results, q=0.25, axis=0)
            q2 = nan_quantile(results, q=0.75, axis=0)
            forecast = (q1 + q2) / 2
        elif point_method == "trimmed_mean_20":
            forecast = trimmed_mean(results, percent=0.2, axis=0)
        elif point_method == "trimmed_mean_40":
            forecast = trimmed_mean(results, percent=0.4, axis=0)
        elif point_method == "closest":
            forecast = results[0]
        else:
            raise ValueError(f"point_method {point_method} not recognized.")

        pred_int = round((1 - self.prediction_interval) / 2, 5)
        upper_forecast = nan_quantile(results, q=(1 - pred_int), axis=0)
        lower_forecast = nan_quantile(results, q=pred_int, axis=0)

        forecast = pd.DataFrame(forecast, index=test_index, columns=self.column_names)
        lower_forecast = pd.DataFrame(
            lower_forecast, index=test_index, columns=self.column_names
        )
        upper_forecast = pd.DataFrame(
            upper_forecast, index=test_index, columns=self.column_names
        )
        self.result_windows = results
        if just_point_forecast:
            return forecast
        else:
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
        """Returns dict of new parameters for parameter tuning"""
        metric_list = [
            'mae',
            'mqae',
            'mse',
            'canberra',
            'minkowski',
            'chebyshev',
        ]
        if method == "event_risk":
            k_choice = random.choices(
                [10, 15, 20, 50, 100], [0.3, 0.1, 0.1, 0.05, 0.05]
            )[0]
        else:
            k_choice = random.choices(
                [1, 3, 5, 10, 15, 20, 100], [0.2, 0.2, 0.2, 0.5, 0.1, 0.1, 0.1]
            )[0]
        return {
            "window": random.choices(
                [3, 5, 7, 10, 15, 30, 50], [0.01, 0.2, 0.1, 0.5, 0.1, 0.1, 0.1]
            )[0],
            "point_method": random.choices(
                [
                    "weighted_mean",
                    "mean",
                    "median",
                    "midhinge",
                    'closest',
                    'trimmed_mean_20',
                    'trimmed_mean_40',
                ],
                [0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
            )[0],
            "distance_metric": random.choice(metric_list),
            "k": k_choice,
            "datepart_method": random_datepart(method=method),
            "independent": bool(random.getrandbits(1)),
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {
            "window": self.window,
            "point_method": self.point_method,
            "distance_metric": self.distance_metric,
            "k": self.k,
            "datepart_method": self.datepart_method,
            "independent": self.independent,
        }


class FFT(ModelObject):
    def __init__(
        self,
        name: str = "FFT",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2023,
        verbose: int = 0,
        n_harmonics: int = 10,
        detrend: str = "linear",
        **kwargs,
    ):
        """Fast Fourier Transform forecast.

        Args:
            n_harmonics (int): number of frequencies to include
            detrend (str): None, 'linear', or 'quadratic', use if no other detrending already done
        """
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
        )
        assert n_harmonics >= 2, f"n_harmonics {n_harmonics} must be >= 2"
        self.n_harmonics = int(n_harmonics)
        self.detrend = detrend

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            regressor (numpy.Array): additional regressor
        """
        df = self.basic_profile(df)
        self.df = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        forecast = pd.DataFrame(
            fourier_extrapolation(
                self.df.to_numpy(),
                forecast_length,
                n_harm=self.n_harmonics,
                detrend=self.detrend,
            )[-forecast_length:],
            columns=self.df.columns,
        )
        forecast.index = test_index
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df,
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
        """Returns dict of new parameters for parameter tuning"""
        return {
            "n_harmonics": random.choices(
                [2, 3, 4, 5, 6, 10, 20, 100, 1000, 5000],
                [0.1, 0.02, 0.2, 0.02, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1],
            )[0],
            "detrend": random.choices([None, "linear", 'quadratic'], [0.2, 0.7, 0.1])[
                0
            ],
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {
            "n_harmonics": self.n_harmonics,
            "detrend": self.detrend,
        }


class BallTreeMultivariateMotif(ModelObject):
    """Forecasts using a nearest neighbors type model adapted for probabilistic time series.
    Many of these motifs will struggle when the forecast_length is large and history is short.

    Args:
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        n_jobs (int): how many parallel processes to run
        random_seed (int): used in selecting windows if max_windows is less than total available
        window (int): length of forecast history to match on
        point_method (int): how to summarize the nearest neighbors to generate the point forecast
            "weighted_mean", "mean", "median", "midhinge"
        distance_metric (str): all valid values for scipy cdist
        k (int): number of closest neighbors to consider
    """

    def __init__(
        self,
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = 1,
        window: int = 5,
        point_method: str = "mean",
        distance_metric: str = "canberra",
        k: int = 10,
        sample_fraction=None,
        comparison_transformation: dict = None,
        combination_transformation: dict = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            "BallTreeMultivariateMotif",
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.window = window
        self.point_method = point_method
        self.distance_metric = distance_metric
        self.k = k
        self.sample_fraction = sample_fraction
        self.comparison_transformation = comparison_transformation
        self.combination_transformation = combination_transformation

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        phrase_n = self.window + forecast_length
        # fit transform only, no need for inverse as this is only for finding windows
        if self.comparison_transformation is not None:
            self.comparison_transformer = GeneralTransformer(
                **self.comparison_transformation
            )
            compare_df = self.comparison_transformer.fit_transform(self.df)
        else:
            compare_df = self.df
        # applied once, then inversed after windows combined as forecast
        if self.combination_transformation is not None:
            self.combination_transformer = GeneralTransformer(
                **self.combination_transformation
            )
            wind_arr = self.combination_transformer.fit_transform(self.df)
        else:
            wind_arr = self.df

        if False:
            # OLD WAY
            x = sliding_window_view(
                compare_df.to_numpy(dtype=np.float32), phrase_n, axis=0
            )
            Xa = x.reshape(-1, x.shape[-1])
            if self.sample_fraction is not None:
                if 0 < self.sample_fration < 1:
                    sample_size = int(Xa.shape[0] * self.sample_fraction)
                else:
                    sample_size = (
                        int(self.sample_fraction)
                        if Xa.shape[0] < self.sample_fraction
                        else int(Xa.shape[0] - 1)
                    )
                Xa = np.random.default_rng().choice(Xa, size=sample_size, axis=0)
        else:
            # shared with WindowRegression
            Xa = chunk_reshape(
                compare_df.to_numpy(dtype=np.float32),
                phrase_n,
                sample_fraction=self.sample_fraction,
                random_seed=self.random_seed,
            )
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        if self.distance_metric in ["euclidean", 'kdtree']:
            from scipy.spatial import KDTree

            # Build a KDTree for Xb
            tree = KDTree(Xa[:, : self.window], leafsize=40)
        else:
            from sklearn.neighbors import BallTree

            tree = BallTree(Xa[:, : self.window], metric=self.distance_metric)
            # Query the KDTree to find k nearest neighbors for each point in Xa
        Xb = compare_df.iloc[-self.window :].to_numpy().T
        A, self.windows = tree.query(Xb, k=self.k)  # dualtree=True
        if (
            self.combination_transformation is not None
            or self.comparison_transformation is not None
        ):
            del Xa
            Xc = chunk_reshape(
                wind_arr.to_numpy(dtype=np.float32),
                phrase_n,
                sample_fraction=self.sample_fraction,
                random_seed=self.random_seed,
            )
        else:
            Xc = Xa
        # (k, forecast_length, n_series)
        self.result_windows = Xc[self.windows][:, :, self.window :].transpose(1, 2, 0)

        # now aggregate results into point and bound forecasts
        if self.point_method == "weighted_mean":
            weights = np.repeat(A.T[..., np.newaxis, :], forecast_length, axis=1)
            if weights.sum() == 0:
                weights = None
            forecast = np.average(self.result_windows, axis=0, weights=weights)
        elif self.point_method == "mean":
            forecast = np.nanmean(self.result_windows, axis=0)
        elif self.point_method == "median":
            forecast = np.nanmedian(self.result_windows, axis=0)
        elif self.point_method == "midhinge":
            q1 = nan_quantile(self.result_windows, q=0.25, axis=0)
            q2 = nan_quantile(self.result_windows, q=0.75, axis=0)
            forecast = (q1 + q2) / 2
        elif self.point_method == 'closest':
            # assumes the first K is the smallest distance (true when written)
            forecast = self.result_windows[0]

        pred_int = round((1 - self.prediction_interval) / 2, 5)
        upper_forecast = nan_quantile(self.result_windows, q=(1 - pred_int), axis=0)
        lower_forecast = nan_quantile(self.result_windows, q=pred_int, axis=0)

        forecast = pd.DataFrame(forecast, index=test_index, columns=self.column_names)
        lower_forecast = pd.DataFrame(
            lower_forecast, index=test_index, columns=self.column_names
        )
        upper_forecast = pd.DataFrame(
            upper_forecast, index=test_index, columns=self.column_names
        )
        if self.combination_transformation is not None:
            forecast = self.combination_transformer.inverse_transform(forecast)
            lower_forecast = self.combination_transformer.inverse_transform(
                lower_forecast
            )
            upper_forecast = self.combination_transformer.inverse_transform(
                upper_forecast
            )
        if just_point_forecast:
            return forecast
        else:
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=forecast.index,
                forecast_columns=forecast.columns,
                # so it's producing float32 but pandas is better with float64
                lower_forecast=lower_forecast.astype(float),
                forecast=forecast.astype(float),
                upper_forecast=upper_forecast.astype(float),
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning"""
        metric_list = [
            'braycurtis',
            'canberra',
            'chebyshev',
            'cityblock',
            'euclidean',
            'hamming',
            # 'mahalanobis',
            'minkowski',
            'kdtree',
        ]
        metric_probabilities = [
            0.05,
            0.05,
            0.05,
            0.05,
            0.9,
            0.05,
            # 0.05,
            0.05,
            0.05,
        ]
        if method != "deep":
            # evidence suggests 20 million can fit in 5 GB of RAM with a window of 28
            sample_fraction = random.choice([5000000, 50000000])
        else:
            sample_fraction = random.choice([0.2, 0.5, 100000000, None])
        if method == "event_risk":
            k_choice = random.choices(
                [10, 15, 20, 50, 100], [0.3, 0.1, 0.1, 0.05, 0.05]
            )[0]
        else:
            k_choice = random.choices(
                [1, 3, 5, 10, 15, 20, 100], [0.02, 0.2, 0.2, 0.5, 0.1, 0.1, 0.1]
            )[0]
        transformers_none = random.choices([True, False], [0.7, 0.3])[0]
        if transformers_none:
            comparison_transformation = None
            combination_transformation = None
        else:
            comparison_transformation = RandomTransform(
                transformer_list=superfast_transformer_dict,
                transformer_max_depth=1,
                allow_none=True,
            )
            combination_transformation = RandomTransform(
                transformer_list=superfast_transformer_dict,
                transformer_max_depth=1,
                allow_none=True,
            )
        return {
            "window": random.choices(
                [2, 3, 5, 7, 10, 14, 28, 60],
                [0.01, 0.01, 0.01, 0.1, 0.5, 0.1, 0.1, 0.01],
            )[0],
            "point_method": random.choices(
                ["weighted_mean", "mean", "median", "midhinge", "closest"],
                [0.4, 0.2, 0.2, 0.2, 0.2],
            )[0],
            "distance_metric": random.choices(metric_list, metric_probabilities)[0],
            "k": k_choice,
            "sample_fraction": sample_fraction,
            "comparison_transformation": comparison_transformation,
            "combination_transformation": combination_transformation,
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {
            "window": self.window,
            "point_method": self.point_method,
            "distance_metric": self.distance_metric,
            "k": self.k,
            "sample_fraction": self.sample_fraction,
            "comparison_transformation": self.comparison_transformation,
            "combination_transformation": self.combination_transformation,
        }


class BasicLinearModel(ModelObject):
    """Ridge regression of seasonal + trend changepoint + constant + regressor.
    Like a minimal version of Prophet or Cassandra.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): "User" or None. If used, will be as covariate. The ratio of num_series:num_regressor_series will largely determine the impact
        window (int): length of forecast history to match on
        point_method (int): how to summarize the nearest neighbors to generate the point forecast
            "weighted_mean", "mean", "median", "midhinge"
        distance_metric (str): all valid values for scipy cdist + "nan_euclidean" from sklearn
        include_differenced (bool): True to have the distance metric result be an average of the distance on absolute values as well as differenced values
        k (int): number of closest neighbors to consider
        stride_size (int): how many obs to skip between each new window. Higher numbers will reduce the number of matching windows and make the model faster.
    """

    def __init__(
        self,
        name: str = "BasicLinearModel",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2024,
        verbose: int = 0,
        regression_type: str = None,
        datepart_method: str = "common_fourier",
        changepoint_spacing: int = None,
        changepoint_distance_end: int = None,
        lambda_: float = 0.01,
        trend_phi: float = None,
        holiday_countries_used: bool = True,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            regression_type=regression_type,
        )
        self.datepart_method = datepart_method
        self.changepoint_spacing = changepoint_spacing
        self.changepoint_distance_end = changepoint_distance_end
        self.lambda_ = lambda_
        self.trend_phi = trend_phi
        self.holiday_countries_used = holiday_countries_used

        self.regressor_columns = []

    def base_scaler(self, df):
        self.scaler_mean = np.mean(df, axis=0)
        self.scaler_std = np.std(df, axis=0).replace(0, 1)
        return (df - self.scaler_mean) / self.scaler_std

    def scale_data(self, df):
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

    def create_x(
        self,
        df,
        future_regressor=None,
        holiday_country="US",
        holiday_countries_used=True,
    ):
        x_s = date_part(
            df.index,
            method=self.datepart_method,
            set_index=True,
            holiday_country=holiday_country,
            holiday_countries_used=holiday_countries_used,
        )
        x_t = create_changepoint_features(
            df.index,
            changepoint_spacing=self.changepoint_spacing,
            changepoint_distance_end=self.changepoint_distance_end,
        )
        self.last_row = x_t.iloc[-1]
        X = pd.concat([x_s, x_t], axis=1)
        if str(self.regression_type).lower() == "user" and future_regressor is not None:
            temp = future_regressor.reindex(df.index).rename(
                columns=lambda x: "regr_" + str(x)
            )
            self.regressor_columns = temp.columns
            X = pd.concat([X, temp], axis=1)
        X["constant"] = 1

        self.seasonal_columns = x_s.columns.tolist()
        self.trend_columns = x_t.columns.tolist()
        return X

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df = df
        if self.changepoint_spacing is None or self.changepoint_distance_end is None:
            half_yr_space = half_yr_spacing(df)
            if self.changepoint_spacing is None:
                self.changepoint_spacing = int(half_yr_space)
            if self.changepoint_distance_end is None:
                self.changepoint_distance_end = int(half_yr_space / 2)
        if str(self.regression_type).lower() == "user":
            if future_regressor is None:
                raise ValueError(
                    "regression_type=='User' but no future_regressor supplied"
                )
        X = self.create_x(
            df,
            future_regressor,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )

        # Convert X and df (Y) to NumPy arrays for linear regression
        X_values = X.to_numpy().astype(float)
        Y_values = df.to_numpy().astype(float)

        if self.lambda_ is not None:
            I = np.eye(X_values.shape[1])
            # Perform Ridge regression using the modified normal equation
            self.beta = (
                np.linalg.inv(X_values.T @ X_values + self.lambda_ * I)
                @ X_values.T
                @ Y_values
            )
        else:
            # Perform linear regression using the normal equation: (X.T @ X)^(-1) @ X.T @ Y
            self.beta = np.linalg.pinv(X_values.T @ X_values) @ X_values.T @ Y_values

        # Calculate predicted values for Y
        Y_pred = X_values @ self.beta

        # Calculate residuals for each column of Y
        residuals = Y_values - Y_pred

        # Calculate the sum of squared errors (SSE) and standard error (sigma) for each column of Y
        sse = np.sum(
            residuals**2, axis=0
        )  # Sum of squared errors for each column (shape (21,))
        n = Y_values.shape[0]
        p = X_values.shape[1]  # Number of predictors
        self.sigma = np.sqrt(sse / (n - p))

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        x_s = date_part(
            test_index,
            method=self.datepart_method,
            set_index=True,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )
        x_t = changepoint_fcst_from_last_row(self.last_row, int(forecast_length))
        x_t.index = test_index
        X = pd.concat([x_s, x_t], axis=1)
        if str(self.regression_type).lower() == "user":
            X = pd.concat([X, future_regressor.reindex(test_index)], axis=1)
        X["constant"] = 1
        X_values = X.to_numpy().astype(float)
        self.X = X

        if self.trend_phi is None or self.trend_phi == 1 or len(test_index) < 2:
            forecast = pd.DataFrame(
                X_values @ self.beta, columns=self.column_names, index=test_index
            )
        else:
            components = np.einsum('ij,jk->ijk', self.X.to_numpy(), self.beta)
            trend_x_start = x_s.shape[1]
            trend_x_end = x_s.shape[1] + x_t.shape[1]
            trend_components = components[:, trend_x_start:trend_x_end, :]

            req_len = len(test_index) - 1
            phi_series = pd.Series(
                [self.trend_phi] * req_len,
                index=test_index[1:],
            ).pow(range(req_len))

            diff_array = np.diff(trend_components, axis=0)
            diff_scaled_array = (
                diff_array * phi_series.to_numpy()[:, np.newaxis, np.newaxis]
            )
            first_row = trend_components[0:1, :]
            combined_array = np.vstack([first_row, diff_scaled_array])
            components[:, trend_x_start:trend_x_end, :] = np.cumsum(
                combined_array, axis=0
            )

            forecast = pd.DataFrame(
                components.sum(axis=1), columns=self.column_names, index=test_index
            )

        if just_point_forecast:
            return forecast
        else:
            z_value = norm.ppf(1 - (1 - self.prediction_interval) / 2)
            # Vectorized calculation of leverage for all points: diag(X @ (X^T X)^(-1) @ X^T)
            hat_matrix_diag = np.einsum(
                'ij,jk,ik->i',
                X_values,
                np.linalg.pinv(X_values.T @ X_values, rcond=5e-16),
                X_values,
            )
            sigma_expanded = self.sigma[np.newaxis, :]
            # Calculate the margin of error for each prediction in each column
            margin_of_error = pd.DataFrame(
                z_value * sigma_expanded * np.sqrt(1 + hat_matrix_diag[:, np.newaxis]),
                columns=self.column_names,
                index=test_index,
            )
            upper_forecast = forecast + margin_of_error
            lower_forecast = forecast - margin_of_error

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=forecast.index,
                forecast_columns=forecast.columns,
                # so it's producing float32 but pandas is better with float64
                lower_forecast=lower_forecast.astype(float),
                forecast=forecast.astype(float),
                upper_forecast=upper_forecast.astype(float),
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def process_components(self):
        # mimic Cassandra style
        res = []
        components = np.einsum('ij,jk->ijk', self.X.to_numpy(), self.beta)
        for x in range(components.shape[2]):
            df = pd.DataFrame(
                components[:, :, x], index=self.X.index, columns=self.X.columns
            )
            new_level = self.column_names[x]
            df.columns = pd.MultiIndex.from_product([[new_level], df.columns])
            res.append(df)
        return pd.concat(res, axis=1)

    def return_components(self, df):
        # Needs some work
        # doens't handle regressor features
        # recompiles X which is suboptimal
        # could use better naming
        X = self.create_x(
            df,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )
        contribution_seasonality = (
            X[self.seasonal_columns].values @ self.beta[: len(self.seasonal_columns)]
        )
        contribution_changepoints = (
            X[self.trend_columns].values
            @ self.beta[
                len(self.seasonal_columns) : len(self.seasonal_columns)
                + len(self.trend_columns)
            ]
        )
        contribution_constant = X["constant"].values.reshape(-1, 1) @ self.beta[-1:]
        return (
            contribution_seasonality,
            contribution_changepoints,
            contribution_constant,
        )

    def coefficient_summary(self, df):
        """Used in profiler."""
        (
            contribution_seasonality,
            contribution_changepoints,
            contribution_constant,
        ) = self.return_components(df)
        # Total contribution (sum of absolute contributions for each time step)
        total_contribution = (
            np.abs(contribution_seasonality)
            + np.abs(contribution_changepoints)
            + np.abs(contribution_constant)
        )

        # Normalize each contribution by the total contribution
        contrib_seasonality_pct = np.abs(contribution_seasonality) / total_contribution
        contrib_changepoints_pct = (
            np.abs(contribution_changepoints) / total_contribution
        )
        contrib_constant_pct = np.abs(contribution_constant) / total_contribution

        # Calculate the average percentage contribution for each group
        avg_contrib_seasonality = np.mean(contrib_seasonality_pct, axis=0)
        avg_contrib_changepoints = np.mean(contrib_changepoints_pct, axis=0)
        avg_contrib_constant = np.mean(contrib_constant_pct, axis=0)

        # Create a DataFrame to summarize the percentage contributions
        feature_contributions = pd.DataFrame(
            {
                "seasonality_contribution": avg_contrib_seasonality,
                "changepoint_contribution": avg_contrib_changepoints,
                "constant_contribution": avg_contrib_constant,
            },
            index=self.column_names,
        )
        """
        feature_contributions['largest_contributor'] = feature_contributions[[
            'seasonality_contribution', 
            'changepoint_contribution',
            'constant_contribution'
        ]].idxmax(axis=1)
        """
        feature_contributions["season_trend_percent"] = feature_contributions[
            "seasonality_contribution"
        ] / (
            feature_contributions["changepoint_contribution"]
            + feature_contributions["seasonality_contribution"]
        )
        return feature_contributions

    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning"""
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices([None, 'User'], [0.8, 0.2])[0]
        return {
            "datepart_method": random_datepart(method=method),
            "changepoint_spacing": random.choices(
                [None, 6, 28, 60, 90, 120, 180, 360, 5040],
                [0.1, 0.05, 0.1, 0.1, 0.1, 0.05, 0.2, 0.1, 0.2],
            )[0],
            "changepoint_distance_end": random.choices(
                [None, 6, 28, 60, 90, 180, 360, 520, 5040],
                [0.1, 0.05, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.2],
            )[0],
            "regression_type": regression_choice,
            "lambda_": random.choices(
                [None, 0.001, 0.01, 0.1, 1, 2, 10, 100, 1000, 10000],
                [0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                k=1,
            )[0],
            "trend_phi": random.choices(
                [None, 0.995, 0.99, 0.98, 0.97, 0.8], [0.9, 0.05, 0.05, 0.1, 0.02, 0.01]
            )[0],
            "holiday_countries_used": random.choices([True, False], [0.5, 0.5])[0],
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {
            "datepart_method": self.datepart_method,
            "changepoint_spacing": self.changepoint_spacing,
            "changepoint_distance_end": self.changepoint_distance_end,
            "regression_type": self.regression_type,
            "lambda_": self.lambda_,
            "trend_phi": self.trend_phi,
            "holiday_countries_used": self.holiday_countries_used,
        }


class TVVAR(BasicLinearModel):
    """Time Varying VAR

    Notes:
        var_preprocessing will fail with many options, anything that scales/shifts the space
        x_scaled=True seems to fail often when base_scaled=False and VAR components used
    TODO:
        # plot of feature impacts

        # highly correlated, shared hidden factors
        # groups / geos

        # impulse response
        # allow other regression models
        # could run regression twice, setting to zero any X which had low coefficients for the second run

        # feature summarization (dynamic factor is PCA)
        # hierchial by GEO
    """

    def __init__(
        self,
        name: str = "TVVAR",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        regression_type: str = None,
        datepart_method: str = "common_fourier",
        changepoint_spacing: int = None,
        changepoint_distance_end: int = None,
        lambda_: float = 0.01,
        phi: float = None,
        max_cycles: int = 2000,
        trend_phi: float = None,
        var_dampening: float = None,
        lags: list = None,
        rolling_means: list = None,
        apply_pca: bool = False,
        pca_n_components: float = 0.95,
        threshold_method: str = 'std',  # 'std' or 'percentile'
        threshold_value: float = None,  # Multiple of std or percentile value
        base_scaled: bool = True,
        x_scaled: bool = False,
        var_preprocessing: dict = False,
        var_postprocessing: dict = False,
        mode: str = 'additive',
        holiday_countries_used: bool = True,
        **kwargs,
    ):
        super().__init__(
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            regression_type=regression_type,
            datepart_method=datepart_method,
            changepoint_spacing=changepoint_spacing,
            changepoint_distance_end=changepoint_distance_end,
            lambda_=lambda_,
            **kwargs,
        )
        self.phi = phi
        self.lags_list = self.lags = lags
        self.rolling_avg_list = self.rolling_means = rolling_means
        self.apply_pca = apply_pca
        self.pca_n_components = pca_n_components
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.base_scaled = base_scaled
        self.x_scaled = x_scaled
        self.var_preprocessing = var_preprocessing
        self.var_postprocessing = var_postprocessing
        self.trend_phi = trend_phi
        self.var_dampening = var_dampening
        self.max_cycles = max_cycles
        self.mode = str(mode).lower()
        self.holiday_countries_used = holiday_countries_used

    def empty_scaler(self, df):
        self.scaler_std = pd.Series(1.0, index=df.columns)
        self.scaler_mean = 0.0
        return df

    def create_VAR_features(self, df):
        lagged_data = pd.DataFrame(index=df.index)
        # Use triple underscores as separators to avoid conflicts
        if self.lags_list is not None:
            for lag in self.lags_list:
                lagged = df.shift(lag).add_suffix(f"___lag{lag}")
                lagged_data = pd.concat([lagged_data, lagged], axis=1)
        if self.rolling_avg_list is not None:
            for window in self.rolling_avg_list:
                rolling_avg = (
                    df.shift(1)
                    .rolling(window=window)
                    .mean()
                    .add_suffix(f"___ravg{window}")
                )
                lagged_data = pd.concat([lagged_data, rolling_avg], axis=1)
        return lagged_data

    def apply_beta_threshold(self, beta=None):
        if beta is None:
            beta = self.beta
        if self.threshold_value is None:
            return beta
        # Compute absolute values of coefficients
        beta_abs = np.abs(beta)
        # Determine threshold dynamically
        if self.threshold_method == 'std':
            # Use multiple of standard deviation
            beta_std = np.std(beta, axis=0, keepdims=True)
            threshold = self.threshold_value * beta_std
        elif self.threshold_method == 'percentile':
            # Use percentile
            threshold = np.percentile(
                beta_abs, self.threshold_value * 100, axis=0, keepdims=True
            )
        else:
            raise ValueError("threshold_method must be 'std' or 'percentile'")
        # Set coefficients below threshold to zero
        beta = np.where(beta_abs >= threshold, beta, 0)
        return beta

    def fit(self, df, future_regressor=None):
        df = self.basic_profile(df)
        if self.mode == 'multiplicative':
            # could add a PositiveShift here to make this more reliable on all data
            df_scaled = (
                np.log(df.replace(0, np.nan)).replace(-np.inf, np.nan).bfill().ffill()
            )
        else:
            df_scaled = df
        # Scaling df
        if self.base_scaled:
            df_scaled = self.base_scaler(df_scaled)
        else:
            df_scaled = self.empty_scaler(df_scaled)
        if self.changepoint_spacing is None or self.changepoint_distance_end is None:
            half_yr_space = half_yr_spacing(df)
            if self.changepoint_spacing is None:
                self.changepoint_spacing = int(half_yr_space)
            if self.changepoint_distance_end is None:
                self.changepoint_distance_end = int(half_yr_space / 2)
        if str(self.regression_type).lower() == "user":
            if future_regressor is None:
                raise ValueError(
                    "regression_type=='User' but no future_regressor supplied"
                )
        if self.var_preprocessing:
            # the idea here is to filter the data more than the Y values may be
            # for some transfomers, this would also need to be implemented in the predict loop (not implemented)
            self.var_preprocessor = GeneralTransformer(
                n_jobs=self.n_jobs,
                holiday_country=self.holiday_country,
                verbose=self.verbose,
                random_seed=self.random_seed,
                # forecast_length=self.forecast_length,
                **self.var_preprocessing,
            )
            # note uses UNSCALED df
            self.var_history = self.var_preprocessor.fit_transform(df).ffill().bfill()
            if self.base_scaled:
                self.var_history = (
                    self.var_history - self.scaler_mean
                ) / self.scaler_std
        else:
            self.var_history = df_scaled
        # Create VAR features
        VAR_features = self.create_VAR_features(self.var_history)
        VAR_feature_columns = VAR_features.columns.tolist()
        # Create external features
        X_ext = self.create_x(
            df,
            future_regressor,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )
        # Combine features
        X = pd.concat([X_ext, VAR_features], axis=1)
        # Remove rows with NaNs due to lagging
        X = X.dropna()
        # note the DF here not df_scaled so potentially index could be different which is not accounted for
        Y_values = df_scaled.loc[X.index].to_numpy().astype(float)
        # Optionally apply PCA
        if self.apply_pca and (self.lags is not None or self.rolling_means is not None):
            from sklearn.decomposition import PCA

            self.pca = PCA(n_components=self.pca_n_components)
            VAR_data = X[VAR_feature_columns]
            VAR_pca = self.pca.fit_transform(VAR_data)
            VAR_pca_df = pd.DataFrame(VAR_pca, index=VAR_data.index)
            # Exclude VAR_feature_columns from X and add VAR_pca_df
            X = pd.concat([X.drop(columns=VAR_feature_columns), VAR_pca_df], axis=1)
            self.pca_columns = VAR_pca_df.columns.tolist()
        else:
            self.apply_pca = False
            self.pca_columns = None
        # Fit the model
        if self.x_scaled:
            self.x_scaler = StandardScaler()
            X_values = self.x_scaler.fit_transform(X).to_numpy().astype(float)
        else:
            self.x_scaler = EmptyTransformer()
            X_values = X.to_numpy().astype(float)

        if self.phi is None:
            if self.lambda_ is not None:
                I = np.eye(X_values.shape[1])
                self.beta = (
                    np.linalg.inv(X_values.T @ X_values + self.lambda_ * I)
                    @ X_values.T
                    @ Y_values
                )
            else:
                self.beta = (
                    np.linalg.pinv(X_values.T @ X_values) @ X_values.T @ Y_values
                )
            # Post-process coefficients to set small values to zero
            self.beta = self.apply_beta_threshold()
            # Calculate residuals
            Y_pred = X_values @ self.beta
            residuals = Y_values - Y_pred
            sse = np.sum(residuals**2, axis=0)
            n = Y_values.shape[0]
            p = X_values.shape[1]
            self.sigma = np.sqrt(sse / (n - p))
        else:
            # Time-varying estimation with forgetting factor phi
            n_samples, n_features = X_values.shape
            if self.max_cycles is not None:
                start_point = n_samples - int(self.max_cycles)
                start_point = start_point if start_point > 0 else 0
            else:
                start_point = 0
            n_targets = Y_values.shape[1]
            X_np = X_values
            Y_np = Y_values
            S = np.zeros((n_features, n_features))
            r = np.zeros((n_features, n_targets))
            alpha = self.lambda_
            if alpha is None:
                alpha = 0.0001
            for t in range(start_point, n_samples):
                X_t = X_np[t : t + 1].T  # Shape (n_features, 1)
                Y_t = Y_np[t : t + 1].T  # Shape (n_targets, 1)
                if t == 0:
                    S = X_t @ X_t.T
                    r = X_t @ Y_t.T
                else:
                    S = self.phi * S + X_t @ X_t.T
                    r = self.phi * r + X_t @ Y_t.T
                # Regularization term
                S_reg = S + alpha * np.eye(n_features)
                # try block has some cost if it fails routinely
                try:
                    theta_t = np.linalg.solve(S_reg.astype(float), r.astype(float))
                except np.linalg.LinAlgError:
                    theta_t = np.linalg.pinv(S_reg) @ r
            self.beta = theta_t  # Use the last theta_t as beta
            # Post-process coefficients to set small values to zero
            self.beta = self.apply_beta_threshold()
            # Calculate residuals
            Y_pred = X_np @ self.beta
            residuals = Y_np - Y_pred
            sse = np.sum(residuals**2, axis=0)
            n = Y_values.shape[0]
            p = X_values.shape[1]
            self.sigma = np.sqrt(sse / (n - p))
        self.fit_runtime = datetime.datetime.now() - self.startTime
        # Save variables for later
        self.X = X
        self.VAR_feature_columns = VAR_feature_columns
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        # Create external features for the forecast period
        x_s = date_part(
            test_index,
            method=self.datepart_method,
            set_index=True,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )
        x_t = changepoint_fcst_from_last_row(self.last_row, int(forecast_length))
        x_t.index = test_index
        X_ext = pd.concat([x_s, x_t], axis=1)
        if str(self.regression_type).lower() == "user":
            X_ext = pd.concat([X_ext, future_regressor.reindex(test_index)], axis=1)
        X_ext["constant"] = 1

        predictions = pd.DataFrame(
            index=test_index, columns=self.column_names, dtype=float
        )
        extended_df = pd.concat([self.var_history, predictions], axis=0)
        # post processing (these do cleanup, keep it a bit from going off the rails, which sometimes happens)
        if self.var_postprocessing:
            self.var_postprocessor = GeneralTransformer(
                n_jobs=self.n_jobs,
                holiday_country=self.holiday_country,
                verbose=self.verbose,
                random_seed=self.random_seed,
                # forecast_length=self.forecast_length,
                **self.var_postprocessing,
            )
        # For each date in forecast horizon
        x_pred = []
        for t, date in enumerate(test_index):
            # Create VAR features for date
            VAR_features_t = self.create_VAR_features(extended_df).loc[date:date]
            if VAR_features_t.isnull().values.any():
                # Attempt to fill NaN values from last available data
                VAR_features_t = VAR_features_t.ffill().bfill()
            VAR_features_t = VAR_features_t.astype(float)
            # Generate external features for date
            X_ext_t = X_ext.loc[date:date]
            X_t = pd.concat([X_ext_t, VAR_features_t], axis=1)
            # If PCA was applied, transform VAR_features
            self.X_t = X_t
            if self.apply_pca:
                VAR_data_t = X_t[self.VAR_feature_columns]
                VAR_pca_t = self.pca.transform(VAR_data_t)
                VAR_pca_df_t = pd.DataFrame(VAR_pca_t, index=VAR_data_t.index)
                self.pca_columns = VAR_pca_df_t.columns
                # Exclude VAR_feature_columns from X_t and add VAR_pca_df_t
                X_t = pd.concat(
                    [X_t.drop(columns=self.VAR_feature_columns), VAR_pca_df_t], axis=1
                )
            # Make prediction
            Y_pred_t = self.x_scaler.transform(X_t).to_numpy().astype(float) @ self.beta
            x_pred.append(X_t)
            # post processing
            if self.var_postprocessing:
                df = pd.DataFrame(Y_pred_t, index=[date], columns=self.column_names)
                self.var_postprocessor.fit(extended_df).inverse_transform(df)
                predictions.loc[date] = df.to_numpy()
            else:
                # Store prediction
                predictions.loc[date] = Y_pred_t.flatten()
            # Update extended_df with the new prediction
            extended_df.loc[date] = predictions.loc[date]

        # Save X_pred for process_components
        # Recreate VAR features for the entire forecast horizon
        self.X_pred = pd.concat(x_pred, axis=0)
        self.X_pred = self.X_pred.astype(float)
        if len(test_index) < 2 or (
            (self.trend_phi is None or self.trend_phi == 1)
            and (self.var_dampening is None or self.var_dampening == 1)
        ):
            pass
        else:
            components = np.einsum(
                'ij,jk->ijk', self.x_scaler.transform(self.X_pred).to_numpy(), self.beta
            )
            if self.trend_phi is not None and self.trend_phi != 1:
                req_len = len(test_index) - 1
                phi_series = pd.Series(
                    [self.trend_phi] * req_len,
                    index=test_index[1:],
                ).pow(range(req_len))
                trend_x_start = x_s.shape[1]
                trend_x_end = x_s.shape[1] + x_t.shape[1]
                trend_components = components[:, trend_x_start:trend_x_end, :]

                diff_array = np.diff(trend_components, axis=0)
                diff_scaled_array = (
                    diff_array * phi_series.to_numpy()[:, np.newaxis, np.newaxis]
                )
                first_row = trend_components[0:1, :]
                combined_array = np.vstack([first_row, diff_scaled_array])
                components[:, trend_x_start:trend_x_end, :] = np.cumsum(
                    combined_array, axis=0
                )

            if self.var_dampening is not None and self.var_dampening != 1:
                req_len = len(test_index) - 1
                phi_series = pd.Series(
                    [self.var_dampening] * req_len,
                    index=test_index[1:],
                ).pow(range(req_len))
                if self.apply_pca:
                    trend_x_start = -len(self.pca_columns)
                else:
                    trend_x_start = -len(self.VAR_feature_columns)
                trend_x_end = -1
                var_components = components[:, trend_x_start:trend_x_end, :]

                diff_array = np.diff(var_components, axis=0)
                diff_scaled_array = (
                    diff_array * phi_series.to_numpy()[:, np.newaxis, np.newaxis]
                )
                first_row = var_components[0:1, :]
                combined_array = np.vstack([first_row, diff_scaled_array])
                components[:, trend_x_start:trend_x_end, :] = np.cumsum(
                    combined_array, axis=0
                )

            predictions = pd.DataFrame(
                components.sum(axis=1), index=test_index, columns=self.column_names
            )

        # Convert forecast back to original scale
        forecast = predictions * self.scaler_std + self.scaler_mean
        if self.mode == 'multiplicative':
            forecast = np.exp(forecast)
        if just_point_forecast:
            return forecast
        else:
            z_value = norm.ppf(1 - (1 - self.prediction_interval) / 2)
            sigma_expanded = np.tile(self.sigma, (forecast_length, 1))
            margin_of_error = pd.DataFrame(
                z_value * sigma_expanded * self.scaler_std.values,
                columns=self.column_names,
                index=forecast.index,
            )
            if self.mode == 'multiplicative':
                margin_of_error = np.exp(margin_of_error)
            upper_forecast = forecast + margin_of_error
            lower_forecast = forecast - margin_of_error
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=forecast.index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast.astype(float),
                forecast=forecast.astype(float),
                upper_forecast=upper_forecast.astype(float),
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )
            return prediction

    def process_components(self):
        """Return components. Does not account for dampening."""
        # Process components for the forecast period
        res = []
        # Ensure X_pred is of type float
        X_pred_values = self.x_scaler.transform(self.X_pred).to_numpy().astype(float)
        components = np.einsum('ij,jk->ijk', X_pred_values, self.beta)
        for x in range(components.shape[2]):
            df = pd.DataFrame(
                components[:, :, x],
                index=self.X_pred.index,
                columns=self.X_pred.columns,
            )
            new_level = self.column_names[x]
            df.columns = pd.MultiIndex.from_product([[new_level], df.columns])
            res.append(df)
        components_df = pd.concat(res, axis=1)
        # If PCA was applied, transform back to original VAR features
        if self.apply_pca:
            # Get PCA components in the DataFrame
            res = []
            for col in self.column_names:
                # Get the component contribution
                # component_contributions = components_df.xs(col, level=1, axis=1)
                component_contributions = components_df[col][self.pca_columns]
                # Transform back to original VAR features
                original_contributions = (
                    component_contributions.to_numpy() @ self.pca.components_
                )
                # Create DataFrame with original VAR feature names
                original_contributions_df = pd.DataFrame(
                    original_contributions,
                    index=component_contributions.index,
                    columns=self.VAR_feature_columns,
                )
                # Remove PCA component from components_df
                # Add original VAR features contributions
                original_contributions_df.columns = [
                    x.split('___')[0] for x in original_contributions_df.columns
                ]
                new_cols = original_contributions_df.T.groupby(level=0).sum().T
                new_cols.columns = pd.MultiIndex.from_product([[col], new_cols.columns])
                res.append(new_cols)
                # for var_col in original_contributions_df.columns:
                #     idx = (col, var_col.split('___')[0])
                #     if idx not in components_df.columns:
                #         components_df[idx] = original_contributions_df[var_col]
                #     else:
                #         components_df[idx] = components_df[idx] + original_contributions_df[var_col]
            # drop all at once
            components_df = pd.concat(
                [components_df, pd.concat(res, axis=1)], axis=1
            ).reindex(self.column_names, level=0, axis=1)
            components_df = components_df.drop(columns=self.pca_columns, level=1)
        # Combine multiple lags into a single impact for each series
        # Sum over lags for each series
        final_components = {}
        for target_col in self.column_names:
            target_components = components_df[target_col]
            series_impacts = {}
            for col in target_components.columns:
                series_name = col.split('___')[0]
                series_impacts.setdefault(series_name, 0)
                series_impacts[series_name] += target_components[col]
            final_components[target_col] = pd.DataFrame(series_impacts)
        # Combine external features
        external_features = ['constant'] + self.seasonal_columns + self.trend_columns
        if str(self.regression_type).lower() == "user":
            external_features += list(self.regressor_columns)
        for target_col in self.column_names:
            target_components = components_df[target_col]
            for feature in external_features:
                if feature in target_components.columns:
                    if target_col not in final_components:
                        final_components[target_col] = pd.DataFrame()
                    final_components[target_col][feature] = target_components[feature]
        # Combine all components into a single DataFrame
        result = {}
        for target_col in final_components:
            df = final_components[target_col]
            # Scale back to original feature space
            df = df * self.scaler_std[target_col]  # + self.scaler_mean[target_col]
            if self.mode == 'multiplicative':
                df = np.exp(df)
            df.columns = pd.MultiIndex.from_product([[target_col], df.columns])
            result[target_col] = df
        components_df_final = pd.concat(result.values(), axis=1)
        return components_df_final

    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning"""
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices([None, 'User'], [0.8, 0.2])[0]
        if method == "deep":
            transformer_max_depth = 2
        else:
            transformer_max_depth = 1
        use_preprocess = random.choices([True, False], [0.5, 0.5])[0]
        if use_preprocess:
            var_preprocessing = RandomTransform(
                transformer_list=[
                    "ClipOutliers",
                    "bkfilter",
                    "AnomalyRemoval",
                    "FFTFilter",
                ],
                transformer_max_depth=transformer_max_depth,
                allow_none=False,
                fast_params=True,
            )
        else:
            var_preprocessing = False
        use_postprocess = random.choices([True, False], [0.2, 0.8])[0]
        if use_postprocess:
            var_postprocessing = RandomTransform(
                transformer_list=[
                    "HistoricValues",
                    "Constraint",
                    "AlignLastDiff",
                    "AlignLastValue",
                    "FIRFilter",
                    "Round",
                ],
                transformer_max_depth=transformer_max_depth,
                allow_none=False,
                fast_params=True,
            )
        else:
            var_postprocessing = False
        return {
            "datepart_method": random_datepart(method=method),
            "changepoint_spacing": random.choices(
                [None, 6, 28, 60, 90, 120, 180, 360, 5040],
                [0.1, 0.05, 0.1, 0.1, 0.1, 0.05, 0.2, 0.1, 0.2],
            )[0],
            "changepoint_distance_end": random.choices(
                [None, 6, 28, 60, 90, 180, 360, 520, 5040],
                [0.1, 0.05, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.2],
            )[0],
            "regression_type": regression_choice,
            "lags": random.choices(
                [None, [1], [7], [1, 2], [24], [1, 2, 3, 4, 5]],
                [0.4, 0.2, 0.3, 0.1, 0.05, 0.01],
            )[0],
            "rolling_means": random.choices(
                [None, [3], [4], [7], [4, 7], [28], [168]],
                [0.4, 0.05, 0.3, 0.2, 0.1, 0.05, 0.02],
            )[0],
            "lambda_": random.choices(
                [
                    None,
                    0.00001,
                    0.0001,
                    0.001,
                    0.01,
                    0.1,
                    1,
                    2,
                    10,
                    100,
                    1000,
                    10000,
                    50000,
                    100000,
                ],
                [
                    0.2,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.05,
                    0.01,
                ],
                k=1,
            )[0],
            "trend_phi": random.choices(
                [None, 0.995, 0.99, 0.98, 0.97, 0.8], [0.9, 0.05, 0.05, 0.1, 0.02, 0.01]
            )[0],
            "var_dampening": random.choices(
                [None, 0.999, 0.995, 0.99, 0.98, 0.97, 0.8],
                [0.9, 0.05, 0.05, 0.05, 0.1, 0.02, 0.01],
            )[0],
            "phi": random.choices(
                [None, 0.995, 0.99, 0.98, 0.97, 0.9, 0.8, 0.5, 0.2, 0.1],
                [0.75, 0.1, 0.05, 0.05, 0.02, 0.02, 0.01, 0.02, 0.01, 0.02],
            )[0],
            "max_cycles": random.choices([2000, 200, 10000], [0.8, 0.2, 0.01])[0],
            "apply_pca": random.choices([True, False], [0.5, 0.5])[0],
            "pca_n_components": random.choices(
                [None, 0.95, 0.9, 0.8, 10, "mle"], [0.2, 0.4, 0.2, 0.1, 0.1, 0.001]
            )[0],
            "base_scaled": random.choices([True, False], [0.4, 0.6])[0],
            "x_scaled": random.choices([True, False], [0.2, 0.8])[0],
            "var_preprocessing": var_preprocessing,
            "var_postprocessing": var_postprocessing,
            "threshold_value": random.choices(
                [None, 0.1, 0.01, 0.05, 0.001], [0.9, 0.025, 0.025, 0.025, 0.025]
            )[0],
            "mode": random.choices(["additive", "multiplicative"], [0.95, 0.05])[0],
            "holiday_countries_used": random.choices([True, False], [0.5, 0.5])[0],
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {
            "datepart_method": self.datepart_method,
            "changepoint_spacing": self.changepoint_spacing,
            "changepoint_distance_end": self.changepoint_distance_end,
            "regression_type": self.regression_type,
            "lags": self.lags,
            "rolling_means": self.rolling_means,
            "lambda_": self.lambda_,
            "trend_phi": self.trend_phi,
            "var_dampening": self.var_dampening,
            "phi": self.phi,
            "max_cycles": self.max_cycles,
            "apply_pca": self.apply_pca,
            "pca_n_components": self.pca_n_components,
            "base_scaled": self.base_scaled,
            "x_scaled": self.x_scaled,
            "var_preprocessing": self.var_preprocessing,
            "var_postprocessing": self.var_postprocessing,
            "threshold_value": self.threshold_value,
            "mode": self.mode,
            "holiday_countries_used": self.holiday_countries_used,
        }


class BallTreeRegressionMotif(ModelObject):
    """Forecasts using a nearest neighbors type model adapted for probabilistic time series.
    This version uses a feature ala MultivariateRegression but with motifs instead of a regression ML model.
    Many of these motifs will struggle when the forecast_length is large and history is short.

    Args:
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        n_jobs (int): how many parallel processes to run
        random_seed (int): used in selecting windows if max_windows is less than total available
        window (int): length of forecast history to match on
        point_method (int): how to summarize the nearest neighbors to generate the point forecast
            "weighted_mean", "mean", "median", "midhinge"
        distance_metric (str): all valid values for scipy cdist
        k (int): number of closest neighbors to consider
    """

    def __init__(
        self,
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = 1,
        window: int = 5,
        point_method: str = "mean",
        distance_metric: str = "canberra",
        k: int = 10,
        sample_fraction=None,
        comparison_transformation: dict = None,
        combination_transformation: dict = None,
        extend_df: bool = True,
        # multivar params
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
        probabilistic: bool = False,
        scale_full_X: bool = False,
        cointegration: str = None,
        cointegration_lag: int = None,
        series_hash: bool = False,
        frac_slice: float = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            "BallTreeRegressionMotif",
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.window = window
        self.point_method = point_method
        self.distance_metric = distance_metric
        self.k = k
        self.sample_fraction = sample_fraction
        self.comparison_transformation = comparison_transformation
        self.combination_transformation = combination_transformation
        self.extend_df = extend_df
        # multivar params
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
        self.regressor_train = None
        self.regressor_per_series_train = None
        self.static_regressor = None
        self.probabilistic = probabilistic
        self.scale_full_X = scale_full_X
        self.cointegration = cointegration
        self.cointegration_lag = cointegration_lag
        self.series_hash = series_hash
        self.frac_slice = frac_slice

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

        df = self.basic_profile(df)
        self.df = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=None,
        just_point_forecast=False,
        static_regressor=None,
        regressor_per_series=None,
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        # fit transform only, no need for inverse as this is only for finding windows
        if self.comparison_transformation is not None:
            self.comparison_transformer = GeneralTransformer(
                **self.comparison_transformation
            )
            compare_df = self.comparison_transformer.fit_transform(self.df)
        else:
            compare_df = self.df
        # applied once, then inversed after windows combined as forecast
        if self.combination_transformation is not None:
            self.combination_transformer = GeneralTransformer(
                **self.combination_transformation
            )
            wind_arr = self.combination_transformer.fit_transform(self.df)
        else:
            wind_arr = self.df

        ############################
        # fractional slicing to reduce size
        if self.frac_slice is not None:
            slice_size = int(self.df.shape[0] * self.frac_slice)
            self.slice_index = self.df.index[slice_size:]
        else:
            self.slice_index = None
        # handle regressor
        if self.regression_type is not None:
            cut_regr = self.regressor_train
            cut_regr.index = compare_df.index
        else:
            cut_regr = None

        parallel = True
        if self.n_jobs in [0, 1] or compare_df.shape[1] < 20:
            parallel = False
        elif not joblib_present:
            parallel = False
        # joblib multiprocessing to loop through series
        # this might be causing issues, TBD Key Error from Resource Tracker
        if parallel:
            self.Xa = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, timeout=36000)(
                delayed(rolling_x_regressor_regressor)(
                    compare_df[x_col].to_frame(),
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
                        self.static_regressor.loc[x_col].to_frame().T
                        if self.static_regressor is not None
                        else None
                    ),
                    cointegration=self.cointegration,
                    cointegration_lag=self.cointegration_lag,
                    series_id=x_col if self.series_hash else None,
                    slice_index=self.slice_index,
                    series_id_to_multiindex=x_col,
                )
                for x_col in compare_df.columns
            )
            self.Xa = pd.concat(self.Xa)
        else:
            self.Xa = pd.concat(
                [
                    rolling_x_regressor_regressor(
                        compare_df[x_col].to_frame(),
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
                            self.static_regressor.loc[x_col].to_frame().T
                            if self.static_regressor is not None
                            else None
                        ),
                        cointegration=self.cointegration,
                        cointegration_lag=self.cointegration_lag,
                        series_id=x_col if self.series_hash else None,
                        slice_index=self.slice_index,
                        series_id_to_multiindex=x_col,
                    )
                    for x_col in compare_df.columns
                ]
            )
        ############################
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        # filter because we need that last bit
        self.Xb = self.Xa[
            self.Xa.index.get_level_values(0) == self.Xa.index.get_level_values(0).max()
        ]
        # don't include a certain amount of the end as they won't have any usable history
        if self.window is not None:
            toss_bit = self.window
        else:
            toss_bit = int(forecast_length / 2)
        self.Xa = self.Xa[
            self.Xa.index.get_level_values(0).isin(
                self.Xa.index.get_level_values(0).unique().sort_values()[:-toss_bit]
            )
        ]  # int(self.forecast_length / 2)

        if self.distance_metric in ["euclidean", 'kdtree']:
            from scipy.spatial import KDTree

            # Build a KDTree for Xb
            tree = KDTree(self.Xa, leafsize=40)
        else:
            from sklearn.neighbors import BallTree

            tree = BallTree(self.Xa, metric=self.distance_metric)
            # Query the KDTree to find k nearest neighbors for each point in Xa
        A, self.windows = tree.query(self.Xb, k=self.k)

        # extend data to future to assure full length for windows, forward fill
        if self.extend_df:
            extension = pd.DataFrame(
                np.nan,
                index=pd.date_range(
                    start=wind_arr.index[-1],
                    periods=int(forecast_length / 2) + 1,
                    freq=self.frequency,
                )[1:],
                columns=self.column_names,
            )
            wind_arr = pd.concat([wind_arr, extension], axis=0).ffill()

        dt_array = self.Xa.index.get_level_values(0).values  # Datetime array
        series_array = self.Xa.index.get_level_values(1).values  # Series names array

        # Flatten windows array to work with 1D arrays
        n_series, k = self.windows.shape
        N = n_series * k
        dt_selected_flat = dt_array[self.windows.flatten()]
        series_selected_flat = series_array[self.windows.flatten()]

        # Find positions in df.index where dates are greater than selected datetimes
        pos_in_df_index = wind_arr.index.searchsorted(
            dt_selected_flat, side='right'
        )  # Shape: (N,)

        # Create positions for forecast_length ahead
        positions = (
            pos_in_df_index[:, None] + np.arange(forecast_length)[None, :]
        )  # Shape: (N, forecast_length)

        # Handle positions exceeding the length of df.index
        max_index = len(wind_arr.index)
        valid_positions = (positions >= 0) & (positions < max_index)

        # Map series names to column indices
        series_name_to_col_idx = {
            name: idx for idx, name in enumerate(wind_arr.columns)
        }
        col_indices = np.array(
            [series_name_to_col_idx[name] for name in series_selected_flat]
        )

        # Broadcast col_indices to match the shape of positions
        col_indices_broadcasted = np.repeat(col_indices, forecast_length).reshape(
            N, forecast_length
        )

        # Use advanced indexing to extract data
        data = np.full((N, forecast_length), np.nan)
        positions_flat = positions.flatten()
        col_indices_flat = col_indices_broadcasted.flatten()
        valid_mask_flat = valid_positions.flatten()

        # Extract valid data
        positions_flat_valid = positions_flat[valid_mask_flat]
        col_indices_flat_valid = col_indices_flat[valid_mask_flat]
        data_flat = data.flatten()
        data_flat[valid_mask_flat] = wind_arr.values[
            positions_flat_valid, col_indices_flat_valid
        ]

        # Reshape data back to original dimensions
        data = data_flat.reshape(N, forecast_length)
        # (k, forecast_length, n_series)
        self.result_windows = data.reshape(n_series, k, forecast_length).transpose(
            1, 2, 0
        )

        # now aggregate results into point and bound forecasts
        if self.point_method == "weighted_mean":
            weights = np.repeat(A.T[..., np.newaxis, :], forecast_length, axis=1)
            if weights.sum() == 0:
                weights = None
            forecast = np.average(self.result_windows, axis=0, weights=weights)
        elif self.point_method == "mean":
            forecast = np.nanmean(self.result_windows, axis=0)
        elif self.point_method == "median":
            forecast = np.nanmedian(self.result_windows, axis=0)
        elif self.point_method == "midhinge":
            q1 = nan_quantile(self.result_windows, q=0.25, axis=0)
            q2 = nan_quantile(self.result_windows, q=0.75, axis=0)
            forecast = (q1 + q2) / 2
        elif self.point_method == 'closest':
            # assumes the first K is the smallest distance (true when written)
            forecast = self.result_windows[0]

        pred_int = round((1 - self.prediction_interval) / 2, 5)
        upper_forecast = nan_quantile(self.result_windows, q=(1 - pred_int), axis=0)
        lower_forecast = nan_quantile(self.result_windows, q=pred_int, axis=0)

        forecast = pd.DataFrame(forecast, index=test_index, columns=self.column_names)
        lower_forecast = pd.DataFrame(
            lower_forecast, index=test_index, columns=self.column_names
        )
        upper_forecast = pd.DataFrame(
            upper_forecast, index=test_index, columns=self.column_names
        )
        if self.combination_transformation is not None:
            forecast = self.combination_transformer.inverse_transform(forecast)
            lower_forecast = self.combination_transformer.inverse_transform(
                lower_forecast
            )
            upper_forecast = self.combination_transformer.inverse_transform(
                upper_forecast
            )
        if just_point_forecast:
            return forecast
        else:
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=forecast.index,
                forecast_columns=forecast.columns,
                # so it's producing float32 but pandas is better with float64
                lower_forecast=lower_forecast.astype(float),
                forecast=forecast.astype(float),
                upper_forecast=upper_forecast.astype(float),
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning"""
        metric_list = [
            'braycurtis',
            'canberra',
            'chebyshev',
            'cityblock',
            'euclidean',
            'hamming',
            # 'mahalanobis',
            'minkowski',
            'kdtree',
        ]
        metric_probabilities = [
            0.05,
            0.05,
            0.05,
            0.05,
            0.9,
            0.05,
            # 0.05,
            0.05,
            0.05,
        ]
        if method != "deep":
            # evidence suggests 20 million can fit in 5 GB of RAM with a window of 28
            sample_fraction = random.choice([5000000, 50000000])
        else:
            sample_fraction = random.choice([0.2, 0.5, 100000000, None])
        if method == "event_risk":
            k_choice = random.choices(
                [10, 15, 20, 50, 100], [0.3, 0.1, 0.1, 0.05, 0.05]
            )[0]
        else:
            k_choice = random.choices(
                [1, 3, 5, 10, 15, 20, 100], [0.02, 0.2, 0.2, 0.5, 0.1, 0.1, 0.1]
            )[0]
        transformers_none = random.choices([True, False], [0.7, 0.3])[0]
        if transformers_none:
            comparison_transformation = None
            combination_transformation = None
        else:
            comparison_transformation = RandomTransform(
                transformer_list=superfast_transformer_dict,
                transformer_max_depth=1,
                allow_none=True,
            )
            combination_transformation = RandomTransform(
                transformer_list=superfast_transformer_dict,
                transformer_max_depth=1,
                allow_none=True,
            )
        # multivar params
        if method == "deep":
            window = random.choices(
                [None, 3, 7, 10, 14, 28], [0.2, 0.2, 0.05, 0.05, 0.05, 0.05]
            )[0]
            # random.choices([2, 3, 5, 7, 10, 14, 28, 60], [0.01, 0.01, 0.01, 0.1, 0.5, 0.1, 0.1, 0.01])[0]
        else:
            window = random.choices([None, 3, 7, 10], [0.3, 0.3, 0.1, 0.05])[0]
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

        return {
            "window": window,
            "point_method": random.choices(
                ["weighted_mean", "mean", "median", "midhinge", "closest"],
                [0.4, 0.2, 0.2, 0.2, 0.2],
            )[0],
            "distance_metric": random.choices(metric_list, metric_probabilities)[0],
            "k": k_choice,
            "sample_fraction": sample_fraction,
            "comparison_transformation": comparison_transformation,
            "combination_transformation": combination_transformation,
            "extend_df": random.choices([True, False], [True, False])[0],
            # multivar params
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
            'holiday': holiday_choice,
            'scale_full_X': random.choices([True, False], [0.2, 0.8])[0],
            "series_hash": random.choices([True, False], [0.5, 0.5])[0],
            "frac_slice": random.choices(
                [None, 0.8, 0.5, 0.2, 0.1], [0.6, 0.1, 0.1, 0.1, 0.1]
            )[0],
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {
            "window": self.window,
            "point_method": self.point_method,
            "distance_metric": self.distance_metric,
            "k": self.k,
            "sample_fraction": self.sample_fraction,
            "comparison_transformation": self.comparison_transformation,
            "combination_transformation": self.combination_transformation,
            "extend_df": self.extend_df,
            # multivar params
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
            'holiday': self.holiday,
            'scale_full_X': self.scale_full_X,
            "series_hash": self.series_hash,
            "frac_slice": self.frac_slice,
        }
