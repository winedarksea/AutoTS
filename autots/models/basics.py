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
from autots.tools import seasonal_int
from autots.tools.probabilistic import Point_to_Probability, historic_quantile
from autots.tools.window_functions import window_id_maker, sliding_window_view
from autots.tools.percentile import nan_quantile

# these are all optional packages
try:
    from scipy.spatial.distance import cdist
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
        **kwargs
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
        **kwargs
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
        method: str = 'Median',
        window: int = None,
        **kwargs
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
            ],
            [0.3, 0.3, 0.01, 0.1, 0.4, 0.1],
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
        **kwargs
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
                self.tile_values_lag_1 = df.groupby(level=0, axis=0).median()
            else:
                self.tile_values_lag_1 = df.groupby(level=0, axis=0).mean()
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
                        self.tile_values_lag_2 = df.groupby(level=0, axis=0).median()
                    else:
                        self.tile_values_lag_2 = df.groupby(level=0, axis=0).mean()
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
        **kwargs
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
            df = (
                df.pct_change(periods=1, fill_method='ffill')
                .tail(df.shape[0] - 1)
                .fillna(0)
            )
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
            rand_slice = df.iloc[
                z : (z + phrase_n),
            ]
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
            forecasts = pd.concat([forecasts, empty_frame], axis=0, sort=False).fillna(
                method='ffill'
            )
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
                ).fillna(method='ffill')
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
                ).fillna(method='ffill')
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
                    'kulsinski',
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

    A = cdist(Xa, Xb, metric=distance_metric)
    # lowest values
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

    pred_int = (1 - prediction_interval) / 2
    upper_forecast = nan_quantile(results, q=(1 - pred_int), axis=0)
    lower_forecast = nan_quantile(results, q=pred_int, axis=0)
    forecast = pd.Series(forecast)
    forecast.name = name
    upper_forecast = pd.Series(upper_forecast)
    upper_forecast.name = name
    lower_forecast = pd.Series(lower_forecast)
    lower_forecast.name = name
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
        **kwargs
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
            'kulsinski',
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
        ]
        if method == "event_risk":
            k_choice = random.choices(
                [10, 15, 20, 50, 100], [0.3, 0.1, 0.1, 0.05, 0.05]
            )[0]
        else:
            k_choice = random.choices(
                [3, 5, 10, 15, 20, 100], [0.2, 0.2, 0.5, 0.1, 0.1, 0.1]
            )[0]
        return {
            "window": random.choices(
                [3, 5, 7, 10, 14, 28, 60], [0.01, 0.01, 0.1, 0.5, 0.1, 0.1, 0.01]
            )[0],
            "point_method": random.choices(
                ["weighted_mean", "mean", "median", "midhinge"], [0.4, 0.2, 0.2, 0.2]
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
        **kwargs
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
        **kwargs
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

        # the regressor can be tacked on to provide (minor) influence to the distance metric
        if regression_type == "user":
            # here unlagging the regressor to match with history only
            full_regr = pd.concat([self.future_regressor, future_regressor], axis=0)
            full_regr = full_regr.tail(self.df.shape[0])
            full_regr.index = self.df.index
            array = pd.concat([self.df, full_regr], axis=1).to_numpy()
        else:
            array = self.df.to_numpy()
        tlt_len = array.shape[0]
        combined_window_size = window_size + forecast_length
        max_steps = array.shape[0] - combined_window_size
        window_idxs = window_id_maker(
            window_size=combined_window_size,
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
        results = array[window_idxs[res_idx, window_size:]]
        # reshape results to (num_windows, forecast_length, num_series)
        if results.ndim == 4:
            res_shape = results.shape
            results = results.reshape((res_shape[0], res_shape[2], res_shape[3]))
        if regression_type == "user":
            results = results[:, :, : self.df.shape[1]]
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
            # 'jensenshannon',
            'kulsinski',
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
        ]
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
            regression_choice = random.choices([None, "User"], [0.9, 0.1])[0]
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
        }
