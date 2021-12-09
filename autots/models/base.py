# -*- coding: utf-8 -*-
"""
Base model information

@author: Colin
"""
import warnings
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.metrics import (
    smape,
    mae,
    rmse,
    containment,
    contour,
    spl,
    medae,
    mean_absolute_differential_error,
)

# from sklearn.metrics import r2_score


class ModelObject(object):
    """Generic class for holding forecasting models.

    Models should all have methods:
        .fit(df, future_regressor = []) (taking a DataFrame with DatetimeIndex and n columns of n timeseries)
        .predict(forecast_length = int, future_regressor = [], just_point_forecast = False)
        .get_new_params() - return a dictionary of weighted random selected parameters

    Args:
        name (str): Model Name
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        n_jobs (int): used by some models that parallelize to multiple cores
    """

    def __init__(
        self,
        name: str = "Uninitiated Model Name",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        fit_runtime=datetime.timedelta(0),
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = -1,
    ):
        self.name = name
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.regression_type = regression_type
        self.fit_runtime = fit_runtime
        self.holiday_country = holiday_country
        self.random_seed = random_seed
        self.verbose = verbose
        self.verbose_bool = True if self.verbose > 1 else False
        self.n_jobs = n_jobs

    def __repr__(self):
        """Print."""
        return 'ModelObject of ' + self.name + ' uses standard .fit/.predict'

    def basic_profile(self, df):
        """Capture basic training details."""
        self.startTime = datetime.datetime.now()
        self.train_shape = df.shape
        self.column_names = df.columns
        self.train_last_date = df.index[-1]
        if self.frequency == 'infer':
            self.frequency = pd.infer_freq(df.index, warn=False)

        return df

    def create_forecast_index(self, forecast_length: int):
        """Generate a pd.DatetimeIndex appropriate for a new forecast.

        Warnings:
            Requires ModelObject.basic_profile() being called as part of .fit()
        """
        forecast_index = pd.date_range(
            freq=self.frequency, start=self.train_last_date, periods=forecast_length + 1
        )
        forecast_index = forecast_index[1:]
        self.forecast_index = forecast_index
        return forecast_index

    def get_params(self):
        """Return dict of current parameters."""
        return {}

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        return {}


class PredictionObject(object):
    """Generic class for holding forecast information.

    Attributes:
        model_name
        model_parameters
        transformation_parameters
        forecast
        upper_forecast
        lower_forecast

    Methods:
        long_form_results: return complete results in long form
        total_runtime: return runtime for all model components in seconds
        plot
        evaluate
    """

    def __init__(
        self,
        model_name: str = 'Uninitiated',
        forecast_length: int = 0,
        forecast_index=np.nan,
        forecast_columns=np.nan,
        lower_forecast=np.nan,
        forecast=np.nan,
        upper_forecast=np.nan,
        prediction_interval: float = 0.9,
        predict_runtime=datetime.timedelta(0),
        fit_runtime=datetime.timedelta(0),
        model_parameters={},
        transformation_parameters={},
        transformation_runtime=datetime.timedelta(0),
        per_series_metrics=np.nan,
        per_timestamp=np.nan,
        avg_metrics=np.nan,
        avg_metrics_weighted=np.nan,
        full_mae_error=None,
    ):
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.transformation_parameters = transformation_parameters
        self.forecast_length = forecast_length
        self.forecast_index = forecast_index
        self.forecast_columns = forecast_columns
        self.lower_forecast = lower_forecast
        self.forecast = forecast
        self.upper_forecast = upper_forecast
        self.prediction_interval = prediction_interval
        self.predict_runtime = predict_runtime
        self.fit_runtime = fit_runtime
        self.transformation_runtime = transformation_runtime
        # eval attributes
        self.per_series_metrics = per_series_metrics
        self.per_timestamp = per_timestamp
        self.avg_metrics = avg_metrics
        self.avg_metrics_weighted = avg_metrics_weighted
        self.full_mae_error = full_mae_error

    def __repr__(self):
        """Print."""
        if isinstance(self.forecast, pd.DataFrame):
            return "Prediction object: \nReturn .forecast, \n .upper_forecast, \n .lower_forecast \n .model_parameters \n .transformation_parameters"
        else:
            return "Empty prediction object."

    def __bool__(self):
        """bool version of class."""
        if isinstance(self.forecast, pd.DataFrame):
            return True
        else:
            return False

    def long_form_results(
        self,
        id_name="SeriesID",
        value_name="Value",
        interval_name='PredictionInterval',
        update_datetime_name=None,
    ):
        """Export forecasts (including upper and lower) as single 'long' format output

        Args:
            id_name (str): name of column containing ids
            value_name (str): name of column containing numeric values
            interval_name (str): name of column telling you what is upper/lower
            update_datetime_name (str): if not None, adds column with current timestamp and this name

        Returns:
            pd.DataFrame
        """
        try:
            upload = pd.melt(
                self.forecast,
                var_name=id_name,
                value_name=value_name,
                ignore_index=False,
            )
        except Exception:
            raise ImportError("Requires pandas>=1.1.0")
        upload[interval_name] = "50%"
        upload_upper = pd.melt(
            self.upper_forecast,
            var_name=id_name,
            value_name=value_name,
            ignore_index=False,
        )
        upload_upper[
            interval_name
        ] = f"{round(100 - ((1- self.prediction_interval)/2) * 100, 0)}%"
        upload_lower = pd.melt(
            self.lower_forecast,
            var_name=id_name,
            value_name=value_name,
            ignore_index=False,
        )
        upload_lower[
            interval_name
        ] = f"{round(((1- self.prediction_interval)/2) * 100, 0)}%"

        upload = pd.concat([upload, upload_upper, upload_lower], axis=0)
        if update_datetime_name is not None:
            upload[update_datetime_name] = datetime.datetime.utcnow()
        return upload

    def total_runtime(self):
        """Combine runtimes."""
        return self.fit_runtime + self.predict_runtime + self.transformation_runtime

    def plot(
        self,
        df_wide=None,
        series: str = None,
        ax=None,
        remove_zeroes: bool = False,
        start_date: str = None,
        **kwargs,
    ):
        """Generate an example plot of one series. Does not handle non-numeric forecasts.

        Args:
            df_wide (str): historic data for plotting actuals
            series (str): column name of series to plot. Random if None.
            ax: matplotlib axes to pass through to pd.plot()
            remove_zeroes (bool): if True, don't plot any zeroes
            start_date (str): Y-m-d string or Timestamp to remove all data before
            **kwargs passed to pd.DataFrame.plot()
        """
        if series is None:
            import random

            series = random.choice(self.forecast.columns)

        if df_wide is not None:
            plot_df = pd.DataFrame(
                {
                    series: df_wide[series],
                    'up_forecast': self.upper_forecast[series],
                    'low_forecast': self.lower_forecast[series],
                    'forecast': self.forecast[series],
                }
            )
        else:
            plot_df = pd.DataFrame(
                {
                    'up_forecast': self.upper_forecast[series],
                    'low_forecast': self.lower_forecast[series],
                    'forecast': self.forecast[series],
                }
            )
        if remove_zeroes:
            plot_df[plot_df == 0] = np.nan

        if start_date is not None:
            start_date = pd.to_datetime(start_date, infer_datetime_format=True)
            if plot_df.index.max() < pd.to_datetime(
                start_date, infer_datetime_format=True
            ):
                raise ValueError("start_date is more recent than all data provided")
            plot_df[plot_df.index >= start_date].plot(**kwargs)
        else:
            plot_df.plot(**kwargs)

    def evaluate(
        self,
        actual,
        series_weights: dict = None,
        df_train=None,
        per_timestamp_errors: bool = False,
        full_mae_error: bool = True,
    ):
        """Evalute prediction against test actual. Fills out attributes of base object.

        This fails with pd.NA values supplied.

        Args:
            actual (pd.DataFrame): dataframe of actual values of (forecast length * n series)
            series_weights (dict): key = column/series_id, value = weight
            df_train (pd.DataFrame): historical values of series, wide, used for setting scaler for SPL
                if None, actuals are used instead (suboptimal).
            per_timestamp (bool): whether to calculate and return per timestamp direction errors

        Returns:
            per_series_metrics (pandas.DataFrame): contains a column for each series containing accuracy metrics
            per_timestamp (pandas.DataFrame): smape accuracy for each timestamp, avg of all series
            avg_metrics (pandas.Series): average values of accuracy across all input series
            avg_metrics_weighted (pandas.Series): average values of accuracy across all input series weighted by series_weight, if given
            full_mae_errors (numpy.array): abs(actual - forecast)
        """
        A = np.array(actual)
        F = np.array(self.forecast)
        lower_forecast = np.array(self.lower_forecast)
        upper_forecast = np.array(self.upper_forecast)
        if df_train is None:
            df_train = A
        df_train = np.array(df_train)

        # check series_weights information
        if series_weights is None:
            from autots.tools.shaping import clean_weights

            series_weights = clean_weights(weights=False, series=self.forecast.columns)
        # make sure the series_weights are passed correctly to metrics
        if len(series_weights) != F.shape[1]:
            series_weights = {col: series_weights[col] for col in self.forecast.columns}

        # reuse this in several metrics
        self.full_mae_errors = abs(A - F)

        # calculate scaler once for spl
        scaler = np.nanmean(np.abs(np.diff(df_train[-100:], axis=0)), axis=0)
        fill_val = np.nanmax(scaler)
        fill_val = fill_val if fill_val > 0 else 1
        scaler[scaler == 0] = fill_val
        scaler[np.isnan(scaler)] = fill_val

        # concat most recent history to enable full-size diffs
        last_of_array = np.nan_to_num(df_train[df_train.shape[0] - 1: df_train.shape[0], ])
        lA = np.concatenate([last_of_array, A])
        lF = np.concatenate([last_of_array, F])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.per_series_metrics = pd.DataFrame(
                {
                    'smape': smape(A, F, self.full_mae_errors),
                    'mae': mae(self.full_mae_errors),
                    'rmse': rmse(self.full_mae_errors),
                    # 'medae': medae(self.full_mae_errors),
                    # r2 can't handle NaN in history, also uncomment import above
                    # 'r2': r2_score(A, F, multioutput="raw_values").flatten(),
                    # 'correlation': pd.DataFrame(A).corrwith(pd.DataFrame(F), drop=True).to_numpy(),
                    'made': mean_absolute_differential_error(lA, lF, 1, scaler=scaler),
                    'made_unscaled': mean_absolute_differential_error(lA, lF, 1),
                    # 'mad2e': mean_absolute_differential_error(lA, lF, 2),
                    'containment': containment(lower_forecast, upper_forecast, A),
                    'spl': spl(
                        A=A,
                        F=upper_forecast,
                        quantile=self.prediction_interval,
                        scaler=scaler,
                    )
                    + spl(
                        A=A,
                        F=lower_forecast,
                        quantile=(1 - self.prediction_interval),
                        scaler=scaler,
                    ),
                    'contour': contour(lA, lF),
                },
                index=actual.columns,
            ).transpose()

        if per_timestamp_errors:
            smape_df = abs(self.forecast - actual) / (abs(self.forecast) + abs(actual))
            weight_mean = np.mean(list(series_weights.values()))
            wsmape_df = (smape_df * series_weights) / weight_mean
            smape_cons = (np.nansum(wsmape_df, axis=1) * 200) / np.count_nonzero(
                ~np.isnan(actual), axis=1
            )
            per_timestamp = pd.DataFrame({'weighted_smape': smape_cons}).transpose()
            self.per_timestamp = per_timestamp

        # this weighting won't work well if entire metrics are NaN
        # but results should still be comparable
        self.avg_metrics_weighted = (self.per_series_metrics * series_weights).sum(
            axis=1, skipna=True
        ) / sum(series_weights.values())
        self.avg_metrics = self.per_series_metrics.mean(axis=1, skipna=True)
        return self
