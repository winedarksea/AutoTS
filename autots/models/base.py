# -*- coding: utf-8 -*-
"""
Base model information

@author: Colin
"""
import warnings
import datetime
import numpy as np
import pandas as pd
from autots.tools.shaping import infer_frequency, clean_weights
from autots.evaluator.metrics import (  # noqa
    smape,
    mae,
    rmse,
    containment,
    contour,
    spl,
    medae,
    mean_absolute_differential_error,
    msle,
    qae,
    mqae,
    mlvb,
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
        if 0 in df.shape:
            raise ValueError(f"{self.name} training dataframe has no data: {df.shape}")
        self.startTime = datetime.datetime.now()
        self.train_shape = df.shape
        self.column_names = df.columns
        self.train_last_date = df.index[-1]
        if self.frequency == 'infer':
            self.frequency = infer_frequency(df.index)

        return df

    def create_forecast_index(self, forecast_length: int):
        """Generate a pd.DatetimeIndex appropriate for a new forecast.

        Warnings:
            Requires ModelObject.basic_profile() being called as part of .fit()
        """
        if self.frequency == 'infer':
            raise ValueError(
                "create_forecast_index run without specific frequency, run basic_profile first or pass proper frequency to model init"
            )
        self.forecast_index = pd.date_range(
            freq=self.frequency, start=self.train_last_date, periods=forecast_length + 1
        )[
            1:
        ]  # note the disposal of the first (already extant) date
        return self.forecast_index

    def get_params(self):
        """Return dict of current parameters."""
        return {}

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        return {}


def apply_constraints(
    forecast,
    lower_forecast,
    upper_forecast,
    constraint_method,
    constraint_regularization,
    upper_constraint,
    lower_constraint,
    bounds,
    df_train=None,
):
    """Use constraint thresholds to adjust outputs by limit.
    Note that only one method of constraint can be used here, but if different methods are desired,
    this can be run twice, with None passed to the upper or lower constraint not being used.

    Args:
        forecast (pd.DataFrame): forecast df, wide style
        lower_forecast (pd.DataFrame): lower bound forecast df
            if bounds is False, upper and lower forecast dataframes are unused and can be empty
        upper_forecast (pd.DataFrame): upper bound forecast df
        constraint_method (str): one of
            stdev_min - threshold is min and max of historic data +/- constraint * st dev of data
            stdev - threshold is the mean of historic data +/- constraint * st dev of data
            absolute - input is array of length series containing the threshold's final value for each
            quantile - constraint is the quantile of historic data to use as threshold
        constraint_regularization (float): 0 to 1
            where 0 means no constraint, 1 is hard threshold cutoff, and in between is penalty term
        upper_constraint (float): or array, depending on method, None if unused
        lower_constraint (float): or array, depending on method, None if unused
        bounds (bool): if True, apply to upper/lower forecast, otherwise False applies only to forecast
        df_train (pd.DataFrame): required for quantile/stdev methods to find threshold values

    Returns:
        forecast, lower, upper (pd.DataFrame)
    """
    if constraint_method == "stdev_min":
        train_std = df_train.std(axis=0)
        if lower_constraint is not None:
            train_min = df_train.min(axis=0) - (lower_constraint * train_std)
        if upper_constraint is not None:
            train_max = df_train.max(axis=0) + (upper_constraint * train_std)
    elif constraint_method == "stdev":
        train_std = df_train.std(axis=0)
        train_mean = df_train.mean(axis=0)
        if lower_constraint is not None:
            train_min = train_mean - (lower_constraint * train_std)
        if upper_constraint is not None:
            train_max = train_mean + (upper_constraint * train_std)
    elif constraint_method == "absolute":
        train_min = lower_constraint
        train_max = upper_constraint
    elif constraint_method == "quantile":
        if lower_constraint is not None:
            train_min = df_train.quantile(lower_constraint, axis=0)
        if upper_constraint is not None:
            train_max = df_train.quantile(upper_constraint, axis=0)
    else:
        raise ValueError("constraint_method not recognized, adjust constraint")

    if constraint_regularization == 1:
        if lower_constraint is not None:
            forecast = forecast.clip(lower=train_min, axis=1)
        if upper_constraint is not None:
            forecast = forecast.clip(upper=train_max, axis=1)
        if bounds:
            if lower_constraint is not None:
                lower_forecast = lower_forecast.clip(lower=train_min, axis=1)
                upper_forecast = upper_forecast.clip(lower=train_min, axis=1)
            if upper_constraint is not None:
                lower_forecast = lower_forecast.clip(upper=train_max, axis=1)
                upper_forecast = upper_forecast.clip(upper=train_max, axis=1)
    else:
        if lower_constraint is not None:
            forecast.where(
                forecast >= train_min,
                forecast + (train_min - forecast) * constraint_regularization,
                inplace=True,
            )
        if upper_constraint is not None:
            forecast.where(
                forecast <= train_max,
                forecast + (train_max - forecast) * constraint_regularization,
                inplace=True,
            )
        if bounds:
            if lower_constraint is not None:
                lower_forecast.where(
                    lower_forecast >= train_min,
                    lower_forecast
                    + (train_min - lower_forecast) * constraint_regularization,
                    inplace=True,
                )
                upper_forecast.where(
                    upper_forecast >= train_min,
                    upper_forecast
                    + (train_min - upper_forecast) * constraint_regularization,
                    inplace=True,
                )
            if upper_constraint is not None:
                lower_forecast.where(
                    lower_forecast <= train_max,
                    lower_forecast
                    + (train_max - lower_forecast) * constraint_regularization,
                    inplace=True,
                )

                upper_forecast.where(
                    upper_forecast <= train_max,
                    upper_forecast
                    + (train_max - upper_forecast) * constraint_regularization,
                    inplace=True,
                )
    return forecast, lower_forecast, upper_forecast


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
        apply_constraints
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
        model=None,
        transformer=None,
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
        # model attributes, not normally used
        self.model = model
        self.transformer = transformer

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
        scaler=None,
    ):
        """Evalute prediction against test actual. Fills out attributes of base object.

        This fails with pd.NA values supplied.

        Args:
            actual (pd.DataFrame): dataframe of actual values of (forecast length * n series)
            series_weights (dict): key = column/series_id, value = weight
            df_train (pd.DataFrame): historical values of series, wide,
                used for setting scaler for SPL
                necessary for MADE and Contour if forecast_length == 1
                if None, actuals are used instead (suboptimal).
            per_timestamp (bool): whether to calculate and return per timestamp direction errors

        Returns:
            per_series_metrics (pandas.DataFrame): contains a column for each series containing accuracy metrics
            per_timestamp (pandas.DataFrame): smape accuracy for each timestamp, avg of all series
            avg_metrics (pandas.Series): average values of accuracy across all input series
            avg_metrics_weighted (pandas.Series): average values of accuracy across all input series weighted by series_weight, if given
            full_mae_errors (numpy.array): abs(actual - forecast)
            scaler (numpy.array): precomputed scaler for efficiency, avg value of series in order of columns
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
            series_weights = clean_weights(weights=False, series=self.forecast.columns)
        # make sure the series_weights are passed correctly to metrics
        if len(series_weights) != F.shape[1]:
            series_weights = {col: series_weights[col] for col in self.forecast.columns}

        # reuse this in several metrics so precalculate
        full_errors = F - A
        self.full_mae_errors = np.abs(full_errors)
        self.squared_errors = full_errors**2
        log_errors = np.log1p(self.full_mae_errors)

        # calculate scaler once
        if scaler is None:
            scaler = np.nanmean(np.abs(np.diff(df_train[-100:], axis=0)), axis=0)
            fill_val = np.nanmax(scaler)
            fill_val = fill_val if fill_val > 0 else 1
            scaler[scaler == 0] = fill_val
            scaler[np.isnan(scaler)] = fill_val

        # concat most recent history to enable full-size diffs
        last_of_array = np.nan_to_num(df_train[-1:, :])
        lA = np.concatenate([last_of_array, A])
        lF = np.concatenate([last_of_array, F])

        # np.where(A >= F, quantile * (A - F), (1 - quantile) * (F - A))
        inv_prediction_interval = 1 - self.prediction_interval
        upper_diff = A - upper_forecast
        self.upper_pl = np.where(
            A >= upper_forecast,
            self.prediction_interval * upper_diff,
            inv_prediction_interval * -1 * upper_diff,
        )
        # note that the quantile here is the lower quantile
        low_diff = A - lower_forecast
        self.lower_pl = np.where(
            A >= lower_forecast,
            inv_prediction_interval * low_diff,
            self.prediction_interval * -1 * low_diff,
        )

        # test for NaN, this allows faster calculations if no nan
        nan_flag = np.isnan(np.min(full_errors))

        # mage = np.nansum(full_errors, axis=None) / A.shape[1]
        if nan_flag:
            mage = np.nanmean(np.abs(np.nansum(full_errors, axis=1)))
        else:
            mage = np.mean(np.abs(np.sum(full_errors, axis=1)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.per_series_metrics = pd.DataFrame(
                {
                    'smape': smape(A, F, self.full_mae_errors, nan_flag=nan_flag),
                    'mae': mae(self.full_mae_errors),
                    'rmse': rmse(self.squared_errors),
                    'made': mean_absolute_differential_error(lA, lF, 1, scaler=scaler),
                    'mage': mage,
                    'mle': msle(
                        full_errors, self.full_mae_errors, log_errors, nan_flag=nan_flag
                    ),
                    'imle': msle(
                        -full_errors,
                        self.full_mae_errors,
                        log_errors,
                        nan_flag=nan_flag,
                    ),
                    'spl': spl(
                        self.upper_pl + self.lower_pl,
                        scaler=scaler,
                    ),
                    'containment': containment(lower_forecast, upper_forecast, A),
                    'contour': contour(lA, lF),
                    # maximum error point
                    'maxe': np.nanmax(self.full_mae_errors, axis=0),  # TAKE MAX for AGG
                    # origin directional accuracy
                    'oda': np.nansum(
                        np.sign(F - last_of_array) == np.sign(A - last_of_array), axis=0
                    )
                    / F.shape[0],
                    # mean of values less than 85th percentile of error
                    'mqae': mqae(self.full_mae_errors, q=0.85, nan_flag=nan_flag),
                    # 90th percentile of error
                    # here for NaN, assuming that NaN to zero only has minor effect on upper quantile
                    # 'qae': qae(self.full_mae_errors, q=0.9, nan_flag=nan_flag),
                    # mean % last value naive baseline, smaller is better
                    # 'mlvb': mlvb(A=A, F=F, last_of_array=last_of_array),
                    # median absolute error
                    # 'medae': medae(self.full_mae_errors, nan_flag=nan_flag),  # median
                    # variations on the mean absolute differential error
                    # 'made_unscaled': mean_absolute_differential_error(lA, lF, 1),
                    # 'mad2e': mean_absolute_differential_error(lA, lF, 2),
                    # r2 can't handle NaN in history, also uncomment import above
                    # 'r2': r2_score(A, F, multioutput="raw_values").flatten(),
                    # 'correlation': pd.DataFrame(A).corrwith(pd.DataFrame(F), drop=True).to_numpy(),
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

    def apply_constraints(
        self,
        constraint_method="quantile",
        constraint_regularization=0.5,
        upper_constraint=1.0,
        lower_constraint=0.0,
        bounds=True,
        df_train=None,
    ):
        """Use constraint thresholds to adjust outputs by limit.
        Note that only one method of constraint can be used here, but if different methods are desired,
        this can be run twice, with None passed to the upper or lower constraint not being used.

        Args:
            constraint_method (str): one of
                stdev_min - threshold is min and max of historic data +/- constraint * st dev of data
                stdev - threshold is the mean of historic data +/- constraint * st dev of data
                absolute - input is array of length series containing the threshold's final value for each
                quantile - constraint is the quantile of historic data to use as threshold
            constraint_regularization (float): 0 to 1
                where 0 means no constraint, 1 is hard threshold cutoff, and in between is penalty term
            upper_constraint (float): or array, depending on method, None if unused
            lower_constraint (float): or array, depending on method, None if unused
            bounds (bool): if True, apply to upper/lower forecast, otherwise False applies only to forecast
            df_train (pd.DataFrame): required for quantile/stdev methods to find threshold values

        Returns:
            self
        """
        self.forecast, self.lower_forecast, self.upper_forecast = apply_constraints(
            self.forecast,
            self.lower_forecast,
            self.upper_forecast,
            constraint_method,
            constraint_regularization,
            upper_constraint,
            lower_constraint,
            bounds,
            df_train,
        )
        return self
