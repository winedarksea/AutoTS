# -*- coding: utf-8 -*-
"""Generate probabilities of forecastings crossing limit thresholds.
Created on Thu Jan 27 13:36:18 2022
"""
import random
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import model_forecast, back_forecast
from autots.evaluator.auto_ts import AutoTS


def set_limit_forecast(
    df_train,
    forecast_length,
    model_name="SeasonalNaive",
    model_param_dict={
        'method': "median",
        "lag_1": 28,
        "lag_2": None,
    },
    model_transform_dict={
        'fillna': 'nearest',
        'transformations': {},
        'transformation_params': {},
    },
    prediction_interval=0.9,
    frequency='infer',
    model_forecast_kwargs={
        "verbose": 1,
        "n_jobs": "auto",
        "random_seed": 321,
    },
    future_regressor_train=None,
    future_regressor_forecast=None,
):
    """Helper function for forecast limits set by forecast algorithms."""
    forecasts = model_forecast(
        model_name=model_name,
        model_param_dict=model_param_dict,
        model_transform_dict=model_transform_dict,
        df_train=df_train,
        forecast_length=forecast_length,
        frequency=frequency,
        prediction_interval=prediction_interval,
        future_regressor_train=future_regressor_train,
        future_regressor_forecast=future_regressor_forecast,
        return_model=True,
        **model_forecast_kwargs,
    )
    return forecasts.upper_forecast.values, forecasts.lower_forecast.values


def set_limit_forecast_historic(
    df_train,
    forecast_length,
    model_name="SeasonalNaive",
    model_param_dict={
        'method': "median",
        "lag_1": 28,
        "lag_2": None,
    },
    model_transform_dict={
        'fillna': 'nearest',
        'transformations': {},
        'transformation_params': {},
    },
    prediction_interval=0.9,
    frequency='infer',
    model_forecast_kwargs={
        "verbose": 2,
        "n_jobs": "auto",
        "random_seed": 321,
    },
    future_regressor_train=None,
    future_regressor_forecast=None,
    eval_periods=None,
):
    """Helper function for forecast limits set by forecast algorithms."""
    forecasts = back_forecast(
        df=df_train,
        n_splits="auto",
        model_name=model_name,
        model_param_dict=model_param_dict,
        model_transform_dict=model_transform_dict,
        forecast_length=forecast_length,
        frequency=frequency,
        prediction_interval=prediction_interval,
        future_regressor_train=future_regressor_train,
        eval_periods=eval_periods,
        **model_forecast_kwargs,
    )
    return forecasts.upper_forecast.values, forecasts.lower_forecast.values


class EventRiskForecast(object):
    """Generate a risk score (0 to 1, but usually close to 0) for a future event exceeding user specified upper or lower bounds.

    Upper and lower limits can be one of four types, and may each be different.
    1. None (no risk score calculated for this direction)
    2. Float in range [0, 1] historic quantile of series (which is historic min and max at edges) is chosen as limit.
    3. A dictionary of {"model_name": x,  "model_param_dict": y, "model_transform_dict": z, "prediction_interval": 0.9} to generate a forecast as the limits
        Primarily intended for simple forecasts like SeasonalNaive, but can be used with any AutoTS model
    4. a custom input numpy array of shape (forecast_length, num_series)

    This can be used to find the "middle" limit too, flip so upper=lower and lower=upper, then abs(U - (1 - L)).
    In some cases it may help to drop the results from the first forecast timestep or two.

    This functions by generating multiple outcome forecast possiblities in two ways.
    If a 'Motif' type model is passed, it uses all the k neighbors motifs as outcome paths (recommended)
    All other AutoTS models will generate the possible outcomes by utilizing multiple prediction_intervals (more intervals = slower but more resolution).
    The risk score is then the % of outcome forecasts which cross the limit.
    (less than or equal for lower, greater than or equal for upper)

    Only accepts `wide` style dataframe input.
    Methods are class_methods and can be used standalone. They default to __init__ inputs, but can be overriden.
    Results are usually a numpy array of shape (forecast_length, num_series)

    Args:
        df_train (pd.DataFrame): `wide style data, pd.DatetimeIndex for index and one series per column
        forecast_length (int): number of forecast steps to make
        frequency (str): frequency of timesteps
        prediction_interval (float): float or list of floats for probabilistic forecasting
            if a list, the first item in the list is the one used for .fit default
        model_forecast_kwargs (dict): AutoTS kwargs to pass to generaet_result_windows, .fit_forecast, and forecast-style limits
        model_name, model_param_dict, model_transform_dict: for model_forecast in generate_result_windows
        future_regressor_train, future_regressor_forecast: regressor arrays if used

    Methods:
        fit
        predict
        predict_historic
        generate_result_windows
        generate_risk_array
        generate_historic_risk_array
        set_limit
        plot

    Attributes:
        result_windows, forecast_df, up_forecast_df, low_forecast_df
        lower_limit_2d, upper_limit_2d, upper_risk_array, lower_risk_array
    """

    def __init__(
        self,
        df_train,
        forecast_length,
        frequency: str = "infer",
        prediction_interval=0.9,
        lower_limit=0.05,
        upper_limit=0.95,
        model_name="UnivariateMotif",
        model_param_dict={
            'window': 14,
            "pointed_method": "median",
            "distance_metric": "euclidean",
            "k": 10,
            "return_result_windows": True,
        },
        model_transform_dict={
            'fillna': 'pchip',
            'transformations': {
                "0": "Slice",
                "1": "DifferencedTransformer",
                "2": "RollingMeanTransformer",
                "3": "MaxAbsScaler",
            },
            'transformation_params': {
                "0": {"method": 0.5},
                "1": {},
                "2": {"fixed": False, "window": 7},
                "3": {},
            },
        },
        model_forecast_kwargs={
            "max_generations": 30,
            "verbose": 1,
            "n_jobs": "auto",
            "random_seed": 321,
        },
        future_regressor_train=None,
        future_regressor_forecast=None,
    ):
        self.name = "EventRiskForecast"
        self.df_train = df_train
        self.forecast_length = forecast_length
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.model_forecast_kwargs = model_forecast_kwargs
        self.model_name = model_name
        self.model_param_dict = model_param_dict
        self.model_transform_dict = model_transform_dict
        self.future_regressor_train = future_regressor_train
        self.future_regressor_forecast = future_regressor_forecast
        self.outcome_shape = (forecast_length, df_train.shape[1])
        self.outcome_columns = df_train.columns
        self.outcome_index = None
        self.result_windows = None

    def __repr__(self):
        """Print."""
        return f"{self.name} object with window generating model: {self.model_name, self.model_param_dict, self.model_transform_dict}"

    def fit(
        self,
        df_train=None,
        forecast_length=None,
        prediction_interval=None,
        models_mode="event_risk",
        model_list=["UnivariateMotif", "MultivariateMotif", "SectionalMotif"],
        ensemble=None,
        autots_kwargs=None,
        future_regressor_train=None,
    ):
        """Shortcut for generating model params.

        args specified are those suggested for an otherwise normal AutoTS run

        Args:
            df_train (pd.DataFrame): wide style only
            model_method (str): event_risk here is used by motif models
            model_list (list): suggesting the use of motif models
            ensemble (list): must be None or empty list to get motif result windows
            autots_kwargs (dict): all other args passed in as kwargs
                if None, defaults to class model_forecast_kwargs, for blank pass empty dict
        """
        autots_kwargs = (
            self.model_forecast_kwargs if autots_kwargs is None else autots_kwargs
        )
        future_regressor_train = (
            self.future_regressor_train
            if future_regressor_train is None
            else future_regressor_train
        )
        prediction_interval = (
            self.prediction_interval
            if prediction_interval is None
            else prediction_interval
        )
        if isinstance(prediction_interval, list):
            prediction_interval = prediction_interval[0]
        model = AutoTS(
            forecast_length=self.forecast_length
            if forecast_length is None
            else forecast_length,
            prediction_interval=prediction_interval,
            models_mode=models_mode,
            model_list=model_list,
            ensemble=ensemble,
            **autots_kwargs,
        ).fit(
            self.df_train if df_train is None else df_train,
            future_regressor=future_regressor_train,
        )
        self.model_name = model.best_model_name
        self.model_param_dict = model.best_model_params
        self.model_transform_dict = model.best_model_transformation_params
        return self

    def generate_result_windows(
        self,
        df_train=None,
        forecast_length=None,
        frequency=None,
        prediction_interval=None,
        model_name=None,
        model_param_dict=None,
        model_transform_dict=None,
        model_forecast_kwargs=None,
        future_regressor_train=None,
        future_regressor_forecast=None,
    ):
        """For event risk forecasting. Params default to class init but can be overridden here.

        Returns:
            result_windows (numpy.array): (num_samples/k, forecast_length, num_series/columns)
        """
        df_train = self.df_train if df_train is None else df_train
        forecast_length = (
            self.forecast_length if forecast_length is None else forecast_length
        )
        prediction_interval = (
            self.prediction_interval
            if prediction_interval is None
            else prediction_interval
        )
        frequency = self.frequency if frequency is None else frequency
        model_name = self.model_name if model_name is None else model_name
        model_param_dict = (
            self.model_param_dict if model_param_dict is None else model_param_dict
        )
        model_transform_dict = (
            self.model_transform_dict
            if model_transform_dict is None
            else model_transform_dict
        )
        model_forecast_kwargs = (
            self.model_forecast_kwargs
            if model_forecast_kwargs is None
            else model_forecast_kwargs
        )
        future_regressor_train = (
            self.future_regressor_train
            if future_regressor_train is None
            else future_regressor_train
        )
        future_regressor_forecast = (
            self.future_regressor_forecast
            if future_regressor_forecast is None
            else future_regressor_forecast
        )

        all_motif_list = [
            "UnivariateMotif",
            "MultivariateMotif",
            "SectionalMotif",
            "Motif",
        ]
        diff_window_motif_list = ["UnivariateMotif", "MultivariateMotif", "Motif"]
        if model_name in all_motif_list:
            if isinstance(prediction_interval, list):
                prediction_interval = prediction_interval[0]
            if model_name in diff_window_motif_list:
                model_param_dict = {
                    **model_param_dict,
                    **{"return_result_windows": True},
                }
            forecasts = model_forecast(
                model_name=model_name,
                model_param_dict=model_param_dict,
                model_transform_dict=model_transform_dict,
                df_train=df_train,
                forecast_length=forecast_length,
                frequency=frequency,
                prediction_interval=prediction_interval,
                future_regressor_train=future_regressor_train,
                future_regressor_forecast=future_regressor_forecast,
                return_model=True,
                **model_forecast_kwargs,
            )
            result_windows = forecasts.model.result_windows
            if model_name in diff_window_motif_list:
                result_windows = np.moveaxis(
                    np.array(list(result_windows.values())), 0, -1
                )
            transformed_array = []
            for samp in result_windows:
                transformed_array.append(
                    forecasts.transformer.inverse_transform(
                        pd.DataFrame(
                            samp,
                            index=forecasts.forecast.index,
                            columns=forecasts.forecast.columns,
                        )
                    )
                )
            result_windows = np.array(transformed_array)
            lower_forecast = forecasts.lower_forecast
            upper_forecast = forecasts.upper_forecast
        else:
            if isinstance(prediction_interval, float):
                prediction_interval = list(set([prediction_interval, 0.95, 0.8, 0.5]))
            result_windows_list = []
            point_included = False
            for interval in prediction_interval:
                forecasts = model_forecast(
                    model_name=model_name,
                    model_param_dict=model_param_dict,
                    model_transform_dict=model_transform_dict,
                    df_train=df_train,
                    forecast_length=forecast_length,
                    frequency=frequency,
                    prediction_interval=interval,
                    future_regressor_train=future_regressor_train,
                    future_regressor_forecast=future_regressor_forecast,
                    return_model=True,
                    **model_forecast_kwargs,
                )
                if not point_included:
                    result_windows_list.append(forecasts.forecast)
                    lower_forecast = forecasts.lower_forecast
                    upper_forecast = forecasts.upper_forecast
                    point_included = True
                result_windows_list.append(forecasts.upper_forecast)
                result_windows_list.append(forecasts.lower_forecast)
            result_windows = np.array(result_windows_list)
        return result_windows, forecasts.forecast, upper_forecast, lower_forecast

    @staticmethod
    def set_limit(
        limit,
        target_shape,
        df_train,
        direction="upper",
        period="forecast",
        forecast_length=None,
        eval_periods=None,
    ):
        """Handles all limit input styles and returns numpy array.

        Args:
            limit: see class overview for input options
            target_shape (tuple): of (forecast_length, num_series)
            df_train (pd.DataFrame): training data
            direction (str): whether it is the "upper" or "lower" limit
            period (str): "forecast" or "historic" only used for limits defined by forecast algorithm params
            forecast_length (int): needed only for historic of forecast algorithm defined limit
            eval_periods (int): only for historic forecast limit, only runs on the tail n (this) of data
        """
        # handle a predefined array
        if isinstance(limit, np.ndarray):
            assert (
                limit.ndim == 2
            ), f"{direction}_limit, if array, must be 2d np array of shape forecast_length, n_series"
            assert (
                limit.shape == target_shape
            ), f"{direction}_limit, if array, must be 2d np array of shape forecast_length, n_series"
            return limit
        # handle a limit as a quantile defined by float
        elif isinstance(limit, float) or isinstance(limit, int):
            assert (
                limit >= 0 and limit <= 1
            ), f"{direction}_limit if float must be in the range [0, 1], received {limit}"
            return np.repeat(
                np.nanquantile(df_train, limit, axis=0).reshape(1, -1),
                target_shape[0],
                axis=0,
            )
        # handle None
        elif limit is None:
            return None
        # handle a limit defined by a forecast algorithm
        elif isinstance(limit, dict):
            if period == "historic":
                upper, lower = set_limit_forecast_historic(
                    df_train=df_train,
                    forecast_length=forecast_length,
                    model_name=limit.get("model_name", "SeasonalNaive"),
                    model_param_dict=limit.get("model_param_dict", {}),
                    model_transform_dict=limit.get("model_transform_dict", {}),
                    prediction_interval=limit.get("prediction_interval", 0.9),
                    frequency=limit.get("frequency", 'infer'),
                    model_forecast_kwargs=limit.get(
                        "model_forecast_kwargs",
                        {
                            "verbose": 1,
                            "n_jobs": "auto",
                            "random_seed": 321,
                        },
                    ),
                    future_regressor_train=limit.get("future_regressor_train", None),
                    future_regressor_forecast=limit.get(
                        "future_regressor_forecast", None
                    ),
                    eval_periods=eval_periods,
                )
            else:
                upper, lower = set_limit_forecast(
                    df_train=df_train,
                    forecast_length=target_shape[0],
                    model_name=limit.get("model_name", "SeasonalNaive"),
                    model_param_dict=limit.get("model_param_dict", {}),
                    model_transform_dict=limit.get("model_transform_dict", {}),
                    prediction_interval=limit.get("prediction_interval", 0.9),
                    frequency=limit.get("frequency", 'infer'),
                    model_forecast_kwargs=limit.get(
                        "model_forecast_kwargs",
                        {
                            "verbose": 1,
                            "n_jobs": "auto",
                            "random_seed": 321,
                        },
                    ),
                    future_regressor_train=limit.get("future_regressor_train", None),
                    future_regressor_forecast=limit.get(
                        "future_regressor_forecast", None
                    ),
                )
            if direction == "upper":
                return upper
            elif direction == "lower":
                return lower
            elif direction == "both":
                return upper, lower
        else:
            raise ValueError(
                f"{direction}_limit was not recognized dtype, input was {limit}"
            )

    @staticmethod
    def generate_risk_array(result_windows, limit, direction="upper"):
        """Given a df and a limit, returns a 0/1 array of whether limit was equaled or exceeded."""
        if direction == "upper":
            return (result_windows >= limit).astype(int).sum(
                axis=0
            ) / result_windows.shape[0]
        elif direction == "lower":
            return (result_windows <= limit).astype(int).sum(
                axis=0
            ) / result_windows.shape[0]
        else:
            raise ValueError(
                f"arg `direction`: {direction} not recognized in generate_risk_array"
            )

    @staticmethod
    def generate_historic_risk_array(df, limit, direction="upper"):
        """Given a df and a limit, returns a 0/1 array of whether limit was equaled or exceeded."""
        if direction == "upper":
            return (df >= limit).astype(int)
        elif direction == "lower":
            return (df <= limit).astype(int)
        else:
            raise ValueError(
                f"arg `direction`: {direction} not recognized in generate_risk_array"
            )

    def predict(self):
        """Returns forecast upper, lower risk probability arrays for input limits."""
        self.upper_limit_2d = self.set_limit(
            self.upper_limit, self.outcome_shape, self.df_train, direction="upper"
        )
        self.lower_limit_2d = self.set_limit(
            self.lower_limit, self.outcome_shape, self.df_train, direction="lower"
        )
        if self.upper_limit_2d is None and self.lower_limit_2d is None:
            raise ValueError(
                "both upper and lower limits are None, at least one must be specified"
            )

        (
            self.result_windows,
            self.forecast_df,
            self.up_forecast_df,
            self.low_forecast_df,
        ) = self.generate_result_windows()
        self.outcome_index = self.forecast_df.index

        self.upper_risk_array = None
        if self.upper_limit_2d is not None:
            self.upper_risk_array = self.generate_risk_array(
                self.result_windows, self.upper_limit_2d, direction="upper"
            )

        self.lower_risk_array = None
        if self.lower_limit_2d is not None:
            self.lower_risk_array = self.generate_risk_array(
                self.result_windows, self.lower_limit_2d, direction="lower"
            )
        return pd.DataFrame(
            self.upper_risk_array,
            columns=self.outcome_columns,
            index=self.outcome_index,
        ), pd.DataFrame(
            self.lower_risk_array,
            columns=self.outcome_columns,
            index=self.outcome_index,
        )

    def predict_historic(self, upper_limit=None, lower_limit=None, eval_periods=None):
        """Returns upper, lower risk probability arrays for input limits for the historic data.
        If manual numpy array limits are used, the limits will need to be appropriate shape (for df_train and eval_periods if used)

        Args:
            upper_limit: if different than the version passed to init
            lower_limit: if different than the version passed to init
            eval_periods (int): only assess the n most recent periods of history
        """
        upper_limit = self.upper_limit if upper_limit is None else upper_limit
        lower_limit = self.lower_limit if lower_limit is None else lower_limit
        if eval_periods is not None:
            target_shape = (eval_periods, self.df_train.shape[1])
            train_df = self.df_train.tail(eval_periods)
        else:
            target_shape = self.df_train.shape
            train_df = self.df_train
        self.historic_upper_limit_2d = self.set_limit(
            upper_limit,
            target_shape,
            self.df_train,
            direction="upper",
            period="historic",
            forecast_length=self.forecast_length,
            eval_periods=eval_periods,
        )
        self.historic_lower_limit_2d = self.set_limit(
            lower_limit,
            target_shape,
            self.df_train,
            direction="lower",
            period="historic",
            forecast_length=self.forecast_length,
            eval_periods=eval_periods,
        )
        if (
            self.historic_upper_limit_2d is None
            and self.historic_lower_limit_2d is None
        ):
            raise ValueError(
                "both upper and lower limits are None, at least one must be specified"
            )

        self.historic_upper_risk_array = None
        if self.historic_upper_limit_2d is not None:
            self.historic_upper_risk_array = self.generate_historic_risk_array(
                train_df, self.historic_upper_limit_2d, direction="upper"
            )

        self.historic_lower_risk_array = None
        if self.historic_lower_limit_2d is not None:
            self.historic_lower_risk_array = self.generate_historic_risk_array(
                train_df, self.historic_lower_limit_2d, direction="lower"
            )

        return pd.DataFrame(
            self.historic_upper_risk_array,
            columns=self.outcome_columns,
            index=train_df.index,
        ), pd.DataFrame(
            self.historic_lower_risk_array,
            columns=self.outcome_columns,
            index=train_df.index,
        )

    def plot(
        self,
        column_idx=0,
        grays=[
            "#838996",
            "#c0c0c0",
            "#dcdcdc",
            "#a9a9a9",
            "#808080",
            "#989898",
            "#808080",
            "#757575",
            "#696969",
            "#c9c0bb",
            "#c8c8c8",
            "#323232",
            "#e5e4e2",
            "#778899",
            "#4f666a",
            "#848482",
            "#414a4c",
            "#8a7f80",
            "#c4c3d0",
            "#bebebe",
            "#dbd7d2",
        ],
        up_low_color=["#ff4500", "#ff5349"],
        bar_color="#6495ED",
        bar_ylim=[0.0, 0.5],
        figsize=(14, 8),
        result_windows=None,
        lower_limit_2d=None,
        upper_limit_2d=None,
        upper_risk_array=None,
        lower_risk_array=None,
    ):
        """Plot a sample of the risk forecast outcomes.

        Args:
            column_idx (int): positional index of series to sample for plot
            grays (list of str): list of hex codes for colors for the potential forecasts
            up_low_colors (list of str): two hex code colors for lower and upper
            bar_color (str): hex color for bar graph
            bar_ylim (list): passed to ylim of plot, sets scale of axis of barplot
            figsize (tuple): passed to figsize of output figure
        """
        import matplotlib.pyplot as plt

        result_windows = (
            self.result_windows if result_windows is None else result_windows
        )
        lower_limit_2d = (
            self.lower_limit_2d if lower_limit_2d is None else lower_limit_2d
        )
        upper_limit_2d = (
            self.upper_limit_2d if upper_limit_2d is None else upper_limit_2d
        )
        upper_risk_array = (
            self.upper_risk_array if upper_risk_array is None else upper_risk_array
        )
        lower_risk_array = (
            self.lower_risk_array if lower_risk_array is None else lower_risk_array
        )

        column = self.outcome_columns[column_idx]
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 1]}, figsize=figsize
        )
        fig.suptitle(f'{column} Event Risk Forecasting')
        # index=pd.date_range("2022-01-01", periods=result_windows.shape[1], freq="D")
        plot_df = pd.DataFrame(result_windows[:, :, column_idx].T, self.outcome_index)
        if lower_limit_2d is not None:
            plot_df['lower_limit'] = lower_limit_2d[
                :, column_idx
            ]  # np.nanquantile(df, 0.6, axis=0)[column_idx]
        else:
            plot_df['lower_limit'] = np.nan
        if upper_limit_2d is not None:
            plot_df['upper_limt'] = upper_limit_2d[
                :, column_idx
            ]  # np.nanquantile(df, 0.85, axis=0)[column_idx]
        else:
            plot_df['upper_limt'] = np.nan
        colors = random.choices(grays, k=plot_df.shape[1] - 2) + up_low_color
        plot_df.plot(color=colors, ax=ax1, legend=False)
        # handle one being None
        try:
            up_risk = upper_risk_array[:, column_idx]
        except Exception:
            up_risk = 0
        try:
            low_risk = lower_risk_array[:, column_idx]
        except Exception:
            low_risk = 0
        plot_df["upper & lower risk"] = up_risk + low_risk
        # #0095a4   #FA9632  # 3264C8   #6495ED
        plot_df["upper & lower risk"].plot(
            kind="bar",
            xticks=[],
            title="Combined Risk Score",
            ax=ax2,
            color=bar_color,
            ylim=bar_ylim,
        )

    def plot_eval(
        self,
        df_test,
        column_idx=0,
        actuals_color=["#00BFFF"],
        up_low_color=["#ff4500", "#ff5349"],
        bar_color="#6495ED",
        bar_ylim=[0.0, 0.5],
        figsize=(14, 8),
        lower_limit_2d=None,
        upper_limit_2d=None,
        upper_risk_array=None,
        lower_risk_array=None,
    ):
        """Plot a sample of the risk forecast with known value vs risk score.

        Args:
            df_test (pd.DataFrame): dataframe of known values (dt index, series)
            column_idx (int): positional index of series to sample for plot
            actuals_color (list of str): list of one hex code for line of known actuals
            up_low_colors (list of str): two hex code colors for lower and upper
            bar_color (str): hex color for bar graph
            bar_ylim (list): passed to ylim of plot, sets scale of axis of barplot
            figsize (tuple): passed to figsize of output figure
        """
        import matplotlib.pyplot as plt

        lower_limit_2d = (
            self.lower_limit_2d if lower_limit_2d is None else lower_limit_2d
        )
        upper_limit_2d = (
            self.upper_limit_2d if upper_limit_2d is None else upper_limit_2d
        )
        upper_risk_array = (
            self.upper_risk_array if upper_risk_array is None else upper_risk_array
        )
        lower_risk_array = (
            self.lower_risk_array if lower_risk_array is None else lower_risk_array
        )
        col = self.outcome_columns[column_idx]
        plot_df = df_test[col].to_frame()
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 1]}, figsize=figsize
        )
        fig.suptitle(f'{col} Event Risk Forecasting Evaluation')
        # index=pd.date_range("2022-01-01", periods=result_windows.shape[1], freq="D")
        if lower_limit_2d is not None:
            plot_df['lower_limit'] = lower_limit_2d[
                :, column_idx
            ]  # np.nanquantile(df, 0.6, axis=0)[column_idx]
        else:
            plot_df['lower_limit'] = np.nan
        if upper_limit_2d is not None:
            plot_df['upper_limt'] = upper_limit_2d[
                :, column_idx
            ]  # np.nanquantile(df, 0.85, axis=0)[column_idx]
        else:
            plot_df['upper_limt'] = np.nan
        colors = actuals_color + up_low_color
        plot_df.plot(color=colors, ax=ax1, legend=False)
        # handle one being None
        try:
            up_risk = upper_risk_array[:, column_idx]
        except Exception:
            up_risk = 0
        try:
            low_risk = lower_risk_array[:, column_idx]
        except Exception:
            low_risk = 0
        plot_df["upper & lower risk"] = up_risk + low_risk
        plot_df["upper & lower risk"].plot(
            kind="bar",
            xticks=[],
            title="Risk Score",
            ax=ax2,
            color=bar_color,
            ylim=bar_ylim,
        )
