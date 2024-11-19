"""
Constraint generation functions
"""

import random
import numpy as np
import pandas as pd

from autots.tools.impute import FillNA


def constant_growth_rate(periods, final_growth):
    """Take a final target growth rate (ie 2 % over a year) and convert to a daily (etc) value."""
    # Convert final growth rate percentage to a growth factor
    final_growth_factor = 1 + final_growth

    # Calculate the daily growth factor required to achieve the final growth factor in the given days
    daily_growth_factor = final_growth_factor ** (1 / periods)

    # Generate an array of day indices
    day_indices = np.arange(1, periods + 1)

    # Calculate the cumulative growth factor for each day
    cumulative_growth_factors = daily_growth_factor**day_indices

    # Calculate the perceived growth rates relative to the starting value
    perceived_growth_rates = cumulative_growth_factors - 1
    return perceived_growth_rates


constraint_method_dict = {
    "quantile": 0.1,
    "stdev": 0.1,
    "stdev_min": 0.1,
    "last_window": 0.1,
    "slope": 0.1,
    "absolute": 0.1,
    "fixed": 0,
    "historic_growth": 0.1,
    "historic_diff": 0.05,
    "dampening": 0.1,
    "round": 0.05,
}


def constraint_new_params(method: str = "fast"):
    method_choice = random.choices(
        list(constraint_method_dict.keys()), list(constraint_method_dict.values())
    )[0]
    params = {
        "constraint_method": method_choice,
        "constraint_direction": random.choices(["upper", "lower"], [0.5, 0.5])[0],
        "constraint_regularization": random.choices(
            [1.0, 0.2, 0.5, 0.7, 0.9], [0.7, 0.05, 0.1, 0.05, 0.1]
        )[0],
    }
    if method_choice == "quantile":
        if params["constraint_direction"] == "upper":
            params["constraint_value"] = random.choices(
                [1.0, 0.5, 0.7, 0.9, 0.98], [0.2, 0.2, 0.1, 0.2, 0.1]
            )[0]
        else:
            params["constraint_value"] = random.choices(
                [0.01, 0.5, 0.1, 0.2, 0.02], [0.2, 0.2, 0.1, 0.2, 0.1]
            )[0]
    elif method_choice == "slope":
        if params["constraint_direction"] == "upper":
            params["constraint_value"] = random.choices(
                [
                    {
                        "slope": 0.02,
                        "window": 10,
                        "window_agg": "max",
                        "threshold": 0.01,
                    },
                    {
                        "slope": 0.05,
                        "window": 10,
                        "window_agg": "max",
                        "threshold": 0.01,
                    },
                    {
                        "slope": 0.1,
                        "window": 30,
                        "window_agg": "max",
                        "threshold": 0.01,
                    },
                    {
                        "slope": 0.2,
                        "window": 10,
                        "window_agg": "mean",
                        "threshold": 0.01,
                    },
                    {
                        "slope": 0.001,
                        "window": 10,
                        "window_agg": "max",
                        "threshold": 0.1,
                    },
                ],
                [0.2, 0.2, 0.1, 0.2, 0.1],
            )[0]
        else:
            params["constraint_value"] = random.choices(
                [
                    {
                        "slope": -0.02,
                        "window": 7,
                        "window_agg": "min",
                        "threshold": -0.01,
                    },
                    {
                        "slope": -0.05,
                        "window": 10,
                        "window_agg": "min",
                        "threshold": -0.01,
                    },
                    {
                        "slope": -0.1,
                        "window": 30,
                        "window_agg": "min",
                        "threshold": 0.01,
                    },
                    {
                        "slope": -0.2,
                        "window": 10,
                        "window_agg": "mean",
                        "threshold": -0.01,
                    },
                    {
                        "slope": -0.001,
                        "window": 10,
                        "window_agg": "min",
                        "threshold": -0.1,
                    },
                ],
                [0.5, 0.2, 0.1, 0.2, 0.1],
            )[0]
    elif method_choice == "last_window":
        if params["constraint_direction"] == "upper":
            params["constraint_value"] = random.choices(
                [
                    {"window": 10, "threshold": 0.1},
                    {"window": 10, "threshold": 0.2},
                    {"window": 20, "threshold": 0.1},
                    {"window": 364, "threshold": 0.01},
                    {"window": 10, "threshold": -0.01},
                ],
                [0.5, 0.2, 0.1, 0.2, 0.1],
            )[0]
        else:
            params["constraint_value"] = random.choices(
                [
                    {"window": 10, "threshold": -0.1},
                    {"window": 28, "threshold": -0.1},
                    {"window": 364, "threshold": -0.01},
                    {"window": 10, "threshold": -0.1},
                    {"window": 10, "threshold": 0.01},
                ],
                [0.5, 0.2, 0.1, 0.2, 0.1],
            )[0]
    elif method_choice in ["stdev", "stdev_min"]:
        params["constraint_value"] = random.choices(
            [1.0, 0.5, 2.0, 3.0, 4.0], [0.2, 0.2, 0.1, 0.2, 0.1]
        )[0]
    elif method_choice in ["dampening"]:
        params["constraint_value"] = random.choices(
            [0.99, 0.9, 0.8, 0.999, 0.98, 0.9999, 0.95],
            [0.5, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1],
        )[0]
        params['constraint_direction'] = "upper"
        params["constraint_regularization"] = 1.0
    elif method_choice in ["round"]:
        params["constraint_value"] = random.choices([0, 2], [0.4, 0.6])[0]
        # not used, just setting to fixed
        params['constraint_direction'] = "upper"
        params["constraint_regularization"] = 1.0
    elif method_choice in ["absolute", "fixed"]:
        params["constraint_value"] = random.choices([0, 0.1, 1], [0.8, 0.1, 0.1])[0]
    elif method_choice in ["historic_growth"]:
        params["constraint_value"] = {
            "threshold": random.choices([1.0, 0.5, 2.0, 0.2], [0.6, 0.2, 0.2, 0.04])[0]
        }
        window_choice = random.choices(
            [None, 10, 100, 360, 4], [0.8, 0.2, 0.2, 0.2, 0.04]
        )[0]
        if window_choice is not None:
            params["constraint_value"]["window"] = window_choice
        quantile_choice = random.choices(
            [None, 0.99, 0.9, 0.98, 0.75], [0.6, 0.2, 0.2, 0.2, 0.04]
        )[0]
        if quantile_choice is not None:
            params["constraint_value"]["quantile"] = quantile_choice
    elif method_choice in ["historic_diff"]:
        params["constraint_value"] = random.choices(
            [1.0, 0.5, 2.0, 0.2, 1.2],
            [0.6, 0.2, 0.2, 0.04, 0.04],
        )[0]
    return params


def fit_constraint(
    constraint_method,
    constraint_value,
    constraint_direction='upper',
    constraint_regularization=1.0,
    bounds=True,
    df_train=None,
    forecast_length=None,
):
    # check if training data provided
    if df_train is None and constraint_method in [
        "quantile",
        "stdev",
        "stdev_min",
        "last_window",
        "slope",
    ]:
        raise ValueError("this constraint requires df_train to be provided")
    # set direction
    lower_constraint = None
    upper_constraint = None
    if constraint_direction == "lower":
        lower_constraint = True
    elif constraint_direction == "upper":
        upper_constraint = True
    else:
        raise ValueError(f"constraint_direction: {constraint_direction} invalid")
    train_min = None
    train_max = None
    if constraint_method == "stdev_min":
        train_std = df_train.std(axis=0)
        if lower_constraint is not None:
            train_min = df_train.min(axis=0) - (constraint_value * train_std)
        if upper_constraint is not None:
            train_max = df_train.max(axis=0) + (constraint_value * train_std)
    elif constraint_method == "stdev":
        train_std = df_train.std(axis=0)
        train_mean = df_train.mean(axis=0)
        if lower_constraint is not None:
            train_min = train_mean - (constraint_value * train_std)
        if upper_constraint is not None:
            train_max = train_mean + (constraint_value * train_std)
    elif constraint_method in ["absolute", "fixed"]:
        train_min = constraint_value
        train_max = constraint_value
    elif constraint_method == "quantile":
        if lower_constraint is not None:
            train_min = df_train.quantile(constraint_value, axis=0)
        if upper_constraint is not None:
            train_max = df_train.quantile(constraint_value, axis=0)
    elif constraint_method == "last_window":
        if isinstance(constraint_value, dict):
            window = constraint_value.get("window", 3)
            window_agg = constraint_value.get("window_agg", "mean")
            if upper_constraint is not None:
                threshold = constraint_value.get("threshold", 0.05)
            else:
                threshold = constraint_value.get("threshold", -0.05)
        else:
            window = 1
            window_agg = "mean"
            threshold = constraint_value
        if window_agg == "mean":
            end_o_data = df_train.iloc[-window:].mean(axis=0)
        elif window_agg == "max":
            end_o_data = df_train.iloc[-window:].max(axis=0)
        elif window_agg == "min":
            end_o_data = df_train.iloc[-window:].min(axis=0)
        else:
            raise ValueError(f"constraint window_agg not recognized: {window_agg}")
        train_min = train_max = end_o_data + end_o_data * threshold
    elif constraint_method == "slope":
        if isinstance(constraint_value, dict):
            window = constraint_value.get("window", 1)
            window_agg = constraint_value.get("window_agg", "mean")
            slope = constraint_value.get("slope", 0.05)
            threshold = constraint_value.get("threshold", None)
        else:
            window = 1
            window_agg = "mean"
            slope = constraint_value
            threshold = None
        # slope is given as a final max growth, NOT compounding
        changes = constant_growth_rate(forecast_length, slope)
        if window_agg == "mean":
            end_o_data = df_train.iloc[-window:].mean(axis=0)
        elif window_agg == "max":
            end_o_data = df_train.iloc[-window:].max(axis=0)
        elif window_agg == "min":
            end_o_data = df_train.iloc[-window:].min(axis=0)
        else:
            raise ValueError(f"constraint window_agg not recognized: {window_agg}")
        # have the slope start above a threshold to allow more volatility
        if threshold is not None:
            end_o_data = end_o_data + end_o_data * threshold
        train_min = train_max = (
            end_o_data.to_numpy()
            + end_o_data.to_numpy()[np.newaxis, :] * changes[:, np.newaxis]
        )
    elif constraint_method == "historic_growth":
        if isinstance(constraint_value, dict):
            window = constraint_value.get("window", forecast_length)
            threshold = constraint_value.get("threshold", 1.0)
            quantile = constraint_value.get("quantile", None)
        else:
            window = forecast_length
            threshold = constraint_value
            quantile = None
        if window is None:
            window = forecast_length
        rolling_diff = df_train.diff(periods=window)
        # calculate the growth rates (slopes) by dividing the differences by the window size
        slopes = rolling_diff / window
        if quantile is not None:
            slopes_max = slopes.quantile(quantile).to_numpy()
            slopes_min = slopes.quantile(1 - quantile).to_numpy()
        else:
            slopes_max = slopes.max().to_numpy()
            slopes_min = slopes.min().to_numpy()
        # find the maximum growth rate and maximum decline rate for each series
        # and apply a log growth rate to that (to better allow for peaks like holidays)
        t = np.arange(forecast_length + 1).reshape(-1, 1)
        t2 = np.log1p(t) + 1
        train_max = (
            df_train.iloc[-1].to_numpy()
            + (((slopes_max * forecast_length)) * t2 / t2.max())[1:] * threshold
        )
        train_min = (
            df_train.iloc[-1].to_numpy()
            + (((slopes_min * forecast_length)) * t2 / t2.max())[1:] * threshold
        )
    elif constraint_method == "historic_diff":
        if isinstance(constraint_value, dict):
            threshold = constraint_value.get("threshold", 1.0)
        else:
            threshold = constraint_value
        rolling_diff = df_train.diff(periods=forecast_length)

        train_max = df_train.iloc[-1] + rolling_diff.max() * threshold
        train_min = df_train.iloc[-1] + rolling_diff.min() * threshold
    elif constraint_method == "dampening":
        pass
    elif constraint_method == "round":
        pass
    else:
        raise ValueError(
            f"constraint_method {constraint_method} not recognized, adjust constraint"
        )
    return lower_constraint, upper_constraint, train_min, train_max


def apply_fit_constraint(
    forecast,
    lower_forecast,
    upper_forecast,
    constraint_method,
    constraint_value,
    constraint_direction='upper',
    constraint_regularization=1.0,
    bounds=True,
    lower_constraint=None,
    upper_constraint=None,
    train_min=None,
    train_max=None,
    fillna=None,  # only used with regularization of 1 / None
):
    if constraint_method == "dampening":
        # the idea is to make the forecast plateau by gradually forcing the step to step change closer to zero
        trend_phi = constraint_value
        if trend_phi is not None and trend_phi != 1 and forecast.shape[0] > 2:
            req_len = forecast.shape[0] - 1
            phi_series = pd.Series(
                [trend_phi] * req_len,
                index=forecast.index[1:],
            ).pow(range(req_len))

            # adjust all by same margin
            forecast = pd.concat(
                [forecast.iloc[0:1], forecast.diff().iloc[1:].mul(phi_series, axis=0)]
            ).cumsum()

            if bounds:
                lower_forecast = pd.concat(
                    [
                        lower_forecast.iloc[0:1],
                        lower_forecast.diff().iloc[1:].mul(phi_series, axis=0),
                    ]
                ).cumsum()
                upper_forecast = pd.concat(
                    [
                        upper_forecast.iloc[0:1],
                        upper_forecast.diff().iloc[1:].mul(phi_series, axis=0),
                    ]
                ).cumsum()
        return forecast, lower_forecast, upper_forecast
    elif constraint_method == "round":
        decimals = int(constraint_value)
        if bounds:
            return (
                forecast.round(decimals=decimals),
                lower_forecast.round(decimals=decimals),
                upper_forecast.round(decimals=decimals),
            )
        else:
            return forecast.round(decimals=decimals), lower_forecast, upper_forecast
    if constraint_regularization == 1 or constraint_regularization is None:
        if fillna in [None, "None", "none", ""]:
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
            # if FILLNA present, don't clip but replace with NaN then fill NaN
            if lower_constraint is not None:
                forecast = forecast.where(
                    forecast >= train_min,
                    np.nan,
                )
            if upper_constraint is not None:
                forecast = forecast.where(
                    forecast <= train_max,
                    np.nan,
                )
            if bounds:
                if lower_constraint is not None:
                    lower_forecast = lower_forecast.where(
                        lower_forecast >= train_min,
                        np.nan,
                    )
                    upper_forecast = upper_forecast.where(
                        upper_forecast >= train_min,
                        np.nan,
                    )
                if upper_constraint is not None:
                    lower_forecast = lower_forecast.where(
                        lower_forecast <= train_max,
                        np.nan,
                    )

                    upper_forecast = upper_forecast.where(
                        lower_forecast <= train_max, np.nan
                    )
            forecast = FillNA(forecast, method=str(fillna), window=10)
            if bounds:
                lower_forecast = FillNA(lower_forecast, method=str(fillna), window=10)
                upper_forecast = FillNA(upper_forecast, method=str(fillna), window=10)
    else:
        if lower_constraint is not None:
            forecast = forecast.where(
                forecast >= train_min,
                forecast + (train_min - forecast) * constraint_regularization,
            )
        if upper_constraint is not None:
            forecast = forecast.where(
                forecast <= train_max,
                forecast + (train_max - forecast) * constraint_regularization,
            )
        if bounds:
            if lower_constraint is not None:
                lower_forecast = lower_forecast.where(
                    lower_forecast >= train_min,
                    lower_forecast
                    + (train_min - lower_forecast) * constraint_regularization,
                )
                upper_forecast = upper_forecast.where(
                    upper_forecast >= train_min,
                    upper_forecast
                    + (train_min - upper_forecast) * constraint_regularization,
                )
            if upper_constraint is not None:
                lower_forecast = lower_forecast.where(
                    lower_forecast <= train_max,
                    lower_forecast
                    + (train_max - lower_forecast) * constraint_regularization,
                )

                upper_forecast = upper_forecast.where(
                    upper_forecast <= train_max,
                    upper_forecast
                    + (train_max - upper_forecast) * constraint_regularization,
                )
    return forecast, lower_forecast, upper_forecast


def apply_constraint_single(
    forecast,
    lower_forecast,
    upper_forecast,
    constraint_method,
    constraint_value,
    constraint_direction='upper',
    constraint_regularization=1.0,
    bounds=True,
    df_train=None,
):
    # note the Constraint Transformer also uses the same API so adjust changes there too
    lower_constraint, upper_constraint, train_min, train_max = fit_constraint(
        constraint_method=constraint_method,
        constraint_value=constraint_value,
        constraint_direction=constraint_direction,
        constraint_regularization=constraint_regularization,
        bounds=bounds,
        df_train=df_train,
        forecast_length=forecast.shape[0],
    )
    return apply_fit_constraint(
        forecast=forecast,
        lower_forecast=lower_forecast,
        upper_forecast=upper_forecast,
        constraint_method=constraint_method,
        constraint_value=constraint_value,
        constraint_direction=constraint_direction,
        constraint_regularization=constraint_regularization,
        bounds=bounds,
        lower_constraint=lower_constraint,
        upper_constraint=upper_constraint,
        train_min=train_min,
        train_max=train_max,
    )
