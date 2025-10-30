# -*- coding: utf-8 -*-
"""
Base model information

@author: Colin
"""
import json
import random
import warnings
import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict
from autots.tools.constraint import apply_constraint_single
from autots.tools.shaping import infer_frequency, clean_weights
from autots.evaluator.metrics import full_metric_evaluation
from autots.tools.plotting import plot_distributions, plot_forecast_with_intervals
import matplotlib.pyplot as plt


DEFAULT_ALIGN_LAST_VALUE_PARAMS = {
    "rows": 1,
    "lag": 1,
    "method": "additive",
    "strength": 1.0,
    "first_value_only": False,
    "threshold": 3,
    "threshold_method": "max",
}


def create_forecast_index(frequency, forecast_length, train_last_date, last_date=None):
    if frequency == 'infer':
        raise ValueError(
            "create_forecast_index run without specific frequency, run basic_profile first or pass proper frequency to model init"
        )
    return pd.date_range(
        freq=frequency,
        start=train_last_date if last_date is None else last_date,
        periods=int(forecast_length + 1),
    )[
        1:
    ]  # note the disposal of the first (already extant) date


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

    def create_forecast_index(self, forecast_length: int, last_date=None):
        """Generate a pd.DatetimeIndex appropriate for a new forecast.

        Warnings:
            Requires ModelObject.basic_profile() being called as part of .fit()
        """

        return create_forecast_index(
            self.frequency, forecast_length, self.train_last_date, last_date
        )

    def get_params(self):
        """Return dict of current parameters."""
        return {}

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        return {}

    def fit_data(self, df, future_regressor=None):
        self.basic_profile(df)
        if future_regressor is not None:
            self.regressor_train = future_regressor
        return self

    @staticmethod
    def time():
        return datetime.datetime.now()


def apply_constraints(
    forecast,
    lower_forecast,
    upper_forecast,
    constraints=None,
    df_train=None,
    # old args
    constraint_method=None,
    constraint_regularization=None,
    upper_constraint=None,
    lower_constraint=None,
    bounds=True,
):
    """Use constraint thresholds to adjust outputs by limit.

    Args:
        forecast (pd.DataFrame): forecast df, wide style
        lower_forecast (pd.DataFrame): lower bound forecast df
            if bounds is False, upper and lower forecast dataframes are unused and can be empty
        upper_forecast (pd.DataFrame): upper bound forecast df
        constraints (list): list of dictionaries of constraints to apply
            keys: "constraint_method" (same as below, old args), "constraint_regularization", "constraint_value", "constraint_direction" (upper/lower), bounds
        df_train (pd.DataFrame): required for quantile/stdev methods to find threshold values
        # old args
        constraint_method (str): one of
            stdev_min - threshold is min and max of historic data +/- constraint * st dev of data
            stdev - threshold is the mean of historic data +/- constraint * st dev of data
            absolute - input is array of length series containing the threshold's final value for each
            quantile - constraint is the quantile of historic data to use as threshold
            last_window - certain percentage above and below the last n data values
            slope - cannot exceed a certain growth rate from last historical value
        constraint_regularization (float): 0 to 1
            where 0 means no constraint, 1 is hard threshold cutoff, and in between is penalty term
        upper_constraint (float): or array, depending on method, None if unused
        lower_constraint (float): or array, depending on method, None if unused
        bounds (bool): if True, apply to upper/lower forecast, otherwise False applies only to forecast

    Returns:
        forecast, lower, upper (pd.DataFrame)
    """
    # handle old style
    if constraint_method is not None:
        if constraints is not None:
            raise ValueError(
                f"both constraint_method (old way) and constraints (new way) args passed, this will not work. Constraints: {constraints}"
            )
        else:
            constraints = []
            if upper_constraint is not None:
                constraints.append(
                    {
                        "constraint_method": constraint_method,
                        "constraint_value": upper_constraint,
                        "constraint_direction": "upper",
                        "constraint_regularization": constraint_regularization,
                        "bounds": bounds,
                    }
                )
            if lower_constraint is not None:
                constraints.append(
                    {
                        "constraint_method": constraint_method,
                        "constraint_value": lower_constraint,
                        "constraint_direction": "lower",
                        "constraint_regularization": constraint_regularization,
                        "bounds": bounds,
                    }
                )
        print(constraints)
    if constraints is None or not constraints:
        print("no constraint applied")
        return forecast, lower_forecast, upper_forecast
    if isinstance(constraints, dict):
        constraints = [constraints]
    for constraint in constraints:
        forecast, lower_forecast, upper_forecast = apply_constraint_single(
            forecast, lower_forecast, upper_forecast, df_train=df_train, **constraint
        )

    return forecast, lower_forecast, upper_forecast


def apply_adjustments(
    forecast,
    lower_forecast,
    upper_forecast,
    adjustments=None,
    df_train=None,
):
    """Apply post-forecast adjustments such as percentage, additive, and smoothing.
    
    Args:
        forecast (pd.DataFrame): forecast df, wide style
        lower_forecast (pd.DataFrame): lower bound forecast df
        upper_forecast (pd.DataFrame): upper bound forecast df
        adjustments (list): list of dictionaries of adjustments to apply
            keys: "adjustment_method" or "method", "columns", "start_date", "end_date", "apply_bounds", plus method-specific params
        df_train (pd.DataFrame): required for align_last_value method
    
    Returns:
        forecast, lower, upper (pd.DataFrame)
    """
    # TODO: apply adjustment to trend component of the predictions, if present
    # TODO: have a df on the PredictionObject to track the adjustments applied
    if adjustments is None or not adjustments:
        return forecast, lower_forecast, upper_forecast
    if isinstance(adjustments, dict):
        adjustments = [adjustments]
    if not isinstance(forecast, pd.DataFrame):
        raise TypeError("apply_adjustments requires forecast to be a pandas DataFrame.")

    forecast_adj = forecast.copy()
    lower_adj = lower_forecast.copy() if isinstance(lower_forecast, pd.DataFrame) else lower_forecast
    upper_adj = upper_forecast.copy() if isinstance(upper_forecast, pd.DataFrame) else upper_forecast

    def _resolve_columns(adjustment):
        """Get list of valid columns from adjustment specification."""
        cols = adjustment.get("columns")
        if cols is None:
            return list(forecast_adj.columns)
        if isinstance(cols, (str, int)):
            cols = [cols]
        # Filter to only existing columns
        valid_cols = [col for col in cols if col in forecast_adj.columns]
        if not valid_cols and cols:
            warnings.warn(f"Adjustment columns {cols} not found in forecast; skipping.")
        return valid_cols

    def _create_mask(index, adjustment):
        """Create boolean mask for date range filtering."""
        if index.empty:
            return pd.Series(dtype=bool, index=index)
        start = adjustment.get("start_date")
        end = adjustment.get("end_date")
        start = index[0] if start is None else pd.to_datetime(start)
        end = index[-1] if end is None else pd.to_datetime(end)
        if end < start:
            warnings.warn(f"Adjustment end_date {end} precedes start_date {start}; skipping window.")
            return pd.Series(False, index=index)
        return (index >= start) & (index <= end)

    # Process each adjustment
    for adjustment in adjustments:
        if not isinstance(adjustment, dict):
            warnings.warn(f"Adjustment {adjustment} is not a dict and will be skipped.")
            continue
        
        method = adjustment.get("adjustment_method") or adjustment.get("method")
        if method is None:
            warnings.warn("No adjustment_method provided; skipping.")
            continue
        method = str(method).lower()
        
        columns = _resolve_columns(adjustment)
        if not columns:
            continue
        
        apply_bounds = adjustment.get("apply_bounds", True)
        mask = _create_mask(forecast_adj.index, adjustment)
        
        # Apply adjustment based on method
        if method in ("percentage", "percent"):
            # Percentage adjustment - multiply by (1 + percentage)
            start_value = adjustment.get("start_value")
            end_value = adjustment.get("end_value")
            constant_value = adjustment.get("value")
            
            # Handle different value specifications
            if start_value is None and end_value is None and constant_value is not None:
                start_value = end_value = constant_value
            elif start_value is None and end_value is not None:
                start_value = end_value
            elif end_value is None and start_value is not None:
                end_value = start_value
            
            if start_value is None:
                warnings.warn("Percentage adjustment missing value definitions; skipping.")
                continue
            if not mask.any():
                continue
            
            # Create linear interpolation between start and end
            num_periods = int(mask.sum())
            pct_values = np.linspace(start_value, end_value, num=num_periods)
            pct_series = pd.Series(pct_values, index=forecast_adj.index[mask])
            
            # Apply to forecast and bounds
            forecast_adj.loc[mask, columns] = forecast_adj.loc[mask, columns].multiply(
                1 + pct_series, axis=0
            )
            if apply_bounds and isinstance(lower_adj, pd.DataFrame):
                lower_adj.loc[mask, columns] = lower_adj.loc[mask, columns].multiply(
                    1 + pct_series, axis=0
                )
            if apply_bounds and isinstance(upper_adj, pd.DataFrame):
                upper_adj.loc[mask, columns] = upper_adj.loc[mask, columns].multiply(
                    1 + pct_series, axis=0
                )
                
        elif method in ("additive", "add"):
            # Additive adjustment - add constant or array
            value = adjustment.get("value")
            if value is None:
                warnings.warn("Additive adjustment missing 'value'; skipping.")
                continue
            if not mask.any():
                continue
            
            # Apply to forecast and bounds
            forecast_adj.loc[mask, columns] = forecast_adj.loc[mask, columns] + value
            if apply_bounds and isinstance(lower_adj, pd.DataFrame):
                lower_adj.loc[mask, columns] = lower_adj.loc[mask, columns] + value
            if apply_bounds and isinstance(upper_adj, pd.DataFrame):
                upper_adj.loc[mask, columns] = upper_adj.loc[mask, columns] + value
                
        elif method in ("align_last_value", "alignlastvalue", "align"):
            # Align forecast to last historical value
            from autots.tools.transform import AlignLastValue

            if df_train is None or not isinstance(df_train, pd.DataFrame):
                warnings.warn("AlignLastValue adjustment requires df_train dataframe; skipping.")
                continue
            
            # Parse parameters
            params = adjustment.get("parameters") or adjustment.get("transformation_params") or {}
            if isinstance(params, dict) and "0" in params and isinstance(params["0"], dict):
                params = params["0"]
            align_params = {**DEFAULT_ALIGN_LAST_VALUE_PARAMS, **params}
            
            # Fit and apply aligner
            aligner = AlignLastValue(**align_params)
            try:
                aligner.fit(df_train[columns])
            except KeyError as err:
                warnings.warn(f"AlignLastValue could not find columns {err}; skipping.")
                continue
            
            # Apply to forecast
            forecast_adj.loc[:, columns] = aligner.inverse_transform(forecast_adj.loc[:, columns])
            
            # Apply to bounds with same adjustment values
            if apply_bounds:
                adjustment_values = aligner.adjustment
                if isinstance(lower_adj, pd.DataFrame):
                    lower_adj.loc[:, columns] = aligner.inverse_transform(
                        lower_adj.loc[:, columns], adjustment=adjustment_values
                    )
                if isinstance(upper_adj, pd.DataFrame):
                    upper_adj.loc[:, columns] = aligner.inverse_transform(
                        upper_adj.loc[:, columns], adjustment=adjustment_values
                    )
                    
        elif method in ("smoothing", "ewma"):
            # Exponential weighted moving average smoothing
            span = adjustment.get("span", 7)
            if span is None or span <= 0:
                warnings.warn(f"Invalid EWMA span '{span}' in adjustment; skipping.")
                continue
            
            # Apply smoothing
            smoothed_forecast = forecast_adj.loc[:, columns].ewm(span=span, adjust=False).mean()
            if mask.any():
                forecast_adj.loc[mask, columns] = smoothed_forecast.loc[mask, columns]
            else:
                forecast_adj.loc[:, columns] = smoothed_forecast
            
            if apply_bounds:
                if isinstance(lower_adj, pd.DataFrame):
                    smoothed_lower = lower_adj.loc[:, columns].ewm(span=span, adjust=False).mean()
                    if mask.any():
                        lower_adj.loc[mask, columns] = smoothed_lower.loc[mask, columns]
                    else:
                        lower_adj.loc[:, columns] = smoothed_lower
                if isinstance(upper_adj, pd.DataFrame):
                    smoothed_upper = upper_adj.loc[:, columns].ewm(span=span, adjust=False).mean()
                    if mask.any():
                        upper_adj.loc[mask, columns] = smoothed_upper.loc[mask, columns]
                    else:
                        upper_adj.loc[:, columns] = smoothed_upper
        else:
            warnings.warn(f"Unknown adjustment_method '{method}' provided; skipping.")

    return forecast_adj, lower_adj, upper_adj


def extract_single_series_from_horz(series, model_name, model_parameters):
    title_prelim = str(model_name)[0:80]
    if title_prelim == "Ensemble":
        ensemble_type = model_parameters.get('model_name', "Ensemble")
        # horizontal and mosaic ensembles
        if "series" in model_parameters.keys():
            model_id = model_parameters['series'].get(series, "Horizontal")
            if isinstance(model_id, dict):
                model_id = list(model_id.values())
            if not isinstance(model_id, list):
                model_id = [str(model_id)]
            res = []
            for imod in model_id:
                res.append(
                    model_parameters.get("models", {})
                    .get(imod, {})
                    .get('Model', "Horizontal")
                )
            title_prelim = ", ".join(set(res))
            if len(model_id) > 1:
                title_prelim = "Mosaic: " + str(title_prelim)
        else:
            title_prelim = ensemble_type
    return str(title_prelim)


def extract_single_transformer(
    series, model_name, model_parameters, transformation_params
):
    if isinstance(transformation_params, str):
        transformation_params = json.loads(transformation_params)
    if isinstance(model_parameters, str):
        model_parameters = json.loads(model_parameters)
    if model_name == "Ensemble":
        # horizontal and mosaic ensembles
        if "series" in model_parameters.keys():
            model_id = model_parameters['series'].get(series, "Horizontal")
            if isinstance(model_id, dict):
                model_id = list(model_id.values())
            if not isinstance(model_id, list):
                model_id = [str(model_id)]
            res = []
            for imod in model_id:
                chosen_mod = model_parameters.get("models", {}).get(imod, {})
                res.append(
                    extract_single_transformer(
                        series,
                        chosen_mod.get("Model"),
                        chosen_mod.get("ModelParameters"),
                        transformation_params=chosen_mod.get(
                            "TransformationParameters"
                        ),
                    )
                )
            return ", ".join(res)
        allz = []
        for idz, mod in model_parameters.get("models").items():
            allz.append(
                extract_single_transformer(
                    series,
                    mod.get("Model"),
                    mod.get("ModelParameters"),
                    transformation_params=mod.get("TransformationParameters"),
                )
            )
        return ", ".join(allz)
    else:
        if isinstance(transformation_params, dict):
            trans_dict = transformation_params.get("transformations")
            if isinstance(trans_dict, dict):
                return ", ".join(list(trans_dict.values()))
            else:
                return "None"
        else:
            return "None"


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
        copy: return a deep copy with separate memory for all key elements
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
        result_windows=None,
        components=None,
    ):
        self.model_name = self.name = model_name
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
        self.runtime_dict = None
        self.result_windows = result_windows
        self.components = components

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

    def copy(self):
        """Create a deep copy of the PredictionObject with separate memory for all key elements.
        
        Returns:
            PredictionObject: A new PredictionObject with deep copies of all attributes
        """
        import copy
        
        # Create a new instance with copied forecasts and core attributes
        new_obj = PredictionObject(
            model_name=self.model_name,
            forecast_length=self.forecast_length,
            forecast_index=self.forecast_index.copy() if isinstance(self.forecast_index, pd.Index) else self.forecast_index,
            forecast_columns=self.forecast_columns.copy() if isinstance(self.forecast_columns, pd.Index) else self.forecast_columns,
            lower_forecast=self.lower_forecast.copy() if isinstance(self.lower_forecast, pd.DataFrame) else self.lower_forecast,
            forecast=self.forecast.copy() if isinstance(self.forecast, pd.DataFrame) else self.forecast,
            upper_forecast=self.upper_forecast.copy() if isinstance(self.upper_forecast, pd.DataFrame) else self.upper_forecast,
            prediction_interval=self.prediction_interval,
            predict_runtime=self.predict_runtime,
            fit_runtime=self.fit_runtime,
            model_parameters=copy.deepcopy(self.model_parameters),
            transformation_parameters=copy.deepcopy(self.transformation_parameters),
            transformation_runtime=self.transformation_runtime,
            per_series_metrics=self.per_series_metrics.copy() if isinstance(self.per_series_metrics, pd.DataFrame) else self.per_series_metrics,
            per_timestamp=self.per_timestamp.copy() if isinstance(self.per_timestamp, pd.DataFrame) else self.per_timestamp,
            avg_metrics=self.avg_metrics.copy() if isinstance(self.avg_metrics, pd.Series) else self.avg_metrics,
            avg_metrics_weighted=self.avg_metrics_weighted.copy() if isinstance(self.avg_metrics_weighted, pd.Series) else self.avg_metrics_weighted,
            full_mae_error=self.full_mae_error.copy() if isinstance(self.full_mae_error, np.ndarray) else self.full_mae_error,
            model=None,  # Don't copy model objects as they can be complex
            transformer=None,  # Don't copy transformer objects
            result_windows=copy.deepcopy(self.result_windows) if self.result_windows is not None else None,
            components=self.components.copy() if isinstance(self.components, pd.DataFrame) else self.components,
        )
        
        # Copy additional attributes that may have been set after initialization
        if hasattr(self, 'runtime_dict') and self.runtime_dict is not None:
            new_obj.runtime_dict = copy.deepcopy(self.runtime_dict)
        
        if hasattr(self, 'squared_errors'):
            new_obj.squared_errors = self.squared_errors.copy() if isinstance(self.squared_errors, np.ndarray) else self.squared_errors
        
        if hasattr(self, 'upper_pl'):
            new_obj.upper_pl = self.upper_pl.copy() if isinstance(self.upper_pl, np.ndarray) else self.upper_pl
        
        if hasattr(self, 'lower_pl'):
            new_obj.lower_pl = self.lower_pl.copy() if isinstance(self.lower_pl, np.ndarray) else self.lower_pl
        
        return new_obj

    def long_form_results(
        self,
        id_name="SeriesID",
        value_name="Value",
        interval_name='PredictionInterval',
        update_datetime_name=None,
        datetime_column=None,
    ):
        """Export forecasts (including upper and lower) as single 'long' format output

        Args:
            id_name (str): name of column containing ids
            value_name (str): name of column containing numeric values
            interval_name (str): name of column telling you what is upper/lower
            datetime_column (str): if None, is index, otherwise, name of column for datetime
            update_datetime_name (str): if not None, adds column with current timestamp and this name

        Returns:
            pd.DataFrame
        """
        upload = pd.melt(
            self.forecast.rename_axis(index='datetime').reset_index(),
            var_name=id_name,
            value_name=value_name,
            id_vars="datetime",
        ).set_index("datetime")
        upload[interval_name] = "50%"
        upload_upper = pd.melt(
            self.upper_forecast.rename_axis(index='datetime').reset_index(),
            var_name=id_name,
            value_name=value_name,
            id_vars="datetime",
        ).set_index("datetime")
        upload_upper[
            interval_name
        ] = f"{round(100 - ((1- self.prediction_interval)/2) * 100, 0)}%"
        upload_lower = pd.melt(
            self.lower_forecast.rename_axis(index='datetime').reset_index(),
            var_name=id_name,
            value_name=value_name,
            id_vars="datetime",
        ).set_index("datetime")
        upload_lower[
            interval_name
        ] = f"{round(((1- self.prediction_interval)/2) * 100, 0)}%"

        upload = pd.concat([upload, upload_upper, upload_lower], axis=0)
        if datetime_column is not None:
            upload.index.name = str(datetime_column)
            upload = upload.reset_index(drop=False)
        if update_datetime_name is not None:
            upload[update_datetime_name] = datetime.datetime.utcnow()
        return upload

    def total_runtime(self):
        """Combine runtimes."""
        return self.fit_runtime + self.predict_runtime + self.transformation_runtime

    def extract_ensemble_runtimes(self):
        """Return a dataframe of final runtimes per model for standard ensembles."""
        if self.runtime_dict is None or not bool(self.model_parameters):
            return None
        else:
            runtimes = pd.DataFrame(
                self.runtime_dict.items(), columns=['ID', 'Runtime']
            )
            runtimes['TotalRuntimeSeconds'] = runtimes['Runtime'].dt.total_seconds()
            new_models = {
                x: y.get("Model")
                for x, y in self.model_parameters.get("models").items()
            }
            models = pd.DataFrame(new_models.items(), columns=['ID', 'Model'])
            return runtimes.merge(models, how='left', on='ID')

    def plot_ensemble_runtimes(self, xlim_right=None):
        """Plot ensemble runtimes by model type."""
        runtimes_data = self.extract_ensemble_runtimes()

        if runtimes_data is None:
            return None
        else:
            return plot_distributions(
                runtimes_data,
                group_col='Model',
                y_col='TotalRuntimeSeconds',
                xlim=0,
                xlim_right=xlim_right,
                title_suffix=" in Chosen Ensemble",
            )

    def plot_df(
        self,
        df_wide=None,
        series: str = None,
        remove_zeroes: bool = False,
        interpolate: str = None,
        start_date: str = None,
    ):
        if series is None:
            series = random.choice(self.forecast.columns)

        model_name = self.model_name
        if model_name == "Ensemble":
            if 'series' in self.model_parameters.keys():
                if "profile" in self.model_parameters["model_metric"]:
                    from autots.tools.profile import profile_time_series

                    df = df_wide if df_wide is not None else self.forecast
                    profile = profile_time_series(df)
                    # I'm not sure why I made it sometimes coming as ID and sometimes SERIES...
                    if "ID" in profile.columns:
                        key_col = "ID"
                    else:
                        key_col = "SERIES"
                    h_params = self.model_parameters['series'][
                        profile[profile[key_col] == series]["PROFILE"].iloc[0]
                    ]
                else:
                    h_params = self.model_parameters['series'][series]
                if isinstance(h_params, str):
                    model_name = self.model_parameters['models'][h_params]['Model']

        if df_wide is not None:
            plot_df = pd.DataFrame(
                {
                    'actuals': df_wide[series],
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
        if interpolate is not None:
            plot_df["actuals"] = plot_df["actuals"].interpolate(
                method=interpolate, limit_direction="backward"
            )
            plot_df["forecast"] = plot_df["forecast"].interpolate(
                method=interpolate, limit_direction="backward", limit=5
            )

        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            if plot_df.index.max() < start_date:
                raise ValueError("start_date is more recent than all data provided")
            plot_df = plot_df[plot_df.index >= start_date]
        return plot_df

    def plot(
        self,
        df_wide=None,
        series: str = None,
        remove_zeroes: bool = False,
        interpolate: str = None,
        start_date: str = "auto",
        alpha=0.3,
        facecolor="black",
        loc="upper right",
        title=None,
        title_substring=None,
        vline=None,
        colors=None,
        include_bounds=True,
        plot_grid=False,  # if part of plot_grid
        dpi=None,  # allow override
        **kwargs,
    ):
        """Generate an example plot of one series. Does not handle non-numeric forecasts.

        Args:
            df_wide (str): historic data for plotting actuals
            series (str): column name of series to plot. Random if None.
            ax: matplotlib axes to pass through to pd.plot()
            remove_zeroes (bool): if True, don't plot any zeroes
            interpolate (str): if not None, a method to pass to pandas interpolate
            start_date (str): Y-m-d string or Timestamp to remove all data before
            vline (datetime): datetime of dashed vertical line to plot
            colors (dict): colors mapping dictionary col: color
            alpha (float): intensity of bound interval shading
            title (str): title
            title_substring (str): additional title details to pass to existing, moves series name to axis
            include_bounds (bool): if True, shows region of upper and lower forecasts
            dpi (int): dots per inch for figure resolution (default: 100 for display, 150+ for publication)
            **kwargs passed to pd.DataFrame.plot()
        """
        if start_date == "auto":
            if df_wide is not None:
                slx = -self.forecast_length * 3
                if abs(slx) > df_wide.shape[0]:
                    slx = 0
                start_date = df_wide.index[slx]
            else:
                start_date = self.forecast.index[0]

        if series is None:
            series = random.choice(self.forecast.columns)
        plot_df = self.plot_df(
            df_wide=df_wide,
            series=series,
            remove_zeroes=remove_zeroes,
            interpolate=interpolate,
            start_date=start_date,
        )
        if self.forecast_length == 1 and 'actuals' in plot_df.columns:
            if plot_df.shape[0] > 3:
                plot_df['forecast'].iloc[-2] = plot_df['actuals'].iloc[-2]
        if 'actuals' not in plot_df.columns:
            plot_df['actuals'] = np.nan
        if colors is None:
            colors = {
                'low_forecast': '#A5ADAF',
                'up_forecast': '#A5ADAF',
                'forecast': '#003399',  # '#4D4DFF',
                'actuals': '#1E88E5',  # Improved contrast from #AFDBF5
            }
        if title is None:
            title_prelim = extract_single_series_from_horz(
                series,
                model_name=self.model_name,
                model_parameters=self.model_parameters,
            )[0:80]
            if title_substring is None:
                title = f"{series} with model {title_prelim}"
            else:
                title = f"{title_substring} with model {title_prelim}"

        plot_kwargs = kwargs.copy()
        ax_param = plot_kwargs.pop('ax', None)
        user_color = plot_kwargs.pop('color', None)
        
        # Set default linewidth for better visibility if not specified
        if 'linewidth' not in plot_kwargs and 'lw' not in plot_kwargs:
            plot_kwargs['linewidth'] = 1.5
        
        # Determine band color with proper fallback
        if colors and 'low_forecast' in colors:
            band_color = colors['low_forecast']
        else:
            band_color = "#A5ADAF"
        
        # Determine which colors to pass: either the colors dict or None if user provided color kwarg
        colors_to_use = None if user_color else colors
        
        # Create new figure with DPI if ax not provided
        if ax_param is None and not plot_grid:
            fig_dpi = dpi if dpi is not None else 100
            fig = plt.figure(dpi=fig_dpi)
            ax_param = fig.add_subplot(111)
        
        ax = plot_forecast_with_intervals(
            plot_df,
            actual_col='actuals' if 'actuals' in plot_df.columns else None,
            forecast_col='forecast',
            lower_col='low_forecast',
            upper_col='up_forecast',
            title=title,
            colors=colors_to_use,
            include_bounds=include_bounds,
            alpha=alpha,
            band_color=band_color,
            interval_label="Prediction Interval",
            ax=ax_param,
            color=user_color,
            **plot_kwargs,
        )
        
        # Professional styling improvements
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)  # Grid behind data
        
        # Improve spine appearance
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#333333')
        
        # Better tick formatting
        ax.tick_params(axis='both', which='major', labelsize=9, length=4, width=0.8)
        
        if vline is not None:
            ax.vlines(
                x=vline,
                ls='--',
                lw=1.2,
                colors='darkred',
                ymin=plot_df.min().min(),
                ymax=plot_df.max().max(),
                alpha=0.7,
            )
            # ax.text(vline, plot_df.max().max(), "Event", color='darkred', verticalalignment='top')
        if title_substring is not None:
            ax.set_ylabel(series, fontsize=10)
        
        # Add AutoTS watermark if not part of plot_grid
        if not plot_grid:
            ax.text(
                0.98,
                0.02,
                "AutoTS",
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=7,
                alpha=0.08,
                style='italic',
                color='gray',
            )
        
        return ax

    def plot_components(
        self,
        series: str = None,
        start_date=None,
        df_wide=None,
        title: str = None,
        figsize=None,
        include_forecast=True,
        sharex=True,
        **kwargs,
    ):
        """Plot stored component contributions for a single series with Prophet-style subplots.
        
        Args:
            series (str): Series name to plot. Random if None.
            start_date: Filter data from this date onward.
            df_wide (pd.DataFrame): Historical actuals for plotting alongside forecast.
            title (str): Overall figure title.
            figsize (tuple): Figure size. Auto-calculated if None.
            include_forecast (bool): If True, adds top subplot with forecast and actuals.
            sharex (bool): Whether subplots share x-axis.
            **kwargs: Additional arguments passed to matplotlib plot.
            
        Returns:
            fig: matplotlib Figure object
        """
        if self.components is None or not isinstance(self.components, pd.DataFrame):
            raise ValueError("No component data available on this PredictionObject.")
        if series is None:
            series = random.choice(self.components.columns.get_level_values(0).unique())
        if series not in self.components.columns.get_level_values(0):
            raise ValueError(f"Series '{series}' not found in stored components.")
        
        comp_df = self.components[series].copy()
        
        # Filter by start_date if provided
        if start_date is not None:
            comp_df = comp_df[comp_df.index >= start_date]
            if comp_df.empty:
                raise ValueError("No component data remains after applying start_date filter.")
        
        # Determine number of subplots
        component_names = comp_df.columns.tolist()
        n_components = len(component_names)
        n_plots = n_components + 1 if include_forecast else n_components
        
        # Auto-calculate figsize if not provided
        if figsize is None:
            figsize = (12, 3 * n_plots)
        
        # Create subplots
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=sharex)
        if n_plots == 1:
            axes = [axes]
        
        # Overall title
        if title is None:
            model_name = extract_single_series_from_horz(
                series,
                model_name=self.model_name,
                model_parameters=self.model_parameters,
            )
            title = f"Component Breakdown: {series}\nModel: {model_name}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plot_idx = 0
        
        # Plot forecast + actuals if requested
        if include_forecast:
            ax = axes[plot_idx]
            forecast_data = self.forecast[series]
            if start_date is not None:
                forecast_data = forecast_data[forecast_data.index >= start_date]
            
            # Plot actuals if available
            if df_wide is not None and series in df_wide.columns:
                actuals = df_wide[series]
                if start_date is not None:
                    actuals = actuals[actuals.index >= start_date]
                ax.plot(actuals.index, actuals.values, 'o', 
                       markersize=2, label='Actual', color='#AFDBF5', alpha=0.7)
            
            # Plot forecast
            ax.plot(forecast_data.index, forecast_data.values, 
                   label='Forecast', color='#003399', linewidth=2)
            
            # Plot prediction intervals
            upper = self.upper_forecast[series]
            lower = self.lower_forecast[series]
            if start_date is not None:
                upper = upper[upper.index >= start_date]
                lower = lower[lower.index >= start_date]
            ax.fill_between(forecast_data.index, lower.values, upper.values,
                           alpha=0.2, color='#A5ADAF', label='Prediction Interval')
            
            ax.set_ylabel('Value', fontweight='bold')
            ax.legend(loc='best', frameon=True, fancybox=True)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_title('Forecast', fontsize=11, loc='left', fontweight='bold')
            plot_idx += 1
        
        # Plot each component
        for component in component_names:
            ax = axes[plot_idx]
            comp_values = comp_df[component]
            
            ax.plot(comp_values.index, comp_values.values, 
                   linewidth=1.5, color='#2C7BB6', **kwargs)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.set_ylabel('Effect', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_title(component, fontsize=11, loc='left', fontweight='bold')
            
            # Improve y-axis scaling
            y_values = comp_values.values
            if len(y_values) > 0 and not np.all(np.isnan(y_values)):
                y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)
                y_range = y_max - y_min
                if y_range > 0:
                    # Add 10% margin on each side
                    margin = y_range * 0.1
                    ax.set_ylim(y_min - margin, y_max + margin)
                elif y_max == y_min and y_max != 0:
                    # Constant non-zero value
                    ax.set_ylim(y_max * 0.9, y_max * 1.1)
                else:
                    # All zeros or single value at zero
                    ax.set_ylim(-0.5, 0.5)
            
            plot_idx += 1
        
        # Set x-label on bottom plot only
        axes[-1].set_xlabel('Date', fontweight='bold')
        
        plt.tight_layout()
        return fig

    def plot_grid(
        self,
        df_wide=None,
        start_date='auto',
        interpolate=None,
        remove_zeroes=False,
        figsize=(24, 18),
        title="AutoTS Forecasts",
        cols=None,
        series=None,  # alias for above
        colors=None,
        include_bounds=True,
        dpi=100,  # default DPI for grid plots
    ):
        """Plots multiple series in a grid, if present. Mostly identical args to the single plot function.
        
        Args:
            dpi (int): dots per inch for figure resolution (default: 100)
        """
        import matplotlib.pyplot as plt

        if series is not None and cols is None:
            cols = series
        if cols is None:
            cols = self.forecast.columns.tolist()
        num_cols = len(cols)
        if num_cols > 4:
            nrow = 2
            ncol = 3
            num_cols = 6
        elif num_cols > 2:
            nrow = 2
            ncol = 2
        else:
            nrow = 1
            ncol = 2
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, dpi=dpi, constrained_layout=True)
        fig.suptitle(title, fontsize='xx-large')
        count = 0
        if len(cols) != num_cols:
            sample_cols = random.choices(cols, k=num_cols)
        else:
            sample_cols = cols
        for r in range(nrow):
            for c in range(ncol):
                if nrow > 1:
                    ax = axes[r, c]
                else:
                    ax = axes[c]
                if count + 1 > num_cols:
                    pass
                else:
                    col = sample_cols[count]
                    self.plot(
                        df_wide=df_wide,
                        series=col,
                        remove_zeroes=remove_zeroes,
                        interpolate=interpolate,
                        start_date=start_date,
                        colors=colors,
                        include_bounds=include_bounds,
                        plot_grid=True,
                        ax=ax,
                    )
                    count += 1
        return fig

    def evaluate(
        self,
        actual,
        series_weights: dict = None,
        df_train=None,
        per_timestamp_errors: bool = False,
        full_mae_error: bool = True,
        scaler=None,
        cumsum_A=None,
        diff_A=None,
        last_of_array=None,
        column_names=None,
        custom_metric=None,
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
            custom metric (callable): a function to generate a custom metric. Expects func(A, F, df_train, prediction_interval) where the first three are np arrays of wide style 2d.

        Returns:
            per_series_metrics (pandas.DataFrame): contains a column for each series containing accuracy metrics
            per_timestamp (pandas.DataFrame): smape accuracy for each timestamp, avg of all series
            avg_metrics (pandas.Series): average values of accuracy across all input series
            avg_metrics_weighted (pandas.Series): average values of accuracy across all input series weighted by series_weight, if given
            full_mae_errors (numpy.array): abs(actual - forecast)
            scaler (numpy.array): precomputed scaler for efficiency, avg value of series in order of columns
        """
        # some forecasts have incorrect columns (they shouldn't but they do as a bug sometimes)
        if column_names is not None:
            use_cols = column_names
        elif isinstance(df_train, pd.DataFrame):
            use_cols = df_train.columns
        else:
            use_cols = self.forecast_columns
        # arrays are faster for math than pandas dataframes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            (
                self.per_series_metrics,
                self.full_mae_errors,
                self.squared_errors,
                self.upper_pl,
                self.lower_pl,
            ) = full_metric_evaluation(
                A=actual,
                F=self.forecast,
                upper_forecast=self.upper_forecast,
                lower_forecast=self.lower_forecast,
                df_train=df_train,
                prediction_interval=self.prediction_interval,
                columns=use_cols,
                scaler=scaler,
                return_components=True,
                cumsum_A=cumsum_A,
                diff_A=diff_A,
                last_of_array=last_of_array,
                custom_metric=custom_metric,
            )

        if per_timestamp_errors:
            smape_df = abs(self.forecast - actual) / (abs(self.forecast) + abs(actual))
            weight_mean = np.mean(list(series_weights.values()))
            wsmape_df = (smape_df * series_weights) / weight_mean
            smape_cons = (np.nansum(wsmape_df, axis=1) * 200) / np.count_nonzero(
                ~np.isnan(actual), axis=1
            )
            per_timestamp = pd.DataFrame({'weighted_smape': smape_cons}).transpose()
            self.per_timestamp = per_timestamp

        # check series_weights information
        if series_weights is None:
            series_weights = clean_weights(weights=False, series=self.forecast.columns)
        # make sure the series_weights are passed correctly to metrics
        if len(series_weights) != self.forecast.shape[1]:
            series_weights = {col: series_weights[col] for col in self.forecast.columns}

        # this weighting won't work well if entire metrics are NaN
        # but results should still be comparable
        self.avg_metrics_weighted = (self.per_series_metrics * series_weights).sum(
            axis=1, skipna=True
        ) / sum(series_weights.values())
        self.avg_metrics = self.per_series_metrics.mean(axis=1, skipna=True)
        if False:
            # vn1 temporary
            submission = self.forecast
            objective = actual
            abs_err = np.nansum(np.abs(submission - objective))
            err = np.nansum((submission - objective))
            score = abs_err + abs(err)
            epsilon = 1
            big_sum = (
                np.nan_to_num(objective, nan=0.0, posinf=0.0, neginf=0.0).sum().sum()
                + epsilon
            )
            score /= big_sum
            self.avg_metrics["custom"] = score
            self.avg_metrics_weighted["custom"] = score
        return self

    def apply_constraints(
        self,
        constraints=None,
        df_train=None,
        # old args
        constraint_method=None,
        constraint_regularization=None,
        upper_constraint=None,
        lower_constraint=None,
        bounds=True,
    ):
        """Use constraint thresholds to adjust outputs by limit.

        Example:
            apply_constraints(
                constraints=[
                    {  # don't exceed historic max
                        "constraint_method": "quantile",
                        "constraint_value": 1.0,
                        "constraint_direction": "upper",
                        "constraint_regularization": 1.0,
                        "bounds": True,
                    },
                    {  # don't exceed 2% decline by end of forecast horizon
                        "constraint_method": "slope",
                        "constraint_value": {
                            "slope": -0.02,
                            "window": 28,
                            "window_agg": "min",
                            "threshold": -0.01,
                        },
                        "constraint_direction": "lower",
                        "constraint_regularization": 0.9,
                        "bounds": False,
                    },
                    {  # don't exceed 2% growth by end of forecast horizon
                        "constraint_method": "slope",
                        "constraint_value": {"slope": 0.02, "window": 10, "window_agg": "max", "threshold": 0.01},
                        "constraint_direction": "upper",
                        "constraint_regularization": 0.9,
                        "bounds": False,
                    },
                    {  # don't go below the last 10 values - 10%
                        "constraint_method": "last_window",
                        "constraint_value": {"window": 10, "threshold": -0.1},
                        "constraint_direction": "lower",
                        "constraint_regularization": 1.0,
                        "bounds": False,
                    },
                    {  # don't go below zero
                        "constraint_method": "absolute",
                        "constraint_value": 0,  # can also be an array or Series
                        "constraint_direction": "lower",
                        "constraint_regularization": 1.0,
                        "bounds": True,
                    },
                    {  # don't go below historic min  - 1 st dev
                        "constraint_method": "stdev_min",
                        "constraint_value": 1.0,
                        "constraint_direction": "lower",
                        "constraint_regularization": 1.0,
                        "bounds": True,
                    },
                    {  # don't go above historic mean  + 3 st devs, soft limit
                        "constraint_method": "stdev",
                        "constraint_value": 3.0,
                        "constraint_direction": "upper",
                        "constraint_regularization": 0.5,
                        "bounds": True,
                    },
                    {  # round decimals to 2 places
                        "constraint_method": "round",
                        "constraint_value": 2,
                    },
                    {  # apply dampening (gradually flatten out forecast)
                        "constraint_method": "dampening",
                        "constraint_value": 0.98,
                    },
                ]
            )

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
            constraints=constraints,
            df_train=df_train,
            # old args
            constraint_method=constraint_method,
            constraint_regularization=constraint_regularization,
            upper_constraint=upper_constraint,
            lower_constraint=lower_constraint,
            bounds=bounds,
        )
        return self

    def apply_adjustments(self, adjustments=None, df_train=None):
        """Apply post-processing adjustments to the stored forecasts."""
        (
            self.forecast,
            self.lower_forecast,
            self.upper_forecast,
        ) = apply_adjustments(
            self.forecast,
            self.lower_forecast,
            self.upper_forecast,
            adjustments=adjustments,
            df_train=df_train,
        )
        return self

    def query_forecast(
        self,
        dates=None,
        series=None,
        include_bounds=False,
        include_components=False,
        return_json=False,
    ):
        """Query a specific slice of forecast results with minimal token usage.
        
        Designed for LLM-friendly output with compact representation.
        
        Args:
            dates (str, datetime, list, slice): Date(s) to query.
                - Single date: "2024-01-15" or datetime object
                - Date range: slice("2024-01-01", "2024-01-31")
                - List of dates: ["2024-01-15", "2024-01-20"]
                - None: all dates
            series (str, list): Series name(s) to query.
                - Single series: "sales"
                - Multiple series: ["sales", "revenue"]
                - None: all series
            include_bounds (bool): Include upper/lower forecast bounds
            include_components (bool): Include component breakdown if available
            return_json (bool): Return JSON string instead of dict
            
        Returns:
            dict or str: Compact forecast data
            
        Examples:
            >>> # Single series, single date
            >>> pred.query_forecast(dates="2024-01-15", series="sales")
            {'forecast': {'sales': {'2024-01-15': 123.45}}}
            
            >>> # Multiple series, date range with bounds
            >>> pred.query_forecast(
            ...     dates=slice("2024-01-01", "2024-01-07"),
            ...     series=["sales", "revenue"],
            ...     include_bounds=True
            ... )
        """
        if not isinstance(self.forecast, pd.DataFrame):
            raise ValueError("No forecast data available")
        
        # Handle series selection
        if series is None:
            selected_series = self.forecast.columns.tolist()
        elif isinstance(series, str):
            selected_series = [series]
        else:
            selected_series = list(series)
        
        # Validate series exist
        missing = set(selected_series) - set(self.forecast.columns)
        if missing:
            raise ValueError(f"Series not found in forecast: {missing}")
        
        # Handle date selection
        if dates is None:
            date_slice = self.forecast.index
        elif isinstance(dates, slice):
            date_slice = self.forecast.loc[dates.start:dates.stop].index
        elif isinstance(dates, (list, pd.Index)):
            date_slice = pd.DatetimeIndex(dates)
        else:
            # Single date
            date_slice = pd.DatetimeIndex([pd.to_datetime(dates)])
        
        # Build result dictionary
        result = {
            'model': self.model_name,
            'prediction_interval': self.prediction_interval,
        }
        
        # Extract forecast values
        forecast_data = {}
        for col in selected_series:
            col_data = {}
            for dt in date_slice:
                if dt in self.forecast.index:
                    val = self.forecast.loc[dt, col]
                    # Convert to native Python type for JSON serialization
                    col_data[dt.isoformat()] = float(val) if pd.notna(val) else None
            forecast_data[col] = col_data
        result['forecast'] = forecast_data
        
        # Add bounds if requested
        if include_bounds:
            if isinstance(self.upper_forecast, pd.DataFrame):
                upper_data = {}
                for col in selected_series:
                    col_data = {}
                    for dt in date_slice:
                        if dt in self.upper_forecast.index:
                            val = self.upper_forecast.loc[dt, col]
                            col_data[dt.isoformat()] = float(val) if pd.notna(val) else None
                    upper_data[col] = col_data
                result['upper_forecast'] = upper_data
            
            if isinstance(self.lower_forecast, pd.DataFrame):
                lower_data = {}
                for col in selected_series:
                    col_data = {}
                    for dt in date_slice:
                        if dt in self.lower_forecast.index:
                            val = self.lower_forecast.loc[dt, col]
                            col_data[dt.isoformat()] = float(val) if pd.notna(val) else None
                    lower_data[col] = col_data
                result['lower_forecast'] = lower_data
        
        # Add components if requested and available
        if include_components and self.components is not None:
            if isinstance(self.components, pd.DataFrame):
                components_data = {}
                for col in selected_series:
                    if col in self.components.columns.get_level_values(0):
                        col_components = {}
                        comp_names = self.components[col].columns.tolist()
                        for comp_name in comp_names:
                            comp_data = {}
                            for dt in date_slice:
                                if dt in self.components.index:
                                    val = self.components.loc[dt, (col, comp_name)]
                                    comp_data[dt.isoformat()] = float(val) if pd.notna(val) else None
                            col_components[comp_name] = comp_data
                        components_data[col] = col_components
                result['components'] = components_data
        
        if return_json:
            import json
            return json.dumps(result, indent=2)
        
        return result


def stack_component_frames(component_frames):
    """Convert dict of component DataFrames into a unified MultiIndex DataFrame."""
    frames = []
    for name, df in component_frames.items():
        if df is None:
            continue
        comp_df = df.copy()
        comp_df.columns = pd.MultiIndex.from_product(
            [comp_df.columns, [name]]
        )
        frames.append(comp_df)
    if not frames:
        return None
    result = pd.concat(frames, axis=1)
    # Ensure columns are grouped by series
    result = result.sort_index(axis=1, level=0)
    return result


def sum_component_frames(component_frames):
    """Helper to sum matching component DataFrames."""
    total = None
    for df in component_frames.values():
        if df is None:
            continue
        total = df if total is None else total.add(df, fill_value=0.0)
    return total
