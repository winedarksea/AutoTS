"""Tools for calculating forecast errors."""
import warnings
import math
import numpy as np
import pandas as pd


def smape(actual, forecast):
    """Expect two, 2-D numpy arrays of forecast_length * n series.
    Allows NaN in actuals, and corresponding NaN in forecast, but not unmatched NaN in forecast
    Also doesn't like zeroes in either forecast or actual - results in poor error value even if forecast is accurate
    
    Returns a 1-D array of results in len n series
    
    Args: 
        actual (numpy.array): known true values
        forecast (numpy.array): predicted values
        
    References:
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        smape = (np.nansum((abs(forecast - actual) / (abs(forecast) + abs(actual))), axis=0) * 200) / np.count_nonzero(~np.isnan(actual), axis=0)
    return smape


def mae(A, F):
    """Expects two, 2-D numpy arrays of forecast_length * n series.

    Returns a 1-D array of results in len n series

    Args:
        A (numpy.array): known true values
        F (numpy.array): predicted values
    """
    mae_result = abs(A - F)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mae_result = np.nanmean(mae_result, axis=0)
    return mae_result


def pinball_loss(A, F, quantile):
    """Bigger is bad-er."""
    pl = np.where(A >= F, quantile * (A - F), (1 - quantile) * (F - A))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmean(pl, axis=0)
    return result


def SPL(A, F, df_train, quantile):
    """Scaled pinball loss."""
    # scaler = df_train.tail(1000).diff().abs().mean(axis=0)
    scaler = np.abs(np.diff(df_train[-1000:], axis=0)).mean(axis=0)
    # need to handle zeroes to prevent div 0 errors.
    # this will tend to make that series irrelevant to the overall evaluation
    fill_val = scaler.max()
    fill_val = fill_val if fill_val > 0 else 1
    scaler[scaler == 0] = fill_val
    return pinball_loss(A=A, F=F, quantile=quantile) / scaler


def rmse(actual, forecast):
    """Expects two, 2-D numpy arrays of forecast_length * n series.

    Returns a 1-D array of results in len n series

    Args:
        actual (numpy.array): known true values
        forecast (numpy.array): predicted values
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rmse_result = np.sqrt(np.nanmean(((actual - forecast) ** 2),
                                         axis=0))
    return rmse_result


def containment(lower_forecast, upper_forecast, actual):
    """Expects two, 2-D numpy arrays of forecast_length * n series.

    Returns a 1-D array of results in len n series

    Args:
        actual (numpy.array): known true values
        forecast (numpy.array): predicted values
    """
    result = np.count_nonzero((upper_forecast >= actual) & (lower_forecast <= actual), axis=0)/actual.shape[0]
    return result


def contour(A, F):
    """A measure of how well the actual and forecast follow the same pattern of change.
    *Note:* If actual values are unchanging, will match positive changing forecasts.
    Expects two, 2-D numpy arrays of forecast_length * n series
    Returns a 1-D array of results in len n series

    Args:
        A (numpy.array): known true values
        F (numpy.array): predicted values
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            X = np.nan_to_num(np.diff(A, axis=0))
            Y = np.nan_to_num(np.diff(F, axis=0))
            # On the assumption flat lines common in forecasts,
            # but exceedingly rare in real world
            X = X >= 0
            Y = Y > 0
            contour_result = np.sum(X == Y, axis=0)/X.shape[0]
        except Exception:
            contour_result = np.nan
    return contour_result


class EvalObject(object):
    """Object to contain all your failures!."""

    def __init__(self, model_name: str = 'Uninitiated',
                 per_series_metrics=np.nan,
                 per_timestamp=np.nan,
                 avg_metrics=np.nan, avg_metrics_weighted=np.nan):
        self.model_name = model_name
        self.per_series_metrics = per_series_metrics
        self.per_timestamp = per_timestamp
        self.avg_metrics = avg_metrics
        self.avg_metrics_weighted = avg_metrics_weighted


def PredictionEval(PredictionObject, actual,
                   series_weights: dict = {},
                   df_train=np.nan,
                   per_timestamp_errors: bool = False,
                   dist_n: int = None):
    """Evalute prediction against test actual.

    Args:
        PredictionObject (autots.PredictionObject): Prediction from AutoTS model object
        actual (pd.DataFrame): dataframe of actual values of (forecast length * n series)
        series_weights (dict): key = column/series_id, value = weight
        per_timestamp (bool): whether to calculate and return per timestamp direction errors
        dist_n (int): if not None, calculates two part rmse on head(n) and tail(remainder) of forecast.
    """
    A = np.array(actual)
    F = np.array(PredictionObject.forecast)
    lower_forecast = np.array(PredictionObject.lower_forecast)
    upper_forecast = np.array(PredictionObject.upper_forecast)
    df_train=np.array(df_train)

    errors = EvalObject(model_name=PredictionObject.model_name)

    per_series = pd.DataFrame({
            'smape': smape(A, F),
            'mae': mae(A, F),
            'rmse': rmse(A, F),
            'containment': containment(lower_forecast, upper_forecast, A),
            'spl': SPL(A=A, F=upper_forecast, df_train=df_train,
                       quantile=PredictionObject.prediction_interval) +
            SPL(A=A, F=lower_forecast, df_train=df_train,
                quantile=(1-PredictionObject.prediction_interval)),
            'contour': contour(A, F)
            }).transpose()
    per_series.columns = actual.columns

    if per_timestamp_errors:
        smape_df = (abs(PredictionObject.forecast - actual
                        ) / (abs(PredictionObject.forecast) + abs(actual)))
        weight_mean = np.mean(list(series_weights.values()))
        wsmape_df = (smape_df * series_weights) / weight_mean
        smape_cons = (np.nansum(wsmape_df, axis=1) * 200
                      ) / np.count_nonzero(~np.isnan(actual), axis=1)
        per_timestamp = pd.DataFrame({
            'weighted_smape': smape_cons
             }).transpose()
        errors.per_timestamp = per_timestamp
    """
    'mae': np.nanmean(abs(actual - PredictionObject.forecast), axis=1),
    'rmse': np.sqrt(np.nanmean(((actual - PredictionObject.forecast) ** 2),axis=1)),
    'containment':np.count_nonzero((PredictionObject.upper_forecast > actual) & (PredictionObject.lower_forecast < actual), axis=1)/actual.shape[1]
    """
    # this weighting won't work well if entire metrics are NaN
    # but results should still be comparable
    errors.avg_metrics_weighted = (per_series * series_weights).sum(
        axis=1, skipna=True) / sum(series_weights.values())
    errors.avg_metrics = per_series.mean(axis=1)

    if str(dist_n).isdigit():
        per_series_d = pd.DataFrame({
            'mae1': mae(A[:dist_n], F[:dist_n]),
            'mae2': mae(A[dist_n:], F[dist_n:])
            }).transpose()
        per_series_d.columns = actual.columns
        per_series = pd.concat([per_series, per_series_d], axis=0)
    errors.per_series_metrics = per_series
    return errors
