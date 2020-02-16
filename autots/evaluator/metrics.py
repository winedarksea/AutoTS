"""
Metrics
"""
import math
import numpy as np
import pandas as pd

def smape_old(actual, forecast):
    """Expects two, 2-D numpy arrays of forecast_length * n series
    Doesn't handle negatives well
    
    Returns a 1-D array of results in len n series
    
    Args: 
        actual (numpy.array): known true values
        forecast (numpy.array): predicted values
        
    References:
        https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/37232
    """
    out_array = np.zeros(actual.shape[1])
    for r in range(actual.shape[1]):
#        try:
        y_true = actual[:,r]
        y_pred = forecast[:,r]
        y_pred = y_pred[~np.isnan(y_true)]
        y_true = y_true[~np.isnan(y_true)]

        out = 0
        for i in range(y_true.shape[0]):
            a = y_true[i]
            b = math.fabs(y_pred[i])
            c = a+b
            if c == 0:
                continue
            out += math.fabs(a - b) / c
        out *= (200.0 / y_true.shape[0])
#        except Exception:
#            out = np.nan
        out_array[r] = out
    return out_array

def smape(actual, forecast):
    """Expects two, 2-D numpy arrays of forecast_length * n series
    Allows NaN in actuals, and corresponding NaN in forecast, but not unmatched NaN in forecast
    Also doesn't like zeroes in either forecast or actual - results in poor error value even if forecast is accurate
    
    Returns a 1-D array of results in len n series
    
    Args: 
        actual (numpy.array): known true values
        forecast (numpy.array): predicted values
        
    References:
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    return (np.nansum((abs(forecast - actual) / (abs(forecast) + abs(actual))), axis = 0)* 200) / np.count_nonzero(~np.isnan(actual), axis = 0)
    
def mae(A, F):
    """Expects two, 2-D numpy arrays of forecast_length * n series
    
    Returns a 1-D array of results in len n series
    
    Args: 
        A (numpy.array): known true values
        F (numpy.array): predicted values
    """
    try:
        mae_result = abs(A - F)
    except Exception:
        mae_result = np.nan
    return np.nanmean(mae_result, axis=0)

def rmse(actual, forecast):
    """Expects two, 2-D numpy arrays of forecast_length * n series
    
    Returns a 1-D array of results in len n series
    
    Args: 
        actual (numpy.array): known true values
        forecast (numpy.array): predicted values
    """
    try:
        mae_result = np.sqrt(np.nanmean(((actual - forecast) ** 2),axis = 0))
    except Exception:
        mae_result = np.nan
    return mae_result

def containment(lower_forecast, upper_forecast, actual):
    """Expects two, 2-D numpy arrays of forecast_length * n series
    
    Returns a 1-D array of results in len n series
    
    Args: 
        actual (numpy.array): known true values
        forecast (numpy.array): predicted values
    """
    return np.count_nonzero((upper_forecast > actual) & (lower_forecast < actual), axis = 0)/actual.shape[0]
    
def contour(A, F):
    """A measure of how well the actual and forecast follow the same pattern of change.
    *Note:* If actual values are unchanging, will match positive changing forecasts.
    Expects two, 2-D numpy arrays of forecast_length * n series   
    Returns a 1-D array of results in len n series
    
    Args: 
        A (numpy.array): known true values
        F (numpy.array): predicted values
    """
    try:
        X = np.diff(A, axis = 0)
        Y = np.diff(F, axis = 0)
        
        # On the assumption flat lines common in forecasts, but exceedingly rare in real world
        X = X>=0
        Y = Y>0
        contour_result = np.sum(X == Y, axis = 0)/X.shape[0]
    except Exception:
        contour_result = np.nan
    return contour_result   

class EvalObject(object):
    """Object to contain all your failures!
    """
    def __init__(self, model_name: str = 'Uninitiated', residuals = np.nan, per_series_metrics = np.nan,per_timestamp_metrics=np.nan, weights = np.nan, avg_metrics = np.nan, avg_metrics_weighted = np.nan):
        self.model_name = model_name
        self.residuals = residuals
        self.per_series_metrics = per_series_metrics
        self.per_timestamp_metrics = per_timestamp_metrics
        self.weights = weights
        self.avg_metrics = avg_metrics
        self.avg_metrics_weighted = avg_metrics_weighted

def PredictionEval(PredictionObject, actual, series_weights: dict = {}, per_timestamp_errors: bool = True, per_series_errors: bool = True):
    """Evalute prediction against test actual.
    
    Args:
        PredictionObject (autots.PredictionObject): Prediction from AutoTS model object
        actual (pandas.DataFrame): dataframe of actual values of (forecast length * n series)
        series_weights (dict): key = column/series_id, value = weight
        per_timestamp (bool): Whether to calculate and return per timestamp direction errors

    """
    if series_weights == {}:
        series_weights = {x:1 for x in actual.columns}

    errors = EvalObject()
    errors.model_name = PredictionObject.model_name
    errors.residuals = PredictionObject.forecast - actual
    errors.weights = series_weights
    

    per_series = pd.DataFrame({
            'smape': smape(actual, PredictionObject.forecast),
            'mae': mae(actual, PredictionObject.forecast),
            'rmse': rmse(actual, PredictionObject.forecast),
            'containment': containment(PredictionObject.lower_forecast, PredictionObject.upper_forecast, actual),
            'lower_mae': mae(actual, PredictionObject.lower_forecast),
            'upper_mae': mae(actual, PredictionObject.upper_forecast),
            'contour': contour(actual, PredictionObject.forecast)
            }).transpose()
    per_series.columns = actual.columns
    errors.per_series_metrics = per_series
    
    if per_timestamp_errors:
        per_timestamp = pd.DataFrame({
                'smape': (np.nansum((abs(PredictionObject.forecast - actual) / (abs(PredictionObject.forecast) + abs(actual))), axis = 1)* 200) / np.count_nonzero(~np.isnan(actual), axis = 1),
                'mae': np.nanmean(abs(actual - PredictionObject.forecast), axis=1),
                'rmse': np.sqrt(np.nanmean(((actual - PredictionObject.forecast) ** 2),axis = 1)),
                'containment':np.count_nonzero((PredictionObject.upper_forecast > actual) & (PredictionObject.lower_forecast < actual), axis = 1)/actual.shape[1]
                }).transpose()
        errors.per_timestamp_metrics = per_timestamp
    
    # this weighting won't work well if entire metrics are NaN, but results should still be comparable
    errors.avg_metrics_weighted = (per_series * series_weights).sum(axis = 1, skipna = True) / sum(series_weights.values())
    errors.avg_metrics = per_series.mean(axis = 1)
    
    return errors

