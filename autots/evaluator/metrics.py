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

def containment():
    """Expects two, 2-D numpy arrays of forecast_length * n series
    
    Returns a 1-D array of results in len n series
    
    Args: 
        actual (numpy.array): known true values
        forecast (numpy.array): predicted values
    """
    

class EvalObject(object):
    """Object to contain all your failures!
    """
    def __init__(self, model_name: str = 'Uninitiated', residuals = np.nan, per_series_metrics = np.nan, per_series_metrics_weighted = np.nan, avg_metrics = np.nan, avg_metrics_weighted = np.nan):
        self.model_name = model_name
        self.residuals = residuals
        self.per_series_metrics = per_series_metrics
        # self.per_series_metrics_weighted = per_series_metrics_weighted
        self.weights
        self.avg_metrics = avg_metrics
        self.avg_metrics_weighted = avg_metrics_weighted

def PredictionEval(PredictionObject, actual, series_weights = {}):
    """Evalute prediction against test actual.
    
    Args:
        PredictionObject (autots.PredictionObject): Prediction from AutoTS model object
        actual (pandas.DataFrame): dataframe of actual values of (forecast length * n series)

    """
    errors = EvalObject()
    errors.residuals = PredictionObject.forecast - actual
    
    smape(actual, PredictionObject.forecast)
    mae(actual, PredictionObject.forecast)
    return errors

"""
actual = df_test
actual_np = actual.values
forecast_np = df_forecast.forecast
forecast = pd.DataFrame(forecast_np, columns = actual.columns, index = actual.index)

temp = mae(actual, forecast)
mae(actual_np, forecast_np)

temp = smape(actual, forecast)
smape(actual_np, forecast_np)

temp = smape_old(actual, forecast)
smape_old(actual_np, forecast_np)

temp = rmse(actual, forecast)
rmse(actual_np, forecast_np)
"""

"""
Transformation Dict
ModelName
Parameter Dict
Residuals (date * n series)
Per Series:
    smape
    mae
    rmse
    containment
Avg Error:   np.nanmean()
    smape
    mae
    rmse
    containment
Weighted
"""