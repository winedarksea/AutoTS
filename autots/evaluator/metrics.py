"""
Metrics
"""

import math

def smape(actual, forecast):
    """
    Expects a 2-D numpy array of forecast_length * n series
    
    Returns a 1-D array of results in len n series
    """
    out_array = np.zeros(actual.shape[1])
    for r in range(actual.shape[1]):
        try:
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
        except Exception:
            out = np.nan
        out_array[r] = out
    return out_array

def mae(A, F):
    """
    Expects a 2-D numpy array of forecast_length * n series
    
    Returns a 1-D array of results in len n series
    """
    try:
        mae_result = abs(A - F)
    except Exception:
        mae_result = np.nan
    return mae_result

