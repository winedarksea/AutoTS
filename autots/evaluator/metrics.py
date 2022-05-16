"""Tools for calculating forecast errors.

Some common args:
    A or actual (np.array): actuals ndim 2 (timesteps, series)
    F or forecast (np.array): forecast values ndim 2 (timesteps, series)
    ae (np.array): precalculated np.abs(A - F)
"""
import warnings
import numpy as np


def symmetric_mean_absolute_percentage_error(actual, forecast):
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
        smape = (
            np.nansum((abs(forecast - actual) / (abs(forecast) + abs(actual))), axis=0)
            * 200
        ) / np.count_nonzero(~np.isnan(actual), axis=0)
    return smape


def mean_absolute_error(A, F):
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


def median_absolute_error(A, F):
    """Expects two, 2-D numpy arrays of forecast_length * n series.

    Returns a 1-D array of results in len n series

    Args:
        A (numpy.array): known true values
        F (numpy.array): predicted values
    """
    mae_result = abs(A - F)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mae_result = np.nanmedian(mae_result, axis=0)
    return mae_result


def mean_absolute_differential_error(A, F, order: int = 1, df_train=None, scaler=None):
    """Expects two, 2-D numpy arrays of forecast_length * n series.

    Returns a 1-D array of results in len n series

    Args:
        A (numpy.array): known true values
        F (numpy.array): predicted values
        order (int): order of differential
        df_train (np.array): if provided, uses this as starting point for first diff step.
            Tail(1) must be most recent historical point before forecast.
            Must be numpy Array not DataFrame.
            Highly recommended if using this as the sole optimization metric.
            Without, it is an "unanchored" shape fitting metric.
            This will also allow this to work on forecast_length = 1 forecasts
        scaler (np.array): if provided, metrics are scaled by this. 1d array of shape (num_series,)
    """
    # scaler = np.mean(A, axis=0)  # debate over whether to make this scaled
    if df_train is not None:
        last_of_array = np.nan_to_num(
            df_train[
                df_train.shape[0] - 1 : df_train.shape[0],
            ]
        )
        # last_of_array = df_train.tail(1).fillna(0).to_numpy()
        # assigning to new because I'm paranoid about overwrite existing objects
        lA = np.concatenate([last_of_array, A])
        lF = np.concatenate([last_of_array, F])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if scaler is None:
                return np.nanmean(
                    abs(np.diff(lA, order, axis=0) - np.diff(lF, order, axis=0)), axis=0
                )
            else:
                return (
                    np.nanmean(
                        abs(np.diff(lA, order, axis=0) - np.diff(lF, order, axis=0)),
                        axis=0,
                    )
                    / scaler
                )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if scaler is None:
                return np.nanmean(
                    abs(np.diff(A, order, axis=0) - np.diff(F, order, axis=0)), axis=0
                )
            else:
                return (
                    np.nanmean(
                        abs(np.diff(A, order, axis=0) - np.diff(F, order, axis=0)),
                        axis=0,
                    )
                    / scaler
                )


def pinball_loss(A, F, quantile):
    """Bigger is bad-er."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pl = np.where(A >= F, quantile * (A - F), (1 - quantile) * (F - A))
        result = np.nanmean(pl, axis=0)
    return result


def scaled_pinball_loss(A, F, df_train, quantile):
    """Scaled pinball loss.

    Args:
        A (np.array): actual values
        F (np.array): forecast values
        df_train (np.array): values of historic data for scaling
        quantile (float): which bound of upper/lower forecast this is
    """
    # scaler = df_train.tail(1000).diff().abs().mean(axis=0)
    # scaler = np.abs(np.diff(df_train[-1000:], axis=0)).mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scaler = np.nanmean(np.abs(np.diff(df_train[-1000:], axis=0)), axis=0)
    # need to handle zeroes to prevent div 0 errors.
    # this will tend to make that series irrelevant to the overall evaluation
    fill_val = np.nanmax(scaler)
    fill_val = fill_val if fill_val > 0 else 1
    scaler[scaler == 0] = fill_val
    scaler[np.isnan(scaler)] = fill_val
    pl = pinball_loss(A=A, F=F, quantile=quantile)
    # for those cases where an entire series is NaN...
    # if any(np.isnan(pl)):
    #     pl[np.isnan(pl)] = np.nanmax(pl)
    return pl / scaler


def root_mean_square_error(actual, forecast):
    """Expects two, 2-D numpy arrays of forecast_length * n series.

    Returns a 1-D array of results in len n series

    Args:
        actual (numpy.array): known true values
        forecast (numpy.array): predicted values
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rmse_result = np.sqrt(np.nanmean(((actual - forecast) ** 2), axis=0))
    return rmse_result


def containment(lower_forecast, upper_forecast, actual):
    """Expects two, 2-D numpy arrays of forecast_length * n series.

    Returns a 1-D array of results in len n series

    Args:
        actual (numpy.array): known true values
        forecast (numpy.array): predicted values
    """
    return (
        np.count_nonzero(
            (upper_forecast >= actual) & (lower_forecast <= actual), axis=0
        )
        / actual.shape[0]
    )


def contour(A, F):
    """A measure of how well the actual and forecast follow the same pattern of change.
    *Note:* If actual values are unchanging, will match positive changing forecasts.
    This is faster, and because if actuals are a flat line, contour probably isn't a concern regardless.

    Expects two, 2-D numpy arrays of forecast_length * n series
    Returns a 1-D array of results in len n series

    NaNs diffs are filled with 0, essentially equiavelent to assuming a forward fill of NaN

    Concat the last row of history to head of both A and F (req for 1 step)

    Args:
        A (numpy.array): known true values
        F (numpy.array): predicted values
    """

    try:
        # On the assumption flat lines common in forecasts,
        # but exceedingly rare in real world
        contour_result = np.sum(
            (np.nan_to_num(np.diff(A, axis=0)) >= 0)
            == (np.nan_to_num(np.diff(F, axis=0)) > 0),
            axis=0,
        ) / (F.shape[0] - 1)
    except Exception:
        contour_result = np.nan
    return contour_result


def mda(A, F):
    """A measure of how well the actual and forecast follow the same pattern of change.
    Expects two, 2-D numpy arrays of forecast_length * n series
    Returns a 1-D array of results in len n series

    NaNs diffs are filled with 0, essentially equiavelent to assuming a forward fill of NaN

    Concat the last row of history to head of both A and F (req for 1 step)

    Args:
        A (numpy.array): known true values
        F (numpy.array): predicted values
    """

    X = np.nan_to_num(np.diff(A, axis=0))
    Y = np.nan_to_num(np.diff(F, axis=0))
    return np.sum(np.sign(X) == np.sign(Y), axis=0) / F.shape[0]


def rps(predictions, observed):
    """Vectorized version of Ranked Probability Score.
    A lower value is a better score.
    From: Colin Catlin, https://syllepsis.live/2022/01/22/ranked-probability-score-in-python/

    Args:
        predictions (pd.DataFrame): each column is an outcome category, with values as the 0 to 1 probability of that category
        observed (pd.DataFrame): each column is an outcome category, with values of 0 OR 1 with 1 being that category occurred
    """
    ncat = predictions.shape[1] - 1
    return (
        np.sum(
            (np.cumsum(predictions, axis=1) - np.cumsum(observed, axis=1)) ** 2, axis=1
        )
        / ncat
    )


def rmse(sqe):
    """Accepting squared error already calculated"""
    return np.sqrt(np.nanmean(sqe, axis=0))


def mae(ae):
    """Accepting abs error already calculated"""
    return np.nanmean(ae, axis=0)


def medae(ae, nan_flag=True):
    """Accepting abs error already calculated"""
    if nan_flag:
        return np.nanmedian(ae, axis=0)
    else:
        return np.median(ae, axis=0)


def smape(actual, forecast, ae, nan_flag=True):
    """Accepting abs error already calculated"""
    if nan_flag:
        return (
            np.nansum((ae / (abs(forecast) + abs(actual))), axis=0) * 200
        ) / np.count_nonzero(~np.isnan(actual), axis=0)
    else:
        return (
            np.sum((ae / (abs(forecast) + abs(actual))), axis=0) * 200
        ) / actual.shape[0]


def _spl(A, F, quantile, scaler):
    """Accepting scaler already calculated"""
    return (
        np.nanmean(
            np.where(A >= F, quantile * (A - F), (1 - quantile) * (F - A)), axis=0
        )
        / scaler
    )


def spl(precomputed_spl, scaler):
    """Accepting most of it already calculated"""
    return np.nanmean(precomputed_spl, axis=0) / scaler


def msle(full_errors, ae, le, nan_flag=True):
    """input is array of y_pred - y_true to over-penalize underestimate.
    Use instead y_true - y_pred to over-penalize overestimate.
    AE used here for the log just to avoid divide by zero warnings (values aren't used either way)
    """
    if nan_flag:
        return np.nanmean(np.where(full_errors > 0, le, ae), axis=0)
    else:
        return np.mean(np.where(full_errors > 0, le, ae), axis=0)


def oda(A, F, last_of_array):
    """Origin Directional Accuracy, the accuracy of growth or decline relative to most recent data."""
    return (
        np.nansum(np.sign(F - last_of_array) == np.sign(A - last_of_array), axis=0)
        / F.shape[0]
    )


def qae(ae, q=0.9, nan_flag=True):
    """Return the q quantile of the errors per series.
    np.nans count as smallest values and will push more values into the exclusion group.
    """
    if nan_flag:
        return np.quantile(np.nan_to_num(ae), q, axis=0)
    else:
        return np.quantile(ae, q, axis=0)


def mqae(ae, q=0.85, nan_flag=True):
    """Return the mean of errors less than q quantile of the errors per series.
    np.nans count as largest values, and so are removed as part of the > q group.
    """
    if ae.shape[0] <= 1:
        vals = ae
    else:
        qi = int(ae.shape[0] * q)
        qi = qi if qi > 1 else 1
        vals = np.partition(ae, qi, axis=0)[:qi]
    if nan_flag:
        return np.nanmean(vals, axis=0)
    else:
        return np.mean(vals, axis=0)


def mlvb(A, F, last_of_array):
    """Mean last value baseline, the % difference of forecast vs last value naive forecast.
    Does poorly with near-zero values.

    Args:
        A (np.array): actuals
        F (np.array): forecast values
        last_of_array (np.array): the last row of the historic training data, most recent values
    """
    a_diff = A - last_of_array
    a_diff[a_diff == 0] = np.nan
    return np.nanmean(np.abs((A - F) / a_diff), axis=0)
