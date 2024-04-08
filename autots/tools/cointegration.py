"""Cointegration

Johansen heavily based on Statsmodels source code

BTCD heavily based on D. Barba
https://towardsdatascience.com/canonical-decomposition-a-forgotten-method-for-time-series-cointegration-and-beyond-4d1213396da1

"""

import datetime
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power

# np.allclose(np.matmul(trans.components_, (df.values - trans.mean_).T).T, trans.transform(df))
# np.allclose(np.matmul((df.values - trans.mean_), (trans.components_.T)), trans.transform(df))
# np.allclose((np.matmul(transformed, trans.components_) + trans.mean_), trans.inverse_transform(transformed))


def lagmat(
    x,
    maxlag: int,
    trim='forward',
    original="ex",
):
    """
    Create 2d array of lags. Modified from Statsmodels.
    """
    orig = x
    trim = "none" if trim is None else trim
    trim = trim.lower()
    is_pandas = isinstance(x, pd.DataFrame)

    dropidx = 0
    nobs, nvar = x.shape
    if original in ["ex", "sep"]:
        dropidx = nvar
    if maxlag >= nobs:
        raise ValueError("maxlag should be < nobs")
    lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)))
    for k in range(0, int(maxlag + 1)):
        lm[
            maxlag - k : nobs + maxlag - k,
            nvar * (maxlag - k) : nvar * (maxlag - k + 1),
        ] = x

    if trim in ("none", "forward"):
        startobs = 0
    elif trim in ("backward", "both"):
        startobs = maxlag
    else:
        raise ValueError("trim option not valid")

    if trim in ("none", "backward"):
        stopobs = len(lm)
    else:
        stopobs = nobs

    if is_pandas:
        x = orig
        x_columns = x.columns if isinstance(x, pd.DataFrame) else [x.name]
        columns = [str(col) for col in x_columns]
        for lag in range(maxlag):
            lag_str = str(lag + 1)
            columns.extend([str(col) + ".L." + lag_str for col in x_columns])
        lm = pd.DataFrame(lm[:stopobs], index=x.index, columns=columns)
        lags = lm.iloc[startobs:]
        if original in ("sep", "ex"):
            leads = lags[x_columns]
            lags = lags.drop(x_columns, axis=1)
    else:
        lags = lm[startobs:stopobs, dropidx:]
        if original == "sep":
            leads = lm[startobs:stopobs, :dropidx]

    if original == "sep":
        return lags, leads
    else:
        return lags


def coint_johansen(endog, det_order=-1, k_ar_diff=1, return_eigenvalues=False):
    """Johansen cointegration test of the cointegration rank of a VECM, abbreviated from Statsmodels"""

    def detrend(y, order):
        if order == -1:
            return y
        else:
            from statsmodels.regression.linear_model import OLS
        return OLS(y, np.vander(np.linspace(-1, 1, len(y)), order + 1)).fit().resid

    def resid(y, x):
        if x.size == 0:
            return y
        r = y - np.dot(x, np.dot(np.linalg.pinv(x), y))
        return r

    endog = np.asarray(endog)

    # f is detrend transformed series, det_order is detrend data
    if det_order > -1:
        f = 0
    else:
        f = det_order

    endog = detrend(endog, det_order)
    dx = np.diff(endog, 1, axis=0)
    z = lagmat(dx, k_ar_diff)
    z = z[k_ar_diff:]
    z = detrend(z, f)

    dx = dx[k_ar_diff:]

    dx = detrend(dx, f)
    r0t = resid(dx, z)
    # GH 5731, [:-0] does not work, need [:t-0]
    lx = endog[: (endog.shape[0] - k_ar_diff)]
    lx = lx[1:]
    dx = detrend(lx, f)
    rkt = resid(dx, z)  # level on lagged diffs
    # Level covariance after filtering k_ar_diff
    skk = np.dot(rkt.T, rkt) / rkt.shape[0]
    # Covariacne between filtered and unfiltered
    sk0 = np.dot(rkt.T, r0t) / rkt.shape[0]
    s00 = np.dot(r0t.T, r0t) / r0t.shape[0]
    sig = np.dot(sk0, np.dot(np.linalg.pinv(s00), sk0.T))
    tmp = np.linalg.pinv(skk)
    au, du = np.linalg.eig(np.dot(tmp, sig))  # au is eval, du is evec

    temp = np.linalg.pinv(np.linalg.cholesky(np.dot(du.T, np.dot(skk, du))))
    dt = np.dot(du, temp)
    if return_eigenvalues:
        return au, dt
    else:
        return dt


def btcd_decompose(
    p_mat: np.ndarray,
    regression_model,
    max_lag: int = 1,
    return_eigenvalues=False,
):
    """Calculate decomposition.
    p_mat is of shape(t,n), wide style data.
    """
    B_sqrt_inv = _get_b_sqrt_inv(p_mat)
    A = _get_A(p_mat, regression_model, max_lag=max_lag)
    D = np.matmul(np.matmul(B_sqrt_inv, A), B_sqrt_inv)
    eigenvalues, eigenvectors = np.linalg.eigh(D)
    eigenvectors = np.matmul(B_sqrt_inv, eigenvectors)
    if return_eigenvalues:
        return eigenvalues, eigenvectors
    else:
        return eigenvectors


def _get_expected_dyadic_prod(V):
    return (1.0 / V.shape[0]) * np.matmul(V.T, V)


def _get_b_sqrt_inv(p_mat):
    """Rows of p_mat represent t index, columns represent each path."""
    B = _get_expected_dyadic_prod(p_mat)
    B_sqrt = fractional_matrix_power(B, 0.5)
    return np.linalg.pinv(B_sqrt)


def _get_y(p_mat: np.ndarray, p_mat_col_idx: int, max_lag: int):
    """
    Returns a 1D array which corresonds to a specific column of p_mat,
    with the first max_lag idxs trimmed.
    the index of this column is p_mat_col_idx
    """
    return p_mat[max_lag:, p_mat_col_idx]


def _get_q_t(regression_model, X: np.ndarray, y: np.ndarray):
    """
    Expected value for p_t (q model) using RegressionModel.
    - X is a numpy 2D array of shape (T-max_lag, n_features)
    - y is a numpy 1D array of shape (T-max_lag,)
    """
    regression_model.fit(X, y)
    return regression_model.predict(X)


def _get_A(p_mat: np.ndarray, regression_model, max_lag: int = 1):
    """Estimate A using an instance of RegressionModel."""
    X = np.concatenate(
        [p_mat[max_lag - lag : -lag, :] for lag in range(1, max_lag + 1)], axis=1
    )
    qs = []
    # model each column j of p_mat.
    for j in range(p_mat.shape[1]):
        y = _get_y(p_mat, j, max_lag)
        q_j = _get_q_t(regression_model, X, y)
        qs.append(q_j)
    q_mat = np.asarray(qs).T
    return _get_expected_dyadic_prod(q_mat)


def fourier_series(dates, period, series_order):
    """Provides Fourier series components with the specified frequency
    and order.

    Parameters
    ----------
    dates: pd.Series containing timestamps.
    period: Number of days of the period.
    series_order: Number of components.

    Returns
    -------
    Matrix with seasonality features.
    """
    # Fourier Detrend
    # periods, order, start_shift, and scaling (multi or univariate)
    # then just subtract

    # convert to days since epoch
    dates = pd.date_range("2020-01-01", "2022-01-01", freq="D")
    t = np.array(
        (dates - datetime.datetime(1970, 1, 1)).total_seconds().astype(float)
    ) / (3600 * 24.0)
    result = np.column_stack(
        [
            fun((2.0 * (i + 1) * np.pi * t / period))
            for i in range(series_order)
            for fun in (np.sin, np.cos)
        ]
    )
