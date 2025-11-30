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


def coint_fast(endog, k_ar_diff=1):
    """A fast estimation of cointegration vectors for VECM.

    Args:
        endog (np.array): should be (nobs, n_vars)
        k_ar_diff (int): number of lags to use in VECM.
    """
    endog = np.asarray(endog)
    if endog.shape[0] < k_ar_diff + 2:
        raise ValueError("Not enough observations for the specified lag order")

    # this is equivalent to Johansen MLE, but faster
    # this is also equivalent to Canonical Correlation Analysis (CCA)
    dx = np.diff(endog, axis=0)
    # lagged levels
    x_level = endog[:-1]

    # Handle lagged differences more efficiently
    if k_ar_diff > 1:
        dx_lag = lagmat(dx, k_ar_diff - 1)
        # Ensure we have enough observations after lagging
        if dx_lag.shape[0] <= k_ar_diff:
            raise ValueError("Not enough observations after creating lags")
        dx_lag = dx_lag[k_ar_diff - 1 :]
    else:
        dx_lag = np.empty((dx.shape[0] - k_ar_diff + 1, 0))

    # Align arrays properly
    dx = dx[k_ar_diff - 1 :]
    x_level = x_level[k_ar_diff - 1 :]

    # project out lagged differences using more stable computation
    if dx_lag.shape[1] > 0:
        # Use QR decomposition for better numerical stability
        Q, R = np.linalg.qr(dx_lag)
        if R.shape[0] > 0:
            proj_matrix = Q @ Q.T
            dx = dx - proj_matrix @ dx
            x_level = x_level - proj_matrix @ x_level

    # Add small regularization for numerical stability
    reg = 1e-12

    # get covariance matrices more efficiently
    n_obs = dx.shape[0]
    dx_centered = dx - np.mean(dx, axis=0)
    x_level_centered = x_level - np.mean(x_level, axis=0)

    c0 = (dx_centered.T @ dx_centered) / (n_obs - 1) + reg * np.eye(dx.shape[1])
    c1 = (x_level_centered.T @ x_level_centered) / (n_obs - 1) + reg * np.eye(
        x_level.shape[1]
    )
    c01 = (dx_centered.T @ x_level_centered) / (n_obs - 1)

    # solve generalized eigenvalue problem with better numerical stability
    from scipy.linalg import eigh, LinAlgError

    try:
        # Use Cholesky decomposition for better numerical stability
        L0 = np.linalg.cholesky(c0)
        L1 = np.linalg.cholesky(c1)

        # Transform to standard eigenvalue problem
        A = np.linalg.solve(L1, c01.T)
        A = np.linalg.solve(L0, A.T).T
        A = A @ A.T

        eigenvalues, eigenvectors_temp = eigh(A)
        # Transform back
        eigenvectors = np.linalg.solve(L1, eigenvectors_temp)

    except LinAlgError:
        # Fallback to regularized approach if Cholesky fails
        eigenvalues, eigenvectors = eigh(c01 @ np.linalg.solve(c0, c01.T), c1)

    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    return eigenvectors


def coint_johansen(
    endog, det_order=-1, k_ar_diff=1, return_eigenvalues=False, fast: bool = True
):
    """Johansen cointegration test of the cointegration rank of a VECM, abbreviated from Statsmodels"""
    if fast:
        dt = coint_fast(endog, k_ar_diff=k_ar_diff)
        if return_eigenvalues:
            # For fast method, we don't compute eigenvalues separately
            return None, dt
        else:
            return dt

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
    dx_level = detrend(lx, f)
    rkt = resid(dx_level, z)  # level on lagged diffs

    # Level covariance after filtering k_ar_diff - use more efficient matrix operations
    n_obs = rkt.shape[0]
    skk = rkt.T @ rkt / n_obs
    sk0 = rkt.T @ r0t / n_obs
    s00 = r0t.T @ r0t / n_obs

    # Add small regularization for numerical stability
    reg = 1e-12
    s00 += reg * np.eye(s00.shape[0])
    skk += reg * np.eye(skk.shape[0])

    sig = sk0 @ np.linalg.solve(s00, sk0.T)

    # Use solve instead of pinv for better numerical stability
    au, du = np.linalg.eig(np.linalg.solve(skk, sig))

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(au.real)[::-1]
    au = au[idx]
    du = du[:, idx]

    try:
        temp = np.linalg.solve(np.linalg.cholesky(du.T @ skk @ du), np.eye(du.shape[1]))
        dt = du @ temp
    except np.linalg.LinAlgError:
        # Fallback if Cholesky decomposition fails
        dt = du

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
    p_mat = np.asarray(p_mat)
    if p_mat.shape[0] < max_lag + 2:
        raise ValueError("Not enough observations for the specified lag order")

    B_sqrt_inv = _get_b_sqrt_inv(p_mat)
    A = _get_A(p_mat, regression_model, max_lag=max_lag)

    # More efficient matrix multiplication using @ operator
    D = B_sqrt_inv @ A @ B_sqrt_inv

    eigenvalues, eigenvectors = np.linalg.eigh(D)

    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Transform eigenvectors back
    eigenvectors = B_sqrt_inv @ eigenvectors

    if return_eigenvalues:
        return eigenvalues, eigenvectors
    else:
        return eigenvectors


def _get_expected_dyadic_prod(V):
    return (1.0 / V.shape[0]) * np.matmul(V.T, V)


def _get_b_sqrt_inv(p_mat):
    """Rows of p_mat represent t index, columns represent each path."""
    B = _get_expected_dyadic_prod(p_mat)

    # Use eigendecomposition for more stable matrix square root
    eigenvals, eigenvecs = np.linalg.eigh(B)

    # Add small regularization to avoid numerical issues
    eigenvals = np.maximum(eigenvals, 1e-12)

    # Compute B^(-1/2) = Q * diag(lambda^(-1/2)) * Q^T
    sqrt_inv_eigenvals = 1.0 / np.sqrt(eigenvals)
    B_sqrt_inv = eigenvecs @ np.diag(sqrt_inv_eigenvals) @ eigenvecs.T

    return B_sqrt_inv


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
    """Estimate A using an instance of RegressionModel - vectorized version."""
    # Create lagged matrix more efficiently
    if max_lag == 1:
        X = p_mat[:-1, :]
    else:
        X = np.concatenate(
            [p_mat[max_lag - lag : -lag, :] for lag in range(1, max_lag + 1)], axis=1
        )

    y_mat = p_mat[max_lag:, :]  # All target variables at once

    # Try to use multioutput regression if available
    try:
        # Check if the model supports multioutput
        if hasattr(regression_model, 'fit'):
            regression_model.fit(X, y_mat)
            q_mat = regression_model.predict(X)
        else:
            # Fallback to loop if multioutput not supported
            qs = []
            for j in range(p_mat.shape[1]):
                y = y_mat[:, j]
                q_j = _get_q_t(regression_model, X, y)
                qs.append(q_j)
            q_mat = np.column_stack(qs)
    except (ValueError, TypeError):
        # Fallback to original loop-based approach
        qs = []
        for j in range(p_mat.shape[1]):
            y = _get_y(p_mat, j, max_lag)
            q_j = _get_q_t(regression_model, X, y)
            qs.append(q_j)
        q_mat = np.column_stack(qs)

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
