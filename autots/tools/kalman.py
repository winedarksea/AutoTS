#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:29:58 2025
"""

import numpy as np
import pandas as pd
try:
    from scipy.stats import norm
except Exception:
    pass


def robust_pinv(M, rcond=1e-15, reg=1e-8):
    # Add a small regularization term to the diagonal
    M_reg = M + reg * np.eye(M.shape[0])
    return np.linalg.pinv(M_reg, rcond=rcond)


def kalman_fusion_forecasts(
    F: np.ndarray,
    index,
    columns,
    coverage: float = 0.90,
    method: str = "multi_series",
    Q_init=0.1,
    R_init=1.0,
    adapt_Q: str = None,    # "none" or "spread"
    adapt_R: str = None,    # "none" or "spread"
    initial_x=None,
    initial_P=None,
    min_std=1e-15,
    scale: bool = True,
    a: float = 1.0  # autoregressive term
):
    """
    Fuse multiple forecasts using a Kalman Filter for each forecast step.

    Args:
        F (np.ndarray): Forecasts of shape (n, S, T)
            - n = number of models
            - S = forecast length (time steps)
            - T = number of time series
        index (pd.Index): Index for the returned DataFrames (length=S).
        columns (list-like): Column names for the returned DataFrames (length=T).
        coverage (float): e.g. 0.90 => ~1.645 standard deviations for intervals.
        method (str): Either "multi_series" or "per_series".
            - "multi_series": single (T-dim) Kalman Filter for all T series together
            - "per_series": run T separate 1D Kalman Filters, one per series
        Q_init (float or np.ndarray): Base process noise (or matrix).
        R_init (float or np.ndarray): Base measurement noise (or matrix).
        adapt_Q (str): If "spread", adapt Q each step based on model spread.
        adapt_R (str): If "spread", adapt R each step based on model spread.
        initial_x (np.ndarray): initial state guess.
            - For multi_series: shape (T,)
            - For per_series: shape (T,)
        initial_P (np.ndarray): initial covariance guess.
            - For multi_series: shape (T, T)
            - For per_series: shape (T,)
        min_std (float): Small floor value for numerical stability in standard deviations.
        scale (bool): If True, each series is scaled (by its standard deviation) before filtering.
                      The fused forecasts are converted back to the original scale.

    Returns:
        df_point (pd.DataFrame): Kalman-fused point forecasts, shape (S, T).
        df_lower (pd.DataFrame): Lower bound, shape (S, T).
        df_upper (pd.DataFrame): Upper bound, shape (S, T).

    Notes:
        - In "multi_series" mode the state is a T-dimensional vector that is updated with an (n*T)-dimensional
          measurement (the stacked forecasts from the n models).
        - In "per_series" mode the algorithm runs T separate 1D Kalman Filters (one per series). In that case,
          for each forecast step (loop over S) you update each of the T filters independently.
        - When scale=True, each series is divided by its standard deviation (computed over all forecasts)
          before filtering; after filtering the outputs are multiplied by the same scale so that the returned forecasts
          are in the original feature space.
    """
    # shapes
    n, S, T = F.shape

    # Optional scaling: compute per-series scales and rescale F (and initial values)
    if scale:
        # Compute a scale (here, standard deviation) for each time series (last axis)
        scales = np.std(F, axis=(0, 1), ddof=1)
        scales[scales == 0] = 1.0  # avoid division by zero

        # Scale the forecasts: broadcast scales to shape (1, 1, T)
        F = F / scales[None, None, :]

        # Adjust initial state (if provided) to be in the scaled space.
        if initial_x is not None:
            initial_x = np.asarray(initial_x, dtype=float) / scales

        # Adjust the initial covariance.
        if initial_P is not None:
            if method == "multi_series":
                initial_P = np.array(initial_P, dtype=float)
                # If initial_P is a full matrix, scale it as: P_scaled = D^-1 P D^-1
                D_inv = np.diag(1.0 / scales)
                initial_P = D_inv @ initial_P @ D_inv
            else:  # per_series mode: initial_P is a vector
                initial_P = np.array(initial_P, dtype=float) / (scales**2)

    # z-value for coverage (assuming normal)
    z = norm.ppf(0.5 + coverage / 2.0)

    # -------------
    # Helper: build Q, R each step, possibly adapting
    # -------------
    def compute_process_noise(k, base_Q, x_dim, spread_value=1.0):
        """
        Return Q_k for time step k, shape = (x_dim, x_dim).
        If adapt_Q=="spread", scale by spread_value.
        """
        if np.isscalar(base_Q):
            val = base_Q
            if adapt_Q == "spread":
                val *= spread_value  # scale
            return val * np.eye(x_dim)
        else:
            # base_Q is a matrix
            if adapt_Q == "spread":
                return base_Q * spread_value
            else:
                return base_Q

    def compute_measurement_noise(k, base_R, meas_dim, spread_value=1.0):
        """
        Return R_k for time step k, shape = (meas_dim, meas_dim).
        If adapt_R=="spread", scale by spread_value.
        """
        if np.isscalar(base_R):
            val = base_R
            if adapt_R == "spread":
                val *= spread_value
            return val * np.eye(meas_dim)
        else:
            # base_R is a matrix
            if adapt_R == "spread":
                return base_R * spread_value
            else:
                return base_R

    # -------------
    # MODE 1: MULTI-SERIES (method="multi_series")
    # A single T-dimensional Kalman filter is used; the measurement at each step
    # is an (n*T)-dimensional vector formed by stacking all n models’ forecasts.
    # -------------
    if method == "multi_series":
        # State dimension = T
        x_dim = T
        # Measurement dimension = n*T
        meas_dim = n * T

        # Build measurement matrix H of shape (n*T, T) by stacking n identity matrices.
        H = np.tile(np.eye(T), (n, 1))  # shape = (n*T, T)

        # Transition matrix A = I (random walk)
        A = a * np.eye(T)
        # it woulds be useful to add a local linear trend version as an option here

        if initial_x is None:
            x_0 = np.mean(F[:, 0, :], axis=0)  # shape (T,)
        else:
            x_0 = np.asarray(initial_x, dtype=float).reshape((T,))

        # Set up initial covariance P_0
        if initial_P is None:
            P_0 = np.eye(T)
        else:
            P_0 = np.array(initial_P, dtype=float)
            if P_0.shape != (T, T):
                raise ValueError("initial_P must be shape (T,T) for multi_series mode")

        # Prepare arrays for outputs
        X_kalman = np.zeros((S, T))
        X_lower = np.zeros((S, T))
        X_upper = np.zeros((S, T))

        # Initialize current state and covariance
        x_curr = x_0
        P_curr = P_0

        for k in range(S):
            # Compute spread value if adapting Q or R:
            if adapt_R == "spread" or adapt_Q == "spread":
                # F[:, k, :] has shape (n, T); compute variance per series and average them.
                var_per_series = np.var(F[:, k, :], axis=0, ddof=1)
                spread_value = float(np.mean(var_per_series))
            else:
                spread_value = 1.0

            # Time update (prediction)
            x_pred = A @ x_curr
            Q_k = compute_process_noise(k, Q_init, x_dim, spread_value=spread_value)
            P_pred = A @ P_curr @ A.T + Q_k

            # Measurement update
            z_k = F[:, k, :].reshape(n * T)  # stack the forecasts into a (n*T,) vector
            y_k = H @ x_pred
            R_k = compute_measurement_noise(k, R_init, meas_dim, spread_value=spread_value)
            S_k = H @ P_pred @ H.T + R_k
            K_k = P_pred @ H.T @ np.linalg.pinv(S_k)
            x_update = x_pred + K_k @ (z_k - y_k)
            I_t = np.eye(T)
            tmp = I_t - K_k @ H
            # Joseph form for stability:
            P_update = tmp @ P_pred @ tmp.T + K_k @ R_k @ K_k.T

            # Store the results for time step k
            X_kalman[k, :] = x_update
            stds = np.sqrt(np.maximum(np.diag(P_update), min_std))
            X_lower[k, :] = x_update - z * stds
            X_upper[k, :] = x_update + z * stds

            # Move to next step
            x_curr = x_update
            P_curr = P_update

        df_point = pd.DataFrame(X_kalman, index=index, columns=columns)
        df_lower = pd.DataFrame(X_lower, index=index, columns=columns)
        df_upper = pd.DataFrame(X_upper, index=index, columns=columns)

        # Unscale the results if scaling was applied.
        if scale:
            # Multiply each column i by its scale factor.
            scale_series = pd.Series(scales, index=columns)
            df_point = df_point.multiply(scale_series, axis=1)
            df_lower = df_lower.multiply(scale_series, axis=1)
            df_upper = df_upper.multiply(scale_series, axis=1)

        return df_point, df_lower, df_upper

    # -------------
    # MODE 2: PER SERIES (method="per_series")
    # Run T independent 1D Kalman filters—one for each time series.
    # For each forecast step k (looping over S), we update each series (looping over T).
    # -------------
    elif method == "per_series":
        # Prepare output arrays: each is of shape (S, T)
        X_kalman = np.zeros((S, T))
        X_lower = np.zeros((S, T))
        X_upper = np.zeros((S, T))

        # Set up initial states.
        if initial_x is None:
            # Default: each series gets the average of the n models’ forecast at time 0.
            x_0 = np.mean(F[:, 0, :], axis=0)  # shape (T,)
        else:
            x_0 = np.asarray(initial_x, dtype=float)
            if x_0.shape != (T,):
                raise ValueError("initial_x must be shape (T,) for 'per_series' mode")

        if initial_P is None:
            P_0 = np.ones(T)  # one variance per series
        else:
            P_0 = np.array(initial_P, dtype=float)
            if P_0.shape != (T,):
                raise ValueError("initial_P must be shape (T,) for 'per_series' mode")

        # For each series i we store the current state and variance.
        x_curr_all = x_0.copy()
        P_curr_all = P_0.copy()

        # Define the measurement model: since state is scalar, let H be a 1D vector of ones (length n)
        H = np.ones(n)  # now H is shape (n,)

        for k in range(S):
            # For each forecast step, update each series i
            for i in range(T):
                # Time update (prediction)
                x_pred = x_curr_all[i]
                if adapt_Q == "spread" or adapt_R == "spread":
                    var_models = np.var(F[:, k, i], ddof=1)
                else:
                    var_models = 1.0

                base_Q = Q_init if np.isscalar(Q_init) else Q_init[i]
                Q_k = base_Q * var_models if adapt_Q == "spread" else base_Q
                P_pred = P_curr_all[i] + Q_k

                # Measurement update:
                z_k = F[:, k, i]  # n measurements for series i at time k (shape (n,))
                y_k = H * x_pred  # predicted measurement (shape (n,))
                base_R = R_init if np.isscalar(R_init) else R_init[i]
                if adapt_R == "spread":
                    R_k = base_R * var_models * np.eye(n)
                else:
                    R_k = base_R * np.eye(n)
                # Innovation covariance: note use of np.outer to form H*H^T.
                S_k = P_pred * np.outer(H, H) + R_k  # shape (n, n)
                # Kalman gain: K_k is a 1D vector of length n.
                K_k = (P_pred * H) @ robust_pinv(S_k)
                # K_k, residuals, rank, s = np.linalg.lstsq(S_k, P_pred @ H.T, rcond=None)

                # Compute the scalar update; ensure a scalar is returned.
                innovation = z_k - y_k  # shape (n,)
                x_update = x_pred + np.dot(K_k, innovation)
                x_update = float(x_update)  # ensure scalar
                # Update covariance (using a simplified Joseph form)
                tmp = 1 - np.dot(K_k, H)
                P_update = tmp * P_pred + np.dot(K_k, R_k @ K_k)
                P_update = float(P_update)

                # Store and update the state for series i.
                x_curr_all[i] = x_update
                P_curr_all[i] = P_update
                X_kalman[k, i] = x_update
                std_i = np.sqrt(max(P_update, min_std))
                X_lower[k, i] = x_update - z * std_i
                X_upper[k, i] = x_update + z * std_i

        df_point = pd.DataFrame(X_kalman, index=index, columns=columns)
        df_lower = pd.DataFrame(X_lower, index=index, columns=columns)
        df_upper = pd.DataFrame(X_upper, index=index, columns=columns)

        # Unscale the results if scaling was applied.
        if scale:
            scale_series = pd.Series(scales, index=columns)
            df_point = df_point.multiply(scale_series, axis=1)
            df_lower = df_lower.multiply(scale_series, axis=1)
            df_upper = df_upper.multiply(scale_series, axis=1)

        return df_point, df_lower, df_upper

    else:
        raise ValueError(f"Unrecognized method: {method}")
