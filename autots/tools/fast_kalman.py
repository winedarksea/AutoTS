# encoding: utf-8
"""From SIMD KALMAN, (c) 2017 Otto Seiskari (MIT License)

Some other resources that I have found useful:
    https://kevinkotze.github.io/ts-4-state-space/
    https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_dfm_coincident.html
    an Introduction to State Space Time Series Analysis, Commandeur and Koopman, chp 8
    Forecasting, structural time series models, and the Kalman Filter, Andrew Harvey

Following the notation in [1]_, the Kalman filter framework consists of
a *dynamic model* (state transition model)

.. math::

    x_k = A x_{k-1} + q_{k-1}, \\qquad q_{k-1} \\sim N(0, Q)

and a *measurement model* (observation model)

.. math::

    y_k = H x_k + r_k, \\qquad r_k \\sim N(0, R),

where the vector :math:`x` is the (hidden) state of the system and
:math:`y` is an observation. `A` and `H` are matrices of suitable shape
and :math:`Q`, :math:`R` are positive-definite noise covariance matrices.


.. [1] Simo Sarkkä (2013).
   Bayesian Filtering and Smoothing. Cambridge University Press.
   https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf


Usage example
---------------


   import numpy.random
   numpy.random.seed(0)

Define model

   import simdkalman
   import numpy as np

   kf = simdkalman.KalmanFilter(
       state_transition = [[1,1],[0,1]],        # matrix A
       process_noise = np.diag([0.1, 0.01]),    # Q
       observation_model = np.array([[1,0]]),   # H
       observation_noise = 1.0)                 # R

Generate some fake data

   import numpy.random as random

   # 100 independent time series
   data = random.normal(size=(100, 200))

   # with 10% of NaNs denoting missing values
   data[random.uniform(size=data.shape) < 0.1] = np.nan


Smooth all data

   smoothed = kf.smooth(data,
                        initial_value = [1,0],
                        initial_covariance = np.eye(2) * 0.5)

   # second timeseries, third time step, hidden state x
   print('mean')
   print(smoothed.states.mean[1,2,:])

   print('covariance')
   print(smoothed.states.cov[1,2,:,:])


    mean
    [ 0.29311384 -0.06948961]
    covariance
    [[ 0.19959416 -0.00777587]
     [-0.00777587  0.02528967]]

Predict new data for a single series (1d case)


   predicted = kf.predict(data[1,:], 123)

   # predicted observation y, third new time step
   pred_mean = predicted.observations.mean[2]
   pred_stdev = np.sqrt(predicted.observations.cov[2])

   print('%g +- %g' % (pred_mean, pred_stdev))

   1.71543 +- 1.65322

Low-level Kalman filter computation steps with multi-dimensional input arrays.
Unlike with the `KalmanFilter <index.html#simdkalman.KalmanFilter>`_ class,
all inputs must be numpy arrays. However, their dimensions can flexibly vary
form 1 to 3 as long as they are reasonable from the point of view of matrix
multiplication and numpy broadcasting rules. Matrix operations are applied on
the *last* two axes of the arrays.
"""
import random
import numpy as np
from functools import wraps


def random_state_space_original():
    """Return randomly generated statespace models."""
    n_dims = random.choices([1, 2, 3, 4, 8], [0.1, 0.2, 0.3, 0.4, 0.3])[0]
    if n_dims == 1:
        st = np.array([[1]])
        obsmod = np.random.randint(1, 3, (1, n_dims))
        procnois = np.diag(np.random.exponential(0.01, size=(n_dims)).round(3))
    else:
        st = np.random.choice([0, 1, -1], p=[0.75, 0.2, 0.05], size=(n_dims, n_dims))
        st[0, 0] = 1
        st[0, -1] = 0
        if n_dims == 2:
            obsmod = np.array([[1, 0]])
        else:
            obsmod = np.array([[1, 1] + [0] * (n_dims - 2)])
        procnois = (
            np.diag([0.2 / random.choice([1, 5, 10]), 0.001] + [0] * (n_dims - 2)) ** 2
        ).round(3)
    obsnois = random.choices(
        [1.0, 10.0, 2.0, 0.5, 0.2, 0.05, 0.001],
        [0.8, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    )[0]
    while (
        st.size > 1
        and np.all(st == 1)
        # or (st.size > 3 and np.isin(st.sum(axis=1), [0]).any())
    ):
        st, procnois, obsmod, obsnois = random_state_space()
    return st, procnois, obsmod, obsnois


def ensure_stability(st):
    eigenvalues, eigenvectors = np.linalg.eig(st)
    # Scale eigenvalues to ensure their absolute values are less than 1
    stable_eigenvalues = eigenvalues / (np.abs(eigenvalues) + 1e-5)
    st_stable = eigenvectors @ np.diag(stable_eigenvalues) @ np.linalg.inv(eigenvectors)
    return st_stable.real


def random_matrix(rows, cols, density=0.2):
    matrix = np.random.randn(rows, cols)
    sparsity_mask = np.random.rand(rows, cols) < density
    return np.where(sparsity_mask, matrix, 0)


def random_state_space(tries=15):
    for _ in range(tries):
        try:
            n_dims = random.choices([1, 2, 3, 4, 8], [0.1, 0.2, 0.3, 0.4, 0.3])[0]
            st = random_matrix(n_dims, n_dims, density=0.5)
            st = ensure_stability(st)
            obsmod = random_matrix(
                1, n_dims, density=1.0
            )  # Full observation for simplicity
            procnois = np.diag(np.random.exponential(0.01, size=n_dims)).round(3)
            obsnois = np.random.exponential(1.0)

            if np.all(np.abs(np.linalg.eigvals(st)) < 1):  # Check stability
                return st, procnois, obsmod, obsnois
        except Exception:
            pass
    # fallback
    return random_state_space_original()


def holt_winters_damped_matrices(M, alpha, beta, gamma, phi=1.0):
    """Not sure if this is correct. It's close, at least."""
    # State Transition Matrix F
    # Level & Trend Equations

    F_lt = np.array(
        [  # not sure about having the alpha and beta here
            [1 + (alpha * (1 - phi)), phi],
            [beta * (1 - phi), phi],
        ]
    )
    # Seasonal Equation
    F_s = np.eye(M, M, -1)  # This creates an identity matrix and shifts it down
    first_row = np.zeros((1, M))
    first_row[0, -1] = 1
    F_s = np.vstack([first_row, F_s[:-1]])
    F_top = np.hstack([F_lt, np.zeros((2, M))])
    F_bottom = np.hstack([np.zeros((M, 2)), F_s])
    F = np.vstack([F_top, F_bottom])
    """
    F = np.zeros((M+2, M+2))
    F[0, 0] = 1
    F[0, 1] = phi
    F[1, 0] = 1
    F[1, 1] = phi
    for i in range(2, M+2):
        F[i, i] = 1
    """

    # Process Noise Covariance Q
    Q = np.zeros((M + 2, M + 2))
    # Assuming the same variance for all states for simplicity.
    # Modify these values based on specific requirements.
    Q[0, 0] = alpha
    Q[1, 1] = beta
    for i in range(2, M + 2):
        Q[i, i] = gamma

    # Observation Matrix H
    H = np.zeros((1, M + 2))
    H[0, 0] = 1
    H[0, 1] = 1
    H[0, -1] = 1

    # Observation Noise Covariance R
    R = np.array([[1]])

    return F, Q, H, R


def new_kalman_params(method=None, allow_auto=True):
    if method in ['fast']:
        em_iter = random.choices([None, 10], [0.8, 0.2])[0]
    elif method == "superfast":
        em_iter = None
    elif method == "deep":
        em_iter = random.choices([None, 10, 20, 50, 100], [0.3, 0.6, 0.1, 0.1, 0.1])[0]
    else:
        em_iter = random.choices([None, 10, 30], [0.3, 0.7, 0.1])[0]

    K = random.choices([2, 3, 6, 8], [0.1, 0.6, 0.1, 0.1])[0]
    sigma_level2 = 1e-2  # Placeholder
    sigma_slope2 = 1e-3  # Placeholder
    sigma_weekly2 = 1e-2  # Placeholder
    sigma_fourier2 = 1e-2  # Placeholder

    Q = np.block(
        [
            [sigma_level2, 0, np.zeros((1, 6)), np.zeros((1, 2 * K))],
            [0, sigma_slope2, np.zeros((1, 6)), np.zeros((1, 2 * K))],
            [np.zeros((6, 2)), sigma_weekly2 * np.eye(6), np.zeros((6, 2 * K))],
            [
                np.zeros((2 * K, 2)),
                np.zeros((2 * K, 6)),
                sigma_fourier2 * np.eye(2 * K),
            ],
        ]
    )

    # Deterministic Trend
    deterministic_trend = np.array([[1, 1], [0, 1]])

    weekly = np.eye(6, k=1)
    weekly[-1, 0] = -1  # Link the last state to the first to maintain cyclicality

    harmonics = []
    for k in range(1, K + 1):
        angle = 2 * np.pi * k / 365.25
        matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        harmonics.append(matrix)

    # Stack Fourier matrices
    fourier_yearly = np.block([harmonics[i] for i in range(K)])

    # The top part
    # F_top is of shape (2, 2+6+2K)
    F_top = np.hstack(
        [
            deterministic_trend,  # shape: (2, 2)
            np.zeros((2, 6)),  # shape: (2, 6)
            np.zeros((2, 2 * K)),  # shape: (2, 2K)
        ]
    )

    # The middle part (weekly seasonality)
    # F_mid is of shape (6, 2+6+2K)
    F_mid = np.hstack(
        [
            np.zeros((6, 2)),  # shape: (6, 2)
            weekly,  # shape: (6, 6)
            np.zeros((6, 2 * K)),  # shape: (6, 2K)
        ]
    )

    # The bottom part (yearly Fourier seasonality)
    # F_bot is of shape (2K, 2+6+2K)
    F_bot = np.hstack(
        [
            np.zeros((2 * K, 2)),  # shape: (2K, 2)
            np.zeros((2 * K, 6)),  # shape: (2K, 6)
            fourier_yearly.T,  # shape: (2K, 2K)
            np.zeros((2 * K, K * 2 - 2)),
        ]
    )
    # Vertically stack
    F = np.vstack([F_top, F_mid, F_bot])
    # Generating H for Fourier terms
    H_fourier = []
    for k in range(1, K + 1):
        H_fourier.extend(
            [np.cos(2 * np.pi * k / 365.25), np.sin(2 * np.pi * k / 365.25)]
        )
    H = np.hstack(([1, 0], np.ones(6), H_fourier))

    F_hw, Q_hw, H_hw, R_hw = holt_winters_damped_matrices(
        M=7,
        alpha=random.choice([0.9, 1.0, 0, 0.3]),
        beta=random.choice([0.9, 1.0, 0, 0.3]),
        gamma=random.choice([0.9, 1.0, 0, 0.3]),
        phi=random.choice([0.9, 1.0, 0.995, 0.98]),
    )

    params = random.choices(
        # the same model can sometimes be defined in various matrix forms
        [
            # floats are phi
            'ets_aan',
            {
                'model_name': 'local linear stochastic seasonal dummy',
                'state_transition': [
                    [1, 0, 0, 0],
                    [0, -1, -1, -1],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                ],
                'process_noise': [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                'observation_model': [[1, 1, 0, 0]],
                'observation_noise': random.choice([0.25, 'auto']),
            },
            {
                'model_name': 'local linear stochastic seasonal 7',
                'state_transition': [
                    [1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                'process_noise': [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                'observation_model': [[1, 0, 1, 0, 0, 0, 0, 0]],
                'observation_noise': random.choice([0.25, 'auto']),
            },
            {
                'model_name': 'MA',
                'state_transition': [[1, 0], [1, 0]],
                'process_noise': [[0.2, 0.0], [0.0, 0]],
                'observation_model': [[1, 0.1]],
                'observation_noise': 1.0,
            },
            {
                'model_name': 'AR(2)',
                'state_transition': [[1, 1], [0.1, 0]],
                'process_noise': [[1, 0], [0, 0]],
                'observation_model': [[1, 0]],
                'observation_noise': 1.0,
            },
            {
                'model_name': 'ucm_deterministic_trend',
                'state_transition': [[1, 1], [0, 1]],
                'process_noise': [[0.01, 0], [0, 0.01]],  # these would be tuned
                'observation_model': [[1, 0]],
                'observation_noise': 0.1,  # this would be tuned
            },
            {
                'model_name': 'X1',
                'state_transition': [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
                'process_noise': [
                    [0.1, 0.0, 0.0],
                    [0.0, 0.01, 0.0],
                    [0.0, 0.0, 0.1],
                ],
                'observation_model': [[1, 1, 1]],
                'observation_noise': 1.0,
            },
            # I believe this is a seasonal ETS model but I would like confirmation on that
            {
                'model_name': "local linear hidden state with seasonal 7",
                'state_transition': [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                'process_noise': [
                    [0.0016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                'observation_model': [[1, 1, 0, 0, 0, 0, 0, 0]],
                'observation_noise': random.choice(
                    [0.25, 0.5, 1.0, 0.04, 0.02, 'auto']
                ),
            },
            {
                'model_name': "factor",
                'state_transition': [
                    [1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                'process_noise': [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                'observation_model': [[1, 0, 0, 0, 0, 0]],
                'observation_noise': 0.04,
            },
            {
                'model_name': "ucm_deterministictrend_seasonal7",
                'state_transition': [
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, -1, -1, -1, -1, -1, -1],
                ],
                'process_noise': [
                    [0.001, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0.001, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0.001, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0.001, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.001, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0.001, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0.001, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                'observation_model': [[1, 0, 1, 1, 1, 1, 1, 1]],
                'observation_noise': random.choice([0.03, 'auto']),
            },
            "random",
            364,
            12,
            {
                'model_name': 'spline',
                'state_transition': [[2, -1], [1, 0]],
                'process_noise': [[1, 0], [0, 0]],
                'observation_model': [[1, 0]],
                'observation_noise': random.choice([0.1, 'auto']),
            },
            {
                'model_name': 'locallinear_weekly_fourier',
                'state_transition': F.tolist(),
                'process_noise': Q.tolist(),
                'observation_model': H[np.newaxis, ...].tolist(),
                'observation_noise': random.choice([0.01, 0.1, 'auto']),
            },
            {
                'model_name': 'holt_winters_damped',
                'state_transition': F_hw.tolist(),
                'process_noise': Q_hw.tolist(),
                'observation_model': H_hw.tolist(),
                'observation_noise': random.choice([0.01, 0.1, 1, 'auto']),
            },
            "dynamic_linear",
            "random_original",
        ],
        [
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.2,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ],
    )[0]
    if params in [364] and method not in ['deep']:
        params = 7
    if params == "random":
        st, procnois, obsmod, obsnois = random_state_space()
        params = {
            'model_name': 'randomly generated',
            'state_transition': st.tolist(),
            'process_noise': procnois.tolist(),
            'observation_model': obsmod.tolist(),
            'observation_noise': obsnois,
        }
    elif params == "random_original":
        st, procnois, obsmod, obsnois = random_state_space_original()
        params = {
            'model_name': 'randomly generated_original',
            'state_transition': st.tolist(),
            'process_noise': procnois.tolist(),
            'observation_model': obsmod.tolist(),
            'observation_noise': obsnois,
        }
    elif params == "dynamic_linear":
        choices = [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
        ]
        params = {
            'model_name': 'dynamic linear',
            'state_transition': [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, random.choice(choices), 1],
                [0, 0, random.choice(choices), 0],
            ],
            'process_noise': [
                [random.choice(choices), 0, 0, 0],
                [0, random.choice(choices), 0, 0],
                [0, 0, random.choice(choices), 0],
                [0, 0, 0, 0],
            ],
            'observation_model': [[1, 0, 1, 0]],
            'observation_noise': 0.25,
            'em_iter': 10,
        }
    elif params == "ets_aan":
        choices = [
            0.0,
            0.01,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
        ]
        params = {
            'model_name': 'local_linear_trend_ets_aan',
            'state_transition': [[1, 1], [0, 1]],
            'process_noise': [
                [random.choice(choices), 0.0],
                [0.0, random.choice(choices)],
            ],
            'observation_model': [[1, 0]],
            'observation_noise': random.choice([0.25, 0.5, 1.0, 0.05]),
        }
    elif isinstance(params, int):
        state_transition = np.zeros((params + 1, params + 1))
        state_transition[0, 0] = 1
        state_transition[1, 1:-1] = [-1.0] * (params - 1)
        state_transition[2:, 1:-1] = np.eye(params - 1)
        observation_model = [[1, 1] + [0] * (params - 1)]
        level_noise = 0.2 / random.choice([0.2, 0.5, 1, 5, 10, 200])
        season_noise = random.choice([1e-4, 1e-3, 1e-2, 1e-1])
        process_noise_cov = (
            np.diag([level_noise, season_noise] + [0] * (params - 1)) ** 2
        )
        params = {
            'model_name': f'local linear hidden state with seasonal {params}',
            'state_transition': state_transition.tolist(),
            'process_noise': process_noise_cov.tolist(),
            'observation_model': observation_model,
            'observation_noise': 0.04,
        }
    params['em_iter'] = em_iter
    if not allow_auto:
        if params['observation_noise'] == 'auto':
            params['observation_noise'] = 0.1
    return params


class Gaussian:
    def __init__(self, mean, cov):
        self.mean = mean
        if cov is not None:
            self.cov = cov
        else:
            self.cov = None

    @staticmethod
    def empty(n_states, n_vars, n_measurements, cov=True):
        mean = np.empty((n_vars, n_measurements, n_states))
        if cov:
            cov = np.empty(
                (n_vars, n_measurements, n_states, n_states), dtype=np.float32
            )
        else:
            cov = None
        return Gaussian(mean, cov)

    def unvectorize_state(self):
        n_states = self.mean.shape[-1]
        assert n_states == 1

        mean = self.mean
        cov = self.cov

        mean = mean[..., 0]
        if cov is not None:
            cov = cov[..., 0, 0]

        return Gaussian(mean, cov)

    def unvectorize_vars(self):
        n_vars = self.mean.shape[0]
        assert n_vars == 1

        mean = self.mean
        cov = self.cov

        mean = mean[0, ...]
        if cov is not None:
            cov = cov[0, ...]

        return Gaussian(mean, cov)

    def __str__(self):
        s = "mean:\n  %s" % str(self.mean).replace("\n", "\n  ")
        if self.cov is not None:
            s += "\ncov:\n  %s" % str(self.cov).replace("\n", "\n  ")
        return s


class KalmanFilter(object):
    """
    The main Kalman filter class providing convenient interfaces to
    vectorized smoothing and filtering operations on multiple independent
    time series.

    As long as the shapes of the given parameters match reasonably according
    to the rules of matrix multiplication, this class is flexible in their
    exact nature accepting

     * scalars: ``process_noise = 0.1``
     * (2d) numpy matrices: ``process_noise = numpy.eye(2)``
     * 2d arrays: ``observation_model = [[1,2]]``
     * 3d arrays and matrices for vectorized computations. Unlike the other
       options, this locks the shape of the inputs that can be processed
       by the smoothing and prediction methods.

    :param state_transition:
        State transition matrix :math:`A`

    :param process_noise:
        Process noise (state transition covariance) matrix :math:`Q`

    :param observation_model:
        Observation model (measurement model) matrix :math:`H`

    :param observation_noise:
        Observation noise (measurement noise covariance) matrix :math:`R`
    """

    # pylint: disable=W0232
    class Result:
        def __str__(self):
            s = ""
            for k, v in self.__dict__.items():
                if len(s) > 0:
                    s += "\n"
                s += "%s:\n" % k
                s += "  " + str(v).replace("\n", "\n  ")
            return s

    def __init__(
        self, state_transition, process_noise, observation_model, observation_noise
    ):
        state_transition = ensure_matrix(state_transition)
        n_states = state_transition.shape[-2]  # Allow different transitions

        process_noise = ensure_matrix(process_noise, n_states)
        observation_model = ensure_matrix(observation_model)
        n_obs = observation_model.shape[-2]
        observation_noise = ensure_matrix(observation_noise, n_obs)

        try:
            assert state_transition.shape[-2:] == (n_states, n_states)
            assert process_noise.shape[-2:] == (n_states, n_states)
            assert observation_model.shape[-2:] == (n_obs, n_states)
            assert observation_noise.shape[-2:] == (n_obs, n_obs)
        except Exception as e:
            raise ValueError(
                f"dimension mismatch: n_states: {n_states}, n_obs: {n_obs}"
            ) from e

        self.state_transition = state_transition
        self.process_noise = process_noise
        self.observation_model = observation_model
        self.observation_noise = observation_noise

    def predict_next(self, m, P):
        """
        Single prediction step

        :param m: :math:`{\\mathbb E}[x_{j-1}]`, the previous mean
        :param P: :math:`{\\rm Cov}[x_{j-1}]`, the previous covariance

        :rtype: ``(prior_mean, prior_cov)`` predicted mean and covariance
            :math:`{\\mathbb E}[x_j]`, :math:`{\\rm Cov}[x_j]`
        """
        return predict(m, P, self.state_transition, self.process_noise)

    def update(self, m, P, y, log_likelihood=False):
        """
        Single update step with NaN check.

        :param m: :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_{j-1}]`,
            the prior mean of :math:`x_j`
        :param P: :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_{j-1}]`,
            the prior covariance of :math:`x_j`
        :param y: observation :math:`y_j`
        :param log_likelihood: compute log-likelihood?
        :type states: boolean

        :rtype: ``(posterior_mean, posterior_covariance, log_likelihood)``
            posterior mean :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_j]`
            & covariance :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_j]`
            and, if requested, log-likelihood. If :math:`y_j` is NaN, returns
            the prior mean and covariance instead
        """
        return priv_update_with_nan_check(
            m,
            P,
            self.observation_model,
            self.observation_noise,
            y,
            log_likelihood=log_likelihood,
        )

    def predict_observation(self, m, P):
        """
        Probability distribution of observation :math:`y` for a given
        distribution of :math:`x`

        :param m: :math:`{\\mathbb E}[x]`
        :param P: :math:`{\\rm Cov}[x]`
        :rtype: mean :math:`{\\mathbb E}[y]` and
            covariance :math:`{\\rm Cov}[y]`
        """
        return predict_observation(m, P, self.observation_model, self.observation_noise)

    def smooth_current(self, m, P, ms, Ps):
        """
        Simgle Kalman smoother backwards step

        :param m: :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_j]`,
            the filtered mean of :math:`x_j`
        :param P: :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_j]`,
            the filtered covariance of :math:`x_j`
        :param ms:
            :math:`{\\mathbb E}[x_{j+1}|y_1,\\ldots,y_T]`
        :param Ps:
            :math:`{\\rm Cov}[x_{j+1}|y_1,\\ldots,y_T]`

        :rtype: ``(smooth_mean, smooth_covariance, smoothing_gain)``
            smoothed mean :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_T]`,
            and covariance :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_T]`
            & smoothing gain :math:`C`
        """
        return priv_smooth(m, P, self.state_transition, self.process_noise, ms, Ps)

    def predict(
        self,
        data,
        n_test,
        initial_value=None,
        initial_covariance=None,
        states=True,
        observations=True,
        covariances=True,
        verbose=False,
    ):
        """
        Filter past data and predict a given number of future values.
        The data can be given as either of

          * 1d array, like ``[1,2,3,4]``. In this case, one Kalman filter is
            used and the return value structure will contain an 1d array of
            ``observations`` (both ``.mean``  and ``.cov`` will be 1d).

          * 2d matrix, whose each row is interpreted as an independent time
            series, all of which are filtered independently. The returned
            ``observations`` members will be 2-dimensional in this case.

          * 3d matrix, whose the last dimension can be used for multi-dimensional
            observations, i.e, ``data[1,2,:]`` defines the components of the
            third observation of the second series. In the-multi-dimensional
            case the returned ``observations.mean`` will be 3-dimensional and
            ``observations.cov`` 4-dimensional.

        Initial values and covariances can be given as scalars or 2d matrices
        in which case the same initial states will be used for all rows or
        3d arrays for different initial values.

        :param data: Past data

        :param n_test:  number of future steps to predict.
        :type n_test: integer

        :param initial_value: Initial value :math:`{\\mathbb E}[x_0]`
        :param initial_covariance: Initial uncertainty :math:`{\\rm Cov}[x_0]`

        :param states: predict states :math:`x`?
        :type states: boolean
        :param observations: predict observations :math:`y`?
        :type observations: boolean

        :param covariances: include covariances in predictions?
        :type covariances: boolean

        :rtype: Result object with fields
            ``states`` and ``observations``, if the respective parameter flags
            are set to True. Both are ``Gaussian`` result objects with fields
            ``mean`` and ``cov`` (if the *covariances* flag is True)
        """

        return self.compute(
            data,
            n_test,
            initial_value,
            initial_covariance,
            smoothed=False,
            states=states,
            covariances=covariances,
            observations=observations,
            verbose=verbose,
        ).predicted

    def smooth(
        self,
        data,
        initial_value=None,
        initial_covariance=None,
        observations=True,
        states=True,
        covariances=True,
        verbose=False,
    ):
        """
        Smooth given data, which can be either

          * 1d array, like ``[1,2,3,4]``. In this case, one Kalman filter is
            used and the return value structure will contain an 1d array of
            ``observations`` (both ``.mean``  and ``.cov`` will be 1d).

          * 2d matrix, whose each row is interpreted as an independent time
            series, all of which are smoothed independently. The returned
            ``observations`` members will be 2-dimensional in this case.

          * 3d matrix, whose the last dimension can be used for multi-dimensional
            observations, i.e, ``data[1,2,:]`` defines the components of the
            third observation of the second series. In the-multi-dimensional
            case the returned ``observations.mean`` will be 3-dimensional and
            ``observations.cov`` 4-dimensional.

        Initial values and covariances can be given as scalars or 2d matrices
        in which case the same initial states will be used for all rows or
        3d arrays for different initial values.

        :param data: 1d or 2d data, see above
        :param initial_value: Initial value :math:`{\\mathbb E}[x_0]`
        :param initial_covariance: Initial uncertainty :math:`{\\rm Cov}[x_0]`

        :param states: return smoothed states :math:`x`?
        :type states: boolean
        :param observations: return smoothed observations :math:`y`?
        :type observations: boolean
        :param covariances: include covariances results?
        :type covariances: boolean

        :rtype: Result object with fields
            ``states`` and ``observations``, if the respective parameter flags
            are set to True. Both are ``Gaussian`` result objects with fields
            ``mean`` and ``cov`` (if the *covariances* flag is True)
        """

        return self.compute(
            data,
            0,
            initial_value,
            initial_covariance,
            smoothed=True,
            states=states,
            covariances=covariances,
            observations=observations,
            verbose=verbose,
        ).smoothed

    def compute(
        self,
        data,
        n_test,
        initial_value=None,
        initial_covariance=None,
        smoothed=True,
        filtered=False,
        states=True,
        covariances=True,
        observations=True,
        likelihoods=False,
        gains=False,
        log_likelihood=False,
        verbose=False,
    ):
        """
        Smoothing, filtering and prediction at the same time. Used internally
        by other methods, but can also be used directly if, e.g., both smoothed
        and predicted data is wanted.

        See **smooth** and **predict** for explanation of the common parameters.
        With this method, there also exist the following flags.

        :param smoothed: compute Kalman smoother (used by **smooth**)
        :type smoothed: boolean
        :param filtered: return (one-way) filtered data
        :type filtered: boolean
        :param likelihoods: return likelihoods of each step
        :type likelihoods: boolean
        :param gains: return Kalman gains and pairwise covariances (used by
            the EM algorithm). If true, the gains are provided as a member of
            the relevant subresult ``filtered.gains`` and/or ``smoothed.gains``.
        :type gains: boolean
        :param log_likelihood: return the log-likelihood(s) for the entire
            series. If matrix data is given, this will be a vector where each
            element is the log-likelihood of a single row.
        :type log_likelihood: boolean

        :rtype: result object whose fields depend on of the above parameter
            flags are True. The possible values are:
            ``smoothed`` (the return value of **smooth**, may contain ``smoothed.gains``),
            ``filtered`` (like ``smoothed``, may also contain ``filtered.gains``),
            ``predicted`` (the return value of **predict** if ``n_test > 0``)
            ``pairwise_covariances``, ``likelihoods`` and
            ``log_likelihood``.
        """

        # pylint: disable=W0201
        result = KalmanFilter.Result()

        data = ensure_matrix(data)
        single_sequence = len(data.shape) == 1
        if single_sequence:
            data = data[np.newaxis, :]

        n_vars = data.shape[0]
        n_measurements = data.shape[1]
        n_states = self.state_transition.shape[0]
        n_obs = self.observation_model.shape[-2]

        def empty_gaussian(
            n_states=n_states, n_measurements=n_measurements, cov=covariances
        ):
            return Gaussian.empty(n_states, n_vars, n_measurements, cov)

        def auto_flat_observations(obs_gaussian):
            r = obs_gaussian
            if n_obs == 1:
                r = r.unvectorize_state()
            if single_sequence:
                r = r.unvectorize_vars()
            return r

        def auto_flat_states(obs_gaussian):
            if single_sequence:
                return obs_gaussian.unvectorize_vars()
            return obs_gaussian

        if initial_value is None:
            initial_value = np.zeros((n_states, 1))
        initial_value = ensure_matrix(initial_value)
        if len(initial_value.shape) == 1:
            initial_value = initial_value.reshape((n_states, 1))

        if initial_covariance is None:
            initial_covariance = ensure_matrix(
                np.trace(ensure_matrix(self.observation_model)) * (5**2), n_states
            )

        initial_covariance = ensure_matrix(initial_covariance, n_states)
        initial_value = ensure_matrix(initial_value)
        assert initial_value.shape[-2:] == (n_states, 1)
        assert initial_covariance.shape[-2:] == (n_states, n_states)

        if len(initial_value.shape) == 2:
            initial_value = np.vstack([initial_value[np.newaxis, ...]] * n_vars)

        if len(initial_covariance.shape) == 2:
            initial_covariance = np.vstack(
                [initial_covariance[np.newaxis, ...]] * n_vars
            )

        m = initial_value
        P = initial_covariance

        keep_filtered = filtered or smoothed
        if filtered or gains:
            result.filtered = KalmanFilter.Result()

        if log_likelihood:
            result.log_likelihood = np.zeros((n_vars,))
            if likelihoods:
                result.log_likelihoods = np.empty((n_vars, n_measurements))

        if keep_filtered:
            if observations:
                filtered_observations = empty_gaussian(n_states=n_obs)
            filtered_states = empty_gaussian(cov=True)

        if gains:
            result.filtered.gains = np.empty((n_vars, n_measurements, n_states, n_obs))

        for j in range(n_measurements):
            if verbose:
                print("filtering %d/%d" % (j + 1, n_measurements))

            y = data[:, j, ...].reshape((n_vars, n_obs, 1))

            tup = self.update(m, P, y, log_likelihood)
            m, P, K = tup[:3]
            if log_likelihood:
                lx = tup[-1]
                result.log_likelihood += lx
                if likelihoods:
                    result.log_likelihoods[:, j] = lx

            if keep_filtered:
                if observations:
                    obs_mean, obs_cov = self.predict_observation(m, P)
                    filtered_observations.mean[:, j, :] = obs_mean[..., 0]
                    if covariances:
                        filtered_observations.cov[:, j, :, :] = obs_cov

                filtered_states.mean[:, j, :] = m[..., 0]
                filtered_states.cov[:, j, :, :] = P

            if gains:
                result.filtered.gains[:, j, :, :] = K

            m, P = self.predict_next(m, P)

        if smoothed:
            result.smoothed = KalmanFilter.Result()
            if states:
                result.smoothed.states = empty_gaussian()

                # lazy trick to keep last filtered = last smoothed
                result.smoothed.states.mean = filtered_states.mean
                if covariances:
                    result.smoothed.states.cov = filtered_states.cov

            if observations:
                result.smoothed.observations = empty_gaussian(n_states=n_obs)
                result.smoothed.observations.mean = filtered_observations.mean
                if covariances:
                    result.smoothed.observations.cov = filtered_observations.cov

            if gains:
                result.smoothed.gains = np.zeros(
                    (n_vars, n_measurements, n_states, n_states)
                )
                result.pairwise_covariances = np.zeros(
                    (n_vars, n_measurements, n_states, n_states)
                )

            ms = filtered_states.mean[:, -1, :][..., np.newaxis]
            Ps = filtered_states.cov[:, -1, :, :]

            for j in range(n_measurements)[-2::-1]:
                if verbose:
                    print("smoothing %d/%d" % (j + 1, n_measurements))
                m0 = filtered_states.mean[:, j, :][..., np.newaxis]
                P0 = filtered_states.cov[:, j, :, :]

                PsNext = Ps
                ms, Ps, Cs = self.smooth_current(m0, P0, ms, Ps)

                if states:
                    result.smoothed.states.mean[:, j, :] = ms[..., 0]
                    if covariances:
                        result.smoothed.states.cov[:, j, :, :] = Ps

                if observations:
                    obs_mean, obs_cov = self.predict_observation(ms, Ps)
                    result.smoothed.observations.mean[:, j, :] = obs_mean[..., 0]
                    if covariances:
                        result.smoothed.observations.cov[:, j, :, :] = obs_cov

                if gains:
                    result.smoothed.gains[:, j, :, :] = Cs
                    result.pairwise_covariances[:, j, :, :] = ddot_t_right(PsNext, Cs)

        if filtered:
            if states:
                result.filtered.states = Gaussian(filtered_states.mean, None)
                if covariances:
                    result.filtered.states.cov = filtered_states.cov
                result.filtered.states = auto_flat_states(result.filtered.states)
            if observations:
                result.filtered.observations = auto_flat_observations(
                    filtered_observations
                )

        if smoothed:
            if observations:
                result.smoothed.observations = auto_flat_observations(
                    result.smoothed.observations
                )
            if states:
                result.smoothed.states = auto_flat_states(result.smoothed.states)

        if n_test > 0:
            result.predicted = KalmanFilter.Result()
            if observations:
                result.predicted.observations = empty_gaussian(
                    n_measurements=n_test, n_states=n_obs
                )
            if states:
                result.predicted.states = empty_gaussian(n_measurements=n_test)

            for j in range(n_test):
                if verbose:
                    print("predicting %d/%d" % (j + 1, n_test))
                if states:
                    result.predicted.states.mean[:, j, :] = m[..., 0]
                    if covariances:
                        result.predicted.states.cov[:, j, :, :] = P
                if observations:
                    obs_mean, obs_cov = self.predict_observation(m, P)
                    result.predicted.observations.mean[:, j, :] = obs_mean[..., 0]
                    if covariances:
                        result.predicted.observations.cov[:, j, :, :] = obs_cov

                m, P = self.predict_next(m, P)

            if observations:
                result.predicted.observations = auto_flat_observations(
                    result.predicted.observations
                )
            if states:
                result.predicted.states = auto_flat_states(result.predicted.states)

        return result

    def em_process_noise(self, result, verbose=False):
        n_vars, n_measurements, n_states = result.smoothed.states.mean.shape

        res = np.zeros((n_vars, n_states, n_states))

        ms0 = result.smoothed.states.mean[:, 0, :][..., np.newaxis]
        Ps0 = result.smoothed.states.cov[:, 0, ...]
        for j in range(n_measurements):
            if verbose:
                print(
                    "computing ML process noise, step %d/%d" % (j + 1, n_measurements)
                )

            ms1 = result.smoothed.states.mean[:, j, :][..., np.newaxis]
            Ps1 = result.smoothed.states.cov[:, j, ...]

            if j > 0:
                # pylint: disable=E0601
                V1 = result.pairwise_covariances[:, j, ...]
                err = ms1 - ddot(self.state_transition, ms0)
                Vt1tA = ddot_t_right(V1, self.state_transition)
                res += (
                    douter(err, err)
                    + ddot(
                        self.state_transition, ddot_t_right(Ps0, self.state_transition)
                    )
                    + Ps1
                    - Vt1tA
                    - Vt1tA.transpose((0, 2, 1))
                )

            ms0 = ms1
            Ps0 = Ps1

        return (1.0 / (n_measurements - 1)) * res

    def em_observation_noise(self, result, data, verbose=False):
        n_vars, n_measurements, _ = result.smoothed.states.mean.shape
        n_obs = self.observation_model.shape[-2]

        res = np.zeros((n_vars, n_obs, n_obs))
        n_not_nan = np.zeros((n_vars,))

        for j in range(n_measurements):
            if verbose:
                print(
                    "computing ML observation noise, step %d/%d"
                    % (j + 1, n_measurements)
                )

            ms = result.smoothed.states.mean[:, j, :][..., np.newaxis]
            Ps = result.smoothed.states.cov[:, j, ...]

            y = data[:, j, ...].reshape((n_vars, n_obs, 1))
            not_nan = np.ravel(np.all(~np.isnan(y), axis=1))
            n_not_nan += not_nan
            err = y - ddot(self.observation_model, ms)

            r = douter(err, err) + ddot(
                self.observation_model, ddot_t_right(Ps, self.observation_model)
            )
            res[not_nan, ...] += r[not_nan, ...]

        res /= np.maximum(n_not_nan, 1)[:, np.newaxis, np.newaxis]

        return res.reshape((n_vars, n_obs, n_obs))

    def em(
        self, data, n_iter=5, initial_value=None, initial_covariance=None, verbose=False
    ):
        if n_iter <= 0:
            return self

        data = ensure_matrix(data)
        if len(data.shape) == 1:
            data = data[np.newaxis, :]

        n_vars = data.shape[0]
        n_states = self.state_transition.shape[-2]  # Allow different transitions

        if initial_value is None:
            initial_value = np.zeros((n_vars, n_states, 1))

        if verbose:
            print("--- EM algorithm %d iteration(s) to go" % n_iter)
            print(" * E step")

        e_step = self.compute(
            data,
            n_test=0,
            initial_value=initial_value,
            initial_covariance=initial_covariance,
            smoothed=True,
            filtered=False,
            states=True,
            observations=True,
            covariances=True,
            likelihoods=False,
            gains=True,
            log_likelihood=False,
            verbose=verbose,
        )

        if verbose:
            print(" * M step")

        process_noise = self.em_process_noise(e_step, verbose=verbose)
        observation_noise = self.em_observation_noise(e_step, data, verbose=verbose)
        initial_value, initial_covariance = em_initial_state(e_step, initial_value)

        new_model = KalmanFilter(
            self.state_transition,
            process_noise,
            self.observation_model,
            observation_noise,
        )

        return new_model.em(
            data, n_iter - 1, initial_value, initial_covariance, verbose
        )


def em_initial_state(result, initial_means):
    x0 = result.smoothed.states.mean[:, 0, :][..., np.newaxis]
    P0 = result.smoothed.states.cov[:, 0, ...]
    x0_x0 = P0 + douter(x0, x0)

    m = x0
    P = (
        x0_x0
        - douter(initial_means, x0)
        - douter(x0, initial_means)
        + douter(initial_means, initial_means)
    )

    return m, P


def ddot(A, B):
    "Matrix multiplication over last two axes"
    return np.matmul(A, B)


def ddot_t_right_old(A, B):
    "Matrix multiplication over last 2 axes with right operand transposed"
    return np.einsum("...ij,...kj->...ik", A, B)


def ddot_t_right(A, B):
    "Matrix multiplication over last 2 axes with right operand transposed"
    # see previous for original code
    # return np.matmul(A, np.transpose(B, axes=(-1, -2)))
    #  (A @ B.swapaxes(-1, -2)).swapaxes(-1, -2)
    return np.matmul(A, np.swapaxes(B, -1, -2))


def douter(a, b):
    "Outer product, last two axes"
    return a * b.transpose((0, 2, 1))


def stable_pinv(A, tol=1e-5, regularization=1e-4):
    n = A.shape[1]
    U, s, Vt = np.linalg.svd((A + regularization * np.eye(n)), full_matrices=False)
    s_inv = np.where(s > tol, 1 / s, 0)
    return Vt.T @ np.diag(s_inv) @ U.T


def dinv(A):
    "Matrix inverse applied to last two axes"
    try:
        res = np.linalg.inv(
            np.nan_to_num(A)
        )  # can cause kernel death in OpenBLAS with NaN
    except Exception:
        try:
            res = np.linalg.pinv(A)  # slower but more robust
        except np.linalg.LinAlgError:
            print("SVD did not converge, attempting more robust approach...")
            res = stable_pinv(A)
    return res


def autoshape(func):
    "Automatically shape arguments and return values"

    def to_3d_array(v):
        if len(v.shape) == 1:
            return v[np.newaxis, :, np.newaxis]
        elif len(v.shape) == 2:
            return v[np.newaxis, ...]
        else:
            return v

    @wraps(func)
    def reshaped_func(*args, **kwargs):
        any_tensor = any([len(x.shape) > 2 for x in args])
        outputs = func(*[to_3d_array(a) for a in args], **kwargs)
        if not any_tensor:
            outputs = [mat[0, ...] for mat in outputs]
        return outputs

    return reshaped_func


@autoshape
def predict(mean, covariance, state_transition, process_noise):
    """
    Kalman filter prediction step

    :param mean: :math:`{\\mathbb E}[x_{j-1}]`,
        the filtered mean form the previous step
    :param covariance: :math:`{\\rm Cov}[x_{j-1}]`,
        the filtered covariance form the previous step
    :param state_transition: matrix :math:`A`
    :param process_noise: matrix :math:`Q`

    :rtype: ``(prior_mean, prior_cov)`` predicted mean and covariance
        :math:`{\\mathbb E}[x_j]`, :math:`{\\rm Cov}[x_j]`
    """

    n = mean.shape[1]

    assert covariance.shape[-2:] == (n, n)
    assert covariance.shape[-2:] == (n, n)
    assert process_noise.shape[-2:] == (n, n)
    assert state_transition.shape[-2:] == (n, n)

    # mp = A * m
    prior_mean = ddot(state_transition, mean)
    # Pp = A * P * A.t + Q
    prior_cov = (
        ddot(state_transition, ddot_t_right(covariance, state_transition))
        + process_noise
    )

    return prior_mean, prior_cov


@autoshape
def _update(
    prior_mean,
    prior_covariance,
    observation_model,
    observation_noise,
    measurement,
    log_likelihood=False,
):
    n = prior_mean.shape[1]
    m = observation_model.shape[1]

    assert measurement.shape[-2:] == (m, 1)
    assert prior_covariance.shape[-2:] == (n, n)
    assert observation_model.shape[-2:] == (m, n)
    assert observation_noise.shape[-2:] == (m, m)

    # y - H * mp
    v = measurement - ddot(observation_model, prior_mean)

    # H * Pp * H.t + R
    S = (
        ddot(observation_model, ddot_t_right(prior_covariance, observation_model))
        + observation_noise
    )
    invS = dinv(S)

    # Kalman gain: Pp * H.t * invS
    K = ddot(ddot_t_right(prior_covariance, observation_model), invS)

    # K * v + mp
    posterior_mean = ddot(K, v) + prior_mean

    # Pp - K * H * Pp
    posterior_covariance = prior_covariance - ddot(
        K, ddot(observation_model, prior_covariance)
    )

    # inv-chi2 test var
    # outlier_test = np.sum(v * ddot(invS, v), axis=0)
    if log_likelihood:
        lx = np.ravel(ddot(v.transpose((0, 2, 1)), ddot(invS, v)))
        lx += np.log(np.linalg.det(S))
        lx *= -0.5
        return posterior_mean, posterior_covariance, K, lx

    return posterior_mean, posterior_covariance, K


def update(
    prior_mean, prior_covariance, observation_model, observation_noise, measurement
):
    """
    Kalman filter update step

    :param prior_mean: :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_{j-1}]`,
        the prior mean of :math:`x_j`
    :param prior_covariance: :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_{j-1}]`,
        the prior covariance of :math:`x_j`
    :param observation_model: matrix :math:`H`
    :param observation_noise: matrix :math:`R`
    :param measurement: observation :math:`y_j`

    :rtype: ``(posterior_mean, posterior_covariance)``
        posterior mean and covariance
        :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_j]`,
        :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_j]`
        after observing :math:`y_j`
    """
    return _update(
        prior_mean, prior_covariance, observation_model, observation_noise, measurement
    )[:2]


@autoshape
def priv_smooth(
    posterior_mean,
    posterior_covariance,
    state_transition,
    process_noise,
    next_smooth_mean,
    next_smooth_covariance,
):
    n = posterior_mean.shape[1]

    assert posterior_covariance.shape[-2:] == (n, n)
    assert process_noise.shape[-2:] == (n, n)
    assert state_transition.shape[-2:] == (n, n)

    assert next_smooth_mean.shape == posterior_mean.shape
    assert next_smooth_covariance.shape == posterior_covariance.shape

    # re-predict a priori estimates for the next state
    # A * m
    mp = ddot(state_transition, posterior_mean)
    # A * P * A.t + Q
    Pp = (
        ddot(state_transition, ddot_t_right(posterior_covariance, state_transition))
        + process_noise
    )

    # Kalman smoothing gain: P * A.t * inv(Pp)
    C = ddot(ddot_t_right(posterior_covariance, state_transition), dinv(Pp))

    # m + C * (ms - mp)
    smooth_mean = posterior_mean + ddot(C, next_smooth_mean - mp)
    # P + C * (Ps - Pp) * C.t
    smooth_covariance = posterior_covariance + ddot(
        C, ddot_t_right(next_smooth_covariance - Pp, C)
    )

    return smooth_mean, smooth_covariance, C


def smooth(
    posterior_mean,
    posterior_covariance,
    state_transition,
    process_noise,
    next_smooth_mean,
    next_smooth_covariance,
):
    """
    Kalman smoother backwards step

    :param posterior_mean: :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_j]`,
        the filtered mean of :math:`x_j`
    :param posterior_covariance: :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_j]`,
        the filtered covariance of :math:`x_j`
    :param state_transition: matrix :math:`A`
    :param process_noise: matrix :math:`Q`
    :param next_smooth_mean:
        :math:`{\\mathbb E}[x_{j+1}|y_1,\\ldots,y_T]`
    :param next_smooth_covariance:
        :math:`{\\rm Cov}[x_{j+1}|y_1,\\ldots,y_T]`

    :rtype: ``(smooth_mean, smooth_covariance, smoothing_gain)``
        smoothed mean :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_T]`,
        and covariance :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_T]`
    """
    return priv_smooth(
        posterior_mean,
        posterior_covariance,
        state_transition,
        process_noise,
        next_smooth_mean,
        next_smooth_covariance,
    )[:2]


@autoshape
def predict_observation(mean, covariance, observation_model, observation_noise):
    """
    Compute probability distribution of the observation :math:`y`, given
    the distribution of :math:`x`.

    :param mean: :math:`{\\mathbb E}[x]`
    :param covariance: :math:`{\\rm Cov}[x]`
    :param observation_model: matrix :math:`H`
    :param observation_noise: matrix :math:`R`

    :rtype: mean :math:`{\\mathbb E}[y]` and covariance :math:`{\\rm Cov}[y]`
    """

    n = mean.shape[1]
    m = observation_model.shape[1]
    assert observation_model.shape[-2:] == (m, n)
    assert covariance.shape[-2:] == (n, n)
    assert observation_model.shape[-2:] == (m, n)

    # H * m
    obs_mean = ddot(observation_model, mean)

    # H * P * H^T + R
    obs_cov = (
        ddot(observation_model, ddot_t_right(covariance, observation_model))
        + observation_noise
    )

    return obs_mean, obs_cov


@autoshape
def priv_update_with_nan_check(
    prior_mean,
    prior_covariance,
    observation_model,
    observation_noise,
    measurement,
    log_likelihood=False,
):
    tup = _update(
        prior_mean,
        prior_covariance,
        observation_model,
        observation_noise,
        measurement,
        log_likelihood=log_likelihood,
    )

    m1, P1, K = tup[:3]

    is_nan = np.ravel(np.any(np.isnan(m1), axis=1))

    m1[is_nan, ...] = prior_mean[is_nan, ...]
    P1[is_nan, ...] = prior_covariance[is_nan, ...]
    K[is_nan, ...] = 0

    if log_likelihood:
        lx = tup[-1]
        lx[is_nan] = 0
        return m1, P1, K, lx
    else:
        return m1, P1, K


def update_with_nan_check(
    prior_mean, prior_covariance, observation_model, observation_noise, measurement
):
    """
    Kalman filter update with a check for NaN observations. Like ``update`` but
    returns ``(prior_mean, prior_covariance)`` if ``measurement`` is NaN
    """

    return priv_update_with_nan_check(
        prior_mean, prior_covariance, observation_model, observation_noise, measurement
    )[:2]


def ensure_matrix(x, dim=1):
    # pylint: disable=W0702,W0104,E1136
    try:
        y = np.array(x)
        y.shape[0]  # for reasons I don't understand, this line is critical
        x = y
    except Exception:
        x = np.eye(dim) * x
    return x


"""
n_seasons = 7
state_transition = np.zeros((n_seasons+1, n_seasons+1))
state_transition[0,0] = 1
state_transition[1,1:-1] = [-1.0] * (n_seasons-1)
state_transition[2:,1:-1] = np.eye(n_seasons-1)
level_noise = 0.05
observation_noise = 0.2
season_noise = 1e-3

process_noise_cov = np.diag([level_noise, season_noise] + [0]*(n_seasons-1))**2
observation_noise_cov = observation_noise**2

kf = KalmanFilter(
    state_transition,
    process_noise_cov,
    [[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0,0,0,0,0]],
    observation_noise_cov)
data = np.concatenate((
    np.expand_dims(temp, axis=2), 
    np.expand_dims(np.repeat(temp[-1:, :], temp.shape[0], axis=0), axis=2)
), axis=2)

result = kf.predict(data, 12)
res = pd.DataFrame(result.observations.mean.T[0])
res.iloc[:, -1].plot()
"""
