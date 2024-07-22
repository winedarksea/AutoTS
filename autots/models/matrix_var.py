# -*- coding: utf-8 -*-
"""VAR models based on matrix factorization and related methods.

Heavily borrowing on the work of Xinyu Chen
See https://github.com/xinychen/transdim and corresponding Medium articles

Thrown around a lot of np.nan_to_num before pinv to prevent the following crash:
On entry to DLASCL parameter number  4 had an illegal value

"""
import datetime
import random
import numpy as np
import pandas as pd
from autots.tools.shaping import wide_to_3d
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.tools.seasonal import seasonal_int


def rrvar(data, R, pred_step, maxiter=100):
    """Reduced-rank VAR algorithm using ALS."""

    N, T = data.shape
    X1 = data[:, :-1]
    X2 = data[:, 1:]
    V = np.random.randn(R, N)
    X1_pinv = np.linalg.pinv(np.nan_to_num(X1))
    for it in range(maxiter):
        W = X2 @ np.linalg.pinv((V @ X1))
        V = np.linalg.pinv(W) @ X2 @ X1_pinv
    mat = np.append(data, np.zeros((N, pred_step)), axis=1)
    for s in range(pred_step):
        mat[:, T + s] = W @ V @ mat[:, T + s - 1]
    return mat[:, -pred_step:]


def var(X, pred_step):
    """Simple VAR."""
    N, T = X.shape
    temp1 = np.zeros((N, N))
    temp2 = np.zeros((N, N))
    for t in range(1, T):
        temp1 += np.outer(X[:, t], X[:, t - 1])
        temp2 += np.outer(X[:, t - 1], X[:, t - 1])
    A = temp1 @ np.linalg.pinv((temp2))
    mat = np.append(X, np.zeros((N, pred_step)), axis=1)
    for s in range(pred_step):
        mat[:, T + s] = A @ mat[:, T + s - 1]
    return mat[:, -pred_step:]


def dmd(data, r):
    """Dynamic Mode Decomposition (DMD) algorithm."""

    # Build data matrices
    X1 = data[:, :-1]
    X2 = data[:, 1:]
    # Perform singular value decomposition on X1
    u, s, v = np.linalg.svd(X1, full_matrices=False)
    # Compute the Koopman matrix
    A_tilde = u[:, :r].conj().T @ X2 @ v[:r, :].conj().T * np.reciprocal(s[:r])
    # Perform eigenvalue decomposition on A_tilde
    Phi, Q = np.linalg.eig(A_tilde)
    # Compute the coefficient matrix
    Psi = X2 @ v[:r, :].conj().T @ np.diag(np.reciprocal(s[:r])) @ Q
    A = Psi @ np.diag(Phi) @ np.linalg.pinv(np.nan_to_num(Psi))

    return A_tilde, Phi, A


def dmd4cast(data, r, pred_step):
    N, T = data.shape
    _, _, A = dmd(data, r)
    mat = np.append(data, np.zeros((N, pred_step)), axis=1)
    for s in range(pred_step):
        mat[:, T + s] = (A @ mat[:, T + s - 1]).real
    return mat[:, -pred_step:]


def mar(X, pred_step, family="gaussian", maxiter=100):
    m, n, T = X.shape
    family = str(family).lower()
    if family == "poisson":
        B = np.random.poisson(size=(n, n))
    elif family == "gamma":
        B = np.random.standard_gamma(2, size=(n, n))
    elif family == "negativebinomial":
        B = np.random.negative_binomial(1, 0.5, size=(n, n))
    elif family == "chi2":
        B = np.random.chisquare(1, size=(n, n))
    elif family == "uniform":
        B = np.random.uniform(size=(n, n))
    else:  # 'Gaussian'
        B = np.random.randn(n, n)

    for it in range(maxiter):
        temp0 = B.T @ B
        temp1 = np.zeros((m, m))
        temp2 = np.zeros((m, m))
        for t in range(1, T):
            temp1 += X[:, :, t] @ B @ X[:, :, t - 1].T
            temp2 += X[:, :, t - 1] @ temp0 @ X[:, :, t - 1].T
        A = temp1 @ np.linalg.pinv(np.nan_to_num(temp2))
        temp0 = A.T @ A
        temp1 = np.zeros((n, n))
        temp2 = np.zeros((n, n))
        for t in range(1, T):
            temp1 += X[:, :, t].T @ A @ X[:, :, t - 1]
            temp2 += X[:, :, t - 1].T @ temp0 @ X[:, :, t - 1]
        B = temp1 @ np.linalg.pinv(np.nan_to_num(temp2))
    tensor = np.append(X, np.zeros((m, n, pred_step)), axis=2)
    for s in range(pred_step):
        tensor[:, :, T + s] = A @ tensor[:, :, T + s - 1] @ B.T
    return tensor[:, :, -pred_step:]


class RRVAR(ModelObject):
    """Reduced Rank VAR models based on the code of Xinyu Chen.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): type of regression (None, 'User', or 'Holiday')
        n_jobs (int): passed to joblib for multiprocessing. Set to none for context manager.

    """

    def __init__(
        self,
        name: str = "RRVAR",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        method: str = "als",
        rank: float = 0.1,
        maxiter: int = 200,
        holiday_country: str = 'US',
        random_seed: int = 2022,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.method = method
        self.rank = rank
        self.maxiter = maxiter

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied .

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """

        df = self.basic_profile(df)
        self.regressor_train = None
        self.verbose_bool = False
        if self.verbose > 1:
            self.verbose_bool = True

        if self.rank < 1 and self.rank > 0:
            self.rank = int(self.rank * df.shape[1])
            self.rank = self.rank if self.rank > 0 else 1
        if self.rank > df.shape[1]:
            pass

        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        data = self.df_train.to_numpy().T
        if self.rank == 0:
            forecast = var(data, forecast_length).T
        elif self.method == "als":
            forecast = rrvar(data, self.rank, forecast_length).T
        elif self.method == "dmd":
            if np.isnan(np.sum(data)):
                raise ValueError("DMD method does not allow NaN")
            forecast = dmd4cast(data, self.rank, forecast_length).T

        forecast = pd.DataFrame(forecast, index=test_index, columns=self.column_names)
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=test_index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        return {
            'method': random.choices(['als', 'dmd'], [0.7, 0.3])[0],
            'rank': random.choice([2, 4, 8, 16, 32, 0.1, 0.2, 0.5]),
            'maxiter': 200,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'method': self.method,
            'rank': self.rank,
            'maxiter': self.maxiter,
        }


class MAR(ModelObject):
    """Matrix Autoregressive model based on the code of Xinyu Chen.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): type of regression (None, 'User', or 'Holiday')
        n_jobs (int): passed to joblib for multiprocessing. Set to none for context manager.

    """

    def __init__(
        self,
        name: str = "MAR",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        seasonality: float = 7,
        family: str = "gaussian",
        maxiter: int = 200,
        holiday_country: str = 'US',
        random_seed: int = 2022,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.seasonality = seasonality
        self.family = family
        self.maxiter = maxiter

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied .

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """

        df = self.basic_profile(df)
        self.regressor_train = None

        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        shifted = wide_to_3d(self.df_train.to_numpy())
        pred_steps = int(np.ceil(forecast_length / self.seasonality) + 1)
        forecast = np.hstack(
            mar(shifted, pred_steps, family=self.family, maxiter=self.maxiter).T
        ).T[:forecast_length]

        forecast = pd.DataFrame(forecast, index=test_index, columns=self.column_names)
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=test_index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        return {
            'seasonality': seasonal_int(include_one=False, very_small=True),
            'family': random.choices(
                ['gaussian', 'poisson', 'negativebinomial', 'gamma', 'chi2', 'uniform'],
                [0.6, 0.05, 0.02, 0.1, 0.05, 0.05],
            )[0],
            'maxiter': 200,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'seasonality': self.seasonality,
            'family': self.family,
            'maxiter': self.maxiter,
        }


def update_cg(var, r, q, Aq, rold):
    alpha = rold / np.inner(q, Aq)
    var = var + alpha * q
    r = r - alpha * Aq
    rnew = np.inner(r, r)
    q = r + (rnew / rold) * q
    return var, r, q, rnew


def ell_w(ind, W, X, rho):
    return X @ ((W.T @ X) * ind).T + rho * W


def conj_grad_w(sparse_mat, ind, W, X, rho, maxiter=5):
    rank, dim1 = W.shape
    w = np.reshape(W, -1, order="F")
    r = np.reshape(X @ sparse_mat.T - ell_w(ind, W, X, rho), -1, order="F")
    q = r.copy()
    rold = np.inner(r, r)
    for it in range(maxiter):
        Q = np.reshape(q, (rank, dim1), order="F")
        Aq = np.reshape(ell_w(ind, Q, X, rho), -1, order="F")
        w, r, q, rold = update_cg(w, r, q, Aq, rold)
    return np.reshape(w, (rank, dim1), order="F")


def ell_x(ind, W, X, A, Psi, d, lambda0, rho):
    rank, dim2 = X.shape
    temp = np.zeros((d * rank, Psi[0].shape[0]))
    for k in range(1, d + 1):
        temp[(k - 1) * rank : k * rank, :] = X @ Psi[k].T
    temp1 = X @ Psi[0].T - A @ temp
    temp2 = np.zeros((rank, dim2))
    for k in range(d):
        temp2 += A[:, k * rank : (k + 1) * rank].T @ temp1 @ Psi[k + 1]
    return W @ ((W.T @ X) * ind) + rho * X + lambda0 * (temp1 @ Psi[0] - temp2)


def conj_grad_x(sparse_mat, ind, W, X, A, Psi, d, lambda0, rho, maxiter=5):
    rank, dim2 = X.shape
    x = np.reshape(X, -1, order="F")
    r = np.reshape(
        W @ sparse_mat - ell_x(ind, W, X, A, Psi, d, lambda0, rho), -1, order="F"
    )
    q = r.copy()
    rold = np.inner(r, r)
    for it in range(maxiter):
        Q = np.reshape(q, (rank, dim2), order="F")
        Aq = np.reshape(ell_x(ind, W, Q, A, Psi, d, lambda0, rho), -1, order="F")
        x, r, q, rold = update_cg(x, r, q, Aq, rold)
    return np.reshape(x, (rank, dim2), order="F")


def generate_Psi(T, d):
    Psi = []
    for k in range(0, d + 1):
        if k == 0:
            Psi.append(np.append(np.zeros((T - d, d)), np.eye(T - d), axis=1))
        else:
            Psi.append(
                np.append(
                    np.append(np.zeros((T - d, d - k)), np.eye(T - d), axis=1),
                    np.zeros((T - d, k)),
                    axis=1,
                )
            )
    return Psi


def tmf(sparse_mat, rank, d, lambda0, rho, maxiter=50, inner_maxiter=10):
    dim1, dim2 = sparse_mat.shape
    # prevent failure of constant matrix
    if np.all(sparse_mat == sparse_mat[0, 0]):
        raise ValueError("TMF fails on constant arrays")
    ind = sparse_mat != 0
    W = 0.01 * np.random.randn(rank, dim1)
    X = 0.01 * np.random.randn(rank, dim2)
    A = 0.01 * np.random.randn(rank, d * rank)
    Psi = generate_Psi(dim2, d)
    temp = np.zeros((d * rank, dim2 - d))
    for it in range(maxiter):
        W = conj_grad_w(sparse_mat, ind, W, X, rho, inner_maxiter)
        X = conj_grad_x(sparse_mat, ind, W, X, A, Psi, d, lambda0, rho, inner_maxiter)
        for k in range(1, d + 1):
            temp[(k - 1) * rank : k * rank, :] = X @ Psi[k].T
        A = X @ Psi[0].T @ np.linalg.pinv((temp))
        mat_hat = W.T @ X
    return mat_hat, W, X, A


def var4cast(X, A, d, delta):
    dim1, dim2 = X.shape
    X_hat = np.append(X, np.zeros((dim1, delta)), axis=1)
    for t in range(delta):
        X_hat[:, dim2 + t] = A @ X_hat[:, dim2 + t - np.arange(1, d + 1)].T.reshape(
            dim1 * d
        )
    return X_hat[:, -delta:]


class TMF(ModelObject):
    """Temporal Matrix Factorization VAR model based on the code of Xinyu Chen.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): type of regression (None, 'User', or 'Holiday')
        n_jobs (int): passed to joblib for multiprocessing. Set to none for context manager.

    """

    def __init__(
        self,
        name: str = "TMF",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        d: int = 1,
        lambda0: float = 1,
        rho: float = 1,
        rank: float = 0.4,
        maxiter: int = 100,
        inner_maxiter: int = 10,
        holiday_country: str = 'US',
        random_seed: int = 2022,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.d = d
        self.lambda0 = lambda0
        self.rho = rho
        self.rank = rank
        self.maxiter = maxiter
        self.inner_maxiter = inner_maxiter

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied .

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """

        df = self.basic_profile(df)
        self.regressor_train = None

        if self.rank < 1 and self.rank > 0:
            self.rank = int(self.rank * df.shape[1])
            self.rank = self.rank if self.rank > 0 else 1
        if self.rank > df.shape[1]:
            pass

        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        _, W, X, A = tmf(
            np.nan_to_num(self.df_train.to_numpy().T),
            self.rank,
            self.d,
            self.lambda0,
            self.rho,
            self.maxiter,
            self.inner_maxiter,
        )
        forecast = (W.T @ var4cast(X, A, self.d, forecast_length)).T

        forecast = pd.DataFrame(forecast, index=test_index, columns=self.column_names)
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=test_index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        rho = random.choices([1, 1e-7, 1e-6, 1e-5, 5e-4], [0.5, 0.1, 0.1, 0.1, 0.1])[0]
        return {
            "d": random.choice([1, 2]),
            "lambda0": random.choice([1, 0, 0.1 * rho, 0.5 * rho, rho, 10 * rho]),
            "rho": rho,
            'rank': random.choice([2, 4, 0.1, 0.2, 0.5]),
            'maxiter': 100,
            'inner_maxiter': 10,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            "d": self.d,
            "lambda0": self.lambda0,
            "rho": self.rho,
            'rank': self.rank,
            'maxiter': self.maxiter,
            'inner_maxiter': self.inner_maxiter,
        }


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order="F")


def mat2ten(mat, dim, mode):
    index = list()
    index.append(mode)
    for i in range(dim.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(dim[index]), order="F"), 0, mode)


def svt_tnn(mat, tau, theta):
    [m, n] = mat.shape
    if 2 * m < n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices=0)
        s = np.sqrt(s)
        idx = np.sum(s > tau)
        mid = np.zeros(idx)
        mid[:theta] = 1
        mid[theta:idx] = (s[theta:idx] - tau) / s[theta:idx]
        return (u[:, :idx] @ np.diag(mid)) @ (u[:, :idx].T @ mat)
    elif m > 2 * n:
        return svt_tnn(mat.T, tau, theta).T
    u, s, v = np.linalg.svd(mat, full_matrices=0)
    idx = np.sum(s > tau)
    vec = s[:idx].copy()
    vec[theta:idx] = s[theta:idx] - tau
    return u[:, :idx] @ np.diag(vec) @ v[:idx, :]


def latc_imputer(
    sparse_tensor,
    time_lags,
    alpha,
    rho0,
    lambda0,
    theta,
    epsilon,
    maxiter,
):
    """Low-Rank Autoregressive Tensor Completion, LATC-imputer.
    Recognizes 0 as NaN.
    """
    dim = np.array(sparse_tensor.shape)
    dim_time = int(np.prod(dim) / dim[0])
    d = len(time_lags)
    max_lag = np.max(time_lags)
    sparse_mat = ten2mat(sparse_tensor, 0)
    pos_missing = np.where(sparse_mat == 0)

    X = np.zeros(np.insert(dim, 0, len(dim)))
    T = np.zeros(np.insert(dim, 0, len(dim)))
    Z = sparse_mat.copy()
    Z[pos_missing] = np.mean(sparse_mat[sparse_mat != 0])
    A = 0.001 * np.random.rand(dim[0], d)
    it = 0
    ind = np.zeros((d, dim_time - max_lag), dtype=int)
    for i in range(d):
        ind[i, :] = np.arange(max_lag - time_lags[i], dim_time - time_lags[i])
    last_mat = sparse_mat.copy()
    snorm = np.linalg.norm(sparse_mat, "fro")
    rho = rho0
    while True:
        rho = min(rho * 1.05, 1e5)
        for k in range(len(dim)):
            X[k] = mat2ten(
                svt_tnn(
                    ten2mat(mat2ten(Z, dim, 0) - T[k] / rho, k), alpha[k] / rho, theta
                ),
                dim,
                k,
            )
        tensor_hat = np.einsum("k, kmnt -> mnt", alpha, X)
        mat_hat = ten2mat(tensor_hat, 0)
        mat0 = np.zeros((dim[0], dim_time - max_lag))
        if lambda0 > 0:
            for m in range(dim[0]):
                Qm = mat_hat[m, ind].T
                A[m, :] = np.linalg.pinv(Qm) @ Z[m, max_lag:]
                mat0[m, :] = Qm @ A[m, :]
            mat1 = ten2mat(np.mean(rho * X + T, axis=0), 0)
            Z[pos_missing] = np.append(
                (mat1[:, :max_lag] / rho),
                (mat1[:, max_lag:] + lambda0 * mat0) / (rho + lambda0),
                axis=1,
            )[pos_missing]
        else:
            Z[pos_missing] = (ten2mat(np.mean(X + T / rho, axis=0), 0))[pos_missing]
        T = T + rho * (
            X - np.broadcast_to(mat2ten(Z, dim, 0), np.insert(dim, 0, len(dim)))
        )
        tol = np.linalg.norm((mat_hat - last_mat), "fro") / snorm
        last_mat = mat_hat.copy()
        it += 1
        if it % 100 == 0:
            pass
            # print("Iter: {}".format(it))
            # print("Tolerance: {:.6}".format(tol))
        if (tol < epsilon) or (it >= maxiter):
            break

    # print("Total iteration: {}".format(it))
    # print("Tolerance: {:.6}".format(tol))

    return tensor_hat


def latc_predictor(
    sparse_mat,
    pred_time_steps,
    time_horizon,
    time_intervals,
    time_lags,
    alpha,
    rho,
    lambda0,
    theta,
    window,
    epsilon,
    maxiter,
):
    """LATC-predictor kernel."""
    num_series = sparse_mat.shape[0]

    pred_cycles = np.ceil(pred_time_steps / time_horizon)
    mat_hat = []
    pred_cycles = int(pred_cycles)
    for t in range(pred_cycles):
        if window is not None:
            temp2 = np.concatenate(
                [sparse_mat[:, -window:], np.zeros((num_series, time_horizon))], axis=1
            )
        else:
            temp2 = np.concatenate(
                [sparse_mat, np.zeros((num_series, time_horizon))], axis=1
            )
        cuts = int(temp2.shape[1] / (time_intervals))
        start_2 = temp2.shape[1] % time_intervals
        dim = np.array([num_series, time_intervals, cuts])
        temp2 = mat2ten(temp2[:, start_2:], dim, 0)
        if (temp2 == 0)[:, 1:].all():
            raise ValueError("LATC cannot accept any arrays that are all 0")
        # if np.all(temp2 == temp2[0, 0, 0])
        tensor = latc_imputer(
            temp2,
            time_lags,
            alpha,
            rho,
            lambda0,
            theta,
            epsilon,
            maxiter,
        )
        res = (ten2mat(tensor, 0))[:, -time_horizon:]
        sparse_mat = np.concatenate([sparse_mat, res], axis=1)
        mat_hat.append(res)
    return np.concatenate(mat_hat, axis=1)[:, :pred_time_steps]


class LATC(ModelObject):
    """Low Rank Autoregressive Tensor Completion.
    Based on https://arxiv.org/abs/2104.14936
    and https://github.com/xinychen/tensor-learning/blob/master/mats/LATC-predictor.ipynb
    rho: learning rate
    lambda: weight parameter

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): type of regression (None, 'User', or 'Holiday')
        n_jobs (int): passed to joblib for multiprocessing. Set to none for context manager.

    """

    def __init__(
        self,
        name: str = "LATC",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        time_horizon: float = 1,
        seasonality: int = 7,
        time_lags: list = [1],
        lambda0: float = 1,
        learning_rate: float = 1,
        theta: float = 1,
        window: int = 30,
        epsilon: float = 1e-4,
        alpha: list = [0.33333333, 0.33333333, 0.33333333],
        maxiter: int = 100,
        holiday_country: str = 'US',
        random_seed: int = 2022,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.time_horizon = time_horizon
        self.seasonality = seasonality
        self.time_lags = time_lags
        self.lambda0 = lambda0
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.theta = theta
        self.window = window
        self.alpha = alpha
        self.maxiter = maxiter

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied .

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """

        df = self.basic_profile(df)
        self.regressor_train = None

        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        if self.time_horizon < 1 and self.time_horizon > 0:
            self.time_horizon = int(np.ceil(self.time_horizon * forecast_length))
        data = self.df_train.fillna(0).to_numpy().T
        mat_hat = latc_predictor(
            sparse_mat=data,
            pred_time_steps=forecast_length,  # forecast_length
            time_horizon=self.time_horizon,  # must be % of pred_time_steps
            time_intervals=self.seasonality,  # seasonality
            time_lags=self.time_lags,
            alpha=np.array(self.alpha),
            rho=self.learning_rate,
            lambda0=self.lambda0,
            theta=self.theta,
            window=self.window,
            epsilon=self.epsilon,
            maxiter=self.maxiter,
        )
        forecast = mat_hat.T

        forecast = pd.DataFrame(forecast, index=test_index, columns=self.column_names)
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=test_index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        learning_rate = random.choices(
            [1, 1e-7, 1e-6, 1e-5, 5e-4, 1e-4], [0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
        )[0]
        lags = random.choice([1, 2])
        time_lags = sorted(
            [
                seasonal_int(include_one=True, very_small=True),
                seasonal_int(include_one=True, very_small=True),
            ]
        )
        if lags == 1:
            time_lags = time_lags[:1]
        return {
            "time_horizon": random.choice([1, 2, 0.25, 0.5]),
            'seasonality': seasonal_int(include_one=True, very_small=True),
            'time_lags': time_lags,
            "lambda0": random.choice(
                [
                    1,
                    0,
                    0.1 * learning_rate,
                    0.5 * learning_rate,
                    learning_rate,
                    10 * learning_rate,
                ]
            ),
            "learning_rate": learning_rate,
            'theta': random.choice([1, 2, 4]),
            'window': random.choice([None, 14, 30, 90]),
            'epsilon': 1e-4,
            'alpha': [0.33333333, 0.33333333, 0.33333333],
            'maxiter': random.choice([25, 50, 100, 150]),
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            "time_horizon": self.time_horizon,
            'seasonality': self.seasonality,
            'time_lags': self.time_lags,
            "lambda0": self.lambda0,
            "learning_rate": self.learning_rate,
            'theta': self.theta,
            'window': self.window,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'maxiter': self.maxiter,
        }


def _DMD(
    data,
    r,
    alpha=0.0,
    amplitude_threshold=None,
    eigenvalue_threshold=None,
    ecr_threshold=0.95,
):
    X1 = data[:, :-1]
    X2 = data[:, 1:]
    u, s, v = np.linalg.svd(X1, full_matrices=False)
    if r in ['ecr', 'auto']:
        total_energy = np.sum(s**2)
        # Calculate captured energy for each singular value
        captured_energy = np.cumsum(s**2) / total_energy
        r = np.searchsorted(captured_energy, ecr_threshold)
        print(f"ECR rank is {r}")
    elif r > 0 and r < 1:
        r = int(data.shape[0] * r)
        # print(f"Rational rank is {r}")

    regularized_s = s[:r] + alpha
    A_tilde = u[:, :r].conj().T @ X2 @ v[:r, :].conj().T * np.reciprocal(regularized_s)
    Phi, Q = np.linalg.eig(A_tilde)

    if amplitude_threshold is not None:
        # Calculate mode amplitudes
        b = np.linalg.pinv(Q) @ u[:, :r].conj().T @ X1[:, 0]
        amplitudes = np.abs(b)
        amp_filter = amplitudes > amplitude_threshold
    else:
        amp_filter = np.ones_like(Phi, dtype=bool)

    if eigenvalue_threshold is not None:
        # Calculate eigenvalue magnitudes
        eigenvalue_magnitudes = np.abs(Phi)
        eigen_filter = eigenvalue_magnitudes <= eigenvalue_threshold
    else:
        eigen_filter = np.ones_like(Phi, dtype=bool)

    # Filter modes based on amplitudes and eigenvalue magnitudes
    filter_mask = amp_filter & eigen_filter
    Phi = Phi[filter_mask]
    Q = Q[:, filter_mask]

    # Reconstruct dynamics with filtered modes
    Psi = X2 @ v[:r, :].conj().T @ np.diag(np.reciprocal(regularized_s)) @ Q
    A = Psi @ np.diag(Phi) @ np.linalg.pinv(Psi)
    return A_tilde, Phi, A


def dmd_forecast(
    data, r, pred_step, alpha=0.0, amplitude_threshold=None, eigenvalue_threshold=None
):
    N, T = data.shape
    _, _, A = _DMD(
        data,
        r,
        alpha,
        amplitude_threshold=amplitude_threshold,
        eigenvalue_threshold=eigenvalue_threshold,
    )
    mat = np.append(data, np.zeros((N, pred_step)), axis=1)
    for s in range(pred_step):
        mat[:, T + s] = (A @ mat[:, T + s - 1]).real
    return mat[:, -pred_step:]


class DMD(ModelObject):
    """Dynamic Mode Decomposition

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): type of regression (None, 'User', or 'Holiday')
        n_jobs (int): passed to joblib for multiprocessing. Set to none for context manager.

    """

    def __init__(
        self,
        name: str = "DMD",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        alpha: float = 0.0,
        rank: float = 0.1,
        amplitude_threshold: float = None,
        eigenvalue_threshold: float = None,
        holiday_country: str = 'US',
        random_seed: int = 2022,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.alpha = alpha
        self.rank = rank
        self.amplitude_threshold = amplitude_threshold
        self.eigenvalue_threshold = eigenvalue_threshold

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied .

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """

        df = self.basic_profile(df)
        self.regressor_train = None
        self.verbose_bool = False
        if self.verbose > 1:
            self.verbose_bool = True

        if isinstance(self.rank, float):
            if self.rank < 1 and self.rank > 0:
                self.rank = int(self.rank * df.shape[1])
                self.rank = self.rank if self.rank > 0 else 1

        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        data = self.df_train.to_numpy().T
        forecast = dmd_forecast(
            data,
            r=self.rank,
            pred_step=forecast_length,
            alpha=self.alpha,
            amplitude_threshold=self.amplitude_threshold,
            eigenvalue_threshold=self.eigenvalue_threshold,
        ).T

        forecast = pd.DataFrame(forecast, index=test_index, columns=self.column_names)
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=test_index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        return {
            'rank': random.choices(
                [2, 3, 4, 6, 10, 0.1, 0.2, 0.5, "ecr"],
                [0.4, 0.1, 0.3, 0.1, 0.1, 0.1, 0.2, 0.2, 0.6],
            )[0],
            'alpha': random.choice([0.0, 0.001, 0.1, 1]),
            'amplitude_threshold': random.choices(
                [None, 0.1, 1, 10],
                [0.7, 0.1, 0.1, 0.1],
            )[0],
            'eigenvalue_threshold': random.choices(
                [None, 0.1, 1, 10],
                [0.7, 0.1, 0.1, 0.1],
            )[0],
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'rank': self.rank,
            'alpha': self.alpha,
            'amplitude_threshold': self.amplitude_threshold,
            'eigenvalue_threshold': self.eigenvalue_threshold,
        }
