import numpy as np


class BayesianMultiOutputRegression:
    """Matrix-normal/Inverse-Wishart Bayesian linear regression.

    Computes an analytical posterior for multivariate linear regression with
    Gaussian weight noise and unknown Gaussian observation noise. The posterior
    mean matches ridge regression, while the stored covariance allows callers
    to recover coefficient standard deviations, prediction intervals, and
    posterior samples without resorting to numerical sampling during training.
    """

    def __init__(
        self,
        gaussian_prior_mean=0.0,
        alpha=1.0,
        wishart_prior_scale=1.0,
        wishart_dof_excess=0,
    ):
        self.gaussian_prior_mean = gaussian_prior_mean
        self.alpha = alpha
        self.wishart_prior_scale = wishart_prior_scale
        self.wishart_dof_excess = wishart_dof_excess

        self.params = None
        self.coef_mean_ = None
        self.coef_cov_ = None
        self.coef_std_ = None
        self.noise_covariance_ = None
        self.nu_ = None
        self.n_outputs_ = None
        self._one_dim_target = False

    def _prepare_prior_mean(self, n_features, n_outputs):
        mean = self.gaussian_prior_mean
        if np.isscalar(mean):
            return np.full((n_features, n_outputs), float(mean))
        mean = np.asarray(mean, dtype=float)
        if mean.ndim == 1:
            if mean.shape[0] != n_features:
                raise ValueError("Prior mean vector must match n_features.")
            mean = mean[:, None]
        if mean.shape == (n_features, 1):
            return np.repeat(mean, n_outputs, axis=1)
        if mean.shape != (n_features, n_outputs):
            raise ValueError(
                "Prior mean must broadcast to shape (n_features, n_outputs)."
            )
        return mean

    def _prepare_prior_cov(self, n_features):
        alpha = self.alpha
        if np.isscalar(alpha):
            return float(alpha) * np.eye(n_features)
        alpha = np.asarray(alpha, dtype=float)
        if alpha.ndim == 1:
            if alpha.shape[0] != n_features:
                raise ValueError("Diagonal prior covariance must match n_features.")
            return np.diag(alpha)
        if alpha.shape != (n_features, n_features):
            raise ValueError(
                "Prior covariance must be scalar, diagonal length n_features, "
                "or full matrix with shape (n_features, n_features)."
            )
        return alpha

    def _prepare_noise_scale(self, n_outputs):
        scale = self.wishart_prior_scale
        if np.isscalar(scale):
            return float(scale) * np.eye(n_outputs)
        scale = np.asarray(scale, dtype=float)
        if scale.ndim == 1:
            if scale.shape[0] != n_outputs:
                raise ValueError("Diagonal noise scale must match n_outputs.")
            return np.diag(scale)
        if scale.shape != (n_outputs, n_outputs):
            raise ValueError(
                "Noise scale must be scalar, diagonal length n_outputs, "
                "or full matrix with shape (n_outputs, n_outputs)."
            )
        return scale

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if Y.ndim == 1:
            Y = Y[:, None]
            self._one_dim_target = True
        elif Y.ndim == 2:
            self._one_dim_target = False
        else:
            raise ValueError("Y must be 1D or 2D.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows.")

        n_samples, n_features = X.shape
        n_outputs = Y.shape[1]
        self.n_outputs_ = n_outputs

        m0 = self._prepare_prior_mean(n_features, n_outputs)
        S0 = self._prepare_prior_cov(n_features)
        S0_inv = np.linalg.solve(S0, np.eye(n_features))

        nu0 = n_outputs + self.wishart_dof_excess
        W0 = self._prepare_noise_scale(n_outputs)

        XtX = X.T @ X
        XtY = X.T @ Y

        Sn_inv = S0_inv + XtX
        Sn = np.linalg.solve(Sn_inv, np.eye(n_features))
        rhs = S0_inv @ m0 + XtY
        m_n = np.linalg.solve(Sn_inv, rhs)

        residual = Y - X @ m_n
        delta = m_n - m0
        Wn = W0 + residual.T @ residual + delta.T @ S0_inv @ delta

        nu_n = nu0 + n_samples

        self.coef_mean_ = m_n
        self.coef_cov_ = Sn
        self.coef_prior_cov_ = S0
        self.coef_prior_mean_ = m0
        self.params = m_n
        self.nu_ = nu_n

        denom = nu_n - n_outputs - 1
        if denom > 0:
            noise_cov = Wn / denom
        else:
            noise_cov = Wn / max(nu_n, 1)
        noise_cov = (noise_cov + noise_cov.T) / 2.0
        self.noise_scale_ = Wn
        self.noise_covariance_ = noise_cov

        coef_sd_features = np.sqrt(np.maximum(np.diag(Sn), 0.0))
        noise_sd_outputs = np.sqrt(np.maximum(np.diag(noise_cov), 0.0))
        self.coef_std_ = np.outer(coef_sd_features, noise_sd_outputs)

        return self

    def predict(self, X, return_std=False):
        if self.coef_mean_ is None:
            raise ValueError("Model has not been fitted.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.coef_mean_.shape[0]:
            raise ValueError("X has wrong number of features.")

        Y_pred = X @ self.coef_mean_

        if return_std:
            x_cov = np.einsum('ij,jk,ik->i', X, self.coef_cov_, X)
            denom = self.nu_ - self.n_outputs_ - 1
            scale = (denom + x_cov) / denom if denom > 0 else 1.0 + x_cov
            noise_diag = np.diag(self.noise_covariance_)
            Y_std = np.sqrt(scale[:, None] * noise_diag[None, :])
            if self._one_dim_target:
                return Y_pred.ravel(), Y_std.ravel()
            return Y_pred, Y_std

        if self._one_dim_target:
            return Y_pred.ravel()
        return Y_pred

    def coefficient_interval(self, z_value=1.96):
        if self.coef_mean_ is None:
            raise ValueError("Model has not been fitted.")
        lower = self.coef_mean_ - z_value * self.coef_std_
        upper = self.coef_mean_ + z_value * self.coef_std_
        return lower, upper

    def sample_posterior(self, n_samples=1):
        if self.coef_mean_ is None:
            raise ValueError("Model has not been fitted.")

        n_features, n_outputs = self.coef_mean_.shape
        samples = np.zeros((n_samples, n_features, n_outputs))

        try:
            chol_coef = np.linalg.cholesky(self.coef_cov_)
        except np.linalg.LinAlgError:
            chol_coef = np.linalg.cholesky(
                self.coef_cov_ + 1e-9 * np.eye(n_features)
            )

        try:
            chol_noise = np.linalg.cholesky(self.noise_covariance_)
        except np.linalg.LinAlgError:
            chol_noise = np.linalg.cholesky(
                self.noise_covariance_ + 1e-9 * np.eye(n_outputs)
            )

        for i in range(n_samples):
            z = np.random.normal(size=(n_features, n_outputs))
            samples[i] = self.coef_mean_ + chol_coef @ z @ chol_noise.T

        return samples
