# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 08:25:34 2022

@author: Colin
"""
import numpy as np
import pandas as pd

import pymc as pm
from pymc.distributions.timeseries import GaussianRandomWalk
from scipy import optimize
import matplotlib.pyplot as plt

df = pd.read_csv(
    "holidays.csv", index_col=0, parse_dates=[0], infer_datetime_format=True
)
x = df.index.to_julian_date()
y = df['wiki_all'].values

LARGE_NUMBER = 1e5

model = pm.Model()
with model:
    smoothing_param = 0.9
    mu = pm.Normal("mu", sigma=LARGE_NUMBER)
    tau = pm.Exponential("tau", 1.0 / LARGE_NUMBER)
    #  τ = 1/σ2
    z = GaussianRandomWalk("z", mu=mu, sigma=LARGE_NUMBER / (1.0 - smoothing_param), shape=y.shape)
    obs = pm.Normal("obs", mu=z, tau=tau / smoothing_param, observed=y)

def infer_z():
    with model:
        # smoothing_param.set_value(smoothing)
        res = pm.find_MAP(vars=[z], fmin=optimize.fmin_l_bfgs_b)
        return res["z"]

smoothing = 0.5
z_val = infer_z()

plt.plot(x, y)
plt.plot(x, z_val)
plt.show()

npl = df.copy()
npl['date'] = df.index
npl['gkv'] = np.exp(
    -(((npl.index - pd.Timestamp(2020, 10, 27)).days) ** 2) / (2 * (2 ** 2))
)

smoothed_cases = []
for date in sorted(npl['date']):
    npl['gkv'] = np.exp(
        -(((npl.index - date).days) ** 2) / (2 * (2 ** 2))
    )
    npl['gkv'] /= npl['gkv'].sum()
    smoothed_cases.append(round(npl['wiki_all'] * npl['gkv']).sum())

npl['smoothed_new_cases'] = smoothed_cases

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

from numpy.linalg import inv

def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
    Computes the suffifient statistics of the posterior distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)
    
    # Equation (7)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, cov_s

X = pd.date_range(df.index[-1], periods=15, freq='D')[1:].to_julian_date().values.reshape(-1, 1)

# Mean and covariance of the prior
mu = np.zeros(X.shape)
cov = kernel(X, X)

X_train = df.index.to_julian_date().values.reshape(-1, 1)
Y_train = df['wiki_all'].values.reshape(-1, 1)
mu_s, cov_s = posterior(X, X_train, Y_train)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)





t = df.index.to_julian_date().to_numpy()
t_min = np.min(t)
t_max = np.max(t)
t = (t - t_min) / (t_max - t_min)

y = df["wiki_all"].to_numpy()
y_max = np.max(y)
y = y / y_max

series_order = 10
period = 365.25

t = np.array(
            (df.index - pd.Timestamp(1970, 1, 1))
                .total_seconds()
                .astype(float)
        ) / (3600 * 24.)
fourier_features = np.column_stack([
    fun((2.0 * (i + 1) * np.pi * t / period))
    for i in range(series_order)
    for fun in (np.sin, np.cos)
])
fourier_features
coords = {"fourier_features": np.arange(2 * series_order)}
with pm.Model(check_bounds=False, coords=coords) as linear_with_seasonality:
    α = pm.Normal("α", mu=0, sigma=0.5)
    β = pm.Normal("β", mu=0, sigma=0.5)
    trend = pm.Deterministic("trend", α + β * t)

    β_fourier = pm.Normal("β_fourier", mu=0, sigma=0.1, dims="fourier_features")
    seasonality = pm.Deterministic(
        "seasonality", pm.math.dot(β_fourier, np.array(fourier_features).T)
    )

    μ = trend * (1 + seasonality)
    σ = pm.HalfNormal("σ", sigma=0.1)
    pm.Normal("likelihood", mu=μ, sigma=σ, observed=y)

    linear_seasonality_prior = pm.sample_prior_predictive()
    print(linear_with_seasonality)

with linear_with_seasonality:
    linear_seasonality_trace = pm.sample(return_inferencedata=True)
    linear_seasonality_posterior = pm.sample_posterior_predictive(trace=linear_seasonality_trace)


import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
df['Month'] = df.index

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(8, 6))
ax[0].plot(
    df["Month"],
    az.extract_dataset(linear_seasonality_posterior, group="posterior_predictive", num_samples=100)[
        "likelihood"
    ]
    * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="wiki_all", color="k", ax=ax[0])
ax[0].set_title("Posterior predictive")
ax[1].plot(
    df["Month"],
    az.extract_dataset(linear_seasonality_trace, group="posterior", num_samples=100)["trend"] * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="wiki_all", color="k", ax=ax[1])
ax[1].set_title("Posterior trend lines")
ax[2].plot(
    df["Month"].iloc[:12],
    az.extract_dataset(linear_seasonality_trace, group="posterior", num_samples=100)["seasonality"][
        :12
    ]
    * 100,
    color="blue",
    alpha=0.05,
)
ax[2].set_title("Posterior seasonality")
ax[2].set_ylabel("Percent change")
formatter = mdates.DateFormatter("%b")
ax[2].xaxis.set_major_formatter(formatter);
