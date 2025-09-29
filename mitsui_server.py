# -*- coding: utf-8 -*-
"""
Mitsui submission using AutoTS + template (no export)
- Lazy-load on first predict()
- Forecasts 1 step per batch

Location of train labels: /kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv
Location of template: /kaggle/input/autots_forecast_template_mitsui/other/default/1
"""
# !pip install autots

import os
import glob
import json
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

import kaggle_evaluation.mitsui_inference_server

# --- AutoTS imports ---
from autots import AutoTS, create_regressor

def custom_ranked_sharpe_metric(A, F, df_train=None, prediction_interval=None):
    """
    Compute a Sharpe-style score based on per-timestep rank correlation
    between forecasts and actuals, in a loss-compatible (lower is better) form.

    Parameters:
        A: np.ndarray, shape (timesteps, series) — actual values
        F: np.ndarray, same shape as A — forecasted values
        df_train: unused
        prediction_interval: unused

    Returns:
        float — a score where lower is better
    """
    epsilon = 1e-8
    T, N = A.shape

    rank_corrs = []

    for t in range(T):
        actual_t = A[t, :]
        pred_t = F[t, :]

        # Remove any pair where actual or forecast is NaN
        valid_mask = np.isfinite(actual_t) & np.isfinite(pred_t)

        if np.sum(valid_mask) < 2:
            continue  # skip if not enough valid entries

        actual_ranks = rankdata(actual_t[valid_mask])
        pred_ranks = rankdata(pred_t[valid_mask])

        # Compute Pearson correlation of ranks (Spearman correlation)
        cov = np.cov(actual_ranks, pred_ranks, ddof=0)
        var_actual = cov[0, 0]
        var_pred = cov[1, 1]
        if var_actual < epsilon or var_pred < epsilon:
            continue  # skip if one rank vector has no variance

        corr = cov[0, 1] / (np.sqrt(var_actual * var_pred) + epsilon)
        rank_corrs.append(corr)

    if len(rank_corrs) < 2:
        return 1e6  # Not enough valid steps for meaningful correlation

    rank_corrs = np.array(rank_corrs)
    mean_corr = np.nanmean(rank_corrs)
    std_corr = np.nanstd(rank_corrs)

    # Sharpe-style ratio of rank correlation (higher is better)
    sharpe = mean_corr / (std_corr + epsilon)

    # Convert to lower-is-better score by inverting and offsetting
    # score = 1 / (sharpe + 1e-6)

    return -(sharpe + 1e-6)

# ==============================
# Config (aligns with your script)
# ==============================
NUM_TARGET_COLUMNS = 424
FREQUENCY = "B"          # business day (matches competition labels cadence)
FORECAST_LENGTH = 1      # predict exactly one step per batch
DROP_MOST_RECENT = 1     # same as your training setting
PREDICTION_INTERVAL = 0.95
N_JOBS = "auto"
MODELS_MODE = "default"

# Use your model + ensemble selections as desired — you can trim for speed
MODEL_LIST = {
    'ETS': 1,
    # 'GLM': 1,
    # 'UnobservedComponents': 1,
    # 'UnivariateMotif': 1,
    # 'MultivariateMotif': 1,
    # 'Theta': 1,
    'ARDL': 1,
    # 'ARCH': 1,
    'ConstantNaive': 1,
    # 'LastValueNaive': 1.5,
    # 'AverageValueNaive': 1,
    # 'GLS': 1,
    'SeasonalNaive': 1,
    # 'VAR': 0.8,
    # 'VECM': 0.8,
    # 'WindowRegression': 0.5,
    # 'DatepartRegression': 0.8,
    # 'SectionalMotif': 1,
    # 'NVAR': 0.3,
    # 'MAR': 0.25,
    'RRVAR': 0.4,
    # 'KalmanStateSpace': 0.4,
    # 'MetricMotif': 1,
    # 'Cassandra': 0.6,
    'SeasonalityMotif': 1.5,
    # 'FFT': 0.8,
    # 'BallTreeMultivariateMotif': 0.4,
    "DMD": 0.4,
    # "BasicLinearModel": 1.2,
    # "MultivariateRegression": 0.8,
    # "TVVAR": 0.8,
    # "BallTreeRegressionMotif": 0.8,
}

ENSEMBLE = None
[
    'horizontal-min-20',
    'horizontal-min-40',
    "mosaic-mae-crosshair-0-20",
    "mosaic-weighted-crosshair-0-40",
    "mosaic-weighted-0-20",
    "mosaic-weighted-0-10",
    "mosaic-weighted-3-20",
    "mosaic-weighted-0-40",
    "mosaic-weighted-crosshair_lite-0-30",
    "mosaic-mae-profile-0-10",
    "mosaic-spl-unpredictability_adjusted-0-30",
    "mosaic-mae-median-profile",
    "mosaic-mae-0-horizontal",
    'mosaic-weighted-median-0-30',
    "mosaic-mae-median-profile-crosshair_lite-horizontal",
]

TRANSFORMER_LIST = "scalable"
TRANSFORMER_MAX_DEPTH = 8

METRIC_WEIGHTING = {
    'smape_weighting': 2,
    'mae_weighting': 2,
    'rmse_weighting': 1.5,
    'made_weighting': 2,
    'mage_weighting': 0,
    'mate_weighting': 0.01,
    'mle_weighting': 0,
    'imle_weighting': 0.0001,
    'spl_weighting': 3,
    'dwae_weighting': 1,
    'uwmse_weighting': 1,
    'ewmae_weighting': 1,
    'dwd_weighting': 1,
    "oda_weighting": 0.1,
    'runtime_weighting': 0.05,
    'custom_weighting': 4,
}

# Name of your uploaded template file (or pattern).
# Put the CSV in a dataset you add under /kaggle/input/, or set TEMPLATE_PATH via env var.
TEMPLATE_BASENAME = "autots_forecast_template_mitsui.csv"

# ==============================
# Globals (lazy-loaded)
# ==============================
_MODEL = None
_DF_TRAIN = None
_REGR_TRAIN = None
_HISTORICAL_DATA = None  # Accumulate new data here

# ==============================
# Utilities
# ==============================

def _find_competition_root():
    # Default mount for published competition data
    default_root = "/kaggle/input/mitsui-commodity-prediction-challenge"
    if os.path.exists(default_root):
        return default_root
    # Fallback: pick any kaggle input path that has train_labels.csv
    candidates = glob.glob("/kaggle/input/*")
    for c in candidates:
        if os.path.exists(os.path.join(c, "train_labels.csv")):
            return c
    raise FileNotFoundError("Could not locate competition data mount with train_labels.csv")

def _find_template_path():
    # Optionally set via env var
    envp = os.getenv("TEMPLATE_PATH")
    if envp and os.path.exists(envp):
        return envp
    # Otherwise search under /kaggle/input for the template filename
    for p in glob.glob(f"/kaggle/input/**/{TEMPLATE_BASENAME}", recursive=True):
        return p
    # As a last resort, accept any .csv that looks like an AutoTS template
    for p in glob.glob("/kaggle/input/**/*.csv", recursive=True):
        if "autots" in os.path.basename(p).lower() and "template" in os.path.basename(p).lower():
            return p
    raise FileNotFoundError(
        f"AutoTS template CSV not found. Ensure {TEMPLATE_BASENAME} exists in an input dataset "
        "or set TEMPLATE_PATH env var to its full path."
    )

def _load_training_df(comp_root: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(comp_root, "train_labels.csv"), index_col=0)
    # Competition labels are daily business days starting 2018-01-01
    df.index = pd.date_range("2018-01-01", periods=df.shape[0], freq=FREQUENCY)
    # Sanity: keep only target_0..target_423 if extra cols exist
    cols = [f"target_{i}" for i in range(NUM_TARGET_COLUMNS)]
    df = df.loc[:, cols]
    return df

def _build_future_regressor(df_wide: pd.DataFrame):
    # Match your training approach: build both train and forecast regressors
    # Ensure we have enough data for frequency inference (minimum 7 days to be safe)
    if len(df_wide) < 10:
        raise ValueError(f"Insufficient data for regressor creation: {len(df_wide)} rows, need at least 10")
    
    try:
        regr_train, regr_fcst = create_regressor(
            df_wide,
            forecast_length=FORECAST_LENGTH,
            frequency=FREQUENCY,
            drop_most_recent=DROP_MOST_RECENT,
            scale=True,
            summarize="auto",
            backfill="bfill",
            fill_na="spline",
            holiday_countries={"US": None},
            encode_holiday_type=True,
        )
        # Because create_regressor drops first `forecast_length` rows in returned regressors,
        # align training df likewise (mirror your training script behavior).
        df_aligned = df_wide.iloc[FORECAST_LENGTH:]
        regr_train = regr_train.iloc[FORECAST_LENGTH:]
        return df_aligned, regr_train, regr_fcst
    except Exception as e:
        # Fallback: if regressor creation fails, use minimal approach
        print(f"Warning: Failed to create regressors: {e}. Using simple approach.")
        df_aligned = df_wide.iloc[FORECAST_LENGTH:] if len(df_wide) > FORECAST_LENGTH else df_wide
        return df_aligned, None, None

def _lazy_init_model():
    global _MODEL, _DF_TRAIN, _REGR_TRAIN
    if _MODEL is not None:
        return

    comp_root = _find_competition_root()
    template_path = _find_template_path()

    # Load training labels
    df_raw = _load_training_df(comp_root)

    # Build regressors (aligned to training df)
    df_train_aligned, regr_train, regr_fcst = _build_future_regressor(df_raw)

    # Instantiate model with your settings but NO training (we'll load template)
    model = AutoTS(
        forecast_length=FORECAST_LENGTH,
        frequency=FREQUENCY,
        prediction_interval=PREDICTION_INTERVAL,
        ensemble=ENSEMBLE,
        model_list=MODEL_LIST,
        transformer_list=TRANSFORMER_LIST,
        transformer_max_depth=TRANSFORMER_MAX_DEPTH,
        metric_weighting=METRIC_WEIGHTING,
        initial_template=None,                 # we are importing, not generating
        aggfunc="first",
        models_to_validate=0.01,               # irrelevant here
        model_interrupt=True,
        num_validations=1,                     # irrelevant here
        validation_method="backwards",         # irrelevant here
        constraint=None,
        drop_most_recent=DROP_MOST_RECENT,
        models_mode=MODELS_MODE,
        current_model_file="current_model_mitsui",
        generation_timeout=30,                 # irrelevant here
        n_jobs=N_JOBS,
        verbose=1,
        custom_metric=custom_ranked_sharpe_metric,
    )

    # Import your best model(s) from the template and bind data quickly
    # If you exported many, you can also use import_template(..., method="only")
    try:
        model.import_best_model(template_path)  # fastest
    except Exception:
        # Fallback: load template entries as candidates
        model.import_template(template_path, method="only")

    # Fit data without re-search (fast), mirroring your branch:
    model.fit_data(df_train_aligned, future_regressor=regr_train)

    _MODEL = model
    _DF_TRAIN = df_train_aligned
    _REGR_TRAIN = regr_train
    # We won't persist regr_fcst; we’ll re-make the 1-step fcst regressor each call
    # based on up-to-date df index (keeps it simple & robust)

def _one_step_future_regressor(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Make a 1-step future regressor for the next timestamp."""
    _, _, regr_fcst = create_regressor(
        df_wide,
        forecast_length=FORECAST_LENGTH,
        frequency=FREQUENCY,
        drop_most_recent=DROP_MOST_RECENT,
        scale=True,
        summarize="auto",
        backfill="bfill",
        fill_na="spline",
        holiday_countries={"US": None},
        encode_holiday_type=True,
    )
    return regr_fcst

def _predict_next_row() -> pd.DataFrame:
    """Return a 1xN DataFrame with columns target_0..target_423."""
    # Build a 1-step future regressor from the current training window
    regr_fcst = _one_step_future_regressor(_DF_TRAIN)

    # Forecast 1 step
    pred = _MODEL.predict(future_regressor=regr_fcst, verbose=0, fail_on_forecast_nan=False)
    fcst_df = pred.forecast

    # We need exactly one row; select the last row (the 1-step ahead)
    # Ensure ordering and column presence
    cols = [f"target_{i}" for i in range(NUM_TARGET_COLUMNS)]
    if not all(c in fcst_df.columns for c in cols):
        # If model renamed columns, force align by best effort
        # (Should not happen if training columns match exactly)
        fcst_df = fcst_df.reindex(columns=cols)
    row = fcst_df.tail(1).astype(float)
    row.index = [0]  # explicit single row index for clean conversion
    return row

# ==============================
# Required predict() for server
# ==============================
def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    """
    Minimal live inference:
    - Lazy initialize model (load template + bind train data)
    - Ignore lag batches for now (model is purely time-series forward)
    - Return 1x424 next-step prediction
    """
    _lazy_init_model()

    # If you want to incorporate lag info later, you can use these batches to
    # update a rolling state. For now, we skip and rely on model forecasting.
    next_row = _predict_next_row()

    # Return as Polars for speed (server accepts pandas or polars)
    out_pl = pl.from_pandas(next_row)
    assert isinstance(out_pl, (pl.DataFrame, pd.DataFrame))
    assert len(out_pl) == 1
    return out_pl


# ==============================
# Server bootstrap
# ==============================
inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))
