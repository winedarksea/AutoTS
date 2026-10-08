# -*- coding: utf-8 -*-
"""
TVA benchmark harness — the gate every TVA change is measured against.

This script:
- Loads 3 real/example datasets (load_daily, load_monthly, load_artificial)
  plus a locally generated factor/hierarchy panel with known latent factors,
  hierarchy, and short-history series.
- Runs baseline models via model_forecast with a fill-only transform dict,
  the TimeSeriesFeatureDetector's own forecast() (the baseline that matters),
  and TVAModel (only if torch is importable).
- Evaluates rolling-origin folds with PredictionObject.evaluate() plus
  harness-local coherence metrics.
- Prints a markdown summary table (skill vs SeasonalNaive) and writes a tidy
  JSON of every row.

Usage:
  python examples/tva_benchmark.py --baselines-only          # no torch needed
  python examples/tva_benchmark.py --out bench.json          # full incl. TVA
  python examples/tva_benchmark.py --smoke                   # tiny CI-style run
  python examples/tva_benchmark.py --models SeasonalNaive,TVAModel --folds 2

LRP 180-day protocol (0d/0f/0h):
  # iteration folds (0,1,2) only — folds 3,4 stay held back
  python examples/tva_benchmark.py --data lrp_forecast_data_202502.csv --horizon 180
  # promotion run (loud banner; do not tune against these)
  python examples/tva_benchmark.py --data ... --horizon 180 --promotion
  # arbitration ceiling diagnostic
  python examples/tva_benchmark.py --data ... --arbitration-probe \
      --arbitration-models 'TVAModel[factor],SeasonalNaive'

  --data builds a single dataset named after the file stem (season_m=7) and
  skips the built-in datasets unless they are named in --datasets. Origins are
  frozen by fixed_origins() and do not depend on --folds. Series are tagged
  base / derived_ratio / frozen_tail from the static, checked-in map in
  examples/lrp_series_tags.py; gates use the base columns only. No
  series_metadata or prior_adjacency is ever passed on these runs.

Notes:
- Point skill = metric_SeasonalNaive / metric_model (>1 means better than
  SeasonalNaive), aggregated as geometric mean over dataset x fold x horizon.
- dca_error: |directional-coherence-agreement(forecast) - same(actuals)| for
  pairs whose (smoothed, differenced) training trends correlate > 0.5.
  Reported as an error so a flat forecast cannot trivially win.
- agg_error: ||S @ bottom_fc - agg_fc||_1 / ||actual_agg||_1 on the factor
  panel only (aggregate columns are part of the panel, so every model
  forecasts them; coherent models should keep them consistent).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from autots.datasets import load_daily, load_monthly, load_artificial

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# transform dict that only fills NaN (short-history series) — no transformations
FILL_ONLY_TRANSFORM = {
    "fillna": "ffill",
    "transformations": {},
    "transformation_params": {},
}

# per-series normalization, for the pooled cross-learning comparator (0h):
# a global model needs each series on a comparable scale
PER_SERIES_NORM_TRANSFORM = {
    "fillna": "ffill",
    "transformations": {"0": "StandardScaler"},
    "transformation_params": {"0": {}},
}

POOLED_PREFIX = "PooledComparator"
POOLED_COMPARATOR_MODEL = "MultivariateRegression"
POOLED_COMPARATOR_FALLBACK = "WindowRegression"
POOLED_COMPARATOR_LABEL = f"{POOLED_PREFIX}[{POOLED_COMPARATOR_MODEL}]"

BASELINE_MODELS = [
    # (model_name, params) run via model_forecast
    ("SeasonalNaive", {}),
    ("LastValueNaive", {}),
    ("BasicLinearModel", {}),  # channel-independent per-column linear
    ("VAR", {}),
    ("DynamicFactorMQ", {}),
    ("SectionalMotif", {}),
    ("Cassandra", {}),
]

DETECTOR_MODEL_NAME = "FeatureDetectorForecast"
TVA_MODEL_NAME = "TVAModel"


# ---------------------------------------------------------------------------
# Factor / hierarchy panel
# ---------------------------------------------------------------------------


def make_factor_panel(n_days: int = 1095, seed: int = 42):
    """Hierarchical factor panel: 3 latent factors drive 36 bottom series.

    3 metrics x 4 surfaces x 3 geos = 36 bottom series, each
    ``loading * factor(metric) + AR(1) idiosyncratic + seasonality + holidays``
    with per-geo scale multipliers spanning 3 orders of magnitude, 2 level
    shifts, and 2 short-history (responder) series. Aggregate columns (one per
    metric) are appended so aggregation consistency is measurable.

    Returns:
        dict with 'df' (wide, bottom + aggregate cols), 'S' (n_agg, n_bottom),
        'bottom_cols', 'agg_cols', 'series_metadata' (list of dicts),
        'factors' (T, k), 'loadings' (n_bottom, k).
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    metrics = ["demand", "engagement", "revenue"]
    surfaces = ["web", "mobile", "tablet", "tv"]
    geos = ["US", "EU", "APAC"]
    k = len(metrics)

    # latent factors: smoothed random walks with drift (the composite trends)
    factors = np.zeros((n_days, k))
    for j in range(k):
        drift = rng.normal(0.0, 0.004)
        steps = rng.normal(0.0, 0.06, n_days)
        walk = np.cumsum(steps) + drift * t
        factors[:, j] = (
            pd.Series(walk).rolling(14, min_periods=1, center=True).mean().values
        )

    geo_scale = {"US": 100.0, "EU": 10.0, "APAC": 1.0}
    bottom_cols = []
    series_metadata = []
    loadings = np.zeros((len(metrics) * len(surfaces) * len(geos), k))
    values = {}
    weekly = np.sin(2 * np.pi * (t % 7) / 7.0) + 0.4 * np.cos(
        2 * np.pi * (t % 7) / 7.0
    )
    yearly = np.sin(2 * np.pi * (t % 365.25) / 365.25)
    holiday_mask = np.zeros(n_days)
    for month, day in [(12, 25), (7, 4), (11, 27)]:
        holiday_mask += (
            (index.month == month) & (index.day == day)
        ).astype(float)

    i = 0
    for mi, metric in enumerate(metrics):
        for surface in surfaces:
            for geo in geos:
                name = f"{metric}_{surface}_{geo}"
                bottom_cols.append(name)
                series_metadata.append(
                    {"name": name, "metric": metric, "surface": surface, "geo": geo}
                )
                # dominant loading on own-metric factor, small cross-loading
                load = np.zeros(k)
                load[mi] = rng.uniform(0.7, 1.4) * rng.choice([1.0, 1.0, 1.0, -1.0])
                other = (mi + 1) % k
                load[other] = rng.uniform(-0.25, 0.25)
                loadings[i] = load

                phi = 0.6
                idio = np.zeros(n_days)
                shocks = rng.normal(0, 0.25, n_days)
                for s in range(1, n_days):
                    idio[s] = phi * idio[s - 1] + shocks[s]

                base = 5.0 + rng.uniform(0, 3)
                signal = (
                    base
                    + factors @ load
                    + idio
                    + rng.uniform(0.2, 0.8) * weekly
                    + rng.uniform(0.0, 0.5) * yearly
                    + rng.uniform(0.5, 2.0) * holiday_mask
                )
                # occasional level shift
                if i in (5, 17):
                    shift_at = int(n_days * rng.uniform(0.4, 0.8))
                    signal[shift_at:] += rng.choice([-1.5, 1.5])
                values[name] = signal * geo_scale[geo]
                i += 1

    df = pd.DataFrame(values, index=index)

    # two short-history responder series (leading NaN)
    short = list(df.columns[[7, 23]])
    cutoff = int(n_days * 0.65)
    for col in short:
        df.iloc[:cutoff, df.columns.get_loc(col)] = np.nan

    # aggregate columns: one per metric (sum over that metric's bottom series)
    agg_cols = []
    n_bottom = len(bottom_cols)
    S = np.zeros((len(metrics), n_bottom))
    for mi, metric in enumerate(metrics):
        member_idx = [
            j for j, meta in enumerate(series_metadata) if meta["metric"] == metric
        ]
        S[mi, member_idx] = 1.0
        agg_name = f"agg_{metric}"
        agg_cols.append(agg_name)
        df[agg_name] = df[bottom_cols].fillna(0.0).values @ S[mi]

    return {
        "df": df,
        "S": S,
        "bottom_cols": bottom_cols,
        "agg_cols": agg_cols,
        "series_metadata": series_metadata,
        "factors": factors,
        "loadings": loadings,
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def build_datasets(seed: int = 42, smoke: bool = False) -> dict:
    """Return {name: spec} where spec has df, horizons, season_m, freq info."""
    datasets = {}
    factor = make_factor_panel(n_days=420 if smoke else 1095, seed=seed)
    datasets["factor_panel"] = {
        "df": factor["df"],
        "horizons": [14] if smoke else [14, 28],
        "season_m": 7,
        "S": factor["S"],
        "bottom_cols": factor["bottom_cols"],
        "agg_cols": factor["agg_cols"],
        "series_metadata": factor["series_metadata"],
    }
    # latent trend-factor panel: the data type TVA is designed for, with
    # known factors/loadings/response lags (see tva_factor_validation.py)
    from autots.datasets import generate_synthetic_daily_data

    synth = generate_synthetic_daily_data(
        n_days=420 if smoke else 1095,
        n_series=24,
        random_seed=seed,
        noise_level=0.05,
        trend_changepoint_freq=2.0,
        series_type_override='standard',
        n_latent_factors=3,
        factor_strength=0.8,
        factor_response_lag_max=10,
    )
    datasets["synthetic_factor_panel"] = {
        "df": synth.get_data(),
        "horizons": [14] if smoke else [14, 28],
        "season_m": 7,
    }

    if smoke:
        return datasets

    daily = load_daily(long=False)
    # trim very long histories for runtime sanity
    datasets["load_daily"] = {
        "df": daily.iloc[-1460:],
        "horizons": [14, 28],
        "season_m": 7,
    }
    monthly = load_monthly(long=False)
    datasets["load_monthly"] = {
        "df": monthly,
        "horizons": [6, 12],
        "season_m": 12,
    }
    artificial = load_artificial(long=False)
    datasets["load_artificial"] = {
        "df": artificial,
        "horizons": [14, 28],
        "season_m": 7,
    }
    return datasets


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mase_value(actual: pd.DataFrame, forecast: pd.DataFrame, train: pd.DataFrame, m: int) -> float:
    """Mean (over series) of MAE / in-sample seasonal-naive MAE."""
    A = actual.values
    F = forecast.reindex(columns=actual.columns).values
    mae = np.nanmean(np.abs(A - F), axis=0)
    tr = train.values
    if tr.shape[0] > m:
        scale = np.nanmean(np.abs(tr[m:] - tr[:-m]), axis=0)
    else:
        scale = np.nanmean(np.abs(np.diff(tr, axis=0)), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.nan)
    ratio = mae / scale
    return float(np.nanmean(ratio))


def correlated_pairs(train: pd.DataFrame, season_m: int, threshold: float = 0.5, max_pairs: int = 200):
    """Pairs of columns whose smoothed, differenced training trends correlate > threshold."""
    window = max(season_m, 3)
    smooth = train.rolling(window, min_periods=1).mean()
    diffs = smooth.diff().dropna(how="all")
    corr = diffs.corr()
    cols = list(train.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if np.isfinite(c) and c > threshold:
                pairs.append((cols[i], cols[j]))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[x] for x in idx]
    return pairs


def sign_agreement(df: pd.DataFrame, pairs) -> dict:
    """Per-pair rate of first-difference sign agreement."""
    diffs = df.diff().dropna(how="all")
    out = {}
    for a, b in pairs:
        da = np.sign(diffs[a].values)
        db = np.sign(diffs[b].values)
        valid = np.isfinite(da) & np.isfinite(db)
        if valid.sum() == 0:
            continue
        out[(a, b)] = float(np.mean(da[valid] == db[valid]))
    return out


def dca_error(forecast: pd.DataFrame, actual: pd.DataFrame, pairs) -> float:
    """Mean |sign-agreement(forecast) - sign-agreement(actual)| over pairs."""
    if not pairs:
        return np.nan
    fc = sign_agreement(forecast.reindex(columns=actual.columns), pairs)
    ac = sign_agreement(actual, pairs)
    errs = [abs(fc[p] - ac[p]) for p in fc if p in ac]
    return float(np.mean(errs)) if errs else np.nan


def _tva_metrics():
    """Lazily return autots.evaluator.tva.metrics, or None if unavailable."""
    try:
        from autots.evaluator.tva import metrics as _m  # noqa: F401

        return _m
    except ImportError:
        return None
    except Exception:
        return None


def per_series_mase(actual: pd.DataFrame, forecast: pd.DataFrame, train: pd.DataFrame, m: int) -> dict:
    """{column: MASE} — the per-series version of :func:`mase_value`."""
    cols = list(actual.columns)
    F = forecast.reindex(columns=cols)
    A = actual.values
    mae = np.nanmean(np.abs(A - F.values), axis=0)
    tr = train.reindex(columns=cols).values
    if tr.shape[0] > m:
        scale = np.nanmean(np.abs(tr[m:] - tr[:-m]), axis=0)
    else:
        scale = np.nanmean(np.abs(np.diff(tr, axis=0)), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.nan)
    ratio = mae / scale
    return {c: float(v) for c, v in zip(cols, ratio)}


def net_change_direction(frame: pd.DataFrame, window: int = 28) -> pd.Series:
    """sign(mean(last `window` rows) - mean(first `window` rows)) per column."""
    m = _tva_metrics()
    if m is not None and hasattr(m, "net_change_direction"):
        return m.net_change_direction(frame, window=window)
    w = int(min(window, max(1, len(frame) // 2)))
    head = frame.iloc[:w].mean()
    tail = frame.iloc[-w:].mean()
    return np.sign(tail - head)


def _pair_direction_map(frame: pd.DataFrame, pairs, window: int = 28) -> dict:
    """{pair: 1.0 if the two series move the same net direction else 0.0}."""
    d = net_change_direction(frame, window=window)
    out = {}
    for a, b in pairs:
        if a not in d.index or b not in d.index:
            continue
        da, db = d[a], d[b]
        if not (np.isfinite(da) and np.isfinite(db)):
            continue
        out[(a, b)] = float(da == db)
    return out


def direction_coherence(frame: pd.DataFrame, pairs, window: int = 28) -> float:
    """Rate at which paired series share a net direction over `window`."""
    m = _tva_metrics()
    if m is not None and hasattr(m, "direction_coherence"):
        return float(m.direction_coherence(frame, pairs, window=window))
    vals = list(_pair_direction_map(frame, pairs, window).values())
    return float(np.mean(vals)) if vals else np.nan


def direction_coherence_error(forecast: pd.DataFrame, actual: pd.DataFrame, pairs, window: int = 28) -> float:
    """|net-direction agreement rate(forecast) - same(actual)| over pairs."""
    m = _tva_metrics()
    if m is not None and hasattr(m, "direction_coherence_error"):
        return float(m.direction_coherence_error(forecast, actual, pairs, window=window))
    if not pairs:
        return np.nan
    fc = _pair_direction_map(forecast.reindex(columns=actual.columns), pairs, window)
    ac = _pair_direction_map(actual, pairs, window)
    common = [p for p in fc if p in ac]
    if not common:
        return np.nan
    return float(abs(np.mean([fc[p] for p in common]) - np.mean([ac[p] for p in common])))


def real_data_coherence(forecast: pd.DataFrame, actual: pd.DataFrame, pairs, window: int = 28) -> float:
    """Per-pair rate at which the forecast reproduces the ACTUAL pair
    co-movement (both up/both down vs. diverging). Higher is better; unlike
    :func:`direction_coherence_error` a flat/averaged forecast cannot win by
    matching the aggregate rate while getting every pair wrong."""
    m = _tva_metrics()
    if m is not None and hasattr(m, "real_data_coherence"):
        return float(m.real_data_coherence(forecast, actual, pairs, window=window))
    if not pairs:
        return np.nan
    fc = _pair_direction_map(forecast.reindex(columns=actual.columns), pairs, window)
    ac = _pair_direction_map(actual, pairs, window)
    common = [p for p in fc if p in ac]
    if not common:
        return np.nan
    return float(np.mean([fc[p] == ac[p] for p in common]))


def aggregation_error(forecast: pd.DataFrame, actual: pd.DataFrame, spec: dict) -> float:
    """||S @ bottom_fc - agg_fc||_1 / ||actual_agg||_1."""
    S = spec.get("S")
    if S is None:
        return np.nan
    bottom = forecast[spec["bottom_cols"]].values
    agg = forecast[spec["agg_cols"]].values
    implied = bottom @ S.T
    denom = np.abs(actual[spec["agg_cols"]].values).sum()
    if denom <= 0:
        return np.nan
    return float(np.abs(implied - agg).sum() / denom)


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------


def run_model_forecast(model_name, params, train, horizon, seed, transform_dict=None):
    from autots.evaluator.auto_model import model_forecast

    return model_forecast(
        model_name=model_name,
        model_param_dict=params,
        model_transform_dict=FILL_ONLY_TRANSFORM if transform_dict is None else transform_dict,
        df_train=train,
        forecast_length=horizon,
        frequency="infer",
        prediction_interval=0.9,
        random_seed=seed,
        verbose=0,
        n_jobs=1,
        fail_on_forecast_nan=True,
    )


def run_detector_forecast(train, horizon):
    from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

    detector = TimeSeriesFeatureDetector()
    detector.fit(train.ffill().bfill())
    return detector.forecast(horizon, prediction_interval=0.9)


def evaluate_prediction(pred, actual, train, season_m, spec, pairs, dir_window: int = 28):
    """Compute the harness metric row from a PredictionObject.

    If ``spec['tags']`` is present ({tag: [columns]}, see
    ``examples/lrp_series_tags.py``) the row also carries per-tag MASE, a
    per-series MASE dict, and the 180-day net-direction coherence metrics
    computed over the base-column pairs only.
    """
    forecast = pred.forecast.reindex(columns=actual.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred.evaluate(actual=actual, df_train=train)
    avg = pred.avg_metrics
    row = {
        "smape": float(avg.get("smape", np.nan)),
        "mase": mase_value(actual, forecast, train, season_m),
        "spl": float(avg.get("spl", np.nan)),
        "containment": float(avg.get("containment", np.nan)),
        "dca_error": dca_error(forecast, actual, pairs),
        "agg_error": aggregation_error(forecast, actual, spec),
    }
    tags = spec.get("tags")
    if tags:
        ps = per_series_mase(actual, forecast, train, season_m)
        row["per_series_mase"] = ps
        for tag, key in (
            ("base", "mase_base"),
            ("derived_ratio", "mase_derived"),
            ("frozen_tail", "mase_frozen"),
        ):
            cols = [c for c in tags.get(tag, []) if c in ps]
            vals = [ps[c] for c in cols]
            row[key] = float(np.nanmean(vals)) if vals else np.nan
        base_cols = [c for c in tags.get("base", []) if c in actual.columns]
        if base_cols:
            row["dir_coh_error"] = direction_coherence_error(
                forecast[base_cols], actual[base_cols], pairs, window=dir_window
            )
            row["real_data_coherence"] = real_data_coherence(
                forecast[base_cols], actual[base_cols], pairs, window=dir_window
            )
        else:
            row["dir_coh_error"] = np.nan
            row["real_data_coherence"] = np.nan
    return row


# ---------------------------------------------------------------------------
# Benchmark protocol
# ---------------------------------------------------------------------------


def load_wide_csv(path: str) -> pd.DataFrame:
    """Load a wide CSV (date column + one column per series) into a DataFrame.

    The date column is ``ds`` if present, else the first column. Values are
    coerced to numeric, the index is sorted, and nothing else is imputed.
    """
    df = pd.read_csv(path)
    date_col = "ds" if "ds" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = date_col
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, [c for c in df.columns if df[c].notna().any()]]
    return df


def fixed_origins(df: pd.DataFrame, horizon: int, n_origins: int = 5):
    """Yield (fold, train, actual) for NON-OVERLAPPING test windows anchored
    at the end of the data, oldest fold first.

    Frozen layout: fold k covers rows [T - horizon*(n_origins-k),
    T - horizon*(n_origins-k-1)); it does not depend on --folds or on which
    subset of folds is being run, so an iteration-only run and a promotion
    run see byte-identical windows. Fold 0..2 are the iteration folds and
    3..4 the promotion folds by convention (enforced by the caller).

    A fold whose training history is shorter than ``3 * horizon`` is dropped
    with a warning (same minimum-history rule as :func:`rolling_origins`).
    """
    T = len(df)
    for fold in range(n_origins):
        cut = T - horizon * (n_origins - fold)
        if cut < horizon * 3:
            print(
                f"  [fixed_origins] dropping fold {fold}: only {max(cut, 0)} train "
                f"rows < 3*horizon ({horizon * 3})"
            )
            continue
        yield fold, df.iloc[:cut], df.iloc[cut : cut + horizon]


def rolling_origins(df: pd.DataFrame, horizon: int, n_folds: int):
    """Yield (train, actual) splits, oldest fold first."""
    T = len(df)
    for fold in range(n_folds):
        cut = T - horizon * (n_folds - fold)
        if cut < horizon * 3:
            continue
        train = df.iloc[:cut]
        actual = df.iloc[cut : cut + horizon]
        yield fold, train, actual


def run_benchmark(
    datasets: dict,
    models: list,
    n_folds: int = 4,
    seed: int = 42,
    include_tva: bool = True,
    tva_params: dict = None,
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []
    for ds_name, spec in datasets.items():
        df = spec["df"]
        season_m = spec["season_m"]
        tags = spec.get("tags")
        if tags is not None:
            # 0d item 6: no semantic metadata may reach the model on tagged
            # (LRP) runs — the harness must not smuggle in domain knowledge
            assert spec.get("series_metadata") is None, "series_metadata must be None"
            assert spec.get("prior_adjacency") is None, "prior_adjacency must be None"
        for horizon in spec["horizons"]:
            if spec.get("fixed_origins"):
                origin_iter = fixed_origins(
                    df, horizon, spec.get("n_origins", 5)
                )
                folds_allowed = spec.get("folds_allowed")
            else:
                origin_iter = rolling_origins(df, horizon, n_folds)
                folds_allowed = None
            for fold, train, actual in origin_iter:
                if folds_allowed is not None and fold not in folds_allowed:
                    continue
                if tags:
                    base_cols = [c for c in tags.get("base", []) if c in train.columns]
                    pairs = correlated_pairs(train[base_cols], season_m)
                else:
                    pairs = correlated_pairs(train, season_m)
                for model_name, params in models:
                    call_name = model_name
                    if model_name.startswith(TVA_MODEL_NAME):
                        if not include_tva or not HAS_TORCH:
                            continue
                        # shared tva_params are defaults; a variant's own params
                        # (e.g. trend_network) win, so several TVA arms can run
                        # side by side under distinct display names
                        params = {**(tva_params or {}), **(params or {})}
                        call_name = TVA_MODEL_NAME
                    start = time.perf_counter()
                    try:
                        if model_name == DETECTOR_MODEL_NAME:
                            pred = run_detector_forecast(train, horizon)
                        elif model_name.startswith(POOLED_PREFIX):
                            # 0h: pooled cross-learning COMPARATOR (never a gate)
                            inner = model_name[len(POOLED_PREFIX) + 1 : -1]
                            try:
                                pred = run_model_forecast(
                                    inner, params, train, horizon, seed,
                                    transform_dict=PER_SERIES_NORM_TRANSFORM,
                                )
                            except Exception as pooled_exc:
                                if verbose:
                                    print(
                                        f"  [pooled comparator] {inner} failed "
                                        f"({type(pooled_exc).__name__}); falling back to "
                                        f"{POOLED_COMPARATOR_FALLBACK}"
                                    )
                                pred = run_model_forecast(
                                    POOLED_COMPARATOR_FALLBACK, {}, train, horizon,
                                    seed, transform_dict=PER_SERIES_NORM_TRANSFORM,
                                )
                        else:
                            pred = run_model_forecast(
                                call_name, params, train, horizon, seed
                            )
                        elapsed = time.perf_counter() - start
                        row = evaluate_prediction(
                            pred, actual, train, season_m, spec, pairs
                        )
                        row["error"] = None
                    except Exception as exc:
                        elapsed = time.perf_counter() - start
                        row = {
                            k: np.nan
                            for k in (
                                "smape",
                                "mase",
                                "spl",
                                "containment",
                                "dca_error",
                                "agg_error",
                            )
                        }
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        if verbose:
                            traceback.print_exc()
                    row.update(
                        {
                            "dataset": ds_name,
                            "fold": fold,
                            "horizon": horizon,
                            "model": model_name,
                            "fit_sec": round(elapsed, 3),
                        }
                    )
                    rows.append(row)
                    if verbose:
                        status = (
                            f"smape={row['smape']:.2f} mase={row['mase']:.3f}"
                            if row["error"] is None
                            else f"FAILED ({row['error']})"
                        )
                        print(
                            f"[{ds_name} h={horizon} fold={fold}] "
                            f"{model_name:<28} {status}  ({elapsed:.1f}s)"
                        )
    result = pd.DataFrame(rows)
    col_order = [
        "dataset",
        "fold",
        "horizon",
        "model",
        "smape",
        "mase",
        "spl",
        "containment",
        "dca_error",
        "agg_error",
        "fit_sec",
        "error",
    ]
    # extra (tagged-run) columns keep their place at the end; on untagged runs
    # none exist, so the frame is byte-identical to before
    col_order = col_order + [c for c in result.columns if c not in col_order]
    return result[col_order]


# ---------------------------------------------------------------------------
# 0f — arbitration ceiling probe (HARNESS DIAGNOSTIC ONLY)
# ---------------------------------------------------------------------------
# This sizes the prize for a future per-series arbitration/selection layer.
# It is deliberately not a library feature and must never become one: the
# oracle chooser peeks at the held-out truth, and the honest chooser is here
# only to show how much of the oracle gap a legitimate selector can recover.


def _probe_forecast(label, train, horizon, seed, tva_params=None):
    """Run one model by benchmark label (TVAModel[x] / PooledComparator[x] /
    the detector / any model_forecast name) and return its PredictionObject."""
    if label == DETECTOR_MODEL_NAME:
        return run_detector_forecast(train, horizon)
    if label.startswith(POOLED_PREFIX):
        inner = label[len(POOLED_PREFIX) + 1 : -1]
        return run_model_forecast(
            inner, {}, train, horizon, seed, transform_dict=PER_SERIES_NORM_TRANSFORM
        )
    params = {}
    call_name = label
    if label.startswith(TVA_MODEL_NAME):
        params = dict(tva_params or {})
        if label != TVA_MODEL_NAME and label.endswith("]"):
            params["trend_network"] = label[len(TVA_MODEL_NAME) + 1 : -1]
        call_name = TVA_MODEL_NAME
    return run_model_forecast(call_name, params, train, horizon, seed)


def run_arbitration_probe(
    datasets: dict,
    model_a: str,
    model_b: str,
    seed: int = 42,
    n_folds: int = 3,
    tva_params: dict = None,
    verbose: bool = True,
) -> list:
    """Per dataset/fold/horizon: aggregate MASE of each model, of an oracle
    per-series chooser, and of an honest inner-holdout per-series chooser."""
    out = []
    for ds_name, spec in datasets.items():
        df = spec["df"]
        season_m = spec["season_m"]
        for horizon in spec["horizons"]:
            if spec.get("fixed_origins"):
                origin_iter = fixed_origins(df, horizon, spec.get("n_origins", 5))
                folds_allowed = spec.get("folds_allowed")
            else:
                origin_iter = rolling_origins(df, horizon, n_folds)
                folds_allowed = None
            for fold, train, actual in origin_iter:
                if folds_allowed is not None and fold not in folds_allowed:
                    continue
                inner_train = train.iloc[:-horizon]
                inner_val = train.iloc[-horizon:]
                if len(inner_train) < 3 * horizon:
                    if verbose:
                        print(
                            f"  [arbitration] {ds_name} h={horizon} fold={fold}: "
                            "inner split too short, skipped"
                        )
                    continue
                real, inner = {}, {}
                err = None
                for label in (model_a, model_b):
                    try:
                        pred = _probe_forecast(label, train, horizon, seed, tva_params)
                        real[label] = per_series_mase(
                            actual, pred.forecast, train, season_m
                        )
                        ipred = _probe_forecast(
                            label, inner_train, horizon, seed, tva_params
                        )
                        inner[label] = per_series_mase(
                            inner_val, ipred.forecast, inner_train, season_m
                        )
                    except Exception as exc:
                        err = f"{label}: {type(exc).__name__}: {exc}"
                        break
                if err is not None:
                    out.append(
                        {
                            "dataset": ds_name,
                            "fold": fold,
                            "horizon": horizon,
                            "model_a": model_a,
                            "model_b": model_b,
                            "error": err,
                        }
                    )
                    if verbose:
                        print(f"  [arbitration] FAILED {err}")
                    continue
                cols = [
                    c
                    for c in actual.columns
                    if c in real[model_a] and c in real[model_b]
                ]
                tags = spec.get("tags")
                if tags:
                    base_cols = [c for c in tags.get("base", []) if c in cols]
                    if base_cols:
                        cols = base_cols
                a_vals = [real[model_a][c] for c in cols]
                b_vals = [real[model_b][c] for c in cols]
                oracle = [min(real[model_a][c], real[model_b][c]) for c in cols]
                choice = {
                    c: (
                        model_a
                        if not np.isfinite(inner[model_b].get(c, np.nan))
                        or inner[model_a].get(c, np.nan) <= inner[model_b].get(c, np.nan)
                        else model_b
                    )
                    for c in cols
                }
                honest = [real[choice[c]][c] for c in cols]
                row = {
                    "dataset": ds_name,
                    "fold": fold,
                    "horizon": horizon,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_series": len(cols),
                    "mase_a": float(np.nanmean(a_vals)),
                    "mase_b": float(np.nanmean(b_vals)),
                    "mase_oracle": float(np.nanmean(oracle)),
                    "mase_honest_holdout": float(np.nanmean(honest)),
                    "pct_series_chose_a": float(
                        np.mean([choice[c] == model_a for c in cols])
                    ) if cols else np.nan,
                    "error": None,
                }
                out.append(row)
                if verbose:
                    print(
                        f"  [arbitration] {ds_name} h={horizon} fold={fold} "
                        f"{model_a}={row['mase_a']:.3f} {model_b}={row['mase_b']:.3f} "
                        f"oracle={row['mase_oracle']:.3f} "
                        f"honest={row['mase_honest_holdout']:.3f}"
                    )
    return out


def summarize(results: pd.DataFrame, reference: str = "SeasonalNaive") -> pd.DataFrame:
    """Skill summary vs a reference model, per model."""
    key = ["dataset", "fold", "horizon"]
    ref = results[results["model"] == reference].set_index(key)
    summaries = []
    for model, grp in results.groupby("model"):
        grp = grp.set_index(key)
        joined = grp.join(ref, rsuffix="_ref")
        with np.errstate(all="ignore"):
            mase_skill = joined["mase_ref"] / joined["mase"]
            smape_skill = joined["smape_ref"] / joined["smape"]
            spl_skill = joined["spl_ref"] / joined["spl"]
        valid = mase_skill.replace([np.inf, -np.inf], np.nan).dropna()
        geo = float(np.exp(np.log(valid.clip(lower=1e-9)).mean())) if len(valid) else np.nan
        smape_valid = smape_skill.replace([np.inf, -np.inf], np.nan).dropna()
        smape_geo = (
            float(np.exp(np.log(smape_valid.clip(lower=1e-9)).mean()))
            if len(smape_valid)
            else np.nan
        )
        spl_valid = spl_skill.replace([np.inf, -np.inf], np.nan).dropna()
        spl_geo = (
            float(np.exp(np.log(spl_valid.clip(lower=1e-9)).mean()))
            if len(spl_valid)
            else np.nan
        )
        summaries.append(
            {
                "model": model,
                "mase_skill_geo": geo,
                "smape_skill_geo": smape_geo,
                "spl_skill_geo": spl_geo,
                "win_rate_vs_ref": float((mase_skill > 1.0).mean()),
                "containment": float(grp["containment"].mean()),
                "dca_error": float(grp["dca_error"].mean()),
                "agg_error": float(grp["agg_error"].mean()),
                "mean_fit_sec": float(grp["fit_sec"].mean()),
                "n_failed": int(grp["error"].notna().sum()),
            }
        )
    return (
        pd.DataFrame(summaries)
        .sort_values("mase_skill_geo", ascending=False)
        .reset_index(drop=True)
    )


def compute_gate_metrics(results: pd.DataFrame, reference: str = "SeasonalNaive") -> dict:
    """Emit the scorecard's gate keys verbatim, so grading needs no re-derivation.

    The scorecard deliberately does not reconstruct gates from result rows --
    a gate whose definition lives in two places drifts. The harness computes
    them here, once, from the same rows it just scored, and writes them under
    the ``gates`` key where the scorecard's extractor finds them by name.

    Gates are computed on **base** columns only when the dataset carries series
    tags; derived-ratio and frozen-tail series are informational per the plan.
    """
    out = {}
    if results is None or results.empty or "model" not in results.columns:
        return out
    ref_rows = results[results["model"] == reference]
    if ref_rows.empty:
        return out
    tagged = [
        name for name, grp in results.groupby("dataset")
        if grp["mase_base"].notna().any()
    ] if "mase_base" in results.columns else []

    metric = "mase_base" if tagged else "mase"
    candidates = [
        m for m in results["model"].unique()
        if m != reference and str(m).startswith(TVA_MODEL_NAME)
    ]
    if not candidates:
        return out

    key = ["dataset", "fold", "horizon"]
    ref = ref_rows.set_index(key)[metric]
    best = None
    for model in candidates:
        grp = results[results["model"] == model].set_index(key)[metric]
        joined = pd.concat([grp.rename("m"), ref.rename("r")], axis=1).dropna()
        if joined.empty:
            continue
        agg = float(joined["m"].mean() / joined["r"].mean())
        if best is None or agg < best[0]:
            best = (agg, model, joined, grp)
    if best is None:
        return out
    agg, model, joined, _grp = best
    out["lrp_mase_ratio_aggregate"] = agg
    with np.errstate(all="ignore"):
        per_fold = (joined["m"] / joined["r"]).replace([np.inf, -np.inf], np.nan)
    if per_fold.notna().any():
        out["lrp_mase_ratio_per_fold_max"] = float(per_fold.max())

    # per-series ratios, pooled over folds
    if "per_series_mase" in results.columns:
        tva_rows = results[results["model"] == model]["per_series_mase"].dropna()
        ref_series = ref_rows["per_series_mase"].dropna()
        if len(tva_rows) and len(ref_series):
            tva_mean = pd.DataFrame(list(tva_rows)).mean()
            ref_mean = pd.DataFrame(list(ref_series)).mean()
            tags = {}
            if tagged:
                try:
                    from lrp_series_tags import SERIES_TAGS

                    tags = SERIES_TAGS
                except Exception:
                    tags = {}
            keep = [
                c for c in tva_mean.index
                if c in ref_mean.index and (not tags or tags.get(c) == "base")
            ]
            if keep:
                with np.errstate(all="ignore"):
                    ratio = (tva_mean[keep] / ref_mean[keep]).replace(
                        [np.inf, -np.inf], np.nan
                    ).dropna()
                if len(ratio):
                    out["lrp_mase_ratio_p90_series"] = float(
                        np.percentile(ratio.values, 90)
                    )
                    out["lrp_mase_ratio_series_max"] = float(ratio.max())

    if "dir_coh_error" in results.columns:
        tva_dce = results[results["model"] == model]["dir_coh_error"].dropna()
        ref_dce = ref_rows["dir_coh_error"].dropna()
        if len(tva_dce) and len(ref_dce) and float(ref_dce.mean()) > 0:
            out["lrp_direction_coherence_error_ratio"] = float(
                tva_dce.mean() / ref_dce.mean()
            )
    out["gate_model"] = str(model)
    return out


def _load_json_arg(text):
    """Parse a CLI argument that is either inline JSON or a path to a JSON file."""
    text = str(text).strip()
    if not text:
        return {}
    if not text.startswith("{"):
        with open(text) as handle:
            return json.load(handle)
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="TVA benchmark harness")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny 1-dataset run")
    parser.add_argument("--out", type=str, default=None, help="JSON output path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--models", type=str, default=None, help="comma-separated model filter"
    )
    parser.add_argument("--tva-epochs", type=int, default=None)
    parser.add_argument(
        "--tva-factor-config",
        type=str,
        default=None,
        help=(
            "JSON dict (or path to a .json file) merged into TVAModel's "
            "factor_config, e.g. '{\"sn_blend\": true, \"reanchor\": true}'. "
            "This is how a default-off Phase 1/2 mechanism is measured against "
            "its gate before its default is flipped."
        ),
    )
    parser.add_argument(
        "--tva-factor-arms",
        type=str,
        default=None,
        help=(
            "Semicolon-separated 'label=JSON' factor_config arms benchmarked "
            "side by side against the same folds, e.g. "
            "'off={};blend={\"sn_blend\":true}'. Implies trend_network=factor."
        ),
    )
    parser.add_argument(
        "--tva-networks",
        type=str,
        default="default",
        help=(
            "Comma-separated TVA trend_network arms to benchmark side by side "
            "(e.g. 'none,factor,v2'). 'default' uses the wrapper default."
        ),
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="wide CSV (date column + one column per series) to benchmark",
    )
    parser.add_argument(
        "--horizon", type=int, default=None, help="override dataset horizons"
    )
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="comma-separated dataset filter (also re-enables built-ins with --data)",
    )
    parser.add_argument("--origins", type=int, default=5)
    parser.add_argument("--iteration-origins", type=int, default=3)
    parser.add_argument(
        "--promotion", action="store_true",
        help="ALSO run the held-back promotion folds (default: iteration folds only)",
    )
    parser.add_argument(
        "--pooled-comparator", action="store_true",
        help="add the pooled cross-learning comparator (automatic with --data)",
    )
    parser.add_argument(
        "--arbitration-probe", action="store_true",
        help="harness-only diagnostic: oracle / honest per-series chooser ceiling",
    )
    parser.add_argument(
        "--arbitration-models", type=str,
        default=f"{TVA_MODEL_NAME}[factor],SeasonalNaive",
        help="two comma-separated benchmark labels for --arbitration-probe",
    )
    args = parser.parse_args()

    if args.data:
        from pathlib import Path

        wide = load_wide_csv(args.data)
        ds_name = Path(args.data).stem
        horizon = args.horizon or 180
        try:
            import sys as _sys

            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from lrp_series_tags import tag_columns

            tags = tag_columns(wide.columns)
        except Exception as exc:  # pragma: no cover - tags are optional
            print(f"[warn] series tags unavailable ({exc}); running untagged")
            tags = None
        folds_allowed = set(range(args.origins))
        if not args.promotion:
            folds_allowed = set(range(min(args.iteration_origins, args.origins)))
        else:
            print("\n" + "!" * 72)
            print("!! PROMOTION FOLDS ARE RUNNING — these windows are the held-back")
            print("!! evaluation set. Do NOT tune against these numbers.")
            print("!" * 72 + "\n")
        datasets = {
            ds_name: {
                "df": wide,
                "horizons": [horizon],
                "season_m": 7,
                "fixed_origins": True,
                "n_origins": args.origins,
                "folds_allowed": folds_allowed,
                "tags": tags,
                # 0d item 6: never any semantic metadata on these runs
                "series_metadata": None,
                "prior_adjacency": None,
            }
        }
        print(
            f"[{ds_name}] {wide.shape[0]} rows x {wide.shape[1]} cols, "
            f"{wide.index.min().date()} -> {wide.index.max().date()}, horizon={horizon}"
        )
        if tags:
            print(
                "  tags: "
                + ", ".join(f"{k}={len(v)}" for k, v in tags.items())
                + f" | folds: {sorted(folds_allowed)}"
                + (" (iteration only)" if not args.promotion else " (INCL. PROMOTION)")
            )
        for f, tr, ac in fixed_origins(wide, horizon, args.origins):
            mark = "iteration" if f < args.iteration_origins else "PROMOTION"
            run = "run" if f in folds_allowed else "skip"
            print(
                f"  fold {f} [{mark:9s} {run:4s}] train {tr.index.min().date()}"
                f"->{tr.index.max().date()} ({len(tr)}) test {ac.index.min().date()}"
                f"->{ac.index.max().date()} ({len(ac)})"
            )
        if args.datasets:
            datasets.update(build_datasets(seed=args.seed, smoke=args.smoke))
    else:
        datasets = build_datasets(seed=args.seed, smoke=args.smoke)
        if args.horizon:
            for spec in datasets.values():
                spec["horizons"] = [args.horizon]
    if args.datasets:
        wanted_ds = {d.strip() for d in args.datasets.split(",")}
        datasets = {k: v for k, v in datasets.items() if k in wanted_ds}
        if not datasets:
            raise SystemExit(f"--datasets {args.datasets} matched no dataset")

    models = list(BASELINE_MODELS) + [(DETECTOR_MODEL_NAME, {})]
    if args.data or args.pooled_comparator:
        # 0h: comparator only — never a gate
        models.append((POOLED_COMPARATOR_LABEL, {}))
    include_tva = not args.baselines_only
    if include_tva:
        for network in [n.strip() for n in args.tva_networks.split(",") if n.strip()]:
            label = (
                TVA_MODEL_NAME
                if network == "default"
                else f"{TVA_MODEL_NAME}[{network}]"
            )
            params = {} if network == "default" else {"trend_network": network}
            models.append((label, params))
    if args.tva_factor_arms:
        for chunk in args.tva_factor_arms.split(";"):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk:
                continue
            label, payload = chunk.split("=", 1)
            models.append(
                (
                    f"{TVA_MODEL_NAME}[factor:{label.strip()}]",
                    {
                        "trend_network": "factor",
                        "factor_config": _load_json_arg(payload),
                    },
                )
            )
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in models if m[0] in wanted]

    tva_params = {}
    if args.smoke:
        args.folds = 1
        tva_params = {"epochs": 5, "window_size": 60}
    if args.tva_epochs:
        tva_params["epochs"] = args.tva_epochs
    if args.tva_factor_config:
        tva_params["factor_config"] = _load_json_arg(args.tva_factor_config)

    results = run_benchmark(
        datasets,
        models,
        n_folds=args.folds,
        seed=args.seed,
        include_tva=include_tva,
        tva_params=tva_params,
    )

    arbitration = None
    if args.arbitration_probe:
        probe_models = [m.strip() for m in args.arbitration_models.split(",") if m.strip()]
        if len(probe_models) != 2:
            raise SystemExit("--arbitration-models needs exactly two labels")
        print("\n## Arbitration ceiling probe (diagnostic only, never a feature)\n")
        arbitration = run_arbitration_probe(
            datasets,
            probe_models[0],
            probe_models[1],
            seed=args.seed,
            n_folds=args.folds,
            tva_params=tva_params,
        )
        ok = [r for r in arbitration if r.get("error") is None]
        if ok:
            arb = pd.DataFrame(ok)
            print(
                arb.groupby(["dataset", "horizon"])[
                    ["mase_a", "mase_b", "mase_oracle", "mase_honest_holdout"]
                ]
                .mean()
                .round(4)
                .to_markdown()
            )
            best = arb[["mase_a", "mase_b"]].mean().min()
            orc = float(arb["mase_oracle"].mean())
            hon = float(arb["mase_honest_holdout"].mean())
            print(
                f"\nbest single model MASE={best:.4f} | oracle={orc:.4f} "
                f"({100 * (1 - orc / best):.1f}% prize) | honest holdout={hon:.4f} "
                f"({100 * (1 - hon / best):.1f}% realized)"
            )

    summary = summarize(results)
    print("\n## Per-run results (mean over folds)\n")
    pivot = results.groupby(["dataset", "model"])[["smape", "mase", "spl", "containment"]].mean()
    print(pivot.round(3).to_markdown())
    print("\n## Skill summary vs SeasonalNaive (geometric mean; >1 is better)\n")
    print(summary.round(3).to_markdown(index=False))

    if "mase_base" in results.columns:
        print("\n## Per-tag MASE (gates use mase_base only; others informational)\n")
        print(
            results.groupby(["dataset", "model"])[
                [c for c in ("mase", "mase_base", "mase_derived", "mase_frozen",
                             "dir_coh_error", "real_data_coherence")
                 if c in results.columns]
            ].mean().round(4).to_markdown()
        )

    # 0h gate-calibration check, on the iteration folds only
    pooled = results[results["model"].str.startswith(POOLED_PREFIX)]
    if len(pooled):
        col = "mase_base" if "mase_base" in results.columns else "mase"
        it = results[results["fold"] < args.iteration_origins]
        p_m = it[it["model"].str.startswith(POOLED_PREFIX)][col].mean()
        s_m = it[it["model"] == "SeasonalNaive"][col].mean()
        if np.isfinite(p_m) and np.isfinite(s_m) and s_m > 0:
            gain = 1.0 - p_m / s_m
            print(
                f"\n[pooled comparator] {col}: pooled={p_m:.4f} SeasonalNaive="
                f"{s_m:.4f} ({100 * gain:+.1f}% vs SN on iteration folds)"
            )
            if gain > 0.10:
                print("\n" + "*" * 72)
                print("*** GATE CALIBRATION WARNING: the pooled cross-learning")
                print("*** comparator beats SeasonalNaive by >10% aggregate MASE.")
                print("*** SN-parity may be the wrong bar — escalate to the user")
                print("*** before further work aims at it.")
                print("*" * 72)

    if args.out:
        cfg = {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()}
        payload = {
            "config": cfg,
            "results": json.loads(results.to_json(orient="records")),
            "summary": json.loads(summary.to_json(orient="records")),
            "gates": compute_gate_metrics(results),
            "arbitration": arbitration,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nWrote {args.out}")
    return results, summary


if __name__ == "__main__":
    main()
