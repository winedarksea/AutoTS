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


def run_model_forecast(model_name, params, train, horizon, seed):
    from autots.evaluator.auto_model import model_forecast

    return model_forecast(
        model_name=model_name,
        model_param_dict=params,
        model_transform_dict=FILL_ONLY_TRANSFORM,
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


def evaluate_prediction(pred, actual, train, season_m, spec, pairs):
    """Compute the harness metric row from a PredictionObject."""
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
    return row


# ---------------------------------------------------------------------------
# Benchmark protocol
# ---------------------------------------------------------------------------


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
        for horizon in spec["horizons"]:
            for fold, train, actual in rolling_origins(df, horizon, n_folds):
                pairs = correlated_pairs(train, season_m)
                for model_name, params in models:
                    if model_name == TVA_MODEL_NAME:
                        if not include_tva or not HAS_TORCH:
                            continue
                        params = dict(tva_params or {})
                    start = time.perf_counter()
                    try:
                        if model_name == DETECTOR_MODEL_NAME:
                            pred = run_detector_forecast(train, horizon)
                        else:
                            pred = run_model_forecast(
                                model_name, params, train, horizon, seed
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
    return result[col_order]


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
    args = parser.parse_args()

    datasets = build_datasets(seed=args.seed, smoke=args.smoke)
    models = list(BASELINE_MODELS) + [(DETECTOR_MODEL_NAME, {})]
    include_tva = not args.baselines_only
    if include_tva:
        models.append((TVA_MODEL_NAME, {}))
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in models if m[0] in wanted]

    tva_params = {}
    if args.smoke:
        args.folds = 1
        tva_params = {"epochs": 5, "window_size": 60}
    if args.tva_epochs:
        tva_params["epochs"] = args.tva_epochs

    results = run_benchmark(
        datasets,
        models,
        n_folds=args.folds,
        seed=args.seed,
        include_tva=include_tva,
        tva_params=tva_params,
    )

    summary = summarize(results)
    print("\n## Per-run results (mean over folds)\n")
    pivot = results.groupby(["dataset", "model"])[["smape", "mase", "spl", "containment"]].mean()
    print(pivot.round(3).to_markdown())
    print("\n## Skill summary vs SeasonalNaive (geometric mean; >1 is better)\n")
    print(summary.round(3).to_markdown(index=False))

    if args.out:
        payload = {
            "config": vars(args),
            "results": json.loads(results.to_json(orient="records")),
            "summary": json.loads(summary.to_json(orient="records")),
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nWrote {args.out}")
    return results, summary


if __name__ == "__main__":
    main()
