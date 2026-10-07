"""
Evaluate TimeSeriesFeatureDetector.forecast trend options by rolling-origin backtest.

Arms (one detector fit per origin, three forecasts from it):
- baseline:  trend_change_variance=False, slope_shrinkage=False (pre-change behaviour)
- variance:  trend_change_variance=True
- shrink:    trend_change_variance=True, slope_shrinkage=True
- damp<phi>: shrink + trend_damping=phi (per period)

Reports per horizon bucket: 90% interval coverage, scaled Winkler interval score,
and point MASE / sMAPE. Also checks the premise behind the variance term on the
synthetic panel: detected rate * E[slope_change^2] vs the generator's truth.

Usage:
  python examples/detector_trend_forecast_eval.py --datasets synthetic,daily,weekly
  # long-horizon damping check (the LRP review's gains were largest at 1-2 years)
  python examples/detector_trend_forecast_eval.py --datasets daily --horizon 365 \
      --skip-premise --log
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from autots.datasets import load_daily, load_weekly
from autots.datasets.synthetic import SyntheticDailyGenerator
from autots.evaluator.feature_detector import TimeSeriesFeatureDetector
from autots.evaluator.feature_detector.utils.trend_forecast import trend_change_rates

ARMS = {
    'baseline': dict(trend_change_variance=False, slope_shrinkage=False),
    'variance': dict(trend_change_variance=True, slope_shrinkage=False),
    'shrink': dict(trend_change_variance=True, slope_shrinkage=True),
}
DAMPING_GRID = (0.998, 0.995, 0.99)


def set_damping_grid(grid):
    """(Re)build the damp<phi> arms; grid is per period (step)."""
    global DAMPING_GRID
    for arm in [a for a in ARMS if a.startswith('damp')]:
        del ARMS[arm]
    DAMPING_GRID = tuple(grid)
    for phi in DAMPING_GRID:
        ARMS[f'damp{phi}'] = dict(
            trend_change_variance=True, slope_shrinkage=True, trend_damping=phi
        )


set_damping_grid(DAMPING_GRID)
ALPHA = 0.1


def make_synthetic(seed, n_days=1460, n_series=12):
    gen = SyntheticDailyGenerator(n_days=n_days, n_series=n_series, random_seed=seed)
    return gen.get_data(), gen


def load_dataset(name, seed, horizon=None):
    if name == 'synthetic':
        out = make_synthetic(seed)[0], 90, 5, 60
    elif name == 'daily':
        df = load_daily(long=False)
        df = df.loc[:, df.notna().mean() > 0.9]
        out = df, 90, 5, 60
    elif name == 'weekly':
        df = load_weekly(long=False)
        df = df.loc[:, df.notna().mean() > 0.9]
        out = df, 26, 5, 13
    else:
        raise ValueError(name)
    if horizon is not None:
        out = (out[0], int(horizon)) + out[2:]
    return out


def horizon_buckets(horizon):
    edges = sorted({min(e, horizon) for e in (7, 30, 90, 182, 365)} | {horizon})
    buckets, lo = [], 0
    for hi in edges:
        if hi > lo:
            buckets.append((lo, hi))
            lo = hi
    return buckets


def score_origin(train, actual, pred):
    """Per-step-per-series scaled errors; scale = in-sample naive MAE."""
    scale = train.diff().abs().mean().replace(0, np.nan).to_numpy()
    y = actual.to_numpy()
    f = pred.forecast.reindex(actual.index).to_numpy()
    lo = pred.lower_forecast.reindex(actual.index).to_numpy()
    hi = pred.upper_forecast.reindex(actual.index).to_numpy()
    covered = ((y >= lo) & (y <= hi)).astype(float)
    winkler = (hi - lo) + (2 / ALPHA) * (
        np.maximum(lo - y, 0) + np.maximum(y - hi, 0)
    )
    smape = 200 * np.abs(f - y) / (np.abs(f) + np.abs(y)).clip(min=1e-9)
    valid = np.isfinite(y)
    nan = np.where(valid, 1.0, np.nan)
    return {
        'coverage': covered * nan,
        'winkler': winkler / scale * nan,
        'mase': np.abs(f - y) / scale * nan,
        'smape': smape * nan,
    }


def run_dataset(name, seed, horizon=None, log=False, detector_kwargs=None):
    df, horizon, n_origins, spacing = load_dataset(name, seed, horizon)
    df = df.astype(float)
    if log:
        # log space, as the LRP review ran it; scores are then in log units
        df = np.log1p(df.clip(lower=0))
    origins = [len(df) - horizon - k * spacing for k in range(n_origins)]
    buckets = horizon_buckets(horizon)
    rows = []
    for origin in origins:
        train, actual = df.iloc[:origin], df.iloc[origin : origin + horizon]
        det = TimeSeriesFeatureDetector(**(detector_kwargs or {}))
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            det.fit(train)
        fit_s = time.time() - t0
        for arm, kwargs in ARMS.items():
            pred = det.forecast(horizon, prediction_interval=1 - ALPHA, **kwargs)
            scores = score_origin(train, actual, pred)
            for lo, hi in buckets:
                for metric, arr in scores.items():
                    vals = arr[lo:hi]
                    rows.append(
                        dict(
                            dataset=name,
                            origin=origin,
                            arm=arm,
                            bucket=f'{lo + 1}-{hi}',
                            metric=metric,
                            value=float(np.nanmean(vals)),
                            # per-series means kept for regression counting
                            per_series=np.nanmean(vals, axis=0),
                        )
                    )
        print(f'  {name} origin={origin} fit={fit_s:.1f}s', flush=True)
    return pd.DataFrame(rows)


def summarize(results):
    table = (
        results.groupby(['dataset', 'bucket', 'metric', 'arm'])['value']
        .mean()
        .unstack('arm')[list(ARMS)]
    )
    print('\n=== Mean metric by dataset / horizon bucket ===')
    with pd.option_context('display.width', 160, 'display.max_rows', 200):
        print(table.round(4))
    # Per-series point regressions from shrinkage (MASE worse by > 10%).
    print('\n=== Shrink vs variance: per-series MASE (averaged over origins) ===')
    for (ds, bucket), grp in results[results.metric == 'mase'].groupby(
        ['dataset', 'bucket']
    ):
        per = {
            arm: np.nanmean(np.vstack(g['per_series'].to_numpy()), axis=0)
            for arm, g in grp.groupby('arm')
        }
        ratio = per['shrink'] / per['variance']
        ratio = ratio[np.isfinite(ratio)]
        print(
            f'  {ds:9s} h{bucket:7s} median ratio={np.median(ratio):.3f} '
            f'better={np.mean(ratio < 0.999):.2f} worse>10%={np.mean(ratio > 1.1):.2f}'
        )
    print('\n=== Damping vs shrink: MASE ratio (mean-MASE ratio, paired over origins) ===')
    mase = results[results.metric == 'mase']
    for (ds, bucket), grp in mase.groupby(['dataset', 'bucket'], sort=False):
        by_arm = grp.pivot_table(index='origin', columns='arm', values='value')
        base = by_arm['shrink']
        parts = []
        for phi in DAMPING_GRID:
            arm = f'damp{phi}'
            per_origin = by_arm[arm] / base
            parts.append(
                f'{phi}: {by_arm[arm].mean() / base.mean():.3f} '
                f'[{per_origin.min():.3f},{per_origin.max():.3f}]'
            )
        print(f'  {ds:9s} h{bucket:8s} ' + '  '.join(parts))


def premise_check(seed):
    """Detected rate*E[delta^2] (and level-shift analogue) vs generator truth."""
    df, gen = make_synthetic(seed)
    det = TimeSeriesFeatureDetector()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        det.fit(df)
    cols = list(df.columns)
    n = len(df)
    detected = trend_change_rates(det.trend_changepoints, det.level_shifts, cols, n)
    # Saturating-trend series label phases with strings; keep numeric slope pairs.
    true_cp = {
        c: [cp for cp in gen.get_trend_changepoints(c) if not isinstance(cp[1], str)]
        for c in cols
    }
    true_ls = {c: gen.get_level_shifts(c) for c in cols}
    truth = trend_change_rates(true_cp, true_ls, cols, n)
    out = pd.DataFrame(
        {
            'n_cp_true': truth['rate_cp'] * n,
            'n_cp_det': detected['rate_cp'] * n,
            'Ed2_true': truth['mean_sq_slope_change'],
            'Ed2_det': detected['mean_sq_slope_change'],
            'cp_prod_ratio': (detected['rate_cp'] * detected['mean_sq_slope_change'])
            / (truth['rate_cp'] * truth['mean_sq_slope_change']),
            'n_ls_true': truth['rate_ls'] * n,
            'n_ls_det': detected['rate_ls'] * n,
            'ls_prod_ratio': (detected['rate_ls'] * detected['mean_sq_level_shift'])
            / (truth['rate_ls'] * truth['mean_sq_level_shift']),
        },
        index=cols,
    )
    print('\n=== Premise: detected vs true change intensity (synthetic) ===')
    with pd.option_context('display.width', 160):
        print(out.round(5))
        ratios = out[['cp_prod_ratio', 'ls_prod_ratio']].replace(
            [np.inf, -np.inf], np.nan
        )
        print('median ratios:', ratios.median().round(3).to_dict())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default='synthetic,daily,weekly')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--skip-premise', action='store_true')
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--log', action='store_true')
    parser.add_argument(
        '--joint', action='store_true', help="changepoint method 'joint_hinge_step'"
    )
    parser.add_argument(
        '--damping-grid', type=str, default=None, help='comma-separated phi values'
    )
    args = parser.parse_args()
    if args.damping_grid:
        set_damping_grid(float(v) for v in args.damping_grid.split(','))
    if not args.skip_premise:
        premise_check(args.seed)
    detector_kwargs = (
        {'changepoint_params': {'method': 'joint_hinge_step'}}
        if args.joint
        else None
    )
    results = pd.concat(
        [
            run_dataset(name, args.seed, args.horizon, args.log, detector_kwargs)
            for name in args.datasets.split(',')
        ],
        ignore_index=True,
    )
    summarize(results)


if __name__ == '__main__':
    main()
