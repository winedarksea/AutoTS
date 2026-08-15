# -*- coding: utf-8 -*-
"""
TVA ablations — is the graph load-bearing, and does the network earn its keep?

Graph ablation (per seed): the discovered edge set is replaced before network
training with one of several controls:
  - discovered: the real discovery output
  - none:       A = 0 (channel-independent baseline — expect it to be strong)
  - random:     random topology with matched edge count and weight distribution
  - shuffled:   sources and targets independently permuted — preserves degree,
                destroys pairing; the sharpest control
  - transposed: direction reversed — does direction carry information at all

CLAIM RULE (printed at the end): only claim the graph is load-bearing if
`discovered` beats `shuffled` AND `transposed` by more than the seed spread.

Component ablations: structure learning off, priors off, additive fusion,
local_trend 0, coherence in {0, 0.25, 1.0}, and Arm A vs Arm C:
  - Arm A: detector components + damped rolling-trend extrapolation with the
    network output forced to zero (pure numpy, torch-free)
  - Arm C: Arm A + the trained network correction (full TVA)
If Arm C ≈ Arm A, the network should be cut.

Usage:
  python examples/tva_graph_ablation.py --seeds 5 --epochs 60
  python examples/tva_graph_ablation.py --component-ablations --seeds 3
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from autots.datasets.synthetic import make_svar_panel


def mase_value(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray, m: int = 7) -> float:
    mae = np.nanmean(np.abs(actual - forecast), axis=0)
    if train.shape[0] > m:
        scale = np.nanmean(np.abs(train[m:] - train[:-m]), axis=0)
    else:
        scale = np.nanmean(np.abs(np.diff(train, axis=0)), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.nan)
    return float(np.nanmean(mae / scale))


def transform_edges(edges: list, mode: str, n_series: int, rng: np.random.Generator) -> list:
    """Return the control edge set for a graph-ablation mode."""
    if mode == 'discovered':
        return edges
    if mode == 'none':
        return []
    if mode == 'transposed':
        return [
            {**e, 'source': e['target'], 'target': e['source']} for e in edges
        ]
    if mode == 'shuffled':
        # permute sources and targets independently: degree sequence kept,
        # pairing destroyed
        src_perm = rng.permutation(n_series)
        dst_perm = rng.permutation(n_series)
        out = []
        for e in edges:
            s = int(src_perm[e['source']])
            t = int(dst_perm[e['target']])
            if s == t:
                t = (t + 1) % n_series
            out.append({**e, 'source': s, 'target': t})
        return out
    if mode == 'random':
        out = []
        for e in edges:
            s = int(rng.integers(0, n_series))
            t = int(rng.integers(0, n_series))
            if s == t:
                t = (t + 1) % n_series
            out.append({**e, 'source': s, 'target': t})
        return out
    raise ValueError(f"unknown mode {mode}")


def fit_predict_tva(train_df, horizon, seed, epochs, mode=None, tva_kwargs=None):
    """Fit TVA (optionally with a transformed edge set) and forecast."""
    from autots.evaluator.tva.tva import TVA

    kwargs = dict(
        epochs=epochs,
        window_size=60,
        forecast_horizon=horizon,
        verbose=0,
        random_seed=seed,
    )
    kwargs.update(tva_kwargs or {})
    tva = TVA(**kwargs)

    if mode is not None and mode != 'discovered':
        rng = np.random.default_rng(seed)
        original_builder = tva._build_network_discovery_inputs

        def patched_builder():
            built = original_builder()
            if built is not None:
                built['edges'] = transform_edges(
                    built['edges'], mode, train_df.shape[1], rng
                )
            return built

        tva._build_network_discovery_inputs = patched_builder

    tva.fit(train_df)
    return tva.predict().values


def arm_a_forecast(train_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Arm A: detector decomposition + damped rolling-trend extrapolation.

    Pure numpy on top of the decomposer — the network output forced to zero.
    """
    from autots.evaluator.tva.decomposition import NornDecomposer

    decomposer = NornDecomposer()
    decomposer.fit(train_df.ffill().bfill())
    trend = decomposer.get_components()['trend'].values
    T, N = trend.shape
    L = max(min(T, 4 * horizon), 2)
    x = trend[-L:]
    t_idx = np.arange(L) - (L - 1) / 2.0
    denom = float(np.sum(t_idx**2)) or 1.0
    slope = (x * t_idx[:, None]).sum(axis=0) / denom
    phi = 0.9
    damp = np.cumsum(phi ** np.arange(1, horizon + 1))
    trend_fc = trend[-1][None, :] + damp[:, None] * slope[None, :]

    comps = decomposer.get_forecast_components(horizon)
    return (
        trend_fc
        + comps['seasonality'].values
        + comps['holidays'].values
        + comps['level_shifts'].values
    )


def run_arm(name, forecaster, df, horizon, n_folds, rows):
    T = len(df)
    for fold in range(n_folds):
        cut = T - horizon * (n_folds - fold)
        train, actual = df.iloc[:cut], df.iloc[cut : cut + horizon]
        start = time.perf_counter()
        try:
            fc = forecaster(train, actual)
            mase = mase_value(actual.values, fc, train.values)
            err = None
        except Exception as exc:
            mase, err = np.nan, f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                'arm': name,
                'fold': fold,
                'mase': mase,
                'sec': round(time.perf_counter() - start, 2),
                'error': err,
            }
        )
        print(f"  [{name} fold={fold}] mase={mase:.3f}" if err is None else f"  [{name} fold={fold}] FAILED {err}")


def main():
    parser = argparse.ArgumentParser(description="TVA graph/component ablations")
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--folds', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=14)
    parser.add_argument('--n-series', type=int, default=12)
    parser.add_argument('--n-obs', type=int, default=600)
    parser.add_argument('--modes', type=str,
                        default='discovered,none,random,shuffled,transposed')
    parser.add_argument('--component-ablations', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    df, *_ = make_svar_panel(
        n_series=args.n_series, n_obs=args.n_obs, n_factors=2,
        edge_density=0.2, seed=11,
    )
    horizon = args.horizon
    T = len(df)

    # ---- graph ablation ----
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    rows = []
    for mode in modes:
        for seed in range(1, args.seeds + 1):
            for fold in range(args.folds):
                cut = T - horizon * (args.folds - fold)
                train, actual = df.iloc[:cut], df.iloc[cut : cut + horizon]
                start = time.perf_counter()
                try:
                    fc = fit_predict_tva(train, horizon, seed, args.epochs, mode=mode)
                    mase = mase_value(actual.values, fc, train.values)
                    err = None
                except Exception as exc:
                    mase, err = np.nan, f"{type(exc).__name__}: {exc}"
                rows.append(
                    {'mode': mode, 'seed': seed, 'fold': fold, 'mase': mase,
                     'sec': round(time.perf_counter() - start, 2), 'error': err}
                )
                print(
                    f"[{mode} seed={seed} fold={fold}] "
                    + (f"mase={mase:.3f}" if err is None else f"FAILED {err}")
                )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby('mode')['mase']
        .agg(['mean', 'std', 'count'])
        .sort_values('mean')
    )
    print("\n## Graph ablation (MASE, lower is better)\n")
    print(summary.round(4).to_markdown())

    verdict = None
    if {'discovered', 'shuffled', 'transposed'} <= set(summary.index):
        disc = summary.loc['discovered']
        spread = float(results[results['mode'] == 'discovered']['mase'].std())
        beats_shuffled = (
            summary.loc['shuffled', 'mean'] - disc['mean'] > spread
        )
        beats_transposed = (
            summary.loc['transposed', 'mean'] - disc['mean'] > spread
        )
        verdict = bool(beats_shuffled and beats_transposed)
        print(
            f"\nCLAIM RULE: discovered beats shuffled by more than seed spread: "
            f"{beats_shuffled}; beats transposed: {beats_transposed}."
        )
        print(
            "=> The graph IS load-bearing." if verdict
            else "=> The graph is NOT demonstrably load-bearing on this panel."
        )

    # ---- component ablations + Arm A vs Arm C ----
    component_rows = []
    if args.component_ablations:
        print("\n## Component ablations\n")
        variants = {
            'arm_c_full_tva': {},
            'structure_off': {'structure_learning_config': {'enabled': False}},
            'priors_off': {'prior_construction_config': {}},
            'fusion_additive': {'fusion': 'additive'},
            'local_trend_0': {'loss_weights': {'local_trend': 0.0}},
            'coherence_0': {'loss_weights': {'coherence': 0.0}},
            'coherence_0.25': {'loss_weights': {'coherence': 0.25}},
            'coherence_1.0': {'loss_weights': {'coherence': 1.0}},
            'discovery_off': {'discovery_config': {'enabled': False}},
        }
        run_arm(
            'arm_a_no_network',
            lambda train, actual: arm_a_forecast(train, horizon),
            df, horizon, args.folds, component_rows,
        )
        for name, kwargs in variants.items():
            run_arm(
                name,
                lambda train, actual, kw=kwargs: fit_predict_tva(
                    train, horizon, 42, args.epochs, tva_kwargs=kw
                ),
                df, horizon, args.folds, component_rows,
            )
        comp = pd.DataFrame(component_rows)
        comp_summary = comp.groupby('arm')['mase'].agg(['mean', 'std']).sort_values('mean')
        print("\n## Component ablation summary (MASE)\n")
        print(comp_summary.round(4).to_markdown())

    if args.out:
        payload = {
            'config': vars(args),
            'graph_ablation': json.loads(results.to_json(orient='records')),
            'graph_summary': json.loads(summary.reset_index().to_json(orient='records')),
            'graph_load_bearing': verdict,
            'component_ablation': component_rows,
        }
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=1, default=str)
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
