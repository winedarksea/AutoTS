#!/usr/bin/env python
"""Gate the structural-covariance MinT weighting against the identity one.

No benchmark panel carries ``hierarchy_path``, so this builds one: a latent
trend-factor panel from :class:`SyntheticDailyGenerator`, with series grouped by
their *true* dominant factor into mid-level nodes under a ``global`` root. That
makes the cross-series covariance non-diagonal by construction — the condition
under which a structural ``W`` should beat the identity — and it is a fair test
because it can also fail.

Three arms on the same fits:

``unreconciled``
    the bottom-level TVA forecast alongside an independently-produced
    aggregate-level forecast; incoherent by construction.
``mint_identity``
    today's behaviour — ``reconciliation_covariance='identity'``, i.e. ``W = I``.
``mint_structural``
    ``reconciliation_covariance='structural'``: ``W = S Sigma S' + diag(psi_agg)``
    with ``Sigma`` from :meth:`TVA.forecast_covariance`.

Why the aggregate arm has to be forecast independently
------------------------------------------------------
``TVA.reconcile`` normally *synthesizes* the aggregate rows as ``S @ bottom``,
which puts its input exactly in the coherent subspace MinT projects onto — so
reconciliation returns it unchanged for every ``W``, in every trend mode. And
even given incoherent input, ``W = S Sigma S'`` alone reduces MinT's estimator
to ``(S'S)^-1 S'``, the same OLS reconciler ``W = I`` gives: Sigma cancels. It
survives only once the aggregate nodes carry error of their own. Both facts are
covered by tests in ``tests/test_tva_covariance.py``. This harness therefore
fits an independent damped-trend model on the aggregate history and passes its
backtested sigma as ``aggregate_sigma``, which is the only configuration in
which the three arms are actually distinguishable.

Usage
-----
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python examples/tva_reconciliation_gate.py \\
        --seeds 0 1 2 --out examples/tva_reconciliation_gate.json

    python examples/tva_scorecard.py --results examples/tva_reconciliation_gate.json

Exit code is nonzero when a graded gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Gate thresholds (mirrored in examples/tva_scorecard.py)
# ---------------------------------------------------------------------------

GATES = {
    "reconciliation_mase_ratio_aggregate": {
        "description": "structural-W MASE / identity-W MASE, all hierarchy nodes",
        "threshold": 1.00,
        "direction": "<=",
    },
    "reconciliation_coherence_error_ratio": {
        # Measured against the *unreconciled* arm, not the identity-W arm. Both
        # MinT arms return exactly coherent forecasts by construction — the
        # projection is what MinT is — so a structural-vs-identity coherence
        # ratio is 0/0 and says nothing. What "must improve coherence error"
        # can mean here is that reconciliation removes the incoherence the
        # independent aggregate forecast introduced, which this reads directly.
        "description": "structural-W coherence error / unreconciled (must improve)",
        "threshold": 0.999,
        "direction": "<=",
    },
}


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------


def build_panel(seed: int, n_days: int, n_series: int, n_factors: int):
    """Return ``(df, metadata, S, agg_names)`` for a factor-grouped hierarchy.

    ``get_true_factors()['dominant_factor']`` is a scoring oracle, never a model
    input — it is used here only to *define* the hierarchy, which is domain
    knowledge a real user would supply from their own org chart.
    """
    from autots.datasets import generate_synthetic_daily_data
    from autots.evaluator.tva.priors import SeriesMetadata

    gen = generate_synthetic_daily_data(
        n_days=n_days,
        n_series=n_series,
        random_seed=seed,
        noise_level=0.05,
        trend_changepoint_freq=2.0,
        series_type_override="standard",
        n_latent_factors=n_factors,
        factor_strength=0.8,
        factor_response_lag_max=0,
    )
    df = gen.get_data()
    truth = gen.get_true_factors()
    if not truth:
        raise RuntimeError("generator produced no latent factors")
    dominant = truth["dominant_factor"]

    metadata = [
        SeriesMetadata(
            name,
            hierarchy_path=["global", f"factor_{int(dominant.get(name, 0))}", name],
            history_periods=int(df[name].notna().sum()),
        )
        for name in df.columns
    ]

    # the aggregate node order build_hierarchy_matrix uses: (depth, tuple)
    nodes = set()
    for m in metadata:
        for depth in range(1, len(m.hierarchy_path)):
            nodes.add(tuple(m.hierarchy_path[:depth]))
    agg_names = ["/".join(n) for n in sorted(nodes, key=lambda x: (len(x), x))]
    return df, metadata, agg_names


# ---------------------------------------------------------------------------
# Independent aggregate-level forecaster
# ---------------------------------------------------------------------------


def damped_trend_forecast(history: np.ndarray, horizon: int, window: int = 90,
                          phi: float = 0.9) -> np.ndarray:
    """(H, N) damped local-linear continuation — the ``_predict_numpy`` rule.

    Deliberately a *different* model from the bottom-level TVA fit: MinT has
    something to do only when the levels disagree.
    """
    L = max(min(int(window), history.shape[0]), 2)
    x = history[-L:]
    t = np.arange(L) - (L - 1) / 2.0
    slope = (x * t[:, None]).sum(axis=0) / max(float((t**2).sum()), 1e-8)
    damp = np.cumsum(phi ** np.arange(1, horizon + 1))[:, None]
    return history[-1][None, :] + damp * slope[None, :]


def backtest_sigma(history: np.ndarray, horizon: int, window: int = 90,
                   n_origins: int = 60) -> np.ndarray:
    """(N,) rolling-origin forecast-error sigma of ``damped_trend_forecast``."""
    T = history.shape[0]
    lo = max(window, 2)
    hi = T - horizon
    if hi <= lo:
        return np.full(history.shape[1], np.nan)
    origins = np.arange(lo, hi + 1)
    if origins.size > n_origins:
        origins = origins[np.linspace(0, origins.size - 1, n_origins).astype(int)]
    errs = [
        damped_trend_forecast(history[:o], horizon, window) - history[o : o + horizon]
        for o in origins
    ]
    return np.concatenate(errs, axis=0).std(axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def seasonal_naive_scale(train: np.ndarray, m: int) -> np.ndarray:
    """(N,) in-sample seasonal-naive MAE, the MASE denominator."""
    if train.shape[0] > m:
        scale = np.nanmean(np.abs(train[m:] - train[:-m]), axis=0)
    else:
        scale = np.nanmean(np.abs(np.diff(train, axis=0)), axis=0)
    return np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.nan)


def mase(actual: np.ndarray, forecast: np.ndarray, scale: np.ndarray) -> float:
    mae = np.nanmean(np.abs(actual - forecast), axis=0)
    return float(np.nanmean(mae / scale))


def arm_metrics(full_fc: np.ndarray, actual_all: np.ndarray, S: np.ndarray,
                scale_all: np.ndarray, n_agg: int) -> dict:
    """MASE at every level plus the two coherence readings the plan asks for."""
    bottom = full_fc[:, n_agg:]
    implied = (S[:n_agg] @ bottom.T).T  # what the bottom level says the aggregates are
    gap = full_fc[:, :n_agg] - implied  # S.b - a, signed
    return {
        "mase_all": mase(actual_all, full_fc, scale_all),
        "mase_bottom": mase(actual_all[:, n_agg:], bottom, scale_all[n_agg:]),
        "mase_aggregate": mase(actual_all[:, :n_agg], full_fc[:, :n_agg], scale_all[:n_agg]),
        # coherence error, scale-free: the incoherence in MASE-comparable units
        "coherence_error": float(np.nanmean(np.abs(gap) / scale_all[:n_agg])),
        # raw ||S.b - a||, the plan's aggregate-consistency reading
        "aggregate_consistency": float(np.linalg.norm(gap)),
    }


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


def run_seed(seed: int, horizon: int, n_days: int, n_series: int,
             n_factors: int, season_m: int, verbose: int = 0) -> dict:
    from autots.evaluator.tva.tva import TVA

    df, metadata, agg_names = build_panel(seed, n_days, n_series, n_factors)
    train, test = df.iloc[:-horizon], df.iloc[-horizon:]

    started = time.time()
    tva = TVA(
        trend_network="factor",
        series_metadata=metadata,
        forecast_horizon=horizon,
        window_size=91,
        verbose=verbose,
        random_seed=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tva.fit(train)
    fit_seconds = time.time() - started

    S = tva._priors.build_hierarchy_matrix().astype(np.float64)
    n_bottom = S.shape[1]
    n_agg = S.shape[0] - n_bottom
    if n_agg <= 0:
        raise RuntimeError("panel produced a flat hierarchy; nothing to reconcile")

    bottom_fc = tva.predict(horizon)

    # independent aggregate-level base forecast + its own error sigma
    agg_hist = (S[:n_agg] @ train.values.T).T
    agg_fc = damped_trend_forecast(agg_hist, horizon)
    agg_sigma = backtest_sigma(agg_hist, horizon)

    columns = list(agg_names) + list(df.columns)
    full = pd.DataFrame(
        np.concatenate([agg_fc, bottom_fc.values], axis=1),
        index=bottom_fc.index,
        columns=columns,
    )

    actual_all = np.concatenate(
        [(S[:n_agg] @ test.values.T).T, test.values], axis=1
    )
    train_all = np.concatenate([agg_hist, train.values], axis=1)
    scale_all = seasonal_naive_scale(train_all, season_m)

    arms = {"unreconciled": full.values}
    for name, mode, agg_arg in (
        ("mint_identity", "identity", None),
        ("mint_structural", "structural", agg_sigma),
    ):
        tva.reconciliation_covariance = mode
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = tva.reconcile(forecasts=full, aggregate_sigma=agg_arg)
        finally:
            tva.reconciliation_covariance = "auto"
        arms[name] = out[columns].values

    # Control: the same W with its off-diagonals deleted. If this matches the
    # structural arm, the cross-series covariance is inert and the win is
    # ordinary variance weighting (trust the level whose model is better) —
    # which would be worth knowing before crediting Sigma for it.
    from autots.evaluator.tva.reconciliation import ReconciliationBridge

    tva.reconciliation_covariance = "structural"
    try:
        W_full = tva._structural_reconciliation_W(S, aggregate_sigma=agg_sigma)
    finally:
        tva.reconciliation_covariance = "auto"
    if W_full is not None:
        arms["mint_variance_only"] = (
            ReconciliationBridge(method="mint")
            .reconcile(full, S, W=np.diag(np.diag(W_full)))[columns]
            .values
        )

    metrics = {
        name: arm_metrics(values, actual_all, S, scale_all, n_agg)
        for name, values in arms.items()
    }

    cov = tva.forecast_covariance(horizon)
    if cov is None:
        cov_diag = {"available": False}
    else:
        sigma, info = cov
        d = np.sqrt(np.diag(sigma))
        corr = sigma / np.outer(d, d)
        off = corr[~np.eye(len(d), dtype=bool)]
        cov_diag = {
            "available": True,
            "alpha": info["alpha"],
            "beta": info["beta"],
            "n_samples": info["n_samples"],
            "floor_binding_frac": float(np.mean(info["floor_binding"])),
            "mean_abs_offdiag_corr": float(np.abs(off).mean()),
            "max_abs_offdiag_corr": float(np.abs(off).max()),
            "min_eigenvalue": float(np.linalg.eigvalsh(sigma).min()),
        }

    ident, struct = metrics["mint_identity"], metrics["mint_structural"]
    return {
        "seed": seed,
        "n_series": n_bottom,
        "n_aggregate_nodes": n_agg,
        "horizon": horizon,
        "fit_seconds": round(fit_seconds, 1),
        "arms": metrics,
        "covariance": cov_diag,
        "mase_ratio": _ratio(struct["mase_all"], ident["mase_all"]),
        "coherence_error_ratio": _ratio(
            struct["coherence_error"], metrics["unreconciled"]["coherence_error"]
        ),
    }


def _ratio(numerator, denominator):
    if denominator is None or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return None
    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


# ---------------------------------------------------------------------------
# Aggregation + grading
# ---------------------------------------------------------------------------


ARMS = ("unreconciled", "mint_identity", "mint_structural", "mint_variance_only")


def summarize(rows: list) -> dict:
    def arm_mean(arm, key):
        vals = [r["arms"][arm][key] for r in rows if np.isfinite(r["arms"][arm][key])]
        return float(np.mean(vals)) if vals else None

    summary = {}
    for arm in ARMS:
        if not all(arm in r["arms"] for r in rows):
            continue
        summary[arm] = {
            key: arm_mean(arm, key)
            for key in (
                "mase_all",
                "mase_bottom",
                "mase_aggregate",
                "coherence_error",
                "aggregate_consistency",
            )
        }
    summary["reconciliation_mase_ratio_aggregate"] = _ratio(
        summary["mint_structural"]["mase_all"], summary["mint_identity"]["mase_all"]
    )
    summary["reconciliation_coherence_error_ratio"] = _ratio(
        summary["mint_structural"]["coherence_error"],
        summary["unreconciled"]["coherence_error"],
    )
    # diagnostic, not a gate: both MinT arms are exactly coherent, so this is
    # the reading that shows the coherence gate cannot discriminate between them
    summary["structural_vs_identity_coherence_gap"] = abs(
        (summary["mint_structural"]["coherence_error"] or 0.0)
        - (summary["mint_identity"]["coherence_error"] or 0.0)
    )
    if "mint_variance_only" in summary:
        # diagnostic, not a gate: 1.0 means Sigma's off-diagonals bought nothing
        summary["structural_vs_variance_only_mase_ratio"] = _ratio(
            summary["mint_structural"]["mase_all"],
            summary["mint_variance_only"]["mase_all"],
        )
    per_seed = [r["mase_ratio"] for r in rows if r["mase_ratio"] is not None]
    summary["reconciliation_mase_ratio_per_seed_max"] = (
        max(per_seed) if per_seed else None
    )
    summary["n_seeds"] = len(rows)
    return summary


def grade(summary: dict) -> dict:
    rows = {}
    for name, gate in GATES.items():
        value = summary.get(name)
        if value is None:
            status = "SKIP"
        elif gate["direction"] == "<=":
            status = "PASS" if value <= gate["threshold"] else "FAIL"
        else:
            status = "PASS" if value >= gate["threshold"] else "FAIL"
        rows[name] = dict(gate, value=value, status=status)
    return rows


def render(rows: dict, summary: dict) -> str:
    lines = [
        "| arm | MASE (all) | MASE (bottom) | MASE (agg) | coherence err | ||S.b - a|| |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    for arm in ARMS:
        a = summary.get(arm)
        if a is None:
            continue
        lines.append(
            f"| {arm} | {fmt(a['mase_all'])} | {fmt(a['mase_bottom'])} | "
            f"{fmt(a['mase_aggregate'])} | {fmt(a['coherence_error'])} | "
            f"{fmt(a['aggregate_consistency'])} |"
        )
    lines += ["", "| gate | value | threshold | direction | status |",
              "| --- | --- | --- | --- | --- |"]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {fmt(row['value'])} | {row['threshold']} | "
            f"{row['direction']} | {row['status']} |"
        )
    worst = summary.get("reconciliation_mase_ratio_per_seed_max")
    lines += ["", f"seeds: {summary['n_seeds']}   worst per-seed MASE ratio: {fmt(worst)}"]
    control = summary.get("structural_vs_variance_only_mase_ratio")
    if control is not None:
        lines.append(
            f"control — structural / variance-only-W MASE: {fmt(control)} "
            "(1.0 = Sigma's off-diagonals bought nothing)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--n-days", type=int, default=1095)
    parser.add_argument("--n-series", type=int, default=24)
    parser.add_argument("--n-factors", type=int, default=3)
    parser.add_argument("--season-m", type=int, default=7)
    parser.add_argument("--smoke", action="store_true",
                        help="short panel, one seed — wiring check only")
    parser.add_argument("--out", default="examples/tva_reconciliation_gate.json")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)

    seeds = [args.seeds[0]] if args.smoke else args.seeds
    n_days = 420 if args.smoke else args.n_days

    rows = []
    for seed in seeds:
        print(f"[seed {seed}] fitting...", flush=True)
        rows.append(
            run_seed(
                seed,
                args.horizon,
                n_days,
                args.n_series,
                args.n_factors,
                args.season_m,
                verbose=args.verbose,
            )
        )
        last = rows[-1]
        print(
            f"[seed {seed}] MASE ratio {last['mase_ratio']}, "
            f"coherence ratio {last['coherence_error_ratio']} "
            f"({last['fit_seconds']}s)",
            flush=True,
        )

    summary = summarize(rows)
    graded = grade(summary)
    payload = {
        "config": vars(args) | {"seeds": seeds, "n_days": n_days},
        "seeds": rows,
        "summary": summary,
        "gates": graded,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)

    print()
    print(render(graded, summary))
    print(f"\nwrote {args.out}")
    failed = [n for n, r in graded.items() if r["status"] == "FAIL"]
    if failed:
        print(f"FAILED gates: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
