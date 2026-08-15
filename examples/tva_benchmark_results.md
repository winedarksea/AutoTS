# TVA rework — benchmark and ablation results (2026-08-15)

Protocol: `examples/tva_benchmark.py`, rolling origin, 4 folds, horizons
{14, 28} daily / {6, 12} monthly, seed 42, gpu311 (CPU torch 2.7.0).
Raw rows: `tva_bench_baselines.json` (all baselines),
`tva_bench_phase3.json` (TVA v2 after Phases 1–3),
`tva_bench_none.json` (torch-free `trend_network='none'`),
`tva_ablation_results.json` (5-seed graph + component ablations).

## Mean MASE by dataset (lower is better)

| model | factor_panel | load_daily | load_monthly | load_artificial |
|---|---|---|---|---|
| SeasonalNaive | **1.165** | 0.926 | 0.805 | **3.561** |
| LastValueNaive | 1.700 | **0.904** | **0.603** | 2.714* |
| Cassandra | — | 1.252 | 0.757 | 6.777 |
| VAR | (failed) | 1.151 | 0.645 | (failed) |
| SectionalMotif | 1.158 | 2.759 | 0.995 | 5.402 |
| detector-alone | 1.903 | 3.465 | 2.924 | 12.898 |
| **TVA as-is (pre-rework)** | **23.6** | — | — | — |
| TVA v2 (post Phases 1–3) | 1.99 | 3.39 | 2.87 | 11.49 |
| TVA 'none' (torch-free) | 2.00 | 3.39 | 2.87 | 12.63 |

*LastValueNaive value from the baselines run; smaller-is-better throughout.
"TVA as-is" measured on 3 factor-panel folds before any change (SMAPE ≈ 100).

## Graph ablation (SVAR panel, 5 seeds × 2 folds, 60 epochs)

All modes identical within noise: discovered 1.0601, none 1.0602,
random 1.0601, shuffled 1.0601, transposed 1.0600 (± 0.13 across folds,
± 0.0005 across seeds). **Claim rule FAILS** — the graph is not
demonstrably load-bearing. Arm A (no network) 1.0583 vs Arm C (full TVA)
1.0601: **Arm C ≈ Arm A**, so per the plan's kill rule the network does not
earn its complexity, and the torch-free `trend_network='none'` mode ships
as the equivalent configuration.

## Gate evaluation (from the plan)

1. *TVA beats detector-alone by ≥5% MASE skill*: geometric-mean skill ≈
   +3% (0.96 / 1.02 / 1.02 / 1.12 by dataset) — **not met** (marginal).
2. *TVA beats Arm A*: equal everywhere except load_artificial (11.49 vs
   12.63) — **not met** overall.
3. *Never lose to SeasonalNaive*: loses on all four datasets — **not met**.

## What the rework did and didn't fix

- The Phase-1 fundamentals (delta targets + scaling, temporal tokenizer,
  damped baseline + zero-init heads, validation early stopping) took the
  factor panel from MASE 23.6 → 2.0: the pre-rework failure was real and is
  fixed. TVA is no longer pathological; it now sits exactly at
  detector-decomposition quality.
- The binding constraint is now the **detector decomposition itself**:
  detector-alone loses to SeasonalNaive on every dataset, and TVA (with or
  without the network) tracks it closely. Neither the learned graph nor the
  network correction moves point accuracy on these panels — exactly the
  honest outcome the plan told us to design the ablation to catch.
- What survives on its merits: the discovery layer (identifiable named
  factors, conditional stability-selected edges with zero topology
  seed-variance — the explainability ask), MinT reconciliation in the
  forecast path, calibrated sigma intervals, and the torch-free mode.
- Next lever (out of scope here): improve the decomposition's trend/
  seasonality quality (e.g., stronger seasonality handling on load_daily,
  where SeasonalNaive's advantage is pure seasonality), since every TVA
  configuration is now capped by it.
