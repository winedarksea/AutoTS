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

---

# Latent trend-factor validation (2026-08-15)

The results above concluded that decomposition is the binding constraint, but
none of those panels were built the way TVA *assumes* data is generated:
series driven by shared latent trend factors. This section closes that gap.

`SyntheticDailyGenerator` gained a latent-factor mode (`n_latent_factors`,
`factor_strength`, `factor_response_lag_max`, `include_factor_series`), so the
factor paths, loadings, and response lags are known exactly.
`examples/tva_factor_validation.py` scores TVA against that ground truth and
localizes any failure to a stage. Ground truth (`get_true_factors()`) is a
scoring oracle only — in latent mode the factors are never in `get_data()`,
and every headline number comes from the fully blind pipeline.

## Headline: near-perfect on clean trends, collapsed on everything else

Factor recovery = mean |corr| between true and discovered factors after
Hungarian matching on differenced tracks (`discovery.match_factors`),
penalized over `max(K, r)` so a missing factor cannot be hidden.

| discovery input | mean \|corr\| | communities | changepoint-cluster ARI |
|---|---|---|---|
| true trend components (oracle diagnostic) | **0.85 – 0.98** | 1.00 | **1.00** |
| detector's estimated trend (what TVA uses) | **0.23 – 0.25** | ~0.25 | ~0.00 |
| raw daily data | 0.005 | — | — |
| observed factor columns (corr 0.9997 with truth!) | 0.02 – 0.07 | — | — |

Discovery, given a clean trend, recovers the factors nearly perfectly,
assigns every series to its true factor, and its changepoint-proximity
clustering reproduces the true factor partition exactly (ARI 1.00). Fed the
detector's trend instead, all three collapse. The recovery gap (oracle −
estimated) is 0.33 – 0.72 across the grid.

The last row is the surprise, and it redirects the diagnosis: even when the
factors are handed to discovery as observed columns correlated 0.9997 with
the truth, recovery still fails. Whatever is wrong is therefore not only the
decomposition. See the next section.

## Root cause: lag-1 differencing destroys slow trend factors

The obvious reading — "the detector's trend estimate is noisy" — is true but
is not the root cause. The observed-factor mode settles it. With
`include_factor_series=True` the generator appends the factor paths to the
panel as literal `market_factor_*` columns, correlated **0.9997** with the
true factors. Discovery still fails on them (mean |corr| 0.07). Per the
plan's ladder, a failure in the mechanics-check mode is a code problem, not
a statistics problem — and it is:

`discover_structure` differences the panel once (`_difference_and_standardize`)
before extracting factors. A macro trend factor moves over *months*, so its
one-day increment is tiny, while observation noise is fully present at every
lag. On the literally-observed factor columns:

| column | level corr with true factor | std(Δ signal) | std(Δ noise) | **Δ SNR** |
|---|---|---|---|---|
| market_factor_1 | 0.9997 | 0.196 | 0.519 | **0.38** |
| market_factor_2 | 0.9997 | 0.084 | 0.529 | **0.16** |
| market_factor_3 | 0.9997 | 0.156 | 0.528 | **0.30** |

A 0.9997-correlated signal becomes a 0.2-SNR one purely by differencing at
lag 1. The same operator applied to a piecewise-linear trend is worse still:
its difference is a step function that only moves at changepoints, so it
contributes almost no variance at all.

This is a regime mismatch, not a bug in the statistics. Lag-1 differencing is
the right call for **random-walk-like** factors, where day-to-day movement
*is* the signal — which is what the existing `make_svar_panel` and
`TestFactorRecovery` panels contain, and why discovery scores well on them.
It is the wrong call for **slow piecewise-linear macro factors**, which is
what TVA's design documents describe as the target.

The noise budget follows directly. Injecting white noise into the *true*
trend and re-running discovery:

| excess first-difference volatility vs true trend | factor recovery |
|---|---|
| 0× (true trend) | 0.91 |
| 1× | 0.53 |
| 3× | 0.20 |
| 10× | 0.06 |

The detector's estimated trend runs at **5 – 14× excess** difference
volatility (measured per series against the true trend); on some series its
trend correlates *negatively* with the truth (−0.78) and spans 10× the true
range. So the decomposition is genuinely poor *and* the operator downstream
of it is unusually intolerant of that — the two compound.

## What was tried, and the one refinement that ships

Detector-side tuning (`smoothing_window`, lowpass cutoff, PELT penalty and
minimum segment length) trades one failure for another: PELT penalty 50 /
segment 60 brings excess volatility to 0.9× and lifts recovery 0.25 → 0.50,
but over-smooths away real structure — the factor count collapses from 3 to
1–2 and the result is seed-unstable.

The refinement that ships is `factor_hp_lambda` in
`DEFAULT_DISCOVERY_CONFIG`: Hodrick-Prescott pre-smoothing applied to the
panel before differencing, for *factor extraction only* (edges are still
found on the unsmoothed residuals, so short-lag lead-lag structure is not
smoothed away). Measured across 16 configurations (4 seeds × 2 strengths ×
2 noise levels):

| discovery input | HP off | HP λ=1e8 |
|---|---|---|
| raw daily panel | 0.252 | **0.485** |
| oracle trend panel | **0.890** | 0.505 |
| SVAR lasso edge precision | **0.78** | 0.64 |

It roughly doubles recovery on noise-dominated input, with the correct
factor count in 16/16 configurations — and it *hurts* clean input and edge
precision by a similar margin. Read through the root cause above, that is
what you would expect: HP smoothing suppresses exactly the high-frequency
content that lag-1 differencing over-weights, which rescues slow trend
factors and discards the signal when the factors are random-walk-like. It is
a regime switch, not a free win, so it ships **off by default** and opt-in,
with the tradeoff recorded in the config comment and locked by
`TestFactorHPSmoothing`.

On the observed-factor panel it lifts recovery from 0.021 to 0.472 — the
mechanics-check mode still does not reach the near-perfect recovery the plan
predicted, so the differencing operator, not the smoothing knob, is where the
real fix belongs (a longer differencing stride, or extracting factors in
level space with a smoothness penalty). That is the next lever and is left
for a follow-up rather than rushed here.

### Negative result: a longer differencing stride is not the fix

The root cause points at the differencing operator, so differencing at a
stride h (`x[t] - x[t-h]`) instead of lag 1 was tested directly — signal
grows roughly linearly in h while independent noise grows as sqrt(2), so the
SNR argument favors it. Measured over 3 seeds, h in {1, 7, 28, 91, 182}:

| discovery input | h=1 | h=7 | h=28 | h=91 | h=182 |
|---|---|---|---|---|---|
| oracle trend | **0.97** | 0.95 | 0.88 | 0.68 | 0.43 |
| detector trend | 0.50 | 0.51 | 0.53 | 0.53 | **0.56** |
| raw daily | 0.02 | 0.26 | 0.34 | **0.43** | 0.46 |

Same tradeoff shape as HP smoothing and no better: large gains on raw input,
monotone damage to the oracle, and — the point that matters — the detector
trend stays flat around 0.5 at *every* stride. A second knob with the same
profile and no additional benefit is not worth shipping, so it was not added.

That flatness is itself the finding: the detector trend's error is
**structured**, not high-frequency noise, so no frequency-domain treatment
recovers it. Both knobs cap out near 0.5 against an oracle ceiling of 0.9+.
Improving the decomposition itself is the only remaining lever — which is
what the earlier benchmark section already flagged as the next one, now
confirmed causally rather than by elimination.

Adaptive selection of the smoothing was investigated and rejected: the
obvious dial (lag-1 autocorrelation of the differenced panel) cannot
distinguish the detector's trend (+0.96, looks clean) from the true trend
(+0.99), which is exactly the case that needs to be told apart.

## Full-grid results

`examples/tva_factor_validation.py --strengths 0.9,0.5 --noise-levels
0.05,0.15 --lag-modes 0,10 --visibility latent,observed --networks none,v2
--folds 2`, gpu311, 16 cells, 1095 days, 24 series, 3 factors, horizon 28.
Raw rows: `fv_hpoff.json`.

### Forecast skill (mean over all 16 cells; MASE, lower is better)

| model | MASE | skill vs SeasonalNaive | follower-only MASE | directional coherence |
|---|---|---|---|---|
| **SeasonalNaive** | **1.022** | 1.00 | **0.973** | 0.482 |
| LastValueNaive | 1.284 | 0.80 | 1.125 | 0.000* |
| TVA `none` (torch-free) | 4.141 | 0.27 | 2.650 | 0.563 |
| TVA `v2` (network) | 4.142 | 0.27 | 2.652 | 0.565 |
| detector-alone | 4.254 | 0.26 | 2.909 | 0.545 |

*LastValueNaive forecasts are flat, so every pairwise direction is 0 and
never "agrees" — a metric artifact, not a coherence failure.

Three things this settles, on the data type TVA was designed for:

1. **TVA loses to SeasonalNaive by ~4×**, and beats detector-alone by only
   2.7%. Success bar (a) — "TVA beats SeasonalNaive and detector-alone at
   strength ≥ 0.7, noise ≤ 0.15" — **fails**.
2. **The network contributes nothing.** `MASE(v2) − MASE(none)` = **0.0009**,
   i.e. 0.02%. The Phase-4 ablation verdict reproduces exactly on the panel
   type most favorable to it.
3. **The graph does not help followers.** On lagged panels, where leader
   series genuinely predict follower futures and no univariate model can
   access that information, TVA scores 2.650 on followers against
   SeasonalNaive's 0.973. Success bar (b) **fails**.

Directional coherence is 0.56 for TVA against 0.48 for SeasonalNaive — barely
above the 0.5 chance level even at `factor_strength=0.9`, where a coherent
model should approach 1.0. The mixed-direction pathology TVA exists to fix is
not fixed on data built to exhibit it.

### Recovery by regime and factor strength

| regime | strength | mean \|corr\| | community accuracy |
|---|---|---|---|
| oracle trends | 0.9 | **0.930** | **0.921** |
| oracle trends | 0.5 | 0.579 | 0.654 |
| estimated trends | 0.9 | 0.278 | 0.274 |
| estimated trends | 0.5 | 0.278 | 0.264 |
| raw data | 0.9 | 0.004 | 0.387 |
| raw data | 0.5 | 0.004 | 0.381 |

Loading-matrix correlation: 0.809 on oracle trends, −0.014 on estimated.
Note the estimated-trend row is **flat in factor strength**: doubling how
much of the panel's trend movement is factor-driven changes measured
recovery by 0.000. The detector's trend carries no usable factor information
at either strength, which is a stronger statement than "recovery is low".

### Edge recovery — the pure-confounder claim does not hold here

| lag mode | regime | true edges | found | precision | recall |
|---|---|---|---|---|---|
| 0 (contemporaneous) | oracle trends | **0** | 78.8 | — | — |
| 0 (contemporaneous) | estimated trends | **0** | 73.4 | — | — |
| 10 (leader/follower) | oracle trends | 77.5 | 183.3 | 0.119 | 0.263 |
| 10 (leader/follower) | estimated trends | 77.5 | 75.1 | 0.040 | 0.039 |

On contemporaneous panels every series is conditionally independent given
the factors, so the true series→series edge set is **empty** and a correct
discovery should find ~nothing. It finds **73–79 edges**. This does not
contradict `test_pure_confounder_yields_no_edges`, which passes: that test's
panel is factor + iid noise only, while these panels also carry seasonality,
holidays, anomalies, and level shifts — shared structure the factor
deconfounding step does not remove, which then surfaces as spurious
conditional edges. Success bar (a)'s pure-confounder-correctness clause
**fails** on realistic panels.

### Attribution table (mean over cells)

| quantity | value | reading |
|---|---|---|
| decomposition corr (est vs true trend) | 0.475 | the bottleneck |
| changepoint-cluster ARI, true trend | **1.000** | clustering logic is correct |
| changepoint-cluster ARI, estimated trend | −0.004 | destroyed by decomposition |
| recovery gap (oracle − estimated) | 0.477 | attributable to decomposition |
| factor-forecast skill vs flat continuation | **1.759** | factor forecasting works |
| network contribution, MASE(v2) − MASE(none) | 0.0009 | network is inert |
| true-factor collinearity | 0.478 | not the limiting factor here |

Stage by stage: factor forecasting is **good** (76% better than a flat
continuation), the changepoint-clustering path is **exactly right** given a
clean trend (ARI 1.000), factor extraction is **excellent** given a clean
trend (0.93 / 0.92), and the network is **inert**. Everything downstream of
the decomposition works; the decomposition is where it all fails.

## Verdict against the plan's success bars

| bar | result |
|---|---|
| (a) TVA beats SeasonalNaive and detector-alone at strength ≥ 0.7, noise ≤ 0.15 | **fail** (4.14 vs 1.02; +2.7% over detector) |
| (a) oracle recovery ≥ 0.8 | **pass** (0.930 at strength 0.9) |
| (a) discovery finds ~no series edges on pure-confounder panels | **fail** (73–79 spurious) |
| (b) discovered edges recover leader→follower structure | **fail** (recall 0.04 estimated / 0.26 oracle) |
| (b) graph-aware TVA beats univariate baselines on follower series | **fail** (2.65 vs 0.97) |

One bar passes. The single actionable conclusion is unchanged from the
earlier section but now proven rather than inferred: **every TVA stage that
can be tested in isolation works, and all of them are capped by the trend
decomposition.** Until the decomposition produces a trend within roughly 1×
the true trend's difference volatility, no amount of network, graph, or
factor-layer work will move these numbers.

## Bug fixed: `trend_network='none'` was not actually torch-free

The torch-free mode shipped in the previous phase raised
`NameError: name 'torch' is not defined` on any machine without torch — the
device selection in `TVA.__init__` and the training-tensor construction in
`fit()` both ran before the `'none'` early return. Both are now gated
(tensor construction moved below the early return). Caught by this harness on
its first run.

## Part 6 (short-history responder series) not started

The plan gates it on the Part 5 success bars passing on full-history panels.
They do not: TVA does not beat SeasonalNaive on these panels, and factor
recovery from the estimated trend is far below the 0.8 bar. Responder
behavior is not meaningful until the core factor pipeline works, so that
phase remains unstarted by design.

---

# Learned latent-factor trend mode `trend_network='factor'` (2026-08-16)

Protocol: `examples/tva_factor_validation.py --strengths 0.9,0.5
--noise-levels 0.05,0.15 --lag-modes 0,10 --visibility latent,observed
--networks none,v2,factor --folds 2`, seed 42, gpu311. Raw rows:
`examples/fv_factor.json`. Success criteria were frozen before the run
(plan `cryptic-wishing-hummingbird.md`); they are scored below as written,
including the ones that fail.

## Headline

**The factor mode is the first TVA trend network that is not inert.** Across
all 16 grid cells it beats the torch-free `'none'` baseline, and `v2` remains
indistinguishable from `'none'` to three decimals:

| cell group | n | SeasonalNaive | TVA[none] | TVA[v2] | TVA[factor] | Δ(factor − none) | wins |
|---|---|---|---|---|---|---|---|
| latent | 8 | 0.955 | 2.849 | 2.850 | **1.714** | **−1.135** | 8/8 |
| observed | 8 | 1.090 | 5.433 | 5.434 | **2.456** | **−2.977** | 8/8 |
| lag=10 (followers only) | 8 | 1.039 | 4.142 | 4.143 | **2.054** | **−2.088** | 8/8 |

The v2-vs-none column is the point worth keeping: on the *same* cells where
the learned factor mode moves mean MASE by 1.14, the v2 network moves it by
0.001. This reproduces the 2026-08-15 inertness finding and rules out "the
panels are unforecastable" as its explanation — the signal was reachable, v2's
representation could not reach it.

**The caveat that matters:** TVA still loses to SeasonalNaive on these panels
(1.71 vs 0.96 latent). The factor mode closes roughly 60% of the gap between
`'none'` and SeasonalNaive; it does not close it. The residual is seasonal
handling, as the earlier section concluded.

## Frozen criteria, scored as written

| # | criterion | measured | verdict |
|---|---|---|---|
| a | learned recovery ≥0.70 on latent cells; ≥10× discovery | 0.423 learned vs 0.248 discovery (**1.7×**) | **miss** (above the 0.40 kill line) |
| b | ΔMASE ≤ −0.10, wins ≥70%, no observed regression >+0.05 | −1.135, **100%** wins, worst observed cell −2.487 | **pass** |
| c | follower MASE improves ≥0.05; Spearman(lags) >0.5 | followers **−1.121**; Spearman n/a | **partial** — MASE bar met, lag bar not testable (see below) |
| d | directional coherence ≥ coherence(none) − 0.02 | **+0.021** | **pass** |
| e | negative control: variance share <10%, MASE ≤ none+0.02 | MASE −0.747; variance share not scoreable | **partial** — MASE bar met, diagnostic replaced |
| f | real data: MASE(factor) ≤ 1.02× MASE(none) per dataset | 4 of 5 improve 27-48%; **load_daily 2.33×** | **fail** |
| g | contemporaneous found-edges → ~0; lag-10 recall > 0.04 | 70 → 60 edges; recall 0.126 → **0.067** | **fail** |

## Criterion (a): the 0.70 bar was unreachable, and we can prove it

Recovery is 0.42 against a 0.70 bar. Before treating that as a modelling
failure, the ceiling was measured directly: give an estimator the **true
loadings** and the best smoother available, and ask how well *any* linear
method can recover the factor paths from the noise-level-0.05 panel.

| estimator (strength 0.9 / noise 0.05 latent panel) | recovery |
|---|---|
| discovery pipeline on the detector trend (diff space) | 0.239 |
| **oracle loadings + fixed-grid spline smoothing** | 0.57 |
| **oracle loadings + ℓ1 trend filtering** | **0.71** |
| this mode, loadings estimated (true components removed) | 0.65 |
| this mode, loadings estimated (detector components removed) | 0.43 |
| discovery pipeline on the *noiseless* true trend | 0.954 |

The reason is a property of the data, not the model: at `noise_level=0.05`
the per-series observation noise (std 4.3) is **larger than the trend itself**
(std 3.0), and `match_factors` scores lag-1 differences, where a 0.7%
level-space error is a ~100% difference-space error. The 0.70 bar was set
without this number. The mode reaches 91% of the measured ceiling on cleanly
adjusted input; the remaining loss (0.65 → 0.43) is the detector's
component estimates, i.e. the same decomposition bottleneck as before.

Scored honestly: **(a) misses its stated bar and is not threshold-shopped.**
It clears the kill line, and (b) — which the plan named as the criterion the
mode is accountable for — passes by a wide margin.

## What the evidence forced us to change

The design in the plan (spline basis, joint Adam over paths + loadings, l2
smoothness) scored **0.25**. Three measured changes produced the shipped
estimator:

1. **ℓ1 trend filtering, not ℓ2 smoothing.** The generator's factors are
   piecewise linear with sparse changepoints; ℓ2 blurs them. Oracle ceiling
   0.57 → 0.71.
2. **Alternating GLS / trend-filter identification, not joint gradient
   descent.** 0.25 → 0.65. Torch now *refines* (loadings, lags, damping);
   letting gradients touch the factor coefficients measurably degraded
   recovery (0.63 → 0.50), so `lr_coef` defaults to 0.
3. **High-pass the subtracted components.** Seasonality/holidays/anomalies are
   high-frequency by definition, so their slow drift is misattributed trend.
   On a non-seasonal panel the detector puts nearly the whole signal into
   "seasonality" (std 1.03 of a 1.07 panel) and subtracting it raw destroyed
   the factor signal entirely (0.59 → 0.01).

Two real bugs were found on the way and fixed: the SVD warm start used
`np.convolve(mode='same')`, whose zero-padding wrecked the level tracks
(0.82 → 0.51 on noiseless input), and the trend filter dropped the Lasso
intercept while regressing against a centered panel.

## Criterion (c): response lags are not identifiable here

The follower-MASE half of (c) passes (−1.121). The lag-recovery half is not
testable on this data, and that is a data property, not a tuning failure —
measured with **oracle factor paths**:

| input | lags within ±2 | Spearman |
|---|---|---|
| noisy adjusted panel | 0.21 | 0.10 |
| noiseless true trend panel | 0.62 | 0.57 |

A 6-day shift of a slow path changes a series by far less than its noise, so
there is nothing to learn. Rather than fit N×(max_lag+1) parameters to noise,
`factor_max_lag` **defaults to 0**; the mechanism is implemented, tested, and
opt-in. The follower gains above therefore come from better factor paths, not
from lead-lag transfer.

## Criterion (g): the graph rework does not work

`discover_structure(..., external_factors=...)` was built as designed —
deconfound edge discovery against the learned level-space factors instead of
the internal difference-space SVD. With families held equal:

| panel | baseline data-driven edges | deconfounded | leader→follower recall |
|---|---|---|---|
| contemporaneous (true edge set empty) | 70 | 60 | — |
| lag-10 leader/follower | 70 | 57 | 0.126 → **0.067** |

Spurious edges fall 14%, nowhere near the ~0 target, and recall halves. The
capability ships but `factor_deconfound_edges` **defaults to False**, with
these numbers recorded in the docstring. A better confounder estimate is not
sufficient to make series-level edges trustworthy on this data.

## Criterion (e): negative control — no degradation, but the variance-share diagnostic is useless

On panels generated with `n_latent_factors=0` (no shared structure at all):

| model | MASE | vs none |
|---|---|---|
| SeasonalNaive | 0.932 | — |
| TVA[none] | 2.295 | — |
| **TVA[factor]** | **1.548** | **−0.747** |

The MASE half of (e) passes with room to spare — the mode does not degrade a
factor-free panel, it improves it, because a shared piecewise-linear basis is
simply a better trend smoother than the detector trend plus a damped slope
even when the "factors" are only describing independent trends.

The variance-share half **cannot be scored as written**: `factor_variance_share`
reads 1.00 on every panel, with three true factors or zero. That is structural,
not a bug — the idiosyncratic term is capped at 2 dof by design, so essentially
all trend movement routes through the factor layer regardless of whether it is
shared. A second definition (shared vs idiosyncratic contribution to the
reconstructed trend) was implemented and reads 1.00 as well.

The replacement is `factor_network.split_half_stability()`: fit the factors on
two disjoint halves of the series and score the agreement. Real shared factors
replicate across subsets; factors absorbing idiosyncratic trends do not.

| panel (K fit = 3) | split-half stability |
|---|---|
| 3 true factors | 0.56 |
| 1 true factor | 0.43 |
| **0 true factors** | **0.34** |

Monotone and truth-free, but the bands overlap — a soft indicator, documented
as such, not a test.

## Criterion (f): real data — one regression found, diagnosed, and fixed

The first full benchmark run failed (f) on one dataset of five:

| dataset | TVA[none] | TVA[v2] | TVA[factor] | ratio vs none |
|---|---|---|---|---|
| factor_panel | 2.083 | 2.077 | **1.523** | 0.73 |
| synthetic_factor_panel | 2.712 | 2.713 | **1.415** | 0.52 |
| load_monthly | 2.874 | 2.877 | **1.766** | 0.61 |
| load_artificial | 11.790 | 11.754 | **8.086** | 0.69 |
| **load_daily** | **3.468** | 3.466 | **9.576** | **2.76 — fail** |

Per-series breakdown located it precisely: the *median* load_daily series
improved (ratio 0.84) and the mean was already better; the aggregate was
destroyed by a handful of trendless columns, worst of all a precipitation
gauge at 15x. Those series have no low-frequency structure, but an
unconstrained least-squares solve still hands them loadings, and the factor
layer then extrapolates shared drift into a forecast that should stay flat.

Fix: a per-series gate on the ratio of smoothed spread to residual
high-frequency spread — a property of the series, not of the fit. Separation is
wide and was measured before the threshold was set: every series on the
synthetic latent panels scores >= 0.84, while every regressing load_daily
series scores <= 0.53 (precipitation 0.19, wind 0.23, holiday-spike wiki pages
0.40-0.53). The threshold is 0.6.

An in-sample SSE-gain gate was tried first and rejected on evidence: raw SSE is
dominated by irreducible daily noise, so it fired on 19 of 24 series on a panel
with genuine factors. Both the rejected attempt and the reason are recorded in
`_gate_trendless_series`.

### After the gate: better, still failing

Re-running the full benchmark with the gate (3 folds, horizons {14,28}):

| dataset | TVA[none] | TVA[factor] before gate | TVA[factor] after gate | ratio vs none |
|---|---|---|---|---|
| factor_panel | 2.083 | 1.523 | **1.523** | 0.73 |
| synthetic_factor_panel | 2.712 | 1.415 | **1.415** | 0.52 |
| load_monthly | 2.874 | 1.766 | **1.766** | 0.61 |
| load_artificial | 15.342 | 8.086 | **10.870** | 0.71 |
| **load_daily** | **3.468** | 9.576 | **8.072** | **2.33 — still fails** |

The gate removed the precipitation blow-up and touches 0 of 24 series on the
synthetic latent panels, but **criterion (f) still fails on load_daily**. The
earlier single-fold, per-series-mean figure quoted during diagnosis (3.15 ->
1.76) is not the benchmark's own aggregate and should not be read as a fix.

The failure is specific to MASE. On the same dataset the factor mode is
*better* on sMAPE (58.3 vs 70.7) and on interval containment (0.809 vs 0.633),
and worse on SPL (21.3 vs 8.2). MASE divides by each series' in-sample seasonal
naive error, so the aggregate is dominated by a few small-scale, spike-driven
wiki pages where that denominator is tiny; the trend-to-noise gate catches the
trendless ones but not series whose low-frequency structure is real yet
holiday-driven. That is the open item.

Overall skill across all five datasets (geometric mean MASE skill vs
SeasonalNaive, higher is better) still favors the factor mode over every other
TVA arm, at a fraction of v2's cost:

| model | mase_skill_geo | containment | mean fit sec |
|---|---|---|---|
| TVAModel[factor] | **0.416** | **0.832** | 10.0 |
| TVAModel[none] | 0.326 | 0.554 | 5.5 |
| TVAModel[v2] | 0.332 | 0.560 | 163.8 |
| FeatureDetectorForecast | 0.325 | 0.581 | 2.2 |

All TVA arms remain well below SeasonalNaive (1.0) on real data.

## Cost

Capping candidate knots at 200 (widening the spacing beyond ~1400 steps, where
7-day knots are far below the resolution the noise supports) cut the fit from
131 s to 34 s at N=200/T=3000, and 15 s to 8 s at N=24/T=1095. For scale, the
v2 network averaged 164 s per fit on the same benchmark against the factor
mode's 13 s.

## Status of the mode

Shipped and enabled in search (`trend_network='factor'`, weight 0.3). It is the
first TVA trend network that beats the torch-free baseline rather than matching
it. It does **not** yet beat SeasonalNaive on these panels, factor recovery
misses its stated bar against a measured ceiling of 0.71, response lags are not
identifiable at this noise, the graph rework failed, and it still regresses on
load_daily's MASE. Those five are the honest boundary of what this change
accomplished.

Scored against the frozen criteria: **(b) and (d) pass, (c) and (e) pass on
their forecast-quality half and fail or are unscoreable on their
structure-recovery half, and (a), (f) and (g) fail.** The kill rule is not
triggered — it fires on (a) < 0.40 (measured 0.42) or (b) < 0.02 (measured
1.135) — but three of seven bars are misses, and the mode ships with all of
them recorded here rather than renegotiated.

---

# TVA rework: Phases 0-6 (superset plan, 2026-08-17)

Everything below is measured on this tree. New behaviour is behind
`factor_config` flags that default to today's behaviour; nothing here has had
its default flipped yet. Negative results are recorded, not deleted.

## Phase 0 — the evaluation foundation was broken

**0a. The synthetic validation harness was grading forecasts of a panel that
was never observed.** `factor_forecast_skill` regenerated the panel at
`n_days + horizon` to obtain "the future". The generator's changepoint *count*
depends on `n_days` (`_generate_trend`), so the longer generation advances the
RNG stream differently and produces a different realization: measured
`max |diff| = 83.0` on the first 500 rows of a 500-day panel, with different
changepoint counts per factor. Every factor-forecast-skill number published
before this fix was computed against the wrong realization.

Fixed by generating **once** at `n_days + horizon` and slicing. Verified: the
history slice is *byte-identical* to the head of the full realization for
`df`, `truth['factors']` and `true_trends`, and the graded future is
contiguous with it.

**0b. Rotation-insensitive scoring.** A factor model is identified only up to
an invertible rotation, so matching estimated factors to true factors one by
one penalizes a correct fit that landed in a rotated basis. Added matched
canonical correlation, loading-subspace recovery (principal angles), trend
reconstruction error, and same-realization learned-factor forecast skill.
Verified exactly invariant: `canon_corr(F, F@R) = 1.000000` and
`subspace(W, W@inv(R).T) = 1.000000` for a random invertible `R`, with trend
reconstruction error `0.0`.

Generator truth now carries the leader->follower `adjacency`/`edges`,
`dominant_factor`, `observation_mask` and `responder` status, so the harness no
longer re-derives the edge set from loadings and lags.

**0d. LRP 180-day harness.** Five frozen non-overlapping 180-day origins; folds
0-2 are iteration, folds 3-4 are promotion and only run under `--promotion`.
The oldest fold has 973 training rows against the `3 x horizon = 540`
requirement, so all five are valid. Series are tagged from a static,
checked-in inspection of the CSV: **12 base**, 17 derived-ratio, 6 frozen-tail.
Gates use base only. (Correction to the plan's assumption: `Marketplace_DAU` is
*not* frozen — its longest constant run is 5 days.)

**0h. The pooled comparator did not re-base the gates.** A pooled
cross-learning baseline (`MultivariateRegression`, per-series normalized)
scored base MASE 2.9295 against SeasonalNaive's 2.9240 — 0.2% *worse*, well
inside the 10% escalation trigger. SN-parity stands as the right bar.
Side finding: that comparator halves direction-coherence error (0.185 vs
0.458) while being MASE-neutral, so coherence and accuracy are separable on
this panel.

**0f. The arbitration prize is small.** Per-series oracle choosing between two
similar models buys 2.3% (LRP fold 0) to 4.2% (factor panel); an honest
holdout chooser gives all of it back and then some (-0.6% and -8.0%). Whole-
forecast switching stays a diagnostic.

**0e. The coherence metric was at its own ceiling — this rescopes Phase 4.**
Oracle `directional_coherence` of the *realized future* is **0.436**, below
chance, while TVA's fitted forecast scores 0.471 — i.e.
`oracle_normalized_coherence` ~= **1.08**. TVA already exceeds the realized
future's own score on that metric. The cause is in the attribution: over a
28-step window, seasonality accounts for **0.589** of the mean absolute net
change and noise **0.355**, while the entire trend (shared + idio) accounts for
**~0.04**. The windowed net-change form does not rescue it (0.443).

Ruled out: flat factors (continuation retains ~27% of in-sample per-step
movement) and over-gating (gated share 0.000). Confirmed as a real secondary
problem: **loading-sign agreement with truth is 0.490, indistinguishable from
chance**, so any shrinkage prior built on learned loading signs would push
about half the series the wrong way.

Usable metric: trend-only coherence, whose oracle is **0.857** against TVA's
0.51-0.59 — real headroom of ~0.27.

## Phase 1-2 — the safety floor and what actually moved the number

Two evaluation bugs had to be fixed before any of this could be measured:

1. **The inner validation folds were in-sample.** The fitted model was scored
   at earlier origins it had been fit through, so at H=180 it looked far better
   than it was and the blend handed it a weight it had not earned. Fixed with
   `inner_refit`: the *factor stage only* is refit on truncated history (the
   detector's components are reused), so the extrapolation being graded is
   genuinely out of sample, at one extra factor fit rather than a full TVA fit.
2. **The inner folds added back the in-sample *fitted* seasonality** at future
   dates — an oracle worth ~59% of the H-step change (see 0e) — while the
   SeasonalNaive fold got no such help. Replaced with an empirical profile of
   the detrended history up to the origin.

With both fixed, blend weights on LRP fold 1 moved from ~1.0 everywhere to a
mean of 0.34 with 15 of 35 series at pure SeasonalNaive.

### LRP iteration folds, base metrics, aggregate MASE ratio vs SeasonalNaive

| arm | agg | worst fold | p90 series | max series | dir-coh err ratio | mean blend w |
|---|---|---|---|---|---|---|
| unblended `factor` | 2.22 | — | — | — | 0.51 | 1.00 |
| A: Phase 1 safety layer | 1.590 | 1.906 | 4.50 | 9.31 | 0.179 | 0.43 |
| B: + tie tolerance 0.10 | 1.233 | 1.530 | 3.33 | 8.52 | 0.357 | 0.20 |
| C: + tie tolerance 0.25 | 1.123 | 1.279 | 1.67 | 6.14 | 0.548 | 0.10 |
| D: C + error cap x1 | 1.123 | 1.277 | 1.67 | 6.09 | 0.548 | 0.10 |
| E: C + seasonal arbitration | 1.134 | 1.309 | 1.85 | 5.84 | 0.726 | 0.10 |
| **F: E + log space** | **1.011** | **1.045** | 1.35 | 1.93 | **0.548** | 0.10 |
| G: E + 5 inner folds | 1.074 | 1.183 | 1.41 | 4.35 | 0.726 | 0.06 |
| gate | <=1.02 | <=1.05 | <=1.10 | <=1.50 | <=0.90 | — |

**Arm F clears aggregate MASE, per-fold MASE and the coherence gate.** The two
per-series gates (p90 1.35, worst 1.93) still fail.

The single largest lever is **log space** (C -> F changes nothing else):
aggregate 1.123 -> 1.011 and worst-series 6.14 -> 1.93. The LRP panel spans
~1e-2 to 2e8 and its engagement metrics comove in growth-rate rather than
standardized-level terms. Space is now selectable as `'auto'` and chosen by
inner-origin validation rather than asserted.

There is a **real accuracy/coherence trade-off** in the blend: as the blend
gets more conservative (mean weight 0.43 -> 0.10) the direction-coherence error
ratio rises from 0.179 to 0.548. Both still pass the coherence gate by a wide
margin, but the two objectives pull in opposite directions and any future
tuning has to price that.

### 2b — the decomposition gap is real, large, and closable

Mean |corr| to the generator's true trend, 24-series latent-factor panel:

| input | mean abs corr | nRMSE | non-trend energy retained |
|---|---|---|---|
| raw | 0.389 | 5.946 | 1.000 |
| detector-adjusted (`_build_adjusted_panel`) | **0.273** | 4.149 | 0.510 |
| robust joint estimator | **0.897** | 0.941 | 0.090 |
| oracle | 1.000 | 0.000 | 0.000 |

**`_build_adjusted_panel` is worse than doing nothing.** It removes half the
non-trend energy and still ends up *less* correlated with the true trend than
the raw panel, which means the high-pass subtraction is taking trend with it.
This is not a smoothing artifact: rolling means of raw at 91/182/365 days score
0.582/0.667/0.764, and the detector's own trend component scores 0.600 — all
below 0.897.

Also measured: the detector's components add nothing on this panel. The robust
estimator with **no components at all** scores 0.900, marginally better than
with them fully subtracted.

Converting that input into recovery is only a partial win so far:

| input | match corr | canonical corr | subspace | trend recon err | K |
|---|---|---|---|---|---|
| detector (strength 0.9) | 0.400 | 0.532 | 0.454 | 0.941 | 3 |
| robust (strength 0.9) | 0.351 | **0.770** | 0.378 | 24.500 | 4 |
| detector (strength 0.5) | 0.447 | 0.624 | 0.325 | 1.090 | 3 |
| robust (strength 0.5) | 0.355 | **0.745** | 0.478 | 1.951 | 4 |

Rotation-insensitive **span** recovery improves decisively (0.532 -> 0.770,
clearing the 0.70 factor-path gate), while level-space matching and
reconstruction scale get worse and rank selection picks an extra factor. The
robust input fixes what it was built to fix and exposes a separate scale/rank
problem downstream of it.

## Phase 5 — the bounded deconfounding fix failed its kill rule

On a contemporaneous panel whose true series->series edge set is **empty**:

| configuration | false-positive edges |
|---|---|
| baseline | 65 |
| + shared-component deconfounding (5a) | **72** |

Worse, not better, against a kill-rule bar of <10. It ships **default-off**
(`deconfound_components`).

The circular-shift null explains why this is not a screener defect: shifting
each series circularly destroys cross-series timing while preserving its own
autocorrelation, and the screener then finds a mean of **4.0** edges
(95th percentile 7.95) against the 65 observed. So the 65 are driven by genuine
shared structure that the factor deconfounding does not span — the confounding
is real and under-removed, not manufactured by the procedure.

Per 5c, `get_edges()` now documents these as predictive screening rather than
causal claims, and points at the factor->series loading graph as the
trustworthy structural output.

## Generator additions (0g, 3a, 6a, 6b) — all default-inert

Every addition below was verified to leave existing panels bit-identical.

- **0g** `scale_log_range`, `frozen_tail`, `derived_ratio_specs` — the three
  LRP artifacts nothing else reproduced: ~17 orders of magnitude of scale
  spread, exact deterministic ratio columns (identity verified exact, zero
  loading rows), and held-constant tails.
- **3a** `short_history_share`, `missing_share`, `missing_run_max` — ragged
  responder cohorts and observation gaps, with responder status and the
  observation mask in the truth manifest. Verified nothing is backfilled before
  a responder's first valid date.
- **6a** `noise_scale_mode='trend_delta'` — noise scaled to `k x std(daily
  trend increments)` instead of to level. Verified linear control:
  `noise_level` 0.5/2.0/8.0 gives `std(noise)/std(dTrend)` of 0.34/1.38/5.50.
  This is the knob that separates an SNR-determined recovery ceiling from a
  model-determined one.
- **6b** `generate_metric_surface_geo_panel` — the direct Use-Case-2 shape
  (`metric.surface.geo`, one factor per surface, per-geo loading scale, shared
  events, emitted `SeriesMetadata`). Verified the structure is real: mean
  trend-delta correlation is 0.581 within a surface and -0.064 across surfaces,
  and TVA runs on it with metadata priors and MinT reconciliation active.
- **6d** declared ratio identities (`derived_definitions`) — never inferred
  from column names; verified the identity holds exactly after the post-step
  and that parent forecasts are untouched.

## Phase 4 — the shrink works; the graph it is handed does not

Rescoped by the 0e diagnostic: raw `directional_coherence` is not the target
(TVA is already at 108% of its oracle), trend-only coherence is.

Sweep on a 24-series latent panel (K=3, strength 0.9), trend-only coherence and
MASE cost vs the unshrunk baseline, graph built from the **fitted** loadings:

| graph | s=0.01 | 0.03 | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|---|---|
| group — coherence | 0.429 | 0.429 | 0.429 | 0.429 | 0.429 | 0.429 |
| laplacian_k5 — coherence | 0.429 | 0.429 | 0.429 | 0.429 | 0.429 | 0.429 |
| laplacian_k5 — MASE cost | -0.06% | -0.16% | -0.38% | -0.69% | -1.22% | -1.98% |

**Flat at every strength and every graph.** Coherence gain 0.000 against a kill
bar of 0.05, so it is killed as a default. It fails the *gain* half of the rule,
not the accuracy half — the laplacian forms slightly improve MASE.

The blocker is upstream, and the oracle control isolates it exactly. With a
graph built from the generator's **true** loadings, the identical solver gives:

| strength | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|
| trend-only coherence | 0.556 | 0.746 | 0.905 | **1.000** |
| MASE cost | -1.04% | -1.68% | -2.14% | **-2.33%** |

So the mechanism delivers the entire 0.57 of headroom *and pays negative MASE
cost* — the moment it is handed a graph that reflects real structure. It is not
handed one: fitted dominant-factor recovery is **0.458** (chance 0.333 at K=3),
fitted-vs-true loading correlations after optimal matching are +0.074, -0.037,
+0.127, and of the 41 same-factor-same-sign pairs the fitted graph asserts, only
**8 (13%)** are real. On its own asserted pairs the shrink works perfectly
(0.878 -> 1.000).

Honest caveat recorded by the same agent: a **complete** graph (no structure at
all, shrink toward the panel mean) also reaches coherence 1.000 on this window,
because all three true factors happen to drift the same way over 28 days. On
honest inner origins that structure-free shrink is worth only -0.75% scaled MAE.
So the oracle-graph result is a valid upper bound on the *mechanism*, not proof
that structure specifically is what buys it.

The module ships default-off (`coherence: False`) and **re-gated on loading
recovery rather than retired**: it becomes useful exactly when identification
does. `resolve_signs` (mass-weighted factor orientation) is in place as the
better-justified estimator, though on this panel it changes no orientation and
sign agreement with truth is 0.75 under both conventions — so the 0e
sign-at-chance finding is panel-dependent, not universal.

## Phase 7 — not run, per the plan's own condition

Phase 7 (decomposition<->factor iteration) was conditional on 2b's robust input
*not* closing the decomposition gap. It closed it: trend correlation 0.273 ->
0.897 against an oracle of 1.000. Running Phase 7 as well would be spending a
2x fit cost on a gap that is already addressed, so it is deliberately skipped
and recorded here rather than silently dropped.

## Three more levers, all measured, all negative on LRP

Same three iteration folds, base metrics, everything else held at arm F:

| arm | agg | worst fold | p90 series | max series | dir-coh err ratio |
|---|---|---|---|---|---|
| **F: reference (tol 0.25 + seasonal + log)** | **1.011** | **1.045** | 1.35 | 1.93 | 0.548 |
| H: F + blend risk weight 0.5 | 1.022 | 1.045 | 1.35 | 2.01 | 0.548 |
| I: F + blend risk weight 1.0 | 1.026 | 1.048 | 1.35 | 2.01 | 0.548 |
| J: F but `space='auto'` | 1.166 | 1.417 | 2.27 | 5.84 | 0.726 |
| K: F + robust input estimator | 1.021 | 1.087 | 1.96 | 2.91 | 0.643 |
| gate | <=1.02 | <=1.05 | <=1.10 | <=1.50 | <=0.90 |

- **Risk-weighted blend selection does not work.** It was added specifically
  to target the "no series worse than 1.50x" gate by scoring
  `mean + risk * worst_fold` instead of the mean. It moves the worst-series
  ratio the wrong way (1.93 -> 2.01) and costs aggregate MASE. The config key
  `blend_risk_weight` stays at 0 and the mechanism is recorded as tried.
- **Auto space selection underperforms the forced choice badly** (1.166 vs
  1.011). Forcing log is right for this panel and the inner-origin criterion
  does not reliably discover that, so 2c's stated gate ("log selected by inner
  validation on >=2 of 3 folds") is **not met** even though log itself is the
  single biggest win. The selector's criterion, not the option, is what needs
  work. `space='auto'` therefore should not be recommended as-is.
- **Robust input helps recovery and not LRP accuracy** (1.021 vs 1.011, and
  worst-series 1.93 -> 2.91). Taken with the synthetic result — canonical
  correlation 0.532 -> 0.770 — the two measurements are consistent and
  informative rather than contradictory: the robust estimator recovers the
  factor *span* much better while making level-space scale and rank selection
  worse, and on LRP the scale/rank damage outweighs the span gain. It stays
  default-off and is the right input to revisit once rank selection is fixed.

## Where this leaves the promotion gates

On the LRP **iteration** folds (promotion folds 3-4 were never touched):

| gate | bar | best measured | status |
|---|---|---|---|
| aggregate base MASE ratio | <=1.02 | 1.011 | PASS |
| worst iteration fold | <=1.05 | 1.045 | PASS |
| direction-coherence error ratio | <=0.90 | 0.548 | PASS |
| 90th-pct per-series ratio | <=1.10 | 1.35 | FAIL |
| worst per-series ratio | <=1.50 | 1.93 | FAIL |
| factor-path recovery (rotation-insensitive) | >=0.70 | 0.770 (robust input) | PASS |
| loading-subspace recovery | >=0.75 | 0.478 | FAIL |
| coherence shrink gain | >=0.05 | 0.000 | FAIL (killed) |
| edge false positives (empty true set) | <10 | 72 | FAIL (killed) |

Nothing has had its default flipped. Promotion is not claimed: three
aggregate-level gates pass and the per-series gates do not, and the honest
reading is that the safety layer succeeded at bounding the *average* damage
and has not yet bounded the *worst series*.

---

# Loading-structure recovery ladder (C1-C8)

Follow-on to the superset rework, which ended with the Phase-4 coherence shrink
killed by its own gate — not because the solver is wrong but because the graph
it is handed is wrong. With the generator's **true** loading graph the identical
shrink delivered +0.57 trend-only coherence at -1.0% to -2.3% MASE; with the
fitted graph, gain 0.000 at every strength. The ladder's question: can the
fitted loading structure be made good enough for that shrink to pay off?

**Answer: no, and not for any reason on the ladder.** The rotation candidate
(C1) is *correct and powerful* — it reaches 0.97 dominant recovery and 0.93 pair
precision when handed a clean trend panel — but every candidate operating at or
after identification is blocked by an input panel that no longer carries the
loading structure. Nothing had its default flipped.

## Step 0 — the committed metric

`loading_structure_score` (`autots/evaluator/tva/metrics.py`) replaces the
ad-hoc 41/8/13% audit. Truth pairs are unordered `(i, j)` sharing a true
dominant factor **and** its sign, over series with nonzero true loading;
asserted pairs come from the caller's graph
(`coherence._graph_pairs(group_graph(...))`), so pair precision scores the
object the shrink actually consumes rather than a proxy for it. Reports
`pair_precision/recall/f1`, `n_pairs_asserted/true`, `dominant_recovery`,
`sign_agreement`, `matched_loading_corr`.

Two dominance definitions are reported because they diverge under rank
over-specification, and the plan's reference number used the charitable one:

- `dominant_recovery` — the estimated dominant column is mapped back to a true
  factor; a series dominated by a *spurious* extra column counts as a miss.
- `dominant_recovery_matched` — dominance judged only among matched columns.

They are equal whenever `n_est == n_true`. The fitted rank is 4.0 on the
primary cell against a true rank of 3, which is why the strict number (0.194)
sits below the plan's quoted 0.458 while the charitable one (0.278) is closer.
The gap is definitional, not a failure to reproduce: pair precision (0.247 vs
"~0.13") and sign agreement (0.578 vs "~0.49") land in the same
indistinguishable-from-chance regime the audit described.

Harnesses wired: `tva_factor_validation.py` (`graph_structure_scores`, two new
report tables including a K-misspecification tabulation),
`tva_coherence_diagnostic.py` (table iv-b), and a new focused sweep harness
`examples/tva_structure_ladder.py` — named config presets, one TVA fit per
(cell x seed x preset), structure + span + MASE only. Fast enough to sweep a
dozen variants; the full validation harness stays the regression check.

## Baseline lock (24 series, K=3, 3 seeds, latent)

| cell | pair prec | dominant | dominant(matched) | sign | loading corr | canon | MASE | K found |
|---|---|---|---|---|---|---|---|---|
| s0.9 n0.05 (primary) | 0.247 | 0.194 | 0.278 | 0.578 | 0.089 | 0.752 | 1.301 | 4.0 |
| s0.9 n0.15 | 0.278 | 0.306 | 0.306 | 0.542 | 0.008 | 0.671 | 2.133 | 3.0 |
| s0.7 n0.05 | 0.289 | 0.236 | 0.264 | 0.556 | 0.132 | 0.735 | 1.455 | 3.3 |
| s0.7 n0.15 | 0.279 | 0.264 | 0.278 | 0.444 | -0.065 | 0.726 | 1.157 | 3.3 |
| s0.5 n0.05 | 0.210 | 0.222 | 0.278 | 0.486 | 0.022 | 0.705 | 1.697 | 3.3 |
| s0.5 n0.15 | 0.317 | 0.264 | 0.333 | 0.444 | -0.026 | 0.722 | 2.917 | 3.7 |
| K-misspec (fit 6) | 0.249 | 0.278 | 0.333 | 0.519 | -0.039 | 0.842 | 1.373 | 5.0 |

Matched loading correlation is ~0 (and negative in three cells) everywhere.
Fitted rank differs from the true rank in **7/7 cells**, which would trigger
C7 — but C7 selects K, and K is not what is broken (see the root cause).

## The root cause: structure is destroyed before identification runs

The decisive measurement is a stage-attribution probe, primary cell, 3 seeds:
identify factors from each candidate input panel and score the loadings.

| input panel | rotation | dominant | sign | loading corr | pair prec |
|---|---|---|---|---|---|
| generator's true trend | none | 0.389 | 0.972 | 0.518 | 0.526 |
| generator's true trend | **varimax** | **0.972** | **1.000** | **0.873** | **0.967** |
| detector-adjusted (what TVA fits) | none | 0.264 | 0.556 | -0.004 | 0.253 |
| detector-adjusted | varimax | 0.347 | 0.486 | -0.044 | 0.269 |
| robust-adjusted | none | 0.250 | 0.506 | 0.009 | 0.272 |
| robust-adjusted | varimax | 0.236 | 0.478 | -0.024 | 0.274 |

Varimax on a clean input clears **every** structure gate with room to spare
(0.97 vs a 0.70 bar, 0.97 vs 0.50, 0.87 vs 0.50, 1.00 vs 0.75). On either real
input it is worth nothing. The rotation was never the missing piece it looked
like from the fitted-graph audit — it is the right mechanism waiting on an
input that carries the structure.

Second probe, which removes identification from the question entirely: hand
each panel the **true** factors and solve only for loadings by least squares.

| input panel | true-factor R² | dominant | sign | loading corr | pair prec |
|---|---|---|---|---|---|
| generator's true trend | 0.997 | 1.000 | 1.000 | 0.893 | 1.000 |
| raw observed data | 0.243 | 0.264 | 0.764 | 0.257 | 0.262 |
| detector-adjusted | 0.156 | 0.444 | 0.639 | 0.248 | 0.285 |
| robust-adjusted | 0.951 | 0.250 | 0.563 | 0.189 | 0.287 |

Even with the true factors supplied, no observable panel recovers the true
loadings. This is not an identification problem, a rotation problem, a rank
problem or a sparsity problem — the information is gone from the input.

Third probe, attributing the loss to a component (mean over 3 seeds):

| panel variant | true-factor R² | pair prec |
|---|---|---|
| raw | 0.243 | 0.262 |
| minus high-pass seasonality | 0.317 | 0.250 |
| minus high-pass seasonality + holidays + anomalies | 0.338 | 0.250 |
| minus level_shifts only | 0.097 | 0.266 |
| FULL `_build_adjusted_panel` | 0.156 | 0.285 |
| generator's true trend | 0.997 | 1.000 |

**Level-shift subtraction is the single largest loss**: it more than halves the
factor-explained variance (0.338 -> 0.156 in the full adjustment, 0.243 ->
0.097 on its own). The detector is classifying genuine shared factor movement
as per-series level shifts and subtracting it. But the loss is not only there:
pair precision is ~0.26 even on **raw, unadjusted data**, so seasonality and
noise at these generator settings are on their own enough to bury the
cross-sectional structure. Fixing level-shift absorption is necessary and, on
this evidence, not sufficient.

Note the robust input's R² of 0.951 with chance-level loading recovery: three
free coefficients fit any smooth 1095-point curve well, so a high R² there is
evidence of flexible curve-fitting, not of recovered shared structure. This is
also why its far better *span* metric (canonical correlation 0.770 vs 0.532)
never converted into better *basis* recovery.

## Candidate results (primary cell, 3 seeds; gates: prec >=0.50 at >=10 pairs, dominant >=0.70, loading corr >=0.50, sign >=0.75)

| candidate | config | pair prec | asserted | dominant | sign | loading corr | MASE drift | verdict |
|---|---|---|---|---|---|---|---|---|
| — | baseline | 0.247 | 45.3 | 0.194 | 0.578 | 0.089 | — | — |
| C1 | varimax | 0.305 | 40.0 | 0.111 | 0.577 | 0.102 | +1.15% | KILLED |
| C1 | quartimax | 0.252 | 51.0 | 0.083 | 0.562 | 0.143 | +3.19% | KILLED |
| C1 | promax | 0.263 | 46.7 | 0.194 | 0.562 | 0.072 | +1.19% | KILLED |
| C1 | varimax, no Kaiser | 0.264 | 45.0 | 0.194 | 0.480 | 0.069 | +2.72% | KILLED |
| C2 | varimax + margin 1.5 | 0.327 | 14.0 | 0.111 | 0.577 | 0.102 | +1.15% | KILLED |
| C2 | varimax + margin 2.0 | 0.361 | **4.3** | 0.111 | 0.577 | 0.102 | +1.15% | KILLED |
| C2 | varimax + share 0.5 | 0.285 | 30.0 | 0.111 | 0.577 | 0.102 | +1.15% | KILLED |
| C2 | varimax + share 0.6 | 0.334 | 15.0 | 0.111 | 0.577 | 0.102 | +1.15% | KILLED |
| C3 | varimax + l1 0.03 | 0.321 | 38.3 | 0.181 | 0.577 | 0.076 | +5.12% | KILLED |
| C3 | varimax + l1 0.1 | 0.272 | 44.7 | 0.167 | 0.575 | 0.076 | -2.63% | KILLED |
| C3 | + prox 1.0 | 0.201 | 49.0 | 0.333 | 0.646 | 0.123 | -7.11% | KILLED |
| C4 | varimax + stability veto 0.5 | n/a | **0.0** | 0.111 | 0.577 | 0.102 | +1.15% | KILLED |
| C5 | robust structure input | 0.259 | 75.7 | 0.194 | 0.581 | -0.004 | 0.00% | KILLED |
| C5 | varimax + robust structure | 0.298 | 44.0 | 0.236 | 0.538 | 0.053 | +1.15% | KILLED |
| C6 | w_decorr 0.1 | 0.247 | 45.3 | 0.194 | 0.578 | 0.089 | 0.00% | KILLED |
| C6 | varimax + w_decorr 0.1 | 0.305 | 40.0 | 0.111 | 0.577 | 0.102 | +1.15% | KILLED |

Per-candidate notes:

- **C1** fails its own kill rule twice over (loading corr 0.102 < 0.3, dominant
  0.111 < 0.55), and promax — the specified fallback — is no better. The
  canonical-correlation move (-0.084, i.e. canon *rose*) exceeds the 0.02
  "that's a bug" threshold, but it is not one: the rotation is exact at the
  parameter copy (verified at 4.5e-7 relative, and asserted as a test), and the
  reported figure is `match_factors` column matching, which is rotation-
  sensitive by construction and rises with the fitted rank (4.0 -> 4.3).
- **C2** does exactly what it was designed to do — precision rises monotonically
  with the margin — but it buys 0.247 -> 0.361 by shrinking the graph from 45
  asserted pairs to 4.3. Its kill rule requires >=0.50 precision at >=10 pairs;
  the best setting reaching 10+ pairs is 0.334. Abstention cannot manufacture
  precision that isn't in the loadings.
- **C3** never reaches the 0.35 precision bar without breaching the MASE guard
  (l1 0.03 gives 0.321 at +5.12%). The prox companion is the one candidate that
  moves dominance and sign in the right direction (0.111 -> 0.333, 0.577 ->
  0.646) at a *better* MASE (-7.11%), but it does so by collapsing the fitted
  rank to 2.7 and it drops pair precision to 0.201. Recorded as an unexplained
  positive worth revisiting only after the input is fixed. Note the useful prox
  range is ~O(1), not the ~1e-3 the plan proposed: the threshold is
  `w_prox_loadings * lr_aux` against loadings of order 0.3 on a normalized
  panel, so the planned {1e-4, 1e-3, 1e-2} grid is entirely inert.
- **C4** vetoes *every* factor on this cell (0 pairs asserted at
  `min_sign_confidence=0.5`). Correctly conservative — no factor here is
  reproducible across split halves — but it cannot discriminate real from
  spurious columns when none are stable, so its K-misspec separation rule is
  unreachable.
- **C5** is the informative kill. Its first measurement was wrong: the ladder
  scored `get_factors()['loadings']` while C5 substitutes loadings only inside
  `_apply_coherence`, so the robust presets initially came out bitwise identical
  to baseline. After exposing `structure_loadings` through `get_factors()` and
  scoring it, the honest number is 0.259 vs 0.247 — a +0.012 gain against a
  +0.10 bar. The agreement guard behaved as designed (0.426 / 0.346 / 0.683 over
  the three seeds; it correctly fell back on the middle one).
- **C6** has *exactly* zero effect — the rows are identical to their baselines.
  This is the outcome the plan predicted and wrote down in advance: an
  orthogonal rotation of decorrelated factors stays decorrelated, so a
  decorrelation penalty has nothing to act on. It earned its slot by being free.
- **C7** (stability K selection) was triggered by the diagnostic (rank wrong in
  7/7 cells) but not built: the probes show K is not the binding constraint, and
  selecting a better K on an input that carries no loading structure changes
  nothing.
- **C8** (bootstrap consensus dominance) not built, for the same reason.

## Gates not reached

The evaluation protocol runs later gates only after earlier ones pass. Gate 1
(synthetic structure) failed for every candidate, so:

- the **Phase-4 coherence-shrink re-test** was not run — there is no winning
  stack to re-test, and the fitted graph is unchanged in kind from the one that
  already produced gain 0.000;
- the **LRP iteration folds** (4a/4b) were not run, and the **promotion folds
  remain untouched**;
- **no default was flipped.** `rotate`, `loading_l1`, `w_prox_loadings`,
  `factor_stability_reps`, `structure_input`, `dominance_margin` and
  `min_loading_share` all ship default-off/no-op, verified by tests that assert
  bitwise-identical output against the current behavior.

## What is worth doing next

The ladder's negative result is sharp enough to point somewhere specific.
Ranked by the evidence above:

1. **Stop the detector from absorbing shared factor movement into
   `level_shifts`.** This is the largest single measured loss (R² 0.338 ->
   0.156) and the only one attributable to a specific component. A shared
   step across many series is a factor move, not N independent level shifts;
   the detector currently has no cross-series view when it makes that call.
2. **Re-run this exact ladder afterwards.** The oracle-panel row is the target,
   and C1 is already implemented, tested and one config flag away. If the input
   fix lands, the expected payoff is the +0.57 coherence at -1.0% to -2.3% MASE
   that the true-graph shrink already demonstrated.
3. **Do not** retry C2/C4/C6 as-is, and do not retry C5 on the current robust
   estimator. Revisit C3's prox only with the O(1) range recorded above.

## Sparse-code identification (C9) and the level-shift veto (I1)

**Answer: the mechanism works, the input attribution was wrong, and the
blocker is now isolated by elimination.** C9 is the
first candidate in this ladder to clear all four structure gates on any panel
(oracle input, `sparse_alt`: 0.792 / 0.778 / 0.605 / 0.894), it beats varimax
there on every metric, and it improves synthetic MASE by 4-9% on the *real*
panel. It does not clear the gates on the detector-adjusted panel. I1 is dead
twice over: the veto never fires, and the unconditional control shows the thing
it was built to fix is not the blocker. Nothing had its default flipped.

### What C9 is

Series are samples; a series' loading vector *is* its sparse code; the
dictionary is parameterized directly in the hinge trend basis, so `coefs` falls
out and the identification contract is matched by construction:

```
F = (design @ C) / std(diff)          # (T, K) atoms
L = signed_topk(Z, k)                 # (N, K) loadings == the codes
Yhat = F @ L.T + level + slope * t
```

Sparsity is learned jointly with the span rather than rotated in afterwards. A
hard support constraint is not rotation-invariant, so the basis is pinned with
no rotation step at all, and atoms nobody selects fall out -- making the live
atom count an implicit rank estimate. Two tiers: `sparse_alt` (torch-free
coordinate descent, re-selects support every iteration) and `sparse_ae`
(gradient autoencoder over free codes, support locks in after warmup).

### Oracle input: the first passing row on this ladder

Primary cell, 3 seeds, TVA fit on the generator's true trend panel
(`--input oracle`):

| config | pair prec | dominant | loading corr | sign | MASE drift | verdict |
|---|---|---|---|---|---|---|
| baseline | 0.611 | 0.403 | 0.189 | 0.778 | — | fail |
| varimax | 0.564 | 0.556 | 0.469 | 0.889 | +1.53% | fail |
| **sparse_alt** | **0.792** | **0.778** | **0.605** | **0.894** | **-14.50%** | **PASS** |
| sparse_ae | 0.656 | 0.514 | 0.571 | 1.000 | +2.53% | fail |

This is the load-bearing result. It is not that sparsity beats rotation
everywhere -- it is that the mechanism is *sufficient* when the input carries
the structure, which no C1-C8 candidate demonstrated end-to-end through TVA.

### Detector-adjusted input: still short, but the largest move measured

Primary cell, 3 seeds. Gates: prec >=0.50 at >=10 pairs, dominant >=0.70,
loading corr >=0.50, sign >=0.75.

| config | pair prec | asserted | dominant | sign | loading corr | MASE drift |
|---|---|---|---|---|---|---|
| baseline | 0.247 | 45.3 | 0.194 | 0.578 | 0.089 | — |
| varimax (C1) | 0.305 | 40.0 | 0.111 | 0.577 | 0.102 | +1.15% |
| k3 (forced rank) | 0.283 | 62.0 | 0.278 | 0.562 | 0.016 | -4.91% |
| sparse_alt | 0.290 | 28.3 | 0.194 | 0.917 | 0.185 | **-6.36%** |
| sparse_ae | 0.261 | 36.0 | 0.153 | 0.857 | 0.021 | -4.51% |
| sparse_ae, k=2 | 0.267 | 45.7 | 0.194 | 0.594 | 0.020 | -4.20% |
| sparse_ae, no idio | 0.354 | 47.3 | 0.333 | 0.778 | 0.088 | **-8.93%** |
| sparse_ae, no AuxK | 0.261 | 36.0 | 0.153 | 0.857 | 0.021 | -4.51% |
| sparse_ae, no support freeze | 0.329 | 42.3 | 0.139 | 0.578 | 0.001 | -5.49% |
| **veto_all + sparse_alt** | **0.385** | 35.0 | **0.361** | 0.820 | 0.213 | +2.82% |

Four things this settles:

1. **The support projection is load-bearing.** Turning it off (`no support
   freeze`) drops sign agreement 0.857 -> 0.578 and loading correlation
   0.021 -> 0.001, i.e. back to baseline. Stage A's `w_l1_loadings` is a
   subgradient term that never reaches zero, so without projecting back onto
   the identified support the sparsity never reaches `fitted_loadings()` --
   which is what `_apply_coherence` actually reads.
2. **AuxK is inert on this panel.** `no AuxK` is bitwise identical to the
   default: no atom stays dead long enough to trigger revival. The knob earns
   its place only on the clean panel, where collapse is real (see below).
3. **The idiosyncratic line hurts here.** `no idio` is the best pure-C9 row
   (0.354 / 0.333 vs 0.261 / 0.153). That contradicts the design argument for
   including it, which was made on a clean fixture where it measured +0.09
   loading correlation. On the detector-adjusted panel the line competes with
   the factors instead of protecting them. Default stays `idio: True` because
   the oracle result depends on it, but the two panels genuinely disagree and
   the knob should be swept, not assumed.
4. **K over-specification is closed.** Forcing the true rank (`k3`) moves pair
   precision 0.247 -> 0.283 and dominance 0.194 -> 0.278 -- real but nowhere
   near the gates, confirming the C1-C8 conclusion that rank is not the binding
   constraint. C7 remains correctly unbuilt.

**Read the sparse rows' sign agreement with care.** `sign_agreement`'s
denominator is the rows whose estimate is nonzero at the *true* dominant
column; for a 1-sparse fit that is exactly the set of rows whose dominance is
already correct. So `sparse_alt`'s 0.917 is computed over roughly
`0.194 x 24 ~ 5` series, not 24, and answers "given the right factor, was the
sign right" -- a strictly easier question than the one baseline's 0.578
answers. It is reported (`sign_usable_approx`, `mean_nonzeros`) rather than
compared against the dense C1-C8 rows. The `zero_rows` channel of the same
hazard turned out inert (0.3/24 across all configs).

### I1: the level-shift veto is dead, and so is the attribution behind it

The prior write-up's #1 recommendation was to stop the detector absorbing
shared factor movement into `level_shifts`, on the strength of a component
attribution showing true-factor R^2 dropping 0.338 -> 0.156 when they are
subtracted. Both halves fail on measurement.

**The veto cannot fire.** On the primary cell the detector emits 38 step events
across 24 series over 1095 days, and they are not co-timed:

| pooling window | max co-stepping series | share of panel |
|---|---|---|
| 1 day | 2 | 8% |
| 7 days | 4 | 17% |
| 31 days | 6 | 25% |
| 61 days | 7 | 29% |

A shared-event rule needs a quorum that never occurs. `veto` and
`veto_s0.1_w31` are both bit-identical to baseline on every metric.

**And the subtraction is not the blocker anyway.** `veto_all` subtracts no
level shift whatsoever -- the unconditional control the attribution implies
should help most:

| config | pair prec | dominant | loading corr | canon | MASE drift |
|---|---|---|---|---|---|
| baseline | 0.247 | 0.194 | 0.089 | 0.752 | — |
| veto_all | 0.202 | 0.208 | 0.179 | **0.859** | -9.36% |
| veto_all + varimax | 0.234 | 0.208 | 0.090 | 0.859 | -1.88% |
| veto_all + sparse_alt | 0.385 | 0.361 | 0.213 | 0.864 | +2.82% |

Pair precision goes *down* (0.247 -> 0.202). Dominance and loading correlation
move up slightly. Nothing approaches the gates. **Level-shift subtraction is
not the single largest loss in the pipeline** -- at least not measured end to
end through TVA against the committed metric, which is the measurement that
matters. Two candidate explanations for the discrepancy with the recorded R^2
attribution, neither verified here: the detector's `level_shifts` component is
dominated by a per-series *constant* (`shifts[0]` reaches 31.3 on this cell)
which `robust_level_scale`'s median centering removes exactly, so an uncentered
R^2 would attribute variance to it that the factor fit never sees; and R^2
against supplied true factors rewards fit, not basis recovery, which is the
same confound already recorded for the robust input's 0.951.

What `veto_all` *does* buy is span: canonical correlation 0.752 -> 0.859, the
best on the detector panel, at -9.4% MASE. That is worth keeping in view even
though it does not convert into basis recovery.

### P1: the gates are not SNR-limited either

Sweeping the generator's `noise_scale_mode='trend_delta'` dial, which scales
observation noise to the trend-increment std, over a 16x range (3 seeds each):

| noise (trend-increment units) | baseline | varimax | sparse_alt |
|---|---|---|---|
| 0.5 | 0.268 | 0.333 | 0.290 |
| 2.0 | 0.283 | 0.243 | 0.292 |
| 8.0 | 0.253 | 0.237 | 0.239 |

Pair precision is **flat**. A 16-fold reduction in observation noise buys
nothing, and the cleanest cell still tops out at 0.333 against a 0.50 gate. So
the ceiling is not an SNR ceiling, and re-basing the gates on SNR grounds --
which is what this probe was run to test -- is not justified. The gates stay
where they are.

Caveat on the accuracy claim: `sparse_alt`'s MASE is unstable on these cells
(+100% at noise 0.5, -15% at 2.0, +69% at 8.0). The consistent -4% to -9% gain
holds on the primary cell and does not generalize to `trend_delta`-scaled
panels. Treat the accuracy result as cell-specific until it is tested on LRP.

### Where the blocker actually is

Four independent controls now agree that the usual suspects are not
responsible, and they converge on one place:

| suspect | control | result |
|---|---|---|
| rank over-specification | `k3` (force true K) | 0.247 -> 0.283. Not it. |
| level-shift absorption | `veto_all` (subtract none) | 0.247 -> 0.202. Not it. |
| observation noise | 16x SNR sweep | flat at ~0.27. Not it. |
| identification / basis | `sparse_alt` on oracle input | 0.247 -> **0.792, all gates pass**. Not it. |

The estimator is sufficient, the rank is close enough, the level shifts are
irrelevant and the noise is irrelevant -- yet the oracle panel scores 0.792 and
the observed panel scores 0.29. What separates them is the *decomposition's
trend estimate itself*, which the earlier work already measured at ~0.50
correlation against an oracle ceiling of 0.90+, and described as "structured,
not high-frequency noise". That is the remaining candidate, and it is the one
thing none of these controls isolates.

### Cost and reliability

| tier | identification cost | full TVA fit |
|---|---|---|
| baseline | — | 11.7 s |
| `sparse_alt` | ~14 s (3 restarts x ~4.7 s) | 25.5 s |
| `sparse_ae` | ~2.3 s (3 restarts) | 13.2 s |

`sparse_alt` roughly doubles the fit; `sparse_ae` costs ~13%. The torch tier is
6x cheaper and reaches comparable loading correlation, but `sparse_alt` is the
one that clears the oracle gates, so the dependency does not yet pay for
itself on quality.

**Known failure mode, measured and unfixed.** On a well-posed fixture
(orthogonal factors, equal amplitude, 92-99% hinge-representable), 4 of 5 seeds
recover the true rank; the remaining seed reproducibly merges two orthogonal
factors onto one atom at every restart schedule tried, and the acceptance guard
does *not* catch it because the merged fit reconstructs marginally better.
`n_atoms_live` is what surfaces it and is carried out to `info`. Restarts from
different initial rotations fix basin-dependent collapses (measured: a varimax
start collapsing to `[8 16 0 0]` at loss 4.01 while the quartimax start
recovered `[8 8 8 0]` at loss 2.28 -- reconstruction loss ranks them correctly),
but not this one.

### Defaults, and what is worth doing next

Everything ships default-off: `identification='alternating'`,
`level_shift_veto=False`, verified bitwise-identical to the previous behavior
through `fit_latent_factor_model` and through `TVA`. Gates 3-5 of the standing
protocol (Phase-4 coherence re-test, LRP folds, default flips) were not run,
because gate 1 did not pass on the panel TVA actually fits.

1. **Attack the decomposition's trend estimate, not the estimator and not the
   component subtractions.** That is the only suspect the four controls above
   leave standing, and C9 is already implemented, tested and one config flag
   away from re-scoring the moment it improves -- exactly the position C1 was
   left in, but now with a candidate that demonstrably converts on a clean
   input rather than one that inverts on a dirty one.
2. **Sweep `idio` rather than assume it.** It is worth +0.09 loading
   correlation on a clean fixture and -0.09 pair precision on the real panel.
3. **Do not retry** the co-timing veto, `sparse_ae` with `aux_k=0` (inert), or
   `code_topk=2` (worse than k=1 on every panel measured: 0.267 vs 0.290
   precision, and it re-enables the C2 abstention knobs at the cost of the
   sparsity that made the basis identifiable).
4. C9's accuracy effect (-4% to -9% synthetic MASE, consistently, across every
   variant) is larger and more reliable than its structure effect and has not
   been tested on LRP at all. That is a separate and possibly more valuable
   question than the one this ladder was built to answer.

---

# Five-arm network benchmark and the search surface (2026-08-18)

Every `trend_network` value measured on the same protocol for the first time,
including `v1`, which had never been benchmarked at all. Synthetic is the full
5-dataset x 2-horizon x 4-fold protocol; LRP is the three iteration folds at
H=180, base-tagged series (promotion folds 3-4 untouched). One seed per cell.
`factor:autoencoder` is `trend_network='factor'` with
`identification='sparse_ae'` and otherwise stock defaults.

| model | synthetic skill vs SN (higher better) | LRP MASE ratio vs SN (lower better) | synthetic dir-coh err | LRP dir-coh err | synthetic fit s | LRP fit s |
|---|---|---|---|---|---|---|
| SeasonalNaive | 1.000 | 1.000 | 0.178 | 0.458 | 0.01 | 0.09 |
| `factor:autoencoder` | **0.507** | 2.387 | **0.150** | **0.185** | 22.3 | 39.1 |
| `factor` | 0.426 | **2.215** | 0.155 | 0.232 | **10.5** | **19.2** |
| `v2` | 0.338 | 4.433 | 0.178 | 0.137 | 132.6 | 336.8 |
| `none` | 0.338 | 4.482 | 0.176 | 0.185 | 5.8 | 11.8 |
| `v1` | 0.338 | 4.487 | 0.180 | 0.137 | 149.1 | 663.7 |

Synthetic skill is the geometric mean across the five datasets; LRP is
`mase_base / SeasonalNaive mase_base`. Note the two accuracy columns run in
opposite directions. `n_failed = 0` for every TVA arm on both benchmarks.

**The LRP `factor` row is plain `TVAModel` defaults, not arm F.** 2.215 here
reproduces the "unblended factor: 2.22" row from the Phase 1-2 table exactly,
which is the internal-consistency check that the run is sound. Arm F's 1.011
needs `sn_blend` + `inner_refit` + tie tolerance 0.25 + `seasonal_arbitration`
+ `space='log'`, none of which any of these five arms sets.

**`v1` is inert and the most expensive arm ever measured here.** Its synthetic
skill (0.338) is identical to `none` and `v2` to three decimals, its LRP ratio
(4.487) is the worst of the three, and it costs 663.7s per fit against 11.8s
for `none` — i.e. it buys nothing for 56x the cost of doing nothing. The
already-recorded inertness of `v2` extends to `v1`.

**C9's accuracy effect does not transfer to LRP** — the open question left by
the C9 write-up, now answered. On synthetic the autoencoder is the best arm
(0.507 vs 0.426) and wins coherence too, consistent with the -4% to -9% MASE
measured across every C9 variant. On LRP it is *worse* than plain `factor`
(2.387 vs 2.215) while still winning coherence (0.185 vs 0.232). Structure and
accuracy separate again, in the same direction 0h found with the pooled
comparator.

**`load_daily` is where both factor arms regress**, ~8x SeasonalNaive
(`factor` 8.104, autoencoder 7.537) against ~3.66x for `none`/`v1`/`v2` — the
known trendless-series blowup. They still win the aggregate because they lead
the other four datasets by 2-3x.

Per-dataset MASE ratio vs SeasonalNaive:

| dataset | SN MASE | none | v1 | v2 | factor | autoencoder |
|---|---|---|---|---|---|---|
| factor_panel | 1.165 | 1.715 | 1.710 | 1.709 | 1.270 | **1.225** |
| synthetic_factor_panel | 0.978 | 2.710 | 2.721 | 2.710 | 1.461 | **1.285** |
| load_daily | 0.926 | 3.661 | 3.661 | **3.659** | 8.104 | 7.537 |
| load_monthly | 0.805 | 3.567 | 3.569 | 3.570 | 2.482 | **2.251** |
| load_artificial | 3.634 | 3.439 | 3.440 | 3.440 | 2.234 | **2.149** |

## What this changed in the search surface

`TVAModel.get_new_params` only ever exposed `trend_network`, `n_factors`,
`factor_knot_spacing` and `factor_max_lag` from the factor path. Every knob the
Phase 1/2 sweep found — the whole 2.22 -> 1.011 gain — was unreachable by the
optimizer. Three changes, no library defaults touched:

- **`v1` is no longer sampled.** Still constructible; just never worth a fit.
- **`v2` drops to a 0.10 rare escape hatch**, `factor` rises to 0.60, `none`
  holds 0.30.
- **`factor_config` is now sampled in factor mode** by
  `TVAModel._new_factor_config`, profile-anchored on arm F rather than
  independent per key. Independent sampling would reach that corner rarely and
  would draw inert combinations (a blend tie tolerance does nothing without
  `sn_blend`; `inner_refit` is meaningless without inner folds to grade). Five
  profiles: `arm_f` 0.32, `arm_f_sparse` 0.23, `safety` 0.15, `explore` 0.15,
  `default` 0.15, where `default` returns `None` and keeps today's behavior
  reachable as the comparator. `sparse_alt`/`sparse_ae` enter through the
  sparse profiles and at 0.15 each elsewhere.

Deliberately **not** sampled, each on a measured result above: `space='auto'`
(1.166 vs 1.011), `blend_risk_weight` (worst-series moves the wrong way),
`coherence` (shrink gain 0.000), `level_shift_veto` (never fires), and
`code_topk=2` (do-not-retry list). All remain settable by hand.

Cost, on a 12-series/500-day panel: plain 4.4s, arm F 7.2s, arm F + `sparse_alt`
9.3s, arm F + `sparse_ae` 7.6s, and every optional knob at once 47.3s. The
expensive tail is `group_factors` (0.12, never in fast mode), `structure_input`
(0.15) and the 3-restart `init_rotate` — all low-weight for that reason.

### Non-autoencoder knobs added to the search (same day)

Four more, all cheap (no extra fits) and none previously reachable:

| knob | default | grid | why |
|---|---|---|---|
| `min_trend_to_noise` | 0.6 | 0.0-1.5 | zeroes loadings of series with no low-frequency structure — the direct lever on the `load_daily` 8x blowup, which is a trendless series handed factor exposure. Pinned since the mode was written, never swept. |
| `prune_share` | 0.02 | 0.0-0.10 | factor exposure-share floor. Fitted rank was 4.0 against a true 3 in 7/7 ladder cells and bad rank selection is what blocked the robust input estimator. |
| `alpha` | 1e-3 | 3e-4 - 1e-2 | l1 trend-filter smoothness on the factor paths, i.e. how smooth the thing being extrapolated is. Scale-invariant by construction (columns standardized before the Lasso, rescaled after), so it is safe across arbitrary panels. |
| `n_factors` | reweighted | — | `'auto'` drops 0.50 -> 0.35; explicit small K takes the mass, on the same 4.0-vs-3 evidence. |

Verified across the 2x2x2 extreme corners on three panel shapes (trended,
trendless, flat): 24/24 finite, no raises. The gate is measurably live rather
than nominal — on the trendless panel `min_trend_to_noise=1.5` narrows the
28-step forecast band from [43.0, 52.8] to [48.8, 51.2], which is the
suppression `load_daily` needs. `prune_share` showed no effect on these
fixtures (K=3 fit on a K=3 panel leaves every factor above the floor); it is
not inert in general, just unexercised there.

**Deliberately still not sampled**, each on a standing kill or do-not-retry
entry: `lr_coef > 0`, `loading_l1` (C3, +5.12% MASE), `w_prox_loadings` (the
C5 measurement trap), `factor_stability_reps` (C4), `coherence`,
`level_shift_veto`, `space='auto'`, `blend_risk_weight`, `code_topk=2`.

---

## Forecast covariance, MinT wiring, and factor-mode scenarios (2026-08-19)

Harness: `examples/tva_reconciliation_gate.py`, 10 seeds, 24-series/1095-day
latent-factor panel from `SyntheticDailyGenerator` grouped into a
`global -> factor_k -> series` hierarchy by true dominant factor, H=28,
`trend_network='factor'`. Raw rows: `tva_reconciliation_gate.json`.
Graded by `examples/tva_scorecard.py` (two new gate entries).

### Two findings that changed the shape of the work

**1. `TVA.reconcile()` was a no-op, not "OLS under the name MinT".** When it is
handed a bottom-level-only forecast it builds the aggregate rows as `S @ bottom`,
which places its input *exactly* in the coherent subspace MinT projects onto.
`S(S'W⁻¹S)⁻¹S'W⁻¹` then returns it unchanged for **every** `W`, in every trend
mode — v1/v2 included, where a real residual matrix was reaching the bridge all
along. The `W = I` fallback in factor/`none` mode was real but had no symptom
because there was nothing to reconcile. Every in-library caller is on this path.

**2. `W = S Σ Sᵀ` cancels Σ out of MinT.** The prescribed expansion is
rank-deficient (rank M < L), so it needs a ridge; but as the ridge → 0,
`(S'W⁻¹S)⁻¹S'W⁻¹ → (S'S)⁻¹S'`, which is precisely the OLS reconciler `W = I`
already gives. Verified to 1e-12 (`test_s_sigma_st_alone_reconciles_identically_to_identity`).
Σ survives only when the aggregate nodes carry error that is *not* the
aggregated bottom error — i.e. when the aggregate forecast was produced
independently. `TVA.reconcile` therefore gained `aggregate_sigma`, and
`W = S Σ Sᵀ + diag(ψ_agg, 0)`.

The gate harness supplies that configuration: an independent damped-trend model
on the aggregate history, with its own backtested sigma.

### Arms (mean over 10 seeds, MASE lower is better)

| arm | MASE (all nodes) | MASE (bottom) | MASE (agg) | coherence err | ‖S·b − a‖ |
|---|---|---|---|---|---|
| unreconciled | 1.4705 | 1.4174 | 1.7890 | 1.8009 | 781.18 |
| MinT, `W = I` (previous) | 1.6366 | 1.6221 | 1.7234 | 0.0000 | 0.00 |
| **MinT, structural Σ** | **1.3852** | 1.4174 | **1.1918** | 0.0000 | 0.00 |
| control: variance-only W | 1.4087 | 1.4258 | 1.3061 | 0.0000 | 0.00 |

| gate | value | threshold | status |
|---|---|---|---|
| `reconciliation_mase_ratio_aggregate` | 0.8464 | ≤ 1.00 | **PASS** |
| `reconciliation_coherence_error_ratio` | 0.0000 | ≤ 0.999 | **PASS** |

Per-seed MASE ratios: 0.924, 0.866, 1.043, 0.939, 1.031, 0.523, 0.845, 0.850,
0.934, 0.908 — 8/10 wins, median 0.916, worst 1.043.

**The control matters.** Deleting Σ's off-diagonals costs only 1.385 → 1.409
(1.7%). Most of the 1.637 → 1.385 gain is ordinary variance weighting: the
structural W tells MinT the independently-forecast aggregates are the unreliable
level, so it leaves the bottom forecast untouched (structural `mase_bottom` is
bitwise the unreconciled one) and pulls the aggregates to `S·b`. That is close
to bottom-up reconciliation, and it wins here because the TVA bottom model is
much better than the aggregate one. The cross-series covariance adds a further
1.7% on top. Both are real; only the second is attributable to Σ.

The coherence gate is read against the **unreconciled** arm, not the identity
arm: both MinT arms are exactly coherent by construction (the projection *is*
MinT), so a structural-vs-identity coherence ratio is 0/0 and grades nothing.

### Default flipped

`RECONCILIATION_COVARIANCE_AUTO = 'structural'` (was `'identity'`). Note the
scope: on the synthesized-aggregate path — every in-library caller — the
structural W is not even assembled, because no `W` can move already-coherent
input and building one would be pure cost. This setting bites only for a caller
passing a full L-column forecast frame with independently-produced aggregates.

### Covariance diagnostics (10 seeds)

| quantity | min | median | max |
|---|---|---|---|
| `alpha` (Ledoit-Wolf shrinkage intensity) | 0.0039 | 0.0076 | 0.0106 |
| `beta` (structural-target scale) | 0.0000 | 35.16 | 315.12 |
| floor-binding fraction | 1.00 | 1.00 | 1.00 |
| mean \|off-diagonal corr\| | 0.128 | 0.144 | 0.188 |
| max \|off-diagonal corr\| | 0.525 | 0.587 | 0.676 |

Two of these are worth reading as results rather than telemetry. `alpha` sits
under 0.011 on every seed, so Σ is essentially the shrunk empirical covariance
and the low-rank `Λ Σ_f Λᵀ` target contributes almost nothing — consistent with
the standing finding that the factor decomposition is the weak stage. And the
residual floor binds on **100% of series on every seed**: Σ's diagonal is
entirely the decomposition-derived sigma, and only its correlations come from
the model's own rolling-origin residuals. The floor is not a rare safety net
here, it is the whole variance estimate. That is the 2-dof cap on the
idiosyncratic term showing up exactly where `factor_variance_share = 1.00`
predicted it would.

### What_if in factor/`none` mode (bug fix, not a measurement)

`what_if()` raised in both default forecasting modes — `BifrostOptimizer`
dereferences `tva._network`, which is `None` there. Fixed with a sibling
closed-form solver (`ClosedFormScenario`): those forecasts are linear in the
factor paths, so the minimum-disruption update is a Gaussian conditioning solve
`δ = Σ Aᵀ (A Σ Aᵀ)⁻¹ (b − A ŷ)` against the same Σ. Deterministic, torch-free,
no Adam steps, and cross-series aware instead of a proportional top-down split.
`apply_hierarchical_adjustment` keeps the proportional split as its fallback
when Σ is unavailable.

v1/v2 are untouched: `predict()`, `reconcile()`, and both `what_if()` forms are
**bitwise identical** to commit 76206eb (checked in a worktree, 8/8 arrays).

## LRP-review items: origin anchor, forced continuation, fit horizon (2026-10-06)

Arms on top of the factor default (`factor_config={}`), `examples/tva_benchmark.py`
built-in datasets, 4 folds. Geometric-mean MASE ratio vs SeasonalNaive (<1 better):

| arm | factor_panel | load_artificial | load_daily | load_monthly | synthetic_factor |
|---|---|---|---|---|---|
| factor default | 1.302 | 18.06 | 18.90 | 2.680 | 1.602 |
| + continuation_force='constant' | 1.299 | 18.09 | 18.84 | 2.647 | 1.578 |
| + origin_anchor='last_value' | 0.868 | 1.864 | 1.108 | 0.840 | 1.033 |
| + origin_anchor='deseasonalized' | 0.822 | 2.303 | 1.812 | 0.957 | 1.204 |
| + last_value + constant | 0.863 | **1.169** | **1.087** | **0.838** | **1.029** |
| + deseasonalized + constant | **0.821** | 1.451 | 2.366 | 0.957 | 1.198 |
| TVA 'none' | 0.960 | 1.787 | 1.819 | 1.830 | 1.404 |

LRP (`--data lrp_forecast_data_202502.csv --horizon 180`, 3 iteration folds):
- On the factor default, last_value + constant takes MASE skill 0.002 -> 0.69.
- On the best known config (log space + sn_blend tol 0.25 + inner_refit) the new
  options are neutral-to-worse on the two usable folds (fold 0: best 4.735,
  +last_value 4.976, +deseasonalized 10.30, +fixed reanchor 4.545; fold 2: all
  ~= SeasonalNaive 2.710).
- **Fold 1 overflows in log space for every arm, including the unmodified best
  config on clean HEAD** (MASE ~1e303): an older bug, not from these changes.
