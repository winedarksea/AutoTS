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
