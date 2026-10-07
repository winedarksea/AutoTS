# TimeSeriesFeatureDetector: trend, changepoint and event-labeling improvements

**Audience:** AutoTS maintainers
**Code tested:** `autots/evaluator/feature_detector/` and `autots/tools/changepoints.py`, from the vendored copy `fbsource/third-party/pypi/autots/1.0.4/sources` (numpy 2.2, pandas 2.x, scikit-learn, opt build). Please re-check line numbers against upstream HEAD.
**Status:** every finding below was reproduced by running code. Each proposal was prototyped as a subclass or monkeypatch and benchmarked. No upstream files were modified.

## Why this matters

Downstream use: decompose long-range planning metrics into business impacts, seasonality and holidays, and trend. Then detect past trend changepoints and level shifts, and have an LLM label each one with a known story (launches, news, manual labels). That only works if:

1. a detected changepoint corresponds to a real change in trend, not an artifact;
2. its date, size and duration are accurate enough to match against stories; and
3. events shared across metrics show up as one event.

On a pure linear trend with no changepoints, the current default pipeline:
- reports several trend changepoints, mostly in the first week of January;
- reports a level shift roughly 3× too large;
- underestimates the trend slope by about 90%.

Each of these becomes a false "story" to label.

## Summary

| # | Finding | Severity | Status | Proposed fix | Benchmarked? |
|---|---|---|---|---|---|
| F1 | Final seasonality fit absorbs the trend into a yearly sawtooth with a discontinuity near Jan 1 | Critical | Confirmed | Remove trend and level shifts jointly with seasonality before the seasonality fit | Yes |
| F2 | Default PELT-`l2` finds level steps, not slope changes: 6 false positives on a pure linear trend, 0% recall of real slope changes | Critical | Confirmed | Hinge-basis selection with a BIC stop, plus relocation and pruning | Yes |
| F3 | Trend refit fits each segment's slope independently and can't represent level jumps. `LevelShiftMagic` magnitude = true step + slope × window | Critical | Confirmed | One joint least-squares fit over hinges and steps | Yes |
| F4 | `_validate_level_shifts` accepts 99% of false candidates | High | Confirmed | Step t-test inside the joint model (AR(1)-adjusted) | Yes |
| F5 | Event DAG families depend on series units; slope deltas are added to level magnitudes; unrelated patterns merge | High | Confirmed | Scale-normalized, horizon-converted signature without `source_mix` | Yes |
| F6 | Changepoint records have no date window or effect size. The existing `changepoint_confidence_` is unused and covers the true date only 65% of the time | Medium | Confirmed | Profile-likelihood date window (96% coverage) plus richer record fields | Yes (window) |
| F7 | A shared event fragments into about 3.7 Event DAG clusters at the default `cluster_window_periods=1` | Medium | Confirmed | Default window of 14 for daily data, or cluster by overlapping date windows | Yes (window) |
| F8 | No day-of-year recurrence check. Seasonal leakage shows up as yearly repeating events | Low (once F1 and F3 are fixed) | Confirmed | Diagnostic tag only, not a filter | Yes |
| B1 | `_solve_weighted_trend_system_batch` raises under NumPy ≥ 2. The error is swallowed, so L1 methods never filter in individual mode | Bug | Confirmed | `solve(A, chunk[..., None])[..., 0]` | Yes |
| B2 | L1 changepoint extraction uses a scale-free threshold (mean + 1.5·std), so it always flags a roughly fixed share of points | Bug | Confirmed | Absolute threshold relative to noise | Partially |
| B3 | `fit()` crashes on NaN-containing series (e.g. the generator's own `business_day` type) in `_classify_anomaly_type` | Bug | Confirmed | Finite mask before `polyfit`; catch `ValueError`/`SystemError` | Yes (workaround) |

Combined prototype ("full joint" = F1 + F2 + F3 + F4 plus a transient-pair rule) compared with the current default:

| Benchmark | Metric | Current | Full joint |
|---|---|---|---|
| Controlled scenarios (6 × 5 seeds) | Trend + level-shift RMSE | 6.49 | **0.12** |
| | Changepoint recall (±30 d) / false positives per series | 3% / 4.6 | **93% / 0.13** |
| | Level-shift recall / false positives / magnitude error | 50% / 0.5 / 5.45 | **95% / 0.07 / 0.22** |
| | Noise std ÷ true noise std | 5.6 | **0.96** |
| AutoTS `SyntheticDailyGenerator` (3 seeds × 8 series) | Trend + level-shift NRMSE, mean (median) | 0.439 (0.397) | **0.309 (0.303)** |
| | `FeatureDetectionLoss.trend_loss` | 12.31 | **11.10** |
| | `noise_structure_loss` | 2.46 | **1.25** |
| | `level_shift_loss` | **2.28** | 2.54 |
| Fit time (vectorized implementation, see that section) | Ratio to current, 5 datasets | 1.0 | **0.79–1.11** (V3f); **0.74–0.95** (V2c, recommended) |

The results are not uniformly better. On the generator, 7 of 8 series types improve, but `no_level_shifts` gets worse (NRMSE 0.27 → 0.44) and `level_shift_loss` rises slightly. The brute-force prototype was 10–50× slower than the current detector. The vectorized implementation, with identical outputs, is at parity or faster (see "Vectorized implementation").

---

## Deep dive on the high-risk items (F1–F4): are they worth applying?

The results above use synthetic data, where the problems are dramatic. To check that they matter in practice, I re-tested on **real data** from the bundled `load_daily()`: 21 Wikipedia pageview series, 2017–2024. Variants form a ladder from least to most invasive:

| Variant | What changes |
|---|---|
| V0 | Current detector |
| V1 | F1 only (hinge-ridge detrend before the seasonality fit) |
| V2 | F1 + joint hinge/step refit with BIC knots + t-validated steps; step candidates still come from `LevelShiftMagic` (F3 + F4) |
| V3 | Full proposal (F1–F4 + unified selection + transient rule) |

### 1. Real-data diagnostics (3 rolling origins, 1095-day training windows)

| Metric | V0 | V1 | V2 | V3 |
|---|---|---|---|---|
| Forecast at 90 d: relative MAE vs seasonal naive, median | 1.15 | 1.32 | 1.05 | **1.03** |
| Forecast at 90 d: better than V0 (paired) | – | 40% | **63%** | 62% |
| Forecast at 365 d: relative MAE median | 1.40 | 1.50 | 1.74 | **1.32** |
| Forecast at 365 d: better than V0 / worst ratio | – | 41% / 12.8× | 46% / **15.8×** | 49% / 8.0× |
| Events per series-year | 2.03 | 2.06 | 1.80 | 1.07 |
| Share of events in Jan 1–7 | 0.1% | 0.1% | 1.4% | 2.8% |
| Noise lag-7 autocorrelation | 0.50 | 0.50 | **0.13** | 0.22 |
| Noise std ÷ series std | 0.63 | 0.63 | **0.38** | 0.41 |
| **Event stability** (events persisting ±14 d after +91 days of data) | 0.68 | 0.74 | **0.70** | **0.48** |
| Fit time (21 series), brute-force prototype | 2.9 s | 2.9 s | 40 s | 144 s |
| Fit time (21 series), vectorized, interleaved best of 3 | 2.87 s | – | **2.37 s** (V2c) | 2.48 s (V3f) |

### 2. Injected known events on the same real series (semi-synthetic truth)

Fit with and without an injected event; the difference in `trend + level_shift` is the estimated effect.

| Injected event | V0 | V2 | V3 |
|---|---|---|---|
| +20% level step: median end-effect error | 113% | **29%** | **28%** |
| +30%/yr slope change: median end-effect error | 58% | **16%** | 31% |
| 45-day +30% transient: wrongly made a trend event / level shift | **52%** of cases | **5%** | 14% |
| Event dated within ±14 d (step / slope) | 38% / 19% | 14% / 19% | 29–33% / 19% |

### 3. What the deep dive changes

- **The symptoms used to motivate F1 don't appear in this real data.** Jan 1 events are 0.1% of V0's output, there's no Jan 1 discontinuity in seasonality (jump ratio 1.1), and V0 doesn't flood events (about 2 per series-year).
  - The synthetic F1 failure needs a strong, steady trend relative to noise. Steadily growing business metrics would have one; mostly flat pageviews don't.
- **The real problem on real data is effect size.** V0 misestimates how big a real step or slope change is by 58–113%, and turns half of 45-day transients into trend events. Those are exactly the quantities story labeling needs: how big, and is it lasting. V2 and V3 cut these errors 2–4×.
- **F1 on its own is not worth applying.** V1 is no better on forecasts, events or effect sizes. F1 only helps as part of the joint estimation.
- **V3's unified selection hurts event stability** (48% vs 68–70%). Global BIC selection relocates or swaps knots as data arrives, so labels would churn. That's a serious drawback for labeling workflows.
- **Better slopes can hurt long-horizon `forecast()`.** V0's underestimated slope acted as accidental damping. V2 regresses at 365 days in the median and in the tail (worst case 15.8×). Any version must add trend damping to `forecast()` (e.g. φ ≈ 0.98/day, or the median slope over the last two segments) before becoming the default.
- **Robustness:**
  - V3 ran in univariate mode, on weekly and monthly data (with NaNs), on 200- and 2869-point series, and with `holiday_country`; `forecast()` stayed finite.
  - The prototype failed on a constant series (`log(0)` in BIC); that's a trivial guard, already applied in the vectorized version.
  - Both V0 and V3 crash on daily FRED data with NaNs; that's B3, not the proposal.
- **The generator regression (`no_level_shifts`, NRMSE 0.27 → 0.44) is unresolved.** Long-tailed anomaly types (`slope_reversion`, `transient_change`, `noisy_burst`) and seasonal misfit get absorbed as permanent steps, some recurring on the same day each year.
  - Raising the step t-threshold to 6 makes things worse overall (generator NRMSE 0.309 → 0.342).
  - BIC×3 helps slightly (0.309 → 0.292) with no loss on the controlled scenarios.
  - A 60–180-day persistence check has no measurable effect.
  - Parameters are not fragile, but this failure mode needs a different fix, for example a robust (Huber) loss in the selection, or recurrence-aware step rejection.
- **Runtime is solved.** The brute-force prototype was 14× (V2) and 50× (V3) slower than V0. The vectorized implementation produces outputs identical to V2, and to V3's selection, and fits at **0.74–0.95×** V0's time (V2c) and 0.77–1.11× (V3f) across 5 datasets of up to 5000 points. See "Vectorized implementation".

### 4. Revised recommendation

| Item | Verdict | Conditions |
|---|---|---|
| F1 alone | **Don't apply** | No measurable benefit in isolation |
| **V2: F1 + F3 + F4** (joint hinge/step refit, BIC knots, t-validated steps, `LevelShiftMagic` candidates), implemented as **V2c** (vectorized selector + cheap backfit) | **Apply as opt-in now; default later** | (a) add trend damping to `forecast()`; (b) ~~vectorized selection~~ done and confirmed: identical output, 0.74–0.95× V0 fit time; (c) check the 365-day forecast regression is gone on a growth-metric backtest; (d) consider `bic_mult=3` |
| V3: unified hinge + step selection | **Keep experimental (opt-in only)** | Fix event stability first, e.g. anchor existing knots and only re-optimize near new data, or penalize relocation of previously labeled events. Resolve the `no_level_shifts` regression |
| Transient rule, F6 date windows | Apply with V2 | Additive output |

Bottom line: the joint estimation (F3 + F4, which needs F1 to work) is worth applying. On real data it gives 2–4× more accurate effect sizes, keeps transients out of the trend, produces cleaner residuals and better 90-day forecasts, with event stability equal to today's, and with the vectorized implementation it is faster than the current detector. The fully unified selection (V3) is not worth making the default yet.

---

## Vectorized implementation (confirmed: identical output, parity or better speed)

The brute-force prototype refit OLS for every candidate position. That's O(n²·p) per greedy round, and relocation multiplies it further. The vectorized selector computes the score of **every** candidate position at once, from prefix and suffix sums, in O(n·p) per round.

### Math

Let `t = 0…n−1`, scaled to `t/n`. Candidates at position `c`:
- **hinge (slope change):** `h_c(t) = max(0, t − t_c)`
- **step (level shift):** `s_c(t) = 1[t ≥ c] − 1 = −1[t < c]`, which is 0 at the end of the series, matching the `LevelShiftMagic` convention

For any vector `v`, the inner products with all candidates at once are:

```
v · h_c = Σ_{t>c} v_t·t − t_c·Σ_{t>c} v_t        (two reverse cumulative sums)
v · s_c = −Σ_{t<c} v_t                            (one cumulative sum)
‖h_c‖² = K(K+1)(2K+1) / (6n²),  K = n−1−c          (closed form)
‖s_c‖² = c
```

Let `Q` be an orthonormal basis of the current design `[1, t, nuisance, chosen hinges, chosen steps]`, and `r` the current residual (orthogonal to `Q`). By Frisch–Waugh–Lovell, adding candidate `x` reduces SSE by exactly

```
gain(x) = (r·x)² / (‖x‖² − ‖Qᵀx‖²)
```

So one round evaluates `[r, Q]·x_c` for all `c` with the formulas above. That's O(n·p) work and O(n·p) memory, with no n×n candidate matrix.

**Pruning and t-tests** come from one `(XᵀX)⁻¹` per round: dropping term j increases SSE by `β_j² / [(XᵀX)⁻¹]_jj`. The t-stat uses the same diagonal with AR(1)-inflated σ².

**Relocation** of term j uses the same gain formula against the basis without j, restricted to the window between its neighbours.

### Core code (condensed from the benchmarked `fastsel.py`; numeric guards simplified)

```python
def _suffix_after(a):            # out[c] = sum_{t>c} a[t]
    cs = np.cumsum(a[::-1], axis=0)[::-1]; out = np.zeros_like(cs); out[:-1] = cs[1:]; return out

def _prefix_before(a):           # out[c] = sum_{t<c} a[t]
    cs = np.cumsum(a, axis=0); out = np.zeros_like(cs); out[1:] = cs[:-1]; return out

class _Candidates:
    def __init__(self, n):
        self.n, self.t = n, np.arange(n, dtype=float) / n
        K = (n - 1 - np.arange(n)).astype(float)
        self.h_norm2 = K * (K + 1) * (2 * K + 1) / 6.0 / n**2
        self.s_norm2 = np.arange(n, dtype=float)
    def dots(self, V, kind):     # inner products of every candidate with columns of V -> (n, k)
        if kind == "knot":
            return _suffix_after(V * self.t[:, None]) - self.t[:, None] * _suffix_after(V)
        return -_prefix_before(V)
    def norm2(self, kind):
        return self.h_norm2 if kind == "knot" else self.s_norm2
    def column(self, c, kind):
        return np.maximum(0.0, self.t - self.t[c]) if kind == "knot" else (np.arange(self.n) >= c) - 1.0

def _gains(cand, Q, r, kind, mask):
    both = cand.dots(np.column_stack([r, Q]), kind)
    rx, qx = both[:, 0], both[:, 1:]
    den = cand.norm2(kind) - np.einsum("ij,ij->i", qx, qx)
    return np.where(mask & (den > 1e-12), rx**2 / np.maximum(den, 1e-300), -np.inf)

def term_stats(X, y, n_base):    # beta, dSSE-if-dropped, AR(1) t for the knot/step terms
    inv = np.linalg.pinv(X.T @ X); beta = inv @ (X.T @ y); r = y - X @ beta; sse = r @ r
    idx = np.arange(n_base, X.shape[1]); d = np.clip(np.diag(inv)[idx], 1e-300, None)
    rho = np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0, 0.95) if np.std(r) > 0 else 0.0
    s2 = sse / max(len(y) - X.shape[1], 1) * (1 + rho) / (1 - rho)
    return beta[idx], beta[idx] ** 2 / d, beta[idx] / np.sqrt(s2 * d), sse
```

The selection loop is the same algorithm as F2/F3: greedy add with a BIC stop → up to 5 rounds of (relocate each term within its neighbour window → drop the BIC-weakest term → drop the weakest step if |t| < 4) → `split_transients`.

Parameters:
- `step_candidates` restricts step positions. V2 passes `[]` for knot-only search and then validates the exact `LevelShiftMagic` dates.
- `relocate_gap` and `prune_first` reproduce `hinge_binseg`'s exact neighbour window and drop order.
- Degenerate input (SSE ≈ 0, e.g. a constant series) returns no terms, which fixes the prototype's `log(0)` failure.

Full source: `scratch_fdreview/fastsel.py`. Detector wiring: `scratch_fdreview/fastdet.py`.

### Cheap backfit (V2c)

V2 runs its pipeline twice: pass 2 seeds the seasonality fit with pass 1's trend + steps. Re-running all of `fit()` doubles the cost of rough seasonality (100-tree random forest), holiday detection and anomaly detection, none of which depend on the trend estimate. V2c runs `_initial_decomposition` once and loops only the stages that depend on the trend:

```python
def fit(self, df):
    df_work = self._prepare_data(df); self._reset_results()
    rough_residual, rough_seasonality = self._initial_decomposition(df_work)
    saved = (self._holiday_regressors_temp, self._holiday_regressor_columns,
             deepcopy(self._holiday_dates_temp), deepcopy(self._anomaly_records_temp))
    self._trend_override = None
    for p in range(1 + self.backfit_passes):
        if p > 0:
            self._trend_override = self._last_trend_scaled
            self._reset_results(); restore(saved)
        final_residual, seas, strength, hol, hol_coef, hol_splash = self._final_seasonality_fit(df_work, rough_residual, rough_seasonality)
        trend, ls, vls, cps, slopes = self._detect_trend_and_shifts(final_residual, hol)
        noise, anom = self._analyze_noise(df_work, trend, ls, seas, hol)
        self._build_template(self._rescale_all_components(trend, ls, seas, hol, noise, anom), vls, slopes, cps, hol_coef, hol_splash, strength)
    self._trend_override = None
    return self
```

Dropping the backfit entirely is **not** an acceptable shortcut. It was faster (0.53–0.88× V0), but real-data event stability fell from 0.69 to 0.60 (below V0's 0.68) and step-size error rose from 29% to 44%.

### Equivalence evidence

| Check | Result |
|---|---|
| Closed-form inner products and norms vs explicit candidate columns (n=300, all c) | max abs error 1.6e-14 (hinge), 0 (step) |
| Unified selector vs brute-force reference (stride 1), 6 scenarios × 3 seeds × {with, without Fourier nuisance} | **36/36** identical terms (±3 d) |
| Knot search vs `hinge_binseg`, 24 synthetic inputs | **24/24** identical (±1 d) |
| Knot search vs `hinge_binseg`, on the 126 actual real-data inputs V2 generates (21 wiki series × 3 origins × 2 passes) | **125/126** identical (±1 d); 1 differs near the series end |
| **Full detector output, V2c vs brute-force V2**, real wiki data (2 expanding windows of 1408/1499 days + 1095-day injection base; 63 series) | **63/63** identical event lists; trend + level-shift identical (max diff 0, one series 0.047·std) |
| Quality benchmarks rerun with the vectorized detectors | Same as brute force: scenario RMSE 0.334, generator NRMSE 0.320, real-data slope error 16.3%, transient false events 4.8%, 90-day forecast better than V0 in 63%, stability 0.694 |

### Speed: selector alone (best of 3, ms)

| n | Vectorized | Vectorized + 26 Fourier nuisance columns | Brute-force reference (`hinge_binseg`) |
|---|---|---|---|
| 365 | 0.9 | 0.4 | 7.9 |
| 1,100 | 2.3 | 13.1 | 110 |
| 2,869 | 3.9 | 30.3 | 6,305 |
| 5,000 | 5.8 | 48.8 | – |
| 10,000 | 11.6 | 98.1 | – |
| 20,000 | 19.1 | 495 | – |

Against the unified brute-force selection with nuisance (3.65 s at n=1100), the vectorized selector is **289×** faster at 12.6 ms.

### Speed: full `fit()`, end to end

Interleaved in one process, best of 3, `@fbcode//mode/opt`, BLAS pinned to 1 thread. The host was shared with other workloads, which is why I used interleaving plus best-of-3; ratios are the reliable number.

| Dataset | V0 current | **V2c** (recommended) | V2x (V2 + full-pipeline backfit) | V3f (unified) |
|---|---|---|---|---|
| wiki 21 × 1095 | 2.87 s | **2.37 s (0.83×)** | 2.91 s (1.01×) | 2.48 s (0.87×) |
| generator 8 × 1100 | 4.42 s | **4.18 s (0.95×)** | 7.99 s (1.81×) | 4.06 s (0.92×) |
| wiki_all 1 × 2869 | 1.56 s | **1.29 s (0.83×)** | 2.32 s (1.48×) | 1.49 s (0.95×) |
| weekly EIA 8 × 1028 | 1.29 s | **1.13 s (0.87×)** | 1.61 s (1.25×) | 1.44 s (1.11×) |
| random walk 1 × 5000 | 3.44 s | **2.54 s (0.74×)** | 4.34 s (1.26×) | 2.72 s (0.79×) |

Why it's faster than current: the selector replaces PELT plus the old refit and validation, and costs milliseconds. The remaining time is the unchanged rough seasonality, holiday and anomaly stages. V2c's second pass adds only the final seasonality fit and the selector.

### Implementation notes for upstream

- Keep `t` scaled to [0, 1) and use the closed-form norms. At n = 20,000, the hinge norm reaches about 1e8, and the subtraction `‖x‖² − ‖Qᵀx‖²` stayed accurate in double precision in every test.
- With nuisance columns, the per-round cost is dominated by the QR of the n×(2+26+terms) design. At very large n (20k: 0.5 s), rank-one QR updates would reduce it further. That's not needed for typical daily lengths.
- `min_seg`, `edge` and the transient window are in **points**. For weekly or monthly data they should be converted from days (e.g. `min_seg = ceil(28 / days_per_period)`). The prototype ran on weekly and monthly data without errors, but used point units.

---

## F1. The final seasonality fit absorbs the trend (root cause)

**Code.** `components/seasonality.py::_fit_final_seasonality` fits `DatepartRegressionTransformer` (default `SVM`, `datepart_method='common_fourier'`) on the standardized series with only anomalies removed. The trend is still in it. For daily data, `common_fourier` is yearly Fourier (n=10) + weekly (n=3) + interactions (`tools/seasonal.py:358-366`), and there is no trend term. The Fourier origin is `2030-01-01` with p=365.25 (`seasonal.py:86`). A ramp projected onto a periodic basis becomes a sawtooth, which jumps at the period boundary, i.e. near Jan 1 every year.

**Evidence** (1100 daily points, `y = 100 + slope·t + weekly + N(0,1)`, optionally + 5·sin(yearly), 3 seeds):

| True slope | Estimated last slope (current) | Yearly-seasonality RMSE (current) | With detrend-first |
|---|---|---|---|
| 0.02 | 0.001 | 2.0 | 0.018 slope / 0.16 RMSE |
| 0.05 | 0.004 | 5.0 | 0.049 / 0.17 |
| 0.10 | 0.008 | 10.1 | 0.099 / 0.17 |

- About 90% of the trend ends up in `seasonality`, and the seasonality error grows as roughly 100 × slope.
- Across the controlled scenarios, 54 of the current detector's 133 false events fall between Jan 1 and Jan 7.
- Unmodeled level shifts leak the same way. A real step on Aug 24 produced ghost steps on Aug 24 of the other years (F8).

**Proposed change.** Before the seasonality fit, estimate trend and level steps jointly with the Fourier terms. Fit the seasonality model on `y − (trend + steps)`, then add `trend + steps` back to the residual passed to trend detection:

```python
def _fit_final_seasonality(self, df, holiday_regressors=None):
    trend = joint_trend_steps_with_seasonality(df)  # F2/F3 selection, Fourier as fixed nuisance
    res = super()._fit_final_seasonality(df - trend, holiday_regressors)
    return (res[0] + trend,) + tuple(res[1:])       # residual keeps trend for F2/F3
```

where

```python
def seasonal_nuisance(index):
    days = (index - pd.Timestamp(origin_ts)).days
    return np.hstack([fourier_series(days, 365.25, 10), fourier_series(days, 7, 3)])
# select_knots_and_steps(y, nuisance=seasonal_nuisance(df.index)); keep only the [1, t, hinges, steps] part
```

**Alternatives tested:**

| Initial detrend | Linear-trend error | Piecewise-trend error | Notes |
|---|---|---|---|
| Global linear term only | 0.013 | 3.51 | The piecewise remainder still leaks into seasonality (about 5 units of fake yearly pattern) and creates false knots near the true breaks |
| Hinge every 61 d + ridge 0.01 (jointly with Fourier) | 0.08 | 0.28 | Ridge 0 is unstable (collinear with yearly Fourier) |
| Selected knots + steps with Fourier nuisance (proposed) | best | best | Also prevents level-shift leakage. Backfitting alone does not fix a bad first-pass trend |

With the true trend supplied (oracle), the remaining non-weekly seasonal noise is 1.11, the same as the proposed method.

**Effect on `forecast()`.** Seasonality no longer carries trend, so a forecast is trend (extrapolated) + seasonality, and the trend part is no longer counted twice. Not separately benchmarked.

---

## F2. The default changepoint method doesn't detect slope changes

**Code.** The default is `{'method': 'pelt', 'method_params': {'penalty': 8, 'loss_function': 'l2'}, 'min_segment_length': 14}` (`detector.py:228-233`). `_detect_series_individual` passes the raw residual to `_detect_pelt_changepoints` (`changepoints.py:3683-3690`) with no detrending. The `l2` cost is piecewise-constant, so a steady trend gets split into a staircase. Meanwhile, `_detect_trend_changepoints` models the trend as continuous piecewise-linear.

**Evidence.** Input matches the detector: standardized, then the default `GeneralTransformer` (clip + Butterworth). 4 seeds per cell, tolerance ±30 days:

| Config | Linear (0 true breaks): false positives | Piecewise (2 true): recall / false positives | Gentle (3 true): recall / false positives |
|---|---|---|---|
| `pelt l2 pen=8` (current default) | 6.0 | 0.00 / 6.0 | 0.0–0.08 / 5.0 |
| `pelt l2 pen=50` | 2.5–3.0 | 0.00 / 2.0 | 0.00 / 2.0 |
| `l1_total_variation` (λ = 1…10⁴) | 24.5 | 0.50 / 23.5 | 0.75 / 22 |
| `l0_trend_filter` (order 2, max 8) | 6.25 | 0.12 / 6.0 | 0.08 / 6.0 |
| `wbs2` default | 53 | 1.00 / 52 | 1.00 / 47 |
| **Hinge selection + BIC (proposed)** | **0.0** | **1.00 / 0.0** | **0.83 / 0.5** |

The L1 rows are broken by B1 and B2, so treat them as "currently unusable", not as a fair evaluation of trend filtering.

**Proposed method.** Greedy forward selection of hinge knots `max(0, t − c)` under a joint least-squares fit. Stop when `n·log(SSE_old/SSE_new) < 2·bic_mult·log(n)` (`bic_mult=2` absorbs the autocorrelation the low-pass filter introduces). Then relocate each knot by coordinate descent and prune by backward elimination under the same rule. Without relocation and pruning, greedy selection leaves systematic redundant pairs, e.g. `[370, 402]` for a true break at 400 (false positives at ±30 d drop from 1.25 to 0.0). `min_seg=28`.

In the full method, step (level-shift) candidates compete in the same loop (F3), so one selection produces both `trend_changepoints` and `level_shifts`.

---

## F3. Trend refit and level-shift magnitude

**Code.**
- `trend.py:256-286` estimates each segment's slope with its own OLS, fits an intercept for segment 0 only, then joins the slopes with hinges. A level jump at a detected changepoint can't be represented: the slopes on both sides stay about the same, and the whole jump goes to `noise` (`anomalies.py:317-320`).
- Separately, `LevelShiftMagic` (`window_size=364`, overlapping rolling means) runs on a residual that still contains trend, so its magnitude ≈ true step + slope × window.

**Evidence.**
- My replica of the current refit matches `_detect_trend_changepoints` to 5e-13.
- Average over 20 seeds:

  | Case | Current RMSE / end error | Joint hinge | Joint hinge + step |
  |---|---|---|---|
  | Piecewise, true breaks | 0.121 / 0.213 | 0.062 / 0.100 | 0.072 / 0.105 |
  | Piecewise, breaks jittered ±10 d | 0.364 / 0.569 | 0.129 / 0.167 | 0.078 / 0.103 |
  | Linear + 5σ step at its break | 3.377 / 4.996 | 1.246 / 1.356 | **0.057 / 0.079** |

- End to end (slope 0.03, +5 step at day 600):
  - The level shift is reported as **15.82** (= 5 + 0.03 × 364).
  - Trend is nearly flat (122.0 → 121.0) against a true rise of +38.
  - Noise std is 6.0 (true 1.0), and `noise` has a −12.6 step at day 600.

**Proposed change.** Given the selected knots and steps, estimate everything in one regression and report each component from it:

```python
def joint_design(n, knots, steps):
    t = np.arange(n, dtype=float)
    T = np.column_stack([np.ones(n), t] + [np.maximum(0.0, t - k) for k in knots])
    S = (np.column_stack([(t >= s) - 1.0 for s in steps]) if steps else np.zeros((n, 0)))  # 0 at end, like LevelShiftMagic
    return T, S

def joint_fit_with_tstats(y, knots, steps, nuisance=None):
    T, S = joint_design(len(y), knots, steps)
    X = np.hstack([T, S] + ([nuisance] if nuisance is not None else []))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    rho = float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0.0, 0.95))
    sigma2 = r @ r / max(len(y) - X.shape[1], 1) * (1 + rho) / (1 - rho)
    se = np.sqrt(np.clip(np.diag(sigma2 * np.linalg.pinv(X.T @ X)), 1e-18, None))
    nt, ns = T.shape[1], S.shape[1]
    b = beta[nt:nt + ns]
    return T @ beta[:nt], S @ b, b, b / se[nt:nt + ns]   # trend, level_shift component, magnitudes, t-stats
```

Both `trend_component` and `level_shift_component` come from this one fit. `_compute_trend_slopes` is unchanged.

**Selection loop.** Extends F2; greedy search is over both candidate types:

```python
def select_knots_and_steps(y, min_seg=28, max_terms=16, bic_mult=2.0, step_t=4.0, edge=14, stride=3, nuisance=None):
    # 1. greedy: at each round try every hinge position (spaced >= min_seg from other knots) and every
    #    step position (spaced >= min_seg from other steps), on a `stride` grid; add the best if
    #    n*log(SSE/SSE_new) >= 2*bic_mult*log(n), else stop.
    # 2. repeat up to 5x: relocate each term to its best position between neighbours (full resolution);
    #    drop the term whose removal costs least if below the BIC penalty; drop the weakest step if |t| < step_t.
    # 3. split_transients(): an opposite-sign step pair <= 90 days apart whose magnitudes nearly cancel
    #    (|a+b| < 0.5*max(|a|,|b|)) is removed from trend/level-shift and reported as a transient event
    #    with start, end and magnitude.
```

The step t-test must include the nuisance columns. Without them, real yearly seasonality inflates sigma and true steps get rejected; with them, a −4 step was recovered at −4.01 / −4.28.

**Transient rule.** On the generator's holiday-heavy series, the prototype first fitted holiday or anomaly blocks as ±50–80 step pairs a few weeks apart (one matched Ramadan 2023). The transient rule cut generator NRMSE from 0.445 to 0.309. It also gives the labeling workflow what it needs: "this ran for N days, then reverted".

---

## F4. Level-shift validation accepts almost everything

**Code.** `trend.py:141-146` computes `rel_change = |Δ| / max(|median before|, 1e-9)` on a standardized residual that is roughly zero-centered. The relative check is OR'ed with the absolute one, so the denominator is often tiny and the test passes.

**Evidence.** I called `_validate_level_shifts` directly with default params, at random candidate dates, on noise with σ=0.3 (100 reps per cell):

| Input slope | True step / σ | Current acceptance | Proposed acceptance (local step t-test, \|t\| ≥ 4) |
|---|---|---|---|
| 0 | 0 (false candidate) | **0.99** | **0.00** |
| 0 | 0.83 | 0.99 | 0.10 |
| 0 | 1.67 | 1.00 | 0.91 |
| 0 | 3.33 | 1.00 | 1.00 |
| 0.003/d | 0 (false candidate) | 0.68 | 0.00 |
| 0.003/d | 1.67 | 1.00 | 0.87 |

**Proposed change.** Validate a step by its t-statistic in the joint model (F3), using AR(1)-inflated sigma. For a minimal patch without F3: regress `[1, t, step]` over ±90 days around the candidate and require `|t_step| ≥ 4`.

---

## F5. Event DAG magnitudes and signatures

**Code.**
- `event_dag.py::_finalize_cluster` (lines 311-312) adds `magnitude` across families and series. Slope deltas (units/day) get added to level shifts and anomalies (units), across metrics of any scale.
- `_cluster_signature` (lines 376-401) normalizes by that total and appends a `source_mix` vector, which is identical whenever all events come from the same family type.

**Evidence** (synthetic detector stub with known events). Pattern X = A↑ + B↑ on days 100 and 400; pattern Y = A↑ + C↓ on days 700 and 1000:

| Setting | Families |
|---|---|
| All series at unit scale | **1 family** containing X and Y (cosine 0.64 > 0.6, driven by `source_mix`) |
| B multiplied by 1000 | 2 families (only because B's units changed) |
| Proposed, either scale | 2 families: {A,B} and {A,C}, the same at both scales |

Also, a same-day slope change of −0.05/day (−4.5 over 90 days) alongside +1 level shifts gives `net_magnitude = 1.95`, which has the wrong sign over a business horizon.

**Proposed change.**

```python
def _cluster_signature(cluster, member_lookup, series_names, source_families, scales, horizon=90):
    sig = np.zeros(len(series_names)); pos = {s: i for i, s in enumerate(series_names)}
    for mid in cluster["member_ids"]:
        m = member_lookup[mid]
        mag = m["magnitude"] * (horizon if m["source_family"] == "trend_changepoints" else 1.0)
        sig[pos[m["series_name"]]] += mag / scales[m["series_name"]]   # scale = noise sigma or series std
    return sig
```

Apply the same normalization to `net_magnitude` and `abs_magnitude`, and expose `horizon` in `event_dag_params`. `series_noise_levels` × `series_scales` (or the residual sigma) are already available on the detector.

---

## F6. Changepoint records and date windows for story matching

**Code.** `_build_trend_label_entries` (`formatting.py:191-209`) outputs only `date, prior_slope, new_slope`. `ChangepointDetector` computes `changepoint_confidence_` (`changepoints.py:4323-4333`, `4602-4616`), but the detector never reads it. That window is based on a difference in means, i.e. on level change.

**Evidence** (40 reps per cell, slope change of 0.02 or 0.05/day, noise σ 1 or 3, change point estimated by F2):

| Input | Existing window: coverage / width | Profile-likelihood window: coverage / width |
|---|---|---|
| Raw residual | 0.67 / 16 d (0.40 for faint, noisy changes) | **0.96 / 36 d** (10 d for clear, 79 d for faint) |
| Low-pass filtered | 0.65 / 14 d (0.20 worst case) | 1.00 / 53 d (over-conservative) |

**Proposed window.** Move changepoint *j* within its neighbours and keep positions where `ΔSSE/σ²_AR ≤ 3.84` (χ²₁, 95%), with `σ²_AR = σ²(1+ρ)/(1−ρ)`. Compute it on the **unfiltered** residual.

**Proposed record schema** (template and `query_features`):

```json
{"date": "...", "date_window": ["...", "..."], "type": "slope_change|level_shift|transient",
 "prior_slope": 0.0, "new_slope": 0.0, "level_change": 0.0,
 "effect_28d": 0.0, "effect_90d": 0.0, "effect_365d": 0.0, "effect_pct_of_level": 0.0,
 "regime_length_days": 0, "is_active_regime": true, "t_stat": 0.0,
 "transient_end": null, "recurring_doy": false}
```

The effect fields are `Δslope × h + level_change`; `transient_end` comes from the F3 transient rule; `recurring_doy` comes from F8. The schema is a design proposal; only the window was benchmarked.

---

## F7. Shared events across metrics

**Code.** `DEFAULT_EVENT_DAG_PARAMS['cluster_window_periods'] = 1` (`event_dag.py:13`). The detector's own loss uses a 7-day changepoint tolerance (`loss/base.py:51`). `ChangepointDetector.get_market_changepoints` exists but is unused.

**Evidence.** Six metrics with random scales (1–1000×) and noise share a slope change at day 600 (three also have a +3 step). 3 seeds:

| Detector | Series detecting the event | Clusters near day 600 at window 1 / 7 / 14 | Largest cluster (series) at window 1 / 7 / 14 |
|---|---|---|---|
| Current | 1.7 of 6 | 1.3 / 1.3 / 1.3 | 1.3 / 1.3 / 1.3 |
| Full joint | 5 of 6 | **3.7 / 2.0 / 1.0** | 3.7 / 4.7 / **5.0** |

Per-series date jitter (sd) is about 9.4 days.

**Proposed change.**
- Set `cluster_window_periods ≈ 14` for daily data (about 1.5 × the observed date sd), or derive it from frequency.
- Better: cluster events whose F6 date windows overlap, which adapts to how sharp each break is.
- **Not recommended yet:** detecting on the first SVD factor of the standardized residuals. It was exact on 1 of 3 seeds (knot 598, step 600) and off by 31–37 days on the other two.

---

## F8. Day-of-year recurrence

On the current detector's output, tagging events that recur within ±7 days of the same day-of-year in 2 or more years:
- flags 106 of 133 false events (80%), but also 18 of 31 true events (58%), because a real level shift spawns yearly ghost copies of itself (F1);
- after the F1 + F3 fixes, there are only 3 false events and 0 are flagged.

**Recommendation.** Add `recurring_doy` as a diagnostic field for labeling ("possible seasonal artifact"), not as a filter. The root fix is F1.

---

## Bugs

### B1. Batched L1 solve fails under NumPy ≥ 2, and the failure is silent

`changepoints.py:2276`: `A` has shape `(k, n, n)` and `chunk` has shape `(k, n)`. Since NumPy 2.0, `np.linalg.solve` treats a 2-D `b` as a matrix, so this raises `ValueError`. `_vectorized_l1_detection` catches it with `except Exception: fitted_block = data_block.copy()`, so `l1_fused_lasso` and `l1_total_variation` in individual mode extract changepoints from **raw data**. The fitted trend is identical for λ = 0.1…10⁴ (roughness 133.3 = the input).

```python
return np.linalg.solve(A, chunk[..., None])[..., 0]
```

Also narrow the `except Exception` so that numeric failures are visible.

After the fix, the fit responds to λ (roughness 0.089 → 0.000), and the changepoint count on a linear trend falls from 21.7 to 1.3 as λ goes from 1 to 1000. The remaining false positives come from B2.

### B2. Scale-free L1 extraction threshold

`_extract_changepoints_from_trend_batch` keeps `|Δᵒ fitted| > mean + 1.5·std` of the same series. On a nearly perfect linear fit, that flags numerical noise. **Proposal:** threshold in absolute terms, e.g. `|Δ² fitted| > c · σ_resid / segment_scale`, or rank kinks and stop with the F2 BIC rule. Not benchmarked beyond the λ sweep above.

### B3. NaN crash in `_classify_anomaly_type`

`anomalies.py:248-253` calls `np.polyfit` on `extended_devs`, which can contain NaN. With NaN input, NumPy raises `SystemError`/`ValueError`, not the `LinAlgError` that's caught. `fit()` crashes on every `SyntheticDailyGenerator(n_days=1100, n_series=8, random_seed=0..2)` dataset, because `series_0` is `business_day` with 314 NaNs.

```python
mask = np.isfinite(extended_devs)
if mask.sum() >= 3:
    try:
        decay_slope = np.polyfit(np.arange(len(extended_devs))[mask], extended_devs[mask], 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        decay_slope = 0
```

---

## Implementation plan (suggested PR order)

1. **B1, B3** (small, independent). Tests: L1 fit changes with λ; `fit()` on generator seed 0 doesn't raise.
2. **F4** as a minimal patch (local step t-test in `_validate_level_shifts`). Test: random candidates on pure noise accept less than 5%; 3σ steps accept more than 95%.
3. **F5** (Event DAG normalization). Test: multiplying one series by 1000 leaves families unchanged; X and Y patterns separate.
4. **F1 + F3 + F4 (V2c configuration)** as an opt-in `changepoint_params['method'] = 'joint_hinge_step'`: vectorized selector, cheap backfit, and trend damping in `forecast()`. Make it the default only after the conditions in the deep-dive table are met. The unified step search (V3) stays behind a separate flag. Tests:
   - pure linear trend → 0 changepoints, 0 level shifts, last slope within 5%, no Jan 1 discontinuity in `seasonality`;
   - linear + 5σ step → level-shift magnitude within 10%, noise std within 10% of truth;
   - piecewise with 2 slope changes → both found within ±30 d;
   - step pair 30 d apart → transient, not a level shift;
   - **equivalence:** vectorized selector vs a brute-force reference on fixed seeds (identical terms);
   - **performance guard:** `fit()` time ≤ 1.1× the legacy path on a 21 × 1095 panel (interleaved, best of 3).
5. **F6** (date window + record schema), **F7** (window default), **F8** (tag).

**Runtime.** Solved; see "Vectorized implementation". Port `fastsel.py` and the V2c `fit()` loop. Don't port the brute-force prototype.

## Not tested / open questions

- Generator series type `no_level_shifts` gets worse (NRMSE 0.27 → 0.44), and `level_shift_loss` rises 2.28 → 2.54. Likely cause: multi-week anomaly or holiday structure that the transient rule only partly catches. Needs a look before making this the default.
- `tune_with_synthetic` / `FeatureDetectionOptimizer`: not run. The optimizer may already move away from PELT-`l2`, but it can't fix F1, F3, F4, F5 or B1.
- Univariate mode, weekly and monthly data, 200–5000-point series, and `holiday_country` were run without errors with the brute-force V3; finite `forecast()` was checked there. With the vectorized detectors, weekly data and a 5000-point series were timed, and V2c's output is identical to brute-force V2 on real data. Univariate mode and `holiday_country` were not rerun with V2c. Multiplicative and saturating series were not specifically tested beyond the generator's `saturating_trend` type.
- Real data: 21 Wikipedia pageview series only, mostly flat or declining. A backtest on steadily growing business metrics is still needed, especially for the 365-day `forecast()` regression.
- The vendored copy may differ from upstream HEAD; re-check line references.