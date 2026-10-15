# Holiday attribution validation

The lunar calendar fixes, Easter rules, and per-series holiday suppression are
implemented. **Keep the existing detector defaults.** The proposed combination
improves holiday attribution but regresses forecast error and synthetic precision.
TVA therefore continues to receive the existing detector defaults.

Enable the new behavior explicitly:

```python
from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

detector = TimeSeriesFeatureDetector(
    global_holiday_anomaly_suppression=None,
    holiday_params={"use_lunar_holidays": True, "use_easter_holidays": True},
)
```

`True` retains batch-wide suppression; `False` disables suppression. Named
per-series suppression compares the same holiday and signed offset against two
other observed years. It preserves anomalies when evidence is insufficient, when
scores are unavailable, or when either score strength or residual magnitude is
exceptional. The magnitude escape hatch is necessary: rolling z-scores can give
ordinary holiday spikes and much larger isolated outages almost identical scores.
An explicit `anomaly_params={"holiday_override_strength": None}` instead requests
unconditional suppression within the chosen scope.

## Measurement

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m examples.holiday_attribution_benchmark --output /tmp/holiday-results.json
```

The comparison uses the 21 bundled `wiki_*` series, with gaps interpolated, and
`SyntheticDailyGenerator(n_days=1460, n_series=12)` with seeds 1–5. Each fit holds
out the final 28 days and uses seed 42 for detector randomness. Both variants use
the corrected calendar utilities; the legacy variant explicitly selects the old
holiday configuration. These measurements use this checkout's newer detector,
so their counts differ from the vendored 1.0.4 review.

All normalized errors use the original training series' standard deviation,
fixed across configurations. Forecast MAE and synthetic component RMSE are
averaged across series, then seeds. Synthetic precision/recall use the existing
optimizer's one-to-one date matcher: one-day tolerance for anomaly onset and
holiday dates, seven days for level shifts. This is onset recovery, not a new
duration/type score. False level shifts retain the existing truth definition.
Fit times are single runs, not a statistically established speed comparison.

## Wikipedia results

| Metric | Legacy | Proposed |
| --- | ---: | ---: |
| Forecast normalized MAE | 0.2845 | 0.3113 |
| Anomaly flags | 1152 | 1733 |
| Level shifts | 152 | 149 |
| Fit seconds | 9.05 | 9.44 |

| Holiday series | Holiday-day anomaly flags, legacy → proposed | Holiday-day residual normalized RMSE, legacy → proposed |
| --- | ---: | ---: |
| Christmas | 7 → 1 | 7.23 → 3.95 |
| Thanksgiving | 7 → 0 | 11.12 → 1.66 |
| Easter | 8 → 1 | 13.46 → 4.43 |
| Chinese New Year | 7 → 3 | 15.95 → 11.65 |

The residual is original minus seasonality, holidays, trend, and level shifts;
the table reports its absolute holiday-day RMSE, without the review's additional
pre-holiday baseline subtraction. More total flags are expected when one series'
holidays stop clearing another series' candidates. There is no real-data anomaly
truth here, so the increase cannot be interpreted as improved accuracy.

Holiday-related shifts remain: the Easter series has two more detected shifts
despite better holiday attribution. Several changes are date realignments, while
new shifts include Easter-adjacent dates. No level-shift gate was added.

## Synthetic results

| Metric, five-seed mean unless noted | Legacy | Proposed |
| --- | ---: | ---: |
| Anomaly onset precision | 0.5077 | 0.4708 |
| Anomaly onset recall | 0.5264 | 0.6101 |
| Holiday date precision | 0.6032 | 0.5654 |
| Holiday date recall | 0.1665 | 0.1745 |
| Holiday component normalized RMSE | 0.0960 | 0.1032 |
| Forecast normalized MAE | 0.6247 | 0.6402 |
| False level shifts, total | 252 | 256 |
| Fit seconds | 6.66 | 6.92 |

The net four extra false shifts occur in seed 1. Inspection of shifts more than
seven days from any legacy detection found several near synthetic anomaly onsets;
only one was within seven days of a labeled holiday. Holiday gating would not
address most of those changes. They remain false positives in the reported score.

## Regression coverage

Deterministic tests cover known lunar New Years from 2017–2025, scalar/prefix/
forecast calendar invariance, independent Easter offsets, missing observations,
caps and relaxed detection, all holiday output styles, older fitted detectors,
exceptional outages, cross-series scoping, insufficient peers, New Year offsets,
intraday/timezone alignment, shared flags, country calendars, forecast continuity,
and configuration serialization. The controlled attribution test requires at
least 80% of the injected holiday effect in the holiday component.

The broader regression selection covers anomaly detection, feature detection,
transformers, Cassandra, TVA, and detector forecasting: **321 passed, 46 skipped,
56 subtests passed** in 139 seconds. Formatting, lint, Python 3.9 parsing of new
files, and whitespace checks also passed. Three existing failures
were reproduced in an isolated checkout of unchanged `HEAD`: the two calendar
tests requiring the absent optional `holidays` package, and
`TestSparseIdentification.test_atom_usage_is_a_trustworthy_rank_diagnostic`.
The live-data/Prophet anomaly test and long synthetic-tuning integration test were
excluded from the bounded regression run.
