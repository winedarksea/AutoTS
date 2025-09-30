# Synthetic Daily Data Generator - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Features](#features)
4. [Usage Examples](#usage-examples)
5. [Component Details](#component-details)
6. [Accessing Labels](#accessing-labels)
7. [Plotting](#plotting)
8. [API Reference](#api-reference)
9. [Model Evaluation](#model-evaluation)
10. [Recent Updates](#recent-updates)

---

## Overview

The `SyntheticDailyGenerator` creates realistic synthetic daily time series data with fully labeled components for evaluating forecasting models. It's designed to test models' ability to handle changepoints, holidays, anomalies, and various complex patterns.

### Why Use This Generator?

- **Ground Truth Labels**: All events (anomalies, changepoints, level shifts, holidays) are labeled for precise evaluation
- **Realistic Complexity**: Multiple interacting patterns mimicking real-world data
- **Special Cases**: Business days, non-Gregorian holidays, variance regimes, shared events
- **Multi-Scale Data**: Tests model robustness across different magnitudes
- **Reproducible**: Same seed always produces identical data

---

## Quick Start

### 30-Second Example

```python
from autots.datasets import generate_synthetic_daily_data

# Generate data
gen = generate_synthetic_daily_data(
    start_date='2018-01-01',
    n_days=1095,  # 3 years
    n_series=5,
    random_seed=42
)

# Get data and view it
data = gen.get_data()
print(data.head())

# See summary
gen.summary()

# Plot a series
gen.plot('series_0')
```

### Installation Note
The generator is already integrated into AutoTS. Just import it:

```python
from autots.datasets import generate_synthetic_daily_data, SyntheticDailyGenerator
```

Optional dependency for plotting:
```bash
pip install matplotlib
```

---

## Features

### Multi-Component Time Series
Each series contains:
- ✅ **Piecewise linear trends** with labeled changepoints
- ✅ **Nonlinear/saturating trends** (logistic + quadratic)
- ✅ **Level shifts** (instantaneous, 2-day ramp, 3-day ramp)
- ✅ **Weekly seasonality** with stochastic variation
- ✅ **Yearly seasonality** (Fourier basis with amplitude drift)
- ✅ **Time-varying seasonality** (gradually changing)
- ✅ **Seasonality changepoints** (distinct pattern changes)
- ✅ **Holiday effects** with splash and bridge days
- ✅ **Noise** with distribution changepoints
- ✅ **Mean-reverting random walks**
- ✅ **GARCH-like variance regimes**
- ✅ **Anomalies** with post-event patterns (5 types)
- ✅ **Shared events** across multiple series (NEW)
- ✅ **Optional regressor effects**

### Special Series Types

- **series_0**: Business day frequency (NaN on weekends), 10x scale
- **series_1**: Nonlinear/saturating trends, 1x scale
- **series_2**: Time-varying seasonality, 1x scale
- **series_3**: Seasonality changepoints, 10x scale
- **series_4**: No level shifts (control series), 1x scale
- **series_5**: Chinese New Year holidays, 1x scale
- **series_6**: Ramadan holidays, 10x scale
- **series_7**: GARCH-like variance regimes, 1x scale
- **series_8+**: Standard combinations

### Multi-Scale Data
Every 3rd series has values approximately 10x larger to test models' handling of different scales.

---

## Usage Examples

### Basic Usage

```python
from autots.datasets import generate_synthetic_daily_data

# Quick generation with defaults
gen = generate_synthetic_daily_data(
    start_date='2018-01-01',
    n_days=1095,  # 3 years
    n_series=10,
    random_seed=42
)

data = gen.get_data()
gen.summary()
```

### Custom Parameters for Complex Data

```python
from autots.datasets import SyntheticDailyGenerator

gen = SyntheticDailyGenerator(
    start_date='2019-01-01',
    n_days=1460,  # 4 years
    n_series=8,
    random_seed=999,
    trend_changepoint_freq=1.0,      # More frequent trend changes (per year)
    level_shift_freq=0.3,             # More level shifts (per year)
    anomaly_freq=0.1,                 # More anomalies (per week)
    weekly_seasonality_strength=2.0,  # Stronger weekly pattern
    yearly_seasonality_strength=1.5,  # Stronger yearly pattern
    noise_level=0.05,                 # Lower noise (cleaner signal)
    include_regressors=True,          # Include synthetic regressors
)
```

### With Shared Events (NEW)

```python
# Generate data with shared anomalies and level shifts
gen = generate_synthetic_daily_data(
    n_days=1095,
    n_series=10,
    random_seed=42,
    shared_anomaly_prob=0.3,      # 30% of anomalies affect multiple series
    shared_level_shift_prob=0.4   # 40% of level shifts are shared
)

# Check which events are shared
all_anomalies = gen.get_anomalies()
for series, anomalies in all_anomalies.items():
    shared = [a for a in anomalies if a[4]]  # a[4] is is_shared flag
    print(f"{series}: {len(shared)} shared anomalies")
```

### With External Regressors

```python
gen = generate_synthetic_daily_data(
    n_days=730,
    n_series=5,
    include_regressors=True
)

# Get regressors (promotion, temperature, precipitation)
regressors = gen.get_regressors()
print(regressors.head())

# Save with regressors
gen.to_csv('data_with_regressors.csv', include_regressors=True)
```

### Clean Data (Low Noise)

```python
gen = generate_synthetic_daily_data(
    n_days=730,
    n_series=5,
    noise_level=0.02,  # Very low noise
    random_seed=42
)
```

---

## Component Details

### Trend Changepoints
- Piecewise linear trends with randomly placed changepoints
- **Series 1** has nonlinear/saturating trends (logistic + quadratic segments)
- Changepoints labeled with dates and slope changes
- Frequency controlled by `trend_changepoint_freq` parameter
- **Validated meaningful changes**: All slope changes exceed minimum threshold (0.008 * scale) to ensure detectability
  - 80% of changepoints have standard threshold for clear detection
  - 20% have subtle but still meaningful changes (50% of threshold)
  - Prevents imperceptible micro-changes that would be lost in noise

### Level Shifts
- **Instantaneous**: Immediate step change
- **Ramp (2-day)**: Gradual shift over 2 days
- **Ramp (3-day)**: Gradual shift over 3 days
- **Shared**: Can affect multiple series simultaneously (controlled by `shared_level_shift_prob`)
- Configurable frequency via `level_shift_freq` parameter
- **Series 4** has no level shifts by design (control series)

### Seasonality
- **Weekly**: Stochastic day-of-week effects (drawn from distribution, not fixed)
- **Yearly**: Fourier basis with small amplitude drift
- **Series 2**: Time-varying seasonality (gradually changing patterns)
- **Series 3**: Distinct seasonality changepoints
- Strength controlled by `weekly_seasonality_strength` and `yearly_seasonality_strength`

### Holiday Effects
- **Common holidays**: December 25th (all series)
- **Custom holiday**: 3rd Tuesday of July (all series)
- **Chinese New Year**: Series 5 (uses AutoTS `gregorian_to_chinese()`)
- **Ramadan**: Series 6 (uses AutoTS `gregorian_to_islamic()`)
- **Splash effects**: Impact on surrounding days (3-7 days before/after)
- **Bridge effects**: Enhanced impact on workdays adjacent to weekends
  - Thursday holidays create Friday bridge days
  - Tuesday holidays create Monday bridge days
- **Weekend interaction**: 1.5x impact if holiday falls on weekend

### Anomalies (5 Types)

1. **Point Outlier**: Single-day outlier (can be positive or negative)
2. **Persistent Shift**: Multi-day anomaly with same mean distribution (2-3 days)
3. **Impulse-decay**: Exponential decay over 7 days
4. **Ramp-down**: Linear ramp-down over 5 days
5. **Transient Change** (NEW): Temporary level shift that reverts (3-8 days)

**Properties:**
- Magnitude: 5x to 15x noise level
- Guaranteed separation from background noise and holidays
- Minimum 7 days between anomalies
- Can be **shared** across multiple series (controlled by `shared_anomaly_prob`)
- All anomalies include `is_shared` flag in labels

**Important Naming Clarification:**
The names describe **DURATION**, not **DIRECTION**. All anomaly types can be both positive (spike up) or negative (dip down):

| Type | Duration | Direction | Pattern |
|------|----------|-----------|---------|
| `point_outlier` | 1 day | ± | Single outlier point |
| `persistent_shift` | 2-3 days | ± | Multi-day with noise |
| `impulse_decay` | 7 days | ± | Exponential decay |
| `ramp_down` | 5 days | ± | Linear ramp |
| `transient_change` | 3-8 days | ± | Level shift then revert |

*Note: `point_outlier` and `persistent_shift` were previously called "spike" and "dip" which incorrectly suggested directional bias.*

### Noise Features
- Distribution changepoints (normal, Laplace, t-distribution)
- Mean-reverting random walk component
- **Series 7**: GARCH(1,1)-like variance regimes (low/normal/high volatility)
- Never reaches anomaly magnitude levels
- Noise level controlled by `noise_level` parameter

### Optional Regressors
- **Promotion**: Binary flag (5% of days)
- **Temperature**: Seasonal pattern + noise
- **Precipitation**: Gamma distribution
- Variable impact by series:
  - 70% of series respond to promotions
  - 50% respond to temperature
  - 30% respond to precipitation

---

## Accessing Labels

### All Labels at Once

```python
# Get all labels for a specific series
labels = gen.get_all_labels('series_0')

# Available labels:
# - trend_changepoints
# - level_shifts
# - anomalies
# - holiday_impacts
# - noise_changepoints
# - seasonality_changepoints
# - noise_to_signal_ratio
# - series_scale
```

### Individual Label Types

```python
# Trend changepoints
trend_cp = gen.get_trend_changepoints('series_0')
for date, old_slope, new_slope in trend_cp:
    print(f"{date.date()}: slope {old_slope:.4f} → {new_slope:.4f}")

# Level shifts
level_shifts = gen.get_level_shifts('series_1')
for date, magnitude, shift_type, is_shared in level_shifts:
    shared_str = " (SHARED)" if is_shared else ""
    print(f"{date.date()}: {shift_type} shift of {magnitude:.2f}{shared_str}")

# Anomalies
anomalies = gen.get_anomalies('series_2')
for date, magnitude, anom_type, duration, is_shared in anomalies:
    shared_str = " (SHARED)" if is_shared else ""
    print(f"{date.date()}: {anom_type}, mag={magnitude:.2f}, {duration}d{shared_str}")

# Holiday impacts
holidays = gen.get_holiday_impacts('series_5')
for date, impact in holidays.items():
    print(f"{date.date()}: impact {impact:.2f}")

# Get for all series
all_anomalies = gen.get_anomalies()  # Returns dict keyed by series name
```

### Component Analysis

```python
# Get individual components for decomposition
components = gen.get_components('series_1')

# Available components:
# - trend
# - level_shift
# - seasonality
# - holidays
# - noise
# - anomalies
# - regressors (if included)

# Verify components sum to total
import numpy as np
data = gen.get_data()
component_sum = sum(components.values())
print(np.allclose(component_sum, data['series_1'].values, equal_nan=True))  # True
```

---

## Plotting

### Plot Method

The `plot()` method creates a comprehensive 4-panel visualization:

```python
# Plot a specific series
gen.plot(series_name='series_0')

# Plot a random series
gen.plot()  # Will randomly select one

# Save without displaying
gen.plot(series_name='series_1', save_path='my_plot.png', show=False)

# Custom figure size
gen.plot(series_name='series_2', figsize=(20, 14))

# Get figure object for further customization
fig = gen.plot(series_name='series_3', show=False)
fig.savefig('custom_plot.pdf', dpi=300)
```

### Plot Panels

The visualization includes:

1. **Full Time Series with Events**
   - Complete time series data
   - Red dashed lines: Anomalies (with triangular markers)
   - Green solid lines: Trend changepoints (with triangle markers)
   - Purple dotted lines: Level shifts (with star markers)
   - Orange dash-dot lines: Seasonality changepoints

2. **Trend and Level Shifts**
   - Green line: Trend component
   - Black line: Combined trend + level shifts
   - Annotations show level shift magnitudes (e.g., "+5.3", "-2.1")

3. **Seasonality and Holidays**
   - Cyan line: Seasonality component
   - Orange line: Holiday effects
   - Blue line: Combined seasonality + holidays
   - Orange markers on top 5 holidays by impact

4. **Noise and Anomalies**
   - Gray line: Background noise
   - Red line: Anomalies
   - Detailed annotations for each anomaly (type, magnitude)
   - Vertical lines mark noise distribution changepoints

**Summary Box**: Displays scale factor, noise-to-signal ratio, and event counts

---

## API Reference

### Main Functions

```python
# Quick generation
generate_synthetic_daily_data(
    start_date='2015-01-01',
    n_days=2555,
    n_series=10,
    random_seed=42,
    trend_changepoint_freq=0.5,
    level_shift_freq=0.1,
    anomaly_freq=0.05,
    weekly_seasonality_strength=1.0,
    yearly_seasonality_strength=1.0,
    noise_level=0.1,
    include_regressors=False,
    shared_anomaly_prob=0.0,      # NEW
    shared_level_shift_prob=0.0,  # NEW
)

# Full class instantiation
gen = SyntheticDailyGenerator(
    # ... same parameters as above
)
```

### Methods

```python
# Data access
data = gen.get_data()                    # Wide format DataFrame
regressors = gen.get_regressors()        # Regressor DataFrame
gen.to_csv('file.csv', include_regressors=False)

# Labels
labels = gen.get_all_labels(series_name)
trend_cp = gen.get_trend_changepoints(series_name)
shifts = gen.get_level_shifts(series_name)
anomalies = gen.get_anomalies(series_name)
holidays = gen.get_holiday_impacts(series_name)

# Components
components = gen.get_components(series_name)

# Visualization
gen.plot(series_name=None, figsize=(16, 12), save_path=None, show=True)

# Info
gen.summary()
```

### Parameters

- **start_date** (str): Start date (e.g., '2018-01-01')
- **n_days** (int): Number of days to generate
- **n_series** (int): Number of time series
- **random_seed** (int): For reproducibility
- **trend_changepoint_freq** (float): Changepoints per year (default: 0.5)
- **level_shift_freq** (float): Level shifts per year (default: 0.1)
- **anomaly_freq** (float): Anomalies per week (default: 0.05)
- **weekly_seasonality_strength** (float): Weekly seasonality magnitude
- **yearly_seasonality_strength** (float): Yearly seasonality magnitude
- **noise_level** (float): Noise relative to signal (default: 0.1)
- **include_regressors** (bool): Include synthetic regressors
- **shared_anomaly_prob** (float): Probability anomaly is shared (0.0-1.0, default: 0.0)
- **shared_level_shift_prob** (float): Probability level shift is shared (0.0-1.0, default: 0.0)

---

## Model Evaluation

### Testing Anomaly Detection

```python
from autots.datasets import generate_synthetic_daily_data

# Generate data
gen = generate_synthetic_daily_data(n_days=730, n_series=3, random_seed=42)
data = gen.get_data()

# Get ground truth
series_name = 'series_0'
labels = gen.get_all_labels(series_name)
true_anomaly_dates = [date for date, _, _, _, _ in labels['anomalies']]

# Run your detector
# detected_dates = your_anomaly_detector(data[series_name])

# Compare
# precision = calculate_precision(detected_dates, true_anomaly_dates)
# recall = calculate_recall(detected_dates, true_anomaly_dates)
```

### Testing Changepoint Detection

```python
# Get true changepoints (trend + level shifts)
labels = gen.get_all_labels('series_0')
true_cp_dates = [date for date, _, _ in labels['trend_changepoints']]
true_cp_dates.extend([date for date, _, _, _ in labels['level_shifts']])

# Run your detector
# detected_cp = your_changepoint_detector(data['series_0'])

# Evaluate
# accuracy = evaluate_changepoints(detected_cp, true_cp_dates)
```

### Testing Shared Event Detection (NEW)

```python
# Generate data with shared events
gen = generate_synthetic_daily_data(
    n_days=1095,
    n_series=10,
    shared_anomaly_prob=0.3,
    shared_level_shift_prob=0.4
)

# Get all anomalies
all_anomalies = gen.get_anomalies()

# Find shared event dates
shared_dates = {}
for series, anomalies in all_anomalies.items():
    for date, mag, atype, dur, is_shared in anomalies:
        if is_shared:
            if date not in shared_dates:
                shared_dates[date] = []
            shared_dates[date].append(series)

# Test your correlation-aware anomaly detector
# detected_shared = your_multivariate_detector(gen.get_data())
# compare with shared_dates
```

### Example Workflow

```python
# 1. Generate
gen = generate_synthetic_daily_data(n_days=730, n_series=5, random_seed=42)

# 2. Get data and labels
data = gen.get_data()
all_labels = gen.get_all_labels()

# 3. Train/test split
train = data.iloc[:550]
test = data.iloc[550:]

# 4. Run your model
# predictions = your_model.fit(train).predict(len(test))

# 5. Evaluate
# Compare predictions with test
# Compare detected events with true labels

# 6. Visualize results
gen.plot('series_0')
gen.summary()
```

---

## Recent Updates

### September 2025 Enhancements

#### 1. Shared Events (NEW)
- **Shared Anomalies**: Events affecting multiple series simultaneously
  - Controlled by `shared_anomaly_prob` parameter
  - Simulates systemic shocks (e.g., economic events, system outages)
  
- **Shared Level Shifts**: Systemic changes across series
  - Controlled by `shared_level_shift_prob` parameter
  - Models policy changes or market-wide shifts
  
- **Enhanced Labels**: All events include `is_shared` boolean flag

#### 2. New Anomaly Type
- **Transient Change**: Temporary level shift that reverts after 3-8 days
- Models scenarios like temporary equipment outages or short-term promotions
- Provides more complex patterns for advanced anomaly detection

#### 3. Improved Holiday Bridge Logic
- Corrected bridge day effect to only apply on workdays adjacent to weekends
- Thursday holidays now correctly create Friday bridge days
- Tuesday holidays now correctly create Monday bridge days
- More realistic simulation of employee behavior around holidays

#### 4. Enhanced Event Detection
- Anomalies prevented from overlapping with holiday effects
- Better separation of event types for clearer ground truth
- Improved spacing between anomalies (minimum 7 days)

#### 5. Comprehensive Plotting
- Built-in `plot()` method with 4-panel visualization
- Professional formatting suitable for presentations
- All events clearly marked and annotated
- Configurable figure size and save options

---

## Use Cases

- ✅ **Anomaly Detection**: Evaluate detection accuracy with ground truth (including shared events)
- ✅ **Changepoint Detection**: Test trend and level shift detection algorithms
- ✅ **Holiday Modeling**: Assess model handling of calendar effects and bridge days
- ✅ **Seasonality**: Test models on stable vs. time-varying patterns
- ✅ **Forecasting**: General accuracy on complex, realistic patterns
- ✅ **Regressor Impact**: Evaluate external variable incorporation
- ✅ **Scale Robustness**: Test models on data at different magnitudes
- ✅ **Multi-series Correlation** (NEW): Test correlation-aware anomaly detection
- ✅ **Shared Event Detection** (NEW): Evaluate systemic shock identification

---

## Design Principles

1. **Labeled Components**: All events are labeled for precise evaluation
2. **Realistic Complexity**: Multiple interacting patterns mimicking real-world data
3. **Noise Separation**: Background noise always distinguishable from anomalies
4. **Stochastic Variation**: Seasonal patterns have noise, not fixed values
5. **Multiple Scales**: Tests scaling robustness across series
6. **Special Cases**: Business days, non-Gregorian holidays, variance regimes
7. **Reproducibility**: Same seed produces identical data every time
8. **Meaningful Changes**: Trend slope changes validated to exceed minimum thresholds for detectability
9. **Clear Event Separation**: Anomalies prevented from overlapping with holiday effects

---

## Tips and Best Practices

1. **Reproducibility**: Always set `random_seed` for consistent results
2. **Scale Testing**: Every 3rd series is 10x larger - test your scaling methods
3. **Business Days**: series_0 has NaN on weekends - test your NaN handling
4. **Holidays**: Use series_5 (Chinese New Year) and series_6 (Ramadan) for non-Gregorian calendars
5. **Noise Levels**: Start with 0.1, reduce to 0.05 for cleaner data
6. **Complexity**: Increase frequencies for more challenging evaluation data
7. **Shared Events**: Use `shared_anomaly_prob` > 0 to test multi-series detection
8. **Component Analysis**: Use `get_components()` to understand individual effects

---

## Example Output

```python
gen = generate_synthetic_daily_data(n_days=365, n_series=3)
gen.summary()
```

```
======================================================================
Synthetic Daily Data Generator Summary
======================================================================
Date range: 2015-01-01 00:00:00 to 2015-12-31 00:00:00
Number of days: 365
Number of series: 3
Random seed: 42

Series Characteristics:
----------------------------------------------------------------------

series_0:
  Scale factor: 10.0
  Noise-to-signal ratio: 0.100
  Trend changepoints: 1
  Level shifts: 0
  Anomalies: 3
  Holiday impacts: 12

series_1:
  Scale factor: 1.0
  Noise-to-signal ratio: 0.100
  Trend changepoints: 3
  Level shifts: 0
  Anomalies: 0
  Holiday impacts: 12

series_2:
  Scale factor: 1.0
  Noise-to-signal ratio: 0.100
  Trend changepoints: 1
  Level shifts: 0
  Anomalies: 1
  Holiday impacts: 12
  Seasonality changepoints: 2

======================================================================
```

---

## Dependencies

- **Required**: numpy, pandas (already required by AutoTS)
- **Optional**: matplotlib (for plotting only)
  - If matplotlib not installed, everything works except `plot()` method
  - Install with: `pip install matplotlib`


## Future Enhancements (Optional)

Possible extensions:
- Cointegrated series
- Granger causality patterns between series (leading/lagging indicators)
- Multiplicative seasonality

---

## Support

- **Full Examples**: See `autots/datasets/synthetic_examples.py`
- **Test Suite**: See `autots/datasets/test_synthetic.py`
- **Source Code**: See `autots/datasets/synthetic.py`

---

**The SyntheticDailyGenerator is production-ready and fully tested for evaluating AutoTS models' performance on changepoint detection, holiday modeling, anomaly detection (including shared anomalies), and general forecasting accuracy across correlated time series.**
