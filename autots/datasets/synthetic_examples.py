"""
Comprehensive examples for the SyntheticDailyGenerator

This file demonstrates all major features and use cases of the synthetic data generator.
Run this file directly to see all examples: python synthetic_examples.py
"""

from autots.datasets.synthetic import SyntheticDailyGenerator, generate_synthetic_daily_data
import pandas as pd
import numpy as np


def basic_example():
    """Basic usage example."""
    print("=" * 70)
    print("Example 1: Basic Generation")
    print("=" * 70)
    
    # Generate data with default parameters
    gen = generate_synthetic_daily_data(
        start_date='2018-01-01',
        n_days=1095,  # 3 years
        n_series=5,
        random_seed=42
    )
    
    # Get the data
    data = gen.get_data()
    print("\nGenerated data shape:", data.shape)
    print("\nFirst few rows:")
    print(data.head())
    
    # Print summary
    gen.summary()
    
    return gen


def custom_parameters_example():
    """Example with custom parameters for complex data."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Parameters (Complex Data)")
    print("=" * 70)
    
    gen = SyntheticDailyGenerator(
        start_date='2019-01-01',
        n_days=1460,  # 4 years
        n_series=8,
        random_seed=999,
        trend_changepoint_freq=1.0,      # More frequent trend changes
        level_shift_freq=0.3,             # More level shifts
        anomaly_freq=0.1,                 # More anomalies
        weekly_seasonality_strength=2.0,  # Stronger weekly pattern
        yearly_seasonality_strength=1.5,  # Stronger yearly pattern
        noise_level=0.05,                 # Lower noise (cleaner signal)
        include_regressors=False,
    )
    
    gen.summary()
    
    # Show event counts across all series
    print("\nEvent counts across all series:")
    all_labels = gen.get_all_labels()
    
    total_trend_cp = sum(len(cp) for cp in all_labels['trend_changepoints'].values())
    total_level_shifts = sum(len(ls) for ls in all_labels['level_shifts'].values())
    total_anomalies = sum(len(an) for an in all_labels['anomalies'].values())
    
    print(f"  Total trend changepoints: {total_trend_cp}")
    print(f"  Total level shifts: {total_level_shifts}")
    print(f"  Total anomalies: {total_anomalies}")


def shared_events_example():
    """Example with shared events across multiple series."""
    print("\n" + "=" * 70)
    print("Example 3: Shared Events (NEW)")
    print("=" * 70)
    
    gen = generate_synthetic_daily_data(
        start_date='2020-01-01',
        n_days=1095,
        n_series=10,
        random_seed=42,
        shared_anomaly_prob=0.3,      # 30% of anomalies are shared
        shared_level_shift_prob=0.4   # 40% of level shifts are shared
    )
    
    # Check which events are shared
    print("\nShared anomalies by series:")
    all_anomalies = gen.get_anomalies()
    for series, anomalies in all_anomalies.items():
        shared = [a for a in anomalies if a[4]]  # a[4] is is_shared flag
        if shared:
            print(f"  {series}: {len(shared)} shared anomalies")
    
    print("\nShared level shifts by series:")
    all_shifts = gen.get_level_shifts()
    for series, shifts in all_shifts.items():
        shared = [s for s in shifts if s[3]]  # s[3] is is_shared flag
        if shared:
            print(f"  {series}: {len(shared)} shared level shifts")


def regressor_example():
    """Example with external regressors."""
    print("\n" + "=" * 70)
    print("Example 4: With External Regressors")
    print("=" * 70)
    
    gen = generate_synthetic_daily_data(
        start_date='2020-01-01',
        n_days=730,  # 2 years
        n_series=3,
        random_seed=123,
        include_regressors=True
    )
    
    # Get regressors
    regressors = gen.get_regressors()
    print("\nGenerated regressors:")
    print(regressors.head(10))
    print("\nRegressor statistics:")
    print(regressors.describe())
    
    # Save to CSV with regressors
    output_path = '/tmp/synthetic_data_with_regressors.csv'
    gen.to_csv(output_path, include_regressors=True)
    print(f"\nData saved to {output_path}")


def access_labels_example(gen):
    """Comprehensive example of accessing various labels."""
    print("\n" + "=" * 70)
    print("Example 5: Accessing All Label Types")
    print("=" * 70)
    
    series_name = 'series_0'
    labels = gen.get_all_labels(series_name)
    
    print(f"\nLabels for {series_name}:")
    print("-" * 70)
    
    print("\n1. Trend Changepoints:")
    if labels['trend_changepoints']:
        for date, old_slope, new_slope in labels['trend_changepoints']:
            print(f"   {date.date()}: slope {old_slope:.4f} → {new_slope:.4f}")
    else:
        print("   None")
    
    print("\n2. Level Shifts:")
    if labels['level_shifts']:
        for date, magnitude, shift_type, is_shared in labels['level_shifts']:
            shared_str = " (SHARED)" if is_shared else ""
            print(f"   {date.date()}: {shift_type} shift of {magnitude:.2f}{shared_str}")
    else:
        print("   None")
    
    print("\n3. Anomalies:")
    if labels['anomalies']:
        for date, magnitude, anom_type, duration, is_shared in labels['anomalies']:
            shared_str = " (SHARED)" if is_shared else ""
            print(f"   {date.date()}: {anom_type}, mag={magnitude:.2f}, {duration}d{shared_str}")
    else:
        print("   None")
    
    print(f"\n4. Holiday Impacts: {len(labels['holiday_impacts'])} dates affected")
    if labels['holiday_impacts']:
        # Show top 5 holidays by impact
        sorted_holidays = sorted(labels['holiday_impacts'].items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:5]
        print("   Top 5 holidays by impact:")
        for date, impact in sorted_holidays:
            print(f"   {date.date()}: {impact:.2f}")
    
    print(f"\n5. Noise-to-Signal Ratio: {labels['noise_to_signal_ratio']:.3f}")
    print(f"6. Series Scale: {labels['series_scale']:.1f}")
    
    if labels['seasonality_changepoints']:
        print(f"\n7. Seasonality Changepoints: {len(labels['seasonality_changepoints'])}")
        for date, description in labels['seasonality_changepoints']:
            print(f"   {date.date()}: {description}")


def component_analysis_example(gen):
    """Example of decomposing series into components."""
    print("\n" + "=" * 70)
    print("Example 6: Component Decomposition")
    print("=" * 70)
    
    series_name = 'series_1'
    components = gen.get_components(series_name)
    
    print(f"\nComponents for {series_name}:")
    print("-" * 70)
    for component_name, component_array in components.items():
        mean_val = np.nanmean(component_array)
        std_val = np.nanstd(component_array)
        print(f"  {component_name:12s}: mean={mean_val:8.2f}, std={std_val:7.2f}")
    
    # Verify components sum to total
    data = gen.get_data()
    total_series = data[series_name].values
    component_sum = sum(components.values())
    
    matches = np.allclose(component_sum, total_series, equal_nan=True)
    print(f"\nVerification: Components sum to total? {matches}")


def plotting_examples(gen):
    """Examples of plotting with the built-in plot method."""
    print("\n" + "=" * 70)
    print("Example 7: Plotting")
    print("=" * 70)
    
    try:
        # Example 1: Plot a specific series
        print("\n1. Plotting series_0 (Business day series)...")
        gen.plot(
            series_name='series_0',
            save_path='/tmp/plot_series_0.png',
            show=False
        )
        print("   ✓ Saved to /tmp/plot_series_0.png")
        
        # Example 2: Plot series with saturating trend
        print("\n2. Plotting series_1 (Saturating trend)...")
        gen.plot(
            series_name='series_1',
            save_path='/tmp/plot_series_1.png',
            show=False
        )
        print("   ✓ Saved to /tmp/plot_series_1.png")
        
        # Example 3: Plot a random series
        print("\n3. Plotting a randomly selected series...")
        gen.plot(
            save_path='/tmp/plot_random.png',
            show=False
        )
        print("   ✓ Saved to /tmp/plot_random.png")
        
        # Example 4: Plot with custom figure size
        print("\n4. Plotting with custom figure size (20x14)...")
        gen.plot(
            series_name='series_2',
            figsize=(20, 14),
            save_path='/tmp/plot_large.png',
            show=False
        )
        print("   ✓ Saved to /tmp/plot_large.png")
        
        print("\n✓ All plots generated successfully!")
        print("Set show=True to display plots interactively")
        
    except ImportError:
        print("\n✗ matplotlib not available, skipping plotting examples")
        print("Install with: pip install matplotlib")


def evaluation_workflow_example(gen):
    """Example workflow for model evaluation."""
    print("\n" + "=" * 70)
    print("Example 8: Model Evaluation Workflow")
    print("=" * 70)
    
    series_name = 'series_0'
    labels = gen.get_all_labels(series_name)
    data = gen.get_data()
    
    # Get true anomaly dates
    true_anomaly_dates = [date for date, _, _, _, _ in labels['anomalies']]
    print(f"\nTrue anomalies for {series_name}: {len(true_anomaly_dates)} dates")
    if true_anomaly_dates:
        print("First 3 anomalies:")
        for date in true_anomaly_dates[:3]:
            print(f"  {date.date()}")
    
    # Get true changepoint dates (trend + level shifts)
    true_changepoint_dates = []
    true_changepoint_dates.extend([date for date, _, _ in labels['trend_changepoints']])
    true_changepoint_dates.extend([date for date, _, _, _ in labels['level_shifts']])
    true_changepoint_dates = sorted(set(true_changepoint_dates))
    
    print(f"\nTrue changepoints (trend + level shifts): {len(true_changepoint_dates)} dates")
    if true_changepoint_dates:
        print("First 3 changepoints:")
        for date in true_changepoint_dates[:3]:
            print(f"  {date.date()}")
    
    # Get holiday-affected dates
    holiday_dates = list(labels['holiday_impacts'].keys())
    print(f"\nHoliday-affected dates: {len(holiday_dates)} dates")
    
    print("\nThese labels can be used to:")
    print("  1. Calculate precision/recall for anomaly detection")
    print("  2. Evaluate changepoint detection accuracy")
    print("  3. Assess holiday effect modeling")
    print("  4. Compare detected vs. true noise regime changes")
    print("  5. Test shared event detection algorithms")


def special_series_example():
    """Demonstrate special series types."""
    print("\n" + "=" * 70)
    print("Example 9: Special Series Types")
    print("=" * 70)
    
    gen = generate_synthetic_daily_data(
        start_date='2020-01-01',
        n_days=730,
        n_series=8,
        random_seed=42
    )
    
    data = gen.get_data()
    
    print("\nSpecial series characteristics:")
    print("-" * 70)
    
    # Series 0: Business days
    print("\nseries_0 (Business days):")
    print(f"  NaN count: {data['series_0'].isna().sum()} days")
    print(f"  Scale: {gen.get_all_labels('series_0')['series_scale']:.1f}x")
    
    # Series 1: Saturating trend
    print("\nseries_1 (Saturating trend):")
    labels = gen.get_all_labels('series_1')
    print(f"  Trend changepoints: {len(labels['trend_changepoints'])}")
    print(f"  Has nonlinear/logistic components")
    
    # Series 2: Time-varying seasonality
    print("\nseries_2 (Time-varying seasonality):")
    print(f"  Seasonality changes gradually over time")
    
    # Series 3: Seasonality changepoints
    print("\nseries_3 (Seasonality changepoints):")
    labels = gen.get_all_labels('series_3')
    if labels['seasonality_changepoints']:
        print(f"  Changepoints: {len(labels['seasonality_changepoints'])}")
    print(f"  Scale: {labels['series_scale']:.1f}x")
    
    # Series 4: No level shifts
    print("\nseries_4 (No level shifts - control):")
    labels = gen.get_all_labels('series_4')
    print(f"  Level shifts: {len(labels['level_shifts'])} (should be 0)")
    
    # Series 7: GARCH variance
    print("\nseries_7 (GARCH-like variance):")
    print(f"  Has volatility regime switches")


def clean_vs_noisy_example():
    """Compare clean and noisy data generation."""
    print("\n" + "=" * 70)
    print("Example 10: Clean vs. Noisy Data")
    print("=" * 70)
    
    # Clean data
    print("\nGenerating CLEAN data (low noise)...")
    gen_clean = generate_synthetic_daily_data(
        n_days=365,
        n_series=3,
        noise_level=0.02,  # Very low noise
        random_seed=42
    )
    
    # Noisy data
    print("\nGenerating NOISY data (high noise)...")
    gen_noisy = generate_synthetic_daily_data(
        n_days=365,
        n_series=3,
        noise_level=0.2,  # High noise
        random_seed=42
    )
    
    # Compare
    data_clean = gen_clean.get_data()
    data_noisy = gen_noisy.get_data()
    
    print("\nComparison for series_0:")
    print(f"  Clean data std:  {data_clean['series_0'].std():.2f}")
    print(f"  Noisy data std:  {data_noisy['series_0'].std():.2f}")
    
    labels_clean = gen_clean.get_all_labels('series_0')
    labels_noisy = gen_noisy.get_all_labels('series_0')
    
    print(f"  Clean noise-to-signal:  {labels_clean['noise_to_signal_ratio']:.3f}")
    print(f"  Noisy noise-to-signal:  {labels_noisy['noise_to_signal_ratio']:.3f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("SYNTHETIC DAILY DATA GENERATOR - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    gen = basic_example()
    custom_parameters_example()
    shared_events_example()
    regressor_example()
    access_labels_example(gen)
    component_analysis_example(gen)
    plotting_examples(gen)
    evaluation_workflow_example(gen)
    special_series_example()
    clean_vs_noisy_example()
    
    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - Documentation: autots/datasets/SYNTHETIC_DATA_GUIDE.md")
    print("  - Test suite: autots/datasets/test_synthetic.py")
    print("  - Source code: autots/datasets/synthetic.py")


if __name__ == '__main__':
    main()
