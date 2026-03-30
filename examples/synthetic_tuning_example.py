# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from autots.datasets import SyntheticDailyGenerator
from autots.evaluator.feature_detector import (
    TimeSeriesFeatureDetector,
    FeatureDetectionOptimizer,
)
from autots.evaluator.feature_detector.loss import FeatureDetectionLoss

# 1. Define synthetic parameters (known_good_bp)
known_good_bp = {  # second tuning
    'trend_changepoint_freq': 1.1311196948895457,
    'level_shift_freq': 0.11235106798095876,
    'level_shift_strength': 0.10391796521529073,
    'anomaly_freq': 0.04654080084797309,
    'weekly_seasonality_strength': 0.8646796590839558,
    'yearly_seasonality_strength': 1.2232927741109423,
    'noise_level': 0.0016485471075049773,  # about 0.5275%
    'trend_slope_scale': 0.4964492705041623,
    'trend_positive_bias': 0.7313495354794045,
    'level_shift_minimum_pct': 0.0944925100542184,
    'level_shift_max_pct': 0.03147911401873594,
    'noise_ar_coefficient': 0.07932349693757101,
    'volatility_regime_intensity': 1.3960299844487944,
    'holiday_signal_floor_pct': 0.0025,
    'random_dom_holiday_count': 1,
    'random_wkdom_holiday_count': 1,
    'anomaly_types': ['point_outlier', 'transient_change'],
}

# 2. Create synthetic data
print("\nGenerating synthetic data...")
generator = SyntheticDailyGenerator(
    start_date='2020-01-01',
    n_days=365 * 3,
    n_series=5,
    random_seed=42,
    **known_good_bp
)
data = generator.get_data()

# 3. Plot sample of synthetic data
print("\nPlotting synthetic data for series_0...")
generator.plot(series_name='series_0')

# 4. Define starting feature detector parameters (from assure_params.md)
starting_params = {
    'rough_seasonality_params': {
        'regression_model': {
            'model': 'RandomForest',
            'model_params': {
                'n_estimators': 100,
                'min_samples_leaf': 2,
                'bootstrap': True,
                'max_depth': 20,
                'max_features': 1.0,
                'min_samples_split': 2,
                'max_leaf_nodes': None,
                'max_samples': 0.8
            }
        },
        'datepart_method': 'expanded',
        'polynomial_degree': None,
        'transform_dict': None,
        'holiday_countries_used': False,
        'lags': None,
        'forward_lags': None
    },
    'seasonality_params': {
        'regression_model': {
            'model': 'FastRidge',
            'model_params': {}
        },
        'datepart_method': ['dayofweek', 365.25],
        'polynomial_degree': None,
        'transform_dict': None,
        'holiday_countries_used': False,
        'lags': None,
        'forward_lags': None
    },
    'holiday_params': {
        'threshold': 0.6,
        'splash_threshold': 0.45,
        'min_occurrences': 2,
        'use_dayofmonth_holidays': True,
        'use_wkdom_holidays': True,
        'use_wkdeom_holidays': False,
        'use_lunar_holidays': False,
        'use_lunar_weekday': False,
        'use_islamic_holidays': False,
        'use_hebrew_holidays': False,
        'use_hindu_holidays': False,
        'auto_relax': True,
        'relax_threshold_floor': 0.55,
        'relax_splash_threshold': 0.55,
        'relax_rounds': 3,
        'min_holidays_per_series': 1,
        'max_holidays_per_series': 24,
        'holiday_selection_strategy': 'score',
        'anomaly_detector_params': {
            'method': 'EE',
            'transform_dict': {
                'transformations': {0: 'DatepartRegression'},
                'transformation_params': {0: {
                    'datepart_method': 'simple_3',
                    'regression_model': {'model': 'FastRidge', 'model_params': {}}
                }}
            },
            'forecast_params': None,
            'method_params': {
                'contamination': 0.08,
                'assume_centered': False,
                'support_fraction': None
            }
        },
        'output': 'multivariate'
    },
    'anomaly_params': {
        'output': 'multivariate',
        'method': 'rolling_zscore',
        'method_params': {
            'distribution': 'norm',
            'alpha': 0.05,
            'rolling_periods': 90,
            'center': False
        },
        'fillna': 'ffill'
    },
    'changepoint_params': {
        'method': 'pelt',
        'method_params': {
            'penalty': 50,
            'loss_function': 'l2',
            'min_segment_length': 15,
            'pruning_factor': 1.0
        },
        'aggregate_method': 'mean',
        'min_segment_length': 15,
        'probabilistic_output': False
    },
    'level_shift_params': {
        'window_size': 14,
        'alpha': 3.5,
        'grouping_forward_limit': 4,
        'max_level_shifts': 3,
        'alignment': 'rolling_diff',
        'output': 'multivariate',
        'remove_at_shift': False,
        'shift_remove_window': 1,
        'shift_fillna': 'mean',
        'window_method': 'overlap'
    },
    'general_transformer_params': {
        'fillna': 'ffill',
        'transformations': {0: 'FFTFilter'},
        'transformation_params': {0: {
            'cutoff': 0.05,
            'reverse': False,
            'on_transform': True,
            'on_inverse': False
        }}
    },
    'standardize': True,
    'smoothing_window': 7
}

starting_params_newer = {
   'rough_seasonality_params': {'regression_model': {'model': 'RandomForest',
   'model_params': {'n_estimators': 100,
    'min_samples_leaf': 2,
    'bootstrap': True,
    'max_depth': 20,
    'max_features': 1.0,
    'min_samples_split': 2,
    'max_leaf_nodes': None,
    'max_samples': 0.8}},
  'datepart_method': 'expanded',
  'polynomial_degree': None,
  'transform_dict': None,
  'holiday_countries_used': False,
  'lags': None,
  'forward_lags': None},
 'seasonality_params': {'regression_model': {'model': 'ElasticNet',
   'model_params': {'l1_ratio': 0.5,
    'fit_intercept': False,
    'selection': 'cyclic',
    'max_iter': 1000}},
  'datepart_method': [7, 365.25],
  'polynomial_degree': None,
  'transform_dict': {'fillna': None,
   'transformations': {'0': 'ClipOutliers'},
   'transformation_params': {'0': {'method': 'clip', 'std_threshold': 4}}},
  'holiday_countries_used': False,
  'lags': 1,
  'forward_lags': None},
 'holiday_params': {'threshold': 0.9,
  'splash_threshold': None,
  'use_dayofmonth_holidays': True,
  'use_wkdom_holidays': True,
  'use_wkdeom_holidays': False,
  'use_lunar_holidays': False,
  'use_lunar_weekday': False,
  'use_islamic_holidays': False,
  'use_hebrew_holidays': True,
  'use_hindu_holidays': False,
  'anomaly_detector_params': {'method': 'IsolationForest',
   'transform_dict': {'fillna': 'rolling_mean',
    'transformations': {0: 'AlignLastValue'},
    'transformation_params': {0: {'rows': 1,
      'lag': 7,
      'method': 'additive',
      'strength': 1.0,
      'first_value_only': True,
      'threshold': 10,
      'threshold_method': 'max',
      'mean_type': 'arithmetic'}}},
   'forecast_params': None,
   'method_params': {'contamination': 0.05,
    'n_estimators': 50,
    'max_features': 0.5,
    'bootstrap': False}},
  'auto_relax': False,
  'relax_threshold_floor': 0.65,
  'relax_splash_threshold': 0.45,
  'relax_rounds': 3,
  'min_holidays_per_series': 1,
  'max_holidays_per_series': None,
  'holiday_selection_strategy': 'score',
  'output': 'multivariate'},
 'anomaly_params': {'output': 'multivariate',
  'method': 'rolling_zscore',
  'method_params': {'distribution': 'gamma',
   'alpha': 0.05,
   'rolling_periods': 200,
   'center': True},
  'fillna': 'ffill',
  'holiday_proximity_days': 2},
 'changepoint_params': {'method': 'basic',
  'method_params': {'changepoint_spacing': 180,
   'changepoint_distance_end': 360},
  'aggregate_method': 'median',
  'min_segment_length': 5,
  'probabilistic_output': False},
 'level_shift_params': {'window_size': 90,
  'alpha': 3.5,
  'grouping_forward_limit': 6,
  'max_level_shifts': 10,
  'alignment': 'average',
  'output': 'multivariate',
  'remove_at_shift': True,
  'shift_remove_window': 1,
  'shift_fillna': 'mean',
  'window_method': 'diff_overlap'},
 'general_transformer_params': {'fillna': 'ffill',
  'transformations': {0: 'KalmanSmoothing', 1: 'FIRFilter'},
  'transformation_params': {0: {'model_name': 'theta_equivalent',
    'state_transition': [[1, 1], [0, 1]],
    'process_noise': [[0.0001, 0], [0, 0]],
    'observation_model': [[1, 0]],
    'observation_noise': 0.1,
    'em_iter': None,
    'on_transform': True,
    'on_inverse': False},
   1: {'numtaps': 128,
    'cutoff_hz': 0.5,
    'window': 'hann',
    'sampling_frequency': 364,
    'on_transform': True,
    'on_inverse': False,
    'bounds_only': False}}},
 'standardize': False,
 'smoothing_window': 7
}
best_50000 = {
    'rough_seasonality_params': {'regression_model': {'model': 'RandomForest', 'model_params': {'n_estimators': 300, 'min_samples_leaf': 1, 'bootstrap': True, 'max_depth': None, 'max_features': 1.0, 'min_samples_split': 4, 'max_leaf_nodes': None, 'max_samples': 0.8}}, 'datepart_method': 'expanded_binarized', 'polynomial_degree': None, 'transform_dict': None, 'holiday_countries_used': False, 'lags': None, 'forward_lags': None},
    'seasonality_params': {'regression_model': {'model': 'ElasticNet', 'model_params': {'l1_ratio': 0.5, 'fit_intercept': False, 'selection': 'cyclic', 'max_iter': 1000}}, 'datepart_method': 'common_fourier', 'polynomial_degree': None, 'transform_dict': {'fillna': None, 'transformations': {'0': 'EWMAFilter'}, 'transformation_params': {'0': {'span': 7}}}, 'holiday_countries_used': False, 'lags': None, 'forward_lags': None},
    'holiday_params': {
        'threshold': 0.8, 'splash_threshold': 0.85, 'use_dayofmonth_holidays': True, 'use_wkdom_holidays': True, 'use_wkdeom_holidays': False, 'use_lunar_holidays': False, 'use_lunar_weekday': False, 'use_islamic_holidays': False, 'use_hebrew_holidays': False, 'use_hindu_holidays': False,
        'anomaly_detector_params': {'method': 'nonparametric', 'transform_dict': None, 'forecast_params': None, 'method_params': {'p': None, 'z_init': 1.5, 'z_limit': 10, 'z_step': 0.5, 'inverse': False, 'max_contamination': 0.25, 'mean_weight': 25, 'sd_weight': 25, 'anomaly_count_weight': 1.0}}, 'auto_relax': False, 'relax_threshold_floor': 0.65, 'relax_splash_threshold': 0.65, 'relax_rounds': 2, 'min_holidays_per_series': 1, 'max_holidays_per_series': None, 'holiday_selection_strategy': 'coverage', 'output': 'multivariate'
    },
    'anomaly_params': {'output': 'multivariate', 'method': 'rolling_zscore', 'method_params': {'distribution': 'gamma', 'alpha': 0.05, 'rolling_periods': 200, 'center': True}, 'fillna': 'ffill'},
    'changepoint_params': {'method': 'ewma', 'method_params': {'lambda_param': 0.2, 'control_limit': 3.0, 'normalize': False, 'two_sided': False, 'adaptive': False, 'min_distance': 10}, 'aggregate_method': 'mean', 'min_segment_length': 5, 'probabilistic_output': False},
    'level_shift_params': {'window_size': 364, 'alpha': 3.5, 'grouping_forward_limit': 5, 'max_level_shifts': 30, 'alignment': 'rolling_diff_3nn', 'output': 'multivariate', 'remove_at_shift': False, 'shift_remove_window': 0, 'shift_fillna': 'cubic', 'window_method': 'diff_overlap'},
    'general_transformer_params': {'fillna': 'ffill', 'transformations': {0: 'KalmanSmoothing', 1: 'bkfilter'}, 'transformation_params': {0: {'model_name': 'ucm_deterministictrend_seasonal7', 'state_transition': [[1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, -1, -1, -1, -1, -1, -1]], 'process_noise': [[0.001, 0, 0, 0, 0, 0, 0, 0], [0, 0.001, 0, 0, 0, 0, 0, 0], [0, 0, 0.001, 0, 0, 0, 0, 0], [0, 0, 0, 0.001, 0, 0, 0, 0], [0, 0, 0, 0, 0.001, 0, 0, 0], [0, 0, 0, 0, 0, 0.001, 0, 0], [0, 0, 0, 0, 0, 0, 0.001, 0], [0, 0, 0, 0, 0, 0, 0, 0]], 'observation_model': [[1, 0, 1, 1, 1, 1, 1, 1]], 'observation_noise': 0.05, 'em_iter': 10, 'on_transform': True, 'on_inverse': False}, 1: {}}}, 'standardize': False, 'smoothing_window': None
}
best_20k = {
    'rough_seasonality_params': {'regression_model': {'model': 'RandomForest',
      'model_params': {'n_estimators': 100,
       'min_samples_leaf': 1,
       'bootstrap': True,
       'max_depth': None,
       'max_features': 1.0,
       'min_samples_split': 2,
       'max_leaf_nodes': None,
       'max_samples': 0.8}},
     'datepart_method': 'expanded',
     'polynomial_degree': None,
     'transform_dict': None,
     'holiday_countries_used': False,
     'lags': None,
     'forward_lags': 1},
    'seasonality_params': {'regression_model': {'model': 'ElasticNet',
      'model_params': {'l1_ratio': 0.5,
       'fit_intercept': False,
       'selection': 'cyclic',
       'max_iter': 1000}},
     'datepart_method': 'common_fourier',
     'polynomial_degree': None,
     'transform_dict': {'fillna': None,
      'transformations': {'0': 'EWMAFilter'},
      'transformation_params': {'0': {'span': 7}}},
     'holiday_countries_used': False,
     'lags': None,
     'forward_lags': None},
    'holiday_params': {'threshold': 0.8,
     'splash_threshold': 0.85,
     'use_dayofmonth_holidays': True,
     'use_wkdom_holidays': True,
     'use_wkdeom_holidays': False,
     'use_lunar_holidays': False,
     'use_lunar_weekday': False,
     'use_islamic_holidays': False,
     'use_hebrew_holidays': False,
     'use_hindu_holidays': False,
     'anomaly_detector_params': {'method': 'nonparametric',
      'transform_dict': None,
      'forecast_params': None,
      'method_params': {'p': None,
       'z_init': 1.5,
       'z_limit': 10,
       'z_step': 0.5,
       'inverse': False,
       'max_contamination': 0.25,
       'mean_weight': 25,
       'sd_weight': 25,
       'anomaly_count_weight': 1.0}},
     'auto_relax': False,
     'relax_threshold_floor': 0.65,
     'relax_splash_threshold': 0.65,
     'relax_rounds': 2,
     'min_holidays_per_series': 1,
     'max_holidays_per_series': None,
     'holiday_selection_strategy': 'coverage',
     'output': 'multivariate'},
    'anomaly_params': {'output': 'multivariate',
     'method': 'rolling_zscore',
     'method_params': {'distribution': 'gamma',
      'alpha': 0.05,
      'rolling_periods': 200,
      'center': True},
     'fillna': 'ffill'},
    'changepoint_params': {'method': 'ewma', 'method_params': {'lambda_param': 0.2, 'control_limit': 3.0, 'normalize': False, 'two_sided': False, 'adaptive': False, 'min_distance': 10}, 'aggregate_method': 'mean', 'min_segment_length': 5, 'probabilistic_output': False},
    'level_shift_params': {'window_size': 90,
     'alpha': 4.0,
     'grouping_forward_limit': 3,
     'max_level_shifts': 10,
     'alignment': 'average',
     'output': 'multivariate',
     'remove_at_shift': True,
     'shift_remove_window': 0,
     'shift_fillna': 'ffill',
     'window_method': 'overlap'},
    'general_transformer_params': {'fillna': 'pchip',
     'transformations': {0: 'KalmanSmoothing', 1: 'ClipOutliers'},
     'transformation_params': {0: {'model_name': 'local linear hidden state with seasonal 7',
       'state_transition': [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
       'process_noise': [[0.0016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
       'observation_model': [[1, 1, 0, 0, 0, 0, 0, 0]],
       'observation_noise': 1.0,
       'em_iter': None,
       'on_transform': True,
       'on_inverse': False},
      1: {'method': 'clip', 'std_threshold': 4.5, 'fillna': None}}},
    'standardize': False,
    'smoothing_window': 3
}

# 5. Fit feature detector with starting parameters
print("\nFitting feature detector with starting parameters...")
detector = TimeSeriesFeatureDetector(**starting_params)
detector.fit(data)


def _summarize_detector(label, detector_obj, generator_obj):
    """Print truth-vs-detected benchmark metrics for a fitted detector."""
    loss_calc = FeatureDetectionLoss()
    detected = detector_obj.get_detected_features(include_components=True)
    truth = generator_obj.get_all_labels()
    components = generator_obj.get_components()
    loss = loss_calc.calculate_loss(
        detected,
        truth,
        true_components=components,
        date_index=generator_obj.date_index,
    )
    series_name = 'series_0'
    detected_series = {
        'cp': len(detected.get('trend_changepoints', {}).get(series_name, [])),
        'ls': len(detected.get('level_shifts', {}).get(series_name, [])),
        'anom': len(detected.get('anomalies', {}).get(series_name, [])),
        'hol': len(detected.get('holiday_dates', {}).get(series_name, [])),
        'strengths': detected.get('series_seasonality_strengths', {}).get(series_name, {}),
    }
    true_series = {
        'cp': len(truth.get('trend_changepoints', {}).get(series_name, [])),
        'ls': len(truth.get('level_shifts', {}).get(series_name, [])),
        'anom': len(truth.get('anomalies', {}).get(series_name, [])),
        'hol': len(truth.get('holiday_dates', {}).get(series_name, [])),
        'strengths': truth.get('series_seasonality_strengths', {}).get(series_name, {}),
    }
    print("\n" + "-" * 80)
    print(f"{label} benchmark summary")
    print("-" * 80)
    print(
        f"total_loss={loss['total_loss']:.4f}  trend={loss['trend_loss']:.4f}  "
        f"anomaly={loss['anomaly_loss']:.4f}  holiday_event={loss['holiday_event_loss']:.4f}  "
        f"seasonality_pattern={loss['seasonality_pattern_loss']:.4f}"
    )
    print(f"series_0 truth    : {true_series}")
    print(f"series_0 detected : {detected_series}")

# 6. Plot detector result
print("Plotting detector output for series_0...")
detector.plot(series_name='series_0')
_summarize_detector("Starting params", detector, generator)

# 7. Run feature detector tuning (optimization) on the synthetic data
print("\nRunning feature detector tuning (FeatureDetectionOptimizer)...")
# Increase n_iterations for real-world tuning
optimizer = FeatureDetectionOptimizer(
    generator,
    n_iterations=10000, 
    starting_params=best_20k,
    random_seed=42
)

optimizer.optimize()
best_params = optimizer.best_params
optimizer.get_optimization_summary()
results = optimizer.history_df

print("\n" + "=" * 80)
print("Optimization Complete.")
print("Best Parameters:")
print(best_params)

# 8. Fit detector with best params
print("\nFitting feature detector with optimized parameters...")
detector_optimized = TimeSeriesFeatureDetector(**best_params)
detector_optimized.fit(data)
_summarize_detector("Optimized params", detector_optimized, generator)

# 9. Plot result of optimized detector
print("Plotting optimized detector output for series_0...")
generator.plot(series_name='series_0')
detector_optimized.plot(series_name='series_0')

generator.plot(series_name='series_1')
detector_optimized.plot(series_name='series_1')
print("\nExample script finished.")

best_params_ft = optimizer.fine_tune_changepoints(best_params, n_per_stage=1000)

print("\nFitting feature detector with optimized parameters...")
detector_optimized_ft = TimeSeriesFeatureDetector(**best_params_ft)
detector_optimized_ft.fit(data)
_summarize_detector("Fine-tuned changepoints", detector_optimized_ft, generator)

print("Plotting optimized detector output for series_0...")
generator.plot(series_name='series_0')
detector_optimized_ft.plot(series_name='series_0')

generator.plot(series_name='series_1')
detector_optimized_ft.plot(series_name='series_1')
