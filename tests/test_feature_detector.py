# -*- coding: utf-8 -*-
"""
Tests for Feature Detector

@author: Colin
"""

import unittest
from unittest.mock import patch
import ast
from pathlib import Path
import pandas as pd
import numpy as np
import random
import time
import json
from autots.datasets import load_daily
from autots.datasets.synthetic import SyntheticDailyGenerator
from autots.models.base import PredictionObject
from autots.evaluator.anomaly_detector import HolidayDetector
from autots.evaluator.feature_detector import (
    TimeSeriesFeatureDetector,
    FeatureDetectionLoss,
    ReconstructionLoss,
    FeatureDetectionOptimizer,
)


class TestFeatureDetector(unittest.TestCase):
    """Test TimeSeriesFeatureDetector class."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic data once for all tests."""
        cls.generator = SyntheticDailyGenerator(
            start_date='2020-01-01',
            n_days=730,  # 2 years
            n_series=3,
            random_seed=42,
            trend_changepoint_freq=0.5,
            level_shift_freq=0.1,
            anomaly_freq=0.05,
            weekly_seasonality_strength=1.0,
            yearly_seasonality_strength=0.5,
            noise_level=0.1,
        )
        cls.data = cls.generator.get_data()
        cls.labels = cls.generator.get_all_labels()

    def _make_manual_event_dag_detector(self, detection_mode='multivariate'):
        detector = TimeSeriesFeatureDetector(detection_mode=detection_mode)
        detector.df_original = pd.DataFrame(
            0.0,
            index=pd.date_range('2024-01-01', periods=45, freq='D'),
            columns=['a', 'b', 'c'],
        )
        detector.date_index = detector.df_original.index
        detector._reset_results()
        return detector

    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = TimeSeriesFeatureDetector()
        self.assertIsNotNone(detector)
        self.assertTrue(detector.standardize)
        self.assertEqual(detector.detection_mode, 'multivariate')  # Default value

    def test_detection_mode_multivariate(self):
        """Test multivariate detection mode initialization."""
        detector = TimeSeriesFeatureDetector(detection_mode='multivariate')

        # Verify detection_mode is set
        self.assertEqual(detector.detection_mode, 'multivariate')

        # Verify parameters are correctly configured
        self.assertEqual(detector.anomaly_params['output'], 'multivariate')
        self.assertEqual(detector.holiday_params['output'], 'multivariate')
        self.assertEqual(detector.changepoint_params['aggregate_method'], 'individual')

    def test_detection_mode_univariate(self):
        """Test univariate detection mode initialization."""
        detector = TimeSeriesFeatureDetector(detection_mode='univariate')

        # Verify detection_mode is set
        self.assertEqual(detector.detection_mode, 'univariate')

        # Verify parameters are correctly configured
        self.assertEqual(detector.anomaly_params['output'], 'univariate')
        self.assertEqual(detector.holiday_params['output'], 'univariate')
        self.assertEqual(detector.changepoint_params['aggregate_method'], 'mean')

    def test_detection_mode_parameter_override(self):
        """Test that detection_mode overrides manually specified parameters."""
        # Specify univariate mode but provide multivariate params
        detector = TimeSeriesFeatureDetector(
            detection_mode='univariate',
            anomaly_params={'output': 'multivariate', 'method': 'zscore'},
            holiday_params={'output': 'multivariate'},
            changepoint_params={'method': 'pelt'},  # Don't specify aggregate_method
        )

        # Verify that detection_mode takes precedence for anomaly and holiday
        self.assertEqual(detector.anomaly_params['output'], 'univariate')
        self.assertEqual(detector.holiday_params['output'], 'univariate')

        # Verify other params are preserved
        self.assertEqual(detector.anomaly_params['method'], 'zscore')

        # Changepoint should use univariate default since aggregate_method not specified
        self.assertEqual(detector.changepoint_params['aggregate_method'], 'mean')

    def test_detection_mode_changepoint_explicit_override(self):
        """Test that explicitly set aggregate_method is NOT overridden."""
        # When user explicitly sets aggregate_method, it should be respected
        detector = TimeSeriesFeatureDetector(
            detection_mode='univariate',
            changepoint_params={'aggregate_method': 'individual', 'method': 'pelt'},
        )

        # Explicit aggregate_method should be preserved
        self.assertEqual(detector.changepoint_params['aggregate_method'], 'individual')

        # But using 'auto' should trigger override
        detector2 = TimeSeriesFeatureDetector(
            detection_mode='univariate',
            changepoint_params={'aggregate_method': 'auto', 'method': 'pelt'},
        )
        self.assertEqual(detector2.changepoint_params['aggregate_method'], 'mean')

    def test_detection_mode_invalid(self):
        """Test that invalid detection_mode raises ValueError."""
        with self.assertRaises(ValueError) as context:
            TimeSeriesFeatureDetector(detection_mode='invalid')

        self.assertIn('multivariate', str(context.exception).lower())
        self.assertIn('univariate', str(context.exception).lower())

    def test_get_new_params_enforces_multivariate_changepoint_aggregation(self):
        """Sampled params should default to per-series changepoints."""
        for _ in range(20):
            params = TimeSeriesFeatureDetector.get_new_params(method='random')
            self.assertEqual(
                params['changepoint_params'].get('aggregate_method'),
                'individual',
            )

    def test_detector_fit(self):
        """Test detector can fit data."""
        detector = TimeSeriesFeatureDetector()
        detector.fit(self.data)

        # Check that features were detected
        self.assertIsNotNone(detector.trend_changepoints)
        self.assertIsNotNone(detector.anomalies)
        self.assertIsNotNone(detector.seasonality_strength)

    def test_multivariate_detection(self):
        """Test that multivariate mode produces series-specific detections."""
        detector = TimeSeriesFeatureDetector(detection_mode='multivariate')
        detector.fit(self.data)

        # Each series should potentially have different anomalies
        anomaly_lists = [detector.anomalies.get(col, []) for col in self.data.columns]

        # At least check that we got results for each series
        for col in self.data.columns:
            self.assertIn(col, detector.anomalies)
            self.assertIn(col, detector.trend_changepoints)

    def test_univariate_detection(self):
        """Test that univariate mode produces shared detections."""
        detector = TimeSeriesFeatureDetector(detection_mode='univariate')
        detector.fit(self.data)

        # All series should have detections
        series_names = list(self.data.columns)
        if len(series_names) > 1:
            # Get anomalies for first two series
            anomalies_0 = detector.anomalies.get(series_names[0], [])
            anomalies_1 = detector.anomalies.get(series_names[1], [])

            # Extract dates (handle both dict and tuple formats)
            def extract_dates(anomaly_list):
                dates = []
                for a in anomaly_list:
                    if isinstance(a, dict):
                        dates.append(a['date'])
                    elif isinstance(a, (tuple, list)):
                        dates.append(a[0])
                return set(dates)

            dates_0 = extract_dates(anomalies_0)
            dates_1 = extract_dates(anomalies_1)

            # In univariate mode, all series should share the same anomaly dates
            self.assertEqual(
                dates_0,
                dates_1,
                "Univariate mode should produce identical anomaly dates across series",
            )

    def test_template_metadata_includes_detection_mode(self):
        """Test that template metadata includes detection_mode."""
        for mode in ['multivariate', 'univariate']:
            detector = TimeSeriesFeatureDetector(detection_mode=mode)
            detector.fit(self.data)

            template = detector.get_template()
            meta = template['meta']
            self.assertEqual(meta.get('source'), 'TimeSeriesFeatureDetector')
            self.assertIn('config', meta)
            self.assertIn('detection_mode', meta['config'])
            self.assertEqual(meta['config']['detection_mode'], mode)
            self.assertNotIn('detector_config', meta)

    def test_forecast_prediction_object(self):
        """Ensure forecast helper returns a PredictionObject with expected shapes."""
        detector = TimeSeriesFeatureDetector()
        detector.fit(self.data)
        horizon = 7
        prediction = detector.forecast(horizon)
        self.assertIsInstance(prediction, PredictionObject)
        self.assertEqual(prediction.forecast.shape, (horizon, self.data.shape[1]))
        self.assertGreater(prediction.forecast.index[0], self.data.index[-1])
        # Components are in a MultiIndex DataFrame with (series, component) structure
        self.assertIsNotNone(prediction.components)
        self.assertIsInstance(prediction.components.columns, pd.MultiIndex)
        # Check that expected components exist at level 1 of the MultiIndex
        component_names = prediction.components.columns.get_level_values(1).unique()
        for comp in ['trend', 'seasonality', 'holidays']:
            self.assertIn(comp, component_names)
        # Check shape: each series has 4 components (trend, level_shift, seasonality, holidays)
        expected_cols = self.data.shape[1] * 4  # 4 components per series
        self.assertEqual(prediction.components.shape, (horizon, expected_cols))

    def test_level_shift_output_parameter(self):
        """Test that level_shift_params includes output parameter matching detection_mode."""
        # Multivariate mode
        detector_multi = TimeSeriesFeatureDetector(detection_mode='multivariate')
        self.assertEqual(detector_multi.level_shift_params['output'], 'multivariate')

        # Univariate mode
        detector_uni = TimeSeriesFeatureDetector(detection_mode='univariate')
        self.assertEqual(detector_uni.level_shift_params['output'], 'univariate')

        # Override test
        detector_override = TimeSeriesFeatureDetector(
            detection_mode='univariate',
            level_shift_params={'window_size': 30, 'output': 'multivariate'},
        )
        # Output should be overridden to match detection_mode
        self.assertEqual(detector_override.level_shift_params['output'], 'univariate')

    def test_get_detected_features(self):
        """Test getting detected features."""
        detector = TimeSeriesFeatureDetector()
        detector.fit(self.data)

        # Get all features
        all_features = detector.get_detected_features()
        self.assertIn('trend_changepoints', all_features)
        self.assertIn('anomalies', all_features)
        self.assertEqual(
            all_features['holiday_splash_impacts'], detector.holiday_splash_impacts
        )

        # Get features for specific series
        series_name = self.data.columns[0]
        series_features = detector.get_detected_features(series_name)
        self.assertIn('trend_changepoints', series_features)
        self.assertEqual(
            series_features['holiday_splash_impacts'],
            detector.holiday_splash_impacts.get(series_name, {}),
        )

        all_features_meta = detector.get_detected_features(include_metadata=True)
        self.assertIn('series_types', all_features_meta)
        self.assertIn('season_types', all_features_meta)
        self.assertIn(series_name, all_features_meta['series_types'])
        self.assertIn(series_name, all_features_meta['season_types'])

        series_features_meta = detector.get_detected_features(
            series_name, include_metadata=True
        )
        self.assertIn('series_type', series_features_meta)
        self.assertIn('season_type', series_features_meta)
        self.assertIsInstance(series_features_meta['series_type'], str)
        self.assertIsInstance(series_features_meta['season_type'], str)

    def test_event_dag_clusters_same_window_events_and_preserves_raw_records(self):
        detector = self._make_manual_event_dag_detector()
        detector.anomalies = {
            'a': [
                (pd.Timestamp('2024-01-10'), 3.0, 'point_outlier', 1, False),
                (pd.Timestamp('2024-01-20'), 2.5, 'point_outlier', 1, False),
            ],
            'b': [
                (pd.Timestamp('2024-01-11'), 2.0, 'point_outlier', 1, False),
                (pd.Timestamp('2024-01-21'), 1.8, 'point_outlier', 1, False),
            ],
            'c': [
                (pd.Timestamp('2024-01-30'), -3.2, 'point_outlier', 1, False),
            ],
        }
        detector.trend_changepoints = {'a': [], 'b': [], 'c': []}
        detector.level_shifts = {'a': [], 'b': [], 'c': []}

        event_dag = detector._rebuild_event_dag()

        self.assertEqual(len(event_dag['member_events']), 5)
        self.assertEqual(len(event_dag['event_clusters']), 3)
        self.assertEqual(detector.anomalies['a'][0][0], pd.Timestamp('2024-01-10'))
        self.assertEqual(len(detector.anomalies['a']), 2)

        first_cluster = event_dag['event_clusters'][0]
        self.assertEqual(first_cluster['series_count'], 2)
        self.assertTrue(first_cluster['is_shared_root_cause_candidate'])
        self.assertEqual(len(first_cluster['member_ids']), 2)
        self.assertEqual(set(first_cluster['affected_series']), {'a', 'b'})

    def test_event_dag_repeated_clusters_form_families(self):
        detector = self._make_manual_event_dag_detector()
        detector.anomalies = {
            'a': [
                (pd.Timestamp('2024-01-10'), 3.0, 'point_outlier', 1, False),
                (pd.Timestamp('2024-01-20'), 2.5, 'point_outlier', 1, False),
            ],
            'b': [
                (pd.Timestamp('2024-01-11'), 2.0, 'point_outlier', 1, False),
                (pd.Timestamp('2024-01-21'), 1.8, 'point_outlier', 1, False),
            ],
            'c': [
                (pd.Timestamp('2024-01-30'), -3.2, 'point_outlier', 1, False),
            ],
        }
        detector.trend_changepoints = {'a': [], 'b': [], 'c': []}
        detector.level_shifts = {'a': [], 'b': [], 'c': []}

        event_dag = detector._rebuild_event_dag()

        self.assertEqual(len(event_dag['event_families']), 1)
        family = event_dag['event_families'][0]
        self.assertEqual(family['occurrence_count'], 2)
        self.assertEqual(len(family['cluster_ids']), 2)
        self.assertEqual(
            event_dag['event_clusters'][0]['family_id'],
            event_dag['event_clusters'][1]['family_id'],
        )
        self.assertIsNone(event_dag['event_clusters'][2]['family_id'])

    def test_event_dag_is_exposed_in_template_features_query_and_plot(self):
        detector = self._make_manual_event_dag_detector()
        detector.anomalies = {
            'a': [(pd.Timestamp('2024-01-10'), 3.0, 'point_outlier', 1, False)],
            'b': [(pd.Timestamp('2024-01-11'), 2.0, 'point_outlier', 1, False)],
            'c': [],
        }
        detector.trend_changepoints = {'a': [], 'b': [], 'c': []}
        detector.level_shifts = {'a': [], 'b': [], 'c': []}
        detector._rebuild_event_dag()

        all_features = detector.get_detected_features()
        self.assertIn('event_dag', all_features)
        self.assertEqual(detector.template['version'], '1.2')
        self.assertIn('event_dag', detector.get_template())

        series_features = detector.get_detected_features('a')
        self.assertEqual(len(series_features['event_dag']['event_clusters']), 1)

        queried = detector.query_features(
            dates=slice('2024-01-09', '2024-01-12'),
            series=['a', 'b'],
            include_event_dag=True,
        )
        self.assertIn('event_dag', queried)
        self.assertEqual(len(queried['event_dag']['event_clusters']), 1)
        self.assertEqual(queried['event_dag']['member_events'], [])

        queried_members = detector.query_features(
            dates=slice('2024-01-09', '2024-01-12'),
            series=['a', 'b'],
            include_event_dag=True,
            include_event_members=True,
        )
        self.assertEqual(len(queried_members['event_dag']['member_events']), 2)

        fig = detector.plot_event_dag(show=False)
        self.assertIsNotNone(fig)

    def test_event_dag_univariate_uses_broadcast_mode_without_families(self):
        detector = self._make_manual_event_dag_detector(detection_mode='univariate')
        shared_records = [
            (pd.Timestamp('2024-01-10'), 2.0, 'point_outlier', 1, True),
            (pd.Timestamp('2024-01-20'), 2.0, 'point_outlier', 1, True),
        ]
        detector.anomalies = {'a': shared_records, 'b': list(shared_records), 'c': list(shared_records)}
        detector.trend_changepoints = {'a': [], 'b': [], 'c': []}
        detector.level_shifts = {'a': [], 'b': [], 'c': []}

        event_dag = detector._rebuild_event_dag()

        self.assertEqual(event_dag['meta']['construction_mode'], 'broadcast')
        self.assertEqual(len(event_dag['event_families']), 0)
        self.assertTrue(all(x['series_name'] == '__broadcast__' for x in event_dag['member_events']))
        self.assertTrue(all(x['shared_flag'] for x in event_dag['member_events']))

    def test_event_dag_empty_path_returns_valid_empty_structure(self):
        detector = self._make_manual_event_dag_detector()
        detector.anomalies = {'a': [], 'b': [], 'c': []}
        detector.trend_changepoints = {'a': [], 'b': [], 'c': []}
        detector.level_shifts = {'a': [], 'b': [], 'c': []}

        event_dag = detector._rebuild_event_dag()

        self.assertIsInstance(event_dag, dict)
        self.assertEqual(event_dag['member_events'], [])
        self.assertEqual(event_dag['event_clusters'], [])
        self.assertEqual(event_dag['event_families'], [])
        self.assertEqual(event_dag['edges'], [])

    def test_holiday_detector_coverage_cap(self):
        test_gen = SyntheticDailyGenerator(
            start_date='2020-01-01',
            n_days=365 * 3,
            n_series=3,
            random_seed=42,
            noise_level=0.02,
        )
        test_data = test_gen.get_data()
        detector = HolidayDetector(
            threshold=0.6,
            splash_threshold=0.45,
            min_occurrences=2,
            use_dayofmonth_holidays=True,
            use_wkdom_holidays=True,
            use_wkdeom_holidays=False,
            use_lunar_holidays=False,
            use_lunar_weekday=False,
            use_islamic_holidays=False,
            use_hebrew_holidays=False,
            use_hindu_holidays=False,
            auto_relax=True,
            relax_rounds=2,
            max_holidays_per_series=5,
            holiday_selection_strategy='coverage',
            anomaly_detector_params={
                'method': 'rolling_zscore',
                'method_params': {
                    'distribution': 'norm',
                    'alpha': 0.05,
                    'rolling_periods': 90,
                    'center': False,
                },
            },
            output='multivariate',
        )
        detector.detect(test_data)

        counts = pd.Series(0, index=test_data.columns, dtype=float)
        for table in [detector.day_holidays, detector.wkdom_holidays]:
            if table is None or table.empty:
                continue
            counts = counts.add(table.groupby('series').size(), fill_value=0)
        counts = counts.reindex(test_data.columns, fill_value=0).astype(int)
        self.assertTrue((counts <= 5).all())

        stats = detector.get_detection_stats()
        self.assertIn('series_holiday_counts', stats)
        self.assertEqual(len(stats['series_holiday_counts']), test_data.shape[1])

        flags = detector.dates_to_holidays(
            test_data.index, style='series_flag', max_features=10
        )
        self.assertEqual(flags.shape, test_data.shape)

    def test_holiday_detector_strategy_fallback(self):
        detector = HolidayDetector(
            max_holidays_per_series=3, holiday_selection_strategy='unknown'
        )
        self.assertEqual(detector.holiday_selection_strategy, 'score')

    def test_noise_regime_detection_helper(self):
        detector = TimeSeriesFeatureDetector()
        detector.date_index = pd.date_range('2020-01-01', periods=180, freq='D')
        rng = np.random.RandomState(42)
        noise = np.concatenate(
            [rng.normal(0, 0.1, 90), rng.normal(0, 0.8, 90)]
        )
        noise_series = pd.Series(noise, index=detector.date_index)
        cps = detector._detect_noise_regime_changepoints(noise_series)
        self.assertGreaterEqual(len(cps), 1)
        self.assertTrue(
            any(abs((cp - detector.date_index[90]).days) <= 25 for cp in cps),
            f"Expected a changepoint near the regime boundary, got {cps}",
        )

    def test_summary(self):
        """Test summary generation."""
        detector = TimeSeriesFeatureDetector()
        detector.fit(self.data)

        # Should not raise error
        try:
            detector.summary()
            success = True
        except Exception as e:
            print(f"Summary failed: {e}")
            success = False

        self.assertTrue(success)

    def test_plot(self):
        """Test plotting (without showing)."""
        detector = TimeSeriesFeatureDetector()
        detector.fit(self.data)

        # Should not raise error
        try:
            detector.plot(show=False)
            success = True
        except Exception as e:
            print(f"Plot failed: {e}")
            success = False

        self.assertTrue(success)

    def test_plot_top_panel_reconstruction_uses_single_axis(self):
        """Top panel should keep series and reconstruction on the same y-axis."""
        detector = TimeSeriesFeatureDetector()
        detector.fit(self.data)
        fig = detector.plot(show=False)

        top_axis = fig.axes[0]
        top_bounds = top_axis.get_position().bounds

        # Ensure no twinx axis exists for the top panel.
        top_panel_axes = [
            axis
            for axis in fig.axes
            if np.allclose(axis.get_position().bounds, top_bounds, atol=1e-6)
        ]
        self.assertEqual(
            len(top_panel_axes),
            1,
            "Top panel should not have a secondary y-axis.",
        )

        line_labels = [line.get_label() for line in top_axis.get_lines()]
        self.assertIn('Series', line_labels)
        self.assertIn('Reconstructed (No Residual)', line_labels)

    def test_synthetic_plot_top_panel_hides_reconstruction(self):
        """Synthetic generator plot should not show reconstructed overlay."""
        synthetic = SyntheticDailyGenerator(
            start_date='2020-01-01',
            n_days=180,
            n_series=2,
            random_seed=42,
        )
        fig = synthetic.plot(show=False)
        top_axis = fig.axes[0]
        line_labels = [line.get_label() for line in top_axis.get_lines()]
        self.assertIn('Series', line_labels)
        self.assertNotIn('Reconstructed (No Residual)', line_labels)

    def test_tune_with_synthetic_applies_and_fits(self):
        """Test synthetic tuning workflow applies params and returns fitted self."""

        tuned_synth_params = {
            'trend_changepoint_freq': 1.1311196948895457,
            'level_shift_freq': 0.11235106798095876,
            'level_shift_strength': 0.10391796521529073,
            'anomaly_freq': 0.04654080084797309,
            'weekly_seasonality_strength': 0.8646796590839558,
            'yearly_seasonality_strength': 1.2232927741109423,
            'noise_level': 0.0016485471075049773,
            'trend_slope_scale': 0.4964492705041623,
            'trend_positive_bias': 0.7313495354794045,
            'level_shift_minimum_pct': 0.03,
            'level_shift_max_pct': 0.09,
            'noise_ar_coefficient': 0.07932349693757101,
            'volatility_regime_intensity': 1.3960299844487944,
        }

        def _fake_tune_to_data(self, df, n_iterations=15, verbose=True, starting_params=None):
            return {
                'best_params': tuned_synth_params,
                'scale_multiplier': 1.0,
                'target_stats': {'weekly_profile': [1.0] * 7},
            }

        def _fake_optimize(self, starting_params=None):
            best = self._default_detector_params()
            best['standardize'] = False
            best['smoothing_window'] = 5
            self.best_params = best
            self.best_loss = 1.5
            self.best_total_loss = 1.5
            self.baseline_loss = 2.0
            self.optimization_history = [
                {'iteration': 'baseline', 'loss': 2.0, 'balanced_loss': 2.0}
            ]
            return best

        real_df = self.data.iloc[:120, :2]
        detector = TimeSeriesFeatureDetector()
        with patch(
            'autots.evaluator.feature_detector.SyntheticDailyGenerator.tune_to_data',
            new=_fake_tune_to_data,
        ), patch(
            'autots.evaluator.feature_detector.FeatureDetectionOptimizer.optimize',
            new=_fake_optimize,
        ):
            result = detector.tune_with_synthetic(
                real_df=real_df,
                n_synthetic_series=2,
                n_tune_iterations=1,
                n_detector_iterations=1,
                tune_seed=42,
                verbose=False,
            )

        self.assertIs(result, detector)
        self.assertIsNotNone(detector.df_original)
        self.assertIsNotNone(detector.optimized_detector_params)
        self.assertIsNotNone(detector.synthetic_tuning_results)
        self.assertIsNotNone(detector.tuned_synthetic_generator)
        self.assertIsNotNone(detector.detector_optimization_summary)
        self.assertFalse(detector.standardize)
        self.assertEqual(detector.smoothing_window, 5)

    def test_analyze_noise_uses_anomaly_transform(self):
        detector = TimeSeriesFeatureDetector()
        index = pd.date_range('2023-01-01', periods=5, freq='D')
        cols = ['series_0']
        df_work = pd.DataFrame({'series_0': [1.0, 5.0, 1.0, 1.0, 1.0]}, index=index)
        cleaned = pd.DataFrame({'series_0': [1.0, 1.0, 1.0, 1.0, 1.0]}, index=index)

        class _DummyAnomalyDetector:
            def __init__(self, cleaned_df):
                self.cleaned_df = cleaned_df

            def transform(self, df):
                return self.cleaned_df.copy()

        detector.anomaly_detector = _DummyAnomalyDetector(cleaned)
        detector._anomaly_records_temp = {}

        zeros = pd.DataFrame(0.0, index=index, columns=cols)
        noise_component, anomaly_component = detector._analyze_noise(
            df_work,
            zeros,
            zeros,
            zeros,
            zeros,
        )

        self.assertAlmostEqual(float(anomaly_component.iloc[1, 0]), 4.0, places=6)
        self.assertAlmostEqual(float(noise_component.iloc[1, 0]), 1.0, places=6)


class TestFeatureDetectionLoss(unittest.TestCase):
    """Test FeatureDetectionLoss class."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic data once for all tests."""
        cls.generator = SyntheticDailyGenerator(
            start_date='2020-01-01',
            n_days=365,
            n_series=2,
            random_seed=42,
        )
        cls.data = cls.generator.get_data()
        cls.labels = cls.generator.get_all_labels()
        cls.components = cls.generator.get_components()
        cls.date_index = cls.generator.date_index

    def test_loss_initialization(self):
        """Test loss calculator initialization."""
        loss_calc = FeatureDetectionLoss()
        self.assertIsNotNone(loss_calc)
        self.assertEqual(loss_calc.changepoint_tolerance_days, 7)

    def test_calculate_loss(self):
        """Test loss calculation."""
        # Detect features
        detector = TimeSeriesFeatureDetector()
        detector.fit(self.data)
        detected = detector.get_detected_features(include_components=True)

        # Calculate loss
        loss_calc = FeatureDetectionLoss()
        loss = loss_calc.calculate_loss(
            detected,
            self.labels,
            true_components=self.components,
            date_index=self.date_index,
        )

        # Check loss structure
        self.assertIn('total_loss', loss)
        self.assertIn('trend_loss', loss)
        self.assertIn('anomaly_loss', loss)
        self.assertIsInstance(loss['total_loss'], (int, float))
        self.assertGreaterEqual(loss['total_loss'], 0)

    def test_series_specific_loss(self):
        """Test loss calculation for specific series."""
        detector = TimeSeriesFeatureDetector()
        detector.fit(self.data)
        detected = detector.get_detected_features(include_components=True)

        loss_calc = FeatureDetectionLoss()
        series_name = self.data.columns[0]
        loss = loss_calc.calculate_loss(
            detected,
            self.labels,
            series_name=series_name,
            true_components=self.components,
            date_index=self.date_index,
        )

        self.assertIn('total_loss', loss)
        self.assertIsInstance(loss['total_loss'], (int, float))

    def test_seasonality_strength_fuzzy_period_match(self):
        loss_calc = FeatureDetectionLoss()
        exact = loss_calc._seasonality_strength_loss(
            {'period_365': 1.0}, {'period_365': 1.0}
        )
        fuzzy = loss_calc._seasonality_strength_loss(
            {'period_366': 1.0}, {'period_365': 1.0}
        )
        self.assertEqual(exact, 0.0)
        self.assertLess(fuzzy, 0.01)

    def test_holiday_splash_loss_uses_splash_impacts(self):
        loss_calc = FeatureDetectionLoss()
        splash_date = pd.Timestamp('2020-01-15')
        detected_features = {
            'trend_changepoints': {'series_0': []},
            'holiday_impacts': {'series_0': {}},
            'holiday_splash_impacts': {'series_0': {splash_date: 2.0}},
            'anomalies': {'series_0': []},
        }
        true_labels = {
            'trend_changepoints': {'series_0': []},
            'holiday_splash_impacts': {'series_0': {splash_date: 2.0}},
        }
        loss = loss_calc.calculate_loss(
            detected_features=detected_features,
            true_labels=true_labels,
            date_index=pd.date_range('2020-01-01', periods=30, freq='D'),
        )
        self.assertEqual(loss['holiday_splash_loss'], 0.0)

    def test_dynamic_loss_activation_disables_empty_components(self):
        detector = TimeSeriesFeatureDetector()
        detector.fit(self.data)
        detected = detector.get_detected_features(include_components=True)

        loss_calc = FeatureDetectionLoss()
        loss = loss_calc.calculate_loss(
            detected,
            self.labels,
            true_components=self.components,
            date_index=self.date_index,
        )

        self.assertIn('regressor_loss', loss.get('disabled_components', []))
        self.assertEqual(
            loss.get('effective_weights', {}).get('regressor_loss'),
            0.0,
        )

    def test_regressor_loss_supports_nested_schema(self):
        loss_calc = FeatureDetectionLoss()
        date = pd.Timestamp('2022-05-10')
        payload = {
            'by_date': {date: {'promotion': 2.0, 'temperature': -0.4}},
            'coefficients': {'promotion': 1.1, 'temperature': -0.2},
        }
        perfect = loss_calc._regressor_loss(payload, payload)
        self.assertEqual(perfect, 0.0)

        partial = {
            'by_date': {date: {'promotion': 2.0}},
            'coefficients': {'promotion': 1.1},
        }
        self.assertGreater(loss_calc._regressor_loss(partial, payload), 0.0)

    def test_anomaly_loss_handles_nan_magnitudes(self):
        loss_calc = FeatureDetectionLoss()
        detected_anom = [
            {'date': pd.Timestamp('2020-01-10'), 'magnitude': np.nan, 'type': 'point_outlier'}
        ]
        true_anom = [
            {'date': pd.Timestamp('2020-01-10'), 'magnitude': 2.0, 'type': 'point_outlier'}
        ]
        loss = loss_calc._anomaly_loss(detected_anom, true_anom)
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_anomaly_loss_penalizes_one_detection_for_two_true_events(self):
        loss_calc = FeatureDetectionLoss(anomaly_tolerance_days=1)
        true_anom = [
            {
                'date': pd.Timestamp('2020-01-10'),
                'magnitude': 2.0,
                'type': 'point_outlier',
            },
            {
                'date': pd.Timestamp('2020-01-11'),
                'magnitude': 2.0,
                'type': 'point_outlier',
            },
        ]
        one_detected = [
            {
                'date': pd.Timestamp('2020-01-10'),
                'magnitude': 2.0,
                'type': 'point_outlier',
            }
        ]
        two_detected = [
            {
                'date': pd.Timestamp('2020-01-10'),
                'magnitude': 2.0,
                'type': 'point_outlier',
            },
            {
                'date': pd.Timestamp('2020-01-11'),
                'magnitude': 2.0,
                'type': 'point_outlier',
            },
        ]

        under_detect_loss = loss_calc._anomaly_loss(one_detected, true_anom)
        full_detect_loss = loss_calc._anomaly_loss(two_detected, true_anom)
        self.assertGreater(under_detect_loss, full_detect_loss + 0.5)


class TestFeatureDetectionOptimizer(unittest.TestCase):
    """Test FeatureDetectionOptimizer class."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic data once for all tests."""
        cls.generator = SyntheticDailyGenerator(
            start_date='2020-01-01',
            n_days=365,
            n_series=2,
            random_seed=42,
            trend_changepoint_freq=1.0,
            level_shift_freq=0.2,
            anomaly_freq=0.1,
        )

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            n_iterations=3,
        )
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.n_iterations, 3)

    def test_random_search(self):
        """Test random search optimization."""
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            n_iterations=3,
        )

        best_params = optimizer.optimize()

        self.assertIsNotNone(best_params)
        self.assertIn('seasonality_params', best_params)
        self.assertIsNotNone(optimizer.best_loss)

        # History includes baseline + successful iterations (failed ones are excluded)
        # With n_iterations=3, we expect baseline (1) + up to 3 regular iterations = up to 4 entries
        history_len = len(optimizer.optimization_history)
        self.assertGreater(
            history_len, 0, "Optimization history should contain at least the baseline"
        )
        self.assertLessEqual(
            history_len,
            4,
            f"With n_iterations=3, expected at most 4 entries (1 baseline + 3 iterations), "
            f"but got {history_len}. History may include duplicate parameter configurations.",
        )

    def test_grid_search(self):
        """Test grid search optimization."""
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            n_iterations=10,
        )

        best_params = optimizer.optimize()

        self.assertIsNotNone(best_params)
        self.assertIn('anomaly_params', best_params)
        # History only includes successful iterations (failed ones are excluded)
        self.assertGreater(len(optimizer.optimization_history), 0)

    def test_optimization_summary(self):
        """Test optimization summary."""
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            n_iterations=2,
        )
        optimizer.optimize()

        # Should not raise error
        try:
            summary = optimizer.get_optimization_summary()
            success = True
        except Exception as e:
            print(f"Summary failed: {e}")
            success = False

        self.assertTrue(success)
        self.assertIn('component_ranges', summary)
        self.assertIsInstance(summary['component_ranges'], dict)

    def test_optimizer_starting_params_seed(self):
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            n_iterations=1,
        )
        seed_params = optimizer._default_detector_params()
        seed_params['standardize'] = not bool(seed_params['standardize'])

        optimizer.optimize(starting_params=seed_params)
        iterations = [entry.get('iteration') for entry in optimizer.optimization_history]
        self.assertIn('starting', iterations)

    def test_starting_params_are_evaluated_without_rewrite(self):
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            n_iterations=0,
        )
        starting_params = optimizer._default_detector_params()
        starting_params['holiday_params']['use_lunar_holidays'] = True
        starting_params['changepoint_params']['probabilistic_output'] = True
        starting_params['anomaly_params']['method'] = 'IsolationForest'

        fake_loss = {'total_loss': 1.0}
        with patch.object(optimizer, '_evaluate_params', return_value=fake_loss):
            optimizer.optimize(starting_params=starting_params)

        starting_entry = next(
            entry
            for entry in optimizer.optimization_history
            if entry.get('iteration') == 'starting'
        )
        self.assertEqual(starting_entry['params'], starting_params)

    def test_param_signature_handles_numpy_and_ordering(self):
        params_a = {
            'alpha': np.float64(0.2),
            'arr': np.array([1, 2, 3]),
            'flags': {'b', 'a'},
            'when': pd.Timestamp('2020-01-01'),
        }
        params_b = {
            'when': pd.Timestamp('2020-01-01'),
            'flags': {'a', 'b'},
            'arr': np.array([1, 2, 3]),
            'alpha': np.float64(0.2),
        }
        sig_a = FeatureDetectionOptimizer._param_signature(params_a)
        sig_b = FeatureDetectionOptimizer._param_signature(params_b)
        self.assertEqual(sig_a, sig_b)

    def test_mutate_params_numeric_perturbation(self):
        """Test recursive parameter mutation with numerical perturbation."""
        optimizer = FeatureDetectionOptimizer(self.generator)
        detector = TimeSeriesFeatureDetector()

        # Use realistic params from the detector so keys overlap with fresh samples
        params = detector.get_new_params(method='random')
        rng = random.Random(42)

        # Test basic mutation (replaces a top-level key with a fresh random value)
        mutated = optimizer._mutate_params(params, detector, rng)
        self.assertNotEqual(params, mutated)

        # Verify it still has the same top-level structure (keys should be preserved)
        self.assertEqual(set(params.keys()), set(mutated.keys()))

    def test_select_best_uses_recovery_lexicographic_order_by_default(self):
        """Default selector should prefer recovery-floor compliance before raw loss."""
        optimizer = FeatureDetectionOptimizer(self.generator)
        history = [
            {
                'iteration': 0,
                'params': {'p': 1},
                'loss': 1.0,
                'loss_breakdown': {
                    'total_loss': 1.0,
                    'trend_loss': 0.3,
                    'anomaly_loss': 1.9,
                    'recovery_floor_violations': 6,
                    'reconstruction_total_loss': 0.9,
                },
            },
            {
                'iteration': 1,
                'params': {'p': 2},
                'loss': 1.1,
                'loss_breakdown': {
                    'total_loss': 1.1,
                    'trend_loss': 0.2,
                    'anomaly_loss': 0.6,
                    'recovery_floor_violations': 0,
                    'reconstruction_total_loss': 0.4,
                },
            },
        ]
        optimizer.optimization_history = history

        best_params = optimizer._select_best_from_history()

        self.assertEqual(best_params, {'p': 2})
        self.assertEqual(len(optimizer.history_df), 2)
        self.assertIn('balanced_loss', optimizer.history_df.columns)
        self.assertEqual(optimizer.best_total_loss, 1.1)

    def test_select_best_can_use_recovery_lexicographic_order(self):
        """Alternate selector should prefer fewer recovery-floor misses."""
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            selection_strategy='recovery_lexicographic',
        )
        optimizer.optimization_history = [
            {
                'iteration': 0,
                'params': {'p': 1},
                'loss': 1.0,
                'loss_breakdown': {
                    'total_loss': 1.0,
                    'recovery_floor_violations': 5,
                    'reconstruction_total_loss': 1.0,
                },
            },
            {
                'iteration': 1,
                'params': {'p': 2},
                'loss': 1.1,
                'loss_breakdown': {
                    'total_loss': 1.1,
                    'recovery_floor_violations': 0,
                    'reconstruction_total_loss': 0.7,
                },
            },
        ]

        best_params = optimizer._select_best_from_history()
        self.assertEqual(best_params, {'p': 2})

    def test_optimizer_records_recovery_metrics(self):
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            n_iterations=1,
        )
        optimizer.optimize()
        self.assertGreater(len(optimizer.optimization_history), 0)
        breakdown = optimizer.optimization_history[0].get('loss_breakdown') or {}
        metrics = breakdown.get('recovery_metrics') or {}
        self.assertIn('median_weekly_rel_error', metrics)
        self.assertIn('holiday_recall', metrics)
        self.assertIn('trend_f2', metrics)

    def test_running_best_params_survive_final_selection_failure(self):
        """Regression: successful evaluations should populate running best state."""
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            n_iterations=1,
        )

        fake_loss = {'total_loss': 1.5}
        with patch.object(optimizer, '_evaluate_params', return_value=fake_loss):
            with patch.object(
                optimizer,
                '_select_best_from_history',
                side_effect=RuntimeError('selection failed'),
            ):
                with self.assertRaises(RuntimeError):
                    optimizer.optimize()

        self.assertGreaterEqual(len(optimizer.optimization_history), 1)
        self.assertIsNotNone(optimizer.best_params)
        self.assertEqual(optimizer.best_loss, 1.5)
        self.assertEqual(optimizer.best_total_loss, 1.5)

    def test_record_evaluation_updates_best_total_loss(self):
        """Regression: running best bookkeeping should keep total loss aligned."""
        optimizer = FeatureDetectionOptimizer(self.generator, n_iterations=0)
        evaluated_signatures = set()
        first_params = {'candidate': 1}
        second_params = {'candidate': 2}

        with patch.object(
            optimizer,
            '_evaluate_params',
            side_effect=[{'total_loss': 2.0}, {'total_loss': 1.0}],
        ):
            optimizer._record_evaluation('first', first_params, evaluated_signatures)
            optimizer._record_evaluation('second', second_params, evaluated_signatures)

        self.assertEqual(optimizer.best_params, second_params)
        self.assertEqual(optimizer.best_loss, 1.0)
        self.assertEqual(optimizer.best_total_loss, 1.0)

    def test_example_benchmark_selected_matches_recovery_lexicographic_rank(self):
        """Regression check for the example benchmark selection path."""
        example_path = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_tuning_example.py'
        src = example_path.read_text()
        mod = ast.parse(src)
        values = {}
        for node in mod.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {'known_good_bp', 'best_20k'}:
                        values[target.id] = ast.literal_eval(node.value)

        generator = SyntheticDailyGenerator(
            start_date='2020-01-01',
            n_days=365 * 3,
            n_series=5,
            random_seed=42,
            **values['known_good_bp'],
        )
        optimizer = FeatureDetectionOptimizer(
            generator,
            n_iterations=3,
            starting_params=values['best_20k'],
            random_seed=42,
        )
        best = optimizer.optimize()
        self.assertIsNotNone(best)
        ranked_best = min(
            optimizer.optimization_history,
            key=lambda entry: (
                entry.get('loss_breakdown', {}).get('recovery_floor_violations', float('inf')),
                entry['loss'],
                entry.get('loss_breakdown', {}).get('reconstruction_total_loss', float('inf')),
            ),
        )
        self.assertEqual(best, ranked_best['params'])
        metrics = ranked_best.get('loss_breakdown', {}).get('recovery_metrics', {})
        self.assertIn('weekly_profile_correlation', metrics)
        self.assertIn('yearly_profile_correlation', metrics)

    def test_changepoint_distance_penalty_prefers_nearby_matches(self):
        optimizer = FeatureDetectionOptimizer(self.generator)
        true_entries = [
            (pd.Timestamp('2020-01-10'), 0.0, 1.0, 1.0),
            (pd.Timestamp('2020-02-10'), 1.0, -0.5, 1.5),
        ]
        near_entries = [
            (pd.Timestamp('2020-01-12'), 0.0, 1.0, 1.0),
            (pd.Timestamp('2020-02-12'), 1.0, -0.5, 1.5),
        ]
        far_entries = [
            (pd.Timestamp('2021-01-10'), 0.0, 1.0, 1.0),
            (pd.Timestamp('2021-02-10'), 1.0, -0.5, 1.5),
        ]

        near_penalty = optimizer._bounded_distance_penalty(
            near_entries, true_entries, sigma=7.0
        )
        far_penalty = optimizer._bounded_distance_penalty(
            far_entries, true_entries, sigma=7.0
        )

        self.assertLess(near_penalty, far_penalty)
        self.assertGreaterEqual(far_penalty, 0.95)

    def test_legacy_optimize_trend_loss_prefers_nearby_matches(self):
        optimizer = FeatureDetectionOptimizer(self.generator)
        true_cp = [(pd.Timestamp('2020-01-10'), 0.0, 1.0)]
        near_detected = [(pd.Timestamp('2020-01-11'), 0.0, 1.0)]
        far_detected = [(pd.Timestamp('2020-06-11'), 0.0, 1.0)]
        components = {'trend': np.array([0.0, 1.0, 2.0, 3.0])}

        near_loss = optimizer._legacy_optimize_trend_loss(
            near_detected,
            true_cp,
            components,
            components,
        )
        far_loss = optimizer._legacy_optimize_trend_loss(
            far_detected,
            true_cp,
            components,
            components,
        )

        self.assertLess(near_loss, far_loss)

    def test_apply_legacy_changepoint_loss_updates_total_loss(self):
        optimizer = FeatureDetectionOptimizer(self.generator)
        ts = pd.Timestamp('2020-01-10')
        detected_features = {
            'trend_changepoints': {'series_0': [(ts, 0.0, 1.0)]},
            'components': {'series_0': {'trend': np.array([0.0, 1.0, 2.0])}},
        }
        true_labels = {
            'trend_changepoints': {'series_0': [(ts, 0.0, 1.0)]},
            'series_types': {'series_0': 'standard'},
        }
        true_components = {'series_0': {'trend': np.array([0.0, 1.0, 2.0])}}
        input_loss = {
            'trend_loss': 10.0,
            'total_loss': 20.0,
            'effective_weights': {'trend_loss': 2.0},
            'series_breakdown': {'series_0': {'trend_loss': 10.0}},
        }

        output_loss = optimizer._apply_legacy_changepoint_loss_for_optimize(
            loss=input_loss,
            detected_features=detected_features,
            true_labels=true_labels,
            true_components=true_components,
        )

        self.assertAlmostEqual(output_loss['trend_loss'], 0.0, places=7)
        self.assertAlmostEqual(output_loss['total_loss'], 0.0, places=7)
        self.assertAlmostEqual(
            output_loss['series_breakdown']['series_0']['trend_loss'],
            0.0,
            places=7,
        )

    def test_count_calibration_penalty_prefers_small_mismatch(self):
        optimizer = FeatureDetectionOptimizer(self.generator)
        small_mismatch = optimizer._count_calibration_penalty(4, 3)
        slight_under_detection = optimizer._count_calibration_penalty(2, 3)
        severe_over_detection = optimizer._count_calibration_penalty(30, 3)
        severe_under_detection = optimizer._count_calibration_penalty(0, 3)

        self.assertLess(small_mismatch, severe_over_detection)
        self.assertLess(small_mismatch, severe_under_detection)
        self.assertLess(
            small_mismatch,
            slight_under_detection,
            "A slight over-prediction should be cheaper than a slight under-prediction.",
        )
        self.assertGreater(severe_over_detection, severe_under_detection)

    def test_slope_alignment_penalty_prefers_matching_direction(self):
        optimizer = FeatureDetectionOptimizer(self.generator)
        true_entries = [(pd.Timestamp('2020-01-10'), 0.0, 1.0, 1.0)]
        matching_entries = [(pd.Timestamp('2020-01-11'), 0.0, 1.1, 1.1)]
        wrong_direction_entries = [(pd.Timestamp('2020-01-11'), 0.0, -1.0, 1.0)]

        matching_penalty = optimizer._slope_change_alignment_penalty(
            matching_entries, true_entries, sigma=7.0
        )
        wrong_direction_penalty = optimizer._slope_change_alignment_penalty(
            wrong_direction_entries, true_entries, sigma=7.0
        )

        self.assertLess(matching_penalty, wrong_direction_penalty)

    def test_cross_family_partial_credit_prefers_nearby_confusion(self):
        optimizer = FeatureDetectionOptimizer(self.generator)
        true_entries = [(pd.Timestamp('2020-01-10'), 0.0, 1.0, 1.0)]
        nearby_other_family = [(pd.Timestamp('2020-01-11'), 1.5)]
        far_other_family = [(pd.Timestamp('2020-03-15'), 1.5)]

        nearby_credit = optimizer._cross_family_partial_credit(
            nearby_other_family, true_entries, sigma=7.0
        )
        far_credit = optimizer._cross_family_partial_credit(
            far_other_family, true_entries, sigma=7.0
        )

        self.assertGreater(nearby_credit, far_credit)
        self.assertGreater(nearby_credit, 0.0)
        self.assertLessEqual(nearby_credit, 0.12)

    def test_local_mutate_changepoint_params_preserves_individual_mode(self):
        optimizer = FeatureDetectionOptimizer(self.generator)
        params = {
            'method': 'pelt',
            'method_params': {
                'penalty': 50,
                'loss_function': 'l2',
                'min_segment_length': 10,
                'pruning_factor': 1.5,
            },
            'aggregate_method': 'mean',
            'min_segment_length': 10,
            'probabilistic_output': True,
        }

        mutated = optimizer._local_mutate_changepoint_params(
            params, random.Random(42)
        )

        self.assertEqual(mutated['aggregate_method'], 'individual')
        self.assertFalse(mutated['probabilistic_output'])
        self.assertIn('method_params', mutated)

    def test_local_mutate_changepoint_params_ewma_path(self):
        optimizer = FeatureDetectionOptimizer(self.generator)
        params = {
            'method': 'ewma',
            'method_params': {
                'lambda_param': 0.2,
                'control_limit': 3.0,
                'min_distance': 10,
            },
            'aggregate_method': 'mean',
            'probabilistic_output': True,
        }

        class DeterministicRng:
            def random(self):
                return 0.2

            def choice(self, values):
                return values[0]

            def sample(self, values, sample_size):
                return list(values)[:sample_size]

        mutated = optimizer._local_mutate_changepoint_params(
            params, DeterministicRng()
        )

        self.assertEqual(mutated['aggregate_method'], 'individual')
        self.assertFalse(mutated['probabilistic_output'])
        self.assertIn('method_params', mutated)
        self.assertIsInstance(mutated['method_params']['min_distance'], int)

    def test_fine_tune_changepoints_runs_with_ewma_starting_params(self):
        optimizer = FeatureDetectionOptimizer(self.generator, random_seed=2)
        starting_params = optimizer._default_detector_params()
        starting_params['changepoint_params'] = {
            'method': 'ewma',
            'method_params': {
                'lambda_param': 0.2,
                'control_limit': 3.0,
                'min_distance': 10,
            },
            'aggregate_method': 'individual',
            'probabilistic_output': False,
        }

        with patch.object(
            optimizer,
            '_evaluate_changepoint_params',
            return_value=1.0,
        ), patch.object(
            optimizer,
            '_local_mutate_changepoint_params',
            wraps=optimizer._local_mutate_changepoint_params,
        ) as local_mutate:
            best = optimizer.fine_tune_changepoints(
                starting_params,
                n_per_stage=20,
                curriculum_sigmas=[7.0],
                exclude_changepoint_methods=[],
            )

        self.assertIsInstance(best, dict)
        self.assertIn('changepoint_params', best)
        self.assertGreaterEqual(local_mutate.call_count, 1)

class TestScaling(unittest.TestCase):
    """Test that scaling and unscaling work correctly."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic data with known scale once for all tests."""
        cls.generator = SyntheticDailyGenerator(
            start_date='2020-01-01',
            n_days=730,
            n_series=2,
            random_seed=42,
            trend_changepoint_freq=0.8,
            level_shift_freq=0.15,
            anomaly_freq=0.08,
            weekly_seasonality_strength=2.0,  # Strong seasonality
            yearly_seasonality_strength=1.0,
            noise_level=0.5,
        )
        cls.data = cls.generator.get_data()
        cls.labels = cls.generator.get_all_labels()
        cls.components = cls.generator.get_components()

        # Scale the data to test scaling/unscaling
        cls.data_scaled = cls.data * 100 + 1000  # Large scale and offset

    def test_standardize_true_impacts_are_unscaled(self):
        """Test that all impacts are properly unscaled when standardize=True."""
        detector = TimeSeriesFeatureDetector(standardize=True)
        detector.fit(self.data_scaled)

        series_name = self.data_scaled.columns[0]
        detected = detector.get_detected_features(series_name, include_components=True)

        # Get original data statistics
        original_mean = self.data_scaled[series_name].mean()
        original_std = self.data_scaled[series_name].std()

        # Test 1: Anomaly magnitudes should be in original scale
        if detected['anomalies']:
            for anomaly in detected['anomalies']:
                if isinstance(anomaly, dict):
                    magnitude = anomaly.get('magnitude', 0)
                elif isinstance(anomaly, (tuple, list)) and len(anomaly) >= 2:
                    magnitude = anomaly[1]
                else:
                    continue

                # Magnitude should be reasonable compared to original scale
                # Should not be in [-3, 3] range (which is standardized)
                # Should be in a range proportional to original std
                self.assertGreater(
                    abs(magnitude),
                    original_std * 0.01,
                    f"Anomaly magnitude {magnitude} appears to be in standardized scale, not original",
                )

        # Test 2: Level shift magnitudes should be in original scale
        if detected['level_shifts']:
            for shift in detected['level_shifts']:
                if isinstance(shift, dict):
                    magnitude = shift.get('magnitude', 0)
                elif isinstance(shift, (tuple, list)) and len(shift) >= 2:
                    magnitude = shift[1]
                else:
                    continue

                # Level shift should be reasonable in original scale
                self.assertGreater(
                    abs(magnitude),
                    original_std * 0.01,
                    f"Level shift magnitude {magnitude} appears to be in standardized scale",
                )

        # Test 4: Holiday coefficients should be in original scale
        if detected['holiday_coefficients']:
            for holiday_name, coef in detected['holiday_coefficients'].items():
                if isinstance(coef, (int, float)):
                    # Holiday coefficients should not be in standardized scale (typically -3 to +3)
                    # If original_std is large, even small absolute values indicate proper unscaling
                    # Check: if original_std > 100, coefficient should be > 5 (beyond typical standardized range)
                    # Otherwise, use the relative threshold
                    if original_std > 100:
                        self.assertGreater(
                            abs(coef),
                            5,
                            f"Holiday coefficient {coef} for {holiday_name} appears to be in standardized scale",
                        )
                    else:
                        self.assertGreater(
                            abs(coef),
                            original_std * 0.01,
                            f"Holiday coefficient {coef} for {holiday_name} appears to be in standardized scale",
                        )

        # Test 5: Trend slopes should be in original scale
        if detected['trend_changepoints']:
            for cp in detected['trend_changepoints']:
                if isinstance(cp, dict) and 'slope' in cp:
                    slope = cp['slope']
                    # Slope in original scale should be larger than standardized slope
                    # (though this is per-day, so may be small)
                    self.assertTrue(np.isfinite(slope), f"Slope {slope} is not finite")

        # Test 6: Components should be in original scale
        if 'components' in detected and detected['components']:
            components = detected['components']

            # Seasonality component
            if 'seasonality' in components:
                seasonality = components['seasonality']
                if isinstance(seasonality, dict):
                    # Values should be in original scale
                    values = [
                        v for v in seasonality.values() if isinstance(v, (int, float))
                    ]
                    if values:
                        max_seasonal = max(abs(v) for v in values)
                        # Seasonal component in original scale should be larger
                        self.assertGreater(
                            max_seasonal,
                            original_std * 0.01,
                            f"Seasonality component appears to be in standardized scale",
                        )

            # Holiday component
            if 'holidays' in components:
                holidays = components['holidays']
                if isinstance(holidays, dict):
                    values = [
                        v for v in holidays.values() if isinstance(v, (int, float))
                    ]
                    if values:
                        max_holiday = max(abs(v) for v in values)
                        self.assertGreater(
                            max_holiday,
                            original_std * 0.01,
                            f"Holiday component appears to be in standardized scale",
                        )

        # Test 7: Reconstruction should match original data scale
        if detector.reconstructed is not None:
            reconstructed_series = detector.reconstructed[series_name]

            # Mean should be close to original mean
            recon_mean = reconstructed_series.mean()
            self.assertAlmostEqual(
                recon_mean,
                original_mean,
                delta=original_std,
                msg=f"Reconstructed mean {recon_mean} doesn't match original mean {original_mean}",
            )

            # Std should be close to original std
            recon_std = reconstructed_series.std()
            self.assertAlmostEqual(
                recon_std,
                original_std,
                delta=original_std * 0.5,
                msg=f"Reconstructed std {recon_std} doesn't match original std {original_std}",
            )

            # Values should be in same range as original
            self.assertGreater(
                reconstructed_series.min(),
                self.data_scaled[series_name].min() - original_std * 3,
            )
            self.assertLess(
                reconstructed_series.max(),
                self.data_scaled[series_name].max() + original_std * 3,
            )

    def test_standardize_false_impacts_are_original(self):
        """Test that impacts are correct when standardize=False."""
        detector = TimeSeriesFeatureDetector(standardize=False)
        detector.fit(self.data_scaled)

        series_name = self.data_scaled.columns[0]
        detected = detector.get_detected_features(series_name, include_components=True)

        original_std = self.data_scaled[series_name].std()

        # All impacts should still be in original scale
        # (No scaling/unscaling needed)

        if detected['anomalies']:
            for anomaly in detected['anomalies']:
                if isinstance(anomaly, dict):
                    magnitude = anomaly.get('magnitude', 0)
                elif isinstance(anomaly, (tuple, list)) and len(anomaly) >= 2:
                    magnitude = anomaly[1]
                else:
                    continue
                self.assertGreater(abs(magnitude), original_std * 0.01)

        if detected['level_shifts']:
            for shift in detected['level_shifts']:
                if isinstance(shift, dict):
                    magnitude = shift.get('magnitude', 0)
                elif isinstance(shift, (tuple, list)) and len(shift) >= 2:
                    magnitude = shift[1]
                else:
                    continue
                self.assertGreater(abs(magnitude), original_std * 0.01)

    def test_scaling_consistency_across_modes(self):
        """Test that standardize=True and False produce comparable results."""
        detector_scaled = TimeSeriesFeatureDetector(standardize=True)
        detector_unscaled = TimeSeriesFeatureDetector(standardize=False)

        detector_scaled.fit(self.data_scaled)
        detector_unscaled.fit(self.data_scaled)

        series_name = self.data_scaled.columns[0]

        # Both should produce reconstructions in the same scale
        if (
            detector_scaled.reconstructed is not None
            and detector_unscaled.reconstructed is not None
        ):
            scaled_recon = detector_scaled.reconstructed[series_name]
            unscaled_recon = detector_unscaled.reconstructed[series_name]

            # Means should be similar
            self.assertAlmostEqual(
                scaled_recon.mean(),
                unscaled_recon.mean(),
                delta=self.data_scaled[series_name].std() * 0.5,
                msg="Scaled and unscaled reconstruction means differ significantly",
            )

    def test_template_values_in_original_scale(self):
        """Test that template contains values in original scale."""
        detector = TimeSeriesFeatureDetector(standardize=True)
        detector.fit(self.data_scaled)

        template = detector.get_template()
        series_name = self.data_scaled.columns[0]

        if series_name in template['series']:
            series_template = template['series'][series_name]
            original_std = self.data_scaled[series_name].std()

            # Check anomalies in template
            if 'anomalies' in series_template['labels']:
                for anomaly in series_template['labels']['anomalies']:
                    magnitude = anomaly.get('magnitude', 0)
                    self.assertGreater(
                        abs(magnitude),
                        original_std * 0.01,
                        "Template anomaly magnitude appears standardized",
                    )

            # Check level shifts in template
            if 'level_shifts' in series_template['labels']:
                for shift in series_template['labels']['level_shifts']:
                    magnitude = shift.get('magnitude', 0)
                    self.assertGreater(
                        abs(magnitude),
                        original_std * 0.01,
                        "Template level shift magnitude appears standardized",
                    )

            # Check holiday impacts in template
            if 'holidays' in series_template['labels']:
                for holiday in series_template['labels']['holidays']:
                    if 'direct_impact' in holiday:
                        impact = holiday['direct_impact']
                        self.assertGreater(
                            abs(impact),
                            original_std * 0.01,
                            "Template holiday impact appears standardized",
                        )

    def test_metadata_scale_attribute(self):
        """Test that series_scale metadata is correctly stored."""
        detector = TimeSeriesFeatureDetector(standardize=True)
        detector.fit(self.data_scaled)

        series_name = self.data_scaled.columns[0]
        detected = detector.get_detected_features(series_name, include_metadata=True)

        # Should have series_scale in metadata
        self.assertIn('series_scale', detected)

        # Scale should match actual data std
        actual_std = self.data_scaled[series_name].std()
        stored_scale = detected['series_scale']

        self.assertAlmostEqual(
            stored_scale,
            actual_std,
            delta=actual_std * 0.01,
            msg=f"Stored scale {stored_scale} doesn't match actual std {actual_std}",
        )

    def test_small_scale_data(self):
        """Test with data in small scale (e.g., [0, 1] range)."""
        small_scale_data = self.data / 100  # Scale down to small values

        detector = TimeSeriesFeatureDetector(standardize=True)
        detector.fit(small_scale_data)

        series_name = small_scale_data.columns[0]
        detected = detector.get_detected_features(series_name, include_components=True)

        original_std = small_scale_data[series_name].std()

        # Even with small scale, impacts should be in original scale
        if detected['anomalies']:
            for anomaly in detected['anomalies']:
                if isinstance(anomaly, dict):
                    magnitude = anomaly.get('magnitude', 0)
                    # Should be small like the original data
                    self.assertLess(
                        abs(magnitude),
                        original_std * 100,
                        "Anomaly magnitude too large for small-scale data",
                    )

    def test_component_reconstruction_scale(self):
        """Test that all reconstructed components are in original scale."""
        detector = TimeSeriesFeatureDetector(standardize=True)
        detector.fit(self.data_scaled)

        if detector.reconstructed_components is None:
            self.skipTest("No reconstructed components available")

        series_name = self.data_scaled.columns[0]
        original_mean = self.data_scaled[series_name].mean()
        original_std = self.data_scaled[series_name].std()

        # Check each component type
        for component_name in [
            'trend',
            'level_shift',
            'seasonality',
            'holidays',
            'noise',
            'anomalies',
        ]:
            if component_name in detector.reconstructed_components:
                component_df = detector.reconstructed_components[component_name]
                if series_name in component_df.columns:
                    component_series = component_df[series_name]

                    # Component values should be finite
                    self.assertTrue(
                        np.all(np.isfinite(component_series.dropna())),
                        f"{component_name} component has non-finite values",
                    )

                    # Component magnitude should be reasonable relative to original data
                    # (not in standardized [-3, 3] range)
                    component_std = component_series.std()
                    if component_name in [
                        'trend',
                        'level_shift',
                        'seasonality',
                        'holidays',
                    ]:
                        # These should have meaningful magnitudes
                        if (
                            component_std > 0.01
                        ):  # Only check if component is non-trivial
                            self.assertGreater(
                                component_std,
                                original_std * 0.001,
                                f"{component_name} component appears to be in wrong scale",
                            )

        # Sum of all components should equal original data (approximately)
        if all(
            comp in detector.reconstructed_components
            for comp in [
                'trend',
                'level_shift',
                'seasonality',
                'holidays',
                'noise',
                'anomalies',
            ]
        ):
            reconstructed_sum = sum(
                detector.reconstructed_components[comp][series_name]
                for comp in [
                    'trend',
                    'level_shift',
                    'seasonality',
                    'holidays',
                    'noise',
                    'anomalies',
                ]
            )

            # Mean should match
            self.assertAlmostEqual(
                reconstructed_sum.mean(),
                original_mean,
                delta=original_std * 0.5,
                msg="Sum of components doesn't match original data mean",
            )

    def test_seasonality_component_scale(self):
        """Test that seasonality component specifically is in original scale."""
        detector = TimeSeriesFeatureDetector(standardize=True)
        detector.fit(self.data_scaled)

        series_name = self.data_scaled.columns[0]
        detected = detector.get_detected_features(series_name, include_components=True)

        if 'components' not in detected or 'seasonality' not in detected['components']:
            self.skipTest("No seasonality component detected")

        seasonality_component = detected['components']['seasonality']
        original_std = self.data_scaled[series_name].std()

        if isinstance(seasonality_component, dict):
            values = [
                v
                for v in seasonality_component.values()
                if isinstance(v, (int, float)) and np.isfinite(v)
            ]
            if values:
                seasonal_std = np.std(values)
                # Seasonal component should be in original scale, not standardized
                # For strong seasonality (2.0), this should be meaningful
                self.assertGreater(
                    seasonal_std,
                    original_std * 0.01,
                    "Seasonality component appears to be in standardized scale",
                )

    def test_noise_level_in_original_scale(self):
        """Test that noise level metadata is a dimensionless ratio."""
        detector = TimeSeriesFeatureDetector(standardize=True)
        detector.fit(self.data_scaled)

        series_name = self.data_scaled.columns[0]
        detected = detector.get_detected_features(series_name, include_metadata=True)

        if 'series_noise_level' not in detected:
            self.skipTest("No noise level metadata")

        noise_level = detected['series_noise_level']

        # Noise level should be a ratio (dimensionless), typically in [0, 2] range
        # It's the ratio of noise std to original series std
        self.assertGreaterEqual(noise_level, 0, "Noise level should be non-negative")
        self.assertLess(noise_level, 10, "Noise level ratio seems unreasonably large")

        # For our synthetic data with noise_level=0.5, should be reasonable
        self.assertLess(
            noise_level, 2.0, "Noise level too high for configured parameters"
        )

    def test_multiple_series_scaling(self):
        """Test that scaling works correctly for multiple series with different scales."""
        # Create data where each series has different scale
        data_multi_scale = self.data.copy()
        data_multi_scale.iloc[:, 0] = data_multi_scale.iloc[:, 0] * 10 + 50
        data_multi_scale.iloc[:, 1] = data_multi_scale.iloc[:, 1] * 1000 + 5000

        detector = TimeSeriesFeatureDetector(standardize=True)
        detector.fit(data_multi_scale)

        # Check that each series has different scale metadata
        for series_name in data_multi_scale.columns:
            detected = detector.get_detected_features(
                series_name, include_metadata=True
            )
            original_std = data_multi_scale[series_name].std()

            self.assertIn('series_scale', detected)
            self.assertAlmostEqual(
                detected['series_scale'], original_std, delta=original_std * 0.01
            )

            # All impacts should be in that series' original scale
            if detected['anomalies']:
                for anomaly in detected['anomalies']:
                    if isinstance(anomaly, dict):
                        magnitude = anomaly.get('magnitude', 0)
                        # Should be reasonable for this series' scale
                        self.assertGreater(
                            abs(magnitude),
                            original_std * 0.01,
                            f"Anomaly in {series_name} appears standardized",
                        )


class TestReconstructionLoss(unittest.TestCase):
    """Tests for ReconstructionLoss on unlabeled data."""

    def setUp(self):
        periods = 90
        index = pd.date_range('2021-01-01', periods=periods, freq='D')
        trend = np.linspace(0, 3, periods)
        weekly = 1.2 * np.sin(2 * np.pi * index.dayofweek / 7)
        anomalies = np.zeros(periods)
        anomalies[[12, 45, 70]] = [3.5, -2.8, 2.1]
        level_shift = np.zeros(periods)
        level_shift[index >= index[50]] = 1.0
        noise = np.random.default_rng(123).normal(scale=0.2, size=periods)

        series = trend + weekly + anomalies + level_shift + noise
        self.df = pd.DataFrame({'series_1': series}, index=index)

        zeros = np.zeros(periods)
        self.components_balanced = {
            'series_1': {
                'trend': trend,
                'level_shift': level_shift,
                'seasonality': weekly,
                'holidays': zeros,
                'anomalies': anomalies,
                'noise': noise,
            }
        }
        self.components_overfit = {
            'series_1': {
                'trend': series,
                'level_shift': zeros,
                'seasonality': zeros,
                'holidays': zeros,
                'anomalies': zeros,
                'noise': zeros,
            }
        }

    def test_penalizes_trend_overfit(self):
        loss_calc = ReconstructionLoss(
            seasonality_lags=(7,),
            seasonality_improvement_target=0.2,
            anomaly_improvement_target=0.1,
            trend_min_other_variance=0.0,
        )

        balanced_loss = loss_calc.calculate_loss(
            observed_df=self.df,
            detected_features={'components': self.components_balanced},
        )
        overfit_loss = loss_calc.calculate_loss(
            observed_df=self.df,
            detected_features={'components': self.components_overfit},
        )

        self.assertLess(
            balanced_loss['total_loss'],
            overfit_loss['total_loss'],
            "Balanced decomposition should score lower total loss than overfit trend.",
        )
        self.assertGreater(
            overfit_loss['trend_smoothness_loss'],
            balanced_loss['trend_smoothness_loss'],
            "Overfit trend should incur higher smoothness penalty.",
        )
        self.assertGreater(
            overfit_loss['trend_dominance_loss'],
            balanced_loss['trend_dominance_loss'],
            "Overfit trend should have larger dominance penalty.",
        )

    def test_requires_components(self):
        loss_calc = ReconstructionLoss()
        with self.assertRaises(ValueError):
            loss_calc.calculate_loss(
                observed_df=self.df,
                detected_features={'trend_changepoints': []},
            )


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test complete detection and optimization pipeline."""
        # Create synthetic data
        generator = SyntheticDailyGenerator(
            start_date='2020-01-01',
            n_days=365,
            n_series=2,
            random_seed=123,
        )

        # Detect with default params
        detector = TimeSeriesFeatureDetector()
        detector.fit(generator.get_data())

        # Get features
        features = detector.get_detected_features(include_components=True)
        self.assertIsNotNone(features)

        # Calculate loss
        loss_calc = FeatureDetectionLoss()
        labels = generator.get_all_labels()
        components = generator.get_components()
        loss = loss_calc.calculate_loss(
            features,
            labels,
            true_components=components,
            date_index=generator.date_index,
        )

        print(f"\nDefault parameters loss: {loss['total_loss']:.4f}")
        self.assertGreater(loss['total_loss'], 0)

        # Optimize (just a few iterations for testing)
        optimizer = FeatureDetectionOptimizer(
            generator,
            n_iterations=3,
        )
        best_params = optimizer.optimize()

        print(f"Optimized loss: {optimizer.best_loss:.4f}")

        # Verify optimization improved or maintained performance
        self.assertLessEqual(
            optimizer.best_loss, loss['total_loss'] * 1.5
        )  # Allow some variance

    def test_comparison_with_labels(self):
        """Test detailed comparison between detected and true features."""
        generator = SyntheticDailyGenerator(
            start_date='2020-01-01',
            n_days=500,
            n_series=1,
            random_seed=456,
            trend_changepoint_freq=2.0,  # More changepoints
            anomaly_freq=0.15,  # More anomalies
        )

        detector = TimeSeriesFeatureDetector()
        detector.fit(generator.get_data())

        series_name = generator.get_data().columns[0]

        # Get true labels
        true_cp = generator.get_trend_changepoints(series_name)
        true_anom = generator.get_anomalies(series_name)

        # Get detected
        detected = detector.get_detected_features(series_name)

        print(f"\n--- Comparison for {series_name} ---")
        print(f"True changepoints: {len(true_cp)}")
        print(f"Detected changepoints: {len(detected['trend_changepoints'])}")
        print(f"True anomalies: {len(true_anom)}")
        print(f"Detected anomalies: {len(detected['anomalies'])}")

        # Should detect at least some features
        total_detected = (
            len(detected['trend_changepoints'])
            + len(detected['anomalies'])
            + len(detected['level_shifts'])
        )
        self.assertGreater(total_detected, 0)

    def test_tune_with_synthetic_on_load_daily(self):
        """Live test of tune_with_synthetic on real load_daily data with detailed reporting."""
        print("\n" + "=" * 80)
        print("LIVE TEST: tune_with_synthetic ON load_daily DATA")
        print("=" * 80)

        # Loading data and preparing wide format
        df = load_daily()
        df_wide = df.pivot(index='datetime', columns='series_id', values='value')
        # Use a manageable subset for testing but enough for meaningful tuning
        df_subset = df_wide.iloc[-365:, :5]

        detector = TimeSeriesFeatureDetector()

        start_time = time.time()
        # Using smaller iterations for the test to avoid taking too long
        detector.tune_with_synthetic(
            real_df=df_subset,
            n_synthetic_series=5,
            n_tune_iterations=5,
            n_detector_iterations=10,
            verbose=True,
        )
        end_time = time.time()

        # Gather results for reporting
        results = detector.synthetic_tuning_results
        optimizer = detector.detector_optimizer
        summary = detector.detector_optimization_summary

        print("\n" + "=" * 80)
        print("DETAILED TUNING REPORT")
        print("=" * 80)
        print(f"Total time: {end_time - start_time:.2f} seconds")

        print(f"\n[1] Synthetic Data Tuning Results:")
        print(f"    Scale Multiplier: {results.get('scale_multiplier'):.4f}")

        gen_params = results.get('tuning_results', {}).get('best_params', {})
        print(f"    Best Synthetic Generator Params:")
        for k, v in gen_params.items():
            if isinstance(v, (float, int)):
                print(f"      - {k}: {v:.4f}")
            else:
                print(f"      - {k}: {v}")

        print(f"\n[2] Detector Optimization Results:")
        print(f"    Baseline Loss: {summary.get('baseline_loss'):.6f}")
        print(f"    Best Loss: {summary.get('best_loss'):.6f}")
        if summary.get('baseline_loss'):
            improvement = (
                (summary.get('baseline_loss') - summary.get('best_loss'))
                / summary.get('baseline_loss')
                * 100
            )
            print(f"    Improvement: {improvement:.2f}%")

        print(f"\n[3] Best Detector Parameters Found:")
        # Compact JSON representation
        print(json.dumps(detector.optimized_detector_params, indent=4))

        print(f"\n[4] Loss Breakdown (Best Model):")
        # Find the best entry in history to show its breakdown
        best_entry = None
        target_loss = summary.get('best_loss')
        if optimizer and optimizer.optimization_history:
            for entry in optimizer.optimization_history:
                # Need to handle floating point comparison
                entry_loss = entry.get('balanced_loss', entry.get('loss', 999))
                if abs(entry_loss - target_loss) < 1e-9:
                    best_entry = entry
                    break

        if best_entry and 'loss_breakdown' in best_entry:
            breakdown = best_entry['loss_breakdown']
            for component, loss_val in breakdown.items():
                if isinstance(loss_val, (float, int)):
                    print(f"      - {component}: {loss_val:.6f}")
                else:
                    print(f"      - {component}: {loss_val}")

        print(f"\n[5] Component Sensitivity (Importance of each loss component):")
        comp_ranges = summary.get('component_ranges', {})
        # Sort by range to show what varied most during optimization
        for comp, stats in sorted(
            comp_ranges.items(), key=lambda x: x[1]['range'], reverse=True
        ):
            print(
                f"      - {comp}: range={stats['range']:.4f} (min={stats['min']:.4f}, max={stats['max']:.4f})"
            )

        print("\n" + "=" * 80)

        # Basic functional assertions
        self.assertIsNotNone(detector.optimized_detector_params)
        self.assertIn('best_params', results.get('tuning_results', {}))
        self.assertGreater(len(optimizer.optimization_history), 0)


class TestFeatureDetectorCalendarHolidayFusion(unittest.TestCase):
    """Focused tests for calendar-country holiday fusion."""

    @staticmethod
    def _detector_kwargs():
        return {
            'standardize': False,
            'seasonality_params': {
                'regression_model': {
                    'model': 'DecisionTree',
                    'model_params': {'max_depth': 4, 'min_samples_split': 2},
                },
                'datepart_method': 'simple_3',
                'polynomial_degree': None,
                'transform_dict': None,
                'holiday_countries_used': True,
                'lags': None,
                'forward_lags': None,
            },
            'rough_seasonality_params': {
                'regression_model': {
                    'model': 'DecisionTree',
                    'model_params': {'max_depth': 3, 'min_samples_split': 2},
                },
                'datepart_method': 'simple',
                'polynomial_degree': None,
                'transform_dict': None,
                'holiday_countries_used': True,
                'lags': None,
                'forward_lags': None,
            },
            'holiday_params': {
                'threshold': 0.99,
                'min_occurrences': 99,
                'splash_threshold': None,
                'use_dayofmonth_holidays': True,
                'use_wkdom_holidays': False,
                'use_wkdeom_holidays': False,
                'use_lunar_holidays': False,
                'use_lunar_weekday': False,
                'use_islamic_holidays': False,
                'use_hebrew_holidays': False,
                'use_hindu_holidays': False,
                'auto_relax': False,
            },
        }

    @staticmethod
    def _make_calendar_spike_df(end_date="2024-06-20"):
        dates = pd.date_range("2021-01-01", end_date, freq='D')
        weekly = np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 0.25
        us = weekly + 1.0
        ca = weekly + 1.0
        us_spike = (dates.month == 7) & (dates.day == 4)
        ca_spike = (dates.month == 7) & (dates.day == 1)
        us = us + us_spike.astype(float) * 8.0
        ca = ca + ca_spike.astype(float) * 6.0
        return pd.DataFrame({'us_series': us, 'ca_series': ca}, index=dates)

    def test_calendar_only_holiday_regression_creates_nonzero_component(self):
        df = self._make_calendar_spike_df()[['us_series']]
        detector = TimeSeriesFeatureDetector(
            holiday_country='US',
            **self._detector_kwargs(),
        )
        detector.fit(df)
        july_fourth_mask = (df.index.month == 7) & (df.index.day == 4)
        holiday_component = np.asarray(
            detector.components['us_series']['holidays'], dtype=float
        )
        self.assertGreater(np.abs(holiday_component[july_fourth_mask]).sum(), 0.0)

    def test_per_series_country_mapping_uses_distinct_calendar_dates(self):
        df = self._make_calendar_spike_df()
        detector = TimeSeriesFeatureDetector(
            holiday_country='US',
            holiday_countries={'ca_series': 'CA'},
            **self._detector_kwargs(),
        )
        detector.fit(df)
        us_dates = set(detector.holiday_dates['us_series'])
        ca_dates = set(detector.holiday_dates['ca_series'])
        self.assertIn(pd.Timestamp('2023-07-04'), us_dates)
        self.assertNotIn(pd.Timestamp('2023-07-01'), us_dates)
        self.assertIn(pd.Timestamp('2023-07-01'), ca_dates)
        self.assertNotIn(pd.Timestamp('2023-07-04'), ca_dates)

    def test_forecast_holiday_regressor_continuity_uses_calendar_dates(self):
        df = self._make_calendar_spike_df(end_date="2024-06-20")[['us_series']]
        detector = TimeSeriesFeatureDetector(
            holiday_country='US',
            **self._detector_kwargs(),
        )
        detector.fit(df)
        pred = detector.forecast(20)
        holidays = pred.components.xs('holidays', axis=1, level=1)
        self.assertIn(pd.Timestamp('2024-07-04'), holidays.index)
        self.assertGreater(
            float(abs(holidays.loc[pd.Timestamp('2024-07-04'), 'us_series'])),
            0.0,
        )

    def test_calendar_fusion_disables_datepart_holiday_features(self):
        detector = TimeSeriesFeatureDetector(
            holiday_country='US',
            **self._detector_kwargs(),
        )
        self.assertFalse(detector.rough_seasonality_params['holiday_countries_used'])
        self.assertFalse(detector.seasonality_params['holiday_countries_used'])


if __name__ == '__main__':
    unittest.main()
