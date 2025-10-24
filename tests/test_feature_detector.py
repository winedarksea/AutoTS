# -*- coding: utf-8 -*-
"""
Tests for Feature Detector

@author: Colin
"""

import unittest
import pandas as pd
import numpy as np
from autots.datasets.synthetic import SyntheticDailyGenerator
from autots.evaluator.feature_detector import (
    TimeSeriesFeatureDetector,
    FeatureDetectionLoss,
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
            self.assertEqual(dates_0, dates_1,
                           "Univariate mode should produce identical anomaly dates across series")
    
    def test_template_metadata_includes_detection_mode(self):
        """Test that template metadata includes detection_mode."""
        for mode in ['multivariate', 'univariate']:
            detector = TimeSeriesFeatureDetector(detection_mode=mode)
            detector.fit(self.data)
            
            template = detector.get_template()
            self.assertIn('detector_config', template['meta'])
            self.assertIn('detection_mode', template['meta']['detector_config'])
            self.assertEqual(template['meta']['detector_config']['detection_mode'], mode)
    
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
            level_shift_params={'window_size': 30, 'output': 'multivariate'}
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
        
        # Get features for specific series
        series_name = self.data.columns[0]
        series_features = detector.get_detected_features(series_name)
        self.assertIn('trend_changepoints', series_features)
    
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
            optimization_method='random_search',
            n_iterations=3,
        )
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.n_iterations, 3)
    
    def test_random_search(self):
        """Test random search optimization."""
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            optimization_method='random_search',
            n_iterations=3,
        )
        
        best_params = optimizer.optimize()
        
        self.assertIsNotNone(best_params)
        self.assertIn('seasonality_params', best_params)
        self.assertIsNotNone(optimizer.best_loss)
        # History only includes successful iterations (failed ones are excluded)
        self.assertGreater(len(optimizer.optimization_history), 0)
        self.assertLessEqual(len(optimizer.optimization_history), 3)
    
    def test_grid_search(self):
        """Test grid search optimization."""
        optimizer = FeatureDetectionOptimizer(
            self.generator,
            optimization_method='grid_search',
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
            optimization_method='random_search',
            n_iterations=2,
        )
        optimizer.optimize()
        
        # Should not raise error
        try:
            optimizer.get_optimization_summary()
            success = True
        except Exception as e:
            print(f"Summary failed: {e}")
            success = False
        
        self.assertTrue(success)


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
            optimization_method='random_search',
            n_iterations=3,
        )
        best_params = optimizer.optimize()
        
        print(f"Optimized loss: {optimizer.best_loss:.4f}")
        
        # Verify optimization improved or maintained performance
        self.assertLessEqual(optimizer.best_loss, loss['total_loss'] * 1.5)  # Allow some variance
    
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
            len(detected['trend_changepoints']) +
            len(detected['anomalies']) +
            len(detected['level_shifts'])
        )
        self.assertGreater(total_detected, 0)


if __name__ == '__main__':
    unittest.main()
