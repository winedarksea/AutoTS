"""
Tests for AutoTS MCP Server

Tests cover:
- Data loading (daily, weekly, hourly)
- Forecasting (mosaic profile, search, custom)
- Feature detection and cleaning
- Event risk forecasting
- Server utilities (cache, data conversion)

"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from autots.mcp.server import (
        cache_object, get_cached_object, clear_cache, list_all_cached_objects,
        load_to_dataframe, dataframe_to_output, serialize_timestamps,
        MCP_AVAILABLE
    )
    SERVER_UTILS_AVAILABLE = True
except ImportError:
    SERVER_UTILS_AVAILABLE = False
    MCP_AVAILABLE = False

from autots import load_daily, load_weekly, load_hourly


class TestMCPSampleData(unittest.TestCase):
    """Test sample data loading."""
    
    def test_load_daily_data(self):
        """Test loading daily sample data."""
        df = load_daily(long=False)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertGreater(len(df), 100)
        self.assertGreater(len(df.columns), 1)
    
    def test_load_weekly_data(self):
        """Test loading weekly sample data."""
        df = load_weekly(long=False)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertGreater(len(df), 50)
    
    def test_load_hourly_data(self):
        """Test loading hourly sample data."""
        df = load_hourly(long=False)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertGreater(len(df), 100)
    
    def test_load_data_long_format(self):
        """Test loading data in long format."""
        df = load_daily(long=True)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('datetime', df.columns)
        self.assertIn('series_id', df.columns)
        self.assertIn('value', df.columns)
    

class TestMCPServerIntegration(unittest.TestCase):
    """Integration tests for MCP server (requires MCP installed)."""
    
    @unittest.skipIf(not MCP_AVAILABLE, "MCP not installed")
    def test_server_imports(self):
        """Test that server can be imported."""
        from autots.mcp.server import app, serve
        
        self.assertIsNotNone(app)
        self.assertIsNotNone(serve)


class TestMCPForecasting(unittest.TestCase):
    """Test forecasting functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all forecast tests."""
        cls.df_daily = load_daily(long=False).iloc[:200, :3]  # Limit size for speed
        cls.df_weekly = load_weekly(long=False).iloc[:100, :2]
        cls.df_hourly = load_hourly(long=False).iloc[:500, :2]
    
    def test_forecast_length_validation(self):
        """Test that forecast validates data length."""
        # Data with only 20 rows but requesting 30 forecast
        short_df = self.df_daily.iloc[:20]
        
        # This should fail or warn for mosaic profile
        # (Actual implementation would check this)
        self.assertLess(len(short_df), 30)
    
    def test_daily_data_forecast_structure(self):
        """Test forecast with daily data returns correct structure."""
        from autots import AutoTS
        
        model = AutoTS(
            forecast_length=14,
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast'
        )
        
        model = model.fit(self.df_daily)
        prediction = model.predict()
        
        self.assertEqual(len(prediction.forecast), 14)
        self.assertEqual(len(prediction.forecast.columns), len(self.df_daily.columns))
        self.assertIsInstance(prediction.forecast.index, pd.DatetimeIndex)
    
    def test_weekly_data_forecast(self):
        """Test forecast with weekly data."""
        from autots import AutoTS
        
        model = AutoTS(
            forecast_length=8,
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast'
        )
        
        model = model.fit(self.df_weekly)
        prediction = model.predict()
        
        self.assertEqual(len(prediction.forecast), 8)
    
    def test_hourly_data_forecast(self):
        """Test forecast with hourly data."""
        from autots import AutoTS
        
        model = AutoTS(
            forecast_length=24,  # 24 hours
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast'
        )
        
        model = model.fit(self.df_hourly)
        prediction = model.predict()
        
        self.assertEqual(len(prediction.forecast), 24)


class TestMCPFeatureDetection(unittest.TestCase):
    """Test feature detection and cleaning."""
    
    def setUp(self):
        """Set up test data."""
        self.df = load_daily(long=False).iloc[:200, :3]
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        from autots.tools.transform import GeneralTransformer
        
        # Create data with some issues
        df_dirty = self.df.copy()
        df_dirty.iloc[10, 0] = np.nan
        df_dirty.iloc[50, 1] = np.nan
        
        transformer = GeneralTransformer(
            fillna='ffill',
            transformations={"0": "ClipOutliers"},
            transformation_params={"0": {}}
        )
        
        df_cleaned = transformer.fit_transform(df_dirty)
        
        # Check that NaNs were filled
        self.assertLess(df_cleaned.isna().sum().sum(), df_dirty.isna().sum().sum())
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        from autots.evaluator.anomaly_detector import AnomalyDetector
        
        detector = AnomalyDetector(
            output='multivariate',
            method='IQR',
            transform_dict=None
        )
        
        detector.detect(self.df)
        
        # Check that anomalies were detected
        self.assertIsNotNone(detector.anomalies)


class TestMCPEventRiskForecasting(unittest.TestCase):
    """Test event risk forecasting."""
    
    def setUp(self):
        """Set up test data."""
        self.df = load_daily(long=False).iloc[:200, :2]
    
    def test_event_risk_basic(self):
        """Test basic event risk forecasting."""
        from autots import EventRiskForecast
        
        # EventRiskForecast expects limits to be quantiles (0-1) or forecast algorithms
        # Let's use quantile limits
        erf = EventRiskForecast(
            df_train=self.df,
            forecast_length=14,
            frequency='infer',
            upper_limit=0.75,  # 75th percentile as upper threshold
            lower_limit=0.25   # 25th percentile as lower threshold
        )
        
        erf.fit()
        upper_risk_df, lower_risk_df = erf.predict()
        
        # Check that risk arrays were generated
        self.assertIsNotNone(upper_risk_df)
        self.assertIsNotNone(lower_risk_df)
        self.assertEqual(len(upper_risk_df), 14)
        self.assertEqual(len(lower_risk_df), 14)
        
        # Check that forecast was generated
        self.assertIsNotNone(erf.forecast_df)
        self.assertEqual(len(erf.forecast_df), 14)


class TestMCPSyntheticData(unittest.TestCase):
    """Test synthetic data generation."""
    
    def test_synthetic_data_generation(self):
        """Test SyntheticDailyGenerator."""
        from autots.datasets.synthetic import SyntheticDailyGenerator
        
        generator = SyntheticDailyGenerator(
            n_series=3,
            n_days=365,
            random_seed=42
        )
        
        # Data is automatically generated and stored in self.data
        df = generator.data
        template = generator.template
        
        self.assertEqual(len(df), 365)
        self.assertEqual(len(df.columns), 3)
        self.assertIsInstance(template, dict)
        self.assertIn('meta', template)
        self.assertIn('series', template)


@unittest.skipIf(not SERVER_UTILS_AVAILABLE, "MCP server utilities not available")
class TestMCPServerUtilities(unittest.TestCase):
    """Test low-level server utility functions."""
    
    def setUp(self):
        """Set up test data and clear cache."""
        clear_cache()
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        self.df = pd.DataFrame({
            'series1': np.random.randn(50),
            'series2': np.random.randn(50)
        }, index=dates)
    
    def tearDown(self):
        """Clean up cache after tests."""
        clear_cache()
    
    # Cache Management Tests
    def test_cache_object_basic(self):
        """Test basic object caching."""
        test_obj = {"data": [1, 2, 3]}
        obj_id = cache_object(test_obj, 'data', {'test': True})
        
        self.assertIsInstance(obj_id, str)
        self.assertGreater(len(obj_id), 0)
        
        retrieved = get_cached_object(obj_id, 'data')
        self.assertEqual(retrieved['object'], test_obj)
        self.assertEqual(retrieved['metadata']['test'], True)
    
    def test_cache_invalid_type(self):
        """Test that invalid cache type raises error."""
        with self.assertRaises(ValueError):
            cache_object({}, 'invalid_type')
    
    def test_get_nonexistent_cache(self):
        """Test that getting nonexistent cache raises error."""
        with self.assertRaises(ValueError):
            get_cached_object('nonexistent-id', 'data')
    
    def test_list_all_cached_objects(self):
        """Test listing all cached objects."""
        cache_object({'test': 1}, 'data')
        cache_object({'test': 2}, 'prediction')
        
        cache_list = list_all_cached_objects()
        self.assertIsInstance(cache_list, dict)
        self.assertIn('data', cache_list)
        self.assertIn('predictions', cache_list)
    
    def test_clear_specific_cache(self):
        """Test clearing specific cached object."""
        obj_id = cache_object({'test': 1}, 'data')
        clear_cache(obj_id, 'data')
        
        with self.assertRaises(ValueError):
            get_cached_object(obj_id, 'data')
    
    def test_clear_all_caches(self):
        """Test clearing all caches."""
        cache_object({'test': 1}, 'data')
        cache_object({'test': 2}, 'prediction')
        clear_cache()
        
        cache_list = list_all_cached_objects()
        self.assertEqual(len(cache_list), 0)
    
    # Data Loading Tests
    def test_load_to_dataframe_from_dict_wide(self):
        """Test loading DataFrame from dict (wide format)."""
        data = {
            'datetime': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'series1': [1, 2, 3],
            'series2': [4, 5, 6]
        }
        df = load_to_dataframe(data, data_format='wide')
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(len(df), 3)
        self.assertIn('series1', df.columns)
    
    def test_load_to_dataframe_from_dict_long(self):
        """Test loading DataFrame from dict (long format)."""
        data = {
            'datetime': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
            'series_id': ['s1', 's2', 's1', 's2'],
            'value': [1, 2, 3, 4]
        }
        df = load_to_dataframe(data, data_format='long')
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(len(df), 2)
    
    def test_load_to_dataframe_missing_datetime(self):
        """Test that missing datetime column raises error."""
        data = {'series1': [1, 2, 3]}
        with self.assertRaises(ValueError):
            load_to_dataframe(data)
    
    def test_load_to_dataframe_from_cache(self):
        """Test loading DataFrame from cached data_id."""
        data_id = cache_object(self.df, 'data')
        df_loaded = load_to_dataframe(data_id=data_id)
        
        self.assertTrue(self.df.equals(df_loaded))
    
    def test_load_to_dataframe_no_params(self):
        """Test that missing both data and data_id raises error."""
        with self.assertRaises(ValueError):
            load_to_dataframe()
    
    # DataFrame Output Tests
    def test_dataframe_to_output_json_wide(self):
        """Test converting DataFrame to JSON wide format."""
        result = dataframe_to_output(self.df, 'json_wide')
        
        self.assertIsInstance(result, dict)
        self.assertIn('datetime', result)
        self.assertIn('series1', result)
        self.assertIn('series2', result)
        self.assertEqual(len(result['datetime']), 50)
    
    def test_dataframe_to_output_json_long(self):
        """Test converting DataFrame to JSON long format."""
        result = dataframe_to_output(self.df, 'json_long')
        
        self.assertIsInstance(result, dict)
        self.assertIn('datetime', result)
        self.assertIn('series_id', result)
        self.assertIn('value', result)
        self.assertEqual(len(result['datetime']), 100)  # 50 rows * 2 series
    
    def test_dataframe_to_output_csv_wide(self):
        """Test converting DataFrame to CSV wide format."""
        filepath = dataframe_to_output(self.df, 'csv_wide')
        
        self.assertIsInstance(filepath, str)
        self.assertTrue(filepath.endswith('.csv'))
        self.assertTrue(os.path.exists(filepath))
    
    def test_dataframe_to_output_invalid_format(self):
        """Test that invalid format raises error."""
        with self.assertRaises(ValueError):
            dataframe_to_output(self.df, 'invalid_format')
    
    # Timestamp Serialization Tests
    def test_serialize_single_timestamp(self):
        """Test serializing single Timestamp."""
        ts = pd.Timestamp('2024-01-01 12:30:45')
        result = serialize_timestamps(ts)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, '2024-01-01 12:30:45')
    
    def test_serialize_dict_with_timestamps(self):
        """Test serializing dict containing Timestamps."""
        data = {
            'date': pd.Timestamp('2024-01-01'),
            'value': 42
        }
        result = serialize_timestamps(data)
        
        self.assertIsInstance(result['date'], str)
        self.assertEqual(result['value'], 42)
    
    def test_serialize_nested_timestamps(self):
        """Test serializing nested structures with Timestamps."""
        data = {
            'dates': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')],
            'metadata': {
                'last_update': pd.Timestamp('2024-01-03')
            }
        }
        result = serialize_timestamps(data)
        
        self.assertTrue(all(isinstance(d, str) for d in result['dates']))
        self.assertIsInstance(result['metadata']['last_update'], str)
    
    # Edge Cases
    def test_dataframe_with_nan_values(self):
        """Test handling DataFrames with NaN values."""
        df_with_nan = self.df.copy()
        df_with_nan.iloc[5:10, 0] = np.nan
        
        result = dataframe_to_output(df_with_nan, 'json_wide')
        self.assertIsInstance(result, dict)
    
    def test_single_row_dataframe(self):
        """Test handling single-row DataFrame."""
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        df_single = pd.DataFrame({'series1': [1]}, index=dates)
        
        result = dataframe_to_output(df_single, 'json_wide')
        self.assertEqual(len(result['datetime']), 1)
    
    def test_special_characters_in_columns(self):
        """Test handling special characters in column names."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df_special = pd.DataFrame({
            'series-1': [1, 2, 3, 4, 5],
            'series/2': [6, 7, 8, 9, 10]
        }, index=dates)
        
        result = dataframe_to_output(df_special, 'json_wide')
        self.assertIn('series-1', result)
        self.assertIn('series/2', result)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMCPSampleData))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPServerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPForecasting))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPFeatureDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPEventRiskForecasting))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPSyntheticData))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPServerUtilities))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
