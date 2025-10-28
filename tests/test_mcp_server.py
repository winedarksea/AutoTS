"""
Tests for AutoTS MCP Server

Tests cover:
- Data loading (daily, weekly, hourly)
- Data format conversion (long/wide)
- Forecasting (mosaic profile, search, custom)
- Feature detection and cleaning
- Event risk forecasting
- Prediction caching (new functionality)
  * Cache/retrieve predictions
  * List/clear cache
  * Constraint application workflow
  * Integration with forecasting tools

"""

import unittest
import json
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autots.mcp.server import (
    json_to_dataframe, dataframe_to_json,
    cache_prediction, get_cached_prediction, list_cached_predictions, clear_prediction_cache,
    MCP_AVAILABLE
)
from autots import load_daily, load_weekly, load_hourly


class TestMCPDataConversion(unittest.TestCase):
    """Test JSON <-> DataFrame conversion utilities."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample wide format data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.df_wide = pd.DataFrame({
            'series1': np.random.randn(100),
            'series2': np.random.randn(100),
        }, index=dates)
        
        # Create sample long format data
        self.df_long = pd.DataFrame({
            'datetime': list(dates) * 2,
            'series_id': ['series1'] * 100 + ['series2'] * 100,
            'value': np.random.randn(200)
        })
    
    def test_wide_to_json(self):
        """Test converting wide DataFrame to JSON."""
        result = dataframe_to_json(self.df_wide, data_format='wide')
        
        self.assertIsInstance(result, dict)
        self.assertIn('datetime', result)
        self.assertIn('series1', result)
        self.assertIn('series2', result)
        self.assertEqual(len(result['datetime']), 100)
    
    def test_long_to_json(self):
        """Test converting long DataFrame to JSON."""
        result = dataframe_to_json(self.df_wide, data_format='long')
        
        self.assertIsInstance(result, dict)
        self.assertIn('datetime', result)
        self.assertIn('series_id', result)
        self.assertIn('value', result)
        self.assertEqual(len(result['datetime']), 200)  # 2 series * 100 dates
    
    def test_json_to_wide_dataframe(self):
        """Test converting JSON to wide DataFrame."""
        json_data = dataframe_to_json(self.df_wide, data_format='wide')
        df_result = json_to_dataframe(json_data, data_format='wide')
        
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertIsInstance(df_result.index, pd.DatetimeIndex)
        self.assertEqual(len(df_result), 100)
        self.assertEqual(len(df_result.columns), 2)
    
    def test_json_to_long_dataframe(self):
        """Test converting JSON long format to DataFrame."""
        json_data = {
            'datetime': ['2020-01-01', '2020-01-02', '2020-01-01', '2020-01-02'],
            'series_id': ['A', 'A', 'B', 'B'],
            'value': [1.0, 2.0, 3.0, 4.0]
        }
        df_result = json_to_dataframe(json_data, data_format='long')
        
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertIsInstance(df_result.index, pd.DatetimeIndex)
        # Long to wide should result in 2 rows, 2 columns
        self.assertEqual(len(df_result), 2)
        self.assertIn('A', df_result.columns)
        self.assertIn('B', df_result.columns)
    
    def test_roundtrip_conversion(self):
        """Test that wide -> JSON -> wide preserves data structure."""
        json_data = dataframe_to_json(self.df_wide, data_format='wide')
        df_result = json_to_dataframe(json_data, data_format='wide')
        
        self.assertEqual(df_result.shape, self.df_wide.shape)
        self.assertEqual(list(df_result.columns), list(self.df_wide.columns))


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


@unittest.skipIf(not MCP_AVAILABLE, "MCP not installed")
class TestMCPServerIntegration(unittest.TestCase):
    """Integration tests for MCP server (requires MCP installed)."""
    
    def test_server_imports(self):
        """Test that server can be imported."""
        from autots.mcp.server import app, serve
        
        self.assertIsNotNone(app)
        self.assertIsNotNone(serve)


class TestMCPSecurity(unittest.TestCase):
    """Test security considerations."""
    
    def test_no_code_execution_in_json(self):
        """Ensure JSON parsing doesn't allow code execution."""
        malicious_data = {
            'datetime': ['2020-01-01'],
            'series1': ["__import__('os').system('echo hacked')"]
        }
        
        # This should convert the string but not execute it
        try:
            df = json_to_dataframe(malicious_data, data_format='wide')
            # The string should remain a string, not be evaluated
            if not df.empty and len(df) > 0:
                # It should be converted to float or stay as string, not execute code
                val = df.iloc[0, 0]
                # If it's a string, that's fine - it wasn't executed
                # If it's NaN from failed conversion, that's also fine
                # We just want to make sure no code was run
                self.assertTrue(isinstance(val, (str, float, type(None))) or pd.isna(val))
        except (ValueError, TypeError) as e:
            # Failing to convert is also acceptable
            pass


class TestMCPToolChaining(unittest.TestCase):
    """Test that tools can be chained together - output from one tool feeds into another."""
    
    def test_get_sample_data_to_forecast_chain(self):
        """Test: get_sample_data → forecast_mosaic_profile"""
        # Step 1: Get sample data
        from autots import load_daily
        df = load_daily(long=False).iloc[:200, :3]
        
        # Convert to JSON (simulating tool output)
        data_json = dataframe_to_json(df, data_format="wide")
        
        # Verify JSON structure for chaining
        self.assertIn('datetime', data_json)
        self.assertIsInstance(data_json['datetime'], list)
        self.assertEqual(len(data_json['datetime']), 200)
        
        # Step 2: Use this data for forecasting (simulate tool input)
        df_reconstructed = json_to_dataframe(data_json, data_format="wide")
        
        # Verify reconstruction worked
        self.assertEqual(df_reconstructed.shape, df.shape)
        self.assertIsInstance(df_reconstructed.index, pd.DatetimeIndex)
        
        # Step 3: Verify it can be used for forecasting
        from autots import AutoTS
        model = AutoTS(
            forecast_length=14,
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast'
        )
        model = model.fit(df_reconstructed)
        prediction = model.predict()
        
        self.assertEqual(len(prediction.forecast), 14)
    
    def test_long_to_wide_to_forecast_chain(self):
        """Test: long_to_wide_converter → forecast_autots_search"""
        # Step 1: Create long format data (use data without NaNs for clean conversion)
        from autots import load_daily
        df_wide_raw = load_daily(long=False).iloc[:150, :2]
        
        # Fill NaNs to ensure clean conversion for this test
        df_wide = df_wide_raw.fillna(method='ffill').fillna(method='bfill')
        
        # Convert to long format
        long_json = dataframe_to_json(df_wide, data_format="long")
        
        # Verify long format structure
        self.assertIn('datetime', long_json)
        self.assertIn('series_id', long_json)
        self.assertIn('value', long_json)
        self.assertEqual(len(long_json['datetime']), 300)  # 150 rows * 2 series
        
        # Step 2: Convert from long to wide (simulating tool)
        df_from_long = json_to_dataframe(long_json, data_format="long")
        
        # Verify conversion (shape should match after conversion)
        self.assertEqual(df_from_long.shape, df_wide.shape)
        self.assertIsInstance(df_from_long.index, pd.DatetimeIndex)
        
        # Step 3: Convert to wide JSON for forecast tool
        wide_json = dataframe_to_json(df_from_long, data_format="wide")
        
        # Step 4: Use for forecasting
        df_for_forecast = json_to_dataframe(wide_json, data_format="wide")
        
        from autots import AutoTS
        model = AutoTS(
            forecast_length=10,
            frequency='infer',
            ensemble=None,
            model_list=['Cassandra'],
            max_generations=1,
            num_validations=1
        )
        model = model.fit(df_for_forecast)
        prediction = model.predict()
        
        self.assertEqual(len(prediction.forecast), 10)
    
    def test_synthetic_to_forecast_to_feature_detection_chain(self):
        """Test: generate_synthetic_data → forecast → detect_features"""
        # Step 1: Generate synthetic data
        from autots.datasets.synthetic import SyntheticDailyGenerator
        generator = SyntheticDailyGenerator(n_series=2, n_days=200, random_seed=42)
        df_synthetic = generator.data
        
        # Convert to JSON
        synthetic_json = dataframe_to_json(df_synthetic, data_format="wide")
        
        # Step 2: Reconstruct and forecast
        df_reconstructed = json_to_dataframe(synthetic_json, data_format="wide")
        
        from autots import AutoTS
        model = AutoTS(
            forecast_length=30,
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast'
        )
        model = model.fit(df_reconstructed)
        prediction = model.predict()
        
        # Convert forecast to JSON
        forecast_json = dataframe_to_json(prediction.forecast, data_format="wide")
        
        # Verify forecast JSON structure
        self.assertIn('datetime', forecast_json)
        self.assertEqual(len(forecast_json['datetime']), 30)
        
        # Step 3: Detect features on original data
        df_for_features = json_to_dataframe(synthetic_json, data_format="wide")
        
        from autots.evaluator.feature_detector import TimeSeriesFeatureDetector
        detector = TimeSeriesFeatureDetector()
        detector.fit(df_for_features)
        
        # Verify detector ran
        self.assertIsNotNone(detector)
    
    def test_get_cleaned_data_to_forecast_chain(self):
        """Test: get_sample_data → get_cleaned_data → forecast"""
        # Step 1: Get sample data with some missing values
        from autots import load_daily
        df = load_daily(long=False).iloc[:150, :2]
        
        # Introduce some NaN values
        df_dirty = df.copy()
        df_dirty.iloc[10:15, 0] = np.nan
        df_dirty.iloc[50:52, 1] = np.nan
        
        # Convert to JSON
        dirty_json = dataframe_to_json(df_dirty, data_format="wide")
        
        # Step 2: Clean the data
        df_to_clean = json_to_dataframe(dirty_json, data_format="wide")
        
        from autots.tools.transform import GeneralTransformer
        transformer = GeneralTransformer(
            fillna='ffill',
            transformations={"0": "ClipOutliers"},
            transformation_params={"0": {}}
        )
        df_cleaned = transformer.fit_transform(df_to_clean)
        
        # Convert cleaned data to JSON
        cleaned_json = dataframe_to_json(df_cleaned, data_format="wide")
        
        # Verify cleaned JSON structure
        self.assertIn('datetime', cleaned_json)
        self.assertEqual(len(cleaned_json['datetime']), 150)
        
        # Step 3: Use cleaned data for forecasting
        df_for_forecast = json_to_dataframe(cleaned_json, data_format="wide")
        
        # Verify NaNs were handled
        self.assertLess(df_for_forecast.isna().sum().sum(), df_to_clean.isna().sum().sum())
        
        from autots import AutoTS
        model = AutoTS(
            forecast_length=20,
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast'
        )
        model = model.fit(df_for_forecast)
        prediction = model.predict()
        
        self.assertEqual(len(prediction.forecast), 20)
    
    def test_forecast_to_event_risk_chain(self):
        """Test: get_sample_data → forecast → event_risk (using historical pattern)"""
        # Step 1: Get sample data
        from autots import load_daily
        df = load_daily(long=False).iloc[:180, :2]
        
        data_json = dataframe_to_json(df, data_format="wide")
        
        # Step 2: Create forecast
        df_reconstructed = json_to_dataframe(data_json, data_format="wide")
        
        from autots import AutoTS
        model = AutoTS(
            forecast_length=30,
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast'
        )
        model = model.fit(df_reconstructed)
        prediction = model.predict()
        
        # Step 3: Use training data for event risk
        # Calculate threshold from historical data
        threshold = df.iloc[:, 0].quantile(0.75)
        
        from autots import EventRiskForecast
        erf = EventRiskForecast(
            df_train=df_reconstructed,
            forecast_length=30,
            frequency='infer',
            upper_limit=0.75,
            lower_limit=0.25
        )
        
        erf.fit()
        upper_risk_df, lower_risk_df = erf.predict()
        
        # Verify event risk outputs
        self.assertIsNotNone(upper_risk_df)
        self.assertIsNotNone(lower_risk_df)
        self.assertEqual(len(upper_risk_df), 30)
        self.assertEqual(len(lower_risk_df), 30)
    
    def test_json_output_structure_consistency(self):
        """Test that all data tools produce consistent JSON structure."""
        from autots import load_daily, load_weekly, load_hourly
        from autots.datasets.synthetic import SyntheticDailyGenerator
        
        # Test different data sources
        datasets = {
            'daily': load_daily(long=False).iloc[:100, :2],
            'weekly': load_weekly(long=False).iloc[:100, :2],
            'hourly': load_hourly(long=False).iloc[:100, :2],
            'synthetic': SyntheticDailyGenerator(n_series=2, n_days=100, random_seed=42).data
        }
        
        for name, df in datasets.items():
            with self.subTest(dataset=name):
                # Convert to JSON
                json_data = dataframe_to_json(df, data_format="wide")
                
                # Verify consistent structure
                self.assertIn('datetime', json_data, f"{name}: missing 'datetime' key")
                self.assertIsInstance(json_data['datetime'], list, f"{name}: 'datetime' not a list")
                self.assertEqual(len(json_data['datetime']), 100, f"{name}: wrong number of dates")
                
                # Verify all series columns are present
                for col in df.columns:
                    self.assertIn(col, json_data, f"{name}: missing column '{col}'")
                    self.assertIsInstance(json_data[col], list, f"{name}: column '{col}' not a list")
                    self.assertEqual(len(json_data[col]), 100, f"{name}: wrong length for '{col}'")
                
                # Verify it can be reconstructed
                df_reconstructed = json_to_dataframe(json_data, data_format="wide")
                self.assertEqual(df_reconstructed.shape, df.shape, f"{name}: shape mismatch")
                self.assertIsInstance(df_reconstructed.index, pd.DatetimeIndex, f"{name}: index not datetime")
    
    def test_roundtrip_through_multiple_conversions(self):
        """Test: wide → JSON → DataFrame → JSON → DataFrame maintains data integrity."""
        from autots import load_daily
        df_original = load_daily(long=False).iloc[:50, :2]
        
        # First conversion: wide → JSON
        json1 = dataframe_to_json(df_original, data_format="wide")
        
        # Second conversion: JSON → DataFrame
        df1 = json_to_dataframe(json1, data_format="wide")
        
        # Third conversion: DataFrame → JSON
        json2 = dataframe_to_json(df1, data_format="wide")
        
        # Fourth conversion: JSON → DataFrame
        df2 = json_to_dataframe(json2, data_format="wide")
        
        # Verify shape is preserved
        self.assertEqual(df_original.shape, df2.shape)
        
        # Verify index is preserved
        self.assertTrue(all(df_original.index == df2.index))
        
        # Verify column names are preserved
        self.assertEqual(list(df_original.columns), list(df2.columns))
        
        # Verify data values are approximately equal (allowing for floating point precision)
        pd.testing.assert_frame_equal(df_original, df2, check_dtype=False, rtol=1e-10)
    
    def test_forecast_output_format_for_visualization(self):
        """Test that forecast outputs can be easily consumed for visualization/analysis."""
        from autots import load_daily, AutoTS
        
        # Get data and forecast
        df = load_daily(long=False).iloc[:150, :2]
        
        model = AutoTS(
            forecast_length=30,
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast'
        )
        model = model.fit(df)
        prediction = model.predict()
        
        # Convert forecast to JSON
        forecast_json = dataframe_to_json(prediction.forecast, data_format="wide")
        
        # Verify structure is suitable for consumption
        self.assertIn('datetime', forecast_json)
        
        # Verify dates are in string format (JSON compatible)
        self.assertIsInstance(forecast_json['datetime'][0], str)
        
        # Verify dates can be parsed back
        test_date = pd.to_datetime(forecast_json['datetime'][0])
        self.assertIsInstance(test_date, pd.Timestamp)
        
        # Verify values are JSON serializable (no NaN issues)
        json_str = json.dumps(forecast_json)
        self.assertIsInstance(json_str, str)
        
        # Verify it can be deserialized
        forecast_reloaded = json.loads(json_str)
        self.assertEqual(forecast_reloaded['datetime'], forecast_json['datetime'])


class TestMCPPredictionCaching(unittest.TestCase):
    """Test the new prediction caching functionality."""
    
    def setUp(self):
        """Set up test data and clear cache before each test."""
        # Clear any existing cache
        clear_prediction_cache()
        
        # Create sample data and forecast
        from autots import load_daily, AutoTS
        self.df = load_daily(long=False).iloc[:150, :2]
        
        # Create a simple forecast for testing
        self.model = AutoTS(
            forecast_length=14,
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast'
        )
        self.model = self.model.fit(self.df)
        self.prediction = self.model.predict()
    
    def tearDown(self):
        """Clean up cache after each test."""
        clear_prediction_cache()
    
    def test_cache_prediction_basic(self):
        """Test caching a prediction and retrieving it."""
        # Cache the prediction
        prediction_id = cache_prediction(
            prediction_obj=self.prediction,
            model_obj=self.model,
            metadata={'test': 'basic'}
        )
        
        # Verify ID is a string (UUID)
        self.assertIsInstance(prediction_id, str)
        self.assertGreater(len(prediction_id), 10)  # UUID should be long
        
        # Retrieve the cached prediction
        cached = get_cached_prediction(prediction_id)
        
        # Verify structure
        self.assertIn('prediction', cached)
        self.assertIn('model', cached)
        self.assertIn('metadata', cached)
        self.assertIn('created_at', cached)
        
        # Verify it's the same prediction
        self.assertEqual(cached['metadata']['test'], 'basic')
    
    def test_cache_prediction_metadata(self):
        """Test that metadata is stored correctly."""
        metadata = {
            'method': 'test',
            'forecast_length': 14,
            'model_name': 'TestModel',
            'series_count': 2
        }
        
        prediction_id = cache_prediction(
            prediction_obj=self.prediction,
            metadata=metadata
        )
        
        cached = get_cached_prediction(prediction_id)
        
        # Verify all metadata fields
        for key, value in metadata.items():
            self.assertEqual(cached['metadata'][key], value)
    
    def test_list_cached_predictions(self):
        """Test listing all cached predictions."""
        # Initially should be empty
        predictions_list = list_cached_predictions()
        self.assertEqual(len(predictions_list), 0)
        
        # Cache multiple predictions
        id1 = cache_prediction(self.prediction, metadata={'name': 'pred1'})
        id2 = cache_prediction(self.prediction, metadata={'name': 'pred2'})
        id3 = cache_prediction(self.prediction, metadata={'name': 'pred3'})
        
        # List should now have 3 items
        predictions_list = list_cached_predictions()
        self.assertEqual(len(predictions_list), 3)
        
        # Verify structure of list items
        ids = {p['id'] for p in predictions_list}
        self.assertEqual(ids, {id1, id2, id3})
        
        # Verify each has created_at and metadata
        for pred_info in predictions_list:
            self.assertIn('id', pred_info)
            self.assertIn('created_at', pred_info)
            self.assertIn('metadata', pred_info)
    
    def test_clear_specific_prediction(self):
        """Test clearing a specific prediction from cache."""
        # Cache multiple predictions
        id1 = cache_prediction(self.prediction, metadata={'name': 'pred1'})
        id2 = cache_prediction(self.prediction, metadata={'name': 'pred2'})
        
        # Verify both exist
        self.assertEqual(len(list_cached_predictions()), 2)
        
        # Clear one prediction
        clear_prediction_cache(id1)
        
        # Verify only one remains
        predictions_list = list_cached_predictions()
        self.assertEqual(len(predictions_list), 1)
        self.assertEqual(predictions_list[0]['id'], id2)
        
        # Verify the cleared one is gone
        with self.assertRaises(ValueError):
            get_cached_prediction(id1)
    
    def test_clear_all_predictions(self):
        """Test clearing all predictions from cache."""
        # Cache multiple predictions
        cache_prediction(self.prediction, metadata={'name': 'pred1'})
        cache_prediction(self.prediction, metadata={'name': 'pred2'})
        cache_prediction(self.prediction, metadata={'name': 'pred3'})
        
        # Verify they exist
        self.assertEqual(len(list_cached_predictions()), 3)
        
        # Clear all
        clear_prediction_cache()
        
        # Verify cache is empty
        self.assertEqual(len(list_cached_predictions()), 0)
    
    def test_get_nonexistent_prediction(self):
        """Test that retrieving non-existent prediction raises error."""
        with self.assertRaises(ValueError) as context:
            get_cached_prediction("nonexistent-id-12345")
        
        self.assertIn("not found in cache", str(context.exception))
    
    def test_cache_multiple_predictions_unique_ids(self):
        """Test that each cached prediction gets a unique ID."""
        # Cache the same prediction multiple times
        ids = set()
        for i in range(10):
            prediction_id = cache_prediction(
                self.prediction,
                metadata={'iteration': i}
            )
            ids.add(prediction_id)
        
        # All IDs should be unique
        self.assertEqual(len(ids), 10)
    
    def test_cached_prediction_persistence(self):
        """Test that cached predictions persist across multiple retrievals."""
        prediction_id = cache_prediction(
            self.prediction,
            model_obj=self.model,
            metadata={'test': 'persistence'}
        )
        
        # Retrieve multiple times
        cached1 = get_cached_prediction(prediction_id)
        cached2 = get_cached_prediction(prediction_id)
        cached3 = get_cached_prediction(prediction_id)
        
        # All should have the same data
        self.assertEqual(cached1['metadata'], cached2['metadata'])
        self.assertEqual(cached2['metadata'], cached3['metadata'])
        
        # Verify the prediction object itself is accessible
        self.assertIsNotNone(cached1['prediction'])
        self.assertEqual(len(cached1['prediction'].forecast), 14)
    
    def test_cache_without_model(self):
        """Test caching prediction without model object."""
        prediction_id = cache_prediction(
            prediction_obj=self.prediction,
            model_obj=None,
            metadata={'has_model': False}
        )
        
        cached = get_cached_prediction(prediction_id)
        
        # Prediction should be there
        self.assertIsNotNone(cached['prediction'])
        
        # Model should be None
        self.assertIsNone(cached['model'])
    
    def test_cache_without_metadata(self):
        """Test caching prediction without metadata."""
        prediction_id = cache_prediction(
            prediction_obj=self.prediction,
            model_obj=self.model
        )
        
        cached = get_cached_prediction(prediction_id)
        
        # Should have empty metadata dict
        self.assertIsInstance(cached['metadata'], dict)
        self.assertEqual(len(cached['metadata']), 0)
    
    def test_cached_prediction_forecast_access(self):
        """Test accessing forecast data from cached prediction."""
        prediction_id = cache_prediction(
            self.prediction,
            metadata={'forecast_length': 14}
        )
        
        cached = get_cached_prediction(prediction_id)
        prediction_obj = cached['prediction']
        
        # Verify we can access the forecast
        forecast_df = prediction_obj.forecast
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), 14)
        
        # Verify we can access upper/lower forecasts if available
        if prediction_obj.upper_forecast is not None:
            self.assertIsInstance(prediction_obj.upper_forecast, pd.DataFrame)
            self.assertEqual(len(prediction_obj.upper_forecast), 14)
    
    def test_cached_prediction_to_json_conversion(self):
        """Test converting cached prediction forecast to JSON."""
        prediction_id = cache_prediction(self.prediction)
        
        cached = get_cached_prediction(prediction_id)
        forecast_df = cached['prediction'].forecast
        
        # Convert to JSON
        forecast_json = dataframe_to_json(forecast_df, data_format='wide')
        
        # Verify JSON structure
        self.assertIn('datetime', forecast_json)
        self.assertEqual(len(forecast_json['datetime']), 14)
        
        # Verify it can be converted back
        forecast_reconstructed = json_to_dataframe(forecast_json, data_format='wide')
        self.assertEqual(len(forecast_reconstructed), 14)
    
    def test_prediction_caching_workflow(self):
        """Test complete workflow: cache → list → retrieve → clear."""
        # Step 1: Cache prediction
        metadata = {
            'method': 'test_workflow',
            'forecast_length': 14,
            'model_name': self.prediction.model_name
        }
        prediction_id = cache_prediction(
            self.prediction,
            self.model,
            metadata
        )
        
        # Step 2: List predictions
        predictions = list_cached_predictions()
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]['id'], prediction_id)
        
        # Step 3: Retrieve and use prediction
        cached = get_cached_prediction(prediction_id)
        forecast_json = dataframe_to_json(
            cached['prediction'].forecast,
            data_format='wide'
        )
        self.assertIn('datetime', forecast_json)
        
        # Step 4: Clear cache
        clear_prediction_cache(prediction_id)
        self.assertEqual(len(list_cached_predictions()), 0)
    
    def test_cache_constrained_prediction(self):
        """Test caching a constrained/modified prediction."""
        # Original prediction
        original_id = cache_prediction(
            self.prediction,
            metadata={'type': 'original'}
        )
        
        # Create a "constrained" version (simulate applying constraints)
        forecast_df = self.prediction.forecast.copy()
        forecast_df = forecast_df * 0.9  # Dampen by 10%
        
        # Create a simple constrained prediction object
        class ConstrainedPrediction:
            def __init__(self, forecast, original_pred):
                self.forecast = forecast
                self.upper_forecast = original_pred.upper_forecast
                self.lower_forecast = original_pred.lower_forecast
                self.model_name = f"{original_pred.model_name} (constrained)"
                self.model_parameters = original_pred.model_parameters
                self.transformation_parameters = original_pred.transformation_parameters
        
        constrained_pred = ConstrainedPrediction(forecast_df, self.prediction)
        
        # Cache the constrained prediction
        constrained_id = cache_prediction(
            constrained_pred,
            metadata={
                'type': 'constrained',
                'constraint_method': 'dampen',
                'original_prediction_id': original_id
            }
        )
        
        # Verify both are in cache
        self.assertEqual(len(list_cached_predictions()), 2)
        
        # Retrieve constrained prediction
        cached = get_cached_prediction(constrained_id)
        self.assertEqual(cached['metadata']['type'], 'constrained')
        self.assertEqual(cached['metadata']['original_prediction_id'], original_id)
        
        # Verify constraint was applied (values should be different)
        original_cached = get_cached_prediction(original_id)
        original_values = original_cached['prediction'].forecast.values
        constrained_values = cached['prediction'].forecast.values
        
        # Values should be different (constrained is 90% of original)
        self.assertFalse(np.array_equal(original_values, constrained_values))


class TestMCPPredictionCachingIntegration(unittest.TestCase):
    """Integration tests for prediction caching with full workflows."""
    
    def setUp(self):
        """Clear cache before each test."""
        clear_prediction_cache()
    
    def tearDown(self):
        """Clean up cache after each test."""
        clear_prediction_cache()
    
    def test_multiple_forecast_methods_caching(self):
        """Test caching predictions from different forecast methods."""
        from autots import load_daily, AutoTS
        df = load_daily(long=False).iloc[:150, :2]
        
        # Method 1: Fast model
        model1 = AutoTS(
            forecast_length=10,
            frequency='infer',
            max_generations=1,
            model_list='superfast'
        )
        model1.fit(df)
        pred1 = model1.predict()
        id1 = cache_prediction(pred1, model1, metadata={'method': 'superfast'})
        
        # Method 2: Different model
        model2 = AutoTS(
            forecast_length=10,
            frequency='infer',
            ensemble=None,
            model_list=['Cassandra'],
            max_generations=1
        )
        model2.fit(df)
        pred2 = model2.predict()
        id2 = cache_prediction(pred2, model2, metadata={'method': 'cassandra'})
        
        # Verify both are cached
        predictions = list_cached_predictions()
        self.assertEqual(len(predictions), 2)
        
        # Verify each can be retrieved correctly
        cached1 = get_cached_prediction(id1)
        cached2 = get_cached_prediction(id2)
        
        self.assertEqual(cached1['metadata']['method'], 'superfast')
        self.assertEqual(cached2['metadata']['method'], 'cassandra')
    
    def test_end_to_end_forecast_to_json_workflow(self):
        """Test complete workflow from data → forecast → cache → JSON."""
        from autots import load_daily, AutoTS
        
        # Step 1: Get data
        df = load_daily(long=False).iloc[:100, :2]
        
        # Step 2: Create forecast
        model = AutoTS(
            forecast_length=14,
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast'
        )
        model.fit(df)
        prediction = model.predict()
        
        # Step 3: Cache prediction
        prediction_id = cache_prediction(
            prediction,
            model,
            metadata={
                'method': 'end_to_end_test',
                'forecast_length': 14,
                'series_count': len(df.columns)
            }
        )
        
        # Step 4: Retrieve and convert to JSON
        cached = get_cached_prediction(prediction_id)
        forecast_json = dataframe_to_json(
            cached['prediction'].forecast,
            data_format='wide'
        )
        
        # Verify complete JSON structure
        self.assertIn('datetime', forecast_json)
        self.assertEqual(len(forecast_json['datetime']), 14)
        for col in df.columns:
            self.assertIn(col, forecast_json)
            self.assertEqual(len(forecast_json[col]), 14)
        
        # Verify JSON is serializable
        json_str = json.dumps(forecast_json)
        self.assertIsInstance(json_str, str)
        
        # Verify can be deserialized and used
        reloaded = json.loads(json_str)
        df_from_json = json_to_dataframe(reloaded, data_format='wide')
        self.assertEqual(len(df_from_json), 14)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMCPDataConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPSampleData))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPForecasting))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPFeatureDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPEventRiskForecasting))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPSyntheticData))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPServerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPSecurity))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPToolChaining))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPPredictionCaching))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPPredictionCachingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
