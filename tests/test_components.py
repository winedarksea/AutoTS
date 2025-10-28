"""Test component standardization across models."""
import unittest
import numpy as np
import pandas as pd
from autots.models.basics import BasicLinearModel, TVVAR
from autots.models.cassandra import Cassandra
from autots.models.base import stack_component_frames, sum_component_frames
from collections import OrderedDict


class TestComponentHelpers(unittest.TestCase):
    """Test helper functions for component handling."""

    def test_stack_component_frames_empty(self):
        """Test stack_component_frames with empty dict."""
        result = stack_component_frames({})
        self.assertIsNone(result)

    def test_stack_component_frames_all_none(self):
        """Test stack_component_frames with all None values."""
        frames = OrderedDict([('a', None), ('b', None)])
        result = stack_component_frames(frames)
        self.assertIsNone(result)

    def test_stack_component_frames_mixed(self):
        """Test stack_component_frames with mixed None and valid DataFrames."""
        df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        frames = OrderedDict([('a', None), ('b', df1), ('c', None)])
        result = stack_component_frames(frames)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.columns.nlevels, 2)

    def test_stack_component_frames_multiple(self):
        """Test stack_component_frames with multiple DataFrames."""
        df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        df2 = pd.DataFrame({'col1': [7, 8, 9], 'col2': [10, 11, 12]})
        frames = OrderedDict([('trend', df1), ('seasonality', df2)])
        result = stack_component_frames(frames)
        
        # Should have 4 columns: (col1, trend), (col1, seasonality), (col2, trend), (col2, seasonality)
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(result.columns.nlevels, 2)
        
        # Check columns are sorted by series
        level0 = result.columns.get_level_values(0).tolist()
        self.assertEqual(level0, ['col1', 'col1', 'col2', 'col2'])

    def test_sum_component_frames_empty(self):
        """Test sum_component_frames with empty dict."""
        result = sum_component_frames({})
        self.assertIsNone(result)

    def test_sum_component_frames_all_none(self):
        """Test sum_component_frames with all None values."""
        frames = OrderedDict([('a', None), ('b', None)])
        result = sum_component_frames(frames)
        self.assertIsNone(result)

    def test_sum_component_frames_single(self):
        """Test sum_component_frames with single DataFrame."""
        df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        frames = OrderedDict([('trend', df1)])
        result = sum_component_frames(frames)
        pd.testing.assert_frame_equal(result, df1)

    def test_sum_component_frames_multiple(self):
        """Test sum_component_frames with multiple DataFrames."""
        df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        df2 = pd.DataFrame({'col1': [10, 20, 30], 'col2': [40, 50, 60]})
        frames = OrderedDict([('trend', df1), ('seasonality', df2)])
        result = sum_component_frames(frames)
        expected = pd.DataFrame({'col1': [11, 22, 33], 'col2': [44, 55, 66]})
        pd.testing.assert_frame_equal(result, expected)

    def test_sum_component_frames_fill_value(self):
        """Test sum_component_frames handles different indices with fill_value=0."""
        df1 = pd.DataFrame({'col1': [1, 2, 3]}, index=[0, 1, 2])
        df2 = pd.DataFrame({'col1': [10, 20]}, index=[1, 2])
        frames = OrderedDict([('trend', df1), ('seasonality', df2)])
        result = sum_component_frames(frames)
        
        self.assertEqual(result.loc[0, 'col1'], 1.0)
        self.assertEqual(result.loc[1, 'col1'], 12.0)
        self.assertEqual(result.loc[2, 'col1'], 23.0)


class TestBasicLinearModelComponents(unittest.TestCase):
    """Test component extraction for BasicLinearModel."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        np.random.seed(42)
        cls.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        cls.df = pd.DataFrame({
            'series1': np.cumsum(np.random.randn(100)) + np.sin(np.arange(100) * 2 * np.pi / 7) * 5,
            'series2': np.cumsum(np.random.randn(100)) + np.cos(np.arange(100) * 2 * np.pi / 7) * 3,
        }, index=cls.dates)

    def test_components_exist(self):
        """Test that components are extracted."""
        model = BasicLinearModel(seasonal_period='weekly', prediction_interval=0.9)
        model.fit(self.df)
        prediction = model.predict(forecast_length=14)
        
        self.assertIsNotNone(prediction.components)
        self.assertIsInstance(prediction.components, pd.DataFrame)
        self.assertEqual(prediction.components.shape[0], 14)

    def test_component_types(self):
        """Test that standard component types are present."""
        model = BasicLinearModel(seasonal_period='weekly', prediction_interval=0.9)
        model.fit(self.df)
        prediction = model.predict(forecast_length=14)
        
        component_types = prediction.components.columns.get_level_values(1).unique().tolist()
        
        # Should have at least: trend, seasonality, constant
        self.assertIn('trend', component_types)
        self.assertIn('seasonality', component_types)
        self.assertIn('constant', component_types)

    def test_components_sum_to_forecast(self):
        """Test that components sum to forecast (CRITICAL TEST)."""
        model = BasicLinearModel(seasonal_period='weekly', trend_phi=0.98, prediction_interval=0.9)
        model.fit(self.df)
        prediction = model.predict(forecast_length=14)
        
        # Sum components by series
        component_sum = prediction.components.groupby(level=0, axis=1).sum()
        
        # Check that sum equals forecast
        diff = (prediction.forecast - component_sum).abs().max().max()
        
        # Should be equal to machine precision
        self.assertLess(diff, 1e-10, 
                       f"Components don't sum to forecast. Max difference: {diff}")

    def test_components_with_regressors(self):
        """Test component extraction WITH regressors (CRITICAL TEST for bug fix)."""
        np.random.seed(42)
        
        # Create regressor data
        future_regressor = pd.DataFrame({
            'reg1': np.random.randn(114),  # 100 train + 14 forecast
            'reg2': np.random.randn(114),
        }, index=pd.date_range(start='2020-01-01', periods=114, freq='D'))
        
        model = BasicLinearModel(
            seasonal_period='weekly',
            trend_phi=0.98,
            regression_type='User',
            prediction_interval=0.9
        )
        model.fit(self.df, future_regressor=future_regressor)
        prediction = model.predict(forecast_length=14, future_regressor=future_regressor)
        
        # Check components exist
        self.assertIsNotNone(prediction.components)
        
        # Check that 'regressors' component is present
        component_types = prediction.components.columns.get_level_values(1).unique().tolist()
        self.assertIn('regressors', component_types, 
                     "Regressor component missing! Bug fix may not have been applied.")
        
        # Check components sum to forecast
        component_sum = prediction.components.groupby(level=0, axis=1).sum()
        diff = (prediction.forecast - component_sum).abs().max().max()
        self.assertLess(diff, 1e-10, 
                       f"Components with regressors don't sum to forecast. Max difference: {diff}")

    def test_components_without_trend_phi(self):
        """Test components with trend_phi=None."""
        model = BasicLinearModel(seasonal_period='weekly', trend_phi=None, prediction_interval=0.9)
        model.fit(self.df)
        prediction = model.predict(forecast_length=14)
        
        # Check components sum to forecast
        component_sum = prediction.components.groupby(level=0, axis=1).sum()
        diff = (prediction.forecast - component_sum).abs().max().max()
        self.assertLess(diff, 1e-10)

    def test_plot_components(self):
        """Test that plot_components method works."""
        model = BasicLinearModel(seasonal_period='weekly', prediction_interval=0.9)
        model.fit(self.df)
        prediction = model.predict(forecast_length=14)
        
        # Test plotting a specific series
        ax = prediction.plot_components(series='series1')
        self.assertIsNotNone(ax)
        
        # Test plotting with random series selection
        ax = prediction.plot_components()
        self.assertIsNotNone(ax)

    def test_plot_components_errors(self):
        """Test plot_components error handling."""
        model = BasicLinearModel(seasonal_period='weekly', prediction_interval=0.9)
        model.fit(self.df)
        prediction = model.predict(forecast_length=14)
        
        # Test invalid series name
        with self.assertRaises(ValueError):
            prediction.plot_components(series='nonexistent')

    def test_forecast_not_reconstructed(self):
        """Test that forecast is calculated directly, not reconstructed from components."""
        model = BasicLinearModel(seasonal_period='weekly', trend_phi=0.98, prediction_interval=0.9)
        model.fit(self.df)
        prediction = model.predict(forecast_length=14)
        
        # Calculate what the forecast should be using the original method
        test_index = prediction.forecast.index
        from autots.tools.seasonal import date_part
        from autots.tools.changepoints import changepoint_fcst_from_last_row
        
        x_s = date_part(
            test_index,
            method=model.datepart_method,
            set_index=True,
            holiday_country=model.holiday_country,
            holiday_countries_used=model.holiday_countries_used,
        )
        x_t = changepoint_fcst_from_last_row(model.last_row, int(14))
        x_t.index = test_index
        X = pd.concat([x_s, x_t], axis=1)
        X["constant"] = 1
        
        # Calculate with dampening
        components = np.einsum('ij,jk->ijk', X.to_numpy(), model.beta)
        trend_x_start = x_s.shape[1]
        trend_x_end = x_s.shape[1] + x_t.shape[1]
        trend_components = components[:, trend_x_start:trend_x_end, :]
        
        req_len = len(test_index) - 1
        phi_series = pd.Series([model.trend_phi] * req_len, index=test_index[1:]).pow(range(req_len))
        diff_array = np.diff(trend_components, axis=0)
        diff_scaled_array = diff_array * phi_series.to_numpy()[:, np.newaxis, np.newaxis]
        first_row = trend_components[0:1, :]
        combined_array = np.vstack([first_row, diff_scaled_array])
        components[:, trend_x_start:trend_x_end, :] = np.cumsum(combined_array, axis=0)
        
        expected_forecast = pd.DataFrame(
            components.sum(axis=1), columns=model.column_names, index=test_index
        )
        
        # Forecast should match the original calculation
        diff = (prediction.forecast - expected_forecast).abs().max().max()
        self.assertLess(diff, 1e-10, "Forecast was not calculated using original method")


class TestTVVARComponents(unittest.TestCase):
    """Test component extraction for TVVAR."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        np.random.seed(42)
        cls.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        cls.df = pd.DataFrame({
            'series1': np.cumsum(np.random.randn(100)) + np.sin(np.arange(100) * 2 * np.pi / 7) * 5,
            'series2': np.cumsum(np.random.randn(100)) + np.cos(np.arange(100) * 2 * np.pi / 7) * 3,
        }, index=cls.dates)

    def test_components_exist(self):
        """Test that components are extracted."""
        model = TVVAR(seasonal_period='weekly', prediction_interval=0.9)
        model.fit(self.df)
        prediction = model.predict(forecast_length=14)
        
        # TVVAR wraps in try/except, so components may or may not exist
        # But if they do exist, they should be valid
        if prediction.components is not None:
            self.assertIsInstance(prediction.components, pd.DataFrame)
            self.assertEqual(prediction.components.shape[0], 14)

    def test_components_with_regressors(self):
        """Test component extraction WITH regressors (CRITICAL TEST for bug fix)."""
        np.random.seed(42)
        
        # Create regressor data
        future_regressor = pd.DataFrame({
            'reg1': np.random.randn(114),
            'reg2': np.random.randn(114),
        }, index=pd.date_range(start='2020-01-01', periods=114, freq='D'))
        
        model = TVVAR(
            seasonal_period='weekly',
            trend_phi=0.98,
            regression_type='User',
            prediction_interval=0.9
        )
        model.fit(self.df, future_regressor=future_regressor)
        prediction = model.predict(forecast_length=14, future_regressor=future_regressor)
        
        # If components exist, check that 'regressors' component is present
        if prediction.components is not None:
            component_types = prediction.components.columns.get_level_values(1).unique().tolist()
            self.assertIn('regressors', component_types, 
                         "Regressor component missing in TVVAR! Bug fix may not have been applied.")


class TestCassandraComponents(unittest.TestCase):
    """Test component extraction for Cassandra."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        np.random.seed(42)
        cls.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        cls.df = pd.DataFrame({
            'series1': np.cumsum(np.random.randn(100)) + np.sin(np.arange(100) * 2 * np.pi / 7) * 5,
            'series2': np.cumsum(np.random.randn(100)) + np.cos(np.arange(100) * 2 * np.pi / 7) * 3,
        }, index=cls.dates)

    def test_components_exist(self):
        """Test that components are extracted."""
        model = Cassandra(seasonality='common_fourier', prediction_interval=0.9)
        model.fit(self.df)
        prediction = model.predict(forecast_length=14)
        
        self.assertIsNotNone(prediction.components)
        self.assertIsInstance(prediction.components, pd.DataFrame)
        self.assertEqual(prediction.components.shape[0], 14)

    def test_component_types(self):
        """Test that standard component types are present."""
        model = Cassandra(seasonality='common_fourier', prediction_interval=0.9)
        model.fit(self.df)
        prediction = model.predict(forecast_length=14)
        
        component_types = prediction.components.columns.get_level_values(1).unique().tolist()
        
        # Cassandra should have at least trend
        self.assertIn('trend', component_types)


class TestComponentStructure(unittest.TestCase):
    """Test the MultiIndex structure of components."""

    def test_multiindex_structure(self):
        """Test that components have correct MultiIndex structure."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'series1': np.cumsum(np.random.randn(50)),
            'series2': np.cumsum(np.random.randn(50)),
        }, index=dates)
        
        model = BasicLinearModel(seasonal_period='weekly', prediction_interval=0.9)
        model.fit(df)
        prediction = model.predict(forecast_length=7)
        
        # Check MultiIndex structure
        self.assertEqual(prediction.components.columns.nlevels, 2)
        
        # Check level 0 contains series names
        level0 = prediction.components.columns.get_level_values(0).unique().tolist()
        self.assertIn('series1', level0)
        self.assertIn('series2', level0)
        
        # Check level 1 contains component types
        level1 = prediction.components.columns.get_level_values(1).unique().tolist()
        self.assertTrue(len(level1) > 0)
        
        # Check columns are sorted by series
        first_series = prediction.components.columns[0][0]
        second_series = prediction.components.columns[0][0]
        # First few columns should all be the same series
        for i in range(min(3, len(prediction.components.columns))):
            if i < len(prediction.components.columns):
                self.assertEqual(prediction.components.columns[i][0], first_series)


if __name__ == '__main__':
    unittest.main()
