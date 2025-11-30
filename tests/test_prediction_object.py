# -*- coding: utf-8 -*-
"""Test PredictionObject functionality."""
import unittest
import numpy as np
import pandas as pd
from autots.models.base import PredictionObject


class TestPredictionObject(unittest.TestCase):
    """Test suite for PredictionObject class."""

    def setUp(self):
        """Set up test fixtures."""
        self.forecast_length = 5
        self.forecast_index = pd.date_range(
            '2024-01-01', periods=self.forecast_length, freq='D'
        )
        self.forecast_columns = pd.Index(['series_1', 'series_2', 'series_3'])

        # Create sample forecasts
        np.random.seed(42)
        self.forecast_data = pd.DataFrame(
            np.random.randn(self.forecast_length, 3) * 10 + 100,
            index=self.forecast_index,
            columns=self.forecast_columns,
        )
        self.upper_forecast_data = self.forecast_data + 5
        self.lower_forecast_data = self.forecast_data - 5

    def test_copy_basic(self):
        """Test that copy() creates a separate object with same values."""
        original = PredictionObject(
            model_name='TestModel',
            forecast_length=self.forecast_length,
            forecast_index=self.forecast_index,
            forecast_columns=self.forecast_columns,
            forecast=self.forecast_data,
            upper_forecast=self.upper_forecast_data,
            lower_forecast=self.lower_forecast_data,
            prediction_interval=0.9,
            model_parameters={'param1': 'value1'},
            transformation_parameters={'transform1': 'config1'},
        )

        copy_obj = original.copy()

        # Verify values match
        self.assertEqual(original.model_name, copy_obj.model_name)
        self.assertEqual(original.forecast_length, copy_obj.forecast_length)
        self.assertEqual(original.prediction_interval, copy_obj.prediction_interval)
        self.assertTrue(original.forecast.equals(copy_obj.forecast))
        self.assertTrue(original.upper_forecast.equals(copy_obj.upper_forecast))
        self.assertTrue(original.lower_forecast.equals(copy_obj.lower_forecast))

    def test_copy_memory_independence(self):
        """Test that copy() creates separate objects in memory."""
        original = PredictionObject(
            model_name='TestModel',
            forecast_length=self.forecast_length,
            forecast_index=self.forecast_index,
            forecast_columns=self.forecast_columns,
            forecast=self.forecast_data,
            upper_forecast=self.upper_forecast_data,
            lower_forecast=self.lower_forecast_data,
            prediction_interval=0.9,
            model_parameters={'param1': 'value1'},
            transformation_parameters={'transform1': 'config1'},
        )

        copy_obj = original.copy()

        # Verify they are separate in memory
        self.assertIsNot(original.forecast, copy_obj.forecast)
        self.assertIsNot(original.upper_forecast, copy_obj.upper_forecast)
        self.assertIsNot(original.lower_forecast, copy_obj.lower_forecast)
        self.assertIsNot(original.model_parameters, copy_obj.model_parameters)
        self.assertIsNot(
            original.transformation_parameters, copy_obj.transformation_parameters
        )

    def test_copy_modification_independence(self):
        """Test that modifying copy doesn't affect original."""
        original = PredictionObject(
            model_name='TestModel',
            forecast_length=self.forecast_length,
            forecast_index=self.forecast_index,
            forecast_columns=self.forecast_columns,
            forecast=self.forecast_data.copy(),
            upper_forecast=self.upper_forecast_data.copy(),
            lower_forecast=self.lower_forecast_data.copy(),
            prediction_interval=0.9,
            model_parameters={'param1': 'value1', 'nested': {'param2': 'value2'}},
            transformation_parameters={'transform1': 'config1'},
        )

        original_forecast_value = original.forecast.iloc[0, 0]
        original_param_value = original.model_parameters['param1']
        original_nested_value = original.model_parameters['nested']['param2']

        copy_obj = original.copy()

        # Modify the copy
        copy_obj.forecast.iloc[0, 0] = 999.99
        copy_obj.model_parameters['param1'] = 'modified_value'
        copy_obj.model_parameters['nested']['param2'] = 'modified_nested'

        # Verify original is unchanged
        self.assertEqual(original.forecast.iloc[0, 0], original_forecast_value)
        self.assertEqual(original.model_parameters['param1'], original_param_value)
        self.assertEqual(
            original.model_parameters['nested']['param2'], original_nested_value
        )

        # Verify copy is changed
        self.assertEqual(copy_obj.forecast.iloc[0, 0], 999.99)
        self.assertEqual(copy_obj.model_parameters['param1'], 'modified_value')
        self.assertEqual(
            copy_obj.model_parameters['nested']['param2'], 'modified_nested'
        )

    def test_copy_with_components(self):
        """Test that copy() properly handles components."""
        original = PredictionObject(
            model_name='TestModel',
            forecast_length=self.forecast_length,
            forecast_index=self.forecast_index,
            forecast_columns=self.forecast_columns,
            forecast=self.forecast_data,
            upper_forecast=self.upper_forecast_data,
            lower_forecast=self.lower_forecast_data,
            prediction_interval=0.9,
        )

        # Add components
        components_data = pd.DataFrame(
            np.random.randn(self.forecast_length, 6),
            index=self.forecast_index,
            columns=pd.MultiIndex.from_product(
                [self.forecast_columns, ['trend', 'seasonal']]
            ),
        )
        original.components = components_data

        copy_obj = original.copy()

        # Verify components are equal but separate
        self.assertTrue(original.components.equals(copy_obj.components))
        self.assertIsNot(original.components, copy_obj.components)

        # Modify copy's components
        original_comp_value = original.components.iloc[0, 0]
        copy_obj.components.iloc[0, 0] = 888.88

        # Verify original components unchanged
        self.assertEqual(original.components.iloc[0, 0], original_comp_value)
        self.assertEqual(copy_obj.components.iloc[0, 0], 888.88)

    def test_copy_with_metrics(self):
        """Test that copy() properly handles evaluation metrics."""
        original = PredictionObject(
            model_name='TestModel',
            forecast_length=self.forecast_length,
            forecast_index=self.forecast_index,
            forecast_columns=self.forecast_columns,
            forecast=self.forecast_data,
            upper_forecast=self.upper_forecast_data,
            lower_forecast=self.lower_forecast_data,
            prediction_interval=0.9,
        )

        # Add metrics (simulating what evaluate() would do)
        original.per_series_metrics = pd.DataFrame(
            {
                'series_1': {'smape': 0.1, 'mae': 2.0},
                'series_2': {'smape': 0.15, 'mae': 2.5},
                'series_3': {'smape': 0.12, 'mae': 2.2},
            }
        )
        original.avg_metrics = pd.Series({'smape': 0.123, 'mae': 2.233})
        original.full_mae_error = np.random.randn(self.forecast_length, 3)

        copy_obj = original.copy()

        # Verify metrics are equal but separate
        self.assertTrue(original.per_series_metrics.equals(copy_obj.per_series_metrics))
        self.assertTrue(original.avg_metrics.equals(copy_obj.avg_metrics))
        self.assertTrue(
            np.array_equal(original.full_mae_error, copy_obj.full_mae_error)
        )

        self.assertIsNot(original.per_series_metrics, copy_obj.per_series_metrics)
        self.assertIsNot(original.avg_metrics, copy_obj.avg_metrics)
        self.assertIsNot(original.full_mae_error, copy_obj.full_mae_error)

    def test_copy_empty_object(self):
        """Test that copy() works on an empty/uninitiated PredictionObject."""
        original = PredictionObject()
        copy_obj = original.copy()

        # Should not raise an error and should have matching (nan) values
        self.assertEqual(original.model_name, copy_obj.model_name)
        self.assertEqual(original.forecast_length, copy_obj.forecast_length)


if __name__ == '__main__':
    unittest.main()
