# -*- coding: utf-8 -*-
"""Test constraint."""
import unittest
import warnings
import numpy as np
import pandas as pd
from autots import load_daily, ModelPrediction


class TestConstraint(unittest.TestCase):

    def test_constraint(self):
        df = load_daily(long=False)
        if "USW00014771_PRCP" in df.columns:
            # too close to zero, causes one test to fail
            df["USW00014771_PRCP"] = df["USW00014771_PRCP"] + 1
        forecast_length = 30
        constraint_types = {
            "empty": {
                "constraints": None,
            },
            "old_style": {
                "constraint_method": "quantile",
                "constraint_regularization": 0.99,
                "upper_constraint": 0.5,
                "lower_constraint": 0.1,
                "bounds": True,
            },
            "quantile": {
                "constraints": [{
                    "constraint_method": "quantile",
                    "constraint_value": 0.98,
                    "constraint_direction": "upper",
                    "constraint_regularization": 1.0,
                    "bounds": False,
                },]
            },
            "last_value": {
                "constraints": [{
                        "constraint_method": "last_window",
                        "constraint_value": 0.0,
                        "constraint_direction": "upper",
                        "constraint_regularization": 1.0,
                        "bounds": True,
                    },
                    {
                        "constraint_method": "last_window",
                        "constraint_value": 0.0,
                        "constraint_direction": "lower",
                        "constraint_regularization": 1.0,
                        "bounds": True,
                    },
                ]
            },
            "example": {"constraints": [
                {  # don't exceed historic max
                    "constraint_method": "quantile",
                    "constraint_value": 1.0,
                    "constraint_direction": "upper",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
                {  # don't exceed 2% growth by end of forecast horizon
                    "constraint_method": "slope",
                    "constraint_value": {"slope": 0.02, "window": 10, "window_agg": "max", "threshold": 0.01},
                    "constraint_direction": "upper",
                    "constraint_regularization": 0.9,
                    "bounds": False,
                },
                {  # don't go below the last 10 values - 10%
                    "constraint_method": "last_window",
                    "constraint_value": {"window": 10, "threshold": -0.1},
                    "constraint_direction": "lower",
                    "constraint_regularization": 1.0,
                    "bounds": False,
                },
                {  # don't go below zero
                    "constraint_method": "absolute",
                    "constraint_value": 0,  # can also be an array or Series
                    "constraint_direction": "lower",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
                {  # don't go below historic min  - 1 st dev
                    "constraint_method": "stdev_min",
                    "constraint_value": 1.0,
                    "constraint_direction": "lower",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
                {  # don't go above historic mean  + 3 st devs, soft limit
                    "constraint_method": "stdev",
                    "constraint_value": 3.0,
                    "constraint_direction": "upper",
                    "constraint_regularization": 0.5,
                    "bounds": True,
                },
                {  # use a log curve shaped by the historic min/max growth rate to limit
                    "constraint_method": "historic_growth",
                    "constraint_value": 1.0,
                    "constraint_direction": "upper",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
                {  # use a log curve shaped by the historic min/max growth rate to limit
                    "constraint_method": "historic_growth",
                    "constraint_value": {'threshold': 2.0, 'window': 360},
                    "constraint_direction": "upper",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
                {  # like slope but steps
                    'constraint_method': 'historic_diff',
                    'constraint_direction': 'upper',
                    'constraint_regularization': 1.0,
                    'constraint_value': 1.0,
                 },
            ]},
            "dampening": {
                "constraints": [{
                    "constraint_method": "dampening",
                    "constraint_value": 0.98,
                    "bounds": True,
                },]
            },
        }
        for key, constraint in constraint_types.items():
            with self.subTest(i=key):
                model = ModelPrediction(
                    forecast_length=forecast_length,
                    transformation_dict={
                        "fillna": "median",
                        "transformations": {"0": "SinTrend", "1": "QuantileTransformer", "2": "bkfilter"},
                        "transformation_params": {"0": {}, "1": {"output_distribution": "uniform", "n_quantiles": 1000}, "2": {}}
                    },
                    model_str="SeasonalityMotif",
                    parameter_dict={
                        "window": 7, "point_method": "midhinge",
                        "distance_metric": "canberra", "k": 10,
                        "datepart_method": "common_fourier",
                    },
                    no_negatives=True,
                )
                prediction = model.fit_predict(df, forecast_length=forecast_length)
                # apply an artificially low value
                prediction.forecast.iloc[0, 0] = -10
                prediction.forecast.iloc[0, -1] = df.iloc[:, -1].max() * 1.1
                prediction.plot(df, df.columns[-1])
                prediction.plot(df, df.columns[0])

                prediction.apply_constraints(
                    df_train=df,
                    **constraint
                )
                prediction.plot(df, df.columns[-1])
                prediction.plot(df, df.columns[0])
                # assuming all history was positive as example data currently is
                if key in ["empty", "dampening"]:
                    self.assertTrue(prediction.forecast.min().min() == -10)
                else:
                    self.assertTrue((prediction.forecast.sum() > 0).all())

                if key in ["old_style", "quantile"]:
                    pred_max = prediction.forecast.iloc[:, -1].max()
                    hist_max = df.iloc[:, -1].max()
                    print(pred_max)
                    print(hist_max)
                    self.assertTrue(pred_max <= hist_max)
                if key in ["last_value"]:
                    self.assertTrue(prediction.forecast.iloc[0, :].max() == df.iloc[-1, :].max())
                # test for nulls
                self.assertTrue(prediction.forecast.isnull().sum().sum() == 0)

    def test_adjustments(self):
        """Test apply_adjustments functionality."""
        df = load_daily(long=False)
        forecast_length = 30
        
        # Create a basic model and prediction
        model = ModelPrediction(
            forecast_length=forecast_length,
            transformation_dict={
                "fillna": "median",
                "transformations": {"0": "SinTrend"},
                "transformation_params": {"0": {}}
            },
            model_str="SeasonalityMotif",
            parameter_dict={
                "window": 7, "point_method": "midhinge",
                "distance_metric": "canberra", "k": 10,
                "datepart_method": "common_fourier",
            },
        )
        prediction = model.fit_predict(df, forecast_length=forecast_length)
        
        # Store original values for comparison
        orig_forecast = prediction.forecast.copy()
        orig_lower = prediction.lower_forecast.copy()
        orig_upper = prediction.upper_forecast.copy()
        
        # Test 1: Percentage adjustment
        with self.subTest(i="percentage_constant"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            pred.lower_forecast = orig_lower.copy()
            pred.upper_forecast = orig_upper.copy()
            
            pred.apply_adjustments(
                adjustments=[{
                    "adjustment_method": "percentage",
                    "value": 0.1,  # 10% increase
                    "apply_bounds": True,
                }],
                df_train=df,
            )
            # Check that values increased by approximately 10%
            expected = orig_forecast * 1.1
            pd.testing.assert_frame_equal(pred.forecast, expected, rtol=1e-10)
            expected_lower = orig_lower * 1.1
            pd.testing.assert_frame_equal(pred.lower_forecast, expected_lower, rtol=1e-10)
        
        # Test 2: Percentage with gradient
        with self.subTest(i="percentage_gradient"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            pred.lower_forecast = orig_lower.copy()
            pred.upper_forecast = orig_upper.copy()
            
            pred.apply_adjustments(
                adjustments=[{
                    "adjustment_method": "percentage",
                    "start_value": 0.0,
                    "end_value": 0.2,  # 0% to 20% increase over forecast period
                    "apply_bounds": False,
                }],
                df_train=df,
            )
            # First value should be unchanged, last should be ~20% higher
            self.assertTrue((pred.forecast.iloc[0, :] - orig_forecast.iloc[0, :]).abs().max() < 1e-10)
            expected_last = orig_forecast.iloc[-1, :] * 1.2
            pd.testing.assert_series_equal(pred.forecast.iloc[-1, :], expected_last, rtol=1e-10)
            # Bounds should be unchanged
            pd.testing.assert_frame_equal(pred.lower_forecast, orig_lower)
        
        # Test 3: Additive adjustment
        with self.subTest(i="additive"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            pred.lower_forecast = orig_lower.copy()
            pred.upper_forecast = orig_upper.copy()
            
            adjustment_value = 5.0
            pred.apply_adjustments(
                adjustments=[{
                    "adjustment_method": "additive",
                    "value": adjustment_value,
                    "apply_bounds": True,
                }],
                df_train=df,
            )
            expected = orig_forecast + adjustment_value
            pd.testing.assert_frame_equal(pred.forecast, expected)
            expected_lower = orig_lower + adjustment_value
            pd.testing.assert_frame_equal(pred.lower_forecast, expected_lower)
        
        # Test 4: Smoothing adjustment
        with self.subTest(i="smoothing"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            pred.lower_forecast = orig_lower.copy()
            pred.upper_forecast = orig_upper.copy()
            
            pred.apply_adjustments(
                adjustments=[{
                    "adjustment_method": "smoothing",
                    "span": 5,
                    "apply_bounds": True,
                }],
                df_train=df,
            )
            # Check that forecast was smoothed (values should differ)
            self.assertFalse(pred.forecast.equals(orig_forecast))
            # Check no nulls introduced
            self.assertEqual(pred.forecast.isnull().sum().sum(), 0)
        
        # Test 5: AlignLastValue adjustment
        with self.subTest(i="align_last_value"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            pred.lower_forecast = orig_lower.copy()
            pred.upper_forecast = orig_upper.copy()
            
            pred.apply_adjustments(
                adjustments=[{
                    "adjustment_method": "align_last_value",
                    "apply_bounds": True,
                    "parameters": {
                        "rows": 1,
                        "lag": 1,
                        "method": "additive",
                        "strength": 1.0,
                    }
                }],
                df_train=df,
            )
            # First forecast value should align with last training value
            for col in df.columns:
                last_train = df[col].iloc[-1]
                first_forecast = pred.forecast[col].iloc[0]
                # Should be aligned (allowing small numerical differences)
                self.assertAlmostEqual(first_forecast, last_train, delta=abs(last_train * 0.01) + 0.1)
        
        # Test 6: Multiple adjustments in sequence
        with self.subTest(i="multiple_adjustments"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            pred.lower_forecast = orig_lower.copy()
            pred.upper_forecast = orig_upper.copy()
            
            pred.apply_adjustments(
                adjustments=[
                    {
                        "adjustment_method": "percentage",
                        "value": 0.1,
                    },
                    {
                        "adjustment_method": "additive",
                        "value": 2.0,
                    },
                ],
                df_train=df,
            )
            # Should apply both: (orig * 1.1) + 2.0
            expected = (orig_forecast * 1.1) + 2.0
            pd.testing.assert_frame_equal(pred.forecast, expected, rtol=1e-10)
        
        # Test 7: Column-specific adjustment
        with self.subTest(i="column_specific"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            pred.lower_forecast = orig_lower.copy()
            pred.upper_forecast = orig_upper.copy()
            
            test_col = df.columns[0]
            pred.apply_adjustments(
                adjustments=[{
                    "adjustment_method": "additive",
                    "value": 10.0,
                    "columns": [test_col],
                }],
                df_train=df,
            )
            # Only specified column should change
            expected = orig_forecast.copy()
            expected[test_col] = orig_forecast[test_col] + 10.0
            pd.testing.assert_frame_equal(pred.forecast, expected)
        
        # Test 8: Date range filtering
        with self.subTest(i="date_range"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            pred.lower_forecast = orig_lower.copy()
            pred.upper_forecast = orig_upper.copy()
            
            # Apply adjustment only to middle portion
            mid_idx = len(pred.forecast) // 2
            start_date = pred.forecast.index[mid_idx]
            
            pred.apply_adjustments(
                adjustments=[{
                    "adjustment_method": "additive",
                    "value": 5.0,
                    "start_date": start_date,
                }],
                df_train=df,
            )
            # First half should be unchanged
            pd.testing.assert_frame_equal(
                pred.forecast.iloc[:mid_idx, :], 
                orig_forecast.iloc[:mid_idx, :]
            )
            # Second half should be adjusted
            expected_second_half = orig_forecast.iloc[mid_idx:, :] + 5.0
            pd.testing.assert_frame_equal(
                pred.forecast.iloc[mid_idx:, :],
                expected_second_half
            )
        
        # Test 9: Empty adjustments
        with self.subTest(i="empty_adjustments"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            pred.lower_forecast = orig_lower.copy()
            pred.upper_forecast = orig_upper.copy()
            
            # Should not change anything
            pred.apply_adjustments(adjustments=None, df_train=df)
            pd.testing.assert_frame_equal(pred.forecast, orig_forecast)
            
            pred.apply_adjustments(adjustments=[], df_train=df)
            pd.testing.assert_frame_equal(pred.forecast, orig_forecast)
        
        # Test 10: Invalid method handling
        with self.subTest(i="invalid_method"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            
            # Should warn and not crash
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pred.apply_adjustments(
                    adjustments=[{
                        "adjustment_method": "nonexistent_method",
                        "value": 1.0,
                    }],
                    df_train=df,
                )
                self.assertTrue(len(w) > 0)
                self.assertTrue("Unknown adjustment_method" in str(w[-1].message))
            # Forecast should be unchanged
            pd.testing.assert_frame_equal(pred.forecast, orig_forecast)
        
        # Test 11: Percentage with gradient over date range
        with self.subTest(i="percentage_gradient_with_date_range"):
            pred = prediction
            pred.forecast = orig_forecast.copy()
            pred.lower_forecast = orig_lower.copy()
            pred.upper_forecast = orig_upper.copy()
            
            # Apply a gradient from 0% to 10% increase, but only to middle portion
            quarter_idx = len(pred.forecast) // 4
            three_quarter_idx = 3 * len(pred.forecast) // 4
            start_date = pred.forecast.index[quarter_idx]
            end_date = pred.forecast.index[three_quarter_idx]
            
            pred.apply_adjustments(
                adjustments=[{
                    "adjustment_method": "percentage",
                    "start_value": 0.0,   # 0% increase at start of range
                    "end_value": 0.1,      # 10% increase at end of range
                    "start_date": start_date,
                    "end_date": end_date,
                    "apply_bounds": True,
                }],
                df_train=df,
            )
            
            # Before start_date should be unchanged
            pd.testing.assert_frame_equal(
                pred.forecast.iloc[:quarter_idx, :],
                orig_forecast.iloc[:quarter_idx, :]
            )
            
            # After end_date should be unchanged
            if three_quarter_idx + 1 < len(pred.forecast):
                pd.testing.assert_frame_equal(
                    pred.forecast.iloc[three_quarter_idx + 1:, :],
                    orig_forecast.iloc[three_quarter_idx + 1:, :]
                )
            
            # At start of range should be nearly unchanged (0% adjustment)
            expected_start = orig_forecast.iloc[quarter_idx, :] * 1.0
            pd.testing.assert_series_equal(
                pred.forecast.iloc[quarter_idx, :],
                expected_start,
                rtol=1e-10
            )
            
            # At end of range should be increased by ~10%
            expected_end = orig_forecast.iloc[three_quarter_idx, :] * 1.1
            pd.testing.assert_series_equal(
                pred.forecast.iloc[three_quarter_idx, :],
                expected_end,
                rtol=1e-10
            )
            
            # Check bounds were also adjusted
            expected_lower_end = orig_lower.iloc[three_quarter_idx, :] * 1.1
            pd.testing.assert_series_equal(
                pred.lower_forecast.iloc[three_quarter_idx, :],
                expected_lower_end,
                rtol=1e-10
            )

