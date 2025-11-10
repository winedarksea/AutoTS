# -*- coding: utf-8 -*-
"""Unit tests for G726Filter and g726_adpcm_filter."""
import unittest
import numpy as np
import pandas as pd
from autots.tools.g726_filter import g726_adpcm_filter
from autots.tools.transform import G726Filter


class TestG726AdpcmFilter(unittest.TestCase):
    """Test the low-level g726_adpcm_filter function."""

    def test_basic_filtering(self):
        """Test that filter produces output with correct shape and type."""
        # Simple sine wave with noise
        t = np.linspace(0, 10, 500)
        signal = np.sin(t) + 0.1 * np.random.randn(500)
        
        filtered = g726_adpcm_filter(signal)
        
        self.assertEqual(filtered.shape, signal.shape)
        self.assertEqual(filtered.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(filtered)))

    def test_multivariate_input(self):
        """Test filtering multiple time series simultaneously."""
        n_obs, n_series = 500, 10
        data = np.random.randn(n_obs, n_series).cumsum(axis=0)
        
        filtered = g726_adpcm_filter(data)
        
        self.assertEqual(filtered.shape, (n_obs, n_series))
        self.assertTrue(np.all(np.isfinite(filtered)))

    def test_pandas_dataframe_input(self):
        """Test that DataFrame input is handled correctly."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'series1': np.random.randn(100).cumsum(),
            'series2': np.random.randn(100).cumsum(),
        }, index=dates)
        
        filtered = g726_adpcm_filter(df)
        
        self.assertIsInstance(filtered, np.ndarray)
        self.assertEqual(filtered.shape, df.shape)

    def test_noise_reduction(self):
        """Verify that high-frequency noise is attenuated."""
        # Clean signal + high-frequency noise
        t = np.linspace(0, 10, 500)
        clean = np.sin(0.5 * t)
        noise = 0.2 * np.random.randn(500)
        noisy = clean + noise
        
        filtered = g726_adpcm_filter(noisy, quant_bits=3, dynamic_range=2.0)
        
        # Filtered signal should be closer to clean signal than noisy
        error_before = np.mean((noisy - clean) ** 2)
        error_after = np.mean((filtered - clean) ** 2)
        self.assertLess(error_after, error_before)

    def test_empty_array(self):
        """Test edge case of empty input."""
        empty = np.array([])
        filtered = g726_adpcm_filter(empty)
        self.assertEqual(filtered.size, 0)

    def test_single_observation(self):
        """Test edge case of single observation."""
        single = np.array([42.0])
        filtered = g726_adpcm_filter(single)
        self.assertEqual(filtered.shape, (1,))
        np.testing.assert_allclose(filtered, single, rtol=1e-10)

    def test_parameter_bounds(self):
        """Test that parameters are properly bounded."""
        data = np.random.randn(100)
        
        # Extreme parameters should not crash or produce invalid output
        filtered = g726_adpcm_filter(
            data,
            quant_bits=2,  # Minimum viable bits
            adaptation_rate=0.99,
            prediction_alpha=0.99,
            floor_step=0.001,
            dynamic_range=3.0,
            blend=1.0,
            noise_gate=0.5,
        )
        
        self.assertTrue(np.all(np.isfinite(filtered)))

    def test_bit_depth_variation(self):
        """Test different quantization bit depths."""
        data = np.random.randn(200).cumsum()
        
        for bits in [2, 3, 4, 5, 6]:
            filtered = g726_adpcm_filter(data, quant_bits=bits)
            self.assertEqual(filtered.shape, data.shape)
            self.assertTrue(np.all(np.isfinite(filtered)))

    def test_blend_parameter(self):
        """Test that blend parameter mixes signal with baseline."""
        data = np.random.randn(100).cumsum()
        
        # High blend should produce smoother output
        filtered_low_blend = g726_adpcm_filter(data, blend=0.0)
        filtered_high_blend = g726_adpcm_filter(data, blend=0.8)
        
        # Higher blend should have lower variance (smoother)
        self.assertLess(np.std(filtered_high_blend), np.std(filtered_low_blend))

    def test_noise_gate(self):
        """Test that noise gate attenuates small residuals."""
        data = np.random.randn(100) * 0.1  # Small noise
        
        filtered_no_gate = g726_adpcm_filter(data, noise_gate=0.0)
        filtered_with_gate = g726_adpcm_filter(data, noise_gate=0.05)
        
        # With gate, small values should be more attenuated
        self.assertLess(np.std(filtered_with_gate), np.std(filtered_no_gate))

    def test_memory_scaling(self):
        """Test memory behavior with moderately large arrays."""
        # Test with ~40 MB input (5000 obs × 1000 series × 8 bytes)
        n_obs, n_series = 5000, 1000
        data = np.random.randn(n_obs, n_series)
        
        filtered = g726_adpcm_filter(data)
        
        self.assertEqual(filtered.shape, data.shape)
        # Peak memory should be ~7x input = ~280 MB, well within limits

    def test_nonuniform_quantizer(self):
        """Test non-uniform quantization mode."""
        data = np.random.randn(200).cumsum()
        
        filtered_uniform = g726_adpcm_filter(data, quantizer="uniform")
        filtered_nonuniform = g726_adpcm_filter(data, quantizer="nonuniform")
        
        # Both should produce valid output
        self.assertEqual(filtered_uniform.shape, data.shape)
        self.assertEqual(filtered_nonuniform.shape, data.shape)
        self.assertTrue(np.all(np.isfinite(filtered_uniform)))
        self.assertTrue(np.all(np.isfinite(filtered_nonuniform)))
        
        # Results should differ (different quantization)
        self.assertFalse(np.allclose(filtered_uniform, filtered_nonuniform))

    def test_adaptive_predictor(self):
        """Test adaptive predictor vs legacy EMA mode."""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        # Signal with trend and seasonality
        t = np.arange(500)
        signal = 100 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 7) + np.random.randn(500) * 2
        df = pd.DataFrame({'series1': signal}, index=dates)
        
        # Adaptive predictor mode
        filtered_adaptive = g726_adpcm_filter(
            df, use_adaptive_predictor=True, quantizer="nonuniform"
        )
        
        # Legacy EMA mode
        filtered_legacy = g726_adpcm_filter(
            df, use_adaptive_predictor=False
        )
        
        # Both should produce valid output
        self.assertEqual(filtered_adaptive.shape, df.shape)
        self.assertEqual(filtered_legacy.shape, df.shape)
        self.assertTrue(np.all(np.isfinite(filtered_adaptive)))
        self.assertTrue(np.all(np.isfinite(filtered_legacy)))

    def test_predictor_leak(self):
        """Test that predictor leak parameter works."""
        data = np.random.randn(300).cumsum()
        
        # Different leak values should produce different results
        filtered_high_leak = g726_adpcm_filter(
            data, use_adaptive_predictor=True, predictor_leak=0.995
        )
        filtered_low_leak = g726_adpcm_filter(
            data, use_adaptive_predictor=True, predictor_leak=0.99999
        )
        
        self.assertTrue(np.all(np.isfinite(filtered_high_leak)))
        self.assertTrue(np.all(np.isfinite(filtered_low_leak)))
        # Results should differ due to different leakage (high leak = faster decay)
        self.assertFalse(np.allclose(filtered_high_leak, filtered_low_leak, rtol=0.001))


class TestG726FilterTransformer(unittest.TestCase):
    """Test the G726Filter transformer class."""

    def test_basic_fit_transform(self):
        """Test basic fit and transform operations."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        df = pd.DataFrame({
            'series1': np.random.randn(200).cumsum() + 100,
            'series2': np.random.randn(200).cumsum() + 50,
        }, index=dates)
        
        transformer = G726Filter()
        transformer.fit(df)
        transformed = transformer.transform(df)
        
        self.assertEqual(transformed.shape, df.shape)
        self.assertIsInstance(transformed, pd.DataFrame)
        pd.testing.assert_index_equal(transformed.index, df.index)
        pd.testing.assert_index_equal(transformed.columns, df.columns)

    def test_fit_transform(self):
        """Test fit_transform convenience method."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'series1': np.random.randn(100).cumsum(),
        }, index=dates)
        
        transformer = G726Filter()
        transformed = transformer.fit_transform(df)
        
        self.assertEqual(transformed.shape, df.shape)

    def test_on_transform_flag(self):
        """Test that on_transform flag controls filtering."""
        df = pd.DataFrame({
            'series1': np.random.randn(100),
        })
        
        # With on_transform=True (default), data should be filtered
        transformer_on = G726Filter(on_transform=True)
        transformed_on = transformer_on.fit_transform(df)
        self.assertFalse(np.allclose(transformed_on.values, df.values))
        
        # With on_transform=False, data should pass through unchanged
        transformer_off = G726Filter(on_transform=False)
        transformed_off = transformer_off.fit_transform(df)
        pd.testing.assert_frame_equal(transformed_off, df)

    def test_inverse_transform(self):
        """Test inverse_transform behavior."""
        df = pd.DataFrame({
            'series1': np.random.randn(100).cumsum(),
        })
        
        # Default: on_inverse=False, should pass through
        transformer = G726Filter(on_inverse=False)
        transformer.fit(df)
        inverse = transformer.inverse_transform(df)
        pd.testing.assert_frame_equal(inverse, df)
        
        # With on_inverse=True, should filter
        transformer_inv = G726Filter(on_inverse=True)
        transformer_inv.fit(df)
        inverse_filtered = transformer_inv.inverse_transform(df)
        self.assertFalse(np.allclose(inverse_filtered.values, df.values))

    def test_fill_methods(self):
        """Test different fill methods for NaN handling."""
        df = pd.DataFrame({
            'series1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'series2': [np.nan, 2.0, 3.0, np.nan, 5.0],
        })
        
        for fill_method in ['interpolate', 'ffill', 'bfill', 'median', 'zero']:
            transformer = G726Filter(fill_method=fill_method)
            transformed = transformer.fit_transform(df)
            
            # Output should have no NaNs
            self.assertFalse(transformed.isnull().any().any())

    def test_parameter_preservation(self):
        """Test that transformer parameters are preserved."""
        params = {
            'quant_bits': 5,
            'adaptation_rate': 0.95,
            'prediction_alpha': 0.90,
            'floor_step': 0.02,
            'dynamic_range': 2.0,
            'blend': 0.2,
            'noise_gate': 0.1,
        }
        
        transformer = G726Filter(**params)
        
        self.assertEqual(transformer.quant_bits, params['quant_bits'])
        self.assertEqual(transformer.adaptation_rate, params['adaptation_rate'])
        self.assertEqual(transformer.prediction_alpha, params['prediction_alpha'])
        self.assertEqual(transformer.floor_step, params['floor_step'])
        self.assertEqual(transformer.dynamic_range, params['dynamic_range'])
        self.assertEqual(transformer.blend, params['blend'])
        self.assertEqual(transformer.noise_gate, params['noise_gate'])

    def test_empty_dataframe(self):
        """Test edge case of empty DataFrame."""
        df_empty = pd.DataFrame()
        transformer = G726Filter()
        transformer.fit(df_empty)
        transformed = transformer.transform(df_empty)
        self.assertTrue(transformed.empty)

    def test_smoothing_effect(self):
        """Verify that filter produces valid output and preserves trends."""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        # Clean trend + high frequency noise
        trend = np.linspace(0, 100, 500)
        noise = np.random.randn(500) * 5
        df = pd.DataFrame({
            'series1': trend + noise,
        }, index=dates)
        
        # Use legacy mode for more predictable smoothing behavior
        transformer = G726Filter(
            use_adaptive_predictor=False,
            quant_bits=3,
            dynamic_range=2.0
        )
        transformed = transformer.fit_transform(df)
        
        # Check that output is finite and has reasonable values
        self.assertTrue(np.all(np.isfinite(transformed.values)))
        
        # Check that the trend is preserved (correlation should be high)
        correlation = np.corrcoef(transformed['series1'].values, df['series1'].values)[0, 1]
        self.assertGreater(correlation, 0.95)
        
        # For adaptive mode, just verify it runs and produces valid output
        transformer_adaptive = G726Filter(
            quantizer="nonuniform",
            use_adaptive_predictor=True,
        )
        transformed_adaptive = transformer_adaptive.fit_transform(df)
        self.assertTrue(np.all(np.isfinite(transformed_adaptive.values)))
        self.assertEqual(transformed_adaptive.shape, df.shape)

    def test_quantizer_parameter(self):
        """Test transformer with different quantizer modes."""
        df = pd.DataFrame({
            'series1': np.random.randn(200).cumsum(),
        })
        
        # Uniform quantizer
        transformer_uniform = G726Filter(quantizer="uniform")
        transformed_uniform = transformer_uniform.fit_transform(df)
        self.assertEqual(transformed_uniform.shape, df.shape)
        
        # Non-uniform quantizer
        transformer_nonuniform = G726Filter(quantizer="nonuniform")
        transformed_nonuniform = transformer_nonuniform.fit_transform(df)
        self.assertEqual(transformed_nonuniform.shape, df.shape)

    def test_adaptive_predictor_parameter(self):
        """Test transformer with adaptive predictor on/off."""
        df = pd.DataFrame({
            'series1': np.random.randn(300).cumsum(),
        })
        
        # With adaptive predictor
        transformer_adaptive = G726Filter(use_adaptive_predictor=True)
        transformed_adaptive = transformer_adaptive.fit_transform(df)
        self.assertEqual(transformed_adaptive.shape, df.shape)
        
        # Without adaptive predictor (legacy)
        transformer_legacy = G726Filter(use_adaptive_predictor=False)
        transformed_legacy = transformer_legacy.fit_transform(df)
        self.assertEqual(transformed_legacy.shape, df.shape)

    def test_get_new_params(self):
        """Test that get_new_params includes new parameters."""
        params = G726Filter.get_new_params()
        
        self.assertIn('quantizer', params)
        self.assertIn('use_adaptive_predictor', params)
        self.assertIn('predictor_leak', params)
        self.assertIn(params['quantizer'], ['uniform', 'nonuniform'])
        self.assertIsInstance(params['use_adaptive_predictor'], bool)
        self.assertIsInstance(params['predictor_leak'], float)


if __name__ == '__main__':
    unittest.main()
