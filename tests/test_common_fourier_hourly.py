"""Common Fourier features preserve intraday time and calendar phase."""

import unittest

import numpy as np
import pandas as pd

from autots.models.basics import BasicLinearModel
from autots.tools.seasonal import date_part, origin_ts


class TestCommonFourierHourly(unittest.TestCase):
    def test_hourly_rows_are_distinct(self):
        index = pd.date_range('2024-01-01', periods=72, freq='h')
        for method in ('common_fourier', 'common_fourier_rw'):
            with self.subTest(method=method):
                features = date_part(index, method=method)
                self.assertEqual(len(features.iloc[:24].drop_duplicates()), 24)
                self.assertEqual(features.shape[1], 52 + (method.endswith('_rw')))
                self.assertEqual(features.columns[0], 'seasonalitycommonfourier_0')
                np.testing.assert_array_equal(
                    features.columns[:52],
                    [f'seasonalitycommonfourier_{n}' for n in range(52)],
                )

    def test_fractional_hour_daily_harmonics_match_clock_time(self):
        index = pd.date_range('2024-01-01 00:15:30', periods=96, freq='30min')
        elapsed_hours = (index - pd.Timestamp(origin_ts)).total_seconds() / 3600
        phase = 2 * np.pi * np.asarray(elapsed_hours) / 24
        expected = np.column_stack(
            [np.cos(phase * harmonic) for harmonic in (1, 2, 3)]
            + [np.sin(phase * harmonic) for harmonic in (1, 2, 3)]
        )
        for method in ('common_fourier', 'common_fourier_rw'):
            with self.subTest(method=method):
                features = date_part(index, method=method)
                np.testing.assert_allclose(
                    features.iloc[:, 20:26], expected, rtol=0, atol=5.1e-7
                )
                np.testing.assert_allclose(
                    features.iloc[:48, 20:26],
                    features.iloc[48:, 20:26],
                    rtol=0,
                    atol=1e-6,
                )

    def test_overlapping_windows_keep_the_same_phase(self):
        index = pd.date_range('2024-01-01 07:30', periods=240, freq='h')
        for method in ('common_fourier', 'common_fourier_rw'):
            with self.subTest(method=method):
                full = date_part(index, method=method)
                subset = date_part(index[37:137], method=method)
                pd.testing.assert_frame_equal(full.iloc[37:137], subset)

    def test_daily_sine_forecast_recovers_intraday_seasonality(self):
        steps = np.arange(24 * 91)
        values = 100 + 30 * np.sin(2 * np.pi * steps / 24)
        df = pd.DataFrame(
            {'series': values[: 24 * 90]},
            index=pd.date_range('2024-01-01', periods=24 * 90, freq='h'),
        )
        model = BasicLinearModel(holiday_countries_used=False).fit(df)
        forecast = model.predict(24).forecast['series'].to_numpy()
        self.assertLess(np.mean(np.abs(forecast - values[-24:])), 1.0)
