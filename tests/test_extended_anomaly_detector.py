# -*- coding: utf-8 -*-
"""Targeted tests for ExtendedAnomalyDetector multi-day anomaly detection."""

import unittest

import numpy as np
import pandas as pd

from autots.evaluator.feature_detector.extended_anomaly import ExtendedAnomalyDetector


class TestExtendedAnomalyDetector(unittest.TestCase):
    """Validate detection of 7-21 day transient anomalies."""

    @staticmethod
    def _overlap_days(start_a, end_a, start_b, end_b):
        overlap_start = max(pd.Timestamp(start_a), pd.Timestamp(start_b))
        overlap_end = min(pd.Timestamp(end_a), pd.Timestamp(end_b))
        days = (overlap_end - overlap_start).days + 1
        return max(0, days)

    def test_detects_14_day_transient_change(self):
        rng = np.random.RandomState(202)
        dates = pd.date_range("2021-01-01", periods=240, freq="D")
        values = rng.normal(0.0, 0.22, size=len(dates))

        true_start = dates[100]
        true_end = dates[113]  # 14 days
        values[100:114] += 2.4

        residual_df = pd.DataFrame({"series_0": values}, index=dates)
        detector = ExtendedAnomalyDetector(
            sustained_window=7,
            sustained_baseline=45,
            sustained_threshold=2.0,
            cusum_h=3.5,
            min_segment_run=2,
            sustained_hysteresis=0.7,
            segment_max_gap=1,
            merge_distance_days=2,
        )
        detector.fit(residual_df, pass1_records={"series_0": []})
        events = detector.get_events("series_0")

        self.assertTrue(events, "Expected at least one extended anomaly event.")
        has_target = False
        for event in events:
            overlap = self._overlap_days(
                event["date"], event["end_date"], true_start, true_end
            )
            duration = int(event.get("duration", 1) or 1)
            if (
                overlap >= 7
                and 7 <= duration <= 35
                and event.get("type") in {"transient_change", "noisy_burst"}
            ):
                has_target = True
                break
        self.assertTrue(
            has_target,
            "Expected a detected event with substantial overlap on 14-day transient.",
        )

    def test_21_day_transient_with_1_day_dip_remains_contiguous(self):
        rng = np.random.RandomState(303)
        dates = pd.date_range("2021-01-01", periods=260, freq="D")
        values = rng.normal(0.0, 0.20, size=len(dates))

        true_start = dates[120]
        true_end = dates[140]  # 21 days
        values[120:141] += 2.6
        # One-day dip in the middle should not split the run.
        values[130] -= 2.2

        residual_df = pd.DataFrame({"series_0": values}, index=dates)
        detector = ExtendedAnomalyDetector(
            sustained_window=7,
            sustained_baseline=45,
            sustained_threshold=2.0,
            cusum_h=3.5,
            min_segment_run=2,
            sustained_hysteresis=0.7,
            segment_max_gap=1,
            merge_distance_days=2,
        )
        detector.fit(residual_df, pass1_records={"series_0": []})
        events = detector.get_events("series_0")

        self.assertTrue(events, "Expected at least one extended anomaly event.")
        best_overlap = 0
        best_duration = 0
        for event in events:
            duration = int(event.get("duration", 1) or 1)
            if duration > 45:
                continue
            overlap = self._overlap_days(
                event["date"], event["end_date"], true_start, true_end
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_duration = duration

        self.assertGreaterEqual(
            best_overlap,
            10,
            "Expected contiguous detection to cover much of the 21-day transient.",
        )
        self.assertGreaterEqual(
            best_duration,
            10,
            "Expected merged/bridged run to have extended duration, not a point spike.",
        )

    def test_max_anomaly_cap_keeps_strongest_event(self):
        dates = pd.date_range("2021-01-01", periods=120, freq="D")
        values = np.zeros(len(dates), dtype=float)
        residual_df = pd.DataFrame({"series_0": values}, index=dates)

        # Two pass-1 proposals; later event is much stronger.
        pass1 = {
            "series_0": [
                {
                    "date": dates[10],
                    "magnitude": 1.0,
                    "score": 1.0,
                    "type": "point_outlier",
                },
                {
                    "date": dates[100],
                    "magnitude": 6.0,
                    "score": 9.0,
                    "type": "point_outlier",
                },
            ]
        }

        detector = ExtendedAnomalyDetector(
            max_anomalies_per_series=1,
            sustained_threshold=999.0,
            cusum_h=999.0,
            slope_reversion_cumsum_threshold=999.0,
        )
        detector.fit(residual_df, pass1_records=pass1)
        events = detector.get_events("series_0")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["date"], dates[100])
        self.assertGreaterEqual(float(events[0].get("score", 0.0)), 9.0)


if __name__ == "__main__":
    unittest.main()
