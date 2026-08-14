# -*- coding: utf-8 -*-
"""Targeted tests for ExtendedAnomalyDetector multi-day anomaly detection."""

import unittest

import numpy as np
import pandas as pd

from autots.evaluator.feature_detector.extended_anomaly import (
    ExtendedAnomalyDetector,
    ExtendedAnomalyMixin,
)
from autots.tools.anomaly_utils import anomaly_scores_to_strength
from autots.tools.transform import AnomalyRemoval


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

        # Two pass-1 proposals; later event is much stronger. Scores are
        # p-values as the real rolling_zscore pipeline produces (smaller =
        # stronger), so only `strength` orders these correctly.
        pass1 = {
            "series_0": [
                {
                    "date": dates[10],
                    "magnitude": 1.0,
                    "score": 0.04,
                    "strength": 1.1,
                    "type": "point_outlier",
                },
                {
                    "date": dates[100],
                    "magnitude": 6.0,
                    "score": 1e-9,
                    "strength": 5.4,
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
        self.assertGreaterEqual(float(events[0].get("strength", 0.0)), 5.0)

    def test_cap_falls_back_to_magnitude_without_strength(self):
        """Records fed in directly by a caller may carry no strength field."""
        dates = pd.date_range("2021-01-01", periods=120, freq="D")
        residual_df = pd.DataFrame(
            {"series_0": np.zeros(len(dates), dtype=float)}, index=dates
        )
        pass1 = {
            "series_0": [
                {
                    "date": dates[10],
                    "magnitude": 1.0,
                    "score": 0.04,
                    "type": "point_outlier",
                },
                {
                    "date": dates[100],
                    "magnitude": 6.0,
                    "score": 1e-9,
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


class TestAnomalyCapRanking(unittest.TestCase):
    """The cap must keep the strongest events, not the weakest."""

    @staticmethod
    def _ramped_spike_series(n_spikes=30, n=1200):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        values = rng.normal(0, 1, n)
        positions = np.linspace(50, n - 50, n_spikes).astype(int)
        amplitudes = np.linspace(5, 40, n_spikes)
        for pos, amp in zip(positions, amplitudes):
            values[pos] += amp
        return pd.DataFrame({"s": values}, index=idx), idx[positions], amplitudes

    def _pass1_via_anomaly_removal(self, df):
        """Build pass-1 records exactly as the feature detector does."""
        detector = AnomalyRemoval(
            method="rolling_zscore",
            output="multivariate",
            method_params={
                "distribution": "norm",
                "alpha": 0.05,
                "rolling_periods": 200,
                "center": True,
            },
        )
        detector.fit(df)
        strengths = anomaly_scores_to_strength(
            detector.scores,
            detector.method,
            detector.method_params,
            detector.anomalies,
        )
        mask = detector.anomalies["s"] == -1
        return [
            {
                "date": date,
                "magnitude": float(df.at[date, "s"]),
                "score": float(detector.scores.loc[date, "s"]),
                "strength": float(strengths.loc[date, "s"]),
                "type": "point_outlier",
            }
            for date in df.index[mask]
        ]

    def test_cap_keeps_largest_spikes_not_smallest(self):
        """Regression: p-value scores previously inverted the cap ordering."""
        df, _, amplitudes = self._ramped_spike_series()
        records = self._pass1_via_anomaly_removal(df)
        self.assertGreater(len(records), 20)

        detector = ExtendedAnomalyDetector(max_anomalies_per_series=10)
        detector.fit(df, pass1_records={"s": records})
        events = detector.get_events("s")

        kept = [e["magnitude"] for e in events if e["type"] == "point_outlier"]
        self.assertTrue(kept, "expected point spikes to survive the cap")
        # The largest spike in the series must be among the survivors, and the
        # survivors must not all be tiny early spikes.
        self.assertGreater(max(kept), 0.5 * float(amplitudes.max()))

    def test_point_events_keep_reserved_share_of_cap(self):
        """A drifty series must not crowd out every point spike."""
        rng = np.random.default_rng(7)
        n = 900
        idx = pd.date_range("2022-01-01", periods=n, freq="D")
        # Repeated sustained level changes generate many pass-2 candidates
        drift = np.zeros(n)
        for start in range(60, n - 60, 70):
            drift[start : start + 35] += 6.0
        values = rng.normal(0, 1, n) + drift
        spike_positions = [15, 35, 640, 660, 680, 700, 720, 740]
        for offset, pos in enumerate(spike_positions):
            values[pos] += 30.0 + offset
        df = pd.DataFrame({"s": values}, index=idx)

        records = [
            {
                "date": idx[pos],
                "magnitude": float(values[pos]),
                "score": 1e-9,
                "strength": 6.0 + offset,
                "type": "point_outlier",
            }
            for offset, pos in enumerate(spike_positions)
        ]

        cap = 10
        detector = ExtendedAnomalyDetector(
            max_anomalies_per_series=cap, min_point_anomaly_share=0.4
        )
        detector.fit(df, pass1_records={"s": records})
        events = detector.get_events("s")

        point_dates = {e["date"] for e in events if e["type"] == "point_outlier"}
        self.assertGreaterEqual(len(point_dates), int(np.ceil(cap * 0.4)))

    def test_unused_reserved_slots_spill_over(self):
        """With no sustained candidates, point events may use the whole cap."""
        idx = pd.date_range("2022-01-01", periods=600, freq="D")
        events = [
            {
                "date": idx[pos],
                "end_date": idx[pos],
                "duration": 1,
                "magnitude": 10.0 + offset,
                "strength": 2.0 + offset,
                "type": "point_outlier",
                "source": "pass1",
            }
            for offset, pos in enumerate(range(20, 20 + 30 * 15, 15))
        ]

        cap = 12
        detector = ExtendedAnomalyDetector(
            max_anomalies_per_series=cap, min_point_anomaly_share=0.4
        )
        kept = detector._apply_cap(list(events))

        self.assertEqual(len(kept), cap)
        # And they are the strongest, not merely the reserved 40% share
        self.assertEqual(
            sorted(e["strength"] for e in kept),
            sorted(e["strength"] for e in events)[-cap:],
        )

    def test_duplicate_detections_share_one_cap_slot(self):
        """A spike seen as both a point and a window costs one slot, not two."""
        detector = ExtendedAnomalyDetector(max_anomalies_per_series=25)
        point = {
            "date": pd.Timestamp("2023-03-01"),
            "end_date": pd.Timestamp("2023-03-01"),
            "duration": 1,
            "magnitude": 12.0,
            "strength": 4.0,
            "type": "point_outlier",
            "source": "pass1",
        }
        window = {
            "date": pd.Timestamp("2023-02-27"),
            "end_date": pd.Timestamp("2023-03-12"),
            "duration": 14,
            "magnitude": 12.0,
            "strength": 3.0,
            "type": "transient_change",
            "source": "segmented_shift",
        }
        groups = detector._group_duplicate_slots([point, window])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        # Both records survive the cap even though they cost a single slot
        detector.max_anomalies_per_series = 1
        kept = detector._apply_cap([point, window])
        self.assertEqual(len(kept), 2)

    def test_unrelated_small_blip_inside_window_is_own_slot(self):
        """Magnitude mismatch means it is a different phenomenon."""
        detector = ExtendedAnomalyDetector(max_anomalies_per_series=25)
        point = {
            "date": pd.Timestamp("2023-03-01"),
            "end_date": pd.Timestamp("2023-03-01"),
            "duration": 1,
            "magnitude": 1.0,
            "strength": 4.0,
            "type": "point_outlier",
            "source": "pass1",
        }
        window = {
            "date": pd.Timestamp("2023-02-27"),
            "end_date": pd.Timestamp("2023-03-12"),
            "duration": 14,
            "magnitude": 12.0,
            "strength": 3.0,
            "type": "transient_change",
            "source": "segmented_shift",
        }
        groups = detector._group_duplicate_slots([point, window])
        self.assertEqual(len(groups), 2)


class TestOuterAnomalyMerge(unittest.TestCase):
    """_merge_anomaly_records must not absorb short spikes into long windows."""

    def test_short_spike_survives_long_extended_window(self):
        mixin = ExtendedAnomalyMixin()
        base = {
            "s": [
                {
                    "date": pd.Timestamp("2023-03-01"),
                    "magnitude": 12.0,
                    "duration": 1,
                    "type": "point_outlier",
                }
            ]
        }
        extended = {
            "s": [
                {
                    "date": pd.Timestamp("2023-02-01"),
                    "end_date": pd.Timestamp("2023-03-17"),
                    "duration": 45,
                    "magnitude": 3.0,
                    "type": "transient_change",
                }
            ]
        }
        merged = mixin._merge_anomaly_records(base, extended)["s"]
        dates = {e["date"] for e in merged}
        self.assertIn(pd.Timestamp("2023-03-01"), dates)
        self.assertEqual(len(merged), 2)

    def test_comparable_duration_event_is_still_absorbed(self):
        mixin = ExtendedAnomalyMixin()
        base = {
            "s": [
                {
                    "date": pd.Timestamp("2023-03-03"),
                    "magnitude": 12.0,
                    "duration": 10,
                    "type": "noisy_burst",
                }
            ]
        }
        extended = {
            "s": [
                {
                    "date": pd.Timestamp("2023-03-01"),
                    "end_date": pd.Timestamp("2023-03-15"),
                    "duration": 15,
                    "magnitude": 12.0,
                    "type": "transient_change",
                }
            ]
        }
        merged = mixin._merge_anomaly_records(base, extended)["s"]
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["date"], pd.Timestamp("2023-03-01"))


class TestAnomalyScoreStrength(unittest.TestCase):
    """Strength must increase with severity regardless of scoring convention."""

    @staticmethod
    def _severity_series(n=400):
        rng = np.random.default_rng(3)
        idx = pd.date_range("2022-01-01", periods=n, freq="D")
        values = rng.normal(0, 1, n)
        positions = [50, 150, 250, 350]
        amplitudes = [8.0, 14.0, 22.0, 35.0]
        for pos, amp in zip(positions, amplitudes):
            values[pos] += amp
        return pd.DataFrame({"s": values}, index=idx), idx[positions], amplitudes

    def test_p_value_scores_map_to_threshold_multiples(self):
        scores = pd.DataFrame({"s": [0.5, 0.05, 0.001, 1e-12]})
        strength = anomaly_scores_to_strength(
            scores, "rolling_zscore", {"alpha": 0.05}
        )["s"]
        # alpha itself sits exactly on the detection boundary
        self.assertAlmostEqual(float(strength.iloc[1]), 1.0, places=6)
        self.assertTrue(strength.is_monotonic_increasing)

    def test_strength_tracks_severity_for_each_method(self):
        df, spike_dates, amplitudes = self._severity_series()
        for method, params in [
            ("rolling_zscore", {"alpha": 0.05, "rolling_periods": 100}),
            ("zscore", {"alpha": 0.05}),
            ("mad", {"alpha": 0.05}),
            ("IQR", {}),
            ("IsolationForest", {"contamination": 0.05}),
        ]:
            with self.subTest(method=method):
                detector = AnomalyRemoval(
                    method=method, output="multivariate", method_params=params
                )
                detector.fit(df)
                strength = anomaly_scores_to_strength(
                    detector.scores,
                    detector.method,
                    detector.method_params,
                    detector.anomalies,
                )
                observed = [float(strength.loc[d, "s"]) for d in spike_dates]
                finite = [v for v in observed if np.isfinite(v)]
                self.assertEqual(len(finite), len(observed))
                # The biggest spike must not score below the smallest one
                self.assertGreater(observed[-1], observed[0])

    def test_unknown_method_polarity_is_inferred(self):
        """A method missing from the tables must not invert silently."""
        # Lower score = more anomalous, flags mark the two lowest points
        scores = pd.DataFrame({"s": [0.4, -0.9, 0.3, -0.5]})
        flags = pd.DataFrame({"s": [1, -1, 1, -1]})
        strength = anomaly_scores_to_strength(scores, "some_future_method", {}, flags)[
            "s"
        ]
        self.assertGreater(float(strength.iloc[1]), float(strength.iloc[0]))
        self.assertGreater(float(strength.iloc[1]), float(strength.iloc[3]))


if __name__ == "__main__":
    unittest.main()
