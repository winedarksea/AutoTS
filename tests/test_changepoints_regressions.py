# -*- coding: utf-8 -*-
"""
Regression tests for changepoint detection accuracy bugs: PELT pruning and ED
scaling, EWMA start-up, trend-filter index offsets, bottom-up cap, NaN date
mapping, CUSUM re-triggering, and the probabilistic methods.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autots.tools.changepoints import (
    ChangepointDetector,
    _calculate_segment_cost,
    _detect_bottom_up_changepoints,
    _detect_cusum_changepoints,
    _detect_l0_trend_changepoints,
    _detect_l1_trend_changepoints,
    _detect_pelt_changepoints,
    _vectorized_cusum_changepoints,
    _vectorized_ewma_changepoints,
)
from autots.tools.changepoints_probabilistic import (
    bayesian_online_changepoint_probabilities,
)


def _level_shift_series(rng, n=1000, n_shifts=5, noise=1.0):
    spacing = min(40, (n - 100) // n_shifts)
    true_cps = np.sort(rng.choice(np.arange(50, n - 50, spacing), n_shifts, replace=False))
    levels = np.cumsum(rng.choice([-1, 1], n_shifts + 1) * rng.uniform(1.5, 3, n_shifts + 1))
    signal = np.repeat(levels, np.diff(np.r_[0, true_cps, n])) + rng.normal(0, noise, n)
    return signal, true_cps


def _f1(predicted, true_cps, tolerance=3):
    predicted = list(predicted)
    if not predicted:
        return 0.0
    used, hits = set(), 0
    for true_cp in true_cps:
        matches = [p for p in predicted if abs(p - true_cp) <= tolerance and p not in used]
        if matches:
            used.add(matches[0])
            hits += 1
    if hits == 0:
        return 0.0
    precision, recall = hits / len(predicted), hits / len(true_cps)
    return 2 * precision * recall / (precision + recall)


def _brute_force_optimal_partition(data, penalty, min_segment_length, loss):
    """Unpruned optimal partitioning; PELT with exact pruning must match it."""
    n = len(data)
    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    for t in range(min_segment_length, n + 1):
        for s in [0] + list(range(min_segment_length, t - min_segment_length + 1)):
            cost = _calculate_segment_cost(data[s:t], loss)
            if loss == "ed":
                cost /= t - s
            if F[s] + cost + penalty < F[t]:
                F[t] = F[s] + cost + penalty
                last[t] = s
    cps, t = [], n
    while t > 0 and last[t] != 0:
        cps.append(last[t])
        t = last[t]
    return sorted(cps)


class TestPeltRegressions(unittest.TestCase):
    def test_exact_pruning_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for loss in ["l2", "ed", "l1"]:
            for _ in range(4):
                signal, _ = _level_shift_series(rng, n=150, n_shifts=3)
                for min_seg in [1, 5]:
                    self.assertEqual(
                        list(_detect_pelt_changepoints(signal, 10, loss, min_seg, 1.0)),
                        _brute_force_optimal_partition(signal, 10, min_seg, loss),
                        f"loss={loss} min_seg={min_seg}",
                    )

    def test_aggressive_pruning_keeps_accuracy(self):
        """pruning_factor > 1 used to prune recent candidates: F1 0.84 -> 0.28."""
        rng = np.random.default_rng(1)
        scores = {1.0: [], 2.0: [], 3.5: []}
        counts = {1.0: [], 2.0: [], 3.5: []}
        for _ in range(10):
            signal, true_cps = _level_shift_series(rng)
            for pf in scores:
                cps = _detect_pelt_changepoints(signal, 20, "l2", 10, pf)
                scores[pf].append(_f1(cps, true_cps))
                counts[pf].append(len(cps))
        for pf in [2.0, 3.5]:
            self.assertGreaterEqual(np.mean(scores[pf]), np.mean(scores[1.0]) - 0.05)
            self.assertLess(abs(np.mean(counts[pf]) - 5), 1.0)

    def test_aggressive_pruning_fresh_candidate_survives(self):
        rng = np.random.default_rng(77)
        signal = np.concatenate([rng.normal(0, 1, 60), rng.normal(7, 1, 60)])
        for loss in ["l2", "ed"]:
            cps = _detect_pelt_changepoints(signal, 200, loss, 1, 3.0)
            self.assertEqual(list(cps), [60], loss)

    def test_ed_penalty_controls_noise(self):
        """Unnormalized ED cost grew O(L^2): 11+ changepoints on noise at penalty 5000."""
        signal = np.random.default_rng(2).normal(0, 1, 800)
        self.assertEqual(len(_detect_pelt_changepoints(signal, 50, "ed", 5)), 0)

    def test_method_params_min_segment_length_is_used(self):
        rng = np.random.default_rng(3)
        signal = np.repeat([0.0, 4.0, 0.0, 4.0], [100, 8, 100, 100]) + rng.normal(0, 0.3, 308)
        df = pd.DataFrame({"a": signal}, index=pd.date_range("2020-01-01", periods=308))
        found = {}
        for min_seg in [2, 20]:
            detector = ChangepointDetector(
                method="pelt",
                method_params={"penalty": 10, "min_segment_length": min_seg},
                aggregate_method="mean",
                min_segment_length=2,
            )
            detector.detect(df)
            found[min_seg] = list(detector.changepoints_)
        self.assertIn(100, found[2])
        self.assertIn(108, found[2])
        self.assertTrue(np.all(np.diff([0] + found[20] + [308]) >= 20), found[20])


class TestEwmaRegressions(unittest.TestCase):
    def test_no_startup_false_alarms_on_noise(self):
        """EWMA started at x0 flagged t<=3 on 32.5% of pure-noise series."""
        rng = np.random.default_rng(4)
        series_list = [rng.normal(0, 1, 300) for _ in range(300)]
        for adaptive in [True, False]:
            results = _vectorized_ewma_changepoints(
                series_list, 0.2, 3.0, 5, True, True, adaptive, 5
            )
            early_rate = np.mean([np.any(np.asarray(c) <= 3) for c in results])
            self.assertLess(early_rate, 0.03, f"adaptive={adaptive}")


class TestTrendFilterOffsets(unittest.TestCase):
    def test_hinge_reported_at_true_index(self):
        """Orders >= 2 were shifted by +order and reported one step late."""
        t = np.arange(300.0)
        hinge = np.maximum(0, t - 100)
        level = (t >= 100).astype(float) * 5
        cps, _ = _detect_l1_trend_changepoints(level, 1.0, "fused_lasso", difference_order=1)
        self.assertEqual(list(cps), [100])
        cps, _ = _detect_l1_trend_changepoints(hinge, 1.0, "total_variation", difference_order=2)
        self.assertEqual(list(cps), [100])
        for order, signal in [(1, level), (2, hinge), (4, hinge)]:
            cps, _ = _detect_l0_trend_changepoints(
                signal, 1.0, difference_order=order, max_changepoints=1
            )
            self.assertEqual(list(cps), [100], f"order={order}")

    def test_vectorized_path_matches(self):
        t = np.arange(300.0)
        rng = np.random.default_rng(5)
        df = pd.DataFrame(
            {c: np.maximum(0, t - 100) + rng.normal(0, 0.05, 300) for c in "ab"},
            index=pd.date_range("2020-01-01", periods=300),
        )
        detector = ChangepointDetector(
            method="l0_trend_filter",
            method_params={"difference_order": 2, "max_changepoints": 1},
            aggregate_method="individual",
        )
        detector.detect(df)
        for col in "ab":
            self.assertEqual(list(detector.changepoints_[col]), [100])


class TestBottomUpRegressions(unittest.TestCase):
    def test_max_changepoints_enforced(self):
        """The penalty stop used to bypass the cap (83/100 runs exceeded it)."""
        rng = np.random.default_rng(6)
        for penalty_scale in [1.0, 0.5]:
            for _ in range(20):
                signal, _ = _level_shift_series(rng, n=500, n_shifts=8)
                cps, _ = _detect_bottom_up_changepoints(
                    signal, penalty_scale=penalty_scale, max_changepoints=5
                )
                self.assertLessEqual(len(cps), 5)


class TestNaNDateMapping(unittest.TestCase):
    def test_features_use_detection_dates(self):
        """Positions over non-NaN values were looked up in the full index."""
        index = pd.date_range("2020-01-01", periods=300)
        values = np.r_[np.zeros(150), np.full(150, 5.0)]
        values += np.random.default_rng(7).normal(0, 0.3, 300)
        values[:50] = np.nan
        df = pd.DataFrame({"a": values, "b": values}, index=index)
        for aggregate_method in ["individual", "mean"]:
            detector = ChangepointDetector(
                method="pelt", method_params={"penalty": 20}, aggregate_method=aggregate_method
            )
            detector.detect(df)
            features = detector.create_features(forecast_length=5)
            first_nonzero = features.iloc[:, 0].ne(0).idxmax()
            self.assertEqual(first_nonzero, index[151], aggregate_method)


class TestCusumRegressions(unittest.TestCase):
    def test_single_step_reported_once_at_onset(self):
        """Global centering + alarm-time reporting averaged 5.9 changepoints per step."""
        rng = np.random.default_rng(8)
        counts, errors = [], []
        for _ in range(30):
            signal = np.r_[np.zeros(250), np.full(250, 2.0)] + rng.normal(0, 1, 500)
            cps = _detect_cusum_changepoints(signal, threshold=10.0)
            counts.append(len(cps))
            errors.extend(abs(np.asarray(cps) - 250))
        self.assertLess(np.mean(counts), 1.3)
        self.assertLess(np.median(errors), 5)

    def test_multi_step_f1_and_vectorized_agreement(self):
        rng = np.random.default_rng(9)
        series, truths = zip(*[_level_shift_series(rng) for _ in range(20)])
        vectorized = _vectorized_cusum_changepoints(list(series), 10.0, 0.0, 5, True, 5)
        scores = []
        for signal, true_cps, vec_cps in zip(series, truths, vectorized):
            single = _detect_cusum_changepoints(signal, threshold=10.0, min_distance=5)
            self.assertEqual(list(single), list(vec_cps))
            scores.append(_f1(single, true_cps))
        self.assertGreater(np.mean(scores), 0.8)


class TestProbabilisticRegressions(unittest.TestCase):
    def test_bayesian_online_peaks_at_change(self):
        """Evidence underflowed to 0 and returned all zeros."""
        rng = np.random.default_rng(10)
        signal = np.r_[rng.normal(0, 1, 150), rng.normal(4, 1, 150)]
        probs = bayesian_online_changepoint_probabilities(signal, 0.01, 5)
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertLessEqual(abs(int(np.argmax(probs)) - 150), 1)
        # Mass splits between adjacent onsets when the first post-change point is ambiguous.
        self.assertGreater(probs[148:153].sum(), 0.8)
        noise_probs = bayesian_online_changepoint_probabilities(rng.normal(0, 1, 300), 0.01, 5)
        self.assertLess(noise_probs.max(), 0.5)

    def test_bootstrap_preserves_time_order(self):
        """Shuffling observations destroyed the segment structure being estimated."""
        rng = np.random.default_rng(11)
        signal = np.r_[rng.normal(0, 1, 150), rng.normal(5, 1, 150)]
        df = pd.DataFrame({"a": signal}, index=pd.date_range("2020-01-01", periods=300))
        detector = ChangepointDetector(
            method="pelt",
            method_params={"penalty": 20, "probabilistic_method": "bootstrap"},
            aggregate_method="individual",
            probabilistic_output=True,
        )
        detector.detect(df)
        probs = np.asarray(detector.changepoint_probabilities_["a"])
        self.assertEqual(int(np.argmax(probs)), 150)
        self.assertGreater(probs.max(), 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
