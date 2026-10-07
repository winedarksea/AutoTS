# -*- coding: utf-8 -*-
"""Tests for the opt-in joint hinge/step trend mode, t-test level-shift validation,
trend damping and the NaN guard in anomaly typing."""

import os
import random
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autots.evaluator.feature_detector import TimeSeriesFeatureDetector
from autots.evaluator.feature_detector.utils.joint_trend_selection import (
    KNOT,
    STEP,
    HingeStepCandidates,
    joint_fit,
    profile_likelihood_windows,
    select_hinge_step_terms,
    split_transient_step_pairs,
)

JOINT = {'method': 'joint_hinge_step'}


def _brute_force_terms(values, max_terms=4, min_segment=28, edge=14, penalty_mult=2.0):
    """Reference greedy knot selection by explicit OLS refits (no refinement)."""
    n = len(values)
    candidates = HingeStepCandidates(n)
    base = [np.ones(n), candidates.t]
    knots = []

    def sse(columns):
        design = np.column_stack(columns)
        beta, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - design @ beta
        return float(residual @ residual)

    current = sse(base)
    while len(knots) < max_terms:
        best = None
        for c in range(edge, n - edge):
            if any(abs(c - k) < min_segment for k in knots):
                continue
            trial = sse(base + [candidates.column(k, KNOT) for k in knots + [c]])
            if best is None or trial < best[1]:
                best = (c, trial)
        if best is None or n * np.log(current / best[1]) < 2 * penalty_mult * np.log(n):
            break
        knots.append(best[0])
        current = best[1]
    return sorted(knots)


def _scenario_frame(seed=0, n=1100):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    weekly = 2 * np.sin(2 * np.pi * t / 7)
    index = pd.date_range("2021-03-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            'linear': 100 + 0.05 * t + weekly + rng.normal(0, 1, n),
            'step': 100 + 0.03 * t + 5 * (t >= 600) + weekly + rng.normal(0, 1, n),
            'piecewise': 100
            + 0.05 * t
            + 0.08 * np.maximum(0, t - 400)
            - 0.1 * np.maximum(0, t - 800)
            + weekly
            + rng.normal(0, 1, n),
        },
        index=index,
    )


class TestJointTrendSelection(unittest.TestCase):
    def test_closed_form_products_match_explicit_columns(self):
        n = 300
        candidates = HingeStepCandidates(n)
        vectors = np.random.default_rng(0).normal(size=(n, 3))
        for kind in (KNOT, STEP):
            explicit = np.array(
                [candidates.column(c, kind) @ vectors for c in range(n)]
            )
            norms = np.array(
                [
                    candidates.column(c, kind) @ candidates.column(c, kind)
                    for c in range(n)
                ]
            )
            np.testing.assert_allclose(
                candidates.dots(vectors, kind), explicit, atol=1e-10
            )
            np.testing.assert_allclose(candidates.norm2(kind), norms, atol=1e-10)

    def test_greedy_knots_match_brute_force(self):
        rng = np.random.default_rng(3)
        t = np.arange(400)
        values = 0.02 * t + 0.06 * np.maximum(0, t - 150) + rng.normal(0, 0.5, 400)
        reference = _brute_force_terms(values)
        selected = [
            p
            for kind, p in select_hinge_step_terms(
                values, search_steps=False, max_refine_rounds=0
            )
            if kind == KNOT
        ]
        self.assertEqual(len(selected), len(reference))
        for got, expected in zip(selected, reference):
            self.assertLessEqual(abs(got - expected), 1)

    def test_scenarios(self):
        rng = np.random.default_rng(0)
        t = np.arange(1100)
        linear = 0.05 * t + rng.normal(0, 1, 1100)
        self.assertEqual(select_hinge_step_terms(linear), [])
        piecewise = (
            0.05 * t
            + 0.08 * np.maximum(0, t - 400)
            - 0.1 * np.maximum(0, t - 800)
            + rng.normal(0, 1, 1100)
        )
        knots = [p for k, p in select_hinge_step_terms(piecewise, search_steps=False)]
        self.assertEqual(len(knots), 2)
        self.assertLessEqual(abs(knots[0] - 400), 30)
        self.assertLessEqual(abs(knots[1] - 800), 30)
        stepped = 0.03 * t + 5 * (t >= 600) + rng.normal(0, 1, 1100)
        terms = select_hinge_step_terms(stepped)
        self.assertEqual(terms, [(STEP, 600)])
        fit = joint_fit(stepped, terms)
        self.assertAlmostEqual(fit['terms'][0]['coefficient'], 5.0, delta=0.5)

    def test_transient_pair_split(self):
        rng = np.random.default_rng(1)
        t = np.arange(1100)
        values = 0.02 * t + 4 * ((t >= 500) & (t < 530)) + rng.normal(0, 1, 1100)
        fit = joint_fit(values, select_hinge_step_terms(values))
        steps = [d for d in fit['terms'] if d['kind'] == STEP]
        kept, transients = split_transient_step_pairs(steps, max_gap=90)
        self.assertEqual(kept, [])
        self.assertEqual(len(transients), 1)
        self.assertAlmostEqual(transients[0]['magnitude'], 4.0, delta=0.8)

    def test_degenerate_input_returns_no_terms(self):
        self.assertEqual(select_hinge_step_terms(np.ones(500)), [])
        self.assertEqual(select_hinge_step_terms(np.arange(500.0)), [])

    def test_profile_windows_contain_true_break(self):
        covered = 0
        for seed in range(10):
            rng = np.random.default_rng(seed)
            t = np.arange(600)
            values = 0.05 * np.maximum(0, t - 300) + rng.normal(0, 1, 600)
            terms = select_hinge_step_terms(values, search_steps=False)
            if not terms:
                continue
            low, high = profile_likelihood_windows(values, terms)[0]
            covered += low <= 300 <= high
        self.assertGreaterEqual(covered, 8)


class TestJointTrendDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")
        cls.df = _scenario_frame()
        cls.detector = TimeSeriesFeatureDetector(changepoint_params=JOINT).fit(cls.df)

    def test_linear_trend_has_no_events_and_correct_slope(self):
        self.assertEqual(self.detector.trend_changepoints['linear'], [])
        self.assertEqual(self.detector.level_shifts['linear'], [])
        slope = self.detector.trend_slopes['linear'][-1]['slope']
        self.assertAlmostEqual(slope, 0.05, delta=0.0025)

    def test_step_magnitude_and_noise(self):
        shifts = self.detector.level_shifts['step']
        self.assertEqual(len(shifts), 1)
        self.assertLessEqual(abs((shifts[0][0] - self.df.index[600]).days), 3)
        self.assertAlmostEqual(shifts[0][1], 5.0, delta=0.5)
        noise_sd = np.nanstd(self.detector.components['step']['noise'])
        self.assertLess(abs(noise_sd - 1.0), 0.1)

    def test_piecewise_found_with_windows(self):
        dates = [entry[0] for entry in self.detector.trend_changepoints['piecewise']]
        self.assertEqual(len(dates), 2)
        for found, true_pos in zip(dates, (400, 800)):
            self.assertLessEqual(abs((found - self.df.index[true_pos]).days), 30)
        template = self.detector.get_template()
        records = template['series']['piecewise']['labels']['trend_changepoints']
        self.assertTrue(all('date_window' in r and 't_stat' in r for r in records))

    def test_no_jan_first_discontinuity_in_seasonality(self):
        seasonal = pd.Series(
            self.detector.components['linear']['seasonality'], index=self.df.index
        )
        jumps = seasonal.diff().abs()
        jan_first = jumps[(jumps.index.month == 1) & (jumps.index.day == 1)]
        self.assertLess(jan_first.max(), 4 * jumps.median() + 0.5)

    def test_forecast_finite_and_damped(self):
        undamped = self.detector.forecast(365).forecast
        damped = self.detector.forecast(365, trend_damping=0.98).forecast
        self.assertTrue(np.isfinite(damped.to_numpy()).all())
        growth_undamped = undamped['linear'].iloc[-1] - undamped['linear'].iloc[0]
        growth_damped = damped['linear'].iloc[-1] - damped['linear'].iloc[0]
        self.assertLess(abs(growth_damped), abs(growth_undamped))

    def test_default_detector_unchanged_by_options(self):
        default = TimeSeriesFeatureDetector().fit(self.df[['linear']])
        self.assertFalse(default._uses_joint_trend())
        self.assertNotIn(
            'transient_events', default.get_template()['series']['linear']['labels']
        )

    def test_univariate_and_weekly(self):
        uni = TimeSeriesFeatureDetector(
            changepoint_params=JOINT, detection_mode='univariate'
        ).fit(self.df)
        self.assertTrue(np.isfinite(uni.forecast(30).forecast.to_numpy()).all())
        weekly = self.df.resample('W').mean()
        weekly.iloc[10:13, 0] = np.nan
        fitted = TimeSeriesFeatureDetector(changepoint_params=JOINT).fit(weekly)
        self.assertTrue(np.isfinite(fitted.forecast(12).forecast.to_numpy()).all())


class TestJointTrendTransientsAndRecords(unittest.TestCase):
    def test_step_pair_is_transient_not_level_shift(self):
        warnings.filterwarnings("ignore")
        rng = np.random.default_rng(4)
        n = 1100
        t = np.arange(n)
        index = pd.date_range("2021-03-01", periods=n, freq="D")
        df = pd.DataFrame(
            {'a': 100 + 0.02 * t + 6 * ((t >= 500) & (t < 530)) + rng.normal(0, 1, n)},
            index=index,
        )
        detector = TimeSeriesFeatureDetector(
            changepoint_params={
                'method': 'joint_hinge_step',
                'method_params': {'step_search': 'unified'},
            }
        ).fit(df)
        self.assertEqual(detector.level_shifts['a'], [])
        transients = detector.get_detected_features()['transient_events']['a']
        self.assertEqual(len(transients), 1)
        self.assertLessEqual(
            abs((pd.Timestamp(transients[0]['start_date']) - index[500]).days), 3
        )
        self.assertAlmostEqual(transients[0]['magnitude'], 6.0, delta=1.0)
        labels = detector.get_template()['series']['a']['labels']
        self.assertEqual(len(labels['transient_events']), 1)

    def test_changepoint_records_have_effect_fields(self):
        warnings.filterwarnings("ignore")
        df = _scenario_frame()[['piecewise']]
        detector = TimeSeriesFeatureDetector(changepoint_params=JOINT).fit(df)
        records = detector.get_template()['series']['piecewise']['labels'][
            'trend_changepoints'
        ]
        self.assertTrue(records)
        for record in records:
            self.assertEqual(record['type'], 'slope_change')
            change = record['new_slope'] - record['prior_slope']
            self.assertAlmostEqual(record['effect_90d'], change * 90, places=6)
            self.assertGreater(record['regime_length_days'], 0)
        self.assertTrue(records[-1]['is_active_regime'])
        self.assertFalse(records[0]['is_active_regime'])


class TestTTestLevelShiftValidation(unittest.TestCase):
    def test_false_candidates_rejected_true_steps_kept(self):
        detector = TimeSeriesFeatureDetector(
            level_shift_validation={'method': 't_test'}
        )
        rng = np.random.default_rng(0)
        index = pd.date_range("2020-01-01", periods=1100, freq="D")
        detector.date_index = index
        zeros = pd.DataFrame(0.0, index=index, columns=['a'])
        accepted = {0.0: 0, 1.0: 0}
        for step in accepted:
            for _ in range(100):
                values = rng.normal(0, 0.3, 1100)
                candidate = int(rng.integers(100, 1000))
                values[candidate:] += step
                residual = pd.DataFrame({'a': values}, index=index)
                _, validated = detector._validate_level_shifts(
                    residual,
                    zeros,
                    {'a': [{'date': index[candidate], 'magnitude': step}]},
                )
                accepted[step] += len(validated['a'])
        self.assertLess(accepted[0.0], 5)
        self.assertGreater(accepted[1.0], 95)


class TestParamsAndNaN(unittest.TestCase):
    def test_get_new_params_samples_new_options_and_fits(self):
        random.seed(0)
        np.random.seed(0)
        samples = [TimeSeriesFeatureDetector.get_new_params() for _ in range(200)]
        methods = {s['changepoint_params'].get('method') for s in samples}
        self.assertIn('joint_hinge_step', methods)
        self.assertTrue(any(s['level_shift_validation'] for s in samples))
        self.assertTrue(any(s['trend_damping'] for s in samples))
        joint = next(
            s
            for s in samples
            if s['changepoint_params'].get('method') == 'joint_hinge_step'
        )
        df = _scenario_frame(n=500)
        detector = TimeSeriesFeatureDetector(**joint).fit(df)
        self.assertTrue(np.isfinite(detector.forecast(14).forecast.to_numpy()).all())

    def test_joint_param_sampler_and_bounded_mutation(self):
        from autots.evaluator.feature_detector.components.joint_trend import (
            JOINT_TREND_PARAM_SPACE,
            mutate_joint_trend_changepoint_params,
            sample_joint_trend_changepoint_params,
        )

        rng = random.Random(0)
        params = sample_joint_trend_changepoint_params(rng)
        for _ in range(100):
            params = mutate_joint_trend_changepoint_params(params, rng)
            self.assertEqual(params['method'], 'joint_hinge_step')
            for key, value in params['method_params'].items():
                self.assertIn(value, JOINT_TREND_PARAM_SPACE[key][0], key)

    def test_optimizer_local_mutation_keeps_joint_params_in_space(self):
        from autots.evaluator.feature_detector.optimizer import (
            FeatureDetectionOptimizer,
        )
        from autots.evaluator.feature_detector.components.joint_trend import (
            JOINT_TREND_PARAM_SPACE,
            sample_joint_trend_changepoint_params,
        )

        rng = random.Random(1)
        optimizer = FeatureDetectionOptimizer.__new__(FeatureDetectionOptimizer)
        params = sample_joint_trend_changepoint_params(rng)
        for _ in range(50):
            params = optimizer._local_mutate_changepoint_params(params, rng)
            if params.get('method') != 'joint_hinge_step':
                break
            for key, value in params['method_params'].items():
                self.assertIn(value, JOINT_TREND_PARAM_SPACE[key][0], key)

    def test_nan_series_fits(self):
        from autots.datasets.synthetic import SyntheticDailyGenerator

        df = SyntheticDailyGenerator(n_days=500, n_series=3, random_seed=0).get_data()
        self.assertTrue(df.isna().any().any())
        TimeSeriesFeatureDetector(changepoint_params=JOINT).fit(df)


if __name__ == '__main__':
    unittest.main()
