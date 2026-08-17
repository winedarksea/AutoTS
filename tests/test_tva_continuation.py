# -*- coding: utf-8 -*-
"""Tests for the validation-selected factor continuation and seasonal paths."""

import unittest

import numpy as np

from autots.evaluator.tva.continuation import (
    DEFAULT_CONTINUATION_CONFIG,
    apply_choice,
    build_specs,
    candidate_futures,
    continue_factor,
    select_continuations,
)
from autots.evaluator.tva.seasonal import (
    amplitude_scale,
    assemble_choice,
    empirical_profile,
    select_seasonal_paths,
    tile_profile,
)


def _kinked_path(n_time=400, knot=250, up=0.02, down=-0.03):
    t = np.arange(n_time, dtype=float)
    return np.where(t < knot, up * t, up * knot + down * (t - knot))


class TestContinuationCandidates(unittest.TestCase):
    def setUp(self):
        self.specs = {s['name']: s for s in build_specs()}
        self.path = _kinked_path()
        self.knots = np.arange(7, 400 - 7, 7)
        self.coef = np.zeros(len(self.knots) + 1)
        self.coef[0] = 0.02
        self.coef[1 + int(np.searchsorted(self.knots, 250))] = -0.05
        self.origins = np.array([399])

    def _delta(self, name, horizon=60):
        return continue_factor(
            self.path, self.origins, horizon, self.specs[name],
            knot_times=self.knots, coef_k=self.coef,
        )[0, -1]

    def test_constant_candidate_does_not_move(self):
        self.assertEqual(self._delta('constant'), 0.0)

    def test_undamped_slope_matches_the_linear_segment(self):
        # the last 90 days are exactly linear at -0.03/step
        self.assertAlmostEqual(self._delta('damped_w90_p1'), -0.03 * 60, places=6)

    def test_zero_damping_equals_constant(self):
        self.assertEqual(self._delta('damped_w90_p0'), 0.0)

    def test_regime_candidate_beats_a_stale_fixed_window(self):
        """A window straddling the changepoint mixes two regimes; the regime
        candidate measures only since the last active knot."""
        truth = -0.03 * 60
        regime_err = abs(self._delta('regime') - truth)
        stale_err = abs(self._delta('damped_w180_p1') - truth)
        self.assertLess(regime_err, stale_err)
        self.assertAlmostEqual(self._delta('regime'), truth, places=6)

    def test_trust_modifier_damps_more_than_the_raw_regime(self):
        self.assertLess(abs(self._delta('regime_trust')), abs(self._delta('regime')))

    def test_regime_stays_put_when_the_regime_is_too_young(self):
        knots = np.arange(7, 400 - 7, 7)
        last_knot = int(knots[-1])
        path = _kinked_path(knot=last_knot)
        coef = np.zeros(len(knots) + 1)
        # the newest regime boundary the trend filter can express
        coef[len(knots)] = -0.05
        origin = last_knot + 4  # 5 days of regime, below min_age=7
        delta = continue_factor(
            path, np.array([origin]), 30, self.specs['regime'],
            knot_times=knots, coef_k=coef,
        )
        self.assertTrue(np.all(delta == 0.0))

    def test_ridge_ar_never_explodes(self):
        """An explosive AR fit is exactly the runaway this phase prevents."""
        rng = np.random.default_rng(0)
        walk = np.cumsum(rng.normal(0.0, 1.0, 600))
        delta = continue_factor(
            walk, np.array([599]), 180, self.specs['ridge_ar_l5']
        )
        self.assertTrue(np.all(np.isfinite(delta)))
        self.assertLess(np.abs(delta).max(), 50 * np.abs(np.diff(walk)).std() * 180)

    def test_short_history_returns_zeros_rather_than_raising(self):
        delta = continue_factor(
            np.array([1.0, 2.0]), np.array([1]), 10, self.specs['ridge_ar_l5']
        )
        self.assertEqual(delta.shape, (1, 10))
        self.assertTrue(np.all(delta == 0.0))


class TestContinuationSelection(unittest.TestCase):
    def test_selection_recovers_the_candidate_that_matches_the_future(self):
        rng = np.random.default_rng(1)
        n_time, horizon = 500, 40
        paths = np.column_stack([_kinked_path(n_time), np.zeros(n_time)])
        origins = np.array([n_time - 1])
        truth = continue_factor(
            paths[:, 0], origins, horizon,
            {'kind': 'damped', 'window': 90, 'phi': 1.0, 'name': 'x'},
        )

        def score_fn(deltas):
            return float(np.mean(np.abs(deltas[:, :, 0] - truth)))

        model_deltas = np.zeros((1, horizon, 2))
        result = select_continuations(
            paths, origins, horizon, score_fn, model_deltas=model_deltas,
            config={'selection_tolerance': 0.0},
        )
        self.assertLess(result['score'], result['baseline_score'])
        self.assertNotEqual(result['choice'][0], 'model')

    def test_incumbent_is_kept_when_nothing_is_materially_better(self):
        paths = np.column_stack([_kinked_path(300)])
        origins = np.array([299])
        model_deltas = np.zeros((1, 20, 1))
        result = select_continuations(
            paths, origins, 20, lambda d: 1.0, model_deltas=model_deltas
        )
        self.assertEqual(result['choice'][0], 'model')

    def test_apply_choice_reproduces_the_selected_candidate(self):
        paths = np.column_stack([_kinked_path(300)])
        origins = np.array([299])
        specs = build_specs()
        applied = apply_choice(
            paths, origins, 25, {0: 'constant'}, specs=specs,
            model_deltas=np.ones((1, 25, 1)),
        )
        self.assertTrue(np.all(applied == 0.0))

    def test_a_failing_score_function_does_not_raise(self):
        paths = np.column_stack([_kinked_path(200)])

        def bad(_deltas):
            raise RuntimeError('nope')

        result = select_continuations(
            paths, np.array([199]), 10, bad,
            model_deltas=np.zeros((1, 10, 1)),
        )
        self.assertIn('choice', result)

    def test_candidate_futures_covers_every_factor_and_spec(self):
        paths = np.column_stack([_kinked_path(300), _kinked_path(300, 100)])
        specs = [s for s in build_specs() if s['kind'] != 'model']
        futures = candidate_futures(paths, np.array([299]), 12, specs)
        self.assertEqual(len(futures), 2 * len(specs))

    def test_default_config_keys_are_stable(self):
        for key in ('slope_windows', 'damping', 'selection_passes'):
            self.assertIn(key, DEFAULT_CONTINUATION_CONFIG)


class TestSeasonalArbitration(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.m, self.n, self.T = 7, 4, 400
        self.profile = self.rng.normal(0.0, 1.0, (self.m, self.n))
        self.profile -= self.profile.mean(axis=0)
        tiled = np.tile(self.profile, (self.T // self.m + 1, 1))[: self.T]
        self.residual = tiled + self.rng.normal(0.0, 0.05, (self.T, self.n))

    def test_empirical_profile_is_phase_aligned_to_the_next_step(self):
        """Row 0 must be the phase immediately AFTER the history ends."""
        recovered = empirical_profile(self.residual, self.m, 2)
        expected = np.roll(self.profile, -(self.T % self.m), axis=0)
        corr = np.corrcoef(recovered.ravel(), expected.ravel())[0, 1]
        self.assertGreater(corr, 0.99)

    def test_tiled_profile_matches_the_true_future(self):
        tiled = tile_profile(empirical_profile(self.residual, self.m, 2), 28)
        truth = np.array([self.profile[(self.T + h) % self.m] for h in range(28)])
        self.assertGreater(np.corrcoef(tiled.ravel(), truth.ravel())[0, 1], 0.99)

    def test_empirical_profile_is_centered(self):
        profile = empirical_profile(self.residual, self.m, 2)
        self.assertTrue(np.allclose(profile.mean(axis=0), 0.0, atol=1e-9))

    def test_amplitude_scale_moves_toward_the_observed_amplitude(self):
        fitted = np.tile(self.profile, (self.T // self.m + 1, 1))[: self.T]
        alpha = amplitude_scale(fitted, self.residual * 1.5)
        self.assertTrue(np.all(alpha > 1.0))
        self.assertTrue(np.all(alpha < 1.5))  # shrunk toward 1, never overshoots

    def test_amplitude_scale_is_one_for_a_perfect_fit(self):
        fitted = np.tile(self.profile, (self.T // self.m + 1, 1))[: self.T]
        alpha = amplitude_scale(fitted, fitted)
        self.assertTrue(np.allclose(alpha, 1.0, atol=1e-6))

    def test_incumbent_wins_inside_the_selection_margin(self):
        actual = np.tile(self.profile, (5, 1))[:28]
        result = select_seasonal_paths(
            {'datepart': actual, 'empirical': actual * 0.99}, actual,
            default='datepart',
        )
        self.assertEqual(set(result['choice'].values()), {'datepart'})
        self.assertEqual(result['n_changed'], 0)

    def test_a_materially_better_candidate_replaces_the_incumbent(self):
        actual = np.tile(self.profile, (5, 1))[:28]
        result = select_seasonal_paths(
            {'datepart': actual * 0.2, 'empirical': actual}, actual,
            default='datepart',
        )
        self.assertEqual(set(result['choice'].values()), {'empirical'})

    def test_assemble_choice_mixes_per_series(self):
        actual = np.tile(self.profile, (5, 1))[:28]
        candidates = {'datepart': np.zeros_like(actual), 'empirical': actual}
        mixed = assemble_choice(candidates, {0: 'empirical'}, 'datepart')
        self.assertTrue(np.allclose(mixed[:, 0], actual[:, 0]))
        self.assertTrue(np.allclose(mixed[:, 1], 0.0))

    def test_degenerate_inputs_return_rather_than_raise(self):
        self.assertIsNone(empirical_profile(np.zeros((3, 2)), 7, 2))
        self.assertIsNone(tile_profile(None, 10))
        empty = select_seasonal_paths({'datepart': None}, np.zeros((5, 2)))
        self.assertEqual(empty['choice'], {})
        alpha = amplitude_scale(np.zeros((10, 2)), np.zeros((10, 2)))
        self.assertTrue(np.allclose(alpha, 1.0))


if __name__ == '__main__':
    unittest.main()
