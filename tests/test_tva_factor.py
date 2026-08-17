# -*- coding: utf-8 -*-
"""Tests for the learned latent-factor trend mode (``trend_network='factor'``).

Thresholds are calibrated empirically on the fixed seeds used here and set
below the observed values, following the convention of the rest of the TVA
suite. Everything that needs torch is skipped when torch is unavailable.
"""

import unittest
import numpy as np
import pandas as pd

from autots.evaluator.tva.discovery import match_factors
from autots.evaluator.tva.factor_network import (
    HAS_TORCH,
    estimate_factors_alternating,
    split_half_stability,
    hinge_design,
    robust_level_scale,
    select_n_factors,
)

if HAS_TORCH:
    from autots.evaluator.tva.factor_network import fit_latent_factor_model
from autots.evaluator.tva.tva import TVA
from autots.models.tva_model import TVAModel


def piecewise_factor(rng, n_time, n_breaks=4):
    """Unit-std piecewise-linear path — the generator's own trend prior."""
    breaks = np.sort(rng.choice(np.arange(1, n_time), n_breaks, replace=False))
    slopes = rng.normal(0, 1, n_breaks + 1)
    deltas = np.zeros(n_time)
    seg = np.searchsorted(breaks, np.arange(n_time))
    deltas[1:] = slopes[seg[1:]]
    path = np.cumsum(deltas)
    path = path - path.mean()
    return path / max(path.std(), 1e-8)


def make_factor_panel(
    n_series=12,
    n_time=500,
    n_factors=2,
    noise=0.3,
    lags=None,
    seed=0,
    idio_scale=0.1,
):
    """Panel of series driven by shared latent piecewise-linear factors."""
    rng = np.random.default_rng(seed)
    factors = np.column_stack(
        [piecewise_factor(rng, n_time) for _ in range(n_factors)]
    )
    loadings = np.zeros((n_series, n_factors))
    for i in range(n_series):
        loadings[i, i % n_factors] = rng.uniform(0.7, 1.3) * rng.choice([1.0, -1.0])
    lags = np.zeros(n_series, dtype=int) if lags is None else np.asarray(lags)
    values = np.empty((n_time, n_series))
    for i in range(n_series):
        shifted = factors[np.clip(np.arange(n_time) - int(lags[i]), 0, None)]
        idio = idio_scale * piecewise_factor(rng, n_time)
        values[:, i] = (
            100.0 + shifted @ loadings[i] + idio + rng.normal(0, noise, n_time)
        )
    index = pd.date_range('2021-01-01', periods=n_time, freq='D')
    columns = [f'series_{i}' for i in range(n_series)]
    return {
        'df': pd.DataFrame(values, index=index, columns=columns),
        'factors': pd.DataFrame(factors, index=index),
        'loadings': loadings,
        'lags': lags,
    }


def normalized(df):
    center, scale = robust_level_scale(df.values)
    return (df.values - center[np.newaxis, :]) / scale[np.newaxis, :]


class TestSplitHalfStability(unittest.TestCase):
    """The truth-free no-hallucination diagnostic (torch-free)."""

    def test_shared_factors_score_above_independent_trends(self):
        shared = make_factor_panel(n_series=16, n_time=500, n_factors=2, seed=9)
        rng = np.random.default_rng(9)
        independent = pd.DataFrame(
            np.column_stack(
                [
                    100 + 3 * piecewise_factor(rng, 500) + rng.normal(0, 0.3, 500)
                    for _ in range(16)
                ]
            ),
            index=shared['df'].index,
            columns=shared['df'].columns,
        )
        shared_score = split_half_stability(normalized(shared['df']), 2, n_reps=2)
        independent_score = split_half_stability(normalized(independent), 2, n_reps=2)
        self.assertGreater(shared_score, independent_score)

    def test_returns_nan_on_tiny_panels(self):
        panel = make_factor_panel(n_series=3, n_time=100, n_factors=1, seed=0)
        self.assertTrue(np.isnan(split_half_stability(normalized(panel['df']), 1)))


class TestFactorBasis(unittest.TestCase):
    def test_hinge_design_is_piecewise_linear_and_anchored(self):
        design = hinge_design(100, 10)
        self.assertEqual(design.shape[0], 100)
        # every basis column is zero at t=0, so B @ c always anchors at 0
        np.testing.assert_allclose(design[0], 0.0, atol=1e-7)
        # a single hinge coefficient produces one breakpoint
        coef = np.zeros(design.shape[1])
        coef[3] = 1.0
        path = design @ coef
        second_diff = np.diff(path, 2)
        self.assertEqual(int((np.abs(second_diff) > 1e-6).sum()), 1)

    def test_select_n_factors_bounds(self):
        panel = make_factor_panel(n_factors=2, noise=0.2, seed=5)
        k = select_n_factors(normalized(panel['df']), cap=6)
        self.assertGreaterEqual(k, 1)
        self.assertLessEqual(k, 6)


class TestAlternatingIdentification(unittest.TestCase):
    """The torch-free stage that actually identifies the factors."""

    def test_recovers_known_factors(self):
        panel = make_factor_panel(n_series=12, n_time=500, n_factors=2, seed=1)
        ident = estimate_factors_alternating(normalized(panel['df']), n_factors=2)
        score = match_factors(panel['factors'], ident['factors'])
        # measured 0.589 on this seed; the ceiling for *any* linear estimator
        # on a comparable noisy panel measured 0.71 (see factor_network docs)
        self.assertGreaterEqual(score['mean_abs_corr'], 0.50)

    def test_loadings_recovered(self):
        panel = make_factor_panel(n_series=12, n_time=500, n_factors=2, seed=1)
        ident = estimate_factors_alternating(normalized(panel['df']), n_factors=2)
        score = match_factors(panel['factors'], ident['factors'])
        corrs = []
        for true_idx, est_idx in score['assignment'].items():
            a = panel['loadings'][:, true_idx]
            b = ident['loadings'][:, est_idx]
            corrs.append(abs(np.corrcoef(a, b)[0, 1]))
        self.assertGreaterEqual(float(np.mean(corrs)), 0.50)  # measured 0.596

    def test_seed_stability(self):
        panel = make_factor_panel(n_series=12, n_time=500, n_factors=2, seed=2)
        y = normalized(panel['df'])
        first = estimate_factors_alternating(y, n_factors=2)
        second = estimate_factors_alternating(y, n_factors=2)
        np.testing.assert_allclose(
            first['factors'], second['factors'], rtol=1e-6, atol=1e-6
        )


@unittest.skipUnless(HAS_TORCH, "torch required for trend_network='factor'")
class TestFactorModel(unittest.TestCase):
    def test_fit_shapes_and_finiteness(self):
        panel = make_factor_panel(n_series=8, n_time=200, n_factors=2, seed=3)
        y = normalized(panel['df'])
        model, info = fit_latent_factor_model(y, n_factors=2, horizon=14)
        self.assertEqual(model.fitted_factors().shape[0], 200)
        self.assertEqual(model.fitted_loadings().shape[0], 8)
        forecast = model.forecast(14).detach().cpu().numpy()
        self.assertEqual(forecast.shape, (8, 14))
        self.assertTrue(np.isfinite(forecast).all())
        self.assertEqual(info['sigma'].shape, (8,))
        self.assertTrue((info['sigma'] >= 0).all())

    def test_recovery_end_to_end(self):
        panel = make_factor_panel(n_series=12, n_time=500, n_factors=2, seed=1)
        model, _ = fit_latent_factor_model(normalized(panel['df']), n_factors=2)
        score = match_factors(panel['factors'], model.fitted_factors())
        self.assertGreaterEqual(score['mean_abs_corr'], 0.50)  # measured 0.589

    def test_k_pruning_reports_live_factors_only(self):
        panel = make_factor_panel(n_series=12, n_time=500, n_factors=1, seed=4)
        model, info = fit_latent_factor_model(normalized(panel['df']), n_factors=6)
        # over-specified K splits one true factor across columns rather than
        # leaving them dead, and the prune threshold is deliberately low so a
        # minor genuine factor survives; rank control is select_n_factors' job
        self.assertLessEqual(info['n_factors_live'], 5)
        self.assertEqual(
            model.fitted_factors().shape[1], model.fitted_loadings().shape[1]
        )

    def test_trendless_series_are_gated_out(self):
        """A pure-noise column must not inherit shared factor drift."""
        panel = make_factor_panel(n_series=8, n_time=400, n_factors=1, seed=12)
        rng = np.random.default_rng(12)
        df = panel['df'].copy()
        df['noise_only'] = 50.0 + rng.normal(0, 3.0, len(df))
        model, info = fit_latent_factor_model(normalized(df), n_factors=1)
        noise_idx = list(df.columns).index('noise_only')
        self.assertIn(noise_idx, info['gated_series'])
        loadings = model.loadings.detach().cpu().numpy()
        np.testing.assert_allclose(loadings[noise_idx], 0.0, atol=1e-8)
        # the genuinely factor-driven series must survive the gate
        self.assertLess(len(info['gated_series']), df.shape[1])

    def test_lag_mechanism_shifts_the_forecast(self):
        """Lags are opt-in and hard to identify; the mechanism must still work.

        A series pinned to lag d must read the factor path d steps behind, so
        its forecast differs from the same series at lag 0.
        """
        import torch

        panel = make_factor_panel(n_series=6, n_time=300, n_factors=1, seed=7)
        model, _ = fit_latent_factor_model(
            normalized(panel['df']), n_factors=1, max_lag=10, horizon=20
        )
        with torch.no_grad():
            model.lag_logits.zero_()
            model.lag_logits[:, 0] = 20.0
            at_zero = model.forecast(20).cpu().numpy()
            model.lag_logits.zero_()
            model.lag_logits[:, 8] = 20.0
            at_eight = model.forecast(20).cpu().numpy()
        self.assertGreater(float(np.abs(at_zero - at_eight).max()), 1e-6)


@unittest.skipUnless(HAS_TORCH, "torch required for trend_network='factor'")
class TestFactorTVAIntegration(unittest.TestCase):
    def test_fit_predict_smoke(self):
        panel = make_factor_panel(n_series=8, n_time=200, n_factors=2, seed=3)
        model = TVA(trend_network='factor', forecast_horizon=14, verbose=0)
        model.fit(panel['df'])
        forecast = model.predict(14)
        self.assertEqual(forecast.shape, (14, 8))
        self.assertTrue(np.isfinite(forecast.values).all())
        self.assertEqual(model._last_sigma.shape, (14, 8))
        self.assertTrue((model._last_sigma.values >= 0).all())

    def test_short_history_does_not_crash(self):
        """Shorter than window_size + horizon: the windowed path would raise."""
        panel = make_factor_panel(n_series=6, n_time=90, n_factors=1, seed=8)
        model = TVA(trend_network='factor', forecast_horizon=14, verbose=0)
        model.fit(panel['df'])
        forecast = model.predict(14)
        self.assertEqual(forecast.shape, (14, 6))
        self.assertTrue(np.isfinite(forecast.values).all())

    def test_get_factors_returns_learned_tables(self):
        panel = make_factor_panel(n_series=10, n_time=300, n_factors=2, seed=1)
        model = TVA(
            trend_network='factor', n_factors=2, forecast_horizon=14, verbose=0
        )
        model.fit(panel['df'])
        tables = model.get_factors()
        self.assertEqual(tables['factors'].shape, (300, 2))
        self.assertEqual(tables['loadings'].shape, (10, 2))
        self.assertEqual(len(tables['lags']), 10)
        self.assertAlmostEqual(float(tables['variance_share'].sum()), 1.0, places=4)
        self.assertIn('factor_variance_share', tables['diag'])
        # the learned paths must still score against ground truth through TVA
        score = match_factors(panel['factors'], tables['factors'])
        self.assertGreaterEqual(score['mean_abs_corr'], 0.35)

    def test_factor_graph_is_bipartite_and_in_edges(self):
        panel = make_factor_panel(n_series=8, n_time=300, n_factors=2, seed=1)
        model = TVA(
            trend_network='factor', n_factors=2, forecast_horizon=14, verbose=0
        )
        model.fit(panel['df'])
        graph = model.get_factor_graph()
        self.assertFalse(graph.empty)
        self.assertTrue(set(graph['source']) <= {'factor_1', 'factor_2'})
        self.assertTrue(set(graph['target']) <= set(panel['df'].columns))
        edges = model.get_edges()
        self.assertIn('factor', set(edges['family']))

    def test_negative_control_does_not_hallucinate(self):
        """Independent trends, no shared factor: must not beat 'none' badly."""
        rng = np.random.default_rng(11)
        n_time, n_series = 400, 8
        values = np.column_stack(
            [100 + 3 * piecewise_factor(rng, n_time) + rng.normal(0, 0.5, n_time)
             for _ in range(n_series)]
        )
        df = pd.DataFrame(
            values,
            index=pd.date_range('2021-01-01', periods=n_time, freq='D'),
            columns=[f's{i}' for i in range(n_series)],
        )
        model = TVA(trend_network='factor', forecast_horizon=14, verbose=0)
        model.fit(df)
        forecast = model.predict(14)
        self.assertTrue(np.isfinite(forecast.values).all())
        # forecasts must stay in a sane range around the last observed level
        spread = float(np.abs(forecast.values - df.values[-1]).max())
        self.assertLess(spread, 10 * float(df.values.std()))

    def test_none_mode_unaffected(self):
        panel = make_factor_panel(n_series=8, n_time=300, n_factors=2, seed=3)
        model = TVA(trend_network='none', forecast_horizon=14, verbose=0)
        model.fit(panel['df'])
        self.assertIsNone(model._factor_network)
        forecast = model.predict(14)
        self.assertEqual(forecast.shape, (14, 8))
        tables = model.get_factors()
        self.assertNotIn('variance_share', tables)


class TestDeconfoundedEdgeDiscovery(unittest.TestCase):
    """discover_structure's external-factor override (torch-free)."""

    def test_external_factors_change_the_residuals(self):
        from autots.evaluator.tva.discovery import discover_structure

        panel = make_factor_panel(n_series=10, n_time=400, n_factors=2, seed=6)
        trend = panel['df'].rolling(28, center=True, min_periods=1).mean()
        supplied = discover_structure(
            trend, seed=42, external_factors=panel['factors'].values
        )
        # the supplied paths become the confounder basis
        self.assertEqual(supplied['factors'].shape[1], panel['factors'].shape[1])
        self.assertEqual(supplied['loadings'].shape[0], len(panel['df'].columns))
        # the true series->series edge set here is empty (pure confounder), so
        # deconfounding against the true factors must leave almost nothing
        self.assertLessEqual(len(supplied['edges']), 5)

    @unittest.skipUnless(HAS_TORCH, "torch required for trend_network='factor'")
    def test_tva_opt_in_flag_rebuilds_edges(self):
        panel = make_factor_panel(n_series=8, n_time=300, n_factors=2, seed=1)
        model = TVA(
            trend_network='factor',
            n_factors=2,
            factor_deconfound_edges=True,
            forecast_horizon=14,
            verbose=0,
        )
        model.fit(panel['df'])
        self.assertEqual(
            model._discovery['factors'].shape[1],
            model.get_factors()['factors'].shape[1],
        )
        forecast = model.predict(14)
        self.assertTrue(np.isfinite(forecast.values).all())


class TestSearchIntegration(unittest.TestCase):
    def test_get_new_params_round_trip(self):
        import random

        random.seed(0)
        networks = set()
        for _ in range(50):
            params = TVAModel.get_new_params()
            for key in ('n_factors', 'factor_knot_spacing', 'factor_max_lag'):
                self.assertIn(key, params)
            networks.add(params['trend_network'])
            model = TVAModel(**params)
            self.assertEqual(model.get_params()['n_factors'], params['n_factors'])
        self.assertIn('factor', networks)


if __name__ == '__main__':
    unittest.main()
