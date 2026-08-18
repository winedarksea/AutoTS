# -*- coding: utf-8 -*-
"""Tests for the learned latent-factor trend mode (``trend_network='factor'``).

Thresholds are calibrated empirically on the fixed seeds used here and set
below the observed values, following the convention of the rest of the TVA
suite. Everything that needs torch is skipped when torch is unavailable.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from autots.evaluator.tva import factor_network

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


class TestRotateIdentification(unittest.TestCase):
    """C1: resolving the rotational indeterminacy of the identification basis.

    The load-bearing property is that a rotation is a *re-parameterization*:
    ``factors @ loadings.T`` must survive it bit-for-bit-ish, so accuracy and
    every span metric are invariant and only basis-dependent structure moves.
    """

    @staticmethod
    def _panel(seed=0, n_per=8, K=3, T=400):
        """Simple-structure latent-factor panel: each series loads one factor."""
        rng = np.random.default_rng(seed)
        factors = np.cumsum(rng.normal(size=(T, K)), axis=0)
        n = n_per * K
        loadings = np.zeros((n, K))
        for i in range(n):
            sign = 1.0 if (i % 3) else -1.0
            loadings[i, i // n_per] = sign * (0.5 + rng.random())
        values = factors @ loadings.T + 0.05 * rng.normal(size=(T, n))
        return values, factors, loadings

    def _ident(self, seed=0, K=3):
        values, factors, loadings = self._panel(seed=seed, K=K)
        ident = factor_network.estimate_factors_alternating(values, n_factors=K)
        return ident, loadings

    def test_reconstruction_is_invariant(self):
        for method in ('varimax', 'quartimax', 'promax'):
            ident, _ = self._ident()
            rotated = factor_network.rotate_identification(ident, method=method)
            before = ident['factors'] @ ident['loadings'].T
            after = rotated['factors'] @ rotated['loadings'].T
            self.assertLess(
                np.abs(before - after).max(), 1e-8,
                f'{method} changed the reconstruction',
            )

    def test_coefs_still_generate_the_rotated_factors(self):
        # the torch model is initialized from coefs, not factors: if the two
        # disagree the rotation silently doesn't reach the fitted model
        ident, _ = self._ident()
        rotated = factor_network.rotate_identification(ident, method='varimax')
        # _l1_trend_filter returns column-centered paths, so the invariant is
        # up to that centering -- which the rotation preserves because
        # centering is linear and the same map is applied to both objects.
        regenerated = rotated['design'] @ rotated['coefs']
        regenerated = regenerated - regenerated.mean(axis=0, keepdims=True)
        self.assertLess(
            np.abs(regenerated - rotated['factors']).max(), 1e-6
        )

    def test_unit_std_increments(self):
        ident, _ = self._ident()
        rotated = factor_network.rotate_identification(ident, method='varimax')
        sd = np.std(np.diff(rotated['factors'], axis=0), axis=0)
        np.testing.assert_allclose(sd, np.ones_like(sd), atol=1e-8)

    def test_mass_vote_orientation_is_baked_in(self):
        ident, _ = self._ident()
        rotated = factor_network.rotate_identification(ident, method='varimax')
        lam = rotated['loadings']
        mass = (lam * np.abs(lam)).sum(axis=0)
        self.assertTrue(np.all(mass >= 0), 'columns left with negative mass')

    def test_matched_loading_correlation_improves(self):
        from autots.evaluator.tva.metrics import loading_structure_score

        gains = []
        for seed in (0, 1, 2):
            ident, true_loadings = self._ident(seed=seed)
            rotated = factor_network.rotate_identification(ident, method='varimax')
            base = loading_structure_score(true_loadings, ident['loadings'])
            rot = loading_structure_score(true_loadings, rotated['loadings'])
            gains.append(
                abs(rot['matched_loading_corr']) - abs(base['matched_loading_corr'])
            )
            self.assertGreater(rot['dominant_recovery'], 0.9)
        self.assertGreater(float(np.mean(gains)), 0.0)

    def test_unknown_method_and_degenerate_inputs_pass_through(self):
        ident, _ = self._ident()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertIs(
                factor_network.rotate_identification(ident, method='nope'), ident
            )
        self.assertIs(factor_network.rotate_identification(ident, method=None), ident)
        self.assertIs(factor_network.rotate_identification(None, 'varimax'), None)
        single = factor_network.estimate_factors_alternating(
            self._panel(K=1)[0], n_factors=1
        )
        rotated = factor_network.rotate_identification(single, method='varimax')
        self.assertEqual(rotated['loadings'].shape, single['loadings'].shape)

    @unittest.skipUnless(factor_network.HAS_TORCH, 'torch required')
    def test_rotate_none_is_bitwise_identical_to_today(self):
        values, _, _ = self._panel(seed=3)
        kwargs = dict(n_factors=3, horizon=28, seed=11)
        model_a, _ = factor_network.fit_latent_factor_model(values, **kwargs)
        model_b, _ = factor_network.fit_latent_factor_model(
            values, config={'rotate': None}, **kwargs
        )
        np.testing.assert_array_equal(
            model_a.fitted_loadings(), model_b.fitted_loadings()
        )

    @unittest.skipUnless(factor_network.HAS_TORCH, 'torch required')
    def test_rotation_reaches_the_fitted_model(self):
        """The rotated basis must survive stage A into ``fitted_loadings``.

        Stage A refines loadings by gradient descent from the rotated
        initialization; if the rotation were applied after the parameter copy
        (or washed out by the refinement) the graph would still be built on
        the arbitrary basis.
        """
        values, _, _ = self._panel(seed=3)
        kwargs = dict(n_factors=3, horizon=28, seed=11)
        model_a, _ = factor_network.fit_latent_factor_model(values, **kwargs)
        model_b, info_b = factor_network.fit_latent_factor_model(
            values, config={'rotate': 'varimax'}, **kwargs
        )
        # raw parameters, not fitted_loadings(): pruning can drop a different
        # number of columns in each basis, which is a genuine consequence of
        # the rotation but makes the shapes incomparable here
        lam_a = model_a.loadings.detach().cpu().numpy()
        lam_b = model_b.loadings.detach().cpu().numpy()
        self.assertGreater(np.abs(lam_a - lam_b).max(), 1e-4)
        # and it is the *identification's* rotated basis that came through
        np.testing.assert_allclose(
            lam_b, info_b['identification']['loadings'], atol=0.2
        )

    @unittest.skipUnless(factor_network.HAS_TORCH, 'torch required')
    def test_rotation_is_exact_at_model_initialization(self):
        """Where the invariance MUST be exact: the parameter copy.

        The rotation is applied between identification and the copy into the
        torch model, so a freshly-initialized model must reconstruct
        identically in either basis (to float32). Anything larger here is an
        algebra bug, not a fit difference.
        """
        import torch

        values, _, _ = self._panel(seed=3)
        ident = factor_network.estimate_factors_alternating(values, n_factors=3)
        rotated = factor_network.rotate_identification(ident, method='varimax')
        T, N = values.shape

        def initialized(source):
            model = factor_network.LatentFactorTrend(
                n_series=N, n_time=T, n_factors=3, knot_spacing=7,
                max_lag=14, slope_window=90,
            )
            with torch.no_grad():
                model.coef.copy_(torch.tensor(source['coefs'], dtype=torch.float32))
                model.loadings.copy_(
                    torch.tensor(source['loadings'], dtype=torch.float32)
                )
                model.idio_level.copy_(
                    torch.tensor(values.mean(axis=0), dtype=torch.float32)
                )
                return model().cpu().numpy()

        a, b = initialized(ident), initialized(rotated)
        rel = np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a ** 2)), 1e-9)
        self.assertLess(rel, 1e-5)

    @unittest.skipUnless(factor_network.HAS_TORCH, 'torch required')
    def test_reconstruction_drift_through_stage_a_stays_bounded(self):
        """After stage A the two bases may differ, but only slightly.

        Stage A descends from a different starting point under
        basis-dependent regularizers (``w_l1_loadings`` finally has sparse
        structure to preserve rather than fight) and can early-stop at a
        different step, so exact equality is not the claim. Boundedness is:
        a large drift here would mean the rotation changed what the model
        can represent, not merely how it is parameterized.
        """
        import torch

        values, _, _ = self._panel(seed=3)
        kwargs = dict(n_factors=3, horizon=28, seed=11)
        model_a, _ = factor_network.fit_latent_factor_model(values, **kwargs)
        model_b, _ = factor_network.fit_latent_factor_model(
            values, config={'rotate': 'varimax'}, **kwargs
        )
        with torch.no_grad():
            recon_a = model_a().cpu().numpy()
            recon_b = model_b().cpu().numpy()
        rel = np.sqrt(np.mean((recon_a - recon_b) ** 2)) / max(
            np.sqrt(np.mean(recon_a ** 2)), 1e-9
        )
        self.assertLess(rel, 0.10)


class TestSparseLoadingSolve(unittest.TestCase):
    """C3: an l1 loading solve as the identifying restriction lstsq lacks.

    Least squares is rotation-invariant, so it cannot prefer the true
    simple-structure loadings over any rotation of them. An l1 penalty can.
    """

    @staticmethod
    def _panel(seed=0, n_per=4, K=3, T=300, noise=0.05):
        rng = np.random.default_rng(seed)
        factors = np.cumsum(rng.normal(size=(T, K)), axis=0)
        n = n_per * K
        loadings = np.zeros((n, K))
        for i in range(n):
            loadings[i, i // n_per] = (1.0 if (i % 3) else -1.0) * (1.0 + rng.random())
        values = factors @ loadings.T + noise * rng.normal(size=(T, n))
        return values, loadings

    def test_l1_zero_is_bit_identical_to_lstsq(self):
        values, _ = self._panel()
        base = factor_network.estimate_factors_alternating(values, 3)
        same = factor_network.estimate_factors_alternating(
            values, 3, loading_l1=0.0
        )
        np.testing.assert_array_equal(base['loadings'], same['loadings'])
        np.testing.assert_array_equal(base['factors'], same['factors'])

    def test_fit_loadings_matches_lstsq_at_zero_penalty(self):
        rng = np.random.default_rng(1)
        F = rng.normal(size=(120, 3))
        Y = rng.normal(size=(120, 7))
        expected, *_ = np.linalg.lstsq(F, Y, rcond=None)
        np.testing.assert_array_equal(
            factor_network._fit_loadings(F, Y, l1=0.0), expected
        )

    def test_penalty_produces_exact_zeros(self):
        values, _ = self._panel()
        base = factor_network.estimate_factors_alternating(values, 3)
        sparse = factor_network.estimate_factors_alternating(
            values, 3, loading_l1=0.1
        )
        self.assertEqual(float((np.abs(base['loadings']) == 0).mean()), 0.0)
        self.assertGreater(float((np.abs(sparse['loadings']) == 0).mean()), 0.1)

    def test_final_solve_does_not_erase_the_sparsity(self):
        # the post-normalization solve at the end of the estimator is a dense
        # lstsq unless it too goes through _fit_loadings
        values, _ = self._panel()
        sparse = factor_network.estimate_factors_alternating(
            values, 3, loading_l1=0.1
        )
        self.assertGreater(float((np.abs(sparse['loadings']) == 0).mean()), 0.0)

    def test_relaxed_refit_is_unbiased_on_the_support(self):
        rng = np.random.default_rng(2)
        F = rng.normal(size=(400, 3))
        true = np.array([[3.0], [0.0], [0.0]])
        Y = F @ true + 0.01 * rng.normal(size=(400, 1))
        relaxed = factor_network._fit_loadings(F, Y, l1=0.05, relax=True)
        shrunk = factor_network._fit_loadings(F, Y, l1=0.05, relax=False)
        self.assertLess(abs(relaxed[0, 0] - 3.0), abs(shrunk[0, 0] - 3.0))
        self.assertEqual(relaxed[1, 0], 0.0)
        self.assertEqual(relaxed[2, 0], 0.0)

    def test_constant_series_gets_no_loading(self):
        rng = np.random.default_rng(4)
        F = rng.normal(size=(100, 2))
        Y = np.hstack([F @ np.array([[2.0], [0.0]]), np.zeros((100, 1))])
        out = factor_network._fit_loadings(F, Y, l1=0.05)
        np.testing.assert_array_equal(out[:, 1], np.zeros(2))

    @unittest.skipUnless(factor_network.HAS_TORCH, 'torch required')
    def test_prox_defaults_off_and_preserves_zeros_when_on(self):
        values, _ = self._panel()
        kwargs = dict(n_factors=3, horizon=28, seed=5)
        model_off, _ = factor_network.fit_latent_factor_model(
            values, config={'loading_l1': 0.1}, **kwargs
        )
        model_on, _ = factor_network.fit_latent_factor_model(
            values, config={'loading_l1': 0.1, 'w_prox_loadings': 1.0}, **kwargs
        )
        # threshold is w_prox_loadings * lr_aux per step, against loadings of
        # order 0.3 on a normalized panel -- so the useful range is ~O(1),
        # not the ~1e-3 the loading-magnitude-naive guess suggests
        zeros_off = float((model_off.loadings.detach().numpy() == 0).mean())
        zeros_on = float((model_on.loadings.detach().numpy() == 0).mean())
        self.assertGreater(zeros_on, zeros_off)

    @unittest.skipUnless(factor_network.HAS_TORCH, 'torch required')
    def test_prox_zero_is_identical_to_today(self):
        values, _ = self._panel()
        kwargs = dict(n_factors=3, horizon=28, seed=5)
        a, _ = factor_network.fit_latent_factor_model(values, **kwargs)
        b, _ = factor_network.fit_latent_factor_model(
            values, config={'w_prox_loadings': 0.0}, **kwargs
        )
        np.testing.assert_array_equal(
            a.loadings.detach().numpy(), b.loadings.detach().numpy()
        )


class TestSplitHalfFactorStability(unittest.TestCase):
    """C4: which *columns* are shared structure, not just whether any are."""

    @staticmethod
    def _panel(seed=0, n_per=8, K=3, T=400):
        rng = np.random.default_rng(seed)
        factors = np.cumsum(rng.normal(size=(T, K)), axis=0)
        n = n_per * K
        loadings = np.zeros((n, K))
        for i in range(n):
            loadings[i, i // n_per] = 1.0 + rng.random()
        return factors @ loadings.T + 0.05 * rng.normal(size=(T, n))

    def test_real_factors_score_high(self):
        out = factor_network.split_half_factor_stability(self._panel(), 3)
        self.assertEqual(out.shape, (3,))
        self.assertGreater(float(np.min(out)), 0.8)

    def test_overspecified_rank_lowers_the_weakest_columns(self):
        real = factor_network.split_half_factor_stability(self._panel(), 3)
        over = factor_network.split_half_factor_stability(self._panel(), 6)
        self.assertEqual(over.shape, (6,))
        self.assertLess(float(np.min(over)), float(np.min(real)))

    def test_degenerate_panels_return_nan_not_an_exception(self):
        out = factor_network.split_half_factor_stability(
            np.zeros((50, 2)), 2
        )
        self.assertEqual(out.shape, (2,))
        self.assertTrue(np.all(np.isnan(out)))

    def test_is_deterministic_for_a_seed(self):
        panel = self._panel()
        a = factor_network.split_half_factor_stability(panel, 3, seed=9)
        b = factor_network.split_half_factor_stability(panel, 3, seed=9)
        np.testing.assert_array_equal(a, b)

    @unittest.skipUnless(factor_network.HAS_TORCH, 'torch required')
    def test_reps_zero_leaves_info_key_none(self):
        values = self._panel()
        _, info = factor_network.fit_latent_factor_model(
            values, n_factors=3, horizon=28, seed=5
        )
        self.assertIsNone(info['factor_stability'])
        _, info_on = factor_network.fit_latent_factor_model(
            values, n_factors=3, horizon=28, seed=5,
            config={'factor_stability_reps': 2},
        )
        self.assertEqual(np.asarray(info_on['factor_stability']).shape, (3,))


@unittest.skipUnless(factor_network.HAS_TORCH, 'torch required')
class TestStructureLadderKnobsThroughTVA(unittest.TestCase):
    """Every ladder knob must be reachable from ``TVA(factor_config=...)``
    and must default to today's behavior.

    The ladder's whole discipline is that each candidate is independently
    killable by flipping one config value back, which only holds if the knob
    actually travels from the constructor to the estimator.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        T, K, n_per = 500, 3, 5
        factors = np.cumsum(rng.normal(size=(T, K)), axis=0)
        n = n_per * K
        loadings = np.zeros((n, K))
        for i in range(n):
            loadings[i, i // n_per] = (1.0 if (i % 3) else -1.0) * (1.0 + rng.random())
        values = 100.0 + factors @ loadings.T + 0.5 * rng.normal(size=(T, n))
        cls.df = pd.DataFrame(
            values,
            index=pd.date_range('2021-01-01', periods=T, freq='D'),
            columns=[f's{i}' for i in range(n)],
        )

    def _fit(self, factor_config=None, coherence_config=None):
        model = TVA(
            trend_network='factor', forecast_horizon=14, random_seed=42,
            verbose=0, factor_config=factor_config,
            coherence_config=coherence_config,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(self.df)
        return model

    def test_all_new_knobs_default_to_current_behavior(self):
        base = self._fit()
        inert = self._fit({
            'rotate': None,
            'rotate_kaiser': True,
            'loading_l1': 0.0,
            'w_prox_loadings': 0.0,
            'factor_stability_reps': 0,
            'structure_input': None,
        })
        np.testing.assert_array_equal(
            base._factor_network.fitted_loadings(),
            inert._factor_network.fitted_loadings(),
        )

    def test_rotate_changes_the_basis_but_not_the_forecast_much(self):
        base, rotated = self._fit(), self._fit({'rotate': 'varimax'})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fa = base.predict(14)
            fb = rotated.predict(14)
        self.assertGreater(
            np.abs(
                base._factor_network.loadings.detach().numpy()
                - rotated._factor_network.loadings.detach().numpy()
            ).max(), 1e-4,
        )
        rel = np.abs(fa.values - fb.values).mean() / max(
            np.abs(fa.values).mean(), 1e-9
        )
        self.assertLess(rel, 0.05)

    def test_structure_input_leaves_the_forecast_untouched(self):
        base = self._fit()
        two_track = self._fit({'structure_input': 'robust'})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fa = base.predict(14)
            fb = two_track.predict(14)
        # by construction: the second fit only produces a loading matrix
        np.testing.assert_allclose(fa.values, fb.values, rtol=1e-6, atol=1e-6)

    def test_factor_stability_lands_in_info(self):
        model = self._fit({'factor_stability_reps': 2})
        stability = model._factor_info['factor_stability']
        self.assertIsNotNone(stability)
        self.assertEqual(
            np.asarray(stability).shape[0], model._factor_network.K
        )
