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
from autots.evaluator.tva import sparse_factor

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


class TestSignedTopK(unittest.TestCase):
    """C9: the sparsifier. Signs are load-bearing, so this is not a ReLU."""

    def test_keeps_exactly_k_nonzeros_per_row(self):
        z = np.array([[1.0, -3.0, 2.0], [-5.0, 1.0, 1.0]])
        for k in (1, 2):
            out = sparse_factor.signed_topk(z, k)
            np.testing.assert_array_equal((np.abs(out) > 0).sum(axis=1), [k, k])

    def test_preserves_sign_of_the_selected_entries(self):
        z = np.array([[1.0, -3.0, 2.0]])
        out = sparse_factor.signed_topk(z, 1)
        # a non-negative activation would rectify this to +3 and hand the
        # coherence graph the wrong group ('f1+' instead of 'f1-')
        self.assertEqual(out[0, 1], -3.0)

    def test_exact_ties_still_yield_exactly_k(self):
        # a `>= kth value` implementation admits all of these; at init many
        # codes are exactly zero, so the failure is not hypothetical
        z = np.zeros((3, 4))
        out = sparse_factor.signed_topk(z, 2)
        self.assertEqual(int((np.abs(out) > 0).sum()), 0)
        z2 = np.ones((2, 4))
        self.assertTrue((np.abs(sparse_factor.signed_topk(z2, 2)) > 0).sum() == 4)

    def test_k_at_or_above_width_is_the_identity(self):
        z = np.array([[1.0, -3.0, 2.0]])
        np.testing.assert_array_equal(sparse_factor.signed_topk(z, 3), z)
        np.testing.assert_array_equal(sparse_factor.signed_topk(z, 9), z)

    def test_non_positive_k_zeroes_everything(self):
        z = np.array([[1.0, -3.0, 2.0]])
        np.testing.assert_array_equal(sparse_factor.signed_topk(z, 0), np.zeros_like(z))


class TestSparseIdentification(unittest.TestCase):
    """C9 tier 1: torch-free sparse dictionary learning over series."""

    @staticmethod
    def _panel(seed=0, n_per=8, K=3, T=400, noise=0.05):
        """Simple-structure panel whose true factors are near-orthogonal.

        The orthogonalization is load-bearing for a *hard assignment* test.
        Raw ``cumsum`` random walks at T=400 routinely correlate 0.7+ with each
        other (measured: 0.745 at seed 1) and differ 5x in scale; when two true
        factors are that close, folding their series onto one atom is a
        defensible answer rather than a failure, so an un-orthogonalized
        fixture measures the data instead of the estimator.
        """
        rng = np.random.default_rng(seed)
        factors = np.cumsum(rng.normal(size=(T, K)), axis=0)
        factors = factors - factors.mean(axis=0)
        factors, _ = np.linalg.qr(factors)          # decorrelate
        factors = factors / factors.std(axis=0)     # equalize amplitude
        n = n_per * K
        loadings = np.zeros((n, K))
        for i in range(n):
            sign = 1.0 if (i % 3) else -1.0
            loadings[i, i // n_per] = sign * (0.5 + rng.random())
        values = factors @ loadings.T + noise * rng.normal(size=(T, n))
        return values, factors, loadings

    def _identify(self, seed=0, K=3, fit_k=None, config=None, noise=0.05):
        values, factors, loadings = self._panel(seed=seed, K=K, noise=noise)
        fit_k = int(fit_k or K)
        init = estimate_factors_alternating(values, n_factors=fit_k)
        out = sparse_factor.identify(
            values, fit_k, init, 'sparse_alt', config=config, alpha=1e-3
        )
        return out, values, factors, loadings

    def test_returns_the_identification_contract(self):
        out, values, _, _ = self._identify()
        for key in ('factors', 'loadings', 'coefs', 'design', 'weights'):
            self.assertIn(key, out)
        t_len, n_series = values.shape
        self.assertEqual(out['design'].shape[0], t_len)
        self.assertEqual(out['loadings'].shape[0], n_series)
        self.assertEqual(out['coefs'].shape[0], out['design'].shape[1])
        self.assertEqual(out['coefs'].shape[1], out['factors'].shape[1])

    def test_design_matches_the_models_own_basis(self):
        out, values, _, _ = self._identify()
        np.testing.assert_array_equal(
            out['design'], hinge_design(values.shape[0], 7).astype(float)
        )

    def test_factors_keep_the_unit_increment_convention(self):
        out, _, _, _ = self._identify()
        # loadings are only commensurate with info['sigma'] under this exit
        # convention; unit-l2 atoms would rescale every precision weight.
        # An atom nobody uses is a zero path and is exempt by construction.
        live = np.asarray(out['atom_usage']) > 0
        np.testing.assert_allclose(
            np.diff(out['factors'], axis=0).std(axis=0)[live], 1.0, atol=1e-6
        )

    def test_loadings_are_sparse_and_signed(self):
        out, _, _, _ = self._identify()
        counts = (np.abs(out['loadings']) > 0).sum(axis=1)
        self.assertTrue((counts <= 1).all())
        self.assertTrue((out['loadings'] < 0).any())

    def test_dead_atoms_select_the_true_rank(self):
        # fit rank 5 on a rank-3 panel: atoms nobody uses fall out, which is
        # the rank selection C7 was going to be built for
        for seed in (0, 1, 2):
            out, _, _, _ = self._identify(seed=seed, fit_k=5)
            self.assertEqual(
                out['n_atoms_live'], 3,
                f'rank selection wrong on seed {seed}: usage={out["atom_usage"]}',
            )
            self.assertEqual(int((np.asarray(out['atom_usage']) == 0).sum()), 2)

    def test_atom_usage_is_a_trustworthy_rank_diagnostic(self):
        """Usage reports the rank, including when the fit has collapsed.

        Measured on this fixture: 4 of 5 seeds recover all three factors at
        both fitted ranks 4 and 5. The remaining seed reproducibly merges two
        *orthogonal* true factors onto one atom regardless of the restart
        schedule, and the acceptance guard does NOT catch it, because the
        merged fit reconstructs marginally better than the correct one.
        ``n_atoms_live`` is the only thing that surfaces this, which is why it
        is carried all the way out to ``info``.
        """
        live = []
        for seed in (0, 1, 2, 3, 4):
            out, _, _, _ = self._identify(seed=seed, fit_k=4)
            usage = np.asarray(out['atom_usage'])
            self.assertEqual(out['n_atoms_live'], int((usage >= 2).sum()))
            live.append(out['n_atoms_live'])
        self.assertGreaterEqual(sum(x == 3 for x in live), 4)
        self.assertTrue(all(x >= 2 for x in live), f'total collapse: {live}')

    def test_recovers_dominant_structure_better_than_the_initializer(self):
        from autots.evaluator.tva.metrics import loading_structure_score

        gains = []
        for seed in (0, 1, 2):
            values, factors, loadings = self._panel(seed=seed)
            init = estimate_factors_alternating(values, n_factors=4)
            out = sparse_factor.identify(values, 4, init, 'sparse_alt', alpha=1e-3)
            before = loading_structure_score(
                loadings, init['loadings'], factors, init['factors']
            )['matched_loading_corr']
            after = loading_structure_score(
                loadings, out['loadings'], factors, out['factors']
            )['matched_loading_corr']
            gains.append(after - before)
        # measured ~0.08-0.35 depending on seed; bar set below the observed
        # mean, following this file's stated convention
        self.assertGreater(float(np.mean(gains)), 0.05)

    def test_rejected_fit_falls_back_to_the_initializer(self):
        values, _, _ = self._panel()
        init = estimate_factors_alternating(values, n_factors=3)
        out = sparse_factor.identify(
            values, 3, init, 'sparse_alt', config={'accept_margin': 0.0}, alpha=1e-3
        )
        self.assertTrue(out['sparse_rejected'])
        np.testing.assert_array_equal(out['loadings'], init['loadings'])
        np.testing.assert_array_equal(out['coefs'], init['coefs'])

    def test_min_code_share_abstains_with_an_all_zero_row(self):
        # k=1 makes coherence's dominance_margin and min_loading_share
        # unconditional (runner-up is exactly 0, share exactly 1.0), so an
        # all-zero row is the only abstention channel the graph still honors
        values, _, _ = self._panel()
        rng = np.random.default_rng(1)
        # a pure-noise series at the panel's own scale has no honest exposure
        noise_col = rng.normal(scale=float(values.std()), size=(len(values), 1))
        values = np.column_stack([values, noise_col])
        init = estimate_factors_alternating(values, n_factors=3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            base = sparse_factor.identify(
                values, 3, init, 'sparse_alt',
                config={'min_code_share': 0.0, 'accept_margin': 1e9}, alpha=1e-3,
            )
            out = sparse_factor.identify(
                values, 3, init, 'sparse_alt',
                config={'min_code_share': 0.9, 'accept_margin': 1e9}, alpha=1e-3,
            )
        n_zero_base = int((np.abs(base['loadings']).max(axis=1) <= 0).sum())
        n_zero = int((np.abs(out['loadings']).max(axis=1) <= 0).sum())
        self.assertGreater(n_zero, n_zero_base)

    def test_degenerate_panels_never_raise(self):
        rng = np.random.default_rng(0)
        cases = [
            np.zeros((50, 4)),
            np.column_stack([np.ones(50), rng.normal(size=(50, 3))]),
            rng.normal(size=(5, 3)),
        ]
        for values in cases:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                init = estimate_factors_alternating(values, n_factors=2)
                try:
                    sparse_factor.identify(values, 2, init, 'sparse_alt', alpha=1e-3)
                except Exception as exc:  # pragma: no cover - the assertion
                    self.fail(f'sparse identification raised on {values.shape}: {exc}')

    def test_a_broken_panel_falls_back_instead_of_raising(self):
        # the estimator upstream raises on all-NaN; identify() must absorb
        # anything that reaches it and hand back the initializer
        values = np.full((40, 3), np.nan)
        init = {'factors': np.zeros((40, 2)), 'loadings': np.zeros((3, 2)),
                'coefs': np.zeros((6, 2)), 'design': np.zeros((40, 6)),
                'weights': np.ones(3)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = sparse_factor.identify(values, 2, init, 'sparse_alt', alpha=1e-3)
        self.assertTrue(out is None or out.get('sparse_rejected'))


class TestLevelShiftVeto(unittest.TestCase):
    """I1: the detector is univariate, so a shared factor move looks like N shifts."""

    @staticmethod
    def _shifts(n_series=20, n_time=40):
        idx = pd.date_range('2021-01-01', periods=n_time)
        return pd.DataFrame(
            0.0, index=idx, columns=[f's{i}' for i in range(n_series)]
        )

    def test_a_panel_wide_step_is_returned_to_the_panel(self):
        shifts = self._shifts()
        shifts.iloc[15:, :] = 5.0
        out = TVA._veto_shared_shifts(shifts, {})
        np.testing.assert_allclose(out.to_numpy(), 0.0, atol=1e-9)

    def test_a_lone_step_is_still_removed(self):
        shifts = self._shifts()
        shifts.iloc[15:, 3] = 7.0
        out = TVA._veto_shared_shifts(shifts, {})
        self.assertAlmostEqual(float(out.iloc[-1, 3]), 7.0, places=6)

    def test_idiosyncratic_excess_survives_a_shared_step(self):
        shifts = self._shifts()
        shifts.iloc[15:, :] = 5.0     # shared
        shifts.iloc[25:, 0] += 3.0    # this series' own, on top
        out = TVA._veto_shared_shifts(shifts, {})
        self.assertAlmostEqual(float(out.iloc[-1, 0]), 3.0, places=6)
        self.assertAlmostEqual(float(out.iloc[-1, 1]), 0.0, places=6)

    def test_disagreeing_directions_are_not_one_event(self):
        shifts = self._shifts()
        shifts.iloc[15:, :10] = 5.0
        shifts.iloc[15:, 10:] = -5.0
        out = TVA._veto_shared_shifts(shifts, {})
        # half up, half down is not a shared factor move; nothing is vetoed
        self.assertAlmostEqual(float(out.iloc[-1, 1]), 5.0, places=6)

    def test_shrink_scales_how_much_is_returned(self):
        shifts = self._shifts()
        shifts.iloc[15:, :] = 5.0
        out = TVA._veto_shared_shifts(shifts, {'level_shift_veto_shrink': 0.5})
        self.assertAlmostEqual(float(out.iloc[-1, 1]), 2.5, places=6)

    def test_veto_all_returns_every_shift(self):
        shifts = self._shifts()
        shifts.iloc[15:, 3] = 7.0   # a lone step, which the normal veto keeps
        out = TVA._veto_shared_shifts(shifts, {'level_shift_veto': 'all'})
        np.testing.assert_allclose(out.to_numpy(), 0.0, atol=1e-9)

    def test_the_initial_level_is_not_dropped(self):
        # rebuilding from cumsum of the differences loses shifts.iloc[0],
        # which turns the veto into a per-series constant -- and a constant is
        # removed exactly by robust_level_scale's median centering, so the fit
        # would be bit-identical while the panel looked changed
        shifts = self._shifts()
        shifts.iloc[:, :] = 4.0     # a standing offset, no step anywhere
        out = TVA._veto_shared_shifts(shifts, {})
        np.testing.assert_allclose(out.to_numpy(), 4.0, atol=1e-9)

    def test_too_few_series_is_a_no_op(self):
        shifts = self._shifts(n_series=2)
        shifts.iloc[15:, :] = 5.0
        pd.testing.assert_frame_equal(TVA._veto_shared_shifts(shifts, {}), shifts)


@unittest.skipUnless(factor_network.HAS_TORCH, 'torch required')
class TestSparseIdentificationThroughTheModel(unittest.TestCase):
    """C9 reaching the object it targets: fitted_loadings(), not ident['loadings']."""

    @staticmethod
    def _panel(seed=0, n_per=6, K=2, T=300):
        rng = np.random.default_rng(seed)
        factors = np.cumsum(rng.normal(size=(T, K)), axis=0)
        n = n_per * K
        loadings = np.zeros((n, K))
        for i in range(n):
            loadings[i, i // n_per] = (1.0 if i % 2 else -1.0) * (0.5 + rng.random())
        return factors @ loadings.T + 0.05 * rng.normal(size=(T, n))

    def _fit(self, config=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return fit_latent_factor_model(
                self._panel(), n_factors=3, horizon=14, config=config
            )

    def test_default_is_bitwise_identical_to_today(self):
        model_a, _ = self._fit()
        model_b, _ = self._fit({
            'identification': 'alternating',
            'sparse_config': None,
            'sparse_freeze_support': True,
            'level_shift_veto': False,
        })
        # raw parameter, not fitted_loadings(): pruning can change the shape
        np.testing.assert_array_equal(
            model_a.loadings.detach().numpy(), model_b.loadings.detach().numpy()
        )

    def test_sparse_support_survives_stage_a(self):
        model, info = self._fit({'identification': 'sparse_alt'})
        counts = (np.abs(model.loadings.detach().numpy()) > 0).sum(axis=1)
        self.assertTrue(
            (counts <= 1).all(),
            'stage A washed the identified zeros out before the graph reads them',
        )
        self.assertEqual(info['identification_method'], 'sparse_alt')

    def test_without_the_projection_stage_a_destroys_the_support(self):
        # the executable form of the hazard: w_l1_loadings is a subgradient
        # term that never reaches zero, so 600 Adam steps refill every entry
        model, _ = self._fit({
            'identification': 'sparse_alt', 'sparse_freeze_support': False,
        })
        counts = (np.abs(model.loadings.detach().numpy()) > 0).sum(axis=1)
        self.assertTrue((counts > 1).any())

    def test_torch_free_fallback_records_the_tier_it_ran(self):
        from unittest import mock

        with mock.patch.object(sparse_factor, 'HAS_TORCH', False):
            _, info = self._fit({'identification': 'sparse_ae'})
        self.assertEqual(info['identification_method'], 'sparse_alt')

    def test_live_atom_count_lands_in_info(self):
        _, info = self._fit({'identification': 'sparse_alt'})
        self.assertEqual(info['n_atoms_live'], 2)  # true rank, fitted at 3
        self.assertEqual(len(info['atom_usage']), 3)


@unittest.skipUnless(factor_network.HAS_TORCH, 'torch required')
class TestSparseKnobsThroughTVA(unittest.TestCase):
    """The C9/I1 knobs are no-ops at their defaults, end to end."""

    @staticmethod
    def _frame(seed=0, n_per=6, K=2, T=280):
        rng = np.random.default_rng(seed)
        factors = np.cumsum(rng.normal(size=(T, K)), axis=0)
        n = n_per * K
        loadings = np.zeros((n, K))
        for i in range(n):
            loadings[i, i // n_per] = (1.0 if i % 2 else -1.0) * (0.5 + rng.random())
        values = factors @ loadings.T + 0.05 * rng.normal(size=(T, n))
        return pd.DataFrame(
            values,
            index=pd.date_range('2021-01-01', periods=T),
            columns=[f's{i}' for i in range(n)],
        )

    def _fit(self, factor_config=None):
        model = TVA(
            trend_network='factor', forecast_horizon=14, random_seed=0,
            verbose=0, factor_config=factor_config,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(self._frame())
        return model

    def test_all_new_knobs_default_to_current_behavior(self):
        base = self._fit()
        explicit = self._fit({
            'identification': 'alternating',
            'sparse_config': None,
            'sparse_freeze_support': True,
            'level_shift_veto': False,
            'level_shift_veto_shrink': 1.0,
        })
        np.testing.assert_array_equal(
            base._factor_network.loadings.detach().numpy(),
            explicit._factor_network.loadings.detach().numpy(),
        )

    def test_sparse_identification_reaches_the_coherence_loadings(self):
        model = self._fit({'identification': 'sparse_alt'})
        lam = model._factor_network.fitted_loadings(0.02)
        counts = (np.abs(lam) > 0).sum(axis=1)
        self.assertTrue((counts <= 1).all())

    def test_sparse_forecast_still_runs_and_stays_close(self):
        base = self._fit()
        sparse = self._fit({'identification': 'sparse_alt'})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fa = base.predict(14)
            fb = sparse.predict(14)
        self.assertEqual(fa.shape, fb.shape)
        rel = np.abs(fa.values - fb.values).mean() / max(np.abs(fa.values).mean(), 1e-9)
        self.assertLess(rel, 0.25)

    def test_veto_changes_the_input_panel_only_when_enabled(self):
        base = self._fit()
        same = self._fit({'level_shift_veto': False})
        np.testing.assert_array_equal(base._adjusted_raw, same._adjusted_raw)
        self.assertIsNone(same._shift_returned)

    def test_veto_all_changes_the_panel_and_stays_reconstruction_symmetric(self):
        base = self._fit()
        vetoed = self._fit({'level_shift_veto': 'all'})
        self.assertIsNotNone(vetoed._shift_returned)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fa = base.predict(14)
            fb = vetoed.predict(14)
        # whatever the veto leaves in the trend must NOT be added back a second
        # time at predict; asymmetry here showed up as a +59% MASE blowup
        rel = np.abs(fa.values - fb.values).mean() / max(np.abs(fa.values).mean(), 1e-9)
        self.assertLess(rel, 0.25)
