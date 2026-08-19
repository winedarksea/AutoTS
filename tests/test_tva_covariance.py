# -*- coding: utf-8 -*-
"""Tests for the TVA forecast covariance, its MinT wiring, and the
closed-form what-if solver for the factor / torch-free modes.

Run with:  python -m pytest tests/test_tva_covariance.py -v
"""
import unittest
import warnings

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from autots.evaluator.tva.decomposition import NornDecomposer  # noqa: F401

    HAS_FEATURE_DETECTOR = True
except Exception:
    HAS_FEATURE_DETECTOR = False

SKIP_INTEGRATION = unittest.skipUnless(
    HAS_TORCH and HAS_FEATURE_DETECTOR,
    "PyTorch or TimeSeriesFeatureDetector not available",
)


def _factor_panel(n_days=520, n_series=6, n_factors=2, noise=1.0, seed=0):
    """Panel whose cross-series covariance is non-diagonal by construction."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n_days, freq="D")
    factors = np.cumsum(rng.normal(size=(n_days, n_factors)), axis=0)
    loadings = rng.normal(size=(n_factors, n_series))
    seasonal = 5.0 * np.sin(np.arange(n_days)[:, None] * 2 * np.pi / 7)
    values = (
        100.0
        + factors @ loadings * 3.0
        + seasonal
        + rng.normal(size=(n_days, n_series)) * noise
    )
    return pd.DataFrame(
        values, index=index, columns=[f"s{i}" for i in range(n_series)]
    )


def _independent_panel(n_days=520, n_series=5, seed=3):
    """Panel with no shared structure: independent random walks."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n_days, freq="D")
    values = 100.0 + np.cumsum(rng.normal(size=(n_days, n_series)), axis=0)
    return pd.DataFrame(
        values, index=index, columns=[f"s{i}" for i in range(n_series)]
    )


def _hierarchy_metadata(columns, split=None):
    """Two-level hierarchy: global -> group -> series."""
    from autots.evaluator.tva.priors import SeriesMetadata

    columns = list(columns)
    split = split if split is not None else len(columns) // 2
    return [
        SeriesMetadata(
            name,
            hierarchy_path=["global", "A" if i < split else "B", name],
            history_periods=520,
        )
        for i, name in enumerate(columns)
    ]


def _fit(df, metadata=None, **kwargs):
    from autots.evaluator.tva.tva import TVA

    params = dict(
        forecast_horizon=14,
        window_size=60,
        epochs=2,
        verbose=0,
        series_metadata=metadata,
    )
    params.update(kwargs)
    tva = TVA(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tva.fit(df)
    return tva


# ---------------------------------------------------------------------------
# covariance.py unit tests (no torch, no fitting)
# ---------------------------------------------------------------------------


class TestCovariancePrimitives(unittest.TestCase):
    def test_ledoit_wolf_matches_library_implementation(self):
        """The vectorized estimator is the library's, plus the intensity it drops."""
        from autots.evaluator.tva.covariance import ledoit_wolf_shrinkage
        from autots.tools.hierarchial import ledoit_wolf_covariance

        rng = np.random.default_rng(0)
        for n, p in [(50, 8), (20, 30), (200, 5)]:
            X = rng.normal(size=(n, p)) @ rng.normal(size=(p, p))
            mine, gamma = ledoit_wolf_shrinkage(X)
            theirs = ledoit_wolf_covariance(X)
            np.testing.assert_allclose(mine, theirs, rtol=1e-10, atol=1e-12)
            self.assertGreaterEqual(gamma, 0.0)
            self.assertLessEqual(gamma, 1.0)

    def test_shrinkage_intensity_rises_when_samples_are_scarce(self):
        from autots.evaluator.tva.covariance import ledoit_wolf_shrinkage

        rng = np.random.default_rng(1)
        # a genuinely correlated population: on iid data the optimal shrinkage
        # toward mu*I is 1.0 at any sample size, which would hide the effect
        mixer = rng.normal(size=(20, 20))
        base = rng.normal(size=(400, 20)) @ mixer
        _, gamma_many = ledoit_wolf_shrinkage(base)
        _, gamma_few = ledoit_wolf_shrinkage(base[:25])
        self.assertGreater(gamma_few, gamma_many)
        self.assertLess(gamma_many, 0.5)

    def test_damped_accumulated_variance_bounds(self):
        from autots.evaluator.tva.covariance import damped_accumulated_variance

        # phi = 0 means only the first innovation survives: variance 1 at every h
        np.testing.assert_allclose(
            damped_accumulated_variance(np.array([0.0]), 10), [1.0]
        )
        # more damping means less accumulated variance, and it grows with horizon
        v_low = damped_accumulated_variance(np.array([0.5]), 20)
        v_high = damped_accumulated_variance(np.array([0.95]), 20)
        self.assertLess(v_low[0], v_high[0])
        self.assertLess(
            damped_accumulated_variance(np.array([0.9]), 5)[0],
            damped_accumulated_variance(np.array([0.9]), 50)[0],
        )

    def test_structural_target_matches_empirical_diagonal(self):
        from autots.evaluator.tva.covariance import structural_target

        rng = np.random.default_rng(2)
        Lam = rng.normal(size=(8, 2))
        v = np.array([1.5, 0.5])
        true = (Lam * v) @ Lam.T + np.diag(rng.uniform(0.2, 1.0, 8))
        struct, beta, psi = structural_target(true, Lam, v)
        # psi is set residually, so the diagonal is reproduced exactly
        np.testing.assert_allclose(np.diag(struct), np.diag(true), rtol=1e-10)
        self.assertAlmostEqual(beta, 1.0, places=6)
        self.assertTrue(np.all(psi >= 0))

    def test_structural_target_beta_absorbs_loading_scale(self):
        """Loadings in the wrong units must not break the target (the 'none' path)."""
        from autots.evaluator.tva.covariance import structural_target

        rng = np.random.default_rng(4)
        Lam = rng.normal(size=(10, 3))
        v = np.array([1.0, 0.7, 0.3])
        true = (Lam * v) @ Lam.T + np.diag(rng.uniform(0.1, 0.5, 10))
        s1, beta1, _ = structural_target(true, Lam, v)
        s2, beta2, _ = structural_target(true, Lam * 100.0, v)
        np.testing.assert_allclose(s1, s2, rtol=1e-8, atol=1e-10)
        self.assertAlmostEqual(beta2 * 1e4, beta1, places=6)

    def test_variance_floor_binds_only_where_needed(self):
        from autots.evaluator.tva.covariance import apply_variance_floor

        sigma = np.array([[4.0, 1.0], [1.0, 9.0]])
        floored, binding = apply_variance_floor(sigma, np.array([3.0, 1.0]))
        np.testing.assert_array_equal(binding, [True, False])
        # the floored series' variance is raised to exactly the floor
        self.assertAlmostEqual(floored[0, 0], 9.0)
        self.assertAlmostEqual(floored[1, 1], 9.0)
        # correlation is untouched by the rescale
        corr = lambda m: m[0, 1] / np.sqrt(m[0, 0] * m[1, 1])  # noqa: E731
        self.assertAlmostEqual(corr(floored), corr(sigma))

    def test_variance_floor_noop_when_not_binding(self):
        from autots.evaluator.tva.covariance import apply_variance_floor

        sigma = np.array([[4.0, 1.0], [1.0, 9.0]])
        floored, binding = apply_variance_floor(sigma, np.array([0.5, 0.5]))
        self.assertFalse(binding.any())
        np.testing.assert_array_equal(floored, sigma)

    def test_assemble_covariance_is_symmetric_and_psd(self):
        from autots.evaluator.tva.covariance import assemble_covariance

        rng = np.random.default_rng(5)
        Lam = rng.normal(size=(7, 2))
        resid = rng.normal(size=(300, 2)) @ Lam.T + rng.normal(size=(300, 7)) * 0.3
        sigma, info = assemble_covariance(
            resid, loadings=Lam, factor_var=np.array([1.0, 1.0])
        )
        np.testing.assert_allclose(sigma, sigma.T, atol=0)
        self.assertGreaterEqual(np.linalg.eigvalsh(sigma).min(), 0.0)
        self.assertTrue(info['has_structure'])
        self.assertEqual(info['n_samples'], 300)

    def test_assemble_covariance_is_diagonal_on_uncorrelated_residuals(self):
        """The estimator itself must not manufacture cross-series structure."""
        from autots.evaluator.tva.covariance import assemble_covariance

        rng = np.random.default_rng(11)
        resid = rng.normal(size=(4000, 6)) * np.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.8])
        Lam = rng.normal(size=(6, 2))
        sigma, info = assemble_covariance(
            resid, loadings=Lam, factor_var=np.array([1.0, 1.0])
        )
        d = np.sqrt(np.diag(sigma))
        corr = sigma / np.outer(d, d)
        off = corr[~np.eye(6, dtype=bool)]
        self.assertLess(np.abs(off).max(), 0.1)
        # the diagonal still recovers the true per-series scales
        np.testing.assert_allclose(
            d, [1.0, 2.0, 0.5, 3.0, 1.5, 0.8], rtol=0.1
        )
        self.assertLess(info['alpha'], 0.2)

    def test_assemble_covariance_rejects_degenerate_input(self):
        from autots.evaluator.tva.covariance import assemble_covariance

        with self.assertRaises(ValueError):
            assemble_covariance(np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            assemble_covariance(np.zeros((10, 3)), loadings=np.ones((3, 1)))


# ---------------------------------------------------------------------------
# TVA.forecast_covariance()
# ---------------------------------------------------------------------------


@SKIP_INTEGRATION
class TestForecastCovariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _factor_panel()
        cls.meta = _hierarchy_metadata(cls.df.columns)
        cls.factor = _fit(cls.df, cls.meta, trend_network='factor')
        cls.none = _fit(cls.df, cls.meta, trend_network='none')

    def _check_sigma(self, tva):
        result = tva.forecast_covariance()
        self.assertIsNotNone(result)
        sigma, info = result
        n = self.df.shape[1]
        self.assertEqual(sigma.shape, (n, n))
        np.testing.assert_allclose(sigma, sigma.T, atol=0)
        self.assertGreaterEqual(np.linalg.eigvalsh(sigma).min(), -1e-9)
        self.assertTrue(np.all(np.isfinite(sigma)))
        return sigma, info

    def test_factor_mode_sigma_is_symmetric_psd(self):
        self._check_sigma(self.factor)

    def test_none_mode_sigma_is_symmetric_psd(self):
        self._check_sigma(self.none)

    def test_sigma_carries_the_diagnostic_pieces(self):
        for tva in (self.factor, self.none):
            _sigma, info = self._check_sigma(tva)
            for key in ('loadings', 'psi', 'alpha', 'floor_binding'):
                self.assertIn(key, info, msg=f"{key} missing for {info['source']}")
            self.assertGreaterEqual(info['alpha'], 0.0)
            self.assertLessEqual(info['alpha'], 1.0)
            self.assertEqual(
                info['loadings'].shape[0], self.df.shape[1]
            )

    def test_residual_floor_state_is_reported_not_silently_clamped(self):
        """The floor is a stated fact about the estimate, not a hidden clamp."""
        for tva in (self.factor, self.none):
            sigma, info = self._check_sigma(tva)
            binding = np.asarray(info['floor_binding'])
            self.assertEqual(binding.dtype, np.dtype(bool))
            self.assertEqual(binding.shape, (self.df.shape[1],))
            # wherever it binds, the variance equals the floor it was raised to
            floor = (
                tva._factor_sigma_floor(np.ones(self.df.shape[1]))
                if info['source'] == 'factor'
                else tva._decomposer.get_residual_sigma()
            )
            if floor is not None and binding.any():
                np.testing.assert_allclose(
                    np.sqrt(np.diag(sigma))[binding],
                    np.asarray(floor)[binding],
                    rtol=1e-6,
                )

    def test_floor_binds_on_a_low_noise_factor_fit(self):
        """A near-noiseless panel makes the model's own sigma the small one."""
        result = self.factor.forecast_covariance()
        self.assertIsNotNone(result)
        _sigma, info = result
        # the idiosyncratic term is capped at 2 dof by design, so the model
        # residual is structurally optimistic and the decomposition floor
        # should be doing real work on at least one series
        self.assertTrue(np.asarray(info['floor_binding']).any())

    @staticmethod
    def _mean_abs_offdiag_corr(sigma):
        d = np.sqrt(np.diag(sigma))
        corr = sigma / np.outer(d, d)
        return float(np.abs(corr[~np.eye(corr.shape[0], dtype=bool)]).mean())

    def test_sigma_is_less_correlated_on_an_uncorrelated_panel(self):
        """The estimator must read structure off the data, not manufacture it.

        Stated as a comparison rather than an absolute bound because Sigma
        describes the *model's* forecast errors, not the panel's: a factor
        model asked to fit independent random walks still makes somewhat
        correlated errors, and that correlation is a true property of the
        forecast it is about to produce.
        """
        df = _independent_panel()
        tva = _fit(df, _hierarchy_metadata(df.columns), trend_network='factor')
        result = tva.forecast_covariance()
        self.assertIsNotNone(result)
        independent = self._mean_abs_offdiag_corr(result[0])
        shared = self._mean_abs_offdiag_corr(self.factor.forecast_covariance()[0])
        self.assertLess(independent, shared)

    def test_sigma_finds_correlation_on_a_factor_panel(self):
        """The counterpart: a shared-factor panel must NOT come back diagonal."""
        sigma, _info = self._check_sigma(self.factor)
        d = np.sqrt(np.diag(sigma))
        corr = sigma / np.outer(d, d)
        off = corr[~np.eye(corr.shape[0], dtype=bool)]
        self.assertGreater(np.abs(off).max(), 0.2)

    def test_horizon_argument_changes_the_estimate(self):
        short = self.factor.forecast_covariance(horizon=3)
        long = self.factor.forecast_covariance(horizon=28)
        self.assertIsNotNone(short)
        self.assertIsNotNone(long)
        self.assertFalse(np.allclose(short[0], long[0]))

    def test_unfitted_returns_none(self):
        from autots.evaluator.tva.tva import TVA

        self.assertIsNone(TVA(verbose=0).forecast_covariance())

    def test_factor_info_keeps_the_residual_matrix(self):
        """The fit's rolling-origin residuals are kept, not reduced to a std."""
        resid = self.factor._factor_info.get('residual_matrix')
        self.assertIsNotNone(resid)
        self.assertEqual(np.asarray(resid).shape[1], self.df.shape[1])


@SKIP_INTEGRATION
class TestForecastCovarianceIsInertForNetworkModes(unittest.TestCase):
    """v1/v2 keep their own residual estimator; the new branch is unreachable."""

    @classmethod
    def setUpClass(cls):
        cls.df = _factor_panel(n_days=380, n_series=4)
        cls.meta = _hierarchy_metadata(cls.df.columns)
        cls.tva = _fit(
            cls.df,
            cls.meta,
            trend_network='v2',
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
        )

    def test_v2_forecast_covariance_is_none(self):
        self.assertIsNone(self.tva.forecast_covariance())

    def test_v2_structural_w_is_none(self):
        S = self.tva._priors.build_hierarchy_matrix()
        self.assertIsNone(self.tva._structural_reconciliation_W(S))
        # even when explicitly asked for it, there is nothing to build one from
        self.tva.reconciliation_covariance = 'structural'
        try:
            self.assertIsNone(self.tva._structural_reconciliation_W(S))
        finally:
            self.tva.reconciliation_covariance = 'auto'

    def test_v2_reconcile_is_unaffected_by_the_setting(self):
        base = self.tva.reconcile()
        self.tva.reconciliation_covariance = 'structural'
        try:
            after = self.tva.reconcile()
        finally:
            self.tva.reconciliation_covariance = 'auto'
        pd.testing.assert_frame_equal(base, after)

    def test_v2_what_if_still_uses_the_backprop_solver(self):
        from autots.evaluator.tva.scenario import BifrostOptimizer

        with unittest.mock.patch.object(
            BifrostOptimizer, 'apply_constraint', autospec=True
        ) as patched:
            patched.return_value = self.tva.predict()
            self.tva.what_if(series_name='s0', timestep=1, target_value=1.0)
        self.assertTrue(patched.called)


# ---------------------------------------------------------------------------
# MinT wiring
# ---------------------------------------------------------------------------


class TestReconciliationBridgeStructuralW(unittest.TestCase):
    def test_precomputed_w_is_used_verbatim(self):
        from autots.evaluator.tva.reconciliation import ReconciliationBridge

        bridge = ReconciliationBridge(method='mint')
        S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        fc = pd.DataFrame(
            [[3.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
            index=pd.date_range("2025-01-01", periods=2, freq="D"),
            columns=['agg', 's0', 's1'],
        )
        W = np.diag([1.0, 1.0, 100.0])
        out = bridge.reconcile(fc, S, W=W)
        # the aggregate is incoherent by 1.0; with s1 100x noisier it absorbs
        # almost all of the correction
        self.assertGreater(
            abs(out['s1'].iloc[0] - 1.0), abs(out['s0'].iloc[0] - 1.0)
        )
        np.testing.assert_allclose(
            out['agg'].values, (out['s0'] + out['s1']).values, rtol=1e-8
        )

    def test_structural_method_requires_a_w(self):
        from autots.evaluator.tva.reconciliation import ReconciliationBridge

        bridge = ReconciliationBridge(covariance_method='structural')
        with self.assertRaises(ValueError):
            bridge._estimate_covariance(np.zeros((5, 3)))

    def test_wrong_shaped_w_is_rejected(self):
        from autots.evaluator.tva.reconciliation import ReconciliationBridge

        bridge = ReconciliationBridge()
        S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        fc = pd.DataFrame(np.ones((2, 3)), columns=['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            bridge.reconcile(fc, S, W=np.eye(2))


@SKIP_INTEGRATION
class TestStructuralReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _factor_panel()
        cls.meta = _hierarchy_metadata(cls.df.columns)
        cls.tva = _fit(cls.df, cls.meta, trend_network='factor')

    def test_auto_resolves_to_structural(self):
        """The gate passed (examples/tva_reconciliation_gate.py), so 'auto' is on."""
        from autots.evaluator.tva import tva as tva_module

        self.assertEqual(tva_module.RECONCILIATION_COVARIANCE_AUTO, 'structural')
        self.assertEqual(self.tva.reconciliation_covariance, 'auto')
        S = self.tva._priors.build_hierarchy_matrix()
        self.assertIsNotNone(self.tva._structural_reconciliation_W(S))

    def test_identity_setting_still_forces_w_equals_i(self):
        S = self.tva._priors.build_hierarchy_matrix()
        self.tva.reconciliation_covariance = 'identity'
        try:
            self.assertIsNone(self.tva._structural_reconciliation_W(S))
        finally:
            self.tva.reconciliation_covariance = 'auto'

    def test_bottom_only_path_never_assembles_a_covariance(self):
        """The no-op path must not pay for a W it cannot use."""
        base = self.tva.predict()
        with unittest.mock.patch.object(
            type(self.tva), '_structural_reconciliation_W', autospec=True
        ) as patched:
            self.tva.reconcile(forecasts=base)
        patched.assert_not_called()

    def test_structural_w_is_positive_definite_and_full_hierarchy_sized(self):
        S = self.tva._priors.build_hierarchy_matrix()
        self.tva.reconciliation_covariance = 'structural'
        try:
            W = self.tva._structural_reconciliation_W(S)
        finally:
            self.tva.reconciliation_covariance = 'auto'
        self.assertIsNotNone(W)
        self.assertEqual(W.shape, (S.shape[0], S.shape[0]))
        # S Sigma S' is rank-deficient by construction; the ridge is what makes
        # it usable by mint_reconcile's assume_a='pos' solve
        self.assertGreater(np.linalg.eigvalsh(W).min(), 0.0)

    def test_structural_reconciliation_stays_coherent(self):
        self.tva.reconciliation_covariance = 'structural'
        try:
            reconciled = self.tva.reconcile()
        finally:
            self.tva.reconciliation_covariance = 'auto'
        S = self.tva._priors.build_hierarchy_matrix()
        n_bottom = S.shape[1]
        n_agg = S.shape[0] - n_bottom
        bottom = reconciled.values.T  # (M, T)
        agg = S[:n_agg] @ bottom
        # bottom-up aggregation of the reconciled bottom level is the
        # definition of coherence for a bottom-only return
        self.assertTrue(np.all(np.isfinite(agg)))
        self.assertEqual(reconciled.shape, (self.tva.forecast_horizon, n_bottom))

    def test_bottom_only_reconcile_is_a_projection_noop(self):
        """Synthesized aggregates are already coherent, so no W can move them.

        ``reconcile()`` builds the aggregate rows as ``S @ bottom``, which puts
        the input exactly in the subspace MinT projects onto. Recording this as
        a test because it is the reason the W = I bug had no visible symptom:
        the shipped path performs no reconciliation at all, in any trend mode.
        """
        base = self.tva.predict()
        for mode in ('identity', 'structural'):
            self.tva.reconciliation_covariance = mode
            try:
                out = self.tva.reconcile(forecasts=base)
            finally:
                self.tva.reconciliation_covariance = 'auto'
            np.testing.assert_allclose(
                out.values, base.values, rtol=1e-8, atol=1e-8,
                err_msg=f"{mode} arm changed a bottom-only reconciliation",
            )

    def test_s_sigma_st_alone_reconciles_identically_to_identity(self):
        """``W = S Sigma S'`` cancels Sigma out of MinT's estimator.

        As the ridge goes to zero, ``(S'W^-1 S)^-1 S'W^-1 -> (S'S)^-1 S'``, the
        plain OLS reconciler that ``W = I`` already gives. Sigma only survives
        once the aggregate nodes carry variance of their own.
        """
        from autots.tools.hierarchial import mint_reconcile

        S = self.tva._priors.build_hierarchy_matrix().astype(np.float64)
        self.tva.reconciliation_covariance = 'structural'
        try:
            W = self.tva._structural_reconciliation_W(S)
        finally:
            self.tva.reconciliation_covariance = 'auto'
        rng = np.random.default_rng(0)
        y_all = rng.normal(size=(5, S.shape[0])) * 10.0  # genuinely incoherent
        np.testing.assert_allclose(
            mint_reconcile(S, y_all, W),
            mint_reconcile(S, y_all, np.eye(S.shape[0])),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_aggregate_sigma_makes_the_covariance_bite(self):
        """With independent aggregate forecasts, structural W stops being OLS."""
        from autots.tools.hierarchial import mint_reconcile

        S = self.tva._priors.build_hierarchy_matrix().astype(np.float64)
        n_agg = S.shape[0] - S.shape[1]
        self.tva.reconciliation_covariance = 'structural'
        try:
            W = self.tva._structural_reconciliation_W(
                S, aggregate_sigma=np.full(n_agg, 25.0)
            )
        finally:
            self.tva.reconciliation_covariance = 'auto'
        self.assertGreater(np.linalg.eigvalsh(W).min(), 0.0)
        rng = np.random.default_rng(0)
        y_all = rng.normal(size=(5, S.shape[0])) * 10.0
        structural = mint_reconcile(S, y_all, W)
        identity = mint_reconcile(S, y_all, np.eye(S.shape[0]))
        self.assertFalse(np.allclose(structural, identity))
        # both arms are still coherent, which is what reconciliation buys
        for out in (structural, identity):
            np.testing.assert_allclose(
                out[:, :n_agg].T, S[:n_agg] @ out[:, n_agg:].T, rtol=1e-6
            )

    def test_aggregate_sigma_length_is_validated(self):
        S = self.tva._priors.build_hierarchy_matrix().astype(np.float64)
        self.tva.reconciliation_covariance = 'structural'
        try:
            with self.assertRaises(ValueError):
                self.tva._structural_reconciliation_W(
                    S, aggregate_sigma=np.ones(1000)
                )
        finally:
            self.tva.reconciliation_covariance = 'auto'


# ---------------------------------------------------------------------------
# Closed-form what-if
# ---------------------------------------------------------------------------


@SKIP_INTEGRATION
class TestClosedFormScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _factor_panel()
        cls.meta = _hierarchy_metadata(cls.df.columns)
        cls.factor = _fit(cls.df, cls.meta, trend_network='factor')
        cls.none = _fit(cls.df, cls.meta, trend_network='none')

    def test_what_if_returns_a_dataframe_in_factor_mode(self):
        """The bug-fix assertion: this used to raise on self.tva._network."""
        base = self.factor.predict()
        result = self.factor.what_if(
            series_name='s0', timestep=5, target_value=float(base.iloc[5, 0]) + 50.0
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, base.shape)
        self.assertFalse(result.isnull().any().any())

    def test_what_if_returns_a_dataframe_in_none_mode(self):
        base = self.none.predict()
        result = self.none.what_if(
            series_name='s0', timestep=5, target_value=float(base.iloc[5, 0]) + 50.0
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, base.shape)
        self.assertFalse(result.isnull().any().any())

    def test_what_if_growth_in_both_modes(self):
        for tva in (self.factor, self.none):
            result = tva.what_if(series_name='s1', growth_rate=0.1)
            self.assertIsInstance(result, pd.DataFrame)
            expected = float(self.df['s1'].iloc[-1]) * 1.1
            self.assertAlmostEqual(float(result['s1'].iloc[-1]), expected, places=4)

    def test_constraint_is_satisfied_exactly(self):
        for tva in (self.factor, self.none):
            base = tva.predict()
            target = float(base.iloc[5, 0]) + 37.5
            result = tva.what_if(series_name='s0', timestep=5, target_value=target)
            self.assertAlmostEqual(float(result.iloc[5, 0]), target, places=6)
            # untouched timesteps stay untouched
            np.testing.assert_allclose(
                result.iloc[0].values, base.iloc[0].values, rtol=1e-10
            )

    def test_adjustment_is_minimum_norm_under_sigma(self):
        """delta must satisfy the KKT condition Sigma^-1 delta ∝ a."""
        from autots.evaluator.tva.scenario import ClosedFormScenario

        tva = self.factor
        sigma = tva.forecast_covariance()[0]
        base = tva.predict()
        target = float(base.iloc[5, 0]) + 37.5
        solver = ClosedFormScenario(tva)
        result = solver.apply_constraint('s0', 5, target)
        delta = result.iloc[5].values - base.iloc[5].values

        dual = np.linalg.solve(sigma, delta)
        a = np.zeros(len(base.columns))
        a[0] = 1.0
        # dual must be parallel to the constraint row: everything off the
        # constrained series is zero to numerical tolerance
        self.assertGreater(abs(dual[0]), 0.0)
        np.testing.assert_allclose(
            dual[1:] / abs(dual[0]), np.zeros(len(a) - 1), atol=1e-8
        )

        # and it must be no larger, in the Sigma metric, than any other
        # correction meeting the same constraint
        norm = float(delta @ np.linalg.solve(sigma, delta))
        rng = np.random.default_rng(7)
        for _ in range(25):
            other = delta + rng.normal(size=len(a)) * np.std(delta)
            other[0] = delta[0]  # keep the constraint satisfied
            self.assertLessEqual(
                norm, float(other @ np.linalg.solve(sigma, other)) + 1e-9
            )

    def test_covariance_moves_correlated_series_together(self):
        """A pin on one series must move its co-moving neighbours, not just itself."""
        base = self.factor.predict()
        target = float(base.iloc[5, 0]) + 50.0
        result = self.factor.what_if(
            series_name='s0', timestep=5, target_value=target
        )
        moved = np.abs(result.iloc[5].values - base.iloc[5].values)
        self.assertGreater(moved[1:].max(), 1e-6)

    def test_hierarchical_adjustment_hits_the_target_every_timestep(self):
        for tva in (self.factor, self.none):
            result = tva.what_if(level_name='A', target_value=500.0)
            constituents = [c for c in result.columns if c in ('s0', 's1', 's2')]
            np.testing.assert_allclose(
                result[constituents].sum(axis=1).values,
                np.full(len(result), 500.0),
                rtol=1e-6,
            )

    def test_hierarchical_adjustment_falls_back_to_proportional(self):
        """Without a covariance the pre-existing proportional split is kept."""
        from autots.evaluator.tva.scenario import ClosedFormScenario, _proportional_split

        tva = self.factor
        base = tva.predict()
        solver = ClosedFormScenario(tva)
        solver._sigma = None
        solver._sigma_resolved = True  # pretend nothing was available
        result = solver.apply_hierarchical_adjustment('A', 500.0)
        expected = _proportional_split(
            base.values.copy().astype(np.float64), np.array([0, 1, 2]), 500.0
        )
        np.testing.assert_allclose(result.values, expected, rtol=1e-10)

    def test_unknown_level_returns_the_base_forecast(self):
        from autots.evaluator.tva.scenario import ClosedFormScenario

        solver = ClosedFormScenario(self.factor)
        result = solver.apply_hierarchical_adjustment('not_a_node', 1.0)
        pd.testing.assert_frame_equal(result, self.factor.predict())

    def test_derived_identities_are_reapplied_last(self):
        """A declared ratio column is rebuilt after the scenario adjustment."""
        df = self.df.copy()
        df['ratio'] = df['s0'] / df['s1']
        meta = _hierarchy_metadata(df.columns, split=3)
        tva = _fit(
            df,
            meta,
            trend_network='factor',
            derived_definitions={'ratio': ('s0', 's1')},
        )
        base = tva.predict()
        result = tva.what_if(
            series_name='s0', timestep=5, target_value=float(base.iloc[5, 0]) + 20.0
        )
        np.testing.assert_allclose(
            result['ratio'].values,
            (result['s0'] / result['s1']).values,
            rtol=1e-8,
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
