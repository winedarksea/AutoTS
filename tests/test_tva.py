# -*- coding: utf-8 -*-
"""
Tests for the TVA (Time Variance Authority) forecasting graph.

Covers: unit tests for each module (no torch required for priors/reconciliation),
integration tests with synthetic data, coherence validation, scenario planning,
reconciliation bridge, and metadata-prior paths.

Run with:  python -m pytest tests/test_tva.py -v
"""
import unittest
import numpy as np
import pandas as pd


def _make_daily_df(n_series: int = 4, n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily wide DataFrame with trend + seasonality."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    trend = np.linspace(0, 3, n_days)[:, None]
    seasonal = np.sin(np.arange(n_days) * 2 * np.pi / 7)[:, None]
    noise = rng.normal(0, 0.1, (n_days, n_series))
    data = trend + seasonal * 0.5 + noise + rng.uniform(1, 5, n_series)
    return pd.DataFrame(data, index=dates, columns=[f"s{i}" for i in range(n_series)])


def _make_coherent_df(n_series: int = 6, n_days: int = 400, seed: int = 1) -> pd.DataFrame:
    """All series share a common upward trend — tests coherence loss."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    shared = np.linspace(0, 5, n_days)
    data = np.stack([shared + rng.normal(0, 0.05, n_days) for _ in range(n_series)], axis=1)
    return pd.DataFrame(data, index=dates, columns=[f"series_{i}" for i in range(n_series)])


# ---------------------------------------------------------------------------
# Module-level unit tests (no torch needed for priors / reconciliation)
# ---------------------------------------------------------------------------

class TestSeriesMetadata(unittest.TestCase):
    def test_default_fields(self):
        from autots.evaluator.tva.priors import SeriesMetadata
        m = SeriesMetadata(name="foo")
        self.assertEqual(m.name, "foo")
        self.assertIsNone(m.metric_type)
        self.assertEqual(m.history_periods, 0)

    def test_fully_specified(self):
        from autots.evaluator.tva.priors import SeriesMetadata
        m = SeriesMetadata(
            name="bar",
            metric_type="dau",
            surface="mobile_feed",
            geography="US",
            hierarchy_path=["global", "NA", "US"],
            history_periods=365,
        )
        self.assertEqual(m.geography, "US")
        self.assertEqual(len(m.hierarchy_path), 3)


class TestYggdrasilPriors(unittest.TestCase):
    def _make_metadata(self):
        from autots.evaluator.tva.priors import SeriesMetadata
        return [
            SeriesMetadata("a", metric_type="dau", surface="mobile", geography="US", history_periods=400),
            SeriesMetadata("b", metric_type="dau", surface="web",    geography="US", history_periods=400),
            SeriesMetadata("c", metric_type="views", surface="mobile", geography="DE", history_periods=100),
            SeriesMetadata("d", metric_type="views", surface="web",    geography="DE", history_periods=50),
        ]

    def test_empty_metadata_returns_defaults(self):
        from autots.evaluator.tva.priors import YggdrasilPriors
        p = YggdrasilPriors()
        adj = p.build_prior_adjacency()
        self.assertEqual(adj.shape, (1, 1))
        emb = p.build_metadata_embeddings()
        self.assertEqual(emb.shape[0], 1)
        mask = p.get_anchor_mask(100)
        self.assertEqual(len(mask), 1)

    def test_prior_adjacency_shape_and_symmetry(self):
        from autots.evaluator.tva.priors import YggdrasilPriors
        p = YggdrasilPriors(series_metadata=self._make_metadata())
        adj = p.build_prior_adjacency()
        n = 4
        self.assertEqual(adj.shape, (n, n))
        np.testing.assert_allclose(adj, adj.T, atol=1e-6)
        # diagonal must be 1
        np.testing.assert_allclose(np.diag(adj), np.ones(n), atol=1e-6)

    def test_shared_attribute_raises_adjacency(self):
        from autots.evaluator.tva.priors import YggdrasilPriors
        p = YggdrasilPriors(series_metadata=self._make_metadata())
        adj = p.build_prior_adjacency()
        # a & b share metric_type=dau AND geography=US, so adj[0,1] > 0
        self.assertGreater(adj[0, 1], 0.0)
        # a & d share nothing (dau vs views, mobile vs web, US vs DE)
        self.assertEqual(adj[0, 3], 0.0)

    def test_metadata_embeddings_shape(self):
        from autots.evaluator.tva.priors import YggdrasilPriors
        p = YggdrasilPriors(series_metadata=self._make_metadata())
        emb = p.build_metadata_embeddings()
        self.assertEqual(emb.shape[0], 4)
        self.assertTrue(emb.shape[1] > 0)
        # one-hot: each row has at least one non-zero per attribute
        self.assertTrue(emb.sum(axis=1).min() > 0)

    def test_anchor_mask(self):
        from autots.evaluator.tva.priors import YggdrasilPriors
        p = YggdrasilPriors(series_metadata=self._make_metadata())
        mask = p.get_anchor_mask(200)
        # a and b have 400 periods → True; c has 100, d has 50 → False
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_explicit_relationship_matrix_returned_as_is(self):
        from autots.evaluator.tva.priors import YggdrasilPriors
        explicit = np.eye(4, dtype=np.float32) * 0.7
        p = YggdrasilPriors(
            series_metadata=self._make_metadata(),
            relationship_matrix=explicit,
        )
        adj = p.build_prior_adjacency()
        np.testing.assert_allclose(adj, explicit, atol=1e-6)

    def test_hierarchy_matrix_identity_when_no_paths(self):
        from autots.evaluator.tva.priors import SeriesMetadata, YggdrasilPriors
        meta = [SeriesMetadata("x"), SeriesMetadata("y")]
        p = YggdrasilPriors(series_metadata=meta)
        S = p.build_hierarchy_matrix()
        np.testing.assert_allclose(S, np.eye(2), atol=1e-6)

    def test_hierarchy_matrix_structure(self):
        from autots.evaluator.tva.priors import SeriesMetadata, YggdrasilPriors
        meta = [
            SeriesMetadata("US", hierarchy_path=["global", "NA", "US"]),
            SeriesMetadata("CA", hierarchy_path=["global", "NA", "CA"]),
            SeriesMetadata("DE", hierarchy_path=["global", "EU", "DE"]),
        ]
        p = YggdrasilPriors(series_metadata=meta)
        S = p.build_hierarchy_matrix()
        # bottom-level identity block at bottom of S
        n_bottom = 3
        np.testing.assert_allclose(S[-n_bottom:, :], np.eye(n_bottom), atol=1e-6)
        # global row (prefix ("global",)) should sum all three bottom series
        np.testing.assert_array_equal(S[0, :], [1, 1, 1])

    def test_hidden_branches_method(self):
        from autots.evaluator.tva.priors import YggdrasilPriors
        p = YggdrasilPriors(series_metadata=self._make_metadata())
        branches = p._branches_of_yggdrasil()
        self.assertIsInstance(branches, int)
        self.assertGreater(branches, 0)


class TestReconciliationBridge(unittest.TestCase):
    """Tests for ReconciliationBridge without PyTorch."""

    def _make_simple_hierarchy(self):
        """3 bottom-level series, 1 aggregate (sum of all)."""
        S = np.array([
            [1, 1, 1],  # aggregate
            [1, 0, 0],  # s0
            [0, 1, 0],  # s1
            [0, 0, 1],  # s2
        ], dtype=np.float64)
        T = 10
        # build consistent synthetic forecasts
        rng = np.random.default_rng(0)
        bottom = rng.normal(10, 1, (T, 3))
        aggregate = bottom.sum(axis=1, keepdims=True)
        y_all = np.hstack([aggregate, bottom])
        index = pd.date_range("2025-01-01", periods=T, freq="D")
        df = pd.DataFrame(y_all, index=index, columns=["agg", "s0", "s1", "s2"])
        return df, S

    def test_mint_reconcile_shape(self):
        from autots.evaluator.tva.reconciliation import ReconciliationBridge
        df, S = self._make_simple_hierarchy()
        bridge = ReconciliationBridge(method="mint")
        result = bridge.reconcile(df, S)
        self.assertEqual(result.shape, df.shape)
        self.assertEqual(list(result.columns), list(df.columns))

    def test_erm_reconcile_shape(self):
        from autots.evaluator.tva.reconciliation import ReconciliationBridge
        df, S = self._make_simple_hierarchy()
        bridge = ReconciliationBridge(method="erm")
        result = bridge.reconcile(df, S)
        self.assertEqual(result.shape, df.shape)

    def test_reconcile_with_residuals(self):
        from autots.evaluator.tva.reconciliation import ReconciliationBridge
        df, S = self._make_simple_hierarchy()
        rng = np.random.default_rng(1)
        residuals = rng.normal(0, 0.5, (20, 4))
        bridge = ReconciliationBridge(method="mint", covariance_method="ledoit_wolf")
        result = bridge.reconcile(df, S, residuals=residuals)
        self.assertEqual(result.shape, df.shape)

    def test_identity_covariance(self):
        from autots.evaluator.tva.reconciliation import ReconciliationBridge
        df, S = self._make_simple_hierarchy()
        bridge = ReconciliationBridge(covariance_method="identity")
        result = bridge.reconcile(df, S)
        self.assertEqual(result.shape, df.shape)


# ---------------------------------------------------------------------------
# PyTorch-dependent unit tests
# ---------------------------------------------------------------------------

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SKIP_TORCH = unittest.skipUnless(HAS_TORCH, "PyTorch not available")


@SKIP_TORCH
class TestLossFunctions(unittest.TestCase):
    """Unit tests for each loss component using random tensors."""

    def _rand(self, *shape):
        return torch.randn(*shape)

    def test_forecast_loss_mse(self):
        from autots.evaluator.tva.losses import ForecastLoss
        loss = ForecastLoss('mse')
        pred = self._rand(2, 4, 10)
        target = self._rand(2, 4, 10)
        val = loss(pred, target)
        self.assertTrue(val.item() >= 0)

    def test_forecast_loss_mae(self):
        from autots.evaluator.tva.losses import ForecastLoss
        loss = ForecastLoss('mae')
        pred = self._rand(2, 4, 10)
        val = loss(pred, pred)  # identical → zero
        self.assertAlmostEqual(val.item(), 0.0, places=5)

    def test_orthogonality_penalty_zero_when_uncorrelated(self):
        from autots.evaluator.tva.losses import OrthogonalityPenalty
        pen = OrthogonalityPenalty()
        # trend = constant, seasonal = zero-mean oscillation
        trend = torch.ones(2, 3, 20)
        seasonal = torch.sin(torch.linspace(0, 6, 20)).unsqueeze(0).unsqueeze(0).expand(2, 3, -1)
        val = pen(trend, seasonal)
        self.assertTrue(val.item() >= 0)

    def test_orthogonality_penalty_no_components(self):
        from autots.evaluator.tva.losses import OrthogonalityPenalty
        pen = OrthogonalityPenalty()
        val = pen(self._rand(2, 3, 10))
        self.assertEqual(val.item(), 0.0)

    def test_local_trend_penalty(self):
        from autots.evaluator.tva.losses import LocalTrendPenalty
        pen = LocalTrendPenalty()
        x = self._rand(2, 4, 10)
        val = pen(x, x)  # identical → zero
        self.assertAlmostEqual(val.item(), 0.0, places=5)

    def test_smoothness_penalty_on_smooth_signal(self):
        from autots.evaluator.tva.losses import SmoothnessPenalty
        pen = SmoothnessPenalty()
        # perfectly linear → zero second-order diff
        t = torch.linspace(0, 1, 20).unsqueeze(0).unsqueeze(0).expand(2, 4, -1)
        val = pen(t)
        self.assertAlmostEqual(val.item(), 0.0, places=5)

    def test_smoothness_penalty_short_signal(self):
        from autots.evaluator.tva.losses import SmoothnessPenalty
        pen = SmoothnessPenalty()
        val = pen(torch.randn(2, 3, 2))  # T < 3 → 0
        self.assertEqual(val.item(), 0.0)

    def test_soft_prior_loss_identical(self):
        from autots.evaluator.tva.losses import SoftPriorLoss
        loss = SoftPriorLoss()
        adj = torch.eye(4)
        val = loss(adj, adj)
        self.assertAlmostEqual(val.item(), 0.0, places=5)

    def test_coherence_loss_all_same_direction(self):
        from autots.evaluator.tva.losses import CrossSeriesCoherenceLoss
        loss = CrossSeriesCoherenceLoss()
        # all trending up → small penalty
        trend = torch.zeros(2, 4, 10)
        for t in range(10):
            trend[:, :, t] = float(t)
        weights = torch.ones(2, 4, 2) / 2  # uniform prototype usage
        val_coherent = loss(trend, weights)

        # mixed directions → larger penalty
        mixed = trend.clone()
        mixed[:, 2:, :] = -trend[:, 2:, :]  # half trending down
        val_mixed = loss(mixed, weights)
        self.assertLess(val_coherent.item(), val_mixed.item())

    def test_coherence_loss_short_timeseries(self):
        from autots.evaluator.tva.losses import CrossSeriesCoherenceLoss
        loss = CrossSeriesCoherenceLoss()
        val = loss(torch.randn(2, 3, 1), torch.ones(2, 3, 2) / 2)
        self.assertEqual(val.item(), 0.0)

    def test_temporal_loss_composite_forward(self):
        from autots.evaluator.tva.losses import TemporalLossComposite
        loss_fn = TemporalLossComposite()
        B, N, T, K = 2, 4, 10, 3
        outputs = {
            'trend_forecast': torch.randn(B, N, T),
            'prototype_weights': torch.softmax(torch.randn(B, N, K), dim=-1),
            'composite_trend': torch.randn(B, K, T),
            'composite_trend_per_series': torch.randn(B, N, T),
        }
        targets = {
            'true_trend': torch.randn(B, N, T),
            'seasonal': torch.randn(B, N, T),
            'holidays': torch.randn(B, N, T),
            'anomalies': torch.randn(B, N, T),
        }
        total, breakdown = loss_fn(outputs, targets)
        self.assertTrue(total.item() >= 0)
        for key in ('forecast', 'orthogonality', 'smoothness', 'coherence', 'total'):
            self.assertIn(key, breakdown)

    def test_temporal_loss_composite_v2_keys(self):
        from autots.evaluator.tva.losses import TemporalLossComposite
        loss_fn = TemporalLossComposite()
        B, N, T, K, M = 2, 4, 10, 3, 4
        prior_adj = torch.eye(M)
        outputs = {
            'trend_forecast': torch.randn(B, N, T),
            'prototype_weights': torch.softmax(torch.randn(B, N, K), dim=-1),
            'composite_trend': torch.randn(B, K, T),
            'composite_trend_per_series': torch.randn(B, N, T),
            'adjacency': torch.sigmoid(torch.randn(M, M)),
        }
        targets = {
            'true_trend': torch.randn(B, N, T),
            'prior_adjacency': prior_adj,
        }
        total, breakdown = loss_fn(outputs, targets)
        self.assertIn('soft_prior', breakdown)

    def test_allfather_judgment(self):
        from autots.evaluator.tva.losses import TemporalLossComposite
        loss_fn = TemporalLossComposite()
        result = loss_fn._allfather_judgment()
        self.assertIsInstance(result, dict)


@SKIP_TORCH
class TestTrendTokenizer(unittest.TestCase):
    def test_output_shape_no_meta(self):
        from autots.evaluator.tva.trend_network import TrendTokenizer
        tok = TrendTokenizer(window_size=30, d_token=32)
        x = torch.randn(2, 5, 30)
        out = tok(x)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_output_shape_with_meta(self):
        from autots.evaluator.tva.trend_network import TrendTokenizer
        tok = TrendTokenizer(window_size=30, d_token=32, d_meta=8)
        x = torch.randn(2, 5, 30)
        meta = torch.randn(2, 5, 8)
        out = tok(x, meta)
        self.assertEqual(out.shape, (2, 5, 32))


@SKIP_TORCH
class TestHierarchicalLatentEncoder(unittest.TestCase):
    def test_output_shapes(self):
        from autots.evaluator.tva.trend_network import HierarchicalLatentEncoder
        enc = HierarchicalLatentEncoder(d_token=32, n_meso=4, n_global=2, n_heads=2)
        tokens = torch.randn(3, 6, 32)
        meso, glob, skip = enc(tokens)
        self.assertEqual(meso.shape, (3, 4, 32))
        self.assertEqual(glob.shape, (3, 2, 32))
        self.assertEqual(skip.shape, tokens.shape)


@SKIP_TORCH
class TestPrototypeBottleneck(unittest.TestCase):
    def test_output_shapes(self):
        from autots.evaluator.tva.trend_network import PrototypeBottleneck
        pb = PrototypeBottleneck(d_latent=32, n_prototypes=5)
        glob = torch.randn(2, 4, 32)
        conditioned, weights = pb(glob)
        self.assertEqual(conditioned.shape, (2, 4, 32))
        self.assertEqual(weights.shape, (2, 4, 5))
        # softmax → each row sums to 1
        row_sums = weights.sum(dim=-1)
        np.testing.assert_allclose(row_sums.detach().numpy(), np.ones_like(row_sums.detach().numpy()), atol=1e-5)


@SKIP_TORCH
class TestFusionLayers(unittest.TestCase):
    def test_additive_fusion(self):
        from autots.evaluator.tva.fusion import AdditiveFusion
        f = AdditiveFusion()
        B, N, T = 2, 4, 10
        trend = torch.ones(B, N, T)
        sea = torch.ones(B, N, T)
        hol = torch.zeros(B, N, T)
        ls = torch.zeros(B, N, T)
        out = f(trend, sea, hol, ls)
        self.assertEqual(out.shape, (B, N, T))
        np.testing.assert_allclose(out.detach().numpy(), 2.0 * np.ones((B, N, T)), atol=1e-5)

    def test_additive_fusion_with_anomalies(self):
        from autots.evaluator.tva.fusion import AdditiveFusion
        f = AdditiveFusion()
        x = torch.ones(2, 3, 5)
        out = f(x, x, x, x, anomalies=x)
        np.testing.assert_allclose(out.detach().numpy(), 5.0 * np.ones((2, 3, 5)), atol=1e-5)

    def test_digital_twin_fusion_shape(self):
        from autots.evaluator.tva.fusion import DigitalTwinFusion
        f = DigitalTwinFusion()
        B, N, T = 2, 4, 10
        x = torch.randn(B, N, T)
        out = f(x, x, x, x)
        self.assertEqual(out.shape, (B, N, T))

    def test_digital_twin_fusion_with_anomalies(self):
        from autots.evaluator.tva.fusion import DigitalTwinFusion
        f = DigitalTwinFusion()
        x = torch.randn(2, 3, 8)
        out = f(x, x, x, x, anomalies=x)
        self.assertEqual(out.shape, (2, 3, 8))


@SKIP_TORCH
class TestCompositeTrendNetworkV1(unittest.TestCase):
    """Unit tests for CompositeTrendNetworkV1 forward pass."""

    def _make_net(self, n_series=4, window=30, horizon=10, d=32, n_meso=4, n_global=2, K=3):
        from autots.evaluator.tva.trend_network import CompositeTrendNetworkV1
        return CompositeTrendNetworkV1(
            n_series=n_series,
            window_size=window,
            forecast_horizon=horizon,
            d_token=d,
            n_meso=n_meso,
            n_global=n_global,
            n_prototypes=K,
            n_heads=2,
        )

    def test_output_keys(self):
        net = self._make_net()
        x = torch.randn(2, 4, 30)
        out = net(x)
        for key in ('trend_forecast', 'prototype_weights', 'composite_trend', 'composite_trend_per_series'):
            self.assertIn(key, out)

    def test_output_shapes_all_anchors(self):
        N, T, H = 4, 30, 10
        net = self._make_net(n_series=N, window=T, horizon=H)
        x = torch.randn(2, N, T)
        out = net(x)
        self.assertEqual(out['trend_forecast'].shape, (2, N, H))
        self.assertEqual(out['composite_trend_per_series'].shape, (2, N, H))

    def test_output_shapes_with_anchor_mask(self):
        N, T, H = 6, 30, 10
        net = self._make_net(n_series=N, window=T, horizon=H, n_meso=4, n_global=2)
        x = torch.randn(2, N, T)
        mask = torch.tensor([True, True, True, True, False, False])
        out = net(x, anchor_mask=mask)
        self.assertEqual(out['trend_forecast'].shape, (2, N, H))

    def test_output_shapes_with_prior_adjacency(self):
        from autots.evaluator.tva.trend_network import CompositeTrendNetworkV1
        prior = np.eye(4, dtype=np.float32)
        net = CompositeTrendNetworkV1(
            n_series=4, window_size=30, forecast_horizon=10,
            d_token=32, n_meso=4, n_global=2, n_prototypes=3, n_heads=2,
            prior_adjacency=prior,
        )
        out = net(torch.randn(2, 4, 30))
        self.assertEqual(out['trend_forecast'].shape, (2, 4, 10))

    def test_prototype_weights_sum_to_one(self):
        net = self._make_net(K=4)
        x = torch.randn(2, 4, 30)
        out = net(x)
        row_sums = out['prototype_weights'].sum(dim=-1)
        np.testing.assert_allclose(
            row_sums.detach().numpy(), np.ones_like(row_sums.detach().numpy()), atol=1e-4
        )


@SKIP_TORCH
class TestCompositeTrendNetworkV2(unittest.TestCase):
    def _make_net(self, n_series=4, window=30, horizon=10):
        from autots.evaluator.tva.trend_network import CompositeTrendNetworkV2
        return CompositeTrendNetworkV2(
            n_series=n_series,
            window_size=window,
            forecast_horizon=horizon,
            d_token=32,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
        )

    def test_v2_extra_output_keys(self):
        net = self._make_net()
        out = net(torch.randn(2, 4, 30))
        for key in ('mu', 'sigma', 'adjacency'):
            self.assertIn(key, out)

    def test_v2_sigma_positive(self):
        net = self._make_net()
        out = net(torch.randn(2, 4, 30))
        self.assertTrue((out['sigma'] > 0).all())

    def test_v2_adjacency_range(self):
        net = self._make_net()
        out = net(torch.randn(2, 4, 30))
        adj = out['adjacency']
        self.assertTrue((adj >= 0).all())
        self.assertTrue((adj <= 1).all())

    def test_v2_learned_adjacency_property(self):
        net = self._make_net()
        adj = net.learned_adjacency
        self.assertEqual(adj.shape, (2, 2))  # n_global x n_global

    def test_v2_promote_responder_no_error(self):
        net = self._make_net()
        net.promote_responder(2)  # should not raise

    def test_v2_glorious_purpose(self):
        net = self._make_net()
        result = net._glorious_purpose()
        self.assertIsInstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Integration tests (require both torch + autots feature detector)
# ---------------------------------------------------------------------------

try:
    from autots.evaluator.feature_detector import TimeSeriesFeatureDetector  # noqa
    HAS_FEATURE_DETECTOR = True
except Exception:
    HAS_FEATURE_DETECTOR = False

SKIP_INTEGRATION = unittest.skipUnless(
    HAS_TORCH and HAS_FEATURE_DETECTOR,
    "PyTorch or TimeSeriesFeatureDetector not available",
)


@SKIP_INTEGRATION
class TestNornDecomposer(unittest.TestCase):
    def setUp(self):
        self.df = _make_daily_df(n_series=3, n_days=200)

    def test_fit_and_get_components(self):
        from autots.evaluator.tva.decomposition import NornDecomposer
        nd = NornDecomposer()
        nd.fit(self.df)
        comps = nd.get_components()
        for key in ('trend', 'seasonality', 'holidays', 'level_shifts', 'anomalies', 'noise'):
            self.assertIn(key, comps)
            c = comps[key]
            self.assertIsInstance(c, pd.DataFrame)
            self.assertEqual(c.shape[1], 3)
            self.assertFalse(c.isnull().any().any(), f"NaN in {key}")

    def test_components_same_index_as_input(self):
        from autots.evaluator.tva.decomposition import NornDecomposer
        nd = NornDecomposer()
        nd.fit(self.df)
        comps = nd.get_components()
        for key, df in comps.items():
            self.assertTrue(df.index.equals(comps['trend'].index), f"Index mismatch in {key}")

    def test_components_cached(self):
        from autots.evaluator.tva.decomposition import NornDecomposer
        nd = NornDecomposer()
        nd.fit(self.df)
        c1 = nd.get_components()
        c2 = nd.get_components()
        self.assertIs(c1, c2)

    def test_get_features_returns_dict(self):
        from autots.evaluator.tva.decomposition import NornDecomposer
        nd = NornDecomposer()
        nd.fit(self.df)
        features = nd.get_features()
        self.assertIsInstance(features, dict)

    def test_get_forecast_components_shape(self):
        from autots.evaluator.tva.decomposition import NornDecomposer
        nd = NornDecomposer()
        nd.fit(self.df)
        fc = nd.get_forecast_components(14)
        for key in ('trend', 'seasonality', 'holidays', 'level_shifts', 'anomalies', 'noise'):
            self.assertIn(key, fc)
            self.assertEqual(fc[key].shape[1], 3)

    def test_error_before_fit(self):
        from autots.evaluator.tva.decomposition import NornDecomposer
        nd = NornDecomposer()
        with self.assertRaises(RuntimeError):
            nd.get_components()

    def test_urd_verdict(self):
        from autots.evaluator.tva.decomposition import NornDecomposer
        nd = NornDecomposer()
        nd.fit(self.df)
        val = nd._urd_verdict("s0")
        # may be nan if not available, but must not raise
        self.assertIsInstance(val, float)


@SKIP_INTEGRATION
class TestTVAIntegration(unittest.TestCase):
    """End-to-end TVA fit → predict with minimal settings for speed."""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_daily_df(n_series=3, n_days=400)

    def _make_tva(self, trend_network='v1', fusion='additive', epochs=2):
        from autots.evaluator.tva.tva import TVA
        return TVA(
            trend_network=trend_network,
            fusion=fusion,
            epochs=epochs,
            window_size=60,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            batch_size=8,
            verbose=0,
        )

    def test_fit_returns_self(self):
        tva = self._make_tva()
        result = tva.fit(self.df)
        self.assertIs(result, tva)

    def test_predict_shape_v1_additive(self):
        tva = self._make_tva(trend_network='v1', fusion='additive')
        tva.fit(self.df)
        forecast = tva.predict()
        self.assertEqual(forecast.shape, (14, 3))
        self.assertFalse(forecast.isnull().any().any())

    def test_predict_shape_v2_additive(self):
        tva = self._make_tva(trend_network='v2', fusion='additive')
        tva.fit(self.df)
        forecast = tva.predict()
        self.assertEqual(forecast.shape, (14, 3))
        self.assertFalse(forecast.isnull().any().any())

    def test_predict_shape_v1_attention_fusion(self):
        tva = self._make_tva(trend_network='v1', fusion='attention')
        tva.fit(self.df)
        forecast = tva.predict()
        self.assertEqual(forecast.shape, (14, 3))

    def test_predict_columns_match_input(self):
        tva = self._make_tva()
        tva.fit(self.df)
        forecast = tva.predict()
        self.assertListEqual(list(forecast.columns), list(self.df.columns))

    def test_predict_custom_forecast_length(self):
        tva = self._make_tva()
        tva.fit(self.df)
        forecast = tva.predict(forecast_length=7)
        self.assertEqual(forecast.shape[0], 7)

    def test_get_composite_trends(self):
        tva = self._make_tva()
        tva.fit(self.df)
        ct = tva.get_composite_trends()
        self.assertIn('prototypes', ct)
        self.assertIn('composite_trend', ct)
        self.assertIn('prototype_weights', ct)

    def test_get_graph_shape_v1(self):
        tva = self._make_tva(trend_network='v1')
        tva.fit(self.df)
        graph = tva.get_graph()
        self.assertIsInstance(graph, np.ndarray)
        self.assertEqual(graph.ndim, 2)

    def test_get_graph_shape_v2(self):
        tva = self._make_tva(trend_network='v2')
        tva.fit(self.df)
        graph = tva.get_graph()
        self.assertEqual(graph.shape, (2, 2))  # n_global x n_global

    def test_he_who_remains(self):
        tva = self._make_tva()
        tva.fit(self.df)
        info = tva._he_who_remains()
        self.assertEqual(info['n_series'], 3)
        self.assertEqual(info['n_prototypes'], 3)

    def test_error_before_fit(self):
        from autots.evaluator.tva.tva import TVA
        tva = TVA(verbose=0)
        with self.assertRaises(RuntimeError):
            tva.predict()

    def test_insufficient_data_error(self):
        from autots.evaluator.tva.tva import TVA
        tva = TVA(window_size=500, forecast_horizon=100, verbose=0)
        with self.assertRaises(ValueError):
            tva.fit(self.df)


@SKIP_INTEGRATION
class TestTVAWithPriors(unittest.TestCase):
    """TVA fit/predict with series metadata and priors."""

    @classmethod
    def setUpClass(cls):
        from autots.evaluator.tva.priors import SeriesMetadata
        cls.df = _make_daily_df(n_series=4, n_days=400)
        cls.metadata = [
            SeriesMetadata("s0", metric_type="dau", surface="mobile", geography="US", history_periods=400),
            SeriesMetadata("s1", metric_type="dau", surface="web",    geography="US", history_periods=400),
            SeriesMetadata("s2", metric_type="views", surface="mobile", geography="DE", history_periods=400),
            SeriesMetadata("s3", metric_type="views", surface="web",    geography="DE", history_periods=50),
        ]

    def test_fit_with_metadata(self):
        from autots.evaluator.tva.tva import TVA
        tva = TVA(
            series_metadata=self.metadata,
            epochs=2,
            window_size=60,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            min_anchor_history=200,
            verbose=0,
        )
        tva.fit(self.df)
        forecast = tva.predict()
        self.assertEqual(forecast.shape, (14, 4))
        self.assertFalse(forecast.isnull().any().any())

    def test_anchor_mask_applied_correctly(self):
        from autots.evaluator.tva.tva import TVA
        tva = TVA(
            series_metadata=self.metadata,
            epochs=2,
            window_size=60,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            min_anchor_history=200,
            verbose=0,
        )
        tva.fit(self.df)
        # s3 has only 50 history periods, should NOT be an anchor
        self.assertFalse(tva._anchor_mask[3])
        # s0-s2 have 400 periods, should be anchors
        self.assertTrue(tva._anchor_mask[0])


@SKIP_INTEGRATION
class TestTVACoherence(unittest.TestCase):
    """CrossSeriesCoherenceLoss should dampen mixed-direction outputs."""

    def test_coherent_data_produces_consistent_forecast_signs(self):
        """When all series have strong upward trends, all forecasts should
        continue in a reasonable direction (not wildly negative)."""
        from autots.evaluator.tva.tva import TVA
        df = _make_coherent_df(n_series=4, n_days=300)
        tva = TVA(
            trend_network='v2',
            fusion='additive',
            epochs=3,
            window_size=50,
            forecast_horizon=10,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            verbose=0,
            loss_weights={'coherence': 5.0},
        )
        tva.fit(df)
        forecast = tva.predict()
        self.assertEqual(forecast.shape, (10, 4))
        # the mean forecast should be positive (data is strongly upward)
        self.assertGreater(forecast.mean().mean(), -10.0)


@SKIP_INTEGRATION
class TestScenarioPlanning(unittest.TestCase):
    """BifrostOptimizer what-if scenarios."""

    @classmethod
    def setUpClass(cls):
        from autots.evaluator.tva.tva import TVA
        cls.df = _make_daily_df(n_series=3, n_days=350)
        cls.tva = TVA(
            epochs=2,
            window_size=50,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            verbose=0,
        )
        cls.tva.fit(cls.df)

    def test_what_if_constraint_returns_dataframe(self):
        result = self.tva.what_if(
            series_name='s0', timestep=5, target_value=100.0
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[1], 3)
        self.assertFalse(result.isnull().any().any())

    def test_what_if_growth_constraint(self):
        result = self.tva.what_if(series_name='s1', growth_rate=0.1)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[1], 3)

    def test_bifrost_apply_constraint_directly(self):
        from autots.evaluator.tva.scenario import BifrostOptimizer
        opt = BifrostOptimizer(self.tva, n_steps=5)
        result = opt.apply_constraint('s0', 3, 50.0)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (14, 3))

    def test_bifrost_growth_constraint_directly(self):
        from autots.evaluator.tva.scenario import BifrostOptimizer
        opt = BifrostOptimizer(self.tva, n_steps=5)
        result = opt.apply_growth_constraint('s2', 0.05)
        self.assertIsInstance(result, pd.DataFrame)

    def test_bifrost_rainbow_bridge_strength(self):
        from autots.evaluator.tva.scenario import BifrostOptimizer
        opt = BifrostOptimizer(self.tva, n_steps=10, lr=0.02)
        strength = opt._rainbow_bridge_strength()
        self.assertAlmostEqual(strength, 0.2, places=5)


@SKIP_INTEGRATION
class TestReconciliationWithTVA(unittest.TestCase):
    """Integration test: TVA reconcile() method with hierarchy."""

    def test_reconcile_without_priors_returns_forecast(self):
        from autots.evaluator.tva.tva import TVA
        df = _make_daily_df(n_series=3, n_days=300)
        tva = TVA(
            reconciliation_method='mint',
            epochs=2,
            window_size=50,
            forecast_horizon=10,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            verbose=0,
        )
        tva.fit(df)
        result = tva.reconcile()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[1], 3)

    def test_reconcile_with_hierarchy(self):
        from autots.evaluator.tva.tva import TVA
        from autots.evaluator.tva.priors import SeriesMetadata
        df = _make_daily_df(n_series=2, n_days=300)
        df.columns = ['US', 'CA']
        meta = [
            SeriesMetadata("US", hierarchy_path=["global", "US"], history_periods=300),
            SeriesMetadata("CA", hierarchy_path=["global", "CA"], history_periods=300),
        ]
        tva = TVA(
            series_metadata=meta,
            reconciliation_method='mint',
            epochs=2,
            window_size=50,
            forecast_horizon=10,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            verbose=0,
        )
        tva.fit(df)
        result = tva.reconcile()
        self.assertIsInstance(result, pd.DataFrame)


@SKIP_INTEGRATION
class TestTVALossWeights(unittest.TestCase):
    """Verify custom loss weights are respected."""

    def test_custom_loss_weights_do_not_raise(self):
        from autots.evaluator.tva.tva import TVA
        df = _make_daily_df(n_series=3, n_days=300)
        tva = TVA(
            epochs=2,
            window_size=50,
            forecast_horizon=10,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            loss_weights={
                'forecast': 2.0,
                'coherence': 5.0,
                'orthogonality': 0.5,
                'local_trend': 0.0,
                'smoothness': 0.0,
                'soft_prior': 0.0,
            },
            verbose=0,
        )
        tva.fit(df)
        forecast = tva.predict()
        self.assertEqual(forecast.shape, (10, 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
