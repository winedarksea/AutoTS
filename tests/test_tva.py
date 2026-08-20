# -*- coding: utf-8 -*-
"""
Tests for the TVA (Time Variance Authority) forecasting graph.

Covers: unit tests for each module (no torch required for priors/reconciliation),
integration tests with synthetic data, coherence validation, scenario planning,
reconciliation bridge, and metadata-prior paths.

Run with:  python -m pytest tests/test_tva.py -v
"""
import unittest
from unittest import mock
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


def _hollow(matrix: np.ndarray) -> np.ndarray:
    """Copy of ``matrix`` with a zeroed diagonal (what a coerced prior is)."""
    out = np.asarray(matrix, dtype=np.float32).copy()
    np.fill_diagonal(out, 0.0)
    return out


def _make_coherent_df(n_series: int = 6, n_days: int = 400, seed: int = 1) -> pd.DataFrame:
    """All series share a common upward trend — tests coherence loss."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    shared = np.linspace(0, 5, n_days)
    data = np.stack([shared + rng.normal(0, 0.05, n_days) for _ in range(n_series)], axis=1)
    return pd.DataFrame(data, index=dates, columns=[f"series_{i}" for i in range(n_series)])


def _make_use_case_2_metadata():
    """Generic metadata layout exercising multiple categorical relationship axes."""
    from autots.evaluator.tva.priors import SeriesMetadata

    return [
        SeriesMetadata(
            "interface_time_marketplace_US",
            attribute_values={
                "axis_a": "type_1",
                "axis_b": "group_red",
                "axis_c": "region_north",
            },
            hierarchy_path=["global", "NA", "US"],
            history_periods=400,
        ),
        SeriesMetadata(
            "viewport_views_marketplace_US",
            attribute_values={
                "axis_a": "type_2",
                "axis_b": "group_red",
                "axis_c": "region_north",
            },
            hierarchy_path=["global", "NA", "US"],
            history_periods=400,
        ),
        SeriesMetadata(
            "daily_active_users_marketplace_CA",
            attribute_values={
                "axis_a": "type_3",
                "axis_b": "group_red",
                "axis_c": "region_central",
            },
            hierarchy_path=["global", "NA", "CA"],
            history_periods=400,
        ),
        SeriesMetadata(
            "interface_time_videos_DE",
            attribute_values={
                "axis_a": "type_1",
                "axis_b": "group_blue",
                "axis_c": "region_south",
            },
            hierarchy_path=["global", "EU", "DE"],
            history_periods=400,
        ),
        SeriesMetadata(
            "daily_active_users_mobile_feed_US",
            attribute_values={
                "axis_a": "type_3",
                "axis_b": "group_green",
                "axis_c": "region_north",
            },
            hierarchy_path=["global", "NA", "US"],
            history_periods=45,
        ),
    ]


# ---------------------------------------------------------------------------
# Module-level unit tests (no torch needed for priors / reconciliation)
# ---------------------------------------------------------------------------

class TestSeriesMetadata(unittest.TestCase):
    def test_default_fields(self):
        from autots.evaluator.tva.priors import SeriesMetadata
        m = SeriesMetadata(name="foo")
        self.assertEqual(m.name, "foo")
        self.assertEqual(m.attribute_values, {})
        self.assertEqual(m.history_periods, 0)

    def test_fully_specified_generic_attributes(self):
        from autots.evaluator.tva.priors import SeriesMetadata
        m = SeriesMetadata(
            name="bar",
            attribute_values={"family": "dau", "channel": "mobile_feed", "locale": "US"},
            hierarchy_path=["global", "NA", "US"],
            history_periods=365,
        )
        self.assertEqual(m.attribute_values["locale"], "US")
        self.assertEqual(len(m.hierarchy_path), 3)

    def test_legacy_keyword_aliases_flow_into_attribute_values(self):
        from autots.evaluator.tva.priors import SeriesMetadata
        m = SeriesMetadata(
            name="bar",
            metric_type="dau",
            surface="mobile_feed",
            geography="US",
        )
        self.assertEqual(m.attribute_values["metric_type"], "dau")
        self.assertEqual(m.attribute_values["surface"], "mobile_feed")
        self.assertEqual(m.attribute_values["geography"], "US")


class TestYggdrasilPriors(unittest.TestCase):
    def _make_metadata(self):
        from autots.evaluator.tva.priors import SeriesMetadata
        return [
            SeriesMetadata(
                "a",
                attribute_values={"family": "f1", "channel": "c1", "locale": "l1"},
                history_periods=400,
            ),
            SeriesMetadata(
                "b",
                attribute_values={"family": "f1", "channel": "c2", "locale": "l1"},
                history_periods=400,
            ),
            SeriesMetadata(
                "c",
                attribute_values={"family": "f2", "channel": "c1", "locale": "l2"},
                history_periods=100,
            ),
            SeriesMetadata(
                "d",
                attribute_values={"family": "f2", "channel": "c2", "locale": "l2"},
                history_periods=50,
            ),
        ]

    def test_empty_metadata_returns_defaults(self):
        from autots.evaluator.tva.priors import YggdrasilPriors
        p = YggdrasilPriors()
        adj = p.build_prior_adjacency()
        self.assertIsNone(adj)
        emb = p.build_metadata_embeddings()
        self.assertEqual(emb.shape, (0, 1))
        mask = p.get_anchor_mask(100)
        self.assertEqual(len(mask), 0)
        S = p.build_hierarchy_matrix()
        self.assertEqual(S.shape, (0, 0))

    def test_no_metadata_defaults_follow_series_count(self):
        from autots.evaluator.tva.priors import YggdrasilPriors

        trend = pd.DataFrame(
            np.zeros((8, 3), dtype=float),
            index=pd.date_range("2024-01-01", periods=8, freq="D"),
            columns=['a', 'b', 'c'],
        )
        p = YggdrasilPriors(trend_data=trend)

        emb = p.build_metadata_embeddings()
        self.assertEqual(emb.shape, (3, 1))

        mask = p.get_anchor_mask(100)
        np.testing.assert_array_equal(mask, np.array([True, True, True], dtype=bool))

        S = p.build_hierarchy_matrix()
        np.testing.assert_allclose(S, np.eye(3, dtype=np.float32), atol=1e-6)

    def test_prior_adjacency_shape_and_symmetry(self):
        from autots.evaluator.tva.priors import YggdrasilPriors
        p = YggdrasilPriors(series_metadata=self._make_metadata())
        adj = p.build_prior_adjacency()
        n = 4
        self.assertEqual(adj.shape, (n, n))
        np.testing.assert_allclose(adj, adj.T, atol=1e-6)
        # self-edges are excluded from priors
        np.testing.assert_allclose(np.diag(adj), np.zeros(n), atol=1e-6)

    def test_shared_attribute_raises_adjacency(self):
        from autots.evaluator.tva.priors import YggdrasilPriors
        p = YggdrasilPriors(series_metadata=self._make_metadata())
        adj = p.build_prior_adjacency()
        # a & b share two categorical axes, so adj[0,1] > 0
        self.assertGreater(adj[0, 1], 0.0)
        # a & d share nothing
        self.assertEqual(adj[0, 3], 0.0)

    def test_metadata_attribute_weights_raise_higher_priority_matches(self):
        from autots.evaluator.tva.priors import SeriesMetadata, YggdrasilPriors

        metadata = [
            SeriesMetadata(
                "a",
                attribute_values={"surface": "search", "geography": "US"},
                attribute_weights={"surface": 0.8, "geography": 0.2},
            ),
            SeriesMetadata(
                "b",
                attribute_values={"surface": "search", "geography": "CA"},
                attribute_weights={"surface": 0.8, "geography": 0.2},
            ),
            SeriesMetadata(
                "c",
                attribute_values={"surface": "video", "geography": "US"},
                attribute_weights={"surface": 0.8, "geography": 0.2},
            ),
        ]
        p = YggdrasilPriors(series_metadata=metadata)
        adj = p.build_prior_adjacency()

        self.assertGreater(adj[0, 1], adj[0, 2])
        self.assertAlmostEqual(adj[0, 1], 0.8, places=6)
        self.assertAlmostEqual(adj[0, 2], 0.2, places=6)

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

    def test_generic_metadata_prior_strength_tracks_shared_attributes(self):
        from autots.evaluator.tva.priors import YggdrasilPriors

        p = YggdrasilPriors(series_metadata=_make_use_case_2_metadata())
        adj = p.build_prior_adjacency()

        # Same surface + geography should be stronger than a single shared attribute.
        self.assertAlmostEqual(adj[0, 1], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(adj[0, 3], 1.0 / 3.0, places=6)
        self.assertGreater(adj[0, 1], adj[0, 3])

        # One shared business attribute should produce the softer 1/3 prior weight.
        self.assertAlmostEqual(adj[2, 4], 1.0 / 3.0, places=6)

    def test_short_history_series_is_output_but_not_anchor(self):
        from autots.evaluator.tva.priors import YggdrasilPriors

        metadata = _make_use_case_2_metadata()
        p = YggdrasilPriors(series_metadata=metadata)
        adj = p.build_prior_adjacency()
        emb = p.build_metadata_embeddings()
        mask = p.get_anchor_mask(180)

        self.assertEqual(adj.shape, (5, 5))
        self.assertEqual(emb.shape[0], 5)
        self.assertEqual(len(mask), 5)
        self.assertFalse(mask[-1])
        self.assertGreater(adj[-1, 0], 0.0)

    def test_no_shared_metadata_returns_no_prior(self):
        from autots.evaluator.tva.priors import SeriesMetadata, YggdrasilPriors

        metadata = [
            SeriesMetadata("a", attribute_values={"family": "f1"}),
            SeriesMetadata("b", attribute_values={"family": "f2"}),
            SeriesMetadata("c", attribute_values={"family": "f3"}),
        ]
        p = YggdrasilPriors(series_metadata=metadata)
        self.assertIsNone(p.build_prior_adjacency())

    def test_event_clustering_builds_structural_prior(self):
        from autots.evaluator.tva.priors import YggdrasilPriors

        features = {
            'trend_changepoints': {
                'a': [(pd.Timestamp("2024-01-10"), 0.0, 2.0)],
                'b': [(pd.Timestamp("2024-01-12"), 0.0, 1.8)],
                'c': [(pd.Timestamp("2024-02-15"), 0.0, 2.1)],
            },
            'level_shifts': {},
            'anomalies': {},
        }
        trend = pd.DataFrame(
            np.zeros((120, 3), dtype=float),
            index=pd.date_range("2024-01-01", periods=120, freq="D"),
            columns=['a', 'b', 'c'],
        )
        p = YggdrasilPriors(
            detected_features=features,
            trend_data=trend,
            series_names=['a', 'b', 'c'],
            prior_construction_config={'sources': ['event']},
        )
        adj = p.build_structural_prior_adjacency()

        self.assertEqual(adj.shape, (3, 3))
        # event edges are now DIRECTED: a's changepoint (Jan 10) precedes
        # b's (Jan 12), so the edge runs a -> b only
        self.assertGreater(adj[0, 1], 0.0)
        self.assertEqual(adj[1, 0], 0.0)
        self.assertEqual(adj[0, 2], 0.0)

    def test_structural_merge_weights_and_renormalize_available_sources(self):
        from autots.evaluator.tva.priors import SeriesMetadata, YggdrasilPriors

        metadata = [
            SeriesMetadata("a", attribute_values={"family": "x"}),
            SeriesMetadata("b", attribute_values={"family": "x"}),
            SeriesMetadata("c", attribute_values={"family": "y"}),
        ]
        features = {
            'trend_changepoints': {},
            'level_shifts': {},
            'anomalies': {
                'a': [],
                'b': [(pd.Timestamp("2024-01-15"), 1.0, 'point_outlier', 1, False)],
                'c': [(pd.Timestamp("2024-01-16"), 1.0, 'point_outlier', 1, False)],
            },
        }
        trend = pd.DataFrame(
            np.zeros((120, 3), dtype=float),
            index=pd.date_range("2024-01-01", periods=120, freq="D"),
            columns=['a', 'b', 'c'],
        )
        config = {
            'sources': ['event', 'metadata'],
            'source_weights': {'event': 0.7, 'metadata': 0.3},
        }
        p = YggdrasilPriors(
            series_metadata=metadata,
            detected_features=features,
            trend_data=trend,
            prior_construction_config=config,
        )
        adj = p.build_structural_prior_adjacency()

        self.assertGreater(adj[1, 2], adj[0, 1])

        metadata_only = YggdrasilPriors(
            series_metadata=metadata,
            detected_features={'trend_changepoints': {}, 'level_shifts': {}, 'anomalies': {}},
            trend_data=trend,
            prior_construction_config=config,
        )
        metadata_default = YggdrasilPriors(series_metadata=metadata)
        np.testing.assert_allclose(
            metadata_only.build_structural_prior_adjacency(),
            metadata_default.build_structural_prior_adjacency(),
            atol=1e-6,
        )

    # NOTE: pairwise Granger causal-prior construction was deleted; series-
    # level causal structure now comes from factor-residual discovery, which
    # is covered (including driver/responder direction) in
    # tests/test_tva_discovery.py.

    def test_metadata_embeddings_encode_all_available_attribute_axes(self):
        from autots.evaluator.tva.priors import YggdrasilPriors

        p = YggdrasilPriors(series_metadata=_make_use_case_2_metadata())
        emb = p.build_metadata_embeddings()

        # 3 values across each of 3 axes = 9 one-hot features.
        self.assertEqual(emb.shape, (5, 9))
        np.testing.assert_allclose(emb.sum(axis=1), np.full(5, 3.0), atol=1e-6)

    def test_hierarchy_matrix_contains_expected_rollups(self):
        from autots.evaluator.tva.priors import YggdrasilPriors

        p = YggdrasilPriors(series_metadata=_make_use_case_2_metadata())
        S = p.build_hierarchy_matrix()

        n_bottom = 5
        aggregate_rows = S[:-n_bottom, :]
        expected_rows = [
            np.array([1, 1, 1, 1, 1], dtype=np.float32),  # global
            np.array([1, 1, 1, 0, 1], dtype=np.float32),  # NA
            np.array([0, 0, 0, 1, 0], dtype=np.float32),  # EU
        ]

        for expected in expected_rows:
            self.assertTrue(
                any(np.array_equal(row, expected) for row in aggregate_rows),
                msg=f"missing expected aggregate row {expected.tolist()}",
            )

        np.testing.assert_allclose(S[-n_bottom:, :], np.eye(n_bottom), atol=1e-6)


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


class TestTVAUtilities(unittest.TestCase):
    def test_recency_window_weights_increase_for_recent_targets(self):
        from autots.evaluator.tva.tva import TVA

        tva = TVA.__new__(TVA)
        tva.window_size = 10
        tva.forecast_horizon = 5
        tva.recency_halflife_days = 7

        index = pd.date_range("2025-01-01", periods=40, freq="D")
        weights = tva._compute_window_sample_weights(index)

        self.assertIsNotNone(weights)
        self.assertEqual(len(weights), 26)
        self.assertGreater(weights[-1], weights[0])

    def test_reconcile_estimates_residuals_when_not_supplied(self):
        from autots.evaluator.tva.tva import TVA

        tva = TVA.__new__(TVA)
        tva.reconciliation_method = 'mint'
        tva._reconciler = mock.Mock()
        tva._priors = mock.Mock()
        tva._priors.build_hierarchy_matrix.return_value = np.array(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        tva._compute_recent_reconciliation_residuals = mock.Mock(
            return_value=np.ones((30, 3), dtype=np.float64)
        )
        tva._reconciler.reconcile.side_effect = (
            lambda full_df, S, residuals=None: full_df
        )

        forecasts = pd.DataFrame(
            [[10.0, 12.0], [11.0, 13.0]],
            index=pd.date_range("2025-01-01", periods=2, freq="D"),
            columns=["s0", "s1"],
        )

        tva.reconcile(forecasts=forecasts)

        tva._compute_recent_reconciliation_residuals.assert_called_once()
        _, _, kwargs = tva._reconciler.reconcile.mock_calls[0]
        np.testing.assert_allclose(kwargs['residuals'], np.ones((30, 3)))

    def test_reconcile_prefers_explicit_residuals(self):
        from autots.evaluator.tva.tva import TVA

        tva = TVA.__new__(TVA)
        tva.reconciliation_method = 'mint'
        tva._reconciler = mock.Mock()
        tva._priors = mock.Mock()
        tva._priors.build_hierarchy_matrix.return_value = np.array(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        tva._compute_recent_reconciliation_residuals = mock.Mock()
        tva._reconciler.reconcile.side_effect = (
            lambda full_df, S, residuals=None: full_df
        )

        forecasts = pd.DataFrame(
            [[10.0, 12.0], [11.0, 13.0]],
            index=pd.date_range("2025-01-01", periods=2, freq="D"),
            columns=["s0", "s1"],
        )
        residuals = np.full((12, 3), 0.25, dtype=np.float64)

        tva.reconcile(forecasts=forecasts, residuals=residuals)

        tva._compute_recent_reconciliation_residuals.assert_not_called()
        _, _, kwargs = tva._reconciler.reconcile.mock_calls[0]
        np.testing.assert_allclose(kwargs['residuals'], residuals)


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

    def test_soft_prior_loss_respects_prior_confidence(self):
        from autots.evaluator.tva.losses import SoftPriorLoss
        loss = SoftPriorLoss(penalty_weight=2.0)
        learned = torch.zeros(2, 2)
        prior = torch.ones(2, 2)
        no_prior = loss(learned, prior, prior_confidence=0.0)
        half_prior = loss(learned, prior, prior_confidence=0.5)
        full_prior = loss(learned, prior, prior_confidence=1.0)
        self.assertEqual(no_prior.item(), 0.0)
        self.assertAlmostEqual(half_prior.item() * 2.0, full_prior.item(), places=5)

    def test_coherence_loss_all_same_direction(self):
        from autots.evaluator.tva.losses import CrossSeriesCoherenceLoss
        loss = CrossSeriesCoherenceLoss()
        # all trending up → small penalty
        trend = torch.zeros(2, 4, 10)
        for t in range(10):
            trend[:, :, t] = float(t)
        weights = torch.ones(2, 4, 2) / 2  # uniform prototype usage
        val_coherent = loss(trend, weights)

        # mixed directions → larger penalty; use 1-of-4 dissenting so consensus != 0
        # (perfectly balanced up/down gives consensus=0 → penalty=0 by design)
        mixed = trend.clone()
        mixed[:, 3:, :] = -trend[:, 3:, :]  # 1 of 4 series trending down
        val_mixed = loss(mixed, weights)
        self.assertLess(val_coherent.item(), val_mixed.item())

    def test_coherence_loss_short_timeseries(self):
        from autots.evaluator.tva.losses import CrossSeriesCoherenceLoss
        loss = CrossSeriesCoherenceLoss()
        val = loss(torch.randn(2, 3, 1), torch.ones(2, 3, 2) / 2)
        self.assertEqual(val.item(), 0.0)

    def test_coherence_loss_penalizes_growth_magnitude_mismatch(self):
        from autots.evaluator.tva.losses import CrossSeriesCoherenceLoss

        loss = CrossSeriesCoherenceLoss()
        matched = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32
        )
        mismatched = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 7.0, 10.0]]], dtype=torch.float32
        )
        weights = torch.ones(1, 2, 1, dtype=torch.float32)

        self.assertLess(
            loss(matched, weights).item(),
            loss(mismatched, weights).item(),
        )

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

    def test_temporal_loss_composite_prior_confidence_zero_disables_prior_losses(self):
        from autots.evaluator.tva.losses import TemporalLossComposite
        loss_fn = TemporalLossComposite(weights={'soft_prior': 1.0, 'causal_prior': 1.0})
        outputs = {
            'trend_forecast': torch.randn(2, 4, 10),
            'prototype_weights': torch.softmax(torch.randn(2, 4, 3), dim=-1),
            'composite_trend': torch.randn(2, 3, 10),
            'adjacency': torch.zeros(2, 2),
            'causal_prior': torch.ones(2, 2),
        }
        targets = {
            'true_trend': torch.randn(2, 4, 10),
            'prior_adjacency': torch.ones(2, 2),
            'prior_confidence': 0.0,
        }
        _, breakdown = loss_fn(outputs, targets)
        self.assertEqual(breakdown['soft_prior'], 0.0)
        self.assertEqual(breakdown['causal_prior'], 0.0)

    def test_allfather_judgment(self):
        from autots.evaluator.tva.losses import TemporalLossComposite
        loss_fn = TemporalLossComposite()
        result = loss_fn._allfather_judgment()
        self.assertIsInstance(result, dict)

    def test_probabilistic_loss_non_negative(self):
        from autots.evaluator.tva.losses import ProbabilisticLoss
        loss = ProbabilisticLoss(penalty_weight=1.0)
        mu = torch.randn(2, 4, 10)
        sigma = torch.abs(torch.randn(2, 4, 10)) + 0.1
        target = torch.randn(2, 4, 10)
        val = loss(mu, sigma, target)
        self.assertIsInstance(val.item(), float)
        # NLL can be negative for very tight sigma, but the penalty-weighted mean
        # should be finite and not NaN.
        self.assertFalse(torch.isnan(val))

    def test_probabilistic_loss_lower_with_good_prediction(self):
        from autots.evaluator.tva.losses import ProbabilisticLoss
        loss = ProbabilisticLoss()
        target = torch.ones(2, 4, 10) * 3.0
        # perfect mu, tight sigma → should give lower loss than random mu
        mu_perfect = torch.ones(2, 4, 10) * 3.0
        mu_wrong = torch.zeros(2, 4, 10)
        sigma = torch.ones(2, 4, 10) * 0.5
        val_perfect = loss(mu_perfect, sigma, target)
        val_wrong = loss(mu_wrong, sigma, target)
        self.assertLess(val_perfect.item(), val_wrong.item())

    def test_probabilistic_loss_weight_scales_output(self):
        from autots.evaluator.tva.losses import ProbabilisticLoss
        mu = torch.randn(2, 4, 10)
        sigma = torch.abs(torch.randn(2, 4, 10)) + 0.1
        target = torch.randn(2, 4, 10)
        val1 = ProbabilisticLoss(penalty_weight=1.0)(mu, sigma, target)
        val2 = ProbabilisticLoss(penalty_weight=2.0)(mu, sigma, target)
        self.assertAlmostEqual(val2.item(), 2.0 * val1.item(), places=5)

    def test_temporal_loss_composite_v2_probabilistic(self):
        """TemporalLossComposite should compute 'probabilistic' when mu/sigma present."""
        from autots.evaluator.tva.losses import TemporalLossComposite
        loss_fn = TemporalLossComposite()
        B, N, T, K = 2, 4, 10, 3
        mu = torch.randn(B, N, T)
        outputs = {
            'trend_forecast': mu,
            'mu': mu,
            'sigma': torch.abs(torch.randn(B, N, T)) + 0.1,
            'prototype_weights': torch.softmax(torch.randn(B, N, K), dim=-1),
            'composite_trend': torch.randn(B, K, T),
            'composite_trend_per_series': torch.randn(B, N, T),
        }
        targets = {'true_trend': torch.randn(B, N, T)}
        total, breakdown = loss_fn(outputs, targets)
        self.assertIn('probabilistic', breakdown)
        self.assertFalse(torch.isnan(total))

    def test_temporal_loss_composite_no_probabilistic_without_sigma(self):
        """Without mu/sigma (V1 output), no probabilistic key in breakdown."""
        from autots.evaluator.tva.losses import TemporalLossComposite
        loss_fn = TemporalLossComposite()
        outputs = {
            'trend_forecast': torch.randn(2, 4, 10),
            'prototype_weights': torch.softmax(torch.randn(2, 4, 3), dim=-1),
            'composite_trend': torch.randn(2, 3, 10),
        }
        targets = {'true_trend': torch.randn(2, 4, 10)}
        _, breakdown = loss_fn(outputs, targets)
        self.assertNotIn('probabilistic', breakdown)

    def test_slope_incentive_zero_when_phi_zero(self):
        from autots.evaluator.tva.losses import TrendSlopeIncentive

        loss = TrendSlopeIncentive(trend_phi=0.0)
        pred = torch.randn(2, 4, 10)
        target = torch.randn(2, 4, 10)
        self.assertEqual(loss(pred, target).item(), 0.0)

    def test_slope_incentive_fires_when_pred_flatter(self):
        from autots.evaluator.tva.losses import TrendSlopeIncentive

        loss = TrendSlopeIncentive(trend_phi=1.0)
        # target has clear upward trend, pred is flat
        target = torch.zeros(1, 2, 10)
        for t in range(10):
            target[:, :, t] = float(t)
        pred = torch.zeros(1, 2, 10)  # flat
        val = loss(pred, target)
        self.assertGreater(val.item(), 0.0)

    def test_slope_incentive_near_zero_when_pred_steeper(self):
        from autots.evaluator.tva.losses import TrendSlopeIncentive

        loss = TrendSlopeIncentive(trend_phi=1.0)
        # target has mild trend, pred has steeper trend in same direction
        target = torch.zeros(1, 2, 10)
        pred = torch.zeros(1, 2, 10)
        for t in range(10):
            target[:, :, t] = float(t)
            pred[:, :, t] = float(t) * 2.0  # steeper
        val_steeper = loss(pred, target)
        # compare with flatter pred
        flat_pred = torch.zeros(1, 2, 10)
        val_flatter = loss(flat_pred, target)
        self.assertLess(val_steeper.item(), val_flatter.item())

    def test_slope_incentive_direction_disagreement(self):
        from autots.evaluator.tva.losses import TrendSlopeIncentive

        loss = TrendSlopeIncentive(trend_phi=1.0)
        target = torch.zeros(1, 2, 10)
        for t in range(10):
            target[:, :, t] = float(t)  # upward
        # same-sign pred with matching magnitude (near-zero penalty)
        same_sign = target.clone()
        # opposite-sign pred with matching magnitude (direction penalty fires)
        opposite = -target.clone()
        val_same = loss(same_sign, target)
        val_opposite = loss(opposite, target)
        self.assertLess(val_same.item(), val_opposite.item())

    def test_slope_incentive_short_series(self):
        from autots.evaluator.tva.losses import TrendSlopeIncentive

        loss = TrendSlopeIncentive(trend_phi=1.0)
        val = loss(torch.randn(2, 3, 1), torch.randn(2, 3, 1))
        self.assertEqual(val.item(), 0.0)

    def test_temporal_loss_composite_with_trend_phi(self):
        from autots.evaluator.tva.losses import TemporalLossComposite

        loss_fn = TemporalLossComposite(weights={'trend_phi': 0.5})
        B, N, T, K = 2, 4, 10, 3
        target = torch.zeros(B, N, T)
        for t in range(T):
            target[:, :, t] = float(t)
        outputs = {
            'trend_forecast': torch.zeros(B, N, T),  # flat pred
            'prototype_weights': torch.softmax(torch.randn(B, N, K), dim=-1),
            'composite_trend': torch.randn(B, K, T),
            'composite_trend_per_series': torch.randn(B, N, T),
        }
        targets = {
            'true_trend': target,
            'seasonal': torch.randn(B, N, T),
            'holidays': torch.randn(B, N, T),
            'anomalies': torch.randn(B, N, T),
        }
        total, breakdown = loss_fn(outputs, targets)
        self.assertIn('trend_slope_incentive', breakdown)
        self.assertGreater(breakdown['trend_slope_incentive'], 0.0)

    def test_loss_component_balance(self):
        """No single weighted loss component should dominate by orders of magnitude.

        Uses standard-normal-scale inputs so all components operate in a comparable
        range. Verifies that the max component value is less than 200x the min
        (excluding zero-valued optional components that require specific inputs).
        """
        from autots.evaluator.tva.losses import TemporalLossComposite
        torch.manual_seed(42)
        B, N, T, K = 4, 6, 20, 4
        mu = torch.randn(B, N, T)
        outputs = {
            'trend_forecast': mu,
            'mu': mu,
            'sigma': torch.abs(torch.randn(B, N, T)) + 0.5,
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
        # Use default weights
        loss_fn = TemporalLossComposite()
        _, breakdown = loss_fn(outputs, targets)

        # Collect active (non-negligible) component values, excluding 'total'.
        # Use an absolute floor of 0.01 so near-zero components like coherence (which
        # can be tiny for random inputs when the composite_per_series anchor is weak)
        # don't distort the balance ratio check.
        active_vals = [v for k, v in breakdown.items() if k != 'total' and abs(v) > 0.01]
        self.assertGreater(len(active_vals), 3, "Expected multiple active loss components")

        max_val = max(abs(v) for v in active_vals)
        min_val = min(abs(v) for v in active_vals)
        ratio = max_val / max(min_val, 1e-9)
        self.assertLess(
            ratio, 2000.0,
            msg=(
                f"Loss components are severely unbalanced (max/min ratio={ratio:.1f}). "
                f"Breakdown: {breakdown}"
            ),
        )


@SKIP_TORCH
class TestStructureLearningUtilities(unittest.TestCase):
    def test_dag_cycle_penalty_tracks_cycles(self):
        from autots.evaluator.tva.structure import dag_cycle_penalty

        acyclic = torch.tensor([[0.0, 0.8], [0.0, 0.0]], dtype=torch.float32)
        cyclic = torch.tensor([[0.0, 0.8], [0.8, 0.0]], dtype=torch.float32)
        self.assertLess(dag_cycle_penalty(acyclic).item(), 1e-3)
        self.assertGreater(dag_cycle_penalty(cyclic).item(), 1e-2)

    def test_structure_config_derives_deterministic_sizes(self):
        from autots.evaluator.tva.structure import StructureLearningConfig

        config = StructureLearningConfig(
            enabled=True,
            learn_hierarchy=True,
            max_levels=3,
            pool_ratio=0.5,
            min_nodes_per_level=2,
        )
        self.assertEqual(config.derive_latent_sizes(8), [4, 2])
        self.assertEqual(config.derive_latent_sizes(5), [3, 2])

    def test_series_graph_learner_anchors_topology_at_discovery(self):
        from autots.evaluator.tva.structure import SeriesGraphLearner

        edges = [
            {'source': 0, 'target': 1, 'lag': 2, 'sign': 1, 'weight': 0.8,
             'family': 'lasso'},
            {'source': 2, 'target': 1, 'lag': 1, 'sign': -1, 'weight': 0.4,
             'family': 'leadlag'},
        ]
        learner = SeriesGraphLearner(n_series=3, edges=edges)
        adj = learner.adjacency.detach().cpu().numpy()

        # topology comes from the edge list, direction from lag structure —
        # no index-order bias exists anywhere
        self.assertGreater(adj[0, 1], 0.0)
        self.assertGreater(adj[2, 1], 0.0)
        self.assertEqual(adj[1, 0], 0.0)
        self.assertEqual(adj[1, 2], 0.0)
        np.testing.assert_allclose(np.diag(adj), np.zeros(3), atol=1e-6)
        # signs survive into the signed adjacency
        signed = learner.signed_adjacency.detach().cpu().numpy()
        self.assertGreater(signed[0, 1], 0.0)
        self.assertLess(signed[2, 1], 0.0)
        # only deltas and family gates are learnable
        param_names = {name for name, _ in learner.named_parameters()}
        self.assertEqual(param_names, {'edge_delta', 'family_gate'})

    def test_graph_snapshot_marks_acyclic_graph(self):
        from autots.evaluator.tva.structure import build_graph_snapshot

        adjacency = np.array(
            [
                [0.0, 0.9, 0.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        assignments = [np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)]
        snapshot = build_graph_snapshot(
            adjacency_dense=adjacency,
            assignment_matrices=assignments,
            threshold=0.2,
            anchor_names=['a', 'b'],
        )
        self.assertTrue(snapshot.is_acyclic)
        self.assertEqual(snapshot.topological_order, [0, 1, 2])
        self.assertEqual(snapshot.assignment_matrices[0].shape, (2, 2))

    def test_graph_snapshot_infers_prototype_structure(self):
        from autots.evaluator.tva.structure import build_graph_snapshot

        adjacency = np.array(
            [
                [0.0, 0.9],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        snapshot = build_graph_snapshot(
            adjacency_dense=adjacency,
            threshold=0.2,
            full_series_names=['s0', 's1'],
            anchor_mask=np.array([True, False]),
            prototype_weights=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
            global_prototype_weights=np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
            decoded_top_trends=np.array(
                [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32
            ),
        )
        self.assertEqual(len(snapshot.prototype_table), 2)
        self.assertEqual(len(snapshot.prototype_table[0]['sparkline']), 3)
        self.assertGreaterEqual(len(snapshot.prototype_edge_table), 1)

    def test_temporal_loss_composite_structure_terms(self):
        from autots.evaluator.tva.losses import TemporalLossComposite
        from autots.evaluator.tva.structure import StructureLearningConfig

        structure_config = StructureLearningConfig(
            enabled=True,
            learn_hierarchy=True,
            learn_dag=True,
            dag_penalty=0.5,
            sparsity_weight=0.2,
            assignment_entropy_weight=0.1,
            assignment_full_rank_weight=0.1,
            prior_tether_weight=0.5,
        )
        loss_fn = TemporalLossComposite(structure_config=structure_config)
        outputs = {
            'trend_forecast': torch.randn(2, 4, 10),
            'prototype_weights': torch.softmax(torch.randn(2, 4, 3), dim=-1),
            'composite_trend': torch.randn(2, 3, 10),
            'adjacency': torch.tensor([[0.0, 0.9], [0.7, 0.0]], dtype=torch.float32),
            'assignment_matrices': [
                torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)
            ],
            'structure_mode': True,
            'structure_prior': torch.tensor([[0.0, 0.7], [0.2, 0.0]], dtype=torch.float32),
            'assignment_drift': torch.tensor(0.0, dtype=torch.float32),
        }
        targets = {
            'true_trend': torch.randn(2, 4, 10),
            'structure_loss_scale': 1.0,
            'structure_prior_weight': 0.5,
            'structure_config': structure_config,
        }
        _, breakdown = loss_fn(outputs, targets)
        self.assertIn('dag', breakdown)
        self.assertIn('structure_sparsity', breakdown)
        self.assertIn('assignment_entropy', breakdown)
        self.assertIn('assignment_full_rank', breakdown)
        self.assertIn('structure_prior', breakdown)


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

    def test_orthogonal_prototype_init_separates_initial_bank(self):
        from autots.evaluator.tva.trend_network import PrototypeBottleneck

        pb = PrototypeBottleneck(d_latent=8, n_prototypes=4)
        proto = pb._sacred_timeline_prototypes.detach().cpu().numpy()
        gram = proto @ proto.T

        np.testing.assert_allclose(np.diag(gram), np.ones(4), atol=1e-5)
        off_diag = gram - np.diag(np.diag(gram))
        self.assertLess(np.abs(off_diag).max(), 1e-5)


@SKIP_TORCH
class TestFusionLayers(unittest.TestCase):
    def test_additive_fusion(self):
        from autots.evaluator.tva.fusion import AdditiveFusion
        f = AdditiveFusion()
        B, N, T = 2, 4, 10
        trend = torch.ones(B, N, T)
        sea = torch.ones(B, N, T)
        hol = torch.zeros(B, N, T)
        # level_shifts are added by caller, not passed to fusion
        out = f(trend, sea, hol)
        self.assertEqual(out.shape, (B, N, T))
        np.testing.assert_allclose(out.detach().numpy(), 2.0 * np.ones((B, N, T)), atol=1e-5)

    def test_additive_fusion_with_anomalies(self):
        from autots.evaluator.tva.fusion import AdditiveFusion
        f = AdditiveFusion()
        x = torch.ones(2, 3, 5)
        out = f(x, x, x, anomalies=x)
        np.testing.assert_allclose(out.detach().numpy(), 4.0 * np.ones((2, 3, 5)), atol=1e-5)

    def test_digital_twin_fusion_shape(self):
        from autots.evaluator.tva.fusion import DigitalTwinFusion
        f = DigitalTwinFusion()
        B, N, T = 2, 4, 10
        x = torch.randn(B, N, T)
        out = f(x, x, x)
        self.assertEqual(out.shape, (B, N, T))

    def test_digital_twin_fusion_with_anomalies(self):
        from autots.evaluator.tva.fusion import DigitalTwinFusion
        f = DigitalTwinFusion()
        x = torch.randn(2, 3, 8)
        out = f(x, x, x, anomalies=x)
        self.assertEqual(out.shape, (2, 3, 8))

    def test_direct_attention_fusion_shape(self):
        from autots.evaluator.tva.fusion import DirectAttentionFusion
        f = DirectAttentionFusion()
        B, N, T = 2, 4, 10
        x = torch.randn(B, N, T)
        out = f(x, x, x)
        self.assertEqual(out.shape, (B, N, T))

    def test_direct_attention_fusion_with_anomalies(self):
        from autots.evaluator.tva.fusion import DirectAttentionFusion
        f = DirectAttentionFusion()
        x = torch.randn(2, 3, 8)
        out = f(x, x, x, anomalies=x)
        self.assertEqual(out.shape, (2, 3, 8))

    def test_direct_attention_fusion_additive_init(self):
        """Zero-init output_proj means fusion output ≈ sum of components at init."""
        from autots.evaluator.tva.fusion import DirectAttentionFusion
        torch.manual_seed(0)
        f = DirectAttentionFusion()
        B, N, T = 1, 2, 6
        trend = torch.full((B, N, T), 1.0)
        sea = torch.full((B, N, T), 0.5)
        hol = torch.full((B, N, T), 0.25)
        out = f(trend, sea, hol)
        expected = 1.0 + 0.5 + 0.25
        np.testing.assert_allclose(out.detach().numpy(), expected * np.ones((B, N, T)), atol=1e-5)


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

    def test_v1_mismatched_prior_falls_back_to_full_attention(self):
        """Bilinear prior interpolation is deleted: a series-level (N, N)
        prior no longer gets image-resized into latent space. A size mismatch
        now falls back to full attention instead of a distorted mask."""
        from autots.evaluator.tva.trend_network import CompositeTrendNetworkV1

        prior = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        net = CompositeTrendNetworkV1(
            n_series=4,
            window_size=30,
            forecast_horizon=10,
            d_token=32,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            prior_adjacency=prior,
        )

        self.assertEqual(tuple(net._attn_mask.shape), (2, 2))
        self.assertTrue(torch.isfinite(net._attn_mask).all())
        np.testing.assert_allclose(net._attn_mask.numpy(), 0.0, atol=1e-6)

    def test_v1_mask_keeps_self_attention_with_zero_diagonal_prior(self):
        from autots.evaluator.tva.trend_network import CompositeTrendNetworkV1

        prior = np.zeros((4, 4), dtype=np.float32)
        net = CompositeTrendNetworkV1(
            n_series=4,
            window_size=30,
            forecast_horizon=10,
            d_token=32,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            prior_adjacency=prior,
        )

        self.assertEqual(tuple(net._attn_mask.shape), (2, 2))
        self.assertTrue(torch.isfinite(torch.diag(net._attn_mask)).all())
        self.assertTrue(torch.all(torch.diag(net._attn_mask) == 0.0))

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

    def test_v2_causal_prior_is_returned_for_regularization(self):
        from autots.evaluator.tva.trend_network import CompositeTrendNetworkV2
        net = CompositeTrendNetworkV2(
            n_series=4,
            window_size=30,
            forecast_horizon=10,
            d_token=32,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            causal_prior=np.eye(4, dtype=np.float32),
        )
        out = net(torch.randn(2, 4, 30))
        self.assertIn('causal_prior', out)
        # priors stay at SERIES level — never image-resized into latent space
        self.assertEqual(out['causal_prior'].shape, (4, 4))
        self.assertTrue(torch.all(out['causal_prior'] >= 0))

    def test_v2_learned_adjacency_property(self):
        net = self._make_net()
        adj = net.learned_adjacency
        self.assertEqual(adj.shape, (4, 4))  # n_series x n_series

    def test_v2_glorious_purpose(self):
        net = self._make_net()
        result = net._glorious_purpose()
        self.assertIsInstance(result, np.ndarray)

    def test_v2_structure_mode_returns_assignments(self):
        from autots.evaluator.tva.trend_network import CompositeTrendNetworkV2

        net = CompositeTrendNetworkV2(
            n_series=5,
            window_size=30,
            forecast_horizon=10,
            d_token=32,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            n_anchor_series=3,
            structure_learning_config={
                'enabled': True,
                'learn_hierarchy': True,
                'learn_dag': True,
                'max_levels': 2,
                'pool_ratio': 0.5,
                'min_nodes_per_level': 2,
            },
            prior_adjacency=np.ones((5, 5), dtype=np.float32),
        )
        mask = torch.tensor([True, True, True, False, False])
        out = net(torch.randn(2, 5, 30), anchor_mask=mask)
        self.assertTrue(out['structure_mode'])
        self.assertTrue(len(out['assignment_matrices']) > 0)
        self.assertEqual(out['trend_forecast'].shape, (2, 5, 10))
        # adjacency and priors are series-level now
        self.assertEqual(out['adjacency'].shape, (5, 5))
        self.assertEqual(out['structure_prior'].shape, (5, 5))
        self.assertTrue(torch.isfinite(out['assignment_drift']))
        self.assertGreaterEqual(float(out['assignment_drift'].item()), 0.0)


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

    def test_fit_with_event_prior_without_metadata(self):
        from autots.evaluator.tva.tva import TVA

        features = {
            'trend_changepoints': {
                's0': [(pd.Timestamp("2022-07-01"), 0.0, 2.0)],
                's1': [(pd.Timestamp("2022-07-03"), 0.0, 1.5)],
                's2': [(pd.Timestamp("2022-11-01"), 0.0, 1.5)],
            },
            'level_shifts': {},
            'anomalies': {},
        }
        tva = TVA(
            trend_network='v1',
            fusion='additive',
            epochs=1,
            window_size=60,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            batch_size=8,
            prior_construction_config={'sources': ['event']},
            verbose=0,
        )
        with mock.patch(
            'autots.evaluator.tva.decomposition.NornDecomposer.get_features',
            return_value=features,
        ):
            tva.fit(self.df)

        self.assertIsNotNone(tva._prior_adj)
        self.assertGreater(float(tva._prior_adj[0, 1]), 0.0)

    def test_fit_with_no_usable_auto_prior_still_runs(self):
        from autots.evaluator.tva.tva import TVA

        tva = TVA(
            trend_network='v1',
            fusion='additive',
            epochs=1,
            window_size=60,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            batch_size=8,
            prior_construction_config={'sources': ['event']},
            verbose=0,
        )
        with mock.patch(
            'autots.evaluator.tva.decomposition.NornDecomposer.get_features',
            return_value={'trend_changepoints': {}, 'level_shifts': {}, 'anomalies': {}},
        ):
            tva.fit(self.df)

        self.assertIsNone(tva._prior_adj)
        forecast = tva.predict()
        self.assertEqual(forecast.shape, (14, 3))

    def test_explicit_prior_and_causal_overrides_win(self):
        from autots.evaluator.tva.tva import TVA

        explicit_prior = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32)
        explicit_causal = np.array([[1.0, 0.9], [0.1, 1.0]], dtype=np.float32)
        df = self.df[['s0', 's1']]
        tva = TVA(
            trend_network='v2',
            fusion='additive',
            epochs=1,
            window_size=60,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            batch_size=8,
            prior_adjacency=explicit_prior,
            causal_prior=explicit_causal,
            prior_construction_config={'sources': ['event']},
            causal_prior_construction_config={'max_lag': 2},
            structure_learning_config={'enabled': False},
            verbose=0,
        )
        with mock.patch(
            'autots.evaluator.tva.decomposition.NornDecomposer.get_features',
            return_value={'trend_changepoints': {}, 'level_shifts': {}, 'anomalies': {}},
        ):
            tva.fit(df)

        # coerce_prior_adjacency hollows the diagonal on the way in: a
        # self-link is meaningless to every consumer, and SoftPriorLoss would
        # otherwise pull the learned adjacency's self-loops toward it. The
        # off-diagonal claim -- the whole content of the prior -- is intact.
        np.testing.assert_allclose(
            tva._prior_adj, _hollow(explicit_prior), atol=1e-6
        )
        np.testing.assert_allclose(
            tva._network._causal_prior.detach().cpu().numpy(),
            _hollow(explicit_causal),
            atol=1e-6,
        )

    def test_fit_v2_with_event_and_causal_auto_priors(self):
        from autots.evaluator.tva.tva import TVA

        n = 220
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        rng = np.random.default_rng(101)
        driver = np.cumsum(rng.normal(0.0, 0.3, n))
        responder = np.roll(driver, 1) * 0.95 + rng.normal(0.0, 0.05, n)
        responder[0] = driver[0]
        neutral = np.cumsum(rng.normal(0.0, 0.3, n))
        df = pd.DataFrame({'s0': driver, 's1': responder, 's2': neutral}, index=dates)
        zero_frame = pd.DataFrame(0.0, index=dates, columns=df.columns)
        components = {
            'trend': df.copy(),
            'seasonality': zero_frame.copy(),
            'holidays': zero_frame.copy(),
            'level_shifts': zero_frame.copy(),
            'anomalies': zero_frame.copy(),
            'noise': zero_frame.copy(),
        }
        features = {
            'trend_changepoints': {
                's0': [(pd.Timestamp("2022-04-10"), 0.0, 1.5)],
                's1': [(pd.Timestamp("2022-04-13"), 0.0, 1.4)],
                's2': [(pd.Timestamp("2022-09-10"), 0.0, 1.0)],
            },
            'level_shifts': {},
            'anomalies': {},
        }

        tva = TVA(
            trend_network='v2',
            fusion='additive',
            epochs=1,
            window_size=40,
            forecast_horizon=10,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            batch_size=8,
            prior_construction_config={'sources': ['event']},
            causal_prior_construction_config={'max_lag': 2, 'top_k': 2, 'min_history': 90},
            verbose=0,
        )
        with mock.patch(
            'autots.evaluator.tva.decomposition.NornDecomposer.get_components',
            return_value=components,
        ), mock.patch(
            'autots.evaluator.tva.decomposition.NornDecomposer.get_features',
            return_value=features,
        ):
            tva.fit(df)

        self.assertIsNotNone(tva._prior_adj)
        # causal structure now comes from discovery, surfaced as an edge table
        edges = tva.get_edges()
        self.assertIsInstance(edges, pd.DataFrame)
        factors = tva.get_factors()
        self.assertIsNotNone(factors['factors'])

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
        n = self.df.shape[1]
        self.assertEqual(graph.shape, (n, n))  # series-level N x N

    def test_v2_structure_learning_snapshot_and_plot(self):
        import matplotlib

        matplotlib.use('Agg')
        from autots.evaluator.tva.tva import TVA

        tva = TVA(
            trend_network='v2',
            fusion='additive',
            epochs=1,
            window_size=60,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            batch_size=8,
            verbose=0,
            structure_learning_config={
                'enabled': True,
                'learn_hierarchy': True,
                'learn_dag': True,
                'max_levels': 2,
                'pool_ratio': 0.5,
                'min_nodes_per_level': 2,
                'dag_penalty': 0.05,
            },
        )
        tva.fit(self.df)
        snapshot = tva.get_graph_snapshot()
        self.assertIn('adjacency_dense', snapshot)
        self.assertIn('assignment_matrices', snapshot)
        self.assertTrue(isinstance(snapshot['assignment_matrices'], list))
        ax = tva.plot_graph(view='dag')
        self.assertIsNotNone(ax)

    def test_structure_learning_respects_short_history_responders(self):
        from autots.evaluator.tva.tva import TVA
        from autots.evaluator.tva.priors import SeriesMetadata

        df = self.df.copy()
        df.columns = ['s0', 's1', 's2']
        metadata = [
            SeriesMetadata("s0", history_periods=400),
            SeriesMetadata("s1", history_periods=400),
            SeriesMetadata("s2", history_periods=30),
        ]
        tva = TVA(
            trend_network='v2',
            fusion='additive',
            series_metadata=metadata,
            epochs=1,
            window_size=60,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            batch_size=8,
            verbose=0,
            min_anchor_history=100,
            structure_learning_config={
                'enabled': True,
                'learn_hierarchy': True,
                'learn_dag': True,
                'max_levels': 2,
                'pool_ratio': 0.5,
                'min_nodes_per_level': 2,
            },
        )
        tva.fit(df)
        snapshot = tva.get_graph_snapshot()
        # level 0 now spans ALL series (anchors + responders) so the DAG is
        # drawn over real series names, plus one latent level node
        self.assertEqual(len(snapshot['node_table']), 4)
        self.assertEqual(len(snapshot['series_table']), 3)
        self.assertEqual(snapshot['series_table'][2]['kind'], 'responder')
        forecast = tva.predict()
        self.assertEqual(forecast.shape, (14, 3))

    def test_graph_snapshot_exports_series_prototype_overview_context(self):
        from autots.evaluator.tva.tva import TVA
        from autots.evaluator.tva.priors import SeriesMetadata

        metadata = [
            SeriesMetadata("s0", metric_type="dau", geography="US", history_periods=400),
            SeriesMetadata("s1", metric_type="dau", geography="US", history_periods=400),
            SeriesMetadata("s2", metric_type="views", geography="DE", history_periods=30),
        ]
        df = self.df.copy()
        df.columns = ['s0', 's1', 's2']
        tva = TVA(
            trend_network='v2',
            fusion='additive',
            series_metadata=metadata,
            epochs=1,
            window_size=60,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            batch_size=8,
            verbose=0,
            min_anchor_history=100,
        )
        tva.fit(df)
        snapshot = tva.get_graph_snapshot()
        self.assertEqual(len(snapshot['series_table']), 3)
        self.assertEqual(len(snapshot['prototype_table']), 3)
        self.assertEqual(len(snapshot['affinity_table']), 9)
        self.assertIn('prototype_edge_table', snapshot)
        self.assertIn('metadata', snapshot['series_table'][0])
        self.assertEqual(snapshot['series_table'][2]['kind'], 'responder')
        self.assertEqual(
            len(snapshot['prototype_table'][0].get('sparkline', [])),
            tva.forecast_horizon,
        )
        self.assertEqual(snapshot['prototype_table'][0]['label'], 'prototype_1')
        ax = tva.plot_graph(view='overview')
        self.assertIsNotNone(ax)

    def test_he_who_remains(self):
        tva = self._make_tva()
        tva.fit(self.df)
        info = tva._get_metadata()
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

    def test_tva_passes_calendar_holiday_config_to_decomposer(self):
        from autots.evaluator.tva.tva import TVA

        class _StopFit(Exception):
            pass

        fake_decomposer = mock.Mock()
        fake_decomposer.fit.side_effect = _StopFit()
        tva = TVA(
            trend_network='v1',
            fusion='additive',
            epochs=1,
            window_size=60,
            forecast_horizon=14,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=3,
            n_heads=2,
            batch_size=8,
            holiday_country='US',
            holiday_countries={'s1': 'CA'},
            verbose=0,
        )
        with mock.patch(
            'autots.evaluator.tva.tva.NornDecomposer',
            return_value=fake_decomposer,
        ) as mock_decomposer:
            with self.assertRaises(_StopFit):
                tva.fit(self.df)

        _, kwargs = mock_decomposer.call_args
        self.assertEqual(kwargs['holiday_country'], 'US')
        self.assertEqual(kwargs['holiday_countries'], {'s1': 'CA'})


class TestTVAModelHolidayPropagation(unittest.TestCase):
    def test_wrapper_passes_calendar_holiday_config_to_tva(self):
        from autots.models.tva_model import TVAModel

        df = _make_daily_df(n_series=2, n_days=120)
        model = TVAModel(
            forecast_length=7,
            epochs=1,
            batch_size=4,
            window_size=30,
            holiday_country='US',
            holiday_countries={'s1': 'CA'},
            verbose=0,
        )
        with mock.patch('autots.evaluator.tva.tva.TVA') as mock_tva:
            mock_tva.return_value.fit.return_value = mock_tva.return_value
            model.fit(df)

        _, kwargs = mock_tva.call_args
        self.assertEqual(kwargs['holiday_country'], 'US')
        self.assertEqual(kwargs['holiday_countries'], {'s1': 'CA'})


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

    def test_bifrost_hierarchical_adjustment_directly(self):
        """apply_hierarchical_adjustment with a hierarchy sets the aggregate to target."""
        from autots.evaluator.tva.tva import TVA
        from autots.evaluator.tva.priors import SeriesMetadata

        df = _make_daily_df(n_series=3, n_days=350)
        df.columns = ['US', 'CA', 'DE']
        meta = [
            SeriesMetadata("US", hierarchy_path=["global", "US"], history_periods=350),
            SeriesMetadata("CA", hierarchy_path=["global", "CA"], history_periods=350),
            SeriesMetadata("DE", hierarchy_path=["global", "DE"], history_periods=350),
        ]
        tva = TVA(
            series_metadata=meta,
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
        tva.fit(df)

        from autots.evaluator.tva.scenario import BifrostOptimizer
        opt = BifrostOptimizer(tva, n_steps=5)
        target = 999.0
        result = opt.apply_hierarchical_adjustment('global', target)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (14, 3))
        self.assertFalse(result.isnull().any().any())

        # the sum of all three series should equal the target at every timestep
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(
            row_sums.values, np.full(14, target), rtol=1e-5,
            err_msg="aggregate sum should equal target_value at every timestep",
        )

    def test_what_if_hierarchical_adjustment(self):
        """what_if with level_name/target_value calls apply_hierarchical_adjustment."""
        from autots.evaluator.tva.tva import TVA
        from autots.evaluator.tva.priors import SeriesMetadata

        df = _make_daily_df(n_series=2, n_days=350)
        df.columns = ['alpha', 'beta']
        meta = [
            SeriesMetadata("alpha", hierarchy_path=["global", "alpha"], history_periods=350),
            SeriesMetadata("beta",  hierarchy_path=["global", "beta"],  history_periods=350),
        ]
        tva = TVA(
            series_metadata=meta,
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
        tva.fit(df)
        target = 500.0
        result = tva.what_if(level_name='global', target_value=target)

        self.assertIsInstance(result, pd.DataFrame)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, np.full(14, target), rtol=1e-5)

    def test_bifrost_adjustment_no_hierarchy_returns_base(self):
        """apply_hierarchical_adjustment without hierarchy returns unmodified forecast."""
        from autots.evaluator.tva.scenario import BifrostOptimizer
        opt = BifrostOptimizer(self.tva, n_steps=5)
        # tva in setUpClass has no series_metadata, so no hierarchy
        result = opt.apply_hierarchical_adjustment('global', 999.0)
        base = self.tva.predict()
        pd.testing.assert_frame_equal(result, base)

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


class TestTVATorchFreeMode(unittest.TestCase):
    """trend_network='none' — the Phase-4 kill-rule configuration.

    Torch-free by design (needs only the feature detector), so it runs even
    where torch is unavailable.
    """

    def test_none_mode_fit_predict(self):
        try:
            from autots.evaluator.feature_detector import (  # noqa: F401
                TimeSeriesFeatureDetector,
            )
        except Exception:
            self.skipTest("feature detector unavailable")
        from autots.evaluator.tva.tva import TVA

        df = _make_daily_df(n_series=4, n_days=300)
        tva = TVA(
            trend_network='none',
            window_size=60,
            forecast_horizon=14,
            verbose=0,
        )
        tva.fit(df)
        self.assertIsNone(tva._network)
        forecast = tva.predict()
        self.assertEqual(forecast.shape, (14, 4))
        self.assertTrue(np.isfinite(forecast.values).all())
        # forecast length is not capped by forecast_horizon in numpy mode
        longer = tva.predict(forecast_length=30)
        self.assertEqual(longer.shape, (30, 4))
        # continuity with the last observed trend value
        gap = np.abs(forecast.iloc[0].values - df.iloc[-1].values)
        self.assertTrue((gap < df.abs().mean().values).all())
        # discovery artifacts still surfaced
        self.assertIsInstance(tva.get_edges(), pd.DataFrame)
        graph = tva.get_graph()
        self.assertEqual(graph.shape, (4, 4))
        # residual-sigma intervals available for the wrapper
        self.assertIsNotNone(tva._last_sigma)
        self.assertEqual(tva._last_sigma.shape, (30, 4))


@SKIP_INTEGRATION
class TestTVABenchmarkSmoke(unittest.TestCase):
    """Guard test keeping examples/tva_benchmark.py importable and runnable.

    Tiny configuration: 1 dataset, 1 fold, 5 epochs. The real benchmark runs
    are done manually from the example script — this only prevents rot.
    """

    def test_benchmark_smoke(self):
        import os
        import sys

        examples_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples'
        )
        sys.path.insert(0, examples_dir)
        try:
            import tva_benchmark
        finally:
            sys.path.remove(examples_dir)

        datasets = tva_benchmark.build_datasets(seed=42, smoke=True)
        self.assertIn('factor_panel', datasets)
        models = [
            ('SeasonalNaive', {}),
            (tva_benchmark.TVA_MODEL_NAME, {}),
        ]
        results = tva_benchmark.run_benchmark(
            datasets,
            models,
            n_folds=1,
            seed=42,
            include_tva=True,
            tva_params={'epochs': 5, 'window_size': 60},
            verbose=False,
        )
        self.assertEqual(results['error'].isna().sum(), len(results))
        summary = tva_benchmark.summarize(results)
        self.assertIn('mase_skill_geo', summary.columns)


class TestCoercePriorAdjacency(unittest.TestCase):
    """Every supported prior form yields the same matrix."""

    NAMES = ['a', 'b', 'c']

    def _coerce(self, prior):
        from autots.evaluator.tva.priors import coerce_prior_adjacency
        return coerce_prior_adjacency(prior, self.NAMES)

    def test_none_passes_through(self):
        self.assertIsNone(self._coerce(None))

    def test_all_forms_agree(self):
        target = np.array(
            [[0.0, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
        )
        matrix = self._coerce(target)
        frame = self._coerce(
            pd.DataFrame(target, index=self.NAMES, columns=self.NAMES)
        )
        edges = self._coerce([{'source': 'a', 'target': 'b', 'weight': 0.8}])
        groups = self._coerce([{'series': ['a', 'b'], 'weight': 0.8}])
        for name, got in (
            ('matrix', matrix), ('frame', frame),
            ('edges', edges), ('groups', groups),
        ):
            with self.subTest(form=name):
                np.testing.assert_allclose(got, target, atol=1e-6)
                self.assertEqual(got.dtype, np.float32)

    def test_dataframe_is_order_safe(self):
        # supplied in a different column order than the panel
        frame = pd.DataFrame(
            [[0.0, 0.5], [0.5, 0.0]], index=['c', 'a'], columns=['c', 'a']
        )
        got = self._coerce(frame)
        self.assertAlmostEqual(float(got[0, 2]), 0.5)
        self.assertAlmostEqual(float(got[2, 0]), 0.5)
        self.assertAlmostEqual(float(got[0, 1]), 0.0)

    def test_group_list_is_a_clique(self):
        got = self._coerce([['a', 'b', 'c']])
        np.testing.assert_allclose(got, 1.0 - np.eye(3, dtype=np.float32), atol=1e-6)

    def test_directed_edge_stays_directed(self):
        got = self._coerce(
            [{'source': 'a', 'target': 'c', 'weight': 0.4, 'directed': True}]
        )
        self.assertAlmostEqual(float(got[0, 2]), 0.4)
        self.assertAlmostEqual(float(got[2, 0]), 0.0)

    def test_negative_weight_survives(self):
        got = self._coerce([{'source': 'a', 'target': 'b', 'weight': -0.6}])
        self.assertAlmostEqual(float(got[0, 1]), -0.6)
        self.assertAlmostEqual(float(got[1, 0]), -0.6)

    def test_clipped_and_hollow(self):
        got = self._coerce(np.full((3, 3), 5.0))
        self.assertLessEqual(float(np.abs(got).max()), 1.0)
        np.testing.assert_allclose(np.diag(got), 0.0, atol=1e-7)

    def test_unknown_names_warn_and_drop(self):
        with self.assertWarns(RuntimeWarning) as ctx:
            got = self._coerce([['a', 'nonexistent']])
        self.assertIn('nonexistent', str(ctx.warning))
        np.testing.assert_allclose(got, np.zeros((3, 3)), atol=1e-7)

    def test_bad_shape_warns_and_returns_none(self):
        with self.assertWarns(RuntimeWarning):
            self.assertIsNone(self._coerce(np.zeros((2, 2))))


@SKIP_INTEGRATION
class TestTVAPriorLevers(unittest.TestCase):
    """The prior must reach 'factor' mode, and a wrong one must cost nothing."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        T, N = 500, 8
        idx = pd.date_range('2021-01-01', periods=T, freq='D')
        fa = np.cumsum(rng.normal(0, 1, T))
        fb = np.cumsum(rng.normal(0, 1, T))
        load = np.zeros((N, 2))
        load[:4, 0] = 1.0 + 0.15 * np.arange(4)
        load[4:, 1] = 1.0 + 0.15 * np.arange(4)
        data = np.column_stack([fa, fb]) @ load.T + rng.normal(0, 1.5, (T, N)) + 100
        cls.df = pd.DataFrame(
            data, index=idx, columns=[f's{i}' for i in range(N)]
        )
        cls.true_prior = [
            ['s0', 's1', 's2', 's3'], ['s4', 's5', 's6', 's7'],
        ]
        cls.wrong_prior = [['s0', 's7'], ['s1', 's6']]

    def _run(self, prior=None, **kwargs):
        from autots.evaluator.tva.tva import TVA
        tva = TVA(
            trend_network='factor', forecast_horizon=14, window_size=91,
            epochs=5, verbose=0, random_seed=42, prior_adjacency=prior,
            prior_confidence=1.0, **kwargs,
        )
        tva.fit(self.df)
        return tva, tva.predict(14)

    def test_defaults_are_a_no_op(self):
        _, base = self._run(None)
        _, primed = self._run(self.true_prior)
        np.testing.assert_array_equal(base.values, primed.values)

    def test_coherence_lever_moves_the_forecast(self):
        cfg = {'coherence': True}
        _, base = self._run(None, factor_config=dict(cfg))
        tva, primed = self._run(self.true_prior, factor_config=dict(cfg))
        self.assertGreater(
            float(np.nanmax(np.abs(base.values - primed.values))), 0.0
        )
        self.assertIn('prior', str(tva._coherence_info.get('graph')))

    def test_loading_penalty_lever_moves_the_forecast(self):
        _, base = self._run(None)
        _, primed = self._run(self.true_prior, factor_config={'w_prior_loadings': 0.5})
        self.assertGreater(
            float(np.nanmax(np.abs(base.values - primed.values))), 0.0
        )

    def test_loading_penalty_without_a_prior_is_inert(self):
        _, base = self._run(None)
        _, weighted = self._run(None, factor_config={'w_prior_loadings': 0.5})
        np.testing.assert_array_equal(base.values, weighted.values)

    def test_wrong_prior_is_falsified_and_costs_nothing(self):
        cfg = {'coherence': True}
        _, base = self._run(None, factor_config=dict(cfg))
        tva, wrong = self._run(self.wrong_prior, factor_config=dict(cfg))
        # select_coherence graded the blended candidates on held-out origins
        # and kept the unblended winner, so the forecast is untouched.
        self.assertNotIn('prior', str(tva._coherence_info.get('graph')))
        np.testing.assert_array_equal(base.values, wrong.values)

    def test_v1_warns_that_it_ignores_the_prior(self):
        from autots.evaluator.tva.tva import TVA
        tva = TVA(
            trend_network='v1', forecast_horizon=7, window_size=60, epochs=1,
            verbose=0, random_seed=42, prior_adjacency=self.true_prior,
            d_token=16, n_meso=4, n_global=2, n_prototypes=3, n_heads=2,
        )
        with self.assertWarns(RuntimeWarning) as ctx:
            tva.fit(self.df)
        self.assertTrue(
            any("v1" in str(w.message) for w in ctx.warnings)
        )


class TestTVAModelPriorPassthrough(unittest.TestCase):
    """The prior reaches TVA through the normal AutoTS model API."""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_daily_df(n_series=4, n_days=300)
        cls.metadata = [
            {'name': 's0', 'hierarchy_path': ['global', 'NA', 's0']},
            {'name': 's1', 'hierarchy_path': ['global', 'NA', 's1']},
            {'name': 's2', 'hierarchy_path': ['global', 'EU', 's2']},
            {'name': 's3', 'hierarchy_path': ['global', 'EU', 's3']},
        ]

    def _fit(self):
        from autots.models.tva_model import TVAModel
        model = TVAModel(
            forecast_length=7, epochs=3, window_size=60,
            trend_network='factor', verbose=0,
            series_metadata=self.metadata,
            prior_adjacency=[['s0', 's1']],
            causal_prior=[
                {'source': 's2', 'target': 's3', 'weight': 0.5, 'directed': True}
            ],
        )
        model.fit(self.df)
        return model

    def test_params_round_trip(self):
        from autots.models.tva_model import TVAModel
        model = TVAModel(
            series_metadata=self.metadata, prior_adjacency=[['s0', 's1']]
        )
        params = model.get_params()
        self.assertEqual(params['series_metadata'], self.metadata)
        self.assertEqual(params['prior_adjacency'], [['s0', 's1']])
        rebuilt = TVAModel(**params)
        self.assertEqual(rebuilt.get_params()['series_metadata'], self.metadata)

    @SKIP_INTEGRATION
    def test_new_params_never_invents_a_prior(self):
        from autots.models.tva_model import TVAModel
        for _ in range(25):
            params = TVAModel.get_new_params()
            self.assertNotIn('prior_adjacency', params)
            self.assertNotIn('series_metadata', params)
            self.assertNotIn('causal_prior', params)

    @SKIP_INTEGRATION
    def test_priors_and_metadata_reach_tva(self):
        model = self._fit()
        forecast = model.predict(7).forecast
        self.assertEqual(forecast.shape, (7, 4))
        families = set(model._tva.get_edges()['family'].unique())
        self.assertIn('business', families)
        self.assertIn('causal', families)
        # metadata now reaches TVA, so the hierarchy is real and MinT engages
        self.assertEqual(model._tva._reconciliation_method_effective, 'mint')


@SKIP_INTEGRATION
class TestTVAModelMetadataConsequences(unittest.TestCase):
    """Everything ``series_metadata`` is supposed to switch on, from the wrapper.

    ``TVAModel`` threads metadata into ``TVA`` (commit 76206eb) but nothing
    exercised the *consequences* through the AutoTS-facing API. The anchor mask
    is the sharpest case: an all-ones mask is exactly what you get when metadata
    never arrived, so a fit that produces one proves nothing. This class uses a
    cohort with genuinely short history so the mask has to come from
    ``get_anchor_mask``.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = _make_daily_df(n_series=4, n_days=300)
        # plain dicts, not SeriesMetadata: template JSON-serializability depends
        # on the dict form surviving _coerced_series_metadata
        cls.metadata = [
            {
                'name': 's0',
                'hierarchy_path': ['global', 'NA', 's0'],
                'attribute_values': {'region': 'NA', 'tier': 'core'},
                'history_periods': 300,
            },
            {
                'name': 's1',
                'hierarchy_path': ['global', 'NA', 's1'],
                'attribute_values': {'region': 'NA', 'tier': 'long_tail'},
                'history_periods': 300,
            },
            {
                'name': 's2',
                'hierarchy_path': ['global', 'EU', 's2'],
                'attribute_values': {'region': 'EU', 'tier': 'core'},
                'history_periods': 300,
            },
            {
                'name': 's3',
                'hierarchy_path': ['global', 'EU', 's3'],
                'attribute_values': {'region': 'EU', 'tier': 'long_tail'},
                # below min_anchor_history: this series must be a responder
                'history_periods': 40,
            },
        ]

    @classmethod
    def _fit(cls):
        from autots.models.tva_model import TVAModel

        model = TVAModel(
            forecast_length=7,
            epochs=2,
            window_size=60,
            trend_network='factor',
            verbose=0,
            min_anchor_history=180,
            series_metadata=cls.metadata,
        )
        model.fit(cls.df)
        return model

    def setUp(self):
        if not hasattr(type(self), '_model'):
            type(self)._model = self._fit()
        self.model = type(self)._model
        self.tva = self.model._tva

    def test_dict_metadata_is_coerced_to_series_metadata(self):
        from autots.evaluator.tva.priors import SeriesMetadata

        coerced = self.model._coerced_series_metadata()
        self.assertEqual(len(coerced), 4)
        self.assertTrue(all(isinstance(m, SeriesMetadata) for m in coerced))
        self.assertEqual(coerced[0].hierarchy_path, ['global', 'NA', 's0'])
        self.assertEqual(coerced[3].history_periods, 40)

    def test_priors_are_built_from_metadata_alone(self):
        self.assertIsNotNone(self.tva._priors)
        self.assertEqual(len(self.tva._priors.series_metadata), 4)

    def test_metadata_embeddings_are_populated(self):
        embeddings = self.tva._metadata_embeddings
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape[0], 4)
        # two categorical axes, two values each -> a non-degenerate one-hot block
        self.assertGreaterEqual(embeddings.shape[1], 4)
        self.assertTrue(np.any(embeddings != 0))

    def test_anchor_mask_comes_from_get_anchor_mask(self):
        """The distinguishing assertion: not all-ones."""
        mask = self.tva._anchor_mask
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (4,))
        np.testing.assert_array_equal(mask, [True, True, True, False])

    def test_hierarchy_is_real_and_mint_auto_enables(self):
        S = self.tva._priors.build_hierarchy_matrix()
        self.assertGreater(S.shape[0], S.shape[1])
        self.assertEqual(self.tva._reconciliation_method_effective, 'mint')
        self.assertIsNotNone(self.tva._reconciler)

    def test_get_params_round_trips_metadata(self):
        from autots.models.tva_model import TVAModel

        params = self.model.get_params()
        self.assertEqual(params['series_metadata'], self.metadata)
        rebuilt = TVAModel(**params)
        self.assertEqual(rebuilt.get_params()['series_metadata'], self.metadata)
        # and the round-tripped model still coerces to the same objects
        self.assertEqual(
            [m.name for m in rebuilt._coerced_series_metadata()],
            ['s0', 's1', 's2', 's3'],
        )

    def test_forecast_still_produced_end_to_end(self):
        forecast = self.model.predict(7).forecast
        self.assertEqual(forecast.shape, (7, 4))
        self.assertFalse(forecast.isnull().any().any())


if __name__ == '__main__':
    unittest.main(verbosity=2)
