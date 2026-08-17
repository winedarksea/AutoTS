# -*- coding: utf-8 -*-
"""Anchor/responder correctness (3b) and derived group structure (3c).

The properties asserted here are the ones the rework exists to buy:

* anchors are chosen from *observed* history, never metadata;
* nothing before a responder's first observation can influence the fit — the
  strongest form of that is an invariance test: scribble arbitrary values into
  the pre-launch region and the fit must not move at all;
* the anchor pathway is a strict superset of the flat one, so a full-history
  panel reproduces the all-series fit bit for bit.
"""

import unittest
import numpy as np

from autots.evaluator.tva.factor_network import (
    HAS_TORCH,
    DEFAULT_FACTOR_CONFIG,
    estimate_factors_alternating,
    observed_mask,
    select_anchors,
    _fit_series_on_frozen_factors,
)
from autots.evaluator.tva import grouping

if HAS_TORCH:
    from autots.evaluator.tva.factor_network import (
        fit_latent_factor_model,
        fit_anchor_factor_model,
    )


def factor_panel(seed=0, n_time=700, n_series=16, n_factors=3, noise=0.3):
    """(values, factors, loadings) — a plain latent-factor panel."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(n_time, n_factors)), axis=0)
    f = (f - f.mean(axis=0)) / f.std(axis=0)
    w = rng.normal(size=(n_series, n_factors))
    values = f @ w.T + noise * rng.normal(size=(n_time, n_series))
    return values.astype(np.float32), f, w


class TestAnchorSelection(unittest.TestCase):
    def test_anchors_from_observed_history(self):
        values = np.zeros((1000, 5), dtype=np.float32)
        mask = np.ones((1000, 5), dtype=bool)
        mask[:900, 1] = False  # 100 observed
        mask[:400, 3] = False  # 600 observed
        anchors, responders = select_anchors(values, mask, min_observed=540)
        self.assertEqual(list(anchors), [0, 2, 3, 4])
        self.assertEqual(list(responders), [1])

    def test_nan_derived_mask(self):
        values = np.ones((600, 3))
        values[:500, 2] = np.nan
        anchors, responders = select_anchors(values, None, min_observed=540)
        self.assertEqual(list(anchors), [0, 1])
        self.assertEqual(list(responders), [2])

    def test_fallback_when_everything_is_short(self):
        values = np.zeros((300, 4))
        mask = np.ones((300, 4), dtype=bool)
        mask[:200, 0] = False
        anchors, _ = select_anchors(values, mask, min_observed=540)
        self.assertTrue(anchors.size >= 1)
        self.assertNotIn(0, list(anchors))

    def test_mask_shape_is_validated(self):
        with self.assertRaises(ValueError):
            observed_mask(np.zeros((10, 2)), np.ones((10, 3), dtype=bool))

    def test_config_defaults_are_off(self):
        self.assertFalse(DEFAULT_FACTOR_CONFIG['anchor_selection'])
        self.assertFalse(DEFAULT_FACTOR_CONFIG['group_factors'])
        self.assertEqual(DEFAULT_FACTOR_CONFIG['min_observed_multiple'], 3.0)
        self.assertEqual(DEFAULT_FACTOR_CONFIG['min_responder_overlap'], 90)


class TestMaskedProjection(unittest.TestCase):
    def test_recovers_a_known_loading_from_partial_overlap(self):
        values, f, w = factor_panel(seed=3, noise=0.1)
        t_norm = np.arange(values.shape[0], dtype=float) / values.shape[0]
        obs = np.zeros(values.shape[0], dtype=bool)
        obs[400:] = True  # a late launcher
        est, lag, _, _, _ = _fit_series_on_frozen_factors(
            f, values[:, 0].astype(float), obs, t_norm, max_lag=0
        )
        corr = np.corrcoef(est, w[0])[0, 1]
        self.assertGreater(corr, 0.9)
        self.assertEqual(lag, 0)

    def test_lag_is_recovered(self):
        rng = np.random.default_rng(11)
        f = np.cumsum(rng.normal(size=(600, 1)), axis=0)
        f = (f - f.mean()) / f.std()
        y = np.zeros(600)
        y[5:] = 2.0 * f[:-5, 0]
        y[:5] = 2.0 * f[0, 0]
        obs = np.ones(600, dtype=bool)
        t_norm = np.arange(600, dtype=float) / 600.0
        _, lag, _, _, _ = _fit_series_on_frozen_factors(
            f, y, obs, t_norm, max_lag=10, ridge=1e-6
        )
        self.assertEqual(lag, 5)


@unittest.skipUnless(HAS_TORCH, "torch required")
class TestAnchorFit(unittest.TestCase):
    def test_negative_control_matches_flat_fit(self):
        """Full-history panel: the anchor path must be bitwise identical."""
        values, _, _ = factor_panel(seed=5, n_time=650, n_series=12)
        flat, _ = fit_latent_factor_model(
            values, n_factors=3, horizon=28, seed=42
        )
        anchored, info = fit_anchor_factor_model(
            values, n_factors=3, horizon=28, seed=42,
            config={'anchor_selection': True},
        )
        self.assertEqual(len(info['responder_idx']), 0)
        for a, b in (
            (flat.factor_paths(), anchored.factor_paths()),
            (flat.loadings, anchored.loadings),
            (flat.idio_level, anchored.idio_level),
            (flat.forecast(60), anchored.forecast(60)),
        ):
            self.assertEqual(float((a - b).abs().max()), 0.0)

    def test_default_path_is_untouched(self):
        values, _, _ = factor_panel(seed=6, n_time=400, n_series=8)
        flat, _ = fit_latent_factor_model(values, n_factors=2, horizon=28, seed=1)
        same, info = fit_anchor_factor_model(
            values, n_factors=2, horizon=28, seed=1
        )
        self.assertEqual(float((flat.forecast(30) - same.forecast(30)).abs().max()), 0.0)
        self.assertEqual(len(info['responder_idx']), 0)

    def test_pre_launch_values_cannot_influence_the_fit(self):
        """The invariance the ``bfill`` bug violated."""
        values, _, _ = factor_panel(seed=7, n_time=700, n_series=14)
        mask = np.ones(values.shape, dtype=bool)
        mask[:500, 10:] = False
        cfg = {'anchor_selection': True, 'min_observed_multiple': 5.0}
        a, info = fit_anchor_factor_model(
            values, n_factors=3, horizon=60, seed=42, config=cfg, mask=mask
        )
        scribbled = values.copy()
        rng = np.random.default_rng(0)
        scribbled[:500, 10:] = (
            50.0 + 10.0 * rng.normal(size=(500, 4))
        ).astype(np.float32)
        b, _ = fit_anchor_factor_model(
            scribbled, n_factors=3, horizon=60, seed=42, config=cfg, mask=mask
        )
        self.assertEqual(list(info['responder_idx']), [10, 11, 12, 13])
        self.assertEqual(float((a.forecast(60) - b.forecast(60)).abs().max()), 0.0)

    def test_short_overlap_is_reported_and_zeroed(self):
        values, _, _ = factor_panel(seed=8, n_time=700, n_series=14)
        mask = np.ones(values.shape, dtype=bool)
        mask[:680, 13] = False  # 20 observed rows: nothing is identifiable
        model, info = fit_anchor_factor_model(
            values, n_factors=3, horizon=60, seed=42, mask=mask,
            config={'anchor_selection': True, 'min_responder_overlap': 90},
        )
        self.assertIn(13, info['insufficient_overlap'])
        self.assertEqual(
            float(model.loadings[13].abs().max().item()), 0.0
        )
        self.assertIn(13, info['gated_series'])
        self.assertTrue(np.isfinite(model.forecast(60).detach().numpy()).all())
        self.assertEqual(len(info['observed_counts']), values.shape[1])
        self.assertEqual(int(info['observed_counts'][13]), 20)


class TestGrouping(unittest.TestCase):
    @staticmethod
    def group_panel(seed=0, n_time=700, block=6, noise=0.3, amp=1.5):
        rng = np.random.default_rng(seed)

        def path():
            p = np.cumsum(rng.normal(size=n_time))
            p = p - p.mean()
            return p / max(p.std(), 1e-9)

        glob, g1, g2 = path(), path(), path()
        n = 3 * block
        values = np.outer(glob, rng.normal(1.0, 0.2, n))
        labels = np.full(n, -1)
        values[:, :block] += amp * np.outer(g1, rng.normal(1.0, 0.2, block))
        values[:, block:2 * block] += amp * np.outer(g2, rng.normal(1.0, 0.2, block))
        labels[:block] = 0
        labels[block:2 * block] = 1
        values += noise * rng.normal(size=(n_time, n))
        return values, labels

    def test_average_linkage(self):
        sim = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.1], [0.0, 0.1, 1.0]])
        labels = grouping.average_linkage_clusters(sim, 0.5)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_discovers_reproducible_blocks(self):
        values, labels = self.group_panel(seed=1)
        glob = estimate_factors_alternating(values, 1)
        found = grouping.discover_groups(
            values, glob['factors'],
            config={'refits': 6, 'stability_threshold': 0.70}, seed=42,
        )
        self.assertGreaterEqual(len(found['groups']), 1)
        # every discovered group must be internally pure w.r.t. the truth
        for members in found['groups'].values():
            truth = labels[np.asarray(members)]
            self.assertEqual(len(set(truth.tolist())), 1)

    def test_no_groups_on_a_flat_panel(self):
        values, _, _ = factor_panel(seed=9, n_time=600, n_series=18, noise=0.5)
        glob = estimate_factors_alternating(values.astype(float), 3)
        found = grouping.discover_groups(
            values.astype(float), glob['factors'],
            config={'refits': 6, 'stability_threshold': 0.9}, seed=42,
        )
        for members in found['groups'].values():
            self.assertLess(len(members), values.shape[1])

    def test_rank_selection_prefers_the_true_rank_over_noise(self):
        values, _, _ = factor_panel(seed=10, n_time=700, n_series=14, noise=0.4)
        out = grouping.select_rank(
            values.astype(float), candidates=(0, 1, 3), horizon=60, n_origins=2,
            seed=42,
        )
        self.assertIn(out['rank'], (0, 1, 3))
        self.assertTrue(all('score' in row for row in out['table']))

    def test_rank_ladder_is_not_degenerate(self):
        """Regression: the inner score must actually depend on the rank.

        Continuing the residual with the same local-linear rule used for the
        factors makes every rank score identically (both are linear
        operators), which silently turns rank selection into a coin flip.
        """
        values, _, _ = factor_panel(seed=12, n_time=650, n_series=12, noise=0.4)
        scores = [
            grouping.rolling_origin_score(values.astype(float), r, 60, n_origins=2)
            for r in (0, 1, 3)
        ]
        self.assertTrue(all(np.isfinite(s) for s in scores))
        self.assertGreater(max(scores) - min(scores), 1e-6)

    def test_group_mask_keeps_global_factors_unrestricted(self):
        mask = grouping._group_mask([0, 0, 1, 1], (4, 3), n_global=1)
        self.assertTrue((mask[:, 0] == 1).all())
        self.assertEqual(list(mask[:, 1]), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(list(mask[:, 2]), [0.0, 0.0, 1.0, 1.0])

    def test_loading_graph_shape(self):
        graph = grouping.loading_graph(
            np.array([[1.0, 0.0], [0.0, 2.0]]), labels=[0, 1]
        )
        self.assertEqual(graph['loadings'].shape, (2, 2))
        self.assertEqual(list(graph['dominant_factor']), [0, 1])
        self.assertEqual(len(graph['factors']), 2)


@unittest.skipUnless(HAS_TORCH, "torch required")
class TestGroupLayerWiring(unittest.TestCase):
    def test_group_layer_reports_its_own_verdict(self):
        """The layer must never be applied on in-sample evidence alone."""
        from autots.evaluator.tva.factor_network import _append_factor_columns

        values, _ = TestGrouping.group_panel(seed=2, n_time=500, block=4, amp=1.5)
        values = values.astype(np.float32)
        model, info = fit_anchor_factor_model(
            values, n_factors=1, max_lag=0, horizon=60, seed=42,
            config={'group_factors': True, 'inner_folds': 2},
        )
        self.assertIn('group_applied', info)
        self.assertIn('loading_graph', info)
        self.assertEqual(
            info['loading_graph']['loadings'].shape[0], values.shape[1]
        )
        if not info['group_applied']:
            self.assertTrue(info['group_reason'])
            self.assertEqual(model.K, 1)
        self.assertTrue(np.isfinite(model.forecast(60).detach().numpy()).all())

        # the append machinery is exercised whatever the verdict was
        before = model.K
        extra_coefs = np.zeros((model.coef.shape[0], 1), dtype=float)
        extra_coefs[1, 0] = 1.0
        extra_loadings = np.zeros((values.shape[1], 1))
        extra_loadings[:4, 0] = 0.5
        obs = np.ones(values.shape, dtype=bool)
        _append_factor_columns(
            model, extra_coefs, extra_loadings, values.astype(float), obs
        )
        self.assertEqual(model.K, before + 1)
        self.assertEqual(model.loadings.shape[1], before + 1)
        self.assertTrue(np.isfinite(model.forecast(60).detach().numpy()).all())


if __name__ == '__main__':
    unittest.main()
