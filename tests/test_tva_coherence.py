# -*- coding: utf-8 -*-
"""Tests for the TVA post-forecast coherence correction (Phase 4).

Thresholds are calibrated empirically on the fixed seeds used here and set
below the observed values, following the convention of the rest of the TVA
suite. Everything here is torch-free: the module is pure numpy by design so
that every guarantee (bitwise identity, never-raise, one solver) is unit
testable without fitting a model.
"""

import unittest
import numpy as np

from autots.evaluator.tva.coherence import (
    DEFAULT_COHERENCE_CONFIG,
    apply_selection,
    block_adjacency,
    build_candidates,
    coherence_shrink,
    group_graph,
    laplacian_graph,
    net_direction,
    resolve_signs,
    select_coherence,
)


def ramp_panel(slopes, n_time=28, anchor=0.0):
    """(H, N) straight-line trend paths with the given per-series slopes."""
    t = np.arange(int(n_time), dtype=float)[:, None]
    return anchor + t * np.asarray(slopes, dtype=float)[None, :]


def one_factor_loadings(signs):
    """(N, 1) single-factor loadings with the given signs, unit magnitude."""
    return np.asarray(signs, dtype=float)[:, None]


class TestResolveSigns(unittest.TestCase):
    """The sign-identification fix. Loading-sign agreement with truth measured
    0.490 under the incumbent largest-|loading| convention -- a coin flip."""

    def test_majority_of_loading_mass_negative_flips_the_factor(self):
        # one big positive loading, three larger-in-total negative ones:
        # the incumbent convention orients on the single 1.2, the mass vote
        # orients on the 3 x -1.0 that carry more of the movement
        lam = np.array([[1.2], [-1.0], [-1.0], [-1.0]])
        signed, conf = resolve_signs(lam)
        self.assertLess(float(signed[0, 0]), 0.0)  # flipped relative to input
        self.assertGreater(float(signed[1, 0]), 0.0)
        # the incumbent convention would have kept it as-is
        incumbent = np.sign(lam[np.abs(lam).argmax(axis=0), np.arange(1)])
        self.assertEqual(float(incumbent[0]), 1.0)
        self.assertGreater(float(conf[0]), 0.0)

    def test_confidence_is_low_for_a_genuinely_split_factor(self):
        split = resolve_signs(one_factor_loadings([1.0, -1.0, 1.0, -1.0]))[1]
        decided = resolve_signs(one_factor_loadings([1.0, 1.0, 1.0, -1.0]))[1]
        self.assertAlmostEqual(float(split[0]), 0.0, places=9)
        self.assertGreater(float(decided[0]), float(split[0]))
        self.assertLessEqual(float(decided[0]), 1.0)

    def test_unanimous_factor_has_confidence_one(self):
        conf = resolve_signs(one_factor_loadings([0.8, 1.0, 1.3]))[1]
        self.assertAlmostEqual(float(conf[0]), 1.0, places=9)

    def test_stability_discounts_the_confidence(self):
        lam = one_factor_loadings([1.0, 1.0, 1.0])
        full = resolve_signs(lam)[1]
        damped = resolve_signs(lam, stability=[0.25])[1]
        self.assertAlmostEqual(float(damped[0]), 0.25 * float(full[0]), places=9)

    def test_signed_loadings_are_the_same_model(self):
        rng = np.random.default_rng(0)
        lam = rng.normal(size=(12, 3))
        signed, _ = resolve_signs(lam)
        # sign resolution is a per-column flip, nothing else
        ratio = np.abs(signed) - np.abs(lam)
        np.testing.assert_allclose(ratio, 0.0, atol=1e-12)


class TestGroupGraph(unittest.TestCase):
    def test_groups_split_by_dominant_factor_and_sign(self):
        lam = np.array(
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0],
             [0.0, 1.0], [0.0, 1.0]]
        )
        graph = group_graph(lam)
        self.assertEqual(len(graph['groups']), 3)
        sizes = sorted(len(v) for v in graph['groups'].values())
        self.assertEqual(sizes, [2, 2, 2])
        # every series appears exactly once
        members = sorted(i for v in graph['groups'].values() for i in v)
        self.assertEqual(members, list(range(6)))

    def test_gated_series_are_excluded(self):
        lam = one_factor_loadings([1.0, 1.0, 1.0, 1.0])
        graph = group_graph(lam, {'gated': [2]})
        members = sorted(i for v in graph['groups'].values() for i in v)
        self.assertEqual(members, [0, 1, 3])

    def test_zero_loading_series_are_not_grouped(self):
        lam = one_factor_loadings([1.0, 1.0, 0.0])
        graph = group_graph(lam)
        members = sorted(i for v in graph['groups'].values() for i in v)
        self.assertEqual(members, [0, 1])

    def test_precision_weights_are_positive_and_scaled(self):
        lam = one_factor_loadings([1.0, 2.0, 3.0])
        graph = group_graph(lam, {'sigma': [1.0, 2.0, 1.0]})
        w = np.asarray(graph['weights'])
        self.assertEqual(w.shape, (3,))
        self.assertTrue((w > 0).all())
        # the noisier series is trusted less than its loading alone implies
        self.assertLess(w[1] / w[0], 2.0)


class TestLaplacianGraph(unittest.TestCase):
    def test_signed_symmetric_hollow(self):
        lam = np.array([[1.0, 0.1], [0.9, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        adj = laplacian_graph(lam, n_neighbors=2)
        self.assertEqual(adj.shape, (4, 4))
        np.testing.assert_allclose(adj, adj.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(adj), 0.0, atol=1e-12)
        # opposite exposures get a negative link, not a missing one
        self.assertLess(float(adj[0, 2]), 0.0)
        self.assertGreater(float(adj[0, 1]), 0.0)

    def test_neighbor_count_controls_density(self):
        rng = np.random.default_rng(3)
        lam = rng.normal(size=(20, 3))
        sparse = laplacian_graph(lam, n_neighbors=2)
        dense = laplacian_graph(lam, n_neighbors=8)
        self.assertLess(int((sparse != 0).sum()), int((dense != 0).sum()))

    def test_low_confidence_factor_isolates_its_series(self):
        lam = one_factor_loadings([1.0, -1.0, 1.0, -1.0])  # confidence 0.0
        adj = laplacian_graph(lam, {'min_sign_confidence': 0.5}, n_neighbors=2)
        np.testing.assert_allclose(adj, 0.0, atol=1e-12)


class TestCoherenceShrink(unittest.TestCase):
    def test_strength_zero_is_bitwise_identity(self):
        trend = ramp_panel([1.0, -0.5, 0.2, 0.9])
        graph = group_graph(one_factor_loadings([1.0, 1.0, 1.0, 1.0]))
        out = coherence_shrink(trend, graph, 0.0)
        self.assertEqual(np.asarray(out).tobytes(), trend.tobytes())

    def test_shared_direction_panel_becomes_more_coherent(self):
        # a panel that genuinely shares a direction, with one series whose
        # slope error has flipped it the wrong way
        trend = ramp_panel([1.0, 0.9, 1.1, -0.2])
        graph = group_graph(one_factor_loadings([1.0, 1.0, 1.0, 1.0]))
        before = net_direction(trend)
        out = coherence_shrink(trend, graph, 1.0)
        after = net_direction(out)
        self.assertEqual(float(before[3]), -1.0)
        self.assertEqual(float(after[3]), 1.0)
        # and the majority is not dragged the other way
        self.assertTrue((after > 0).all())

    def test_anchor_row_is_preserved_exactly(self):
        trend = ramp_panel([1.0, 0.9, -0.2], anchor=7.5)
        graph = group_graph(one_factor_loadings([1.0, 1.0, 1.0]))
        out = coherence_shrink(trend, graph, 1.0)
        np.testing.assert_allclose(out[0], trend[0], atol=1e-12)

    def test_divergent_panel_is_not_forced_together(self):
        """The decisiveness guard: an evenly split group must not go flat."""
        trend = ramp_panel([1.0, 1.0, -1.0, -1.0])
        graph = group_graph(one_factor_loadings([1.0, 1.0, 1.0, 1.0]))
        cfg = {'decisiveness_floor': 0.8}
        out = coherence_shrink(trend, graph, 1.0, config=cfg)
        self.assertEqual(np.asarray(out).tobytes(), trend.tobytes())
        # without the guard the same group would be pulled to ~flat
        loose = np.asarray(coherence_shrink(trend, graph, 1.0))
        self.assertLess(
            float(np.abs(loose[-1]).mean()), float(np.abs(trend[-1]).mean())
        )

    def test_low_confidence_factor_is_skipped(self):
        """A coin-flip orientation must not move anything (measured 0.490)."""
        lam = one_factor_loadings([1.0, -1.0, 1.0, -1.0])  # confidence 0.0
        trend = ramp_panel([1.0, 0.9, 1.1, -0.2])
        graph = group_graph(lam)
        out = coherence_shrink(trend, graph, 1.0, {'min_sign_confidence': 0.5})
        self.assertEqual(np.asarray(out).tobytes(), trend.tobytes())

    def test_negative_link_pushes_apart_not_together(self):
        lam = one_factor_loadings([1.0, -1.0])
        adj = laplacian_graph(lam, n_neighbors=1)
        trend = ramp_panel([1.0, 0.2])
        out = np.asarray(coherence_shrink(trend, adj, 1.0))
        # the anti-correlated partner is pushed toward the opposite sign
        self.assertLess(float(out[-1, 1]), float(trend[-1, 1]))

    def test_weights_control_who_moves(self):
        trend = ramp_panel([1.0, -0.2])
        adj = np.array([[0.0, 1.0], [1.0, 0.0]])
        out = np.asarray(
            coherence_shrink(trend, adj, 1.0, weights=np.array([100.0, 0.05]))
        )
        # series 0 is precisely measured, so it barely moves and series 1 does
        self.assertLess(abs(out[-1, 0] - trend[-1, 0]), 0.05 * abs(trend[-1, 0]))
        self.assertGreater(abs(out[-1, 1] - trend[-1, 1]), 0.5 * abs(trend[-1, 1]))

    def test_group_view_equals_laplacian_view(self):
        """One solver, two views: the block adjacency reproduces the group."""
        lam = np.array(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        trend = ramp_panel([1.0, 0.4, -0.3, 0.8, -0.1])
        graph = group_graph(lam)
        w = np.asarray(graph['weights'])
        adj = block_adjacency(graph, trend.shape[1])
        via_group = np.asarray(coherence_shrink(trend, graph, 0.3, weights=w))
        via_lap = np.asarray(coherence_shrink(trend, adj, 0.3, weights=w))
        np.testing.assert_allclose(via_group, via_lap, rtol=1e-12, atol=1e-12)


class TestSelection(unittest.TestCase):
    def _folds(self, pred_slopes, actual_slopes, n_folds=3, h=28):
        preds = [ramp_panel(pred_slopes, h) for _ in range(n_folds)]
        acts = [ramp_panel(actual_slopes, h) for _ in range(n_folds)]
        return preds, acts

    def test_zero_strength_is_always_reachable(self):
        preds, acts = self._folds([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        graphs, _ = build_candidates(one_factor_loadings([1.0, 1.0, 1.0]))
        sel = select_coherence(preds, acts, graphs, np.ones(3))
        # the forecast already matches the future exactly: nothing to gain
        self.assertEqual(sel['strength'], 0.0)
        self.assertIsNone(sel['graph'])

    def test_selects_a_shrinkage_that_fixes_a_flipped_series(self):
        preds, acts = self._folds([1.0, 0.9, -0.15], [1.0, 0.9, 0.85])
        graphs, _ = build_candidates(
            one_factor_loadings([1.0, 1.0, 1.0]), {'graph': 'group'}
        )
        sel = select_coherence(preds, acts, graphs, np.ones(3) * 5.0)
        self.assertGreater(sel['strength'], 0.0)
        self.assertEqual(sel['graph'], 'group')
        self.assertLess(
            sel['selected_coherence_error'], sel['baseline_coherence_error']
        )

    def test_mase_guardrail_vetoes_an_accuracy_destroying_candidate(self):
        """Coherence-improving but accuracy-destroying must lose."""
        preds, acts = self._folds([1.0, 0.9, -0.15], [1.0, 0.9, 0.85])
        graphs, _ = build_candidates(
            one_factor_loadings([1.0, 1.0, 1.0]), {'graph': 'group'}
        )
        # a tiny MASE denominator makes any movement look expensive; with the
        # guardrail at 0 nothing that changes the forecast can be admitted
        tight = select_coherence(
            preds, acts, graphs, np.ones(3), {'mase_guardrail': 0.0}
        )
        loose = select_coherence(
            preds, acts, graphs, np.ones(3), {'mase_guardrail': 10.0}
        )
        self.assertGreater(loose['strength'], 0.0)
        # the vetoed run either does nothing or does strictly less
        self.assertLessEqual(tight['strength'], loose['strength'])
        vetoed = [r for r in tight['table'] if not r['admissible']]
        self.assertTrue(vetoed)

    def test_guardrail_rejects_the_wrong_direction_shrinkage(self):
        # the panel truly diverges; pulling it together must cost MASE and be
        # rejected at the default 2% guardrail
        preds, acts = self._folds([1.0, -1.0], [1.0, -1.0])
        graphs, _ = build_candidates(
            one_factor_loadings([1.0, 1.0]), {'graph': 'group'}
        )
        sel = select_coherence(preds, acts, graphs, np.ones(2) * 0.5)
        self.assertEqual(sel['strength'], 0.0)

    def test_apply_selection_reports_what_it_did(self):
        preds, acts = self._folds([1.0, 0.9, -0.15], [1.0, 0.9, 0.85])
        graphs, meta = build_candidates(
            one_factor_loadings([1.0, 1.0, 1.0]), {'graph': 'group'}
        )
        sel = select_coherence(preds, acts, graphs, np.ones(3) * 5.0)
        out, info = apply_selection(preds[0], sel, graphs)
        self.assertTrue(info['applied'])
        self.assertGreater(info['adjustment_rms'], 0.0)
        self.assertEqual(info['graph'], 'group')
        self.assertEqual(info['n_links'], 3)
        self.assertEqual(len(info['sign_confidence']), 1)
        self.assertEqual(len(meta['sign_confidence']), 1)
        self.assertFalse(np.array_equal(np.asarray(out), preds[0]))

    def test_apply_selection_is_json_safe(self):
        import json

        preds, acts = self._folds([1.0, 0.9, -0.15], [1.0, 0.9, 0.85])
        graphs, _ = build_candidates(one_factor_loadings([1.0, 1.0, 1.0]))
        sel = select_coherence(preds, acts, graphs, np.ones(3) * 5.0)
        _out, info = apply_selection(preds[0], sel, graphs)
        json.dumps(info)
        json.dumps(sel)

    def test_null_selection_is_bitwise_passthrough(self):
        trend = ramp_panel([1.0, 0.5])
        out, info = apply_selection(trend, None, {})
        self.assertEqual(np.asarray(out).tobytes(), trend.tobytes())
        self.assertFalse(info['applied'])
        self.assertEqual(info['adjustment_rms'], 0.0)


class TestDegenerateInputs(unittest.TestCase):
    """Nothing in this module raises. Every bad input returns the input."""

    def test_single_series(self):
        trend = ramp_panel([1.0])
        graph = group_graph(one_factor_loadings([1.0]))
        self.assertEqual(graph['groups'], {})
        out = coherence_shrink(trend, graph, 1.0)
        self.assertEqual(np.asarray(out).tobytes(), trend.tobytes())

    def test_all_zero_loadings(self):
        lam = np.zeros((5, 2))
        signed, conf = resolve_signs(lam)
        np.testing.assert_allclose(conf, 0.0)
        self.assertEqual(group_graph(lam)['groups'], {})
        np.testing.assert_allclose(laplacian_graph(lam), 0.0)
        trend = ramp_panel([1.0, 0.5, -0.2, 0.1, 0.0])
        out = coherence_shrink(trend, laplacian_graph(lam), 1.0)
        self.assertEqual(np.asarray(out).tobytes(), trend.tobytes())

    def test_nan_columns(self):
        lam = np.array([[np.nan, 1.0], [1.0, np.nan], [1.0, 0.0]])
        signed, conf = resolve_signs(lam)
        self.assertTrue(np.isfinite(signed).all())
        self.assertTrue(np.isfinite(conf).all())
        trend = ramp_panel([1.0, 0.5, np.nan])
        graph = group_graph(lam)
        out = coherence_shrink(trend, graph, 1.0)
        self.assertEqual(np.asarray(out).tobytes(), trend.tobytes())

    def test_empty_graph_and_disconnected_nodes(self):
        trend = ramp_panel([1.0, 0.5, -0.2])
        self.assertEqual(
            np.asarray(coherence_shrink(trend, {}, 1.0)).tobytes(), trend.tobytes()
        )
        self.assertEqual(
            np.asarray(coherence_shrink(trend, np.zeros((3, 3)), 1.0)).tobytes(),
            trend.tobytes(),
        )
        # one connected pair, one isolated node: the isolate is untouched
        adj = np.zeros((3, 3))
        adj[0, 1] = adj[1, 0] = 1.0
        out = np.asarray(coherence_shrink(trend, adj, 1.0))
        np.testing.assert_allclose(out[:, 2], trend[:, 2], atol=1e-12)

    def test_singular_and_malformed_graphs(self):
        trend = ramp_panel([1.0, 0.5, -0.2])
        for bad in (np.zeros((2, 2)), np.ones((5, 5)), 'nonsense', None, 3.0):
            out = coherence_shrink(trend, bad, 1.0)
            self.assertTrue(np.isfinite(np.asarray(out, dtype=float)).all())

    def test_empty_folds_and_missing_graphs(self):
        sel = select_coherence([], [], {}, np.ones(3))
        self.assertEqual(sel['strength'], 0.0)
        self.assertIsNone(sel['graph'])
        sel = select_coherence(
            [ramp_panel([1.0])], [ramp_panel([1.0])], {}, np.ones(1)
        )
        self.assertEqual(sel['strength'], 0.0)

    def test_zero_and_nan_mase_scale(self):
        preds = [ramp_panel([1.0, 0.9, -0.15])]
        acts = [ramp_panel([1.0, 0.9, 0.85])]
        graphs, _ = build_candidates(one_factor_loadings([1.0, 1.0, 1.0]))
        sel = select_coherence(preds, acts, graphs, np.array([0.0, np.nan, -1.0]))
        self.assertIn(sel['reason'], ('selected',
                                      'no admissible candidate improved coherence'))

    def test_build_candidates_modes(self):
        lam = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        self.assertEqual(build_candidates(lam, {'graph': 'none'})[0], {})
        self.assertEqual(
            list(build_candidates(lam, {'graph': 'group'})[0]), ['group']
        )
        auto = build_candidates(lam, {'graph': 'auto'})[0]
        self.assertIn('group', auto)
        self.assertTrue(any(k.startswith('laplacian') for k in auto))

    def test_defaults_are_present(self):
        for key in (
            'graph', 'strengths', 'neighbors', 'stability_threshold',
            'mase_guardrail', 'min_sign_confidence', 'decisiveness_floor',
        ):
            self.assertIn(key, DEFAULT_COHERENCE_CONFIG)


if __name__ == '__main__':
    unittest.main()


class TestGraphAbstention(unittest.TestCase):
    """C2: a series with no decisive dominant factor should join no group.

    The shrink pulls every in-group member toward the group consensus, so a
    mis-grouped series is moved in the wrong direction — strictly worse than
    leaving it alone. These knobs buy precision by declining to answer.
    """

    # rows 0-1 unambiguous on f0, rows 2-3 unambiguous on f1, row 4 a coin flip
    LAM = np.array([
        [1.0, 0.05, 0.0],
        [0.9, 0.02, 0.0],
        [0.05, 1.0, 0.0],
        [0.02, 0.9, 0.0],
        [0.72, 0.70, 0.0],
    ])

    def test_defaults_are_an_exact_no_op(self):
        base = group_graph(self.LAM)
        for cfg in ({}, {'dominance_margin': 1.0}, {'min_loading_share': 0.0}):
            self.assertEqual(group_graph(self.LAM, cfg)['groups'],
                             base['groups'])

    def test_ambiguous_row_abstains_at_margin(self):
        base = group_graph(self.LAM)
        members = {i for v in base['groups'].values() for i in v}
        self.assertIn(4, members)
        strict = group_graph(self.LAM, {'dominance_margin': 1.5})
        strict_members = {i for v in strict['groups'].values() for i in v}
        self.assertNotIn(4, strict_members)
        # and the decisive series are untouched
        self.assertTrue({0, 1, 2, 3}.issubset(strict_members))

    def test_ambiguous_row_abstains_at_loading_share(self):
        strict = group_graph(self.LAM, {'min_loading_share': 0.6})
        members = {i for v in strict['groups'].values() for i in v}
        self.assertNotIn(4, members)
        self.assertTrue({0, 1, 2, 3}.issubset(members))

    def test_abstention_never_invents_groups(self):
        for margin in (1.0, 1.25, 1.5, 2.0):
            graph = group_graph(self.LAM, {'dominance_margin': margin})
            base = group_graph(self.LAM)
            for key, members in graph['groups'].items():
                self.assertIn(key, base['groups'])
                self.assertTrue(set(members).issubset(set(base['groups'][key])))

    def test_single_factor_panel_ignores_margin(self):
        # with one column there is no runner-up; the margin test must not fire
        lam = np.array([[1.0], [0.8], [-0.9], [-0.7]])
        graph = group_graph(lam, {'dominance_margin': 2.0})
        self.assertTrue(graph['groups'])

    def test_shrink_is_still_inert_at_strength_zero(self):
        rng = np.random.default_rng(0)
        trends = rng.normal(size=(30, 5)).cumsum(axis=0)
        cfg = {'dominance_margin': 1.5, 'min_loading_share': 0.5}
        out = coherence_shrink(
            trends, group_graph(self.LAM, cfg), strength=0.0
        )
        np.testing.assert_array_equal(out, trends)


class TestStabilityVeto(unittest.TestCase):
    """C4: ``resolve_signs``' long-dead ``stability`` argument, finally wired.

    A factor that neither half of the panel can reproduce still gets a
    confident-looking sign from the mass vote, because the vote only measures
    agreement among loadings — not whether the factor exists. Multiplying in
    a stability score is what lets ``min_sign_confidence`` see the difference.
    """

    # two decisive groups, both with a confident mass-vote orientation, so
    # the only thing that can remove one is the stability veto
    LAM = np.array([
        [1.0, 0.10, 0.0],
        [0.9, 0.10, 0.0],
        [0.1, 1.00, 0.0],
        [0.1, 0.90, 0.0],
    ])

    def test_stability_none_is_the_current_behavior(self):
        base = group_graph(self.LAM)
        self.assertEqual(group_graph(self.LAM, {'stability': None})['groups'],
                         base['groups'])

    def test_stability_scales_the_confidence(self):
        _, plain = resolve_signs(self.LAM)
        _, scaled = resolve_signs(self.LAM, stability=np.array([1.0, 0.25, 1.0]))
        np.testing.assert_allclose(scaled[0], plain[0])
        np.testing.assert_allclose(scaled[1], plain[1] * 0.25)

    def test_unstable_factor_drops_out_of_the_graph(self):
        cfg = {'min_sign_confidence': 0.5}
        with_all = group_graph(self.LAM, cfg)
        vetoed = group_graph(
            self.LAM, dict(cfg, stability=np.array([1.0, 0.1, 1.0]))
        )
        self.assertTrue(any(k.startswith('f1') for k in with_all['groups']))
        self.assertFalse(any(k.startswith('f1') for k in vetoed['groups']))

    def test_wrong_length_stability_is_ignored_not_fatal(self):
        base = group_graph(self.LAM)
        for bad in (np.array([1.0]), np.array([1.0] * 9), 'nonsense'):
            try:
                out = group_graph(self.LAM, {'stability': bad})
            except Exception as exc:  # pragma: no cover
                self.fail(f'group_graph raised on stability={bad!r}: {exc}')
            self.assertEqual(out['groups'], base['groups'])

    def test_candidates_and_laplacian_see_the_veto(self):
        cfg = {'min_sign_confidence': 0.5, 'stability': np.array([1.0, 0.1, 1.0])}
        _graphs, meta = build_candidates(self.LAM, cfg)
        self.assertLess(meta['sign_confidence'][1], 0.5)
        adj = laplacian_graph(self.LAM, cfg)
        self.assertEqual(adj.shape, (4, 4))


class TestPriorBlending(unittest.TestCase):
    """A user prior only ever *adds* candidates; it never rewrites the
    existing ones, and the blended adjacency keeps the graph's invariants."""

    LAM = np.array([
        [1.0, 0.05],
        [0.9, 0.10],
        [0.05, 1.0],
        [0.10, 0.9],
    ])

    def _prior(self):
        prior = np.zeros((4, 4), dtype=float)
        prior[0, 3] = prior[3, 0] = 1.0
        return prior

    def test_no_prior_leaves_the_candidate_dict_unchanged(self):
        base, base_meta = build_candidates(self.LAM)
        again, _ = build_candidates(self.LAM, prior=None)
        self.assertEqual(sorted(base), sorted(again))
        for name, graph in base.items():
            if isinstance(graph, np.ndarray):
                np.testing.assert_allclose(graph, again[name], atol=1e-12)
        self.assertIn('n_series', base_meta)

    def test_prior_adds_candidates_without_altering_the_originals(self):
        base, _ = build_candidates(self.LAM)
        primed, _ = build_candidates(self.LAM, prior=self._prior())
        self.assertTrue(set(base).issubset(set(primed)))
        for name, graph in base.items():
            if isinstance(graph, np.ndarray):
                np.testing.assert_allclose(graph, primed[name], atol=1e-12)
        added = set(primed) - set(base)
        self.assertIn('prior_only', added)
        self.assertTrue(any(n.endswith('_prior') for n in added))

    def test_prior_candidates_can_be_switched_off(self):
        base, _ = build_candidates(self.LAM)
        primed, _ = build_candidates(
            self.LAM, {'prior_candidates': False}, prior=self._prior()
        )
        self.assertEqual(sorted(base), sorted(primed))
        zero_weight, _ = build_candidates(
            self.LAM, {'prior_weight': 0.0}, prior=self._prior()
        )
        self.assertEqual(sorted(base), sorted(zero_weight))

    def test_blended_adjacency_stays_signed_symmetric_hollow(self):
        prior = self._prior()
        prior[1, 2] = prior[2, 1] = -1.0  # a substitution link
        adj = laplacian_graph(
            self.LAM, {'prior_weight': 0.8}, n_neighbors=3, prior=prior
        )
        np.testing.assert_allclose(adj, adj.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(adj), 0.0, atol=1e-12)
        self.assertLess(float(adj.min()), 0.0)
        self.assertGreater(float(adj.max()), 0.0)

    def test_blend_moves_the_graph_toward_the_prior(self):
        prior = self._prior()
        plain = laplacian_graph(self.LAM, n_neighbors=1)
        blended = laplacian_graph(
            self.LAM, {'prior_weight': 0.9}, n_neighbors=1, prior=prior
        )
        # s0 and s3 sit on opposite factors, so the unblended graph does not
        # link them; a confident prior does.
        self.assertAlmostEqual(float(plain[0, 3]), 0.0)
        self.assertGreater(float(blended[0, 3]), 0.0)

    def test_defaults_carry_the_new_keys(self):
        self.assertIn('prior_weight', DEFAULT_COHERENCE_CONFIG)
        self.assertIn('prior_candidates', DEFAULT_COHERENCE_CONFIG)

    def test_degenerate_priors_are_ignored(self):
        base, _ = build_candidates(self.LAM)
        for bad in (np.zeros((4, 4)), np.zeros((3, 3)), np.full((4, 4), np.nan)):
            with self.subTest(prior=bad.shape):
                primed, _ = build_candidates(self.LAM, prior=bad)
                self.assertEqual(sorted(base), sorted(primed))
