# -*- coding: utf-8 -*-
"""Tests for the shared TVA metrics module.

Focus is ``loading_structure_score``: the metric that distinguishes "found the
factor span" from "found the true basis within it". Its whole purpose is to be
invariant to the indeterminacies a factor model legitimately has (column order,
column sign) while being sensitive to the one it is supposed to resolve
(which series loads which factor).
"""

import unittest

import numpy as np
import pandas as pd

from autots.evaluator.tva.metrics import loading_structure_score


def simple_structure(n_per_factor=8, n_factors=3, seed=0):
    """(N, K) block loadings: each series loads exactly one factor, mixed signs."""
    rng = np.random.default_rng(seed)
    n = n_per_factor * n_factors
    lam = np.zeros((n, n_factors))
    for i in range(n):
        k = i // n_per_factor
        sign = 1.0 if (i % 3) else -1.0
        lam[i, k] = sign * (0.5 + rng.random())
    return lam


class TestLoadingStructureScore(unittest.TestCase):
    def setUp(self):
        self.lam = simple_structure()

    def test_truth_vs_truth_is_perfect(self):
        out = loading_structure_score(self.lam, self.lam)
        self.assertAlmostEqual(out['matched_loading_corr'], 1.0, places=9)
        self.assertAlmostEqual(out['dominant_recovery'], 1.0, places=9)
        self.assertAlmostEqual(out['sign_agreement'], 1.0, places=9)
        self.assertAlmostEqual(out['pair_precision'], 1.0, places=9)
        self.assertAlmostEqual(out['pair_recall'], 1.0, places=9)
        self.assertAlmostEqual(out['pair_f1'], 1.0, places=9)
        self.assertGreater(out['n_pairs_true'], 0)
        self.assertEqual(out['n_pairs_asserted'], out['n_pairs_true'])

    def test_permuted_and_sign_flipped_truth_is_still_perfect(self):
        # the exact indeterminacy a factor model is allowed to have
        perm = [2, 0, 1]
        est = self.lam[:, perm] * np.array([1.0, -1.0, -1.0])[None, :]
        out = loading_structure_score(self.lam, est)
        self.assertAlmostEqual(out['matched_loading_corr'], 1.0, places=9)
        self.assertAlmostEqual(out['dominant_recovery'], 1.0, places=9)
        self.assertAlmostEqual(out['sign_agreement'], 1.0, places=9)
        self.assertAlmostEqual(out['pair_precision'], 1.0, places=9)
        self.assertAlmostEqual(out['pair_recall'], 1.0, places=9)

    def test_dataframe_inputs_accepted(self):
        names = [f's{i}' for i in range(self.lam.shape[0])]
        frame = pd.DataFrame(self.lam, index=names)
        out = loading_structure_score(frame, frame)
        self.assertAlmostEqual(out['dominant_recovery'], 1.0, places=9)

    def test_random_loadings_score_near_chance(self):
        rng = np.random.default_rng(7)
        est = rng.normal(size=self.lam.shape)
        out = loading_structure_score(self.lam, est)
        # K=3 -> 1/3 dominance, 1/2 sign, by construction
        self.assertLess(out['dominant_recovery'], 0.65)
        self.assertLess(out['sign_agreement'], 0.75)
        self.assertLess(abs(out['matched_loading_corr']), 0.5)
        self.assertLess(out['pair_precision'], 0.6)

    def test_rotation_of_the_true_span_is_detected(self):
        # a 45-degree rotation of two true factors preserves the span exactly
        # but destroys the basis -- the failure this metric exists to catch
        R = np.eye(3)
        c = np.sqrt(0.5)
        R[:2, :2] = [[c, -c], [c, c]]
        out = loading_structure_score(self.lam, self.lam @ R)
        self.assertLess(out['dominant_recovery'], 0.9)
        self.assertLess(out['matched_loading_corr'], 0.95)

    def test_k_mismatch_shapes_handled(self):
        est = np.hstack([self.lam, np.zeros((self.lam.shape[0], 3))])
        out = loading_structure_score(self.lam, est)
        self.assertEqual(out['n_true'], 3)
        self.assertEqual(out['n_est'], 6)
        self.assertAlmostEqual(out['dominant_recovery'], 1.0, places=9)

        fewer = loading_structure_score(self.lam, self.lam[:, :1])
        self.assertEqual(fewer['n_est'], 1)
        # only one factor's series can possibly be recovered
        self.assertLess(fewer['dominant_recovery'], 0.5)

    def test_spurious_dominant_column_counts_as_a_miss(self):
        # a 4th, unmatched column that dominates every row: the K-misspec case
        est = np.hstack([self.lam, np.full((self.lam.shape[0], 1), 10.0)])
        out = loading_structure_score(self.lam, est)
        self.assertAlmostEqual(out['dominant_recovery'], 0.0, places=9)

    def test_asserted_pairs_override_and_drop_negative_links(self):
        truth_pair = (1, 2)          # same factor, same sign
        wrong_pair = (1, 23)         # different factors
        out = loading_structure_score(
            self.lam, self.lam, asserted_pairs=[truth_pair, wrong_pair]
        )
        self.assertEqual(out['n_pairs_asserted'], 2)
        self.assertAlmostEqual(out['pair_precision'], 0.5, places=9)

        signed = loading_structure_score(
            self.lam, self.lam,
            asserted_pairs=[(1, 2, 1.0), (1, 23, -1.0)],
        )
        self.assertEqual(signed['n_pairs_asserted'], 1)
        self.assertAlmostEqual(signed['pair_precision'], 1.0, places=9)

    def test_unexposed_series_excluded_from_truth_pairs(self):
        lam = self.lam.copy()
        lam[:4] = 0.0  # no true exposure -> not a group, not a pair
        out = loading_structure_score(lam, lam)
        self.assertAlmostEqual(out['pair_precision'], 1.0, places=9)
        base = loading_structure_score(self.lam, self.lam)
        self.assertLess(out['n_pairs_true'], base['n_pairs_true'])

    def test_factor_paths_drive_the_matching(self):
        rng = np.random.default_rng(3)
        true_f = np.cumsum(rng.normal(size=(200, 3)), axis=0)
        perm = [1, 2, 0]
        est_f = true_f[:, perm] * np.array([-1.0, 1.0, -1.0])[None, :]
        est_l = self.lam[:, perm] * np.array([-1.0, 1.0, -1.0])[None, :]
        out = loading_structure_score(
            self.lam, est_l, true_factors=true_f, est_factors=est_f
        )
        self.assertAlmostEqual(out['dominant_recovery'], 1.0, places=9)
        self.assertAlmostEqual(out['sign_agreement'], 1.0, places=9)
        self.assertAlmostEqual(out['matched_loading_corr'], 1.0, places=6)

    def test_degenerate_inputs_never_raise(self):
        for bad in (None, np.zeros((0, 0)), np.zeros((5, 0)), np.array(3.0)):
            out = loading_structure_score(bad, self.lam)
            self.assertTrue(np.isnan(out['dominant_recovery']))
            out = loading_structure_score(self.lam, bad)
            self.assertTrue(np.isnan(out['dominant_recovery']))
        nan_lam = self.lam.copy()
        nan_lam[0, 0] = np.nan
        out = loading_structure_score(self.lam, nan_lam)
        self.assertTrue(np.isfinite(out['dominant_recovery']))


if __name__ == '__main__':
    unittest.main()


class TestDominantRecoveryVariants(unittest.TestCase):
    """The two dominance definitions must agree except under K over-specification."""

    def setUp(self):
        self.lam = simple_structure()

    def test_variants_agree_when_k_matches(self):
        for est in (self.lam, self.lam[:, [1, 2, 0]]):
            out = loading_structure_score(self.lam, est)
            self.assertAlmostEqual(
                out['dominant_recovery'], out['dominant_recovery_matched'],
                places=9,
            )

    def test_spurious_column_splits_the_two(self):
        est = np.hstack([self.lam, np.full((self.lam.shape[0], 1), 10.0)])
        out = loading_structure_score(self.lam, est)
        self.assertAlmostEqual(out['dominant_recovery'], 0.0, places=9)
        self.assertAlmostEqual(out['dominant_recovery_matched'], 1.0, places=9)
