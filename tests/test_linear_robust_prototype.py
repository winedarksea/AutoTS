"""Compare the research solver to an independent converged Huber fit."""

import unittest

import numpy as np

from examples.linear_robust_prototype import (
    fit_huber_irls_reference,
    fit_huber_mean_shift,
    huber_objective,
)


class TestHuberMeanShift(unittest.TestCase):
    def test_matches_masked_irls_objective_and_extrapolation(self):
        rng = np.random.default_rng(55)
        time = np.linspace(0, 1, 240)
        design = np.column_stack((np.ones(240), time, np.sin(20 * time)))
        targets = design @ rng.normal(size=(3, 5)) + rng.normal(
            scale=0.2, size=(240, 5)
        )
        targets[30:40] += 8
        targets[70:80, 1] = np.nan
        targets[120:150, 2] = np.nan
        coefficients, diagnostics = fit_huber_mean_shift(design, targets, chunk_size=2)
        reference, convergence = fit_huber_irls_reference(
            design, targets, np.asarray(diagnostics["scale"])
        )
        self.assertTrue(all(row["converged"] for row in convergence))
        thresholds = 1.345 * np.asarray(diagnostics["scale"])[None, :]
        expected_objective = huber_objective(design, targets, reference, thresholds)
        actual_objective = huber_objective(design, targets, coefficients, thresholds)
        np.testing.assert_allclose(actual_objective, expected_objective, rtol=1e-8)
        future = np.column_stack(
            (np.ones(20), np.linspace(1, 1.5, 20), np.sin(np.linspace(20, 30, 20)))
        )
        np.testing.assert_allclose(future @ coefficients, future @ reference, atol=1e-5)
        single, _ = fit_huber_mean_shift(design, targets, chunk_size=512)
        np.testing.assert_allclose(single, coefficients, atol=1e-12)

    def test_degenerate_targets_and_invalid_design_are_reported(self):
        design = np.column_stack((np.ones(100), np.arange(100.0) / 100))
        targets = np.column_stack((np.zeros(100), np.full(100, 5.0), np.zeros(100)))
        targets[::10, 2] = 5
        design[20, 1] = np.nan
        coefficients, diagnostics = fit_huber_mean_shift(design, targets)
        self.assertTrue(np.isfinite(coefficients).all())
        self.assertTrue(diagnostics["collapsed_scale"][0])
        self.assertTrue(np.isfinite(diagnostics["objective"]).all())
        targets[:, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "observations"):
            fit_huber_mean_shift(design, targets)
