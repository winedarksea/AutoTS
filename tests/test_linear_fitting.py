"""Numerical and observation-mask contracts for the linear forecast models."""

import random
import unittest
from statistics import NormalDist
from unittest.mock import patch

import numpy as np
import pandas as pd

from autots.models.basics import BasicLinearModel
from autots.models.cassandra import fit_linear_model, lstsq_solve
from autots.tools.linear import (
    linear_fit_covariance,
    observation_mask_groups,
    solve_linear_coefficients,
    solve_huber_coefficients,
)


class TestLinearCoefficients(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.design = np.column_stack((rng.normal(size=(90, 4)), np.ones(90)))
        self.targets = rng.normal(size=(90, 3))

    def test_objective_matches_augmented_least_squares(self):
        for collinear in (False, True):
            design = self.design.copy()
            if collinear:
                design[:, 1] = design[:, 0]
            for penalty in (None, 0, 2, np.array([2, 2, 2, 2, 0])):
                with self.subTest(collinear=collinear, penalty=penalty):
                    diagonal = np.broadcast_to(0 if penalty is None else penalty, (5,))
                    augmented = np.vstack((design, np.diag(np.sqrt(diagonal))))
                    expected = np.linalg.lstsq(
                        augmented,
                        np.vstack((self.targets, np.zeros((5, 3)))),
                        rcond=None,
                    )[0]
                    actual = solve_linear_coefficients(design, self.targets, penalty)
                    np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_failed_solve_uses_original_augmented_objective(self):
        expected = solve_linear_coefficients(self.design, self.targets, 2)
        with patch("numpy.linalg.solve", side_effect=np.linalg.LinAlgError):
            actual = solve_linear_coefficients(self.design, self.targets, 2)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_huber_fit_resists_outliers_and_handles_missing_targets(self):
        x = np.linspace(-1, 1, 101)
        design = np.column_stack((x, np.ones_like(x)))
        targets = np.column_stack((3 + 2 * x, 4 - x))
        targets[50, 0] += 100
        targets[::7, 1] = np.nan
        ordinary = np.column_stack(
            [
                solve_linear_coefficients(
                    design[np.isfinite(targets[:, column])],
                    targets[np.isfinite(targets[:, column]), column],
                    0,
                )
                for column in range(targets.shape[1])
            ]
        )
        robust = solve_huber_coefficients(design, targets, penalty=0, iterations=50)
        self.assertLess(abs(robust[1, 0] - 3), abs(ordinary[1, 0] - 3))
        self.assertLess(abs(robust[0, 1] + 1), 1e-6)
        self.assertTrue(np.isfinite(robust).all())

    def test_huber_wide_fit_matches_independent_series_with_distinct_masks(self):
        rng = np.random.default_rng(43)
        targets = rng.normal(size=(len(self.design), 600))
        targets[30, :] += 40
        targets[::7, 512:] = np.nan
        design = self.design.copy()
        design[0, 0] = np.nan
        actual = solve_huber_coefficients(design, targets, penalty=2, iterations=8)
        for column in (0, 511, 512, 599):
            with self.subTest(column=column):
                expected = solve_huber_coefficients(
                    design, targets[:, column], penalty=2, iterations=8
                )
                np.testing.assert_allclose(actual[:, column], expected, atol=1e-12)

    def test_basic_linear_huber_selection_is_rare_and_excluded_from_fast(self):
        original_choices = random.choices

        def choose_huber(population, weights=None, **kwargs):
            if population == ["lstsq", "huber"]:
                self.assertEqual(weights, [0.98, 0.02])
                return ["huber"]
            return original_choices(population, weights=weights, **kwargs)

        with patch("autots.models.basics.random.choices", side_effect=choose_huber):
            model = BasicLinearModel(holiday_countries_used=False)
            random_params = model.get_new_params(method="random")
            fast_params = model.get_new_params(method="fast")
        self.assertEqual(random_params["linear_model"], "huber")
        self.assertEqual(random_params["huber_iterations"], 50)
        self.assertEqual(fast_params["linear_model"], "lstsq")

    def test_single_target_and_singular_partially_penalized_design(self):
        design = np.column_stack((self.design, np.zeros(90)))
        penalty = np.array([2, 2, 2, 2, 0, 0])
        actual = solve_linear_coefficients(design, self.targets[:, 0], penalty)
        self.assertEqual(actual.shape, (6,))
        self.assertEqual(actual[-1], 0)
        self.assertTrue(np.isfinite(actual).all())

    def test_covariance_and_degrees_of_freedom_match_hat_matrix(self):
        for penalty in (None, 0, 2):
            for design in (self.design, self.design[:3]):
                with self.subTest(penalty=penalty, rows=len(design)):
                    diagonal = np.full(5, penalty or 0)
                    augmented = np.vstack((design, np.diag(np.sqrt(diagonal))))
                    operator = np.linalg.pinv(augmented)[:, : len(design)]
                    covariance, dof = linear_fit_covariance(design, penalty)
                    hat = design @ operator
                    expected_dof = max(
                        len(design) - 2 * np.trace(hat) + np.sum(hat**2), 1
                    )
                    np.testing.assert_allclose(
                        covariance, operator @ operator.T, atol=1e-12
                    )
                    self.assertAlmostEqual(dof, expected_dof)

    def test_cassandra_weight_and_intercept_conventions(self):
        for model in ("lstsq", "linalg_solve"):
            weights = np.arange(1, 91) ** 0.25
            expected = solve_linear_coefficients(
                self.design * weights[:, None],
                self.targets * weights[:, None],
                np.array([2, 2, 2, 2, 0]),
            )
            with patch(
                "numpy.linalg.cond", side_effect=AssertionError("unneeded SVD")
            ) as condition_check:
                actual = fit_linear_model(
                    self.design,
                    self.targets,
                    {"model": model, "lambda": 2, "recency_weighting": 0.25},
                )
                condition_check.assert_not_called()
            np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(
            lstsq_solve(self.design, self.targets, lamb=None),
            solve_linear_coefficients(self.design, self.targets, [1, 1, 1, 1, 0]),
        )

    def test_cassandra_automatic_ridge_guard_is_preserved(self):
        design = self.design.copy()
        design[:, 1] = design[:, 0]
        for model in ("lstsq", "linalg_solve"):
            actual = fit_linear_model(design, self.targets, {"model": model})
            expected = solve_linear_coefficients(design, self.targets, [1, 1, 1, 1, 0])
            np.testing.assert_allclose(actual, expected)

    def test_cassandra_other_losses_skip_condition_check(self):
        expected = np.ones((5, 3))
        for loss in ("l1_norm", "l1_positive", "quantile_norm", "dwae_norm"):
            with (
                patch("numpy.linalg.cond") as condition_check,
                patch("autots.models.cassandra.lstsq_minimize", return_value=expected),
            ):
                actual = fit_linear_model(self.design, self.targets, {"model": loss})
                self.assertIs(actual, expected)
                condition_check.assert_not_called()


class TestBasicLinearMissingData(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.data = pd.DataFrame(
            20 + rng.normal(size=(400, 4)),
            index=pd.date_range("2022-01-01", periods=400),
            columns=["complete", "gappy", "same_mask", "other_mask"],
        )
        self.data.iloc[20:55, 1:3] = np.nan
        self.data.iloc[80, 3] = np.inf

    def model(self, **kwargs):
        return BasicLinearModel(holiday_countries_used=False, **kwargs)

    def test_missing_groups_match_separate_masked_fits(self):
        original = self.data.copy(deep=True)
        for penalty in (None, 0, 2):
            model = self.model(lambda_=penalty).fit(self.data)
            design = model.create_x(self.data, holiday_countries_used=False).to_numpy()
            self.assertEqual(
                len(list(observation_mask_groups(design, self.data.to_numpy()))), 3
            )
            for j, name in enumerate(self.data.columns):
                rows = np.isfinite(self.data[name])
                expected = solve_linear_coefficients(
                    design[rows], self.data.loc[rows, name], penalty
                )
                np.testing.assert_allclose(model.beta[:, j], expected, atol=1e-9)
                covariance, dof = linear_fit_covariance(design[rows], penalty)
                expected_sigma = np.sqrt(
                    np.sum((self.data.loc[rows, name] - design[rows] @ expected) ** 2)
                    / dof
                )
                self.assertAlmostEqual(model.sigma[j], expected_sigma)
            prediction = model.predict(14)
            self.assertTrue(np.isfinite(prediction.forecast).all().all())
            self.assertTrue(np.isfinite(prediction.upper_forecast).all().all())
            self.assertTrue(
                (prediction.lower_forecast <= prediction.forecast).all().all()
            )

            self.assertTrue(
                (prediction.upper_forecast >= prediction.forecast).all().all()
            )
            components = prediction.components.T.groupby(level=0).sum().T
            np.testing.assert_allclose(
                components[self.data.columns], prediction.forecast, atol=1e-10
            )
        pd.testing.assert_frame_equal(self.data, original)

    def test_huber_mode_fits_and_round_trips_model_parameters(self):
        model = self.model(linear_model="huber", huber_iterations=8).fit(self.data)
        prediction = model.predict(7)
        self.assertEqual(model.get_params()["linear_model"], "huber")
        self.assertEqual(model.get_params()["huber_iterations"], 8)
        self.assertTrue(np.isfinite(prediction.forecast.to_numpy()).all())

    def test_invalid_training_design_rows_are_excluded_before_products(self):
        regressor = pd.DataFrame({"regressor": np.arange(400.0)}, index=self.data.index)
        regressor.iloc[15:18] = np.nan
        model = self.model(regression_type="User").fit(self.data, regressor)
        self.assertTrue(np.isfinite(model.beta).all())
        future = pd.DataFrame(
            {"regressor": np.arange(400.0, 407.0)},
            index=pd.date_range("2023-02-05", periods=7),
        )
        future.index = model.create_forecast_index(7)
        self.assertTrue(np.isfinite(model.predict(7, future).forecast).all().all())
        future.iloc[0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model.predict(7, future)
        with self.assertRaisesRegex(ValueError, "no future_regressor"):
            model.predict(7)

    def test_series_without_observations_does_not_affect_others(self):
        expected = self.model().fit(self.data).predict(7)
        data = self.data.assign(empty=np.nan)
        actual = self.model().fit(data).predict(7)
        for name in ("forecast", "upper_forecast"):
            result = getattr(actual, name)
            pd.testing.assert_frame_equal(
                result[self.data.columns], getattr(expected, name)
            )
            self.assertTrue(result["empty"].isna().all())

    def test_nullable_float_targets_preserve_missingness(self):
        expected = self.model().fit(self.data).predict(7).forecast
        actual = self.model().fit(self.data.astype("Float64")).predict(7).forecast
        pd.testing.assert_frame_equal(actual, expected)

    def test_forecast_regressor_order_matches_training(self):
        rng = np.random.default_rng(77)
        regressor = pd.DataFrame(
            rng.normal(size=(414, 2)),
            columns=["a", "b"],
            index=pd.date_range(self.data.index[0], periods=414),
        )
        model = self.model(regression_type="User").fit(self.data, regressor)
        expected = model.predict(14, regressor).forecast
        actual = model.predict(14, regressor[["b", "a"]]).forecast
        pd.testing.assert_frame_equal(actual, expected)

    def test_short_histories_have_finite_intervals(self):
        for rows in (1, 2, 30):
            for penalty in (None, 0, 2):
                model = self.model(frequency="D", lambda_=penalty).fit(
                    self.data.iloc[:rows, :1]
                )
                prediction = model.predict(7)
                self.assertTrue(np.isfinite(prediction.upper_forecast).all().all())
                self.assertTrue(np.isfinite(model.sigma).all())

    def test_interval_prefix_is_independent_of_requested_horizon(self):
        for damping in (None, 0.98):
            for penalty in (None, 2):
                model = self.model(trend_phi=damping, lambda_=penalty).fit(self.data)
                short, long = model.predict(7), model.predict(30)
                for field in ("forecast", "upper_forecast", "lower_forecast"):
                    np.testing.assert_allclose(
                        getattr(short, field), getattr(long, field).iloc[:7], atol=1e-10
                    )

    def test_interval_uses_training_covariance_and_damped_design(self):
        model = self.model(trend_phi=0.8).fit(self.data[["complete"]])
        prediction = model.predict(9)
        effective = model.X.copy()
        trend = effective[model.trend_columns].to_numpy()
        changes = np.diff(trend, axis=0) * 0.8 ** np.arange(8)[:, None]
        effective[model.trend_columns] = np.cumsum(
            np.vstack((trend[:1], changes)), axis=0
        )
        covariance, _ = linear_fit_covariance(model._linear_training_design, 2)
        leverage = np.einsum("ij,jk,ik->i", effective, covariance, effective)
        expected = NormalDist().inv_cdf(0.95) * model.sigma[0] * np.sqrt(1 + leverage)
        np.testing.assert_allclose(
            (prediction.upper_forecast - prediction.forecast).iloc[:, 0], expected
        )
        second = self.model().fit(self.data[["complete"]].iloc[-100:])
        width = second.predict(9).upper_forecast - second.predict(9).forecast
        self.assertFalse(np.allclose(width.iloc[:, 0], expected))
