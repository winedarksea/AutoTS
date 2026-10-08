# -*- coding: utf-8 -*-
"""Tests for Cassandra's AR helpers and the level/residual AR modes."""
import unittest
import numpy as np
import pandas as pd
from autots.models.cassandra import Cassandra
from autots.models.cassandra_ar import (
    build_lag_array,
    lag_history_tail,
    ar_season_matrix,
    build_ar_design,
    ar_gamma,
    fit_ar_batched,
    stabilize_ar,
    ar_recursive_forecast,
)

FORECAST_LENGTH = 21


def make_panel(n_rows=400, n_series=3, seed=0, ar_coef=0.7):
    """Weekly seasonality + slow random walk + AR(1) noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_rows, freq="D")
    weekly = 3 * np.sin(2 * np.pi * np.arange(n_rows) / 7)[:, None]
    noise = np.zeros((n_rows, n_series))
    shocks = rng.normal(size=(n_rows, n_series))
    for t in range(1, n_rows):
        noise[t] = ar_coef * noise[t - 1] + shocks[t]
    walk = rng.normal(size=(n_rows, n_series)).cumsum(axis=0) * 0.1
    return pd.DataFrame(
        50 + weekly + walk + noise,
        index=idx,
        columns=[f"s{i}" for i in range(n_series)],
    )


def ar_model(**kwargs):
    params = dict(
        seasonalities=[7],
        scaling="BaseScaler",
        regressors_used=False,
        trend_window=None,
        ar_lags=[1, 7],
        n_jobs=1,
        forecast_length=FORECAST_LENGTH,
        linear_model={"model": "lstsq", "lambda": 1, "recency_weighting": None},
    )
    params.update(kwargs)
    return Cassandra(**params)


class ARHelperTest(unittest.TestCase):
    def test_lag_array_shifts_by_date_on_gapped_index(self):
        idx = pd.date_range("2022-01-01", periods=10, freq="D")
        df = pd.DataFrame({"a": np.arange(10.0)}, index=idx).drop(idx[[4, 5]])
        lagged = build_lag_array(df, [1, 2], frequency="D")
        self.assertEqual(lagged.shape, (8, 1, 2))
        lag1 = pd.Series(lagged[:, 0, 0], index=df.index)
        # the day after a dropped row has no lag-1 value, rather than the row two days back
        self.assertTrue(np.isnan(lag1[idx[6]]))
        self.assertEqual(lag1[idx[7]], 6.0)
        self.assertTrue(np.isnan(lagged[0, 0, 0]))
        lag2 = pd.Series(lagged[:, 0, 1], index=df.index)
        self.assertTrue(np.isnan(lag2[idx[6]]) and np.isnan(lag2[idx[7]]))
        self.assertEqual(lag2[idx[8]], 6.0)

    def test_history_tail_on_date_grid(self):
        idx = pd.date_range("2022-01-01", periods=6, freq="D")
        df = pd.DataFrame({"a": np.arange(6.0)}, index=idx).drop(idx[4])
        tail = lag_history_tail(df, 3, frequency="D", fill_value=0.0)
        np.testing.assert_array_equal(tail[:, 0], [3.0, 0.0, 5.0])

    def test_batched_ridge_matches_lstsq(self):
        rng = np.random.default_rng(1)
        T, N, K = 200, 4, 3
        design = rng.normal(size=(T, N, K))
        target = np.einsum("tnk,nk->tn", design, rng.normal(size=(N, K)))
        target += rng.normal(scale=0.1, size=(T, N))
        design[:5, 2, 0] = np.nan  # masked rows only affect series 2
        coef, std = fit_ar_batched(design, target, lam=1e-10)
        for n in range(N):
            valid = np.all(np.isfinite(design[:, n]), axis=1)
            expected = np.linalg.lstsq(design[valid, n], target[valid, n], rcond=None)[0]
            np.testing.assert_allclose(coef[n], expected, rtol=1e-6, atol=1e-8)
        self.assertTrue(np.all((std > 0.05) & (std < 0.2)))

    def test_recursion_matches_naive_loop(self):
        rng = np.random.default_rng(2)
        H, N, lags = 12, 3, [1, 3]
        season = pd.DataFrame(rng.normal(size=(H, 2)), columns=["a", "b"])
        season_matrix, _, _ = ar_season_matrix(season)
        coef = rng.normal(scale=0.2, size=(N, len(lags), season_matrix.shape[1]))
        hist = rng.normal(size=(3, N))
        base = rng.normal(size=(H, N))
        path, contrib, _ = ar_recursive_forecast(hist, base, coef, season_matrix, lags)
        z = list(hist)
        for h in range(H):
            gamma = coef @ season_matrix[h]  # (N, L)
            ar = sum(gamma[:, i] * z[len(z) - lag] for i, lag in enumerate(lags))
            np.testing.assert_allclose(contrib[h], ar)
            z.append(base[h] + ar)
        np.testing.assert_allclose(path, np.array(z[3:]))

    def test_recursion_without_loop_when_lags_exceed_horizon(self):
        rng = np.random.default_rng(3)
        coef = rng.normal(size=(2, 1, 1))
        hist = rng.normal(size=(7, 2))
        base = rng.normal(size=(5, 2))
        path, _, _ = ar_recursive_forecast(hist, base, coef, None, [7])
        expected = base + coef[None, :, 0, 0] * hist[:5]
        np.testing.assert_allclose(path, expected)

    def test_stabilization_guarantees_decay(self):
        rng = np.random.default_rng(4)
        T = 70
        season = pd.DataFrame(
            np.sin(2 * np.pi * np.arange(T) / 7)[:, None], columns=["s"]
        )
        season_matrix, _, _ = ar_season_matrix(season)
        coef = np.array([[[1.1, 0.4]], [[0.6, 0.6]], [[0.3, 0.0]]])  # explosive
        stable = stabilize_ar(coef, season_matrix, T)
        gamma = ar_gamma(stable, season_matrix, T)
        self.assertLessEqual(np.abs(gamma).sum(axis=2).max(), 0.98 + 1e-12)
        # already-stable series are untouched
        np.testing.assert_allclose(stable[2], coef[2])
        hist = rng.normal(size=(1, 3)) * 10
        path, _, psi = ar_recursive_forecast(
            hist, np.zeros((T, 3)), stable, season_matrix, [1]
        )
        self.assertLess(np.abs(path[-1]).max(), np.abs(hist).max() * 0.98 ** (T - 1) + 1e-9)
        self.assertTrue(np.all(np.abs(psi[-1]) < 0.5))

    def test_interaction_design_keeps_plain_lag(self):
        T = 30
        lag_array = np.arange(T, dtype=float).reshape(T, 1, 1)
        season = pd.DataFrame(
            {
                "const": np.ones(T),
                "zero": np.zeros(T),
                "wave": np.sin(np.arange(T)),
            }
        )
        season_matrix, keep, names = ar_season_matrix(season)
        self.assertEqual(names, ["wave"])
        design = build_ar_design(lag_array, season_matrix)
        self.assertEqual(design.shape, (T, 1, 2))
        np.testing.assert_allclose(design[:, 0, 0], lag_array[:, 0, 0])
        np.testing.assert_allclose(design[:, 0, 1], lag_array[:, 0, 0] * season["wave"])


class CassandraARModeTest(unittest.TestCase):
    def test_residual_mode_forecast_decays_to_trend(self):
        df = make_panel()
        model = ar_model(ar_target="residual", ar_lags=[1, 2]).fit(df)
        self.assertFalse(model.loop_required)
        horizon = 200
        model.predict(horizon)
        ar_future = model.predicted_ar.to_numpy()
        self.assertGreater(np.abs(ar_future[0]).max(), 1e-3)
        self.assertLess(np.abs(ar_future[-1]).max(), 1e-3)

    def test_residual_mode_intervals_widen_then_level_off(self):
        df = make_panel()
        model = ar_model(ar_target="residual", ar_lags=[1]).fit(df)
        pred = model.predict(60)
        width = (pred.upper_forecast - pred.lower_forecast).mean(axis=1).to_numpy()
        self.assertGreater(width[10], width[2])
        self.assertLess(abs(width[-1] - width[-2]), 1e-3 * width[-1] + 1e-9)

    def test_components_sum_to_forecast(self):
        df = make_panel()
        for target in ["level", "residual"]:
            for include_history in [False, True]:
                model = ar_model(ar_target=target, ar_interaction_seasonality=7)
                model.fit(df)
                pred = model.predict(FORECAST_LENGTH, include_history=include_history)
                comps = model.return_components(to_origin_space=False)
                total = comps.T.groupby(level=0).sum().T[df.columns]
                origin = total * model.scaler_std + model.scaler_mean
                np.testing.assert_allclose(
                    origin.to_numpy(), pred.forecast.to_numpy(), rtol=1e-8, atol=1e-6
                )
                self.assertIn("lags", comps.columns.get_level_values(1))

    def test_level_mode_with_gapped_index_and_x_scaler(self):
        df = make_panel().drop(make_panel().index[[100, 101, 250]])
        model = ar_model(ar_target="level", x_scaler=True).fit(df)
        pred = model.predict(FORECAST_LENGTH)
        self.assertEqual(pred.forecast.shape, (FORECAST_LENGTH, 3))
        self.assertFalse(pred.forecast.isna().any().any())
        # lags stay out of the scaler: the lag-1 column holds yesterday's scaled y as-is
        x_s0 = model.x_array["s0"]
        day = df.index[50]
        self.assertAlmostEqual(
            x_s0.loc[day, "lag1_"], model.df.loc[df.index[49], "s0"], places=10
        )

    def test_new_df_refreshes_ar_state(self):
        df = make_panel()
        for target in ["level", "residual"]:
            model = ar_model(ar_target=target).fit(df)
            original = model.predict(FORECAST_LENGTH).forecast
            refit = model.predict(FORECAST_LENGTH, new_df=df).forecast
            np.testing.assert_allclose(refit.to_numpy(), original.to_numpy(), rtol=1e-6)

    def test_multivariate_stepping_with_ar(self):
        df = make_panel(n_series=4)
        groups = {"s0": "g1", "s1": "g1", "s2": "g2", "s3": "g2"}
        for target in ["level", "residual"]:
            model = ar_model(ar_target=target, multivariate_feature="group_average")
            model.fit(df, categorical_groups=groups)
            pred = model.predict(FORECAST_LENGTH)
            self.assertEqual(pred.forecast.shape, (FORECAST_LENGTH, 4))
            self.assertFalse(pred.forecast.isna().any().any())
            multivar = model.predict_x_array
            if isinstance(multivar, dict):
                multivar = multivar["s0"]
            mv_cols = [c for c in multivar.columns if c.startswith("multivar_")]
            self.assertFalse(multivar[mv_cols].isna().any().any())

    def test_get_new_params_ar_weights(self):
        params = Cassandra().get_new_params()
        self.assertIn("ar_target", params)
        model = Cassandra(**params)
        self.assertIn(model.get_params()["ar_target"], ("level", "residual"))


if __name__ == "__main__":
    unittest.main()
