"""Behavioural tests for LevelShiftMagic bug fixes and the linear_gap detector."""

import unittest

import numpy as np
import pandas as pd

from autots.tools.level_shift import chow_gap, detect_multiscale, PrefixSums
from autots.tools.transform import LevelShiftMagic

T = 730
INDEX = pd.date_range("2020-01-01", periods=T, freq="D")
TIME = np.arange(T)
LINEAR_GAP = dict(window_method="linear_gap", alpha=4.0, scales=[7, 14, 30, 60])


def _frame(*columns, index=INDEX):
    return pd.DataFrame(np.column_stack(columns), index=index[: len(columns[0])])


def _shift_rows(model, df):
    """{column position: [(first post-shift row, magnitude), ...]}"""
    events = model.extract_level_shift_dates()
    return {
        i: [(df.index.get_loc(e["date"]), e["magnitude"]) for e in events[col]]
        for i, col in enumerate(df.columns)
    }


class TestLinearGapDetection(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def noise(self, sd=1.0):
        return self.rng.normal(0, sd, T)

    def test_steady_trend_no_detection(self):
        df = _frame(*[100 + 0.2 * TIME + self.noise(2) for _ in range(5)])
        model = LevelShiftMagic(**LINEAR_GAP).fit(df)
        self.assertEqual(int(model.shift_mask_.to_numpy().sum()), 0)

    def test_step_against_trend_found_and_sized(self):
        df = _frame(100 + 0.2 * TIME + self.noise(2) - 15 * (TIME >= 400))
        model = LevelShiftMagic(**LINEAR_GAP).fit(df)
        shifts = _shift_rows(model, df)[0]
        self.assertEqual(len(shifts), 1)
        row, magnitude = shifts[0]
        self.assertLessEqual(abs(row - 400), 2)
        self.assertLess(abs(magnitude + 15) / 15, 0.1)

    def test_mid_series_spikes_not_detected(self):
        columns = []
        for width in (3, 10):
            y = 100 + self.noise()
            y[300 : 300 + width] += 8
            columns.append(y)
        model = LevelShiftMagic(**LINEAR_GAP).fit(_frame(*columns))
        self.assertEqual(int(model.shift_mask_.to_numpy().sum()), 0)

    def test_slope_change_not_detected(self):
        y = 100 + self.noise() + 0.05 * np.clip(TIME - 400, 0, None)
        model = LevelShiftMagic(**LINEAR_GAP).fit(_frame(y))
        self.assertEqual(int(model.shift_mask_.to_numpy().sum()), 0)

    def test_close_opposite_steps_both_found(self):
        y = 100 + self.noise() + 8 * ((TIME >= 300) & (TIME < 315))
        df = _frame(y)
        shifts = _shift_rows(LevelShiftMagic(**LINEAR_GAP).fit(df), df)[0]
        rows = sorted(r for r, _ in shifts)
        self.assertEqual(len(rows), 2)
        self.assertLessEqual(abs(rows[0] - 300), 2)
        self.assertLessEqual(abs(rows[1] - 315), 2)

    def test_flag_on_last_pre_shift_row(self):
        y = 100 + 0.01 * self.noise()
        y[400:] += 10
        model = LevelShiftMagic(window_method="linear_gap", alpha=4, scales=[30])
        model.fit(_frame(y))
        flagged = np.flatnonzero(model.shift_mask_.to_numpy()[:, 0])
        self.assertEqual(flagged.tolist(), [399])
        # history before the shift is aligned to the latest regime
        corrected = model.transform(_frame(y)).to_numpy()[:, 0]
        self.assertAlmostEqual(corrected[399], corrected[400], delta=0.1)

    def test_chow_gap_peaks_at_first_post_row(self):
        y = np.r_[np.zeros(100), np.ones(100) * 5][:, None]
        gap, norm = chow_gap(PrefixSums(y), 20)
        self.assertEqual(int(np.nanargmax(np.abs(norm[:, 0]))), 100)
        self.assertAlmostEqual(gap[100, 0], 5.0)

    def test_chow_gap_matches_direct_ols(self):
        """The prefix-sum expansion equals a per-window two-intercept OLS fit."""
        T_small, N = 80, 10
        y = 50 + self.rng.normal(0, 1, (T_small, 2)) + 0.3 * np.arange(T_small)[:, None]
        y[40:, 1] += 5
        y[self.rng.random(y.shape) < 0.15] = np.nan
        gap, norm = chow_gap(PrefixSums(y), N)
        for i in range(N, T_small - N):
            for c in range(2):
                tb, tf = np.arange(i - N, i), np.arange(i, i + N)
                tb, yb = tb[~np.isnan(y[tb, c])], y[tb, c][~np.isnan(y[tb, c])]
                tf, yf = tf[~np.isnan(y[tf, c])], y[tf, c][~np.isnan(y[tf, c])]
                if min(len(tb), len(tf)) < 3:
                    continue
                sxx = ((tb - tb.mean()) ** 2).sum() + ((tf - tf.mean()) ** 2).sum()
                slope = (
                    ((tb - tb.mean()) * (yb - yb.mean())).sum()
                    + ((tf - tf.mean()) * (yf - yf.mean())).sum()
                ) / sxx
                expected = yf.mean() - yb.mean() - slope * (tf.mean() - tb.mean())
                self.assertAlmostEqual(gap[i, c], expected, places=8)

    def test_max_level_shifts_zero_keeps_nothing(self):
        y = 100 + self.noise() + 10 * (TIME >= 400)
        rows, *_ = detect_multiscale(y[:, None], [30], 4.0, 0)
        self.assertEqual(rows.size, 0)

    def test_tail_scan_finds_end_shift(self):
        # 10 post points: too few for the bump test, exactly the tail minimum
        y = 100 + self.noise() + 8 * (TIME >= T - 10)
        df = _frame(y)
        without = LevelShiftMagic(**LINEAR_GAP).fit(df)
        self.assertEqual(len(_shift_rows(without, df)[0]), 0)
        with_tail = LevelShiftMagic(tail_scan=True, **LINEAR_GAP).fit(df)
        shifts = _shift_rows(with_tail, df)[0]
        self.assertEqual(len(shifts), 1)
        self.assertLessEqual(abs(shifts[0][0] - (T - 10)), 2)

    def test_tail_scan_respects_max_level_shifts(self):
        y = 100 + self.noise() + 10 * (TIME >= 400) + 8 * (TIME >= T - 10)
        df = _frame(y)
        model = LevelShiftMagic(tail_scan=True, **dict(LINEAR_GAP, max_level_shifts=1))
        self.assertEqual(len(_shift_rows(model.fit(df), df)[0]), 1)
        model = LevelShiftMagic(tail_scan=True, **dict(LINEAR_GAP, max_level_shifts=2))
        self.assertEqual(len(_shift_rows(model.fit(df), df)[0]), 2)

    def test_deseasonalize_with_weekly_pattern(self):
        weekly = 3 * np.sin(2 * np.pi * TIME / 7)
        y = 100 + weekly + self.noise() + 6 * (TIME >= 350)
        df = _frame(y)
        model = LevelShiftMagic(deseasonalize=True, **LINEAR_GAP).fit(df)
        self.assertEqual(model.season_length_, 7)
        shifts = _shift_rows(model, df)[0]
        self.assertEqual(len(shifts), 1)
        self.assertLessEqual(abs(shifts[0][0] - 350), 2)

    def test_univariate_linear_gap_leaves_flat_series(self):
        df = _frame(
            100 + self.noise() + 8 * (TIME >= 400),
            100 + self.noise() - 8 * (TIME >= 400),
            100 + self.noise(),
        )
        model = LevelShiftMagic(output="univariate", **LINEAR_GAP).fit(df)
        lvlshft = model.lvlshft.to_numpy()
        self.assertGreater(np.abs(lvlshft[:, 0]).max(), 5)
        self.assertGreater(np.abs(lvlshft[:, 1]).max(), 5)
        self.assertEqual(np.abs(lvlshft[:, 2]).max(), 0)

    def test_univariate_shared_shift_not_diluted_by_missing_series(self):
        shifted = [100 + self.noise() + 3 * (TIME >= 400) for _ in range(3)]
        missing = [np.full(T, np.nan) for _ in range(30)]
        df = _frame(*shifted, *missing)
        model = LevelShiftMagic(output="univariate", **LINEAR_GAP).fit(df)
        for i in range(3):
            rows = [row for row, _ in _shift_rows(model, df)[i]]
            self.assertTrue(any(abs(row - 400) <= 2 for row in rows), rows)


class TestLegacyBugFixes(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)

    def test_final_point_spike_does_not_leak(self):
        columns = []
        for _ in range(10):
            y = 100 + self.rng.normal(0, 1, T)
            y[-1] += 8
            columns.append(y)
        for method in ("overlap", "exclusive", "diff_overlap"):
            model = LevelShiftMagic(window_method=method).fit(_frame(*columns))
            np.testing.assert_array_equal(model.lvlshft.iloc[-1].to_numpy(), 0.0)
            self.assertFalse(model.shift_mask_.iloc[-2:].to_numpy().any())

    def test_first_point_spike_does_not_suppress_later_shifts(self):
        y = 100 + self.rng.normal(0, 1, T) + 10 * (TIME >= 400)
        y[0] += 30
        df = _frame(y)
        model = LevelShiftMagic(window_size=30, alpha=2.5).fit(df)
        rows = [r for r, _ in _shift_rows(model, df)[0]]
        self.assertTrue(any(abs(r - 400) <= 2 for r in rows), rows)

    def test_joint_fit_equals_separate_fit(self):
        shared = 10 * (TIME >= 300)
        columns = [
            100 + self.rng.normal(0, 1, T) + shared + 6 * (TIME >= 100 + 40 * i)
            for i in range(6)
        ]
        df = _frame(*columns)
        params = dict(window_size=30, alpha=2.5, max_level_shifts=5)
        joint = LevelShiftMagic(**params).fit(df).lvlshft
        for i, col in enumerate(df.columns):
            alone = LevelShiftMagic(**params).fit(df[[col]]).lvlshft
            np.testing.assert_allclose(joint[col].to_numpy(), alone[col].to_numpy())

    def test_max_level_shifts_respected(self):
        y = 100 + self.rng.normal(0, 1, T)
        for k in range(1, 6):
            y += 10 * (TIME >= 120 * k)
        model = LevelShiftMagic(window_size=30, max_level_shifts=3).fit(_frame(y))
        self.assertEqual(int(model.shift_mask_.to_numpy().sum()), 3)
        none = LevelShiftMagic(window_size=30, max_level_shifts=0).fit(_frame(y))
        self.assertEqual(int(none.shift_mask_.to_numpy().sum()), 0)

    def test_integer_data_one_flag_per_step(self):
        y = np.r_[np.zeros(200), np.full(200, 5.0)]
        for method in ("overlap", "exclusive"):
            model = LevelShiftMagic(window_size=30, window_method=method)
            model.fit(_frame(y))
            flagged = np.flatnonzero(model.shift_mask_.to_numpy()[:, 0])
            self.assertEqual(flagged.tolist(), [199], method)
            shifts = model.extract_level_shift_dates()[0]
            self.assertEqual(len(shifts), 1)

    def test_univariate_opposite_shifts_leave_flat_series(self):
        df = _frame(
            100 + self.rng.normal(0, 1, T) + 8 * (TIME >= 363),
            100 + self.rng.normal(0, 1, T) - 8 * (TIME >= 363),
            100 + self.rng.normal(0, 1, T),
        )
        model = LevelShiftMagic(window_size=30, output="univariate").fit(df)
        lvlshft = model.lvlshft.to_numpy()
        # lvlshft carries -size before the break: an up-step gives a negative value
        self.assertLess(lvlshft[0, 0], -5)
        self.assertGreater(lvlshft[0, 1], 5)
        self.assertEqual(np.abs(lvlshft[:, 2]).max(), 0)

    def test_diff_overlap_sizes_clean_step(self):
        y = 100 + self.rng.normal(0, 0.5, T) + 10 * (TIME >= 400)
        df = _frame(y)
        for alignment in ("rolling_diff", "average", "last_value"):
            model = LevelShiftMagic(
                window_method="diff_overlap", window_size=7, alignment=alignment
            ).fit(df)
            shifts = _shift_rows(model, df)[0]
            near = [m for r, m in shifts if abs(r - 400) <= 1]
            self.assertEqual(len(near), 1, (alignment, shifts))
            self.assertLess(abs(near[0] - 10) / 10, 0.1, alignment)

    def test_remove_at_shift_sizes_from_raw_and_fills_only_masked(self):
        y = 100 + self.rng.normal(0, 0.1, T) + 10 * (TIME >= 400)
        y[50] = np.nan
        df = _frame(y)
        params = dict(window_size=30, alignment="last_value", shift_remove_window=2)
        plain = LevelShiftMagic(**params).fit(df.ffill())
        removed = LevelShiftMagic(remove_at_shift=True, **params).fit(df.ffill())
        np.testing.assert_allclose(plain.lvlshft.to_numpy(), removed.lvlshft.to_numpy())
        self.assertGreater(abs(removed.lvlshft.iloc[0, 0]), 9)
        # unrelated NaN gaps are not imputed by the shift fill
        transformed = removed.transform(df)
        self.assertTrue(np.isnan(transformed.iloc[50, 0]))

    def test_auto_correction_never_negative(self):
        growth = 5 * np.exp(0.006 * TIME)
        y = (
            growth
            * np.where(TIME >= 600, 0.2, 1.0)
            * (1 + 0.01 * self.rng.normal(0, 1, T))
        )
        df = _frame(y)
        params = dict(window_method="linear_gap", alpha=4, scales=[30])
        additive = LevelShiftMagic(**params).fit(df)
        self.assertLess(additive.transform(df).to_numpy().min(), 0)
        auto = LevelShiftMagic(correction_mode="auto", **params).fit(df)
        transformed = auto.transform(df)
        self.assertGreaterEqual(transformed.to_numpy().min(), 0)
        np.testing.assert_allclose(auto.inverse_transform(transformed), df)


class TestFeatureDetectorPrelimPass(unittest.TestCase):
    def test_prelim_level_shift_recovers_year_boundary_step(self):
        """Without the prelim pass the day-of-year seasonality fit absorbs this step."""
        import warnings

        from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

        rng = np.random.default_rng(1)
        t = TIME[:, None]
        base = 10 + np.sin(2 * np.pi * t / 7) + 2 * np.sin(2 * np.pi * t / 365.25)
        positions = np.array([354, 362, 372, 374])
        values = base + rng.normal(0, 1, (T, 4)) + (t >= positions) * -5.0
        df = pd.DataFrame(values, index=INDEX, columns=[f"s{i}" for i in range(4)])
        params = dict(
            window_method="linear_gap",
            alpha=3.5,
            scales=[14, 30, 60],
            alignment="linear_robust",
            max_level_shifts=10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detector = TimeSeriesFeatureDetector(
                level_shift_params=params, prelim_level_shift=True
            ).fit(df)
        found = 0
        for col, position in zip(df.columns, positions):
            # entries are (date, magnitude, status, shared) tuples
            rows = [df.index.get_loc(e[0]) for e in detector.level_shifts.get(col, [])]
            found += any(abs(r - position) <= 3 for r in rows)
        self.assertGreaterEqual(found, 3)

    def test_default_detector_recovers_year_boundary_step(self):
        import warnings

        from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

        rng = np.random.default_rng(1)
        t = TIME[:, None]
        base = 10 + np.sin(2 * np.pi * t / 7) + 2 * np.sin(2 * np.pi * t / 365.25)
        positions = np.array([354, 362, 372, 374])
        values = base + rng.normal(0, 1, (T, 4)) + (t >= positions) * -5.0
        df = pd.DataFrame(values, index=INDEX, columns=[f"s{i}" for i in range(4)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detector = TimeSeriesFeatureDetector().fit(df)
        found = 0
        for col, position in zip(df.columns, positions):
            rows = [df.index.get_loc(e[0]) for e in detector.level_shifts.get(col, [])]
            found += any(abs(r - position) <= 3 for r in rows)
        self.assertGreaterEqual(found, 3)


class TestLevelShiftParams(unittest.TestCase):
    def test_new_params_fit(self):
        rng = np.random.default_rng(3)
        df = _frame(
            100 + rng.normal(0, 1, T) + 6 * (TIME >= 400), 50 + rng.normal(0, 1, T)
        )
        for seed in range(40):
            import random

            random.seed(seed)
            params = LevelShiftMagic.get_new_params()
            model = LevelShiftMagic(**params).fit(df)
            transformed = model.transform(df)
            self.assertTrue(np.isfinite(transformed.to_numpy()).all())
            if not params["remove_at_shift"]:
                np.testing.assert_allclose(model.inverse_transform(transformed), df)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            LevelShiftMagic(window_method="nope")
        with self.assertRaises(ValueError):
            LevelShiftMagic(correction_mode="nope")


if __name__ == "__main__":
    unittest.main()
