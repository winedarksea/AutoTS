# -*- coding: utf-8 -*-
"""Tests for the robust trend-isolation input estimator (``robust_input``).

Thresholds are calibrated empirically on the fixed seeds used here and set
below the observed values, following the convention of the rest of the TVA
suite. The module is numpy-only, so nothing here is skipped.
"""

import unittest

import numpy as np
import pandas as pd

from autots.evaluator.tva.robust_input import (
    DEFAULT_ROBUST_INPUT_CONFIG,
    compare_inputs,
    robust_adjusted_panel,
)


def piecewise_path(rng, n_time, n_breaks=4):
    """Unit-std piecewise-linear path — the estimator's own trend prior."""
    breaks = np.sort(rng.choice(np.arange(1, n_time), n_breaks, replace=False))
    slopes = rng.normal(0, 1, n_breaks + 1)
    deltas = np.zeros(n_time)
    seg = np.searchsorted(breaks, np.arange(n_time))
    deltas[1:] = slopes[seg[1:]]
    path = np.cumsum(deltas)
    path = path - path.mean()
    return path / max(path.std(), 1e-8)


def make_contaminated_panel(
    n_series=12,
    n_time=600,
    n_factors=2,
    noise=0.2,
    idio_scale=0.3,
    n_spikes=3,
    spike_size=6.0,
    seed=0,
):
    """``low_rank_trend + level_shift + sparse_spikes + noise``.

    Returns the observed panel together with every injected piece, so a test
    can ask not only "is the answer better" but "did it recover the thing we
    put there".
    """
    rng = np.random.default_rng(seed)
    factors = np.column_stack(
        [piecewise_path(rng, n_time) for _ in range(n_factors)]
    )
    loadings = rng.normal(0, 1, (n_series, n_factors))
    idio = idio_scale * np.column_stack(
        [piecewise_path(rng, n_time) for _ in range(n_series)]
    )
    trend = 100.0 + factors @ loadings.T + idio

    shifts = np.zeros((n_time, n_series))
    shift_spec = {0: (int(n_time * 0.6), 4.0)}
    if n_series > 5:
        shift_spec[5] = (int(n_time * 0.4), -3.5)
    for col, (start, size) in shift_spec.items():
        shifts[start:, col] = size

    spikes = np.zeros((n_time, n_series))
    for j in range(n_series):
        idx = rng.choice(n_time, n_spikes, replace=False)
        spikes[idx, j] = rng.choice([-1.0, 1.0], n_spikes) * spike_size

    values = trend + shifts + spikes + rng.normal(0, noise, (n_time, n_series))
    return {
        'values': values,
        'trend': trend,
        'shifts': shifts,
        'shift_spec': shift_spec,
        'spikes': spikes,
        'factors': factors,
        'loadings': loadings,
    }


def mean_abs_corr(candidate, truth):
    """Mean |corr| of each column of ``candidate`` with ``truth``."""
    out = []
    for j in range(truth.shape[1]):
        c, t = candidate[:, j], truth[:, j]
        if np.std(c) > 1e-12 and np.std(t) > 1e-12:
            out.append(abs(float(np.corrcoef(c, t)[0, 1])))
    return float(np.mean(out)) if out else float('nan')


class TestRobustAdjustedPanel(unittest.TestCase):
    """The estimator on a panel whose contamination we injected ourselves."""

    @classmethod
    def setUpClass(cls):
        cls.panel = make_contaminated_panel(seed=0)
        cls.result = robust_adjusted_panel(cls.panel['values'])

    def test_returns_full_contract(self):
        for key in (
            'adjusted', 'shared', 'idio', 'anomalies', 'shifts', 'scale',
            'center', 'n_iters', 'converged', 'diagnostics',
        ):
            self.assertIn(key, self.result)
        shape = self.panel['values'].shape
        for key in ('adjusted', 'shared', 'idio', 'anomalies', 'shifts'):
            self.assertEqual(self.result[key].shape, shape, key)
            self.assertTrue(np.isfinite(self.result[key]).all(), key)
        # shared + idio is exactly the returned trend panel
        np.testing.assert_allclose(
            self.result['shared'] + self.result['idio'],
            self.result['adjusted'],
            rtol=1e-8, atol=1e-8,
        )

    def test_adjusted_beats_raw_on_trend_correlation(self):
        truth = self.panel['trend']
        raw_corr = mean_abs_corr(self.panel['values'], truth)
        adj_corr = mean_abs_corr(self.result['adjusted'], truth)
        self.assertGreater(adj_corr, raw_corr + 0.15)
        self.assertGreater(adj_corr, 0.95)

    def test_residual_non_trend_energy_is_reduced(self):
        """Per series, most of the non-trend energy is gone."""
        truth = self.panel['trend']
        adjusted = self.result['adjusted']
        raw = self.panel['values']
        kept = []
        for j in range(truth.shape[1]):
            o = truth[:, j] - truth[:, j].mean()
            a = adjusted[:, j] - adjusted[:, j].mean()
            r = raw[:, j] - raw[:, j].mean()
            kept.append(np.sum((a - o) ** 2) / np.sum((r - o) ** 2))
        self.assertLess(float(np.mean(kept)), 0.2)
        self.assertLess(float(np.max(kept)), 1.0)

    def test_recovered_anomalies_track_the_injected_spikes(self):
        spikes = self.panel['spikes']
        got = self.result['anomalies']
        corr = float(np.corrcoef(got.ravel(), spikes.ravel())[0, 1])
        self.assertGreater(corr, 0.95)
        # and they are sparse: nowhere near a dense residual
        self.assertLess(self.result['diagnostics']['anomaly_fraction'], 0.1)

    def test_recovered_shifts_land_near_the_injected_ones(self):
        shifts = self.result['shifts']
        for col, (start, size) in self.panel['shift_spec'].items():
            before = float(np.median(shifts[max(start - 120, 0):start - 20, col]))
            after = float(np.median(shifts[start + 20:start + 120, col]))
            jump = after - before
            self.assertGreater(
                jump * np.sign(size), 0.5 * abs(size),
                msg=f'column {col} recovered jump {jump:.2f} vs injected {size}',
            )
        # clean columns get no shift term at all
        clean = [j for j in range(shifts.shape[1])
                 if j not in self.panel['shift_spec']]
        n_flagged = sum(1 for j in clean if np.ptp(shifts[:, j]) > 1e-9)
        self.assertEqual(n_flagged, 0)

    def test_shift_times_are_located_exactly(self):
        """The breakpoint the fit kept is the one we injected."""
        times = self.result['diagnostics']['shift_times']
        self.assertEqual(
            set(times), set(self.panel['shift_spec']),
            msg=f'flagged {times}, injected {self.panel["shift_spec"]}',
        )
        for col, (start, _size) in self.panel['shift_spec'].items():
            found = times[col]
            self.assertEqual(len(found), 1)
            self.assertLessEqual(abs(found[0] - start), 5)

    def test_level_shifts_do_not_leak_into_the_shared_part(self):
        """A step in one series must not become a shared 'regime'."""
        n_time, n_series = 600, 10
        rng = np.random.default_rng(3)
        factor = piecewise_path(rng, n_time)
        loadings = rng.uniform(0.7, 1.3, n_series)
        trend = 50.0 + np.outer(factor, loadings)
        base = trend + rng.normal(0, 0.1, (n_time, n_series))
        step = np.zeros(n_time)
        step[300:] = 8.0
        contaminated = base.copy()
        contaminated[:, 0] += step

        clean = robust_adjusted_panel(base)
        dirty = robust_adjusted_panel(contaminated)

        # the step is charged to the shift term, not to the shared factor
        step_in_shared = abs(
            float(np.mean(dirty['shared'][320:, 0]))
            - float(np.mean(dirty['shared'][:280, 0]))
        )
        step_in_shifts = abs(
            float(np.mean(dirty['shifts'][320:, 0]))
            - float(np.mean(dirty['shifts'][:280, 0]))
        )
        self.assertGreater(step_in_shifts, step_in_shared)
        # and the *other* series' shared trends barely notice
        for j in range(1, n_series):
            corr = abs(float(np.corrcoef(
                dirty['shared'][:, j], clean['shared'][:, j]
            )[0, 1]))
            self.assertGreater(corr, 0.95, msg=f'series {j} shared moved')

    def test_missing_mask_is_honoured(self):
        """A filled run must not be treated as evidence of a flat trend."""
        n_time = 600
        rng = np.random.default_rng(7)
        ramp = np.linspace(0.0, 40.0, n_time)
        truth = np.column_stack([100.0 + ramp for _ in range(4)])
        values = truth + rng.normal(0, 0.1, (n_time, 4))
        gap = slice(250, 400)
        holed = values.copy()
        holed[gap, 0] = np.nan
        mask = np.isfinite(holed)

        masked = robust_adjusted_panel(holed, mask=mask)
        # the ffill behaviour of _build_adjusted_panel, reproduced exactly
        ffilled = pd.DataFrame(holed).ffill().bfill().to_numpy()
        naive = robust_adjusted_panel(ffilled, mask=np.ones_like(mask, dtype=bool))

        err_masked = float(np.mean(np.abs(masked['adjusted'][gap, 0] - truth[gap, 0])))
        err_naive = float(np.mean(np.abs(naive['adjusted'][gap, 0] - truth[gap, 0])))
        self.assertLess(err_masked, err_naive)
        # the mask does not damage the columns that were never missing
        self.assertGreater(
            mean_abs_corr(masked['adjusted'][:, 1:], truth[:, 1:]), 0.99
        )

    def test_robust_scaling_isolates_a_huge_column(self):
        """Multiplying one column by 1e6 must not move the others."""
        panel = make_contaminated_panel(seed=1, n_series=8)
        values = panel['values']
        blown = values.copy()
        blown[:, 3] *= 1e6

        base = robust_adjusted_panel(values)
        scaled = robust_adjusted_panel(blown)
        for j in range(values.shape[1]):
            if j == 3:
                continue
            a, b = base['adjusted'][:, j], scaled['adjusted'][:, j]
            denom = max(float(np.std(a)), 1e-9)
            self.assertLess(
                float(np.max(np.abs(a - b))) / denom, 0.25,
                msg=f'column {j} moved when column 3 was rescaled',
            )
        # and the blown-up column comes back on its own scale
        np.testing.assert_allclose(
            scaled['adjusted'][:, 3] / 1e6, base['adjusted'][:, 3],
            rtol=1e-6, atol=1e-6 * float(np.std(base['adjusted'][:, 3])),
        )


class TestShrinkage(unittest.TestCase):
    """The nuisance-shrinkage knob that replaces the high-pass invariant."""

    def setUp(self):
        n_time, n_series = 500, 6
        rng = np.random.default_rng(11)
        t = np.arange(n_time)
        self.trend = 100.0 + np.outer(t / n_time, np.linspace(5, 20, n_series))
        # a "seasonality" estimate that has quietly absorbed a linear drift
        season = np.outer(np.sin(2 * np.pi * t / 91.0), np.ones(n_series)) * 3.0
        self.drift = np.outer(t / n_time, np.full(n_series, 10.0))
        self.season_estimate = season + self.drift
        self.values = self.trend + season + rng.normal(0, 0.05, (n_time, n_series))
        self.components = {'seasonality': self.season_estimate}

    def test_shrink_zero_versus_one_moves_in_the_expected_direction(self):
        full = robust_adjusted_panel(
            self.values, components=self.components, shrink={'seasonality': 1.0}
        )['adjusted']
        none = robust_adjusted_panel(
            self.values, components=self.components, shrink={'seasonality': 0.0}
        )['adjusted']
        self.assertGreater(float(np.max(np.abs(full - none))), 1.0)
        # subtracting the estimate whole also removes the drift it absorbed,
        # so the fully-shrunk panel sits *below* the trend by that drift
        drift_removed = float(np.mean((none - full) - self.drift))
        self.assertLess(abs(drift_removed), 1.0)
        # ignoring it entirely is closer to the true trend here, which is the
        # whole reason the coefficient is exposed rather than hard-coded to 1
        err_full = float(np.mean(np.abs(full - self.trend)))
        err_none = float(np.mean(np.abs(none - self.trend)))
        self.assertLess(err_none, err_full)

    def test_partial_shrink_lands_between_the_extremes(self):
        outs = {
            s: robust_adjusted_panel(
                self.values, components=self.components,
                shrink={'seasonality': s},
            )['adjusted']
            for s in (0.0, 0.5, 1.0)
        }
        mid = float(np.mean(outs[0.5]))
        self.assertLess(mid, float(np.mean(outs[0.0])) + 1e-9)
        self.assertGreater(mid, float(np.mean(outs[1.0])) - 1e-9)

    def test_shrink_defaults_come_from_config(self):
        self.assertEqual(
            set(DEFAULT_ROBUST_INPUT_CONFIG['shrink']),
            {'seasonality', 'holidays', 'anomalies', 'level_shifts'},
        )
        implicit = robust_adjusted_panel(
            self.values, components=self.components
        )['adjusted']
        explicit = robust_adjusted_panel(
            self.values, components=self.components, shrink={'seasonality': 1.0}
        )['adjusted']
        np.testing.assert_allclose(implicit, explicit)


class TestDegenerateInput(unittest.TestCase):
    """Nothing here may raise; the worst case is a passthrough."""

    def _ok(self, result, shape=None):
        self.assertIsInstance(result, dict)
        self.assertIn('adjusted', result)
        self.assertTrue(np.isfinite(result['adjusted']).all())
        if shape is not None:
            self.assertEqual(result['adjusted'].shape, shape)

    def test_all_nan_column(self):
        values = make_contaminated_panel(seed=2, n_series=5, n_time=300)['values']
        values[:, 2] = np.nan
        self._ok(robust_adjusted_panel(values), values.shape)

    def test_all_nan_panel(self):
        values = np.full((200, 4), np.nan)
        self._ok(robust_adjusted_panel(values), values.shape)

    def test_constant_column(self):
        values = make_contaminated_panel(seed=2, n_series=5, n_time=300)['values']
        values[:, 1] = 42.0
        self._ok(robust_adjusted_panel(values), values.shape)

    def test_constant_panel(self):
        values = np.full((120, 3), 7.0)
        self._ok(robust_adjusted_panel(values), values.shape)

    def test_short_panel(self):
        for n_time in (1, 2, 5, 20):
            values = np.random.default_rng(0).normal(0, 1, (n_time, 3)) + 10
            self._ok(robust_adjusted_panel(values), (n_time, 3))

    def test_single_series_and_single_row(self):
        self._ok(robust_adjusted_panel(np.linspace(0, 1, 300).reshape(-1, 1)))
        self._ok(robust_adjusted_panel(np.ones((1, 4))))

    def test_rank_above_series_count(self):
        values = make_contaminated_panel(seed=2, n_series=3, n_time=300)['values']
        self._ok(robust_adjusted_panel(values, config={'rank': 25}), values.shape)

    def test_empty_and_malformed_components(self):
        values = make_contaminated_panel(seed=2, n_series=4, n_time=300)['values']
        self._ok(robust_adjusted_panel(values, components={}), values.shape)
        self._ok(
            robust_adjusted_panel(values, components={'seasonality': None}),
            values.shape,
        )
        self._ok(
            robust_adjusted_panel(
                values, components={'seasonality': np.zeros((10, 2))}
            ),
            values.shape,
        )
        self._ok(
            robust_adjusted_panel(values, components={'nonsense': values}),
            values.shape,
        )

    def test_infinities_and_out_of_range_shrink(self):
        values = make_contaminated_panel(seed=2, n_series=4, n_time=300)['values']
        values[10, 0] = np.inf
        values[11, 1] = -np.inf
        self._ok(
            robust_adjusted_panel(
                values, shrink={'seasonality': 5.0, 'holidays': -2.0}
            ),
            values.shape,
        )

    def test_dataframe_input_and_mask(self):
        panel = make_contaminated_panel(seed=2, n_series=4, n_time=300)
        index = pd.date_range('2021-01-01', periods=300, freq='D')
        df = pd.DataFrame(panel['values'], index=index)
        mask = pd.DataFrame(np.ones((300, 4), dtype=bool), index=index)
        self._ok(robust_adjusted_panel(df, mask=mask), (300, 4))
        self._ok(robust_adjusted_panel(df, mask=np.zeros((3, 3), dtype=bool)))

    def test_determinism(self):
        values = make_contaminated_panel(seed=4, n_series=6, n_time=400)['values']
        first = robust_adjusted_panel(values)['adjusted']
        second = robust_adjusted_panel(values)['adjusted']
        np.testing.assert_array_equal(first, second)


class TestCompareInputs(unittest.TestCase):
    """The measurement that separates decomposition from factor-model loss."""

    @classmethod
    def setUpClass(cls):
        cls.panel = make_contaminated_panel(seed=5, n_series=10, n_time=500)
        cls.raw = cls.panel['values']
        cls.oracle = cls.panel['trend']
        # a plausible "detector adjusted" panel: spikes removed, steps left
        cls.detector = cls.raw - cls.panel['spikes']
        cls.robust = robust_adjusted_panel(cls.raw)['adjusted']

    def test_oracle_adjusted_outranks_raw(self):
        scores = compare_inputs(self.raw, self.detector, self.robust, self.oracle)
        self.assertEqual(scores['ranking'][0], 'oracle')
        self.assertGreater(
            scores['oracle']['mean_abs_corr'], scores['raw']['mean_abs_corr']
        )
        self.assertLess(scores['oracle']['nrmse'], scores['raw']['nrmse'])
        self.assertLess(
            scores['oracle']['residual_energy_retained'],
            scores['raw']['residual_energy_retained'],
        )

    def test_anchor_values_are_exact(self):
        scores = compare_inputs(self.raw, self.detector, self.robust, self.oracle)
        self.assertAlmostEqual(scores['oracle']['mean_abs_corr'], 1.0, places=6)
        self.assertAlmostEqual(scores['oracle']['nrmse'], 0.0, places=6)
        self.assertAlmostEqual(
            scores['oracle']['residual_energy_retained'], 0.0, places=6
        )
        self.assertAlmostEqual(
            scores['raw']['residual_energy_retained'], 1.0, places=6
        )

    def test_robust_input_beats_both_raw_and_detector(self):
        scores = compare_inputs(self.raw, self.detector, self.robust, self.oracle)
        self.assertGreater(
            scores['robust']['mean_abs_corr'], scores['raw']['mean_abs_corr']
        )
        self.assertGreater(
            scores['robust']['mean_abs_corr'], scores['detector']['mean_abs_corr']
        )
        self.assertLess(
            scores['robust']['residual_energy_retained'],
            scores['detector']['residual_energy_retained'],
        )
        self.assertEqual(scores['ranking'][:2], ['oracle', 'robust'])

    def test_all_keys_present_and_degenerate_safe(self):
        scores = compare_inputs(self.raw, None, None, self.oracle)
        for name in ('raw', 'detector', 'robust', 'oracle'):
            self.assertIn(name, scores)
            for key in ('mean_abs_corr', 'nrmse', 'residual_energy_retained'):
                self.assertIn(key, scores[name])
        flat = np.ones_like(self.raw)
        safe = compare_inputs(flat, flat, flat, flat)
        self.assertIn('ranking', safe)

    def test_accepts_dataframes(self):
        index = pd.date_range('2021-01-01', periods=self.raw.shape[0], freq='D')
        frames = [
            pd.DataFrame(a, index=index)
            for a in (self.raw, self.detector, self.robust, self.oracle)
        ]
        scores = compare_inputs(*frames)
        self.assertEqual(scores['ranking'][0], 'oracle')


if __name__ == '__main__':
    unittest.main()
