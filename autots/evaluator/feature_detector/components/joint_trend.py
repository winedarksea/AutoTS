# -*- coding: utf-8 -*-
"""JointTrendMixin: opt-in joint hinge/step trend estimation for TimeSeriesFeatureDetector.

Enabled with ``changepoint_params={'method': 'joint_hinge_step', ...}``. PELT-l2
segments a ramp into a level staircase and the legacy level-shift check accepts
most random candidates; here slope changes (hinges) and level shifts (steps)
are selected and sized in one least-squares model with a BIC stop and
AR(1)-adjusted step t-tests. Also hosts the opt-in t-test level-shift validator
(``level_shift_validation={'method': 't_test'}``) usable with any changepoint method.
"""

import math
import random

import numpy as np
import pandas as pd

from autots.tools.seasonal import fourier_series
from ..utils.joint_trend_selection import (
    KNOT,
    STEP,
    joint_fit,
    profile_likelihood_windows,
    select_hinge_step_terms,
    split_transient_step_pairs,
)

JOINT_TREND_METHOD = 'joint_hinge_step'

DEFAULT_JOINT_TREND_PARAMS = {
    # 'lsm': knots searched, steps proposed by LevelShiftMagic then t-validated.
    # 'unified': steps searched alongside knots (less stable as data arrives).
    'step_search': 'lsm',
    # 3.0 beat 2.0 on SyntheticDailyGenerator (trend NRMSE 0.32 vs 0.42) and
    # avoided 2.0's regression on series with no level shifts.
    'bic_multiplier': 3.0,
    'step_t_threshold': 4.0,
    'min_segment_days': 28,
    'edge_days': 14,
    'max_terms': 16,
    # Opposite-sign step pairs within this many days that nearly cancel are
    # reported as transients rather than two permanent shifts. None disables.
    'transient_max_days': 90,
    # Remove selected knots+steps (fit with Fourier nuisance) before the
    # seasonality fit, so seasonality cannot absorb a piecewise trend.
    'seasonal_prior': True,
    # Refits seasonality and trend seeded with the previous pass's trend.
    'backfit_passes': 1,
    'date_windows': True,
    # BIC pruning on the AR(1) effective sample size (see joint_trend_selection).
    'effective_n_prune': True,
}


# (choices, weights) per sampled key; also the bounded domain for local mutation.
# Not sampled (benchmarked worse, kept as manual options): step_search='unified',
# effective_n_prune=False, backfit_passes=0.
JOINT_TREND_PARAM_SPACE = {
    'bic_multiplier': ([3.0, 2.0, 4.0], [0.5, 0.3, 0.2]),
    'step_t_threshold': ([4.0, 3.0, 5.0], [0.6, 0.2, 0.2]),
    'min_segment_days': ([28, 14, 56], [0.6, 0.2, 0.2]),
    'transient_max_days': ([90, None, 45], [0.6, 0.2, 0.2]),
    'seasonal_prior': ([True, False], [0.85, 0.15]),
    'backfit_passes': ([1, 2], [0.8, 0.2]),
}


def sample_joint_trend_changepoint_params(rng=None):
    """Random ``changepoint_params`` for the joint method (for get_new_params / optimizers)."""
    rng = rng or random
    method_params = {
        key: rng.choices(choices, weights)[0]
        for key, (choices, weights) in JOINT_TREND_PARAM_SPACE.items()
    }
    return {
        'method': JOINT_TREND_METHOD,
        'method_params': method_params,
        'aggregate_method': 'individual',
    }


def mutate_joint_trend_changepoint_params(params, rng=None, n_keys=2):
    """Resample a few method_params within the declared space.

    Generic numeric jitter would push booleans/ints (seasonal_prior,
    backfit_passes) and the BIC multiplier outside meaningful ranges.
    """
    rng = rng or random
    mutated = dict(params or {})
    method_params = dict(mutated.get('method_params') or {})
    for key in rng.sample(list(JOINT_TREND_PARAM_SPACE), n_keys):
        choices, weights = JOINT_TREND_PARAM_SPACE[key]
        method_params[key] = rng.choices(choices, weights)[0]
    mutated['method'] = JOINT_TREND_METHOD
    mutated['method_params'] = method_params
    mutated['aggregate_method'] = 'individual'
    return mutated


def _ar1_inflated_t(design, values):
    """Coefficient of the last design column and its AR(1)-adjusted t statistic."""
    beta, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ beta
    dof = max(len(values) - design.shape[1], 1)
    variance = float(residual @ residual) / dof
    if residual.size > 2 and np.std(residual) > 0:
        rho = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
        rho = float(np.clip(rho if np.isfinite(rho) else 0.0, 0.0, 0.95))
        variance *= (1.0 + rho) / (1.0 - rho)
    gram_inverse = np.linalg.pinv(design.T @ design)
    standard_error = math.sqrt(max(variance * gram_inverse[-1, -1], 1e-300))
    return float(beta[-1]), float(beta[-1] / standard_error)


class JointTrendMixin:
    """Joint hinge/step trend selection and t-test level-shift validation."""

    def _uses_joint_trend(self):
        return (self.changepoint_params or {}).get('method') == JOINT_TREND_METHOD

    def _days_per_period(self):
        index = self.date_index
        if index is None or len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
            return 1.0
        # Timedelta division is resolution-safe (asi8 may be us, not ns, in pandas 2+).
        step_days = np.median(np.diff(index)) / pd.Timedelta(days=1)
        return float(step_days) if step_days > 0 else 1.0

    def _days_to_points(self, days, minimum=1):
        return max(int(math.ceil(float(days) / self._days_per_period())), minimum)

    def _joint_trend_settings(self):
        settings = dict(DEFAULT_JOINT_TREND_PARAMS)
        settings.update((self.changepoint_params or {}).get('method_params') or {})
        settings['min_segment'] = self._days_to_points(settings['min_segment_days'], 2)
        settings['edge'] = self._days_to_points(settings['edge_days'], 1)
        transient_days = settings.get('transient_max_days')
        settings['transient_max_gap'] = (
            self._days_to_points(transient_days) if transient_days else None
        )
        return settings

    def _selection_kwargs(self, settings):
        return {
            'min_segment': settings['min_segment'],
            'edge': settings['edge'],
            'max_terms': int(settings['max_terms']),
            'bic_multiplier': float(settings['bic_multiplier']),
            'step_t_threshold': float(settings['step_t_threshold']),
            'search_steps': settings['step_search'] == 'unified',
            'effective_n_prune': bool(settings['effective_n_prune']),
        }

    @staticmethod
    def _trend_prior_fourier_nuisance(index):
        """Weekly (sub-weekly data) and yearly (1.5+ years) Fourier terms, or None.

        Yearly terms are omitted on short history because a ramp and a yearly
        cycle are not separable there; the ambiguity is attributed to trend.
        """
        if len(index) < 14 or not isinstance(index, pd.DatetimeIndex):
            return None
        days = np.asarray((index - index[0]) / pd.Timedelta(days=1), dtype=float)
        columns = []
        if np.median(np.diff(days)) < 7:
            columns.append(fourier_series(days, p=7, n=3))
        if days[-1] >= 548:
            columns.append(fourier_series(days, p=365.25, n=6))
        return np.column_stack(columns) if columns else None

    def _shared_or_individual_terms(self, values, select):
        """Per-column terms, or one shared set selected on the column mean (univariate)."""
        if self.detection_mode == 'univariate':
            shared = select(np.nanmean(values, axis=1), None)
            return [shared] * values.shape[1]
        return [select(values[:, j], j) for j in range(values.shape[1])]

    def _joint_seasonal_trend_prior(self, df):
        """Selected knots+steps fit jointly with Fourier nuisance; returns the trend+step part."""
        prior = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        nuisance = self._trend_prior_fourier_nuisance(df.index)
        if nuisance is None:
            return prior
        settings = self._joint_trend_settings()
        kwargs = self._selection_kwargs(settings)
        # Step search is always on here: unmodelled level shifts otherwise leak
        # into seasonality as yearly ghost steps on the same calendar day.
        kwargs['search_steps'] = True
        values = (
            df.interpolate(limit_direction='both').fillna(0.0).to_numpy(dtype=float)
        )
        terms_by_column = self._shared_or_individual_terms(
            values,
            lambda y, _: select_hinge_step_terms(y, nuisance=nuisance, **kwargs),
        )
        for j, terms in enumerate(terms_by_column):
            fit = joint_fit(values[:, j], terms, nuisance=nuisance)
            prior.iloc[:, j] = fit['trend'] + fit['level_shift']
        return prior

    def _lsm_step_proposals(self, residual_df):
        """LevelShiftMagic candidate positions per column (union across columns if shared)."""
        _, candidates = self._detect_level_shifts(residual_df)
        positions = {}
        for col in residual_df.columns:
            dates = [entry['date'] for entry in candidates.get(col, [])]
            found = self.date_index.searchsorted(pd.DatetimeIndex(dates))
            positions[col] = sorted(
                {int(p) for p in found if 0 < p < len(self.date_index)}
            )
        if self.detection_mode == 'univariate':
            union = sorted({p for found in positions.values() for p in found})
            positions = {col: union for col in residual_df.columns}
        return positions

    def _detect_joint_trend_and_shifts(self, final_residual, residual_for_trend):
        """Same return contract as ``_detect_trend_and_shifts``."""
        settings = self._joint_trend_settings()
        kwargs = self._selection_kwargs(settings)
        columns = list(residual_for_trend.columns)
        values = residual_for_trend.ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        unfiltered = (
            final_residual.reindex(columns=columns).ffill().bfill().fillna(0.0)
        ).to_numpy(dtype=float)
        proposals = (
            self._lsm_step_proposals(residual_for_trend)
            if settings['step_search'] == 'lsm'
            else {col: [] for col in columns}
        )
        terms_by_column = self._shared_or_individual_terms(
            values,
            lambda y, j: select_hinge_step_terms(
                y,
                fixed_steps=proposals[columns[j] if j is not None else columns[0]],
                **kwargs,
            ),
        )

        n = len(self.date_index)
        trend = np.zeros((n, len(columns)))
        level_shift = np.zeros((n, len(columns)))
        transient_blocks = np.zeros((n, len(columns)))
        changepoints, validated, details = {}, {}, {}
        position_index = np.arange(n)
        for j, col in enumerate(columns):
            terms = terms_by_column[j]
            fit = joint_fit(values[:, j], terms)
            windows = (
                profile_likelihood_windows(
                    unfiltered[:, j], terms, min_segment=settings['min_segment']
                )
                if settings['date_windows'] and terms
                else [(p, p) for _, p in terms]
            )
            for term, window in zip(fit['terms'], windows):
                term['window'] = window
            steps = [term for term in fit['terms'] if term['kind'] == STEP]
            transients = []
            if settings['transient_max_gap']:
                steps, transients = split_transient_step_pairs(
                    steps, settings['transient_max_gap']
                )
            trend[:, j] = fit['trend']
            # Transient pairs leave the level-shift output but are kept for the
            # backfit prior; otherwise the next seasonality fit absorbs them as a
            # pattern repeating on the same calendar days every year.
            transient_blocks[:, j] = fit['level_shift']
            for step in steps:
                level_shift[:, j] += step['coefficient'] * (
                    (position_index >= step['position']) - 1.0
                )
            knots = [term for term in fit['terms'] if term['kind'] == KNOT]
            changepoints[col] = [self.date_index[t['position']] for t in knots]
            validated[col] = [
                {
                    'date': self.date_index[step['position']],
                    'magnitude': step['coefficient'],
                    'validated_change': step['coefficient'],
                    'relative_change': 0.0,
                    't_stat': step['t_stat'],
                    'date_window': self._window_dates(step['window']),
                }
                for step in steps
            ]
            details[col] = {
                'changepoints': {
                    self.date_index[t['position']]: {
                        't_stat': t['t_stat'],
                        'date_window': self._window_dates(t['window']),
                    }
                    for t in knots
                },
                'transients': [
                    {
                        'start_date': self.date_index[item['start']],
                        'end_date': self.date_index[item['end']],
                        'magnitude': item['magnitude'],
                    }
                    for item in transients
                ],
            }
        transient_blocks -= level_shift
        self._joint_trend_details = details
        self._joint_transient_component = pd.DataFrame(
            transient_blocks, index=self.date_index, columns=columns
        )
        trend_component = pd.DataFrame(trend, index=self.date_index, columns=columns)
        level_shift_component = pd.DataFrame(
            level_shift, index=self.date_index, columns=columns
        )
        slope_info = self._compute_trend_slopes(trend_component, changepoints)
        return (
            trend_component,
            level_shift_component,
            validated,
            changepoints,
            slope_info,
        )

    def _window_dates(self, window):
        return [self.date_index[int(window[0])], self.date_index[int(window[1])]]

    def _validate_level_shifts_t_test(self, residual_df, lvlshft, candidates):
        """Keep a candidate step only if |t| >= threshold in a local line+step fit.

        The legacy check compares medians against thresholds OR'd with a relative
        test on a near-zero-centred residual, so most random dates pass. A step
        t-test inside a local linear fit also stops slope from posing as a step.
        Benchmarked as too strict on SyntheticDailyGenerator (keeps ~0.1 shifts per
        series vs ~0.6 true), hence only sampled at low probability.
        """
        params = self.level_shift_validation
        half_window = self._days_to_points(params.get('window_days', 90), 3)
        threshold = float(params.get('t_threshold', 4.0))
        validated_component = lvlshft.copy()
        validated = {}
        for col in residual_df.columns:
            series = residual_df[col].to_numpy(dtype=float)
            entries = []
            for candidate in candidates.get(col, []):
                date, magnitude = candidate['date'], candidate['magnitude']
                try:
                    idx = residual_df.index.get_loc(date)
                except KeyError:
                    continue
                start = max(0, idx - half_window)
                end = min(len(series), idx + half_window)
                local = series[start:end]
                offsets = np.arange(start, end)
                finite = np.isfinite(local)
                before = finite & (offsets < idx)
                after = finite & (offsets >= idx)
                t_stat, change = 0.0, 0.0
                if before.sum() >= 3 and after.sum() >= 3:
                    design = np.column_stack(
                        [
                            np.ones(finite.sum()),
                            (offsets[finite] - idx) / float(half_window),
                            (offsets[finite] >= idx).astype(float),
                        ]
                    )
                    change, t_stat = _ar1_inflated_t(design, local[finite])
                if abs(t_stat) >= threshold:
                    entries.append(
                        {
                            'date': date,
                            'magnitude': magnitude,
                            'validated_change': change,
                            'relative_change': 0.0,
                            't_stat': t_stat,
                        }
                    )
                else:
                    validated_component.loc[
                        validated_component.index < date, col
                    ] += magnitude
            validated[col] = entries
        return validated_component, validated
