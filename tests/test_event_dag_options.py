# -*- coding: utf-8 -*-
"""Tests for opt-in Event DAG options (normalization, auto window, DOY tags)."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from autots.evaluator.feature_detector.event_dag import (
    build_event_dag_from_detector,
    get_new_event_dag_params,
    resolve_event_dag_params,
)


def _stub_detector(
    index,
    columns,
    level_shifts=None,
    trend_changepoints=None,
    anomalies=None,
    scales=None,
    noise_levels=None,
    event_dag_params=None,
):
    return SimpleNamespace(
        detection_mode='multivariate',
        df_original=pd.DataFrame(0.0, index=index, columns=columns),
        date_index=index,
        level_shifts=level_shifts or {},
        trend_changepoints=trend_changepoints or {},
        anomalies=anomalies or {},
        series_scales=scales if scales is not None else {c: 1.0 for c in columns},
        series_noise_levels=(
            noise_levels if noise_levels is not None else {c: 0.1 for c in columns}
        ),
        scale_series=None,
        event_dag_params=event_dag_params,
    )


def _xy_detector(b_factor=1.0, params=None):
    """X = A+,B+ at days 100/400; Y = A+,C- at days 700/1000."""
    index = pd.date_range('2020-01-01', periods=1100, freq='D')
    d = lambda n: index[n]
    ls = {
        'A': [(d(100), 10.0, 'validated', False), (d(400), 10.0, 'validated', False),
              (d(700), 10.0, 'validated', False), (d(1000), 10.0, 'validated', False)],
        'B': [(d(100), 10.0 * b_factor, 'validated', False),
              (d(400), 10.0 * b_factor, 'validated', False)],
        'C': [(d(700), -10.0, 'validated', False), (d(1000), -10.0, 'validated', False)],
    }
    scales = {'A': 1.0, 'B': 1.0 * b_factor, 'C': 1.0}
    return _stub_detector(
        index, ['A', 'B', 'C'], level_shifts=ls, scales=scales, event_dag_params=params
    )


class TestEventDagOptions(unittest.TestCase):
    def test_default_equals_explicit_none(self):
        det = _xy_detector()
        base = build_event_dag_from_detector(det)
        det.event_dag_params = {'magnitude_normalization': 'none'}
        self.assertEqual(base, build_event_dag_from_detector(det))
        self.assertNotIn('normalized_magnitude', base['member_events'][0])
        self.assertNotIn('recurring_doy', base['member_events'][0])
        self.assertNotIn('magnitude_normalization', base['meta'])

    def test_xy_separates_and_is_scale_invariant(self):
        params = {'magnitude_normalization': 'series_scale'}
        dag = build_event_dag_from_detector(_xy_detector(params=params))
        self.assertEqual(len(dag['event_families']), 2)
        dag_big = build_event_dag_from_detector(
            _xy_detector(b_factor=1000.0, params=params)
        )
        self.assertEqual(len(dag_big['event_families']), 2)
        self.assertEqual(
            [f['cluster_ids'] for f in dag['event_families']],
            [f['cluster_ids'] for f in dag_big['event_families']],
        )

    def test_slope_delta_converted_by_horizon(self):
        index = pd.date_range('2021-01-01', periods=200, freq='D')
        d = index[50]
        det = _stub_detector(
            index,
            ['A'],
            level_shifts={'A': [(d, 1.0, 'validated', False)]},
            trend_changepoints={'A': [(d, 0.0, -0.05)]},
            noise_levels={'A': 0.5},
            event_dag_params={
                'magnitude_normalization': 'series_scale',
                'slope_horizon_periods': 90,
            },
        )
        dag = build_event_dag_from_detector(det)
        self.assertEqual(len(dag['event_clusters']), 1)
        cluster = dag['event_clusters'][0]
        self.assertLess(cluster['net_magnitude'], 0)
        # raw (unit-mixed) sum is still positive-looking: -0.05 + 1 > 0
        self.assertGreater(cluster['raw_net_magnitude'], 0)

    def test_auto_window_resolution(self):
        resolved = resolve_event_dag_params({'cluster_window_periods': 'auto'})
        self.assertEqual(resolved['cluster_window_periods'], 'auto')
        for freq, expected in [('D', 14), ('W', 2), ('MS', 1)]:
            index = pd.date_range('2020-01-01', periods=100, freq=freq)
            det = _stub_detector(
                index,
                ['A'],
                level_shifts={'A': [(index[10], 1.0, 'validated', False)]},
                event_dag_params={'cluster_window_periods': 'auto'},
            )
            dag = build_event_dag_from_detector(det)
            self.assertEqual(dag['meta']['cluster_window_periods'], expected, freq)

    def test_auto_window_merges_jittered_events(self):
        index = pd.date_range('2020-01-01', periods=100, freq='D')
        ls = {
            'A': [(index[10], 1.0, 'validated', False)],
            'B': [(index[16], 1.0, 'validated', False)],
        }
        det = _stub_detector(index, ['A', 'B'], level_shifts=ls)
        self.assertEqual(len(build_event_dag_from_detector(det)['event_clusters']), 2)
        det.event_dag_params = {'cluster_window_periods': 'auto'}
        self.assertEqual(len(build_event_dag_from_detector(det)['event_clusters']), 1)

    def test_recurring_doy_tag(self):
        index = pd.date_range('2018-01-01', periods=365 * 4, freq='D')
        ts = pd.Timestamp
        ls = {
            'A': [
                (ts('2018-03-10'), 1.0, 'validated', False),
                (ts('2019-03-14'), 1.0, 'validated', False),  # near-same DOY
                (ts('2018-12-30'), 1.0, 'validated', False),
                (ts('2020-01-02'), 1.0, 'validated', False),  # Dec/Jan wrap
                (ts('2019-07-01'), 1.0, 'validated', False),  # isolated
                (ts('2019-07-03'), 1.0, 'validated', False),  # same year only
            ]
        }
        det = _stub_detector(
            index, ['A'], level_shifts=ls, event_dag_params={'tag_recurring_doy': True}
        )
        dag = build_event_dag_from_detector(det)
        tags = {m['date'][:10]: m['recurring_doy'] for m in dag['member_events']}
        self.assertTrue(tags['2018-03-10'])
        self.assertTrue(tags['2019-03-14'])
        self.assertTrue(tags['2018-12-30'])
        self.assertTrue(tags['2020-01-02'])
        self.assertFalse(tags['2019-07-01'])
        self.assertFalse(tags['2019-07-03'])

    def test_get_new_event_dag_params_resolves(self):
        for _ in range(50):
            sampled = get_new_event_dag_params('random')
            resolved = resolve_event_dag_params(sampled)
            self.assertIn(resolved['magnitude_normalization'], ('none', 'series_scale'))
            copy.deepcopy(resolved)
        self.assertEqual(get_new_event_dag_params('default'), {})


class TestEventDagOptionsRealDetector(unittest.TestCase):
    def test_smoke(self):
        from autots.datasets import load_daily
        from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

        df = load_daily(long=False).iloc[-730:, :4]
        det = TimeSeriesFeatureDetector(
            event_dag_params={
                'magnitude_normalization': 'series_scale',
                'cluster_window_periods': 'auto',
                'tag_recurring_doy': True,
            }
        )
        det.fit(df)
        dag = det.get_event_dag()
        self.assertEqual(dag['meta']['cluster_window_periods'], 14)
        for m in dag['member_events']:
            self.assertIn('recurring_doy', m)
            self.assertIn('normalized_magnitude', m)


if __name__ == '__main__':
    unittest.main()
