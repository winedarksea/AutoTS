# -*- coding: utf-8 -*-
"""Event DAG utilities for TimeSeriesFeatureDetector."""

import copy
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_EVENT_DAG_PARAMS = {
    'enabled': True,
    'source_families': ['anomalies', 'trend_changepoints', 'level_shifts'],
    'cluster_window_periods': 1,
    'family_similarity_threshold': 0.6,
    'min_family_occurrences': 2,
    'build_singleton_clusters': True,
}


def resolve_event_dag_params(params=None):
    """Return normalized Event DAG params."""
    resolved = copy.deepcopy(DEFAULT_EVENT_DAG_PARAMS)
    if params:
        resolved.update(copy.deepcopy(params))
    resolved['enabled'] = bool(resolved.get('enabled', True))
    families = resolved.get(
        'source_families', DEFAULT_EVENT_DAG_PARAMS['source_families']
    )
    if not isinstance(families, (list, tuple)):
        families = DEFAULT_EVENT_DAG_PARAMS['source_families']
    resolved['source_families'] = [str(x) for x in families if x]
    if not resolved['source_families']:
        resolved['source_families'] = copy.deepcopy(
            DEFAULT_EVENT_DAG_PARAMS['source_families']
        )
    resolved['cluster_window_periods'] = max(
        int(resolved.get('cluster_window_periods', 1)), 0
    )
    resolved['family_similarity_threshold'] = float(
        np.clip(resolved.get('family_similarity_threshold', 0.6), -1.0, 1.0)
    )
    resolved['min_family_occurrences'] = max(
        int(resolved.get('min_family_occurrences', 2)), 2
    )
    resolved['build_singleton_clusters'] = bool(
        resolved.get('build_singleton_clusters', True)
    )
    return resolved


def empty_event_dag(
    params=None,
    detection_mode='multivariate',
    construction_mode='full',
    series_names=None,
):
    """Return a valid empty Event DAG container."""
    resolved = resolve_event_dag_params(params)
    series_names = list(series_names or [])
    return {
        'meta': {
            'enabled': bool(resolved.get('enabled', True)),
            'detection_mode': detection_mode,
            'construction_mode': construction_mode,
            'source_families': list(resolved['source_families']),
            'cluster_window_periods': int(resolved['cluster_window_periods']),
            'family_similarity_threshold': float(
                resolved['family_similarity_threshold']
            ),
            'min_family_occurrences': int(resolved['min_family_occurrences']),
            'build_singleton_clusters': bool(resolved['build_singleton_clusters']),
            'series_names': series_names,
        },
        'member_events': [],
        'event_clusters': [],
        'event_families': [],
        'edges': [],
    }


def _infer_step_timedelta(date_index) -> pd.Timedelta:
    if date_index is None or len(date_index) < 2:
        return pd.Timedelta(days=1)
    diffs = pd.Series(date_index).diff().dropna()
    if diffs.empty:
        return pd.Timedelta(days=1)
    try:
        step = pd.to_timedelta(diffs.median())
    except Exception:
        step = pd.Timedelta(days=1)
    if not isinstance(step, pd.Timedelta) or step <= pd.Timedelta(0):
        return pd.Timedelta(days=1)
    return step


def _timestamp_to_iso(value) -> Optional[str]:
    if value is None:
        return None
    return pd.Timestamp(value).isoformat()


def _periods_between(start, end, step: pd.Timedelta) -> int:
    if start is None or end is None:
        return 1
    if step <= pd.Timedelta(0):
        return 1
    delta = pd.Timestamp(end) - pd.Timestamp(start)
    periods = int(round(delta / step)) + 1
    return max(periods, 1)


def _safe_float(value) -> float:
    try:
        result = float(value)
    except Exception:
        result = 0.0
    if not np.isfinite(result):
        return 0.0
    return result


def _family_order_index(source_families):
    return {name: idx for idx, name in enumerate(source_families)}


def _extract_record_fields(family, record, step, shared_default=False):
    shared_flag = bool(shared_default)
    subtype = family
    magnitude = 0.0
    start_date = None
    end_date = None

    if isinstance(record, dict):
        start_date = pd.Timestamp(record.get('date'))
        magnitude = _safe_float(record.get('magnitude', record.get('new_slope', 0.0)))
        shared_flag = bool(record.get('shared', shared_default))
        subtype = str(
            record.get(
                'type',
                record.get(
                    'pattern',
                    record.get('shift_type', record.get('description', family)),
                ),
            )
        )
        if family == 'trend_changepoints':
            prior_slope = _safe_float(record.get('prior_slope', 0.0))
            new_slope = _safe_float(record.get('new_slope', magnitude))
            magnitude = new_slope - prior_slope
        duration = max(int(record.get('duration', 1) or 1), 1)
        end_date = start_date + (duration - 1) * step
    else:
        values = list(record)
        start_date = pd.Timestamp(values[0])
        if family == 'anomalies':
            magnitude = _safe_float(values[1] if len(values) > 1 else 0.0)
            subtype = str(values[2] if len(values) > 2 else 'point_outlier')
            duration = max(int(values[3] if len(values) > 3 else 1), 1)
            shared_flag = bool(values[4] if len(values) > 4 else shared_default)
            end_date = start_date + (duration - 1) * step
        elif family == 'level_shifts':
            magnitude = _safe_float(values[1] if len(values) > 1 else 0.0)
            subtype = str(values[2] if len(values) > 2 else 'validated')
            shared_flag = bool(values[3] if len(values) > 3 else shared_default)
            end_date = start_date
        elif family == 'trend_changepoints':
            prior_slope = _safe_float(values[1] if len(values) > 1 else 0.0)
            new_slope = _safe_float(values[2] if len(values) > 2 else 0.0)
            magnitude = new_slope - prior_slope
            subtype = 'trend_changepoint'
            end_date = start_date
        else:
            magnitude = _safe_float(values[1] if len(values) > 1 else 0.0)
            end_date = start_date

    direction = (
        'positive' if magnitude > 0 else 'negative' if magnitude < 0 else 'neutral'
    )
    return {
        'date': start_date,
        'start_date': start_date,
        'end_date': end_date,
        'magnitude': magnitude,
        'direction': direction,
        'subtype': subtype,
        'shared_flag': shared_flag,
    }


def _extract_member_events(detector, params, step):
    source_families = list(params['source_families'])
    mode = getattr(detector, 'detection_mode', 'multivariate')
    columns = list(getattr(detector.df_original, 'columns', []))
    family_rank = _family_order_index(source_families)
    members = []

    if mode == 'univariate':
        shared_series = columns[0] if columns else '__broadcast__'
        iter_series = [shared_series]
        construction_mode = 'broadcast'
    else:
        iter_series = columns
        construction_mode = 'full'

    for family in source_families:
        family_data = getattr(detector, family, {})
        if not isinstance(family_data, dict):
            continue
        for series_name in iter_series:
            records = family_data.get(series_name, [])
            if not records:
                continue
            for idx, record in enumerate(records):
                fields = _extract_record_fields(
                    family,
                    record,
                    step=step,
                    shared_default=(mode == 'univariate'),
                )
                member_series = series_name if mode != 'univariate' else '__broadcast__'
                members.append(
                    {
                        'member_id': f"{family}:{member_series}:{idx}",
                        'series_name': member_series,
                        'source_family': family,
                        'family_rank': family_rank.get(family, len(family_rank)),
                        **fields,
                    }
                )

    members.sort(
        key=lambda x: (
            pd.Timestamp(x['start_date']),
            x['family_rank'],
            x['series_name'],
            x['member_id'],
        )
    )
    return members, construction_mode


def _serialize_member_event(event):
    return {
        'member_id': event['member_id'],
        'series_name': event['series_name'],
        'source_family': event['source_family'],
        'date': _timestamp_to_iso(event['date']),
        'start_date': _timestamp_to_iso(event['start_date']),
        'end_date': _timestamp_to_iso(event['end_date']),
        'magnitude': _safe_float(event['magnitude']),
        'direction': event['direction'],
        'subtype': event['subtype'],
        'shared_flag': bool(event['shared_flag']),
    }


def _finalize_cluster(cluster_events, cluster_id, step):
    start_date = min(pd.Timestamp(x['start_date']) for x in cluster_events)
    end_date = max(pd.Timestamp(x['end_date']) for x in cluster_events)
    center_ns = int(np.median([pd.Timestamp(x['date']).value for x in cluster_events]))
    center_date = pd.Timestamp(center_ns)
    source_counts = {}
    affected_series = []
    seen_series = set()
    net_magnitude = 0.0
    abs_magnitude = 0.0

    for event in cluster_events:
        source_counts[event['source_family']] = (
            source_counts.get(event['source_family'], 0) + 1
        )
        series_name = event['series_name']
        if series_name not in seen_series:
            affected_series.append(series_name)
            seen_series.add(series_name)
        net_magnitude += _safe_float(event['magnitude'])
        abs_magnitude += abs(_safe_float(event['magnitude']))

    return {
        'cluster_id': cluster_id,
        'start_date': _timestamp_to_iso(start_date),
        'end_date': _timestamp_to_iso(end_date),
        'center_date': _timestamp_to_iso(center_date),
        'member_ids': [x['member_id'] for x in cluster_events],
        'affected_series': sorted(affected_series),
        'series_count': len(affected_series),
        'source_family_counts': source_counts,
        'net_magnitude': _safe_float(net_magnitude),
        'abs_magnitude': _safe_float(abs_magnitude),
        'duration_periods': _periods_between(start_date, end_date, step),
        'is_shared_root_cause_candidate': len(affected_series) >= 2,
    }


def _build_clusters(member_events, params, step):
    if not member_events:
        return [], []
    window_delta = params['cluster_window_periods'] * step
    clusters = []
    edges = []
    current = [member_events[0]]
    current_end = pd.Timestamp(member_events[0]['end_date'])

    for event in member_events[1:]:
        start = pd.Timestamp(event['start_date'])
        if start <= current_end + window_delta:
            current.append(event)
            current_end = max(current_end, pd.Timestamp(event['end_date']))
        else:
            if params['build_singleton_clusters'] or len(current) > 1:
                cluster_id = f"event_cluster:{len(clusters)}"
                cluster = _finalize_cluster(current, cluster_id, step)
                clusters.append(cluster)
                edges.extend(
                    {
                        'source_id': cluster_id,
                        'target_id': member['member_id'],
                        'edge_type': 'contains',
                    }
                    for member in current
                )
            current = [event]
            current_end = pd.Timestamp(event['end_date'])

    if current and (params['build_singleton_clusters'] or len(current) > 1):
        cluster_id = f"event_cluster:{len(clusters)}"
        cluster = _finalize_cluster(current, cluster_id, step)
        clusters.append(cluster)
        edges.extend(
            {
                'source_id': cluster_id,
                'target_id': member['member_id'],
                'edge_type': 'contains',
            }
            for member in current
        )

    return clusters, edges


def _cluster_signature(cluster, member_lookup, series_names, source_families):
    n_series = len(series_names)
    family_map = {name: idx for idx, name in enumerate(source_families)}
    incidence = np.zeros(n_series, dtype=float)
    signed = np.zeros(n_series, dtype=float)
    source_mix = np.zeros(len(source_families), dtype=float)
    series_index = {name: idx for idx, name in enumerate(series_names)}
    total_abs = max(abs(cluster.get('abs_magnitude', 0.0)), 1e-9)

    for member_id in cluster.get('member_ids', []):
        member = member_lookup.get(member_id)
        if member is None:
            continue
        series_name = member.get('series_name')
        if series_name in series_index:
            idx = series_index[series_name]
            incidence[idx] = 1.0
            signed[idx] += _safe_float(member.get('magnitude', 0.0)) / total_abs
        family = member.get('source_family')
        if family in family_map:
            source_mix[family_map[family]] += 1.0

    total_mix = source_mix.sum()
    if total_mix > 0:
        source_mix = source_mix / total_mix
    return np.concatenate([incidence, signed, source_mix])


def _cosine_similarity(left, right):
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm <= 0 or right_norm <= 0:
        return -1.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _build_families(clusters, member_lookup, params, series_names):
    if not clusters:
        return [], []

    source_families = list(params['source_families'])
    groups = []
    for cluster in clusters:
        signature = _cluster_signature(
            cluster, member_lookup, series_names, source_families
        )
        best_idx = None
        best_score = -1.0
        for idx, group in enumerate(groups):
            score = _cosine_similarity(signature, group['centroid'])
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None and best_score >= params['family_similarity_threshold']:
            group = groups[best_idx]
            group['clusters'].append(cluster)
            group['signatures'].append(signature)
            group['centroid'] = np.mean(group['signatures'], axis=0)
        else:
            groups.append(
                {
                    'clusters': [cluster],
                    'signatures': [signature],
                    'centroid': signature,
                }
            )

    event_families = []
    edges = []
    family_id_lookup = {}
    for group in groups:
        if len(group['clusters']) < params['min_family_occurrences']:
            continue
        family_id = f"event_family:{len(event_families)}"
        family_clusters = group['clusters']
        cluster_ids = [cluster['cluster_id'] for cluster in family_clusters]
        first_date = min(
            pd.Timestamp(cluster['start_date']) for cluster in family_clusters
        )
        last_date = max(
            pd.Timestamp(cluster['end_date']) for cluster in family_clusters
        )
        affected_series = sorted(
            {
                series_name
                for cluster in family_clusters
                for series_name in cluster.get('affected_series', [])
            }
        )
        source_counts = {}
        for cluster in family_clusters:
            for source_family, count in cluster.get('source_family_counts', {}).items():
                source_counts[source_family] = (
                    source_counts.get(source_family, 0) + count
                )
            family_id_lookup[cluster['cluster_id']] = family_id
            edges.append(
                {
                    'source_id': family_id,
                    'target_id': cluster['cluster_id'],
                    'edge_type': 'repeats',
                }
            )
        event_families.append(
            {
                'family_id': family_id,
                'cluster_ids': cluster_ids,
                'occurrence_count': len(cluster_ids),
                'first_date': _timestamp_to_iso(first_date),
                'last_date': _timestamp_to_iso(last_date),
                'affected_series': affected_series,
                'source_family_counts': source_counts,
            }
        )

    for cluster in clusters:
        cluster['family_id'] = family_id_lookup.get(cluster['cluster_id'])

    return event_families, edges


def build_event_dag_from_detector(detector):
    """Build an Event DAG from detector public event outputs."""
    params = resolve_event_dag_params(getattr(detector, 'event_dag_params', None))
    series_names = list(getattr(detector.df_original, 'columns', []))
    mode = getattr(detector, 'detection_mode', 'multivariate')
    dag = empty_event_dag(
        params=params,
        detection_mode=mode,
        construction_mode='broadcast' if mode == 'univariate' else 'full',
        series_names=series_names,
    )
    if not params['enabled']:
        return dag

    step = _infer_step_timedelta(getattr(detector, 'date_index', None))
    member_events, construction_mode = _extract_member_events(detector, params, step)
    dag['meta']['construction_mode'] = construction_mode
    if not member_events:
        return dag

    serialized_members = [_serialize_member_event(event) for event in member_events]
    clusters, cluster_edges = _build_clusters(member_events, params, step)
    member_lookup = {event['member_id']: event for event in serialized_members}
    if mode == 'univariate':
        event_families = []
        family_edges = []
        for cluster in clusters:
            cluster['family_id'] = None
    else:
        event_families, family_edges = _build_families(
            clusters,
            member_lookup,
            params,
            series_names=series_names,
        )

    dag['member_events'] = serialized_members
    dag['event_clusters'] = clusters
    dag['event_families'] = event_families
    dag['edges'] = family_edges + cluster_edges
    return dag
