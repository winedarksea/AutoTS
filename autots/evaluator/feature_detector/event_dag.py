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
    # Opt-in options (defaults reproduce the original behavior exactly):
    # 'series_scale' divides each event magnitude by a per-series scale so events
    # from different metrics/families are comparable, and makes family signatures
    # scale-invariant. Slope deltas become level effects via slope_horizon_periods.
    'magnitude_normalization': 'none',
    'slope_horizon_periods': 90,
    # Diagnostic only: tag events that repeat near the same day-of-year across years.
    'tag_recurring_doy': False,
}

# ~1.5x the per-series date jitter (sd ~9 d) of a shared event on daily data; the
# reviewer measured 3.7 / 2.0 / 1.0 clusters per shared event at 1 / 7 / 14 days.
_AUTO_WINDOW_DAYS = 14
_RECURRING_DOY_TOLERANCE_DAYS = 7
# Floor on residual noise as a fraction of series std so near-noiseless series do
# not turn every event into an astronomically large normalized magnitude.
_MIN_NOISE_FRACTION_OF_SCALE = 0.01


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
    window = resolved.get('cluster_window_periods', 1)
    if isinstance(window, str) and window.strip().lower() == 'auto':
        resolved['cluster_window_periods'] = 'auto'
    else:
        resolved['cluster_window_periods'] = max(int(window), 0)
    normalization = str(resolved.get('magnitude_normalization', 'none')).lower()
    if normalization not in ('none', 'series_scale'):
        normalization = 'none'
    resolved['magnitude_normalization'] = normalization
    resolved['slope_horizon_periods'] = max(
        int(resolved.get('slope_horizon_periods', 90)), 1
    )
    resolved['tag_recurring_doy'] = bool(resolved.get('tag_recurring_doy', False))
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
    dag = {
        'meta': {
            'enabled': bool(resolved.get('enabled', True)),
            'detection_mode': detection_mode,
            'construction_mode': construction_mode,
            'source_families': list(resolved['source_families']),
            'cluster_window_periods': (
                'auto'
                if resolved['cluster_window_periods'] == 'auto'
                else int(resolved['cluster_window_periods'])
            ),
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
    # Only emitted when opted in so default output stays identical.
    if resolved['magnitude_normalization'] != 'none':
        dag['meta']['magnitude_normalization'] = resolved['magnitude_normalization']
        dag['meta']['slope_horizon_periods'] = int(resolved['slope_horizon_periods'])
    if resolved['tag_recurring_doy']:
        dag['meta']['tag_recurring_doy'] = True
    return dag


def get_new_event_dag_params(method='random'):
    """Sample event DAG options; non-default choices are deliberately rare."""
    if method not in ('random', 'default'):
        method = 'random'
    if method == 'default':
        return {}
    rng = np.random
    window = rng.choice([1, 'auto', 7, 14], p=[0.7, 0.15, 0.1, 0.05])
    if not isinstance(window, str):
        window = int(window)
    return {
        'magnitude_normalization': (
            'series_scale' if rng.random() < 0.25 else 'none'
        ),
        'cluster_window_periods': window,
        'tag_recurring_doy': bool(rng.random() < 0.2),
        'slope_horizon_periods': int(rng.choice([28, 90, 365], p=[0.2, 0.6, 0.2])),
    }


def _resolve_window_periods(window, step: pd.Timedelta) -> int:
    """Turn 'auto' into ~14 days of periods (min 1); pass ints through."""
    if isinstance(window, str):
        periods = int(round(pd.Timedelta(days=_AUTO_WINDOW_DAYS) / step))
        return max(periods, 1)
    return int(window)


def _resolve_series_scales(detector, series_names):
    """Per-series scale in original units for magnitude normalization.

    Prefers residual noise sigma (noise_level is relative to series std, so
    sigma = noise_level * std) because events are then measured in noise units;
    floored at a fraction of std for stability. Falls back to std, then 1.0.
    """
    noise_levels = getattr(detector, 'series_noise_levels', None) or {}
    stds = getattr(detector, 'series_scales', None) or {}
    scale_series = getattr(detector, 'scale_series', None)
    scales = {}
    for name in series_names:
        std = _safe_float(stds.get(name, 0.0))
        if std <= 0 and scale_series is not None:
            try:
                std = _safe_float(scale_series[name])
            except (KeyError, IndexError, TypeError):
                std = 0.0
        sigma = _safe_float(noise_levels.get(name, 0.0)) * std
        if std > 0:
            sigma = max(sigma, _MIN_NOISE_FRACTION_OF_SCALE * std)
            scale = sigma
        else:
            scale = 0.0
        scales[name] = scale if scale > 0 else 1.0
    return scales


def _tag_recurring_day_of_year(members):
    """Set member['recurring_doy'] for same-series, same-family events that
    recur within +/-7 days of the same day-of-year in another year.

    Circular distance on a 365-day cycle handles the Dec/Jan wrap.
    """
    groups = {}
    for idx, member in enumerate(members):
        member['recurring_doy'] = False
        groups.setdefault((member['series_name'], member['source_family']), []).append(
            idx
        )
    for indices in groups.values():
        if len(indices) < 2:
            continue
        dates = pd.DatetimeIndex([pd.Timestamp(members[i]['date']) for i in indices])
        doy = np.asarray(dates.dayofyear, dtype=float)
        year = np.asarray(dates.year)
        raw = np.abs(doy[:, None] - doy[None, :])
        circular = np.minimum(raw, 365.0 - raw)
        close = (circular <= _RECURRING_DOY_TOLERANCE_DAYS) & (
            year[:, None] != year[None, :]
        )
        for i, flag in zip(indices, close.any(axis=1)):
            members[i]['recurring_doy'] = bool(flag)


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


def _safe_duration(value, max_periods=None) -> int:
    """Coerce a record ``duration`` to a sane period count.

    Records occasionally carry a corrupt or out-of-range ``duration`` (e.g. a
    raw sample count or a value derived from a multi-century / NaT date span).
    Such values overflow int64 nanoseconds when multiplied by ``step`` in
    :func:`_safe_end_date`, so clamp to ``[1, max_periods]`` before use.
    """
    try:
        duration = int(value)
    except (TypeError, ValueError):
        duration = 1
    duration = max(duration, 1)
    if max_periods is not None and max_periods >= 1:
        duration = min(duration, max_periods)
    return duration


def _safe_end_date(start_date, duration: int, step: pd.Timedelta):
    """``start_date + (duration - 1) * step`` that degrades instead of crashing.

    Even after clamping, defend the Timedelta multiply against int64 overflow so
    a pathological record can never abort ``detector.fit``.
    """
    try:
        return start_date + (duration - 1) * step
    except (OverflowError, ValueError):
        return start_date


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


def _extract_record_fields(
    family, record, step, shared_default=False, max_periods=None
):
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
        duration = _safe_duration(record.get('duration', 1) or 1, max_periods)
        end_date = _safe_end_date(start_date, duration, step)
    else:
        values = list(record)
        start_date = pd.Timestamp(values[0])
        if family == 'anomalies':
            magnitude = _safe_float(values[1] if len(values) > 1 else 0.0)
            subtype = str(values[2] if len(values) > 2 else 'point_outlier')
            duration = _safe_duration(values[3] if len(values) > 3 else 1, max_periods)
            shared_flag = bool(values[4] if len(values) > 4 else shared_default)
            end_date = _safe_end_date(start_date, duration, step)
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
    date_index = getattr(detector, 'date_index', None)
    max_periods = max(len(date_index) if date_index is not None else 0, 1)
    members = []
    normalize = params.get('magnitude_normalization', 'none') == 'series_scale'
    if normalize:
        series_scales = _resolve_series_scales(detector, columns)
        horizon = float(params.get('slope_horizon_periods', 90))

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
                    max_periods=max_periods,
                )
                member_series = series_name if mode != 'univariate' else '__broadcast__'
                if normalize:
                    # Slope deltas are units/period; horizon makes them a level effect.
                    level_factor = horizon if family == 'trend_changepoints' else 1.0
                    fields['normalized_magnitude'] = _safe_float(
                        fields['magnitude'] * level_factor / series_scales[series_name]
                    )
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
    if params.get('tag_recurring_doy'):
        _tag_recurring_day_of_year(members)
    return members, construction_mode


def _serialize_member_event(event):
    serialized = {
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
    # 'magnitude' stays raw (original units); the normalized value is additive.
    if 'normalized_magnitude' in event:
        serialized['normalized_magnitude'] = _safe_float(event['normalized_magnitude'])
    if 'recurring_doy' in event:
        serialized['recurring_doy'] = bool(event['recurring_doy'])
    return serialized


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
    raw_net_magnitude = 0.0
    raw_abs_magnitude = 0.0
    normalized = any('normalized_magnitude' in x for x in cluster_events)

    for event in cluster_events:
        source_counts[event['source_family']] = (
            source_counts.get(event['source_family'], 0) + 1
        )
        series_name = event['series_name']
        if series_name not in seen_series:
            affected_series.append(series_name)
            seen_series.add(series_name)
        value = _safe_float(event.get('normalized_magnitude', event['magnitude']))
        net_magnitude += value
        abs_magnitude += abs(value)
        raw_net_magnitude += _safe_float(event['magnitude'])
        raw_abs_magnitude += abs(_safe_float(event['magnitude']))

    cluster = {
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
    if normalized:
        cluster['raw_net_magnitude'] = _safe_float(raw_net_magnitude)
        cluster['raw_abs_magnitude'] = _safe_float(raw_abs_magnitude)
    return cluster


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


def _cluster_signature(
    cluster,
    member_lookup,
    series_names,
    source_families,
    magnitude_normalization='none',
):
    n_series = len(series_names)
    family_map = {name: idx for idx, name in enumerate(source_families)}
    incidence = np.zeros(n_series, dtype=float)
    signed = np.zeros(n_series, dtype=float)
    source_mix = np.zeros(len(source_families), dtype=float)
    series_index = {name: idx for idx, name in enumerate(series_names)}
    total_abs = max(abs(cluster.get('abs_magnitude', 0.0)), 1e-9)
    scale_invariant = magnitude_normalization == 'series_scale'

    for member_id in cluster.get('member_ids', []):
        member = member_lookup.get(member_id)
        if member is None:
            continue
        series_name = member.get('series_name')
        if series_name in series_index:
            idx = series_index[series_name]
            incidence[idx] = 1.0
            if scale_invariant:
                signed[idx] += _safe_float(member.get('normalized_magnitude', 0.0))
            else:
                signed[idx] += _safe_float(member.get('magnitude', 0.0)) / total_abs
        family = member.get('source_family')
        if family in family_map:
            source_mix[family_map[family]] += 1.0

    if scale_invariant:
        # No source_mix (identical for same-type events, it forced spurious merges)
        # and no division by cluster total (keeps relative strength across series).
        return np.concatenate([incidence, signed])
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
            cluster,
            member_lookup,
            series_names,
            source_families,
            params.get('magnitude_normalization', 'none'),
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
    params['cluster_window_periods'] = _resolve_window_periods(
        params['cluster_window_periods'], step
    )
    dag['meta']['cluster_window_periods'] = params['cluster_window_periods']
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
