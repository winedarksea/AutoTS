# -*- coding: utf-8 -*-
"""Event DAG filtering and plotting helpers."""

import copy

import pandas as pd

from .event_dag import empty_event_dag


def _date_overlaps(entry_start, entry_end, start_date=None, end_date=None):
    entry_start = pd.Timestamp(entry_start)
    entry_end = pd.Timestamp(entry_end)
    if start_date is not None and entry_end < pd.Timestamp(start_date):
        return False
    if end_date is not None and entry_start > pd.Timestamp(end_date):
        return False
    return True


def filter_event_dag(
    event_dag,
    series=None,
    start_date=None,
    end_date=None,
    include_members=True,
):
    """Filter an Event DAG view by series and date range."""
    if not event_dag:
        return empty_event_dag()
    filtered = {
        'meta': copy.deepcopy(event_dag.get('meta', {})),
        'member_events': [],
        'event_clusters': [],
        'event_families': [],
        'edges': [],
    }
    series_filter = None if series is None else set(series)

    selected_members = []
    selected_member_ids = set()
    for member in event_dag.get('member_events', []):
        if series_filter is not None and member.get('series_name') not in series_filter:
            continue
        if not _date_overlaps(
            member.get('start_date'),
            member.get('end_date'),
            start_date=start_date,
            end_date=end_date,
        ):
            continue
        selected_member_ids.add(member['member_id'])
        if include_members:
            selected_members.append(copy.deepcopy(member))

    selected_clusters = []
    selected_cluster_ids = set()
    for cluster in event_dag.get('event_clusters', []):
        cluster_series = set(cluster.get('affected_series', []))
        if series_filter is not None and not (cluster_series & series_filter):
            continue
        if not _date_overlaps(
            cluster.get('start_date'),
            cluster.get('end_date'),
            start_date=start_date,
            end_date=end_date,
        ):
            continue
        member_intersection = selected_member_ids.intersection(
            cluster.get('member_ids', [])
        )
        if series_filter is not None and not member_intersection:
            continue
        selected_cluster_ids.add(cluster['cluster_id'])
        cluster_copy = copy.deepcopy(cluster)
        if include_members:
            cluster_copy['member_ids'] = [
                member_id
                for member_id in cluster_copy.get('member_ids', [])
                if member_id in selected_member_ids
            ]
        selected_clusters.append(cluster_copy)

    selected_families = []
    selected_family_ids = set()
    for family in event_dag.get('event_families', []):
        cluster_ids = set(family.get('cluster_ids', []))
        if not (cluster_ids & selected_cluster_ids):
            continue
        family_copy = copy.deepcopy(family)
        family_copy['cluster_ids'] = [
            cluster_id
            for cluster_id in family_copy.get('cluster_ids', [])
            if cluster_id in selected_cluster_ids
        ]
        selected_family_ids.add(family['family_id'])
        selected_families.append(family_copy)

    for edge in event_dag.get('edges', []):
        if edge.get('edge_type') == 'contains':
            if edge.get('source_id') in selected_cluster_ids:
                if include_members and edge.get('target_id') in selected_member_ids:
                    filtered['edges'].append(copy.deepcopy(edge))
        elif edge.get('edge_type') == 'repeats':
            if (
                edge.get('source_id') in selected_family_ids
                and edge.get('target_id') in selected_cluster_ids
            ):
                filtered['edges'].append(copy.deepcopy(edge))

    filtered['member_events'] = selected_members if include_members else []
    filtered['event_clusters'] = selected_clusters
    filtered['event_families'] = selected_families
    return filtered


def plot_event_dag_timeline(
    event_dag,
    series=None,
    start_date=None,
    end_date=None,
    show_members=False,
    figsize=(14, 6),
    save_path=None,
    show=True,
):
    """Render a timeline-first Event DAG view."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise ImportError("matplotlib is required for Event DAG plotting.") from exc

    filtered = filter_event_dag(
        event_dag,
        series=series,
        start_date=start_date,
        end_date=end_date,
        include_members=show_members,
    )
    clusters = filtered.get('event_clusters', [])
    families = filtered.get('event_families', [])
    members = filtered.get('member_events', [])

    if families:
        rows = [(family['family_id'], family['family_id']) for family in families]
    elif clusters:
        rows = [('unassigned', 'Events')]
    else:
        rows = [('empty', 'Events')]

    y_lookup = {row_id: idx for idx, (row_id, _) in enumerate(rows)}
    fig_height = max(figsize[1], 2.5 + 0.45 * len(rows))
    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(rows), 1)))

    family_line_points = {}
    for cluster in clusters:
        cluster_date = pd.Timestamp(cluster['center_date'])
        family_id = cluster.get('family_id')
        row_id = family_id if family_id in y_lookup else rows[0][0]
        y = y_lookup[row_id]
        color = palette[y % len(palette)]
        size = 40 + 35 * max(int(cluster.get('series_count', 1)), 1)
        ax.scatter(
            [cluster_date],
            [y],
            s=size,
            color=color,
            alpha=0.85,
            edgecolors='black',
            linewidths=0.5,
            zorder=3,
        )
        family_line_points.setdefault(row_id, []).append(cluster_date)

    for row_id, points in family_line_points.items():
        if len(points) > 1:
            points = sorted(points)
            y = y_lookup[row_id]
            ax.plot(
                points,
                [y] * len(points),
                color='0.55',
                alpha=0.35,
                linewidth=1.0,
                zorder=1,
            )

    if show_members and members:
        member_y = {cluster['cluster_id']: None for cluster in clusters}
        for cluster in clusters:
            family_id = cluster.get('family_id')
            row_id = family_id if family_id in y_lookup else rows[0][0]
            member_y[cluster['cluster_id']] = y_lookup[row_id]
        member_to_cluster = {
            edge['target_id']: edge['source_id']
            for edge in filtered.get('edges', [])
            if edge.get('edge_type') == 'contains'
        }
        for idx, member in enumerate(members):
            cluster_id = member_to_cluster.get(member['member_id'])
            if cluster_id not in member_y:
                continue
            y = member_y[cluster_id] + ((idx % 3) - 1) * 0.08
            marker = 'x' if member.get('direction') == 'negative' else 'o'
            ax.scatter(
                [pd.Timestamp(member['date'])],
                [y],
                s=24,
                marker=marker,
                color='0.2',
                alpha=0.55,
                linewidths=0.75,
                zorder=4,
            )

    ax.set_yticks(list(y_lookup.values()))
    ax.set_yticklabels([label for _, label in rows])
    ax.set_title("Event DAG Timeline")
    ax.set_xlabel("Time")
    ax.set_ylabel("Event Family")
    ax.grid(axis='x', alpha=0.25, linestyle=':')

    if clusters:
        all_dates = [pd.Timestamp(cluster['center_date']) for cluster in clusters]
        pad = pd.Timedelta(days=1)
        ax.set_xlim(min(all_dates) - pad, max(all_dates) + pad)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    return fig
