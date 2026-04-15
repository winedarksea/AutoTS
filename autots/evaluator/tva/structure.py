# -*- coding: utf-8 -*-
"""Structure learning helpers for TVA.

This module keeps graph discovery, hierarchy discovery, export, and plotting
separate from the forecast orchestration layer so new approaches can be
swapped in without rewriting the TVA training loop.
"""

from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    Line2D = None
    FancyArrowPatch = None
    FancyBboxPatch = None
    Patch = None
    HAS_MATPLOTLIB = False


@dataclass
class StructureLearningConfig:
    """Opt-in configuration for TVA structure discovery."""

    enabled: bool = False
    learn_hierarchy: bool = True
    learn_dag: bool = True
    max_levels: int = 3
    pool_ratio: float = 0.5
    min_nodes_per_level: int = 2
    dag_penalty: float = 0.1
    dag_warmup_epochs: float = 0.2
    sparsity_weight: float = 0.01
    assignment_entropy_weight: float = 0.01
    assignment_full_rank_weight: float = 0.01
    prior_tether_weight: float = 0.05
    temporal_drift_weight: float = 0.0
    threshold_for_export: float = 0.2

    @classmethod
    def from_dict(cls, config: Optional[dict] = None) -> "StructureLearningConfig":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        return cls(**dict(config))

    def warmup_start_epoch(self, total_epochs: int) -> int:
        if self.dag_warmup_epochs <= 1:
            return int(total_epochs * max(self.dag_warmup_epochs, 0.0))
        return int(max(self.dag_warmup_epochs, 0.0))

    def structure_scale(self, epoch_index: int, total_epochs: int) -> float:
        if not self.enabled:
            return 0.0
        warmup_start = self.warmup_start_epoch(total_epochs)
        if epoch_index < warmup_start:
            return 0.0
        ramp_denom = max(total_epochs - warmup_start, 1)
        return float(min(max((epoch_index - warmup_start + 1) / ramp_denom, 0.0), 1.0))

    def derive_latent_sizes(
        self,
        n_anchor: int,
        fallback_sizes: Optional[list] = None,
    ) -> list:
        """Return deterministic latent widths from anchor count.

        The returned list excludes the input anchor level and contains only
        derived latent widths ordered bottom-up.
        """
        if n_anchor <= 0:
            return list(fallback_sizes or [])
        if not self.enabled or not self.learn_hierarchy:
            return list(fallback_sizes or [])

        derived_sizes = []
        current_size = int(n_anchor)
        max_levels = max(int(self.max_levels), 1)
        min_nodes = max(int(self.min_nodes_per_level), 1)
        pool_ratio = min(max(float(self.pool_ratio), 0.05), 0.95)

        for _ in range(max_levels):
            next_size = max(int(np.ceil(current_size * pool_ratio)), 1)
            if derived_sizes and next_size >= derived_sizes[-1]:
                next_size = max(derived_sizes[-1] - 1, 1)
            if next_size >= current_size:
                next_size = max(current_size - 1, 1)
            derived_sizes.append(next_size)
            current_size = next_size
            if current_size <= min_nodes:
                break

        if not derived_sizes:
            derived_sizes = list(fallback_sizes or [])
        return derived_sizes

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphSnapshot:
    """Serializable export of learned TVA structure."""

    node_table: list
    edge_table: list
    adjacency_dense: np.ndarray
    adjacency_thresholded: np.ndarray
    assignment_matrices: list
    topological_order: list
    prior_adjacency: Optional[np.ndarray]
    is_acyclic: bool
    cycle_score: float
    series_table: Optional[list] = None
    prototype_table: Optional[list] = None
    affinity_table: Optional[list] = None
    prototype_edge_table: Optional[list] = None

    def to_dict(self) -> dict:
        return {
            'node_table': list(self.node_table),
            'edge_table': list(self.edge_table),
            'adjacency_dense': np.asarray(self.adjacency_dense, dtype=np.float32),
            'adjacency_thresholded': np.asarray(
                self.adjacency_thresholded, dtype=np.float32
            ),
            'assignment_matrices': [
                np.asarray(matrix, dtype=np.float32)
                for matrix in self.assignment_matrices
            ],
            'topological_order': list(self.topological_order),
            'prior_adjacency': (
                None
                if self.prior_adjacency is None
                else np.asarray(self.prior_adjacency, dtype=np.float32)
            ),
            'is_acyclic': bool(self.is_acyclic),
            'cycle_score': float(self.cycle_score),
            'series_table': list(self.series_table or []),
            'prototype_table': list(self.prototype_table or []),
            'affinity_table': list(self.affinity_table or []),
            'prototype_edge_table': list(self.prototype_edge_table or []),
        }


def _normalize_node_color(color_value, fallback: str = '#4c78a8') -> str:
    if color_value is None:
        return fallback
    return str(color_value)


def _series_metadata_lookup(series_metadata: Optional[list]) -> dict:
    lookup = {}
    for metadata in series_metadata or []:
        name = getattr(metadata, 'name', None)
        if name is not None:
            lookup[name] = metadata
    return lookup


def _metadata_color_mapping(series_table: list, color_by: str = 'auto') -> tuple:
    series_table = list(series_table or [])
    if not series_table:
        return {}, None

    available_keys = []
    for series_row in series_table:
        metadata = dict(series_row.get('metadata') or {})
        for key, value in metadata.items():
            if value not in (None, ''):
                available_keys.append(key)
    available_keys = sorted(set(available_keys))

    selected_key = None
    if color_by == 'auto':
        for key in available_keys:
            distinct_values = {
                row.get('metadata', {}).get(key)
                for row in series_table
                if row.get('metadata', {}).get(key) not in (None, '')
            }
            if 1 < len(distinct_values) <= 10:
                selected_key = key
                break
    elif color_by in available_keys:
        selected_key = color_by

    if selected_key is None:
        return {}, None

    palette = [
        '#4c78a8',
        '#f58518',
        '#54a24b',
        '#e45756',
        '#72b7b2',
        '#b279a2',
        '#ff9da6',
        '#9d755d',
        '#bab0ab',
        '#2f4b7c',
    ]
    categories = sorted(
        {
            row.get('metadata', {}).get(selected_key)
            for row in series_table
            if row.get('metadata', {}).get(selected_key) not in (None, '')
        }
    )
    color_map = {
        category: palette[idx % len(palette)] for idx, category in enumerate(categories)
    }
    return color_map, selected_key


def _build_series_and_prototype_tables(
    full_series_names: Optional[list] = None,
    anchor_mask: np.ndarray = None,
    series_metadata: Optional[list] = None,
    prototype_weights: np.ndarray = None,
    prototype_forecasts: np.ndarray = None,
) -> tuple:
    series_names = list(full_series_names or [])
    metadata_lookup = _series_metadata_lookup(series_metadata)
    if anchor_mask is None:
        resolved_anchor_mask = np.ones(len(series_names), dtype=bool)
    else:
        resolved_anchor_mask = np.asarray(anchor_mask, dtype=bool).reshape(-1)
        if resolved_anchor_mask.shape[0] != len(series_names):
            resolved_anchor_mask = np.ones(len(series_names), dtype=bool)

    affinity = None
    if prototype_weights is not None:
        affinity = np.asarray(prototype_weights, dtype=np.float32)
        if affinity.ndim != 2 or affinity.shape[0] != len(series_names):
            affinity = None

    prototype_table = []
    if affinity is not None:
        n_prototypes = int(affinity.shape[1])
    elif prototype_forecasts is not None:
        prototype_forecasts = np.asarray(prototype_forecasts, dtype=np.float32)
        n_prototypes = (
            int(prototype_forecasts.shape[0]) if prototype_forecasts.ndim == 2 else 0
        )
    else:
        n_prototypes = 0

    forecast_values = None
    if prototype_forecasts is not None:
        forecast_values = np.asarray(prototype_forecasts, dtype=np.float32)
        if forecast_values.ndim != 2 or forecast_values.shape[0] != n_prototypes:
            forecast_values = None

    for prototype_index in range(n_prototypes):
        prototype_row = {
            'node_id': f'prototype_{prototype_index}',
            'label': f'prototype_{prototype_index + 1}',
            'index': int(prototype_index),
        }
        if forecast_values is not None:
            prototype_row['sparkline'] = forecast_values[prototype_index].tolist()
        prototype_table.append(prototype_row)

    series_table = []
    affinity_table = []
    for idx, name in enumerate(series_names):
        metadata = metadata_lookup.get(name)
        metadata_values = dict(getattr(metadata, 'attribute_values', {}) or {})
        history_periods = int(getattr(metadata, 'history_periods', 0) or 0)
        dominant_prototype = None
        dominant_weight = None
        if affinity is not None and affinity.shape[1] > 0:
            dominant_prototype = int(np.argmax(affinity[idx]))
            dominant_weight = float(affinity[idx, dominant_prototype])
            for prototype_index in range(affinity.shape[1]):
                affinity_table.append(
                    {
                        'source': name,
                        'target': f'prototype_{prototype_index}',
                        'weight': float(affinity[idx, prototype_index]),
                        'edge_type': 'affinity',
                        'is_dominant': bool(prototype_index == dominant_prototype),
                    }
                )

        series_table.append(
            {
                'node_id': name,
                'label': name,
                'index': int(idx),
                'kind': 'anchor' if resolved_anchor_mask[idx] else 'responder',
                'history_periods': history_periods,
                'metadata': metadata_values,
                'dominant_prototype': dominant_prototype,
                'dominant_weight': dominant_weight,
            }
        )

    return series_table, prototype_table, affinity_table


def _infer_prototype_graph(
    adjacency_dense: np.ndarray = None,
    global_prototype_weights: np.ndarray = None,
    decoded_top_trends: np.ndarray = None,
    threshold: float = 0.2,
) -> tuple:
    """Project the driver-layer graph into prototype space for interpretation.

    This is an inferred prototype graph, not a directly learned object.
    """
    if global_prototype_weights is None:
        return None, []

    weights = np.asarray(global_prototype_weights, dtype=np.float32)
    if weights.ndim != 2:
        return None, []

    row_sums = weights.sum(axis=1, keepdims=True).clip(min=1e-6)
    weights = weights / row_sums
    n_drivers, n_prototypes = weights.shape

    prototype_trends = None
    if decoded_top_trends is not None:
        decoded = np.asarray(decoded_top_trends, dtype=np.float32)
        if decoded.ndim == 2 and decoded.shape[0] == n_drivers:
            prototype_mass = weights.sum(axis=0, keepdims=True).clip(min=1e-6)
            prototype_trends = (weights.T @ decoded) / prototype_mass.T

    prototype_edge_table = []
    adjacency = np.asarray(adjacency_dense, dtype=np.float32)
    if (
        adjacency.ndim == 2
        and adjacency.shape[0] == n_drivers
        and adjacency.shape[1] == n_drivers
    ):
        prototype_mass = weights.sum(axis=0, keepdims=True)
        projected = weights.T @ adjacency @ weights
        normalization = np.maximum(prototype_mass.T @ prototype_mass, 1e-6)
        projected = np.clip(projected / normalization, 0.0, 1.0)
        np.fill_diagonal(projected, 0.0)
        for source in range(n_prototypes):
            for target in range(n_prototypes):
                weight = float(projected[source, target])
                if source == target or weight < float(threshold):
                    continue
                prototype_edge_table.append(
                    {
                        'source': f'prototype_{source}',
                        'target': f'prototype_{target}',
                        'weight': weight,
                        'edge_type': 'prototype_dag',
                        'is_inferred': True,
                    }
                )

    return prototype_trends, prototype_edge_table


def _safe_cycle_score_numpy(adjacency: np.ndarray) -> float:
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        return 0.0
    if adjacency.size == 0:
        return 0.0
    squared = adjacency * adjacency
    try:
        eigenvalues = np.linalg.eigvals(squared)
        return float(np.real(np.exp(eigenvalues).sum() - adjacency.shape[0]))
    except Exception:
        return float(np.maximum(np.trace(squared), 0.0))


def _threshold_adjacency(adjacency: np.ndarray, threshold: float) -> np.ndarray:
    dense = np.asarray(adjacency, dtype=np.float32).copy()
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        return np.zeros((0, 0), dtype=np.float32)
    np.fill_diagonal(dense, 0.0)
    thresholded = (dense >= float(threshold)).astype(np.float32)
    np.fill_diagonal(thresholded, 0.0)
    return thresholded


def topological_order_from_adjacency(adjacency: np.ndarray) -> list:
    """Return a node ordering if the graph is acyclic, otherwise []."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        return []
    n_nodes = adjacency.shape[0]
    indegree = adjacency.sum(axis=0).astype(int).tolist()
    queue = deque(idx for idx in range(n_nodes) if indegree[idx] == 0)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        outgoing = np.where(adjacency[node] > 0)[0]
        for child in outgoing:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))

    return order if len(order) == n_nodes else []


def build_graph_snapshot(
    adjacency_dense: np.ndarray,
    assignment_matrices: Optional[list] = None,
    threshold: float = 0.2,
    prior_adjacency: np.ndarray = None,
    anchor_names: Optional[list] = None,
    full_series_names: Optional[list] = None,
    anchor_mask: np.ndarray = None,
    series_metadata: Optional[list] = None,
    prototype_weights: np.ndarray = None,
    prototype_forecasts: np.ndarray = None,
    global_prototype_weights: np.ndarray = None,
    decoded_top_trends: np.ndarray = None,
) -> GraphSnapshot:
    """Create a serializable snapshot from TVA structure state."""
    dense = np.asarray(adjacency_dense, dtype=np.float32)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        dense = np.zeros((0, 0), dtype=np.float32)
    np.fill_diagonal(dense, 0.0)

    assignment_arrays = [
        np.asarray(matrix, dtype=np.float32) for matrix in (assignment_matrices or [])
    ]
    thresholded = _threshold_adjacency(dense, threshold=threshold)
    topological_order = topological_order_from_adjacency(thresholded)
    is_acyclic = len(topological_order) == thresholded.shape[0]
    cycle_score = _safe_cycle_score_numpy(dense)

    level_sizes = []
    if assignment_arrays:
        level_sizes = [assignment_arrays[0].shape[0]]
        level_sizes.extend([matrix.shape[1] for matrix in assignment_arrays])
    else:
        level_sizes = [dense.shape[0]]
    level_offsets = np.cumsum([0] + level_sizes).tolist()

    node_table = []
    edge_table = []
    x_positions = np.linspace(0.1, 0.9, max(len(level_sizes), 1))

    for level_index, level_size in enumerate(level_sizes):
        y_positions = np.linspace(0.9, 0.1, max(level_size, 1))
        for node_index in range(level_size):
            if level_index == 0:
                label = (
                    anchor_names[node_index]
                    if anchor_names is not None and node_index < len(anchor_names)
                    else f'anchor_{node_index}'
                )
                kind = 'anchor'
            elif level_index == len(level_sizes) - 1:
                label = f'driver_{node_index + 1}'
                if global_prototype_weights is not None:
                    driver_weights = np.asarray(
                        global_prototype_weights, dtype=np.float32
                    )
                    if (
                        driver_weights.ndim == 2
                        and node_index < driver_weights.shape[0]
                        and driver_weights.shape[1] > 0
                    ):
                        dominant_prototype = (
                            int(np.argmax(driver_weights[node_index])) + 1
                        )
                        label = f'{label} [P{dominant_prototype}]'
                kind = 'driver'
            else:
                label = f'latent_{level_index}_{node_index}'
                kind = 'latent'
            node_table.append(
                {
                    'node_id': label,
                    'level': int(level_index),
                    'index': int(node_index),
                    'kind': kind,
                    'x': float(x_positions[level_index]),
                    'y': float(y_positions[node_index]),
                }
            )

    for level_index, matrix in enumerate(assignment_arrays):
        lower_size, upper_size = matrix.shape
        for lower_idx in range(lower_size):
            for upper_idx in range(upper_size):
                weight = float(matrix[lower_idx, upper_idx])
                if weight < threshold:
                    continue
                edge_table.append(
                    {
                        'source': node_table[level_offsets[level_index] + lower_idx][
                            'node_id'
                        ],
                        'target': node_table[
                            level_offsets[level_index + 1] + upper_idx
                        ]['node_id'],
                        'weight': weight,
                        'edge_type': 'hierarchy',
                    }
                )

    dag_level_offset = level_offsets[-2] if len(level_offsets) >= 2 else 0
    for source in range(dense.shape[0]):
        for target in range(dense.shape[1]):
            weight = float(dense[source, target])
            if source == target or weight < threshold:
                continue
            edge_table.append(
                {
                    'source': (
                        node_table[dag_level_offset + source]['node_id']
                        if dag_level_offset + source < len(node_table)
                        else f'top_{source}'
                    ),
                    'target': (
                        node_table[dag_level_offset + target]['node_id']
                        if dag_level_offset + target < len(node_table)
                        else f'top_{target}'
                    ),
                    'weight': weight,
                    'edge_type': 'dag',
                }
            )

    inferred_prototype_trends, prototype_edge_table = _infer_prototype_graph(
        adjacency_dense=dense,
        global_prototype_weights=global_prototype_weights,
        decoded_top_trends=decoded_top_trends,
        threshold=threshold,
    )
    if inferred_prototype_trends is not None:
        prototype_forecasts = inferred_prototype_trends

    series_table, prototype_table, affinity_table = _build_series_and_prototype_tables(
        full_series_names=full_series_names,
        anchor_mask=anchor_mask,
        series_metadata=series_metadata,
        prototype_weights=prototype_weights,
        prototype_forecasts=prototype_forecasts,
    )

    return GraphSnapshot(
        node_table=node_table,
        edge_table=edge_table,
        adjacency_dense=dense,
        adjacency_thresholded=thresholded,
        assignment_matrices=assignment_arrays,
        topological_order=topological_order,
        prior_adjacency=(
            None
            if prior_adjacency is None
            else np.asarray(prior_adjacency, dtype=np.float32)
        ),
        is_acyclic=is_acyclic,
        cycle_score=cycle_score,
        series_table=series_table,
        prototype_table=prototype_table,
        affinity_table=affinity_table,
        prototype_edge_table=prototype_edge_table,
    )


def plot_graph_snapshot(
    snapshot: GraphSnapshot,
    view: str = 'dag',
    max_edges: int = 50,
    show_priors: bool = False,
    ax=None,
    metadata_color_by: str = 'auto',
):
    """Render a TVA structure snapshot with matplotlib."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for TVA graph plotting")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.clear()
    view = str(view).lower()
    if view == 'heatmap':
        ax.imshow(
            snapshot.adjacency_dense, cmap='Blues', aspect='auto', vmin=0.0, vmax=1.0
        )
        ax.set_title("TVA Learned Adjacency")
        ax.set_xlabel("Target node")
        ax.set_ylabel("Source node")
        return ax
    if view == 'overview':
        return _plot_overview_snapshot(
            snapshot=snapshot,
            max_edges=max_edges,
            ax=ax,
            metadata_color_by=metadata_color_by,
        )

    nodes_by_id = {node['node_id']: node for node in snapshot.node_table}
    if view == 'hierarchy':
        candidate_edges = [
            edge for edge in snapshot.edge_table if edge['edge_type'] == 'hierarchy'
        ]
        title = "TVA Learned Hierarchy"
    else:
        candidate_edges = [
            edge for edge in snapshot.edge_table if edge['edge_type'] == 'dag'
        ]
        title = "TVA Learned DAG"
        if show_priors and snapshot.prior_adjacency is not None:
            prior = np.asarray(snapshot.prior_adjacency, dtype=np.float32)
            offset = (
                max(node['level'] for node in snapshot.node_table)
                if snapshot.node_table
                else 0
            )
            top_nodes = [
                node for node in snapshot.node_table if node['level'] == offset
            ]
            for source in range(min(prior.shape[0], len(top_nodes))):
                for target in range(min(prior.shape[1], len(top_nodes))):
                    if source == target or prior[source, target] <= 0:
                        continue
                    left = top_nodes[source]
                    right = top_nodes[target]
                    _draw_arrow(
                        ax,
                        left['x'],
                        left['y'],
                        right['x'],
                        right['y'],
                        color='lightgray',
                        linewidth=1.0,
                        alpha=float(min(prior[source, target], 0.8)),
                        linestyle='--',
                        mutation_scale=10,
                        zorder=1,
                    )

    candidate_edges = sorted(candidate_edges, key=lambda x: x['weight'], reverse=True)[
        :max_edges
    ]
    for edge in candidate_edges:
        source = nodes_by_id.get(edge['source'])
        target = nodes_by_id.get(edge['target'])
        if source is None or target is None:
            continue
        line_color = '#1f77b4' if edge['edge_type'] == 'dag' else '#2ca02c'
        if edge['edge_type'] == 'dag' and snapshot.prior_adjacency is not None:
            prior = np.asarray(snapshot.prior_adjacency, dtype=np.float32)
            source_index = int(source.get('index', -1))
            target_index = int(target.get('index', -1))
            forward_prior = 0.0
            reverse_prior = 0.0
            if (
                0 <= source_index < prior.shape[0]
                and 0 <= target_index < prior.shape[1]
            ):
                forward_prior = float(prior[source_index, target_index])
            if (
                0 <= target_index < prior.shape[0]
                and 0 <= source_index < prior.shape[1]
            ):
                reverse_prior = float(prior[target_index, source_index])
            if reverse_prior > 0 and forward_prior <= 0:
                line_color = '#d62728'
            elif forward_prior <= 0:
                line_color = '#ff7f0e'
        _draw_arrow(
            ax,
            source['x'],
            source['y'],
            target['x'],
            target['y'],
            linewidth=1.0 + (2.5 * edge['weight']),
            color=line_color,
            alpha=0.3 + (0.6 * edge['weight']),
            mutation_scale=12,
            zorder=2,
        )

    for node in snapshot.node_table:
        ax.scatter(node['x'], node['y'], s=70, c='#111111', zorder=3)
        ax.text(
            node['x'] + 0.01,
            node['y'],
            node['node_id'],
            fontsize=8,
            va='center',
            ha='left',
            zorder=4,
        )

    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')
    return ax


def _draw_arrow(
    ax,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    color: str,
    linewidth: float,
    alpha: float,
    linestyle: str = '-',
    mutation_scale: float = 12,
    zorder: int = 2,
    connection_rad: float = 0.03,
):
    if FancyArrowPatch is None:
        ax.plot(
            [start_x, end_x],
            [start_y, end_y],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=zorder,
        )
        return

    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle='-|>',
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        shrinkA=10,
        shrinkB=10,
        connectionstyle=f'arc3,rad={float(connection_rad)}',
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _draw_prototype_sparkline(ax, x_center: float, y_center: float, values, width=0.16):
    if values is None:
        return
    sparkline = np.asarray(values, dtype=np.float32).reshape(-1)
    if sparkline.size < 2:
        return
    xmin = x_center - (width / 2.0)
    xmax = x_center + (width / 2.0)
    x_values = np.linspace(xmin, xmax, sparkline.size)
    min_value = float(np.min(sparkline))
    max_value = float(np.max(sparkline))
    span = max(max_value - min_value, 1e-6)
    y_values = y_center - 0.035 + (0.07 * ((sparkline - min_value) / span))
    ax.plot(x_values, y_values, color='#222222', linewidth=1.1, alpha=0.9, zorder=5)


def _plot_overview_snapshot(
    snapshot: GraphSnapshot,
    max_edges: int = 50,
    ax=None,
    metadata_color_by: str = 'auto',
):
    series_table = list(snapshot.series_table or [])
    prototype_table = list(snapshot.prototype_table or [])
    affinity_table = list(snapshot.affinity_table or [])
    prototype_edge_table = list(snapshot.prototype_edge_table or [])

    if not series_table or not prototype_table:
        ax.text(
            0.5,
            0.5,
            "Overview requires series metadata and prototype affinities.",
            ha='center',
            va='center',
            fontsize=10,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis('off')
        return ax

    prototype_positions = {}
    prototype_y_positions = np.linspace(0.85, 0.15, len(prototype_table))
    for row, y_pos in zip(prototype_table, prototype_y_positions):
        prototype_positions[row['node_id']] = (0.76, float(y_pos))

    grouped_series = {row['node_id']: row for row in series_table}
    for row in grouped_series.values():
        if row.get('dominant_prototype') is None:
            row['_cluster_key'] = (10**6, row['label'])
        else:
            row['_cluster_key'] = (int(row['dominant_prototype']), row['label'])
    ordered_series = sorted(
        grouped_series.values(), key=lambda row: row['_cluster_key']
    )

    cluster_counts = {}
    for row in ordered_series:
        prototype_index = row.get('dominant_prototype')
        cluster_counts[prototype_index] = cluster_counts.get(prototype_index, 0) + 1

    cluster_offsets = {}
    for prototype_index, count in cluster_counts.items():
        if prototype_index is None or prototype_index >= len(prototype_y_positions):
            base_y = 0.5
        else:
            base_y = float(prototype_y_positions[prototype_index])
        if count == 1:
            cluster_offsets[prototype_index] = [base_y]
        else:
            spread = min(0.12 + (0.02 * (count - 1)), 0.22)
            cluster_offsets[prototype_index] = np.linspace(
                base_y + spread / 2.0,
                base_y - spread / 2.0,
                count,
            ).tolist()

    color_map, selected_key = _metadata_color_mapping(
        series_table=ordered_series, color_by=metadata_color_by
    )
    cluster_seen = {key: 0 for key in cluster_counts}
    series_positions = {}
    for row in ordered_series:
        prototype_index = row.get('dominant_prototype')
        seen = cluster_seen[prototype_index]
        cluster_seen[prototype_index] += 1
        y_pos = cluster_offsets[prototype_index][seen]
        x_pos = 0.16 if row.get('kind') == 'anchor' else 0.08
        series_positions[row['node_id']] = (x_pos, float(y_pos))

    dominant_affinity_edges = [
        edge for edge in affinity_table if edge.get('is_dominant')
    ]
    dominant_affinity_edges = sorted(
        dominant_affinity_edges,
        key=lambda edge: edge.get('weight', 0.0),
        reverse=True,
    )[:max_edges]

    for edge in dominant_affinity_edges:
        source_pos = series_positions.get(edge['source'])
        target_pos = prototype_positions.get(edge['target'])
        if source_pos is None or target_pos is None:
            continue
        edge_color = '#7f8c8d'
        _draw_arrow(
            ax,
            source_pos[0],
            source_pos[1],
            target_pos[0],
            target_pos[1],
            color=edge_color,
            linewidth=0.8 + (2.6 * float(edge['weight'])),
            alpha=0.25 + (0.55 * float(edge['weight'])),
            mutation_scale=12,
            zorder=1,
        )

    prototype_edge_table = sorted(
        prototype_edge_table,
        key=lambda edge: edge.get('weight', 0.0),
        reverse=True,
    )[:max_edges]
    for edge in prototype_edge_table:
        source_pos = prototype_positions.get(edge['source'])
        target_pos = prototype_positions.get(edge['target'])
        if source_pos is None or target_pos is None:
            continue
        curve = 0.28 if source_pos[1] > target_pos[1] else -0.28
        _draw_arrow(
            ax,
            source_pos[0] + 0.07,
            source_pos[1],
            target_pos[0] + 0.07,
            target_pos[1],
            color='#c44e52',
            linewidth=1.0 + (2.2 * float(edge['weight'])),
            alpha=0.30 + (0.55 * float(edge['weight'])),
            mutation_scale=12,
            zorder=2,
            connection_rad=curve,
        )

    for row in prototype_table:
        x_pos, y_pos = prototype_positions[row['node_id']]
        if FancyBboxPatch is not None:
            ax.add_patch(
                FancyBboxPatch(
                    (x_pos - 0.085, y_pos - 0.055),
                    0.17,
                    0.11,
                    boxstyle='round,pad=0.012,rounding_size=0.02',
                    facecolor='#f7f4ea',
                    edgecolor='#5b5b5b',
                    linewidth=1.0,
                    zorder=3,
                )
            )
        else:
            ax.scatter(x_pos, y_pos, s=420, c='#f7f4ea', edgecolors='#5b5b5b', zorder=3)
        _draw_prototype_sparkline(
            ax,
            x_center=x_pos,
            y_center=y_pos + 0.006,
            values=row.get('sparkline'),
        )
        ax.text(
            x_pos,
            y_pos - 0.062,
            row.get('label', row['node_id']),
            fontsize=8,
            ha='center',
            va='top',
            zorder=5,
        )

    for row in ordered_series:
        x_pos, y_pos = series_positions[row['node_id']]
        metadata_value = (
            row.get('metadata', {}).get(selected_key) if selected_key else None
        )
        fill_color = color_map.get(metadata_value, '#4c78a8')
        marker = 'o' if row.get('kind') == 'anchor' else 's'
        size = (
            80 + min(max(float(row.get('history_periods', 0) or 0), 0.0), 365.0) * 0.35
        )
        ax.scatter(
            x_pos,
            y_pos,
            s=size,
            c=_normalize_node_color(fill_color),
            marker=marker,
            edgecolors='#111111',
            linewidths=0.6,
            zorder=4,
        )
        ax.text(
            x_pos + 0.018,
            y_pos,
            row.get('label', row['node_id']),
            fontsize=8,
            ha='left',
            va='center',
            zorder=5,
        )

    legend_handles = []
    if Line2D is not None:
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label='anchor',
                    markerfacecolor='#888888',
                    markeredgecolor='#111111',
                    markersize=7,
                ),
                Line2D(
                    [0],
                    [0],
                    marker='s',
                    color='w',
                    label='responder',
                    markerfacecolor='#888888',
                    markeredgecolor='#111111',
                    markersize=7,
                ),
            ]
        )
    if Patch is not None and selected_key is not None and color_map:
        legend_handles.extend(
            [
                Patch(facecolor=color, edgecolor='none', label=str(category))
                for category, color in color_map.items()
            ]
        )
    if legend_handles:
        legend_title = "Series role"
        if selected_key is not None and color_map:
            legend_title = f"Role and {selected_key}"
        ax.legend(
            handles=legend_handles,
            title=legend_title,
            loc='lower left',
            bbox_to_anchor=(0.01, 0.01),
            frameon=False,
            fontsize=8,
            title_fontsize=8,
        )

    ax.text(0.08, 0.95, "Series", fontsize=10, ha='left', va='center')
    ax.text(0.76, 0.95, "Shared Prototypes", fontsize=10, ha='center', va='center')
    ax.text(
        0.02,
        0.98,
        "Prototype arrows are inferred from driver adjacency, not directly learned.",
        fontsize=8,
        ha='left',
        va='top',
        color='#444444',
    )
    ax.set_title("TVA Series-Prototype Overview")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')
    return ax


if HAS_TORCH:

    class DirectedGraphLearner(nn.Module):
        """Learn a dense directed adjacency matrix over the top latent layer."""

        def __init__(self, n_nodes: int, prior_adjacency: np.ndarray = None):
            super().__init__()
            self.n_nodes = int(max(n_nodes, 1))
            if prior_adjacency is None:
                init = np.full((self.n_nodes, self.n_nodes), 0.5, dtype=np.float32)
                init = self._break_symmetry(init)
            else:
                init = np.asarray(prior_adjacency, dtype=np.float32)
                if init.shape != (self.n_nodes, self.n_nodes):
                    init = np.full((self.n_nodes, self.n_nodes), 0.5, dtype=np.float32)
                if np.allclose(init, init.T, atol=1e-6):
                    init = self._break_symmetry(init)
            np.fill_diagonal(init, 0.0)
            init = np.clip(init, 1e-4, 1 - 1e-4)
            self.edge_logits = nn.Parameter(
                torch.logit(torch.tensor(init, dtype=torch.float32))
            )

        @staticmethod
        def _break_symmetry(
            init: np.ndarray, off_diagonal_bias: float = 0.075
        ) -> np.ndarray:
            """Inject a small deterministic direction bias into symmetric priors."""
            init = np.asarray(init, dtype=np.float32).copy()
            if init.ndim != 2 or init.shape[0] != init.shape[1]:
                return init
            upper = np.triu(np.ones_like(init, dtype=np.float32), k=1)
            lower = np.tril(np.ones_like(init, dtype=np.float32), k=-1)
            init = init + off_diagonal_bias * upper - off_diagonal_bias * lower
            np.fill_diagonal(init, 0.0)
            return np.clip(init, 1e-4, 1 - 1e-4)

        @property
        def adjacency(self) -> torch.Tensor:
            logits = torch.clamp(self.edge_logits, min=-8.0, max=8.0)
            adjacency = torch.sigmoid(logits)
            adjacency = adjacency * (
                1.0 - torch.eye(self.n_nodes, device=adjacency.device)
            )
            return adjacency

        def attention_mask(self) -> torch.Tensor:
            return -10.0 * (1.0 - self.adjacency)

    class DynamicHierarchyLearner(nn.Module):
        """Learn a bounded latent hierarchy with soft parent-child assignments."""

        def __init__(
            self, n_anchor: int, d_token: int, config: StructureLearningConfig
        ):
            super().__init__()
            self.n_anchor = int(max(n_anchor, 1))
            self.d_token = int(d_token)
            self.config = config
            self.level_sizes = config.derive_latent_sizes(self.n_anchor)
            if not self.level_sizes:
                self.level_sizes = [max(int(np.ceil(self.n_anchor * 0.5)), 1)]

            layer_sizes = [self.n_anchor] + self.level_sizes
            self.assignment_logits = nn.ParameterList()
            self.encoder_norms = nn.ModuleList()
            self.decoder_norms = nn.ModuleList()
            # Keep the backbone static while allowing a small token-conditioned
            # assignment adaptation used for diagnostics and optional drift loss.
            self.dynamic_assignment_mix = 0.2

            for lower_size, upper_size in zip(layer_sizes[:-1], layer_sizes[1:]):
                logits = torch.zeros(lower_size, upper_size, dtype=torch.float32)
                self.assignment_logits.append(nn.Parameter(logits))
                self.encoder_norms.append(nn.LayerNorm(d_token))
                self.decoder_norms.append(nn.LayerNorm(d_token))

        def assignment_matrices(self) -> list:
            matrices = []
            for logits in self.assignment_logits:
                clamped = torch.clamp(logits, min=-8.0, max=8.0)
                matrices.append(F.softmax(clamped, dim=-1))
            return matrices

        def _batch_conditioned_assignments(
            self,
            lower_tokens: torch.Tensor,
            base_assignment: torch.Tensor,
        ) -> torch.Tensor:
            """Return per-batch assignments blended with the static backbone."""
            batch_size = lower_tokens.shape[0]
            base_expanded = base_assignment.unsqueeze(0).expand(batch_size, -1, -1)
            if batch_size <= 1:
                return base_expanded

            base_column_norm = base_assignment.sum(dim=0, keepdim=True).clamp(min=1e-6)
            seed_upper = torch.einsum(
                'bld,lu->bud',
                lower_tokens,
                base_assignment / base_column_norm,
            )
            lower_unit = F.normalize(lower_tokens, p=2, dim=-1)
            upper_unit = F.normalize(seed_upper, p=2, dim=-1)
            logits = torch.einsum('bld,bud->blu', lower_unit, upper_unit)
            adaptive = F.softmax(logits, dim=-1)

            mixed = ((1.0 - self.dynamic_assignment_mix) * base_expanded) + (
                self.dynamic_assignment_mix * adaptive
            )
            return mixed / mixed.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        def forward(self, anchor_tokens: torch.Tensor) -> dict:
            levels = [anchor_tokens]
            assignments = self.assignment_matrices()
            dynamic_assignments = []
            drift_total = torch.tensor(0.0, device=anchor_tokens.device)
            current = anchor_tokens
            for assignment, norm in zip(assignments, self.encoder_norms):
                batch_assignment = self._batch_conditioned_assignments(
                    current, assignment
                )
                dynamic_assignments.append(batch_assignment)
                drift_total = (
                    drift_total
                    + ((batch_assignment - assignment.unsqueeze(0)) ** 2).mean()
                )

                column_norm = batch_assignment.sum(dim=1, keepdim=True).clamp(min=1e-6)
                pooled = torch.einsum(
                    'bld,blu->bud',
                    current,
                    batch_assignment / column_norm,
                )
                current = norm(pooled)
                levels.append(current)
            return {
                'levels': levels,
                'assignment_matrices': assignments,
                'dynamic_assignment_matrices': dynamic_assignments,
                'top_latent': levels[-1],
                'assignment_drift': drift_total / max(len(assignments), 1),
            }

        def decode(
            self, top_latent: torch.Tensor, skip_levels: list, assignment_matrices=None
        ) -> list:
            assignments = assignment_matrices or self.assignment_matrices()
            current = top_latent
            decoded_levels = [None] * len(skip_levels)
            decoded_levels[-1] = current
            for idx in reversed(range(len(assignments))):
                assignment = assignments[idx]
                if assignment.ndim == 3:
                    upsampled = torch.einsum('blu,bud->bld', assignment, current)
                else:
                    upsampled = torch.einsum('lu,bud->bld', assignment, current)
                current = self.decoder_norms[idx](upsampled + skip_levels[idx])
                decoded_levels[idx] = current
            return decoded_levels

    def dag_cycle_penalty(adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            return torch.tensor(0.0, device=adjacency.device)
        squared = adjacency * adjacency
        return torch.trace(torch.matrix_exp(squared)) - adjacency.shape[0]

    def adjacency_sparsity_penalty(adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.numel() == 0:
            return torch.tensor(0.0, device=adjacency.device)
        mask = 1.0 - torch.eye(adjacency.shape[0], device=adjacency.device)
        return (adjacency * mask).mean()

    def assignment_entropy_penalty(assignment_matrices: list) -> torch.Tensor:
        if not assignment_matrices:
            return torch.tensor(0.0)
        device = assignment_matrices[0].device
        penalty = torch.tensor(0.0, device=device)
        for matrix in assignment_matrices:
            entropy = -(matrix * torch.log(matrix.clamp(min=1e-8))).sum(dim=-1).mean()
            penalty = penalty + entropy
        return penalty / max(len(assignment_matrices), 1)

    def assignment_full_rank_penalty(assignment_matrices: list) -> torch.Tensor:
        if not assignment_matrices:
            return torch.tensor(0.0)
        device = assignment_matrices[0].device
        penalty = torch.tensor(0.0, device=device)
        used = 0
        for matrix in assignment_matrices:
            if matrix.shape[1] <= 1:
                continue
            normalized = matrix / matrix.norm(dim=0, keepdim=True).clamp(min=1e-6)
            gram = normalized.transpose(0, 1) @ normalized
            identity = torch.eye(gram.shape[0], device=device)
            penalty = penalty + F.mse_loss(gram, identity)
            used += 1
        if used == 0:
            return torch.tensor(0.0, device=device)
        return penalty / used
