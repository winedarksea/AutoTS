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

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
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
        }


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
                        'source': node_table[
                            level_offsets[level_index] + lower_idx
                        ]['node_id'],
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
    )


def plot_graph_snapshot(
    snapshot: GraphSnapshot,
    view: str = 'dag',
    max_edges: int = 50,
    show_priors: bool = False,
    ax=None,
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
                    ax.plot(
                        [left['x'], right['x']],
                        [left['y'], right['y']],
                        linestyle='--',
                        linewidth=1.0,
                        color='lightgray',
                        alpha=float(min(prior[source, target], 1.0)),
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
        ax.plot(
            [source['x'], target['x']],
            [source['y'], target['y']],
            linewidth=1.0 + (2.5 * edge['weight']),
            color='#1f77b4' if edge['edge_type'] == 'dag' else '#2ca02c',
            alpha=0.3 + (0.6 * edge['weight']),
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


if HAS_TORCH:

    class DirectedGraphLearner(nn.Module):
        """Learn a dense directed adjacency matrix over the top latent layer."""

        def __init__(self, n_nodes: int, prior_adjacency: np.ndarray = None):
            super().__init__()
            self.n_nodes = int(max(n_nodes, 1))
            if prior_adjacency is None:
                init = np.full((self.n_nodes, self.n_nodes), 0.5, dtype=np.float32)
            else:
                init = np.asarray(prior_adjacency, dtype=np.float32)
                if init.shape != (self.n_nodes, self.n_nodes):
                    init = np.full((self.n_nodes, self.n_nodes), 0.5, dtype=np.float32)
            np.fill_diagonal(init, 0.0)
            init = np.clip(init, 1e-4, 1 - 1e-4)
            self.edge_logits = nn.Parameter(
                torch.logit(torch.tensor(init, dtype=torch.float32))
            )

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
