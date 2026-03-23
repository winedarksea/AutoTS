# -*- coding: utf-8 -*-
"""
YggdrasilPriors — Prior specification and encoding for the TVA graph.

Yggdrasil is the world tree connecting all nine realms in Norse mythology.
Here it connects time series through shared metadata and business priors.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SeriesMetadata:
    """Metadata descriptor for a single time series.

    Attributes:
        name: Series identifier matching DataFrame column name.
        metric_type: Category of metric (e.g. 'interface_time', 'viewport_views', 'dau').
        surface: Product surface (e.g. 'marketplace', 'videos', 'mobile_feed').
        geography: Geographic region (e.g. 'US', 'UK', 'DE').
        hierarchy_path: Ordered path from root to leaf (e.g. ['global', 'NA', 'US']).
        history_periods: Number of observed time periods available.
    """
    name: str
    metric_type: Optional[str] = None
    surface: Optional[str] = None
    geography: Optional[str] = None
    hierarchy_path: Optional[list] = field(default_factory=list)
    history_periods: int = 0


class YggdrasilPriors:
    """Constructs prior adjacency matrices, metadata embeddings, and hierarchy
    matrices from series metadata. All methods return sensible defaults when
    metadata is absent, so TVA runs with or without priors.

    Args:
        series_metadata: List of SeriesMetadata, one per series.
        relationship_matrix: Optional (N, N) soft prior adjacency override.
        prior_confidence: Weight of prior vs learned structure (0=ignore, 1=rigid).
    """

    def __init__(
        self,
        series_metadata: list = None,
        relationship_matrix: np.ndarray = None,
        prior_confidence: float = 0.3,
    ):
        self.series_metadata = series_metadata or []
        self.relationship_matrix = relationship_matrix
        self.prior_confidence = prior_confidence
        self._series_names = [m.name for m in self.series_metadata] if self.series_metadata else []

    @property
    def n_series(self):
        return len(self._series_names)

    def build_prior_adjacency(self) -> np.ndarray:
        """Construct a soft prior adjacency matrix from shared attributes.

        Series sharing metric_type, surface, or geography get non-zero edges.
        Each shared attribute contributes equally. Returns (N, N) in [0, 1].
        Falls back to uniform matrix if no metadata is provided.
        """
        if self.relationship_matrix is not None:
            return self.relationship_matrix.copy()

        n = self.n_series
        if n == 0:
            return np.ones((1, 1), dtype=np.float32)

        adj = np.zeros((n, n), dtype=np.float32)
        _attributes = ['metric_type', 'surface', 'geography']
        n_attrs = len(_attributes)

        for attr in _attributes:
            values = [getattr(m, attr, None) for m in self.series_metadata]
            for i in range(n):
                for j in range(i + 1, n):
                    if values[i] is not None and values[j] is not None and values[i] == values[j]:
                        adj[i, j] += 1.0 / n_attrs
                        adj[j, i] += 1.0 / n_attrs

        # self-connections
        np.fill_diagonal(adj, 1.0)

        # if no metadata produced any edges, fall back to uniform
        if adj.max() <= 1.0 and np.count_nonzero(adj - np.eye(n)) == 0:
            adj = np.full((n, n), 1.0 / n, dtype=np.float32)
            np.fill_diagonal(adj, 1.0)

        return adj

    def build_metadata_embeddings(self) -> np.ndarray:
        """Build (N, D_meta) one-hot embedding matrix from categorical metadata.

        Encodes metric_type, surface, and geography as concatenated one-hot vectors.
        Returns zeros if no metadata is available.
        """
        if not self.series_metadata:
            return np.zeros((1, 1), dtype=np.float32)

        n = self.n_series
        _attributes = ['metric_type', 'surface', 'geography']
        vocabs = {}
        for attr in _attributes:
            unique_vals = sorted(set(
                getattr(m, attr) for m in self.series_metadata
                if getattr(m, attr, None) is not None
            ))
            vocabs[attr] = {v: i for i, v in enumerate(unique_vals)}

        d_meta = sum(len(v) for v in vocabs.values())
        if d_meta == 0:
            return np.zeros((n, 1), dtype=np.float32)

        embeddings = np.zeros((n, d_meta), dtype=np.float32)
        offset = 0
        for attr in _attributes:
            vocab = vocabs[attr]
            for i, m in enumerate(self.series_metadata):
                val = getattr(m, attr, None)
                if val is not None and val in vocab:
                    embeddings[i, offset + vocab[val]] = 1.0
            offset += len(vocab)

        return embeddings

    def get_anchor_mask(self, min_history: int) -> np.ndarray:
        """Return boolean mask (N,) where True = series has enough history to be an anchor.

        Anchors shape the core latent trend graph. Responders inherit composite
        trends but do not perturb the core geometry.
        """
        if not self.series_metadata:
            return np.ones(1, dtype=bool)
        return np.array(
            [m.history_periods >= min_history for m in self.series_metadata],
            dtype=bool,
        )

    def build_hierarchy_matrix(self) -> np.ndarray:
        """Build summing matrix S for hierarchical reconciliation.

        S has shape (L, M) where L = total nodes (aggregates + bottom),
        M = number of bottom-level series. Compatible with mint_reconcile(S, y_all, W).

        Returns identity if no hierarchy paths are specified.
        """
        if not self.series_metadata:
            return np.eye(1, dtype=np.float32)

        paths = [m.hierarchy_path for m in self.series_metadata if m.hierarchy_path]
        if not paths:
            n = self.n_series
            return np.eye(n, dtype=np.float32)

        # collect all unique aggregate nodes (non-leaf path prefixes)
        _aggregate_nodes = set()
        for path in paths:
            for depth in range(1, len(path)):
                _aggregate_nodes.add(tuple(path[:depth]))
        _aggregate_nodes = sorted(_aggregate_nodes, key=lambda x: (len(x), x))

        n_bottom = self.n_series
        n_agg = len(_aggregate_nodes)
        n_total = n_agg + n_bottom

        S = np.zeros((n_total, n_bottom), dtype=np.float32)

        # bottom-level identity block
        S[n_agg:, :] = np.eye(n_bottom, dtype=np.float32)

        # aggregate rows: sum over bottom series whose path starts with the aggregate prefix
        _agg_index = {node: i for i, node in enumerate(_aggregate_nodes)}
        for j, m in enumerate(self.series_metadata):
            if not m.hierarchy_path:
                continue
            path = m.hierarchy_path
            for depth in range(1, len(path)):
                prefix = tuple(path[:depth])
                if prefix in _agg_index:
                    S[_agg_index[prefix], j] = 1.0

        return S

    def get_series_names(self) -> list:
        """Return ordered list of series names."""
        return list(self._series_names)

    def _branches_of_yggdrasil(self):
        """Hidden: count the branches (edges) in the prior graph."""
        adj = self.build_prior_adjacency()
        return int(np.count_nonzero(adj - np.diag(np.diag(adj))) // 2)
