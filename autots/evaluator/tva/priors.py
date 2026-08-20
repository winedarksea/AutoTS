# -*- coding: utf-8 -*-
"""
YggdrasilPriors — Prior specification and encoding for the TVA graph.

Acts as the 'world tree' for the forecasting graph, connecting
related time series across different domains through shared metadata.
"""

import warnings

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(init=False)
class SeriesMetadata:
    """Metadata descriptor for a single time series.

    Attributes:
        name: Series identifier matching DataFrame column name.
        attribute_values: Arbitrary categorical metadata for the series.
        hierarchy_path: Ordered path from root to leaf (e.g. ['global', 'NA', 'US']).
        history_periods: Number of observed time periods available.
    """

    name: str
    attribute_values: Dict[str, str] = field(default_factory=dict)
    attribute_weights: Dict[str, float] = field(default_factory=dict)
    hierarchy_path: Optional[list] = field(default_factory=list)
    history_periods: int = 0

    def __init__(
        self,
        name: str,
        attribute_values: Optional[Dict[str, str]] = None,
        attribute_weights: Optional[Dict[str, float]] = None,
        hierarchy_path: Optional[list] = None,
        history_periods: int = 0,
        metric_type: Optional[str] = None,
        surface: Optional[str] = None,
        geography: Optional[str] = None,
    ):
        self.name = name
        self.attribute_values = dict(attribute_values or {})
        self.attribute_weights = {
            key: float(value)
            for key, value in dict(attribute_weights or {}).items()
            if value is not None
        }
        self.hierarchy_path = list(hierarchy_path or [])
        self.history_periods = history_periods

        # Backward-compatible aliases for older TVA experiments.
        legacy_attributes = {
            'metric_type': metric_type,
            'surface': surface,
            'geography': geography,
        }
        for key, value in legacy_attributes.items():
            if value is not None and key not in self.attribute_values:
                self.attribute_values[key] = value

    @property
    def metric_type(self) -> Optional[str]:
        return self.attribute_values.get('metric_type')

    @property
    def surface(self) -> Optional[str]:
        return self.attribute_values.get('surface')

    @property
    def geography(self) -> Optional[str]:
        return self.attribute_values.get('geography')


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
        detected_features: dict = None,
        trend_data: pd.DataFrame = None,
        observed_history: Dict[str, int] = None,
        prior_construction_config: dict = None,
        causal_prior_construction_config: dict = None,
        series_names: list = None,
    ):
        self.series_metadata = series_metadata or []
        self.relationship_matrix = relationship_matrix
        self.prior_confidence = prior_confidence
        self.detected_features = detected_features or {}
        self.trend_data = trend_data
        self.observed_history = dict(observed_history or {})
        self.prior_construction_config = prior_construction_config
        self.causal_prior_construction_config = causal_prior_construction_config

        if self.series_metadata:
            self._series_names = [m.name for m in self.series_metadata]
        elif self.trend_data is not None:
            self._series_names = list(self.trend_data.columns)
        else:
            self._series_names = list(series_names or [])

    @property
    def n_series(self):
        return len(self._series_names)

    def _attribute_names(self) -> list:
        """Return sorted union of categorical metadata keys across series."""
        attribute_names = set()
        for metadata in self.series_metadata:
            attribute_names.update(metadata.attribute_values.keys())
        return sorted(attribute_names)

    def _resolve_structural_config(self) -> dict:
        config = self.prior_construction_config
        if not config:
            return {}
        resolved = dict(config)
        resolved['sources'] = list(resolved.get('sources', ['event', 'metadata']))
        resolved['source_weights'] = dict(
            resolved.get('source_weights', {'event': 0.7, 'metadata': 0.3})
        )
        resolved['max_distance_days'] = int(resolved.get('max_distance_days', 7))
        resolved['min_series_per_cluster'] = int(
            resolved.get('min_series_per_cluster', 2)
        )
        resolved['event_family_weights'] = dict(
            resolved.get(
                'event_family_weights',
                {
                    'trend_changepoints': 1.0,
                    'level_shifts': 1.0,
                    'anomalies': 0.6,
                },
            )
        )
        return resolved

    @staticmethod
    def _normalize_off_diagonal(adjacency: np.ndarray) -> Optional[np.ndarray]:
        adjacency = np.asarray(adjacency, dtype=np.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            return None
        if adjacency.shape[0] == 0:
            return None
        result = adjacency.copy()
        np.fill_diagonal(result, 0.0)
        max_off_diag = float(np.nanmax(result)) if result.size else 0.0
        if max_off_diag <= 0:
            return None
        result = result / max_off_diag
        result = np.clip(result, 0.0, 1.0)
        np.fill_diagonal(result, 0.0)
        return result.astype(np.float32)

    @staticmethod
    def _coerce_timestamp(value):
        try:
            return pd.Timestamp(value)
        except Exception:
            return None

    @staticmethod
    def _magnitude_from_value(value) -> float:
        try:
            magnitude = abs(float(value))
        except Exception:
            magnitude = 0.0
        if not np.isfinite(magnitude):
            return 0.0
        return magnitude

    def _series_index(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self._series_names)}

    def _build_metadata_prior_adjacency(self) -> Optional[np.ndarray]:
        """Construct a soft prior adjacency matrix from shared attributes."""
        n = self.n_series
        if n == 0 or not self.series_metadata:
            return None

        adj = np.zeros((n, n), dtype=np.float32)
        attribute_names = self._attribute_names()
        attribute_weights = self._resolve_metadata_attribute_weights(attribute_names)
        if not attribute_weights:
            return None

        for attr, attr_weight in attribute_weights.items():
            values = [m.attribute_values.get(attr) for m in self.series_metadata]
            for i in range(n):
                for j in range(i + 1, n):
                    if (
                        values[i] is not None
                        and values[j] is not None
                        and values[i] == values[j]
                    ):
                        adj[i, j] += attr_weight
                        adj[j, i] += attr_weight

        if np.count_nonzero(adj) == 0:
            return None
        adj = np.clip(adj, 0.0, 1.0)
        np.fill_diagonal(adj, 0.0)
        return adj.astype(np.float32)

    def _resolve_metadata_attribute_weights(
        self, attribute_names: list
    ) -> Dict[str, float]:
        """Resolve normalized per-attribute weights for metadata priors."""
        if not attribute_names:
            return {}

        config_weights = {}
        if isinstance(self.prior_construction_config, dict):
            config_weights = dict(
                self.prior_construction_config.get('metadata_attribute_weights', {})
            )

        series_weight_sums = {}
        series_weight_counts = {}
        for metadata in self.series_metadata:
            for attr, value in getattr(metadata, 'attribute_weights', {}).items():
                if attr not in attribute_names:
                    continue
                try:
                    numeric = float(value)
                except Exception:
                    continue
                if not np.isfinite(numeric):
                    continue
                series_weight_sums[attr] = series_weight_sums.get(attr, 0.0) + numeric
                series_weight_counts[attr] = series_weight_counts.get(attr, 0) + 1

        resolved = {}
        explicit_weight_found = False
        for attr in attribute_names:
            weight = config_weights.get(attr)
            if weight is None and series_weight_counts.get(attr, 0) > 0:
                weight = series_weight_sums[attr] / float(series_weight_counts[attr])
            if weight is not None:
                try:
                    weight = float(weight)
                except Exception:
                    weight = None
            if weight is not None and np.isfinite(weight) and weight > 0:
                explicit_weight_found = True
                resolved[attr] = weight

        if not explicit_weight_found:
            # Preserve equal weighting by default; only apply directional heuristics
            # when TVA sees the common surface/geography-only metadata shape.
            if set(attribute_names) == {'surface', 'geography'}:
                resolved = {'surface': 0.7, 'geography': 0.3}
            else:
                resolved = {attr: 1.0 for attr in attribute_names}
        else:
            for attr in attribute_names:
                if attr not in resolved:
                    resolved[attr] = 1.0

        total_weight = float(sum(resolved.values()))
        if total_weight <= 0:
            return {}
        return {
            attr: float(weight) / total_weight
            for attr, weight in resolved.items()
            if float(weight) > 0
        }

    def _extract_event_records(self) -> list:
        features = self.detected_features or {}
        series_index = self._series_index()
        if not series_index:
            return []

        event_specs = [
            ('trend_changepoints', self._parse_trend_changepoint_record),
            ('level_shifts', self._parse_level_shift_record),
            ('anomalies', self._parse_anomaly_record),
        ]
        records = []

        for family, parser in event_specs:
            family_events = features.get(family, {})
            if not isinstance(family_events, dict):
                continue
            family_magnitudes = []
            parsed_records = []
            for series_name, events in family_events.items():
                if series_name not in series_index or not events:
                    continue
                for event in events:
                    parsed = parser(event)
                    if parsed is None:
                        continue
                    parsed['series_name'] = series_name
                    parsed['event_family'] = family
                    family_magnitudes.append(parsed['raw_magnitude'])
                    parsed_records.append(parsed)

            if not parsed_records:
                continue

            scale = (
                float(np.nanmedian(np.abs(family_magnitudes)))
                if family_magnitudes
                else 0.0
            )
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0

            for parsed in parsed_records:
                parsed['scaled_magnitude'] = parsed['raw_magnitude'] / scale
                records.append(parsed)

        return records

    def _parse_trend_changepoint_record(self, event) -> Optional[dict]:
        if isinstance(event, dict):
            timestamp = self._coerce_timestamp(event.get('date'))
            prior_slope = event.get('prior_slope', 0.0)
            new_slope = event.get('new_slope', 0.0)
        elif isinstance(event, (tuple, list)) and len(event) >= 3:
            timestamp = self._coerce_timestamp(event[0])
            prior_slope = event[1]
            new_slope = event[2]
        else:
            return None
        if timestamp is None:
            return None
        return {
            'timestamp': timestamp,
            'raw_magnitude': self._magnitude_from_value(
                float(new_slope) - float(prior_slope)
            ),
        }

    def _parse_level_shift_record(self, event) -> Optional[dict]:
        if isinstance(event, dict):
            timestamp = self._coerce_timestamp(event.get('date'))
            magnitude = event.get('magnitude', 0.0)
        elif isinstance(event, (tuple, list)) and len(event) >= 2:
            timestamp = self._coerce_timestamp(event[0])
            magnitude = event[1]
        else:
            return None
        if timestamp is None:
            return None
        return {
            'timestamp': timestamp,
            'raw_magnitude': self._magnitude_from_value(magnitude),
        }

    def _parse_anomaly_record(self, event) -> Optional[dict]:
        if isinstance(event, dict):
            timestamp = self._coerce_timestamp(event.get('date'))
            magnitude = event.get('magnitude', 0.0)
        elif isinstance(event, (tuple, list)) and len(event) >= 2:
            timestamp = self._coerce_timestamp(event[0])
            magnitude = event[1]
        else:
            return None
        if timestamp is None:
            return None
        return {
            'timestamp': timestamp,
            'raw_magnitude': self._magnitude_from_value(magnitude),
        }

    @staticmethod
    def _cluster_records_by_time(records: list, max_distance_days: int) -> list:
        """Cluster event records by timestamp with a bounded cluster diameter.

        Complete-linkage agglomerative clustering with a distance cap: no two
        events in a cluster are more than max_distance_days apart. This
        replaces single-linkage gap chaining, where a busy panel chains into
        one giant cluster and yields an all-ones prior (D-7b).
        """
        if not records:
            return []
        if len(records) == 1:
            return [list(records)]
        day_values = np.array(
            [record['timestamp'].value for record in records], dtype=float
        ) / (86400.0 * 1e9)
        labels = None
        try:
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=float(max_distance_days) + 1e-9,
                linkage='complete',
            )
            labels = clustering.fit_predict(day_values.reshape(-1, 1))
        except Exception:
            pass
        if labels is None:
            # fallback: sorted gap chaining (legacy behavior)
            order = np.argsort(day_values)
            labels = np.zeros(len(records), dtype=int)
            current = 0
            for prev, nxt in zip(order[:-1], order[1:]):
                if day_values[nxt] - day_values[prev] > max_distance_days:
                    current += 1
                labels[nxt] = current

        clusters = {}
        for record, label in zip(records, labels):
            clusters.setdefault(int(label), []).append(record)
        return [
            sorted(cluster, key=lambda x: x['timestamp'])
            for _, cluster in sorted(clusters.items())
        ]

    def _build_event_prior_adjacency(self, config: dict) -> Optional[np.ndarray]:
        n = self.n_series
        if n == 0:
            return None

        records = self._extract_event_records()
        if not records:
            return None

        adjacency = np.zeros((n, n), dtype=np.float32)
        series_index = self._series_index()
        family_weights = config.get('event_family_weights', {})
        max_distance_days = max(int(config.get('max_distance_days', 7)), 1)
        min_series_per_cluster = max(int(config.get('min_series_per_cluster', 2)), 2)

        by_family = {}
        for record in records:
            by_family.setdefault(record['event_family'], []).append(record)

        for family, family_records in by_family.items():
            family_weight = float(family_weights.get(family, 1.0))
            if family_weight <= 0:
                continue
            for cluster in self._cluster_records_by_time(
                family_records, max_distance_days
            ):
                per_series = {}
                for event in cluster:
                    existing = per_series.get(event['series_name'])
                    if (
                        existing is None
                        or event['scaled_magnitude'] > existing['scaled_magnitude']
                    ):
                        per_series[event['series_name']] = event
                if len(per_series) < min_series_per_cluster:
                    continue
                # D-7c: drop market-wide clusters — a shock that hits (nearly)
                # every series belongs to the shared factor, not to pairwise
                # edges. Keeping it would re-inject the common-driver
                # confounder the factor layer just removed.
                market_wide_fraction = float(config.get('market_wide_fraction', 0.8))
                if n >= 4 and len(per_series) >= market_wide_fraction * n:
                    continue

                cluster_events = list(per_series.values())
                cluster_center = pd.Timestamp(
                    np.median([event['timestamp'].value for event in cluster_events])
                )
                mean_distance_days = float(
                    np.mean(
                        [
                            abs((event['timestamp'] - cluster_center).total_seconds())
                            / 86400.0
                            for event in cluster_events
                        ]
                    )
                )
                tightness = 1.0 / (1.0 + (mean_distance_days / max_distance_days))
                cluster_strength = float(
                    np.mean([event['scaled_magnitude'] for event in cluster_events])
                )
                weight = family_weight * tightness * cluster_strength

                # D-7a: DIRECT edges — earlier event -> later event. Ties emit
                # both directions at half weight. No blanket symmetrization.
                series_names = sorted(per_series.keys())
                for i, left_name in enumerate(series_names):
                    left_idx = series_index[left_name]
                    left_time = per_series[left_name]['timestamp']
                    for right_name in series_names[i + 1 :]:
                        right_idx = series_index[right_name]
                        right_time = per_series[right_name]['timestamp']
                        if left_time < right_time:
                            adjacency[left_idx, right_idx] += weight
                        elif right_time < left_time:
                            adjacency[right_idx, left_idx] += weight
                        else:
                            adjacency[left_idx, right_idx] += weight / 2.0
                            adjacency[right_idx, left_idx] += weight / 2.0

        return self._normalize_off_diagonal(adjacency)

    def _blend_structural_sources(
        self, source_matrices: dict, config: dict
    ) -> Optional[np.ndarray]:
        if not source_matrices:
            return None
        if len(source_matrices) == 1:
            matrix = next(iter(source_matrices.values()))
            return matrix.copy() if matrix is not None else None

        weights = config.get('source_weights', {})
        total_weight = 0.0
        blended = None
        for source_name, matrix in source_matrices.items():
            if matrix is None:
                continue
            weight = float(weights.get(source_name, 1.0))
            if weight <= 0:
                continue
            total_weight += weight
            if blended is None:
                blended = weight * matrix.astype(np.float32)
            else:
                blended = blended + weight * matrix.astype(np.float32)

        if blended is None or total_weight <= 0:
            return None
        blended = blended / total_weight
        blended = np.clip(blended, 0.0, 1.0)
        np.fill_diagonal(blended, 0.0)
        return blended.astype(np.float32)

    def build_structural_prior_adjacency(self) -> Optional[np.ndarray]:
        """Construct a soft structural prior from explicit, metadata, and event sources."""
        if self.relationship_matrix is not None:
            return np.asarray(self.relationship_matrix, dtype=np.float32).copy()

        config = self._resolve_structural_config()

        if not config:
            return self._build_metadata_prior_adjacency()

        source_matrices = {}
        for source_name in config.get('sources', []):
            if source_name == 'metadata':
                source_matrices[source_name] = self._build_metadata_prior_adjacency()
            elif source_name == 'event':
                source_matrices[source_name] = self._build_event_prior_adjacency(config)

        return self._blend_structural_sources(source_matrices, config)

    def build_prior_adjacency(self) -> Optional[np.ndarray]:
        """Backward-compatible alias for structural prior construction."""
        return self.build_structural_prior_adjacency()

    def build_metadata_embeddings(self) -> np.ndarray:
        """Build (N, D_meta) one-hot embedding matrix from categorical metadata.

        Encodes arbitrary categorical metadata keys as concatenated one-hot vectors.
        Returns zeros if no metadata is available.
        """
        if not self.series_metadata:
            n = self.n_series
            return np.zeros((n, 1), dtype=np.float32)

        n = self.n_series
        vocabs = {}
        for attr in self._attribute_names():
            unique_vals = sorted(
                set(
                    m.attribute_values.get(attr)
                    for m in self.series_metadata
                    if m.attribute_values.get(attr) is not None
                )
            )
            vocabs[attr] = {v: i for i, v in enumerate(unique_vals)}

        d_meta = sum(len(v) for v in vocabs.values())
        if d_meta == 0:
            return np.zeros((n, 1), dtype=np.float32)

        embeddings = np.zeros((n, d_meta), dtype=np.float32)
        offset = 0
        for attr in self._attribute_names():
            vocab = vocabs[attr]
            for i, m in enumerate(self.series_metadata):
                val = m.attribute_values.get(attr)
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
            return np.ones(self.n_series, dtype=bool)
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
            return np.eye(self.n_series, dtype=np.float32)

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
        if adj is None:
            return 0
        return int(np.count_nonzero(adj - np.diag(np.diag(adj))) // 2)


# ---------------------------------------------------------------------------
# One usable prior format
# ---------------------------------------------------------------------------


def _prior_warn_unknown(unknown, source: str):
    if not unknown:
        return
    warnings.warn(
        "TVA prior references series not present in the panel and dropped "
        f"them ({source}): {sorted(unknown)}",
        RuntimeWarning,
        stacklevel=3,
    )


def _prior_from_dataframe(prior, series_names, index) -> np.ndarray:
    unknown = set()
    for label in list(prior.index) + list(prior.columns):
        if label not in index:
            unknown.add(str(label))
    _prior_warn_unknown(unknown, 'DataFrame labels')
    aligned = prior.reindex(index=series_names, columns=series_names)
    return aligned.astype(float).fillna(0.0).to_numpy()


def _prior_from_edge_list(prior, series_names, index) -> np.ndarray:
    n = len(series_names)
    adjacency = np.zeros((n, n), dtype=float)
    unknown = set()
    for row in prior:
        source = row.get('source', row.get('from'))
        target = row.get('target', row.get('to'))
        if source is None or target is None:
            continue
        if source not in index:
            unknown.add(str(source))
        if target not in index:
            unknown.add(str(target))
        if source not in index or target not in index:
            continue
        i, j = index[source], index[target]
        if i == j:
            continue
        try:
            weight = float(row.get('weight', 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if not np.isfinite(weight) or weight == 0.0:
            continue
        adjacency[i, j] = weight
        if not bool(row.get('directed', False)):
            adjacency[j, i] = weight
    _prior_warn_unknown(unknown, 'edge list')
    return adjacency


def _prior_from_groups(prior, series_names, index) -> np.ndarray:
    n = len(series_names)
    adjacency = np.zeros((n, n), dtype=float)
    unknown = set()
    for group in prior:
        if isinstance(group, dict):
            members = group.get('series', group.get('members', []))
            try:
                weight = float(group.get('weight', 1.0))
            except (TypeError, ValueError):
                weight = 1.0
        else:
            members, weight = group, 1.0
        if isinstance(members, str):
            members = [members]
        idx = []
        for name in members or []:
            if name not in index:
                unknown.add(str(name))
                continue
            idx.append(index[name])
        idx = sorted(set(idx))
        if len(idx) < 2 or not np.isfinite(weight) or weight == 0.0:
            continue
        block = np.asarray(idx, dtype=int)
        sub = adjacency[np.ix_(block, block)]
        adjacency[np.ix_(block, block)] = np.where(
            np.abs(sub) >= abs(weight), sub, weight
        )
    _prior_warn_unknown(unknown, 'group list')
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def coerce_prior_adjacency(prior, series_names) -> Optional[np.ndarray]:
    """Normalize any supported user prior spec into a signed (N, N) matrix.

    The single entry point for user-supplied graph priors, so every downstream
    consumer reads one shape regardless of how the prior was expressed.

    Supported forms:

    * ``None`` -> ``None``
    * ``(N, N)`` array-like -> validated and passed through
    * ``pd.DataFrame`` -> reindexed onto ``series_names`` (order-safe)
    * list of ``{'source', 'target', 'weight', 'directed'}`` dicts -> symmetric
      unless the row sets ``directed: True``
    * list of name groups (``[['a', 'b'], ...]``) -> each group is a clique at
      weight 1.0; ``{'series': [...], 'weight': 0.6}`` for a weaker group

    Unknown series names raise a ``RuntimeWarning`` naming them and are
    dropped, never silently ignored.

    Args:
        prior: user-supplied prior in any of the forms above.
        series_names: panel column order defining the matrix axes.

    Returns:
        (N, N) float32, signed, clipped to [-1, 1], hollow diagonal; or None.
    """
    if prior is None:
        return None
    series_names = list(series_names or [])
    n = len(series_names)
    if n == 0:
        return None
    index = {name: i for i, name in enumerate(series_names)}

    if isinstance(prior, pd.DataFrame):
        adjacency = _prior_from_dataframe(prior, series_names, index)
    elif isinstance(prior, dict):
        # {'a': {'b': 0.8}} nested mapping -- a DataFrame in disguise
        adjacency = _prior_from_dataframe(pd.DataFrame(prior).T, series_names, index)
    elif (
        isinstance(prior, (list, tuple))
        and len(prior)
        and isinstance(prior[0], dict)
        and ('source' in prior[0] or 'from' in prior[0])
    ):
        adjacency = _prior_from_edge_list(prior, series_names, index)
    elif (
        isinstance(prior, (list, tuple))
        and len(prior)
        and isinstance(prior[0], (list, tuple, set, dict))
    ):
        adjacency = _prior_from_groups(prior, series_names, index)
    else:
        adjacency = np.asarray(prior, dtype=float)

    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape != (n, n):
        warnings.warn(
            "TVA prior_adjacency could not be interpreted as an "
            f"({n}, {n}) matrix over the panel (got shape "
            f"{getattr(adjacency, 'shape', None)}); ignoring it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    adjacency = np.clip(adjacency, -1.0, 1.0)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency.astype(np.float32)
