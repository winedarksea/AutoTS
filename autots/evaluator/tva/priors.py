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

try:
    from statsmodels.tsa.stattools import grangercausalitytests

    HAS_STATSMODELS = True
except Exception:
    grangercausalitytests = None
    HAS_STATSMODELS = False


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
    hierarchy_path: Optional[list] = field(default_factory=list)
    history_periods: int = 0

    def __init__(
        self,
        name: str,
        attribute_values: Optional[Dict[str, str]] = None,
        hierarchy_path: Optional[list] = None,
        history_periods: int = 0,
        metric_type: Optional[str] = None,
        surface: Optional[str] = None,
        geography: Optional[str] = None,
    ):
        self.name = name
        self.attribute_values = dict(attribute_values or {})
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

    def _resolve_causal_config(self) -> dict:
        config = self.causal_prior_construction_config
        if not config:
            return {}
        resolved = dict(config)
        resolved['max_lag'] = int(resolved.get('max_lag', 3))
        resolved['min_history'] = int(resolved.get('min_history', 90))
        resolved['top_k'] = int(resolved.get('top_k', 8))
        resolved['alpha'] = float(resolved.get('alpha', 0.05))
        resolved['difference'] = bool(resolved.get('difference', True))
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
        n_attrs = len(attribute_names)
        if n_attrs == 0:
            return None

        for attr in attribute_names:
            values = [m.attribute_values.get(attr) for m in self.series_metadata]
            for i in range(n):
                for j in range(i + 1, n):
                    if (
                        values[i] is not None
                        and values[j] is not None
                        and values[i] == values[j]
                    ):
                        adj[i, j] += 1.0 / n_attrs
                        adj[j, i] += 1.0 / n_attrs

        if np.count_nonzero(adj) == 0:
            return None
        adj = np.clip(adj, 0.0, 1.0)
        np.fill_diagonal(adj, 0.0)
        return adj.astype(np.float32)

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
        if not records:
            return []
        sorted_records = sorted(records, key=lambda x: x['timestamp'])
        clusters = [[sorted_records[0]]]
        max_gap = pd.Timedelta(days=max_distance_days)

        for record in sorted_records[1:]:
            previous = clusters[-1][-1]
            if record['timestamp'] - previous['timestamp'] <= max_gap:
                clusters[-1].append(record)
            else:
                clusters.append([record])
        return clusters

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

                series_names = sorted(per_series.keys())
                for i, left_name in enumerate(series_names):
                    left_idx = series_index[left_name]
                    for right_name in series_names[i + 1 :]:
                        right_idx = series_index[right_name]
                        adjacency[left_idx, right_idx] += weight
                        adjacency[right_idx, left_idx] += weight

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

    def build_causal_prior_adjacency(self) -> Optional[np.ndarray]:
        """Construct a directed causal prior from decomposed trend components."""
        config = self._resolve_causal_config()
        if not config:
            return None
        if not HAS_STATSMODELS or self.trend_data is None or self.n_series < 2:
            return None

        trend_data = self.trend_data.copy()
        min_history = int(config['min_history'])
        alpha = float(config['alpha'])
        max_lag = max(int(config['max_lag']), 1)
        top_k = max(int(config['top_k']), 1)

        valid_columns = []
        for column in trend_data.columns:
            history = int(
                self.observed_history.get(column, trend_data[column].notna().sum())
            )
            if history >= min_history:
                valid_columns.append(column)
        if len(valid_columns) < 2:
            return None

        working = trend_data[valid_columns].astype(float)
        if bool(config.get('difference', True)):
            working = working.diff()
        working = working.replace([np.inf, -np.inf], np.nan)
        if len(working.dropna(how='all')) <= max_lag + 2:
            return None

        diff_corr = working.corr().abs().fillna(0.0)
        adjacency = np.zeros((self.n_series, self.n_series), dtype=np.float32)
        series_index = self._series_index()
        found_signal = False

        for target in valid_columns:
            candidate_scores = (
                diff_corr[target]
                .drop(labels=[target], errors='ignore')
                .sort_values(ascending=False)
            )
            if top_k:
                candidate_scores = candidate_scores.head(top_k)

            for source, corr_strength in candidate_scores.items():
                if float(corr_strength) <= 0:
                    continue
                pair = working[[target, source]].dropna()
                if len(pair) <= max_lag + 2:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'ignore',
                            message='verbose is deprecated',
                            category=FutureWarning,
                        )
                        results = grangercausalitytests(
                            pair,
                            maxlag=max_lag,
                            verbose=False,
                        )
                except Exception:
                    continue

                best_pvalue = None
                best_lag = None
                for lag, result in results.items():
                    try:
                        pvalue = float(result[0]['ssr_ftest'][1])
                    except Exception:
                        continue
                    if best_pvalue is None or pvalue < best_pvalue:
                        best_pvalue = pvalue
                        best_lag = int(lag)

                if best_pvalue is None or best_pvalue >= alpha:
                    continue

                lag_factor = 1.0 / max(best_lag, 1)
                significance = 1.0 - min(best_pvalue / alpha, 1.0)
                weight = float(
                    np.clip(corr_strength * significance * lag_factor, 0.0, 1.0)
                )
                if weight <= 0:
                    continue

                adjacency[series_index[source], series_index[target]] = max(
                    adjacency[series_index[source], series_index[target]],
                    weight,
                )
                found_signal = True

        if not found_signal:
            return None

        max_off_diag = float(np.nanmax(adjacency)) if adjacency.size else 0.0
        if max_off_diag > 0:
            adjacency = adjacency / max_off_diag
        adjacency = np.clip(adjacency, 0.0, 1.0)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency.astype(np.float32)

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
