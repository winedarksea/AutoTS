"""State management & global caches for the AutoTS MCP server."""

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Global Caches
# ============================================================================

PREDICTION_CACHE: Dict[str, Dict[str, Any]] = {}
AUTOTS_CACHE: Dict[str, Dict[str, Any]] = {}
EVENT_RISK_CACHE: Dict[str, Dict[str, Any]] = {}
FEATURE_DETECTOR_CACHE: Dict[str, Dict[str, Any]] = {}
DATA_CACHE: Dict[str, Dict[str, Any]] = {}

try:
    CACHE_MAX_OBJECTS = max(1, int(os.environ.get("AUTOTS_MCP_CACHE_MAX", 60)))
except (TypeError, ValueError):
    CACHE_MAX_OBJECTS = 60

CACHE_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
    'prediction': PREDICTION_CACHE,
    'autots': AUTOTS_CACHE,
    'event_risk': EVENT_RISK_CACHE,
    'feature_detector': FEATURE_DETECTOR_CACHE,
    'data': DATA_CACHE,
}

CACHE_SUMMARY_KEYS = {
    'prediction': 'predictions',
    'autots': 'autots_models',
    'event_risk': 'event_risk',
    'feature_detector': 'feature_detectors',
    'data': 'data',
}


def _resolve_cache(cache_type: str) -> Dict[str, Dict[str, Any]]:
    try:
        return CACHE_REGISTRY[cache_type]
    except KeyError:
        raise ValueError(f"Unknown cache type: {cache_type}") from None


def _enforce_cache_limit(cache: Dict[str, Dict[str, Any]]):
    while len(cache) > CACHE_MAX_OBJECTS:
        oldest_id = next(iter(cache))
        logger.debug(
            f"Cache limit ({CACHE_MAX_OBJECTS}) reached, evicting oldest entry: {oldest_id[:8]}. "
            f"Increase cache size with env var AUTOTS_MCP_CACHE_MAX"
        )
        cache.pop(oldest_id, None)


def cache_object(obj: Any, cache_type: str, metadata: dict = None) -> str:
    """Cache an object and return a unique ID."""
    obj_id = str(uuid.uuid4())
    cache_entry = {
        'object': obj,
        'metadata': metadata or {},
        'created_at': datetime.now().isoformat(),
    }
    cache = _resolve_cache(cache_type)
    cache[obj_id] = cache_entry
    _enforce_cache_limit(cache)
    return obj_id


def get_cached_object(obj_id: str, cache_type: str) -> Dict[str, Any]:
    """Retrieve a cached object by ID and type."""
    cache = _resolve_cache(cache_type)
    if obj_id not in cache:
        raise ValueError(f"{cache_type} ID {obj_id} not found in cache")
    return cache[obj_id]


def list_all_cached_objects() -> dict:
    """List all cached objects across all cache types."""
    result = {}
    for cache_type, cache in CACHE_REGISTRY.items():
        if not cache:
            continue
        summary_key = CACHE_SUMMARY_KEYS[cache_type]
        result[summary_key] = [
            {'id': k, 'created_at': v['created_at'], 'metadata': v['metadata']}
            for k, v in cache.items()
        ]
    return result


def clear_cache(obj_id: Optional[str] = None, cache_type: Optional[str] = None):
    """Clear cache - specific ID, specific type, or all if both None."""
    if obj_id and cache_type:
        cache = _resolve_cache(cache_type)
        cache.pop(obj_id, None)
    elif cache_type:
        _resolve_cache(cache_type).clear()
    else:
        for cache in CACHE_REGISTRY.values():
            cache.clear()
