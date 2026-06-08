"""Pyodide-facing API for the AutoTS PWA.

This is the thin layer the in-browser Web Worker calls. It:

  * Restricts forecasting to a **Pyodide-safe** set of pure-Python models
    (no tensorflow / torch / prophet / xgboost / lightgbm / arch, all of which
    are unavailable or impractical in WASM), and forces ``n_jobs=1`` since
    Pyodide has no process parallelism.
  * Maps the PWA's three buttons ("make forecast", "search for best forecast",
    "search all night") to concrete tool calls, keeping that business logic in
    Python so the Rust/Leptos (or any) frontend stays a thin view layer.
  * Exposes a single JSON-in / JSON-out boundary (:func:`run_command_json`) the
    worker can call, routing ``progress_cb`` to ``postMessage``.

It deliberately avoids importing ``mcp`` so it runs where the MCP SDK is absent.
"""

import json
import math

from autots.models.model_list import superfast as _superfast
from autots.mcp.core import run_tool, available_tools

# ---------------------------------------------------------------------------
# Pyodide-safe model sets
# ---------------------------------------------------------------------------
# ``superfast`` is entirely pure-Python (numpy / pandas / statsmodels / sklearn).
PYODIDE_FAST_MODELS = list(_superfast)

# A broader but still pure-Python set for in-browser model search. Every entry
# below lives in basics.py / statsmodels.py / cassandra.py / matrix_var.py and
# imports none of the native/heavy backends.
PYODIDE_SEARCH_MODELS = [
    'ConstantNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'SeasonalNaive',
    'SeasonalityMotif',
    'SectionalMotif',
    'UnivariateMotif',
    'MetricMotif',
    'BasicLinearModel',
    'ETS',
    'GLM',
    'VAR',
    'Cassandra',
    'FFT',
    'DMD',
    'RRVAR',
    'KalmanStateSpace',
]

# Button -> (generations, validations). "make_forecast" is handled separately
# (it uses the fast feature-detector forecast, not a model search).
SEARCH_PRESETS = {
    "search_forecast": (3, 2),
    "search_all_night": (20, 3),
}

PRESET_COMMANDS = {"make_forecast", *SEARCH_PRESETS}


async def _make_forecast(arguments, progress_cb):
    """Fast forecast via the feature detector (the PWA's default 'Make forecast')."""
    data_id = arguments.get("data_id")
    if not data_id:
        return {"error": "make_forecast requires 'data_id'"}
    forecast_length = arguments.get("forecast_length", 30)

    det = await run_tool("detect_features", {"data_id": data_id}, progress_cb)
    if "error" in det:
        return det
    return await run_tool(
        "forecast_from_features",
        {"detector_id": det["detector_id"], "forecast_length": forecast_length},
        progress_cb,
    )


async def _search_forecast(arguments, progress_cb, generations, validations):
    """Model search restricted to the Pyodide-safe model set."""
    data_id = arguments.get("data_id")
    if not data_id:
        return {"error": "search requires 'data_id'"}
    forecast_length = arguments.get("forecast_length", 30)
    # Allow the caller to widen the budget (generations/validations/etc.) but
    # always force n_jobs=1 and the safe model list so extra params can't
    # smuggle in native-dependency models.
    autots_params = dict(arguments.get("autots_params", {}) or {})
    autots_params.setdefault("max_generations", generations)
    autots_params.setdefault("num_validations", validations)
    autots_params.setdefault("ensemble", None)
    autots_params["model_list"] = PYODIDE_SEARCH_MODELS
    autots_params["n_jobs"] = 1
    return await run_tool(
        "forecast_custom",
        {
            "data_id": data_id,
            "forecast_length": forecast_length,
            "autots_params": autots_params,
        },
        progress_cb,
    )


async def dispatch(command, arguments=None, progress_cb=None):
    """Route a PWA command to a preset or directly to a core tool.

    ``command`` is either a preset ("make_forecast", "search_forecast",
    "search_all_night") or any registered tool name (see
    :func:`list_commands`).
    """
    arguments = arguments or {}
    if command == "make_forecast":
        return await _make_forecast(arguments, progress_cb)
    if command in SEARCH_PRESETS:
        generations, validations = SEARCH_PRESETS[command]
        return await _search_forecast(arguments, progress_cb, generations, validations)
    return await run_tool(command, arguments, progress_cb)


def list_commands():
    """All commands the PWA may call: presets plus every core tool."""
    return sorted(PRESET_COMMANDS) + available_tools()


def _nan_to_null(obj):
    """Recursively replace NaN/±Inf floats with None so json.dumps produces valid JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_null(v) for v in obj]
    return obj


async def run_command_json(command, arguments_json="{}", progress_cb=None):
    """JSON-in / JSON-out boundary for the Pyodide Web Worker.

    Args:
        command: Preset or tool name.
        arguments_json: JSON string (or dict) of arguments.
        progress_cb: Optional callable taking a progress-message string. JS
            functions passed from the worker are invoked synchronously to
            ``postMessage`` progress updates.

    Returns:
        A JSON string of the result.
    """
    if isinstance(arguments_json, str):
        arguments = json.loads(arguments_json) if arguments_json else {}
    else:
        arguments = arguments_json or {}
    result = await dispatch(command, arguments, progress_cb)
    return json.dumps(_nan_to_null(result), default=str)
