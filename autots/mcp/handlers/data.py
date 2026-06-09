"""Handlers for cache management and data loading/conversion tools.

Handlers return plain, JSON-serializable data. Transport layers (the MCP server,
the Pyodide worker) are responsible for wrapping the result.
"""

import base64
import sys

import pandas as pd

from autots import (
    load_daily,
    load_hourly,
    load_weekly,
    load_monthly,
    load_yearly,
    load_live_daily,
    load_linear,
    load_sine,
    load_artificial,
)
from autots.datasets.synthetic import SyntheticDailyGenerator
from autots.tools.transform import GeneralTransformer

from autots.mcp.cache import (
    cache_object,
    get_cached_object,
    list_all_cached_objects,
    clear_cache,
)
from autots.mcp.data_utils import (
    load_to_dataframe,
    dataframe_to_output,
    build_csv_metadata,
)
from autots.mcp.ingest import smart_load


async def handle_list_cache(arguments: dict, log_progress) -> dict:
    return list_all_cached_objects()


async def handle_clear_cache(arguments: dict, log_progress) -> dict:
    obj_id = arguments.get("object_id")
    cache_type = arguments.get("cache_type")
    clear_cache(obj_id, cache_type)
    return {"success": True}


async def handle_load_sample_data(arguments: dict, log_progress) -> dict:
    dataset = arguments.get("dataset", "daily")
    long = arguments.get("long", False)

    loaders = {
        "daily": load_daily,
        "hourly": load_hourly,
        "weekly": load_weekly,
        "monthly": load_monthly,
        "yearly": load_yearly,
        "linear": load_linear,
        "sine": load_sine,
        "artificial": load_artificial,
    }
    df = loaders[dataset](long=long)

    data_id = cache_object(
        df,
        'data',
        {
            'source': dataset,
            'format': 'long' if long else 'wide',
            'rows': len(df),
            'columns': len(df.columns),
        },
    )

    return {
        "data_id": data_id,
        "source": dataset,
        "rows": len(df),
        "cols": len(df.columns),
    }


# Parameters of load_live_daily that the PWA / general API may control. Absent
# keys fall back to library defaults; a value of None for any source-selecting
# parameter skips that source (load_live_daily's contract), so the frontend
# sends null for de-selected sources.
_LIVE_PARAM_KEYS = (
    "observation_start",
    "observation_end",
    "fred_key",
    "fred_series",
    "tickers",
    "trends_list",
    "trends_geo",
    "weather_data_types",
    "weather_stations",
    "weather_years",
    "noaa_cdo_token",
    "london_air_stations",
    "london_air_species",
    "london_air_days",
    "earthquake_days",
    "earthquake_min_magnitude",
    "gsa_key",
    "nasa_api_key",
    "gov_domain_list",
    "gov_domain_limit",
    "wikipedia_pages",
    "wiki_language",
    "weather_event_types",
    "caiso_query",
    "eia_key",
    "eia_respondents",
    "sleep_seconds",
)


async def handle_load_live_data(arguments: dict, log_progress) -> dict:
    long = arguments.get("long", False)

    # Forward only recognized parameters straight through to load_live_daily.
    kwargs = {k: arguments[k] for k in _LIVE_PARAM_KEYS if k in arguments}

    status_log = []

    # load_live_daily is synchronous and blocking. Under Pyodide we drive the
    # (sync-backed) async progress wrapper to completion so its postMessage
    # flushes to the UI thread even while the worker thread is busy. Off Pyodide
    # (e.g. the MCP server) the progress callback genuinely awaits I/O, so we
    # skip mid-call progress and rely on the returned per-source status list.
    if sys.platform == "emscripten":

        def _prog(message):
            try:
                log_progress(message).send(None)
            except StopIteration:
                pass
            except Exception:
                pass

    else:
        _prog = None

    try:
        df = load_live_daily(
            long=long,
            progress_cb=_prog,
            status_log=status_log,
            **kwargs,
        )
    except ValueError as e:
        # Every source failed (e.g. CORS/network in-browser). Surface the
        # per-source failures rather than erroring out so the UI can explain
        # what happened.
        return {
            "data_id": None,
            "rows": 0,
            "cols": 0,
            "sources": status_log,
            "error": str(e),
        }

    data_id = cache_object(
        df,
        'data',
        {
            'source': 'live',
            'format': 'long' if long else 'wide',
            'rows': len(df),
            'columns': len(df.columns),
            'sources': status_log,
        },
    )

    return {
        "data_id": data_id,
        "rows": len(df),
        "cols": len(df.columns),
        "sources": status_log,
    }


async def handle_generate_synthetic_data(arguments: dict, log_progress) -> dict:
    n_series = arguments.get("n_series", 5)
    generator = SyntheticDailyGenerator(n_series=n_series, random_seed=42)
    df = generator.data

    data_id = cache_object(
        df,
        'data',
        {
            'source': 'synthetic',
            'n_series': n_series,
            'rows': len(df),
            'columns': len(df.columns),
        },
    )

    return {
        "data_id": data_id,
        "n_series": n_series,
        "rows": len(df),
        "cols": len(df.columns),
    }


async def handle_load_data_from_file(arguments: dict, log_progress) -> dict:
    filepath = arguments.get("filepath")
    df = load_to_dataframe(filepath)

    data_id = cache_object(
        df,
        'data',
        {
            'source': 'file',
            'filepath': filepath,
            'rows': len(df),
            'columns': len(df.columns),
        },
    )

    return {
        "data_id": data_id,
        "source": filepath,
        "rows": len(df),
        "cols": len(df.columns),
    }


async def handle_smart_load(arguments: dict, log_progress) -> dict:
    """Load messy user data (paste / upload / URL) with auto cleanup + detection.

    Accepts pasted ``text`` (CSV/TSV), a ``url``, or base64-encoded
    ``content_base64`` (CSV or XLSX; ``filename`` disambiguates). Returns a
    ``data_id`` plus a ``report`` describing what was cleaned and detected.
    """
    text = arguments.get("text")
    url = arguments.get("url")
    filename = arguments.get("filename")
    content_base64 = arguments.get("content_base64")
    data_format = arguments.get("data_format", "auto")
    long_cols = arguments.get("long_cols")

    csv_bytes = base64.b64decode(content_base64) if content_base64 else None
    if text is None and url is None and csv_bytes is None:
        raise ValueError(
            "Must provide one of: 'text', 'url', or 'content_base64'"
        )

    await log_progress("smart_load: parsing and cleaning uploaded data")
    df, report = smart_load(
        text=text,
        csv_bytes=csv_bytes,
        url=url,
        filename=filename,
        data_format=data_format,
        long_cols=long_cols,
    )

    data_id = cache_object(
        df,
        'data',
        {
            'source': 'smart_load',
            'detected_format': report.get('detected_format'),
            'rows': len(df),
            'columns': len(df.columns),
        },
    )

    return {
        "data_id": data_id,
        "report": report,
        "rows": len(df),
        "cols": len(df.columns),
    }


async def handle_get_data(arguments: dict, log_progress) -> dict:
    data_id = arguments.get("data_id")
    output_format = arguments.get("output_format", "json_wide")

    cached = get_cached_object(data_id, 'data')
    df = cached['object']

    if output_format.startswith("csv"):
        filepath = dataframe_to_output(df, output_format)
        is_long = output_format == "csv_long"
        return build_csv_metadata(filepath, df, is_long)
    else:
        return dataframe_to_output(df, output_format)


async def handle_convert_long_to_wide(arguments: dict, log_progress) -> dict:
    data = arguments.get("data")
    data_id = arguments.get("data_id")

    if data is None and data_id is None:
        raise ValueError("Must provide either 'data' or 'data_id' parameter")

    if data_id:
        df = load_to_dataframe(data_id=data_id)
    elif data:
        df = load_to_dataframe(data, data_format="long")
    else:
        raise ValueError("Must provide data or data_id")

    new_data_id = cache_object(
        df,
        'data',
        {
            'source': 'converted',
            'format': 'wide',
            'rows': len(df),
            'columns': len(df.columns),
        },
    )

    return {
        "data_id": new_data_id,
        "rows": len(df),
        "cols": len(df.columns),
    }


async def handle_clean_data(arguments: dict, log_progress) -> dict:
    data = arguments.get("data")
    data_id = arguments.get("data_id")
    fillna = arguments.get("fillna", "ffill")

    if data is None and data_id is None:
        raise ValueError("Must provide either 'data' or 'data_id' parameter")

    df = load_to_dataframe(data, data_id=data_id)

    transformer = GeneralTransformer(fillna=fillna)
    df_clean = transformer.fit_transform(df)

    clean_data_id = cache_object(
        df_clean,
        'data',
        {
            'source': 'cleaned',
            'fillna': fillna,
            'rows': len(df_clean),
            'columns': len(df_clean.columns),
        },
    )

    return {
        "data_id": clean_data_id,
        "rows": len(df_clean),
        "cols": len(df_clean.columns),
    }


DATA_HANDLERS = {
    "list_cache": handle_list_cache,
    "clear_cache": handle_clear_cache,
    "load_sample_data": handle_load_sample_data,
    "load_live_data": handle_load_live_data,
    "generate_synthetic_data": handle_generate_synthetic_data,
    "load_data_from_file": handle_load_data_from_file,
    "smart_load": handle_smart_load,
    "get_data": handle_get_data,
    "convert_long_to_wide": handle_convert_long_to_wide,
    "clean_data": handle_clean_data,
}
