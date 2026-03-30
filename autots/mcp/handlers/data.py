"""Handlers for cache management and data loading/conversion tools."""

import json

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

try:
    from mcp.types import TextContent

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

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


async def handle_list_cache(arguments: dict, log_progress) -> list:
    cache_info = list_all_cached_objects()
    return [
        TextContent(type="text", text=json.dumps(cache_info, separators=(',', ':')))
    ]


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


async def handle_load_live_data(arguments: dict, log_progress) -> dict:
    fred_key = arguments.get("fred_key")
    fred_series = arguments.get("fred_series")
    tickers = arguments.get("tickers")
    long = arguments.get("long", False)

    df = load_live_daily(
        long=long,
        fred_key=fred_key,
        fred_series=fred_series,
        tickers=tickers,
    )

    data_id = cache_object(
        df,
        'data',
        {
            'source': 'live',
            'format': 'long' if long else 'wide',
            'rows': len(df),
            'columns': len(df.columns),
        },
    )

    return {
        "data_id": data_id,
        "rows": len(df),
        "cols": len(df.columns),
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


async def handle_get_data(arguments: dict, log_progress) -> list:
    data_id = arguments.get("data_id")
    output_format = arguments.get("output_format", "json_wide")

    cached = get_cached_object(data_id, 'data')
    df = cached['object']

    if output_format.startswith("csv"):
        filepath = dataframe_to_output(df, output_format)
        is_long = output_format == "csv_long"
        metadata = build_csv_metadata(filepath, df, is_long)
        return [
            TextContent(type="text", text=json.dumps(metadata, separators=(',', ':')))
        ]
    else:
        result = dataframe_to_output(df, output_format)
        return [
            TextContent(type="text", text=json.dumps(result, separators=(',', ':')))
        ]


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
    "get_data": handle_get_data,
    "convert_long_to_wide": handle_convert_long_to_wide,
    "clean_data": handle_clean_data,
}
