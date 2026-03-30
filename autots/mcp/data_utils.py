"""DataFrame loading and CSV formatting utilities for the AutoTS MCP server."""

import os
import tempfile
import uuid
from typing import Optional, Union

import pandas as pd

from autots import long_to_wide


def serialize_timestamps(obj):
    """Recursively convert pandas Timestamp objects to strings for JSON serialization."""
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, dict):
        return {k: serialize_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_timestamps(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_timestamps(item) for item in obj)
    else:
        return obj


def load_to_dataframe(
    data: Optional[Union[dict, str]] = None,
    data_format: str = "wide",
    data_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load data to pandas DataFrame from multiple sources.

    Args:
        data: JSON dict, CSV file path, or URL. If None, must provide data_id.
        data_format: "wide" or "long" (for JSON dict input).
        data_id: Optional cached data ID to load from cache.

    Returns:
        DataFrame with DatetimeIndex.
    """
    if data_id:
        from autots.mcp.cache import get_cached_object

        cached = get_cached_object(data_id, 'data')
        return cached['object']

    if data is None:
        raise ValueError("Must provide either data or data_id")

    if isinstance(data, str):
        df = pd.read_csv(data, parse_dates=True, index_col=0)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    if isinstance(data, dict):
        df = pd.DataFrame(data)

        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        else:
            raise ValueError("Data must include 'datetime' column")

        if data_format == "long":
            if 'series_id' not in df.columns or 'value' not in df.columns:
                raise ValueError("Long format requires 'series_id' and 'value' columns")
            df = long_to_wide(
                df.reset_index(),
                date_col='datetime',
                value_col='value',
                id_col='series_id',
                aggfunc='first',
            )

        return df

    raise ValueError(f"Unsupported data type: {type(data)}")


def dataframe_to_output(
    df: pd.DataFrame, output_format: str = "json_wide", save_path: Optional[str] = None
) -> Union[dict, str]:
    """
    Convert DataFrame to requested output format.

    Args:
        df: DataFrame with DatetimeIndex.
        output_format: "json_wide", "json_long", "csv_wide", or "csv_long".
        save_path: Optional path to save CSV (returns path).

    Returns:
        Dictionary (JSON) or string (CSV path).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df_copy = df.copy()
    df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')

    if output_format == "json_wide":
        result = df_copy.reset_index().to_dict(orient='list')
        result['datetime'] = result.pop('index', result.get('datetime'))
        return result

    elif output_format == "json_long":
        df_reset = df_copy.reset_index()
        index_col = df_reset.columns[0]
        df_long = df_reset.melt(
            id_vars=[index_col], var_name='series_id', value_name='value'
        )
        df_long = df_long.rename(columns={index_col: 'datetime'})
        return df_long.to_dict(orient='list')

    elif output_format in ["csv_wide", "csv_long"]:
        if save_path is None:
            save_path = save_temp_csv(df, is_long=(output_format == "csv_long"))
        else:
            if output_format == "csv_long":
                df_reset = df_copy.reset_index()
                index_col = df_reset.columns[0]
                df_long = df_reset.melt(
                    id_vars=[index_col], var_name='series_id', value_name='value'
                )
                df_long = df_long.rename(columns={index_col: 'datetime'})
                df_long.to_csv(save_path, index=False)
            else:
                df_copy.to_csv(save_path)
        return save_path

    raise ValueError(f"Unknown output format: {output_format}")


def save_temp_csv(df: pd.DataFrame, is_long: bool = False) -> str:
    """Save DataFrame to a temporary CSV file and return the path."""
    temp_dir = tempfile.gettempdir()
    file_id = str(uuid.uuid4())[:8]
    filename = f"autots_{file_id}_{'long' if is_long else 'wide'}.csv"
    filepath = os.path.join(temp_dir, filename)

    if is_long:
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index)
        df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
        df_reset = df_copy.reset_index()
        index_col = df_reset.columns[0]
        df_long = df_reset.melt(
            id_vars=[index_col], var_name='series_id', value_name='value'
        )
        df_long = df_long.rename(columns={index_col: 'datetime'})
        df_long.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath)

    return filepath


def build_csv_metadata(filepath: str, df: pd.DataFrame, is_long: bool = False) -> dict:
    """Build metadata dict for a CSV export with loading instructions."""
    if is_long:
        columns_info = ['datetime', 'series_id', 'value']
        description = 'Long format: datetime,series_id,value columns'
        pandas_cmd = f"pd.read_csv('{filepath}')"
        autots_mcp_cmd = (
            f"Use load_to_dataframe('{filepath}') then convert_long_to_wide"
        )
    else:
        columns_info = list(df.columns)
        description = 'Wide format: datetime index, series as columns'
        pandas_cmd = f"pd.read_csv('{filepath}',parse_dates=True,index_col=0)"
        autots_mcp_cmd = f"Use load_to_dataframe('{filepath}') to load this CSV file"

    return {
        'filepath': filepath,
        'format': 'long' if is_long else 'wide',
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'columns': columns_info,
        'loading_instructions': {
            'description': description,
            'pandas': pandas_cmd,
            'autots_mcp': autots_mcp_cmd,
        },
    }
