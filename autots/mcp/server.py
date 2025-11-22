"""
MCP Server for AutoTS Time Series Forecasting

This server exposes AutoTS forecasting and analysis functions as MCP tools
for integration with LLM environments like VS Code.
"""
import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Optional, Dict, Union

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP not available. Install with: pip install autots[mcp]")

from autots import (
    load_daily, load_hourly, load_weekly, load_monthly, load_yearly,
    load_live_daily, load_linear, load_sine, load_artificial,
    AutoTS, EventRiskForecast, long_to_wide
)
from autots.datasets.synthetic import SyntheticDailyGenerator
from autots.evaluator.anomaly_detector import AnomalyDetector, HolidayDetector
from autots.evaluator.feature_detector import TimeSeriesFeatureDetector
from autots.tools.transform import GeneralTransformer

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def serialize_timestamps(obj):
    """
    Recursively convert pandas Timestamp objects to strings for JSON serialization.
    
    Args:
        obj: Object that may contain Timestamp objects
    
    Returns:
        Object with Timestamps converted to strings
    """
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


# ============================================================================
# Global Caches - Store objects for later retrieval
# ============================================================================

PREDICTION_CACHE: Dict[str, Dict[str, Any]] = {}
AUTOTS_CACHE: Dict[str, Dict[str, Any]] = {}
EVENT_RISK_CACHE: Dict[str, Dict[str, Any]] = {}
FEATURE_DETECTOR_CACHE: Dict[str, Dict[str, Any]] = {}
DATA_CACHE: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Cache Management Functions
# ============================================================================

def cache_object(obj: Any, cache_type: str, metadata: dict = None) -> str:
    """
    Cache an object and return a unique ID.
    
    Args:
        obj: Object to cache
        cache_type: Type of cache ('prediction', 'autots', 'event_risk', 'feature_detector', 'data')
        metadata: Optional metadata
    
    Returns:
        Unique object ID
    """
    obj_id = str(uuid.uuid4())
    cache_entry = {
        'object': obj,
        'metadata': metadata or {},
        'created_at': datetime.now().isoformat()
    }
    
    if cache_type == 'prediction':
        PREDICTION_CACHE[obj_id] = cache_entry
    elif cache_type == 'autots':
        AUTOTS_CACHE[obj_id] = cache_entry
    elif cache_type == 'event_risk':
        EVENT_RISK_CACHE[obj_id] = cache_entry
    elif cache_type == 'feature_detector':
        FEATURE_DETECTOR_CACHE[obj_id] = cache_entry
    elif cache_type == 'data':
        DATA_CACHE[obj_id] = cache_entry
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
    
    return obj_id


def get_cached_object(obj_id: str, cache_type: str) -> Dict[str, Any]:
    """Retrieve a cached object by ID and type."""
    if cache_type == 'prediction':
        cache = PREDICTION_CACHE
    elif cache_type == 'autots':
        cache = AUTOTS_CACHE
    elif cache_type == 'event_risk':
        cache = EVENT_RISK_CACHE
    elif cache_type == 'feature_detector':
        cache = FEATURE_DETECTOR_CACHE
    elif cache_type == 'data':
        cache = DATA_CACHE
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
    
    if obj_id not in cache:
        raise ValueError(f"{cache_type} ID {obj_id} not found in cache")
    return cache[obj_id]


def list_all_cached_objects() -> dict:
    """List all cached objects across all cache types."""
    result = {}
    
    if PREDICTION_CACHE:
        result['predictions'] = [
            {'id': k, 'created_at': v['created_at'], 'metadata': v['metadata']}
            for k, v in PREDICTION_CACHE.items()
        ]
    if AUTOTS_CACHE:
        result['autots_models'] = [
            {'id': k, 'created_at': v['created_at'], 'metadata': v['metadata']}
            for k, v in AUTOTS_CACHE.items()
        ]
    if EVENT_RISK_CACHE:
        result['event_risk'] = [
            {'id': k, 'created_at': v['created_at'], 'metadata': v['metadata']}
            for k, v in EVENT_RISK_CACHE.items()
        ]
    if FEATURE_DETECTOR_CACHE:
        result['feature_detectors'] = [
            {'id': k, 'created_at': v['created_at'], 'metadata': v['metadata']}
            for k, v in FEATURE_DETECTOR_CACHE.items()
        ]
    if DATA_CACHE:
        result['data'] = [
            {'id': k, 'created_at': v['created_at'], 'metadata': v['metadata']}
            for k, v in DATA_CACHE.items()
        ]
    
    return result


def clear_cache(obj_id: Optional[str] = None, cache_type: Optional[str] = None):
    """Clear cache - specific ID, specific type, or all if both None."""
    if obj_id and cache_type:
        cache = {
            'prediction': PREDICTION_CACHE,
            'autots': AUTOTS_CACHE,
            'event_risk': EVENT_RISK_CACHE,
            'feature_detector': FEATURE_DETECTOR_CACHE,
            'data': DATA_CACHE
        }.get(cache_type)
        if cache and obj_id in cache:
            del cache[obj_id]
    elif cache_type:
        if cache_type == 'prediction':
            PREDICTION_CACHE.clear()
        elif cache_type == 'autots':
            AUTOTS_CACHE.clear()
        elif cache_type == 'event_risk':
            EVENT_RISK_CACHE.clear()
        elif cache_type == 'feature_detector':
            FEATURE_DETECTOR_CACHE.clear()
        elif cache_type == 'data':
            DATA_CACHE.clear()
    else:
        PREDICTION_CACHE.clear()
        AUTOTS_CACHE.clear()
        EVENT_RISK_CACHE.clear()
        FEATURE_DETECTOR_CACHE.clear()
        DATA_CACHE.clear()


# ============================================================================
# Data Loading and Conversion Functions
# ============================================================================

def load_to_dataframe(
    data: Optional[Union[dict, str]] = None,
    data_format: str = "wide",
    data_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data to pandas DataFrame from multiple sources.
    
    Args:
        data: JSON dict, CSV file path, or URL. If None, must provide data_id
        data_format: "wide" or "long" (for JSON input)
        data_id: Optional cached data ID to load from cache
    
    Returns:
        DataFrame with DatetimeIndex
    """
    if data_id:
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
                aggfunc='first'
            )
        
        return df
    
    raise ValueError(f"Unsupported data type: {type(data)}")


def dataframe_to_output(
    df: pd.DataFrame,
    output_format: str = "json_wide",
    save_path: Optional[str] = None
) -> Union[dict, str]:
    """
    Convert DataFrame to requested output format (token-efficient).
    
    Args:
        df: DataFrame with DatetimeIndex
        output_format: "json_wide", "json_long", "csv_wide", "csv_long"
        save_path: Optional path to save CSV (returns path)
    
    Returns:
        Dictionary (JSON) or string (CSV path)
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
            id_vars=[index_col],
            var_name='series_id',
            value_name='value'
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
                    id_vars=[index_col],
                    var_name='series_id',
                    value_name='value'
                )
                df_long = df_long.rename(columns={index_col: 'datetime'})
                df_long.to_csv(save_path, index=False)
            else:
                df_copy.to_csv(save_path)
        return save_path
    
    raise ValueError(f"Unknown output format: {output_format}")


def save_temp_csv(df: pd.DataFrame, is_long: bool = False) -> str:
    """
    Save DataFrame to temporary CSV file.
    
    Args:
        df: DataFrame to save
        is_long: Convert to long format before saving
    
    Returns:
        Full path to saved CSV
    """
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
            id_vars=[index_col],
            var_name='series_id',
            value_name='value'
        )
        df_long = df_long.rename(columns={index_col: 'datetime'})
        df_long.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath)
    
    return filepath


def build_csv_metadata(filepath: str, df: pd.DataFrame, is_long: bool = False) -> dict:
    """
    Build metadata for CSV export with loading instructions.
    
    Args:
        filepath: Path to CSV file
        df: DataFrame that was saved
        is_long: Whether CSV is in long format
    
    Returns:
        Metadata dictionary with loading instructions
    """
    if is_long:
        columns_info = ['datetime', 'series_id', 'value']
        description = 'Long format: datetime,series_id,value columns'
        pandas_cmd = f"pd.read_csv('{filepath}')"
        autots_mcp_cmd = f"Use load_to_dataframe('{filepath}') then convert_long_to_wide"
    else:
        columns_info = list(df.columns)
        description = 'Wide format: datetime index, series as columns'
        pandas_cmd = f"pd.read_csv('{filepath}',parse_dates=True,index_col=0)"
        autots_mcp_cmd = f"Use load_to_dataframe('{filepath}') to load this CSV file"
    
    metadata = {
        'filepath': filepath,
        'format': 'long' if is_long else 'wide',
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'columns': columns_info,
        'loading_instructions': {
            'description': description,
            'pandas': pandas_cmd,
            'autots_mcp': autots_mcp_cmd
        }
    }
    return metadata


# ============================================================================
# MCP Server Implementation
# ============================================================================
# TODO: add a forecast adjustment tool for specific date ranges (e.g., "increase growth by 2% for Q3 2026")
# TODO: enhance component extraction for more model types (currently limited to Cassandra/TVVAR)

if MCP_AVAILABLE:
    app = Server("autots")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available AutoTS tools."""
        return [
            # Cache management
            Tool(
                name="list_cache",
                description="List all cached objects (predictions, autots models, event_risk, feature_detectors, data)",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="clear_cache",
                description="Clear cache: specific object by ID and type, entire cache type, or all caches",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "object_id": {"type": "string", "description": "Specific object ID to clear"},
                        "cache_type": {
                            "type": "string",
                            "enum": ["prediction", "autots", "event_risk", "feature_detector", "data"],
                            "description": "Cache type to clear (omit both params to clear all)"
                        }
                    }
                }
            ),
            
            # Data loading
            Tool(
                name="load_sample_data",
                description="Load sample time series dataset. Returns data_id for use in other tools",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "string",
                            "enum": ["daily", "hourly", "weekly", "monthly", "yearly", "linear", "sine", "artificial"],
                            "default": "daily",
                            "description": "Sample dataset to load"
                        },
                        "long": {"type": "boolean", "default": False, "description": "Return long format (default: wide)"}
                    }
                }
            ),
            Tool(
                name="load_live_data",
                description="Load live data from FRED, stocks, etc. Returns data_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "fred_key": {"type": "string", "description": "FRED API key"},
                        "fred_series": {"type": "array", "items": {"type": "string"}, "description": "FRED series codes"},
                        "tickers": {"type": "array", "items": {"type": "string"}, "description": "Stock tickers"},
                        "long": {"type": "boolean", "default": False, "description": "Return long format"}
                    }
                }
            ),
            Tool(
                name="generate_synthetic_data",
                description="Generate synthetic time series with labeled components. Returns data_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "n_series": {"type": "integer", "default": 5, "description": "Number of series to generate"}
                    }
                }
            ),
            Tool(
                name="load_data_from_file",
                description="Load CSV from local path or URL. Returns data_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Local file path or URL to CSV"}
                    },
                    "required": ["filepath"]
                }
            ),
            Tool(
                name="get_data",
                description="Get cached data as JSON (wide/long) or save as CSV with metadata",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_id": {"type": "string", "description": "Cached data ID"},
                        "output_format": {
                            "type": "string",
                            "enum": ["json_wide", "json_long", "csv_wide", "csv_long"],
                            "default": "json_wide",
                            "description": "Output format"
                        }
                    },
                    "required": ["data_id"]
                }
            ),
            Tool(
                name="convert_long_to_wide",
                description="Convert long format to wide. Returns new data_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Long format data with datetime,series_id,value"},
                        "data_id": {"type": "string", "description": "Cached data ID (alternative to data)"}
                    }
                }
            ),
            Tool(
                name="clean_data",
                description="Clean time series data (handle missing values, outliers). Returns data_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Wide format data"},
                        "data_id": {"type": "string", "description": "Cached data ID"},
                        "fillna": {
                            "type": "string",
                            "enum": ["ffill", "mean", "median", "rolling_mean", "linear"],
                            "default": "ffill",
                            "description": "Missing value fill method"
                        }
                    }
                }
            ),
            
            # Forecasting
            Tool(
                name="forecast_mosaic",
                description="FAST: Pre-configured mosaic ensemble forecast. Use data or data_id. Returns prediction_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Wide format data"},
                        "data_id": {"type": "string", "description": "Cached data ID"},
                        "forecast_length": {"type": "integer", "default": 30, "description": "Periods to forecast"},
                        "profile_template": {"type": "object", "description": "Optional custom mosaic profile JSON"}
                    }
                }
            ),
            Tool(
                name="forecast_explainable",
                description="MODERATE: AutoTS model search with EXPLAINABLE models only (Cassandra, TVVAR, BasicLinearModel). Convenience wrapper for interpretable forecasts. Use when you need to understand model components. Returns prediction_id and autots_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Wide format data"},
                        "data_id": {"type": "string", "description": "Cached data ID"},
                        "forecast_length": {"type": "integer", "default": 30, "description": "Periods to forecast"}
                    }
                }
            ),
            Tool(
                name="forecast_custom",
                description="CUSTOM: AutoTS with user-specified parameters or template. Defaults to 'scalable' model_list for speed and accuracy. For explainable models, use forecast_explainable instead. Returns prediction_id and autots_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Wide format data"},
                        "data_id": {"type": "string", "description": "Cached data ID"},
                        "forecast_length": {"type": "integer", "default": 30, "description": "Periods to forecast"},
                        "autots_params": {"type": "object", "description": "AutoTS initialization parameters (defaults: model_list='scalable')"},
                        "model_template": {"type": "object", "description": "Specific model template to run"}
                    }
                }
            ),
            Tool(
                name="get_autots_docs",
                description="Get documentation for AutoTS custom forecast parameters. Use this before forecast_custom to understand available options",
                inputSchema={"type": "object", "properties": {}}
            ),
            
            # Prediction object tools
            Tool(
                name="get_forecast",
                description="Get forecast from cached prediction as JSON or CSV",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prediction_id": {"type": "string", "description": "Cached prediction ID"},
                        "output": {
                            "type": "string",
                            "enum": ["forecast", "upper_forecast", "lower_forecast"],
                            "default": "forecast",
                            "description": "Which forecast to return"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json_wide", "json_long", "csv_wide", "csv_long"],
                            "default": "json_wide",
                            "description": "Output format"
                        }
                    },
                    "required": ["prediction_id"]
                }
            ),
            Tool(
                name="plot_forecast",
                description="Plot forecast from prediction. Returns base64 PNG image. Defaults to first series only.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prediction_id": {"type": "string", "description": "Cached prediction ID"},
                        "include_history": {"type": "boolean", "default": True, "description": "Include historical data"},
                        "series": {"type": "array", "items": {"type": "string"}, "description": "Specific series to plot (by name/id)"},
                        "plot_all": {"type": "boolean", "default": False, "description": "Plot all series (overrides series parameter)"}
                    },
                    "required": ["prediction_id"]
                }
            ),
            Tool(
                name="apply_constraints",
                description="Apply constraints to forecast (dampen, bounds, quantiles). Returns new prediction_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prediction_id": {"type": "string", "description": "Cached prediction ID"},
                        "constraint_method": {
                            "type": "string",
                            "enum": ["dampen", "upper", "lower", "quantile"],
                            "description": "Constraint type"
                        },
                        "constraint_value": {"type": "number", "description": "Constraint value"},
                        "constraint_direction": {
                            "type": "string",
                            "enum": ["upper", "lower"],
                            "description": "Direction for bounds"
                        }
                    },
                    "required": ["prediction_id", "constraint_method"]
                }
            ),
            Tool(
                name="get_model_params",
                description="Get model parameters from cached prediction",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prediction_id": {"type": "string", "description": "Cached prediction ID"}
                    },
                    "required": ["prediction_id"]
                }
            ),
            Tool(
                name="get_forecast_components",
                description="Get decomposed forecast components (trend, seasonality) if available. Only for Cassandra/TVVAR models",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prediction_id": {"type": "string", "description": "Cached prediction ID"}
                    },
                    "required": ["prediction_id"]
                }
            ),
            
            # AutoTS model tools
            Tool(
                name="get_validation_results",
                description="Get validation results summary from AutoTS search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "autots_id": {"type": "string", "description": "Cached AutoTS ID"}
                    },
                    "required": ["autots_id"]
                }
            ),
            Tool(
                name="plot_validation",
                description="Plot validation forecasts from AutoTS search. Returns base64 PNG",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "autots_id": {"type": "string", "description": "Cached AutoTS ID"}
                    },
                    "required": ["autots_id"]
                }
            ),
            Tool(
                name="plot_generation_loss",
                description="Plot unpredictability score from AutoTS search. Returns base64 PNG",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "autots_id": {"type": "string", "description": "Cached AutoTS ID"}
                    },
                    "required": ["autots_id"]
                }
            ),
            
            # Event Risk tools
            Tool(
                name="forecast_event_risk",
                description="Forecast probability of crossing threshold. Returns event_risk_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Wide format data"},
                        "data_id": {"type": "string", "description": "Cached data ID"},
                        "forecast_length": {"type": "integer", "default": 30, "description": "Periods to forecast"},
                        "threshold": {"type": "number", "description": "Threshold value"},
                        "direction": {
                            "type": "string",
                            "enum": ["upper", "lower"],
                            "default": "upper",
                            "description": "Detect crossing above (upper) or below (lower)"
                        },
                        "tune": {"type": "boolean", "default": False, "description": "Enable model tuning (slower but more accurate)"}
                    },
                    "required": ["threshold"]
                }
            ),
            Tool(
                name="get_event_risk_results",
                description="Get event risk probabilities from cached EventRiskForecast",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "event_risk_id": {"type": "string", "description": "Cached event risk ID"},
                        "format": {
                            "type": "string",
                            "enum": ["json_wide", "json_long", "csv_wide", "csv_long"],
                            "default": "json_wide",
                            "description": "Output format"
                        }
                    },
                    "required": ["event_risk_id"]
                }
            ),
            Tool(
                name="plot_event_risk",
                description="Plot event risk probabilities. Returns base64 PNG",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "event_risk_id": {"type": "string", "description": "Cached event risk ID"}
                    },
                    "required": ["event_risk_id"]
                }
            ),
            
            # Feature detection tools
            Tool(
                name="detect_features",
                description="Detect anomalies, changepoints, holidays, patterns. Returns detector_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Wide format data"},
                        "data_id": {"type": "string", "description": "Cached data ID"}
                    }
                }
            ),
            Tool(
                name="get_detected_features",
                description="Get detected features (anomalies, changepoints, holidays, seasonality) from cached detector. Supports filtering by date range, specific dates, and by series name. Use this to answer queries like 'was there an anomaly on Christmas 2024' or 'when was the first level shift'. Parameters: detector_id (required), date_start/date_end for ranges, series_name for filtering, include_components and include_metadata flags.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "detector_id": {"type": "string", "description": "Cached detector ID"},
                        "date_start": {"type": "string", "description": "Optional start date for filtering (YYYY-MM-DD format)"},
                        "date_end": {"type": "string", "description": "Optional end date for filtering (YYYY-MM-DD format)"},
                        "specific_date": {"type": "string", "description": "Optional specific single date to query (YYYY-MM-DD format)"},
                        "series_name": {"type": "string", "description": "Optional series name to filter results"},
                        "include_components": {"type": "boolean", "description": "Include component time series values (default: false)"},
                        "include_metadata": {"type": "boolean", "description": "Include metadata like noise levels, scales (default: false)"}
                    },
                    "required": ["detector_id"]
                }
            ),
            Tool(
                name="plot_features",
                description="Plot detected features. Returns base64 PNG",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "detector_id": {"type": "string", "description": "Cached detector ID"},
                        "series": {"type": "array", "items": {"type": "string"}, "description": "Specific series to plot"}
                    },
                    "required": ["detector_id"]
                }
            ),
            Tool(
                name="forecast_from_features",
                description="Create forecast using detected features (EXPERIMENTAL: use only after feature detection, not for standalone forecasts). Returns prediction_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "detector_id": {"type": "string", "description": "Cached detector ID"},
                        "forecast_length": {"type": "integer", "default": 30, "description": "Periods to forecast"}
                    },
                    "required": ["detector_id"]
                }
            ),
        ]

    # ========================================================================
    # Tool Implementations
    # ========================================================================

    @app.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Execute the requested tool."""
        try:
            # Cache management tools
            if name == "list_cache":
                cache_info = list_all_cached_objects()
                return [TextContent(type="text", text=json.dumps(cache_info, separators=(',', ':')))]
            
            elif name == "clear_cache":
                obj_id = arguments.get("object_id")
                cache_type = arguments.get("cache_type")
                clear_cache(obj_id, cache_type)
                return [TextContent(type="text", text=json.dumps({"success": True}, separators=(',', ':')))]
            
            # Data loading tools
            elif name == "load_sample_data":
                dataset = arguments.get("dataset", "daily")
                long = arguments.get("long", False)
                
                loaders = {
                    "daily": load_daily, "hourly": load_hourly, "weekly": load_weekly,
                    "monthly": load_monthly, "yearly": load_yearly, "linear": load_linear,
                    "sine": load_sine, "artificial": load_artificial
                }
                df = loaders[dataset](long=long)
                
                data_id = cache_object(df, 'data', {
                    'source': dataset, 'format': 'long' if long else 'wide',
                    'rows': len(df), 'columns': len(df.columns)
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "data_id": data_id, "source": dataset, "rows": len(df), "cols": len(df.columns)
                }, separators=(',', ':')))]
            
            elif name == "load_live_data":
                fred_key = arguments.get("fred_key")
                fred_series = arguments.get("fred_series")
                tickers = arguments.get("tickers")
                long = arguments.get("long", False)
                
                df = load_live_daily(long=long, fred_key=fred_key, fred_series=fred_series, tickers=tickers)
                
                data_id = cache_object(df, 'data', {
                    'source': 'live', 'format': 'long' if long else 'wide',
                    'rows': len(df), 'columns': len(df.columns)
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "data_id": data_id, "rows": len(df), "cols": len(df.columns)
                }, separators=(',', ':')))]
            
            elif name == "generate_synthetic_data":
                n_series = arguments.get("n_series", 5)
                generator = SyntheticDailyGenerator(n_series=n_series, random_seed=42)
                df = generator.data
                
                data_id = cache_object(df, 'data', {
                    'source': 'synthetic', 'n_series': n_series,
                    'rows': len(df), 'columns': len(df.columns)
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "data_id": data_id, "n_series": n_series, "rows": len(df), "cols": len(df.columns)
                }, separators=(',', ':')))]
            
            elif name == "load_data_from_file":
                filepath = arguments.get("filepath")
                df = load_to_dataframe(filepath)
                
                data_id = cache_object(df, 'data', {
                    'source': 'file', 'filepath': filepath,
                    'rows': len(df), 'columns': len(df.columns)
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "data_id": data_id, "source": filepath, "rows": len(df), "cols": len(df.columns)
                }, separators=(',', ':')))]
            
            elif name == "get_data":
                data_id = arguments.get("data_id")
                output_format = arguments.get("output_format", "json_wide")
                
                cached = get_cached_object(data_id, 'data')
                df = cached['object']
                
                if output_format.startswith("csv"):
                    filepath = dataframe_to_output(df, output_format)
                    is_long = output_format == "csv_long"
                    metadata = build_csv_metadata(filepath, df, is_long)
                    return [TextContent(type="text", text=json.dumps(metadata, separators=(',', ':')))]
                else:
                    result = dataframe_to_output(df, output_format)
                    return [TextContent(type="text", text=json.dumps(result, separators=(',', ':')))]
            
            elif name == "convert_long_to_wide":
                data = arguments.get("data")
                data_id = arguments.get("data_id")
                
                if data_id:
                    df = load_to_dataframe(data_id=data_id)
                elif data:
                    df = load_to_dataframe(data, data_format="long")
                else:
                    raise ValueError("Must provide data or data_id")
                
                new_data_id = cache_object(df, 'data', {
                    'source': 'converted', 'format': 'wide',
                    'rows': len(df), 'columns': len(df.columns)
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "data_id": new_data_id, "rows": len(df), "cols": len(df.columns)
                }, separators=(',', ':')))]
            
            elif name == "clean_data":
                data = arguments.get("data")
                data_id = arguments.get("data_id")
                fillna = arguments.get("fillna", "ffill")
                
                df = load_to_dataframe(data, data_id=data_id)
                
                transformer = GeneralTransformer(fillna=fillna)
                df_clean = transformer.fit_transform(df)
                
                clean_data_id = cache_object(df_clean, 'data', {
                    'source': 'cleaned', 'fillna': fillna,
                    'rows': len(df_clean), 'columns': len(df_clean.columns)
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "data_id": clean_data_id, "rows": len(df_clean), "cols": len(df_clean.columns)
                }, separators=(',', ':')))]
            
            # Forecasting tools
            elif name == "forecast_mosaic":
                data = arguments.get("data")
                data_id = arguments.get("data_id")
                forecast_length = arguments.get("forecast_length", 30)
                profile_template = arguments.get("profile_template")
                
                df = load_to_dataframe(data, data_id=data_id)
                
                if profile_template:
                    model = AutoTS(
                        forecast_length=forecast_length,
                        frequency='infer',
                        ensemble='mosaic',
                        model_list='no_shared',
                        max_generations=0,
                        num_validations=0,
                        validation_method='backwards'
                    )
                    model = model.import_template(profile_template, method='only')
                    prediction = model.predict()
                else:
                    model = AutoTS(
                        forecast_length=forecast_length,
                        frequency='infer',
                        ensemble='mosaic-window',
                        model_list='fast',
                        max_generations=0,
                        num_validations=0,
                        validation_method='backwards'
                    )
                    model.fit(df)
                    prediction = model.predict()
                
                prediction_id = cache_object(prediction, 'prediction', {
                    'method': 'mosaic', 'forecast_length': forecast_length,
                    'series_count': len(df.columns),
                    'historical_data_id': data_id
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "prediction_id": prediction_id, "forecast_length": forecast_length
                }, separators=(',', ':')))]
            
            elif name == "forecast_explainable":
                data = arguments.get("data")
                data_id = arguments.get("data_id")
                forecast_length = arguments.get("forecast_length", 30)
                
                df = load_to_dataframe(data, data_id=data_id)
                
                model = AutoTS(
                    forecast_length=forecast_length,
                    frequency='infer',
                    ensemble=None,
                    model_list=['Cassandra', 'TVVAR', 'BasicLinearModel'],
                    max_generations=3,
                    num_validations=2,
                    validation_method='backwards'
                )
                model.fit(df)
                prediction = model.predict()
                
                prediction_id = cache_object(prediction, 'prediction', {
                    'method': 'explainable', 'forecast_length': forecast_length,
                    'series_count': len(df.columns),
                    'historical_data_id': data_id
                })
                autots_id = cache_object(model, 'autots', {
                    'forecast_length': forecast_length, 'series_count': len(df.columns)
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "prediction_id": prediction_id, "autots_id": autots_id,
                    "forecast_length": forecast_length
                }, separators=(',', ':')))]
            
            elif name == "forecast_custom":
                data = arguments.get("data")
                data_id = arguments.get("data_id")
                forecast_length = arguments.get("forecast_length", 30)
                autots_params = arguments.get("autots_params", {})
                model_template = arguments.get("model_template")
                
                df = load_to_dataframe(data, data_id=data_id)
                
                if 'forecast_length' not in autots_params:
                    autots_params['forecast_length'] = forecast_length
                if 'frequency' not in autots_params:
                    autots_params['frequency'] = 'infer'
                if 'model_list' not in autots_params:
                    autots_params['model_list'] = 'scalable'
                
                model = AutoTS(**autots_params)
                
                if model_template:
                    model = model.import_template(model_template, method='only')
                    prediction = model.predict()
                else:
                    model.fit(df)
                    prediction = model.predict()
                
                prediction_id = cache_object(prediction, 'prediction', {
                    'method': 'custom', 'forecast_length': forecast_length,
                    'series_count': len(df.columns),
                    'historical_data_id': data_id
                })
                autots_id = cache_object(model, 'autots', {
                    'forecast_length': forecast_length, 'series_count': len(df.columns)
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "prediction_id": prediction_id, "autots_id": autots_id,
                    "forecast_length": forecast_length
                }, separators=(',', ':')))]
            
            elif name == "get_autots_docs":
                docs = {
                    "AutoTS_Parameters": {
                        "forecast_length": "Number of periods to forecast (required)",
                        "frequency": "Pandas frequency string ('D','H','W','MS',etc.) or 'infer'",
                        "ensemble": "Ensemble method: 'simple','distance','horizontal','mosaic',None",
                        "model_list": "List of models or preset: 'fast','superfast','all','default'",
                        "transformer_list": "Transformations: 'fast','superfast','all'",
                        "max_generations": "Number of genetic algorithm generations",
                        "num_validations": "Number of cross-validation splits",
                        "validation_method": "'backwards','even','seasonal',etc.",
                        "models_to_validate": "Fraction of models to fully validate (0.0-1.0)",
                        "n_jobs": "Parallel processes: 'auto',-1,or specific number"
                    },
                    "Example": {
                        "autots_params": {
                            "forecast_length": 30,
                            "frequency": "D",
                            "ensemble": "simple",
                            "model_list": "fast",
                            "max_generations": 5
                        }
                    },
                    "Documentation": "See extended_tutorial.md for complete documentation"
                }
                return [TextContent(type="text", text=json.dumps(docs, separators=(',', ':')))]
            
            # Prediction object tools
            elif name == "get_forecast":
                prediction_id = arguments.get("prediction_id")
                output = arguments.get("output", "forecast")
                format_type = arguments.get("format", "json_wide")
                
                cached = get_cached_object(prediction_id, 'prediction')
                prediction = cached['object']
                
                if output == "forecast":
                    df = prediction.forecast
                elif output == "upper_forecast":
                    df = prediction.upper_forecast
                elif output == "lower_forecast":
                    df = prediction.lower_forecast
                else:
                    raise ValueError(f"Unknown output type: {output}")
                
                if format_type.startswith("csv"):
                    filepath = dataframe_to_output(df, format_type)
                    is_long = format_type == "csv_long"
                    metadata = build_csv_metadata(filepath, df, is_long)
                    return [TextContent(type="text", text=json.dumps(metadata, separators=(',', ':')))]
                else:
                    result = dataframe_to_output(df, format_type)
                    return [TextContent(type="text", text=json.dumps(result, separators=(',', ':')))]
            
            elif name == "plot_forecast":
                prediction_id = arguments.get("prediction_id")
                include_history = arguments.get("include_history", True)
                series = arguments.get("series")
                plot_all = arguments.get("plot_all", False)
                
                cached = get_cached_object(prediction_id, 'prediction')
                prediction = cached['object']
                metadata = cached.get('metadata', {})
                
                # Determine which series to plot
                if plot_all:
                    forecast_df = prediction.forecast
                    series_names = list(prediction.forecast.columns)
                elif series:
                    forecast_df = prediction.forecast[series]
                    series_names = series if isinstance(series, list) else [series]
                else:
                    # Default to first series only
                    first_col = prediction.forecast.columns[0]
                    forecast_df = prediction.forecast[[first_col]]
                    series_names = [first_col]
                
                # Adjust figure size based on legend needs
                if len(series_names) > 5:
                    fig, ax = plt.subplots(figsize=(14, 6))
                else:
                    fig, ax = plt.subplots(figsize=(12, 6))
                
                # Try to get historical data if available
                history_df = None
                if include_history and 'historical_data_id' in metadata:
                    try:
                        history_cached = get_cached_object(metadata['historical_data_id'], 'data')
                        history_df = history_cached['object'][series_names]
                    except:
                        pass
                
                # Plot historical data
                if history_df is not None:
                    for col in series_names:
                        ax.plot(history_df.index, history_df[col], label=f'{col} (history)', alpha=0.7)
                
                # Plot forecast
                for col in series_names:
                    ax.plot(forecast_df.index, forecast_df[col], label=f'{col} (forecast)', linestyle='--', linewidth=2)
                
                # Plot prediction intervals
                if hasattr(prediction, 'upper_forecast') and prediction.upper_forecast is not None:
                    for col in series_names:
                        ax.fill_between(
                            forecast_df.index,
                            prediction.lower_forecast[col],
                            prediction.upper_forecast[col],
                            alpha=0.2
                        )
                
                ax.set_title(f'Forecast ({len(series_names)} series)' if len(series_names) > 1 else f'Forecast: {series_names[0]}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                
                # Place legend outside plot area if many series, otherwise inside
                if len(series_names) > 5:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                else:
                    ax.legend(loc='best')
                    plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                return [ImageContent(type="image", data=img_base64, mimeType="image/png")]
            
            elif name == "apply_constraints":
                prediction_id = arguments.get("prediction_id")
                constraint_method = arguments.get("constraint_method")
                constraint_value = arguments.get("constraint_value")
                constraint_direction = arguments.get("constraint_direction", "upper")
                
                cached = get_cached_object(prediction_id, 'prediction')
                prediction = cached['object']
                
                if constraint_method == "dampen":
                    prediction = prediction.apply_constraints(
                        constraint_method="dampen",
                        constraint_value=constraint_value
                    )
                elif constraint_method in ["upper", "lower"]:
                    prediction = prediction.apply_constraints(
                        constraint_method="constraint",
                        constraint_value=constraint_value,
                        constraint_direction=constraint_direction
                    )
                elif constraint_method == "quantile":
                    prediction = prediction.apply_constraints(
                        constraint_method="quantile",
                        constraint_value=constraint_value
                    )
                else:
                    raise ValueError(f"Unknown constraint method: {constraint_method}")
                
                new_prediction_id = cache_object(prediction, 'prediction', {
                    'method': 'constrained', 'constraint_method': constraint_method,
                    'original_prediction_id': prediction_id
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "prediction_id": new_prediction_id, "constraint_method": constraint_method
                }, separators=(',', ':')))]
            
            elif name == "get_model_params":
                prediction_id = arguments.get("prediction_id")
                
                cached = get_cached_object(prediction_id, 'prediction')
                prediction = cached['object']
                
                params = {
                    'model_name': prediction.model_name,
                    'model_parameters': prediction.model_parameters,
                    'transformation_parameters': prediction.transformation_parameters,
                    'forecast_length': len(prediction.forecast)
                }
                
                return [TextContent(type="text", text=json.dumps(params, separators=(',', ':')))]
            
            elif name == "get_forecast_components":
                prediction_id = arguments.get("prediction_id")
                
                cached = get_cached_object(prediction_id, 'prediction')
                prediction = cached['object']
                
                if hasattr(prediction, 'model') and hasattr(prediction.model, 'get_params_components'):
                    components = prediction.model.get_params_components()
                    result = {}
                    for key, df in components.items():
                        result[key] = dataframe_to_output(df, "json_wide")
                    return [TextContent(type="text", text=json.dumps(result, separators=(',', ':')))]
                else:
                    return [TextContent(type="text", text=json.dumps({
                        "error": "Components not available for this model type"
                    }, separators=(',', ':')))]
            
            # AutoTS model tools
            elif name == "get_validation_results":
                autots_id = arguments.get("autots_id")
                
                cached = get_cached_object(autots_id, 'autots')
                model = cached['object']
                
                if hasattr(model, 'results'):
                    results_summary = model.results().head(10).to_string()
                    return [TextContent(type="text", text=results_summary)]
                else:
                    return [TextContent(type="text", text="No validation results available")]
            
            elif name == "plot_validation":
                autots_id = arguments.get("autots_id")
                
                cached = get_cached_object(autots_id, 'autots')
                model = cached['object']
                
                fig = model.plot_validations()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                return [ImageContent(type="image", data=img_base64, mimeType="image/png")]
            
            elif name == "plot_generation_loss":
                autots_id = arguments.get("autots_id")
                
                cached = get_cached_object(autots_id, 'autots')
                model = cached['object']
                
                fig = model.plot_generation_loss()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                return [ImageContent(type="image", data=img_base64, mimeType="image/png")]
            
            # Event Risk tools
            elif name == "forecast_event_risk":
                data = arguments.get("data")
                data_id = arguments.get("data_id")
                forecast_length = arguments.get("forecast_length", 30)
                threshold = arguments.get("threshold")
                direction = arguments.get("direction", "upper")
                tune = arguments.get("tune", False)
                
                df = load_to_dataframe(data, data_id=data_id)
                
                erf = EventRiskForecast(
                    df=df,
                    forecast_length=forecast_length,
                    frequency='infer',
                    threshold=threshold,
                    direction=direction,
                    model_name='default',
                    include_differenced=True,
                    regression_type=None
                )
                
                if tune:
                    erf.fit(constraint=None)
                else:
                    erf.fit_no_tune()
                
                event_risk_id = cache_object(erf, 'event_risk', {
                    'threshold': threshold, 'direction': direction,
                    'forecast_length': forecast_length, 'tuned': tune
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "event_risk_id": event_risk_id, "threshold": threshold,
                    "direction": direction, "forecast_length": forecast_length
                }, separators=(',', ':')))]
            
            elif name == "get_event_risk_results":
                event_risk_id = arguments.get("event_risk_id")
                format_type = arguments.get("format", "json_wide")
                
                cached = get_cached_object(event_risk_id, 'event_risk')
                erf = cached['object']
                
                df = erf.predict_historic_risk()
                
                if format_type.startswith("csv"):
                    filepath = dataframe_to_output(df, format_type)
                    is_long = format_type == "csv_long"
                    metadata = build_csv_metadata(filepath, df, is_long)
                    return [TextContent(type="text", text=json.dumps(metadata, separators=(',', ':')))]
                else:
                    result = dataframe_to_output(df, format_type)
                    return [TextContent(type="text", text=json.dumps(result, separators=(',', ':')))]
            
            elif name == "plot_event_risk":
                event_risk_id = arguments.get("event_risk_id")
                
                cached = get_cached_object(event_risk_id, 'event_risk')
                erf = cached['object']
                
                fig = erf.plot()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                return [ImageContent(type="image", data=img_base64, mimeType="image/png")]
            
            # Feature detection tools
            elif name == "detect_features":
                data = arguments.get("data")
                data_id = arguments.get("data_id")
                
                df = load_to_dataframe(data, data_id=data_id)
                
                detector = TimeSeriesFeatureDetector()
                detector.fit(df)
                
                detector_id = cache_object(detector, 'feature_detector', {
                    'series_count': len(df.columns), 'data_rows': len(df)
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "detector_id": detector_id, "series_count": len(df.columns)
                }, separators=(',', ':')))]
            
            elif name == "get_detected_features":
                detector_id = arguments.get("detector_id")
                date_start = arguments.get("date_start")
                date_end = arguments.get("date_end")
                specific_date = arguments.get("specific_date")
                series_name = arguments.get("series_name")
                include_components = arguments.get("include_components", False)
                include_metadata = arguments.get("include_metadata", False)
                
                cached = get_cached_object(detector_id, 'feature_detector')
                detector = cached['object']
                
                # Parse dates argument to appropriate format for query_features
                date_filter = None
                if specific_date:
                    # Single specific date
                    date_filter = specific_date
                elif date_start or date_end:
                    # Date range
                    date_filter = slice(date_start, date_end)
                
                # Call query_features with the parsed arguments
                try:
                    results = detector.query_features(
                        dates=date_filter,
                        series=series_name,
                        include_components=include_components,
                        include_metadata=include_metadata,
                        return_json=False
                    )
                    
                    # Add summary information at the top level
                    summary = {
                        "date_range": {
                            "start": detector.df_original.index[0].strftime('%Y-%m-%d'),
                            "end": detector.df_original.index[-1].strftime('%Y-%m-%d')
                        },
                        "num_series": len(detector.df_original.columns),
                        "num_observations": len(detector.df_original),
                        "series_names": list(detector.df_original.columns)
                    }
                    
                    # Add detection counts for quick reference
                    detection_counts = {}
                    for series_name in results.get('series', {}).keys():
                        series_data = results['series'][series_name]
                        counts = {
                            "trend_changepoints": len(series_data.get('trend_changepoints', [])),
                            "level_shifts": len(series_data.get('level_shifts', [])),
                            "anomalies": len(series_data.get('anomalies', [])),
                            "holidays": len(series_data.get('holiday_dates', []))
                        }
                        if 'seasonality_strength' in series_data:
                            counts['seasonality_strength'] = series_data['seasonality_strength']
                        detection_counts[series_name] = counts
                    
                    output = {
                        'summary': summary,
                        'detection_counts': detection_counts,
                        'features': results
                    }
                    
                    return [TextContent(type="text", text=json.dumps(output, indent=2))]
                    
                except Exception as e:
                    # Fallback to basic feature extraction if query_features fails
                    import traceback
                    logger.warning(f"query_features failed, falling back to basic extraction: {e}")
                    logger.warning(f"Traceback: {traceback.format_exc()}")
                    
                    # Get all features using the old method
                    all_features = detector.get_detected_features(
                        series_name=series_name,
                        include_components=include_components,
                        include_metadata=include_metadata
                    )
                    
                    summary = {
                        "date_range": {
                            "start": detector.df_original.index[0].strftime('%Y-%m-%d'),
                            "end": detector.df_original.index[-1].strftime('%Y-%m-%d')
                        },
                        "num_series": len(detector.df_original.columns),
                        "num_observations": len(detector.df_original),
                        "series_names": list(detector.df_original.columns)
                    }
                    
                    results = {
                        'summary': summary,
                        'features': all_features,
                        'note': 'Date filtering not applied due to error'
                    }
                    
                    return [TextContent(type="text", text=json.dumps(results, indent=2))]
            
            elif name == "plot_features":
                detector_id = arguments.get("detector_id")
                series = arguments.get("series")
                
                cached = get_cached_object(detector_id, 'feature_detector')
                detector = cached['object']
                
                if series:
                    fig = detector.plot(series=series[0])
                else:
                    fig = detector.plot()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                return [ImageContent(type="image", data=img_base64, mimeType="image/png")]
            
            elif name == "forecast_from_features":
                detector_id = arguments.get("detector_id")
                forecast_length = arguments.get("forecast_length", 30)
                
                cached = get_cached_object(detector_id, 'feature_detector')
                detector = cached['object']
                
                prediction = detector.predict(forecast_length=forecast_length)
                
                prediction_id = cache_object(prediction, 'prediction', {
                    'method': 'feature_detector', 'forecast_length': forecast_length,
                    'detector_id': detector_id
                })
                
                return [TextContent(type="text", text=json.dumps({
                    "prediction_id": prediction_id, "forecast_length": forecast_length,
                    "note": "This forecast is based on detected features and is experimental"
                }, separators=(',', ':')))]
            
            else:
                return [TextContent(type="text", text=json.dumps({
                    "error": f"Unknown tool: {name}"
                }, separators=(',', ':')))]
        
        except Exception as e:
            logger.exception(f"Error in tool {name}")
            return [TextContent(type="text", text=json.dumps({
                "error": str(e), "tool": name
            }, separators=(',', ':')))]

    @app.list_resources()
    async def list_resources() -> list[Any]:
        """List available documentation resources."""
        from os.path import dirname, join, exists
        
        resources = []
        base_path = dirname(dirname(dirname(__file__)))
        mcp_path = dirname(__file__)
        
        doc_files = [
            ("README.md", "MCP Server README", mcp_path),
            ("extended_tutorial.md", "Extended AutoTS Tutorial", base_path),
            ("production_example.py", "Production Example Script", base_path),
            ("README.md", "AutoTS Main README", base_path),
            ("TODO.md", "AutoTS Development Roadmap", base_path),
        ]
        
        for filename, description, base in doc_files:
            filepath = join(base, filename)
            if exists(filepath):
                resources.append(
                    EmbeddedResource(
                        type="resource",
                        resource={
                            "uri": f"file://{filepath}",
                            "name": filename,
                            "description": description,
                            "mimeType": "text/markdown" if filename.endswith('.md') else "text/plain"
                        }
                    )
                )
        
        return resources

    @app.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a documentation resource."""
        if uri.startswith("file://"):
            filepath = uri[7:]
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading {filepath}: {str(e)}"
        else:
            return f"Unknown resource URI: {uri}"


def serve():
    """Start the MCP server."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP package not installed. Install with: pip install autots[mcp]")
    
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    
    asyncio.run(main())


if __name__ == "__main__":
    serve()
