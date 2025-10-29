"""
MCP Server for AutoTS Time Series Forecasting

This server exposes AutoTS forecasting and analysis functions as MCP tools
for integration with LLM environments like VS Code.
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional, Dict
import pandas as pd
import numpy as np
import uuid
import io
import base64

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
# Prediction Cache - Store prediction objects for later retrieval
# ============================================================================

# Global cache to store prediction objects and their metadata
PREDICTION_CACHE: Dict[str, Dict[str, Any]] = {}

def cache_prediction(prediction_obj: Any, model_obj: Any = None, metadata: dict = None) -> str:
    """
    Cache a prediction object and return a unique ID.
    
    Args:
        prediction_obj: The AutoTS prediction object to cache
        model_obj: Optional fitted model object
        metadata: Optional additional metadata
    
    Returns:
        Unique prediction ID
    """
    prediction_id = str(uuid.uuid4())
    PREDICTION_CACHE[prediction_id] = {
        'prediction': prediction_obj,
        'model': model_obj,
        'metadata': metadata or {},
        'created_at': datetime.now().isoformat()
    }
    return prediction_id

def get_cached_prediction(prediction_id: str) -> Dict[str, Any]:
    """Retrieve a cached prediction by ID."""
    if prediction_id not in PREDICTION_CACHE:
        raise ValueError(f"Prediction ID {prediction_id} not found in cache")
    return PREDICTION_CACHE[prediction_id]

def list_cached_predictions() -> list:
    """List all cached prediction IDs with metadata."""
    return [
        {
            'id': pid,
            'created_at': cache['created_at'],
            'metadata': cache['metadata']
        }
        for pid, cache in PREDICTION_CACHE.items()
    ]

def clear_prediction_cache(prediction_id: Optional[str] = None):
    """Clear cache - specific ID or all if None."""
    if prediction_id:
        if prediction_id in PREDICTION_CACHE:
            del PREDICTION_CACHE[prediction_id]
    else:
        PREDICTION_CACHE.clear()


# ============================================================================
# Utility Functions for JSON/DataFrame conversion
# ============================================================================

def json_to_dataframe(data: dict, data_format: str = "wide") -> pd.DataFrame:
    """
    Convert JSON data to pandas DataFrame with DatetimeIndex.
    
    Args:
        data: Dictionary with either:
            - Wide format: {"datetime": [...], "series1": [...], "series2": [...]}
            - Long format: {"datetime": [...], "series_id": [...], "value": [...]}
        data_format: "wide" or "long"
    
    Returns:
        DataFrame with DatetimeIndex
    """
    try:
        df = pd.DataFrame(data)
        
        # Convert datetime column to DatetimeIndex
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        else:
            raise ValueError("Data must include 'datetime' column")
        
        # Convert from long to wide if needed
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
    except Exception as e:
        raise ValueError(f"Error converting JSON to DataFrame: {str(e)}")


def dataframe_to_json(df: pd.DataFrame, data_format: str = "wide") -> dict:
    """
    Convert pandas DataFrame to JSON-compatible dictionary.
    
    Args:
        df: DataFrame with DatetimeIndex
        data_format: "wide" or "long"
    
    Returns:
        Dictionary suitable for JSON serialization
    """
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df_copy = df.copy()
        df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
        
        if data_format == "wide":
            result = df_copy.reset_index().to_dict(orient='list')
            result['datetime'] = result.pop('index', result.get('datetime'))
        else:
            # Convert to long format
            df_reset = df_copy.reset_index()
            # Get the name of the index column (might be 'index' or 'datetime' or something else)
            index_col = df_reset.columns[0]
            
            df_long = df_reset.melt(
                id_vars=[index_col],
                var_name='series_id',
                value_name='value'
            )
            df_long = df_long.rename(columns={index_col: 'datetime'})
            result = df_long.to_dict(orient='list')
        
        return result
    except Exception as e:
        raise ValueError(f"Error converting DataFrame to JSON: {str(e)}")


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
            # Data loading tools
            Tool(
                name="get_sample_data",
                description=(
                    "Get sample time series data for testing. Returns load_daily by default (wide format). "
                    "Options: daily, hourly, weekly, monthly, yearly, linear, sine, artificial. "
                    "Set long=false for wide format (default), long=true for long format."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "string",
                            "enum": ["daily", "hourly", "weekly", "monthly", "yearly", "linear", "sine", "artificial"],
                            "default": "daily",
                            "description": "Which sample dataset to load"
                        },
                        "long": {
                            "type": "boolean",
                            "default": False,
                            "description": "Return data in long format (true) or wide format (false)"
                        }
                    }
                }
            ),
            Tool(
                name="load_live_daily",
                description=(
                    "Load live daily data from various sources (FRED, stocks, Google Trends, weather, etc.). "
                    "Requires API keys for some sources. Returns wide format by default."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "fred_key": {
                            "type": "string",
                            "description": "FRED API key for economic data"
                        },
                        "fred_series": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of FRED series codes"
                        },
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock tickers"
                        },
                        "long": {
                            "type": "boolean",
                            "default": False,
                            "description": "Return data in long format"
                        }
                    }
                }
            ),
            Tool(
                name="generate_synthetic_data",
                description=(
                    "Generate synthetic daily time series data with labeled components using SyntheticDailyGenerator. "
                    "Useful for testing and demonstrations. Only n_series parameter is exposed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "n_series": {
                            "type": "integer",
                            "default": 5,
                            "description": "Number of time series to generate"
                        }
                    }
                }
            ),
            Tool(
                name="long_to_wide_converter",
                description=(
                    "Convert long-format time series data to wide format. "
                    "Long format has columns: datetime, series_id, value. "
                    "Wide format has datetime as index and one column per series."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Long format data with datetime, series_id, and value"
                        }
                    },
                    "required": ["data"]
                }
            ),
            
            # Forecasting tools
            Tool(
                name="forecast_mosaic_profile",
                description=(
                    "FAST: Run a pre-configured mosaic profile forecast. This is the fastest forecasting method. "
                    "Uses a pre-trained ensemble model from a JSON profile template. "
                    "Best for quick forecasts when data length >= requested forecast length."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Wide format time series data (datetime + series columns)"
                        },
                        "forecast_length": {
                            "type": "integer",
                            "description": "Number of periods to forecast",
                            "default": 30
                        },
                        "profile_template": {
                            "type": "object",
                            "description": "Optional: Custom mosaic profile as JSON. If not provided, uses default."
                        }
                    },
                    "required": ["data"]
                }
            ),
            Tool(
                name="forecast_autots_search",
                description=(
                    "MODERATE SPEED: Run AutoTS model search with hard-coded fast parameters. "
                    "Searches across multiple model types but is optimized for speed. "
                    "Use this for explainable forecasts (no ensembles, specific model types only). "
                    "Models limited to: Cassandra, TVVAR, BasicLinearModel."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Wide format time series data"
                        },
                        "forecast_length": {
                            "type": "integer",
                            "description": "Number of periods to forecast",
                            "default": 30
                        }
                    },
                    "required": ["data"]
                }
            ),
            Tool(
                name="forecast_custom",
                description=(
                    "CUSTOM: Run AutoTS with custom parameters or a specific model template. "
                    "Use this ONLY when user provides specific AutoTS parameters. "
                    "This tool accepts either AutoTS init parameters or a model template to run."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Wide format time series data"
                        },
                        "forecast_length": {
                            "type": "integer",
                            "description": "Number of periods to forecast",
                            "default": 30
                        },
                        "autots_params": {
                            "type": "object",
                            "description": "AutoTS initialization parameters (forecast_length, frequency, etc.)"
                        },
                        "model_template": {
                            "type": "object",
                            "description": "Specific model template to run"
                        }
                    },
                    "required": ["data"]
                }
            ),
            Tool(
                name="get_autots_docs",
                description=(
                    "Get documentation for AutoTS custom forecast parameters. "
                    "Returns explanation of commonly used parameters for the forecast_custom tool."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            
            # Analysis tools
            Tool(
                name="detect_features",
                description=(
                    "Use TimeSeriesFeatureDetector to detect anomalies, holidays, changepoints, and patterns. "
                    "Runs with default parameters. Returns comprehensive detected features including:\n"
                    "- Trend changepoints (dates, slope changes)\n"
                    "- Level shifts (dates, magnitudes)\n"
                    "- Anomalies (dates, magnitudes, types)\n"
                    "- Holidays (dates, impacts, splash effects)\n"
                    "- Seasonality strength and patterns\n"
                    "- Noise characteristics\n"
                    "Results are organized per series with detection counts and full feature details."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Wide format time series data"
                        }
                    },
                    "required": ["data"]
                }
            ),
            Tool(
                name="get_cleaned_data",
                description=(
                    "Clean time series data using AutoTS transformers. "
                    "Handles missing values, outliers, and data quality issues."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Wide format time series data"
                        },
                        "fillna": {
                            "type": "string",
                            "enum": ["ffill", "mean", "median", "rolling_mean", "linear"],
                            "default": "ffill",
                            "description": "Method to fill missing values"
                        }
                    },
                    "required": ["data"]
                }
            ),
            
            # Event Risk Forecasting tools
            Tool(
                name="forecast_event_risk_default",
                description=(
                    "Forecast the probability of crossing a threshold value. "
                    "Uses default parameters optimized for general use cases."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Wide format time series data"
                        },
                        "forecast_length": {
                            "type": "integer",
                            "description": "Number of periods to forecast",
                            "default": 30
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Threshold value to monitor"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["upper", "lower"],
                            "default": "upper",
                            "description": "Whether to detect crossing above (upper) or below (lower) threshold"
                        }
                    },
                    "required": ["data", "threshold"]
                }
            ),
            Tool(
                name="forecast_event_risk_tuning",
                description=(
                    "Forecast event risk with model tuning. "
                    "Slower than default but provides more accurate probability estimates. "
                    "Tunes the forecasting model specifically for event risk prediction."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Wide format time series data"
                        },
                        "forecast_length": {
                            "type": "integer",
                            "description": "Number of periods to forecast",
                            "default": 30
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Threshold value to monitor"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["upper", "lower"],
                            "default": "upper",
                            "description": "Whether to detect crossing above or below threshold"
                        }
                    },
                    "required": ["data", "threshold"]
                }
            ),
            
            # Prediction Object Management Tools
            Tool(
                name="list_predictions",
                description=(
                    "List all cached prediction objects with their IDs and metadata. "
                    "Use this to see available predictions that can be further analyzed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="get_prediction_forecast",
                description=(
                    "Get the forecast data from a cached prediction as JSON. "
                    "This is the primary way to retrieve forecast values after running a forecast."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prediction_id": {
                            "type": "string",
                            "description": "ID of the cached prediction"
                        },
                        "output": {
                            "type": "string",
                            "enum": ["forecast", "upper_forecast", "lower_forecast"],
                            "default": "forecast",
                            "description": "Which forecast to return (point, upper bound, or lower bound)"
                        }
                    },
                    "required": ["prediction_id"]
                }
            ),
            Tool(
                name="plot_prediction",
                description=(
                    "Generate a plot of the forecast from a cached prediction. "
                    "Returns a base64-encoded PNG image of the forecast visualization."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prediction_id": {
                            "type": "string",
                            "description": "ID of the cached prediction"
                        },
                        "include_history": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include historical data in the plot"
                        },
                        "series": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific series to plot (default: all series)"
                        }
                    },
                    "required": ["prediction_id"]
                }
            ),
            Tool(
                name="apply_constraints",
                description=(
                    "Apply constraints to a cached forecast (e.g., enforce bounds, dampening). "
                    "Returns a new prediction ID with the constrained forecast."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prediction_id": {
                            "type": "string",
                            "description": "ID of the cached prediction"
                        },
                        "constraint_method": {
                            "type": "string",
                            "enum": ["dampen", "upper", "lower", "quantile"],
                            "description": "Type of constraint to apply"
                        },
                        "constraint_value": {
                            "type": "number",
                            "description": "Value for the constraint (e.g., dampening factor, upper/lower bound)"
                        },
                        "constraint_direction": {
                            "type": "string",
                            "enum": ["upper", "lower"],
                            "description": "Direction for bound constraints"
                        }
                    },
                    "required": ["prediction_id", "constraint_method"]
                }
            ),
            Tool(
                name="get_prediction_components",
                description=(
                    "Get decomposed forecast components (trend, seasonality, etc.) if available. "
                    "Only works with models that support decomposition (e.g., Cassandra, TVVAR)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prediction_id": {
                            "type": "string",
                            "description": "ID of the cached prediction"
                        }
                    },
                    "required": ["prediction_id"]
                }
            ),
            Tool(
                name="clear_predictions",
                description=(
                    "Clear cached predictions to free memory. "
                    "Can clear a specific prediction or all predictions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prediction_id": {
                            "type": "string",
                            "description": "ID of prediction to clear (omit to clear all)"
                        }
                    }
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
            if name == "get_sample_data":
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
                
                # Convert to JSON
                if long:
                    result = dataframe_to_json(df, data_format="long")
                else:
                    result = dataframe_to_json(df, data_format="wide")
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "dataset": dataset,
                        "format": "long" if long else "wide",
                        "shape": {"rows": len(df), "columns": len(df.columns) if not long else "3"},
                        "data": result
                    }, indent=2)
                )]
            
            elif name == "load_live_daily":
                fred_key = arguments.get("fred_key")
                fred_series = arguments.get("fred_series")
                tickers = arguments.get("tickers")
                long = arguments.get("long", False)
                
                df = load_live_daily(
                    long=long,
                    fred_key=fred_key,
                    fred_series=fred_series,
                    tickers=tickers
                )
                
                result = dataframe_to_json(df, data_format="long" if long else "wide")
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "source": "live_daily",
                        "format": "long" if long else "wide",
                        "shape": {"rows": len(df), "columns": len(df.columns)},
                        "data": result
                    }, indent=2)
                )]
            
            elif name == "generate_synthetic_data":
                n_series = arguments.get("n_series", 5)
                
                generator = SyntheticDailyGenerator(
                    n_series=n_series,
                    random_seed=42
                )
                # Data is automatically generated in __init__
                df = generator.data
                template = generator.template
                
                result = dataframe_to_json(df, data_format="wide")
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "source": "synthetic",
                        "n_series": n_series,
                        "shape": {"rows": len(df), "columns": len(df.columns)},
                        "data": result,
                        "labels": {
                            "info": "Generated with labeled components for evaluation",
                            "template_available": True,
                            "series_types": list(template.get('meta', {}).get('series_type_descriptions', {}).keys())
                        }
                    }, indent=2)
                )]
            
            elif name == "long_to_wide_converter":
                data = arguments.get("data")
                df_long = json_to_dataframe(data, data_format="long")
                df_wide = long_to_wide(
                    df_long.reset_index(),
                    date_col='datetime',
                    value_col='value',
                    id_col='series_id'
                )
                
                result = dataframe_to_json(df_wide, data_format="wide")
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "format": "wide",
                        "shape": {"rows": len(df_wide), "columns": len(df_wide.columns)},
                        "data": result
                    }, indent=2)
                )]
            
            elif name == "forecast_mosaic_profile":
                data = arguments.get("data")
                forecast_length = arguments.get("forecast_length", 30)
                profile_template = arguments.get("profile_template")
                
                df = json_to_dataframe(data, data_format="wide")
                
                # Check if data length is sufficient
                if len(df) < forecast_length:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": f"Insufficient data. Need at least {forecast_length} rows, got {len(df)}. Use forecast_autots_search instead."
                        })
                    )]
                
                # Use provided profile or load default
                if profile_template is None:
                    # Load the mosaic_profile_template.json from the mcp folder
                    import os
                    from os.path import dirname, join
                    template_path = join(dirname(__file__), 'mosaic_profile_template.json')
                    if os.path.exists(template_path):
                        with open(template_path, 'r') as f:
                            profile_template = json.load(f)
                    else:
                        return [TextContent(
                            type="text",
                            text=json.dumps({
                                "error": "No profile template provided and default not found"
                            })
                        )]
                
                # Initialize AutoTS with mosaic profile
                model = AutoTS(
                    forecast_length=forecast_length,
                    frequency='infer',
                    ensemble='mosaic',
                    model_list='no_shared',
                    transformer_list='fast',
                    max_generations=0,
                    num_validations=0,
                    validation_method='backwards'
                )
                
                # Import the template
                model.import_template(
                    profile_template,
                    method='only',
                    enforce_model_list=True
                )
                
                # Fit and predict
                model = model.fit(df)
                prediction = model.predict()
                
                # Cache the prediction object
                prediction_id = cache_prediction(
                    prediction_obj=prediction,
                    model_obj=model,
                    metadata={
                        "method": "mosaic_profile",
                        "forecast_length": forecast_length,
                        "model_name": prediction.model_name,
                        "series_count": len(df.columns),
                        "data_points": len(df)
                    }
                )
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "prediction_id": prediction_id,
                        "method": "mosaic_profile",
                        "forecast_length": forecast_length,
                        "model_name": prediction.model_name,
                        "series": list(df.columns),
                        "message": f"Forecast created. Use 'get_prediction_forecast' with prediction_id to retrieve forecast data, or 'plot_prediction' to visualize."
                    }, indent=2)
                )]
            
            elif name == "forecast_autots_search":
                data = arguments.get("data")
                forecast_length = arguments.get("forecast_length", 30)
                
                df = json_to_dataframe(data, data_format="wide")
                
                # Fast AutoTS search with limited models for explainability
                model = AutoTS(
                    forecast_length=forecast_length,
                    frequency='infer',
                    ensemble=None,  # No ensembles for explainability
                    model_list=['Cassandra', 'TVVAR', 'BasicLinearModel'],
                    transformer_list='fast',
                    max_generations=3,
                    num_validations=2,
                    validation_method='backwards',
                    models_to_validate=0.2,
                    n_jobs='auto'
                )
                
                model = model.fit(df)
                prediction = model.predict()
                
                # Cache the prediction object
                prediction_id = cache_prediction(
                    prediction_obj=prediction,
                    model_obj=model,
                    metadata={
                        "method": "autots_search",
                        "forecast_length": forecast_length,
                        "model_name": prediction.model_name,
                        "series_count": len(df.columns),
                        "data_points": len(df),
                        "explainable": True
                    }
                )
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "prediction_id": prediction_id,
                        "method": "autots_search",
                        "forecast_length": forecast_length,
                        "model_name": prediction.model_name,
                        "model_params": str(prediction.model_parameters),
                        "transformation": str(prediction.transformation_parameters),
                        "series": list(df.columns),
                        "message": f"Forecast created with explainable model. Use 'get_prediction_forecast' to retrieve data, 'plot_prediction' to visualize, or 'get_prediction_components' for decomposition."
                    }, indent=2)
                )]
            
            elif name == "forecast_custom":
                data = arguments.get("data")
                forecast_length = arguments.get("forecast_length", 30)
                autots_params = arguments.get("autots_params", {})
                model_template = arguments.get("model_template")
                
                df = json_to_dataframe(data, data_format="wide")
                
                # Build AutoTS parameters
                params = {
                    "forecast_length": forecast_length,
                    "frequency": "infer",
                    **autots_params
                }
                
                model = AutoTS(**params)
                
                # Import template if provided
                if model_template is not None:
                    model.import_template(model_template, method='only')
                
                model = model.fit(df)
                prediction = model.predict()
                
                # Cache the prediction object
                prediction_id = cache_prediction(
                    prediction_obj=prediction,
                    model_obj=model,
                    metadata={
                        "method": "custom",
                        "forecast_length": forecast_length,
                        "model_name": prediction.model_name,
                        "series_count": len(df.columns),
                        "data_points": len(df),
                        "custom_params": autots_params
                    }
                )
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "prediction_id": prediction_id,
                        "method": "custom",
                        "forecast_length": forecast_length,
                        "model_name": prediction.model_name,
                        "series": list(df.columns),
                        "message": f"Custom forecast created. Use 'get_prediction_forecast' to retrieve data or 'plot_prediction' to visualize."
                    }, indent=2)
                )]
            
            elif name == "get_autots_docs":
                docs = {
                    "AutoTS Parameters": {
                        "forecast_length": "Number of periods to forecast (required)",
                        "frequency": "Pandas frequency string ('D', 'H', 'W', 'MS', etc.) or 'infer'",
                        "ensemble": "Ensemble method: 'simple', 'distance', 'horizontal', 'mosaic', None",
                        "model_list": "List of models or preset: 'fast', 'superfast', 'all', 'default'",
                        "transformer_list": "Transformations: 'fast', 'superfast', 'all'",
                        "max_generations": "Number of genetic algorithm generations",
                        "num_validations": "Number of cross-validation splits",
                        "validation_method": "'backwards', 'even', 'seasonal', etc.",
                        "models_to_validate": "Fraction of models to fully validate (0.0-1.0)",
                        "n_jobs": "Parallel processes: 'auto', -1, or specific number"
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
                
                return [TextContent(
                    type="text",
                    text=json.dumps(docs, indent=2)
                )]
            
            elif name == "detect_features":
                data = arguments.get("data")
                df = json_to_dataframe(data, data_format="wide")
                
                # Run feature detection
                detector = TimeSeriesFeatureDetector()
                detector.fit(df)
                
                # Get detected features using the detector's public API
                features = detector.get_detected_features(
                    include_components=False,
                    include_metadata=True
                )
                
                # Convert all Timestamp objects to strings for JSON serialization
                def serialize_timestamps(obj):
                    """Recursively convert Timestamp objects to strings."""
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
                
                features_serialized = serialize_timestamps(features)
                
                # Add summary statistics
                summary = {
                    "date_range": {
                        "start": detector.date_index[0].strftime('%Y-%m-%d'),
                        "end": detector.date_index[-1].strftime('%Y-%m-%d')
                    },
                    "num_series": len(detector.df_original.columns),
                    "num_observations": len(detector.df_original),
                    "series_names": list(detector.df_original.columns),
                }
                
                # Count detections per series
                detection_counts = {}
                for series_name in detector.df_original.columns:
                    detection_counts[series_name] = {
                        "trend_changepoints": len(detector.trend_changepoints.get(series_name, [])),
                        "level_shifts": len(detector.level_shifts.get(series_name, [])),
                        "anomalies": len(detector.anomalies.get(series_name, [])),
                        "holidays": len(detector.holiday_dates.get(series_name, [])),
                        "seasonality_strength": float(detector.seasonality_strength.get(series_name, 0.0)),
                    }
                
                results = {
                    "summary": summary,
                    "detection_counts": detection_counts,
                    "features": features_serialized,
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )]
            
            elif name == "get_cleaned_data":
                data = arguments.get("data")
                fillna = arguments.get("fillna", "ffill")
                
                df = json_to_dataframe(data, data_format="wide")
                
                # Apply cleaning transformation
                transformer = GeneralTransformer(
                    fillna=fillna,
                    transformations={"0": "ClipOutliers"},
                    transformation_params={"0": {}}
                )
                
                df_cleaned = transformer.fit_transform(df)
                result = dataframe_to_json(df_cleaned, data_format="wide")
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "cleaned": True,
                        "method": fillna,
                        "shape": {"rows": len(df_cleaned), "columns": len(df_cleaned.columns)},
                        "data": result
                    }, indent=2)
                )]
            
            elif name == "forecast_event_risk_default":
                data = arguments.get("data")
                forecast_length = arguments.get("forecast_length", 30)
                threshold = arguments.get("threshold")
                direction = arguments.get("direction", "upper")
                
                df = json_to_dataframe(data, data_format="wide")
                
                # Convert absolute threshold to quantile
                # Find which quantile this threshold represents in the historical data
                if threshold is not None:
                    # Calculate quantile for each series
                    quantiles = []
                    for col in df.columns:
                        series_vals = df[col].dropna()
                        if len(series_vals) > 0:
                            q = (series_vals <= threshold).sum() / len(series_vals)
                            quantiles.append(q)
                    # Use average quantile across series
                    quantile_limit = np.mean(quantiles) if quantiles else 0.5
                    # Ensure it's in valid range
                    quantile_limit = max(0.01, min(0.99, quantile_limit))
                else:
                    # Default quantiles
                    quantile_limit = 0.95 if direction == "upper" else 0.05
                
                # Determine quantile limit from threshold
                upper_limit = quantile_limit if direction == "upper" else None
                lower_limit = quantile_limit if direction == "lower" else None
                
                # Initialize EventRiskForecast with defaults
                erf = EventRiskForecast(
                    df_train=df,
                    forecast_length=forecast_length,
                    frequency='infer',
                    upper_limit=upper_limit,
                    lower_limit=lower_limit
                )
                
                # Fit the model
                erf.fit()
                
                # Generate predictions - this returns (upper_risk_df, lower_risk_df)
                upper_risk_df, lower_risk_df = erf.predict()
                
                # Get the relevant risk dataframe
                risk_df = upper_risk_df if direction == "upper" else lower_risk_df
                
                # Convert risk array to serializable format
                risk_data = {
                    "threshold": threshold,
                    "threshold_quantile": quantile_limit,
                    "direction": direction,
                    "probabilities": dataframe_to_json(risk_df, data_format="wide") if risk_df is not None else None
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps(risk_data, indent=2)
                )]
            
            elif name == "forecast_event_risk_tuning":
                data = arguments.get("data")
                forecast_length = arguments.get("forecast_length", 30)
                threshold = arguments.get("threshold")
                direction = arguments.get("direction", "upper")
                
                df = json_to_dataframe(data, data_format="wide")
                
                # Convert absolute threshold to quantile
                if threshold is not None:
                    quantiles = []
                    for col in df.columns:
                        series_vals = df[col].dropna()
                        if len(series_vals) > 0:
                            q = (series_vals <= threshold).sum() / len(series_vals)
                            quantiles.append(q)
                    quantile_limit = np.mean(quantiles) if quantiles else 0.5
                    quantile_limit = max(0.01, min(0.99, quantile_limit))
                else:
                    quantile_limit = 0.95 if direction == "upper" else 0.05
                
                # Determine limit configuration
                upper_limit = quantile_limit if direction == "upper" else None
                lower_limit = quantile_limit if direction == "lower" else None
                
                # Initialize with tuning parameters
                erf = EventRiskForecast(
                    df_train=df,
                    forecast_length=forecast_length,
                    frequency='infer',
                    upper_limit=upper_limit,
                    lower_limit=lower_limit,
                    # Tuning parameters for better accuracy
                    model_name='BallTreeMultivariateMotif',
                    model_param_dict={
                        'window': 5,
                        'point_method': 'median',
                        'distance_metric': 'canberra',
                        'k': 10
                    }
                )
                
                erf.fit()
                upper_risk_df, lower_risk_df = erf.predict()
                
                # Get the relevant risk dataframe
                risk_df = upper_risk_df if direction == "upper" else lower_risk_df
                
                risk_data = {
                    "threshold": threshold,
                    "threshold_quantile": quantile_limit,
                    "direction": direction,
                    "method": "tuned",
                    "probabilities": dataframe_to_json(risk_df, data_format="wide") if risk_df is not None else None
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps(risk_data, indent=2)
                )]
            
            elif name == "list_predictions":
                predictions = list_cached_predictions()
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "cached_predictions": predictions,
                        "count": len(predictions)
                    }, indent=2)
                )]
            
            elif name == "get_prediction_forecast":
                prediction_id = arguments.get("prediction_id")
                output = arguments.get("output", "forecast")
                
                cached = get_cached_prediction(prediction_id)
                prediction = cached['prediction']
                
                # Get the requested output
                if output == "forecast":
                    df = prediction.forecast
                elif output == "upper_forecast":
                    df = prediction.upper_forecast
                elif output == "lower_forecast":
                    df = prediction.lower_forecast
                else:
                    raise ValueError(f"Unknown output type: {output}")
                
                result = dataframe_to_json(df, data_format="wide")
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "prediction_id": prediction_id,
                        "output_type": output,
                        "model_name": prediction.model_name,
                        "metadata": cached['metadata'],
                        "forecast": result
                    }, indent=2)
                )]
            
            elif name == "plot_prediction":
                prediction_id = arguments.get("prediction_id")
                include_history = arguments.get("include_history", True)
                series = arguments.get("series")
                
                cached = get_cached_prediction(prediction_id)
                prediction = cached['prediction']
                model = cached.get('model')
                
                # Import matplotlib
                try:
                    import matplotlib
                    matplotlib.use('Agg')  # Non-interactive backend
                    import matplotlib.pyplot as plt
                except ImportError:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": "Matplotlib not installed. Install with: pip install matplotlib"
                        })
                    )]
                
                # Create the plot
                fig = plt.figure(figsize=(12, 6))
                
                forecast_df = prediction.forecast
                
                # Filter series if specified
                if series:
                    forecast_df = forecast_df[series]
                
                # Plot forecast
                for col in forecast_df.columns:
                    plt.plot(forecast_df.index, forecast_df[col], label=f'{col} (forecast)', linewidth=2)
                
                # Plot historical data if available and requested
                if include_history and model is not None:
                    try:
                        if hasattr(model, 'df_wide_numeric'):
                            hist_df = model.df_wide_numeric
                            if series:
                                hist_df = hist_df[series]
                            for col in hist_df.columns:
                                plt.plot(hist_df.index, hist_df[col], label=f'{col} (history)', 
                                        alpha=0.6, linestyle='--')
                    except:
                        pass  # Skip if historical data not available
                
                # Plot confidence intervals if available
                if prediction.upper_forecast is not None:
                    upper_df = prediction.upper_forecast
                    lower_df = prediction.lower_forecast
                    if series:
                        upper_df = upper_df[series]
                        lower_df = lower_df[series]
                    for col in upper_df.columns:
                        plt.fill_between(
                            upper_df.index,
                            lower_df[col],
                            upper_df[col],
                            alpha=0.2
                        )
                
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.title(f'Forecast - {prediction.model_name}')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode()
                plt.close(fig)
                
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "prediction_id": prediction_id,
                            "model_name": prediction.model_name,
                            "series_plotted": list(forecast_df.columns),
                            "image_format": "png"
                        }, indent=2)
                    ),
                    ImageContent(
                        type="image",
                        data=image_base64,
                        mimeType="image/png"
                    )
                ]
            
            elif name == "apply_constraints":
                prediction_id = arguments.get("prediction_id")
                constraint_method = arguments.get("constraint_method")
                constraint_value = arguments.get("constraint_value")
                constraint_direction = arguments.get("constraint_direction")
                
                cached = get_cached_prediction(prediction_id)
                prediction = cached['prediction']
                
                # Apply constraints to the forecast
                forecast_df = prediction.forecast.copy()
                
                if constraint_method == "dampen":
                    # Apply dampening to forecast
                    dampen_factor = constraint_value if constraint_value else 0.9
                    # Simple dampening: reduce deviation from last historical value
                    for col in forecast_df.columns:
                        last_val = forecast_df[col].iloc[0]
                        for i in range(len(forecast_df)):
                            forecast_df[col].iloc[i] = last_val + (forecast_df[col].iloc[i] - last_val) * (dampen_factor ** i)
                
                elif constraint_method in ["upper", "lower"]:
                    # Apply upper/lower bounds
                    if constraint_value is None:
                        raise ValueError("constraint_value required for upper/lower bounds")
                    if constraint_method == "upper":
                        forecast_df = forecast_df.clip(upper=constraint_value)
                    else:
                        forecast_df = forecast_df.clip(lower=constraint_value)
                
                elif constraint_method == "quantile":
                    # Constraint to historical quantile
                    if constraint_value is None:
                        raise ValueError("constraint_value (quantile) required")
                    # This would need historical data from model
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": "Quantile constraints require historical data (not yet implemented)"
                        })
                    )]
                
                # Create a new prediction-like object with constrained forecast
                # We'll cache this as a new prediction
                class ConstrainedPrediction:
                    def __init__(self, original_prediction, constrained_forecast):
                        self.forecast = constrained_forecast
                        self.upper_forecast = original_prediction.upper_forecast
                        self.lower_forecast = original_prediction.lower_forecast
                        self.model_name = f"{original_prediction.model_name} (constrained)"
                        self.model_parameters = original_prediction.model_parameters
                        self.transformation_parameters = original_prediction.transformation_parameters
                
                constrained_pred = ConstrainedPrediction(prediction, forecast_df)
                
                # Cache the constrained prediction
                new_prediction_id = cache_prediction(
                    prediction_obj=constrained_pred,
                    metadata={
                        **cached['metadata'],
                        "constrained": True,
                        "constraint_method": constraint_method,
                        "constraint_value": constraint_value,
                        "original_prediction_id": prediction_id
                    }
                )
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "new_prediction_id": new_prediction_id,
                        "original_prediction_id": prediction_id,
                        "constraint_applied": constraint_method,
                        "constraint_value": constraint_value,
                        "message": "Constraints applied. Use 'get_prediction_forecast' with new_prediction_id to retrieve constrained forecast."
                    }, indent=2)
                )]
            
            elif name == "get_prediction_components":
                prediction_id = arguments.get("prediction_id")
                
                cached = get_cached_prediction(prediction_id)
                prediction = cached['prediction']
                model = cached.get('model')
                
                # Try to get decomposition/components
                components = {}
                
                # Check if model supports component extraction
                if hasattr(prediction, 'model_parameters'):
                    model_name = prediction.model_name
                    
                    # For models like Cassandra that have components
                    if 'cassandra' in model_name.lower():
                        # Try to access Cassandra components if available
                        try:
                            if hasattr(model, 'best_model'):
                                best_model = model.best_model
                                if hasattr(best_model, 'seasonal_coef'):
                                    components['seasonality'] = "Available (Cassandra model)"
                                if hasattr(best_model, 'trend_coef'):
                                    components['trend'] = "Available (Cassandra model)"
                        except:
                            pass
                    
                    # For seasonal decomposition models
                    if 'seasonal' in model_name.lower():
                        components['info'] = "Seasonal decomposition model - components embedded in forecast"
                
                if not components:
                    components['message'] = f"Model {prediction.model_name} does not expose decomposed components via MCP. Consider using models like Cassandra or TVVAR for component extraction."
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "prediction_id": prediction_id,
                        "model_name": prediction.model_name,
                        "components": components,
                        "note": "Full component extraction is model-specific. This feature is under development."
                    }, indent=2)
                )]
            
            elif name == "clear_predictions":
                prediction_id = arguments.get("prediction_id")
                
                if prediction_id:
                    clear_prediction_cache(prediction_id)
                    message = f"Cleared prediction {prediction_id}"
                else:
                    clear_prediction_cache()
                    message = "Cleared all cached predictions"
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": message
                    }, indent=2)
                )]
            
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"})
                )]
        
        except Exception as e:
            logger.exception(f"Error in tool {name}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": str(e),
                    "tool": name,
                    "traceback": str(e.__traceback__)
                }, indent=2)
            )]

    @app.list_resources()
    async def list_resources() -> list[Any]:
        """List available documentation resources."""
        import os
        from os.path import dirname, join, exists
        
        resources = []
        base_path = dirname(dirname(dirname(__file__)))
        mcp_path = dirname(__file__)
        
        # Documentation files to expose
        doc_files = [
            ("extended_tutorial.md", "Extended AutoTS Tutorial", base_path),
            ("production_example.py", "Production Example Script", base_path),
            ("README.md", "AutoTS README", base_path),
            ("docs/metric_weighting_guide.md", "Metric Weighting Guide", base_path),
            ("docs/catlin_m6_paper.tex", "M6 Forecasting Competition Paper (LaTeX)", base_path),
            ("QUICK_REFERENCE.md", "⭐ MCP Quick Reference - START HERE", mcp_path),
            ("PREDICTION_CACHING_GUIDE.md", "MCP Prediction Caching Pattern Guide", mcp_path),
            ("ARCHITECTURE_DIAGRAM.md", "MCP Architecture and Data Flow Diagrams", mcp_path),
            ("IMPLEMENTATION_SUMMARY.md", "MCP Implementation Summary", mcp_path),
            ("example_prediction_workflow.py", "MCP Prediction Workflow Examples", mcp_path),
        ]
        
        for filename, description, base in doc_files:
            filepath = join(base, filename)
            if exists(filepath):
                # Determine MIME type
                if filename.endswith(".md"):
                    mime_type = "text/markdown"
                elif filename.endswith(".py"):
                    mime_type = "text/x-python"
                elif filename.endswith(".tex"):
                    mime_type = "text/x-tex"
                else:
                    mime_type = "text/plain"
                
                resources.append({
                    "uri": f"file://{filepath}",
                    "name": filename,
                    "description": description,
                    "mimeType": mime_type
                })
        
        return resources

    @app.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a documentation resource."""
        if uri.startswith("file://"):
            filepath = uri[7:]  # Remove "file://"
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except Exception as e:
                return f"Error reading {filepath}: {str(e)}"
        else:
            return f"Unknown resource URI: {uri}"


def serve():
    """Start the MCP server."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP package not installed. Install with: pip install autots[mcp]"
        )
    
    import asyncio
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    
    asyncio.run(main())


if __name__ == "__main__":
    serve()
