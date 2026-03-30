"""MCP Tool schemas for cache management and data loading/conversion tools."""

try:
    from mcp.types import Tool, ToolAnnotations

    DATA_TOOLS = [
        # ----------------------------------------------------------------
        # Cache management
        # ----------------------------------------------------------------
        Tool(
            name="list_cache",
            title="List Cached Objects",
            description="List all cached objects across all cache types (predictions, autots_models, event_risk, feature_detectors, data). Call this to discover existing IDs before calling any get_*, plot_*, or apply_* tools.",
            inputSchema={"type": "object", "additionalProperties": False},
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        Tool(
            name="clear_cache",
            title="Clear Cache",
            description="Clear cache: specific object by ID+type, an entire cache type, or all caches. Destructive — freed objects cannot be recovered.",
            inputSchema={
                "type": "object",
                "properties": {
                    "object_id": {
                        "type": "string",
                        "description": "Specific object ID to clear",
                    },
                    "cache_type": {
                        "type": "string",
                        "enum": [
                            "prediction",
                            "autots",
                            "event_risk",
                            "feature_detector",
                            "data",
                        ],
                        "description": "Cache type to clear (omit both params to clear all)",
                    },
                },
            },
            outputSchema={
                "type": "object",
                "properties": {"success": {"type": "boolean"}},
                "required": ["success"],
            },
            annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True),
        ),
        # ----------------------------------------------------------------
        # Data loading
        # ----------------------------------------------------------------
        Tool(
            name="load_sample_data",
            title="Load Sample Dataset",
            description="Load a built-in sample time series dataset. Returns data_id for use as the data_id parameter in forecast_*, detect_features, forecast_event_risk, clean_data, and get_data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "enum": [
                            "daily",
                            "hourly",
                            "weekly",
                            "monthly",
                            "yearly",
                            "linear",
                            "sine",
                            "artificial",
                        ],
                        "default": "daily",
                        "description": "Sample dataset to load",
                    },
                    "long": {
                        "type": "boolean",
                        "default": False,
                        "description": "Return long format (default: wide)",
                    },
                },
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "Cache ID — pass as data_id to other tools",
                    },
                    "source": {"type": "string"},
                    "rows": {"type": "integer"},
                    "cols": {"type": "integer"},
                },
                "required": ["data_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
        Tool(
            name="load_live_data",
            title="Load Live Data (FRED / Stocks)",
            description="Load live data from external sources (FRED economic data, stock tickers). Returns data_id. Requires network access.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fred_key": {"type": "string", "description": "FRED API key"},
                    "fred_series": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "FRED series codes",
                    },
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stock tickers",
                    },
                    "long": {
                        "type": "boolean",
                        "default": False,
                        "description": "Return long format",
                    },
                },
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "Cache ID — pass as data_id to other tools",
                    },
                    "rows": {"type": "integer"},
                    "cols": {"type": "integer"},
                },
                "required": ["data_id"],
            },
            annotations=ToolAnnotations(
                readOnlyHint=False, idempotentHint=False, openWorldHint=True
            ),
        ),
        Tool(
            name="generate_synthetic_data",
            title="Generate Synthetic Time Series",
            description="Generate synthetic time series data with labeled components for testing. Returns data_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n_series": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of series to generate",
                    }
                },
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "Cache ID — pass as data_id to other tools",
                    },
                    "n_series": {"type": "integer"},
                    "rows": {"type": "integer"},
                    "cols": {"type": "integer"},
                },
                "required": ["data_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
        Tool(
            name="load_data_from_file",
            title="Load Data from File or URL",
            description="Load a CSV from a local file path or URL. Returns data_id. CSV must have a datetime column as the first column (index).",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Local file path or URL to CSV",
                    }
                },
                "required": ["filepath"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "Cache ID — pass as data_id to other tools",
                    },
                    "source": {"type": "string"},
                    "rows": {"type": "integer"},
                    "cols": {"type": "integer"},
                },
                "required": ["data_id"],
            },
            annotations=ToolAnnotations(
                readOnlyHint=False, idempotentHint=False, openWorldHint=True
            ),
        ),
        Tool(
            name="get_data",
            title="Get Cached Data",
            description="Retrieve cached data as JSON (wide or long) or save as CSV. Requires data_id from load_sample_data, load_live_data, load_data_from_file, generate_synthetic_data, convert_long_to_wide, or clean_data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_id": {"type": "string", "description": "Cached data ID"},
                    "output_format": {
                        "type": "string",
                        "enum": ["json_wide", "json_long", "csv_wide", "csv_long"],
                        "default": "json_wide",
                        "description": "Output format",
                    },
                },
                "required": ["data_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        Tool(
            name="convert_long_to_wide",
            title="Convert Long Format to Wide",
            description="Convert long-format data (datetime, series_id, value columns) to wide format. Provide either data (inline dict) or data_id. Returns new data_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Long format data with datetime, series_id, value columns",
                    },
                    "data_id": {
                        "type": "string",
                        "description": "Cached data ID (alternative to data)",
                    },
                },
                "oneOf": [{"required": ["data"]}, {"required": ["data_id"]}],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "Cache ID for converted wide-format data",
                    },
                    "rows": {"type": "integer"},
                    "cols": {"type": "integer"},
                },
                "required": ["data_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
        Tool(
            name="clean_data",
            title="Clean Time Series Data",
            description="Clean time series data: fill missing values and handle outliers. Provide either data (inline dict) or data_id. Returns new data_id with cleaned data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "object", "description": "Wide format data"},
                    "data_id": {"type": "string", "description": "Cached data ID"},
                    "fillna": {
                        "type": "string",
                        "enum": [
                            "ffill",
                            "mean",
                            "median",
                            "rolling_mean",
                            "linear",
                        ],
                        "default": "ffill",
                        "description": "Missing value fill method",
                    },
                },
                "oneOf": [{"required": ["data"]}, {"required": ["data_id"]}],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "Cache ID for cleaned data",
                    },
                    "rows": {"type": "integer"},
                    "cols": {"type": "integer"},
                },
                "required": ["data_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
    ]

except ImportError:
    DATA_TOOLS = []
