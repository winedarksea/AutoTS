"""MCP Tool schemas for event risk forecasting and feature detection tools."""

try:
    from mcp.types import Tool, ToolAnnotations

    FEATURES_TOOLS = [
        # ----------------------------------------------------------------
        # Event Risk tools
        # ----------------------------------------------------------------
        Tool(
            name="forecast_event_risk",
            title="Forecast Event Risk Probabilities",
            description="Forecast the probability of crossing a threshold over future periods (e.g., stockout risk, capacity breach). Threshold in [0,1] is treated as a historical quantile; values outside [0,1] are absolute. Provide data or data_id. Returns event_risk_id for use in get_event_risk_results and plot_event_risk.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "object", "description": "Wide format data"},
                    "data_id": {"type": "string", "description": "Cached data ID"},
                    "forecast_length": {
                        "type": "integer",
                        "default": 30,
                        "description": "Periods to forecast",
                    },
                    "threshold": {
                        "oneOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": "number"}},
                        ],
                        "description": "Threshold value (required). Float in [0,1] = historical quantile. Outside [0,1] = absolute threshold. Can be 2D array of shape (forecast_length, num_series).",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["upper", "lower"],
                        "default": "upper",
                        "description": "Detect crossing above (upper) or below (lower) the threshold",
                    },
                    "tune": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable model tuning (slower but more accurate)",
                    },
                },
                "required": ["threshold"],
                "oneOf": [{"required": ["data"]}, {"required": ["data_id"]}],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "event_risk_id": {
                        "type": "string",
                        "description": "Cache ID — pass to get_event_risk_results or plot_event_risk",
                    },
                    "threshold": {},
                    "direction": {"type": "string"},
                    "forecast_length": {"type": "integer"},
                },
                "required": ["event_risk_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
        Tool(
            name="get_event_risk_results",
            title="Get Event Risk Probabilities",
            description="Get event risk probability values from a cached EventRiskForecast. Requires event_risk_id from forecast_event_risk. Returns probabilities per period per series.",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_risk_id": {
                        "type": "string",
                        "description": "Cached event risk ID",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json_wide", "json_long", "csv_wide", "csv_long"],
                        "default": "json_wide",
                        "description": "Output format",
                    },
                },
                "required": ["event_risk_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        Tool(
            name="plot_event_risk",
            title="Plot Event Risk Probabilities",
            description="Plot event risk probabilities over the forecast horizon. Requires event_risk_id from forecast_event_risk. Returns base64-encoded PNG.",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_risk_id": {
                        "type": "string",
                        "description": "Cached event risk ID",
                    }
                },
                "required": ["event_risk_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        # ----------------------------------------------------------------
        # Feature detection tools
        # ----------------------------------------------------------------
        Tool(
            name="detect_features",
            title="Detect Time Series Features",
            description="Detect anomalies, changepoints, level shifts, holidays, and seasonality patterns across all series. Provide data or data_id. Returns detector_id for use in get_detected_features, plot_features, and forecast_from_features.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "object", "description": "Wide format data"},
                    "data_id": {"type": "string", "description": "Cached data ID"},
                },
                "oneOf": [{"required": ["data"]}, {"required": ["data_id"]}],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "detector_id": {
                        "type": "string",
                        "description": "Cache ID — pass to get_detected_features, plot_features, forecast_from_features",
                    },
                    "series_count": {"type": "integer"},
                },
                "required": ["detector_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
        Tool(
            name="get_detected_features",
            title="Get Detected Features",
            description="Query detected features (anomalies, changepoints, level shifts, holidays, seasonality) from a cached detector. Requires detector_id from detect_features. Supports filtering by date range, specific date, or series name. Use include_components=true for time-series component values. Examples: 'was there an anomaly on 2024-12-25?', 'when was the first level shift?'",
            inputSchema={
                "type": "object",
                "properties": {
                    "detector_id": {
                        "type": "string",
                        "description": "Cached detector ID",
                    },
                    "date_start": {
                        "type": "string",
                        "description": "Optional start date filter (YYYY-MM-DD)",
                    },
                    "date_end": {
                        "type": "string",
                        "description": "Optional end date filter (YYYY-MM-DD)",
                    },
                    "specific_date": {
                        "type": "string",
                        "description": "Optional single date to query (YYYY-MM-DD)",
                    },
                    "series_name": {
                        "type": "string",
                        "description": "Optional series name to filter results",
                    },
                    "include_components": {
                        "type": "boolean",
                        "description": "Include component time-series values (default: false)",
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include metadata like noise levels and scales (default: false)",
                    },
                },
                "required": ["detector_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        Tool(
            name="plot_features",
            title="Plot Detected Features",
            description="Plot detected features overlaid on the time series. Requires detector_id from detect_features. Returns base64-encoded PNG.",
            inputSchema={
                "type": "object",
                "properties": {
                    "detector_id": {
                        "type": "string",
                        "description": "Cached detector ID",
                    },
                    "series": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific series to plot",
                    },
                },
                "required": ["detector_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        Tool(
            name="forecast_from_features",
            title="Forecast from Detected Features (Experimental)",
            description="EXPERIMENTAL: Create a forecast using the decomposed components from a feature detector. Only use after detect_features, not as a standalone forecasting method. Requires detector_id from detect_features. Returns prediction_id for use in get_forecast, plot_forecast.",
            inputSchema={
                "type": "object",
                "properties": {
                    "detector_id": {
                        "type": "string",
                        "description": "Cached detector ID",
                    },
                    "forecast_length": {
                        "type": "integer",
                        "default": 30,
                        "description": "Periods to forecast",
                    },
                },
                "required": ["detector_id"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cache ID — pass to get_forecast or plot_forecast",
                    },
                    "forecast_length": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "required": ["prediction_id", "forecast_length"],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
    ]

except ImportError:
    FEATURES_TOOLS = []
