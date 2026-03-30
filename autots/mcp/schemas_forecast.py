"""MCP Tool schemas for forecasting, prediction, event risk, and feature detection tools."""

try:
    from mcp.types import Tool, ToolAnnotations

    FORECAST_TOOLS = [
        # ----------------------------------------------------------------
        # Forecasting
        # ----------------------------------------------------------------
        Tool(
            name="forecast_fast",
            title="Fast Mosaic Ensemble Forecast",
            description="FAST: Pre-configured mosaic ensemble forecast using fit_data (no model search). Use for quick results. Provide data or data_id. Returns prediction_id for use in get_forecast, plot_forecast, apply_constraints, apply_adjustments, get_model_params.",
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
                    "profile_template": {
                        "type": "object",
                        "description": "Optional custom mosaic profile JSON",
                    },
                },
                "oneOf": [{"required": ["data"]}, {"required": ["data_id"]}],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cache ID — pass to get_forecast, plot_forecast, apply_constraints, apply_adjustments, get_model_params",
                    },
                    "data_id": {
                        "type": "string",
                        "description": "Cache ID for the input data — pass as data_id to apply_adjustments when using align_last_value",
                    },
                    "forecast_length": {"type": "integer"},
                },
                "required": ["prediction_id", "data_id", "forecast_length"],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
        Tool(
            name="forecast_explainable",
            title="Explainable Model Forecast",
            description="MODERATE: AutoTS model search restricted to EXPLAINABLE models (Cassandra, TVVAR, BasicLinearModel). Use when interpretability matters. Provide data or data_id. Returns prediction_id and autots_id. Use get_forecast_components on prediction_id for component decomposition.",
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
                },
                "oneOf": [{"required": ["data"]}, {"required": ["data_id"]}],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cache ID — pass to get_forecast, plot_forecast, get_forecast_components, apply_constraints, apply_adjustments",
                    },
                    "autots_id": {
                        "type": "string",
                        "description": "Cache ID — pass to get_validation_results, plot_validation",
                    },
                    "data_id": {
                        "type": "string",
                        "description": "Cache ID for the input data — pass as data_id to apply_adjustments when using align_last_value",
                    },
                    "forecast_length": {"type": "integer"},
                },
                "required": [
                    "prediction_id",
                    "autots_id",
                    "data_id",
                    "forecast_length",
                ],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
        Tool(
            name="forecast_custom",
            title="Custom AutoTS Forecast",
            description="CUSTOM: AutoTS with user-specified parameters or template. Use when forecast_fast results are insufficient. Read the autots://docs/forecast_custom_params resource for available parameters. Defaults to 'scalable' model_list. Provide data or data_id. Returns prediction_id and autots_id.",
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
                    "autots_params": {
                        "type": "object",
                        "description": "AutoTS initialization parameters (defaults: model_list='scalable')",
                    },
                    "model_template": {
                        "type": "object",
                        "description": "Specific model template to run",
                    },
                    "future_regressor_train": {
                        "type": "object",
                        "description": "Future regressor for training (wide format DataFrame)",
                    },
                    "future_regressor_forecast": {
                        "type": "object",
                        "description": "Future regressor for forecast period (wide format DataFrame)",
                    },
                },
                "oneOf": [{"required": ["data"]}, {"required": ["data_id"]}],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cache ID — pass to get_forecast, plot_forecast, get_forecast_components, apply_constraints, apply_adjustments",
                    },
                    "autots_id": {
                        "type": "string",
                        "description": "Cache ID — pass to get_validation_results, plot_validation",
                    },
                    "data_id": {
                        "type": "string",
                        "description": "Cache ID for the input data — pass as data_id to apply_adjustments when using align_last_value",
                    },
                    "forecast_length": {"type": "integer"},
                },
                "required": [
                    "prediction_id",
                    "autots_id",
                    "data_id",
                    "forecast_length",
                ],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
        # ----------------------------------------------------------------
        # Prediction object tools
        # ----------------------------------------------------------------
        Tool(
            name="get_forecast",
            title="Get Forecast Data",
            description="Retrieve forecast values from a cached prediction as JSON or CSV. Requires prediction_id from forecast_fast, forecast_explainable, forecast_custom, or forecast_from_features. Use output='all' to get point, upper, and lower forecasts combined with a forecast_type column.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cached prediction ID",
                    },
                    "output": {
                        "type": "string",
                        "enum": [
                            "forecast",
                            "upper_forecast",
                            "lower_forecast",
                            "all",
                        ],
                        "default": "forecast",
                        "description": "Which forecast to return. 'all' returns point, upper, and lower combined in long format with forecast_type column",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json_wide", "json_long", "csv_wide", "csv_long"],
                        "default": "json_wide",
                        "description": "Output format. Note: 'all' output uses long format automatically",
                    },
                },
                "required": ["prediction_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        Tool(
            name="plot_forecast",
            title="Plot Forecast",
            description="Plot forecast with optional history and prediction intervals. Requires prediction_id from forecast_fast, forecast_explainable, forecast_custom, or forecast_from_features. Returns base64-encoded PNG image. Defaults to first series only.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cached prediction ID",
                    },
                    "include_history": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include historical data",
                    },
                    "series": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific series to plot (by name/id)",
                    },
                    "plot_all": {
                        "type": "boolean",
                        "default": False,
                        "description": "Plot all series (overrides series parameter)",
                    },
                },
                "required": ["prediction_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        Tool(
            name="apply_constraints",
            title="Apply Forecast Constraints",
            description="Apply constraints to a forecast (dampen growth, enforce upper/lower bounds, or quantile clipping). Requires prediction_id from a forecast tool. Returns a new prediction_id with constrained values.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cached prediction ID",
                    },
                    "constraint_method": {
                        "type": "string",
                        "enum": ["dampen", "upper", "lower", "quantile"],
                        "description": "Constraint type",
                    },
                    "constraint_value": {
                        "type": "number",
                        "description": "Constraint value",
                    },
                    "constraint_direction": {
                        "type": "string",
                        "enum": ["upper", "lower"],
                        "description": "Direction for bounds",
                    },
                },
                "required": ["prediction_id", "constraint_method"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cache ID for the constrained forecast",
                    },
                    "constraint_method": {"type": "string"},
                },
                "required": ["prediction_id", "constraint_method"],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
        Tool(
            name="apply_adjustments",
            title="Apply Forecast Adjustments",
            description="Apply post-hoc adjustments to a forecast. Requires prediction_id from a forecast tool. Three types: 1) 'basic'/'linear'/'ramp' — linear ramp between start/end dates with start/end values (additive or multiplicative); 2) 'align_last_value'/'alignlastvalue' — align forecast to recent history (also requires data_id); 3) 'smoothing'/'ewma' — exponential smoothing with span parameter. Optionally restrict to specific series_ids. Returns new prediction_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cached prediction ID",
                    },
                    "data_id": {
                        "type": "string",
                        "description": "Cached data ID for training data (required for align_last_value)",
                    },
                    "adjustment_method": {
                        "type": "string",
                        "enum": [
                            "basic",
                            "linear",
                            "ramp",
                            "align_last_value",
                            "alignlastvalue",
                            "smoothing",
                            "ewma",
                        ],
                        "description": "Adjustment type",
                    },
                    "adjustment_params": {
                        "type": "object",
                        "description": "Adjustment parameters. For basic: {start_date, end_date, start_value, end_value, value, method='additive'|'multiplicative'}. For align_last_value: {rows, lag, method, strength}. For smoothing: {span}",
                    },
                    "series_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Series IDs to apply adjustment to. If omitted, applies to all series.",
                    },
                },
                "required": ["prediction_id", "adjustment_method"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cache ID for the adjusted forecast",
                    },
                    "adjustment_method": {"type": "string"},
                    "series_ids_applied": {},
                },
                "required": ["prediction_id", "adjustment_method"],
            },
            annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        ),
        Tool(
            name="get_model_params",
            title="Get Model Parameters",
            description="Get model name, parameters, and transformation parameters from a cached prediction. Requires prediction_id from any forecast tool.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cached prediction ID",
                    }
                },
                "required": ["prediction_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        Tool(
            name="get_forecast_components",
            title="Get Forecast Component Decomposition",
            description="Get decomposed forecast components (trend, seasonality, etc.) if available. Only works for Cassandra and TVVAR models — use forecast_explainable to guarantee these model types. Requires prediction_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "Cached prediction ID (must be from a Cassandra or TVVAR model)",
                    }
                },
                "required": ["prediction_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        # ----------------------------------------------------------------
        # AutoTS model tools
        # ----------------------------------------------------------------
        Tool(
            name="get_validation_results",
            title="Get Validation Results",
            description="Get cross-validation results and top model rankings from an AutoTS model search. Requires autots_id from forecast_explainable or forecast_custom.",
            inputSchema={
                "type": "object",
                "properties": {
                    "autots_id": {
                        "type": "string",
                        "description": "Cached AutoTS ID",
                    }
                },
                "required": ["autots_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
        Tool(
            name="plot_validation",
            title="Plot Validation Forecasts",
            description="Plot cross-validation forecasts from an AutoTS model search. Requires autots_id from forecast_explainable or forecast_custom. Returns base64-encoded PNG.",
            inputSchema={
                "type": "object",
                "properties": {
                    "autots_id": {
                        "type": "string",
                        "description": "Cached AutoTS ID",
                    }
                },
                "required": ["autots_id"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        ),
    ]

except ImportError:
    FORECAST_TOOLS = []
