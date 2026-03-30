"""Handlers for retrieving and modifying cached predictions and AutoTS model results."""

import base64
import copy
import io
import json
import os
import tempfile
import uuid

import pandas as pd

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from mcp.types import TextContent, ImageContent

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from autots.tools.constraint import apply_adjustment_single

from autots.mcp.cache import cache_object, get_cached_object
from autots.mcp.data_utils import dataframe_to_output, build_csv_metadata


async def handle_get_forecast(arguments: dict, log_progress) -> list:
    prediction_id = arguments.get("prediction_id")
    output = arguments.get("output", "forecast")
    format_type = arguments.get("format", "json_wide")

    cached = get_cached_object(prediction_id, 'prediction')
    prediction = cached['object']

    if output == "all":
        dfs_to_combine = []

        df_forecast = prediction.forecast.copy()
        df_forecast.index = df_forecast.index.strftime('%Y-%m-%d %H:%M:%S')
        df_forecast_long = df_forecast.reset_index().melt(
            id_vars=[df_forecast.reset_index().columns[0]],
            var_name='series_id',
            value_name='value',
        )
        df_forecast_long = df_forecast_long.rename(
            columns={df_forecast_long.columns[0]: 'datetime'}
        )
        df_forecast_long['forecast_type'] = 'point'
        dfs_to_combine.append(df_forecast_long)

        if (
            hasattr(prediction, 'upper_forecast')
            and prediction.upper_forecast is not None
        ):
            df_upper = prediction.upper_forecast.copy()
            df_upper.index = df_upper.index.strftime('%Y-%m-%d %H:%M:%S')
            df_upper_long = df_upper.reset_index().melt(
                id_vars=[df_upper.reset_index().columns[0]],
                var_name='series_id',
                value_name='value',
            )
            df_upper_long = df_upper_long.rename(
                columns={df_upper_long.columns[0]: 'datetime'}
            )
            df_upper_long['forecast_type'] = 'upper'
            dfs_to_combine.append(df_upper_long)

        if (
            hasattr(prediction, 'lower_forecast')
            and prediction.lower_forecast is not None
        ):
            df_lower = prediction.lower_forecast.copy()
            df_lower.index = df_lower.index.strftime('%Y-%m-%d %H:%M:%S')
            df_lower_long = df_lower.reset_index().melt(
                id_vars=[df_lower.reset_index().columns[0]],
                var_name='series_id',
                value_name='value',
            )
            df_lower_long = df_lower_long.rename(
                columns={df_lower_long.columns[0]: 'datetime'}
            )
            df_lower_long['forecast_type'] = 'lower'
            dfs_to_combine.append(df_lower_long)

        df_combined = pd.concat(dfs_to_combine, ignore_index=True)
        df_combined = df_combined[['datetime', 'series_id', 'forecast_type', 'value']]

        if format_type.startswith("csv"):
            temp_dir = tempfile.gettempdir()
            file_id = str(uuid.uuid4())[:8]
            filepath = os.path.join(temp_dir, f"autots_{file_id}_all_forecasts.csv")
            df_combined.to_csv(filepath, index=False)

            metadata = {
                'filepath': filepath,
                'format': 'long_with_forecast_type',
                'shape': {
                    'rows': len(df_combined),
                    'columns': len(df_combined.columns),
                },
                'columns': ['datetime', 'series_id', 'forecast_type', 'value'],
                'forecast_types': df_combined['forecast_type'].unique().tolist(),
                'loading_instructions': {
                    'description': 'Long format with forecast_type: datetime, series_id, forecast_type (point/upper/lower), value columns',
                    'pandas': f"pd.read_csv('{filepath}')",
                    'autots_mcp': f"Use load_to_dataframe('{filepath}') then pivot on forecast_type if needed",
                },
            }
            return [
                TextContent(
                    type="text", text=json.dumps(metadata, separators=(',', ':'))
                )
            ]
        else:
            result = df_combined.to_dict(orient='list')
            return [
                TextContent(type="text", text=json.dumps(result, separators=(',', ':')))
            ]

    elif output == "forecast":
        df = prediction.forecast
    elif output == "upper_forecast":
        df = prediction.upper_forecast
    elif output == "lower_forecast":
        df = prediction.lower_forecast
    else:
        raise ValueError(f"Unknown output type: {output}")

    if format_type.startswith("csv"):
        filepath_out = str(dataframe_to_output(df, format_type))
        is_long = format_type == "csv_long"
        metadata = build_csv_metadata(filepath_out, df, is_long)
        return [
            TextContent(type="text", text=json.dumps(metadata, separators=(',', ':')))
        ]
    else:
        result = dataframe_to_output(df, format_type)
        return [
            TextContent(type="text", text=json.dumps(result, separators=(',', ':')))
        ]


async def handle_plot_forecast(arguments: dict, log_progress) -> list:
    prediction_id = arguments.get("prediction_id")
    include_history = arguments.get("include_history", True)
    series = arguments.get("series")
    plot_all = arguments.get("plot_all", False)

    cached = get_cached_object(prediction_id, 'prediction')
    prediction = cached['object']
    metadata = cached.get('metadata', {})

    if plot_all:
        series_names = list(prediction.forecast.columns)
        forecast_df = prediction.forecast
    elif series:
        series_names = series if isinstance(series, list) else [series]
        forecast_df = prediction.forecast[series_names]
    else:
        first_col = prediction.forecast.columns[0]
        forecast_df = prediction.forecast[[first_col]]
        series_names = [first_col]

    if len(series_names) > 5:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    history_df = None
    if include_history and 'historical_data_id' in metadata:
        try:
            history_cached = get_cached_object(metadata['historical_data_id'], 'data')
            history_df = history_cached['object'][series_names]
        except Exception:
            pass

    if history_df is not None:
        for col in series_names:
            ax.plot(
                history_df.index, history_df[col], label=f'{col} (history)', alpha=0.7
            )

    for col in series_names:
        ax.plot(
            forecast_df.index, forecast_df[col], label=f'{col} (forecast)', linewidth=2
        )

    if hasattr(prediction, 'upper_forecast') and prediction.upper_forecast is not None:
        for col in series_names:
            ax.fill_between(
                forecast_df.index,
                prediction.lower_forecast[col],
                prediction.upper_forecast[col],
                alpha=0.2,
            )

    ax.set_title(
        f'Forecast ({len(series_names)} series)'
        if len(series_names) > 1
        else f'Forecast: {series_names[0]}'
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

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


async def handle_apply_constraints(arguments: dict, log_progress) -> dict:
    prediction_id = arguments.get("prediction_id")
    constraint_method = arguments.get("constraint_method")
    constraint_value = arguments.get("constraint_value")
    constraint_direction = arguments.get("constraint_direction", "upper")

    cached = get_cached_object(prediction_id, 'prediction')
    prediction = cached['object']

    if constraint_method == "dampen":
        prediction = prediction.apply_constraints(
            constraints={
                "constraint_method": "dampening",
                "constraint_value": constraint_value,
            }
        )
    elif constraint_method in ["upper", "lower"]:
        prediction = prediction.apply_constraints(
            constraints={
                "constraint_method": "absolute",
                "constraint_value": constraint_value,
                "constraint_direction": constraint_direction,
            }
        )
    elif constraint_method == "quantile":
        prediction = prediction.apply_constraints(
            constraints={
                "constraint_method": "quantile",
                "constraint_value": constraint_value,
            }
        )
    else:
        raise ValueError(f"Unknown constraint method: {constraint_method}")

    new_prediction_id = cache_object(
        prediction,
        'prediction',
        {
            'method': 'constrained',
            'constraint_method': constraint_method,
            'original_prediction_id': prediction_id,
        },
    )

    return {
        "prediction_id": new_prediction_id,
        "constraint_method": constraint_method,
    }


async def handle_apply_adjustments(arguments: dict, log_progress) -> dict:
    prediction_id = arguments.get("prediction_id")
    adjustment_method = arguments.get("adjustment_method")
    adjustment_params = arguments.get("adjustment_params", {})
    series_ids = arguments.get("series_ids")
    data_id = arguments.get("data_id")

    cached = get_cached_object(prediction_id, 'prediction')
    prediction = cached['object']

    df_train = None
    if adjustment_method in ["align_last_value", "alignlastvalue"]:
        if data_id is None:
            raise ValueError("data_id is required for align_last_value adjustment")
        train_cached = get_cached_object(data_id, 'data')
        df_train = train_cached['object']

    forecast_adj, lower_adj, upper_adj = apply_adjustment_single(
        forecast=prediction.forecast.copy(),
        adjustment_method=adjustment_method,
        adjustment_params=adjustment_params,
        df_train=df_train,
        series_ids=series_ids,
        lower_forecast=(
            prediction.lower_forecast.copy()
            if hasattr(prediction, 'lower_forecast')
            and prediction.lower_forecast is not None
            else None
        ),
        upper_forecast=(
            prediction.upper_forecast.copy()
            if hasattr(prediction, 'upper_forecast')
            and prediction.upper_forecast is not None
            else None
        ),
    )

    new_prediction = copy.deepcopy(prediction)
    new_prediction.forecast = forecast_adj
    if lower_adj is not None:
        new_prediction.lower_forecast = lower_adj
    if upper_adj is not None:
        new_prediction.upper_forecast = upper_adj

    new_prediction_id = cache_object(
        new_prediction,
        'prediction',
        {
            'method': 'adjusted',
            'adjustment_method': adjustment_method,
            'adjustment_params': adjustment_params,
            'series_ids': series_ids,
            'original_prediction_id': prediction_id,
        },
    )

    return {
        "prediction_id": new_prediction_id,
        "adjustment_method": adjustment_method,
        "series_ids_applied": series_ids if series_ids else "all",
    }


async def handle_get_model_params(arguments: dict, log_progress) -> list:
    prediction_id = arguments.get("prediction_id")

    cached = get_cached_object(prediction_id, 'prediction')
    prediction = cached['object']

    params = {
        'model_name': prediction.model_name,
        'model_parameters': prediction.model_parameters,
        'transformation_parameters': prediction.transformation_parameters,
        'forecast_length': len(prediction.forecast),
    }

    return [TextContent(type="text", text=json.dumps(params, separators=(',', ':')))]


async def handle_get_forecast_components(arguments: dict, log_progress) -> list:
    prediction_id = arguments.get("prediction_id")

    cached = get_cached_object(prediction_id, 'prediction')
    prediction = cached['object']

    if hasattr(prediction, 'components') and prediction.components is not None:
        components_df = prediction.components

        if isinstance(components_df, pd.DataFrame):
            if isinstance(components_df.columns, pd.MultiIndex):
                component_types = (
                    components_df.columns.get_level_values(1).unique().tolist()
                )
                series_names = (
                    components_df.columns.get_level_values(0).unique().tolist()
                )
                components_flat = components_df.copy()
                components_flat.columns = [
                    f"{series}_{comp}" for series, comp in components_df.columns
                ]
                result = {
                    "format": "multiindex",
                    "series": series_names,
                    "component_types": component_types,
                    "components": dataframe_to_output(components_flat, "json_wide"),
                    "note": "Column names are formatted as 'series_component_type'",
                }
            else:
                result = {
                    "format": "simple",
                    "components": dataframe_to_output(components_df, "json_wide"),
                }
            return [
                TextContent(type="text", text=json.dumps(result, separators=(',', ':')))
            ]
        else:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Components has unexpected type: {type(components_df)}"
                        },
                        separators=(',', ':'),
                    ),
                )
            ]
    else:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Components not available for this model type. Only certain models (Cassandra, TVVAR) provide component decomposition."
                    },
                    separators=(',', ':'),
                ),
            )
        ]


async def handle_get_validation_results(arguments: dict, log_progress) -> list:
    autots_id = arguments.get("autots_id")

    cached = get_cached_object(autots_id, 'autots')
    model = cached['object']
    metadata = cached.get('metadata', {})

    if not hasattr(model, 'results'):
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "No validation results available"}, indent=2),
            )
        ]

    results_df = model.results()

    best_model = {
        'name': model.best_model_name,
        'parameters': model.best_model_params,
        'transformation': model.best_model_transformation_params,
    }

    top_models = []
    for idx, row in results_df.head(10).iterrows():
        model_info = {
            'model_name': row.get('Model', 'Unknown'),
            'smape': float(row.get('smape', 0)) if pd.notna(row.get('smape')) else None,
            'mae': float(row.get('mae', 0)) if pd.notna(row.get('mae')) else None,
            'rmse': float(row.get('rmse', 0)) if pd.notna(row.get('rmse')) else None,
            'runtime_seconds': (
                float(row.get('TotalRuntimeSeconds', 0))
                if pd.notna(row.get('TotalRuntimeSeconds'))
                else None
            ),
            'validation_round': (
                int(row.get('ValidationRound', 0))
                if pd.notna(row.get('ValidationRound'))
                else None
            ),
        }
        for col in results_df.columns:
            if col not in [
                'Model',
                'smape',
                'mae',
                'rmse',
                'TotalRuntimeSeconds',
                'ValidationRound',
            ]:
                val = row.get(col)
                if pd.notna(val):
                    try:
                        model_info[col] = (
                            float(val) if isinstance(val, (int, float)) else str(val)
                        )
                    except Exception:
                        model_info[col] = str(val)
        top_models.append(model_info)

    train_info = {
        'forecast_length': metadata.get('forecast_length', model.forecast_length),
        'frequency': model.frequency,
        'ensemble': model.ensemble,
        'max_generations': model.max_generations,
        'num_validations': model.num_validations,
        'validation_method': model.validation_method,
        'models_count': len(results_df),
    }

    df_wide = getattr(model, "df_wide_numeric", None)
    if df_wide is not None:
        train_info['train_shape'] = {
            'rows': df_wide.shape[0],
            'columns': df_wide.shape[1],
        }
        train_info['date_range'] = {
            'start': df_wide.index[0].strftime('%Y-%m-%d'),
            'end': df_wide.index[-1].strftime('%Y-%m-%d'),
        }

    result = {
        'best_model': best_model,
        'train_info': train_info,
        'top_10_models': top_models,
        'available_metrics': list(results_df.columns),
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_plot_validation(arguments: dict, log_progress) -> list:
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


PREDICTION_HANDLERS = {
    "get_forecast": handle_get_forecast,
    "plot_forecast": handle_plot_forecast,
    "apply_constraints": handle_apply_constraints,
    "apply_adjustments": handle_apply_adjustments,
    "get_model_params": handle_get_model_params,
    "get_forecast_components": handle_get_forecast_components,
    "get_validation_results": handle_get_validation_results,
    "plot_validation": handle_plot_validation,
}
