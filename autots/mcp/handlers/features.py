"""Handlers for event risk forecasting and time series feature detection tools."""

import base64
import io
import json

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

from autots import EventRiskForecast
from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

from autots.mcp.cache import cache_object, get_cached_object
from autots.mcp.data_utils import (
    load_to_dataframe,
    dataframe_to_output,
    build_csv_metadata,
)


async def handle_forecast_event_risk(arguments: dict, log_progress) -> dict:
    data = arguments.get("data")
    data_id = arguments.get("data_id")
    forecast_length = arguments.get("forecast_length", 30)
    threshold = arguments.get("threshold")
    direction = arguments.get("direction", "upper")
    tune = arguments.get("tune", False)

    if data is None and data_id is None:
        raise ValueError("Must provide either 'data' or 'data_id' parameter")
    if threshold is None:
        raise ValueError("Must provide 'threshold' parameter")

    df = load_to_dataframe(data, data_id=data_id)

    if direction == "lower":
        upper_limit = None
        lower_limit = threshold
    else:
        upper_limit = threshold
        lower_limit = None

    tune_kwargs = {"max_generations": 50, "num_validations": 2} if tune else None
    await log_progress(
        f"forecast_event_risk: fitting event risk model on {len(df.columns)} series × {len(df)} rows"
        + (" (tuned — slower)" if tune else "")
    )
    erf = EventRiskForecast(
        df_train=df,
        forecast_length=forecast_length,
        frequency='infer',
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )

    erf.fit(autots_kwargs=tune_kwargs)
    erf.predict()
    await log_progress("forecast_event_risk: event risk forecast complete")

    event_risk_id = cache_object(
        erf,
        'event_risk',
        {
            'threshold': threshold,
            'direction': direction,
            'forecast_length': forecast_length,
            'tuned': tune,
        },
    )

    return {
        "event_risk_id": event_risk_id,
        "threshold": threshold,
        "direction": direction,
        "forecast_length": forecast_length,
    }


async def handle_get_event_risk_results(arguments: dict, log_progress) -> list:
    event_risk_id = arguments.get("event_risk_id")
    format_type = arguments.get("format", "json_wide")

    cached = get_cached_object(event_risk_id, 'event_risk')
    erf = cached['object']
    metadata = cached['metadata']

    upper_risk_df, lower_risk_df = erf.predict()

    direction = metadata.get('direction', 'upper')
    if direction == 'upper':
        df = upper_risk_df
        risk_type = 'upper_risk'
    elif direction == 'lower':
        df = lower_risk_df
        risk_type = 'lower_risk'
    else:
        result = {
            'threshold': metadata.get('threshold'),
            'forecast_length': metadata.get('forecast_length'),
            'upper_risk': (
                dataframe_to_output(upper_risk_df, format_type)
                if upper_risk_df is not None
                else None
            ),
            'lower_risk': (
                dataframe_to_output(lower_risk_df, format_type)
                if lower_risk_df is not None
                else None
            ),
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if format_type.startswith("csv"):
        filepath = dataframe_to_output(df, format_type)
        is_long = format_type == "csv_long"
        csv_metadata = build_csv_metadata(filepath, df, is_long)
        csv_metadata['risk_type'] = risk_type
        csv_metadata['threshold'] = metadata.get('threshold')
        csv_metadata['forecast_length'] = metadata.get('forecast_length')
        return [TextContent(type="text", text=json.dumps(csv_metadata, indent=2))]
    else:
        result = {
            'risk_type': risk_type,
            'threshold': metadata.get('threshold'),
            'forecast_length': metadata.get('forecast_length'),
            'probabilities': dataframe_to_output(df, format_type),
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_plot_event_risk(arguments: dict, log_progress) -> list:
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


async def handle_detect_features(arguments: dict, log_progress) -> dict:
    data = arguments.get("data")
    data_id = arguments.get("data_id")

    if data is None and data_id is None:
        raise ValueError("Must provide either 'data' or 'data_id' parameter")

    df = load_to_dataframe(data, data_id=data_id)

    await log_progress(
        f"detect_features: analysing {len(df.columns)} series × {len(df)} rows "
        f"for anomalies, changepoints, seasonality, and holidays"
    )
    detector = TimeSeriesFeatureDetector()
    detector.fit(df)
    await log_progress("detect_features: detection complete")

    detector_id = cache_object(
        detector,
        'feature_detector',
        {'series_count': len(df.columns), 'data_rows': len(df)},
    )

    return {
        "detector_id": detector_id,
        "series_count": len(df.columns),
    }


async def handle_get_detected_features(arguments: dict, log_progress) -> list:
    detector_id = arguments.get("detector_id")
    date_start = arguments.get("date_start")
    date_end = arguments.get("date_end")
    specific_date = arguments.get("specific_date")
    series_name = arguments.get("series_name")
    include_components = arguments.get("include_components", False)
    include_metadata = arguments.get("include_metadata", False)

    cached = get_cached_object(detector_id, 'feature_detector')
    detector = cached['object']

    date_filter = None
    if specific_date:
        date_filter = specific_date
    elif date_start or date_end:
        date_filter = slice(date_start, date_end)

    results = detector.query_features(
        dates=date_filter,
        series=series_name,
        include_components=include_components,
        include_metadata=include_metadata,
        return_json=False,
    )

    summary = {
        "date_range": {
            "start": detector.df_original.index[0].strftime('%Y-%m-%d'),
            "end": detector.df_original.index[-1].strftime('%Y-%m-%d'),
        },
        "num_series": len(detector.df_original.columns),
        "num_observations": len(detector.df_original),
        "series_names": list(detector.df_original.columns),
    }

    detection_counts = {}
    for sname in results.get('series', {}).keys():
        series_data = results['series'][sname]
        counts = {
            "trend_changepoints": len(series_data.get('trend_changepoints', [])),
            "level_shifts": len(series_data.get('level_shifts', [])),
            "anomalies": len(series_data.get('anomalies', [])),
            "holidays": len(series_data.get('holiday_dates', [])),
        }
        if 'seasonality_strength' in series_data:
            counts['seasonality_strength'] = series_data['seasonality_strength']
        detection_counts[sname] = counts

    output = {
        'summary': summary,
        'detection_counts': detection_counts,
        'features': results,
    }

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def handle_plot_features(arguments: dict, log_progress) -> list:
    detector_id = arguments.get("detector_id")
    series = arguments.get("series")

    cached = get_cached_object(detector_id, 'feature_detector')
    detector = cached['object']

    if series:
        fig = detector.plot(series_name=series[0])
    else:
        fig = detector.plot()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return [ImageContent(type="image", data=img_base64, mimeType="image/png")]


async def handle_forecast_from_features(arguments: dict, log_progress) -> dict:
    detector_id = arguments.get("detector_id")
    forecast_length = arguments.get("forecast_length", 30)

    cached = get_cached_object(detector_id, 'feature_detector')
    detector = cached['object']

    prediction = detector.forecast(forecast_length=forecast_length)

    historical_data_id = None
    if hasattr(detector, 'df_original') and detector.df_original is not None:
        historical_data_id = cache_object(
            detector.df_original,
            'data',
            {
                'source': 'feature_detector_history',
                'rows': len(detector.df_original),
                'columns': len(detector.df_original.columns),
            },
        )

    prediction_id = cache_object(
        prediction,
        'prediction',
        {
            'method': 'feature_detector',
            'forecast_length': forecast_length,
            'detector_id': detector_id,
            'historical_data_id': historical_data_id,
        },
    )

    return {
        "prediction_id": prediction_id,
        "forecast_length": forecast_length,
        "note": "This forecast is based on detected features and is experimental",
    }


FEATURES_HANDLERS = {
    "forecast_event_risk": handle_forecast_event_risk,
    "get_event_risk_results": handle_get_event_risk_results,
    "plot_event_risk": handle_plot_event_risk,
    "detect_features": handle_detect_features,
    "get_detected_features": handle_get_detected_features,
    "plot_features": handle_plot_features,
    "forecast_from_features": handle_forecast_from_features,
}
