"""Handlers for running forecasts: forecast_fast, forecast_explainable, forecast_custom."""

import json
from pathlib import Path

import pandas as pd

try:
    from mcp.types import TextContent

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from autots import AutoTS

from autots.mcp.cache import cache_object
from autots.mcp.data_utils import load_to_dataframe


async def handle_forecast_fast(arguments: dict, log_progress) -> dict:
    data = arguments.get("data")
    data_id = arguments.get("data_id")
    forecast_length = arguments.get("forecast_length", 30)
    profile_template = arguments.get("profile_template")

    if data is None and data_id is None:
        raise ValueError("Must provide either 'data' or 'data_id' parameter")

    df = load_to_dataframe(data, data_id=data_id)

    if not data_id:
        data_id = cache_object(
            df,
            'data',
            {
                'source': 'forecast_fast_input',
                'rows': len(df),
                'columns': len(df.columns),
            },
        )

    if not profile_template:
        template_path = str(
            Path(__file__).parent.parent / 'mosaic_profile_template.json'
        )
        with open(template_path, 'r') as f:
            profile_template = json.load(f)

    ensemble_params = {
        'Model': 'Ensemble',
        'ModelParameters': json.dumps(profile_template),
        'TransformationParameters': json.dumps(
            profile_template.get('transformation', {})
        ),
        'Ensemble': 2,
    }
    ensemble_template = pd.DataFrame(ensemble_params, index=[0])

    await log_progress(
        f"forecast_fast: fitting mosaic ensemble on {len(df.columns)} series × {len(df)} rows"
    )
    model = AutoTS(
        forecast_length=forecast_length,
        frequency='infer',
        ensemble='horizontal-profile',
        model_list='scalable',
        max_generations=0,
        num_validations=0,
        validation_method='backwards',
    )
    model.fit_data(df)
    model.import_best_model(
        ensemble_template, enforce_model_list=False, include_ensemble=True
    )
    prediction = model.predict()

    prediction_id = cache_object(
        prediction,
        'prediction',
        {
            'method': 'fast',
            'forecast_length': forecast_length,
            'series_count': len(df.columns),
            'historical_data_id': data_id,
        },
    )

    return {
        "prediction_id": prediction_id,
        "data_id": data_id,
        "forecast_length": forecast_length,
    }


async def handle_forecast_explainable(arguments: dict, log_progress) -> dict:
    data = arguments.get("data")
    data_id = arguments.get("data_id")
    forecast_length = arguments.get("forecast_length", 30)

    if data is None and data_id is None:
        raise ValueError("Must provide either 'data' or 'data_id' parameter")

    df = load_to_dataframe(data, data_id=data_id)

    if not data_id:
        data_id = cache_object(
            df,
            'data',
            {
                'source': 'forecast_explainable_input',
                'rows': len(df),
                'columns': len(df.columns),
            },
        )

    await log_progress(
        f"forecast_explainable: starting model search on {len(df.columns)} series × {len(df)} rows "
        f"(3 generations, 2 validations — may take several minutes)"
    )
    model = AutoTS(
        forecast_length=forecast_length,
        frequency='infer',
        ensemble=None,
        model_list=['Cassandra', 'TVVAR', 'BasicLinearModel'],
        max_generations=3,
        num_validations=2,
        validation_method='backwards',
    )
    model.fit(df)
    await log_progress(
        "forecast_explainable: model search complete, generating forecast"
    )
    prediction = model.predict()

    prediction_id = cache_object(
        prediction,
        'prediction',
        {
            'method': 'explainable',
            'forecast_length': forecast_length,
            'series_count': len(df.columns),
            'historical_data_id': data_id,
        },
    )
    autots_id = cache_object(
        model,
        'autots',
        {
            'forecast_length': forecast_length,
            'series_count': len(df.columns),
        },
    )

    return {
        "prediction_id": prediction_id,
        "autots_id": autots_id,
        "data_id": data_id,
        "forecast_length": forecast_length,
    }


async def handle_forecast_custom(arguments: dict, log_progress) -> dict:
    data = arguments.get("data")
    data_id = arguments.get("data_id")
    forecast_length = arguments.get("forecast_length", 30)
    autots_params = arguments.get("autots_params", {})
    model_template = arguments.get("model_template")
    future_regressor_train = arguments.get("future_regressor_train")
    future_regressor_forecast = arguments.get("future_regressor_forecast")

    if data is None and data_id is None:
        raise ValueError("Must provide either 'data' or 'data_id' parameter")

    df = load_to_dataframe(data, data_id=data_id)

    future_regressor_train_df = None
    future_regressor_forecast_df = None
    if future_regressor_train:
        future_regressor_train_df = load_to_dataframe(
            future_regressor_train, data_format="wide"
        )
    if future_regressor_forecast:
        future_regressor_forecast_df = load_to_dataframe(
            future_regressor_forecast, data_format="wide"
        )

    if not data_id:
        data_id = cache_object(
            df,
            'data',
            {
                'source': 'forecast_custom_input',
                'rows': len(df),
                'columns': len(df.columns),
            },
        )

    if 'forecast_length' not in autots_params:
        autots_params['forecast_length'] = forecast_length
    if 'frequency' not in autots_params:
        autots_params['frequency'] = 'infer'
    if 'model_list' not in autots_params:
        autots_params['model_list'] = 'scalable'

    model = AutoTS(**autots_params)

    gens = autots_params.get('max_generations', 10)
    vals = autots_params.get('num_validations', 2)
    await log_progress(
        f"forecast_custom: starting model search on {len(df.columns)} series × {len(df)} rows "
        f"({gens} generations, {vals} validations — may take several minutes)"
    )
    if model_template:
        model = model.import_template(model_template, method='only')
        model.fit(df, future_regressor=future_regressor_train_df)
        prediction = model.predict(future_regressor=future_regressor_forecast_df)
    else:
        model.fit(df, future_regressor=future_regressor_train_df)
        prediction = model.predict(future_regressor=future_regressor_forecast_df)
    await log_progress("forecast_custom: model search complete, forecast generated")

    prediction_id = cache_object(
        prediction,
        'prediction',
        {
            'method': 'custom',
            'forecast_length': forecast_length,
            'series_count': len(df.columns),
            'historical_data_id': data_id,
        },
    )
    autots_id = cache_object(
        model,
        'autots',
        {
            'forecast_length': forecast_length,
            'series_count': len(df.columns),
        },
    )

    return {
        "prediction_id": prediction_id,
        "autots_id": autots_id,
        "data_id": data_id,
        "forecast_length": forecast_length,
    }


FORECAST_HANDLERS = {
    "forecast_fast": handle_forecast_fast,
    "forecast_explainable": handle_forecast_explainable,
    "forecast_custom": handle_forecast_custom,
}
