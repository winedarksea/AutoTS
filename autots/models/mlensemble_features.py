"""Base-model forecasts and stacked regressor features for MLEnsemble.

collect_base_forecasts feeds the per-series correction methods; create_feature
builds the pooled feature matrix used only by correction_method='regressor'.
"""

import json
import sys
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import model_forecast
from autots.models.model_list import diff_window_motif_list, all_result_path
from autots.evaluator.event_forecasting import extract_result_windows
from autots.tools.seasonal import date_part


def _coerce_params(config):
    """Normalize model/config dictionaries that might arrive as JSON strings."""
    if config in (None, "", {}):
        return {}
    if isinstance(config, str):
        return json.loads(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"Unsupported parameter type for MLEnsemble: {type(config)}")


def _ensure_feature_block(array_like, target_columns):
    """Cast feature data to 3D array with axis order (feature, horizon, series)."""
    if isinstance(array_like, pd.DataFrame):
        matrix = array_like.reindex(columns=target_columns, fill_value=0).to_numpy(
            copy=False
        )
    else:
        matrix = np.asarray(array_like)
    if matrix.ndim == 2:
        return matrix[np.newaxis, :, :]
    if matrix.ndim == 3:
        return matrix
    raise ValueError(
        f"Features must be 2D or 3D after alignment, received shape {matrix.shape}."
    )


def _align_result_windows(result_windows, source_columns, target_columns):
    """Reorder or pad extracted motif windows to match training column order."""
    if list(source_columns) == list(target_columns):
        return result_windows
    indexer = pd.Index(source_columns).get_indexer(target_columns)
    aligned = np.zeros(
        (result_windows.shape[0], result_windows.shape[1], len(target_columns)),
        dtype=result_windows.dtype,
    )
    valid = indexer >= 0
    if valid.any():
        aligned[:, :, valid] = result_windows[:, :, indexer[valid]]
    return aligned


def _sanitize_feature_matrix(feature_matrix):
    """Replace NaN/inf features (from unfilled history stats) so any regressor accepts them."""
    if not np.isfinite(feature_matrix).all():
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_matrix


def _forecast_base_model(
    model_name,
    model_param_dict,
    transform_params,
    df_train,
    forecast_length,
    future_regressor_train=None,
    future_regressor_forecast=None,
    random_seed=2020,
    verbose=0,
    n_jobs=1,
    return_model=False,
):
    return model_forecast(
        model_name=model_name,
        model_param_dict=model_param_dict,
        model_transform_dict=transform_params,
        df_train=df_train,
        forecast_length=forecast_length,
        frequency='infer',
        prediction_interval=0.9,
        no_negatives=False,
        future_regressor_train=future_regressor_train,
        future_regressor_forecast=future_regressor_forecast,
        random_seed=random_seed,
        verbose=verbose,
        n_jobs=n_jobs,
        return_model=return_model,
        fail_on_forecast_nan=True,
    )


def collect_base_forecasts(
    df_train,
    models,
    forecast_length,
    future_regressor_train=None,
    future_regressor_forecast=None,
    random_seed=2020,
    verbose=0,
    n_jobs=1,
):
    """Point, upper and lower base forecasts as (M, H, N) arrays plus the forecast index."""
    points, uppers, lowers = [], [], []
    for model in models:
        forecasts = _forecast_base_model(
            model['Model'],
            _coerce_params(model.get('ModelParameters')),
            _coerce_params(model.get('TransformationParameters')),
            df_train,
            forecast_length,
            future_regressor_train=future_regressor_train,
            future_regressor_forecast=future_regressor_forecast,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        for store, frame in zip(
            (points, uppers, lowers),
            (forecasts.forecast, forecasts.upper_forecast, forecasts.lower_forecast),
        ):
            store.append(frame.reindex(columns=df_train.columns).to_numpy(dtype=float))
    return (
        np.stack(points),
        np.stack(uppers),
        np.stack(lowers),
        forecasts.forecast.index,
    )


def create_feature(
    df_train,
    models,
    forecast_length,
    future_regressor_train=None,
    future_regressor_forecast=None,
    datepart_method=None,
    random_seed=2020,
    verbose=0,
    n_jobs=1,
    base_return_model=False,
):
    target_columns = df_train.columns
    num_series = len(target_columns)
    feature_blocks = [
        _ensure_feature_block(
            np.repeat(df_train.iloc[-1:,].to_numpy(), forecast_length, axis=0),
            target_columns,
        ),
        _ensure_feature_block(
            np.repeat(
                df_train.mean().to_numpy()[np.newaxis, :], forecast_length, axis=0
            ),
            target_columns,
        ),
        _ensure_feature_block(
            np.repeat(
                df_train.std().to_numpy()[np.newaxis, :], forecast_length, axis=0
            ),
            target_columns,
        ),
    ]

    forecast_index = None
    for model in models:
        model_name = model['Model']
        model_param_dict = _coerce_params(model.get('ModelParameters'))
        transform_params = _coerce_params(model.get('TransformationParameters'))
        if model_name in diff_window_motif_list and not model_param_dict.get(
            "return_result_windows"
        ):
            model_param_dict = {**model_param_dict, "return_result_windows": True}

        need_components = model_name == 'Cassandra'
        need_result_windows = model_name in all_result_path
        forecasts = _forecast_base_model(
            model_name,
            model_param_dict,
            transform_params,
            df_train,
            forecast_length,
            future_regressor_train=future_regressor_train,
            future_regressor_forecast=future_regressor_forecast,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            return_model=base_return_model or need_components or need_result_windows,
        )
        forecast_index = forecasts.forecast.index
        feature_blocks.extend(
            [
                _ensure_feature_block(forecasts.forecast, target_columns),
                _ensure_feature_block(forecasts.upper_forecast, target_columns),
                _ensure_feature_block(forecasts.lower_forecast, target_columns),
            ]
        )
        if need_components and hasattr(forecasts.model, "return_components"):
            comps = forecasts.model.return_components()
            ncomps = comps.columns.get_level_values(1)
            for cp in set(ncomps):
                comp_df = (
                    comps.loc[:, comps.columns.get_level_values(1) == cp]
                    .droplevel(1, 1)
                    .reindex(forecasts.forecast.index)
                )
                feature_blocks.append(_ensure_feature_block(comp_df, target_columns))
        if need_result_windows:
            windows = extract_result_windows(forecasts, model_name=model_name)
            aligned_windows = _align_result_windows(
                windows, forecasts.forecast.columns, target_columns
            )
            feature_blocks.append(aligned_windows)

    if future_regressor_forecast is not None:
        reg_array = np.repeat(
            future_regressor_forecast.to_numpy().T[:, :, np.newaxis],
            num_series,
            axis=2,
        )
        feature_blocks.append(reg_array)

    time_feature = np.broadcast_to(
        np.arange(forecast_length)[np.newaxis, :, np.newaxis],
        (1, forecast_length, num_series),
    )
    feature_blocks.append(time_feature)

    series_id_feature = np.broadcast_to(
        np.arange(num_series)[np.newaxis, np.newaxis, :],
        (1, forecast_length, num_series),
    )
    feature_blocks.append(series_id_feature)

    if datepart_method not in [None, "None", "none"] and forecast_index is not None:
        date_parts = date_part(forecast_index, method=datepart_method)
        date_features = np.repeat(
            date_parts.to_numpy().T[:, :, np.newaxis],
            num_series,
            axis=2,
        )
        feature_blocks.append(date_features)

    res = np.concatenate(feature_blocks, axis=0)
    sys.stdout.flush()
    return res


def default_regression_model():
    """Original tuned XGBoost (RandomForest if xgboost is missing) for the 'regressor' path."""
    try:
        import xgboost  # noqa

        return {
            "model": 'XGBRegressor',
            "model_params": {
                "base_score": 0.5,
                "booster": 'gbtree',
                "colsample_bylevel": 0.50502979,
                "colsample_bynode": 1,
                "colsample_bytree": 0.6164668,
                "early_stopping_rounds": None,
                "enable_categorical": False,
                "eval_metric": None,
                "feature_types": None,
                "gamma": 0,
                "grow_policy": 'depthwise',
                "importance_type": None,
                "interaction_constraints": '',
                "learning_rate": 0.00266327,
                "max_bin": 256,
                "max_cat_threshold": 64,
                "max_cat_to_onehot": 4,
                "max_delta_step": 0,
                "max_depth": 10,
                "max_leaves": 0,
                "min_child_weight": 0.0104586,
                "monotone_constraints": '()',
                "n_estimators": 1356,
                "num_parallel_tree": 1,
            },
        }
    except Exception:
        return {
            "model": 'RandomForest',
            "model_params": {
                "max_features": 0.16322,
                "max_leaf_nodes": 10,
                "n_estimators": 5,
            },
        }
