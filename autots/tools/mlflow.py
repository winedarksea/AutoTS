"""MLflow autolog support for AutoTS and ModelObject-based models.

This module intentionally keeps MLflow optional. All MLflow imports are
performed inside try/except blocks, and integration remains a no-op when
MLflow is not installed.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import tempfile
import threading
from typing import Any, Dict

import numpy as np
import pandas as pd


_AUTOLOG_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "tried_models_logging": "single",  # single | individual | none
    "log_model_runs": True,
    "log_model_forecasts": True,
    "silent": True,
}

_STATE = threading.local()


def _get_mlflow():
    """Return mlflow module if available, else None."""
    try:
        import mlflow

        return mlflow
    except Exception:
        return None


def _state_get(name: str, default):
    return getattr(_STATE, name, default)


def _state_set(name: str, value):
    setattr(_STATE, name, value)


def _is_modelobject_suppressed() -> bool:
    return _state_get("modelobject_suppress_count", 0) > 0


def _suppress_modelobject_logging():
    _state_set(
        "modelobject_suppress_count", _state_get("modelobject_suppress_count", 0) + 1
    )


def _unsuppress_modelobject_logging():
    current = _state_get("modelobject_suppress_count", 0)
    _state_set("modelobject_suppress_count", max(current - 1, 0))


def _safe_float(value):
    try:
        if isinstance(value, (np.floating, float, np.integer, int)):
            v = float(value)
            if math.isfinite(v):
                return v
        return None
    except Exception:
        return None


def _timedelta_seconds(value):
    try:
        if isinstance(value, datetime.timedelta):
            return float(value.total_seconds())
        if hasattr(value, "total_seconds"):
            return float(value.total_seconds())
        return None
    except Exception:
        return None


def _short_param_value(value, max_len: int = 480):
    if isinstance(value, (dict, list, tuple, set)):
        try:
            value = json.dumps(value, sort_keys=True)
        except Exception:
            value = str(value)
    elif value is None:
        value = "None"
    else:
        value = str(value)
    if len(value) > max_len:
        value = value[: max_len - 3] + "..."
    return value


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return {str(k): _to_jsonable(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.Index):
        return [_to_jsonable(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(x) for x in value]
    return str(value)


def _safe_mlflow_set_tag(mlflow, key: str, value: Any):
    try:
        mlflow.set_tag(key, _short_param_value(value, max_len=5000))
    except Exception:
        pass


def _safe_mlflow_log_param(mlflow, key: str, value: Any):
    try:
        mlflow.log_param(key, _short_param_value(value))
    except Exception:
        pass


def _safe_mlflow_log_metric(mlflow, key: str, value: Any):
    val = _safe_float(value)
    if val is None:
        return
    try:
        mlflow.log_metric(key, val)
    except Exception:
        pass


def _safe_mlflow_log_dict(mlflow, payload: dict, artifact_file: str):
    artifact_file = str(artifact_file).lstrip("/")
    artifact_dir = os.path.dirname(artifact_file)
    file_name = os.path.basename(artifact_file) or "autots_payload.json"
    if not file_name.endswith(".json"):
        file_name = f"{file_name}.json"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, file_name)
            with open(file_path, "w", encoding="utf-8") as tmp:
                json.dump(_to_jsonable(payload), tmp, indent=2, sort_keys=True)
            mlflow.log_artifact(file_path, artifact_path=artifact_dir or None)
    except Exception:
        # Fall back to direct API if available
        try:
            mlflow.log_dict(_to_jsonable(payload), artifact_file)
        except Exception:
            pass


def _safe_mlflow_log_dataframe(mlflow, df: pd.DataFrame, artifact_file: str):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    artifact_file = str(artifact_file).lstrip("/")
    artifact_dir = os.path.dirname(artifact_file)
    file_name = os.path.basename(artifact_file) or "autots_results.csv"
    if not file_name.endswith(".csv"):
        file_name = f"{file_name}.csv"
    local_df = df.copy()
    for col in local_df.columns:
        try:
            if hasattr(local_df[col].dtype, 'kind') and local_df[col].dtype.kind == 'm':
                local_df[col] = local_df[col].astype(str)
        except Exception:
            pass
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, file_name)
            local_df.to_csv(file_path, index=False)
            mlflow.log_artifact(file_path, artifact_path=artifact_dir or None)
    except Exception:
        try:
            mlflow.log_table(local_df, artifact_file)
        except Exception:
            pass


def _start_run(mlflow, run_name: str):
    """Start a run and return context details."""
    started = False
    run = None
    try:
        if mlflow.active_run() is None:
            run = mlflow.start_run(run_name=run_name)
        else:
            run = mlflow.start_run(run_name=run_name, nested=True)
        started = True
    except Exception:
        run = mlflow.active_run()
        started = False
    return {
        "run": run,
        "started": started,
        "run_id": None if run is None else run.info.run_id,
    }


def _end_run(mlflow, run_ctx: dict, status: str = "FINISHED"):
    if not run_ctx or not run_ctx.get("started", False):
        return
    try:
        mlflow.end_run(status=status)
    except Exception:
        pass


def _build_model_as_code_payload(model_obj, extra: dict | None = None):
    payload = {
        "model_name": getattr(model_obj, "name", model_obj.__class__.__name__),
        "model_class": model_obj.__class__.__name__,
        "model_module": model_obj.__class__.__module__,
        "refit_expected": True,
        "model_as_code": True,
    }
    try:
        params = model_obj.get_params()
        if isinstance(params, dict):
            payload["model_parameters"] = params
    except Exception:
        pass
    last_date = getattr(model_obj, "train_last_date", None)
    if last_date is not None:
        payload["last_obs_date"] = _to_jsonable(last_date)
    if extra:
        payload.update(extra)
    return payload


def _build_autots_model_as_code_payload(autots_obj, best_row=None):
    payload = {
        "model_name": "AutoTS",
        "model_class": autots_obj.__class__.__name__,
        "model_module": autots_obj.__class__.__module__,
        "refit_expected": True,
        "model_as_code": True,
        "best_model_name": getattr(autots_obj, "best_model_name", None),
        "best_model_parameters": getattr(autots_obj, "best_model_params", None),
        "best_model_transformation_parameters": getattr(
            autots_obj, "best_model_transformation_params", None
        ),
    }
    try:
        if (
            hasattr(autots_obj, "df_wide_numeric")
            and autots_obj.df_wide_numeric is not None
        ):
            payload["last_obs_date"] = _to_jsonable(
                autots_obj.df_wide_numeric.index[-1]
            )
    except Exception:
        pass
    if isinstance(best_row, pd.Series):
        payload["best_model_id"] = _to_jsonable(best_row.get("ID"))
        payload["best_model_generation"] = _to_jsonable(best_row.get("Generation"))
        payload["best_model_ensemble"] = _to_jsonable(best_row.get("Ensemble"))
    return payload


def _extract_autots_best_row(autots_obj):
    try:
        val = autots_obj.results("validation")
        if isinstance(val, pd.DataFrame) and not val.empty:
            best_id = getattr(autots_obj, "best_model_id", None)
            if best_id is not None and "ID" in val.columns:
                matched = val[val["ID"] == best_id]
                if not matched.empty:
                    return matched.iloc[0]
            return val.sort_values("Score", ascending=True).iloc[0]
    except Exception:
        pass
    try:
        ini = autots_obj.results("initial")
        if isinstance(ini, pd.DataFrame) and not ini.empty:
            return ini.sort_values("Score", ascending=True).iloc[0]
    except Exception:
        pass
    return None


def _extract_autots_results_df(autots_obj, result_set="initial"):
    try:
        res = autots_obj.results(result_set)
        if isinstance(res, pd.DataFrame):
            return res
    except Exception:
        pass
    return pd.DataFrame()


def _log_autots_common(mlflow, autots_obj):
    _safe_mlflow_set_tag(mlflow, "autots.flavor", "AutoTS")
    _safe_mlflow_set_tag(mlflow, "autots.class", autots_obj.__class__.__name__)
    _safe_mlflow_set_tag(mlflow, "autots.module", autots_obj.__class__.__module__)

    param_keys = [
        "forecast_length",
        "frequency",
        "prediction_interval",
        "ensemble",
        "max_generations",
        "num_validations",
        "validation_method",
        "models_to_validate",
        "model_list",
        "transformer_list",
        "transformer_max_depth",
        "metric_weighting",
        "n_jobs",
        "random_seed",
    ]
    for key in param_keys:
        _safe_mlflow_log_param(mlflow, key, getattr(autots_obj, key, None))

    try:
        dfw = autots_obj.df_wide_numeric
        if isinstance(dfw, pd.DataFrame):
            _safe_mlflow_log_param(mlflow, "train_rows", int(dfw.shape[0]))
            _safe_mlflow_log_param(mlflow, "train_cols", int(dfw.shape[1]))
            _safe_mlflow_log_param(mlflow, "last_obs_date", _to_jsonable(dfw.index[-1]))
    except Exception:
        pass


def _log_autots_summary(mlflow, autots_obj):
    initial_df = _extract_autots_results_df(autots_obj, "initial")
    validation_df = _extract_autots_results_df(autots_obj, "validation")

    if not initial_df.empty:
        _safe_mlflow_log_metric(mlflow, "autots.models_tested", initial_df.shape[0])
        if "Exceptions" in initial_df.columns:
            passed = initial_df["Exceptions"].isna().sum()
            _safe_mlflow_log_metric(mlflow, "autots.models_passed", passed)
            _safe_mlflow_log_metric(
                mlflow,
                "autots.failure_rate",
                (initial_df.shape[0] - passed) / max(initial_df.shape[0], 1),
            )

    best_row = _extract_autots_best_row(autots_obj)
    if isinstance(best_row, pd.Series):
        for col in [
            "smape",
            "mae",
            "rmse",
            "spl",
            "contour",
            "containment",
            "Score",
            "TotalRuntimeSeconds",
        ]:
            if col in best_row.index:
                _safe_mlflow_log_metric(mlflow, f"best.{col}", best_row[col])
        if "Model" in best_row.index:
            _safe_mlflow_set_tag(mlflow, "best.model_name", best_row["Model"])
        if "ID" in best_row.index:
            _safe_mlflow_set_tag(mlflow, "best.model_id", best_row["ID"])

    _safe_mlflow_log_param(
        mlflow, "best_model_name", getattr(autots_obj, "best_model_name", None)
    )
    _safe_mlflow_log_param(
        mlflow,
        "best_model_ensemble",
        getattr(autots_obj, "best_model_ensemble", None),
    )

    model_as_code = _build_autots_model_as_code_payload(autots_obj, best_row=best_row)
    _safe_mlflow_log_dict(mlflow, model_as_code, "autots/model_as_code.json")

    if _AUTOLOG_CONFIG.get("tried_models_logging", "single") == "single":
        _safe_mlflow_log_dataframe(
            mlflow, initial_df, "autots/model_results_initial.csv"
        )
        if not validation_df.empty:
            _safe_mlflow_log_dataframe(
                mlflow, validation_df, "autots/model_results_validation.csv"
            )
        score_breakdown = getattr(autots_obj, "score_breakdown", None)
        if isinstance(score_breakdown, pd.DataFrame) and not score_breakdown.empty:
            _safe_mlflow_log_dataframe(
                mlflow,
                score_breakdown.reset_index(drop=False),
                "autots/score_breakdown.csv",
            )


def _log_autots_individual_model_runs(mlflow, autots_obj, max_runs: int = 400):
    rows = _extract_autots_results_df(autots_obj, "initial")
    if rows.empty:
        return

    if "ValidationRound" in rows.columns:
        rows = rows[rows["ValidationRound"] == 0]
    rows = rows.reset_index(drop=True)
    if len(rows) > max_runs:
        rows = rows.head(max_runs)

    create_id = None
    try:
        from autots.evaluator.auto_model import create_model_id as create_id
    except Exception:
        create_id = None

    for idx, row in rows.iterrows():
        run_name = f"AutoTSCandidate::{row.get('Model', 'Model')}::{idx}"
        run_ctx = _start_run(mlflow, run_name)
        try:
            _safe_mlflow_set_tag(mlflow, "autots.flavor", "AutoTSModelCandidate")
            _safe_mlflow_set_tag(mlflow, "autots.parent", "AutoTS")
            _safe_mlflow_set_tag(mlflow, "autots.model_name", row.get("Model"))
            _safe_mlflow_set_tag(mlflow, "autots.model_id", row.get("ID"))

            for key in ["Ensemble", "Generation", "ValidationRound", "Runs"]:
                if key in row.index:
                    _safe_mlflow_log_param(mlflow, key.lower(), row.get(key))

            for metric in [
                "Score",
                "smape",
                "mae",
                "rmse",
                "spl",
                "containment",
                "contour",
                "TotalRuntimeSeconds",
            ]:
                if metric in row.index:
                    _safe_mlflow_log_metric(
                        mlflow,
                        metric.lower().replace(
                            "totalruntimeseconds", "runtime_seconds"
                        ),
                        row.get(metric),
                    )

            model_payload = {
                "model_name": row.get("Model"),
                "model_id": row.get("ID"),
                "ensemble": row.get("Ensemble"),
                "generation": row.get("Generation"),
                "validation_round": row.get("ValidationRound"),
                "model_parameters": row.get("ModelParameters"),
                "transformation_parameters": row.get("TransformationParameters"),
                "refit_expected": True,
                "model_as_code": True,
            }
            if create_id is not None and model_payload.get("model_id") is None:
                try:
                    model_payload["model_id"] = create_id(
                        model_payload["model_name"],
                        model_payload["model_parameters"],
                        model_payload["transformation_parameters"],
                    )
                except Exception:
                    pass
            _safe_mlflow_log_dict(
                mlflow,
                model_payload,
                f"autots/candidate_{idx}_model_as_code.json",
            )
        finally:
            _end_run(mlflow, run_ctx, status="FINISHED")


def _start_autots_fit_run(autots_obj):
    if not _AUTOLOG_CONFIG.get("enabled", False):
        return None
    mlflow = _get_mlflow()
    if mlflow is None:
        if not _AUTOLOG_CONFIG.get("silent", True):
            print("AutoTS MLflow autolog enabled, but mlflow import failed.")
        return None

    run_ctx = _start_run(mlflow, "AutoTS.fit")
    _suppress_modelobject_logging()
    _log_autots_common(mlflow, autots_obj)
    return {"mlflow": mlflow, "run_ctx": run_ctx, "closed": False}


def _end_autots_fit_run(autots_obj, ctx, error=None):
    if ctx is None or ctx.get("closed", False):
        return
    ctx["closed"] = True
    mlflow = ctx.get("mlflow")
    run_ctx = ctx.get("run_ctx")
    status = "FAILED" if error is not None else "FINISHED"
    try:
        if mlflow is not None:
            if error is not None:
                _safe_mlflow_set_tag(mlflow, "autots.run_status", "failed")
                _safe_mlflow_set_tag(mlflow, "autots.error", repr(error))
            else:
                _safe_mlflow_set_tag(mlflow, "autots.run_status", "finished")
            _log_autots_summary(mlflow, autots_obj)
            if _AUTOLOG_CONFIG.get("tried_models_logging", "single") == "individual":
                _log_autots_individual_model_runs(mlflow, autots_obj)
    finally:
        _unsuppress_modelobject_logging()
        if mlflow is not None:
            _end_run(mlflow, run_ctx, status=status)


def _patch_autots_fit():
    """Patch AutoTS.fit so autologging works automatically."""
    try:
        from autots.evaluator.auto_ts import AutoTS
    except Exception:
        return False

    fit_method = getattr(AutoTS, "fit", None)
    if fit_method is None:
        return False
    if getattr(fit_method, "_autots_mlflow_wrapped", False):
        return True

    from functools import wraps

    @wraps(fit_method)
    def wrapped_fit(self, *args, **kwargs):
        ctx = _start_autots_fit_run(self)
        result = None
        err = None
        try:
            result = fit_method(self, *args, **kwargs)
            return result
        except Exception as exc:
            err = exc
            raise
        finally:
            _end_autots_fit_run(self, ctx, error=err)

    wrapped_fit._autots_mlflow_wrapped = True
    wrapped_fit._autots_mlflow_original = fit_method
    AutoTS.fit = wrapped_fit
    return True


def _patch_autots_predict():
    """Patch AutoTS.predict to suppress inner ModelObject logs and emit class-level logs."""
    try:
        from autots.evaluator.auto_ts import AutoTS
    except Exception:
        return False

    predict_method = getattr(AutoTS, "predict", None)
    if predict_method is None:
        return False
    if getattr(predict_method, "_autots_mlflow_wrapped", False):
        return True

    from functools import wraps

    @wraps(predict_method)
    def wrapped_predict(self, *args, **kwargs):
        if not _AUTOLOG_CONFIG.get("enabled", False):
            return predict_method(self, *args, **kwargs)

        mlflow = _get_mlflow()
        run_ctx = None
        err = None
        result = None
        if mlflow is not None:
            run_ctx = _start_run(mlflow, "AutoTS.predict")
            _safe_mlflow_set_tag(mlflow, "autots.flavor", "AutoTS")
            _safe_mlflow_set_tag(mlflow, "autots.stage", "predict")
            _safe_mlflow_log_param(
                mlflow, "best_model_name", getattr(self, "best_model_name", None)
            )
            fl = (
                kwargs.get("forecast_length")
                if "forecast_length" in kwargs
                else (args[0] if args else None)
            )
            if fl is not None and fl != "self":
                _safe_mlflow_log_param(mlflow, "forecast_length", fl)

        _suppress_modelobject_logging()
        try:
            result = predict_method(self, *args, **kwargs)
            return result
        except Exception as exc:
            err = exc
            raise
        finally:
            _unsuppress_modelobject_logging()
            if mlflow is not None:
                if err is not None:
                    _safe_mlflow_set_tag(mlflow, "autots.run_status", "failed")
                    _safe_mlflow_set_tag(mlflow, "autots.error", repr(err))
                else:
                    _safe_mlflow_set_tag(mlflow, "autots.run_status", "finished")
                    forecast_df = getattr(result, "forecast", None)
                    if isinstance(forecast_df, pd.DataFrame):
                        _safe_mlflow_log_param(
                            mlflow, "forecast_rows", forecast_df.shape[0]
                        )
                        _safe_mlflow_log_param(
                            mlflow, "forecast_cols", forecast_df.shape[1]
                        )
                _end_run(
                    mlflow, run_ctx, status="FAILED" if err is not None else "FINISHED"
                )

    wrapped_predict._autots_mlflow_wrapped = True
    wrapped_predict._autots_mlflow_original = predict_method
    AutoTS.predict = wrapped_predict
    return True


def autolog(
    disable: bool = False,
    tried_models_logging: str = "single",
    log_model_runs: bool = True,
    log_model_forecasts: bool = True,
    silent: bool = True,
):
    """Enable or disable AutoTS MLflow autologging.

    Args:
        disable (bool): if True, disables autologging.
        tried_models_logging (str): one of "single", "individual", or "none".
            "single" logs one artifact table for all tried models (default).
            "individual" logs a nested run for each tried model row.
            "none" logs only run summary + best model metadata.
        log_model_runs (bool): if True, direct ModelObject.fit() calls are logged.
        log_model_forecasts (bool): if True, direct ModelObject.predict() calls are logged.
        silent (bool): suppress non-fatal integration messages.

    Returns:
        dict: active autolog configuration.
    """
    mode = str(tried_models_logging).lower().strip()
    if mode not in {"single", "individual", "none"}:
        raise ValueError(
            "tried_models_logging must be one of: 'single', 'individual', 'none'"
        )

    _AUTOLOG_CONFIG["enabled"] = not bool(disable)
    _AUTOLOG_CONFIG["tried_models_logging"] = mode
    _AUTOLOG_CONFIG["log_model_runs"] = bool(log_model_runs)
    _AUTOLOG_CONFIG["log_model_forecasts"] = bool(log_model_forecasts)
    _AUTOLOG_CONFIG["silent"] = bool(silent)

    if _AUTOLOG_CONFIG["enabled"]:
        _patch_autots_fit()
        _patch_autots_predict()

    cfg = _AUTOLOG_CONFIG.copy()
    cfg["mlflow_available"] = _get_mlflow() is not None
    return cfg


def modelobject_fit_start(model_obj, args=None, kwargs=None):
    """Called by ModelObject wrappers before fit()."""
    if not _AUTOLOG_CONFIG.get("enabled", False):
        return None
    if not _AUTOLOG_CONFIG.get("log_model_runs", True):
        return None
    if _is_modelobject_suppressed():
        return None

    mlflow = _get_mlflow()
    if mlflow is None:
        return None

    run_ctx = _start_run(
        mlflow,
        f"ModelObject.fit::{getattr(model_obj, 'name', model_obj.__class__.__name__)}",
    )
    _safe_mlflow_set_tag(mlflow, "autots.flavor", "ModelObject")
    _safe_mlflow_set_tag(mlflow, "autots.stage", "fit")
    _safe_mlflow_set_tag(mlflow, "autots.model_class", model_obj.__class__.__name__)
    _safe_mlflow_set_tag(mlflow, "autots.model_module", model_obj.__class__.__module__)

    _safe_mlflow_log_param(mlflow, "model_name", getattr(model_obj, "name", None))
    _safe_mlflow_log_param(mlflow, "frequency", getattr(model_obj, "frequency", None))
    _safe_mlflow_log_param(
        mlflow,
        "prediction_interval",
        getattr(model_obj, "prediction_interval", None),
    )

    df = None
    if args and len(args) > 0 and isinstance(args[0], pd.DataFrame):
        df = args[0]
    if isinstance(df, pd.DataFrame):
        _safe_mlflow_log_param(mlflow, "train_rows", df.shape[0])
        _safe_mlflow_log_param(mlflow, "train_cols", df.shape[1])
        try:
            _safe_mlflow_log_param(mlflow, "last_obs_date", _to_jsonable(df.index[-1]))
        except Exception:
            pass

    return {"mlflow": mlflow, "run_ctx": run_ctx, "closed": False}


def modelobject_fit_end(model_obj, context=None, result=None, error=None):
    """Called by ModelObject wrappers after fit()."""
    if context is None or context.get("closed", False):
        return
    context["closed"] = True
    mlflow = context.get("mlflow")
    run_ctx = context.get("run_ctx")
    status = "FAILED" if error is not None else "FINISHED"

    try:
        if mlflow is None:
            return
        if error is not None:
            _safe_mlflow_set_tag(mlflow, "autots.run_status", "failed")
            _safe_mlflow_set_tag(mlflow, "autots.error", repr(error))
        else:
            _safe_mlflow_set_tag(mlflow, "autots.run_status", "finished")
            fit_runtime = _timedelta_seconds(getattr(model_obj, "fit_runtime", None))
            if fit_runtime is not None:
                _safe_mlflow_log_metric(mlflow, "fit_runtime_seconds", fit_runtime)
            payload = _build_model_as_code_payload(model_obj)
            _safe_mlflow_log_dict(mlflow, payload, "modelobject/model_as_code.json")
    finally:
        if mlflow is not None:
            _end_run(mlflow, run_ctx, status=status)


def modelobject_predict_start(model_obj, args=None, kwargs=None):
    """Called by ModelObject wrappers before predict()."""
    if not _AUTOLOG_CONFIG.get("enabled", False):
        return None
    if not _AUTOLOG_CONFIG.get("log_model_forecasts", True):
        return None
    if _is_modelobject_suppressed():
        return None

    mlflow = _get_mlflow()
    if mlflow is None:
        return None

    run_ctx = _start_run(
        mlflow,
        f"ModelObject.predict::{getattr(model_obj, 'name', model_obj.__class__.__name__)}",
    )
    _safe_mlflow_set_tag(mlflow, "autots.flavor", "ModelObject")
    _safe_mlflow_set_tag(mlflow, "autots.stage", "predict")
    _safe_mlflow_set_tag(mlflow, "autots.model_class", model_obj.__class__.__name__)

    if kwargs and "forecast_length" in kwargs:
        _safe_mlflow_log_param(mlflow, "forecast_length", kwargs.get("forecast_length"))
    elif args:
        _safe_mlflow_log_param(mlflow, "forecast_length", args[0])

    return {"mlflow": mlflow, "run_ctx": run_ctx, "closed": False}


def modelobject_predict_end(model_obj, context=None, result=None, error=None):
    """Called by ModelObject wrappers after predict()."""
    if context is None or context.get("closed", False):
        return
    context["closed"] = True
    mlflow = context.get("mlflow")
    run_ctx = context.get("run_ctx")
    status = "FAILED" if error is not None else "FINISHED"

    try:
        if mlflow is None:
            return
        if error is not None:
            _safe_mlflow_set_tag(mlflow, "autots.run_status", "failed")
            _safe_mlflow_set_tag(mlflow, "autots.error", repr(error))
        else:
            _safe_mlflow_set_tag(mlflow, "autots.run_status", "finished")
            if result is not None:
                pred_runtime = _timedelta_seconds(
                    getattr(
                        result,
                        "predict_runtime",
                        getattr(model_obj, "predict_runtime", None),
                    )
                )
                if pred_runtime is not None:
                    _safe_mlflow_log_metric(
                        mlflow, "predict_runtime_seconds", pred_runtime
                    )
                forecast = getattr(result, "forecast", None)
                if isinstance(forecast, pd.DataFrame):
                    _safe_mlflow_log_param(mlflow, "forecast_rows", forecast.shape[0])
                    _safe_mlflow_log_param(mlflow, "forecast_cols", forecast.shape[1])
                    if not forecast.empty:
                        _safe_mlflow_log_param(
                            mlflow,
                            "forecast_start_date",
                            _to_jsonable(forecast.index[0]),
                        )
                        _safe_mlflow_log_param(
                            mlflow,
                            "forecast_end_date",
                            _to_jsonable(forecast.index[-1]),
                        )
            payload = _build_model_as_code_payload(model_obj)
            _safe_mlflow_log_dict(
                mlflow, payload, "modelobject/model_as_code_predict.json"
            )
    finally:
        if mlflow is not None:
            _end_run(mlflow, run_ctx, status=status)


def load_model(run_id: str, artifact_path: str = None) -> Any:
    """Load an AutoTS or ModelObject instance from an MLflow run.

    Args:
        run_id (str): MLflow run ID.
        artifact_path (str): Path to the JSON artifact.
            If None, tries 'autots/model_as_code.json' then 'modelobject/model_as_code.json'.

    Returns:
        AutoTS or ModelObject: The reconstructed model instance.
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        raise ImportError("mlflow is required for load_model")

    import json
    import tempfile

    # Selection logic for finding the right artifact
    search_paths = (
        [artifact_path]
        if artifact_path
        else ["autots/model_as_code.json", "modelobject/model_as_code.json"]
    )

    payload = None
    with tempfile.TemporaryDirectory() as tmpdir:
        for p in search_paths:
            try:
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=p, dst_path=tmpdir
                )
                with open(local_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                break
            except Exception:
                continue

    if payload is None:
        raise FileNotFoundError(f"No model artifact found for run {run_id}")

    model_class = payload.get("model_class")
    if model_class == "AutoTS":
        from autots import AutoTS

        # Get params from MLflow tracking to initialize correctly
        try:
            run = mlflow.get_run(run_id)
            params = run.data.params
        except Exception:
            params = {}

        def get_p(key, default, transform=lambda x: x):
            val = params.get(key)
            if val is None or val == "None":
                return default
            try:
                return transform(val)
            except Exception:
                return default

        # Start with original params, but default max_generations=0 to avoid search on refit
        model = AutoTS(
            forecast_length=get_p("forecast_length", 14, int),
            frequency=get_p("frequency", "infer"),
            prediction_interval=get_p("prediction_interval", 0.9, float),
            max_generations=get_p("max_generations", 0, int),
            num_validations=get_p("num_validations", 0, int),
            ensemble=get_p("ensemble", None),
            random_seed=get_p("random_seed", 2022, int),
        )
        # Restore winner state
        model.best_model_name = payload.get("best_model_name")
        model.best_model_params = payload.get("best_model_parameters")
        model.best_model_transformation_params = payload.get(
            "best_model_transformation_params",
            payload.get("best_model_transformation_parameters"),
        )
        model.best_model_ensemble = payload.get("best_model_ensemble", 0)
        model.best_model_id = payload.get("best_model_id")
        return model
    else:
        from autots.evaluator.auto_model import ModelMonster

        # Try to restore important top-level attributes
        return ModelMonster(
            model=model_class,
            parameters=payload.get("model_parameters", {}),
            frequency=payload.get("frequency", "infer"),
            prediction_interval=payload.get("prediction_interval", 0.9),
        )


__all__ = [
    "autolog",
    "load_model",
    "modelobject_fit_start",
    "modelobject_fit_end",
    "modelobject_predict_start",
    "modelobject_predict_end",
]
