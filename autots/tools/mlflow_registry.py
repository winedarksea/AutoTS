"""Version AutoTS Top-N templates in the MLflow Model Registry.

The template, not a model weight, is the state that carries across production runs:
each run seeds its search with the last Top-N and exports a new one. Registering it
gives every template a version, and aliases select which one a run uses:

    rolling (evolve): import "models:/demand_seed@rolling", fit, log_template(alias="rolling")
    approved (frozen): mlflow_load_model("models:/demand_seed@approved"), fit_data, predict

Rolling back is moving the alias to an earlier version in the MLflow UI.

MLflow registers only real models (an MLmodel file), so the template is logged as a
pyfunc (autots/tools/mlflow_pyfunc.py). The same version can be served: its predict
refits the winning model on the history it is given, because a fitted AutoTS cannot
be pickled.

Artifacts of the logged model: template.csv (best model first, with validation
results) and autots_config.json (same payload as the autologged model_as_code.json).
"""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd

from autots.tools.mlflow import (
    _build_autots_model_as_code_payload,
    _extract_autots_best_row,
    _get_mlflow,
    _to_jsonable,
)

TEMPLATE_FILE = "template.csv"
CONFIG_FILE = "autots_config.json"
_PYFUNC_CODE_PATH = os.path.join(os.path.dirname(__file__), "mlflow_pyfunc.py")


def _require_mlflow():
    mlflow = _get_mlflow()
    if mlflow is None:
        raise ImportError("mlflow is required for the AutoTS template registry")
    return mlflow


def _pyfunc_signature():
    """Declare predict params; MLflow drops any param missing from the signature.

    No input schema, since the series columns differ between datasets.
    """
    from mlflow.models import ModelSignature
    from mlflow.types.schema import ParamSchema, ParamSpec

    return ModelSignature(
        inputs=None,
        outputs=None,
        params=ParamSchema(
            [
                ParamSpec("forecast_length", "long", None),
                ParamSpec("prediction_interval", "double", None),
                ParamSpec("date_col", "string", None),
                ParamSpec("output", "string", "long"),
            ]
        ),
    )


def _best_model_first(template: pd.DataFrame, best_model_id) -> pd.DataFrame:
    """import_best_model takes the first row, but export_template sorts by Score only."""
    is_best = template["ID"] == best_model_id
    return pd.concat([template[is_best], template[~is_best]]).reset_index(drop=True)


def _download_artifact(uri: str, file_name: str, dst_path: str) -> str:
    mlflow = _require_mlflow()
    try:
        # pyfunc log_model stores its `artifacts` dict under artifacts/ in the model
        return mlflow.artifacts.download_artifacts(
            artifact_uri=f"{uri.rstrip('/')}/artifacts/{file_name}", dst_path=dst_path
        )
    except Exception as e:
        # a missing alias/model is the expected first-run case, so give it one type
        raise FileNotFoundError(f"No AutoTS {file_name} found at {uri}: {e}") from e


def download_template(uri: str) -> pd.DataFrame:
    """Top-N template DataFrame from a logged template, e.g. "models:/demand_seed@rolling".

    Raises FileNotFoundError if the alias, version, or model does not exist.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        return pd.read_csv(_download_artifact(uri, TEMPLATE_FILE, tmpdir))


def download_config(uri: str) -> dict:
    """AutoTS rebuild payload (init_params + best model) from a logged template."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(
            _download_artifact(uri, CONFIG_FILE, tmpdir), "r", encoding="utf-8"
        ) as f:
            return json.load(f)


def _version_tags(autots_model, template: pd.DataFrame, survival_rate=None) -> dict:
    from autots.tools.mlflow_insights import ensemble_leaf_models, seed_drift_metrics

    tags = {
        "autots.best_model_name": autots_model.best_model_name,
        "autots.best_model_id": autots_model.best_model_id,
        "autots.best_model_ensemble": autots_model.best_model_ensemble,
        "autots.n_models": len(template),
    }
    if autots_model.best_model_ensemble >= 1:
        families = set(ensemble_leaf_models(autots_model.best_model_params).values())
        tags["autots.ensemble_families"] = json.dumps(sorted(families))
    try:
        drift = seed_drift_metrics(
            autots_model, getattr(autots_model, "mlflow_seed_ids", None)
        )
        tags.update(drift)
    except Exception:
        pass
    if survival_rate is not None:
        tags["drift.seed_survival_rate"] = round(survival_rate, 4)
    return {k: str(v) for k, v in tags.items()}


def log_template(
    autots_model,
    registry_name: str = None,
    alias: str = None,
    n: int = 40,
    run_id: str = None,
    pip_requirements: list = None,
    name: str = "autots_template",
    **export_kwargs,
) -> dict:
    """Log a fitted AutoTS's Top-N template as a pyfunc model, optionally registered.

    Args:
        autots_model (AutoTS): fitted AutoTS.
        registry_name (str): registered model name. None logs without registering.
        alias (str): alias to point at the new version, e.g. "rolling". Needs registry_name.
            Leave None for frozen mode, where an approved alias is moved by hand.
        n (int): models in the template, passed to export_template.
        run_id (str): run to attach to. Defaults to the autologged fit run of this
            model, then the active run, else a new run.
        pip_requirements (list): serving requirements; defaults to this autots version.
        name (str): name of the logged model within the run.
        **export_kwargs: passed to AutoTS.export_template, e.g. max_per_model_class=6,
            min_metrics, max_metrics, focus_models.

    Returns:
        dict: model_uri, run_id, and (when registered) registered_model_uri and version.
    """
    mlflow = _require_mlflow()
    if alias is not None and registry_name is None:
        raise ValueError("alias requires registry_name")
    if getattr(autots_model, "best_model", pd.DataFrame()).empty:
        raise ValueError("AutoTS has no best model; run fit() before log_template")

    from autots import __version__

    template = autots_model.export_template(
        filename=None, models="best", n=n, include_results=True, **export_kwargs
    )
    template = _best_model_first(template, autots_model.best_model_id)
    from autots.tools.mlflow_insights import seed_survival_rate

    survival_rate = seed_survival_rate(
        getattr(autots_model, "mlflow_seed_ids", None), template
    )
    payload = _build_autots_model_as_code_payload(
        autots_model, best_row=_extract_autots_best_row(autots_model)
    )

    run_id = run_id or getattr(autots_model, "mlflow_run_id", None)
    active = mlflow.active_run()
    started = False
    if run_id is not None and (active is None or active.info.run_id != run_id):
        mlflow.start_run(run_id=run_id, nested=active is not None)
        started = True
    elif active is None:
        mlflow.start_run(run_name="AutoTS.template")
        started = True
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = os.path.join(tmpdir, TEMPLATE_FILE)
            config_path = os.path.join(tmpdir, CONFIG_FILE)
            template.to_csv(template_path, index=False)
            # same jsonable conversion as the autologged model_as_code.json
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)
            model_info = mlflow.pyfunc.log_model(
                name=name,
                python_model=_PYFUNC_CODE_PATH,
                artifacts={"template": template_path, "config": config_path},
                signature=_pyfunc_signature(),
                pip_requirements=pip_requirements or [f"autots=={__version__}"],
                registered_model_name=registry_name,
            )
        if survival_rate is not None:
            mlflow.log_metric("drift.seed_survival_rate", survival_rate)
        logged_run_id = mlflow.active_run().info.run_id
    finally:
        if started:
            mlflow.end_run()

    result = {"model_uri": model_info.model_uri, "run_id": logged_run_id}
    if registry_name is None:
        return result

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    version = str(model_info.registered_model_version)
    for key, value in _version_tags(autots_model, template, survival_rate).items():
        client.set_model_version_tag(registry_name, version, key, value)
    if alias is not None:
        client.set_registered_model_alias(registry_name, alias, version)
    result["version"] = version
    result["registered_model_uri"] = f"models:/{registry_name}/{version}"
    return result


__all__ = [
    "log_template",
    "download_template",
    "download_config",
    "TEMPLATE_FILE",
    "CONFIG_FILE",
]
