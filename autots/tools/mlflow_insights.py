"""Explain an AutoTS search in MLflow: convergence, per-series error, routing, seed drift.

Called by the autolog fit wrapper in autots.tools.mlflow (`log_insights=True`).
Every item is computed after fit from results AutoTS already keeps, so fit itself
needs no hooks, and each item fails independently without breaking the fit.

Logged to the AutoTS.fit run:
    metrics (step = generation): autots.generation_best_score, autots.generation_best_smape
        Scores are relative to the final pool of models, so compare them within a run only.
    tables (mlflow.log_table, sortable in the UI):
        autots/validation_results.json, autots/per_series_best.json,
        autots/ensemble_routing.json and autots/mosaic_routing.json (horizontal/mosaic winners)
    best.worst_series_mape, best.median_series_mape, best.worst_series,
    best.ensemble_families, best.ensemble_model_count
    seed drift (only when the search was seeded with an exported template):
        drift.seed_count, drift.winner_seed_share, drift.winner_origin
        drift.seed_survival_rate is logged by mlflow_log_template, since it compares the
        seed with the new exported template and so needs the export settings.
        Result IDs are rebuilt from the parameters a model actually used, so a few seeds
        can count as new.
"""

from __future__ import annotations

import json

import pandas as pd

from autots.tools.mlflow import (
    _safe_float,
    _safe_mlflow_log_metric,
    _safe_mlflow_set_tag,
)


def _as_dict(params) -> dict:
    if isinstance(params, dict):
        return params
    try:
        loaded = json.loads(params)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def ensemble_leaf_models(model_params) -> dict:
    """Map of constituent model ID -> model name, recursing through nested ensembles."""
    leaves = {}
    for model_id, spec in _as_dict(model_params).get("models", {}).items():
        if not isinstance(spec, dict):
            continue
        if spec.get("Model") == "Ensemble":
            leaves.update(ensemble_leaf_models(spec.get("ModelParameters", {})))
        else:
            leaves[str(model_id)] = spec.get("Model")
    return leaves


def collect_seed_ids(autots_obj) -> list:
    """Single-model IDs of an imported template, read before fit overwrites initial_template.

    Exported templates always carry an ID; general/random templates have none, so an
    unseeded search returns []. Ensemble rows contribute their constituents, because the
    ensemble itself is rebuilt (with a new ID) in every search.
    """
    seed_ids = set()
    for frame in [
        getattr(autots_obj, "initial_template", None),
        getattr(autots_obj, "validate_import", None),
    ]:
        if not isinstance(frame, pd.DataFrame) or "ID" not in frame.columns:
            continue
        seed_ids.update(template_model_ids(frame[frame["ID"].notna()]))
    return sorted(seed_ids)


def template_model_ids(template: pd.DataFrame) -> set:
    """Single-model IDs in an exported template, with ensembles expanded to constituents."""
    is_ensemble = pd.to_numeric(template["Ensemble"], errors="coerce") >= 1
    model_ids = set(template.loc[~is_ensemble, "ID"].astype(str))
    for model_params in template.loc[is_ensemble, "ModelParameters"]:
        model_ids.update(ensemble_leaf_models(model_params))
    return model_ids


def seed_survival_rate(seed_ids, template: pd.DataFrame):
    """Share of seed models kept in the new exported template; None when unseeded.

    Compares like with like only if both templates use the same export settings.
    It is below 1 even on unchanged data, because every search also finds new models
    (0.42 measured on a 2-generation superfast refit), so watch its trend across runs,
    where a sharp drop suggests the data changed (concept drift).
    """
    if not seed_ids:
        return None
    seed = set(seed_ids)
    return len(seed & template_model_ids(template)) / len(seed)


def seed_drift_metrics(autots_obj, seed_ids) -> dict:
    """Whether the winner came from the incoming seed. Empty when unseeded."""
    if not seed_ids:
        return {}
    seed = set(seed_ids)
    if autots_obj.best_model_ensemble >= 1:
        winner_ids = list(ensemble_leaf_models(autots_obj.best_model_params))
    else:
        winner_ids = [str(autots_obj.best_model_id)]
    winner_seed_share = (
        sum(x in seed for x in winner_ids) / len(winner_ids) if winner_ids else 0.0
    )
    if winner_seed_share == 1:
        winner_origin = "seed"
    elif winner_seed_share == 0:
        winner_origin = "new"
    else:
        winner_origin = "mixed"
    return {
        "drift.seed_count": len(seed),
        "drift.winner_seed_share": winner_seed_share,
        "drift.winner_origin": winner_origin,
    }


def _safe_log_table(mlflow, df: pd.DataFrame, artifact_file: str):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    table = df.copy()
    table.columns = [str(x) for x in table.columns]
    for col in table.columns:
        # timedeltas and other objects don't serialize to JSON
        if table[col].dtype.kind in "mMO":
            table[col] = table[col].astype(str)
    try:
        mlflow.log_table(table, artifact_file)
    except Exception:
        pass


def _log_step_series(mlflow, key: str, series: pd.Series):
    """One metric point per generation, batched so long searches don't make 1000s of calls."""
    points = [
        (int(step), _safe_float(value)) for step, value in series.items() if pd.notna(step)
    ]
    points = [(step, value) for step, value in points if value is not None]
    if not points:
        return
    try:
        import time

        from mlflow.entities import Metric
        from mlflow.tracking import MlflowClient

        timestamp = int(time.time() * 1000)
        MlflowClient().log_batch(
            mlflow.active_run().info.run_id,
            metrics=[Metric(key, value, timestamp, step) for step, value in points],
        )
    except Exception:
        pass


def log_generation_convergence(mlflow, autots_obj):
    """Best single-model score per generation, the plot_generation_loss curve as metrics."""
    results = autots_obj.initial_results.model_results
    singles = results[(results["ValidationRound"] == 0) & (results["Ensemble"] < 1)]
    if singles.empty or "Generation" not in singles.columns:
        return
    by_generation = singles.groupby("Generation")
    _log_step_series(
        mlflow, "autots.generation_best_score", by_generation["Score"].min().cummin()
    )
    if "smape" in singles.columns:
        _log_step_series(
            mlflow,
            "autots.generation_best_smape",
            by_generation["smape"].min().cummin(),
        )


def log_per_series_errors(mlflow, autots_obj):
    """Per-series error of the winner; a good overall score can hide one bad series."""
    mape = autots_obj.best_model_per_series_mape()
    table = mape.rename("MAPE").to_frame()
    try:
        table["Score"] = autots_obj.best_model_per_series_score()
    except Exception:
        pass
    _safe_log_table(
        mlflow, table.rename_axis("Series").reset_index(), "autots/per_series_best.json"
    )
    if mape.notna().any():
        _safe_mlflow_log_metric(mlflow, "best.worst_series_mape", mape.max())
        _safe_mlflow_log_metric(mlflow, "best.median_series_mape", mape.median())
        _safe_mlflow_set_tag(mlflow, "best.worst_series", mape.idxmax())


def log_ensemble_routing(mlflow, autots_obj):
    """Which models the winning ensemble uses, and which series each one serves."""
    if autots_obj.best_model_ensemble < 1:
        return
    leaves = ensemble_leaf_models(autots_obj.best_model_params)
    _safe_mlflow_set_tag(
        mlflow, "best.ensemble_families", sorted(set(leaves.values()))
    )
    _safe_mlflow_log_metric(mlflow, "best.ensemble_model_count", len(leaves))
    if autots_obj.best_model_ensemble != 2:
        return
    try:
        _safe_log_table(
            mlflow, autots_obj.horizontal_to_df(), "autots/ensemble_routing.json"
        )
    except Exception:
        pass
    if "mosaic" in str(autots_obj.best_model_params.get("model_name", "")).lower():
        try:
            mosaic = autots_obj.mosaic_to_df().rename_axis("ForecastStep")
            _safe_log_table(
                mlflow, mosaic.reset_index(), "autots/mosaic_routing.json"
            )
        except Exception:
            pass


def log_autots_insights(mlflow, autots_obj, seed_ids=None):
    """Log every insight; each one is skipped on its own if it cannot be computed."""
    for log_item in [
        log_generation_convergence,
        log_per_series_errors,
        log_ensemble_routing,
    ]:
        try:
            log_item(mlflow, autots_obj)
        except Exception:
            pass
    try:
        _safe_log_table(
            mlflow,
            autots_obj.validation_results.model_results,
            "autots/validation_results.json",
        )
    except Exception:
        pass
    try:
        for key, value in seed_drift_metrics(autots_obj, seed_ids).items():
            if isinstance(value, str):
                _safe_mlflow_set_tag(mlflow, key, value)
            else:
                _safe_mlflow_log_metric(mlflow, key, value)
    except Exception:
        pass
