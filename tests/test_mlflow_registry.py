# -*- coding: utf-8 -*-
"""MLflow template registry, pyfunc serving, and fit insights for AutoTS."""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from autots import AutoTS, mlflow_load_model, mlflow_log_template
from autots.tools.mlflow import autolog
from autots.tools.mlflow_insights import (
    collect_seed_ids,
    ensemble_leaf_models,
    seed_survival_rate,
)
from autots.tools.mlflow_registry import _best_model_first, download_template

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None

REGISTRY_NAME = "demand_seed"
ROLLING_URI = f"models:/{REGISTRY_NAME}@rolling"


def _make_seasonal_df(rows=120, cols=3, seed=7):
    """Weekly-seasonal noisy series, so models disagree (same as test_mlflow.py)."""
    rng = np.random.default_rng(seed)
    t = np.arange(rows)[:, None]
    data = (
        50
        + 0.2 * t * np.arange(1, cols + 1)
        + 10 * np.sin(2 * np.pi * t / 7 + np.arange(cols))
        + rng.normal(0, 2, (rows, cols))
    )
    return pd.DataFrame(
        data,
        index=pd.date_range("2020-01-01", periods=rows, freq="D"),
        columns=[f"series_{i}" for i in range(cols)],
    )


def _use_temp_tracking_store(experiment_name):
    """Fresh sqlite store with its own artifact root (same as test_mlflow.py)."""
    tmpdir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(tmpdir, 'mlflow.db')}")
    mlflow.create_experiment(
        experiment_name, artifact_location=Path(tmpdir, "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_name)
    return tmpdir


def _autots_kwargs(**overrides):
    kwargs = dict(
        forecast_length=7,
        frequency="D",
        model_list="superfast",
        ensemble="horizontal-max",
        num_validations=1,
        max_generations=2,
        verbose=0,
        random_seed=1,
        n_jobs=1,
    )
    kwargs.update(overrides)
    return kwargs


def _end_active_runs():
    while mlflow.active_run() is not None:
        mlflow.end_run()


class TestRegistryHelpers(unittest.TestCase):
    """Pure helpers; no MLflow tracking store needed."""

    def test_best_model_first_when_score_is_not_lowest(self):
        template = pd.DataFrame(
            {"ID": ["a", "b", "winner"], "Score": [1.0, 2.0, 3.0]}
        )
        ordered = _best_model_first(template, "winner")
        self.assertEqual(ordered["ID"].tolist(), ["winner", "a", "b"])

    def test_ensemble_leaf_models_recurses_nested_ensembles(self):
        params = {
            "models": {
                "m1": {"Model": "ETS"},
                "nested": {
                    "Model": "Ensemble",
                    "ModelParameters": '{"models": {"m2": {"Model": "Theta"}}}',
                },
            }
        }
        self.assertEqual(ensemble_leaf_models(params), {"m1": "ETS", "m2": "Theta"})

    def test_seed_ids_skip_unseeded_templates(self):
        model = AutoTS(**_autots_kwargs())
        self.assertEqual(collect_seed_ids(model), [])
        self.assertIsNone(seed_survival_rate([], pd.DataFrame()))

    def test_seed_survival_rate_expands_ensembles(self):
        template = pd.DataFrame(
            {
                "ID": ["a", "ens"],
                "Ensemble": [0, 2],
                "ModelParameters": ["{}", '{"models": {"b": {"Model": "ETS"}}}'],
            }
        )
        self.assertEqual(seed_survival_rate(["a", "b", "c", "d"], template), 0.5)

    def test_csv_templates_still_load(self):
        template = pd.DataFrame(
            {
                "ID": ["x"],
                "Model": ["LastValueNaive"],
                "ModelParameters": ["{}"],
                "TransformationParameters": ['{"fillna": "ffill"}'],
                "Ensemble": [0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "template.csv")
            template.to_csv(path, index=False)
            loaded = AutoTS(**_autots_kwargs()).load_template(path)
        self.assertEqual(loaded["ID"].tolist(), ["x"])


@unittest.skipIf(mlflow is None, "mlflow not installed")
class TestTemplateRegistryLoop(unittest.TestCase):
    """The production loop: initial fit -> @rolling v1, seeded refit -> v2."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        _end_active_runs()
        cls.tmpdir = _use_temp_tracking_store("test_registry")
        autolog(tried_models_logging="single", silent=True)
        cls.df = _make_seasonal_df(rows=150, cols=4)
        cls.client = MlflowClient()

        cls.model_v1 = AutoTS(**_autots_kwargs())
        cls.model_v1.fit(cls.df.iloc[:130])
        cls.info_v1 = mlflow_log_template(
            cls.model_v1, REGISTRY_NAME, alias="rolling", n=20
        )

        cls.model_v2 = AutoTS(**_autots_kwargs())
        cls.model_v2.import_template(
            ROLLING_URI, method="addon", force_validation=True
        )
        cls.model_v2.fit(cls.df.iloc[:140])
        cls.info_v2 = mlflow_log_template(
            cls.model_v2, REGISTRY_NAME, alias="rolling", n=20
        )
        cls.forecast_v2 = cls.model_v2.predict().forecast

    @classmethod
    def tearDownClass(cls):
        autolog(disable=True)
        _end_active_runs()

    def tearDown(self):
        # rollback tests move the alias; restore it for the others
        self.client.set_registered_model_alias(
            REGISTRY_NAME, "rolling", self.info_v2["version"]
        )

    def test_alias_moves_to_newest_version(self):
        self.assertEqual(self.info_v1["version"], "1")
        self.assertEqual(self.info_v2["version"], "2")
        alias_version = self.client.get_model_version_by_alias(REGISTRY_NAME, "rolling")
        self.assertEqual(str(alias_version.version), "2")
        # the template is attached to its autologged fit run
        self.assertEqual(self.info_v2["run_id"], self.model_v2.mlflow_run_id)

    def test_rollback_by_moving_alias(self):
        self.client.set_registered_model_alias(
            REGISTRY_NAME, "rolling", self.info_v1["version"]
        )
        template = AutoTS(**_autots_kwargs()).load_template(ROLLING_URI)
        self.assertEqual(template["ID"].iloc[0], self.model_v1.best_model_id)

    def test_import_best_model_from_alias_picks_winner(self):
        model = AutoTS(**_autots_kwargs())
        model.import_best_model(ROLLING_URI, enforce_model_list=False)
        self.assertEqual(model.best_model_id, self.model_v2.best_model_id)

    def test_frozen_mode_refits_winner_without_search(self):
        loaded = mlflow_load_model(ROLLING_URI)
        loaded.fit_data(self.df.iloc[:140])
        forecast = loaded.predict().forecast
        self.assertTrue(loaded.initial_results.model_results.empty)
        pd.testing.assert_frame_equal(forecast, self.forecast_v2)

    def test_pyfunc_matches_autots_forecast(self):
        served = mlflow.pyfunc.load_model(self.info_v2["registered_model_uri"])
        long_out = served.predict(self.df.iloc[:140], params={"forecast_length": 5})
        self.assertEqual(
            list(long_out.columns),
            ["datetime", "SeriesID", "Value", "PredictionInterval"],
        )
        self.assertEqual(long_out["datetime"].nunique(), 5)

        # REST/JSON payloads lose the index, so dates arrive as a string column
        rest_input = self.df.iloc[:140].rename_axis("date").reset_index()
        rest_input["date"] = rest_input["date"].astype(str)
        wide_out = served.predict(
            rest_input, params={"output": "forecast", "date_col": "date"}
        )
        np.testing.assert_allclose(
            wide_out.drop(columns="datetime").to_numpy(), self.forecast_v2.to_numpy()
        )

    def test_fit_insights_logged(self):
        run_id = self.info_v2["run_id"]
        artifacts = {a.path for a in self.client.list_artifacts(run_id, "autots")}
        for path in [
            "autots/per_series_best.json",
            "autots/validation_results.json",
            "autots/ensemble_routing.json",
        ]:
            self.assertIn(path, artifacts)
        history = self.client.get_metric_history(
            run_id, "autots.generation_best_score"
        )
        self.assertGreater(len(history), 1)
        values = [m.value for m in sorted(history, key=lambda m: m.step)]
        self.assertEqual(values, sorted(values, reverse=True))
        tags = mlflow.get_run(run_id).data.tags
        self.assertIn("best.ensemble_families", tags)
        per_series = mlflow.load_table("autots/per_series_best.json", run_ids=[run_id])
        self.assertEqual(sorted(per_series["Series"]), sorted(self.df.columns))

    def test_seed_drift_logged_only_when_seeded(self):
        seeded = mlflow.get_run(self.info_v2["run_id"]).data
        survival = seeded.metrics["drift.seed_survival_rate"]
        self.assertTrue(0 <= survival <= 1)
        self.assertIn(seeded.tags["drift.winner_origin"], {"seed", "new", "mixed"})
        version_tags = self.client.get_model_version(REGISTRY_NAME, "2").tags
        self.assertIn("drift.seed_survival_rate", version_tags)

        unseeded = mlflow.get_run(self.info_v1["run_id"]).data
        self.assertFalse(any(k.startswith("drift.") for k in unseeded.metrics))
        self.assertFalse(any(k.startswith("drift.") for k in unseeded.tags))

    def test_missing_alias_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            download_template(f"models:/{REGISTRY_NAME}@does_not_exist")

    def test_alias_requires_registry_name(self):
        with self.assertRaises(ValueError):
            mlflow_log_template(self.model_v2, alias="rolling")


@unittest.skipIf(mlflow is None, "mlflow not installed")
class TestInsightsDisabled(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        _end_active_runs()
        _use_temp_tracking_store("test_insights_disabled")
        autolog(tried_models_logging="single", silent=True, log_insights=False)

    def tearDown(self):
        autolog(disable=True)
        _end_active_runs()

    def test_no_insight_artifacts(self):
        model = AutoTS(**_autots_kwargs(ensemble=None, max_generations=1))
        model.fit(_make_seasonal_df(rows=100, cols=2))
        run_id = model.mlflow_run_id
        artifacts = {a.path for a in MlflowClient().list_artifacts(run_id, "autots")}
        self.assertIn("autots/model_as_code.json", artifacts)
        self.assertNotIn("autots/per_series_best.json", artifacts)
        self.assertNotIn(
            "autots.generation_best_score", mlflow.get_run(run_id).data.metrics
        )


if __name__ == "__main__":
    unittest.main()
