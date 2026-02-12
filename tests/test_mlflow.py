# -*- coding: utf-8 -*-
"""MLflow integration tests for AutoTS autolog support."""

import tempfile
import unittest

import pandas as pd

from autots import AutoTS
from autots.models.basics import LastValueNaive
from autots.tools.mlflow import autolog

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None


class AutoTSMLflowTest(unittest.TestCase):
    def test_autolog_configuration(self):
        cfg = autolog(disable=False)
        self.assertIn("enabled", cfg)
        self.assertIn("tried_models_logging", cfg)
        self.assertEqual(cfg["tried_models_logging"], "single")
        cfg = autolog(disable=True)
        self.assertFalse(cfg["enabled"])

    @unittest.skipIf(mlflow is None, "mlflow not installed")
    def test_modelobject_autolog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            mlflow.set_experiment("autots_modelobject_autolog")
            autolog(disable=False, tried_models_logging="single")

            df = pd.DataFrame(
                {"series": [1, 2, 3, 4, 5, 6, 7, 8]},
                index=pd.date_range("2020-01-01", periods=8, freq="D"),
            )
            model = LastValueNaive(frequency="infer", prediction_interval=0.9)
            model.fit(df)
            model.predict(forecast_length=2)

            client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
            exp = client.get_experiment_by_name("autots_modelobject_autolog")
            runs = client.search_runs([exp.experiment_id])
            self.assertTrue(
                any(run.data.tags.get("autots.flavor") == "ModelObject" for run in runs)
            )
            autolog(disable=True)

    @unittest.skipIf(mlflow is None, "mlflow not installed")
    def test_autots_autolog_individual_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            mlflow.set_experiment("autots_autolog_individual")
            autolog(disable=False, tried_models_logging="individual")

            df = pd.DataFrame(
                {
                    "series_1": list(range(1, 40)),
                    "series_2": list(range(2, 41)),
                },
                index=pd.date_range("2020-01-01", periods=39, freq="D"),
            )

            model = AutoTS(
                forecast_length=3,
                frequency="infer",
                max_generations=0,
                num_validations=0,
                model_list=["LastValueNaive"],
                transformer_list="superfast",
                transformer_max_depth=1,
                initial_template="Random",
                ensemble=None,
                models_to_validate=1,
                verbose=-1,
            )
            model.fit(df)
            model.predict(forecast_length=3)

            client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
            exp = client.get_experiment_by_name("autots_autolog_individual")
            runs = client.search_runs([exp.experiment_id])
            flavors = {run.data.tags.get("autots.flavor") for run in runs}

            self.assertIn("AutoTS", flavors)
            self.assertIn("AutoTSModelCandidate", flavors)
            self.assertNotIn("ModelObject", flavors)
            autolog(disable=True)


if __name__ == "__main__":
    unittest.main()
