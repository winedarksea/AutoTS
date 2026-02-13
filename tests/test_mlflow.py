# -*- coding: utf-8 -*-
"""MLflow integration tests for AutoTS autolog support."""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from autots import AutoTS
from autots.models.basics import LastValueNaive, AverageValueNaive
from autots.tools.mlflow import (
    autolog,
    _AUTOLOG_CONFIG,
    _get_mlflow,
    _safe_float,
    _to_jsonable,
    _short_param_value,
    _build_model_as_code_payload,
    _build_autots_model_as_code_payload,
    modelobject_fit_start,
    modelobject_fit_end,
    modelobject_predict_start,
    modelobject_predict_end,
)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None


def _make_df(rows=40, cols=2, freq="D"):
    """Create a small wide-form DataFrame for testing."""
    data = {f"series_{i}": np.arange(1, rows + 1) + i * 0.1 for i in range(cols)}
    return pd.DataFrame(
        data, index=pd.date_range("2020-01-01", periods=rows, freq=freq)
    )


class TestHelperFunctions(unittest.TestCase):
    """Tests for standalone helper functions that don't require MLflow."""

    def test_safe_float_valid(self):
        self.assertEqual(_safe_float(3.14), 3.14)
        self.assertEqual(_safe_float(42), 42.0)
        self.assertEqual(_safe_float(np.float64(2.5)), 2.5)
        self.assertEqual(_safe_float(np.int32(7)), 7.0)

    def test_safe_float_invalid(self):
        self.assertIsNone(_safe_float(float("inf")))
        self.assertIsNone(_safe_float(float("nan")))
        self.assertIsNone(_safe_float("abc"))
        self.assertIsNone(_safe_float(None))
        self.assertIsNone(_safe_float([1, 2]))

    def test_short_param_value(self):
        self.assertEqual(_short_param_value("hello"), "hello")
        self.assertEqual(_short_param_value(None), "None")
        self.assertEqual(_short_param_value({"a": 1}), '{"a": 1}')
        # Truncation
        long_val = "x" * 1000
        result = _short_param_value(long_val, max_len=50)
        self.assertEqual(len(result), 50)
        self.assertTrue(result.endswith("..."))

    def test_to_jsonable_types(self):
        self.assertEqual(_to_jsonable("str"), "str")
        self.assertEqual(_to_jsonable(42), 42)
        self.assertEqual(_to_jsonable(None), None)
        self.assertIsInstance(_to_jsonable(pd.Timestamp("2020-01-01")), str)
        self.assertIsInstance(
            _to_jsonable(np.float64(1.5)), float
        )
        self.assertEqual(_to_jsonable(np.array([1, 2])), [1, 2])
        nested = {"a": np.int32(1), "b": [np.float64(2.0)]}
        result = _to_jsonable(nested)
        self.assertEqual(result, {"a": 1, "b": [2.0]})
        # Verify JSON-serializable
        json.dumps(result)

    def test_to_jsonable_pandas_types(self):
        s = pd.Series([1, 2], index=["a", "b"])
        result = _to_jsonable(s)
        self.assertIsInstance(result, dict)
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
        result = _to_jsonable(idx)
        self.assertIsInstance(result, list)

    def test_build_model_as_code_payload(self):
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        df = _make_df(10, 1)
        model.fit(df)
        payload = _build_model_as_code_payload(model)
        self.assertEqual(payload["model_name"], "LastValueNaive")
        self.assertTrue(payload["model_as_code"])
        self.assertTrue(payload["refit_expected"])
        self.assertIn("last_obs_date", payload)
        # Verify JSON-serializable
        json.dumps(_to_jsonable(payload))

    def test_build_model_as_code_payload_with_extra(self):
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        payload = _build_model_as_code_payload(model, extra={"custom_key": "value"})
        self.assertEqual(payload["custom_key"], "value")


class TestAutologConfiguration(unittest.TestCase):
    """Test autolog() configuration behavior."""

    def setUp(self):
        autolog(disable=True)

    def tearDown(self):
        autolog(disable=True)

    def test_default_config(self):
        cfg = autolog(disable=False)
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["tried_models_logging"], "single")
        self.assertTrue(cfg["log_model_runs"])
        self.assertTrue(cfg["log_model_forecasts"])
        self.assertTrue(cfg["silent"])
        self.assertIn("mlflow_available", cfg)

    def test_disable(self):
        autolog(disable=False)
        cfg = autolog(disable=True)
        self.assertFalse(cfg["enabled"])

    def test_tried_models_modes(self):
        for mode in ["single", "individual", "none"]:
            cfg = autolog(disable=False, tried_models_logging=mode)
            self.assertEqual(cfg["tried_models_logging"], mode)

    def test_invalid_tried_models_mode_raises(self):
        with self.assertRaises(ValueError):
            autolog(tried_models_logging="invalid_mode")

    def test_log_flags(self):
        cfg = autolog(
            disable=False,
            log_model_runs=False,
            log_model_forecasts=False,
            silent=False,
        )
        self.assertFalse(cfg["log_model_runs"])
        self.assertFalse(cfg["log_model_forecasts"])
        self.assertFalse(cfg["silent"])


class TestNoMlflowFallback(unittest.TestCase):
    """Test that everything is a no-op when autolog is disabled."""

    def setUp(self):
        autolog(disable=True)

    def test_modelobject_fit_start_returns_none_when_disabled(self):
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        result = modelobject_fit_start(model)
        self.assertIsNone(result)

    def test_modelobject_fit_end_noop_with_none_context(self):
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        # Should not raise
        modelobject_fit_end(model, context=None, result=None, error=None)

    def test_modelobject_predict_start_returns_none_when_disabled(self):
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        result = modelobject_predict_start(model)
        self.assertIsNone(result)

    def test_modelobject_predict_end_noop_with_none_context(self):
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        modelobject_predict_end(model, context=None, result=None, error=None)

    def test_model_still_works_when_disabled(self):
        """Models must work normally with autolog disabled."""
        df = _make_df(20, 2)
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        model.fit(df)
        result = model.predict(forecast_length=3)
        self.assertEqual(result.forecast.shape, (3, 2))


class TestModelObjectFitPredictDisabled(unittest.TestCase):
    """ModelObject fit/predict must not fail with autolog disabled."""

    def setUp(self):
        autolog(disable=True)

    def tearDown(self):
        autolog(disable=True)

    def test_fit_predict_no_side_effects(self):
        df = _make_df(30, 2)
        model = AverageValueNaive(frequency="D", prediction_interval=0.9)
        result = model.fit(df)
        self.assertIs(result, model)
        pred = model.predict(forecast_length=3)
        self.assertIsNotNone(pred.forecast)
        self.assertEqual(pred.forecast.shape[0], 3)


@unittest.skipIf(mlflow is None, "mlflow not installed")
class TestModelObjectAutolog(unittest.TestCase):
    """Test ModelObject-level autologging with MLflow."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file://{self.tmpdir}")
        mlflow.set_experiment("test_modelobject")
        autolog(disable=False, tried_models_logging="single")

    def tearDown(self):
        autolog(disable=True)

    def test_fit_creates_run(self):
        df = _make_df(15, 1)
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        model.fit(df)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_modelobject")
        runs = client.search_runs([exp.experiment_id])

        mo_runs = [r for r in runs if r.data.tags.get("autots.flavor") == "ModelObject"]
        self.assertGreaterEqual(len(mo_runs), 1)

        fit_run = mo_runs[0]
        self.assertEqual(fit_run.data.tags.get("autots.stage"), "fit")
        self.assertEqual(fit_run.data.tags.get("autots.run_status"), "finished")
        self.assertIn("model_name", fit_run.data.params)
        self.assertIn("train_rows", fit_run.data.params)

    def test_predict_creates_run(self):
        df = _make_df(15, 1)
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        model.fit(df)
        model.predict(forecast_length=3)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_modelobject")
        runs = client.search_runs([exp.experiment_id])

        pred_runs = [
            r for r in runs
            if r.data.tags.get("autots.flavor") == "ModelObject"
            and r.data.tags.get("autots.stage") == "predict"
        ]
        self.assertGreaterEqual(len(pred_runs), 1)
        self.assertEqual(pred_runs[0].data.tags.get("autots.run_status"), "finished")

    def test_model_as_code_artifact(self):
        df = _make_df(15, 1)
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        model.fit(df)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_modelobject")
        runs = client.search_runs([exp.experiment_id])
        fit_runs = [
            r for r in runs
            if r.data.tags.get("autots.stage") == "fit"
        ]
        self.assertTrue(len(fit_runs) > 0)
        artifacts = client.list_artifacts(fit_runs[0].info.run_id, "modelobject")
        artifact_names = [a.path for a in artifacts]
        self.assertTrue(
            any("model_as_code" in n for n in artifact_names),
            f"Expected model_as_code artifact, got {artifact_names}",
        )

    def test_log_model_runs_false_suppresses(self):
        autolog(disable=False, log_model_runs=False)
        df = _make_df(15, 1)
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        model.fit(df)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_modelobject")
        runs = client.search_runs([exp.experiment_id])
        mo_fit_runs = [
            r for r in runs
            if r.data.tags.get("autots.flavor") == "ModelObject"
            and r.data.tags.get("autots.stage") == "fit"
        ]
        self.assertEqual(len(mo_fit_runs), 0)


@unittest.skipIf(mlflow is None, "mlflow not installed")
class TestAutoTSAutologSingleMode(unittest.TestCase):
    """Test AutoTS-level autologging in 'single' mode (default)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file://{self.tmpdir}")
        mlflow.set_experiment("test_autots_single")
        autolog(disable=False, tried_models_logging="single")

    def tearDown(self):
        autolog(disable=True)

    def _make_model(self):
        return AutoTS(
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

    def test_fit_creates_parent_run(self):
        df = _make_df(39, 2)
        model = self._make_model()
        model.fit(df)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_autots_single")
        runs = client.search_runs([exp.experiment_id])

        autots_runs = [r for r in runs if r.data.tags.get("autots.flavor") == "AutoTS"]
        self.assertGreaterEqual(len(autots_runs), 1)

        parent = [
            r for r in autots_runs if r.data.tags.get("autots.stage") != "predict"
        ]
        self.assertGreaterEqual(len(parent), 1)
        run = parent[0]
        self.assertEqual(run.data.tags.get("autots.run_status"), "finished")
        self.assertIn("forecast_length", run.data.params)
        self.assertIn("best_model_name", run.data.params)

    def test_no_modelobject_runs_during_fit(self):
        """Inner ModelObject runs should be suppressed during AutoTS.fit."""
        df = _make_df(39, 2)
        model = self._make_model()
        model.fit(df)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_autots_single")
        runs = client.search_runs([exp.experiment_id])

        mo_runs = [r for r in runs if r.data.tags.get("autots.flavor") == "ModelObject"]
        self.assertEqual(len(mo_runs), 0, "ModelObject runs should be suppressed")

    def test_no_individual_candidate_runs_in_single_mode(self):
        df = _make_df(39, 2)
        model = self._make_model()
        model.fit(df)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_autots_single")
        runs = client.search_runs([exp.experiment_id])

        candidate_runs = [
            r for r in runs
            if r.data.tags.get("autots.flavor") == "AutoTSModelCandidate"
        ]
        self.assertEqual(len(candidate_runs), 0)

    def test_best_model_metrics_logged(self):
        df = _make_df(39, 2)
        model = self._make_model()
        model.fit(df)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_autots_single")
        runs = client.search_runs([exp.experiment_id])

        parent_runs = [
            r for r in runs
            if r.data.tags.get("autots.flavor") == "AutoTS"
            and r.data.tags.get("autots.stage") != "predict"
        ]
        self.assertTrue(len(parent_runs) > 0)
        metrics = parent_runs[0].data.metrics
        # At least some metric should be logged
        metric_keys = set(metrics.keys())
        self.assertTrue(
            metric_keys.intersection({"best.smape", "best.mae", "best.Score", "autots.models_tested"}),
            f"Expected some best.* metrics, got {metric_keys}",
        )

    def test_predict_creates_run(self):
        df = _make_df(39, 2)
        model = self._make_model()
        model.fit(df)
        model.predict()

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_autots_single")
        runs = client.search_runs([exp.experiment_id])

        pred_runs = [
            r for r in runs
            if r.data.tags.get("autots.flavor") == "AutoTS"
            and r.data.tags.get("autots.stage") == "predict"
        ]
        self.assertGreaterEqual(len(pred_runs), 1)

    def test_model_as_code_artifact_logged(self):
        df = _make_df(39, 2)
        model = self._make_model()
        model.fit(df)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_autots_single")
        runs = client.search_runs([exp.experiment_id])

        parent_runs = [
            r for r in runs
            if r.data.tags.get("autots.flavor") == "AutoTS"
            and r.data.tags.get("autots.stage") != "predict"
        ]
        self.assertTrue(len(parent_runs) > 0)
        artifacts = client.list_artifacts(parent_runs[0].info.run_id, "autots")
        artifact_names = [a.path for a in artifacts]
        self.assertTrue(
            any("model_as_code" in n for n in artifact_names),
            f"Expected model_as_code artifact, got {artifact_names}",
        )

    def test_results_csv_artifact_logged(self):
        df = _make_df(39, 2)
        model = self._make_model()
        model.fit(df)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_autots_single")
        runs = client.search_runs([exp.experiment_id])

        parent_runs = [
            r for r in runs
            if r.data.tags.get("autots.flavor") == "AutoTS"
            and r.data.tags.get("autots.stage") != "predict"
        ]
        self.assertTrue(len(parent_runs) > 0)
        artifacts = client.list_artifacts(parent_runs[0].info.run_id, "autots")
        artifact_names = [a.path for a in artifacts]
        self.assertTrue(
            any("model_results" in n for n in artifact_names),
            f"Expected model_results artifact, got {artifact_names}",
        )


@unittest.skipIf(mlflow is None, "mlflow not installed")
class TestAutoTSAutologIndividualMode(unittest.TestCase):
    """Test AutoTS autologging in 'individual' mode."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file://{self.tmpdir}")
        mlflow.set_experiment("test_autots_individual")
        autolog(disable=False, tried_models_logging="individual")

    def tearDown(self):
        autolog(disable=True)

    def test_individual_creates_candidate_runs(self):
        df = _make_df(39, 2)
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

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_autots_individual")
        runs = client.search_runs([exp.experiment_id])
        flavors = {r.data.tags.get("autots.flavor") for r in runs}
        self.assertIn("AutoTS", flavors)
        self.assertIn("AutoTSModelCandidate", flavors)
        self.assertNotIn("ModelObject", flavors)


@unittest.skipIf(mlflow is None, "mlflow not installed")
class TestAutoTSAutologNoneMode(unittest.TestCase):
    """Test AutoTS autologging in 'none' mode."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file://{self.tmpdir}")
        mlflow.set_experiment("test_autots_none")
        autolog(disable=False, tried_models_logging="none")

    def tearDown(self):
        autolog(disable=True)

    def test_none_mode_no_results_artifacts(self):
        df = _make_df(39, 2)
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

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_autots_none")
        runs = client.search_runs([exp.experiment_id])

        autots_runs = [
            r for r in runs
            if r.data.tags.get("autots.flavor") == "AutoTS"
            and r.data.tags.get("autots.stage") != "predict"
        ]
        candidate_runs = [
            r for r in runs
            if r.data.tags.get("autots.flavor") == "AutoTSModelCandidate"
        ]
        self.assertGreaterEqual(len(autots_runs), 1)
        self.assertEqual(len(candidate_runs), 0)


@unittest.skipIf(mlflow is None, "mlflow not installed")
class TestDisableStopsLogging(unittest.TestCase):
    """Verify that autolog(disable=True) actually stops all logging."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file://{self.tmpdir}")
        mlflow.set_experiment("test_disable")

    def tearDown(self):
        autolog(disable=True)

    def test_disable_stops_modelobject_logging(self):
        autolog(disable=False)
        autolog(disable=True)

        df = _make_df(15, 1)
        model = LastValueNaive(frequency="D", prediction_interval=0.9)
        model.fit(df)
        model.predict(forecast_length=3)

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        exp = client.get_experiment_by_name("test_disable")
        runs = client.search_runs([exp.experiment_id])
        self.assertEqual(len(runs), 0, "No runs should be created when disabled")


@unittest.skipIf(mlflow is None, "mlflow not installed")
class TestAutoTSModelAsCodePayload(unittest.TestCase):
    """Test the model-as-code payload content for AutoTS."""

    def test_payload_structure(self):
        df = _make_df(39, 2)
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

        payload = _build_autots_model_as_code_payload(model)
        self.assertEqual(payload["model_name"], "AutoTS")
        self.assertTrue(payload["model_as_code"])
        self.assertTrue(payload["refit_expected"])
        self.assertIn("best_model_name", payload)
        self.assertIn("last_obs_date", payload)
        # Verify JSON-serializable
        json.dumps(_to_jsonable(payload))


class TestMLflowRoundTrip(unittest.TestCase):
    """Tests for loading models from MLflow and verifying consistency."""

    def setUp(self):
        if mlflow is None:
            self.skipTest("mlflow not installed")
        try:
            mlflow.end_run()
        except Exception:
            pass
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tracking_uri = f"file://{os.path.abspath(self.tmp_dir.name)}"
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment("test_roundtrip")
        autolog(tried_models_logging="single", silent=True)

    def tearDown(self):
        autolog(disable=True)
        try:
            mlflow.end_run()
        except Exception:
            pass
        self.tmp_dir.cleanup()

    def test_modelobject_roundtrip_consistency(self):
        df = _make_df(30, 2)
        # 1. Fit original (autolog will create the run)
        original = LastValueNaive(frequency="D")
        original.fit(df)

        # 2. Get the run ID from MLflow
        client = MlflowClient()
        experiment = client.get_experiment_by_name("test_roundtrip")
        runs = client.search_runs(
            experiment.experiment_id, order_by=["attributes.start_time DESC"]
        )
        run_id = runs[0].info.run_id

        # 3. Get original forecast
        orig_forecast = original.predict(forecast_length=5).forecast

        # 4. Load from MLflow
        from autots.tools.mlflow import load_model

        loaded = load_model(run_id)

        # 5. Refit and predict loaded
        loaded.fit(df)
        loaded_forecast = loaded.predict(forecast_length=5).forecast

        # 6. Assert consistency
        pd.testing.assert_frame_equal(orig_forecast, loaded_forecast)
        self.assertIsInstance(loaded, LastValueNaive)

    def test_autots_roundtrip_consistency(self):
        df = _make_df(30, 2)
        # 1. Fit original AutoTS
        model = AutoTS(
            forecast_length=5,
            frequency="D",
            model_list=["LastValueNaive", "AverageValueNaive"],
            max_generations=0,
            num_validations=0,
            random_seed=42,
        )
        model.fit(df)

        # 2. Get the run ID from MLflow
        client = MlflowClient()
        experiment = client.get_experiment_by_name("test_roundtrip")
        runs = client.search_runs(
            experiment.experiment_id, order_by=["attributes.start_time DESC"]
        )
        # Note: AutoTS might have multiple runs if individual logging is on,
        # but the top-most (summary) run should be most recent if nested correctly.
        run_id = runs[0].info.run_id

        # 3. Get original forecast
        orig_forecast = model.predict().forecast

        # 4. Load from MLflow
        from autots.tools.mlflow import load_model

        loaded = load_model(run_id)

        # 5. Refit and predict with loaded
        loaded.fit(df)
        loaded_forecast = loaded.predict().forecast

        # 6. Assert consistency
        pd.testing.assert_frame_equal(orig_forecast, loaded_forecast)
        self.assertEqual(loaded.best_model_name, model.best_model_name)


if __name__ == "__main__":
    unittest.main()
