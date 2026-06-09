"""
Tests for AutoTS MCP Server

Tests cover:
- Data loading (daily, weekly, hourly, monthly, yearly, synthetic, live)
- Forecasting (mosaic profile, search, custom, explainable)
- Feature detection and cleaning
- Event risk forecasting
- Server utilities (cache, data conversion, serialization)
- Tool handler invocations (async)
- Tool schema / list validation
- Package structure and entry-point verification

Resource-optimised: expensive operations (forecast_fast, detect_features,
EventRiskForecast) are shared across tests via lazy class-level fixtures so
they execute at most once per class.  Data is sliced to small sizes before
being fed into heavy operations.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Conditional import of server utilities
# ---------------------------------------------------------------------------
try:
    from autots.mcp.server import (
        CACHE_REGISTRY,
        MCP_AVAILABLE,
        _resolve_cache,
        build_csv_metadata,
        cache_object,
        call_tool,
        clear_cache,
        dataframe_to_output,
        get_cached_object,
        list_all_cached_objects,
        list_prompts,
        list_resources,
        list_tools,
        load_to_dataframe,
        read_resource,
        save_temp_csv,
        serialize_timestamps,
    )

    SERVER_UTILS_AVAILABLE = True
except ImportError:
    SERVER_UTILS_AVAILABLE = False
    MCP_AVAILABLE = False

from autots import load_daily, load_hourly, load_monthly, load_weekly, load_yearly

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = {
    "list_cache",
    "clear_cache",
    "load_sample_data",
    "load_live_data",
    "generate_synthetic_data",
    "load_data_from_file",
    "smart_load",
    "get_data",
    "convert_long_to_wide",
    "clean_data",
    "forecast_fast",
    "forecast_explainable",
    "forecast_custom",
    "get_forecast",
    "plot_forecast",
    "apply_constraints",
    "apply_adjustments",
    "get_model_params",
    "get_forecast_components",
    "get_validation_results",
    "plot_validation",
    "forecast_event_risk",
    "get_event_risk_results",
    "plot_event_risk",
    "detect_features",
    "get_detected_features",
    "plot_features",
    "forecast_from_features",
}


def _extract(result):
    """Return plain dict from a call_tool result regardless of return style.

    Tools return either:
      * a plain dict
      * a list[TextContent]  ->  JSON-parsed
      * a list[ImageContent] ->  returned as-is (caller checks .type)
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, list) and result:
        item = result[0]
        if hasattr(item, 'text'):
            return json.loads(item.text)
        return item
    return result


# ===========================================================================
# Sample data loading (no server required)
# ===========================================================================


class TestMCPSampleData(unittest.TestCase):
    """Test built-in sample data loaders."""

    def test_load_daily_data(self):
        df = load_daily(long=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertGreater(len(df), 100)
        self.assertGreater(len(df.columns), 1)

    def test_load_weekly_data(self):
        df = load_weekly(long=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertGreater(len(df), 50)

    def test_load_hourly_data(self):
        df = load_hourly(long=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertGreater(len(df), 100)

    def test_load_monthly_data(self):
        df = load_monthly(long=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertGreater(len(df), 10)

    def test_load_yearly_data(self):
        df = load_yearly(long=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertGreater(len(df), 5)

    def test_load_data_long_format(self):
        df = load_daily(long=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('datetime', df.columns)
        self.assertIn('series_id', df.columns)
        self.assertIn('value', df.columns)


# ===========================================================================
# Server integration / schema
# ===========================================================================


@unittest.skipIf(not MCP_AVAILABLE, "MCP not installed")
class TestMCPServerIntegration(unittest.TestCase):
    """Smoke tests for the MCP server."""

    def test_server_imports(self):
        from autots.mcp.server import app, serve

        self.assertIsNotNone(app)
        self.assertIsNotNone(serve)

    def test_serve_is_callable(self):
        from autots.mcp.server import serve

        self.assertTrue(callable(serve))

    def test_tool_list_completeness(self):
        """Every expected tool must appear in the server's tool list."""
        tools = asyncio.run(list_tools())
        registered = {t.name for t in tools}
        missing = EXPECTED_TOOLS - registered
        self.assertSetEqual(missing, set(), msg=f"Missing tools: {missing}")

    def test_tool_list_no_unknown(self):
        """No extra / renamed tools should appear unexpectedly."""
        tools = asyncio.run(list_tools())
        registered = {t.name for t in tools}
        extra = registered - EXPECTED_TOOLS
        self.assertSetEqual(extra, set(), msg=f"Unexpected extra tools: {extra}")

    def test_all_tools_have_descriptions(self):
        tools = asyncio.run(list_tools())
        for t in tools:
            self.assertTrue(
                t.description and len(t.description) > 10,
                msg=f"Tool '{t.name}' has a missing/short description",
            )

    def test_all_tools_have_input_schema(self):
        tools = asyncio.run(list_tools())
        for t in tools:
            self.assertIsNotNone(
                t.inputSchema, msg=f"Tool '{t.name}' has no inputSchema"
            )

    def test_readonly_tools_flagged(self):
        """Read-only tools should carry the readOnlyHint annotation."""
        read_only_tools = {"list_cache", "get_forecast"}
        tools = asyncio.run(list_tools())
        tool_map = {t.name: t for t in tools}
        for name in read_only_tools:
            t = tool_map.get(name)
            if t and t.annotations:
                self.assertTrue(
                    t.annotations.readOnlyHint,
                    msg=f"Tool '{name}' should be readOnlyHint=True",
                )


# ===========================================================================
# Forecasting business logic (no MCP required)
# ===========================================================================


class TestMCPForecasting(unittest.TestCase):
    """Test AutoTS forecasting underlying the MCP tools."""

    @classmethod
    def setUpClass(cls):
        cls.df_daily = load_daily(long=False).iloc[:200, :3]
        cls.df_weekly = load_weekly(long=False).iloc[:100, :2]
        cls.df_hourly = load_hourly(long=False).iloc[:500, :2]

    def test_forecast_length_validation(self):
        short_df = self.df_daily.iloc[:20]
        self.assertLess(len(short_df), 30)

    def test_daily_data_forecast_structure(self):
        from autots import AutoTS

        model = AutoTS(
            forecast_length=14,
            frequency='infer',
            max_generations=1,
            num_validations=1,
            model_list='superfast',
        )
        model = model.fit(self.df_daily)
        prediction = model.predict()

        self.assertEqual(len(prediction.forecast), 14)
        self.assertEqual(len(prediction.forecast.columns), len(self.df_daily.columns))
        self.assertIsInstance(prediction.forecast.index, pd.DatetimeIndex)

    def test_weekly_data_forecast(self):
        from autots import AutoTS

        # gens=0, vals=0: tests forecast output shape without expensive model search
        model = AutoTS(
            forecast_length=8,
            frequency='infer',
            max_generations=0,
            num_validations=0,
            model_list='superfast',
        )
        model = model.fit(self.df_weekly)
        prediction = model.predict()
        self.assertEqual(len(prediction.forecast), 8)

    def test_hourly_data_forecast(self):
        from autots import AutoTS

        # gens=0, vals=0: tests forecast output shape without expensive model search
        model = AutoTS(
            forecast_length=24,
            frequency='infer',
            max_generations=0,
            num_validations=0,
            model_list='superfast',
        )
        model = model.fit(self.df_hourly)
        prediction = model.predict()
        self.assertEqual(len(prediction.forecast), 24)

    def test_horizontal_profile_import_best_model(self):
        """Ensures forecast_fast mosaic path works end-to-end."""
        from autots import AutoTS
        from autots.evaluator.auto_model import create_model_id

        model_params = {'method': 'median', 'window': None}
        model_id = create_model_id('AverageValueNaive', model_params, {})
        profile_template = {
            "model_name": "Horizontal",
            "model_metric": "horizontal-profile",
            "model_count": 1,
            "models": {
                model_id: {
                    "Model": "AverageValueNaive",
                    "ModelParameters": json.dumps(model_params),
                    "TransformationParameters": "{}",
                }
            },
            "series": {
                "overall": model_id,
                "smooth": model_id,
                "binary": model_id,
            },
            "transformation": {
                "fillna": "ffill",
                "transformations": {},
                "transformation_params": {},
            },
        }
        ensemble_params = {
            'Model': 'Ensemble',
            'ModelParameters': json.dumps(profile_template),
            'TransformationParameters': json.dumps(
                profile_template.get('transformation', {})
            ),
            'Ensemble': 2,
        }
        ensemble_template = pd.DataFrame(ensemble_params, index=[0])

        model = AutoTS(
            forecast_length=14,
            frequency='infer',
            ensemble='horizontal-profile',
            model_list='scalable',
            max_generations=0,
            num_validations=0,
            validation_method='backwards',
        )
        model.fit_data(self.df_daily.iloc[:120, :2])
        model.import_best_model(
            ensemble_template, enforce_model_list=False, include_ensemble=True
        )
        prediction = model.predict()

        self.assertEqual(len(prediction.forecast), 14)
        self.assertEqual(prediction.forecast.shape[1], 2)

    def test_prediction_has_intervals(self):
        """Upper/lower forecasts should be available."""
        from autots import AutoTS

        model = AutoTS(
            forecast_length=7,
            frequency='infer',
            max_generations=0,
            num_validations=0,
            model_list='superfast',
        )
        model.fit(self.df_daily.iloc[:100, :2])
        prediction = model.predict()

        self.assertIsNotNone(prediction.upper_forecast)
        self.assertIsNotNone(prediction.lower_forecast)
        self.assertEqual(prediction.upper_forecast.shape, prediction.forecast.shape)


# ===========================================================================
# Feature detection (no MCP required)
# ===========================================================================


class TestMCPFeatureDetection(unittest.TestCase):
    def setUp(self):
        self.df = load_daily(long=False).iloc[:200, :3]

    def test_data_cleaning(self):
        from autots.tools.transform import GeneralTransformer

        df_dirty = self.df.copy()
        df_dirty.iloc[10, 0] = np.nan
        df_dirty.iloc[50, 1] = np.nan

        transformer = GeneralTransformer(
            fillna='ffill',
            transformations={"0": "ClipOutliers"},
            transformation_params={"0": {}},
        )
        df_cleaned = transformer.fit_transform(df_dirty)
        self.assertLess(df_cleaned.isna().sum().sum(), df_dirty.isna().sum().sum())

    def test_anomaly_detection(self):
        from autots.evaluator.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(
            output='multivariate', method='IQR', transform_dict=None
        )
        detector.detect(self.df)
        self.assertIsNotNone(detector.anomalies)

    def test_feature_detector_fit(self):
        from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

        detector = TimeSeriesFeatureDetector()
        detector.fit(self.df.iloc[:, :2])
        self.assertTrue(hasattr(detector, 'df_original'))
        self.assertIsNotNone(detector.df_original)

    def test_feature_detector_query(self):
        from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

        detector = TimeSeriesFeatureDetector()
        detector.fit(self.df.iloc[:, :2])
        results = detector.query_features(return_json=False)
        self.assertIn('series', results)

    def test_feature_detector_forecast(self):
        from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

        detector = TimeSeriesFeatureDetector()
        detector.fit(self.df.iloc[:100, :2])
        prediction = detector.forecast(forecast_length=14)
        self.assertIsNotNone(prediction)
        self.assertEqual(len(prediction.forecast), 14)


# ===========================================================================
# Smart ingestion (no MCP required)
# ===========================================================================


class TestMCPSmartLoad(unittest.TestCase):
    """Tests for autots.mcp.ingest.smart_load auto-clean / auto-detect."""

    def setUp(self):
        from autots.mcp.ingest import smart_load

        self.smart_load = smart_load

    def test_messy_wide_cleanup(self):
        """Empty padding rows/cols and off-center tables are stripped."""
        messy = (
            ",,,\n"
            ",datetime,sales,inventory\n"
            ",2024-01-01,100,50\n"
            ",2024-01-02,102,48\n"
            ",2024-01-03,99,51\n"
            ",,,\n"
        )
        df, rep = self.smart_load(text=messy)
        self.assertEqual(rep["detected_format"], "wide")
        self.assertEqual(list(df.columns), ["sales", "inventory"])
        self.assertEqual(len(df), 3)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertGreaterEqual(rep["dropped"]["columns"], 1)

    def test_single_series_two_columns(self):
        df, rep = self.smart_load(
            text="date,price\n2024-01-01,10\n2024-01-02,11\n2024-01-03,12\n"
        )
        self.assertEqual(rep["detected_format"], "wide")
        self.assertTrue(rep["single_series"])
        self.assertEqual(df.shape, (3, 1))

    def test_long_format_detection(self):
        long_text = (
            "datetime,series_id,value\n"
            "2024-01-01,a,1\n2024-01-01,b,5\n"
            "2024-01-02,a,2\n2024-01-02,b,6\n"
            "2024-01-03,a,3\n2024-01-03,b,7\n"
        )
        df, rep = self.smart_load(text=long_text)
        self.assertEqual(rep["detected_format"], "long")
        self.assertEqual(sorted(df.columns), ["a", "b"])
        self.assertEqual(len(df), 3)

    def test_tsv_paste_wide(self):
        tsv = "datetime\tA\tB\tC\n2024-01-01\t1\t2\t3\n2024-01-02\t4\t5\t6\n2024-01-03\t7\t8\t9\n"
        df, rep = self.smart_load(text=tsv)
        self.assertEqual(rep["detected_format"], "wide")
        self.assertEqual(list(df.columns), ["A", "B", "C"])
        self.assertEqual(rep["inferred_frequency"], "D")

    def test_no_header_numeric_grid(self):
        df, rep = self.smart_load(
            text="2024-01-01,1,2\n2024-01-02,3,4\n2024-01-03,5,6\n"
        )
        self.assertFalse(rep["had_header"])
        self.assertEqual(len(df), 3)

    def test_report_is_json_serializable(self):
        df, rep = self.smart_load(
            text="date,price\n2024-01-01,10\n2024-01-02,11\n2024-01-03,12\n"
        )
        json.dumps(rep)  # must not raise

    def test_no_date_column_raises(self):
        with self.assertRaises(ValueError):
            self.smart_load(text="a,b\nfoo,bar\nbaz,qux\n")

    def test_no_input_raises(self):
        with self.assertRaises(ValueError):
            self.smart_load()


# ===========================================================================
# Pyodide API (no MCP required)
# ===========================================================================


class TestMCPPyodideAPI(unittest.IsolatedAsyncioTestCase):
    """Tests for the Pyodide-facing PWA boundary (autots.mcp.pyodide_api)."""

    def setUp(self):
        from autots.mcp import pyodide_api as P

        self.P = P

    def test_safe_model_sets_exclude_native(self):
        native = {
            "GluonTS", "NeuralForecast", "MambaSSM", "pMLP", "TiDE",
            "PytorchForecasting", "Prophet", "ARCH",
            "MultivariateRegression", "WindowRegression", "DatepartRegression",
            "PreprocessingRegression",
        }
        self.assertEqual(set(self.P.PYODIDE_FAST_MODELS) & native, set())
        self.assertEqual(set(self.P.PYODIDE_SEARCH_MODELS) & native, set())
        self.assertIn("BasicLinearModel", self.P.PYODIDE_FAST_MODELS)
        self.assertIn("Cassandra", self.P.PYODIDE_SEARCH_MODELS)

    def test_list_commands_has_presets_and_tools(self):
        commands = self.P.list_commands()
        for preset in ("make_forecast", "search_forecast", "search_all_night"):
            self.assertIn(preset, commands)
        self.assertIn("smart_load", commands)
        self.assertIn("get_forecast", commands)

    async def test_run_command_json_passthrough(self):
        from autots.mcp.cache import clear_cache

        out = await self.P.run_command_json(
            "load_sample_data", json.dumps({"dataset": "sine"})
        )
        self.assertIsInstance(out, str)  # JSON string boundary
        data = json.loads(out)
        self.assertIn("data_id", data)
        clear_cache(data["data_id"], "data")

    async def test_make_forecast_requires_data_id(self):
        out = await self.P.run_command_json("make_forecast", json.dumps({}))
        self.assertIn("error", json.loads(out))

    async def test_make_forecast_preserves_prediction_interval(self):
        calls = []

        async def fake_run_tool(name, arguments, progress_cb):
            calls.append((name, arguments))
            if name == "detect_features":
                return {"detector_id": "feature-detector"}
            return {"prediction_id": "prediction"}

        orig = self.P.run_tool
        self.P.run_tool = fake_run_tool
        try:
            result = await self.P.dispatch(
                "make_forecast",
                {
                    "data_id": "d",
                    "forecast_length": 12,
                    "prediction_interval": 0.8,
                },
            )
        finally:
            self.P.run_tool = orig

        self.assertEqual(result["prediction_id"], "prediction")
        self.assertEqual(calls[1][0], "forecast_from_features")
        self.assertEqual(calls[1][1]["forecast_length"], 12)
        self.assertEqual(calls[1][1]["prediction_interval"], 0.8)

    async def test_restore_data_snapshot_rebuilds_wide_datetime_data(self):
        from autots.mcp.cache import clear_cache, get_cached_object

        out = await self.P.run_command_json(
            "restore_data_snapshot",
            json.dumps(
                {
                    "data": {
                        "datetime": ["2026-01-01", "2026-01-02"],
                        "sales": [10.0, None],
                        "returns": [1.0, 2.0],
                    },
                    "metadata": {"source": "test_restore"},
                }
            ),
        )
        result = json.loads(out)
        restored = get_cached_object(result["data_id"], "data")
        self.assertEqual(restored["metadata"]["source"], "test_restore")
        self.assertEqual(list(restored["object"].columns), ["sales", "returns"])
        self.assertEqual(str(restored["object"].index.dtype), "datetime64[ns]")
        self.assertTrue(pd.isna(restored["object"].iloc[1, 0]))
        clear_cache(result["data_id"], "data")

    async def test_search_pins_safe_models(self):
        """_search_forecast must force the safe model list even if extra params try to override."""
        captured = {}

        async def fake_run_tool(name, arguments, progress_cb):
            captured["name"] = name
            captured["arguments"] = arguments
            return {"prediction_id": "x"}

        orig = self.P.run_tool
        self.P.run_tool = fake_run_tool
        try:
            await self.P.dispatch(
                "search_forecast",
                {
                    "data_id": "d",
                    "autots_params": {"model_list": ["GluonTS"], "n_jobs": -1},
                },
            )
        finally:
            self.P.run_tool = orig

        params = captured["arguments"]["autots_params"]
        self.assertEqual(params["model_list"], self.P.PYODIDE_SEARCH_MODELS)
        self.assertEqual(params["n_jobs"], 1)

    async def test_search_preserves_prediction_interval(self):
        captured = {}

        async def fake_run_tool(name, arguments, progress_cb):
            captured["arguments"] = arguments
            return {"prediction_id": "x"}

        orig = self.P.run_tool
        self.P.run_tool = fake_run_tool
        try:
            await self.P.dispatch(
                "search_forecast",
                {
                    "data_id": "d",
                    "autots_params": {"prediction_interval": 0.8},
                },
            )
        finally:
            self.P.run_tool = orig

        params = captured["arguments"]["autots_params"]
        self.assertEqual(params["prediction_interval"], 0.8)
        self.assertEqual(params["max_generations"], 3)
        self.assertEqual(params["n_jobs"], 1)

    async def test_all_night_search_preserves_timeout_and_large_generation_limit(self):
        captured = {}

        async def fake_run_tool(name, arguments, progress_cb):
            captured["name"] = name
            captured["arguments"] = arguments
            return {"prediction_id": "x"}

        orig = self.P.run_tool
        self.P.run_tool = fake_run_tool
        try:
            await self.P.dispatch(
                "search_all_night",
                {
                    "data_id": "d",
                    "autots_params": {
                        "prediction_interval": 0.95,
                        "generation_timeout": 240,
                        "max_generations": 1_000_000,
                    },
                },
            )
        finally:
            self.P.run_tool = orig

        params = captured["arguments"]["autots_params"]
        self.assertEqual(captured["name"], "forecast_custom")
        self.assertEqual(params["prediction_interval"], 0.95)
        self.assertEqual(params["generation_timeout"], 240)
        self.assertEqual(params["max_generations"], 1_000_000)
        self.assertEqual(params["model_list"], self.P.PYODIDE_SEARCH_MODELS)
        self.assertEqual(params["n_jobs"], 1)


# ===========================================================================
# Live data loading (no network — loader is mocked)
# ===========================================================================


class TestMCPLiveDataLoader(unittest.TestCase):
    """load_live_daily's optional progress/status instrumentation (network-free)."""

    def test_all_disabled_raises_with_empty_status(self):
        from autots.datasets._live import load_live_daily

        status, prog = [], []
        with self.assertRaises(ValueError):
            load_live_daily(
                fred_key=None, fred_series=None, tickers=None, trends_list=None,
                weather_stations=None, london_air_stations=None,
                earthquake_min_magnitude=None, nasa_api_key=None,
                gov_domain_list=None, wikipedia_pages=None,
                weather_event_types=None, caiso_query=None,
                eia_key=None, eia_respondents=None,
                progress_cb=lambda m: prog.append(m), status_log=status,
            )
        # No sources were enabled, so nothing should have been attempted.
        self.assertEqual(status, [])
        self.assertEqual(prog, [])


class TestMCPLiveDataHandler(unittest.IsolatedAsyncioTestCase):
    """handle_load_live_data forwards params and reports per-source status."""

    async def _noop(self, msg):
        return None

    async def test_status_passthrough_and_partial_failure(self):
        from autots.mcp.handlers import data as data_handlers
        from autots.mcp.cache import clear_cache

        captured = {}

        def fake_loader(long=False, progress_cb=None, status_log=None, **kwargs):
            captured["kwargs"] = kwargs
            if status_log is not None:
                status_log.append({"source": "FRED", "status": "ok", "series": 3})
                status_log.append(
                    {"source": "Stock tickers", "status": "failed", "error": "boom"}
                )
            return pd.DataFrame(
                {"a": [1, 2, 3]},
                index=pd.date_range("2024-01-01", periods=3, name="datetime"),
            )

        orig = data_handlers.load_live_daily
        data_handlers.load_live_daily = fake_loader
        try:
            res = await data_handlers.handle_load_live_data(
                {"fred_key": "k", "fred_series": ["DGS10"], "tickers": ["MSFT"]},
                self._noop,
            )
        finally:
            data_handlers.load_live_daily = orig

        self.assertIsNotNone(res["data_id"])
        self.assertEqual(res["cols"], 1)
        statuses = {s["source"]: s["status"] for s in res["sources"]}
        self.assertEqual(statuses["FRED"], "ok")
        self.assertEqual(statuses["Stock tickers"], "failed")
        # Recognized params are forwarded straight through to the loader.
        self.assertEqual(captured["kwargs"]["fred_key"], "k")
        self.assertEqual(captured["kwargs"]["tickers"], ["MSFT"])
        clear_cache(res["data_id"], "data")

    async def test_all_sources_fail_returns_error_not_raise(self):
        from autots.mcp.handlers import data as data_handlers

        def fake_loader(long=False, progress_cb=None, status_log=None, **kwargs):
            if status_log is not None:
                status_log.append({"source": "FRED", "status": "failed", "error": "nope"})
            raise ValueError("No data successfully downloaded!")

        orig = data_handlers.load_live_daily
        data_handlers.load_live_daily = fake_loader
        try:
            res = await data_handlers.handle_load_live_data(
                {"fred_key": "k"}, self._noop
            )
        finally:
            data_handlers.load_live_daily = orig

        self.assertIsNone(res["data_id"])
        self.assertIn("error", res)
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["status"], "failed")


# ===========================================================================
# Event risk (no MCP required)
# ===========================================================================


class TestMCPEventRiskForecasting(unittest.TestCase):
    def setUp(self):
        self.df = load_daily(long=False).iloc[:200, :2]

    def test_event_risk_basic(self):
        from autots import EventRiskForecast

        erf = EventRiskForecast(
            df_train=self.df,
            forecast_length=14,
            frequency='infer',
            upper_limit=0.75,
            lower_limit=0.25,
        )
        erf.fit()
        upper_risk_df, lower_risk_df = erf.predict()

        self.assertIsNotNone(upper_risk_df)
        self.assertIsNotNone(lower_risk_df)
        self.assertEqual(len(upper_risk_df), 14)
        self.assertEqual(len(lower_risk_df), 14)
        self.assertIsNotNone(erf.forecast_df)
        self.assertEqual(len(erf.forecast_df), 14)

    def test_event_risk_upper_direction(self):
        from autots import EventRiskForecast

        erf = EventRiskForecast(
            df_train=self.df,
            forecast_length=7,
            frequency='infer',
            upper_limit=0.9,
        )
        erf.fit()
        upper_risk_df, _ = erf.predict()
        # Probabilities must be in [0, 1]
        self.assertTrue((upper_risk_df.values >= 0).all())
        self.assertTrue((upper_risk_df.values <= 1).all())


# ===========================================================================
# Synthetic data
# ===========================================================================


class TestMCPSyntheticData(unittest.TestCase):
    def test_synthetic_data_generation(self):
        from autots.datasets.synthetic import SyntheticDailyGenerator

        generator = SyntheticDailyGenerator(n_series=3, n_days=365, random_seed=42)
        df = generator.data
        template = generator.template

        self.assertEqual(len(df), 365)
        self.assertEqual(len(df.columns), 3)
        self.assertIsInstance(template, dict)
        self.assertIn('meta', template)
        self.assertIn('series', template)


# ===========================================================================
# Server utility functions
# ===========================================================================


@unittest.skipIf(not SERVER_UTILS_AVAILABLE, "MCP server utilities not available")
class TestMCPServerUtilities(unittest.TestCase):
    """Low-level utility function tests."""

    def setUp(self):
        clear_cache()
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        self.df = pd.DataFrame(
            {'series1': np.random.randn(50), 'series2': np.random.randn(50)},
            index=dates,
        )

    def tearDown(self):
        clear_cache()

    # ---- Cache management ------------------------------------------------

    def test_cache_object_basic(self):
        test_obj = {"data": [1, 2, 3]}
        obj_id = cache_object(test_obj, 'data', {'test': True})

        self.assertIsInstance(obj_id, str)
        self.assertGreater(len(obj_id), 0)
        retrieved = get_cached_object(obj_id, 'data')
        self.assertEqual(retrieved['object'], test_obj)
        self.assertTrue(retrieved['metadata']['test'])

    def test_cache_stores_created_at(self):
        obj_id = cache_object({'x': 1}, 'data')
        entry = get_cached_object(obj_id, 'data')
        self.assertIn('created_at', entry)

    def test_cache_invalid_type(self):
        with self.assertRaises(ValueError):
            cache_object({}, 'invalid_type')

    def test_cache_all_valid_types(self):
        for cache_type in ('prediction', 'autots', 'event_risk', 'feature_detector', 'data'):
            obj_id = cache_object({'x': 1}, cache_type)
            self.assertIsNotNone(obj_id)

    def test_resolve_cache_unknown(self):
        with self.assertRaises(ValueError):
            _resolve_cache('nonexistent')

    def test_get_nonexistent_cache(self):
        with self.assertRaises(ValueError):
            get_cached_object('nonexistent-id', 'data')

    def test_list_all_cached_objects(self):
        cache_object({'test': 1}, 'data')
        cache_object({'test': 2}, 'prediction')

        cache_list = list_all_cached_objects()
        self.assertIsInstance(cache_list, dict)
        self.assertIn('data', cache_list)
        self.assertIn('predictions', cache_list)

    def test_list_cached_objects_empty(self):
        cache_list = list_all_cached_objects()
        self.assertEqual(len(cache_list), 0)

    def test_clear_specific_cache(self):
        obj_id = cache_object({'test': 1}, 'data')
        clear_cache(obj_id, 'data')
        with self.assertRaises(ValueError):
            get_cached_object(obj_id, 'data')

    def test_clear_cache_by_type(self):
        cache_object({'test': 1}, 'data')
        cache_object({'test': 2}, 'data')
        clear_cache(cache_type='data')
        cache_list = list_all_cached_objects()
        self.assertNotIn('data', cache_list)

    def test_clear_all_caches(self):
        cache_object({'test': 1}, 'data')
        cache_object({'test': 2}, 'prediction')
        clear_cache()
        self.assertEqual(len(list_all_cached_objects()), 0)

    def test_cache_eviction(self):
        """Cache should evict oldest entries when at capacity."""
        cache = {}
        for i in range(5):
            cache[str(i)] = {'created_at': f'2024-01-0{i+1}', 'metadata': {}}

        original_max = 3
        # Manually test _enforce_cache_limit logic
        while len(cache) > original_max:
            oldest = next(iter(cache))
            cache.pop(oldest)
        self.assertLessEqual(len(cache), original_max)

    # ---- Data loading ----------------------------------------------------

    def test_load_to_dataframe_from_dict_wide(self):
        data = {
            'datetime': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'series1': [1, 2, 3],
            'series2': [4, 5, 6],
        }
        df = load_to_dataframe(data, data_format='wide')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(len(df), 3)
        self.assertIn('series1', df.columns)

    def test_load_to_dataframe_from_dict_long(self):
        data = {
            'datetime': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
            'series_id': ['s1', 's2', 's1', 's2'],
            'value': [1, 2, 3, 4],
        }
        df = load_to_dataframe(data, data_format='long')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(len(df), 2)

    def test_load_to_dataframe_missing_datetime(self):
        data = {'series1': [1, 2, 3]}
        with self.assertRaises(ValueError):
            load_to_dataframe(data)

    def test_load_to_dataframe_from_cache(self):
        data_id = cache_object(self.df, 'data')
        df_loaded = load_to_dataframe(data_id=data_id)
        self.assertTrue(self.df.equals(df_loaded))

    def test_load_to_dataframe_no_params(self):
        with self.assertRaises(ValueError):
            load_to_dataframe()

    def test_load_to_dataframe_from_csv(self):
        with tempfile.NamedTemporaryFile(
            suffix='.csv', mode='w', delete=False
        ) as f:
            self.df.to_csv(f.name)
            tmp_path = f.name
        try:
            df_loaded = load_to_dataframe(tmp_path)
            self.assertIsInstance(df_loaded, pd.DataFrame)
            self.assertIsInstance(df_loaded.index, pd.DatetimeIndex)
        finally:
            os.unlink(tmp_path)

    # ---- DataFrame output -----------------------------------------------

    def test_dataframe_to_output_json_wide(self):
        result = dataframe_to_output(self.df, 'json_wide')
        self.assertIsInstance(result, dict)
        self.assertIn('datetime', result)
        self.assertIn('series1', result)
        self.assertIn('series2', result)
        self.assertEqual(len(result['datetime']), 50)

    def test_dataframe_to_output_json_long(self):
        result = dataframe_to_output(self.df, 'json_long')
        self.assertIsInstance(result, dict)
        self.assertIn('datetime', result)
        self.assertIn('series_id', result)
        self.assertIn('value', result)
        self.assertEqual(len(result['datetime']), 100)  # 50 rows * 2 series

    def test_dataframe_to_output_csv_wide(self):
        filepath = dataframe_to_output(self.df, 'csv_wide')
        self.assertIsInstance(filepath, str)
        self.assertTrue(filepath.endswith('.csv'))
        self.assertTrue(os.path.exists(filepath))

    def test_dataframe_to_output_csv_long(self):
        filepath = dataframe_to_output(self.df, 'csv_long')
        self.assertIsInstance(filepath, str)
        self.assertTrue(os.path.exists(filepath))
        df_loaded = pd.read_csv(filepath)
        self.assertIn('series_id', df_loaded.columns)
        self.assertIn('value', df_loaded.columns)

    def test_dataframe_to_output_invalid_format(self):
        with self.assertRaises(ValueError):
            dataframe_to_output(self.df, 'invalid_format')

    def test_save_temp_csv_wide(self):
        filepath = save_temp_csv(self.df, is_long=False)
        self.assertTrue(os.path.exists(filepath))
        df_loaded = pd.read_csv(filepath, parse_dates=True, index_col=0)
        self.assertEqual(len(df_loaded), len(self.df))

    def test_save_temp_csv_long(self):
        filepath = save_temp_csv(self.df, is_long=True)
        self.assertTrue(os.path.exists(filepath))
        df_loaded = pd.read_csv(filepath)
        self.assertIn('series_id', df_loaded.columns)

    def test_build_csv_metadata_wide(self):
        filepath = save_temp_csv(self.df)
        meta = build_csv_metadata(filepath, self.df, is_long=False)
        self.assertEqual(meta['format'], 'wide')
        self.assertEqual(meta['shape']['rows'], len(self.df))
        self.assertIn('loading_instructions', meta)
        self.assertIn('pandas', meta['loading_instructions'])

    def test_build_csv_metadata_long(self):
        filepath = save_temp_csv(self.df, is_long=True)
        meta = build_csv_metadata(filepath, self.df, is_long=True)
        self.assertEqual(meta['format'], 'long')
        self.assertIn('series_id', meta['columns'])

    # ---- Timestamp serialization ----------------------------------------

    def test_serialize_single_timestamp(self):
        ts = pd.Timestamp('2024-01-01 12:30:45')
        result = serialize_timestamps(ts)
        self.assertIsInstance(result, str)
        self.assertEqual(result, '2024-01-01 12:30:45')

    def test_serialize_dict_with_timestamps(self):
        data = {'date': pd.Timestamp('2024-01-01'), 'value': 42}
        result = serialize_timestamps(data)
        self.assertIsInstance(result['date'], str)
        self.assertEqual(result['value'], 42)

    def test_serialize_nested_timestamps(self):
        data = {
            'dates': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')],
            'metadata': {'last_update': pd.Timestamp('2024-01-03')},
        }
        result = serialize_timestamps(data)
        self.assertTrue(all(isinstance(d, str) for d in result['dates']))
        self.assertIsInstance(result['metadata']['last_update'], str)

    def test_serialize_non_timestamp_passthrough(self):
        data = {'num': 42, 'lst': [1, 2, 3], 'tpl': (4, 5)}
        result = serialize_timestamps(data)
        self.assertEqual(result['num'], 42)
        self.assertEqual(result['lst'], [1, 2, 3])

    # ---- Edge cases ------------------------------------------------------

    def test_dataframe_with_nan_values(self):
        df_with_nan = self.df.copy()
        df_with_nan.iloc[5:10, 0] = np.nan
        result = dataframe_to_output(df_with_nan, 'json_wide')
        self.assertIsInstance(result, dict)

    def test_single_row_dataframe(self):
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        df_single = pd.DataFrame({'series1': [1]}, index=dates)
        result = dataframe_to_output(df_single, 'json_wide')
        self.assertEqual(len(result['datetime']), 1)

    def test_special_characters_in_columns(self):
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df_special = pd.DataFrame(
            {'series-1': [1, 2, 3, 4, 5], 'series/2': [6, 7, 8, 9, 10]},
            index=dates,
        )
        result = dataframe_to_output(df_special, 'json_wide')
        self.assertIn('series-1', result)
        self.assertIn('series/2', result)


# ===========================================================================
# Async tool handler tests (call_tool directly)
#
# Expensive operations (forecast_fast, detect_features) are shared via lazy
# class-level fixtures.  Data is sliced to small sizes (<=150 rows, 2-3 cols)
# and cached directly to avoid loading full datasets into heavy operations.
# ===========================================================================


@unittest.skipIf(not SERVER_UTILS_AVAILABLE, "MCP server utilities not available")
class TestMCPToolHandlers(unittest.IsolatedAsyncioTestCase):
    """Invoke call_tool handlers directly and verify return values."""

    # Shared fixture IDs (populated lazily, persist across tests in the class)
    _weekly_data_id = None
    _weekly_prediction_id = None
    _daily_data_id = None
    _detector_id = None

    @classmethod
    def setUpClass(cls):
        clear_cache()
        # Pre-load small sliced DataFrames so tool handlers operate on small data.
        # This avoids loading full datasets (2869x27 daily, 1028x9 weekly) which
        # caused memory exhaustion when repeated across many tests.
        cls._small_weekly = load_weekly(long=False).iloc[:80, :2]
        cls._small_daily = load_daily(long=False).iloc[:150, :3]

    @classmethod
    def tearDownClass(cls):
        clear_cache()
        cls._weekly_data_id = None
        cls._weekly_prediction_id = None
        cls._daily_data_id = None
        cls._detector_id = None

    # ---- Lazy shared helpers (run at most once, reused across tests) ------

    async def _ensure_weekly_data(self):
        """Cache small weekly DataFrame; return data_id."""
        cls = self.__class__
        if cls._weekly_data_id is None or cls._weekly_data_id not in CACHE_REGISTRY['data']:
            cls._weekly_data_id = cache_object(
                cls._small_weekly, 'data',
                {'source': 'test_weekly', 'rows': len(cls._small_weekly)},
            )
        return cls._weekly_data_id

    async def _ensure_weekly_forecast(self):
        """Run forecast_fast once on small weekly data; return prediction_id."""
        cls = self.__class__
        if cls._weekly_prediction_id is None or cls._weekly_prediction_id not in CACHE_REGISTRY['prediction']:
            data_id = await self._ensure_weekly_data()
            result = await call_tool(
                "forecast_fast", {"data_id": data_id, "forecast_length": 8}
            )
            cls._weekly_prediction_id = _extract(result)["prediction_id"]
        return cls._weekly_prediction_id

    async def _ensure_daily_data(self):
        """Cache small daily DataFrame; return data_id."""
        cls = self.__class__
        if cls._daily_data_id is None or cls._daily_data_id not in CACHE_REGISTRY['data']:
            cls._daily_data_id = cache_object(
                cls._small_daily, 'data',
                {'source': 'test_daily', 'rows': len(cls._small_daily)},
            )
        return cls._daily_data_id

    async def _ensure_detector(self):
        """Run detect_features once on small daily data; return detector_id."""
        cls = self.__class__
        if cls._detector_id is None or cls._detector_id not in CACHE_REGISTRY['feature_detector']:
            data_id = await self._ensure_daily_data()
            result = await call_tool("detect_features", {"data_id": data_id})
            cls._detector_id = _extract(result)["detector_id"]
        return cls._detector_id

    # ---- Cache tools (lightweight, manage their own state) ---------------

    async def test_list_cache_empty(self):
        # Temporarily snapshot caches, clear, test, then restore so shared
        # fixtures remain valid for other tests in this class
        saved = {k: dict(v) for k, v in CACHE_REGISTRY.items()}
        try:
            clear_cache()
            result = await call_tool("list_cache", {})
            data = _extract(result)
            self.assertIsInstance(data, dict)
            self.assertEqual(len(data), 0)
        finally:
            for k in CACHE_REGISTRY:
                CACHE_REGISTRY[k].clear()
                CACHE_REGISTRY[k].update(saved[k])

    async def test_clear_cache_tool(self):
        # Clear only a temp entry — not the shared fixtures
        temp_id = cache_object({'x': 1}, 'data')
        result = await call_tool(
            "clear_cache", {"object_id": temp_id, "cache_type": "data"}
        )
        data = _extract(result)
        self.assertTrue(data.get("success", True))

    # ---- Data loading tools (lightweight, use load_sample_data API) ------

    async def test_load_sample_data_daily(self):
        result = await call_tool("load_sample_data", {"dataset": "daily"})
        data = _extract(result)
        self.assertIn("data_id", data)
        self.assertGreater(data.get("rows", 0), 100)
        self.assertGreater(data.get("cols", 0), 1)
        # Clean up full-sized data immediately to save memory
        clear_cache(data["data_id"], "data")

    async def test_load_sample_data_datasets(self):
        for dataset in ("weekly", "monthly", "yearly", "linear", "sine"):
            result = await call_tool("load_sample_data", {"dataset": dataset})
            data = _extract(result)
            self.assertIn("data_id", data, msg=f"Missing data_id for dataset={dataset}")
            # Clean up full-sized data immediately
            clear_cache(data["data_id"], "data")

    async def test_generate_synthetic_data(self):
        result = await call_tool("generate_synthetic_data", {"n_series": 3})
        data = _extract(result)
        self.assertIn("data_id", data)
        self.assertEqual(data.get("n_series"), 3)
        clear_cache(data["data_id"], "data")

    async def test_load_data_from_file(self):
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        df = pd.DataFrame({'s1': np.random.randn(30)}, index=dates)
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            df.to_csv(f.name)
            tmp_path = f.name
        try:
            result = await call_tool("load_data_from_file", {"filepath": tmp_path})
            data = _extract(result)
            self.assertIn("data_id", data)
            clear_cache(data["data_id"], "data")
        finally:
            os.unlink(tmp_path)

    # ---- Data manipulation tools (use shared weekly data) ----------------

    async def test_get_data_json_wide(self):
        data_id = await self._ensure_weekly_data()
        result = await call_tool(
            "get_data", {"data_id": data_id, "output_format": "json_wide"}
        )
        data = _extract(result)
        self.assertIn("datetime", data)

    async def test_get_data_json_long(self):
        data_id = await self._ensure_weekly_data()
        result = await call_tool(
            "get_data", {"data_id": data_id, "output_format": "json_long"}
        )
        data = _extract(result)
        self.assertIn("datetime", data)
        self.assertIn("series_id", data)
        self.assertIn("value", data)

    async def test_get_data_csv_wide(self):
        data_id = await self._ensure_weekly_data()
        result = await call_tool(
            "get_data", {"data_id": data_id, "output_format": "csv_wide"}
        )
        data = _extract(result)
        self.assertIn("filepath", data)
        self.assertTrue(os.path.exists(data["filepath"]))

    async def test_clean_data(self):
        data_id = await self._ensure_daily_data()
        result = await call_tool(
            "clean_data", {"data_id": data_id, "fillna": "ffill"}
        )
        data = _extract(result)
        self.assertIn("data_id", data)
        self.assertNotEqual(data["data_id"], data_id)
        # Clean up derived data
        clear_cache(data["data_id"], "data")

    async def test_convert_long_to_wide(self):
        long_data = {
            'datetime': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
            'series_id': ['s1', 's2', 's1', 's2'],
            'value': [1.0, 2.0, 3.0, 4.0],
        }
        result = await call_tool("convert_long_to_wide", {"data": long_data})
        data = _extract(result)
        self.assertIn("data_id", data)
        self.assertEqual(data.get("rows"), 2)
        self.assertEqual(data.get("cols"), 2)
        clear_cache(data["data_id"], "data")

    # ---- Documentation resource -------------------------------------------

    async def test_autots_docs_resource(self):
        """AutoTS parameter docs are served as a resource, not a tool."""
        resources = await list_resources()
        uris = {str(r.uri) for r in resources}
        self.assertIn("autots://docs/forecast_custom_params", uris)

        content = await read_resource("autots://docs/forecast_custom_params")
        data = json.loads(content)
        self.assertIn("AutoTS_Parameters", data)
        self.assertIn("forecast_length", data["AutoTS_Parameters"])

    # ---- Prompts ----------------------------------------------------------

    async def test_list_prompts(self):
        """Server exposes at least the two built-in workflow prompts."""
        prompts = await list_prompts()
        names = {p.name for p in prompts}
        self.assertIn("sample_forecast", names)
        self.assertIn("explainable_forecast", names)

    async def test_get_prompt_sample_forecast(self):
        from autots.mcp.server import get_prompt

        result = await get_prompt("sample_forecast", {"dataset": "weekly"})
        self.assertIsNotNone(result.messages)
        self.assertGreater(len(result.messages), 0)
        text = result.messages[0].content.text
        self.assertIn("weekly", text)
        self.assertIn("forecast_fast", text)
        self.assertIn("plot_forecast", text)

    async def test_get_prompt_explainable_forecast(self):
        from autots.mcp.server import get_prompt

        result = await get_prompt(
            "explainable_forecast", {"filepath": "/tmp/test.csv", "forecast_length": "14"}
        )
        self.assertIsNotNone(result.messages)
        text = result.messages[0].content.text
        self.assertIn("/tmp/test.csv", text)
        self.assertIn("14", text)
        self.assertIn("forecast_explainable", text)
        self.assertIn("get_forecast_components", text)
        self.assertIn("get_validation_results", text)

    async def test_get_prompt_explainable_forecast_missing_filepath(self):
        from autots.mcp.server import get_prompt

        result = await get_prompt("explainable_forecast", {})
        text = result.messages[0].content.text
        self.assertIn("Error", text)

    # ---- Feature detection tools (shared detector fixture) ---------------

    async def test_detect_features(self):
        data_id = await self._ensure_daily_data()
        result = await call_tool("detect_features", {"data_id": data_id})
        data = _extract(result)
        self.assertIn("detector_id", data)
        self.assertGreater(data.get("series_count", 0), 0)
        # Promote to shared fixture if none exists; otherwise clean up the duplicate.
        cls = self.__class__
        if cls._detector_id is None or cls._detector_id not in CACHE_REGISTRY['feature_detector']:
            cls._detector_id = data["detector_id"]
        else:
            clear_cache(data["detector_id"], "feature_detector")

    async def test_get_detected_features(self):
        detector_id = await self._ensure_detector()
        result = await call_tool(
            "get_detected_features", {"detector_id": detector_id}
        )
        data = _extract(result)
        self.assertIn("summary", data)
        self.assertIn("detection_counts", data)
        self.assertIn("features", data)

    async def test_get_detected_features_with_date_filter(self):
        detector_id = await self._ensure_detector()
        result = await call_tool(
            "get_detected_features",
            {
                "detector_id": detector_id,
                "date_start": "2015-01-01",
                "date_end": "2016-01-01",
            },
        )
        data = _extract(result)
        self.assertIn("summary", data)

    async def test_forecast_from_features(self):
        from autots.mcp.cache import get_cached_object

        detector_id = await self._ensure_detector()
        result = await call_tool(
            "forecast_from_features",
            {
                "detector_id": detector_id,
                "forecast_length": 14,
                "prediction_interval": 0.8,
            },
        )
        data = _extract(result)
        self.assertIn("prediction_id", data)
        self.assertEqual(data.get("forecast_length"), 14)
        self.assertEqual(data.get("prediction_interval"), 0.8)
        prediction = get_cached_object(data["prediction_id"], "prediction")["object"]
        self.assertTrue(
            (prediction.upper_forecast.values >= prediction.forecast.values).all()
        )
        self.assertTrue(
            (prediction.forecast.values >= prediction.lower_forecast.values).all()
        )
        self.assertGreater(
            (prediction.upper_forecast - prediction.lower_forecast)
            .to_numpy()
            .max(),
            0,
        )
        # Clean up derived prediction
        clear_cache(data["prediction_id"], "prediction")

    # ---- Forecast fast pipeline (shared forecast fixture) ----------------

    async def test_forecast_fast_and_get_forecast(self):
        """End-to-end: load -> forecast_fast -> get_forecast."""
        prediction_id = await self._ensure_weekly_forecast()

        forecast_result = await call_tool(
            "get_forecast",
            {"prediction_id": prediction_id, "output": "forecast"},
        )
        fdata = _extract(forecast_result)
        self.assertIn("datetime", fdata)

    async def test_forecast_fast_get_all(self):
        """forecast_fast -> get_forecast(output='all')."""
        prediction_id = await self._ensure_weekly_forecast()

        result = await call_tool(
            "get_forecast",
            {"prediction_id": prediction_id, "output": "all"},
        )
        data = _extract(result)
        self.assertIn("datetime", data)
        self.assertIn("forecast_type", data)
        types_present = set(data["forecast_type"])
        self.assertIn("point", types_present)

    async def test_forecast_fast_get_model_params(self):
        prediction_id = await self._ensure_weekly_forecast()

        result = await call_tool(
            "get_model_params", {"prediction_id": prediction_id}
        )
        data = _extract(result)
        self.assertIn("model_name", data)
        self.assertIn("forecast_length", data)
        self.assertEqual(data["forecast_length"], 8)

    async def test_plot_forecast_single_series_string(self):
        prediction_id = await self._ensure_weekly_forecast()
        series_name = self.__class__._small_weekly.columns[0]
        result = await call_tool(
            "plot_forecast",
            {"prediction_id": prediction_id, "series": series_name},
        )
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertEqual(getattr(result[0], "type", None), "image")

    async def test_apply_constraints_dampen(self):
        prediction_id = await self._ensure_weekly_forecast()

        result = await call_tool(
            "apply_constraints",
            {
                "prediction_id": prediction_id,
                "constraint_method": "dampen",
                "constraint_value": 0.9,
            },
        )
        data = _extract(result)
        self.assertIn("prediction_id", data)
        self.assertNotEqual(data["prediction_id"], prediction_id)
        # Clean up derived prediction
        clear_cache(data["prediction_id"], "prediction")

    async def test_apply_adjustments_ewma(self):
        prediction_id = await self._ensure_weekly_forecast()

        result = await call_tool(
            "apply_adjustments",
            {
                "prediction_id": prediction_id,
                "adjustment_method": "smoothing",
                "adjustment_params": {"span": 5},
            },
        )
        data = _extract(result)
        self.assertIn("prediction_id", data)
        self.assertEqual(data.get("adjustment_method"), "smoothing")
        # Clean up derived prediction
        clear_cache(data["prediction_id"], "prediction")

    async def test_list_cache_after_operations(self):
        # Ensure shared fixtures exist so cache has entries to list
        await self._ensure_weekly_forecast()
        await self._ensure_daily_data()
        result = await call_tool("list_cache", {})
        data = _extract(result)
        self.assertIn("data", data)
        self.assertGreaterEqual(len(data["data"]), 2)


# ===========================================================================
# Event risk tool handlers (async)
#
# Shares data and lazily creates event risk fixture to avoid running the
# expensive EventRiskForecast fit multiple times on full-sized data.
# ===========================================================================


@unittest.skipIf(not SERVER_UTILS_AVAILABLE, "MCP server utilities not available")
class TestMCPEventRiskHandlers(unittest.IsolatedAsyncioTestCase):

    _daily_data_id = None
    _upper_event_risk_id = None

    @classmethod
    def setUpClass(cls):
        clear_cache()
        # Pre-cache a small slice of daily data (150 rows, 2 cols)
        cls._small_daily = load_daily(long=False).iloc[:150, :2]

    @classmethod
    def tearDownClass(cls):
        clear_cache()
        cls._daily_data_id = None
        cls._upper_event_risk_id = None

    async def _ensure_daily_data(self):
        cls = self.__class__
        if cls._daily_data_id is None or cls._daily_data_id not in CACHE_REGISTRY['data']:
            cls._daily_data_id = cache_object(
                cls._small_daily, 'data', {'source': 'test_daily_erf'}
            )
        return cls._daily_data_id

    async def _ensure_upper_event_risk(self):
        cls = self.__class__
        if cls._upper_event_risk_id is None or cls._upper_event_risk_id not in CACHE_REGISTRY['event_risk']:
            data_id = await self._ensure_daily_data()
            result = await call_tool(
                "forecast_event_risk",
                {
                    "data_id": data_id,
                    "threshold": 0.75,
                    "direction": "upper",
                    "forecast_length": 14,
                },
            )
            cls._upper_event_risk_id = _extract(result)["event_risk_id"]
        return cls._upper_event_risk_id

    async def test_forecast_event_risk_and_results(self):
        event_risk_id = await self._ensure_upper_event_risk()

        # Verify the cached entry is valid and has correct direction
        cached = get_cached_object(event_risk_id, 'event_risk')
        self.assertEqual(cached['metadata'].get('direction'), 'upper')

        results = await call_tool(
            "get_event_risk_results", {"event_risk_id": event_risk_id}
        )
        rdata = _extract(results)
        self.assertIn("probabilities", rdata)
        self.assertIn("risk_type", rdata)

    async def test_forecast_event_risk_lower(self):
        data_id = await self._ensure_daily_data()

        result = await call_tool(
            "forecast_event_risk",
            {
                "data_id": data_id,
                "threshold": 0.25,
                "direction": "lower",
                "forecast_length": 7,
            },
        )
        data = _extract(result)
        self.assertIn("event_risk_id", data)
        self.assertEqual(data.get("direction"), "lower")


# ===========================================================================
# Package structure verification
# ===========================================================================


class TestMCPPackaging(unittest.TestCase):
    """Verify that the package is set up correctly for PyPI / uv installs."""

    def _pyproject(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, 'pyproject.toml')
        with open(path) as f:
            return f.read()

    def test_pyproject_exists(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertTrue(os.path.exists(os.path.join(root, 'pyproject.toml')))

    def test_entry_point_defined(self):
        content = self._pyproject()
        self.assertIn('autots-mcp', content)
        self.assertIn('autots.mcp.server:serve', content)

    def test_mcp_optional_dependency_listed(self):
        content = self._pyproject()
        self.assertIn('mcp>=1.0.0', content)

    def test_package_data_json_included(self):
        content = self._pyproject()
        self.assertIn('"autots.mcp"', content)
        self.assertIn('*.json', content)

    def test_mosaic_profile_template_exists(self):
        """The mosaic_profile_template.json file must ship with the package."""
        import autots.mcp as mcp_pkg

        pkg_dir = os.path.dirname(mcp_pkg.__file__)
        template_path = os.path.join(pkg_dir, 'mosaic_profile_template.json')
        self.assertTrue(
            os.path.exists(template_path),
            msg=f"mosaic_profile_template.json not found at {template_path}",
        )
        # Validate JSON is parseable
        with open(template_path) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_mcp_init_importable(self):
        import autots.mcp  # noqa: F401

    def test_server_module_importable(self):
        import autots.mcp.server  # noqa: F401

    def test_main_module_importable(self):
        """__main__.py must exist so `python -m autots.mcp` works."""
        import autots.mcp.__main__  # noqa: F401

    def test_serve_function_callable(self):
        from autots.mcp.server import serve

        self.assertTrue(callable(serve))

    def test_no_hard_mcp_import_at_toplevel(self):
        """Server must import cleanly even when mcp is present; MCP_AVAILABLE=True here."""
        import autots.mcp.server as srv

        # The flag must be a bool
        self.assertIsInstance(srv.MCP_AVAILABLE, bool)

    def test_matplotlib_backend_is_agg(self):
        """Server forces Agg backend so it doesn't need a display."""
        import matplotlib

        self.assertEqual(matplotlib.get_backend().lower(), 'agg')


# ===========================================================================
# Runner
# ===========================================================================


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in [
        TestMCPSampleData,
        TestMCPServerIntegration,
        TestMCPServerUtilities,
        TestMCPForecasting,
        TestMCPFeatureDetection,
        TestMCPSmartLoad,
        TestMCPPyodideAPI,
        TestMCPEventRiskForecasting,
        TestMCPSyntheticData,
        TestMCPToolHandlers,
        TestMCPEventRiskHandlers,
        TestMCPPackaging,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
