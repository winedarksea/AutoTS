# -*- coding: utf-8 -*-
"""Tests for the PreprocessingExperts composite model."""

import unittest
import numpy as np
import pandas as pd

from autots.models.composite import PreprocessingExperts


def _build_training_frame(rows: int = 120, cols: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="D")
    values = rng.normal(loc=50.0, scale=5.0, size=(rows, cols))
    values += np.linspace(0, 3, rows)[:, None]
    values += 2.5 * np.sin(np.arange(rows) * 2 * np.pi / 7)[:, None]
    return pd.DataFrame(
        values, index=dates, columns=[f"series_{i}" for i in range(cols)]
    )


def _base_model_params() -> dict:
    return {
        "model_str": "LastValueNaive",
        "parameter_dict": {},
        "transformation_dict": {
            "fillna": None,
            "transformations": {},
            "transformation_params": {},
        },
    }


class TestPreprocessingExperts(unittest.TestCase):
    def test_basic_fit_predict_closest_point_method(self):
        df = _build_training_frame()
        forecast_length = 7
        transformation_dict = {
            "fillna": "ffill",
            "transformations": {0: "StandardScaler", 1: "DifferencedTransformer"},
            "transformation_params": {0: {}, 1: {}},
        }

        model = PreprocessingExperts(
            forecast_length=forecast_length,
            transformation_dict=transformation_dict,
            model_params=_base_model_params(),
            point_method="closest",
            random_seed=123,
        )

        model.fit(df)
        prediction = model.predict(forecast_length=forecast_length)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.forecast.shape, (forecast_length, df.shape[1]))
        self.assertFalse(prediction.forecast.isnull().any().any())
        self.assertEqual(
            model.result_windows.shape[0],
            len(transformation_dict["transformations"]) + 1,
        )
        np.testing.assert_allclose(
            prediction.forecast.values,
            model.result_windows[-1],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_requires_transformation_dict(self):
        df = _build_training_frame(rows=40, cols=2)
        model = PreprocessingExperts(
            forecast_length=5,
            transformation_dict=None,
            model_params=_base_model_params(),
        )

        with self.assertRaises(ValueError):
            model.fit(df)

    def test_string_indexed_transformations_preserve_numeric_order(self):
        df = _build_training_frame(rows=90, cols=3)
        transformation_dict = {
            "fillna": "ffill",
            "transformations": {
                "0": "StandardScaler",
                "10": "RollingMeanTransformer",
                "2": "PositiveShift",
            },
            "transformation_params": {
                "0": {},
                "10": {"window": 5, "fixed": True},
                "2": {},
            },
        }

        model = PreprocessingExperts(
            forecast_length=6,
            transformation_dict=transformation_dict,
            model_params=_base_model_params(),
            point_method="mean",
            random_seed=321,
        )

        model.fit(df)
        applied_keys = list(model.transformer_object.transformers.keys())
        self.assertListEqual(applied_keys, [0, 2, 10])

        prediction = model.predict(forecast_length=6)
        self.assertEqual(prediction.forecast.shape[0], 6)
        self.assertFalse(np.isnan(prediction.forecast.values).any())


if __name__ == "__main__":
    unittest.main()
