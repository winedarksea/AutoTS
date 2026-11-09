# -*- coding: utf-8 -*-
"""export_template focus_models and import_template argument handling."""

import unittest
import warnings

import numpy as np
import pandas as pd

from autots import AutoTS


def _make_seasonal_df(rows=100, cols=3, seed=7):
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


class TestTemplateExportImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.model = AutoTS(
            forecast_length=7,
            frequency="D",
            model_list="superfast",
            ensemble=None,
            num_validations=1,
            max_generations=1,
            verbose=0,
            random_seed=1,
            n_jobs=1,
        ).fit(_make_seasonal_df())

    def test_focus_models_adds_best_of_each_model(self):
        results = self.model.validation_results.model_results
        validated = results[results["Runs"] >= 2]
        losing_model = (
            validated[validated["Model"] != self.model.best_model_name]
            .sort_values("Score")["Model"]
            .iloc[-1]
        )
        # include_results avoids the n == 1 shortcut that returns best_model directly
        export_kwargs = dict(models="best", n=1, include_results=True)
        plain = self.model.export_template(min_metrics=None, **export_kwargs)
        self.assertNotIn(losing_model, plain["Model"].tolist())

        focused = self.model.export_template(
            min_metrics=None, focus_models=[losing_model], **export_kwargs
        )
        self.assertIn(losing_model, focused["Model"].tolist())
        # metric loops inside the focus branch must not need a prior metric variable
        focused_metrics = self.model.export_template(
            min_metrics=["smape"],
            max_metrics=["containment"],
            focus_models=[losing_model],
            **export_kwargs,
        )
        self.assertIn(losing_model, focused_metrics["Model"].tolist())

    def test_import_template_rejects_unknown_method(self):
        template = self.model.export_template(models="best", n=1)
        with self.assertRaises(ValueError):
            AutoTS(forecast_length=7).import_template(template, method="replace")


if __name__ == "__main__":
    unittest.main()
