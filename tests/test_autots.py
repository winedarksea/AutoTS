# -*- coding: utf-8 -*-
"""Overall testing."""
import unittest
import json
import pandas as pd
from autots.datasets import (
    load_daily
)
from autots import AutoTS
from autots.evaluator.auto_ts import fake_regressor


class AutoTSTest(unittest.TestCase):

    def test_autots(self):
        forecast_length = 8
        long = False
        df = load_daily(long=long).drop(columns=['US.Total.Covid.Tests'], errors='ignore')
        n_jobs = 'auto'
        verbose = 0
        validation_method = "backwards"
        generations = 0
        num_validations = 2
        models_to_validate = 0.35  # must be a decimal percent for this test

        model_list = [
            'ZeroesNaive',
            'LastValueNaive',
            'AverageValueNaive',
            'SeasonalNaive',
        ]

        transformer_list = "fast"  # ["SinTrend", "MinMaxScaler"]
        transformer_max_depth = 3

        metric_weighting = {
            'smape_weighting': 3,
            'mae_weighting': 1,
            'rmse_weighting': 1,
            'containment_weighting': 0,
            'runtime_weighting': 0,
            'spl_weighting': 1,
            'contour_weighting': 1,
        }

        model = AutoTS(
            forecast_length=forecast_length,
            frequency='infer',
            prediction_interval=0.9,
            ensemble=["horizontal-max,horizontal-min"],
            constraint=None,
            max_generations=generations,
            num_validations=num_validations,
            validation_method=validation_method,
            model_list=model_list,
            transformer_list=transformer_list,
            transformer_max_depth=transformer_max_depth,
            initial_template='General+Random',
            metric_weighting=metric_weighting,
            models_to_validate=models_to_validate,
            max_per_model_class=None,
            model_interrupt=False,
            no_negatives=True,
            subset=100,
            n_jobs=n_jobs,
            drop_most_recent=1,
            verbose=verbose,
        )
        future_regressor_train2d, future_regressor_forecast2d = fake_regressor(
            df,
            dimensions=4,
            forecast_length=forecast_length,
            date_col='datetime' if long else None,
            value_col='value' if long else None,
            id_col='series_id' if long else None,
            drop_most_recent=model.drop_most_recent,
            aggfunc=model.aggfunc,
            verbose=model.verbose,
        )
        model = model.fit(
            df,
            future_regressor=future_regressor_train2d,
            date_col='datetime' if long else None,
            value_col='value' if long else None,
            id_col='series_id' if long else None,
        )
        prediction = model.predict(future_regressor=future_regressor_forecast2d, verbose=0)
        forecasts_df = prediction.forecast
        initial_results = model.results()
        validation_results = model.results("validation")

        validated_count = (validation_results['Runs'] == (num_validations + 1)).sum()

        # so these account for DROP MOST RECENT = 1
        expected_idx = pd.date_range(
            start=df.index[-2], periods=forecast_length + 1, freq='D'
        )[1:]
        expected_val1 = pd.date_range(
            end=df.index[-(forecast_length + 2)], periods=forecast_length, freq='D'
        )
        expected_val2 = pd.date_range(
            end=df.index[-(forecast_length * 2 + 2)], periods=forecast_length, freq='D'
        )

        template_dict = json.loads(model.best_model['ModelParameters'].iloc[0])

        # check there were no failed models in this simple setup (fancier models are expected to fail sometimes!)
        self.assertTrue(initial_results['Exceptions'].isna().all())
        self.assertEqual(validated_count, model.models_to_validate)
        self.assertGreater(model.models_to_validate, (initial_results['ValidationRound'] == 0).sum() * models_to_validate - 2)
        # check the generated forecasts look right
        self.assertEqual(forecasts_df.shape[0], forecast_length)
        self.assertEqual(forecasts_df.shape[1], df.shape[1])
        self.assertFalse(forecasts_df.isna().any().any())
        self.assertTrue((forecasts_df >= 0).all().all())
        self.assertEqual(forecast_length, len(forecasts_df.index))
        self.assertTrue((expected_idx == pd.DatetimeIndex(forecasts_df.index)).all())
        # these next two could potentiall fail if any inputs have a strong trend
        self.assertTrue((forecasts_df.mean() <= (df.max()) + df.std()).all())
        self.assertTrue((forecasts_df.mean() >= (df.min()) - df.std()).all())
        # check all the checks work
        self.assertEqual(model.ensemble_check, 1)
        self.assertFalse(model.weighted)
        self.assertFalse(model.used_regressor_check)
        self.assertFalse(model.subset_flag)
        # assess 'backwards' validation
        self.assertEqual(len(model.validation_test_indexes), num_validations)
        self.assertEqual(model.validation_train_indexes[0].shape[0], df.shape[0] - (forecast_length * 2 + 1))  # +1 via drop most recent
        self.assertTrue((model.validation_test_indexes[0].index == expected_val1).all())
        self.assertTrue((model.validation_test_indexes[1].index == expected_val2).all())
        # assess Horizontal Ensembling
        self.assertTrue('horizontal' in template_dict['model_name'].lower())
        self.assertEqual(len(template_dict['series'].keys()), df.shape[1])
        self.assertEqual(len(set(template_dict['series'].values())), template_dict['model_count'])
        self.assertEqual(len(template_dict['models'].keys()), template_dict['model_count'])
