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
from autots.models.model_list import default as default_model_list


class AutoTSTest(unittest.TestCase):

    def test_autots(self):
        forecast_length = 8
        long = False
        df = load_daily(long=long).drop(columns=['US.Total.Covid.Tests'], errors='ignore')
        n_jobs = 'auto'
        verbose = 0
        validation_method = "backwards"
        generations = 1
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
        back_forecast = model.back_forecast(n_splits=2, verbose=0).forecast
        # validated_count = (validation_results['Runs'] == (num_validations + 1)).sum()

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
        best_model_result = validation_results[validation_results['ID'] == model.best_model['ID'].iloc[0]]

        # check there were few failed models in this simple setup (fancier models are expected to fail sometimes!)
        self.assertGreater(initial_results['Exceptions'].isnull().mean(), 0.95, "Too many 'superfast' models failed. This can occur by random chance, try running again.")
        # check general model setup
        # self.assertEqual(validated_count, model.models_to_validate)
        self.assertGreater(model.models_to_validate, (initial_results['ValidationRound'] == 0).sum() * models_to_validate - 2)
        self.assertEqual(set(initial_results['Model'].unique().tolist()) - {'Ensemble'}, set(model.model_list))
        self.assertFalse(model.best_model.empty)
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
        self.assertTrue(model.validation_test_indexes[0].intersection(model.validation_train_indexes[0]).empty)
        self.assertTrue(model.validation_test_indexes[1].intersection(model.validation_train_indexes[1]).empty)
        self.assertEqual(model.validation_train_indexes[0].shape[0], df.shape[0] - (forecast_length * 2 + 1))  # +1 via drop most recent
        self.assertTrue((model.validation_test_indexes[0] == expected_val1).all())
        self.assertTrue((model.validation_test_indexes[1] == expected_val2).all())
        # assess Horizontal Ensembling
        self.assertTrue('horizontal' in template_dict['model_name'].lower())
        self.assertEqual(len(template_dict['series'].keys()), df.shape[1])
        self.assertEqual(len(set(template_dict['series'].values())), template_dict['model_count'])
        self.assertEqual(len(template_dict['models'].keys()), template_dict['model_count'])
        # test that actually the best model (or nearly) was chosen
        self.assertGreater(validation_results['Score'].quantile(0.05), best_model_result['Score'].iloc[0])
        # test back_forecast
        self.assertTrue((back_forecast.index == model.df_wide_numeric.index).all(), msg="Back forecasting failed to have equivalent index to train.")

        # a
        # b
        # c
        # d
        # e
        # f
        # g
    def test_all_default_models(self):
        forecast_length = 8
        long = False
        df = load_daily(long=long).drop(columns=['US.Total.Covid.Tests'], errors='ignore')
        # to make it faster
        df = df[df.columns[0:2]]
        n_jobs = 'auto'
        verbose = -1
        validation_method = "backwards"
        generations = 1
        num_validations = 1
        models_to_validate = 0.10  # must be a decimal percent for this test

        model_list = "default"

        transformer_list = "fast"  # ["SinTrend", "MinMaxScaler"]
        transformer_max_depth = 1

        model = AutoTS(
            forecast_length=forecast_length,
            frequency='infer',
            prediction_interval=0.9,
            ensemble=["horizontal-max"],
            constraint=None,
            max_generations=generations,
            num_validations=num_validations,
            validation_method=validation_method,
            model_list=model_list,
            transformer_list=transformer_list,
            transformer_max_depth=transformer_max_depth,
            initial_template='Random',
            models_to_validate=models_to_validate,
            max_per_model_class=None,
            n_jobs=n_jobs,
            drop_most_recent=1,
            verbose=verbose,
        )
        model = model.fit(
            df,
            date_col='datetime' if long else None,
            value_col='value' if long else None,
            id_col='series_id' if long else None,
        )
        prediction = model.predict(verbose=0)
        forecasts_df = prediction.forecast
        initial_results = model.results()
        validation_results = model.results("validation")

        # validated_count = (validation_results['Runs'] == (num_validations + 1)).sum()
        validated_count = (validation_results['Runs'] > 1).sum()

        # so these account for DROP MOST RECENT = 1
        expected_idx = pd.date_range(
            start=df.index[-2], periods=forecast_length + 1, freq='D'
        )[1:]
        expected_val1 = pd.date_range(
            end=df.index[-(forecast_length + 2)], periods=forecast_length, freq='D'
        )

        template_dict = json.loads(model.best_model['ModelParameters'].iloc[0])
        best_model_result = validation_results[validation_results['ID'] == model.best_model['ID'].iloc[0]]

        check_fails = initial_results.groupby("Model")["mae"].count() > 0

        # check that all models had at least 1 success
        self.assertEqual(set(initial_results['Model'].unique().tolist()) - {'Ensemble'}, set(default_model_list), msg="Not all models used in initial template.")
        self.assertTrue(check_fails.all(), msg=f"These models failed: {check_fails[~check_fails].index.tolist()}. It is more likely a package install problem than a code problem")
        # check general model setup
        self.assertGreaterEqual(validated_count, model.models_to_validate)
        self.assertGreater(model.models_to_validate, (initial_results['ValidationRound'] == 0).sum() * models_to_validate - 2)
        self.assertFalse(model.best_model.empty)
        # check the generated forecasts look right
        self.assertEqual(forecasts_df.shape[0], forecast_length)
        self.assertEqual(forecasts_df.shape[1], df.shape[1])
        self.assertFalse(forecasts_df.isna().any().any())
        self.assertEqual(forecast_length, len(forecasts_df.index))
        self.assertTrue((expected_idx == pd.DatetimeIndex(forecasts_df.index)).all())
        # these next two could potentiall fail if any inputs have a strong trend
        self.assertTrue((forecasts_df.mean() <= (df.max()) + df.std()).all())
        self.assertTrue((forecasts_df.mean() >= (df.min()) - df.std()).all())
        # check all the checks work
        self.assertEqual(model.ensemble_check, 1)
        self.assertFalse(model.weighted)
        self.assertFalse(model.subset_flag)
        # assess 'backwards' validation
        self.assertEqual(len(model.validation_test_indexes), num_validations)
        self.assertTrue(model.validation_test_indexes[0].intersection(model.validation_train_indexes[0]).empty)
        self.assertEqual(model.validation_train_indexes[0].shape[0], df.shape[0] - (forecast_length * 2 + 1))  # +1 via drop most recent
        self.assertTrue((model.validation_test_indexes[0] == expected_val1).all())
        # assess Horizontal Ensembling
        self.assertTrue('horizontal' in template_dict['model_name'].lower())
        self.assertEqual(len(template_dict['series'].keys()), df.shape[1])
        self.assertEqual(len(set(template_dict['series'].values())), template_dict['model_count'])
        self.assertEqual(len(template_dict['models'].keys()), template_dict['model_count'])
        # test that actually the best model (or nearly) was chosen
        self.assertGreater(validation_results['Score'].quantile(0.05), best_model_result['Score'].iloc[0])
        # test metrics
        self.assertTrue(initial_results['Score'].min() > 0)
        self.assertTrue(initial_results['mae'].min() >= 0)
        self.assertTrue(initial_results['smape'].min() >= 0)
        self.assertTrue(initial_results['rmse'].min() >= 0)
        self.assertTrue(initial_results['contour'].min() >= 0)
        self.assertTrue(initial_results['containment'].min() >= 0)
        self.assertTrue(initial_results['TotalRuntimeSeconds'].min() >= 0)
        self.assertTrue(initial_results['spl'].min() >= 0)
        self.assertTrue(initial_results['contour'].min() <= 1)
        self.assertTrue(initial_results['containment'].min() <= 1)

        # test all same on univariate input, non-horizontal, with regressor, and different frequency, with forecast_length = 1 !

        # the big ones are:
        # 1. that validations are sampled correctly
        # 2. that accuracy metrics are performed and aggregated correctly

        # test template import and export
        # test result saving and import
        # test seasonal validation
        # test score generation + metric_weighting
        # test very short training data and/or lots of NaNs in data
        # test on all models that for each model, failure rate is < 100%
