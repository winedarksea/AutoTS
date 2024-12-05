# -*- coding: utf-8 -*-
"""Overall testing."""
import unittest
import json
import time
import timeit
import tempfile
import os
import numpy as np
import pandas as pd
from autots.datasets import (
    load_daily, load_monthly, load_artificial, load_sine
)
from autots import AutoTS, model_forecast, ModelPrediction
from autots.evaluator.auto_ts import fake_regressor
from autots.evaluator.auto_model import ModelMonster
from autots.models.ensemble import full_ensemble_test_list
from autots.models.model_list import default as default_model_list
from autots.models.model_list import all_models
from autots.evaluator.benchmark import Benchmark
from autots.templates.general import general_template
from autots.tools.cpu_count import cpu_count, set_n_jobs


class AutoTSTest(unittest.TestCase):

    def test_autots(self):
        print("Starting AutoTS class tests")
        forecast_length = 8
        long = False
        df = load_daily(long=long)
        n_jobs = 'auto'
        verbose = 0
        validation_method = "backwards"
        generations = 1
        num_validations = 2
        models_to_validate = 0.25  # must be a decimal percent for this test

        model_list = [
            'ConstantNaive',
            'LastValueNaive',
            'AverageValueNaive',
            'SeasonalNaive',
            'DatepartRegression',
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
        ensemble = full_ensemble_test_list

        model = AutoTS(
            forecast_length=forecast_length,
            frequency='infer',
            prediction_interval=0.9,
            ensemble=ensemble,
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
            model_interrupt="end_generation",
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
        # first test multiple prediction intervals
        prediction = model.predict(future_regressor=future_regressor_forecast2d, prediction_interval=[0.6, 0.9], verbose=0)
        prediction = model.predict(future_regressor=future_regressor_forecast2d, verbose=0)
        long_form = prediction.long_form_results()
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
        self.assertGreater(model.validation_template.size, (initial_results['ValidationRound'] == 0).sum() * models_to_validate - 2)
        self.assertEqual(set(initial_results['Model'].unique().tolist()) - {'Ensemble', 'MLEnsemble'}, set(model.model_list))
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
        self.assertFalse(model.subset_flag)
        # assess 'backwards' validation
        self.assertEqual(len(model.validation_test_indexes), num_validations + 1)
        self.assertTrue(model.validation_test_indexes[1].intersection(model.validation_train_indexes[1]).empty)
        self.assertTrue(model.validation_test_indexes[2].intersection(model.validation_train_indexes[2]).empty)
        self.assertEqual(model.validation_train_indexes[1].shape[0], df.shape[0] - (forecast_length * 2 + 1))  # +1 via drop most recent
        self.assertTrue((model.validation_test_indexes[1] == expected_val1).all())
        self.assertTrue((model.validation_test_indexes[2] == expected_val2).all())
        # assess Horizontal Ensembling
        tested_horizontal = 'horizontal' in template_dict['model_name'].lower()
        tested_mosaic = 'mosaic' in template_dict['model_name'].lower()
        print(f"chosen model was mosaic: {tested_mosaic} or was horizontal: {tested_horizontal}")
        self.assertTrue(tested_horizontal or tested_mosaic)
        self.assertEqual(len(template_dict['series'].keys()), df.shape[1])
        if tested_horizontal:
            self.assertEqual(len(set(template_dict['series'].values())), template_dict['model_count'])
        self.assertEqual(len(template_dict['models'].keys()), template_dict['model_count'])
        # check that the create number of models were available that were requested
        one_mos = initial_results[initial_results["ModelParameters"].str.contains("mosaic-spl-3-10")]
        res = []
        for x in json.loads(one_mos["ModelParameters"].iloc[0])["series"].values():
            for y in x.values():
                res.append(y)
        self.assertLessEqual(len(set(res)), 10)
        # check all mosaic and horizontal styles were created
        count_horz = len([x for x in ensemble if "horizontal" in x or "mosaic" in x])
        self.assertEqual(len(initial_results[initial_results["Ensemble"] == 2]["ModelParameters"].unique()), count_horz)
        # check the mosaic details were equal
        self.assertTrue(len(model.initial_results.full_mae_errors) == len(model.initial_results.full_mae_ids) == len(model.initial_results.full_mae_vals))
        # check at least 1 'simple' ensemble worked
        self.assertGreater(initial_results[initial_results["Ensemble"] == 1]['Exceptions'].isnull().sum(), 0)
        # test that actually the best model (or nearly) was chosen
        self.assertGreater(validation_results['Score'].quantile(0.05), best_model_result['Score'].iloc[0])
        # test back_forecast
        # self.assertTrue((back_forecast.index == model.df_wide_numeric.index).all(), msg="Back forecasting failed to have equivalent index to train.")
        self.assertFalse(np.any(back_forecast.isnull()))
        self.assertEqual(long_form.shape[0], forecasts_df.shape[0] * forecasts_df.shape[1] * 3)
        # results present
        self.assertGreater(model.initial_results.per_series_metrics.shape[0], 1)
        # assert that per_series results have appropriate column names
        self.assertCountEqual(model.initial_results.per_series_mae.columns.tolist(), df.columns.tolist())

        # TEST EXPORTING A TEMPLATE THEN USING THE BEST MODEL AS A PREDICTION
        df_train = df.iloc[:-forecast_length]
        df_test = df.iloc[-forecast_length:]
        tf = tempfile.NamedTemporaryFile(suffix='.csv', prefix=os.path.basename("autots_test"), delete=False)
        time.sleep(1)
        name = tf.name
        model.export_template(name, models="best", n=20, max_per_model_class=3)
        future_regressor_train2d, future_regressor_forecast2d = fake_regressor(
            df,
            dimensions=4,
            forecast_length=forecast_length,
            date_col='datetime' if long else None,
            value_col='value' if long else None,
            id_col='series_id' if long else None,
            drop_most_recent=0,
            aggfunc=model.aggfunc,
            verbose=model.verbose,
        )
        
        model2 = AutoTS(
            forecast_length=forecast_length,
            frequency='infer',
            prediction_interval=0.9,
            ensemble='all',
            constraint=None,
            max_generations=generations,
            num_validations=num_validations,
            validation_method=validation_method,
            model_list="update_fit",
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
            drop_most_recent=0,
            verbose=2,
        )
        # TEST MODEL PREDICT WITH LOWER LEVEL MODEL TRAINED ON PREVIOUS DATA ONLY
        model2.import_best_model(tf.name, include_ensemble=False)
        model2.fit_data(df_train, future_regressor=future_regressor_train2d.reindex(df_train.index))
        prediction = model2.predict(future_regressor=future_regressor_train2d.reindex(df_test.index), verbose=0)
        prediction.evaluate(df_test, df_train=df_train)
        smape1 = prediction.avg_metrics['smape']
        
        model2.fit_data(df, future_regressor=future_regressor_train2d)
        try:
            prediction2 = model2.predict(future_regressor=future_regressor_forecast2d, verbose=0)
        except Exception as e:
            raise ValueError(f"prediction failed with model {model2.best_model}") from e
        forecasts_df2 = prediction2.forecast
        
        # now retrain on full data
        model2.model = None
        model2.fit_data(df, future_regressor=future_regressor_train2d)
        prediction2 = model2.predict(future_regressor=future_regressor_forecast2d, verbose=0)
        # and see if it got better on past holdout
        model2.fit_data(df_train, future_regressor=future_regressor_train2d.reindex(df_train.index))
        prediction = model2.predict(future_regressor=future_regressor_train2d.reindex(df_test.index), verbose=0)
        prediction.evaluate(df_test, df_train=df_train)
        smape2 = prediction.avg_metrics['smape']
        print("=====================================================")
        # smape2 should be better because it is trained on the very data it is supposed to predict
        print(f"fit 1 SMAPE {smape1}, then refit with history SMAPE: {smape2}")

        tf.close()
        os.unlink(tf.name)

        
        self.assertEqual(forecasts_df2.shape[0], forecast_length)
        self.assertEqual(forecasts_df2.shape[1], df.shape[1])
        self.assertFalse(forecasts_df2.isna().any().any())
        self.assertFalse(model.initial_results.per_series_metrics.empty)

    def test_all_default_models(self):
        print("Starting test_all_default_models")
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
        constraint = {
            "constraint_method": "quantile",
            "constraint_regularization": 0.9,
            "upper_constraint": 0.99,
            "lower_constraint": 0.01,
            "bounds": True,
        }

        model = AutoTS(
            forecast_length=forecast_length,
            frequency='infer',
            prediction_interval=0.9,
            ensemble=["horizontal-max"],
            constraint=constraint,
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
            model_interrupt=True,
            drop_most_recent=1,
            verbose=verbose,
            random_seed=1918,
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
        lvl1 = initial_results[initial_results["Exceptions"].isnull()]
        self.assertGreater(model.models_to_validate, (lvl1[lvl1["Ensemble"] == 0]['ValidationRound'] == 0).sum() * models_to_validate - 1)
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
        self.assertFalse(model.used_regressor_check)
        # assess 'backwards' validation
        val_1 = model.validation_test_indexes[1]
        self.assertEqual(len(model.validation_test_indexes), num_validations + 1)
        self.assertTrue(val_1.intersection(model.validation_train_indexes[1]).empty)
        self.assertEqual(model.validation_train_indexes[1].shape[0], df.shape[0] - (forecast_length * 2 + 1))  # +1 via drop most recent
        self.assertTrue((val_1 == expected_val1).all())
        # assess Horizontal Ensembling
        self.assertTrue('horizontal' in template_dict['model_name'].lower())
        self.assertEqual(len(template_dict['series'].keys()), df.shape[1])
        self.assertEqual(len(set(template_dict['series'].values())), template_dict['model_count'])
        self.assertEqual(len(template_dict['models'].keys()), template_dict['model_count'])
        # test that actually the best model (or nearly) was chosen
        self.assertGreater(validation_results[validation_results['Runs'] > model.num_validations]['Score'].quantile(0.05), best_model_result['Score'].iloc[0])
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
        
        # Test that generate_score is actually picking the lowest value on a single metric
        # minimizing metrics only
        # no weights present on metrics
        for target_metric in ['smape', 'mae', 'rmse', 'made', 'spl', 'mage', 'mle', 'imle', 'dwae', 'mqae', 'uwmse', 'wasserstein', 'dwd']:
            with self.subTest(i=target_metric):
                new_weighting = {
                    str(target_metric) + '_weighting': 1,
                }
                temp_cols = ['ID', 'Model', 'ModelParameters', 'TransformationParameters', 'Ensemble', target_metric]
                new_mod = model._return_best_model(metric_weighting=new_weighting, template_cols=temp_cols)
                new_mod_non = new_mod[1]
                new_mod = new_mod[0]
                if new_mod['Ensemble'].iloc[0] == 2:
                    min_pos = validation_results[validation_results['Ensemble'] == 2][target_metric].min()
                    min_pos_non = validation_results[(validation_results['Ensemble'] < 2) & (validation_results['Runs'] > model.num_validations)][target_metric].min()
                    chos_pos = new_mod[target_metric].iloc[0]
                    # print(min_pos)
                    # print(chos_pos)
                    self.assertTrue(np.allclose(chos_pos, min_pos))
                    self.assertTrue(np.allclose(new_mod_non[target_metric].iloc[0], min_pos_non))
                    # print(json.loads(new_mod['ModelParameters'].iloc[0])['model_name'])
                    # print(json.loads(new_mod['ModelParameters'].iloc[0])['model_metric'])

    def test_load_datasets(self):
        df = load_artificial(long=True)
        df = load_monthly(long=True)
        df = load_sine(long=False)
        df = load_daily(long=False)
        df = load_daily(long=True)  # noqa

    def test_new_params(self):
        params = AutoTS.get_new_params()
        self.assertIsInstance(params, dict)
        model = AutoTS(**params)  # noqa

        params = AutoTS.get_new_params(method='fast')
        self.assertIsInstance(params, dict)
        model = AutoTS(**params)  # noqa

    def test_univariate1step(self):
        print("Starting test_univariate1step")
        df = load_artificial(long=False)
        df.iloc[:, :1]
        forecast_length = 1
        n_jobs = 1
        verbose = -1
        validation_method = "backwards"
        generations = 1
        model_list = [
            'ConstantNaive',
            'LastValueNaive',
            'AverageValueNaive',
            'SeasonalNaive',
        ]

        model = AutoTS(
            forecast_length=forecast_length,
            frequency='infer',
            max_generations=generations,
            validation_method=validation_method,
            model_list=model_list,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        model = model.fit(
            df,
        )
        prediction = model.predict(verbose=0)
        forecasts_df = prediction.forecast
        initial_results = model.results()

        expected_idx = pd.date_range(
            start=df.index[-1], periods=forecast_length + 1, freq='D'
        )[1:]
        check_fails = initial_results.groupby("Model")["mae"].count() > 0
        self.assertTrue(check_fails.all(), msg=f"These models failed: {check_fails[~check_fails].index.tolist()}. It is more likely a package install problem than a code problem")
        # check the generated forecasts look right
        self.assertEqual(forecasts_df.shape[0], forecast_length)
        self.assertEqual(forecasts_df.shape[1], df.shape[1])
        self.assertFalse(forecasts_df.isna().any().any())
        self.assertEqual(forecast_length, len(forecasts_df.index))
        self.assertTrue((expected_idx == pd.DatetimeIndex(forecasts_df.index)).all())

    def test_subset_expansion(self):
        # probably has room for testing some more things as well
        long = True
        df = load_daily(long=long)
        forecast_length = 28
        model = AutoTS(
            forecast_length=forecast_length,
            frequency='infer',
            max_generations=10,
            validation_method="seasonal",
            model_list="superfast",
            ensemble = [
                "horizontal-max",
                "mosaic-weighted-0-10",
                "mosaic-mae-crosshair-0-20",
            ],
            n_jobs=2,
            verbose=-1,
            subset=4,
            remove_leading_zeroes=True,
        )
        model = model.fit(
            df,
            date_col="datetime" if long else None,
            value_col="value" if long else None,
            id_col="series_id" if long else None,
        )
        model.expand_horizontal()
        orig_param = json.loads(model.best_model_original.iloc[0]['ModelParameters'])
        new_param = json.loads(model.best_model.iloc[0]['ModelParameters'])
        diff_mods = [x for x in orig_param['models'].keys() if x not in new_param['models'].keys()]
        if diff_mods:
            details = orig_param['models'][diff_mods]
        else:
            details = ""
        self.assertCountEqual(
            orig_param['models'].keys(),
            new_param['models'].keys(),
            msg=f"model expansion failed to use the same models {details}"
        )
        num_series = len(df['series_id'].unique().tolist()) if long else df.shape[1]
        self.assertEqual(
            len(json.loads(model.best_model.iloc[0]['ModelParameters'])['series'].keys()),
            num_series,
            msg="model expansion failed to expand to all df columns"
        )
        prediction = model.predict(verbose=0)
        forecasts_df = prediction.forecast
        initial_results = model.results()

        check_fails = initial_results.groupby("Model")["mae"].count() > 0
        self.assertTrue(check_fails.all(), msg=f"These models failed: {check_fails[~check_fails].index.tolist()}. It is more likely a package install problem than a code problem")
        # check the generated forecasts look right
        self.assertEqual(forecasts_df.shape[0], forecast_length)
        self.assertEqual(forecasts_df.shape[1], num_series)
        self.assertFalse(forecasts_df.isna().any().any())
        self.assertEqual(forecast_length, len(forecasts_df.index))

    def test_all_models_load(self):
        print("Starting test_all_models_load")
        # make sure it can at least load a template of all models
        forecast_length = 8
        n_jobs = 'auto'
        verbose = 4
        generations = 0

        model_list = "all"
        transformer_list = "all"  # ["SinTrend", "MinMaxScaler"]
        transformer_max_depth = 10

        model = AutoTS(
            forecast_length=forecast_length,
            frequency='infer',
            prediction_interval=0.9,
            ensemble=["horizontal-max"],
            constraint=None,
            max_generations=generations,
            model_list=model_list,
            transformer_list=transformer_list,
            transformer_max_depth=transformer_max_depth,
            initial_template='Random',
            max_per_model_class=None,
            n_jobs=n_jobs,
            model_interrupt=True,
            drop_most_recent=1,
            verbose=verbose,
        )
        self.assertFalse(model.initial_template.empty)

    def test_benchmark(self):
        print("Starting test_benchmark")
        bench = Benchmark()
        bench.run(times=1, verbose=-1)
        self.assertGreater(bench.total_runtime, 0)
        print(f"Benchmark total_runtime: {bench.total_runtime}")
        print(bench.results)
        time.sleep(5)

    def test_template(self):
        for index, row in general_template.iterrows():
            model = row["Model"]
            with self.subTest(i=model):
                mod = json.loads(row['ModelParameters'])
                trans = json.loads(row['TransformationParameters'])
                ensemble = row["Ensemble"]
                self.assertIsInstance(mod, dict)
                self.assertIsInstance(trans, dict)
                self.assertIsNotNone(ensemble)

    def test_custom_validations(self):
        long = False
        df = load_daily(long=long)
        forecast_length = 28
        model = AutoTS(
            forecast_length=forecast_length,
            frequency='D',
            max_generations=10,
            validation_method="custom",
            model_list="superfast",
            ensemble=None,
            n_jobs=1,
            verbose=2,
            subset=4,
            remove_leading_zeroes=True,
            generation_timeout=1,
            horizontal_ensemble_validation=True,
        )
        custom_idx = [
            pd.date_range(df.index[0], df.index[-100]),
            pd.date_range(df.index[0], df.index[-(100 + forecast_length)]),
        ]
        model = model.fit(
            df,
            validation_indexes=custom_idx
        )
        self.assertEqual(model.ensemble_check, 0)

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


class ModelTest(unittest.TestCase):
    
    def test_models_get_params(self):
        """See if new random params can be generated without error."""
        default_methods = ['deep', 'fast', 'random', 'default', 'superfast', 'regressor', 'event_risk']
        for method in default_methods:
            for model_str in all_models:
                ModelMonster(model_str).get_new_params(method=method)
        

    def test_models(self):
        """Test if models are the same as saved comparisons."""
        print("Starting test_models")
        n_jobs = 1
        random_seed = 300
        df = load_daily(long=False).iloc[:, 0:5]
        df = df[df.index < "2022-10-04"]  # update dataset and have not yet updated stored model results
        df = df[df.index > "2017-10-04"]  # update dataset and have not yet updated stored model results
        models = [
            'SectionalMotif', 'MultivariateMotif', 'AverageValueNaive',
            'NVAR', "LastValueNaive", 'Theta', 'FBProphet', 'SeasonalNaive',
            'GLM', 'ETS', "ConstantNaive", 'WindowRegression',
            'DatepartRegression', 'MultivariateRegression',
            'Cassandra', 'MetricMotif', 'SeasonalityMotif', 'KalmanStateSpace',
            'ARDL', 'UnivariateMotif', 'VAR', 'MAR', 'TMF', 'RRVAR', 'VECM',
            'BallTreeMultivariateMotif', 'FFT',
            "DMD",  # 0.6.12
            "BasicLinearModel", "TVVAR",  # 0.6.16
            # "BallTreeRegressionMotif",  # 0.6.17
        ]
        # models that for whatever reason arne't consistent across test sessions
        run_only_no_score = ['FBProphet', 'RRVAR', "TMF"]

        timings = {}
        forecasts = {}
        upper_forecasts = {}
        lower_forecasts = {}
        # load the comparison source
        with open("./tests/model_forecasts.json", "r") as file:
            loaded = json.load(file)
            for x in models:
                forecasts[x] = pd.DataFrame.from_dict(loaded['forecasts'][x], orient="columns")
                forecasts[x]['index'] = pd.to_datetime(forecasts[x]['index'])
                forecasts[x] = forecasts[x].set_index("index")
                upper_forecasts[x] = pd.DataFrame.from_dict(loaded['upper_forecasts'][x], orient="columns")
                upper_forecasts[x]['index'] = pd.to_datetime(upper_forecasts[x]['index'])
                upper_forecasts[x] = upper_forecasts[x].set_index("index")
                lower_forecasts[x] = pd.DataFrame.from_dict(loaded['lower_forecasts'][x], orient="columns")
                lower_forecasts[x]['index'] = pd.to_datetime(lower_forecasts[x]['index'])
                lower_forecasts[x] = lower_forecasts[x].set_index("index")
            timings = loaded['timing']

        timings2 = {}
        forecasts2 = {}
        upper_forecasts2 = {}
        lower_forecasts2 = {}
        # following are not consistent with seed:
        # "MotifSimulation"

        for x in models:
            print(x)
            try:
                start_time = timeit.default_timer()
                df_forecast = model_forecast(
                    model_name=x,
                    model_param_dict={},  # 'return_result_windows': True
                    model_transform_dict={
                        "fillna": "ffill",
                        "transformations": {"0": "StandardScaler"},
                        "transformation_params": {"0": {}},
                    },
                    df_train=df,
                    forecast_length=5,
                    frequency="D",
                    prediction_interval=0.9,
                    random_seed=random_seed,
                    verbose=0,
                    # bug in sklearn 1.1.2 for n_jobs for RandomForest
                    n_jobs=n_jobs if x != "WindowRegression" else 2,
                    return_model=True,
                )
                forecasts2[x] = df_forecast.forecast.round(2)
                upper_forecasts2[x] = df_forecast.upper_forecast.round(2)
                lower_forecasts2[x] = df_forecast.lower_forecast.round(2)
                timings2[x] = (timeit.default_timer() - start_time)
            except Exception as e:
                raise ValueError(f"model {x} failed with {repr(e)}")

        print(sum(timings.values()))

        pass_probabilistic = ['FBProphet']  # not yet reproducible in upper/lower with seed
        for x in models:
            if x not in run_only_no_score:
                with self.subTest(i=x):
                    res = (forecasts2[x].round(2) == forecasts[x].round(2)).all().all()
                    if x not in pass_probabilistic:
                        res_u = (upper_forecasts2[x].round(2) == upper_forecasts[x].round(2)).all().all()
                        res_l = (lower_forecasts2[x].round(2) == lower_forecasts[x].round(2)).all().all()
                    else:
                        res_u = True
                        res_l = True
                    self.assertTrue(
                        res,
                        f"Model '{x}' forecasts diverged from sample forecasts."
                    )
                    self.assertTrue(
                        res_u,
                        f"Model '{x}' upper forecasts diverged from sample forecasts."
                    )
                    self.assertTrue(
                        res_l,
                        f"Model '{x}' lower forecasts diverged from sample forecasts."
                    )
                    print(f"{res & res_u & res_l} model '{x}' ran successfully in {round(timings2[x], 4)} (bench: {round(timings[x], 4)})")

        """
        for x in models:
            forecasts[x].index = forecasts[x].index.strftime("%Y-%m-%d")
            forecasts[x] = forecasts[x].reset_index(drop=False).to_dict(orient="list")
            upper_forecasts[x].index = upper_forecasts[x].index.strftime("%Y-%m-%d")
            upper_forecasts[x] = upper_forecasts[x].reset_index(drop=False).to_dict(orient="list")
            lower_forecasts[x].index = lower_forecasts[x].index.strftime("%Y-%m-%d")
            lower_forecasts[x] = lower_forecasts[x].reset_index(drop=False).to_dict(orient="list")

        with open("./tests/model_forecasts.json", "w") as file:
            json.dump(
                {
                    'forecasts': forecasts,
                    "upper_forecasts": upper_forecasts,
                    "lower_forecasts": lower_forecasts,
                    "timing": timings,
                }, file
            )
        """

    def test_transforms(self):
        """Test if transformers meet saved comparison outputs."""
        print("Starting test_transforms")
        n_jobs = 1
        random_seed = 300
        df = load_monthly(long=False)[['CSUSHPISA', 'EMVOVERALLEMV', 'EXCAUS']]
        transforms = [
            'MinMaxScaler', 'PowerTransformer', 'QuantileTransformer',
            'MaxAbsScaler', 'StandardScaler', 'RobustScaler',
            'PCA', 'FastICA', "DatepartRegression",
            "EWMAFilter", 'STLFilter', 'HPFilter', 'Detrend', 'Slice',
            'ScipyFilter', 'Round', 'ClipOutliers', 'IntermittentOccurrence',
            'CenterLastValue', 'Discretize', 'SeasonalDifference',
            'RollingMeanTransformer', 'bkfilter', 'cffilter', 'Log',
            'DifferencedTransformer', 'PctChangeTransformer', 'PositiveShift',
            'SineTrend', 'convolution_filter', 'CumSumTransformer',
            'AlignLastValue',  # new 0.4.3
            'AnomalyRemoval', "HolidayTransformer",  # new 0.5.0
            'LocalLinearTrend',  # new 0.5.1
            "KalmanSmoothing",  # new 0.5.1
            "RegressionFilter",   # new 0.5.7
            "LevelShiftTransformer",  # new 0.6.0
            "CenterSplit",   # new 0.6.1
            "FFTFilter", "ReplaceConstant", "AlignLastDiff",  # new 0.6.2
            "FFTDecomposition",  # new in 0.6.2
            "HistoricValues",  # new in 0.6.7
            "BKBandpassFilter",  # new in 0.6.8
            "Constraint",  # new in 0.6.15
            "DiffSmoother",  # new in 0.6.15
            "FIRFilter",  # new in 0.6.16
            "ShiftFirstValue",  # new in 0.6.16
            "ThetaTransformer",  # new in 0.6.16
            "ChangepointDetrend",  # new in 0.6.16
            "MeanPercentSplitter",  # new in 0.6.16
        ]

        timings = {}
        forecasts = {}
        upper_forecasts = {}
        lower_forecasts = {}
        # load the comparison source
        with open("./tests/transform_forecasts.json", "r") as file:
            loaded = json.load(file)
            for x in transforms:
                forecasts[x] = pd.DataFrame.from_dict(loaded['forecasts'][x], orient="columns")
                forecasts[x]['index'] = pd.to_datetime(forecasts[x]['index'])
                forecasts[x] = forecasts[x].set_index("index")
                upper_forecasts[x] = pd.DataFrame.from_dict(loaded['upper_forecasts'][x], orient="columns")
                upper_forecasts[x]['index'] = pd.to_datetime(upper_forecasts[x]['index'])
                upper_forecasts[x] = upper_forecasts[x].set_index("index")
                lower_forecasts[x] = pd.DataFrame.from_dict(loaded['lower_forecasts'][x], orient="columns")
                lower_forecasts[x]['index'] = pd.to_datetime(lower_forecasts[x]['index'])
                lower_forecasts[x] = lower_forecasts[x].set_index("index")
            timings = loaded['timing']

        timings2 = {}
        forecasts2 = {}
        upper_forecasts2 = {}
        lower_forecasts2 = {}

        for x in transforms:
            print(x)
            param = {} if x not in ['QuantileTransformer'] else {"n_quantiles": 100}
            start_time = timeit.default_timer()
            model = ModelPrediction(
                forecast_length=5,
                transformation_dict={
                    "fillna": "ffill",
                    "transformations": {"0": x},
                    "transformation_params": {"0": param},
                },
                model_str="LastValueNaive",
                parameter_dict={},
                frequency="infer",
                prediction_interval=0.9,
                random_seed=random_seed,
                verbose=-1,
                fail_on_forecast_nan=True,
                n_jobs=n_jobs,
                return_model=True,
            )
            model = model.fit(df)
            df_forecast = model.predict(forecast_length=5)
            forecasts2[x] = df_forecast.forecast.round(2)
            upper_forecasts2[x] = df_forecast.upper_forecast.round(2)
            lower_forecasts2[x] = df_forecast.lower_forecast.round(2)
            timings2[x] = (timeit.default_timer() - start_time)

        print(sum(timings2.values()))

        pass_probabilistic = ['FastICA']  # not reproducible in upper/lower with seed
        for x in transforms:
            with self.subTest(i=x):
                res = (forecasts2[x].round(2) == forecasts[x].round(2)).all().all()
                if x not in pass_probabilistic:
                    res_u = (upper_forecasts2[x].round(2) == upper_forecasts[x].round(2)).all().all()
                    res_l = (lower_forecasts2[x].round(2) == lower_forecasts[x].round(2)).all().all()
                else:
                    res_u = True
                    res_l = True
                self.assertTrue(
                    res,
                    f"Model '{x}' forecasts diverged from sample forecasts."
                )
                self.assertTrue(
                    res_u,
                    f"Model '{x}' upper forecasts diverged from sample forecasts."
                )
                self.assertTrue(
                    res_l,
                    f"Model '{x}' lower forecasts diverged from sample forecasts."
                )
                print(f"{res & res_u & res_l} model '{x}' ran successfully in {round(timings2[x], 4)} (bench: {round(timings[x], 4)})")

        """
        for x in transforms:
            forecasts[x].index = forecasts[x].index.strftime("%Y-%m-%d")
            forecasts[x] = forecasts[x].reset_index(drop=False).to_dict(orient="list")
            upper_forecasts[x].index = upper_forecasts[x].index.strftime("%Y-%m-%d")
            upper_forecasts[x] = upper_forecasts[x].reset_index(drop=False).to_dict(orient="list")
            lower_forecasts[x].index = lower_forecasts[x].index.strftime("%Y-%m-%d")
            lower_forecasts[x] = lower_forecasts[x].reset_index(drop=False).to_dict(orient="list")

        with open("./tests/transform_forecasts.json", "w") as file:
            json.dump(
                {
                    'forecasts': forecasts,
                    "upper_forecasts": upper_forecasts,
                    "lower_forecasts": lower_forecasts,
                    "timing": timings,
                }, file
            )
        """

    def test_sklearn(self):
        from autots import load_daily
        from autots import create_regressor
        from autots.models.sklearn import MultivariateRegression, DatepartRegression, WindowRegression

        df = load_daily(long=False).bfill().ffill()
        forecast_length = 8
        df_train = df.iloc[:-forecast_length]
        df_test = df.iloc[-forecast_length:]
        future_regressor_train, future_regressor_forecast = create_regressor(
            df_train,
            forecast_length=forecast_length,
            frequency="infer",
            drop_most_recent=0,
            scale=True,
            summarize="auto",
            backfill="bfill",
            fill_na="spline",
            holiday_countries={"US": None},  # requires holidays package
            encode_holiday_type=True,
        )

        random_seed = 300
        frequency = 'D'
        prediction_interval = 0.9
        verbose = -1
        n_jobs = 2

        params = MultivariateRegression().get_new_params()
        params = {
            'regression_model': {'model': 'LightGBM',
            'model_params': {
                'objective': 'regression',
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': 10,
                'boosting_type': 'goss',
                 'n_estimators': 250,
                'linear_tree': False
            }},
            'mean_rolling_periods': 90,
            'macd_periods': 12,
            'std_rolling_periods': 7,
            'max_rolling_periods': None,
            'min_rolling_periods': None,
            'quantile90_rolling_periods': 7,
            'quantile10_rolling_periods': 10,
            'ewm_alpha': 0.8,
            'ewm_var_alpha': None,
            'additional_lag_periods': None,
            'abs_energy': False,
            'rolling_autocorr_periods': None,
            'datepart_method': 'expanded',
            'polynomial_degree': None,
            'regression_type': None,
            'window': 3,
            'holiday': True,
            'probabilistic': False,
            'cointegration': None,
            'cointegration_lag': 1
        }
        model = MultivariateRegression(
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **params
        )
        model.fit(df_train)
        first_forecast = model.predict(future_regressor=future_regressor_forecast)
        self.assertListEqual(first_forecast.forecast.index.tolist(), df_test.index.tolist())
        model.fit_data(df)
        updated_forecast = model.predict()
        self.assertEqual(updated_forecast.forecast.shape[0], forecast_length)
        self.assertTrue(updated_forecast.forecast.index[0] > df.index[-1])

        params = WindowRegression().get_new_params()
        params = {}
        model = WindowRegression(
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **params
        )
        model.fit(df_train)
        first_forecast = model.predict(future_regressor=future_regressor_forecast)
        # first_forecast.plot_grid(df)
        self.assertListEqual(first_forecast.forecast.index.tolist(), df_test.index.tolist())
        model.fit_data(df)
        updated_forecast = model.predict()
        # updated_forecast.plot_grid(df)
        self.assertEqual(updated_forecast.forecast.shape[0], forecast_length)
        self.assertTrue(updated_forecast.forecast.index[0] > df.index[-1])


        params = {
            'regression_model': {
                'model': 'ExtraTrees',
                'model_params': {
                    'n_estimators': 500,
                    'min_samples_leaf': 1,
                    'max_depth': 20
            }},
            'datepart_method': 'simple_binarized',
            'polynomial_degree': None,
            'regression_type': None,
        }
        model = DatepartRegression(
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **params
        )
        model.fit(df_train)
        first_forecast = model.predict(future_regressor=future_regressor_forecast)
        self.assertListEqual(first_forecast.forecast.index.tolist(), df_test.index.tolist())
        model.fit_data(df)
        updated_forecast = model.predict()
        self.assertEqual(updated_forecast.forecast.shape[0], forecast_length)
        self.assertTrue(updated_forecast.forecast.index[0] > df.index[-1])

    def test_corecount(self):
        auto_count = cpu_count()
        half = int(auto_count * 0.5) if int(auto_count * 0.5) > 1 else 1
        self.assertEqual(half, set_n_jobs(0.5))
        self.assertGreater(auto_count, 0)
        self.assertGreater(set_n_jobs(-4), 0)
        self.assertEqual(set_n_jobs(8.0), 8)
        self.assertIsInstance(set_n_jobs("auto"), int)
        self.assertEqual(set_n_jobs("auto"), set_n_jobs(-1))
