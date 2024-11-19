# -*- coding: utf-8 -*-
"""Cassandra testing."""
import unittest
import random
import numpy as np
import pandas as pd
from autots.models.cassandra import Cassandra


class CassandraTest(unittest.TestCase):
    def test_model(self):
        print("Starting Cassandra model tests")
        from autots import load_daily
        import matplotlib.pyplot as plt

        categorical_groups = {
            "wiki_United_States": "country",
            "wiki_Germany": "country",
            "wiki_Jesus": "holiday",
            "wiki_Michael_Jackson": "person",
            "wiki_Easter": "holiday",
            "wiki_Christmas": "holiday",
            "wiki_Chinese_New_Year": "holiday",
            "wiki_Thanksgiving": "holiday",
            "wiki_Elizabeth_II": "person",
            "wiki_William_Shakespeare": "person",
            "wiki_George_Washington": "person",
            "wiki_Cleopatra": "person",
        }
        holiday_countries = {
            "wiki_Elizabeth_II": "uk",
            "wiki_United_States": "us",
            "wiki_Germany": "de",
        }
        df_daily = load_daily(long=False)
        # so it expects these first in the column order for the tests
        cols = ['wiki_United_States', 'wiki_Germany'] + [col for col in df_daily.columns if col not in ['wiki_United_States', 'wiki_Germany']]
        df_daily = df_daily[cols]
        forecast_length = 180
        include_history = True
        df_train = df_daily[:-forecast_length].iloc[:, 1:]
        df_test = df_daily[-forecast_length:].iloc[:, 1:]
        fake_regr = df_daily[:-forecast_length].iloc[:, 0:1]
        fake_regr_fcst = (
            df_daily.iloc[:, 0:1]
            if include_history
            else df_daily[-forecast_length:].iloc[:, 0:1]
        )
        flag_regressor = pd.DataFrame(1, index=fake_regr.index, columns=["flag_test"])
        flag_regressor_fcst = pd.DataFrame(
            1, index=fake_regr_fcst.index, columns=["flag_test"]
        )
        regr_per_series = {
            str(df_train.columns[0]): pd.DataFrame(
                np.random.normal(size=(len(df_train), 1)), index=df_train.index
            )
        }
        regr_per_series_fcst = {
            str(df_train.columns[0]): pd.DataFrame(
                np.random.normal(size=(forecast_length, 1)), index=df_test.index
            )
        }
        constraint = {
            "constraints": [{
                    "constraint_method": "last_window",
                    "constraint_value": 0.5,
                    "constraint_direction": "upper",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
                {
                    "constraint_method": "last_window",
                    "constraint_value": -0.5,
                    "constraint_direction": "lower",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
            ]
        }
        past_impacts = pd.DataFrame(0, index=df_train.index, columns=df_train.columns)
        past_impacts.iloc[-10:, 0] = np.geomspace(1, 10)[0:10] / 100
        past_impacts.iloc[-30:, -1] = np.linspace(1, 10)[0:30] / -100
        future_impacts = pd.DataFrame(0, index=df_test.index, columns=df_test.columns)
        future_impacts.iloc[0:10, 0] = (np.linspace(1, 10)[0:10] + 10) / 100

        c_params = Cassandra().get_new_params()
        c_params = {
            "preprocessing_transformation": {
                "fillna": "ffill",
                "transformations": {0: "Slice", 1: "AlignLastValue"},
                "transformation_params": {
                    0: {"method": 0.5},
                    1: {
                        "rows": 1,
                        "lag": 1,
                        "method": "additive",
                        "strength": 1.0,
                        "first_value_only": False,
                    },
                },
            },
            "scaling": "BaseScaler",
            "seasonalities": ["weekdayofmonth", "common_fourier"],
            "ar_lags": None,
            "ar_interaction_seasonality": None,
            "anomaly_detector_params": {
                "method": "rolling_zscore",
                "transform_dict": {
                    "fillna": None,
                    "transformations": {"0": "ClipOutliers"},
                    "transformation_params": {
                        "0": {"method": "clip", "std_threshold": 6}
                    },
                },
                "method_params": {
                    "distribution": "chi2",
                    "alpha": 0.05,
                    "rolling_periods": 200,
                    "center": False,
                },
                "fillna": "rolling_mean_24",
            },
            "anomaly_intervention": "remove",
            "holiday_detector_params": None,
            "holiday_countries_used": True,
            "multivariate_feature": None,
            "multivariate_transformation": {
                "fillna": "ffill",
                "transformations": {0: "QuantileTransformer", 1: "QuantileTransformer"},
                "transformation_params": {
                    0: {"output_distribution": "uniform", "n_quantiles": 1000},
                    1: {"output_distribution": "uniform", "n_quantiles": 1000},
                },
            },
            "regressor_transformation": {
                "fillna": "rolling_mean_24",
                "transformations": {0: "QuantileTransformer"},
                "transformation_params": {
                    0: {"output_distribution": "uniform", "n_quantiles": 60}
                },
            },
            "regressors_used": True,
            "linear_model": {"model": "lstsq", "lambda": 10, "recency_weighting": None},
            "randomwalk_n": 10,
            "trend_window": 3,
            "trend_standin": "random_normal",
            "trend_anomaly_detector_params": {
                "method": "IQR",
                "transform_dict": {
                    "fillna": "rolling_mean_24",
                    "transformations": {0: "AlignLastValue", 1: "RobustScaler"},
                    "transformation_params": {
                        0: {
                            "rows": 1,
                            "lag": 1,
                            "method": "additive",
                            "strength": 1.0,
                            "first_value_only": False,
                        },
                        1: {},
                    },
                },
                "method_params": {"iqr_threshold": 1.5, "iqr_quantiles": [0.25, 0.75]},
                "fillna": "rolling_mean_24",
            },
            "trend_transformation": {
                "fillna": "ffill",
                "transformations": {0: "SeasonalDifference"},
                "transformation_params": {0: {"lag_1": 7, "method": "LastValue"}},
            },
            "trend_model": {
                "Model": "MetricMotif",
                "ModelParameters": {
                    "window": 30,
                    "point_method": "weighted_mean",
                    "distance_metric": "mqae",
                    "k": 3,
                    "comparison_transformation": {
                        "fillna": "rolling_mean",
                        "transformations": {0: "KalmanSmoothing"},
                        "transformation_params": {
                            0: {
                                "state_transition": [[1]],
                                "process_noise": [[0.064]],
                                "observation_model": [[2]],
                                "observation_noise": 10.0,
                            }
                        },
                    },
                    "combination_transformation": {
                        "fillna": "rolling_mean_24",
                        "transformations": {0: "KalmanSmoothing"},
                        "transformation_params": {
                            0: {
                                "state_transition": [[1]],
                                "process_noise": [[0.246]],
                                "observation_model": [[1]],
                                "observation_noise": 0.5,
                            }
                        },
                    },
                },
            },
            "trend_phi": None,
        }

        mod = Cassandra(
            n_jobs=1,
            **c_params,
            constraint=constraint,
            holiday_countries=holiday_countries,
            max_multicolinearity=0.0001,
        )
        mod.fit(
            df_train,
            categorical_groups=categorical_groups,
            future_regressor=fake_regr,
            regressor_per_series=regr_per_series,
            past_impacts=past_impacts,
            flag_regressors=flag_regressor,
        )
        pred = mod.predict(
            forecast_length=forecast_length,
            include_history=include_history,
            future_regressor=fake_regr_fcst,
            regressor_per_series=regr_per_series_fcst,
            future_impacts=future_impacts,
            flag_regressors=flag_regressor_fcst,
        )
        result = pred.forecast
        series = random.choice(mod.column_names)
        mod.regressors_used
        mod.holiday_countries_used
        start_date = "2019-07-01"
        if False:
            with plt.style.context("seaborn-white"):
                mod.plot_forecast(
                    pred,
                    actuals=df_daily if include_history else df_test,
                    series=series,
                    start_date=start_date,
                )
                plt.show()
                mod.plot_components(
                    pred, series=series, to_origin_space=True, start_date=start_date
                )
                mod.plot_trend(series=series, vline=df_test.index[0], start_date=start_date)
        pred.evaluate(
            df_daily.reindex(result.index)[df_train.columns]
            if include_history
            else df_test[df_train.columns]
        )
        dates = df_daily.index.union(
            mod.create_forecast_index(forecast_length, last_date=df_daily.index[-1])
        )
        regr_ps = {
            "wiki_Germany": regr_per_series_fcst["wiki_Germany"].reindex(
                dates, fill_value=0
            )
        }
        mod.past_impacts_intervention = None
        pred2 = mod.predict(
            forecast_length=forecast_length,
            include_history=True,
            new_df=df_daily[df_train.columns],
            flag_regressors=flag_regressor_fcst.reindex(dates, fill_value=0),
            future_regressor=fake_regr_fcst.reindex(dates, fill_value=0),
            regressor_per_series=regr_ps,
        )
        mod.plot_forecast(pred2, actuals=df_daily, series=series, start_date=start_date)
        mod.return_components()
        print(pred.avg_metrics.round(1))

        self.assertFalse(pred.forecast.isna().all().all())

    def test_another_version(self):
        from autots import load_daily

        df = load_daily(long=False)
        params = {
            'frequency': 'D',
          'preprocessing_transformation': {'fillna': 'ffill',
           'transformations': {'0': 'AlignLastValue', '1': 'ClipOutliers'},
           'transformation_params': {'0': {'rows': 1,
             'lag': 1,
             'method': 'additive',
             'strength': 1.0,
             'first_value_only': False},
            '1': {'method': 'clip', 'std_threshold': 4, 'fillna': None}}},
          'scaling': 'BaseScaler',
          'past_impacts_intervention': 'remove',
          'seasonalities': ['month', 'dayofweek', 'weekdayofmonth'],
          'ar_lags': None,
          'ar_interaction_seasonality': None,
          'anomaly_detector_params': {'method': 'zscore',
           'transform_dict': {'transformations': {'0': 'DatepartRegression'},
            'transformation_params': {'0': {'datepart_method': 'simple_3',
              'regression_model': {'model': 'ElasticNet', 'model_params': {}}}}},
           'method_params': {'distribution': 'gamma', 'alpha': 0.05},
           'fillna': 'rolling_mean_24'},
          'anomaly_intervention': None,
          'holiday_detector_params': None,
          'holiday_countries_used': True,
          'multivariate_feature': None,
          'multivariate_transformation': None,
          'regressor_transformation': None,
          'regressors_used': False,
          'linear_model': {'model': 'l1_positive',
           'recency_weighting': None,
           'maxiter': 15000},
          'randomwalk_n': 10,
          'trend_window': None,
          'trend_standin': None,
          'trend_anomaly_detector_params': {'method': 'LOF',
           'method_params': {'contamination': 'auto',
            'n_neighbors': 10,
            'metric': 'minkowski'},
           'fillna': 'mean',
           'transform_dict': None},
          'trend_transformation2': {'fillna': 'zero',
           'transformations': {'0': 'HPFilter', '1': 'ClipOutliers'},
           'transformation_params': {'0': {'part': 'trend', 'lamb': 6.25},
            '1': {'method': 'clip', 'std_threshold': 2, 'fillna': None}}},
          'trend_model2': {'Model': 'WindowRegression',
           'ModelParameters': {'window_size': 12,
            'input_dim': 'univariate',
            'output_dim': '1step',
            'normalize_window': False,
            'max_windows': 8000,
            'regression_type': None,
            'regression_model': {'model': 'ExtraTrees',
             'model_params': {'n_estimators': 100,
              'min_samples_leaf': 1,
              'max_depth': 20}}}},
          'trend_transformation': {'fillna': 'zero',
           'transformations': {'0': 'HPFilter', '1': 'ClipOutliers'},
           'transformation_params': {'0': {'part': 'trend', 'lamb': 6.25},
            '1': {'method': 'clip', 'std_threshold': 2, 'fillna': None}}},
          'trend_model': {'Model': 'SeasonalityMotif',
           'ModelParameters': {'window': 10,
            'point_method': 'weighted_mean',
            'distance_metric': 'mqae',
            'k': 10,
            'datepart_method': 'expanded_binarized'}},
          'trend_phi': None
        }
        mod = Cassandra(**params)
        
        mod.fit(df)
        pred = mod.predict(forecast_length=10)

        self.assertFalse(pred.forecast.isna().all().all())
        self.assertEqual(pred.forecast.shape[0], 10)
