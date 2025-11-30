"""Starting templates for models."""

import pandas as pd

general_template_dict = {
    "1": {
        "Model": "ARIMA",
        "ModelParameters": '{"p": 4, "d": 0, "q": 12, "regression_type": null}',
        "TransformationParameters": '{"fillna": "cubic", "transformations": {"0": "bkfilter"}, "transformation_params": {"0": {}}}',
        "Ensemble": 0,
    },
    "2": {
        "Model": "AverageValueNaive",
        "ModelParameters": "{\"method\": \"Mean\"}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"DifferencedTransformer\", \"1\": \"SinTrend\"}, \"transformation_params\": {\"0\": {}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "3": {
        "Model": "AverageValueNaive",
        "ModelParameters": "{\"method\": \"Mean\"}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "4": {
        "Model": "AverageValueNaive",
        "ModelParameters": "{\"method\": \"Mean\"}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"Round\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"lag_1\": 7, \"method\": \"Mean\"}, \"1\": {\"model\": \"middle\", \"decimals\": 2, \"on_transform\": true, \"on_inverse\": false}, \"2\": {\"model\": \"GLS\"}}}",
        "Ensemble": 0,
    },
    "5": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"DecisionTree\", \"model_params\": {\"max_depth\": 3, \"min_samples_split\": 2}}, \"datepart_method\": \"recurring\", \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "6": {
        "Model": "ETS",
        "ModelParameters": "{\"damped_trend\": false, \"trend\": \"additive\", \"seasonal\": null, \"seasonal_periods\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "7": {
        "Model": "GLM",
        "ModelParameters": "{\"family\": \"Binomial\", \"constant\": false, \"regression_type\": \"datepart\"}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 4, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "8": {
        "Model": "GLS",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"median\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"RobustScaler\", \"3\": \"Round\", \"4\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3.5, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}, \"3\": {\"model\": \"middle\", \"decimals\": 2, \"on_transform\": true, \"on_inverse\": true}, \"4\": {}}}",
        "Ensemble": 0,
    },
    "9": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"DeepAR\", \"epochs\": 150, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"PowerTransformer\", \"1\": \"DifferencedTransformer\", \"2\": \"bkfilter\"}, \"transformation_params\": {\"0\": {}, \"1\": {}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "10": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"NPTS\", \"epochs\": 20, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"Detrend\", \"1\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {\"model\": \"Linear\"}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "11": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"WaveNet\", \"epochs\": 40, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "12": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"Transformer\", \"epochs\": 40, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "13": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"SFF\", \"epochs\": 40, \"learning_rate\": 0.01, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "14": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"PositiveShift\", \"1\": \"SinTrend\", \"2\": \"bkfilter\"}, \"transformation_params\": {\"0\": {}, \"1\": {}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "15": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"SinTrend\"}, \"transformation_params\": {\"0\": {\"lag_1\": 7, \"method\": \"LastValue\"}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "16": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 1, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "17": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"LastValue\", \"lag_1\": 2, \"lag_2\": 1}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"QuantileTransformer\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"lag_1\": 12, \"method\": \"Median\"}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"model\": \"GLS\"}}}",
        "Ensemble": 0,
    },
    "18": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"LastValue\", \"lag_1\": 7, \"lag_2\": 2}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"ClipOutliers\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"1\": {\"method\": \"clip\", \"std_threshold\": 2, \"fillna\": null}}}",
        "Ensemble": 0,
    },
    "19": {
        "Model": "VAR",
        "ModelParameters": "{\"regression_type\": null, \"maxlags\": 5, \"ic\": \"fpe\"}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "20": {
        "Model": "ConstantNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"PowerTransformer\", \"1\": \"QuantileTransformer\", \"2\": \"SeasonalDifference\"}, \"transformation_params\": {\"0\": {}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"lag_1\": 7, \"method\": \"LastValue\"}}}",
        "Ensemble": 0,
    },
    # Gen 2
    "21": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"ExtraTrees\", \"model_params\": {\"n_estimators\": 500, \"min_samples_leaf\": 1, \"max_depth\": 10}}, \"datepart_method\": \"expanded\", \"polynomial_degree\": null, \"regression_type\": \"User\"}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"MaxAbsScaler\", \"1\": \"MinMaxScaler\"}, \"transformation_params\": {\"0\": {}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "22": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"lastvalue\", \"lag_1\": 364, \"lag_2\": 28}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"MaxAbsScaler\", \"2\": \"Round\"}, \"transformation_params\": {\"0\": {\"lag_1\": 7, \"method\": \"LastValue\"}, \"1\": {}, \"2\": {\"decimals\": 0, \"on_transform\": false, \"on_inverse\": true}}}",
        "Ensemble": 0,
    },
    "23": {
        "Model": "ETS",
        "ModelParameters": "{\"damped_trend\": false, \"trend\": null, \"seasonal\": \"additive\", \"seasonal_periods\": 28}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"MaxAbsScaler\", \"1\": \"Slice\"}, \"transformation_params\": {\"0\": {}, \"1\": {\"method\": 100}}}",
        "Ensemble": 0,
    },
    "24": {
        "Model": "ARDL",
        "ModelParameters": "{\"lags\": 4, \"trend\": \"n\", \"order\": 1, \"regression_type\": \"holiday\"}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"IntermittentOccurrence\"}, \"transformation_params\": {\"0\": {\"center\": \"mean\"}}}",
        "Ensemble": 0,
    },
    "25": {
        "Model": "MultivariateMotif",
        "ModelParameters": "{\"window\": 10, \"point_method\": \"median\", \"distance_metric\": \"mahalanobis\", \"k\": 20, \"max_windows\": null}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"QuantileTransformer\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {\"output_distribution\": \"normal\", \"n_quantiles\": 1000}, \"2\": {\"model\": \"Linear\", \"phi\": 1, \"window\": null}}}",
        "Ensemble": 0,
    },
    "26": {
        "Model": "MultivariateMotif",
        "ModelParameters": "{\"window\": 10, \"point_method\": \"median\", \"distance_metric\": \"sqeuclidean\", \"k\": 10, \"max_windows\": 10000}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean\", \"transformations\": {\"0\": \"Detrend\", \"1\": \"bkfilter\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"model\": \"GLS\", \"phi\": 1, \"window\": null}, \"1\": {}, \"2\": {\"model\": \"Linear\", \"phi\": 1, \"window\": null}}}",
        "Ensemble": 0,
    },
    "27": {
        "Model": "UnivariateMotif",
        "ModelParameters": "{\"window\": 60, \"point_method\": \"median\", \"distance_metric\": \"canberra\", \"k\": 10, \"max_windows\": 10000}",
        "TransformationParameters": "{\"fillna\": \"KNNImputer\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"SeasonalDifference\", \"2\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 100}, \"1\": {\"lag_1\": 7, \"method\": \"Mean\"}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "28": {
        "Model": "UnivariateMotif",
        "ModelParameters": "{\"window\": 14, \"point_method\": \"median\", \"distance_metric\": \"minkowski\", \"k\": 5, \"max_windows\": 10000}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"SeasonalDifference\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {\"lag_1\": 7, \"method\": \"LastValue\"}}}",
        "Ensemble": 0,
    },
    "29": {
        "Model": "SectionalMotif",
        "ModelParameters": "{\"window\": 10, \"point_method\": \"weighted_mean\", \"distance_metric\": \"sokalmichener\", \"include_differenced\": true, \"k\": 20, \"stride_size\": 1, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": null}, \"transformation_params\": {\"0\": {}}}",
        "Ensemble": 0,
    },
    "30": {
        "Model": "SectionalMotif",
        "ModelParameters": "{\"window\": 5, \"point_method\": \"midhinge\", \"distance_metric\": \"canberra\", \"include_differenced\": false, \"k\": 10, \"stride_size\": 1, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"QuantileTransformer\", \"2\": \"RobustScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "31": {
        "Model": "FBProphet",
        "ModelParameters": "{\"holiday\": true, \"regression_type\": null, \"growth\": \"linear\", \"n_changepoints\": 25, \"changepoint_prior_scale\": 30, \"seasonality_mode\": \"multiplicative\", \"changepoint_range\": 0.9, \"seasonality_prior_scale\": 10.0, \"holidays_prior_scale\": 10.0}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"Slice\"}, \"transformation_params\": {\"0\": {\"method\": 0.5}}}",
        "Ensemble": 0,
    },
    "32": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"lastvalue\", \"lag_1\": 364, \"lag_2\": 30}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"MaxAbsScaler\", \"2\": \"Round\"}, \"transformation_params\": {\"0\": {\"lag_1\": 7, \"method\": \"LastValue\"}, \"1\": {}, \"2\": {\"decimals\": 0, \"on_transform\": false, \"on_inverse\": true}}}",
        "Ensemble": 0,
    },
    "33": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"RandomForest\", \"model_params\": {\"n_estimators\": 100, \"min_samples_leaf\": 2, \"bootstrap\": true}}, \"datepart_method\": \"expanded\", \"polynomial_degree\": null, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"Detrend\", \"2\": \"RobustScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {\"model\": \"Linear\", \"phi\": 1, \"window\": null}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "34": {
        'Model': 'NVAR',
        'ModelParameters': '{"k": 1, "ridge_param": 0.002, "warmup_pts": 1, "seed_pts": 1, "seed_weighted": null, "batch_size": 5, "batch_method": "input_order"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "MinMaxScaler"}, "transformation_params": {"0": {}}}',
        "Ensemble": 0,
    },
    "35": {
        'Model': 'Theta',
        'ModelParameters': '{"deseasonalize": true, "difference": false, "use_test": false, "method": "auto", "period": null, "theta": 2.5, "use_mle": true}',
        'TransformationParameters': '{"fillna": "mean", "transformations": {"0": "Detrend"}, "transformation_params": {"0": {"model": "Linear", "phi": 1, "window": 90}}}',
        "Ensemble": 0,
    },
    "36": {
        'Model': 'RollingRegression',
        'ModelParameters': '{"regression_model": {"model": "ExtraTrees", "model_params": {"n_estimators": 100, "min_samples_leaf": 1, "max_depth": 10}}, "holiday": false, "mean_rolling_periods": null, "macd_periods": null, "std_rolling_periods": null, "max_rolling_periods": null, "min_rolling_periods": 7, "ewm_var_alpha": null, "quantile90_rolling_periods": null, "quantile10_rolling_periods": null, "ewm_alpha": null, "additional_lag_periods": 95, "abs_energy": true, "rolling_autocorr_periods": null, "add_date_part": "expanded", "polynomial_degree": null, "x_transform": null, "regression_type": "User"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "MaxAbsScaler"}, "transformation_params": {"0": {}}}',
        'Ensemble': 0,
    },
    "37": {
        'Model': 'MotifSimulation',
        'ModelParameters': '{"phrase_len": 10, "comparison": "magnitude_pct_change_sign", "shared": false, "distance_metric": "sokalmichener", "max_motifs": 0.2, "recency_weighting": 0.01, "cutoff_minimum": 20, "point_method": "median"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "MaxAbsScaler", "1": "Detrend"}, "transformation_params": {"0": {}, "1": {"model": "Linear", "phi": 1, "window": null}}}',
        'Ensemble': 0,
    },
    "38": {
        'Model': 'DynamicFactor',
        'ModelParameters': '{"k_factors": 0, "factor_order": 0, "regression_type": "User"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "MaxAbsScaler", "1": "EWMAFilter", "2": "QuantileTransformer"}, "transformation_params": {"0": {}, "1": {"span": 3}, "2": {"output_distribution": "uniform", "n_quantiles": 1000}}}',
        'Ensemble': 0,
    },
    "39": {
        'Model': 'RollingRegression',
        'ModelParameters': '{"regression_model": {"model": "ExtraTrees", "model_params": {"n_estimators": 100, "min_samples_leaf": 1, "max_depth": 10}}, "holiday": false, "mean_rolling_periods": null, "macd_periods": null, "std_rolling_periods": null, "max_rolling_periods": 420, "min_rolling_periods": 7, "ewm_var_alpha": null, "quantile90_rolling_periods": null, "quantile10_rolling_periods": null, "ewm_alpha": null, "additional_lag_periods": 363, "abs_energy": false, "rolling_autocorr_periods": null, "add_date_part": "expanded", "polynomial_degree": null, "x_transform": null, "regression_type": null}',
        'TransformationParameters': '{"fillna": "ffill_mean_biased", "transformations": {"0": "RobustScaler"}, "transformation_params": {"0": {}}}',
        'Ensemble': 0,
    },
    "40": {
        'Model': 'ARCH',
        'ModelParameters': '{"mean": "Zero", "lags": 1, "vol": "GARCH", "p": 4, "o": 1, "q": 2, "power": 1.5, "dist": "studentst", "rescale": true, "simulations": 1000, "maxiter": 200, "regression_type": null}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "SeasonalDifference"}, "transformation_params": {"0": {"lag_1": 7, "method": "LastValue"}}}',
        'Ensemble': 0,
    },
    "41": {
        'Model': 'Cassandra',
        'ModelParameters': '''{
            "preprocessing_transformation": {"fillna": "ffill", "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 3.5, "fillna": null}}}, 
            "scaling": {"fillna": null, "transformations": {"0": "QuantileTransformer"}, "transformation_params": {"0": {"output_distribution": "uniform", "n_quantiles": 308}}},
            "past_impacts_intervention": null,
            "seasonalities": ["month", "dayofweek", "weekdayofmonth"],
            "ar_lags": null, "ar_interaction_seasonality": null,
            "anomaly_detector_params": {"method": "zscore", "transform_dict": {"transformations": {"0": "DatepartRegression"}, "transformation_params": {"0": {"datepart_method": "simple_3", "regression_model": {"model": "ElasticNet", "model_params": {}}}}}, "method_params": {"distribution": "gamma", "alpha": 0.05}, "fillna": "rolling_mean_24"}, "anomaly_intervention": null,
            "holiday_detector_params": {"threshold": 0.9, "splash_threshold": null, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": false, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": false, "anomaly_detector_params": {"method": "IQR", "transform_dict": {"fillna": "rolling_mean_24", "transformations": {"0": "bkfilter", "1": "PCA"}, "transformation_params": {"0": {}, "1": {"whiten": true}}}, "method_params": {"iqr_threshold": 2.0, "iqr_quantiles": [0.25, 0.75]}, "fillna": "linear"}, "remove_excess_anomalies": false, "impact": null, "regression_params": null, "output": "multivariate"},
            "holiday_countries_used": false,
            "multivariate_feature": null, "multivariate_transformation": null, "regressor_transformation": null,
            "regressors_used": false, "linear_model": {"model": "linalg_solve", "lambda": 0.1, "recency_weighting": 0.1}, "randomwalk_n": 10, "trend_window": 90,
            "trend_standin": null,
            "trend_anomaly_detector_params": {"method": "zscore", "transform_dict": {"transformations": {"0": "DifferencedTransformer"}, "transformation_params": {"0": {}}}, "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "rolling_mean_24"},
            "trend_transformation": {"fillna": "ffill_mean_biased", "transformations": {"0": "bkfilter"}, "transformation_params": {"0": {}}},
            "trend_model": {"Model": "MetricMotif", "ModelParameters": {"window": 7, "point_method": "mean", "distance_metric": "mae", "k": 10, "comparison_transformation": {"fillna": "quadratic", "transformations": {"0": null}, "transformation_params": {"0": {}}}, "combination_transformation": {"fillna": "ffill", "transformations": {"0": "AlignLastValue"}, "transformation_params": {"0": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false}}}}},
            "trend_phi": null
        }''',
        'TransformationParameters': '{"fillna": null, "transformations": {}, "transformation_params": {}}',
        'Ensemble': 0,
    },
    "42": {
        'Model': 'SeasonalityMotif',
        'ModelParameters': '{"window": 5, "point_method": "weighted_mean", "distance_metric": "mae", "k": 10, "datepart_method": "common_fourier"}',
        'TransformationParameters': '{"fillna": "nearest", "transformations": {"0": "AlignLastValue"}, "transformation_params": {"0": {"rows": 1, "lag": 1, "method": "multiplicative", "strength": 1.0, "first_value_only": false}}}',
        'Ensemble': 0,
    },
    "43": {
        'Model': 'TiDE',
        'ModelParameters': '''{
            "learning_rate": 0.000999999, "transform": true,
            "layer_norm": false, "holiday": false, "dropout_rate": 0.5,
            "batch_size": 32, "hidden_size": 256, "num_layers": 1,
            "hist_len": 21, "decoder_output_dim": 8, "final_decoder_hidden": 64,
            "num_split": 4, "min_num_epochs": 5, "train_epochs": 40,
            "epoch_len": null, "permute": true, "normalize": false
        }''',
        'TransformationParameters': '''
        {"fillna": "ffill", "transformations": {"0": "AlignLastValue", "1": "Log", "2": "AlignLastValue", "3": "MinMaxScaler"}, "transformation_params": {"0": {"rows": 4, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false}, "1": {}, "2": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false}, "3": {}}}
        ''',
        'Ensemble': 0,
    },
    "44": {
        'Model': 'Cassandra',
        'ModelParameters': '''{"preprocessing_transformation": {"fillna": "ffill", "transformations": {"0": "SeasonalDifference", "1": "SeasonalDifference"}, "transformation_params": {"0": {"lag_1": 12, "method": "Median"}, "1": {"lag_1": 7, "method": "LastValue"}}}, "scaling": "BaseScaler", "past_impacts_intervention": null, "seasonalities": ["dayofmonthofyear", "weekend"], "ar_lags": null, "ar_interaction_seasonality": null, "anomaly_detector_params": {"method": "zscore", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "rolling_mean_24", "transform_dict": {"transformations": {"0": "DatepartRegression"}, "transformation_params": {"0": {"datepart_method": "simple_3", "regression_model": {"model": "DecisionTree", "model_params": {"max_depth": null, "min_samples_split": 0.1}}}}}}, "anomaly_intervention": "remove", "holiday_detector_params": {"threshold": 0.8, "splash_threshold": 0.85, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": false, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": false, "anomaly_detector_params": {"method": "mad", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "fake_date", "transform_dict": {"fillna": null, "transformations": {"0": "EWMAFilter"}, "transformation_params": {"0": {"span": 7}}}}, "remove_excess_anomalies": false, "impact": null, "regression_params": null, "output": "multivariate"}, "holiday_countries_used": false, "multivariate_feature": null, "multivariate_transformation": null, "regressor_transformation": null, "regressors_used": false, "linear_model": {"model": "lstsq", "lambda": 1, "recency_weighting": null}, "randomwalk_n": 10, "trend_window": 15, "trend_standin": "rolling_trend", "trend_anomaly_detector_params": {"method": "nonparametric", "method_params": {"p": null, "z_init": 1.5, "z_limit": 12, "z_step": 0.25, "inverse": false, "max_contamination": 0.25, "mean_weight": 25, "sd_weight": 25, "anomaly_count_weight": 1.0}, "fillna": "ffill", "transform_dict": {"fillna": "time", "transformations": {"0": "MinMaxScaler"}, "transformation_params": {"0": {}}}}, "trend_transformation": {"fillna": "ffill", "transformations": {"0": "AlignLastDiff", "1": "HPFilter"}, "transformation_params": {"0": {"rows": 1, "displacement_rows": 1, "quantile": 0.7, "decay_span": 3}, "1": {"part": "trend", "lamb": 129600}}}, "trend_model": {"Model": "LastValueNaive", "ModelParameters": {}}, "trend_phi": null}''',
        'TransformationParameters': '{"fillna": "linear", "transformations": {"0": "Round"}, "transformation_params": {"0": {"decimals": 0, "on_transform": true, "on_inverse": true}}}',
        'Ensemble': 0,
    },
    "45": {
        'Model': 'Cassandra',
        'ModelParameters': '''{"preprocessing_transformation": {"fillna": "mean", "transformations": {"0": "ClipOutliers", "1": "bkfilter"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 3.5, "fillna": null}, "1": {}}}, "scaling": "BaseScaler", "past_impacts_intervention": null, "seasonalities": [7, 365.25], "ar_lags": [7], "ar_interaction_seasonality": null, "anomaly_detector_params": null, "anomaly_intervention": null, "holiday_detector_params": {"threshold": 0.9, "splash_threshold": null, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": false, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": true, "anomaly_detector_params": {"method": "mad", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "rolling_mean_24", "transform_dict": {"fillna": null, "transformations": {"0": "EWMAFilter"}, "transformation_params": {"0": {"span": 7}}}}, "remove_excess_anomalies": false, "impact": null, "regression_params": null, "output": "multivariate"}, "holiday_countries_used": true, "multivariate_feature": null, "multivariate_transformation": null, "regressor_transformation": null, "regressors_used": false, "linear_model": {"model": "linalg_solve", "lambda": 1, "recency_weighting": 0.1}, "randomwalk_n": null, "trend_window": 365, "trend_standin": "rolling_trend", "trend_anomaly_detector_params": {"method": "mad", "method_params": {"distribution": "uniform", "alpha": 0.05}, "fillna": "rolling_mean_24", "transform_dict": {"fillna": null, "transformations": {"0": "EWMAFilter"}, "transformation_params": {"0": {"span": 7}}}}, "trend_transformation": {"fillna": "nearest", "transformations": {"0": "StandardScaler", "1": "AnomalyRemoval"}, "transformation_params": {"0": {}, "1": {"method": "IQR", "transform_dict": {"fillna": null, "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 6}}}, "method_params": {"iqr_threshold": 2.5, "iqr_quantiles": [0.25, 0.75]}, "fillna": "ffill"}}}, "trend_model": {"Model": "SectionalMotif", "ModelParameters": {"window": 50, "point_method": "mean", "distance_metric": "dice", "include_differenced": true, "k": 5, "stride_size": 1, "regression_type": null}}, "trend_phi": null}''',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "EWMAFilter", "1": "LevelShiftTransformer", "2": "StandardScaler", "3": "DatepartRegression"}, "transformation_params": {"0": {"span": 7}, "1": {"window_size": 7, "alpha": 2.0, "grouping_forward_limit": 2, "max_level_shifts": 5, "alignment": "average"}, "2": {}, "3": {"regression_model": {"model": "ElasticNet", "model_params": {}}, "datepart_method": "recurring", "polynomial_degree": null, "transform_dict": {"fillna": null, "transformations": {"0": "ScipyFilter"}, "transformation_params": {"0": {"method": "savgol_filter", "method_args": {"window_length": 31, "polyorder": 3, "deriv": 0, "mode": "interp"}}}}, "holiday_countries_used": false}}}',
        'Ensemble': 0,
    },
    "46": {  # optimized on M5, 58.5 SMAPE
        'Model': 'NeuralForecast',
        'ModelParameters': '''{"model": "MLP", "scaler_type": "minmax", "loss": "MQLoss", "learning_rate": 0.001, "max_steps": 100, "input_size": 28, "model_args": {"num_layers": 1, "hidden_size": 2560}, "regression_type": null}''',
        'TransformationParameters': '''{"fillna": "ffill", "transformations": {"0": "ClipOutliers", "1": "QuantileTransformer", "2": "SeasonalDifference", "3": "RobustScaler", "4": "ClipOutliers", "5": "MaxAbsScaler"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 3.5, "fillna": null}, "1": {"output_distribution": "normal", "n_quantiles": 100}, "2": {"lag_1": 7, "method": "Mean"}, "3": {}, "4": {"method": "clip", "std_threshold": 4, "fillna": null}, "5": {}}}''',
        'Ensemble': 0,
    },
    "47": {  # from production_example, mosaic most common, 2024-02-21
        'Model': 'Cassandra',
        'ModelParameters': '{"preprocessing_transformation": {"fillna": "ffill", "transformations": {"0": "RobustScaler", "1": "SeasonalDifference"}, "transformation_params": {"0": {}, "1": {"lag_1": 7, "method": "LastValue"}}}, "scaling": {"fillna": null, "transformations": {"0": "StandardScaler"}, "transformation_params": {"0": {}}}, "past_impacts_intervention": null, "seasonalities": ["simple_binarized"], "ar_lags": null, "ar_interaction_seasonality": null, "anomaly_detector_params": null, "anomaly_intervention": null, "holiday_detector_params": {"threshold": 1.0, "splash_threshold": null, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": true, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": false, "anomaly_detector_params": {"method": "zscore", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "mean", "transform_dict": null, "isolated_only": false}, "remove_excess_anomalies": false, "impact": null, "regression_params": null, "output": "multivariate"}, "holiday_countries_used": false, "multivariate_feature": null, "multivariate_transformation": null, "regressor_transformation": {"fillna": "nearest", "transformations": {"0": "AlignLastDiff"}, "transformation_params": {"0": {"rows": 7, "displacement_rows": 7, "quantile": 1.0, "decay_span": 2}}}, "regressors_used": true, "linear_model": {"model": "lstsq", "lambda": null, "recency_weighting": null}, "randomwalk_n": 10, "trend_window": 3, "trend_standin": null, "trend_anomaly_detector_params": null, "trend_transformation": {"fillna": "pchip", "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 3.5, "fillna": null}}}, "trend_model": {"Model": "ARDL", "ModelParameters": {"lags": 1, "trend": "c", "order": 0, "causal": false, "regression_type": "holiday"}}, "trend_phi": null}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "MaxAbsScaler", "1": "FFTDecomposition", "2": "bkfilter"}, "transformation_params": {"0": {}, "1": {"n_harmonics": 10, "detrend": "linear"}, "2": {}}}',
        'Ensemble': 0,
    },
    "48": {  # optimized 200 minutes on initial model import on load_daily
        "Model": "DMD",
        'ModelParameters': '{"rank": 10, "alpha": 1, "amplitude_threshold": null, "eigenvalue_threshold": null}',
        "TransformationParameters": '''{"fillna": "linear", "transformations": {"0": "HistoricValues", "1": "AnomalyRemoval", "2": "SeasonalDifference", "3": "AnomalyRemoval"},"transformation_params": {"0": {"window": 10}, "1": {"method": "zscore", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "ffill", "transform_dict": {"fillna": null, "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 6}}}, "isolated_only": false}, "2": {"lag_1": 7, "method": "Mean"}, "3": {"method": "zscore", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "fake_date", "transform_dict": {"transformations": {"0": "DifferencedTransformer"}, "transformation_params": {"0": {}}}, "isolated_only": false}}}''',
        "Ensemble": 0,
    },
    "49": {  # short optimization on M5
        "Model": "DMD",
        "ModelParameters": '''{"rank": 2, "alpha": 1, "amplitude_threshold": null, "eigenvalue_threshold": 1}''',
        "TransformationParameters": '''{"fillna": "ffill", "transformations": {"0": "SeasonalDifference", "1": "AlignLastValue", "2": "Round", "3": "Round", "4": "MinMaxScaler"}, "transformation_params": {"0": {"lag_1": 7, "method": "LastValue"}, "1": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false}, "2": {"decimals": 0, "on_transform": false, "on_inverse": true}, "3": {"decimals": 0, "on_transform": false, "on_inverse": true}, "4": {}}}''',
        "Ensemble": 0,
    },
    "50": {
        'Model': 'NVAR',
        'ModelParameters': '{"k": 2, "ridge_param": 2e-06, "warmup_pts": 1, "seed_pts": 1, "seed_weighted": null, "batch_size": 5, "batch_method": "std_sorted"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "HolidayTransformer", "1": "PositiveShift"}, "transformation_params": {"0": {"threshold": 0.9, "splash_threshold": null, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": false, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": false, "anomaly_detector_params": {"method": "rolling_zscore", "method_params": {"distribution": "uniform", "alpha": 0.03, "rolling_periods": 300, "center": true}, "fillna": "ffill", "transform_dict": {"fillna": "nearest", "transformations": {"0": null}, "transformation_params": {"0": {}}}, "isolated_only": false}, "remove_excess_anomalies": true, "impact": "datepart_regression", "regression_params": {"regression_model": {"model": "ElasticNet", "model_params": {"l1_ratio": 0.1, "fit_intercept": true, "selection": "cyclic"}}, "datepart_method": "simple", "polynomial_degree": null, "transform_dict": null, "holiday_countries_used": false}}, "1": {}}}',
        "Ensemble": 0,
    },
    "51": {  # optimized on M5, 60 SMAPE 2024-06-06
        'Model': 'BallTreeMultivariateMotif',
        'ModelParameters': '{"window": 28, "point_method": "median", "distance_metric": "euclidean", "k": 15}',
        'TransformationParameters': '{"fillna": "ffill_mean_biased", "transformations": {"0": "QuantileTransformer", "1": "QuantileTransformer", "2": "DatepartRegression"}, "transformation_params": 	{"0": {"output_distribution": "uniform", "n_quantiles": 1000}, "1": {"output_distribution": "normal", "n_quantiles": 100}, "2": {"regression_model": {"model": "ElasticNet", "model_params": {"l1_ratio": 0.1, "fit_intercept": true, "selection": "cyclic"}}, "datepart_method": "expanded", "polynomial_degree": null, "transform_dict": null, "holiday_countries_used": false}}}',
        "Ensemble": 0,
    },
    "52": {  # optimized on dap, 2.43 SMAPE 2024-06-20
        'Model': 'SectionalMotif',
        'ModelParameters': '{"window": 7, "point_method": "median", "distance_metric": "canberra", "include_differenced": true, "k": 5, "stride_size": 1, "regression_type": null}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "SeasonalDifference", "1": "AlignLastValue", "2": "SeasonalDifference", "3": "LevelShiftTransformer"}, "transformation_params": {"0": {"lag_1": 7, "method": "Median"}, "1": {"rows": 1, "lag": 2, "method": "additive", "strength": 1.0, "first_value_only": false}, "2": {"lag_1": 12, "method": 5}, "3": {"window_size": 30, "alpha": 2.0, "grouping_forward_limit": 4, "max_level_shifts": 10, "alignment": "average"}}}',
        "Ensemble": 0,
    },
    "53": {  # optimized on dap, 662000 MAE 2024-06-20
        'Model': 'FBProphet',
        'ModelParameters': '{"holiday": {"threshold": 1.0, "splash_threshold": null, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": false, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": false, "anomaly_detector_params": {"method": "IQR", "transform_dict": {"fillna": "time", "transformations": {"0": "Slice"}, "transformation_params": {"0": {"method": 0.5}}}, "forecast_params": {"model_name": "RRVAR", "model_param_dict": {"method": "als", "rank": 0.2, "maxiter": 200}, "model_transform_dict": {"fillna": "mean", "transformations": {"0": "DifferencedTransformer", "1": "bkfilter", "2": "AlignLastValue", "3": "ClipOutliers", "4": "SeasonalDifference"}, "transformation_params": {"0": {"lag": 1, "fill": "zero"}, "1": {}, "2": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false}, "3": {"method": "clip", "std_threshold": 2, "fillna": null}, "4": {"lag_1": 12, "method": "LastValue"}}}}, "method_params": {"iqr_threshold": 2.0, "iqr_quantiles": [0.4, 0.6]}}}, "regression_type": null, "growth": "linear", "n_changepoints": 10, "changepoint_prior_scale": 0.1, "seasonality_mode": "additive", "changepoint_range": 0.85, "seasonality_prior_scale": 0.01, "holidays_prior_scale": 10.0, "trend_phi": 1}',
        'TransformationParameters': '{"fillna": "rolling_mean", "transformations": {"0": "SeasonalDifference", "1": "bkfilter", "2": "SeasonalDifference", "3": "LevelShiftTransformer", "4": "AlignLastDiff"}, "transformation_params": {"0": {"lag_1": 7, "method": "Median"}, "1": {}, "2": {"lag_1": 12, "method": 5}, "3": {"window_size": 90, "alpha": 3.0, "grouping_forward_limit": 4, "max_level_shifts": 10, "alignment": "rolling_diff_3nn"}, "4": {"rows": null, "displacement_rows": 1, "quantile": 1.0, "decay_span": null}}}',
        "Ensemble": 0,
    },
    "54": {  # optimized on prod example daily, 13.5 SMAPE, 0.64 ODA, 42750 MAE, 2024-06-22
        'Model': 'ARDL',
        'ModelParameters': '{"lags": 3, "trend": "c", "order": 0, "causal": false, "regression_type": "simple_3"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "ClipOutliers", "1": "Detrend"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 4.5, "fillna": null}, "1": {"model": "Linear", "phi": 1, "window": 90, "transform_dict": {"fillna": null, "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 4}}}}}}',
        "Ensemble": 0,
    },
    "55": {  # optimized on prod example daily, 15.67 SMAPE 2024-06-22
        'Model': 'FFT',
        'ModelParameters': '{"n_harmonics": 4, "detrend": "linear"}',
        'TransformationParameters': '{"fillna": "time", "transformations": {"0": "AlignLastValue"}, "transformation_params": {"0": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.9, "first_value_only": false}}}',
        "Ensemble": 0,
    },
    "56": {  # optimized on wiki example daily, 27 SMAPE 2024-10-05
        'Model': 'BasicLinearModel',
        'ModelParameters': '{"datepart_method": "simple_binarized", "changepoint_spacing": 360, "changepoint_distance_end": 360, "regression_type": null, "lambda_": 0.01}',
        'TransformationParameters': '{"fillna": "ffill_mean_biased", "transformations": {"0": "Round", "1": "AlignLastValue", "2": "HistoricValues", "3": "ClipOutliers", "4": "bkfilter"}, "transformation_params": {"0": {"decimals": -1, "on_transform": true, "on_inverse": true}, "1": {"rows": 7, "lag": 1, "method": "additive", "strength": 0.2, "first_value_only": true, "threshold": null, "threshold_method": "mean"}, "2": {"window": 10}, "3": {"method": "clip", "std_threshold": 3, "fillna": null}, "4": {}}}',
        "Ensemble": 0,
    },
    "57": {  # optimized on wiki example daily 26.8 SMAPE 2024-10-05
        'Model': 'BasicLinearModel',
        'ModelParameters': '{"datepart_method": ["dayofweek", [365.25, 14]], "changepoint_spacing": 90, "changepoint_distance_end": 360, "regression_type": null, "lambda_": null, "trend_phi": 0.98}',
        'TransformationParameters': '{"fillna": "piecewise_polynomial", "transformations": {"0": "AlignLastValue", "1": "IntermittentOccurrence", "2": "RobustScaler", "3": "Log"}, "transformation_params": {"0": {"rows": 1, "lag": 2, "method": "multiplicative", "strength": 0.9, "first_value_only": false, "threshold": 1, "threshold_method": "mean"}, "1": {"center": "mean"}, "2": {}, "3": {}}}',
        "Ensemble": 0,
    },
    "58": {  # optimized on VN1, best theta, 50 smape 0.44 competition
        'Model': 'Theta',
        'ModelParameters': '{"deseasonalize": true, "difference": false, "use_test": true, "method": "auto", "period": null, "theta": 1.4, "use_mle": false}',
        'TransformationParameters': '{"fillna": "quadratic", "transformations": {"0": "AlignLastValue", "1": "MinMaxScaler", "2": "HistoricValues", "3": "AlignLastValue"}, "transformation_params": {"0": {"rows": 7, "lag": 1, "method": "additive", "strength": 0.7, "first_value_only": false, "threshold": 10, "threshold_method": "max"}, "1": {}, "2": {"window": 10}, "3": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.2, "first_value_only": false, "threshold": null, "threshold_method": "max"}}}',
        "Ensemble": 0,
    },
    "59": {  # optimized on VN1, best wasserstein 80.5 (sample of 600)
        'Model': 'MultivariateRegression',
        'ModelParameters': '{"regression_model": {"model": "ExtraTrees", "model_params": {"n_estimators": 100, "min_samples_leaf": 1, "min_samples_split": 1.0, "max_depth": null, "criterion": "friedman_mse", "max_features": 1}}, "mean_rolling_periods": 12, "macd_periods": 2, "std_rolling_periods": null, "max_rolling_periods": 7, "min_rolling_periods": 364, "quantile90_rolling_periods": 10, "quantile10_rolling_periods": 5, "ewm_alpha": 0.2, "ewm_var_alpha": 0.5, "additional_lag_periods": null, "abs_energy": false, "rolling_autocorr_periods": null, "nonzero_last_n": null, "datepart_method": "recurring", "polynomial_degree": null, "regression_type": "User", "window": null, "holiday": false, "probabilistic": false, "scale_full_X": true, "cointegration": null, "cointegration_lag": 1, "series_hash": false, "frac_slice": null}',
        'TransformationParameters': '{"fillna": "ffill_mean_biased", "transformations": {"0": "AlignLastValue", "1": "PositiveShift", "2": "AlignLastValue"}, "transformation_params": {"0": {"rows": 7, "lag": 1, "method": "additive", "strength": 0.7, "first_value_only": false, "threshold": 10, "threshold_method": "max"}, "1": {}, "2": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false, "threshold": 10, "threshold_method": "max"}}}',
        "Ensemble": 0,
    },
    "60": {  # optimized on VN1 best competition 0.43, good smape 0.49
        'Model': 'SeasonalityMotif',
        'ModelParameters': '{"window": 5, "point_method": "trimmed_mean_40", "distance_metric": "mae", "k": 5, "datepart_method": ["simple_binarized_poly"], "independent": true}',
        'TransformationParameters': '{"fillna": "rolling_mean", "transformations": {"0": "Constraint", "1": "QuantileTransformer", "2": "AlignLastValue"}, "transformation_params": {"0": {"constraint_method": "historic_diff", "constraint_direction": "lower", "constraint_regularization": 1.0, "constraint_value": 0.2, "bounds_only": false, "fillna": null}, "1": {"output_distribution": "uniform", "n_quantiles": 43}, "2": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.2, "first_value_only": false, "threshold": null, "threshold_method": "max"}}}',
        "Ensemble": 0,
    },
    "61": {  # optimized on wiki daily, 25.6 smape, 991k mqae, initial fit
        'Model': 'TVVAR',
        'ModelParameters': '{"datepart_method": "expanded", "changepoint_spacing": 90, "changepoint_distance_end": 520, "regression_type": null, "lags": [7], "rolling_means": [4], "lambda_": 0.001, "trend_phi": 0.99, "var_dampening": 0.98, "phi": null, "max_cycles": 2000, "apply_pca": true, "base_scaled": false, "x_scaled": false, "var_preprocessing": {"fillna": "rolling_mean", "transformations": {"0": "FFTFilter"}, "transformation_params": {"0": {"cutoff": 0.4, "reverse": false, "on_transform": true, "on_inverse": false}}}, "threshold_value": null}',
        'TransformationParameters': '{"fillna": "linear", "transformations": {"0": "QuantileTransformer", "1": "RobustScaler", "2": "PositiveShift", "3": "MinMaxScaler", "4": "AlignLastValue", "5": "ClipOutliers"}, "transformation_params": {"0": {"output_distribution": "normal", "n_quantiles": 100}, "1": {}, "2": {}, "3": {}, "4": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.5, "first_value_only": false, "threshold": 1, "threshold_method": "max"}, "5": {"method": "clip", "std_threshold": 4, "fillna": null}}}',
        "Ensemble": 0,
    },
    "62": {
        'Model': 'TVVAR',
        'ModelParameters': '{"datepart_method": "expanded", "changepoint_spacing": 6, "changepoint_distance_end": 520, "regression_type": null, "lags": [7], "rolling_means": [4], "lambda_": 0.001, "trend_phi": 0.99, "var_dampening": 0.98, "phi": null, "max_cycles": 2000, "apply_pca": true, "base_scaled": false, "x_scaled": false, "var_preprocessing": {"fillna": "rolling_mean", "transformations": {"0": "FFTFilter"}, "transformation_params": {"0": {"cutoff": 0.4, "reverse": false, "on_transform": true, "on_inverse": false}}}, "threshold_value": null}',
        'TransformationParameters': '{"fillna": "linear", "transformations": {"0": "QuantileTransformer", "1": "RobustScaler", "2": "AlignLastValue", "3": "MinMaxScaler", "4": "MinMaxScaler"}, "transformation_params": {"0": {"output_distribution": "normal", "n_quantiles": 100}, "1": {}, "2": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.2, "first_value_only": false, "threshold": 1, "threshold_method": "max"}, "3": {}, "4": {}}}',
        'Ensemble': 0,
    },
    "63": {  # VPV best score
        'Model': 'BasicLinearModel',
        'ModelParameters': '{"datepart_method": "recurring", "changepoint_spacing": 28, "changepoint_distance_end": 90, "regression_type": null, "lambda_": 0.01, "trend_phi": 0.98, "holiday_countries_used": true}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "Slice", "1": "ClipOutliers", "2": "SeasonalDifference", "3": "LevelShiftTransformer", "4": "LevelShiftTransformer", "5": "PositiveShift"}, "transformation_params": {"0": {"method": 0.2}, "1": {"method": "clip", "std_threshold": 3, "fillna": null}, "2": {"lag_1": 12, "method": 5}, "3": {"window_size": 7, "alpha": 2.0, "grouping_forward_limit": 5, "max_level_shifts": 30, "alignment": "rolling_diff_3nn"}, "4": {"window_size": 7, "alpha": 2.0, "grouping_forward_limit": 5, "max_level_shifts": 30, "alignment": "rolling_diff_3nn"}, "5": {}}}',
        'Ensemble': 0,
    },
    "64": {  # load daily best on some
        'Model': 'PreprocessingExperts',
        'ModelParameters': '{"point_method": "midhinge", "model_params": {"model_str": "MetricMotif", "parameter_dict": {"window": 7, "point_method": "median", "distance_metric": "canberra", "k": 5, "comparison_transformation": {"fillna": "akima", "transformations": {"0": "DifferencedTransformer"}, "transformation_params": {"0": {"lag": 1, "fill": "bfill"}}}, "combination_transformation": {"fillna": "quadratic", "transformations": {"0": "MinMaxScaler"}, "transformation_params": {"0": {}}}}, "transformation_dict": {"fillna": "pchip", "transformations": {"0": "MinMaxScaler", "1": "AlignLastValue"}, "transformation_params": {"0": {}, "1": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.7, "first_value_only": false, "threshold": null, "threshold_method": "mean"}}}}, "transformation_dict": {"fillna": "time", "transformations": {"0": "DatepartRegression", "1": "SeasonalDifference", "2": "Detrend", "3": "QuantileTransformer", "4": "QuantileTransformer", "5": "ClipOutliers"}, "transformation_params": {"0": {"regression_model": {"model": "RandomForest", "model_params": {"n_estimators": 1000, "min_samples_leaf": 1, "bootstrap": true}}, "datepart_method": ["morlet_365.25_12_12", "ricker_7_7_1"], "polynomial_degree": null, "transform_dict": {"fillna": null, "transformations": {"0": "EWMAFilter"}, "transformation_params": {"0": {"span": 7}}}, "holiday_countries_used": true, "lags": null, "forward_lags": null}, "1": {"lag_1": 24, "method": "LastValue"}, "2": {"model": "Linear", "phi": 0.998, "window": null, "transform_dict": null}, "3": {"output_distribution": "normal", "n_quantiles": 537}, "4": {"output_distribution": "uniform", "n_quantiles": 1000}, "5": {"method": "remove", "std_threshold": 3.5, "fillna": "ffill"}}}}',
        'TransformationParameters': '{"fillna": "time", "transformations": {"0": "DatepartRegression", "1": "SeasonalDifference", "2": "Detrend", "3": "QuantileTransformer", "4": "QuantileTransformer", "5": "ClipOutliers"}, "transformation_params": {"0": {"regression_model": {"model": "RandomForest", "model_params": {"n_estimators": 1000, "min_samples_leaf": 1, "bootstrap": true}}, "datepart_method": ["morlet_365.25_12_12", "ricker_7_7_1"], "polynomial_degree": null, "transform_dict": {"fillna": null, "transformations": {"0": "EWMAFilter"}, "transformation_params": {"0": {"span": 7}}}, "holiday_countries_used": true, "lags": null, "forward_lags": null}, "1": {"lag_1": 24, "method": "LastValue"}, "2": {"model": "Linear", "phi": 0.998, "window": null, "transform_dict": null}, "3": {"output_distribution": "normal", "n_quantiles": 537}, "4": {"output_distribution": "uniform", "n_quantiles": 1000}, "5": {"method": "remove", "std_threshold": 3.5, "fillna": "ffill"}}}',
        'Ensemble': 0,
    },
    "65": {  # load daily best single model overall
        'Model': 'PreprocessingExperts',
        'ModelParameters': '{"point_method": "weighted_mean", "model_params": {"model_str": "SectionalMotif", "parameter_dict": {"window": 10, "point_method": "median", "distance_metric": "hamming", "include_differenced": true, "k": 20, "stride_size": 1, "regression_type": "User", "comparison_transformation": {"fillna": "ffill", "transformations": {"0": "DifferencedTransformer"}, "transformation_params": {"0": {"lag": 7, "fill": "bfill"}}}, "combination_transformation": null}, "transformation_dict": {"fillna": "akima", "transformations": {"0": "MinMaxScaler"}, "transformation_params": {"0": {}}}}, "transformation_dict": {"fillna": "fake_date", "transformations": {"0": "AlignLastValue", "1": "ClipOutliers", "2": "PowerTransformer", "3": "StandardScaler", "4": "Discretize", "5": "Discretize", "6": "bkfilter"}, "transformation_params": {"0": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false, "threshold": null, "threshold_method": "mean"}, "1": {"method": "clip", "std_threshold": 3.5, "fillna": null}, "2": {}, "3": {}, "4": {"discretization": "upper", "n_bins": 10}, "5": {"discretization": "upper", "n_bins": 10}, "6": {}}}}',
        'TransformationParameters': '{"fillna": "fake_date", "transformations": {"0": "AlignLastValue", "1": "ClipOutliers", "2": "PowerTransformer", "3": "StandardScaler", "4": "Discretize", "5": "Discretize", "6": "bkfilter"}, "transformation_params": {"0": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false, "threshold": null, "threshold_method": "mean"}, "1": {"method": "clip", "std_threshold": 3.5, "fillna": null}, "2": {}, "3": {}, "4": {"discretization": "upper", "n_bins": 10}, "5": {"discretization": "upper", "n_bins": 10}, "6": {}}}',
        'Ensemble': 0,
    },
    "66": {  # load daily part of best small horizontal
        'Model': 'PreprocessingRegression',
        'ModelParameters': '{"window_size": 20, "max_history": 1000, "one_step": false, "normalize_window": false, "processed_y": true, "transformation_dict": {"fillna": "fake_date", "transformations": {"0": "SeasonalDifference", "1": "AlignLastValue", "2": "QuantileTransformer", "3": "AnomalyRemoval"}, "transformation_params": {"0": {"lag_1": 364, "method": "Mean"}, "1": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.7, "first_value_only": false, "threshold": 10, "threshold_method": "mean"}, "2": {"output_distribution": "uniform", "n_quantiles": 100}, "3": {"method": "EE", "method_params": {"contamination": 0.1, "assume_centered": false, "support_fraction": 0.2}, "fillna": "rolling_mean_24", "transform_dict": {"fillna": null, "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 6}}}, "isolated_only": false, "on_inverse": false}}}, "datepart_method": "simple_2", "regression_type": "User", "regression_model": {"model": "LightGBM", "model_params": {"colsample_bytree": 0.1645, "learning_rate": 0.0203, "max_bin": 1023, "min_child_samples": 16, "n_estimators": 1794, "num_leaves": 15, "reg_alpha": 0.00098, "reg_lambda": 0.686}}}',
        'TransformationParameters': '{"fillna": "fake_date", "transformations": {"0": "SeasonalDifference", "1": "AlignLastValue", "2": "QuantileTransformer", "3": "AnomalyRemoval"}, "transformation_params": {"0": {"lag_1": 364, "method": "Mean"}, "1": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.7, "first_value_only": false, "threshold": 10, "threshold_method": "mean"}, "2": {"output_distribution": "uniform", "n_quantiles": 100}, "3": {"method": "EE", "method_params": {"contamination": 0.1, "assume_centered": false, "support_fraction": 0.2}, "fillna": "rolling_mean_24", "transform_dict": {"fillna": null, "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 6}}}, "isolated_only": false, "on_inverse": false}}}',
        'Ensemble': 0,
    },
    "67": {  # load daily part of best small horizontal
        'Model': 'PreprocessingRegression',
        'ModelParameters': '{"window_size": 28, "max_history": null, "one_step": true, "normalize_window": false, "processed_y": true, "transformation_dict": {"fillna": "fake_date", "transformations": {"0": "DifferencedTransformer", "1": "QuantileTransformer", "2": "SeasonalDifference", "3": "Discretize"}, "transformation_params": {"0": {"lag": 1, "fill": "bfill"}, "1": {"output_distribution": "uniform", "n_quantiles": 268}, "2": {"lag_1": 2, "method": "LastValue"}, "3": {"discretization": "upper", "n_bins": 10}}}, "datepart_method": "simple", "regression_type": null, "regression_model": {"model": "LightGBM", "model_params": {"colsample_bytree": 0.1645, "learning_rate": 0.0203, "max_bin": 1023, "min_child_samples": 16, "n_estimators": 1794, "num_leaves": 15, "reg_alpha": 0.00098, "reg_lambda": 0.686}}}',
        'TransformationParameters': '{"fillna": "fake_date", "transformations": {"0": "DifferencedTransformer", "1": "QuantileTransformer", "2": "SeasonalDifference", "3": "Discretize"}, "transformation_params": {"0": {"lag": 1, "fill": "bfill"}, "1": {"output_distribution": "uniform", "n_quantiles": 268}, "2": {"lag_1": 2, "method": "LastValue"}, "3": {"discretization": "upper", "n_bins": 10}}}',
        'Ensemble': 0,
    },
    "68": {  # load daily best overall, moderately deep search, 26 smape in val and holdout, 2025-03-03
        'Model': 'FBProphet',
        'ModelParameters': '{"holiday": false, "regression_type": null, "growth": "linear", "n_changepoints": 20, "changepoint_prior_scale": 30, "seasonality_mode": "additive", "changepoint_range": 0.8, "seasonality_prior_scale": 0.01, "holidays_prior_scale": 10.0, "trend_phi": 1}',
        'TransformationParameters': '{"fillna": "rolling_mean_24", "transformations": {"0": "Log", "1": "AlignLastValue", "2": "LevelShiftTransformer", "3": "SinTrend", "4": "AlignLastValue"}, "transformation_params": {"0": {}, "1": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false, "threshold": 10, "threshold_method": "max"}, "2": {"window_size": 30, "alpha": 2.5, "grouping_forward_limit": 2, "max_level_shifts": 5, "alignment": "average"}, "3": {}, "4": {"rows": 4, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false, "threshold": null, "threshold_method": "max"}}}',
        'Ensemble': 0,
    },
    "69": {  # optimized on prod example
        'Model': 'BallTreeRegressionMotif',
        'ModelParameters': '{"window": 3, "point_method": "midhinge", "distance_metric": "canberra", "k": 100, "sample_fraction": 5000000, "comparison_transformation": {"fillna": "cubic", "transformations": {"0": "AlignLastDiff"}, "transformation_params": {"0": {"rows": 364, "displacement_rows": 1, "quantile": 1.0, "decay_span": null}}}, "combination_transformation": {"fillna": "time", "transformations": {"0": "AlignLastDiff"}, "transformation_params": {"0": {"rows": 7, "displacement_rows": 1, "quantile": 1.0, "decay_span": 2}}}, "extend_df": true, "mean_rolling_periods": 12, "macd_periods": 74, "std_rolling_periods": 30, "max_rolling_periods": 364, "min_rolling_periods": null, "quantile90_rolling_periods": 10, "quantile10_rolling_periods": 10, "ewm_alpha": null, "ewm_var_alpha": null, "additional_lag_periods": null, "abs_energy": false, "rolling_autocorr_periods": null, "nonzero_last_n": null, "datepart_method": null, "polynomial_degree": null, "regression_type": null, "holiday": false, "scale_full_X": false, "series_hash": true, "frac_slice": null}',
        'TransformationParameters': '{"fillna": "akima", "transformations": {"0": "Log", "1": "SinTrend", "2": "ChangepointDetrend"}, "transformation_params": {"0": {}, "1": {}, "2": {"model": "Linear", "changepoint_spacing": 5040, "changepoint_distance_end": 520, "datepart_method": "common_fourier"}}}',
        'Ensemble': 0,
    },
    "70": {  # from mosaic profile template (subset of prod example)
        'Model': 'DatepartRegression',
        'ModelParameters': '{"regression_model": {"model": "ElasticNet", "model_params": {"l1_ratio": 0.1, "fit_intercept": true, "selection": "cyclic", "max_iter": 1000}}, "datepart_method": "anchored_warped_fourier:us_school", "polynomial_degree": null, "holiday_countries_used": true, "lags": null, "forward_lags": null, "regression_type": null}',
        'TransformationParameters': '{"fillna": "linear", "transformations": {"0": "AlignLastValue", "1": "Slice", "2": "Constraint", "3": "Log", "4": "AlignLastValue"}, "transformation_params": {"0": {"rows": 1, "lag": 1, "method": "multiplicative", "strength": 1.0, "first_value_only": false, "threshold": null, "threshold_method": "max"}, "1": {"method": 0.5}, "2": {"constraint_method": "slope", "constraint_direction": "upper", "constraint_regularization": 0.7, "constraint_value": {"slope": 0.1, "window": 30, "window_agg": "max", "threshold": 0.01}, "bounds_only": false, "fillna": null}, "3": {}, "4": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.7, "first_value_only": false, "threshold": 3, "threshold_method": "max", "mean_type": "arithmetic"}}}',
        'Ensemble': 0,
    },
    "71": {  # VECM best on subset of prod example
        'Model': 'VECM',
        'ModelParameters': '{"deterministic": "n", "k_ar_diff": 3, "seasons": 0, "coint_rank": 1, "regression_type": "Holiday"}',
        'TransformationParameters': '{"fillna": "zero", "transformations": {"0": "Log", "1": "SeasonalDifference", "2": "AnomalyRemoval", "3": "SeasonalDifference", "4": "EWMAFilter", "5": "SinTrend"}, "transformation_params": {"0": {}, "1": {"lag_1": 7, "method": "Mean"}, "2": {"method": "rolling_zscore", "method_params": {"distribution": "gamma", "alpha": 0.05, "rolling_periods": 200, "center": false}, "fillna": "ffill", "transform_dict": null, "isolated_only": false, "on_inverse": false}, "3": {"lag_1": 7, "method": 5}, "4": {"span": 10}, "5": {}}}',
        'Ensemble': 0,
    },
    "72": {  # VECM best overall smape on prod example
        'Model': 'VECM',
        'ModelParameters': '{"deterministic": "n", "k_ar_diff": 0, "seasons": 0, "coint_rank": 1, "regression_type": null}',
        'TransformationParameters': '{"fillna": "quadratic", "transformations": {"0": "QuantileTransformer", "1": "SeasonalDifference", "2": "AlignLastDiff", "3": "Constraint", "4": "convolution_filter"}, "transformation_params": {"0": {"output_distribution": "uniform", "n_quantiles": 1000}, "1": {"lag_1": 7, "method": "Mean"}, "2": {"rows": 1, "displacement_rows": 1, "quantile": 1.0, "decay_span": 3}, "3": {"constraint_method": "slope", "constraint_direction": "upper", "constraint_regularization": 0.7, "constraint_value": {"slope": 0.1, "window": 30, "window_agg": "max", "threshold": 0.01}, "bounds_only": false, "fillna": null}, "4": {}}}',
        'Ensemble': 0,
    },
    "73": {  # Cassandra common_fourier trend
        'Model': 'Cassandra',
        'ModelParameters': '{"preprocessing_transformation": {"fillna": "ffill", "transformations": {"0": "AlignLastDiff"}, "transformation_params": {"0": {"rows": 90, "displacement_rows": 1, "quantile": 1.0, "decay_span": 2}}}, "scaling": "BaseScaler", "past_impacts_intervention": null, "seasonalities": ["common_fourier"], "ar_lags": null, "ar_interaction_seasonality": null, "anomaly_detector_params": null, "anomaly_intervention": null, "holiday_detector_params": {"threshold": 0.8, "splash_threshold": 0.65, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": true, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": false, "use_hindu_holidays": false, "anomaly_detector_params": {"method": "rolling_zscore", "method_params": {"distribution": "chi2", "alpha": 0.05, "rolling_periods": 300, "center": true}, "fillna": "rolling_mean_24", "transform_dict": {"transformations": {"0": "DifferencedTransformer"}, "transformation_params": {"0": {}}}, "isolated_only": false, "on_inverse": false}, "remove_excess_anomalies": false, "impact": null, "regression_params": null, "output": "multivariate"}, "holiday_countries_used": false, "multivariate_feature": null, "multivariate_transformation": null, "regressor_transformation": null, "regressors_used": false, "linear_model": {"model": "lstsq", "lambda": 1, "recency_weighting": 0.5}, "randomwalk_n": null, "trend_window": 3, "trend_standin": null, "trend_anomaly_detector_params": null, "trend_transformation": {"fillna": "ffill", "transformations": {"0": "AnomalyRemoval", "1": "RobustScaler"}, "transformation_params": {"0": {"method": "IQR", "transform_dict": {"fillna": null, "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 6}}}, "method_params": {"iqr_threshold": 2.5, "iqr_quantiles": [0.25, 0.75]}, "fillna": "ffill"}, "1": {}}}, "trend_model": {"Model": "ARDL", "ModelParameters": {"lags": 3, "trend": "t", "order": 0, "causal": false, "regression_type": "common_fourier"}}, "trend_phi": 0.98, "x_scaler": false}',
        'TransformationParameters': '{"fillna": "linear", "transformations": {"0": "AlignLastValue", "1": "Slice", "2": "Constraint", "3": "Log", "4": "AlignLastValue", "5": "SeasonalDifference", "6": "AlignLastValue"}, "transformation_params": {"0": {"rows": 1, "lag": 1, "method": "multiplicative", "strength": 1.0, "first_value_only": false, "threshold": null, "threshold_method": "max"}, "1": {"method": 0.5}, "2": {"constraint_method": "slope", "constraint_direction": "upper", "constraint_regularization": 0.7, "constraint_value": {"slope": 0.1, "window": 30, "window_agg": "max", "threshold": 0.01}, "bounds_only": false, "fillna": null}, "3": {}, "4": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.7, "first_value_only": false, "threshold": 3, "threshold_method": "max", "mean_type": "arithmetic"}, "5": {"lag_1": 12, "method": "Median"}, "6": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false, "threshold": 1, "threshold_method": "mean"}}}',
        'Ensemble': 0,
    },
    # Add a WindowRegression, add new UnobservedComponents, add a pMLP, a new MultivariateRegression
}

general_template = pd.DataFrame.from_dict(general_template_dict, orient='index')


"""
# Basic Template Construction Code
# transformer_max_depth = 6 and transformer_list = "fast"
from autots.evaluator.auto_model import unpack_ensemble_models
max_per_model_class = 1
export_template = model.validation_results.model_results
export_template = export_template[
    export_template['Runs'] >= (model.num_validations + 1)
]
export_template = (
    export_template.sort_values('Score', ascending=True)
    .groupby('Model')
    .head(max_per_model_class)
    .reset_index()
)
import json
export2 = unpack_ensemble_models(model.best_model, keep_ensemble=False, recursive=True)
export_final = pd.concat([export_template, export2])
export_final = export_final[export_final['Ensemble'] < 1]
export_final[["Model", "ModelParameters", "TransformationParameters", "Ensemble"]].reset_index(drop=True).to_json(orient='index')

import pprint
import json

imported = pd.read_csv("autots_forecast_template_gen.csv")
export = unpack_ensemble_models(imported, keep_ensemble=False, recursive=True)
export[export['Ensemble'] < 1].to_json("template.json", orient="records")
with open("template.json", "r") as jsn:
    json_temp = json.loads(jsn.read())
print(json_temp)
with open("template.txt", "w") as txt:
    txt.write(json.dumps(json_temp, indent=4, sort_keys=False))
"""
