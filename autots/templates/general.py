"""Starting templates for models."""

import pandas as pd

general_template_dict = {
    "1": {
        "Model": "ARIMA",
        "ModelParameters": '{"p": 4, "d": 0, "q": 12, "regression_type": null}',
        "TransformationParameters": '{"fillna": "cubic", "transformations": {"0": "bkfilter"}, "transformation_params": {"0": {}}}',
        "Ensemble": 0,
    },
    "3": {
        "Model": "AverageValueNaive",
        "ModelParameters": "{\"method\": \"Mean\"}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"DifferencedTransformer\", \"1\": \"SinTrend\"}, \"transformation_params\": {\"0\": {}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "4": {
        "Model": "AverageValueNaive",
        "ModelParameters": "{\"method\": \"Mean\"}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "5": {
        "Model": "AverageValueNaive",
        "ModelParameters": "{\"method\": \"Mean\"}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"Round\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"lag_1\": 7, \"method\": \"Mean\"}, \"1\": {\"model\": \"middle\", \"decimals\": 2, \"on_transform\": true, \"on_inverse\": false}, \"2\": {\"model\": \"GLS\"}}}",
        "Ensemble": 0,
    },
    "6": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"DecisionTree\", \"model_params\": {\"max_depth\": 3, \"min_samples_split\": 2}}, \"datepart_method\": \"recurring\", \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "7": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"SVM\", \"model_params\": {}}, \"datepart_method\": \"recurring\", \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"RollingMeanTransformer\", \"2\": \"Detrend\", \"3\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3.5, \"fillna\": null}, \"1\": {\"fixed\": true, \"window\": 10}, \"2\": {\"model\": \"Linear\"}, \"3\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "8": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"MLP\", \"model_params\": {\"hidden_layer_sizes\": [25, 15, 25], \"max_iter\": 1000, \"activation\": \"tanh\", \"solver\": \"lbfgs\", \"early_stopping\": false, \"learning_rate_init\": 0.001}}, \"datepart_method\": \"recurring\", \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "9": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"KerasRNN\", \"model_params\": {\"kernel_initializer\": \"glorot_uniform\", \"epochs\": 50, \"batch_size\": 32, \"optimizer\": \"adam\", \"loss\": \"Huber\", \"hidden_layer_sizes\": [32, 32, 32], \"rnn_type\": \"GRU\"}}, \"datepart_method\": \"recurring\", \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"Round\", \"1\": \"QuantileTransformer\", \"2\": \"QuantileTransformer\", \"3\": \"QuantileTransformer\", \"4\": \"PowerTransformer\"}, \"transformation_params\": {\"0\": {\"model\": \"middle\", \"decimals\": 1, \"on_transform\": true, \"on_inverse\": false}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 20}, \"3\": {\"output_distribution\": \"normal\", \"n_quantiles\": 1000}, \"4\": {}}}",
        "Ensemble": 0,
    },
    "10": {
        "Model": "ETS",
        "ModelParameters": "{\"damped_trend\": false, \"trend\": \"additive\", \"seasonal\": null, \"seasonal_periods\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "11": {
        "Model": "ETS",
        "ModelParameters": "{\"damped_trend\": false, \"trend\": null, \"seasonal\": \"additive\", \"seasonal_periods\": 7}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"Round\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"model\": \"middle\", \"decimals\": 0, \"on_transform\": false, \"on_inverse\": true}, \"2\": {\"model\": \"GLS\"}}}",
        "Ensemble": 0,
    },
    "12": {
        "Model": "GLM",
        "ModelParameters": "{\"family\": \"Binomial\", \"constant\": false, \"regression_type\": \"datepart\"}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 4, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "13": {
        "Model": "GLM",
        "ModelParameters": "{\"family\": \"Binomial\", \"constant\": false, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "14": {
        "Model": "GLS",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean\", \"transformations\": {\"0\": \"RollingMeanTransformer\", \"1\": \"DifferencedTransformer\", \"2\": \"Detrend\", \"3\": \"Slice\"}, \"transformation_params\": {\"0\": {\"fixed\": true, \"window\": 3}, \"1\": {}, \"2\": {\"model\": \"Linear\"}, \"3\": {\"method\": 100}}}",
        "Ensemble": 0,
    },
    "15": {
        "Model": "GLS",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"median\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"RobustScaler\", \"3\": \"Round\", \"4\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3.5, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}, \"3\": {\"model\": \"middle\", \"decimals\": 2, \"on_transform\": true, \"on_inverse\": true}, \"4\": {}}}",
        "Ensemble": 0,
    },
    "16": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"DeepAR\", \"epochs\": 150, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"PowerTransformer\", \"1\": \"DifferencedTransformer\", \"2\": \"bkfilter\"}, \"transformation_params\": {\"0\": {}, \"1\": {}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "17": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"NPTS\", \"epochs\": 20, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"Detrend\", \"1\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {\"model\": \"Linear\"}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "18": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"WaveNet\", \"epochs\": 40, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "19": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"Transformer\", \"epochs\": 40, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "20": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"SFF\", \"epochs\": 40, \"learning_rate\": 0.01, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "21": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"bkfilter\", \"1\": \"SinTrend\", \"2\": \"Detrend\", \"3\": \"PowerTransformer\"}, \"transformation_params\": {\"0\": {}, \"1\": {}, \"2\": {\"model\": \"Linear\"}, \"3\": {}}}",
        "Ensemble": 0,
    },
    "22": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"PositiveShift\", \"1\": \"SinTrend\", \"2\": \"bkfilter\"}, \"transformation_params\": {\"0\": {}, \"1\": {}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "23": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"SinTrend\"}, \"transformation_params\": {\"0\": {\"lag_1\": 7, \"method\": \"LastValue\"}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "24": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 1, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "25": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"LastValue\", \"lag_1\": 2, \"lag_2\": 7}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"SinTrend\", \"1\": \"Round\", \"2\": \"PowerTransformer\"}, \"transformation_params\": {\"0\": {}, \"1\": {\"model\": \"middle\", \"decimals\": 2, \"on_transform\": false, \"on_inverse\": true}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "26": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"LastValue\", \"lag_1\": 2, \"lag_2\": 1}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"QuantileTransformer\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"lag_1\": 12, \"method\": \"Median\"}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"model\": \"GLS\"}}}",
        "Ensemble": 0,
    },
    "27": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"LastValue\", \"lag_1\": 7, \"lag_2\": 2}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"ClipOutliers\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"1\": {\"method\": \"clip\", \"std_threshold\": 2, \"fillna\": null}}}",
        "Ensemble": 0,
    },
    "28": {
        "Model": "UnobservedComponents",
        "ModelParameters": "{\"level\": true, \"trend\": false, \"cycle\": true, \"damped_cycle\": true, \"irregular\": true, \"stochastic_trend\": false, \"stochastic_level\": true, \"stochastic_cycle\": true, \"regression_type\": \"Holiday\"}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"PositiveShift\", \"1\": \"Detrend\", \"2\": \"bkfilter\", \"3\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {}, \"1\": {\"model\": \"Linear\"}, \"2\": {}, \"3\": {}}}",
        "Ensemble": 0,
    },
    "29": {
        "Model": "UnobservedComponents",
        "ModelParameters": "{\"level\": false, \"trend\": false, \"cycle\": true, \"damped_cycle\": false, \"irregular\": false, \"stochastic_trend\": false, \"stochastic_level\": true, \"stochastic_cycle\": false, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"RobustScaler\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "30": {
        "Model": "UnobservedComponents",
        "ModelParameters": "{\"level\": true, \"trend\": false, \"cycle\": false, \"damped_cycle\": false, \"irregular\": false, \"stochastic_trend\": false, \"stochastic_level\": true, \"stochastic_cycle\": true, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"ClipOutliers\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 5, \"fillna\": null}}}",
        "Ensemble": 0,
    },
    "31": {
        "Model": "VAR",
        "ModelParameters": "{\"regression_type\": null, \"maxlags\": 5, \"ic\": \"fpe\"}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "32": {
        "Model": "VAR",
        "ModelParameters": "{\"regression_type\": null, \"maxlags\": 15, \"ic\": \"aic\"}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"RollingMeanTransformer\", \"1\": \"QuantileTransformer\", \"2\": \"RollingMeanTransformer\"}, \"transformation_params\": {\"0\": {\"fixed\": true, \"window\": 10}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"fixed\": false, \"window\": 10}}}",
        "Ensemble": 0,
    },
    "33": {
        "Model": "VECM",
        "ModelParameters": "{\"deterministic\": \"cili\", \"k_ar_diff\": 2, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"Detrend\", \"2\": \"Detrend\", \"3\": \"PowerTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"model\": \"GLS\"}, \"2\": {\"model\": \"Linear\"}, \"3\": {}}}",
        "Ensemble": 0,
    },
    "34": {
        "Model": "VECM",
        "ModelParameters": "{\"deterministic\": \"li\", \"k_ar_diff\": 3, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"model\": \"Linear\"}}}",
        "Ensemble": 0,
    },
    "35": {
        "Model": "WindowRegression",
        "ModelParameters": "{\"window_size\": 10, \"regression_model\": {\"model\": \"MLP\", \"model_params\": {\"hidden_layer_sizes\": [72, 36, 72], \"max_iter\": 250, \"activation\": \"relu\", \"solver\": \"lbfgs\", \"early_stopping\": false, \"learning_rate_init\": 0.001}}, \"input_dim\": \"univariate\", \"output_dim\": \"forecast_length\", \"normalize_window\": false, \"shuffle\": true, \"max_windows\": 5000}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"MinMaxScaler\", \"2\": \"RobustScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 100}, \"1\": {}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "36": {
        "Model": "ConstantNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"PowerTransformer\", \"1\": \"QuantileTransformer\", \"2\": \"SeasonalDifference\"}, \"transformation_params\": {\"0\": {}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"lag_1\": 7, \"method\": \"LastValue\"}}}",
        "Ensemble": 0,
    },
    # Gen 2
    "37": {
        "Model": "FBProphet",
        "ModelParameters": "{\"holiday\": true, \"regression_type\": null, \"growth\": \"linear\", \"n_changepoints\": 30, \"changepoint_prior_scale\": 0.05, \"seasonality_mode\": \"additive\", \"changepoint_range\": 0.98, \"seasonality_prior_scale\": 10.0, \"holidays_prior_scale\": 10.0}",
        "TransformationParameters": "{\"fillna\": \"quadratic\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"Slice\", \"2\": \"SeasonalDifference\", \"3\": \"Round\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 1000}, \"1\": {\"method\": 0.5}, \"2\": {\"lag_1\": 7, \"method\": \"LastValue\"}, \"3\": {\"decimals\": -1, \"on_transform\": false, \"on_inverse\": true}}}",
        "Ensemble": 0,
    },
    "38": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"Transformer\", \"epochs\": 80, \"learning_rate\": 0.01, \"context_length\": \"1ForecastLength\", \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "39": {
        "Model": "MultivariateRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"RandomForest\", \"model_params\": {\"n_estimators\": 200, \"min_samples_leaf\": 1, \"bootstrap\": true}}, \"mean_rolling_periods\": 12, \"macd_periods\": 94, \"std_rolling_periods\": null, \"max_rolling_periods\": null, \"min_rolling_periods\": 7, \"quantile90_rolling_periods\": 10, \"quantile10_rolling_periods\": 30, \"ewm_alpha\": 0.1, \"ewm_var_alpha\": 0.2, \"additional_lag_periods\": null, \"abs_energy\": false, \"rolling_autocorr_periods\": null, \"datepart_method\": null, \"polynomial_degree\": null, \"regression_type\": null, \"window\": null, \"holiday\": true, \"probabilistic\": false}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"MaxAbsScaler\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {}, \"2\": {\"model\": \"Linear\", \"phi\": 1, \"window\": null}}}",
        "Ensemble": 0,
    },
    "40": {
        "Model": "MultivariateRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"LightGBM\", \"model_params\": {\"objective\": \"regression\", \"learning_rate\": 0.1, \"num_leaves\": 70, \"max_depth\": -1, \"boosting_type\": \"gbdt\", \"n_estimators\": 100}}, \"mean_rolling_periods\": 12, \"macd_periods\": null, \"std_rolling_periods\": 30, \"max_rolling_periods\": null, \"min_rolling_periods\": 28, \"quantile90_rolling_periods\": 10, \"quantile10_rolling_periods\": null, \"ewm_alpha\": null, \"ewm_var_alpha\": null, \"additional_lag_periods\": null, \"abs_energy\": false, \"rolling_autocorr_periods\": null, \"datepart_method\": null, \"polynomial_degree\": null, \"regression_type\": \"User\", \"window\": 3, \"holiday\": false, \"probabilistic\": false}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"MaxAbsScaler\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {}, \"2\": {\"model\": \"Linear\", \"phi\": 1, \"window\": null}}}",
        "Ensemble": 0,
    },
    "41": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"ExtraTrees\", \"model_params\": {\"n_estimators\": 500, \"min_samples_leaf\": 1, \"max_depth\": 10}}, \"datepart_method\": \"expanded\", \"polynomial_degree\": null, \"regression_type\": \"User\"}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"MaxAbsScaler\", \"1\": \"MinMaxScaler\"}, \"transformation_params\": {\"0\": {}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "42": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"lastvalue\", \"lag_1\": 364, \"lag_2\": 28}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"MaxAbsScaler\", \"2\": \"Round\"}, \"transformation_params\": {\"0\": {\"lag_1\": 7, \"method\": \"LastValue\"}, \"1\": {}, \"2\": {\"decimals\": 0, \"on_transform\": false, \"on_inverse\": true}}}",
        "Ensemble": 0,
    },
    "43": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"ExtraTrees\", \"model_params\": {\"n_estimators\": 100, \"min_samples_leaf\": 1, \"max_depth\": null}}, \"datepart_method\": \"recurring\", \"polynomial_degree\": null, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {}}}",
        "Ensemble": 0,
    },
    "44": {
        "Model": "UnobservedComponents",
        "ModelParameters": "{\"level\": false, \"maxiter\": 100, \"cov_type\": \"opg\", \"method\": \"lbfgs\", \"autoregressive\": null, \"regression_type\": \"Holiday\"}",
        "TransformationParameters": "{\"fillna\": \"median\", \"transformations\": {\"0\": \"Slice\", \"1\": \"SeasonalDifference\"}, \"transformation_params\": {\"0\": {\"method\": 0.2}, \"1\": {\"lag_1\": 7, \"method\": \"LastValue\"}}}",
        "Ensemble": 0,
    },
    "45": {
        "Model": "UnobservedComponents",
        "ModelParameters": "{\"level\": \"random walk with drift\", \"maxiter\": 50, \"cov_type\": \"opg\", \"method\": \"lbfgs\", \"autoregressive\": null, \"regression_type\": \"Holiday\"}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"SeasonalDifference\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {\"lag_1\": 7, \"method\": \"LastValue\"}}}",
        "Ensemble": 0,
    },
    "46": {
        "Model": "ETS",
        "ModelParameters": "{\"damped_trend\": false, \"trend\": null, \"seasonal\": \"additive\", \"seasonal_periods\": 28}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"MaxAbsScaler\", \"1\": \"Slice\"}, \"transformation_params\": {\"0\": {}, \"1\": {\"method\": 100}}}",
        "Ensemble": 0,
    },
    "47": {
        "Model": "VECM",
        "ModelParameters": "{\"deterministic\": \"n\", \"k_ar_diff\": 2, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean\", \"transformations\": {\"0\": \"PCA\", \"1\": \"Detrend\", \"2\": \"HPFilter\"}, \"transformation_params\": {\"0\": {}, \"1\": {\"model\": \"GLS\", \"phi\": 1, \"window\": null}, \"2\": {\"part\": \"trend\", \"lamb\": 129600}}}",
        "Ensemble": 0,
    },
    "48": {
        "Model": "ARDL",
        "ModelParameters": "{\"lags\": 4, \"trend\": \"n\", \"order\": 1, \"regression_type\": \"holiday\"}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"IntermittentOccurrence\"}, \"transformation_params\": {\"0\": {\"center\": \"mean\"}}}",
        "Ensemble": 0,
    },
    "49": {
        "Model": "MultivariateMotif",
        "ModelParameters": "{\"window\": 10, \"point_method\": \"median\", \"distance_metric\": \"mahalanobis\", \"k\": 20, \"max_windows\": null}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"QuantileTransformer\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {\"output_distribution\": \"normal\", \"n_quantiles\": 1000}, \"2\": {\"model\": \"Linear\", \"phi\": 1, \"window\": null}}}",
        "Ensemble": 0,
    },
    "50": {
        "Model": "MultivariateMotif",
        "ModelParameters": "{\"window\": 10, \"point_method\": \"median\", \"distance_metric\": \"sqeuclidean\", \"k\": 10, \"max_windows\": 10000}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean\", \"transformations\": {\"0\": \"Detrend\", \"1\": \"bkfilter\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"model\": \"GLS\", \"phi\": 1, \"window\": null}, \"1\": {}, \"2\": {\"model\": \"Linear\", \"phi\": 1, \"window\": null}}}",
        "Ensemble": 0,
    },
    "51": {
        "Model": "UnivariateMotif",
        "ModelParameters": "{\"window\": 60, \"point_method\": \"median\", \"distance_metric\": \"canberra\", \"k\": 10, \"max_windows\": 10000}",
        "TransformationParameters": "{\"fillna\": \"KNNImputer\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"SeasonalDifference\", \"2\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 100}, \"1\": {\"lag_1\": 7, \"method\": \"Mean\"}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "52": {
        "Model": "UnivariateMotif",
        "ModelParameters": "{\"window\": 14, \"point_method\": \"median\", \"distance_metric\": \"minkowski\", \"k\": 5, \"max_windows\": 10000}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"SeasonalDifference\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {\"lag_1\": 7, \"method\": \"LastValue\"}}}",
        "Ensemble": 0,
    },
    "53": {
        "Model": "SectionalMotif",
        "ModelParameters": "{\"window\": 10, \"point_method\": \"weighted_mean\", \"distance_metric\": \"sokalmichener\", \"include_differenced\": true, \"k\": 20, \"stride_size\": 1, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": null}, \"transformation_params\": {\"0\": {}}}",
        "Ensemble": 0,
    },
    "54": {
        "Model": "SectionalMotif",
        "ModelParameters": "{\"window\": 5, \"point_method\": \"midhinge\", \"distance_metric\": \"canberra\", \"include_differenced\": false, \"k\": 10, \"stride_size\": 1, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"QuantileTransformer\", \"2\": \"RobustScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "55": {
        "Model": "MultivariateRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"DecisionTree\", \"model_params\": {\"max_depth\": null, \"min_samples_split\": 2}}, \"mean_rolling_periods\": 30, \"macd_periods\": 7, \"std_rolling_periods\": 90, \"max_rolling_periods\": null, \"min_rolling_periods\": null, \"quantile90_rolling_periods\": 10, \"quantile10_rolling_periods\": 90, \"ewm_alpha\": null, \"ewm_var_alpha\": null, \"additional_lag_periods\": null, \"abs_energy\": false, \"rolling_autocorr_periods\": null, \"datepart_method\": \"recurring\", \"polynomial_degree\": null, \"regression_type\": null, \"window\": 10, \"holiday\": true, \"probabilistic\": false}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"SeasonalDifference\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {\"lag_1\": 7, \"method\": \"LastValue\"}, \"2\": {\"model\": \"Linear\", \"phi\": 1, \"window\": null}}}",
        "Ensemble": 0,
    },
    "56": {
        "Model": "FBProphet",
        "ModelParameters": "{\"holiday\": true, \"regression_type\": null, \"growth\": \"linear\", \"n_changepoints\": 25, \"changepoint_prior_scale\": 30, \"seasonality_mode\": \"multiplicative\", \"changepoint_range\": 0.9, \"seasonality_prior_scale\": 10.0, \"holidays_prior_scale\": 10.0}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"Slice\"}, \"transformation_params\": {\"0\": {\"method\": 0.5}}}",
        "Ensemble": 0,
    },
    "57": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"lastvalue\", \"lag_1\": 364, \"lag_2\": 30}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"MaxAbsScaler\", \"2\": \"Round\"}, \"transformation_params\": {\"0\": {\"lag_1\": 7, \"method\": \"LastValue\"}, \"1\": {}, \"2\": {\"decimals\": 0, \"on_transform\": false, \"on_inverse\": true}}}",
        "Ensemble": 0,
    },
    "58": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"RandomForest\", \"model_params\": {\"n_estimators\": 100, \"min_samples_leaf\": 2, \"bootstrap\": true}}, \"datepart_method\": \"expanded\", \"polynomial_degree\": null, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"Detrend\", \"2\": \"RobustScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"normal\", \"n_quantiles\": 20}, \"1\": {\"model\": \"Linear\", \"phi\": 1, \"window\": null}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "59": {
        'Model': 'NVAR',
        'ModelParameters': '{"k": 1, "ridge_param": 0.002, "warmup_pts": 1, "seed_pts": 1, "seed_weighted": null, "batch_size": 5, "batch_method": "input_order"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "MinMaxScaler"}, "transformation_params": {"0": {}}}',
        "Ensemble": 0,
    },
    "60": {
        'Model': 'Theta',
        'ModelParameters': '{"deseasonalize": true, "difference": false, "use_test": false, "method": "auto", "period": null, "theta": 2.5, "use_mle": true}',
        'TransformationParameters': '{"fillna": "mean", "transformations": {"0": "Detrend"}, "transformation_params": {"0": {"model": "Linear", "phi": 1, "window": 90}}}',
        "Ensemble": 0,
    },
    "61": {
        'Model': 'RollingRegression',
        'ModelParameters': '{"regression_model": {"model": "ExtraTrees", "model_params": {"n_estimators": 100, "min_samples_leaf": 1, "max_depth": 10}}, "holiday": false, "mean_rolling_periods": null, "macd_periods": null, "std_rolling_periods": null, "max_rolling_periods": null, "min_rolling_periods": 7, "ewm_var_alpha": null, "quantile90_rolling_periods": null, "quantile10_rolling_periods": null, "ewm_alpha": null, "additional_lag_periods": 95, "abs_energy": true, "rolling_autocorr_periods": null, "add_date_part": "expanded", "polynomial_degree": null, "x_transform": null, "regression_type": "User"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "MaxAbsScaler"}, "transformation_params": {"0": {}}}',
        'Ensemble': 0,
    },
    "62": {
        'Model': 'UnivariateRegression',
        'ModelParameters': '{"regression_model": {"model": "DecisionTree", "model_params": {"max_depth": null, "min_samples_split": 1.0}}, "holiday": false, "mean_rolling_periods": null, "macd_periods": null, "std_rolling_periods": null, "max_rolling_periods": 12, "min_rolling_periods": 58, "ewm_var_alpha": null, "ewm_alpha": 0.5, "additional_lag_periods": 363, "abs_energy": false, "rolling_autocorr_periods": null, "add_date_part": "simple_2_poly", "polynomial_degree": null, "x_transform": null, "regression_type": null, "window": null}',
        'TransformationParameters': '{"fillna": "rolling_mean", "transformations": {"0": "MaxAbsScaler", "1": "SeasonalDifference", "2": "Round"}, "transformation_params": {"0": {}, "1": {"lag_1": 7, "method": "LastValue"}, "2": {"decimals": -2, "on_transform": false, "on_inverse": true}}}',
        'Ensemble': 0,
    },
    "63": {
        'Model': 'MotifSimulation',
        'ModelParameters': '{"phrase_len": 10, "comparison": "magnitude_pct_change_sign", "shared": false, "distance_metric": "sokalmichener", "max_motifs": 0.2, "recency_weighting": 0.01, "cutoff_minimum": 20, "point_method": "median"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "MaxAbsScaler", "1": "Detrend"}, "transformation_params": {"0": {}, "1": {"model": "Linear", "phi": 1, "window": null}}}',
        'Ensemble': 0,
    },
    "64": {
        'Model': 'DynamicFactor',
        'ModelParameters': '{"k_factors": 0, "factor_order": 0, "regression_type": "User"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "MaxAbsScaler", "1": "EWMAFilter", "2": "QuantileTransformer"}, "transformation_params": {"0": {}, "1": {"span": 3}, "2": {"output_distribution": "uniform", "n_quantiles": 1000}}}',
        'Ensemble': 0,
    },
    "65": {
        'Model': 'RollingRegression',
        'ModelParameters': '{"regression_model": {"model": "ExtraTrees", "model_params": {"n_estimators": 100, "min_samples_leaf": 1, "max_depth": 10}}, "holiday": false, "mean_rolling_periods": null, "macd_periods": null, "std_rolling_periods": null, "max_rolling_periods": 420, "min_rolling_periods": 7, "ewm_var_alpha": null, "quantile90_rolling_periods": null, "quantile10_rolling_periods": null, "ewm_alpha": null, "additional_lag_periods": 363, "abs_energy": false, "rolling_autocorr_periods": null, "add_date_part": "expanded", "polynomial_degree": null, "x_transform": null, "regression_type": null}',
        'TransformationParameters': '{"fillna": "ffill_mean_biased", "transformations": {"0": "RobustScaler"}, "transformation_params": {"0": {}}}',
        'Ensemble': 0,
    },
    "66": {
        'Model': 'ARCH',
        'ModelParameters': '{"mean": "Zero", "lags": 1, "vol": "GARCH", "p": 4, "o": 1, "q": 2, "power": 1.5, "dist": "studentst", "rescale": true, "simulations": 1000, "maxiter": 200, "regression_type": null}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "SeasonalDifference"}, "transformation_params": {"0": {"lag_1": 7, "method": "LastValue"}}}',
        'Ensemble': 0,
    },
    "67": {
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
    "68": {
        'Model': 'SeasonalityMotif',
        'ModelParameters': '{"window": 5, "point_method": "weighted_mean", "distance_metric": "mae", "k": 10, "datepart_method": "common_fourier"}',
        'TransformationParameters': '{"fillna": "nearest", "transformations": {"0": "AlignLastValue"}, "transformation_params": {"0": {"rows": 1, "lag": 1, "method": "multiplicative", "strength": 1.0, "first_value_only": false}}}',
        'Ensemble': 0,
    },
    "69": {
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
    "70": {
        'Model': 'Cassandra',
        'ModelParameters': '''{"preprocessing_transformation": {"fillna": "ffill", "transformations": {"0": "SeasonalDifference", "1": "SeasonalDifference"}, "transformation_params": {"0": {"lag_1": 12, "method": "Median"}, "1": {"lag_1": 7, "method": "LastValue"}}}, "scaling": "BaseScaler", "past_impacts_intervention": null, "seasonalities": ["dayofmonthofyear", "weekend"], "ar_lags": null, "ar_interaction_seasonality": null, "anomaly_detector_params": {"method": "zscore", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "rolling_mean_24", "transform_dict": {"transformations": {"0": "DatepartRegression"}, "transformation_params": {"0": {"datepart_method": "simple_3", "regression_model": {"model": "DecisionTree", "model_params": {"max_depth": null, "min_samples_split": 0.1}}}}}}, "anomaly_intervention": "remove", "holiday_detector_params": {"threshold": 0.8, "splash_threshold": 0.85, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": false, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": false, "anomaly_detector_params": {"method": "mad", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "fake_date", "transform_dict": {"fillna": null, "transformations": {"0": "EWMAFilter"}, "transformation_params": {"0": {"span": 7}}}}, "remove_excess_anomalies": false, "impact": null, "regression_params": null, "output": "multivariate"}, "holiday_countries_used": false, "multivariate_feature": null, "multivariate_transformation": null, "regressor_transformation": null, "regressors_used": false, "linear_model": {"model": "lstsq", "lambda": 1, "recency_weighting": null}, "randomwalk_n": 10, "trend_window": 15, "trend_standin": "rolling_trend", "trend_anomaly_detector_params": {"method": "nonparametric", "method_params": {"p": null, "z_init": 1.5, "z_limit": 12, "z_step": 0.25, "inverse": false, "max_contamination": 0.25, "mean_weight": 25, "sd_weight": 25, "anomaly_count_weight": 1.0}, "fillna": "ffill", "transform_dict": {"fillna": "time", "transformations": {"0": "MinMaxScaler"}, "transformation_params": {"0": {}}}}, "trend_transformation": {"fillna": "ffill", "transformations": {"0": "AlignLastDiff", "1": "HPFilter"}, "transformation_params": {"0": {"rows": 1, "displacement_rows": 1, "quantile": 0.7, "decay_span": 3}, "1": {"part": "trend", "lamb": 129600}}}, "trend_model": {"Model": "LastValueNaive", "ModelParameters": {}}, "trend_phi": null}''',
        'TransformationParameters': '{"fillna": "linear", "transformations": {"0": "Round"}, "transformation_params": {"0": {"decimals": 0, "on_transform": true, "on_inverse": true}}}',
        'Ensemble': 0,
    },
    "71": {
        'Model': 'Cassandra',
        'ModelParameters': '''{"preprocessing_transformation": {"fillna": "mean", "transformations": {"0": "ClipOutliers", "1": "bkfilter"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 3.5, "fillna": null}, "1": {}}}, "scaling": "BaseScaler", "past_impacts_intervention": null, "seasonalities": [7, 365.25], "ar_lags": [7], "ar_interaction_seasonality": null, "anomaly_detector_params": null, "anomaly_intervention": null, "holiday_detector_params": {"threshold": 0.9, "splash_threshold": null, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": false, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": true, "anomaly_detector_params": {"method": "mad", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "rolling_mean_24", "transform_dict": {"fillna": null, "transformations": {"0": "EWMAFilter"}, "transformation_params": {"0": {"span": 7}}}}, "remove_excess_anomalies": false, "impact": null, "regression_params": null, "output": "multivariate"}, "holiday_countries_used": true, "multivariate_feature": null, "multivariate_transformation": null, "regressor_transformation": null, "regressors_used": false, "linear_model": {"model": "linalg_solve", "lambda": 1, "recency_weighting": 0.1}, "randomwalk_n": null, "trend_window": 365, "trend_standin": "rolling_trend", "trend_anomaly_detector_params": {"method": "mad", "method_params": {"distribution": "uniform", "alpha": 0.05}, "fillna": "rolling_mean_24", "transform_dict": {"fillna": null, "transformations": {"0": "EWMAFilter"}, "transformation_params": {"0": {"span": 7}}}}, "trend_transformation": {"fillna": "nearest", "transformations": {"0": "StandardScaler", "1": "AnomalyRemoval"}, "transformation_params": {"0": {}, "1": {"method": "IQR", "transform_dict": {"fillna": null, "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 6}}}, "method_params": {"iqr_threshold": 2.5, "iqr_quantiles": [0.25, 0.75]}, "fillna": "ffill"}}}, "trend_model": {"Model": "SectionalMotif", "ModelParameters": {"window": 50, "point_method": "mean", "distance_metric": "dice", "include_differenced": true, "k": 5, "stride_size": 1, "regression_type": null}}, "trend_phi": null}''',
        'TransformationParameters': '{"fillna": "pad", "transformations": {"0": "EWMAFilter", "1": "LevelShiftTransformer", "2": "StandardScaler", "3": "DatepartRegression"}, "transformation_params": {"0": {"span": 7}, "1": {"window_size": 7, "alpha": 2.0, "grouping_forward_limit": 2, "max_level_shifts": 5, "alignment": "average"}, "2": {}, "3": {"regression_model": {"model": "ElasticNet", "model_params": {}}, "datepart_method": "recurring", "polynomial_degree": null, "transform_dict": {"fillna": null, "transformations": {"0": "ScipyFilter"}, "transformation_params": {"0": {"method": "savgol_filter", "method_args": {"window_length": 31, "polyorder": 3, "deriv": 0, "mode": "interp"}}}}, "holiday_countries_used": false}}}',
        'Ensemble': 0,
    },
    "72": {  # optimized on M5, 58.5 SMAPE
        'Model': 'NeuralForecast',
        'ModelParameters': '''{"model": "MLP", "scaler_type": "minmax", "loss": "MQLoss", "learning_rate": 0.001, "max_steps": 100, "input_size": 28, "model_args": {"num_layers": 1, "hidden_size": 2560}, "regression_type": null}''',
        'TransformationParameters': '''{"fillna": "ffill", "transformations": {"0": "ClipOutliers", "1": "QuantileTransformer", "2": "SeasonalDifference", "3": "RobustScaler", "4": "ClipOutliers", "5": "MaxAbsScaler"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 3.5, "fillna": null}, "1": {"output_distribution": "normal", "n_quantiles": 100}, "2": {"lag_1": 7, "method": "Mean"}, "3": {}, "4": {"method": "clip", "std_threshold": 4, "fillna": null}, "5": {}}}''',
        'Ensemble': 0,
    },
    "73": {  # from production_example, mosaic most common, 2024-02-21
        'Model': 'Cassandra',
        'ModelParameters': '{"preprocessing_transformation": {"fillna": "ffill", "transformations": {"0": "RobustScaler", "1": "SeasonalDifference"}, "transformation_params": {"0": {}, "1": {"lag_1": 7, "method": "LastValue"}}}, "scaling": {"fillna": null, "transformations": {"0": "StandardScaler"}, "transformation_params": {"0": {}}}, "past_impacts_intervention": null, "seasonalities": ["simple_binarized"], "ar_lags": null, "ar_interaction_seasonality": null, "anomaly_detector_params": null, "anomaly_intervention": null, "holiday_detector_params": {"threshold": 1.0, "splash_threshold": null, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": true, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": false, "anomaly_detector_params": {"method": "zscore", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "mean", "transform_dict": null, "isolated_only": false}, "remove_excess_anomalies": false, "impact": null, "regression_params": null, "output": "multivariate"}, "holiday_countries_used": false, "multivariate_feature": null, "multivariate_transformation": null, "regressor_transformation": {"fillna": "nearest", "transformations": {"0": "AlignLastDiff"}, "transformation_params": {"0": {"rows": 7, "displacement_rows": 7, "quantile": 1.0, "decay_span": 2}}}, "regressors_used": true, "linear_model": {"model": "lstsq", "lambda": null, "recency_weighting": null}, "randomwalk_n": 10, "trend_window": 3, "trend_standin": null, "trend_anomaly_detector_params": null, "trend_transformation": {"fillna": "pchip", "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 3.5, "fillna": null}}}, "trend_model": {"Model": "ARDL", "ModelParameters": {"lags": 1, "trend": "c", "order": 0, "causal": false, "regression_type": "holiday"}}, "trend_phi": null}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "MaxAbsScaler", "1": "FFTDecomposition", "2": "bkfilter"}, "transformation_params": {"0": {}, "1": {"n_harmonics": 10, "detrend": "linear"}, "2": {}}}',
        'Ensemble': 0,
    },
    "74": {  # optimized 200 minutes on initial model import on load_daily
        "Model": "DMD",
        'ModelParameters': '{"rank": 10, "alpha": 1, "amplitude_threshold": null, "eigenvalue_threshold": null}',
        "TransformationParameters": '''{"fillna": "linear", "transformations": {"0": "HistoricValues", "1": "AnomalyRemoval", "2": "SeasonalDifference", "3": "AnomalyRemoval"},"transformation_params": {"0": {"window": 10}, "1": {"method": "zscore", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "ffill", "transform_dict": {"fillna": null, "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 6}}}, "isolated_only": false}, "2": {"lag_1": 7, "method": "Mean"}, "3": {"method": "zscore", "method_params": {"distribution": "norm", "alpha": 0.05}, "fillna": "fake_date", "transform_dict": {"transformations": {"0": "DifferencedTransformer"}, "transformation_params": {"0": {}}}, "isolated_only": false}}}''',
        "Ensemble": 0,
    },
    "75": {  # short optimization on M5
        "Model": "DMD",
        "ModelParameters": '''{"rank": 2, "alpha": 1, "amplitude_threshold": null, "eigenvalue_threshold": 1}''',
        "TransformationParameters": '''{"fillna": "ffill", "transformations": {"0": "SeasonalDifference", "1": "AlignLastValue", "2": "Round", "3": "Round", "4": "MinMaxScaler"}, "transformation_params": {"0": {"lag_1": 7, "method": "LastValue"}, "1": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false}, "2": {"decimals": 0, "on_transform": false, "on_inverse": true}, "3": {"decimals": 0, "on_transform": false, "on_inverse": true}, "4": {}}}''',
        "Ensemble": 0,
    },
    "76": {
        'Model': 'NVAR',
        'ModelParameters': '{"k": 2, "ridge_param": 2e-06, "warmup_pts": 1, "seed_pts": 1, "seed_weighted": null, "batch_size": 5, "batch_method": "std_sorted"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "HolidayTransformer", "1": "PositiveShift"}, "transformation_params": {"0": {"threshold": 0.9, "splash_threshold": null, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": false, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": false, "anomaly_detector_params": {"method": "rolling_zscore", "method_params": {"distribution": "uniform", "alpha": 0.03, "rolling_periods": 300, "center": true}, "fillna": "ffill", "transform_dict": {"fillna": "nearest", "transformations": {"0": null}, "transformation_params": {"0": {}}}, "isolated_only": false}, "remove_excess_anomalies": true, "impact": "datepart_regression", "regression_params": {"regression_model": {"model": "ElasticNet", "model_params": {"l1_ratio": 0.1, "fit_intercept": true, "selection": "cyclic"}}, "datepart_method": "simple", "polynomial_degree": null, "transform_dict": null, "holiday_countries_used": false}}, "1": {}}}',
        "Ensemble": 0,
    },
    "77": {  # optimized on M5, 60 SMAPE 2024-06-06
        'Model': 'BallTreeMultivariateMotif',
        'ModelParameters': '{"window": 28, "point_method": "median", "distance_metric": "euclidean", "k": 15}',
        'TransformationParameters': '{"fillna": "ffill_mean_biased", "transformations": {"0": "QuantileTransformer", "1": "QuantileTransformer", "2": "DatepartRegression"}, "transformation_params": 	{"0": {"output_distribution": "uniform", "n_quantiles": 1000}, "1": {"output_distribution": "normal", "n_quantiles": 100}, "2": {"regression_model": {"model": "ElasticNet", "model_params": {"l1_ratio": 0.1, "fit_intercept": true, "selection": "cyclic"}}, "datepart_method": "expanded", "polynomial_degree": null, "transform_dict": null, "holiday_countries_used": false}}}',
        "Ensemble": 0,
    },
    "78": {  # optimized on dap, 2.43 SMAPE 2024-06-20
        'Model': 'SectionalMotif',
        'ModelParameters': '{"window": 7, "point_method": "median", "distance_metric": "canberra", "include_differenced": true, "k": 5, "stride_size": 1, "regression_type": null}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "SeasonalDifference", "1": "AlignLastValue", "2": "SeasonalDifference", "3": "LevelShiftTransformer"}, "transformation_params": {"0": {"lag_1": 7, "method": "Median"}, "1": {"rows": 1, "lag": 2, "method": "additive", "strength": 1.0, "first_value_only": false}, "2": {"lag_1": 12, "method": 5}, "3": {"window_size": 30, "alpha": 2.0, "grouping_forward_limit": 4, "max_level_shifts": 10, "alignment": "average"}}}',
        "Ensemble": 0,
    },
    "79": {  # optimized on dap, 662000 MAE 2024-06-20
        'Model': 'FBProphet',
        'ModelParameters': '{"holiday": {"threshold": 1.0, "splash_threshold": null, "use_dayofmonth_holidays": true, "use_wkdom_holidays": true, "use_wkdeom_holidays": false, "use_lunar_holidays": false, "use_lunar_weekday": false, "use_islamic_holidays": false, "use_hebrew_holidays": false, "anomaly_detector_params": {"method": "IQR", "transform_dict": {"fillna": "time", "transformations": {"0": "Slice"}, "transformation_params": {"0": {"method": 0.5}}}, "forecast_params": {"model_name": "RRVAR", "model_param_dict": {"method": "als", "rank": 0.2, "maxiter": 200}, "model_transform_dict": {"fillna": "mean", "transformations": {"0": "DifferencedTransformer", "1": "bkfilter", "2": "AlignLastValue", "3": "ClipOutliers", "4": "SeasonalDifference"}, "transformation_params": {"0": {"lag": 1, "fill": "zero"}, "1": {}, "2": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false}, "3": {"method": "clip", "std_threshold": 2, "fillna": null}, "4": {"lag_1": 12, "method": "LastValue"}}}}, "method_params": {"iqr_threshold": 2.0, "iqr_quantiles": [0.4, 0.6]}}}, "regression_type": null, "growth": "linear", "n_changepoints": 10, "changepoint_prior_scale": 0.1, "seasonality_mode": "additive", "changepoint_range": 0.85, "seasonality_prior_scale": 0.01, "holidays_prior_scale": 10.0, "trend_phi": 1}',
        'TransformationParameters': '{"fillna": "rolling_mean", "transformations": {"0": "SeasonalDifference", "1": "bkfilter", "2": "SeasonalDifference", "3": "LevelShiftTransformer", "4": "AlignLastDiff"}, "transformation_params": {"0": {"lag_1": 7, "method": "Median"}, "1": {}, "2": {"lag_1": 12, "method": 5}, "3": {"window_size": 90, "alpha": 3.0, "grouping_forward_limit": 4, "max_level_shifts": 10, "alignment": "rolling_diff_3nn"}, "4": {"rows": null, "displacement_rows": 1, "quantile": 1.0, "decay_span": null}}}',
        "Ensemble": 0,
    },
    "80": {  # optimized on prod example daily, 13.5 SMAPE, 0.64 ODA, 42750 MAE, 2024-06-22
        'Model': 'ARDL',
        'ModelParameters': '{"lags": 3, "trend": "c", "order": 0, "causal": false, "regression_type": "simple_3"}',
        'TransformationParameters': '{"fillna": "ffill", "transformations": {"0": "ClipOutliers", "1": "Detrend"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 4.5, "fillna": null}, "1": {"model": "Linear", "phi": 1, "window": 90, "transform_dict": {"fillna": null, "transformations": {"0": "ClipOutliers"}, "transformation_params": {"0": {"method": "clip", "std_threshold": 4}}}}}}',
        "Ensemble": 0,
    },
    "81": {  # optimized on prod example daily, 15.67 SMAPE 2024-06-22
        'Model': 'FFT',
        'ModelParameters': '{"n_harmonics": 4, "detrend": "linear"}',
        'TransformationParameters': '{"fillna": "time", "transformations": {"0": "AlignLastValue"}, "transformation_params": {"0": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.9, "first_value_only": false}}}',
        "Ensemble": 0,
    },
    "82": {  # optimized on wiki example daily, 27 SMAPE 2024-10-05
        'Model': 'BasicLinearModel',
        'ModelParameters': '{"datepart_method": "simple_binarized", "changepoint_spacing": 360, "changepoint_distance_end": 360, "regression_type": null, "lambda_": 0.01}',
        'TransformationParameters': '{"fillna": "ffill_mean_biased", "transformations": {"0": "Round", "1": "AlignLastValue", "2": "HistoricValues", "3": "ClipOutliers", "4": "bkfilter"}, "transformation_params": {"0": {"decimals": -1, "on_transform": true, "on_inverse": true}, "1": {"rows": 7, "lag": 1, "method": "additive", "strength": 0.2, "first_value_only": true, "threshold": null, "threshold_method": "mean"}, "2": {"window": 10}, "3": {"method": "clip", "std_threshold": 3, "fillna": null}, "4": {}}}',
        "Ensemble": 0,
    },
    "83": {  # optimized on wiki example daily 26.8 SMAPE 2024-10-05
        'Model': 'BasicLinearModel',
        'ModelParameters': '{"datepart_method": ["dayofweek", [365.25, 14]], "changepoint_spacing": 90, "changepoint_distance_end": 360, "regression_type": null, "lambda_": null, "trend_phi": 0.98}',
        'TransformationParameters': '{"fillna": "piecewise_polynomial", "transformations": {"0": "AlignLastValue", "1": "IntermittentOccurrence", "2": "RobustScaler", "3": "Log"}, "transformation_params": {"0": {"rows": 1, "lag": 2, "method": "multiplicative", "strength": 0.9, "first_value_only": false, "threshold": 1, "threshold_method": "mean"}, "1": {"center": "mean"}, "2": {}, "3": {}}}',
        "Ensemble": 0,
    },
    "84": {  # optimized on VN1, best theta, 50 smape 0.44 competition
        'Model': 'Theta',
        'ModelParameters': '{"deseasonalize": true, "difference": false, "use_test": true, "method": "auto", "period": null, "theta": 1.4, "use_mle": false}',
        'TransformationParameters': '{"fillna": "quadratic", "transformations": {"0": "AlignLastValue", "1": "MinMaxScaler", "2": "HistoricValues", "3": "AlignLastValue"}, "transformation_params": {"0": {"rows": 7, "lag": 1, "method": "additive", "strength": 0.7, "first_value_only": false, "threshold": 10, "threshold_method": "max"}, "1": {}, "2": {"window": 10}, "3": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.2, "first_value_only": false, "threshold": null, "threshold_method": "max"}}}',
        "Ensemble": 0,
    },
    "85": {  # optimized on VN1, best wasserstein 80.5 (sample of 600)
        'Model': 'MultivariateRegression',
        'ModelParameters': '{"regression_model": {"model": "ExtraTrees", "model_params": {"n_estimators": 100, "min_samples_leaf": 1, "min_samples_split": 1.0, "max_depth": null, "criterion": "friedman_mse", "max_features": 1}}, "mean_rolling_periods": 12, "macd_periods": 2, "std_rolling_periods": null, "max_rolling_periods": 7, "min_rolling_periods": 364, "quantile90_rolling_periods": 10, "quantile10_rolling_periods": 5, "ewm_alpha": 0.2, "ewm_var_alpha": 0.5, "additional_lag_periods": null, "abs_energy": false, "rolling_autocorr_periods": null, "nonzero_last_n": null, "datepart_method": "recurring", "polynomial_degree": null, "regression_type": "User", "window": null, "holiday": false, "probabilistic": false, "scale_full_X": true, "cointegration": null, "cointegration_lag": 1, "series_hash": false, "frac_slice": null}',
        'TransformationParameters': '{"fillna": "ffill_mean_biased", "transformations": {"0": "AlignLastValue", "1": "PositiveShift", "2": "AlignLastValue"}, "transformation_params": {"0": {"rows": 7, "lag": 1, "method": "additive", "strength": 0.7, "first_value_only": false, "threshold": 10, "threshold_method": "max"}, "1": {}, "2": {"rows": 1, "lag": 1, "method": "additive", "strength": 1.0, "first_value_only": false, "threshold": 10, "threshold_method": "max"}}}',
        "Ensemble": 0,
    },
    "86": {  # optimized on VN1 best competition 0.43, good smape 0.49
        'Model': 'SeasonalityMotif',
        'ModelParameters': '{"window": 5, "point_method": "trimmed_mean_40", "distance_metric": "mae", "k": 5, "datepart_method": ["simple_binarized_poly"], "independent": true}',
        'TransformationParameters': '{"fillna": "rolling_mean", "transformations": {"0": "Constraint", "1": "QuantileTransformer", "2": "AlignLastValue"}, "transformation_params": {"0": {"constraint_method": "historic_diff", "constraint_direction": "lower", "constraint_regularization": 1.0, "constraint_value": 0.2, "bounds_only": false, "fillna": null}, "1": {"output_distribution": "uniform", "n_quantiles": 43}, "2": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.2, "first_value_only": false, "threshold": null, "threshold_method": "max"}}}',
        "Ensemble": 0,
    },
    "87": {  # optimized on wiki daily, 25.6 smape, 991k mqae, initial fit
        'Model': 'TVVAR',
        'ModelParameters': '{"datepart_method": "expanded", "changepoint_spacing": 90, "changepoint_distance_end": 520, "regression_type": null, "lags": [7], "rolling_means": [4], "lambda_": 0.001, "trend_phi": 0.99, "var_dampening": 0.98, "phi": null, "max_cycles": 2000, "apply_pca": true, "base_scaled": false, "x_scaled": false, "var_preprocessing": {"fillna": "rolling_mean", "transformations": {"0": "FFTFilter"}, "transformation_params": {"0": {"cutoff": 0.4, "reverse": false, "on_transform": true, "on_inverse": false}}}, "threshold_value": null}',
        'TransformationParameters': '{"fillna": "linear", "transformations": {"0": "QuantileTransformer", "1": "RobustScaler", "2": "PositiveShift", "3": "MinMaxScaler", "4": "AlignLastValue", "5": "ClipOutliers"}, "transformation_params": {"0": {"output_distribution": "normal", "n_quantiles": 100}, "1": {}, "2": {}, "3": {}, "4": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.5, "first_value_only": false, "threshold": 1, "threshold_method": "max"}, "5": {"method": "clip", "std_threshold": 4, "fillna": null}}}',
        "Ensemble": 0,
    },
    "88": {
        'Model': 'TVVAR',
        'ModelParameters': '{"datepart_method": "expanded", "changepoint_spacing": 6, "changepoint_distance_end": 520, "regression_type": null, "lags": [7], "rolling_means": [4], "lambda_": 0.001, "trend_phi": 0.99, "var_dampening": 0.98, "phi": null, "max_cycles": 2000, "apply_pca": true, "base_scaled": false, "x_scaled": false, "var_preprocessing": {"fillna": "rolling_mean", "transformations": {"0": "FFTFilter"}, "transformation_params": {"0": {"cutoff": 0.4, "reverse": false, "on_transform": true, "on_inverse": false}}}, "threshold_value": null}',
        'TransformationParameters': '{"fillna": "linear", "transformations": {"0": "QuantileTransformer", "1": "RobustScaler", "2": "AlignLastValue", "3": "MinMaxScaler", "4": "MinMaxScaler"}, "transformation_params": {"0": {"output_distribution": "normal", "n_quantiles": 100}, "1": {}, "2": {"rows": 1, "lag": 1, "method": "additive", "strength": 0.2, "first_value_only": false, "threshold": 1, "threshold_method": "max"}, "3": {}, "4": {}}}',
        'Ensemble': 0,
    },
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
