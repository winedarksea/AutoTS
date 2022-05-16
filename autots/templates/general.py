"""Starting templates for models."""
import pandas as pd

general_template_dict = {
    "0": {
        "Model": "ARIMA",
        "ModelParameters": "{\"p\": 7, \"d\": 1, \"q\": 4, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"ClipOutliers\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3.5, \"fillna\": null}}}",
        "Ensemble": 0,
    },
    "1": {
        "Model": "ARIMA",
        "ModelParameters": "{\"p\": 4, \"d\": 1, \"q\": 7, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "2": {
        "Model": "ARIMA",
        "ModelParameters": "{\"p\": 12, \"d\": 1, \"q\": 7, \"regression_type\": \"Holiday\"}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}}}",
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
    ### Gen 2
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
        'Model': 'UnivariateRegression',
        'ModelParameters': '{"regression_model": {"model": "KNN", "model_params": {"n_neighbors": 3, "weights": "uniform"}}, "holiday": false, "mean_rolling_periods": 5, "macd_periods": 28, "std_rolling_periods": null, "max_rolling_periods": 24, "min_rolling_periods": null, "ewm_var_alpha": null, "ewm_alpha": 0.5, "additional_lag_periods": 11, "abs_energy": false, "rolling_autocorr_periods": null, "add_date_part": null, "polynomial_degree": null, "x_transform": null, "regression_type": null, "window": null}',
        'TransformationParameters': '{"fillna": "rolling_mean", "transformations": {"0": "QuantileTransformer", "1": "QuantileTransformer"}, "transformation_params": {"0": {"output_distribution": "normal", "n_quantiles": 1000}, "1": {"output_distribution": "uniform", "n_quantiles": 20}}}',
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
