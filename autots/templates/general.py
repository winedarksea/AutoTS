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
        "ModelParameters": "{\"p\": 12, \"d\": 3, \"q\": 7, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "3": {
        "Model": "ARIMA",
        "ModelParameters": "{\"p\": 12, \"d\": 1, \"q\": 7, \"regression_type\": \"Holiday\"}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}}}",
        "Ensemble": 0,
    },
    "4": {
        "Model": "ARIMA",
        "ModelParameters": "{\"p\": 12, \"d\": 0, \"q\": 12, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"Detrend\", \"1\": \"DifferencedTransformer\", \"2\": \"ClipOutliers\", \"3\": \"RobustScaler\", \"4\": \"StandardScaler\"}, \"transformation_params\": {\"0\": {\"model\": \"GLS\"}, \"1\": {}, \"2\": {\"method\": \"clip\", \"std_threshold\": 4, \"fillna\": null}, \"3\": {}, \"4\": {}}}",
        "Ensemble": 0,
    },
    "5": {
        "Model": "AverageValueNaive",
        "ModelParameters": "{\"method\": \"Mean\"}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"DifferencedTransformer\", \"1\": \"SinTrend\"}, \"transformation_params\": {\"0\": {}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "6": {
        "Model": "AverageValueNaive",
        "ModelParameters": "{\"method\": \"Mean\"}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "7": {
        "Model": "AverageValueNaive",
        "ModelParameters": "{\"method\": \"Mean\"}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"Round\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"lag_1\": 7, \"method\": \"Mean\"}, \"1\": {\"model\": \"middle\", \"decimals\": 2, \"on_transform\": true, \"on_inverse\": false}, \"2\": {\"model\": \"GLS\"}}}",
        "Ensemble": 0,
    },
    "8": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"DecisionTree\", \"model_params\": {\"max_depth\": 3, \"min_samples_split\": 2}}, \"datepart_method\": \"recurring\", \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "9": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"SVM\", \"model_params\": {}}, \"datepart_method\": \"recurring\", \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"RollingMeanTransformer\", \"2\": \"Detrend\", \"3\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3.5, \"fillna\": null}, \"1\": {\"fixed\": true, \"window\": 10}, \"2\": {\"model\": \"Linear\"}, \"3\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "10": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"MLP\", \"model_params\": {\"hidden_layer_sizes\": [25, 15, 25], \"max_iter\": 1000, \"activation\": \"tanh\", \"solver\": \"lbfgs\", \"early_stopping\": false, \"learning_rate_init\": 0.001}}, \"datepart_method\": \"recurring\", \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "11": {
        "Model": "DatepartRegression",
        "ModelParameters": "{\"regression_model\": {\"model\": \"KerasRNN\", \"model_params\": {\"kernel_initializer\": \"glorot_uniform\", \"epochs\": 50, \"batch_size\": 32, \"optimizer\": \"adam\", \"loss\": \"Huber\", \"hidden_layer_sizes\": [32, 32, 32], \"rnn_type\": \"GRU\"}}, \"datepart_method\": \"recurring\", \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"Round\", \"1\": \"QuantileTransformer\", \"2\": \"QuantileTransformer\", \"3\": \"QuantileTransformer\", \"4\": \"PowerTransformer\"}, \"transformation_params\": {\"0\": {\"model\": \"middle\", \"decimals\": 1, \"on_transform\": true, \"on_inverse\": false}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 20}, \"3\": {\"output_distribution\": \"normal\", \"n_quantiles\": 1000}, \"4\": {}}}",
        "Ensemble": 0,
    },
    "12": {
        "Model": "ETS",
        "ModelParameters": "{\"damped_trend\": false, \"trend\": \"additive\", \"seasonal\": null, \"seasonal_periods\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "13": {
        "Model": "ETS",
        "ModelParameters": "{\"damped_trend\": false, \"trend\": null, \"seasonal\": \"additive\", \"seasonal_periods\": 7}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"Round\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"model\": \"middle\", \"decimals\": 0, \"on_transform\": false, \"on_inverse\": true}, \"2\": {\"model\": \"GLS\"}}}",
        "Ensemble": 0,
    },
    "14": {
        "Model": "GLM",
        "ModelParameters": "{\"family\": \"Binomial\", \"constant\": false, \"regression_type\": \"datepart\"}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 4, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "15": {
        "Model": "GLM",
        "ModelParameters": "{\"family\": \"Binomial\", \"constant\": false, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "16": {
        "Model": "GLS",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean\", \"transformations\": {\"0\": \"RollingMeanTransformer\", \"1\": \"DifferencedTransformer\", \"2\": \"Detrend\", \"3\": \"Slice\"}, \"transformation_params\": {\"0\": {\"fixed\": true, \"window\": 3}, \"1\": {}, \"2\": {\"model\": \"Linear\"}, \"3\": {\"method\": 100}}}",
        "Ensemble": 0,
    },
    "17": {
        "Model": "GLS",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"median\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"RobustScaler\", \"3\": \"Round\", \"4\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3.5, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}, \"3\": {\"model\": \"middle\", \"decimals\": 2, \"on_transform\": true, \"on_inverse\": true}, \"4\": {}}}",
        "Ensemble": 0,
    },
    "18": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"DeepAR\", \"epochs\": 150, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"PowerTransformer\", \"1\": \"DifferencedTransformer\", \"2\": \"bkfilter\"}, \"transformation_params\": {\"0\": {}, \"1\": {}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "19": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"NPTS\", \"epochs\": 20, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"Detrend\", \"1\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {\"model\": \"Linear\"}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "20": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"WaveNet\", \"epochs\": 40, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "21": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"NPTS\", \"epochs\": 20, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"PowerTransformer\", \"3\": \"SeasonalDifference\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}, \"3\": {\"lag_1\": 7, \"method\": \"Median\"}}}",
        "Ensemble": 0,
    },
    "22": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"Transformer\", \"epochs\": 40, \"learning_rate\": 0.001, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"MaxAbsScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "23": {
        "Model": "GluonTS",
        "ModelParameters": "{\"gluon_model\": \"SFF\", \"epochs\": 40, \"learning_rate\": 0.01, \"context_length\": 10}",
        "TransformationParameters": "{\"fillna\": \"ffill_mean_biased\", \"transformations\": {\"0\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "24": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"bkfilter\", \"1\": \"SinTrend\", \"2\": \"Detrend\", \"3\": \"PowerTransformer\"}, \"transformation_params\": {\"0\": {}, \"1\": {}, \"2\": {\"model\": \"Linear\"}, \"3\": {}}}",
        "Ensemble": 0,
    },
    "25": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"PositiveShift\", \"1\": \"SinTrend\", \"2\": \"bkfilter\"}, \"transformation_params\": {\"0\": {}, \"1\": {}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "26": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"SinTrend\"}, \"transformation_params\": {\"0\": {\"lag_1\": 7, \"method\": \"LastValue\"}, \"1\": {}}}",
        "Ensemble": 0,
    },
    "27": {
        "Model": "LastValueNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 1, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "28": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"LastValue\", \"lag_1\": 2, \"lag_2\": 7}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"SinTrend\", \"1\": \"Round\", \"2\": \"PowerTransformer\"}, \"transformation_params\": {\"0\": {}, \"1\": {\"model\": \"middle\", \"decimals\": 2, \"on_transform\": false, \"on_inverse\": true}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "29": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"LastValue\", \"lag_1\": 96, \"lag_2\": 4}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"MinMaxScaler\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "30": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"LastValue\", \"lag_1\": 2, \"lag_2\": 1}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"SeasonalDifference\", \"1\": \"QuantileTransformer\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"lag_1\": 12, \"method\": \"Median\"}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"model\": \"GLS\"}}}",
        "Ensemble": 0,
    },
    "31": {
        "Model": "SeasonalNaive",
        "ModelParameters": "{\"method\": \"LastValue\", \"lag_1\": 7, \"lag_2\": 2}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"ClipOutliers\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"1\": {\"method\": \"clip\", \"std_threshold\": 2, \"fillna\": null}}}",
        "Ensemble": 0,
    },
    "32": {
        "Model": "UnobservedComponents",
        "ModelParameters": "{\"level\": true, \"trend\": false, \"cycle\": true, \"damped_cycle\": true, \"irregular\": true, \"stochastic_trend\": false, \"stochastic_level\": true, \"stochastic_cycle\": true, \"regression_type\": \"Holiday\"}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"PositiveShift\", \"1\": \"Detrend\", \"2\": \"bkfilter\", \"3\": \"DifferencedTransformer\"}, \"transformation_params\": {\"0\": {}, \"1\": {\"model\": \"Linear\"}, \"2\": {}, \"3\": {}}}",
        "Ensemble": 0,
    },
    "33": {
        "Model": "UnobservedComponents",
        "ModelParameters": "{\"level\": false, \"trend\": false, \"cycle\": true, \"damped_cycle\": false, \"irregular\": false, \"stochastic_trend\": false, \"stochastic_level\": true, \"stochastic_cycle\": false, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"RobustScaler\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "34": {
        "Model": "UnobservedComponents",
        "ModelParameters": "{\"level\": true, \"trend\": false, \"cycle\": false, \"damped_cycle\": false, \"irregular\": false, \"stochastic_trend\": false, \"stochastic_level\": true, \"stochastic_cycle\": true, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"rolling_mean_24\", \"transformations\": {\"0\": \"ClipOutliers\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 5, \"fillna\": null}}}",
        "Ensemble": 0,
    },
    "35": {
        "Model": "VAR",
        "ModelParameters": "{\"regression_type\": null, \"maxlags\": null, \"ic\": \"fpe\"}",
        "TransformationParameters": "{\"fillna\": \"fake_date\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3.5, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "36": {
        "Model": "VAR",
        "ModelParameters": "{\"regression_type\": null, \"maxlags\": 5, \"ic\": \"fpe\"}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}}}",
        "Ensemble": 0,
    },
    "37": {
        "Model": "VAR",
        "ModelParameters": "{\"regression_type\": null, \"maxlags\": 15, \"ic\": \"aic\"}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"RollingMeanTransformer\", \"1\": \"QuantileTransformer\", \"2\": \"RollingMeanTransformer\"}, \"transformation_params\": {\"0\": {\"fixed\": true, \"window\": 10}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"fixed\": false, \"window\": 10}}}",
        "Ensemble": 0,
    },
    "38": {
        "Model": "VECM",
        "ModelParameters": "{\"deterministic\": \"cili\", \"k_ar_diff\": 2, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"Detrend\", \"2\": \"Detrend\", \"3\": \"PowerTransformer\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"model\": \"GLS\"}, \"2\": {\"model\": \"Linear\"}, \"3\": {}}}",
        "Ensemble": 0,
    },
    "39": {
        "Model": "VECM",
        "ModelParameters": "{\"deterministic\": \"li\", \"k_ar_diff\": 3, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"model\": \"Linear\"}}}",
        "Ensemble": 0,
    },
    "40": {
        "Model": "VECM",
        "ModelParameters": "{\"deterministic\": \"lo\", \"k_ar_diff\": 2, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"ClipOutliers\", \"2\": \"QuantileTransformer\", \"3\": \"Discretize\", \"4\": \"RobustScaler\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"2\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"3\": {\"discretization\": \"lower\", \"n_bins\": 10}, \"4\": {}}}",
        "Ensemble": 0,
    },
    "41": {
        "Model": "VECM",
        "ModelParameters": "{\"deterministic\": \"colo\", \"k_ar_diff\": 3, \"regression_type\": null}",
        "TransformationParameters": "{\"fillna\": \"zero\", \"transformations\": {\"0\": \"ClipOutliers\", \"1\": \"QuantileTransformer\", \"2\": \"Detrend\"}, \"transformation_params\": {\"0\": {\"method\": \"clip\", \"std_threshold\": 3, \"fillna\": null}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"model\": \"Linear\"}}}",
        "Ensemble": 0,
    },
    "42": {
        "Model": "WindowRegression",
        "ModelParameters": "{\"window_size\": 10, \"regression_model\": {\"model\": \"MLP\", \"model_params\": {\"hidden_layer_sizes\": [72, 36, 72], \"max_iter\": 250, \"activation\": \"relu\", \"solver\": \"lbfgs\", \"early_stopping\": false, \"learning_rate_init\": 0.001}}, \"input_dim\": \"univariate\", \"output_dim\": \"forecast_length\", \"normalize_window\": false, \"shuffle\": true, \"max_windows\": 5000}",
        "TransformationParameters": "{\"fillna\": \"mean\", \"transformations\": {\"0\": \"QuantileTransformer\", \"1\": \"MinMaxScaler\", \"2\": \"RobustScaler\"}, \"transformation_params\": {\"0\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 100}, \"1\": {}, \"2\": {}}}",
        "Ensemble": 0,
    },
    "43": {
        "Model": "ZeroesNaive",
        "ModelParameters": "{}",
        "TransformationParameters": "{\"fillna\": \"ffill\", \"transformations\": {\"0\": \"PowerTransformer\", \"1\": \"QuantileTransformer\", \"2\": \"SeasonalDifference\"}, \"transformation_params\": {\"0\": {}, \"1\": {\"output_distribution\": \"uniform\", \"n_quantiles\": 1000}, \"2\": {\"lag_1\": 7, \"method\": \"LastValue\"}}}",
        "Ensemble": 0,
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
"""
