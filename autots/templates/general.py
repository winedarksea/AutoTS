import pandas as pd

general_template_dict = {
    0: {
        'Model': 'UnobservedComponents',
        'ModelParameters': '{"level": true, "trend": true, "cycle": true, "damped_cycle": true, "irregular": true, "stochastic_trend": true, "stochastic_level": true, "stochastic_cycle": false, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": "clip", "outlier_threshold": 3, "outlier_position": "first", "fillna": "rolling mean", "transformation": "PowerTransformer", "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": null, "transformation_param2": null, "fourth_transformation": null, "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    1: {
        'Model': 'UnobservedComponents',
        'ModelParameters': '{"level": true, "trend": false, "cycle": false, "damped_cycle": false, "irregular": false, "stochastic_trend": true, "stochastic_level": true, "stochastic_cycle": false, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": "clip", "outlier_threshold": 2, "outlier_position": "first", "fillna": "rolling mean", "transformation": "PowerTransformer", "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": null, "transformation_param2": null, "fourth_transformation": null, "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    2: {
        'Model': 'ETS',
        'ModelParameters': '{"damped": false, "trend": "additive", "seasonal": null, "seasonal_periods": null}',
        'TransformationParameters': '{"outlier_method": "remove", "outlier_threshold": 3, "outlier_position": "first", "fillna": "median", "transformation": "PowerTransformer", "second_transformation": null, "transformation_param": null, "detrend": "Linear", "third_transformation": "RollingMean", "transformation_param2": 10, "fourth_transformation": null, "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    3: {
        'Model': 'VECM',
        'ModelParameters': '{"deterministic": "ci", "k_ar_diff": 1, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": "remove", "outlier_threshold": 3, "outlier_position": "first", "fillna": "ffill", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": "RollingMean", "transformation_param2": "100thN", "fourth_transformation": "10", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    4: {
        'Model': 'VECM',
        'ModelParameters': '{"deterministic": "colo", "k_ar_diff": 1, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": null, "outlier_threshold": null, "outlier_position": "first", "fillna": "rolling mean", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": "Linear", "third_transformation": "RollingMean", "transformation_param2": "100thN", "fourth_transformation": "DifferencedTransformer", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    5: {
        'Model': 'VARMAX',
        'ModelParameters': '{"order": [2, 0], "trend": [1, 1, 0, 1]}',
        'TransformationParameters': '{"outlier_method": null, "outlier_threshold": null, "outlier_position": "first", "fillna": "fake date", "transformation": "PowerTransformer", "second_transformation": null, "transformation_param": null, "detrend": "Linear", "third_transformation": null, "transformation_param2": null, "fourth_transformation": null, "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    6: {
        'Model': 'FBProphet',
        'ModelParameters': '{"holiday": false, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": null, "outlier_threshold": null, "outlier_position": "first", "fillna": "fake date", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": "Linear", "third_transformation": "RollingMean", "transformation_param2": "100thN", "fourth_transformation": "DifferencedTransformer", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    7: {
        'Model': 'AverageValueNaive',
        'ModelParameters': '{"method": "Median"}',
        'TransformationParameters': '{"outlier_method": null, "outlier_threshold": null, "outlier_position": "first", "fillna": "fake date", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": null, "transformation_param2": null, "fourth_transformation": null, "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    8: {
        'Model': 'LastValueNaive',
        'ModelParameters': '{}',
        'TransformationParameters': '{"outlier_method": null, "outlier_threshold": null, "outlier_position": "first", "fillna": "mean", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": null, "transformation_param2": null, "fourth_transformation": null, "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    9: {
        'Model': 'LastValueNaive',
        'ModelParameters': '{}',
        'TransformationParameters': '{"outlier_method": "clip", "outlier_threshold": 4, "outlier_position": "first", "fillna": "fake date", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": "Linear", "third_transformation": null, "transformation_param2": null, "fourth_transformation": null, "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": "HalfMax"}',
        'Ensemble': 0,
    },
    10: {
        'Model': 'RollingRegression',
        'ModelParameters': '{"regression_model": {"model": "BayesianRidge", "model_params": {}}, "holiday": true, "mean_rolling_periods": 7, "macd_periods": null, "std_rolling_periods": 5, "max_rolling_periods": 10, "min_rolling_periods": 30, "ewm_alpha": 0.8, "additional_lag_periods": 30, "abs_energy": false, "rolling_autocorr_periods": null, "add_date_part": "expanded", "polynomial_degree": null, "x_transform": null, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": "clip", "outlier_threshold": 2, "outlier_position": "first", "fillna": "ffill", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": null, "transformation_param2": null, "fourth_transformation": "PowerTransformer", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    11: {
        'Model': 'RollingRegression',
        'ModelParameters': '{"regression_model": {"model": "DecisionTree", "model_params": {"max_depth": 3, "min_samples_split": 2}}, "holiday": true, "mean_rolling_periods": 7, "macd_periods": null, "std_rolling_periods": 5, "max_rolling_periods": 30, "min_rolling_periods": 7, "ewm_alpha": 0.8, "additional_lag_periods": 6, "abs_energy": true, "rolling_autocorr_periods": null, "add_date_part": "simple", "polynomial_degree": null, "x_transform": null, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": "clip", "outlier_threshold": 3, "outlier_position": "first", "fillna": "ffill", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": null, "transformation_param2": null, "fourth_transformation": "PowerTransformer", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": "6ForecastLength"}',
        'Ensemble': 0,
    },
    12: {
        'Model': 'RollingRegression',
        'ModelParameters': '{"regression_model": {"model": "MLP", "model_params": {"hidden_layer_sizes": [25, 15, 25], "max_iter": 250, "activation": "tanh", "solver": "lbfgs", "early_stopping": false, "learning_rate_init": 0.001}}, "holiday": false, "mean_rolling_periods": 5, "macd_periods": 30, "std_rolling_periods": 7, "max_rolling_periods": 30, "min_rolling_periods": null, "ewm_alpha": null, "additional_lag_periods": 2, "abs_energy": false, "rolling_autocorr_periods": 6, "add_date_part": null, "polynomial_degree": null, "x_transform": null, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": null, "outlier_threshold": null, "outlier_position": null, "fillna": "mean", "transformation": "DifferencedTransformer", "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": null, "transformation_param2": null, "fourth_transformation": "QuantileTransformer", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    13: {
        'Model': 'RollingRegression',
        'ModelParameters': '{"regression_model": {"model": "DecisionTree", "model_params": {"max_depth": null, "min_samples_split": 0.05}}, "holiday": false, "mean_rolling_periods": 30, "macd_periods": 7, "std_rolling_periods": 10, "max_rolling_periods": 7, "min_rolling_periods": 2, "ewm_alpha": 0.8, "additional_lag_periods": null, "abs_energy": false, "rolling_autocorr_periods": null, "add_date_part": null, "polynomial_degree": null, "x_transform": null, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": null, "outlier_threshold": null, "outlier_position": null, "fillna": "ffill", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": null, "transformation_param2": null, "fourth_transformation": "MaxAbsScaler", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    14: {
        'Model': 'RollingRegression',
        'ModelParameters': '{"regression_model": {"model": "KerasRNN", "model_params": {"kernel_initializer": "lecun_uniform", "epochs": 50, "batch_size": 32, "optimizer": "adam", "loss": "mae", "hidden_layer_sizes": [32, 32, 32], "rnn_type": "LSTM"}}, "holiday": false, "mean_rolling_periods": 5, "macd_periods": 7, "std_rolling_periods": null, "max_rolling_periods": 4, "min_rolling_periods": 28, "ewm_alpha": null, "additional_lag_periods": 3, "abs_energy": false, "rolling_autocorr_periods": null, "add_date_part": null, "polynomial_degree": null, "x_transform": null, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": null, "outlier_threshold": null, "outlier_position": null, "fillna": "ffill", "transformation": "DifferencedTransformer", "second_transformation": null, "transformation_param": null, "detrend": "Linear", "third_transformation": "FixedRollingMean", "transformation_param2": "100thN", "fourth_transformation": "StandardScaler", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    15: {
        'Model': 'GLM',
        'ModelParameters': '{"family": "NegativeBinomial", "constant": false, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": null, "outlier_threshold": null, "outlier_position": "first", "fillna": "ffill", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": null, "transformation_param2": null, "fourth_transformation": "MinMaxScaler", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": "HalfMax"}',
        'Ensemble': 0,
    },
    16: {
        'Model': 'GLM',
        'ModelParameters': '{"family": "Gaussian", "constant": true, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": "remove", "outlier_threshold": 4, "outlier_position": "first;last", "fillna": "zero", "transformation": "DifferencedTransformer", "second_transformation": null, "transformation_param": null, "detrend": "Linear", "third_transformation": "SeasonalDifferenceMean", "transformation_param2": "28", "fourth_transformation": "IntermittentOccurrence", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    17: {
        'Model': 'MotifSimulation',
        'ModelParameters': '{"phrase_len": "20", "comparison": "pct_change_sign", "shared": false, "distance_metric": "hamming", "max_motifs": 0.2, "recency_weighting": 0.0,"cutoff_threshold": 0.9, "cutoff_minimum": 50, "point_method": "median"}',
        'TransformationParameters': '{"outlier_method": null, "outlier_threshold": null, "outlier_position": "first", "fillna": "ffill", "transformation": "StandardScaler", "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": "RollingMean", "transformation_param2": "10", "fourth_transformation": null, "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    18: {
        'Model': 'GLS',
        'ModelParameters': '{"family": "Gaussian", "constant": true, "regression_type": null}',
        'TransformationParameters': '{"outlier_method": "remove", "outlier_threshold": 3, "outlier_position": "middle", "fillna": "median", "transformation": "QuantileTransformer", "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": "RobustScaler", "transformation_param2": null, "fourth_transformation": "PowerTransformer", "discretization": "upper", "n_bins": 10, "coerce_integer": false, "context_slicer": "6ForecastLength"}',
        'Ensemble': 0,
    },
    19: {
        'Model': 'TensorflowSTS',
        'ModelParameters': '{"fit_method": "hmc", "num_steps": 200, "ar_order": 7, "seasonal_periods": 7, "trend": "semilocal"}',
        'TransformationParameters': '{"outlier_method": "clip", "outlier_threshold": 3, "outlier_position": "first;middle", "fillna": "mean", "transformation": "PowerTransformer", "second_transformation": null, "transformation_param": null, "detrend": null, "third_transformation": "FixedRollingMean", "transformation_param2": "100thN", "fourth_transformation": "RobustScaler", "discretization": null, "n_bins": null, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
    20: {
        'Model': 'TensorflowSTS',
        'ModelParameters': '{"fit_method": "variational", "num_steps": 50, "ar_order": 1, "seasonal_periods": 364, "trend": "local"}',
        'TransformationParameters': '{"outlier_method": "clip", "outlier_threshold": 3, "outlier_position": "first", "fillna": "ffill", "transformation": null, "second_transformation": null, "transformation_param": null, "detrend": "Linear", "third_transformation": "FixedRollingMean", "transformation_param2": "14", "fourth_transformation": "SinTrend", "discretization": "sklearn-uniform", "n_bins": 5, "coerce_integer": false, "context_slicer": null}',
        'Ensemble': 0,
    },
}

general_template = pd.DataFrame.from_dict(general_template_dict, orient='index')
