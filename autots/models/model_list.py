"""Lists of models grouped by aspects."""

all_models = [
    'ConstantNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'GLM',
    'ETS',
    'ARIMA',
    'FBProphet',
    'RollingRegression',
    'GluonTS',
    'SeasonalNaive',
    'UnobservedComponents',
    # 'VARMAX',
    'VECM',
    'DynamicFactor',
    'MotifSimulation',
    'WindowRegression',
    'VAR',
    'DatepartRegression',
    "UnivariateRegression",
    # "Greykite",
    'UnivariateMotif',
    'MultivariateMotif',
    'NVAR',
    'MultivariateRegression',
    'SectionalMotif',
    'Theta',
    'ARDL',
    'NeuralProphet',
    'DynamicFactorMQ',
    'PytorchForecasting',
    'ARCH',
    'RRVAR',
    'MAR',
    'TMF',
    'LATC',
    'KalmanStateSpace',
    'MetricMotif',
    'Cassandra',
    'SeasonalityMotif',
    'MLEnsemble',
    'PreprocessingRegression',
    'FFT',
    "BallTreeMultivariateMotif",
    "TiDE",
    "NeuralForecast",
    "DMD",  # 45 models
    "BasicLinearModel",
    "TVVAR",
    "BallTreeRegressionMotif",
]
# used for graphing, not for model selection
model_classes = {
    'ARDL': 'stat',
    'DatepartRegression': 'ML',
    'ETS': 'stat',
    'FBProphet': 'stat',
    'GLM': 'stat',
    'GLS': 'stat',
    'MAR': 'stat',
    'MultivariateMotif': 'motif',
    'MultivariateRegression': 'ML',
    'NVAR': 'stat',
    'RRVAR': 'stat',
    'SectionalMotif': 'motif',
    'Theta': 'stat',
    'UnivariateMotif': 'motif',
    'UnivariateRegression': 'ML',
    'VAR': 'stat',
    'VECM': 'stat',
    'WindowRegression': 'ML',
    'BallTreeMultivariateMotif': 'motif',
    'MetricMotif': 'motif',
    'ARCH': 'stat',
    'KalmanStateSpace': 'stat',
    'ARIMA': 'stat',
    'BasicLinearModel': "stat",
    "Cassandra": "stat",
    "DMD": 'stat',
    "DynamicFactor": "stat",
    "DynamicFactorMQ": "stat",
    "FFT": "stat",
    "GluonTS": "DL",
    "LATC": "stat",
    "MotifSimulation": "motif",
    "NeuralForecast": "DL",
    "NeuralProphet": "DL",
    "PreprocessingRegression": "ML",
    "PytorchForecasting": "DL",
    "RollingRegression": "ML",
    "SeasonalityMotif": "motif",
    "TMF": "stat",
    "TiDE": "DL",
    "UnobservedComponents": "stat",
    'AverageValueNaive': 'naive',
    'ConstantNaive': 'naive',
    'LastValueNaive': 'naive',
    'SeasonalNaive': 'naive',
    'ZeroesNaive': 'naive',
    "TVVAR": "stat",
    "BallTreeRegressionMotif": "motif",
}
all_pragmatic = list((set(all_models) - set(['MLEnsemble', 'VARMAX', 'Greykite'])))
# downweight slower models
default = {
    'ConstantNaive': 1,
    'LastValueNaive': 1,
    'AverageValueNaive': 1,
    'GLS': 1,
    'SeasonalNaive': 1,
    'GLM': 1,
    'ETS': 1,
    'FBProphet': 0.5,
    # 'GluonTS': 0.5,
    'UnobservedComponents': 0.6,
    'VAR': 1,
    'VECM': 1,
    'ARIMA': 0.3,
    'WindowRegression': 0.8,
    'DatepartRegression': 1,
    # 'UnivariateRegression': 0.1,
    'MultivariateRegression': 0.4,
    'UnivariateMotif': 1,
    'MultivariateMotif': 1,
    'SectionalMotif': 1,
    'NVAR': 0.4,
    'Theta': 1,
    'ARDL': 1,
    'ARCH': 1,
    'MetricMotif': 1,
    'SeasonalityMotif': 1,
    'DMD': 0.3,
    'RRVAR': 0.8,
    'FFT': 0.8,
    'Cassandra': 0.8,
    'BasicLinearModel': 0.8,
    'TVVAR': 0.4,
    "BallTreeMultivariateMotif": 0.4,
    # "BallTreeRegressionMotif",
}
# fastest models at any scale
superfast = [
    'ConstantNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'SeasonalNaive',
    'SeasonalityMotif',
    'SectionalMotif',  # not entirely sure but so far this is pretty fast
    'BasicLinearModel',  # also on the riskier end
]
# relatively fast
fast = {
    'ConstantNaive': 1,
    'LastValueNaive': 1.5,
    'AverageValueNaive': 1,
    'GLS': 1,
    'SeasonalNaive': 1,
    'GLM': 1,
    'ETS': 1,
    'VAR': 0.8,
    'VECM': 0.8,  # a bit on the higher side for memory usage
    'WindowRegression': 0.5,  # this gets slow with Transformer, KerasRNN
    'DatepartRegression': 0.8,
    'UnivariateMotif': 1,
    # 'MultivariateMotif': 0.8,  # RAM issues at scale it seems
    'SectionalMotif': 1,
    'NVAR': 0.3,
    'MAR': 0.25,
    'RRVAR': 0.4,
    'KalmanStateSpace': 0.4,
    'MetricMotif': 1,
    'Cassandra': 0.6,
    'SeasonalityMotif': 1.5,
    'FFT': 0.8,
    "BallTreeMultivariateMotif": 0.5,  # keep an eye on RAM, not the fastest at scale but works...
    "BasicLinearModel": 0.6,
    # "TVVAR": 0.6,
}
# models that can scale well if many CPU cores are available
parallel = {
    'ETS': 1,
    'FBProphet': 0.8,
    'ARIMA': 0.2,  # slow
    'GLM': 1,
    'UnobservedComponents': 1,
    # "Greykite": 0.3,
    'UnivariateMotif': 1,
    'MultivariateMotif': 1,
    'Theta': 1,
    'ARDL': 1,
    'ARCH': 1,
}
# models that should be fast given many CPU cores
fast_parallel = {**parallel, **fast}
fast_parallel_no_arima = {
    i: fast_parallel[i]
    for i in fast_parallel
    if i
    not in [
        'ARIMA',
        'NVAR',
        "UnobservedComponents",
        "KalmanStateSpace",
        "MultivariateMotif",
        'Theta',
        "VECM",
        "MAR",
        "BallTreeMultivariateMotif",  # might need sample_fraction tuning
        # "WindowRegression"  # same base shaping as BallTreeMM
    ]
}
# so this opiniated and not fully updated always
best = list(
    set(
        list(fast_parallel_no_arima.keys())
        + ['MultivariateRegression', 'NeuralForecast', 'PytorchForecasting']
    )
)

# models that are explicitly not production ready
experimental = [
    'MotifSimulation',
    'TensorflowSTS',
    'ComponentAnalysis',
    'TFPRegression',
]
# models that perform slowly at scale
slow = list((set(all_models) - set(fast.keys())) - set(experimental))
# use GPU
gpu = [
    'GluonTS',
    'WindowRegression',
    'PytorchForecasting',
    "TiDE",
    "NeuralForecast",
    "NeuralProphet",
]
# models with model-based upper/lower forecasts
probabilistic = [
    'ARIMA',
    'GluonTS',
    'FBProphet',
    'AverageValueNaive',
    # 'VARMAX',  # yes but so slow
    'DynamicFactor',
    'VAR',
    'UnivariateMotif',
    "MultivariateMotif",
    'SectionalMotif',
    'NVAR',
    'Theta',
    'ARDL',
    'UnobservedComponents',
    'DynamicFactorMQ',
    'PytorchForecasting',
    # 'MultivariateRegression',
    'ARCH',
    'KalmanStateSpace',
    'MetricMotif',
    'Cassandra',
    'SeasonalityMotif',
    "NeuralForecast",  # mostly
    "BasicLinearModel",
    "TVVAR",
    "BallTreeRegressionMotif",
]
# models that use the shared information of multiple series to improve accuracy
multivariate = [
    'VECM',
    'DynamicFactor',
    'GluonTS',
    # 'VARMAX',  # yes but so slow
    'RollingRegression',
    'WindowRegression',
    'VAR',
    "MultivariateMotif",
    'NVAR',
    'MultivariateRegression',
    'SectionalMotif',
    'DynamicFactorMQ',
    'PytorchForecasting',
    'RRVAR',
    'MAR',
    'TMF',
    'LATC',
    'Cassandra',  # depends
    'BallTreeMultivariateMotif',
    "TiDE",
    "NeuralForecast",
    "DMD",
    "TVVAR",
    "BallTreeRegressionMotif",
]
univariate = list((set(all_models) - set(multivariate)) - set(experimental))
# USED IN AUTO_MODEL, models with no parameters
no_params = ['LastValueNaive']
# USED IN AUTO_MODEL, ONLY MODELS WHICH CAN ACCEPT RANDOM MIXING OF PARAMS
recombination_approved = [
    'GLS',
    'SeasonalNaive',
    'MotifSimulation',
    "ETS",
    'DynamicFactor',
    'VECM',
    'VARMAX',
    'GLM',
    'ARIMA',
    'FBProphet',
    'GluonTS',
    'RollingRegression',
    'VAR',
    # 'WindowRegression',
    'TensorflowSTS',
    'TFPRegression',
    'UnivariateRegression',
    "Greykite",
    'UnivariateMotif',
    "MultivariateMotif",
    'NVAR',
    'MultivariateRegression',
    'SectionalMotif',
    'Theta',
    'ARDL',
    'NeuralProphet',
    'DynamicFactorMQ',
    'PytorchForecasting',
    'ARCH',
    'RRVAR',
    'MAR',
    'TMF',
    'LATC',
    # 'KalmanStateSpace', # matrix sizes must match
    'MetricMotif',
    'Cassandra',
    'SeasonalityMotif',
    'PreprocessingRegression',
    'FFT',
    'BallTreeMultivariateMotif',
    "TiDE",
    "DMD",
    "BasicLinearModel",
    "TVVAR",
    "BallTreeRegressionMotif",
]
# USED IN AUTO_MODEL for models that don't share information among series
no_shared = [
    'ConstantNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLM',
    'ETS',
    'ARIMA',
    'FBProphet',
    'SeasonalNaive',
    'UnobservedComponents',
    'TensorflowSTS',
    "GLS",
    "UnivariateRegression",
    "Greykite",
    'UnivariateMotif',
    'Theta',
    'ARDL',
    'NeuralProphet',
    'ARCH',
    'KalmanStateSpace',
    'MetricMotif',
    'SeasonalityMotif',
    'FFT',
    "BasicLinearModel",
]
# allow the use of a regressor, need to accept "User" (fail if not given), have 'regressor' param method
regressor = [
    'GLM',
    'ARIMA',
    'FBProphet',
    'RollingRegression',
    'UnobservedComponents',
    'VECM',
    'DynamicFactor',
    'WindowRegression',
    'VAR',
    'DatepartRegression',
    "GluonTS",
    "UnivariateRegression",
    'MultivariateRegression',
    'SectionalMotif',  # kinda
    'ARDL',
    'NeuralProphet',
    'ARCH',
    'Cassandra',
    'PreprocessingRegression',
    "NeuralForecast",
    "BasicLinearModel",
    "TVVAR",
    "BallTreeRegressionMotif",
]
motifs = [
    'UnivariateMotif',
    "MultivariateMotif",
    'SectionalMotif',
    'MotifSimulation',
    'MetricMotif',
    'SeasonalityMotif',
    'BallTreeMultivariateMotif',
    "BallTreeRegressionMotif",
]
regressions = [
    'RollingRegression',
    'WindowRegression',
    'DatepartRegression',
    'UnivariateRegression',
    'MultivariateRegression',
    'PreprocessingRegression',
]
no_shared_fast = list(set(no_shared).intersection(set(fast_parallel_no_arima)))
# this should be implementable with some models in gluonts
all_result_path = [
    "UnivariateMotif",
    "MultivariateMotif",
    "SectionalMotif",
    'MetricMotif',
    "SeasonalityMotif",
    "Motif",
    "ARCH",  # simulations not motifs but similar
    "PytorchForecasting",
    # "BallTreeRegressionMotif",
    # "BallTreeMultivariateMotif",
]
# these are those that require a parameter, and return a dict
diff_window_motif_list = [
    "UnivariateMotif",
    "MultivariateMotif",
    "Motif",
    "ARCH",
]
# models that fit and then have updated predicts without updated model fits (just data update)
update_fit = [
    'MultivariateRegression',
    "DatepartRegression",
    "GluonTS",
    'WindowRegression',
    'Cassandra',
    'PreprocessingRegression',
]
model_lists = {
    "all": all_models,
    "default": default,
    "fast": fast,
    "superfast": superfast,
    "parallel": parallel,
    "fast_parallel": fast_parallel,
    "fast_parallel_no_arima": fast_parallel_no_arima,
    "scalable": fast_parallel_no_arima,
    "probabilistic": probabilistic,
    "multivariate": multivariate,
    "univariate": univariate,
    "no_params": no_params,
    "recombination_approved": recombination_approved,
    "no_shared": no_shared,
    "no_shared_fast": no_shared_fast,
    "experimental": experimental,
    "slow": slow,
    "gpu": gpu,
    "regressor": regressor,
    "best": best,
    "motifs": motifs,
    "all_result_path": all_result_path,
    "regressions": regressions,
    "all_pragmatic": all_pragmatic,
    "update_fit": update_fit,
}


def auto_model_list(n_jobs, n_series, frequency):
    pass


def model_list_to_dict(model_list):
    """Convert various possibilities to dict."""
    if model_list in list(model_lists.keys()):
        model_list = model_lists[model_list]

    if isinstance(model_list, dict):
        model_prob = list(model_list.values())
        model_list = [*model_list]
    elif isinstance(model_list, list):
        trs_len = len(model_list)
        model_prob = [1 / trs_len] * trs_len
    else:
        raise ValueError("model_list type not recognized.")
    return model_list, model_prob
