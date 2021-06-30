"""Lists of models grouped by aspects."""
all_models = [
    'ZeroesNaive',
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
    'VARMAX',
    'VECM',
    'DynamicFactor',
    'MotifSimulation',
    'WindowRegression',
    'VAR',
    'TFPRegression',
    'ComponentAnalysis',
    'DatepartRegression',
    "UnivariateRegression",
]
default = [
    'ZeroesNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'SeasonalNaive',
    'GLM',
    'ETS',
    'FBProphet',
    'RollingRegression',
    'GluonTS',
    'UnobservedComponents',
    'VAR',
    'VECM',
    'WindowRegression',
    'DatepartRegression',
]
# fastest models at any scale
superfast = [
    'ZeroesNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'SeasonalNaive',
]
# relatively fast at any scale
fast = [
    'ZeroesNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'GLM',
    'ETS',
    'WindowRegression',
    'GluonTS',
    'VAR',
    'SeasonalNaive',
    'VECM',
    'ComponentAnalysis',
]
# models that can scale well if many CPU cores are available
parallel = [
    'ETS',
    'FBProphet',
    'ARIMA',
    'GLM',
    'UnobservedComponents',
    'MotifSimulation',
]
# models that should be fast given many CPU cores
fast_parallel = list(set(parallel + fast))
fast_parallel.remove('MotifSimulation')
# models that are explicitly not production ready
experimental = [
    'MotifSimulation',
    'TensorflowSTS',
    'ComponentAnalysis',
    'TFPRegression',
]
# models that perform slowly at scale
slow = list((set(all_models) - set(fast)) - set(experimental))
# use GPU
gpu = ['GluonTS']
# models with model-based upper/lower forecasts
probabilistic = [
    'ARIMA',
    'GluonTS',
    'FBProphet',
    'AverageValueNaive',
    'MotifSimulation',
    'VARMAX',
    'DynamicFactor',
    'VAR',
]
# models that use the shared information of multiple series to improve accuracy
multivariate = [
    'VECM',
    'DynamicFactor',
    'GluonTS',
    'VARMAX',
    'RollingRegression',
    'WindowRegression',
    'VAR',
    'ComponentAnalysis',
]
# USED IN AUTO_MODEL, models with no parameters
no_params = ['ZeroesNaive', 'LastValueNaive', 'GLS']
# USED IN AUTO_MODEL, ONLY MODELS WHICH CAN ACCEPT RANDOM MIXING OF PARAMS
recombination_approved = [
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
    'WindowRegression',
    'TensorflowSTS',
    'TFPRegression',
    'UnivariateRegression'
]
# USED IN AUTO_MODEL for models that don't share information among series
no_shared = [
    'ZeroesNaive',
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
    "UnivariateRegression"
    # 'MotifSimulation',
    # 'DatepartRegression',
]
# allow the use of a regressor
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
    # "UnivariateRegression",
]
no_shared_fast = list(set(no_shared).intersection(set(fast_parallel)))
model_lists = {
    "all": all_models,
    "default": default,
    "fast": fast,
    "superfast": superfast,
    "parallel": parallel,
    "fast_parallel": fast_parallel,
    "probabilistic": probabilistic,
    "multivariate": multivariate,
    "no_params": no_params,
    "recombination_approved": recombination_approved,
    "no_shared": no_shared,
    "no_shared_fast": no_shared_fast,
    "experimental": experimental,
    "slow": slow,
    "gpu": gpu,
    "regressor": regressor,
}
