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
    'TensorflowSTS',
    'TFPRegression',
    'ComponentAnalysis',
    'DatepartRegression',
]
default = [
    'ZeroesNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'SeasonalNaive',
    'GLM',
    'ETS',
    'ARIMA',
    'FBProphet',
    'RollingRegression',
    'GluonTS',
    'UnobservedComponents',
    'VAR',
    'VECM',
    'WindowRegression',
    'DatepartRegression',
]
superfast =  [
    'ZeroesNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'SeasonalNaive',
]
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
    # 'MotifSimulation',
    # 'DatepartRegression',
]
model_lists = {
    "all": all_models,
    "default": default,
    "fast": fast,
    "superfast": superfast,
    "probabilistic": probabilistic,
    "multivariate": multivariate,
    "no_params": no_params,
    "recombination_approved": recombination_approved,
    "no_shared": no_shared,
}
