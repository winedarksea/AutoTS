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
    "Greykite",
    'UnivariateMotif',
    'MultivariateMotif',
    'NVAR',
    'MultivariateRegression',
    'SectionalMotif',
    'Theta',
    'ARDL',
    'NeuralProphet',
    'DynamicFactorMQ',
]
default = [
    'ConstantNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'SeasonalNaive',
    'GLM',
    'ETS',
    'FBProphet',
    # 'RollingRegression',  # maybe not?
    # 'GluonTS',  # downweight if that becomes an option
    'UnobservedComponents',
    'VAR',
    'VECM',
    'WindowRegression',
    'DatepartRegression',
    # 'UnivariateRegression',  # this has been crashing on 1135
    'MultivariateRegression',  # downweight if that becomes an option
    'UnivariateMotif',
    'MultivariateMotif',
    'SectionalMotif',
    'NVAR',
    'Theta',
    'ARDL',
    # 'DynamicFactorMQ',
]
best = [
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'GLM',
    'ETS',
    # 'ARIMA',
    'FBProphet',
    # 'RollingRegression',
    'GluonTS',
    'SeasonalNaive',
    'UnobservedComponents',
    # 'VARMAX',
    'VECM',
    # 'MotifSimulation',
    # 'UnivariateRegression',
    'MultivariateRegression',
    'WindowRegression',
    'VAR',
    'DatepartRegression',
    'UnivariateMotif',
    'MultivariateMotif',
    'NVAR',
    'SectionalMotif',
    'Theta',
    'ARDL',
]
# fastest models at any scale
superfast = [
    'ConstantNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'SeasonalNaive',
]
# relatively fast
fast = [
    'ConstantNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'SeasonalNaive',
    'GLM',
    'ETS',
    # 'UnobservedComponents',  # it's fast enough but I'll leave for parallel
    'VAR',
    'VECM',
    'WindowRegression',  # well, this gets slow with Transformer, KerasRNN
    'DatepartRegression',
    'UnivariateMotif',
    'MultivariateMotif',
    'SectionalMotif',
    'NVAR',
]
# models that can scale well if many CPU cores are available
parallel = [
    'ETS',
    'FBProphet',
    'ARIMA',
    'GLM',
    'UnobservedComponents',
    "Greykite",
    'UnivariateMotif',
    'MultivariateMotif',
    'Theta',
    'ARDL',
]
# models that should be fast given many CPU cores
fast_parallel = list(set(parallel + fast))
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
gpu = ['GluonTS', 'WindowRegression']
# models with model-based upper/lower forecasts
probabilistic = [
    'ARIMA',
    'GluonTS',
    'FBProphet',
    'AverageValueNaive',
    'VARMAX',
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
    # 'MultivariateRegression',
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
    "MultivariateMotif",
    'NVAR',
    'MultivariateRegression',
    'SectionalMotif',
    'DynamicFactorMQ',
]
univariate = list((set(all_models) - set(multivariate)) - set(experimental))
# USED IN AUTO_MODEL, models with no parameters
no_params = ['LastValueNaive', 'GLS']
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
}


def auto_model_list(n_jobs, n_series, frequency):
    pass
