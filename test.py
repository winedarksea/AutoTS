"""Informal testing script."""
from time import sleep
import timeit
import os
import platform
import pandas as pd
from autots.datasets import (  # noqa
    load_daily,
    load_hourly,
    load_monthly,
    load_yearly,
    load_weekly,
    load_weekdays,
    load_zeroes,
    load_linear,
    load_sine,
    load_artificial,
)
from autots import AutoTS, create_regressor, model_forecast  # noqa
import matplotlib.pyplot as plt

# raise ValueError("aaargh!")
use_template = False
save_template = False
force_univariate = False  # long = False
back_forecast = False
graph = True
template_import_method = "addon"  # "only" "addon"
models_to_validate = 0.35  # 0.99 to validate every tried (use with template import)

# this is the template file imported:
template_filename = "template_" + str(platform.node()) + ".csv"
template_filename = "template_categories_1.csv"
name = template_filename.replace('.csv', '')
random_seed = 2022
forecast_length = 28
long = False
# df = load_linear(long=long, shape=(400, 1000), introduce_nan=None)
df = load_sine(
    long=long, shape=(400, 1000), start_date="2021-01-01", introduce_random=100
).iloc[:, 2:]
# df = load_artificial(long=long, date_start="2018-01-01")
# df = load_weekly(long=long)
interest_series = [
    'arima220_outliers',
    'lumpy',
    'out-of-stock',
    "sine_seasonality_monthweek",
    "intermittent_weekly",
    "arima017",
    "old_to_new",
]
prediction_interval = 0.9
n_jobs = "auto"
verbose = 2
validation_method = "backwards"  # "similarity"
frequency = "infer"
drop_most_recent = 0
generations = 1
num_validations = 0  # "auto"
initial_template = "Random"  # "General+Random"
if use_template:
    initial_training = not os.path.exists(template_filename)
    if initial_training:
        print("No existing template found.")
    else:
        print("Existing template found.")

if force_univariate:
    df = df.iloc[:, 0]

transformer_list = "fast"  # "fast", "all", "superfast"
# transformer_list = ["Round", "Slice", "EWMAFilter", 'Cointegration', "MeanDifference", "BTCD"]
transformer_max_depth = 1
models_mode = "default"  # "default", "regressor", "neuralnets"
model_list = "superfast"
# model_list = "fast"  # fast_parallel, all
model_list = ["UnivariateRegression", "LastValueNaive"]
preclean = None
{
    "fillna": None,  # mean or median one of few consistent things
    "transformations": {"0": "EWMAFilter"},
    "transformation_params": {
        "0": {"span": 14},
    },
}
ensemble = [
    "simple",
    'horizontal',
]  # "dist", "subsample", "mosaic-window", "horizontal-max"
# ensemble = None
metric_weighting = {
    'smape_weighting': 5,
    'mae_weighting': 2,
    'rmse_weighting': 2,
    'made_weighting': 1,
    'mage_weighting': 0,
    'mle_weighting': 0,
    'imle_weighting': 0,
    'spl_weighting': 3,
    'containment_weighting': 0,
    'contour_weighting': 0,
    'runtime_weighting': 0.05,
    'maxe_weighting': 0,
    'oda_weighting': 0,
    'mqae_weighting': 0,
}
constraint = {
    "constraint_method": "quantile",
    "constraint_regularization": 0.9,
    "upper_constraint": 0.9,
    "lower_constraint": 0.1,
    "bounds": True,
}
constraint = None

model = AutoTS(
    forecast_length=forecast_length,
    frequency=frequency,
    prediction_interval=prediction_interval,
    ensemble=ensemble,
    constraint=constraint,
    max_generations=generations,
    num_validations=num_validations,
    validation_method=validation_method,
    model_list=model_list,
    transformer_list=transformer_list,
    transformer_max_depth=transformer_max_depth,
    initial_template=initial_template,
    metric_weighting=metric_weighting,
    models_to_validate=models_to_validate,
    max_per_model_class=None,
    model_interrupt=True,
    n_jobs=n_jobs,
    drop_most_recent=drop_most_recent,
    introduce_na=None,
    preclean=preclean,
    # prefill_na=0,
    # subset=5,
    verbose=verbose,
    models_mode=models_mode,
    random_seed=random_seed,
    current_model_file=f"current_model_{name}",
)


regr_train, regr_fcst = create_regressor(
    df,
    forecast_length=forecast_length,
    frequency=frequency,
    drop_most_recent=drop_most_recent,
    scale=True,
    summarize="auto",
    backfill="bfill",
    fill_na="pchip",
    holiday_countries=["US"],
    datepart_method="recurring",
)


# model = model.import_results('test.pickle')
if use_template:
    model = model.import_template(
        template_filename, method=template_import_method, enforce_model_list=True
    )

start_time_for = timeit.default_timer()
model = model.fit(
    df,
    future_regressor=regr_train,
    # weights="mean",
    # result_file='test.pickle',
    validation_indexes=[
        pd.date_range("2021-01-01", "2022-05-02"),
        pd.date_range("2021-01-01", "2022-02-02"),
        pd.date_range("2021-01-01", "2022-03-03"),
    ],
    date_col="datetime" if long else None,
    value_col="value" if long else None,
    id_col="series_id" if long else None,
)

elapsed_for = timeit.default_timer() - start_time_for

prediction = model.predict(
    future_regressor=regr_fcst, verbose=1, fail_on_forecast_nan=True
)
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
initial_results = model.results()
# validation results
validation_results = model.results("validation")

initial_results["TransformationRuntime"] = initial_results[
    "TransformationRuntime"
].dt.total_seconds()
initial_results["FitRuntime"] = initial_results["FitRuntime"].dt.total_seconds()
initial_results["PredictRuntime"] = initial_results["PredictRuntime"].dt.total_seconds()
initial_results["TotalRuntime"] = initial_results["TotalRuntime"].dt.total_seconds()

sleep(5)
print(model)
print(model.validation_test_indexes)
print(f"Model failure rate is {model.failure_rate() * 100:.1f}%")
print(f'The following model types failed completely {model.list_failed_model_types()}')
print("Slowest models:")
print(
    initial_results[initial_results["Ensemble"] < 1]
    .groupby("Model")
    .agg({"TotalRuntimeSeconds": ["mean", "max"]})
    .idxmax()
)

if save_template:
    model.export_template(
        template_filename,
        models="best",
        n=20,
        max_per_model_class=5,
        include_results=True,
    )

if graph:
    prediction.plot(
        model.df_wide_numeric,
        series=model.df_wide_numeric.columns[2],
        remove_zeroes=False,
        start_date="2018-09-26",
    )
    plt.show()
    model.plot_generation_loss()
    plt.show()
    # plt.savefig("improvement_over_generations.png", dpi=300)

    model.plot_per_series_smape(kind="pie")
    plt.show()

    model.plot_per_series_error()
    plt.show()

    if model.best_model_ensemble == 2:
        model.plot_horizontal_per_generation()
        plt.show()

        plt.show()
        model.plot_horizontal_transformers(method="fillna")
        plt.show()
        model.plot_horizontal_transformers()
        plt.show()
        model.plot_horizontal()
        plt.savefig(f"horizontal_{name}.png", dpi=300)
        # plt.show()
        if "mosaic" in model.best_model["ModelParameters"].iloc[0].lower():
            mosaic_df = model.mosaic_to_df()
            print(mosaic_df[mosaic_df.columns[0:5]].head(5))

    plt.show()
    if back_forecast:
        model.plot_backforecast(n_splits="auto", start_date="2019-01-01")

df_wide_numeric = model.df_wide_numeric

df = df_wide_numeric.tail(100).fillna(0).astype(float)

print("test run complete")

if interest_series not in model.df_wide_numeric.columns.tolist():
    interest_series = model.df_wide_numeric.columns.tolist()[0:5]
if model.best_model["Ensemble"].iloc[0] == 2:
    interest_models = []
    for x, y in model.best_model_params['series'].items():
        if x in interest_series:
            if isinstance(y, str):
                interest_models.append(y)
            else:
                interest_models.extend(list(y.values()))
            if graph:
                prediction.plot(
                    model.df_wide_numeric,
                    series=x,
                    remove_zeroes=False,
                    start_date="2018-09-26",
                )
    interest_models = pd.Series(interest_models).value_counts().head(10)
    print(interest_models)
    print(
        [
            y
            for x, y in model.best_model_params['models'].items()
            if x in interest_models.index.to_list()
        ]
    )
else:
    for x in interest_series:
        if graph:
            prediction.plot(
                model.df_wide_numeric,
                series=x,
                remove_zeroes=False,
                start_date="2018-09-26",
            )

"""
forecasts = model_forecast(
    model_name="UnivariateMotif",
    model_param_dict={'window': 10, "pointed_method":"weighted_mean", "distance_metric": "cosine", "k": 10, "return_result_windows": True},
    model_transform_dict={
        'fillna': 'rolling_mean',
        'transformations': {'0': 'MinMaxScaler', "1": "PCA"},
        'transformation_params': {'0': {}, '1': {"whiten": True}}
    },
    df_train=model.df_wide_numeric,
    forecast_length=forecast_length,
    frequency='infer',
    prediction_interval=prediction_interval,
    no_negatives=False,
    # future_regressor_train=future_regressor_train2d,
    # future_regressor_forecast=future_regressor_forecast2d,
    random_seed=321,
    verbose=1,
    n_jobs="auto",
    return_model=True,
)
result = forecasts.forecast.head(5)
print(result)
print(forecasts.upper_forecast.head(5))
print(forecasts.lower_forecast.head(5))
result_windows = forecasts.model.result_windows
"""

"""
# default save location of files is apparently root
systemd-run --unit=background_cmd_service --remain-after-exit /home/colin/miniconda3/envs/openblas/bin/python /home/colin/AutoTS/test.py
systemd-run --unit=background_cmd_service --remain-after-exit /home/colin/miniconda3/envs/openblas/bin/python /home/colin/AutoTS/local_example.py
journalctl -r -n 10 -u background_cmd_service
journalctl -f -u background_cmd_service
journalctl -b -u background_cmd_service

systemctl stop background_cmd_service
systemctl reset-failed
systemctl kill background_cmd_service

scp colin@192.168.1.122:/home/colin/AutoTS/general_template_colin-1135.csv ./Documents/AutoTS
scp colin@192.168.1.122:/general_template_colin-1135.csv ./Documents/AutoTS


PACKAGE RELEASE
# update version in setup.py, /docs/conf.py, /autots/_init__.py

cd to AutoTS
set PYTHONPATH=%PYTHONPATH%;C:/Users/Colin/Documents/AutoTS
python -m unittest discover ./tests
python -m unittest tests.test_autots.ModelTest.test_models
python -m unittest tests.test_impute.TestImpute.test_impute

python ./autots/evaluator/benchmark.py > benchmark.txt

cd <project dir>
black ./autots -l 88 -S

mistune==0.8.4 markupsafe==2.0.1 jinja2==2.11.3
https://github.com/sphinx-doc/sphinx/issues/3382
# pip install sphinx==2.4.4
# m2r does not yet work on sphinx 3.0
# pip install m2r
cd <project dir>
# delete docs/source and /build (not tutorial or intro.rst)
sphinx-apidoc -f -o docs/source autots
cd ./docs
make html

https://winedarksea.github.io/AutoTS/build/index.html
"""
"""
https://packaging.python.org/tutorials/packaging-projects/

python -m pip install --user --upgrade setuptools wheel
cd /to project directory
python setup.py sdist bdist_wheel
twine upload dist/*
To use this API token:
    Set your username to __token__
    Set your password to the token value, including the pypi- prefix


Merge dev to master on GitHub and create release (include .tar.gz)
Update conda-forge
Update fb third-party (and default)
"""

# Help correlate errors with parameters
"""
test = initial_results[initial_results['TransformationParameters'].str.contains('FastICA')]

cols = ['Model', 'ModelParameters', 'TransformationParameters', 'Exceptions']
if (~initial_results['Exceptions'].isna()).sum() > 0:
    test_corr = error_correlations(
        initial_results[cols], result='corr'
    )  # result='poly corr'

python -m cProfile -o testpy.pstats test.py
gprof2dot -f pstats testpy.pstats | "C:/Program Files (x86)/Graphviz/bin/dot.exe" -Tpng -o test_pstat_output.png
gprof2dot -f pstats testpy.pstats | dot -Tpng -o test_pstat_output.png
"""
