"""Informal testing script."""
from time import sleep
import timeit
import platform
import pandas as pd
from autots.datasets import (
    load_daily,
    load_hourly,
    load_monthly,
    load_yearly,
    load_weekly,
    load_weekdays,
    load_zeroes,
    load_linear,
    load_sine,
)
from autots import AutoTS, create_regressor
import matplotlib.pyplot as plt

# raise ValueError("aaargh!")
use_template = False
force_univariate = False  # long = False
back_forecast = False
graph = True

# this is the template file imported:
example_filename = "example_export.csv"  # .csv/.json
forecast_length = 8
long = False
# df = load_linear(long=long, shape=(200, 500), introduce_nan=0.2)
df = load_daily(long=long)
n_jobs = "auto"
verbose = 2
validation_method = "similarity"
frequency = 'infer'
drop_most_recent = 0
if use_template:
    generations = 5
    num_validations = 0
else:
    generations = 3
    num_validations = 2

if force_univariate:
    df = df.iloc[:, 0]

transformer_list = "fast"  # ["bkfilter", "STLFilter", "HPFilter", 'StandardScaler']
transformer_max_depth = 1
model_list = "default"
model_list = 'superfast'  # fast_parallel
# model_list = ["NVAR", "SectionalMotif"]

metric_weighting = {
    'smape_weighting': 3,
    'mae_weighting': 1,
    'rmse_weighting': 1,
    'containment_weighting': 0,
    'runtime_weighting': 0.1,
    'spl_weighting': 1,
    'contour_weighting': 1,
}

model = AutoTS(
    forecast_length=forecast_length,
    frequency=frequency,
    prediction_interval=0.9,
    ensemble=['horizontal-max'],
    constraint=None,
    max_generations=generations,
    num_validations=num_validations,
    validation_method=validation_method,
    model_list=model_list,
    transformer_list=transformer_list,
    transformer_max_depth=transformer_max_depth,
    initial_template='Random',
    metric_weighting=metric_weighting,
    models_to_validate=0.35,
    max_per_model_class=None,
    model_interrupt=True,
    n_jobs=n_jobs,
    drop_most_recent=drop_most_recent,
    introduce_na=True,
    # prefill_na=0,
    # subset=5,
    verbose=verbose,
)


regr_train, regr_fcst = create_regressor(
    df,
    forecast_length=forecast_length,
    frequency=frequency,
    drop_most_recent=drop_most_recent,
    scale=True,
    summarize="auto",
    backfill='bfill',
    fill_na='pchip',
    holiday_countries=["US"],
    datepart_method="recurring",
)


# model = model.import_results('test.pickle')
if use_template:
    model = model.import_template(example_filename, method='only', enforce_model_list=True)

start_time_for = timeit.default_timer()
model = model.fit(
    df,
    future_regressor=regr_train,
    weights="mean",
    # result_file='test.pickle',
    validation_indexes=[pd.date_range("2021-01-01", "2022-02-02"), pd.date_range("2021-01-01", "2022-03-03")],
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
)

elapsed_for = timeit.default_timer() - start_time_for

prediction = model.predict(future_regressor=regr_fcst, verbose=1)
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
initial_results = model.results()
# validation results
validation_results = model.results("validation")

initial_results['TransformationRuntime'] = initial_results['TransformationRuntime'].dt.total_seconds()
initial_results['FitRuntime'] = initial_results['FitRuntime'].dt.total_seconds()
initial_results['PredictRuntime'] = initial_results['PredictRuntime'].dt.total_seconds()
initial_results['TotalRuntime'] = initial_results['TotalRuntime'].dt.total_seconds()

sleep(5)
print(model)
print(f"Model failure rate is {model.failure_rate() * 100:.1f}%")
print("Slowest models:")
print(initial_results.groupby("Model").agg({'TotalRuntimeSeconds': ['mean', 'max']}).idxmax())

initial_results.to_csv("general_template_" + str(platform.node()) + ".csv")

if graph:
    prediction.plot(model.df_wide_numeric,
                    series=model.df_wide_numeric.columns[2],
                    remove_zeroes=False,
                    start_date="2018-09-26")
    plt.show()
    model.plot_generation_loss()

    if model.best_model['Ensemble'].iloc[0] == 2:
        plt.show()
        model.plot_horizontal_transformers(method="fillna")
        plt.show()
        model.plot_horizontal_transformers()
        plt.show()
        model.plot_horizontal()
        plt.show()
        if 'mosaic' in model.best_model['ModelParameters'].iloc[0].lower():
            mosaic_df = model.mosaic_to_df()
            print(mosaic_df[mosaic_df.columns[0:5]].head(5))

    plt.show()
    if back_forecast:
        model.plot_backforecast(n_splits="auto", start_date="2019-01-01")

df_wide_numeric = model.df_wide_numeric

df = df_wide_numeric.tail(100).fillna(0).astype(float)

print('test run complete')

"""
# Import/Export
model.export_template(example_filename, models='all',
                      n=15, max_per_model_class=3)
del(model)
model = model.import_template(example_filename, method='only')
print("Overwrite template is: {}".format(str(model.initial_template)))

# default save location of files is apparently root
systemd-run --unit=background_cmd_service --remain-after-exit /home/colin/miniconda3/envs/openblas/bin/python /home/colin/AutoTS/test.py
journalctl -r -n 10 -u background_cmd_service
journalctl -f -u background_cmd_service
journalctl -b -u background_cmd_service

systemctl stop background_cmd_service
systemctl reset-failed
systemctl kill background_cmd_service

scp colin@192.168.1.122:/home/colin/AutoTS/general_template_colin-1135.csv ./Documents/AutoTS
scp colin@192.168.1.122:/general_template_colin-1135.csv ./Documents/AutoTS


Edgey Cases:
    Single Time Series
    Forecast Length of 1
    Very short training data
    Lots of NaN


PACKAGE RELEASE
# update version in setup.py, /docs/conf.py, /autots/_init__.py

set PYTHONPATH=%PYTHONPATH%;C:/Users/Colin/Documents/AutoTS
python -m unittest discover ./tests

cd <project dir>
black ./autots -l 88 -S

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

Merge dev to master on GitHub and create release (include .tar.gz)
"""

# Help correlate errors with parameters
"""
test = initial_results[initial_results['TransformationParameters'].str.contains('FastICA')]

cols = ['Model', 'ModelParameters', 'TransformationParameters', 'Exceptions']
if (~initial_results['Exceptions'].isna()).sum() > 0:
    test_corr = error_correlations(
        initial_results[cols], result='corr'
    )  # result='poly corr'
"""
