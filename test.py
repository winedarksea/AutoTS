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
)
from autots import AutoTS, create_lagged_regressor
import matplotlib.pyplot as plt

# raise ValueError("aaargh!")
use_template = False
use_m5 = False  # long = False
force_univariate = True  # long = False
back_forecast = False

# this is the template file imported:
example_filename = "example_export.csv"  # .csv/.json
forecast_length = 8
long = False
df = load_monthly(long=long)
n_jobs = "auto"
verbose = 1
validation_method = "backwards"
if use_template:
    generations = 0
    num_validations = 0
else:
    generations = 3
    num_validations = 2

if use_m5:
    long = False
    df = pd.read_csv("m5_sample.gz")
    df['datetime'] = pd.DatetimeIndex(df['datetime'])
    df = df.set_index("datetime", drop=True)
    # df = df.iloc[:, 0:40]
if force_univariate:
    df = df.iloc[:, 0]

weights_hourly = {'traffic_volume': 10}
weights_monthly = {'GS10': 5}
weights_weekly = {
    'Weekly Minnesota Midgrade Conventional Retail Gasoline Prices  (Dollars per Gallon)': 2
}

transformer_list = "all"  # ["bkfilter", "STLFilter", "HPFilter", 'StandardScaler']
transformer_max_depth = 3
model_list = "default"
model_list = 'superfast'  # fast_parallel
# model_list = ["UnivariateRegression", "WindowRegression", "DatepartRegression", "RollingRegression"]

metric_weighting = {
    'smape_weighting': 3,
    'mae_weighting': 1,
    'rmse_weighting': 1,
    'containment_weighting': 0,
    'runtime_weighting': 0,
    'spl_weighting': 1,
    'contour_weighting': 1,
}


model = AutoTS(
    forecast_length=forecast_length,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=["simple", "distance", "horizontal-max", "horizontal-min", "mosaic"],
    constraint=None,
    max_generations=generations,
    num_validations=num_validations,
    validation_method=validation_method,
    model_list=model_list,
    transformer_list=transformer_list,
    transformer_max_depth=transformer_max_depth,
    initial_template='General+Random',
    metric_weighting=metric_weighting,
    models_to_validate=0.35,
    max_per_model_class=None,
    model_interrupt=True,
    n_jobs=n_jobs,
    drop_most_recent=1,
    # prefill_na=0,
    subset=None,
    verbose=verbose,
)


future_regressor_train2d, future_regressor_forecast2d  = create_lagged_regressor(
    df,
    forecast_length=forecast_length,
    summarize=None,
    backfill='datepartregression',
    fill_na='ffill'
)


# model = model.import_results('test.pickle')
if use_template:
    model = model.import_template(example_filename, method='only', enforce_model_list=True)

start_time_for = timeit.default_timer()
model = model.fit(
    df,
    future_regressor=future_regressor_train2d,
    weights="mean",
    # grouping_ids=grouping_monthly,
    # result_file='test.pickle',
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
)

elapsed_for = timeit.default_timer() - start_time_for

prediction = model.predict(future_regressor=future_regressor_forecast2d, verbose=1)
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

initial_results.to_csv("general_template_" + str(platform.node()) + ".csv")

prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                remove_zeroes=False,
                start_date="2019-01-01")

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

"""
# Import/Export
model.export_template(example_filename, models='all',
                      n=15, max_per_model_class=3)

del(model)
model = model.import_template(example_filename, method='only')
print("Overwrite template is: {}".format(str(model.initial_template)))

future_regressor_train2d, future_regressor_forecast2d = fake_regressor(
    df,
    dimensions=4,
    forecast_length=forecast_length,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
    drop_most_recent=model.drop_most_recent,
    aggfunc=model.aggfunc,
    verbose=model.verbose,
)
"""

"""
Things needing testing:
    With and without regressor
    With and without weighting
    Different frequencies
    Various verbose inputs

Edgey Cases:
        Single Time Series
        Forecast Length of 1
        Very short training data
        Lots of NaN
"""

# %%

"""
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

# %%

# Help correlate errors with parameters
"""
test = initial_results[initial_results['TransformationParameters'].str.contains('FastICA')]

cols = ['Model', 'ModelParameters', 'TransformationParameters', 'Exceptions']
if (~initial_results['Exceptions'].isna()).sum() > 0:
    test_corr = error_correlations(
        initial_results[cols], result='corr'
    )  # result='poly corr'
"""
