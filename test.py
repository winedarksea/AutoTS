"""Informal testing script."""
import timeit
import numpy as np
import pandas as pd
from autots.datasets import (
    load_daily,
    load_hourly,
    load_monthly,
    load_yearly,
    load_weekly,
)
from autots import AutoTS
from autots.evaluator.auto_ts import fake_regressor, error_correlations

# raise ValueError("aaargh!")

example_filename = "example_export2.csv"  # .csv/.json
forecast_length = 12
long = False
df = load_weekly(long=long)
n_jobs = 'auto'
generations = 2

df = pd.read_csv("m5_sample.gz")
df['datetime'] = pd.DatetimeIndex(df['datetime'])
df = df.set_index("datetime", drop=True)
df = df.iloc[:, 0:100]
# df = df[df['series_id'] == 'GS10']

weights_hourly = {'traffic_volume': 10}
weights_monthly = {'GS10': 5}
weights_weekly = {
    'Weekly Minnesota Midgrade Conventional Retail Gasoline Prices  (Dollars per Gallon)': 20
}
grouping_monthly = {
    'CSUSHPISA': 'A',
    'EMVOVERALLEMV': 'A',
    'EXCAUS': 'exchange rates',
    'EXCHUS': 'exchange rates',
    'EXUSEU': 'exchange rates',
    'MCOILWTICO': 'C',
    'T10YIEM': 'C',
    'wrong': 'C',
    'USEPUINDXM': 'C',
}

model_list = [
    'ZeroesNaive',
    'LastValueNaive',
    'AverageValueNaive',
    'GLS',
    'SeasonalNaive',
    'GLM',
    'ETS',
    'FBProphet',
    # 'RollingRegression',
    # 'GluonTS',
    'UnobservedComponents',
    'DatepartRegression',
    # 'VAR',
    # 'VECM',
    # 'WindowRegression',
]
# model_list = 'superfast'
# model_list = ['GLM', 'DatepartRegression']
# model_list = ['ARIMA', 'ETS', 'FBProphet', 'LastValueNaive', 'GLM']

metric_weighting = {
    'smape_weighting': 2,
    'mae_weighting': 1,
    'rmse_weighting': 2,
    'containment_weighting': 0,
    'runtime_weighting': 0,
    'spl_weighting': 1,
    'contour_weighting': 0,
}


model = AutoTS(
    forecast_length=forecast_length,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,  # "simple,distance,horizontal-max",
    constraint=2,
    max_generations=generations,
    num_validations=2,
    validation_method='backwards',
    model_list=model_list,
    initial_template='Random',
    metric_weighting=metric_weighting,
    models_to_validate=0.3,
    max_per_model_class=None,
    model_interrupt=True,
    n_jobs=n_jobs,
    drop_most_recent=0,
    subset=None,
    verbose=1,
)


future_regressor_train, future_regressor_forecast = fake_regressor(
    df,
    dimensions=1,
    forecast_length=forecast_length,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
    drop_most_recent=model.drop_most_recent,
    aggfunc=model.aggfunc,
    verbose=model.verbose,
)
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

# model = model.import_results('test.pickle')
# model = model.import_template(example_filename, method='only', enforce_model_list=True)

start_time_for = timeit.default_timer()
model = model.fit(
    df,
    future_regressor=future_regressor_train2d,
    # weights=weights_weekly,
    # grouping_ids=grouping_monthly,
    # result_file='test.pickle',
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
)
elapsed_for = timeit.default_timer() - start_time_for


"""
del model
model = AutoTS(
    forecast_length=forecast_length,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,
    constraint=2,
    max_generations=generations,
    num_validations=2,
    validation_method='backwards',
    model_list=model_list,
    initial_template='General+Random',
    metric_weighting=metric_weighting,
    models_to_validate=0.1,
    max_per_model_class=None,
    model_interrupt=True,
    n_jobs=None,
    drop_most_recent=0,
    verbose=1,
)
# model = model.import_template(example_filename, method='only')
import time

time.sleep(30)
import joblib

with joblib.parallel_backend("loky", n_jobs=n_jobs):
    start_time_cxt = timeit.default_timer()
    model = model.fit(
        df,
        future_regressor=future_regressor_train2d,
        # grouping_ids=grouping_monthly,
        # result_file='test.pickle',
        date_col='datetime' if long else None,
        value_col='value' if long else None,
        id_col='series_id' if long else None,
    )
    elapsed_cxt = timeit.default_timer() - start_time_cxt
print(f"With Context {elapsed_cxt}\nWithout Context {elapsed_for}")
"""

print(model.best_model['Model'].iloc[0])
print(model.best_model['ModelParameters'].iloc[0])
print(model.best_model['TransformationParameters'].iloc[0])

"""
prediction_ints = model.predict(
    future_regressor=future_regressor_forecast2d,
    prediction_interval=[0.99, 0.5],
    verbose=0,
)
"""
prediction = model.predict(future_regressor=future_regressor_forecast2d, verbose=0)
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

print(f"Model failure rate is {model.failure_rate() * 100:.1f}%")
# import platform
# initial_results.to_csv("bigger_speedtest" + str(platform.node()) + "_openblas.csv")

"""
# Import/Export
model.export_template(example_filename, models='all',
                      n=15, max_per_model_class=3)

del(model)
model = model.import_template(example_filename, method='only')
print("Overwrite template is: {}".format(str(model.initial_template)))
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
df_wide_numeric = model.df_wide_numeric

df = df_wide_numeric.tail(50).fillna(0).astype(float)

"""
PACKAGE RELEASE
# update version in setup.py, /docs/conf.py, /autots/_init__.py

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

#%%

# Help correlate errors with parameters
"""
# test = initial_results[initial_results['TransformationParameters'].str.contains('Detrend')]
cols = ['Model', 'ModelParameters', 'TransformationParameters', 'Exceptions']
if (~initial_results['Exceptions'].isna()).sum() > 0:
    test_corr = error_correlations(
        initial_results[cols], result='corr'
    )  # result='poly corr'
"""
"""
prediction_intervals = [0.99, 0.67]
model_list = 'superfast'  # ['FBProphet', 'VAR', 'AverageValueNaive']
from autots.evaluator.auto_ts import AutoTSIntervals

intervalModel = AutoTSIntervals().fit(
    prediction_intervals=prediction_intervals,
    import_template=None,
    forecast_length=forecast_length,
    df=df,
    max_generations=1,
    num_validations=2,
    import_results='test.pickle',
    result_file='testProb.pickle',
    validation_method='seasonal 12',
    models_to_validate=0.2,
    interval_models_to_validate=50,
    date_col='datetime',
    value_col='value',
    id_col='series_id',
    model_list=model_list,
    future_regressor=[],
    constraint=2,
    no_negatives=True,
    remove_leading_zeroes=True,
    random_seed=2020,
)  # weights, future_regressor, metrics
intervalForecasts = intervalModel.predict()
intervalForecasts[0.99].upper_forecast
"""
