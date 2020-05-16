"""Informal testing script."""
# pragma pylint: disable=W293,E251,D407,E501
import numpy as np
import pandas as pd
from autots.datasets import load_toy_daily
from autots.datasets import load_toy_hourly
from autots.datasets import load_toy_monthly
from autots.datasets import load_toy_yearly
from autots.datasets import load_toy_weekly
from autots import AutoTS
from autots.evaluator.auto_ts import fake_regressor, error_correlations

forecast_length = 36
df_long = load_toy_hourly()

# df_long = df_long[df_long['series_id'] == 'GS10']

weights_hourly = {'traffic_volume': 10}

model_list = [
              'ZeroesNaive', 'LastValueNaive', 'AverageValueNaive',
              'GLS', 'GLM', 'SeasonalNaive'
              # 'ETS', 'RollingRegression', 'ARIMA',
              'FBProphet', 'VAR', 'GluonTS'
              # , 'VECM', 'DynamicFactor'
              # ,'VARMAX', 'GluonTS'
              ]
model_list = 'superfast'
# model_list = ['AverageValueNaive', 'LastValueNaive', 'ZeroesNaive']
# model_list = ['GLM', 'GLS']  # 'TensorflowSTS', 'TFPRegression'

metric_weighting = {'smape_weighting': 2, 'mae_weighting': 1,
                    'rmse_weighting': 2, 'containment_weighting': 0,
                    'runtime_weighting': 0, 'spl_weighting': 1,
                    'contour_weighting': 0
                    }


model = AutoTS(forecast_length=forecast_length, frequency='infer',
               prediction_interval=0.9,
               ensemble='simple,distance,probabilistic-max,horizontal-max',
               constraint=2,
               max_generations=2, num_validations=2,
               validation_method='backwards',
               model_list=model_list, initial_template='General+Random',
               metric_weighting=metric_weighting, models_to_validate=0.2,
               max_per_model_class=None,
               drop_most_recent=0, verbose=0)


preord_regressor_train, preord_regressor_forecast = fake_regressor(
    df_long, dimensions=1, forecast_length=forecast_length,
    date_col='datetime', value_col='value', id_col='series_id')
preord_regressor_train2d, preord_regressor_forecast2d = fake_regressor(
    df_long, dimensions=4, forecast_length=forecast_length,
    date_col='datetime', value_col='value', id_col='series_id')

# model = model.import_results('04222020test.csv')
model = model.fit(df_long,
                  preord_regressor=preord_regressor_train2d,
                  # weights=weights_hourly,
                  # result_file='04222027test.csv',
                  date_col='datetime', value_col='value', id_col='series_id')

print(model.best_model['Model'].iloc[0])
print(model.best_model['ModelParameters'].iloc[0])
print(model.best_model['TransformationParameters'].iloc[0])

prediction = model.predict(preord_regressor=preord_regressor_forecast2d)
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
initial_results = model.initial_results.model_results
# validation results
validation_results = model.validation_results.model_results
# just errors
error_results = model.error_templates

"""
Import/Export

example_filename = "example_export.csv" #.csv/.json
model.export_template(example_filename, models='best',
                      n=15, max_per_model_class=3)

del(model)
model = model.import_template(example_filename, method='add on')
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
"""

# %%
df_wide_numeric = model.df_wide_numeric

df = df_wide_numeric.tail(50).fillna(0).astype(float)


"""
https://packaging.python.org/tutorials/packaging-projects/

python -m pip install --user --upgrade setuptools wheel
cd /to project directory
python setup.py sdist bdist_wheel
twine upload dist/*

Merge dev to master on GitHub and create release (include .tar.gz)
"""
# *args, **kwargs

#%%
"""
Help correlate errors with parameters
"""
cols = ['Model', 'ModelParameters',
        'TransformationParameters', 'Exceptions']
all_results = pd.concat([initial_results[cols], error_results[cols]], axis=0)

# test = initial_results[ initial_results['TransformationParameters'].str.contains('IntermittentOccurrence')]

if error_results.shape[0] > 0:
    test_corr = error_correlations(all_results,
                                   result='corr')  # result='poly corr'
