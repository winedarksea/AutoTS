"""
Informal testing script
"""
import numpy as np
import pandas as pd

from autots.datasets import load_toy_daily
from autots.datasets import load_toy_hourly
from autots.datasets import load_toy_monthly
from autots.datasets import load_toy_yearly
from autots.datasets import load_toy_weekly

#%% 
forecast_length = 5
df_long = load_toy_daily()

# df_long = df_long[df_long['series_id'] == 'GS10']

weights_daily = {'categoricalDayofWeek': 5,
           'randomNegative': 1,
         'sp500high': 2,
         'wabashaTemp': 1}

weights_hourly = {'traffic_volume': 10}

model_list = [
              # 'ZeroesNaive', 'LastValueNaive', 'AverageValueNaive', 'GLS',
              'GLM', 'ETS','RollingRegression', 'ARIMA',
               'FBProphet', 'UnobservedComponents'
               ,'VECM', 'DynamicFactor'
              #,'VARMAX', 'GluonTS'
              ]
# model_list = 'superfast'
# model_list = ['MofitSimulation', 'GLM','ZeroesNaive', 'LastValueNaive', 'AverageValueNaive', 'GLS', 'SeasonalNaive']
model_list = ['RollingRegression']

metric_weighting = {'smape_weighting' : 10, 'mae_weighting' : 1,
            'rmse_weighting' : 5, 'containment_weighting' : 1, 'runtime_weighting' : 0,
            'lower_mae_weighting': 0, 'upper_mae_weighting': 1, 'contour_weighting': 2}

from autots import AutoTS
model = AutoTS(forecast_length = forecast_length, frequency = 'infer',
               prediction_interval = 0.9, ensemble = False, weighted = False,
               max_generations = 0, num_validations = 2, validation_method = 'even',
               model_list = model_list, initial_template = 'General+Random',
               metric_weighting = metric_weighting, models_to_validate = 100,
               max_per_model_class = 10,
               drop_most_recent = 1, verbose = 1)

from autots.evaluator.auto_ts import fake_regressor
preord_regressor_train, preord_regressor_forecast = fake_regressor(df_long, dimensions= 1, forecast_length = forecast_length, date_col = 'datetime', value_col = 'value', id_col = 'series_id')
preord_regressor_train2d, preord_regressor_forecast2d = fake_regressor(df_long, dimensions= 4, forecast_length = forecast_length, date_col = 'datetime', value_col = 'value', id_col = 'series_id')

# model = model.fit(df_long, date_col = 'datetime', value_col = 'value', id_col = 'series_id')
"""
model = model.fit(df_long, weights = weights_hourly,
                  result_file = 'test_results.csv',
                  date_col = 'datetime', value_col = 'value', id_col = 'series_id') # and weighted = True
"""
model = model.fit(df_long, preord_regressor = preord_regressor_train2d, date_col = 'datetime', value_col = 'value', id_col = 'series_id')

print(model.best_model['Model'].iloc[0])
print(model.best_model['ModelParameters'].iloc[0])
print(model.best_model['TransformationParameters'].iloc[0])

prediction = model.predict(preord_regressor = preord_regressor_forecast2d)
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
initial_results = model.initial_results.model_results
# validation results
validation_results = model.validation_results.model_results

#%%

"""
Import/Export

example_filename = "example_export.csv" #.csv/.json
model.export_template(example_filename, models = 'best', n = 15, max_per_model_class = 3)
model = model.import_template(example_filename, method = 'add on')
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
"""

from autots.tools.shaping import long_to_wide
df_wide = long_to_wide(df_long, date_col = 'datetime', value_col = 'value',
                       id_col='series_id', frequency='infer', aggfunc='first')

# df = df_wide[df_wide.columns[0:3]].fillna(0).astype(float)

from autots.tools.shaping import values_to_numeric
categorical_transformer = values_to_numeric(df_wide)
df_wide_numeric = categorical_transformer.dataframe

df = df_wide_numeric.tail(50)
"""


"""
https://packaging.python.org/tutorials/packaging-projects/

python -m pip install --user --upgrade setuptools wheel
cd /to project directory
python setup.py sdist bdist_wheel
twine upload dist/*

Merge dev to master on GitHub and create release (include .tar.gz)
"""
# *args, **kwargs
