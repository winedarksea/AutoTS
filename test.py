"""
Things needing testing:
    With and without regressor
    With and without weighting
    Different frequencies
    Various verbose inputs
    
    Passing in Start Dates - (Test)
    Holidays on non-daily datas
"""
import numpy as np
import pandas as pd

forecast_length = 3
from autots.datasets import load_toy_daily
from autots.datasets import load_toy_hourly
from autots.datasets import load_toy_yearly

df_long = load_toy_yearly()

weights_daily = {'categoricalDayofWeek': 5,
           'randomNegative': 1,
         'sp500high': 2,
         'wabashaTemp': 1}

weights_hourly = {'traffic_volume': 10}

from autots import AutoTS
model = AutoTS(forecast_length = forecast_length, frequency = 'infer',
               prediction_interval = 0.9, ensemble = True, weighted = False,
               max_generations = 1, num_validations = 2, validation_method = 'even',
               drop_most_recent = 1)

from autots.evaluator.auto_ts import fake_regressor
preord_regressor_train, preord_regressor_forecast = fake_regressor(df_long, forecast_length = forecast_length, date_col = 'datetime', value_col = 'value', id_col = 'series_id')

model = model.fit(df_long, date_col = 'datetime', value_col = 'value', id_col = 'series_id')
# model = model.fit(df_long, date_col = 'datetime', value_col = 'value', id_col = 'series_id', weights = weights_hourly) # and weighted = True
# model = model.fit(df_long, date_col = 'datetime', value_col = 'value', id_col = 'series_id', preord_regressor = preord_regressor_train)

print(model.best_model['Model'].iloc[0])
print(model.best_model['ModelParameters'].iloc[0])

prediction = model.predict(preord_regressor = preord_regressor_forecast)
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
initial_results = model.initial_results.model_results
# validation results
validation_results = model.validation_results.model_results

"""
Import/Export

example_filename = "example_export.csv" #.csv/.json
model.export_template(example_filename, models = 'best', n = 5)
model = model.import_template(example_filename, method = 'add on')
print("Overwrite template is: {}".format(str(model.initial_template)))
"""

"""
from autots.tools.shaping import long_to_wide
df_wide = long_to_wide(df_long, date_col = 'datetime', value_col = 'value',
                       id_col = 'series_id', frequency = 'infer', aggfunc = 'first')
"""


"""
https://packaging.python.org/tutorials/packaging-projects/

python -m pip install --user --upgrade setuptools wheel
cd /to project directory
python setup.py sdist bdist_wheel
twine upload dist/*

Merge dev to master on GitHub and create release (include .tar.gz)
"""

"""
pip install fredapi # if using samples
conda install -c conda-forge fbprophet
pip install mxnet==1.4.1
    pip install mxnet-cu90mkl==1.4.1 # if you want GPU and have Intel CPU
pip install gluonts==0.4.0
    pip install git+https://github.com/awslabs/gluon-ts.git #if you want dev version
pip install pmdarima==1.4.0 
pip uninstall numpy # might be necessary, even twice, followed by the following
pip install numpy==1.17.4 # gluonts likes to force numpy back to 1.14, but 1.17 seems to be fine with it
pip install sktime==0.3.1
"""
# to gluon ds
# to xgboost ds
# trim series to first actual value
    # gluon start
    # per series, trim before first na
    # from regressions, remove rows based on % columns that are NaN
# *args, **kwargs
