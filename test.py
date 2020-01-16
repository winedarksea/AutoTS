import numpy as np
import pandas as pd

forecast_length = 12
from autots.datasets import load_toy_daily
from autots.datasets import load_toy_monthly

df_long = load_toy_monthly()

weights_daily = {'categoricalDayofWeek': 5,
           'randomNegative': 1,
         'sp500high': 2,
         'wabashaTemp': 1}

from autots import AutoTS
model = AutoTS(forecast_length = forecast_length, frequency = 'infer',
               prediction_interval = 0.9, ensemble = True, weighted = False,
               max_generations = 2, num_validations = 2, validation_method = 'even')

from autots.evaluator.auto_ts import fake_regressor
preord_regressor_train, preord_regressor_forecast = fake_regressor(df_long, forecast_length = forecast_length)

model = model.fit(df_long, date_col = 'date', value_col = 'value', id_col = 'series_id')
# model = model.fit(df_long, date_col = 'date', value_col = 'value', id_col = 'series_id', weights = weights_daily) # and weighted = True
# model = model.fit(df_long, date_col = 'date', value_col = 'value', id_col = 'series_id', preord_regressor = preord_regressor_train)

print(model.best_model['Model'].iloc[0])
print(model.best_model['ModelParameters'].iloc[0])

prediction = model.predict(preord_regressor = preord_regressor_forecast)
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
initial_results = model.initial_results.model_results
# validation results
try:
    validation_results = model.validation_results.model_results
except Exception:
    pass

"""
Import/Export
"""
example_filename = "example_export.csv" #.csv/.json
# model.export_template(example_filename, models = 'best', n = 5)
# model = model.import_template(example_filename, method = 'add on')
# print("Overwrite template is: {}".format(str(model.initial_template)))


"""
Recombine best two of each model, if two or more present
Inf appearing in MAE and RMSE (possibly all NaN in test)
Na Tolerance for test in simple_train_test_split
Relative/Absolute Imports and reduce package reloading
User regressor to sklearn model regression_type
Annual, Monthly, Weekly, Hourly sample data
Format of Regressor - allow multiple input to at least sklearn models
'Age' regressor as an option in addition to User/Holiday
Handle categorical forecasts where forecast leaves range of known values

Things needing testing:
    With and without regressor
    With and without weighting
    Passing in Start Dates - (Test)
    Different frequencies
    Various verbose inputs
    Test holidays on non-daily data
   

https://packaging.python.org/tutorials/packaging-projects/
python -m pip install --user --upgrade setuptools wheel
cd /to project directory
python setup.py sdist bdist_wheel
twine upload dist/*
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
# GENERATOR of series for per series methods
# trim series to first actual value
    # gluon start
    # per series, trim before first na
    # from regressions, remove rows based on % columns that are NaN
# *args, **kwargs
