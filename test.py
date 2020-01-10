import numpy as np
import pandas as pd


from autots.datasets import load_toy_daily
df_long = load_toy_daily()

from autots import AutoTS
model = AutoTS(max_generations = 5, num_validations = 3, validation_method = 'even')
model = model.fit(df_long)
prediction = model.predict()
prediction.forecast

weights = {'categoricalDayofWeek': 1,
           'randomNegative': 1,
         'sp500high': 4,
         'wabashaTemp': 1}



"""
Recombine best two of each model, if two or more present
Duplicates still seem to be occurring in the genetic template runs
Inf appearing in MAE and RMSE (possibly all NaN in test)
Na Tolerance for test in simple_train_test_split
Relative/Absolute Imports and reduce package reloading
User regressor to sklearn model regression_type

Predict method
    PredictWitch + Inverse Categorical
    Regressor Flag - warns if Regressor provided in train but not forecast (regression_type == 'User')
    Best Model
Import/export template
Sklearn models

ARIMA + Detrend fails

Things needing testing:
    Confirm per_series weighting works properly
    Passing in Start Dates - (Test)
    Different frequencies
    Various verbose inputs
    Test holidays on non-daily data
    Handle categorical forecasts where forecast leaves known values


"""

"""
The verbosity level: if non zero, progress messages are printed. Above 50, the output is sent to stdout. The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are reported."
In addition, the Glossary (search for "verbose") says this:
"Logging is not handled very consistently in Scikit-learn at present, but when it is provided as an option, the verbose parameter is usually available to choose no logging (set to False). Any True value should enable some logging, but larger integers (e.g. above 10) may be needed for full verbosity. Verbose logs are usually printed to Standard Output. Estimators should not produce any output on Standard Output with the default verbose setting."
https://stackoverflow.com/questions/29995249/verbose-argument-in-scikit-learn
"""
# to gluon ds
# to xgboost ds
# GENERATOR of series for per series methods
# trim series to first actual value
    # gluon start
    # per series, trim before first na
    # from regressions, remove rows based on % columns that are NaN
# *args, **kwargs
