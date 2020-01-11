import numpy as np
import pandas as pd


from autots.datasets import load_toy_daily
df_long = load_toy_daily()

from autots import AutoTS
model = AutoTS(forecast_length = 14, frequency = 'infer',
               prediction_interval = 0.9, ensemble = True, weighted = False,
               max_generations = 5, num_validations = 2, validation_method = 'even')
model = model.fit(df_long, date_col = 'date', value_col = 'value', id_col = 'series_id' )

print(model.best_model['Model'].iloc[0])
print(model.best_model['ModelParameters'].iloc[0])

prediction = model.predict()
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
model_results = model.main_results.model_results
# validation results
try:
    validation_results = model.validation_results.model_results
except:
    pass

weights = {'categoricalDayofWeek': 1,
           'randomNegative': 1,
         'sp500high': 4,
         'wabashaTemp': 1}

# preord_regressor = pd.Series(np.random.randint(0, 100, size = len(df_wide.index)), index = df_wide.index )

"""
Recombine best two of each model, if two or more present
Duplicates still seem to be occurring in the genetic template runs
Inf appearing in MAE and RMSE (possibly all NaN in test)
Na Tolerance for test in simple_train_test_split
Relative/Absolute Imports and reduce package reloading
User regressor to sklearn model regression_type
Import/export template
ARIMA + Detrend fails

Things needing testing:
    Confirm per_series weighting works properly
    Passing in Start Dates - (Test)
    Different frequencies
    Various verbose inputs
    Test holidays on non-daily data
    Handle categorical forecasts where forecast leaves known values

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
