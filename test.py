import numpy as np
import pandas as pd

forecast_length = 14

weighted = False
frequency = '1D'
prediction_interval = 0.9
no_negatives = False
seed = 425
holiday_country = 'US'

df_long = pd.read_csv("SampleTimeSeries.csv")
df_long['date'] = pd.to_datetime(df_long['date'], infer_datetime_format = True)


#from autots.datasets.fred import get_fred_data
#df_long = get_fred_data('XXXXXXXXXXXxx')


from autots.tools.shaping import long_to_wide

df_wide = long_to_wide(df_long, date_col = 'date', value_col = 'value',
                       id_col = 'series_id', frequency = frequency, na_tolerance = 0.95,
                       drop_data_older_than_periods = 1000, aggfunc = 'first')


preord_regressor = pd.Series(np.random.randint(0, 100, size = len(df_wide.index)), index = df_wide.index )





from autots.tools.shaping import values_to_numeric

categorical_transformer = values_to_numeric(df_wide)
df_wide_numeric = categorical_transformer.dataframe

from autots.tools.shaping import categorical_inverse
df_cat_inverse = categorical_inverse(categorical_transformer, df_wide_numeric)


from autots.tools.shaping import simple_train_test_split
df_train, df_test = simple_train_test_split(df_wide_numeric, forecast_length = forecast_length)
preord_regressor_train = preord_regressor[df_train.index]
preord_regressor_test = preord_regressor[df_test.index]

if weighted == False:
    weights = {x:1 for x in df_train.columns}


transformation_dict = {'outlier': 'clip2std',
                       'fillNA' : 'ffill', 
                       'transformation' : 'Detrend',
                       'context_slicer' : 'None'}
model_str = "FBProphet"
parameter_dict = {'holiday':True,
                  'regression_type' : 'User'}

from autots.evaluator.auto_model import ModelPrediction
df_forecast = ModelPrediction(df_train, forecast_length,transformation_dict, 
                              model_str, parameter_dict, frequency=frequency, 
                              prediction_interval=prediction_interval, 
                              no_negatives=no_negatives,
                              preord_regressor_train = preord_regressor_train,
                              preord_regressor_forecast = preord_regressor_test, 
                              holiday_country = holiday_country)

from autots.evaluator.metrics import PredictionEval
model_error = PredictionEval(df_forecast, df_test, series_weights = weights)





import json
result = pd.DataFrame({
        'Model': model_str,
        'ModelParameters': json.dumps(df_forecast.model_parameters),
        'TransformationParameters': json.dumps(df_forecast.transformation_parameters),
        'FitRuntime': df_forecast.fit_runtime,
        'PredictRuntime': df_forecast.predict_runtime,
        'TotalRuntime': df_forecast.fit_runtime + df_forecast.predict_runtime
        }, index = [0])
a = pd.DataFrame(model_error.avg_metrics_weighted.rename(lambda x: x + '_weighted')).transpose()
result = pd.concat([result, pd.DataFrame(model_error.avg_metrics).transpose(), a], axis = 1)

model_error.per_series_metrics.loc['smape']

model_error.per_series_metrics.loc['mae']



"""
Transformation Dict
ModelName
Parameter Dict
Capture Errors, Total Runtime
Model name * Series, all SMAPE
Dict of Forecast Values

Ensemble
Multiple validation
Point to probability isn't working particularly well
"""
"""
Managing template errors...

Confirm per_series weighting

"""

















from autots.tools.profile import data_profile
# currently doesn't ignore nans
# profile_df = data_profile(df_wide)

from autots.tools.shaping import subset_series

df_subset = subset_series(df_wide_numeric, n = 10, na_tolerance = 0.5, random_state = 425)




# to gluon ds
# to xgboost ds

# train/test split + cross validation
    # to be more sensitive to NaNs in train/test split count
    # option to skip if na_tolerance is not met in both train and test.
# GENERATOR of series for per series methods
# trim series to first actual value
    # gluon start
    # per series, trim before first na
    # from regressions, remove rows based on % columns that are NaN
# then fill na
# *args, **kwargs
