import numpy as np
import pandas as pd
import datetime
import json

forecast_length = 14

weighted = False
frequency = '1D'
prediction_interval = 0.9
no_negatives = False
seed = 425
holiday_country = 'US'
ensemble = True

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

from autots.tools.profile import data_profile
profile_df = data_profile(df_wide_numeric)

from autots.tools.shaping import categorical_inverse
df_cat_inverse = categorical_inverse(categorical_transformer, df_wide_numeric)


from autots.tools.shaping import simple_train_test_split
df_train, df_test = simple_train_test_split(df_wide_numeric, forecast_length = forecast_length)
preord_regressor_train = preord_regressor[df_train.index]
preord_regressor_test = preord_regressor[df_test.index]

if weighted == False:
    weights = {x:1 for x in df_train.columns}

model_results = pd.DataFrame()
model_results_per_series_mae = pd.DataFrame()
model_results_per_series_smape = pd.DataFrame()

forecasts = []
upper_forecasts = []
lower_forecasts = []
forecasts_list = []

from autots.evaluator.auto_model import ModelNames
from autots.tools.transform import RandomTransform
from autots.evaluator.auto_model import ModelMonster
def RandomTemplate(n: int = 10):
    """"
    Returns a template dataframe of randomly generated transformations, models, and hyperparameters
    
    Args:
        n (int): number of random models to return
    """
    n = abs(int(n))
    template = pd.DataFrame()
    counter = 0
    while (len(template.index) < n):
        model_str = np.random.choice(ModelNames)
        param_dict = ModelMonster(model_str).get_new_params()
        trans_dict = RandomTransform()
        row = pd.DataFrame({
                'Model': model_str,
                'ModelParameters': json.dumps(param_dict),
                'TransformationParameters': json.dumps(trans_dict),
                'Ensemble': 0
                }, index = [0])
        template = pd.concat([template, row], axis = 0, ignore_index = True)
        template.drop_duplicates(inplace = True)
        counter += 1
        if counter > (n * 3):
            break
    return template
    

# 
transformation_dict = {'outlier': 'clip2std',
                       'fillNA' : 'ffill', 
                       'transformation' : 'RollingMean10',
                       'context_slicer' : 'None'}
model_str = "FBProphet"
parameter_dict = {'holiday':True,
                  'regression_type' : 'User'}
model_str = "ARIMA"
parameter_dict = {'p': 1,
                  'd': 0,
                  'q': 1,
                  'regression_type' : 'User'}

template = RandomTemplate(20)
for index, row in template.iterrows():
    model_str = row['Model']
    parameter_dict = json.loads(row['ModelParameters'])
    transformation_dict = json.loads(row['TransformationParameters'])
    print("Row: {} with model {}".format(str(index), model_str))
    try:
        from autots.evaluator.auto_model import ModelPrediction
        df_forecast = ModelPrediction(df_train, forecast_length,transformation_dict, 
                                      model_str, parameter_dict, frequency=frequency, 
                                      prediction_interval=prediction_interval, 
                                      no_negatives=no_negatives,
                                      preord_regressor_train = preord_regressor_train,
                                      preord_regressor_forecast = preord_regressor_test, 
                                      holiday_country = holiday_country,
                                      startTimeStamps = profile_df.loc['FirstDate'])
        
        from autots.evaluator.metrics import PredictionEval
        model_error = PredictionEval(df_forecast, df_test, series_weights = weights)
        
        result = pd.DataFrame({
                'Model': model_str,
                'ModelParameters': json.dumps(df_forecast.model_parameters),
                'TransformationParameters': json.dumps(df_forecast.transformation_parameters),
                'TransformationRuntime': df_forecast.transformation_runtime,
                'FitRuntime': df_forecast.fit_runtime,
                'PredictRuntime': df_forecast.predict_runtime,
                'TotalRuntime': df_forecast.fit_runtime + df_forecast.predict_runtime + df_forecast.transformation_runtime,
                'Ensemble': 0,
                'Exceptions': np.nan
                }, index = [0])
        a = pd.DataFrame(model_error.avg_metrics_weighted.rename(lambda x: x + '_weighted')).transpose()
        result = pd.concat([result, pd.DataFrame(model_error.avg_metrics).transpose(), a], axis = 1)
        
        model_results = pd.concat([model_results, result], axis = 0).reset_index(drop = True)
        model_results_per_series_smape = model_results_per_series_smape.append(model_error.per_series_metrics.loc['smape'], ignore_index = True)
        model_results_per_series_mae = model_results_per_series_mae.append(model_error.per_series_metrics.loc['mae'], ignore_index = True)
        
        if ensemble:
            forecasts_list.extend([model_str])
            forecasts.extend([df_forecast.forecast])
            upper_forecasts.extend([df_forecast.upper_forecast])
            lower_forecasts.extend([df_forecast.lower_forecast])
    
    except Exception as e:
        result = pd.DataFrame({
            'Model': model_str,
            'ModelParameters': json.dumps(parameter_dict),
            'TransformationParameters': json.dumps(transformation_dict),
            'Ensemble': 0,
            'TransformationRuntime': datetime.timedelta(0),
            'FitRuntime': datetime.timedelta(0),
            'PredictRuntime': datetime.timedelta(0),
            'TotalRuntime': datetime.timedelta(0),
            'Exceptions': str(e)
            }, index = [0])
        model_results = pd.concat([model_results, result], axis = 0).reset_index(drop = True)


"""
Transformation Dict
ModelName
Parameter Dict
Model name * Series, all SMAPE
Dict of Forecast Values
Errors in Results DF
Passing in Start Dates

Ensemble
Multiple validation
Point to probability isn't working particularly well

ARIMA + Detrend fails

Combine multiple metrics into 'score'
Ranked by score
"""

"""
Managing template errors...

Confirm per_series weighting

"""



















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
