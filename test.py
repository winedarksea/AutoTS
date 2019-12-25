import numpy as np
import pandas as pd

forecast_length = 14

weighted = False
transformation_dict = {'outlier': 'clip2std',
                       'fillNA' : 'fake date', 
                       'transformation' : 'PowerTransformer',
                       'context_slicer' : 'None'}
model_str = "LastValueNaive"
parameter_dict = {}

df_long = pd.read_csv("SampleTimeSeries.csv")
df_long['date'] = pd.to_datetime(df_long['date'], infer_datetime_format = True)


#from autots.datasets.fred import get_fred_data
#df_long = get_fred_data('XXXXXXXXXXXxx')


from autots.tools.shaping import long_to_wide

df_wide = long_to_wide(df_long, date_col = 'date', value_col = 'value',
                       id_col = 'series_id', frequency = '1D', na_tolerance = 0.95,
                       drop_data_older_than_periods = 1000, aggfunc = 'first')

from autots.tools.shaping import values_to_numeric

categorical_transformer = values_to_numeric(df_wide)
df_wide_numeric = categorical_transformer.dataframe

from autots.tools.shaping import categorical_inverse
df_cat_inverse = categorical_inverse(categorical_transformer, df_wide_numeric)


from autots.tools.shaping import simple_train_test_split
df_train, df_test = simple_train_test_split(df_wide_numeric, forecast_length = forecast_length)

if weighted == False:
    weights = {x:1 for x in df_train.columns}


def ModelPrediction(df_train, forecast_length: int, frequency: str = 'infer', prediction_interval: float = 0.9, transformation_dict: dict, model_str: str, parameter_dict: dict):
    """Feed parameters into modeling pipeline
    
    Args:
        df_train (pandas.DataFrame): numeric training dataset of DatetimeIndex and series as cols
      
    Returns:
        PredictionObject (autots.PredictionObject): Prediction from AutoTS model object
    """
    
    from autots.tools.transform import GeneralTransformer
    transformer_object = GeneralTransformer(outlier=transformation_dict['outlier'],
                                            fillNA = transformation_dict['fillNA'], 
                                            transformation = transformation_dict['transformation']).fit(df_train)
    df_train_transformed = transformer_object.transform(df_train)
    
    if transformation_dict['context_slicer'] in ['2ForecastLength','HalfMax']:
        from autots.tools.transform import simple_context_slicer
        df_train_transformed = simple_context_slicer(df_train_transformed, method = transformation_dict['context_slicer'], forecast_length = forecast_length)
    
    from autots.evaluator.auto_model import ModelMonster
    model = ModelMonster(model_str, parameter_dict, frequency = frequency, prediction_interval = prediction_interval)
    model = model.fit(df_train_transformed)
    df_forecast = model.predict(forecast_length = forecast_length, regressor = "007")
    
    df_forecast.lower_forecast = transformer_object.inverse_transform(df_forecast.lower_forecast)
    df_forecast.forecast = transformer_object.inverse_transform(df_forecast.forecast)
    df_forecast.upper_forecast = transformer_object.inverse_transform(df_forecast.upper_forecast)
    
    return df_forecast

from autots.evaluator.metrics import PredictionEval
model_error = PredictionEval(df_forecast, df_test)

"""
Managing template errors...
"""

















df4 = df_wide_numeric.copy()
from autots.tools.impute import FillNA
df4 = FillNA(df4)

from autots.tools.transform import RollingMeanTransformer
meaner = RollingMeanTransformer(window = 10).fit(df4)
temp2 = meaner.transform(df4)
test = temp2.tail(21)

meaner = RollingMeanTransformer(window = 10).fit(df4.head(120))
temp = meaner.transform(df4.head(120))
testtemp = meaner.inverse_transform(test, trans_method = 'forecast')
testDF = pd.concat([df4.tail(21), testtemp], axis = 1)
testDF2 = pd.concat([temp2.head(120), temp.head(120)], axis = 1)


from autots.tools.transform import Detrend
detrender = Detrend().fit(df4)
temp = detrender.transform(df4)
temp = detrender.inverse_transform(temp)
test = pd.concat([df4, temp], axis = 1)

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
