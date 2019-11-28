# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:51:13 2019

@author: Owner
"""
import numpy as np
import pandas as pd

df_long = pd.read_csv("SampleTimeSeries.csv")
df_long['date'] = pd.to_datetime(df_long['date'], infer_datetime_format = True)


from autots.datasets.fred import get_fred_data
df_long = get_fred_data('93873d40f10c20fe6f6e75b1ad0aed4d')



from autots.tools.shaping import long_to_wide

df_wide = long_to_wide(df_long, date_col = 'date', value_col = 'value',
                       id_col = 'series_id', frequency = '1D', na_tolerance = 0.95,
                       drop_data_older_than_periods = 1000, aggfunc = 'first')

from autots.tools.profile import data_profile

profile_df = data_profile(df_wide)

from autots.tools.shaping import subset_series

df_subset = subset_series(df_wide, n = 10, na_tolerance = 0.5, random_state = 425)

from autots.tools.shaping import values_to_numeric

transformer_result = values_to_numeric(df_subset)
df_subset_numeric = transformer_result.dataframe
# transformer_result.encoder.inverse_transform(df_subset_numeric['categoricalDayofWeek'].values.reshape(-1, 1))

from autots.tools.shaping import simple_train_test_split

df_train, df_test = simple_train_test_split(df_subset_numeric, forecast_length = 14)

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
