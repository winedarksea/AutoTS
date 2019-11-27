# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:51:13 2019

@author: Owner
"""

from autots.datasets.fred import get_fred_data

df_long = get_fred_data('93873d40f10c20fe6f6e75b1ad0aed4d')

from autots.tools.shaping import long_to_wide

df_wide = long_to_wide(df_long, date_col = 'date', value_col = 'value',
                       id_col = 'series_id', frequency = '1D', na_tolerance = 0.5,
                       drop_data_older_than_periods = 1000)


# train/test split + cross validation
# trim series to first actual value
# then fill na
