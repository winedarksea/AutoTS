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

from autots.tools.shaping import values_to_numeric

categorical_transformer = values_to_numeric(df_wide)
df_wide_numeric = categorical_transformer.dataframe
# categorical_transformer.encoder.inverse_transform(df_wide_numeric['categoricalDayofWeek'].values.reshape(-1, 1))

df4 = df_wide_numeric.copy()
from autots.tools.impute import fill_na
df4 = fill_na(df4)

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

from autots.tools.shaping import simple_train_test_split

df_train, df_test = simple_train_test_split(df_subset, forecast_length = 14)


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
