"""
Reshape
"""
import numpy as np
import pandas as pd
# import typing
# id_col: typing.Optional[str]=None

def long_to_wide(df, date_col: str, value_col: str, id_col: str, 
                 frequency: str = "infer", na_tolerance: float = 0.25,
                 drop_data_older_than_periods: int = 10000, 
                 drop_most_recent: bool = False):
    """
    Takes long data and converts into wide, cleaner data
    
    args:
        df - a pandas dataframe having three columns:
            date_col - the name of the column containing dates, preferrably already in pandas datetime format
            value_col - the name of the column with the values of the time series (ie sales $)
            id_col - name of the id column, unique for each time series
        
        frequency - frequency in string of alias for DateOffset object, normally "1D" -daily, "MS" -month start etc.
            currently, aliases are listed somewhere in here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        
        na_tolerance - allow up to this percent of values to be NaN, else drop the entire series
        
        drop_data_older_than_periods - cut off older data because eventually you just get too much
            10,000 is meant to be rather high, normally for daily data I'd use only the last couple of years, say 1500 samples
            
        drop_most_recent - if to drop the most recent data point
            useful if you pull monthly data before month end, and you don't want an incomplete month appearing complete
    """
    timeseries_long = df.copy()
    timeseries_long['value'] = timeseries_long[value_col]
    timeseries_long['date'] = timeseries_long[date_col]
    
    # Attempt to convert to datetime format if not already
    try:
        timeseries_long['date'] = pd.to_datetime(timeseries_long['date'], infer_datetime_format = True)
    except Exception:
        raise ValueError("Could not convert date to datetime format. Incorrect column name or preformat with pandas to_datetime")
    
    # handle no id_col for if only one time series
    # this isn't particularly robust, hence an id_col is required
    if id_col is None:
        print("No id_col passed, using only first time series...")
        timeseries_long['series_id'] = 'First'
        timeseries_long.drop_duplicates(subset = 'date', keep = 'first', inplace = True)
    else:
        timeseries_long['series_id'] = timeseries_long[id_col]
    
    # drop any unnecessary columns
    timeseries_long = timeseries_long[['date','series_id','value']]
    
    # pivot to different wide shape
    timeseries_seriescols = timeseries_long.pivot_table(values='value', index='date', columns='series_id')
    timeseries_seriescols = timeseries_seriescols.sort_index(ascending=True)
    
    # drop older data, because too much of a good thing...
    # okay, so technically it may not be periods until after asfreq, but whateva
    timeseries_seriescols = timeseries_seriescols.tail(drop_data_older_than_periods)
    
    # infer frequency, not recommended
    if frequency == 'infer':
        frequency = pd.infer_freq(timeseries_seriescols.index, warn = True)
    
    # fill missing dates in index with NaN
    timeseries_seriescols = timeseries_seriescols.asfreq(frequency, fill_value=np.nan)
   
    # remove series with way too many NaNs - probably those of a different frequency, or brand new
    na_threshold = int(len(timeseries_seriescols.index) * (1 - na_tolerance))
    timeseries_seriescols = timeseries_seriescols.dropna(axis = 1, thresh=na_threshold)
    
    if len(timeseries_seriescols.columns) < 1:
        raise ValueError("All series filtered, probably the na_tolerance is too low or frequency is incorrect")
    
    # drop most recent value when desired
    if drop_most_recent:
        timeseries_seriescols.drop(timeseries_seriescols.tail(1).index, inplace = True)
    
    return timeseries_seriescols