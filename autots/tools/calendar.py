# -*- coding: utf-8 -*-
"""Calendar conversion functions.
Created on Thu Jul 14 16:41:16 2022

@author: Colin
"""
import numpy as np
import pandas as pd
from autots.tools.lunar import moon_phase_df


def lunar_from_lunar(new_moon):
    """Assumes continuous daily data and pre-needed start."""
    new_moon_dates = new_moon[new_moon == 1]
    # assuming the rule "first new moon on or after January 21st"
    filtered = new_moon_dates[~((new_moon_dates.index.month == 1) & (new_moon_dates.index.day < 21))].to_frame()
    filtered['year'] = filtered.index.year
    filtered['datetime'] = filtered.index
    new_years = filtered.groupby('year')['datetime'].min()
    new_years = pd.Series(1, index=new_years, name='new_years').to_frame()
    new_years['syear'] = np.arange(0, new_years.shape[0])
    new_years = pd.concat([new_moon_dates, new_years], axis=1)
    new_years['syear'] = new_years['syear'].ffill()
    new_years['lunar_month'] = new_years.groupby('syear').cumcount() + 1
    return new_years


def gregorian_to_chinese(datetime_index):
    """Convert a pandas DatetimeIndex to Chinese Lunar calendar. Potentially has errors."""
    if isinstance(datetime_index, (str, list)):
        datetime_input = pd.to_datetime(
            datetime_index, infer_datetime_format=True
        ).sort_values()
    else:
        datetime_input = datetime_index.sort_values()
    expanded_dates = pd.date_range(
        datetime_input[0] - pd.Timedelta(days=365), datetime_input[-1], freq='D'
    )
    min_year = np.min(expanded_dates.year)
    moon_df = moon_phase_df(expanded_dates, epoch=2444238.5)
    lunar_months = lunar_from_lunar(moon_df['new_moon'])
    expanded_dates = pd.concat([pd.Series(0, index=expanded_dates, name="date"), lunar_months], axis=1)
    expanded_dates['syear'] = expanded_dates['syear'].ffill()
    expanded_dates['lunar_month'] = expanded_dates['lunar_month'].ffill()
    expanded_dates['lunar_day'] = expanded_dates.groupby(['syear', 'lunar_month']).cumcount() + 1
    expanded_dates['lunar_year'] = expanded_dates['syear'] + min_year
    return expanded_dates.loc[datetime_index, ['lunar_year', 'lunar_month', 'lunar_day']].astype(int)


def to_jd(year, month, day):
    """Determine Julian day count from Islamic date. From convertdate by fitnr."""
    return (
        day + np.ceil(29.5 * (month - 1)) + (year - 1) * 354
        + np.floor((3 + (11 * year)) / 30) + 1948439.5
    ) - 1


def gregorian_to_islamic(date):
    """Calculate Islamic dates for pandas DatetimeIndex. Approximately. From convertdate by fitnr."""
    if isinstance(date, (str, list)):
        date = pd.to_datetime(date, infer_datetime_format=True)
    jd = date.to_julian_date()
    jd = np.floor(jd) + 1.5
    year = np.floor(((30 * (jd - 1948439.5)) + 10646) / 10631)
    month = np.minimum(12, np.ceil((jd - (29 + to_jd(year, 1, 1))) / 29.5) + 1)
    day = (jd - to_jd(year, month, 1)).astype(int) + 1
    return pd.DataFrame({'year': year, 'month': month, 'day': day}, index=date).astype(int)
