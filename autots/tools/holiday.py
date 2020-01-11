import numpy as np
import pandas as pd
import holidays
from pandas.tseries.holiday import USFederalHolidayCalendar

def holiday_flag(DTindex, country: str =  'US'):
    """
    Creates a 0/1 flag for any datetime
    
    Args:
        DTindex (panda.DatetimeIndex): DatetimeIndex of dates to create flags for
        country (str): to pass through to python package Holidays
    
    Returns:
        pandas.Series() with DatetimeIndex and column 'HolidayFlag'
    """
    if country.upper() == 'US':
        try:
            country_holidays = holidays.CountryHoliday('US')
            country_holidays = country_holidays[DTindex[0]:DTindex[-1]]
            all_days = pd.Series(np.repeat(0, len(DTindex)), index = DTindex)
            holi_days = pd.Series(np.repeat(1, len(country_holidays)), index = pd.DatetimeIndex(country_holidays))
            holi_days = all_days.combine(holi_days, func = max)
            holi_days.rename("HolidayFlag", inplace = True)
        except:
            
            holi_days = USFederalHolidayCalendar().holidays().to_series()[DTindex[0]:DTindex[-1]]
            all_days = pd.Series(np.repeat(0, len(DTindex)), index = DTindex)
            holi_days = pd.Series(np.repeat(1, len(holi_days)), index = holi_days)
            holi_days = all_days.combine(holi_days, func = max)
            holi_days.rename("HolidayFlag", inplace = True)
    else:
        country_holidays = holidays.CountryHoliday(country.upper())
        country_holidays = country_holidays[DTindex[0]:DTindex[-1]]
        all_days = pd.Series(np.repeat(0, len(DTindex)), index = DTindex)
        holi_days = pd.Series(np.repeat(1, len(country_holidays)), index = pd.DatetimeIndex(country_holidays))
        holi_days = all_days.combine(holi_days, func = max)
        holi_days.rename("HolidayFlag", inplace = True)
        
    return holi_days