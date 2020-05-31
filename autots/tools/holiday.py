"""Manage holiday features."""
import numpy as np
import pandas as pd


def holiday_flag(DTindex, country: str = 'US'):
    """Create a 0/1 flag for given datetime index.

    Args:
        DTindex (panda.DatetimeIndex): DatetimeIndex of dates to create flags
        country (str): to pass through to python package Holidays

    Returns:
        pandas.Series() with DatetimeIndex and column 'HolidayFlag'
    """
    if country.upper() == 'US':
        try:
            import holidays
            country_holidays = holidays.CountryHoliday('US')
            country_holidays = country_holidays[DTindex[0]:DTindex[-1]]
            all_days = pd.Series(np.repeat(0, len(DTindex)), index = DTindex)
            holi_days = pd.Series(np.repeat(1, len(country_holidays)),
                                  index = pd.DatetimeIndex(country_holidays))
            holi_days = all_days.combine(holi_days, func = max).fillna(0)
            holi_days.rename("HolidayFlag", inplace = True)
        except Exception:
            from pandas.tseries.holiday import USFederalHolidayCalendar
            # uses pandas calendar as backup in the event holidays fails
            holi_days = USFederalHolidayCalendar().holidays().to_series()[DTindex[0]:DTindex[-1]]
            all_days = pd.Series(np.repeat(0, len(DTindex)), index = DTindex)
            holi_days = pd.Series(np.repeat(1, len(holi_days)), index = holi_days)
            holi_days = all_days.combine(holi_days, func = max).fillna(0)
            holi_days.rename("HolidayFlag", inplace = True)
    else:
        import holidays
        country_holidays = holidays.CountryHoliday(country.upper())
        country_holidays = country_holidays[DTindex[0]:DTindex[-1]]
        all_days = pd.Series(np.repeat(0, len(DTindex)), index = DTindex)
        holi_days = pd.Series(np.repeat(1, len(country_holidays)),
                              index = pd.DatetimeIndex(country_holidays))
        holi_days = all_days.combine(holi_days, func = max).fillna(0)
        holi_days.rename("HolidayFlag", inplace = True)

    return holi_days[DTindex]
