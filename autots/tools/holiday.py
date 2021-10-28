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
    years = list(range(DTindex[0].year, DTindex[-1].year + 1))
    if country.upper() in ['US', "USA", "United States"]:
        try:
            import holidays

            country_holidays_base = holidays.CountryHoliday('US', years=years)
            country_holidays = country_holidays_base.keys()
            holi_days = pd.Series(
                np.repeat(1, len(country_holidays)),
                index=pd.DatetimeIndex(country_holidays),
                name="HolidayFlag",
            )
            holi_days = holi_days.reindex(DTindex).fillna(0)
        except Exception:
            from pandas.tseries.holiday import USFederalHolidayCalendar

            # uses pandas calendar as backup in the event holidays fails
            holi_days = (
                USFederalHolidayCalendar()
                .holidays()
                .to_series()[DTindex[0]: DTindex[-1]]
            )
            holi_days = pd.Series(np.repeat(1, len(holi_days)), index=holi_days)
            holi_days = holi_days.reindex(DTindex).fillna(0)
            holi_days.rename("HolidayFlag", inplace=True)
    else:
        import holidays

        country_holidays_base = holidays.CountryHoliday(country.upper())
        country_holidays = country_holidays_base.keys()
        # country_holidays = country_holidays_base[DTindex[0]: DTindex[-1]]
        # all_days = pd.Series(np.repeat(0, len(DTindex)), index=DTindex)
        holi_days = pd.Series(
            np.repeat(1, len(country_holidays)),
            index=pd.DatetimeIndex(country_holidays),
            name="HolidayFlag",
        )
        holi_days = holi_days.reindex(DTindex).fillna(0)
        # holi_days = all_days.combine(holi_days, func=max).fillna(0)
        # holi_days.rename("HolidayFlag", inplace=True)

    return holi_days[DTindex]
