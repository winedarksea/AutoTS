"""Manage holiday features."""
import numpy as np
import pandas as pd


def holiday_flag(DTindex, country: str = 'US', encode_holiday_type: bool = False, holidays_subdiv=None):
    """Create a 0/1 flag for given datetime index. Includes fallback to pandas for US holidays if holidays package unavailable.

    Args:
        DTindex (panda.DatetimeIndex): DatetimeIndex of dates to create flags
        country (str): to pass through to python package Holidays
        encode_holiday_type (bool): if True, each holiday gets a unique integer column, if False, 0/1 for all holidays

    Returns:
        pandas.Series() with DatetimeIndex and name 'HolidayFlag'
    """
    country = str(country).upper()
    if country in ['US', "USA", "UNITED STATES"]:
        try:
            holi_days = query_holidays(
                DTindex, country="US", encode_holiday_type=encode_holiday_type, holidays_subdiv=holidays_subdiv
            )
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
            holi_days = holi_days.rename("holiday_flag")
    else:
        holi_days = query_holidays(
            DTindex, country=country, encode_holiday_type=encode_holiday_type, holidays_subdiv=holidays_subdiv
        )

    return holi_days


def query_holidays(DTindex, country: str, encode_holiday_type: bool = False, holidays_subdiv=None):
    """Query holidays package for dates.

    Args:
        DTindex (panda.DatetimeIndex): DatetimeIndex of dates to create flags
        country (str): to pass through to python package Holidays
        encode_holiday_type (bool): if True, each holiday gets a unique integer column, if False, 0/1 for all holidays
    """
    import holidays

    years = list(range(DTindex[0].year, DTindex[-1].year + 1))
    country_holidays_base = holidays.CountryHoliday(country, years=years, subdiv=holidays_subdiv)
    if encode_holiday_type:
        # sorting to hopefully get consistent encoding across runs (requires long period...)
        country_holidays = pd.Series(country_holidays_base).sort_values()
        """
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=999)
        holi_days = pd.Series(
            encoder.fit_transform(country_holidays.to_numpy().reshape(-1, 1)).flatten(),
            name="HolidayFlag",
        )
        # since zeroes are reserved for non-holidays
        holi_days = holi_days + 1
        holi_days.index = country_holidays.index
        """
        holi_days = pd.get_dummies(country_holidays)
    else:
        country_holidays = country_holidays_base.keys()
        holi_days = pd.Series(
            np.repeat(1, len(country_holidays)),
            index=pd.DatetimeIndex(country_holidays),
            name="HolidayFlag",
        )
    # do some messy stuff to make sub daily data (hourly) align with daily holidays
    try:
        holi_days.index = pd.DatetimeIndex(holi_days.index).normalize()
        holi_days = holi_days.reindex(pd.DatetimeIndex(DTindex).normalize()).fillna(0)
        holi_days.index = DTindex
    except Exception:
        holi_days = holi_days.reindex(DTindex).fillna(0)
    return holi_days
