"""Manage holiday features."""

import numpy as np
import pandas as pd
from autots.tools.shaping import infer_frequency


def holiday_flag(
    DTindex,
    country: str = 'US',
    encode_holiday_type: bool = False,
    holidays_subdiv=None,
):
    """Create a 0/1 flag for given datetime index. Includes fallback to pandas for US holidays if holidays package unavailable.

    Args:
        DTindex (panda.DatetimeIndex): DatetimeIndex of dates to create flags
        country (str): to pass through to python package Holidays
            also accepts a list of countries, but not a list of subdivisions
        encode_holiday_type (bool): if True, each holiday gets a unique integer column, if False, 0/1 for all holidays
        holidays_subdiv (str): subdivision (ie state), if used

    Returns:
        pd.DataFrame with DatetimeIndex
    """
    use_index = DTindex.copy()
    # extend the index to make sure all holidays are captured in holiday flag
    if encode_holiday_type:
        frequency = infer_frequency(use_index)
        new_index = pd.date_range(
            use_index[-1], end=use_index[-1] + pd.Timedelta(days=900), freq=frequency
        )
        # just new index wasn't enough, although another option might be to add more than 1 year to new index
        prev_index = pd.date_range(
            use_index[0] - pd.Timedelta(days=365), end=use_index[0], freq=frequency
        )
        use_index = prev_index[:-1].append(use_index.append(new_index[1:]))

    if isinstance(country, str):
        country = [country]
    elif isinstance(country, dict):
        country = list(country.keys())
        # subdivisions = list(country.values())

    holiday_list = []
    for hld in country:
        if hld == "RU":
            hld = "UA"
        elif hld == 'CN':
            hld = 'TW'
        hld = str(hld).upper()
        if hld in ['US', "USA", "UNITED STATES"]:
            try:
                holi_days = query_holidays(
                    use_index,
                    country="US",
                    encode_holiday_type=encode_holiday_type,
                    holidays_subdiv=holidays_subdiv,
                )
            except Exception:
                from pandas.tseries.holiday import USFederalHolidayCalendar

                # uses pandas calendar as backup in the event holidays fails
                holi_days = (
                    USFederalHolidayCalendar()
                    .holidays()
                    .to_series()[use_index[0] : use_index[-1]]
                )
                holi_days = pd.Series(np.repeat(1, len(holi_days)), index=holi_days)
                holi_days = holi_days.reindex(use_index).fillna(0)
                holi_days = holi_days.rename("holiday_flag")
        else:
            holi_days = query_holidays(
                use_index,
                country=hld,
                encode_holiday_type=encode_holiday_type,
                holidays_subdiv=holidays_subdiv,
            )
        if not encode_holiday_type:
            holi_days.name = str(holi_days.name) + '_' + str(hld)
        holiday_list.append(holi_days.reindex(DTindex))

    return_df = pd.concat(holiday_list, axis=1, ignore_index=False)
    if encode_holiday_type:
        return return_df.loc[:, ~return_df.columns.duplicated()]
    else:
        return return_df


def query_holidays(
    DTindex, country: str, encode_holiday_type: bool = False, holidays_subdiv=None
):
    """Query holidays package for dates.

    Args:
        DTindex (panda.DatetimeIndex): DatetimeIndex of dates to create flags
        country (str): to pass through to python package Holidays
        encode_holiday_type (bool): if True, each holiday gets a unique integer column, if False, 0/1 for all holidays
    """
    import holidays

    # need the extra years to make sure it captures less common holidays
    # mostly it is the (Observed) holiday flags showing up that cause issues
    years = list(range(DTindex[0].year - 2, DTindex[-1].year + 4))
    try:
        country_holidays_base = holidays.country_holidays(
            country, years=years, subdiv=holidays_subdiv
        )
    except Exception:
        print(
            f'country {country} not recognized. Filter holiday_countries by holidays.utils.list_supported_countries() to remove this warning'
        )
        country_holidays_base = {}
    if encode_holiday_type:
        # sorting to hopefully get consistent encoding across runs (requires long period...)
        if not country_holidays_base:
            country_holidays = pd.Series('HolidayFlag', index=DTindex)
        else:
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
        holi_days = pd.get_dummies(country_holidays, dtype=float)
    else:
        country_holidays = country_holidays_base.keys()
        if not country_holidays:
            holi_days = pd.Series(0, name='HolidayFlag', dtype=int)
        else:
            holi_days = pd.Series(
                np.repeat(1, len(country_holidays)),
                index=pd.DatetimeIndex(country_holidays),
                name="HolidayFlag",
                dtype=int,
            )
    # do some messy stuff to make sub daily data (hourly) align with daily holidays
    try:
        holi_days.index = pd.DatetimeIndex(holi_days.index).normalize()
        holi_days = holi_days.reindex(pd.DatetimeIndex(DTindex).normalize()).fillna(0)
        holi_days.index = DTindex
    except Exception:
        holi_days = holi_days.reindex(DTindex).fillna(0)
    return holi_days
