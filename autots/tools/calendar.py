# -*- coding: utf-8 -*-
"""Calendar conversion functions.

Includes Lunar, Chinese lunar, and Arabic lunar
"""
import numpy as np
import pandas as pd
from autots.tools.lunar import moon_phase_df


def lunar_from_lunar(new_moon):
    """Assumes continuous daily data and pre-needed start."""
    new_moon_dates = new_moon[new_moon == 1]
    # assuming the rule "first new moon on or after January 21st"
    filtered = new_moon_dates[
        ~((new_moon_dates.index.month == 1) & (new_moon_dates.index.day < 21))
    ].to_frame()
    filtered['year'] = filtered.index.year
    filtered['datetime'] = filtered.index
    new_years = filtered.groupby('year')['datetime'].min()
    new_years = pd.Series(1, index=new_years, name='new_years').to_frame()
    new_years['syear'] = np.arange(0, new_years.shape[0])
    new_years = pd.concat([new_moon_dates, new_years], axis=1)
    new_years['syear'] = new_years['syear'].ffill()
    new_years['lunar_month'] = new_years.groupby('syear').cumcount() + 1
    return new_years


def lunar_from_lunar_full(full_moon):
    """Assumes continuous daily data and pre-needed start."""
    full_moon_dates = full_moon[full_moon == 1]
    # assuming the rule "first full moon on or after January 21st"
    filtered = full_moon_dates[
        ~((full_moon_dates.index.month <= 3) & (full_moon_dates.index.day < 23))
    ].to_frame()
    filtered['year'] = filtered.index.year
    filtered['datetime'] = filtered.index
    new_years = filtered.groupby('year')['datetime'].min()
    new_years = pd.Series(1, index=new_years, name='new_years').to_frame()
    new_years['syear'] = np.arange(0, new_years.shape[0])
    new_years = pd.concat([full_moon_dates, new_years], axis=1)
    new_years['syear'] = new_years['syear'].ffill()
    new_years['lunar_month'] = new_years.groupby('syear').cumcount() + 1
    return new_years


def gregorian_to_christian_lunar(datetime_index):
    """Convert a pandas DatetimeIndex to Christian Lunar calendar. Aspiration it doesn't work exactly."""
    if isinstance(datetime_index, (str, list)):
        datetime_input = pd.to_datetime(datetime_index).sort_values()
    else:
        datetime_input = datetime_index.sort_values()
    expanded_dates = pd.date_range(
        datetime_input[0] - pd.Timedelta(days=365), datetime_input[-1], freq='D'
    )
    min_year = np.min(expanded_dates.year)
    moon_df = moon_phase_df(expanded_dates, epoch=2444238.5)
    lunar_months = lunar_from_lunar_full(moon_df['full_moon'])
    expanded_dates = pd.concat(
        [pd.Series(0, index=expanded_dates, name="date"), lunar_months], axis=1
    )
    expanded_dates['syear'] = expanded_dates['syear'].ffill()
    expanded_dates['lunar_month'] = expanded_dates['lunar_month'].ffill()
    expanded_dates['lunar_day'] = (
        expanded_dates.groupby(['syear', 'lunar_month']).cumcount() + 1
    )
    expanded_dates['lunar_year'] = expanded_dates['syear'] + min_year
    expanded_dates["weekofmonth"] = (expanded_dates["lunar_day"] - 1) // 7 + 1
    expanded_dates['dayofweek'] = expanded_dates.index.dayofweek
    return expanded_dates.loc[
        datetime_index,
        ['lunar_year', 'lunar_month', 'lunar_day', 'weekofmonth', 'dayofweek'],
    ].astype(int)


def gregorian_to_chinese(datetime_index):
    """Convert a pandas DatetimeIndex to Chinese Lunar calendar. Potentially has errors."""
    if isinstance(datetime_index, (str, list)):
        datetime_input = pd.to_datetime(datetime_index).sort_values()
    else:
        datetime_input = datetime_index.sort_values()
    expanded_dates = pd.date_range(
        datetime_input[0] - pd.Timedelta(days=365), datetime_input[-1], freq='D'
    )
    min_year = np.min(expanded_dates.year)
    moon_df = moon_phase_df(expanded_dates, epoch=2444238.5)
    lunar_months = lunar_from_lunar(moon_df['new_moon'])
    expanded_dates = pd.concat(
        [pd.Series(0, index=expanded_dates, name="date"), lunar_months], axis=1
    )
    expanded_dates['syear'] = expanded_dates['syear'].ffill()
    expanded_dates['lunar_month'] = expanded_dates['lunar_month'].ffill()
    expanded_dates['lunar_day'] = (
        expanded_dates.groupby(['syear', 'lunar_month']).cumcount() + 1
    )
    expanded_dates['lunar_year'] = expanded_dates['syear'] + min_year
    return (
        expanded_dates.loc[datetime_index, ['lunar_year', 'lunar_month', 'lunar_day']]
        .astype(int)
        .rename_axis(index='date')
    )


def to_jd(year, month, day):
    """Determine Julian day count from Islamic date. From convertdate by fitnr."""
    return (
        day
        + np.ceil(29.5 * (month - 1))
        + (year - 1) * 354
        + np.floor((3 + (11 * year)) / 30)
        + 1948439.5
    ) - 1


def gregorian_to_islamic(date, epoch_adjustment=1.5):
    """Calculate Islamic dates for pandas DatetimeIndex. Approximately. From convertdate by fitnr.

    Args:
        epoch_adjustment (float): 1.0 and that needs to be adjusted by about +/- 0.5 to account for timezone
    """
    if isinstance(date, (str, list)):
        date = pd.to_datetime(date)
    jd = date.to_julian_date()
    jd = np.floor(jd) + epoch_adjustment
    year = np.floor(((30 * (jd - 1948439.5)) + 10646) / 10631)
    month = np.minimum(12, np.ceil((jd - (29 + to_jd(year, 1, 1))) / 29.5) + 1)
    day = (jd - to_jd(year, month, 1)).astype(int) + 1
    return (
        pd.DataFrame({'year': year, 'month': month, 'day': day}, index=date)
        .astype(int)
        .rename_axis(index='date')
    )


def heb_is_leap(year):
    if (((7 * year) + 1) % 19) < 7:
        return True
    return False


def _elapsed_months(year):
    return (235 * year - 234) // 19


def _elapsed_days(year):
    months_elapsed = _elapsed_months(year)
    parts_elapsed = 204 + 793 * (months_elapsed % 1080)
    hours_elapsed = (
        5 + 12 * months_elapsed + 793 * (months_elapsed // 1080) + parts_elapsed // 1080
    )
    conjunction_day = 1 + 29 * months_elapsed + hours_elapsed // 24
    conjunction_parts = 1080 * (hours_elapsed % 24) + parts_elapsed % 1080

    if (
        (conjunction_parts >= 19440)
        or (
            (conjunction_day % 7 == 2)
            and (conjunction_parts >= 9924)
            and not heb_is_leap(year)
        )
        or (
            (conjunction_day % 7 == 1)
            and conjunction_parts >= 16789
            and heb_is_leap(year - 1)
        )
    ):
        alt_day = conjunction_day + 1
    else:
        alt_day = conjunction_day
    if alt_day % 7 in [0, 3, 5]:
        alt_day += 1

    return alt_day


def _days_in_year(year):
    return _elapsed_days(year + 1) - _elapsed_days(year)


def _long_cheshvan(year):
    """Returns True if Cheshvan has 30 days"""
    return _days_in_year(year) % 10 == 5


def _short_kislev(year):
    """Returns True if Kislev has 29 days"""
    return _days_in_year(year) % 10 == 3


def _month_length(year, month):
    """Months start with Nissan (Nissan is 1 and Tishrei is 7)"""
    if month in [1, 3, 5, 7, 11]:
        return 30
    if month in [2, 4, 6, 10, 13]:
        return 29
    if month == 12:
        if heb_is_leap(year):
            return 30
        return 29
    if month == 8:  # if long Cheshvan return 30, else return 29
        if _long_cheshvan(year):
            return 30
        return 29
    if month == 9:  # if short Kislev return 29, else return 30
        if _short_kislev(year):
            return 29
        return 30
    raise ValueError("Invalid month")


def gregorian_to_hebrew(dates):
    """Convert pd.Datetimes to a Hebrew date. From pyluach by simlist.

    This is the slowest of the lot and needs to be improved.
    """
    if isinstance(dates, (str, list)):
        day = pd.to_datetime(dates).to_julian_date()
    else:
        day = dates.to_julian_date()
    if (day <= 347997).any():
        raise ValueError("According to this calendar, this time doesn't exist")

    jd = (day + 0.5).astype(int)  # Try to account for half day
    jd -= 347997
    years = (jd // 365).astype(int) + 2  # try that to debug early years
    date_list = []
    for idx in range(len(jd)):
        c_jd = jd[idx]
        year = years[idx]
        first_day = _elapsed_days(year)

        while first_day > c_jd:
            year -= 1
            first_day = _elapsed_days(year)

        months = [7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6]
        if not heb_is_leap(year):
            months.remove(13)

        days_remaining = c_jd - first_day
        for month in months:
            if days_remaining >= _month_length(year, month):
                days_remaining -= _month_length(year, month)
            else:
                date_list.append(
                    pd.DataFrame(
                        {"year": year, "month": month, 'day': days_remaining + 1},
                        index=[dates[idx]],
                    )
                )
                break
    return pd.concat(date_list, axis=0).rename_axis(index='date')


def gregorian_to_hindu(datetime_index):
    """Convert a pandas DatetimeIndex to Hindu calendar date components.
    Hindu calendar has numerous regional variations.

    Used an llm to put this one together.
    It gets the dates wrong, but it does appear to have correlated consistency so may still work for modeling.
    Suggestions for improvement welcomed.
    """
    if isinstance(datetime_index, (str, list)):
        datetime_input = pd.to_datetime(datetime_index).sort_values()
    else:
        datetime_input = datetime_index.sort_values()
    # Expand date range to cover previous year for new moons
    expanded_dates = pd.date_range(
        datetime_input[0] - pd.Timedelta(days=365), datetime_input[-1], freq='D'
    )
    min_year = np.min(expanded_dates.year)
    # Get moon phases
    moon_df = moon_phase_df(expanded_dates, epoch=2444238.5)
    # Use new moon dates to define lunar months (Amanta system)
    lunar_months = lunar_from_lunar(moon_df['new_moon'])
    # Merge with expanded dates
    expanded_dates = pd.concat(
        [pd.Series(0, index=expanded_dates, name="date"), lunar_months], axis=1
    )
    expanded_dates['syear'] = expanded_dates['syear'].ffill()
    expanded_dates['lunar_month'] = expanded_dates['lunar_month'].ffill()
    # Calculate lunar day (tithi)
    expanded_dates['lunar_day'] = (
        expanded_dates.groupby(['syear', 'lunar_month']).cumcount() + 1
    )
    expanded_dates['lunar_year'] = expanded_dates['syear'] + min_year
    # Assign approximate Hindu month names
    hindu_month_names = {
        1: 'Chaitra',
        2: 'Vaishakha',
        3: 'Jyeshtha',
        4: 'Ashadha',
        5: 'Shravana',
        6: 'Bhadrapada',
        7: 'Ashwin',
        8: 'Kartika',
        9: 'Margashirsha',
        10: 'Pausha',
        11: 'Magha',
        12: 'Phalguna',
    }
    # Adjust lunar_month to fit within 12 months
    expanded_dates['hindu_month_number'] = (
        (expanded_dates['lunar_month'] - 1) % 12
    ) + 1
    expanded_dates['hindu_month_name'] = expanded_dates['hindu_month_number'].map(
        hindu_month_names
    )
    # Return the data for the input dates
    return expanded_dates.loc[
        datetime_input,
        ['lunar_year', 'hindu_month_number', 'hindu_month_name', 'lunar_day'],
    ].rename_axis(index='date')
