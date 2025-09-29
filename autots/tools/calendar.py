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


def gregorian_to_chinese(datetime_index, epoch=2444238.5):
    """Convert a pandas DatetimeIndex to Chinese Lunar calendar. Potentially has errors."""
    if isinstance(datetime_index, (str, list)):
        datetime_input = pd.to_datetime(datetime_index).sort_values()
    else:
        datetime_input = datetime_index.sort_values()
    expanded_dates = pd.date_range(
        datetime_input[0] - pd.Timedelta(days=365), datetime_input[-1], freq='D'
    )
    min_year = np.min(expanded_dates.year)
    moon_df = moon_phase_df(expanded_dates, epoch=epoch)
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


def gregorian_to_hindu(datetime_index, method: str = "lunar"):
    """Convert a pandas DatetimeIndex to Hindu calendar date components.
    Hindu calendar has numerous regional variations. This implementation
    aims for compatibility with common internationally recognized Hindu holidays.

    Args:
        datetime_index (pd.DatetimeIndex): pandas DatetimeIndex
        method (str): 'simple' or 'lunar'. Simple is faster. Lunar is more accurate.
    """
    if isinstance(datetime_index, (str, list)):
        datetime_input = pd.to_datetime(datetime_index).sort_values()
    else:
        datetime_input = datetime_index.sort_values()
    # Expand date range to cover previous year for new moons
    expanded_dates = pd.date_range(
        datetime_input[0] - pd.Timedelta(days=400), datetime_input[-1], freq='D'
    )

    # Calculate the Hindu year offset from the minimum Gregorian year
    min_year = np.min(expanded_dates.year)
    # Get moon phases
    moon_df = moon_phase_df(expanded_dates, epoch=2444238.5)

    # Use the original lunar function but adjust the month numbering for Hindu calendar
    lunar_months = lunar_from_lunar(moon_df['new_moon'])
    # Merge with expanded dates
    expanded_dates = pd.concat(
        [pd.Series(0, index=expanded_dates, name="date"), lunar_months], axis=1
    )

    # Forward fill the data
    expanded_dates['syear'] = expanded_dates['syear'].ffill()
    expanded_dates['lunar_month'] = expanded_dates['lunar_month'].ffill()

    # Calculate lunar day (tithi) - days since the start of the lunar month
    expanded_dates['lunar_day'] = (
        expanded_dates.groupby(['syear', 'lunar_month']).cumcount() + 1
    )

    # Calculate the Hindu year
    expanded_dates['hindu_calendar_year'] = expanded_dates['syear'] + min_year - 57

    # Hindu month names in traditional order
    hindu_month_names = {
        1: 'Chaitra',  # March-April
        2: 'Vaishakha',  # April-May
        3: 'Jyeshtha',  # May-June
        4: 'Ashadha',  # June-July
        5: 'Shravana',  # July-August
        6: 'Bhadrapada',  # August-September
        7: 'Ashwin',  # September-October
        8: 'Kartika',  # October-November
        9: 'Margashirsha',  # November-December
        10: 'Pausha',  # December-January
        11: 'Magha',  # January-February
        12: 'Phalguna',  # February-March
    }

    # Adjust month numbering for Hindu calendar
    # Base shift aligns new moons with traditional month starts (Amanta system)
    base_month = ((expanded_dates['lunar_month'] + 10 - 1) % 12) + 1

    # Apply a secondary shift for days after the full moon (Krishna Paksha)
    # to better match the widely used Purnimanta naming convention
    if method == "simple":
        # Improved logic: shift earlier for dates that are clearly in Krishna Paksha
        waning_shift = np.where(expanded_dates['lunar_day'] >= 16, 1, 0)
    elif method == "lunar":
        # Get full moon data from astronomical calculations
        # Use multiple epoch values to account for timing uncertainty
        epochs = [2444238.0, 2444238.5]  # neutral and Asian timezones
        
        # Start with the original date range index
        date_range_index = pd.date_range(
            datetime_input[0] - pd.Timedelta(days=400), datetime_input[-1], freq='D'
        )
        full_moon_combined = pd.Series(0, index=date_range_index)
        
        for epoch in epochs:
            moon_df_epoch = moon_phase_df(date_range_index, epoch=epoch)
            full_moon_combined = full_moon_combined | moon_df_epoch['full_moon']
        
        expanded_dates = pd.concat([expanded_dates, full_moon_combined.rename('full_moon')], axis=1)
        expanded_dates['full_moon'] = expanded_dates['full_moon'].fillna(0)

        # Use actual full moon occurrences to determine Krishna Paksha
        # In Purnimanta system, the full moon day is the LAST day of the month
        # Only days AFTER the full moon shift to the next month (Krishna Paksha)
        waning_shift = expanded_dates.groupby(['syear', 'lunar_month'])[
            'full_moon'
        ].transform('cumsum')
        # Shift only occurs on days AFTER the full moon (not on the full moon day itself)
        waning_shift = (waning_shift > 0) & (expanded_dates['full_moon'] == 0)
        waning_shift = np.where(waning_shift, 1, 0)
        
        # Special adjustment for known calendar edge cases where the base month calculation
        # doesn't align with established cultural/religious calendar practices
        # This specifically handles Holi dates that fall on full moons in late February/early March
        # which should be in Phalguna according to traditional Hindu calendar usage
        full_moon_days = expanded_dates['full_moon'] == 1
        gregorian_months = pd.Series(expanded_dates.index.month, index=expanded_dates.index)
        is_late_feb_early_mar = (gregorian_months == 2) | (gregorian_months == 3)
        needs_phalguna_correction = full_moon_days & (base_month == 11) & is_late_feb_early_mar
        waning_shift = np.where(needs_phalguna_correction, 1, waning_shift)
    else:
        raise ValueError("method must be one of 'simple' or 'lunar'")
    expanded_dates['hindu_month_number'] = (
        (base_month + waning_shift - 1) % 12
    ) + 1

    expanded_dates['hindu_month_name'] = expanded_dates['hindu_month_number'].map(
        hindu_month_names
    )
    # Return the data for the input dates
    result = expanded_dates.loc[
        datetime_input,
        ['hindu_calendar_year', 'hindu_month_number', 'hindu_month_name', 'lunar_day'],
    ].rename(columns={'hindu_calendar_year': 'lunar_year'}).rename_axis(index='date')

    return result
