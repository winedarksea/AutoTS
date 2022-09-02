# -*- coding: utf-8 -*-
"""
seasonal

@author: Colin
"""
import random
import pandas as pd
from autots.tools.lunar import moon_phase


def seasonal_int(include_one: bool = False, small=False, very_small=False):
    """Generate a random integer of typical seasonalities.

    Args:
        include_one (bool): whether to include 1 in output options
        small (bool): if True, keep below 364
        very_small (bool): if True keep below 30
    """
    prob_dict = {
        -1: 0.1,  # random int
        1: 0.05,  # previous day
        2: 0.1,
        4: 0.05,  # quarters
        7: 0.15,  # week
        10: 0.01,
        12: 0.1,  # months
        24: 0.1,  # months or hours
        28: 0.1,  # days in month to weekday
        60: 0.05,
        96: 0.04,  # quarter in days
        168: 0.01,
        364: 0.1,  # year to weekday
        1440: 0.01,
        420: 0.01,
        52: 0.01,
        84: 0.01,
    }
    lag = random.choices(
        list(prob_dict.keys()),
        list(prob_dict.values()),
        k=1,
    )[0]
    if lag == -1:
        lag = random.randint(2, 100)
    if not include_one and lag == 1:
        lag = seasonal_int(include_one=include_one, small=small, very_small=very_small)
    if small:
        lag = lag if lag < 364 else 364
    if very_small:
        while lag > 30:
            lag = seasonal_int(include_one=include_one, very_small=very_small)
    return int(lag)


def date_part(
    DTindex,
    method: str = 'simple',
    set_index: bool = True,
    polynomial_degree: int = None,
):
    """Create date part columns from pd.DatetimeIndex.

    Args:
        DTindex (pd.DatetimeIndex): datetime index to provide dates
        method (str): expanded, recurring, or simple
            simple - just day, year, month, weekday
            expanded - all available futures
            recurring - all features that should commonly repeat without aging
            simple_2
            simple_3
            simple_binarized
            expanded_binarized
        set_index (bool): if True, return DTindex as index of df
        polynomial_degree (int): add this degree of sklearn polynomial features if not None

    Returns:
        pd.Dataframe with DTindex
    """
    if "_poly" in method:
        method = method.replace("_poly", "")
        polynomial_degree = 2
    if method == 'recurring':
        date_part_df = pd.DataFrame(
            {
                'month': DTindex.month,
                'day': DTindex.day,
                'weekday': DTindex.weekday,
                'weekend': (DTindex.weekday > 4).astype(int),
                'hour': DTindex.hour,
                'quarter': DTindex.quarter,
                'midyear': (
                    (DTindex.dayofyear > 74) & (DTindex.dayofyear < 258)
                ).astype(
                    int
                ),  # 2 season
            }
        )
    elif method in ["simple_2", "simple_2_poly"]:
        date_part_df = pd.DataFrame(
            {
                'month': DTindex.month,
                'day': DTindex.day,
                'weekday': DTindex.weekday,
                'weekend': (DTindex.weekday > 4).astype(int),
                'epoch': pd.to_numeric(
                    DTindex, errors='coerce', downcast='integer'
                ).values / 100000000000,
            }
        )
    elif method in ["simple_3", "lunar_phase"]:
        # trying to *prevent* it from learning holidays for this one
        date_part_df = pd.DataFrame(
            {
                'month': pd.Categorical(
                    DTindex.month, categories=list(range(12)), ordered=True
                ),
                'weekday': pd.Categorical(
                    DTindex.weekday, categories=list(range(7)), ordered=True
                ),
                'weekend': (DTindex.weekday > 4).astype(int),
                'quarter': DTindex.quarter,
                'epoch': DTindex.to_julian_date(),
            }
        )
        date_part_df['weekday'] = date_part_df['month'].astype(
            pd.CategoricalDtype(categories=list(range(6)))
        )
        date_part_df = pd.get_dummies(date_part_df, columns=['month', 'weekday'])
        if method == "lunar_phase":
            date_part_df['phase'] = moon_phase(DTindex)
    elif "simple_binarized" in method:
        date_part_df = pd.DataFrame(
            {
                'month': pd.Categorical(
                    DTindex.month, categories=list(range(12)), ordered=True
                ),
                'weekday': pd.Categorical(
                    DTindex.weekday, categories=list(range(7)), ordered=True
                ),
                'day': DTindex.day,
                'weekend': (DTindex.weekday > 4).astype(int),
                'epoch': DTindex.to_julian_date(),
            }
        )
        date_part_df = pd.get_dummies(date_part_df, columns=['month', 'weekday'])
    elif method in "expanded_binarized":
        date_part_df = pd.DataFrame(
            {
                'month': pd.Categorical(
                    DTindex.month, categories=list(range(12)), ordered=True
                ),
                'weekday': pd.Categorical(
                    DTindex.weekday, categories=list(range(7)), ordered=True
                ),
                'day': pd.Categorical(
                    DTindex.day, categories=list(range(31)), ordered=True
                ),
                'weekdayofmonth': pd.Categorical(
                    (DTindex.day - 1) // 7 + 1,
                    categories=list(range(5)), ordered=True,
                ),
                'weekend': (DTindex.weekday > 4).astype(int),
                'quarter': DTindex.quarter,
                'epoch': DTindex.to_julian_date(),
            }
        )
        date_part_df = pd.get_dummies(
            date_part_df, columns=['month', 'weekday', 'day', 'weekdayofmonth']
        )
    else:
        # method == "simple"
        date_part_df = pd.DataFrame(
            {
                'year': DTindex.year,
                'month': DTindex.month,
                'day': DTindex.day,
                'weekday': DTindex.weekday,
            }
        )
        if method == 'expanded':
            try:
                weekyear = DTindex.isocalendar().week.to_numpy()
            except Exception:
                weekyear = DTindex.week
            date_part_df2 = pd.DataFrame(
                {
                    'hour': DTindex.hour,
                    'week': weekyear,
                    'quarter': DTindex.quarter,
                    'dayofyear': DTindex.dayofyear,
                    'midyear': (
                        (DTindex.dayofyear > 74) & (DTindex.dayofyear < 258)
                    ).astype(
                        int
                    ),  # 2 season
                    'weekend': (DTindex.weekday > 4).astype(int),
                    'weekdayofmonth': (DTindex.day - 1) // 7 + 1,
                    'month_end': (DTindex.is_month_end).astype(int),
                    'month_start': (DTindex.is_month_start).astype(int),
                    "quarter_end": (DTindex.is_quarter_end).astype(int),
                    'year_end': (DTindex.is_year_end).astype(int),
                    'daysinmonth': DTindex.daysinmonth,
                    'epoch': pd.to_numeric(
                        DTindex, errors='coerce', downcast='integer'
                    ).values
                    - 946684800000000000,
                    'us_election_year': (DTindex.year % 4 == 0).astype(int),
                }
            )
            date_part_df = pd.concat([date_part_df, date_part_df2], axis=1)
    if polynomial_degree is not None:
        from sklearn.preprocessing import PolynomialFeatures

        date_part_df = pd.DataFrame(
            PolynomialFeatures(polynomial_degree, include_bias=False).fit_transform(
                date_part_df
            )
        )
        date_part_df.columns = ['dp' + str(x) for x in date_part_df.columns]
    if set_index:
        date_part_df.index = DTindex
    return date_part_df
