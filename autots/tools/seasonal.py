# -*- coding: utf-8 -*-
"""
seasonal

@author: Colin
"""
import random
import pandas as pd


def seasonal_int(include_one: bool = False, small=False):
    """Generate a random integer of typical seasonalities."""
    prob_dict = {
        'random_int': 0.1,
        1: 0.05,
        2: 0.05,
        4: 0.05,
        7: 0.15,
        10: 0.01,
        12: 0.1,
        24: 0.1,
        28: 0.1,
        60: 0.1,
        96: 0.04,
        168: 0.01,
        364: 0.1,
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
    if not include_one and str(lag) == '1':
        lag = 'random_int'
    if lag == 'random_int':
        lag = random.randint(2, 100)
    if small:
        lag = lag if lag < 364 else 364
    return int(lag)


def date_part(DTindex, method: str = 'simple', set_index: bool = True, polynomial_degree: int = None):
    """Create date part columns from pd.DatetimeIndex.

    Args:
        DTindex (pd.DatetimeIndex): datetime index to provide dates
        method (str): expanded, recurring, or simple
            simple - just day, year, month, weekday
            expanded - all available futures
            recurring - all features that should commonly repeat without aging
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
                ).values,
            }
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
                    'month_end': (DTindex.is_month_end).astype(int),
                    'month_start': (DTindex.is_month_start).astype(int),
                    "quarter_end": (DTindex.is_quarter_end).astype(int),
                    'year_end': (DTindex.is_year_end).astype(int),
                    'daysinmonth': DTindex.daysinmonth,
                    'epoch': pd.to_numeric(
                        DTindex, errors='coerce', downcast='integer'
                    ).values,
                }
            )
            date_part_df = pd.concat([date_part_df, date_part_df2], axis=1)
    if polynomial_degree is not None:
        from sklearn.preprocessing import PolynomialFeatures

        date_part_df = pd.DataFrame(
            PolynomialFeatures(
                polynomial_degree, include_bias=False
            ).fit_transform(date_part_df)
        )
        date_part_df.columns = ['dp' + str(x) for x in date_part_df.columns]
    if set_index:
        date_part_df.index = DTindex
    return date_part_df
