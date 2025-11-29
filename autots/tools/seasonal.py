# -*- coding: utf-8 -*-
"""
seasonal

@author: Colin
"""
import random
import numpy as np
import pandas as pd
from autots.tools.lunar import moon_phase
from autots.tools.window_functions import sliding_window_view
from autots.tools.holiday import holiday_flag
from autots.tools.wavelet import offset_wavelet, create_narrowing_wavelets
from autots.tools.shaping import infer_frequency
from autots.tools.changepoints import (
    create_changepoint_features,
    half_yr_spacing,
)


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
        288: 0.001,  # daily at 5 minute intervals
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


date_part_methods = [
    "recurring",
    "simple",
    "expanded",
    "simple_2",
    "simple_3",
    'lunar_phase',
    "simple_binarized",
    "simple_binarized_poly",
    "expanded_binarized",
    'common_fourier',
    'common_fourier_rw',
    'common_fourier_rw_lag',
    'anchored_warped_fourier:quarter_end',
    'anchored_segment_fourier:quarter_end',
    'anchored_warped_fourier:us_school',
    'anchored_segment_fourier:us_school',
]

origin_ts = "2030-01-01"

COMMON_FOURIER_RW_LAG_ORDER = 3  # mirror ARDL order for exog lag depth


ANCHOR_SCHEMES = {
    'quarter_end': {
        'type': 'static_month_day',
        'dates': [(3, 31), (6, 30), (9, 30), (12, 31)],
    },
    'us_school': {
        'type': 'holiday',
        'country': 'US',
        'holidays': [
            ['Memorial Day'],
            ['Labor Day'],
            ['Thanksgiving Day'],
            ['Christmas Day', 'Christmas Day (observed)'],
        ],
        'approximate_dayofyear': [145, 247, 329, 359],
    },
}


def _is_seasonality_order_list(data):
    """[365.25, 14] would be true and [7, 365.25] would be false"""
    # Check if the data is a list with exactly two items
    if not isinstance(data, list) or len(data) != 2:
        return False

    # Check if the first item is either a float or integer
    first_item = data[0]
    if not isinstance(first_item, (int, float)):
        return False

    # Check if the second item is an integer
    second_item = data[1]
    if not isinstance(second_item, int):
        return False
    elif second_item > 120:
        # unlikely there would be a request for more than 120 fourier orders
        return False

    return True


def date_part(
    DTindex,
    method: str = 'simple',
    set_index: bool = True,
    polynomial_degree: int = None,
    holiday_country: str = None,
    holiday_countries_used: bool = True,
    lags: int = None,  # inspired by ARDL
    forward_lags: int = None,
):
    """Create date part columns from pd.DatetimeIndex.

    If you date_part isn't recognized, you will see a ['year', 'month' 'day', 'weekday'] output

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
            common_fourier
        set_index (bool): if True, return DTindex as index of df
        polynomial_degree (int): add this degree of sklearn polynomial features if not None
        holdiay_country (list or str): names of countries to pull calendar holidays for
        holiday_countries_used (bool): to use holiday_country if given
        lags (int): if not None, include the past N previous index date parts
        forward_lags (int): if not None, include the future N index date parts
    Returns:
        pd.Dataframe with DTindex
    """
    # recursive
    is_seasonality_list = _is_seasonality_order_list(method)
    if is_seasonality_list:
        # because JSON can't do tuples and it's list, but want to have a pair of (seasonality, order) for fouriers
        date_part_df = fourier_df(DTindex, seasonality=method[0], order=method[1])
    elif isinstance(method, list):
        all_seas = []
        for seas in method:
            all_seas.append(date_part(DTindex, method=seas, set_index=True))
        date_part_df = pd.concat(all_seas, axis=1)
    elif "_poly" in str(method):
        method = method.replace("_poly", "")
        polynomial_degree = 2

    # add extra time to make sure full flags are captured
    # this is more a backup for the Cateogorical flags which handle this directly
    expansion_flag = False
    if isinstance(method, str):
        if "binarized" in method and set_index:
            expansion_flag = True

    if expansion_flag:
        # code shared with holiday_flag
        frequency = infer_frequency(DTindex)
        backup = DTindex.copy()
        new_index = pd.date_range(
            DTindex[-1], end=DTindex[-1] + pd.Timedelta(days=900), freq=frequency
        )
        prev_index = pd.date_range(
            DTindex[0] - pd.Timedelta(days=365), end=DTindex[0], freq=frequency
        )
        DTindex = prev_index[:-1].append(DTindex.append(new_index[1:]))

    if isinstance(method, (int, float)):
        date_part_df = fourier_df(DTindex, seasonality=method, order=6)
    elif is_seasonality_list:
        pass
    elif isinstance(method, tuple):
        date_part_df = fourier_df(DTindex, seasonality=method[0], order=method[1])
    elif isinstance(method, list):
        # this handles it already having been run recursively
        # remove duplicate columns if present
        date_part_df = date_part_df.loc[:, ~date_part_df.columns.duplicated()]
    elif isinstance(method, str) and method.startswith('anchored_warped_fourier'):
        date_part_df = anchored_warped_fourier_features(DTindex, method)
    elif isinstance(method, str) and method.startswith('anchored_segment_fourier'):
        date_part_df = anchored_segment_fourier_features(DTindex, method)
    elif method in datepart_components:
        date_part_df = create_datepart_components(DTindex, method)
    elif method == 'recurring':
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
                ).values
                / 100000000000,
            }
        )
    elif method in ["simple_3", "lunar_phase"]:
        # trying to *prevent* it from learning holidays for this one
        date_part_df = pd.DataFrame(
            {
                'month': pd.Categorical(
                    DTindex.month, categories=list(range(1, 13)), ordered=True
                ),
                'weekday': pd.Categorical(
                    DTindex.weekday, categories=list(range(7)), ordered=True
                ),
                'weekend': (DTindex.weekday > 4).astype(int),
                'quarter': DTindex.quarter,
                'epoch': DTindex.to_julian_date(),
            }
        )
        # date_part_df['weekday'] = date_part_df['month'].astype(pd.CategoricalDtype(categories=list(range(6))))
        date_part_df = pd.get_dummies(
            date_part_df, columns=['month', 'weekday'], dtype=float
        )
        if method == "lunar_phase":
            date_part_df['phase'] = moon_phase(DTindex)
    elif "simple_binarized2" in method:
        date_part_df = pd.DataFrame(
            {
                'isoweek': pd.Categorical(
                    DTindex.isocalendar().week,
                    categories=list(range(1, 54)),
                    ordered=True,
                ),
                'weekday': pd.Categorical(
                    DTindex.weekday, categories=list(range(7)), ordered=True
                ),
                'day': DTindex.day,
                'weekend': (DTindex.weekday > 4).astype(int),
                'epoch': DTindex.to_julian_date(),
            }
        )
        date_part_df = pd.get_dummies(
            date_part_df, columns=['isoweek', 'weekday'], dtype=float
        )
    elif "simple_binarized" in method:
        date_part_df = pd.DataFrame(
            {
                'month': pd.Categorical(
                    DTindex.month, categories=list(range(1, 13)), ordered=True
                ),
                'weekday': pd.Categorical(
                    DTindex.weekday, categories=list(range(7)), ordered=True
                ),
                'day': DTindex.day,
                'weekend': (DTindex.weekday > 4).astype(int),
                'epoch': DTindex.to_julian_date(),
            }
        )
        date_part_df = pd.get_dummies(
            date_part_df, columns=['month', 'weekday'], dtype=float
        )
    elif method in "expanded_binarized":
        date_part_df = pd.DataFrame(
            {
                'month': pd.Categorical(
                    DTindex.month, categories=list(range(1, 13)), ordered=True
                ),
                'weekday': pd.Categorical(
                    DTindex.weekday, categories=list(range(7)), ordered=True
                ),
                'day': pd.Categorical(
                    DTindex.day, categories=list(range(1, 32)), ordered=True
                ),
                'weekdayofmonth': pd.Categorical(
                    (DTindex.day - 1) // 7 + 1,
                    categories=list(range(1, 6)),
                    ordered=True,
                ),
                'weekend': (DTindex.weekday > 4).astype(int),
                'quarter': DTindex.quarter,
                'epoch': DTindex.to_julian_date(),
            }
        )
        date_part_df = pd.get_dummies(
            date_part_df,
            columns=['month', 'weekday', 'day', 'weekdayofmonth'],
            dtype=float,
        )
    elif method in ["common_fourier", "common_fourier_rw"]:
        seasonal_list = []
        DTmin = DTindex.min()
        DTmax = DTindex.max()
        # 1 time step will always not work with this
        # for new seasonal_ratio, worse case scenario is 2 steps ahead so one season / 2
        # in order to assure error on wrong seasonality choice in train vs test, we must have different sum orders for each seasonality
        if len(DTindex) <= 1:
            seasonal_ratio = 1  # assume daily, but weekly or monthly seems more likely for 1 step forecasts
        else:
            seasonal_ratio = ((DTmax - DTmin).days + 1) / len(DTindex)
        # seasonal_ratio = (DTmax.year - DTmin.year + 1) / len(DTindex)  # old ratio
        # hourly
        # if seasonal_ratio < 0.001:  # 0.00011 to 0.00023
        if seasonal_ratio < 0.75:  # 0.00011 to 0.00023
            t = DTindex - pd.Timestamp(origin_ts)
            t = (t.days * 24) + (t.components['minutes'] / 60)
            # add hourly, weekly, yearly
            seasonal_list.append(fourier_series(t, p=8766, n=10))
            seasonal_list.append(fourier_series(t, p=24, n=3))
            seasonal_list.append(fourier_series(t, p=168, n=5))
            # interactions
            seasonal_list.append(
                fourier_series(t, p=168, n=5) * fourier_series(t, p=24, n=5)
            )
            seasonal_list.append(
                fourier_series(t, p=168, n=3) * fourier_series(t, p=8766, n=3)
            )
        # daily (+ business day)
        # elif seasonal_ratio < 0.012:  # 0.0027 to 0.0055
        elif seasonal_ratio < 3.5:  # 0.0027 to 0.0055
            t = (DTindex - pd.Timestamp(origin_ts)).days
            # add yearly and weekly seasonality
            seasonal_list.append(fourier_series(t, p=365.25, n=10))
            seasonal_list.append(fourier_series(t, p=7, n=3))
            # interaction
            seasonal_list.append(
                fourier_series(t, p=7, n=5) * fourier_series(t, p=365.25, n=5)
            )
        # weekly
        # elif seasonal_ratio < 0.05:  # 0.019 to 0.038
        elif seasonal_ratio < 12:  # 0.019 to 0.038
            t = (DTindex - pd.Timestamp(origin_ts)).days
            seasonal_list.append(fourier_series(t, p=365.25, n=10))
            seasonal_list.append(fourier_series(t, p=28, n=4))
        # monthly
        # elif seasonal_ratio < 0.5:  # 0.083 to 0.154
        elif seasonal_ratio < 182:  # 0.083 to 0.154
            t = (DTindex - pd.Timestamp(origin_ts)).days
            seasonal_list.append(fourier_series(t, p=365.25, n=3))
            seasonal_list.append(fourier_series(t, p=1461, n=10))
        # yearly
        else:
            t = (DTindex - pd.Timestamp(origin_ts)).days
            seasonal_list.append(fourier_series(t, p=1461, n=10))
        date_part_df = (
            pd.DataFrame(np.concatenate(seasonal_list, axis=1))
            .rename(columns=lambda x: "seasonalitycommonfourier_" + str(x))
            .round(6)
        )
        if method == "common_fourier_rw":
            date_part_df['epoch'] = (DTindex.to_julian_date() ** 0.65).astype(int)
    elif isinstance(method, str) and method == "common_fourier_rw_lag":
        date_part_df = _common_fourier_rw_lag_features(
            DTindex, lag_count=COMMON_FOURIER_RW_LAG_ORDER
        )
    elif "morlet" in method:
        parts = method.split("_")
        if len(parts) >= 2:
            p = parts[1]
        else:
            p = 7
        if len(parts) >= 3:
            order = parts[2]
        else:
            order = 7
        if len(parts) >= 4:
            sigma = parts[3]
        else:
            sigma = 4.0
        date_part_df = seasonal_repeating_wavelet(
            DTindex, p=p, order=order, sigma=sigma, wavelet_type='morlet'
        )
    elif "ricker" in method:
        parts = method.split("_")
        if len(parts) >= 2:
            p = parts[1]
        else:
            p = 7
        if len(parts) >= 3:
            order = parts[2]
        else:
            order = 7
        if len(parts) >= 4:
            sigma = parts[3]
        else:
            sigma = 4.0
        date_part_df = seasonal_repeating_wavelet(
            DTindex, p=p, order=order, sigma=sigma, wavelet_type='ricker'
        )
    elif "db2" in method:
        parts = method.split("_")
        if len(parts) >= 2:
            p = parts[1]
        else:
            p = 7
        if len(parts) >= 3:
            order = parts[2]
        else:
            order = 7
        if len(parts) >= 4:
            sigma = parts[3]
        else:
            sigma = 4.0
        date_part_df = seasonal_repeating_wavelet(
            DTindex, p=p, order=order, sigma=sigma, wavelet_type='db2'
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
                    'us_election_year': (DTindex.year % 4 == 0).astype(
                        int
                    ),  # also Olympics
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
    if isinstance(date_part_df, pd.Index):
        date_part_df = pd.Series(date_part_df)
    if set_index:
        date_part_df.index = DTindex
        if expansion_flag:
            date_part_df = date_part_df.reindex(backup)
    if holiday_country is not None and holiday_countries_used:
        date_part_df = pd.concat(
            [
                date_part_df,
                holiday_flag(
                    DTindex, country=holiday_country, encode_holiday_type=True
                ),
            ],
            axis=1,
            ignore_index=not set_index,
        )
    # recursive
    if lags is not None:
        frequency = infer_frequency(DTindex)
        longer_idx = pd.date_range(
            end=DTindex[-1], periods=len(DTindex) + lags, freq=frequency
        )
        for laggy in range(lags):
            add_X = date_part(
                longer_idx[lags - (laggy + 1) :][0 : len(DTindex)],
                method=method,
                polynomial_degree=polynomial_degree,
            ).rename(columns=lambda x: str(x) + f"_lag{laggy}")
            add_X.index = DTindex
            date_part_df = pd.concat([date_part_df, add_X], axis=1)
    if forward_lags is not None:
        frequency = infer_frequency(DTindex)
        longer_idx = pd.date_range(
            start=DTindex[0], periods=len(DTindex) + forward_lags, freq=frequency
        )
        for laggy in range(forward_lags):
            add_X = date_part(
                longer_idx[laggy + 1 :][0 : len(DTindex)],
                method=method,
                polynomial_degree=polynomial_degree,
            ).rename(columns=lambda x: str(x) + f"_flag{laggy}")
            add_X.index = DTindex
            date_part_df = pd.concat([date_part_df, add_X], axis=1)
    return date_part_df


def _parse_anchor_method(method: str, default_order: int = 6):
    parts = method.split(':')
    if len(parts) < 2:
        raise ValueError(f"Anchored method `{method}` is missing a scheme name.")
    scheme_key = parts[1]
    order = default_order
    if len(parts) >= 3 and parts[2]:
        try:
            order = int(float(parts[2]))
        except Exception:
            pass
    order = max(1, int(order))
    return scheme_key, order


def _resolve_anchor_scheme(scheme_key: str):
    if scheme_key not in ANCHOR_SCHEMES:
        raise ValueError(
            f"Anchored scheme `{scheme_key}` not recognized. Available: {list(ANCHOR_SCHEMES.keys())}"
        )
    return ANCHOR_SCHEMES[scheme_key]


def _align_timestamp_to_tz(ts, tz):
    ts = pd.Timestamp(ts)
    if tz is None:
        if ts.tzinfo is not None:
            return ts.tz_localize(None)
        return ts
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    if ts.tzinfo == tz:
        return ts
    return ts.tz_convert(tz)


def _holiday_anchors_for_year(year, tz, scheme, holiday_cache):
    holidays_list = scheme.get('holidays', [])
    if not holidays_list:
        return []
    if year not in holiday_cache:
        start = pd.Timestamp(year=year, month=1, day=1, tz=tz)
        end = pd.Timestamp(year=year, month=12, day=31, tz=tz)
        anchor_index = pd.date_range(start, end, freq='D')
        holiday_cache[year] = holiday_flag(
            anchor_index, country=[scheme.get('country', 'US')], encode_holiday_type=True
        )
    flags = holiday_cache[year]
    approx = scheme.get('approximate_dayofyear', [])
    start_year = pd.Timestamp(year=year, month=1, day=1, tz=tz)
    anchors = []
    for idx, names in enumerate(holidays_list):
        if isinstance(names, str):
            names = [names]
        selected = None
        for name in names:
            if name in flags.columns:
                hits = flags.index[flags[name] > 0]
                if len(hits) > 0:
                    selected = _align_timestamp_to_tz(hits[0], tz)
                    break
        if selected is None:
            if idx < len(approx):
                selected = start_year + pd.Timedelta(days=max(0, approx[idx] - 1))
            else:
                # spread evenly if approximation missing
                span_days = (
                    pd.Timestamp(year=year + 1, month=1, day=1, tz=tz)
                    - start_year
                ).days
                frac = (idx + 1) / (len(holidays_list) + 1)
                selected = start_year + pd.Timedelta(days=int(round(frac * span_days)))
        anchors.append(_align_timestamp_to_tz(selected, tz))
    return anchors


def _anchor_year_boundaries(year, tz, scheme, holiday_cache):
    start_year = pd.Timestamp(year=year, month=1, day=1, tz=tz)
    next_year = pd.Timestamp(year=year + 1, month=1, day=1, tz=tz)
    anchors = []
    if scheme.get('type') == 'static_month_day':
        for month, day in scheme.get('dates', []):
            anchors.append(pd.Timestamp(year=year, month=month, day=day, tz=tz))
    elif scheme.get('type') == 'holiday':
        anchors.extend(_holiday_anchors_for_year(year, tz, scheme, holiday_cache))
    else:
        raise ValueError(f"Unsupported anchor scheme type `{scheme.get('type')}`.")
    anchors = sorted(anchor for anchor in anchors if start_year <= anchor < next_year)
    boundaries = [start_year]
    for anchor in anchors:
        if anchor > boundaries[-1]:
            boundaries.append(anchor)
    if boundaries[-1] != next_year:
        boundaries.append(next_year)
    return boundaries


def _compute_anchor_positions(DTindex, scheme_key):
    if len(DTindex) == 0:
        return (
            np.array([]),
            np.array([], dtype=int),
            np.array([]),
            0,
        )
    scheme = _resolve_anchor_scheme(scheme_key)
    tz = DTindex.tz
    min_year = DTindex.min().year
    max_year = DTindex.max().year
    holiday_cache = {}
    years = list(range(min_year - 1, max_year + 2))
    
    all_boundaries = []
    for year in years:
        all_boundaries.extend(_anchor_year_boundaries(year, tz, scheme, holiday_cache))
    
    all_boundaries = pd.Series(all_boundaries).drop_duplicates().sort_values().to_list()
    base_length = len(_anchor_year_boundaries(years[0], tz, scheme, holiday_cache))

    if len(all_boundaries) < 2:
        return (
            np.zeros(len(DTindex), dtype=float),
            np.zeros(len(DTindex), dtype=int),
            np.zeros(len(DTindex), dtype=float),
            0,
        )

    # Use pd.cut to find which segment each timestamp belongs to
    segment_indices = pd.cut(
        DTindex,
        bins=all_boundaries,
        right=False,
        labels=False,
        include_lowest=True,
    )
    # fill any NaNs from dates outside the boundary range
    segment_indices = (
        pd.Series(segment_indices, index=DTindex)
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )
    max_segment_index = len(all_boundaries) - 2
    if max_segment_index < 0:
        return (
            np.zeros(len(DTindex), dtype=float),
            np.zeros(len(DTindex), dtype=int),
            np.zeros(len(DTindex), dtype=float),
            0,
        )
    segment_indices = np.clip(segment_indices, 0, max_segment_index).astype(int)

    start_int = np.fromiter(
        (all_boundaries[i].value for i in segment_indices),
        dtype=np.int64,
        count=len(segment_indices),
    )
    end_int = np.fromiter(
        (all_boundaries[i + 1].value for i in segment_indices),
        dtype=np.int64,
        count=len(segment_indices),
    )

    dt_int = DTindex.view('i8')
    durations = end_int - start_int
    time_from_start = dt_int - start_int

    segment_frac = np.divide(
        time_from_start,
        durations,
        out=np.zeros_like(time_from_start, dtype=float),
        where=durations != 0,
    )
    segment_frac = np.clip(segment_frac, 0.0, 1.0)
    
    segment_idx = segment_indices % (base_length - 1)

    target_positions = np.linspace(0, 1, base_length)
    warped = target_positions[segment_idx] + segment_frac * (
        target_positions[segment_idx + 1] - target_positions[segment_idx]
    )

    return warped, segment_idx, segment_frac, base_length - 1


def _fourier_column_names(prefix: str, order: int):
    cos_cols = [f"{prefix}_cos{idx}" for idx in range(1, order + 1)]
    sin_cols = [f"{prefix}_sin{idx}" for idx in range(1, order + 1)]
    return cos_cols + sin_cols


def anchored_warped_fourier_features(DTindex, method: str):
    scheme_key, order = _parse_anchor_method(method, default_order=6)
    warped, _, _, _ = _compute_anchor_positions(DTindex, scheme_key)
    if warped.size == 0:
        return pd.DataFrame()
    data = fourier_series(warped, p=1.0, n=order)
    columns = _fourier_column_names(
        f"anchored_warped_{scheme_key}_fourier", order
    )
    return pd.DataFrame(data, columns=columns)


def anchored_segment_fourier_features(DTindex, method: str):
    scheme_key, order = _parse_anchor_method(method, default_order=4)
    warped, segment_idx, segment_frac, n_segments = _compute_anchor_positions(
        DTindex, scheme_key
    )
    if warped.size == 0 or n_segments <= 0:
        return pd.DataFrame()
    rows = len(DTindex)
    feature_arrays = []
    columns = []
    # segment-specific fourier terms
    fourier_data = np.zeros((rows, order * 2 * n_segments), dtype=float)
    for seg in range(n_segments):
        mask = segment_idx == seg
        col_slice = slice(seg * order * 2, (seg + 1) * order * 2)
        if mask.any():
            fourier_data[mask, col_slice] = fourier_series(
                segment_frac[mask], p=1.0, n=order
            )
        columns.extend(
            _fourier_column_names(
                f"anchored_segment_{scheme_key}_segment{seg}_fourier", order
            )
        )
    feature_arrays.append(fourier_data)
    # day-of-week gating
    weekday = DTindex.weekday
    dow_data = np.zeros((rows, n_segments * 7), dtype=float)
    for seg in range(n_segments):
        mask = segment_idx == seg
        base = seg * 7
        for day in range(7):
            col_idx = base + day
            if mask.any():
                dow_data[mask, col_idx] = (weekday[mask] == day).astype(float)
            columns.append(
                f"anchored_segment_{scheme_key}_segment{seg}_dow_{day}"
            )
    feature_arrays.append(dow_data)
    # hourly gating when grains are hourly or faster
    include_hour = False
    if len(DTindex) > 1:
        diffs = np.diff(DTindex.asi8) / 1e9
        if diffs.size > 0:
            delta = float(np.median(np.abs(diffs)))
            include_hour = delta <= 3600 + 1e-6
    if include_hour:
        hours = DTindex.hour
        hour_data = np.zeros((rows, n_segments * 24), dtype=float)
        for seg in range(n_segments):
            mask = segment_idx == seg
            base = seg * 24
            for hour in range(24):
                col_idx = base + hour
                if mask.any():
                    hour_data[mask, col_idx] = (hours[mask] == hour).astype(float)
                columns.append(
                    f"anchored_segment_{scheme_key}_segment{seg}_hour_{hour}"
                )
        feature_arrays.append(hour_data)
    if not feature_arrays:
        return pd.DataFrame()
    assembled = np.column_stack(feature_arrays)
    return pd.DataFrame(assembled, columns=columns)


def fourier_series(t, p=365.25, n=10):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n + 1) / p
    # 2 pi n / p * t
    x = x * np.asarray(t)[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x


def fourier_df(DTindex, seasonality, order=10, t=None, history_days=None):
    # if history_days is None:
    #     history_days = (DTindex.max() - DTindex.min()).days
    if t is None:
        # Calculate the time difference in days as a float to preserve the exact time
        t = (DTindex - pd.Timestamp(origin_ts)).total_seconds() / 86400
        # for only daily: t = (DTindex - pd.Timestamp(origin_ts)).days
        # for nano seconds: t = (DTindex - pd.Timestamp(origin_ts)).to_numpy(dtype=np.int64) // (1000 * 1000 * 1000) / (3600 * 24.)
    # formerly seasonality / history_days below
    return pd.DataFrame(fourier_series(np.asarray(t), seasonality, n=order)).rename(
        columns=lambda x: f"seasonality{seasonality}_" + str(x)
    )


datepart_components = [
    "dayofweek",
    "month",
    'day',
    "weekend",
    "weekdayofmonth",
    "hour",
    "daysinmonth",
    "quarter",
    "dayofyear",
    "weekdaymonthofyear",
    "dayofmonthofyear",
    "is_month_end",
    "is_month_start",
    "is_quarter_start",
    "is_quarter_end",
    "days_from_epoch",
    "isoweek",
    "isoweek_binary",
    "isoday",
    "quarterlydayofweek",
    "hourlydayofweek",
    "constant",
    "week",
    "year",
]


def create_datepart_components(DTindex, seasonality):
    """single date part one-hot flags."""
    if seasonality == "dayofweek":
        return pd.get_dummies(
            pd.Categorical(DTindex.weekday, categories=list(range(7)), ordered=True),
            dtype=np.uint8,
        ).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "month":
        return pd.get_dummies(
            pd.Categorical(DTindex.month, categories=list(range(1, 13)), ordered=True),
            dtype=np.uint8,
        ).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "day":
        return pd.get_dummies(
            pd.Categorical(DTindex.day, categories=list(range(1, 32)), ordered=True),
            dtype=np.uint8,
        ).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "weekend":
        return pd.DataFrame((DTindex.weekday > 4).astype(int), columns=["weekend"])
    elif seasonality == "weekdayofmonth":
        return pd.get_dummies(
            pd.Categorical(
                (DTindex.day - 1) // 7 + 1,
                categories=list(range(1, 6)),
                ordered=True,
            ),
            dtype=float,
        ).rename(columns=lambda x: f"{seasonality}_" + str(x))
    # recommend used in combination with some combination like dayofweek and quarter
    elif seasonality == "weekdaymonthofyear":
        monweek = (
            ((DTindex.day - 1) // 7 + 1).astype(str) + (DTindex.weekday).astype(str)
        ).astype(int)
        # because not much data for last week (week 5) of months unless many years of data
        monweek = monweek.where(monweek <= 50, 50)
        strs = DTindex.month.astype(str) + monweek.astype(str)
        cat_index = pd.date_range(
            "2020-01-01", "2021-01-01", freq='D'
        )  # must be a leap year
        catweek = (
            ((cat_index.day - 1) // 7 + 1).astype(str) + (cat_index.weekday).astype(str)
        ).astype(int)
        catweek = catweek.where(catweek <= 50, 50)
        cats = (cat_index.month.astype(str) + catweek.astype(str)).unique()
        return pd.get_dummies(
            pd.Categorical(
                strs,
                categories=cats,
                ordered=False,
            ),
            dtype=float,
        ).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "dayofmonthofyear":
        strs = DTindex.month.astype(str) + DTindex.day.astype(str)
        cat_index = pd.date_range(
            "2020-01-01", "2021-01-01", freq='D'
        )  # must be a leap yaer
        cats = (cat_index.month.astype(str) + cat_index.day.astype(str)).unique()
        return pd.get_dummies(
            pd.Categorical(
                strs,
                categories=cats,
                ordered=False,
            ),
            dtype=float,
        ).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "hour":
        return pd.get_dummies(
            pd.Categorical(DTindex.hour, categories=list(range(1, 25)), ordered=True),
            dtype=np.uint8,
        ).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "daysinmonth":
        return pd.DataFrame({'daysinmonth': DTindex.daysinmonth})
    elif seasonality == "quarter":
        return pd.get_dummies(
            pd.Categorical(DTindex.quarter, categories=list(range(1, 5)), ordered=True),
            dtype=np.uint8,
        ).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "dayofyear":
        return pd.get_dummies(
            pd.Categorical(
                DTindex.dayofyear, categories=list(range(1, 367)), ordered=True
            ),
            dtype=np.uint16,
        ).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "is_month_end":
        return pd.DataFrame({'is_month_end': DTindex.is_month_end})
    elif seasonality == "is_month_start":
        return pd.DataFrame({'is_month_start': DTindex.is_month_start})
    elif seasonality == "is_quarter_start":
        return pd.DataFrame({'is_quarter_start': DTindex.is_quarter_start})
    elif seasonality == "is_quarter_end":
        return pd.DataFrame({'is_quarter_end': DTindex.is_quarter_end})
    elif seasonality == "days_from_epoch":
        return (DTindex - pd.Timestamp('2000-01-01')).days.astype('int32')
    elif seasonality in ["isoweek", "week"]:
        return DTindex.isocalendar().week
    elif seasonality in ["year"]:
        return DTindex.year.rename("year")
    elif seasonality == "isoweek_binary":
        return pd.get_dummies(
            pd.Categorical(
                DTindex.isocalendar().week, categories=list(range(1, 54)), ordered=True
            ),
            dtype=np.uint16,
        ).rename(columns=lambda x: f"{seasonality}_" + str(x))
    elif seasonality == "isoday":
        return DTindex.isocalendar().day
    elif seasonality == "quarterlydayofweek":
        day_dummies = pd.get_dummies(
            pd.Categorical(DTindex.weekday, categories=list(range(7)), ordered=True),
            dtype=np.uint8,
            prefix='day',
        )
        quarter_dummies = pd.get_dummies(
            pd.Categorical(DTindex.quarter, categories=list(range(1, 5)), ordered=True),
            dtype=np.uint8,
            prefix='Q',
        )
        # Create interaction terms by multiplying day and quarter dummy variables
        interaction_features = (
            day_dummies.values[:, :, np.newaxis]
            * quarter_dummies.values[:, np.newaxis, :]
        )
        # Reshape the interaction_features to a 2D array
        interaction_features = interaction_features.reshape(DTindex.shape[0], -1)
        # Create column names for interaction features
        interaction_feature_names = [
            f"{day_col}_{quarter_col}"
            for day_col in day_dummies.columns
            for quarter_col in quarter_dummies.columns
        ]
        # Create a DataFrame for interaction features
        return pd.DataFrame(
            interaction_features, columns=interaction_feature_names
        ).astype(int)
    elif seasonality == "hourlydayofweek":
        day_dummies = pd.get_dummies(
            pd.Categorical(DTindex.weekday, categories=list(range(7)), ordered=True),
            dtype=np.uint8,
            prefix='day',
        )
        quarter_dummies = pd.get_dummies(
            pd.Categorical(DTindex.hour, categories=list(range(1, 25)), ordered=True),
            dtype=np.uint8,
            prefix='h',
        )
        # Create interaction terms by multiplying day and quarter dummy variables
        interaction_features = (
            day_dummies.values[:, :, np.newaxis]
            * quarter_dummies.values[:, np.newaxis, :]
        )
        # Reshape the interaction_features to a 2D array
        interaction_features = interaction_features.reshape(DTindex.shape[0], -1)
        # Create column names for interaction features
        interaction_feature_names = [
            f"{day_col}_{quarter_col}"
            for day_col in day_dummies.columns
            for quarter_col in quarter_dummies.columns
        ]
        # Create a DataFrame for interaction features
        return pd.DataFrame(
            interaction_features, columns=interaction_feature_names
        ).astype(int)
    elif seasonality == "constant":
        return pd.DataFrame(1, columns=["constant"], index=range(len(DTindex)))
    else:
        raise ValueError(
            f"create_datepart_components `{seasonality}` is not recognized"
        )


def create_seasonality_feature(DTindex, t, seasonality, history_days=None):
    """Cassandra-designed feature generator."""
    # for consistency, all must have a range index, not date index
    # fourier orders
    if isinstance(seasonality, (int, float)):
        return fourier_df(
            DTindex, seasonality=seasonality, t=t, history_days=history_days
        )
    if isinstance(seasonality, tuple):
        return fourier_df(
            DTindex,
            seasonality=seasonality[0],
            order=seasonality[1],
            t=t,
            history_days=history_days,
        )
    # dateparts
    elif seasonality in datepart_components:
        return create_datepart_components(DTindex, seasonality)
    elif seasonality in date_part_methods:
        return date_part(DTindex, method=seasonality, set_index=False)
    else:
        return ValueError(
            f"Seasonality `{seasonality}` not recognized. Must be int, float, or a select type string such as 'dayofweek'"
        )


def _common_fourier_rw_lag_features(DTindex, lag_count=COMMON_FOURIER_RW_LAG_ORDER):
    """Construct common_fourier_rw features with deterministic lags."""
    lag_count = 0 if lag_count is None else int(max(0, lag_count))
    base_length = len(DTindex)
    if base_length == 0:
        return pd.DataFrame(index=pd.RangeIndex(0))
    if lag_count == 0:
        return date_part(DTindex, method="common_fourier_rw", set_index=False)
    # For small date ranges, fall back to shift-based lagging
    if base_length < 3:
        base_features = date_part(DTindex, method="common_fourier_rw", set_index=False)
        lagged_frames = []
        for lag in range(1, lag_count + 1):
            lag_slice = base_features.shift(lag)
            lag_slice = lag_slice.bfill().fillna(0.0)
            lag_slice.columns = [f"{col}_lag{lag}" for col in lag_slice.columns]
            lagged_frames.append(lag_slice)
        return pd.concat([base_features] + lagged_frames, axis=1)
    frequency = infer_frequency(DTindex)
    tz = getattr(DTindex, 'tz', None)
    if frequency is not None:
        longer_idx = pd.date_range(
            end=DTindex[-1],
            periods=base_length + lag_count,
            freq=frequency,
            tz=tz,
        )
        extended = date_part(longer_idx, method="common_fourier_rw", set_index=False)
        base_features = extended.iloc[lag_count:].reset_index(drop=True)
        lagged_frames = []
        for lag in range(1, lag_count + 1):
            start = lag_count - lag
            stop = start + base_length
            lag_slice = extended.iloc[start:stop].reset_index(drop=True)
            lag_slice.columns = [f"{col}_lag{lag}" for col in lag_slice.columns]
            lagged_frames.append(lag_slice)
        return pd.concat([base_features] + lagged_frames, axis=1)
    base_features = date_part(DTindex, method="common_fourier_rw", set_index=False)
    lagged_frames = []
    for lag in range(1, lag_count + 1):
        lag_slice = base_features.shift(lag)
        lag_slice = lag_slice.bfill().fillna(0.0)
        lag_slice.columns = [f"{col}_lag{lag}" for col in lag_slice.columns]
        lagged_frames.append(lag_slice)
    return pd.concat([base_features] + lagged_frames, axis=1)


base_seasonalities = [  # this needs to be a list
    "recurring",
    "simple",
    "expanded",
    "simple_2",
    "simple_binarized",
    "expanded_binarized",
    'common_fourier',
    'common_fourier_rw',
    'common_fourier_rw_lag',
    'anchored_warped_fourier:us_school',
    'anchored_segment_fourier:us_school',
    "simple_poly",
    # it is critical for this to work with the fourier order option that the FLOAT COME second if the list is length 2
    [7, 365.25],
    ["dayofweek", 365.25],
    ['weekdayofmonth', 'common_fourier'],
    [52, 'quarter'],
    [168, "hour"],
    ["morlet_365.25_12_12", "ricker_7_7_1"],
    [
        "db2_365.25_12_0.5",
        "morlet_7_7_1",
    ],  # this actually is working surprisingly well, could probably be expanded upon
    ["weekdaymonthofyear", "quarter", "dayofweek"],
    "lunar_phase",
    ["dayofweek", (365.25, 4)],
    ["dayofweek", (365.25, 14)],
    ["dayofweek", (365.25, 24)],
    [
        "dayofweek",
        (365.25, 14),
        (354.37, 10),
    ],  # 354.37 should be islamic calendar avg length
    "other",
]


def random_datepart(method='random'):
    """New random parameters for seasonality."""
    seasonalities = random.choices(
        base_seasonalities,
        [
            0.4,
            0.3,
            0.3,
            0.3,
            0.4,
            0.35,
            0.45,
            0.1,
            0.2,
            0.25,
            0.25,
            0.1,
            0.1,
            0.05,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
            0.05,
            0.02,
            0.02,
            0.3,
        ],
    )[0]
    if seasonalities == "other":
        predefined = random.choices([True, False], [0.5, 0.5])[0]
        if predefined:
            seasonalities = [random.choice(date_part_methods)]
        else:
            comp_opts = datepart_components + [7, 365.25, 12, 52, 168, 24]
            seasonalities = random.choices(comp_opts, k=3)
    return seasonalities


def seasonal_window_match(
    DTindex,
    k,
    window_size=10,
    forecast_length=None,
    datepart_method="common_fourier_rw",
    distance_metric="mae",
    full_sort=False,
):
    array = date_part(DTindex, method=datepart_method).to_numpy()

    # when k is larger, can be more aggressive on allowing a longer portion into view
    min_k = 5
    if k > min_k:
        n_tail = min(window_size, forecast_length)
    else:
        n_tail = forecast_length
    # finding sliding windows to compare
    temp = sliding_window_view(array[:-n_tail, :], window_size, axis=0)
    # compare windows by metrics
    last_window = array[-window_size:, :]
    if distance_metric == "mae":
        scores = np.mean(np.abs(temp - last_window.T), axis=2)
    elif distance_metric == "canberra":
        divisor = np.abs(temp) + np.abs(last_window.T)
        divisor[divisor == 0] = 1
        scores = np.mean(np.abs(temp - last_window.T) / divisor, axis=2)
    elif distance_metric == "minkowski":
        p = 2
        scores = np.sum(np.abs(temp - last_window.T) ** p, axis=2) ** (1 / p)
    elif distance_metric == "euclidean":
        scores = np.sqrt(np.sum((temp - last_window.T) ** 2, axis=2))
    elif distance_metric == "chebyshev":
        scores = np.max(np.abs(temp - last_window.T), axis=2)
    elif distance_metric == "mqae":
        q = 0.85
        ae = np.abs(temp - last_window.T)
        if ae.shape[2] <= 1:
            vals = ae
        else:
            qi = int(ae.shape[2] * q)
            qi = qi if qi > 1 else 1
            vals = np.partition(ae, qi, axis=2)[..., :qi]
        scores = np.mean(vals, axis=2)
    elif distance_metric == "mse":
        scores = np.mean((temp - last_window.T) ** 2, axis=2)
    else:
        raise ValueError(f"distance_metric: {distance_metric} not recognized")

    # select smallest windows
    if full_sort:
        min_idx = np.argsort(scores.mean(axis=1), axis=0)[:k]
    else:
        min_idx = np.argpartition(scores.mean(axis=1), k - 1, axis=0)[:k]
    # take the period starting AFTER the window
    test = (
        np.broadcast_to(
            np.arange(0, forecast_length)[..., None],
            (forecast_length, min_idx.shape[0]),
        )
        + min_idx
        + window_size
    )
    # for data over the end, fill last value
    if k > min_k:
        test = np.where(test >= len(DTindex), -1, test)
    return test, scores


def seasonal_independent_match(
    DTindex,
    DTindex_future,
    k,
    datepart_method='simple_binarized',
    distance_metric='canberra',
    full_sort=False,
    nan_array=None,
):
    array = date_part(DTindex, method=datepart_method)
    if nan_array is not None:
        array[nan_array] = np.inf
    future_array = date_part(DTindex_future, method=datepart_method).to_numpy()

    # when k is larger, can be more aggressive on allowing a longer portion into view
    min_k = 5
    # compare windows by metrics
    a = array.to_numpy()[:, None]
    b = future_array
    if distance_metric == "mae":
        scores = np.abs(a - b).mean(axis=2)
    elif distance_metric == "canberra":
        divisor = np.abs(a) + np.abs(b)
        divisor[divisor == 0] = 1
        scores = np.mean(np.abs(a - b) / divisor, axis=2)
    elif distance_metric == "minkowski":
        p = 2
        scores = np.sum(np.abs(a - b) ** p, axis=2) ** (1 / p)
    elif distance_metric == "euclidean":
        scores = np.sqrt(np.sum((a - b) ** 2, axis=2))
    elif distance_metric == "chebyshev":
        scores = np.max(np.abs(a - b), axis=2)
    elif distance_metric == "mqae":
        q = 0.85
        ae = np.abs(a - b)
        if ae.shape[2] <= 1:
            vals = ae
        else:
            qi = int(ae.shape[2] * q)
            qi = qi if qi > 1 else 1
            vals = np.partition(ae, qi, axis=2)[..., :qi]
        scores = np.mean(vals, axis=2)
    elif distance_metric == "mse":
        scores = np.mean((a - b) ** 2, axis=2)
    else:
        raise ValueError(f"distance_metric: {distance_metric} not recognized")

    # select smallest windows
    if full_sort:
        min_idx = np.argsort(scores, axis=0)[:k]
    else:
        min_idx = np.argpartition(scores, k - 1, axis=0)[:k]
    # take the period starting AFTER the window
    test = min_idx.T
    # for data over the end, fill last value
    if k > min_k:
        test = np.where(test >= len(DTindex), -1, test)
    return test, scores


def seasonal_repeating_wavelet(DTindex, p, order=12, sigma=4.0, wavelet_type='morlet'):
    t = (DTindex - pd.Timestamp(origin_ts)).total_seconds() / 86400

    if wavelet_type == "db2":
        wavelets = create_narrowing_wavelets(
            p=float(p), max_order=int(order), t=t, sigma=float(sigma)
        )
    else:
        wavelets = offset_wavelet(
            p=float(p),  # Weekly period
            t=t,  # A full year (365 days)
            # origin_ts=origin_ts,
            order=int(order),  # One offset for each day of the week
            # frequency=2 * np.pi / p,  # Frequency for weekly pattern
            sigma=float(sigma),  # Smaller sigma for tighter weekly spread
            wavelet_type=wavelet_type,
        )
    return pd.DataFrame(wavelets, index=DTindex).rename(
        columns=lambda x: f"wavelet_{p}_" + str(x)
    )
