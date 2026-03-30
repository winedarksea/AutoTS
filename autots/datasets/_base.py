"""Loading example datasets."""

from os.path import dirname, join
import time
import datetime
import io
import json
import numpy as np
import pandas as pd


def load_daily(long: bool = True):
    """Daily sample data.

    ```
    # most of the wiki data was chosen to show holidays or holiday-like patterns
    wiki = [
        'United_States',
        'Germany',
        'List_of_highest-grossing_films',
        'Jesus',
        'Michael_Jackson',
        'List_of_United_States_cities_by_population',
        'Microsoft_Office',
        'Google_Chrome',
        'Periodic_table',
        'Standard_deviation',
        'Easter',
        'Christmas',
        'Chinese_New_Year',
        'Thanksgiving',
        'List_of_countries_that_have_gained_independence_from_the_United_Kingdom',
        'History_of_the_hamburger',
        'Elizabeth_II',
        'William_Shakespeare',
        'George_Washington',
        'Cleopatra',
        'all'
    ]

    df2 = load_live_daily(
        observation_start="2017-01-01", weather_years=7, trends_list=None,
        gov_domain_list=None, wikipedia_pages=wiki,
        fred_series=['DGS10', 'T5YIE', 'SP500','DEXUSEU'], sleep_seconds=10,
        fred_key = "93873d40f10c20fe6f6e75b1ad0aed4d",
        weather_data_types = ["WSF2", "PRCP"],
        weather_stations = ["USW00014771"],  # looking for intermittent
        tickers=None, london_air_stations=None,
        weather_event_types=None, earthquake_min_magnitude=None,
    )
    data_file_name = join("autots", "datasets", 'data', 'holidays.zip')
    df2.to_csv(
        data_file_name,
        index=True,
        compression={
            'method': 'zip',
            'archive_name': 'holidays.csv',
            'compresslevel': 9  # Maximum compression level (0-9)
        }
    )
    ```

    Sources: Wikimedia Foundation

    Args:
        long (bool): if True, return data in long format. Otherwise return wide
    """
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', 'holidays.zip')

    df_wide = pd.read_csv(data_file_name, index_col=0, parse_dates=True)
    if not long:
        return df_wide
    else:
        df_wide.index.name = 'datetime'
        df_long = df_wide.reset_index(drop=False).melt(
            id_vars=['datetime'], var_name='series_id', value_name='value'
        )
        return df_long


def load_fred_monthly():
    """
    Federal Reserve of St. Louis.
    from autots.datasets.fred import get_fred_data
    SeriesNameDict = {'GS10':'10-Year Treasury Constant Maturity Rate',
                              'MCOILWTICO':'Crude Oil West Texas Intermediate Cushing Oklahoma',
                              'CSUSHPISA': ' U.S. National Home Price Index',
                              'EXUSEU': 'US Euro Foreign Exchange Rate',
                              'EXCHUS': 'China US Foreign Exchange Rate',
                              'EXCAUS' : 'Canadian to US Dollar Exchange Rate Daily',
                              'EMVOVERALLEMV': 'Equity Market Volatility Tracker Overall',  # this is a more irregular series
                              'T10YIEM' : '10 Year Breakeven Inflation Rate',
                              'USEPUINDXM': 'Economic Policy Uncertainty Index for United States' # also very irregular
                              }
    monthly_data = get_fred_data(fredkey = 'XXXXXXXXX', SeriesNameDict = SeriesNameDict)
    """
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', 'fred_monthly.zip')

    df_long = pd.read_csv(data_file_name, compression='zip')
    df_long['datetime'] = pd.to_datetime(df_long['datetime'])

    return df_long


def load_monthly(long: bool = True):
    """Federal Reserve of St. Louis monthly economic indicators."""
    if long:
        return load_fred_monthly()
    else:
        from autots.tools.shaping import long_to_wide

        df_long = load_fred_monthly()
        df_wide = long_to_wide(
            df_long,
            date_col='datetime',
            value_col='value',
            id_col='series_id',
            aggfunc='first',
        )
        return df_wide


def load_fred_yearly():
    """
    Federal Reserve of St. Louis.
    from autots.datasets.fred import get_fred_data
    SSeriesNameDict = {'GDPA':"Gross Domestic Product",
                  'ACOILWTICO':'Crude Oil West Texas Intermediate Cushing Oklahoma',
                  'AEXUSEU': 'US Euro Foreign Exchange Rate',
                  'AEXCHUS': 'China US Foreign Exchange Rate',
                  'AEXCAUS' : 'Canadian to US Dollar Exchange Rate Daily',
                  'MEHOINUSA672N': 'Real Median US Household Income',
                  'CPALTT01USA661S': 'Consumer Price Index All Items',
                  'FYFSD': 'Federal Surplus or Deficit',
                  'DDDM01USA156NWDB': 'Stock Market Capitalization to US GDP',
                  'LEU0252881600A': 'Median Weekly Earnings for Salary Workers',
                  'LFWA64TTUSA647N': 'US Working Age Population',
                  'IRLTLT01USA156N' : 'Long Term Government Bond Yields'
                  }
    monthly_data = get_fred_data(fredkey = 'XXXXXXXXX', SeriesNameDict = SeriesNameDict)
    """
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', 'fred_yearly.zip')

    df_long = pd.read_csv(data_file_name)
    df_long['datetime'] = pd.to_datetime(df_long['datetime'])

    return df_long


def load_yearly(long: bool = True):
    """Federal Reserve of St. Louis annual economic indicators."""
    if long:
        return load_fred_yearly()
    else:
        from autots.tools.shaping import long_to_wide

        df_long = load_fred_yearly()
        df_wide = long_to_wide(
            df_long,
            date_col='datetime',
            value_col='value',
            id_col='series_id',
            aggfunc='first',
        )
        return df_wide


def load_traffic_hourly(long: bool = True):
    """
    From the MN DOT via the UCI data repository.
    Yes, Minnesota is the best state of the Union.
    """
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', 'traffic_hourly.zip')

    df_wide = pd.read_csv(
        data_file_name, index_col=0, parse_dates=True, compression='zip'
    )
    if not long:
        return df_wide
    else:
        df_long = df_wide.reset_index(drop=False).melt(
            id_vars=['datetime'], var_name='series_id', value_name='value'
        )
        return df_long


def load_hourly(long: bool = True):
    """Traffic data from the MN DOT via the UCI data repository."""
    return load_traffic_hourly(long=long)


def load_eia_weekly():
    """Weekly petroleum industry data from the EIA."""
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', 'eia_weekly.zip')

    df_long = pd.read_csv(data_file_name, compression='zip')
    df_long['datetime'] = pd.to_datetime(df_long['datetime'])
    return df_long


def load_weekly(long: bool = True):
    """Weekly petroleum industry data from the EIA."""
    if long:
        return load_eia_weekly()
    else:
        from autots.tools.shaping import long_to_wide

        df_long = load_eia_weekly()
        df_wide = long_to_wide(
            df_long,
            date_col='datetime',
            value_col='value',
            id_col='series_id',
            aggfunc='first',
        )
        return df_wide


def load_weekdays(long: bool = False, categorical: bool = True, periods: int = 180):
    """Test edge cases by creating a Series with values as day of week.

    Args:
        long (bool):
            if True, return a df with columns "value" and "datetime"
            if False, return a Series with dt index
        categorical (bool): if True, return str/object, else return int
        periods (int): number of periods, ie length of data to generate
    """
    idx = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq="D")
    df_wide = pd.Series(idx.weekday, index=idx, name="value")
    df_wide.index.name = "datetime"
    if categorical:
        df_wide = df_wide.replace(
            {
                0: "Mon",
                1: "Tues",
                2: "Wed",
                3: "Thor's",
                4: "Fri",
                5: "Sat",
                6: "Sun",
                7: "Mon",
            }
        )
    if long:
        return df_wide.reset_index()
    else:
        return df_wide


# load_live_daily moved to _live.py for single-responsibility; re-exported here for backward compatibility
from autots.datasets._live import load_live_daily  # noqa: F401


def load_zeroes(long=False, shape=None, start_date: str = "2021-01-01"):
    """Create a dataset of just zeroes for testing edge case."""
    if shape is None:
        shape = (200, 5)
    df_wide = pd.DataFrame(
        np.zeros(shape), index=pd.date_range(start_date, periods=shape[0], freq="D")
    )
    if not long:
        return df_wide
    else:
        df_wide.index.name = "datetime"
        df_long = df_wide.reset_index(drop=False).melt(
            id_vars=['datetime'], var_name='series_id', value_name='value'
        )
        return df_long


def load_linear(
    long=False,
    shape=None,
    start_date: str = "2021-01-01",
    introduce_nan: float = None,
    introduce_random: float = None,
    random_seed: int = 123,
):
    """Create a dataset of just zeroes for testing edge case.

    Args:
        long (bool): whether to make long or wide
        shape (tuple): shape of output dataframe
        start_date (str): first date of index
        introduce_nan (float): percent of rows to make null. 0.2 = 20%
        introduce_random (float): shape of gamma distribution
        random_seed (int): seed for random
    """
    if shape is None:
        shape = (500, 5)
    idx = pd.date_range(start_date, periods=shape[0], freq="D")
    df_wide = pd.DataFrame(np.ones(shape), index=idx)
    df_wide = (df_wide * list(range(0, shape[1]))).cumsum()
    if introduce_nan is not None:
        df_wide = df_wide.sample(
            frac=(1 - introduce_nan), random_state=random_seed
        ).reindex(idx)
    if introduce_random is not None:
        df_wide = df_wide + np.random.default_rng(random_seed).gamma(
            introduce_random, size=shape
        )
    if not long:
        return df_wide
    else:
        df_wide.index.name = "datetime"
        df_long = df_wide.reset_index(drop=False).melt(
            id_vars=['datetime'], var_name='series_id', value_name='value'
        )
        return df_long


def load_sine(
    long=False,
    shape=None,
    start_date: str = "2021-01-01",
    introduce_random: float = None,
    random_seed: int = 123,
):
    """Create a dataset of just zeroes for testing edge case."""
    if shape is None:
        shape = (500, 5)
    df_wide = pd.DataFrame(
        np.ones(shape),
        index=pd.date_range(start_date, periods=shape[0], freq="D"),
        columns=range(shape[1]),
    )
    X = pd.to_numeric(df_wide.index, errors='coerce', downcast='integer').values

    def sin_func(a, X):
        return a * np.sin(a * X) + a

    for column in df_wide.columns:
        df_wide[column] = sin_func(column, X)
    if introduce_random is not None:
        df_wide = (
            df_wide
            + np.random.default_rng(random_seed).gamma(introduce_random, size=shape)
        ).clip(lower=0.1)
    if not long:
        return df_wide
    else:
        df_wide.index.name = "datetime"
        df_long = df_wide.reset_index(drop=False).melt(
            id_vars=['datetime'], var_name='series_id', value_name='value'
        )
        return df_long


def load_artificial(long=False, date_start=None, date_end=None):
    """Load artifically generated series from random distributions.

    Args:
        long (bool): if True long style data, if False, wide style data
        date_start: str or datetime.datetime of start date
        date_end: str or datetime.datetime of end date
    """
    import scipy.signal
    from scipy.ndimage import maximum_filter1d
    from autots.tools.wavelet import create_mexican_hat_wavelet, create_morlet_wavelet

    if date_end is None:
        date_end = datetime.datetime.now().date()
    if isinstance(date_end, datetime.datetime):
        date_end = date_end.date()
    if date_start is None:
        if isinstance(date_end, datetime.date):
            date_start = date_end - datetime.timedelta(days=740)
        else:
            date_start = datetime.datetime.now().date() - datetime.timedelta(days=740)
    if isinstance(date_start, datetime.datetime):
        date_start = date_start.date()
    dates = pd.date_range(date_start, date_end)
    size = dates.size
    new_size = int(size / 10)
    rng = np.random.default_rng()
    holiday = pd.Series(
        np.arange(size) * 0.025
        + rng.normal(0, 0.2, size)
        + (np.sin((np.pi / 7) * np.arange(size)) * 0.5),
        index=dates,
        name='holiday',
    )
    # January 1st
    holiday[holiday.index.month == 1 & (holiday.index.day == 1)] += 10
    # December 25th
    holiday[(holiday.index.month == 12) & (holiday.index.day == 25)] += -4
    # Second Tuesday of April
    # Find all Tuesdays in April
    second_tuesday_of_april = (
        (holiday.index.month == 4)
        & (holiday.index.weekday == 1)
        & (holiday.index.day >= 8)
        & (holiday.index.day <= 14)
    )
    holiday[second_tuesday_of_april] += 10
    # Last Monday of August
    last_monday_of_august = (
        (holiday.index.month == 8)
        & (holiday.index.weekday == 0)
        & ((holiday.index + pd.Timedelta(7, unit='D')).month == 9)
    )
    holiday[last_monday_of_august] += 12

    df_wide = pd.DataFrame(
        {
            'white_noise': rng.normal(0, 1, size),
            "white_noise_trend": rng.normal(0, 1, size) + np.arange(size) * 0.01,
            "random_walk": np.random.choice(a=[-0.8, 0, 0.8], size=size).cumsum() * 0.8,
            "arima007_trend": np.convolve(
                np.random.choice(a=[-0.4, 0, 0.4], size=size + 6),
                np.ones(7, dtype=int),
                'valid',
            )
            + np.arange(size) * 0.01,
            "arima017": np.convolve(
                np.random.choice(a=[-0.4, 0, 0.4], size=size + 6),
                np.ones(7, dtype=int),
                'valid',
            ).cumsum()
            / 12,
            "arima200_gamma": scipy.signal.lfilter(
                [1], [1.0, -0.75, 0.25], 1 * rng.gamma(1, size=size), axis=0
            ),  # ma order is first, then ar order
            "arima220_outliers": np.where(
                rng.poisson(20, size) >= 30,
                rng.gamma(5, size=size) + 10,
                scipy.signal.lfilter(
                    [1.0, 0.65, 0.25],
                    [1.0, -0.85, 0.25],
                    1 * rng.normal(2, size=size),
                    axis=0,
                )
                / 2,
            ),
            "linear": np.arange(size) * 0.025,
            "flat": 1,
            "new_product": np.concatenate(
                [
                    np.zeros(int(size - new_size)),
                    np.random.choice(a=[-0.8, 0, 0.8], size=new_size).cumsum(),
                ]
            ),
            "sine_wave": np.sin(np.arange(size)),
            "sine_seasonality_monthweek": (
                (np.sin((np.pi / 7) * np.arange(size)) * 0.25 + 0.25)
                + (np.sin((np.pi / 28) * np.arange(size)) * 1 + 1)
                + rng.normal(0, 0.15, size)
            ),
            "wavelet_ricker": np.tile(
                create_mexican_hat_wavelet(33, sigma=1.0), int(np.ceil(size / 33))
            )[:size],
            "wavelet_morlet": np.real(
                np.tile(create_morlet_wavelet(100, 6.0, 6.0), int(np.ceil(size / 100)))[
                    :size
                ]
                * 10
            ),
            "lumpy": np.stack(
                [
                    rng.gamma(1, size=size),
                    rng.gamma(1, size=size),
                    rng.gamma(1, size=size),
                    rng.gamma(4, size=size),
                    rng.gamma(6, size=size),
                    rng.gamma(7, size=size),
                    rng.gamma(5, size=size),
                ],
                axis=0,
            ).T.ravel()[:size]
            + (np.sin((np.pi / 182) * np.arange(size)) * 0.75 + 1),
            "intermittent_random": rng.poisson(0.3, size=size),
            "intermittent_weekly": np.stack(
                [
                    np.random.choice(a=[0, 1], p=[0.98, 0.02], size=size),
                    np.random.choice(a=[0, 1], p=[0.96, 0.04], size=size),
                    np.random.choice(a=[0, 1], p=[0.94, 0.06], size=size),
                    np.random.choice(a=[0, 1], p=[0.94, 0.06], size=size),
                    np.random.choice(a=[0, 2, 1], p=[0.8, 0.1, 0.1], size=size),
                    np.random.choice(
                        a=[0, 3, 2, 1], p=[0.5, 0.05, 0.1, 0.35], size=size
                    ),
                    np.random.choice(
                        a=[0, 3, 2, 1], p=[0.25, 0.2, 0.3, 0.25], size=size
                    ),
                ],
                axis=0,
            ).T.ravel()[:size],
            "out_of_stock": np.where(
                -maximum_filter1d(-rng.negative_binomial(1, 0.04, size=size), 8) == 0,
                0,
                # moving average of a sine + gamma random
                np.convolve(
                    (
                        (np.sin((np.pi / 182) * np.arange(size + 2)) * 2 + 2)
                        + rng.gamma(1, 0.5, size=size + 2)
                    ),
                    np.ones(3, dtype=int),
                    'valid',
                )
                / 2,
            ),
            "cubic_root": np.cbrt(np.arange(-int(size / 2), size - int(size / 2))),
            "logistic_growth": np.log(np.arange(2, size + 2)),
            "recent_spike": np.where(
                np.arange(size) < (9 * size) / 10,
                np.arange(size) * 0.01,
                abs(np.arange(size) - (9 * size) / 10) ** 1.01 + (9 * size) / 10 * 0.01,
            )
            + rng.normal(0, 0.05, size),
            "recent_plateau": np.where(
                np.arange(size) < (8.5 * size) / 10,
                np.arange(size) * 0.01,
                (8.5 * size * 0.01) / 10,
            )
            + rng.normal(0, 0.05, size),
            "old_to_new": np.where(
                np.arange(size) < (4 * size) / 5,
                np.real(
                    np.tile(
                        create_morlet_wavelet(50, 6.0, 6.0), int(np.ceil(size / 50))
                    )[:size]
                    * 10
                ),
                np.real(
                    np.tile(
                        create_morlet_wavelet(50, 6.0, 0.0), int(np.ceil(size / 50))
                    )[:size]
                    * 10
                ),
            )
            + (np.sin((np.pi / 182) * np.arange(size)) * 1 + 1),
        },
        index=dates,
    )
    df_wide = df_wide.merge(holiday, left_index=True, right_index=True)

    if not long:
        return df_wide
    else:
        df_wide.index.name = "datetime"
        df_long = df_wide.reset_index(drop=False).melt(
            id_vars=['datetime'], var_name='series_id', value_name='value'
        )
        return df_long
