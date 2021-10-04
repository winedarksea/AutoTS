"""Loading example datasets."""
from os.path import dirname, join
import datetime
import io
import requests
import numpy as np
import pandas as pd


def load_daily(long: bool = True):
    """2020 Covid, Air Pollution, and Economic Data.

    Sources: Covid Tracking Project, EPA, and FRED

    Args:
        long (bool): if True, return data in long format. Otherwise return wide
    """
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', 'covid_daily.zip')

    df_wide = pd.read_csv(data_file_name, index_col=0, parse_dates=True)
    if not long:
        return df_wide
    else:
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
    df_long['datetime'] = pd.to_datetime(
        df_long['datetime'], infer_datetime_format=True
    )

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
    df_long['datetime'] = pd.to_datetime(
        df_long['datetime'], infer_datetime_format=True
    )

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
    df_long['datetime'] = pd.to_datetime(
        df_long['datetime'], infer_datetime_format=True
    )
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


def load_live_daily(
    long: bool = False,
    fred_key: str = None,
    fred_series: list = ["DGS10", "T5YIE", "SP500", "DCOILWTICO", "DEXUSEU"],
    tickers: list = ["MSFT"],
    trends_list: list = ["forecasting", "cycling", "cpu", "microsoft"],
    weather_data_types: list = ["AWND", "WSF2", "TAVG"],
    weather_stations: list = ["USW00094846", "USW00014925"],
    weather_years: int = 10,
    london_air_stations: list = ['CT3', 'SK8'],
    london_air_species: str = "PM25",
    london_air_days: int = 180,
    earthquake_days: int = 180,
    earthquake_min_magnitude: int = 5,
):
    """Generates a dataframe of data up to the present day.

    Args:
        long (bool): whether to return in long format or wide
        fred_key (str): https://fred.stlouisfed.org/docs/api/api_key.html
        fred_series (list): list of FRED series IDs. This requires fredapi package
        tickers (list): list of stock tickers, requires yfinance
        trends (list): list of search keywords, requires pytrends.
        weather_data_types (list): from NCEI NOAA api data types, GHCN Daily Weather Elements
            PRCP, SNOW, TMAX, TMIN, TAVG, AWND, WSF1, WSF2, WSF5, WSFG
        weather_stations (list): from NCEI NOAA api station ids
        london_air_stations (list): londonair.org.uk source station IDs
        london_species (str): what measurement to pull from London Air. Not all stations have all metrics.\
        earthquake_min_magnitude (int): smallest earthquake magnitude to pull from earthquake.usgs.gov
    """
    dataset_lists = []
    current_date = datetime.datetime.utcnow()

    try:
        if fred_key is not None:
            from fredapi import Fred  # noqa
            from autots.datasets.fred import get_fred_data

            fred_df = get_fred_data(fred_key, fred_series, long=False)
            fred_df.index = fred_df.index.tz_localize(None)
            dataset_lists.append(fred_df)
    except ModuleNotFoundError:
        print("pip install fredapi (and you'll also need an api key)")
    except Exception as e:
        print(f"FRED data failed: {repr(e)}")

    for ticker in tickers:
        try:
            import yfinance as yf

            msft = yf.Ticker(ticker)
            # get historical market data
            msft_hist = msft.history(period="max")
            msft_hist = msft_hist.rename(columns=lambda x: x.lower().replace(" ", "_"))
            msft_hist = msft_hist.rename(columns=lambda x: ticker.lower() + "_" + x)
            try:
                msft_hist.index = msft_hist.index.tz_localize(None)
            except Exception:
                pass
            dataset_lists.append(msft_hist)
        except ModuleNotFoundError:
            print("You need to: pip install yfinance")
        except Exception as e:
            print(f"yfinance data failed: {repr(e)}")

    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360)
        # pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
        pytrends.build_payload(trends_list, timeframe="all")
        gtrends = pytrends.interest_over_time()
        gtrends.index = gtrends.index.tz_localize(None)
        gtrends.drop(columns="isPartial", inplace=True, errors="ignore")
        dataset_lists.append(gtrends)
    except ImportError:
        print("You need to: pip install pytrends")
    except Exception as e:
        print(f"pytrends data failed: {repr(e)}")

    str_end_time = current_date.strftime("%Y-%m-%d")
    start_date = (current_date - datetime.timedelta(days=360 * weather_years)).strftime(
        "%Y-%m-%d"
    )
    for wstation in weather_stations:
        try:
            wbase = "https://www.ncei.noaa.gov/access/services/data/v1/?dataset=daily-summaries"
            wargs = f"&dataTypes={','.join(weather_data_types)}&stations={wstation}"
            wargs = (
                wargs
                + f"&startDate={start_date}&endDate={str_end_time}&boundingBox=90,-180,-90,180&units=standard&format=csv"
            )
            wdf = pd.read_csv(wbase + wargs)
            wdf['DATE'] = pd.to_datetime(wdf['DATE'], infer_datetime_format=True)
            wdf = wdf.set_index('DATE').drop(columns=['STATION'])
            wdf.rename(columns=lambda x: wstation + "_" + x, inplace=True)
            dataset_lists.append(wdf)
        except Exception as e:
            print(f"weather data failed: {repr(e)}")

    str_end_time = current_date.strftime("%d-%b-%Y")
    start_date = (current_date - datetime.timedelta(days=london_air_days)).strftime(
        "%d-%b-%Y"
    )
    for asite in london_air_stations:
        try:
            # abase = "http://api.erg.ic.ac.uk/AirQuality/Data/Site/Wide/"
            # aargs = "SiteCode=CT8/StartDate=2021-07-01/EndDate=2021-07-30/csv"
            abase = 'https://www.londonair.org.uk/london/asp/downloadsite.asp'
            aargs = f"?site={asite}&species1={london_air_species}m&species2=&species3=&species4=&species5=&species6=&start={start_date}&end={str_end_time}&res=6&period=daily&units=ugm3"
            s = requests.get(abase + aargs).content
            adf = pd.read_csv(io.StringIO(s.decode('utf-8')))
            acol = adf['Site'].iloc[0] + "_" + adf['Species'].iloc[0]
            adf['Datetime'] = pd.to_datetime(adf['ReadingDateTime'], dayfirst=True)
            adf[acol] = adf['Value']
            dataset_lists.append(adf[['Datetime', acol]].set_index("Datetime"))
            # "/Data/Traffic/Site/SiteCode={SiteCode}/StartDate={StartDate}/EndDate={EndDate}/Json"
        except Exception as e:
            print(f"London Air data failed: {repr(e)}")

    try:
        str_end_time = current_date.strftime("%Y-%m-%d")
        start_date = (current_date - datetime.timedelta(days=earthquake_days)).strftime(
            "%Y-%m-%d"
        )
        # is limited to ~1000 rows of data, ie individual earthquakes
        ebase = "https://earthquake.usgs.gov/fdsnws/event/1/query?"
        eargs = f"format=csv&starttime={start_date}&endtime={str_end_time}&minmagnitude={earthquake_min_magnitude}"
        eq = pd.read_csv(ebase + eargs)
        eq["time"] = pd.to_datetime(eq["time"], infer_datetime_format=True)
        eq["time"] = eq["time"].dt.tz_localize(None)
        eq.set_index("time", inplace=True)
        global_earthquakes = eq.resample("1D").agg({"mag": "mean", "depth": "count"})
        global_earthquakes["mag"] = global_earthquakes["mag"].fillna(
            earthquake_min_magnitude
        )
        global_earthquakes = global_earthquakes.rename(
            columns={
                "mag": "largest_magnitude_earthquake",
                "depth": "count_large_earthquakes",
            }
        )
        dataset_lists.append(global_earthquakes)
    except Exception as e:
        print(f"earthquake data failed: {repr(e)}")

    if len(dataset_lists) < 1:
        raise ValueError("No data successfully downloaded!")
    elif len(dataset_lists) == 1:
        df = dataset_lists[0]
    else:
        from functools import reduce

        df = reduce(
            lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"),
            dataset_lists,
        )
    print(f"{df.shape[1]} series downloaded.")

    if not long:
        return df
    else:
        df_long = df.reset_index(drop=False).melt(
            id_vars=['datetime'], var_name='series_id', value_name='value'
        )
        return df_long


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


def load_linear(long=False, shape=None, start_date: str = "2021-01-01"):
    """Create a dataset of just zeroes for testing edge case."""
    if shape is None:
        shape = (500, 5)
    df_wide = pd.DataFrame(
        np.ones(shape), index=pd.date_range(start_date, periods=shape[0], freq="D")
    )
    df_wide = (df_wide * list(range(0, shape[1]))).cumsum()
    if not long:
        return df_wide
    else:
        df_wide.index.name = "datetime"
        df_long = df_wide.reset_index(drop=False).melt(
            id_vars=['datetime'], var_name='series_id', value_name='value'
        )
        return df_long


def load_sine(long=False, shape=None, start_date: str = "2021-01-01"):
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
        return a * np.sin(1 * X) + a

    for column in df_wide.columns:
        df_wide[column] = sin_func(column, X)
    if not long:
        return df_wide
    else:
        df_wide.index.name = "datetime"
        df_long = df_wide.reset_index(drop=False).melt(
            id_vars=['datetime'], var_name='series_id', value_name='value'
        )
        return df_long
