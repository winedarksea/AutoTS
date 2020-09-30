"""Loading example datasets."""
from os.path import dirname, join
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


def load_monthly():
    """Federal Reserve of St. Louis monthly economic indicators."""
    return load_fred_monthly()


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


def load_yearly():
    """Federal Reserve of St. Louis annual economic indicators."""
    return load_fred_yearly()


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


def load_weekly():
    """Weekly petroleum industry data from the EIA."""
    return load_eia_weekly()
