"""
FRED (Federal Reserve Economic Data) Data Import

requires API key from FRED
and pip install fredapi
"""

import time
import pandas as pd

try:
    from fredapi import Fred
except Exception:  # except ImportError
    _has_fred = False
else:
    _has_fred = True


def get_fred_data(
    fredkey: str,
    SeriesNameDict: dict = None,
    long=True,
    observation_start=None,
    sleep_seconds: int = 1,
    **kwargs,
):
    """Imports Data from Federal Reserve.
    For simplest results, make sure requested series are all of the same frequency.

    args:
        fredkey (str): an API key from FRED
        SeriesNameDict (dict): pairs of FRED Series IDs and Series Names like: {'SeriesID': 'SeriesName'} or a list of FRED IDs.
            Series id must match Fred IDs, but name can be anything
            if None, several default series are returned
        long (bool): if True, return long style data, else return wide style data with dt index
        observation_start (datetime): passed to Fred get_series
        sleep_seconds (int): seconds to sleep between each series call, reduces failure chance usually
    """
    if not _has_fred:
        raise ImportError("Package fredapi is required")

    fred = Fred(api_key=fredkey)

    if SeriesNameDict is None:
        SeriesNameDict = {
            'T10Y2Y': '10 Year Treasury Constant Maturity Minus 2 Year Treasury Constant Maturity',
            'DGS10': '10 Year Treasury Constant Maturity Rate',
            'DCOILWTICO': 'Crude Oil West Texas Intermediate Cushing Oklahoma',
            'SP500': 'S&P 500',
            'DEXUSEU': 'US Euro Foreign Exchange Rate',
            'DEXCHUS': 'China US Foreign Exchange Rate',
            'DEXCAUS': 'Canadian to US Dollar Exchange Rate Daily',
            'VIXCLS': 'CBOE Volatility Index: VIX',  # this is a more irregular series
            'T10YIE': '10 Year Breakeven Inflation Rate',
            'USEPUINDXD': 'Economic Policy Uncertainty Index for United States',  # also very irregular
        }

    if isinstance(SeriesNameDict, dict):
        series_desired = list(SeriesNameDict.keys())
    else:
        series_desired = list(SeriesNameDict)

    if long:
        fred_timeseries = pd.DataFrame(
            columns=['date', 'value', 'series_id', 'series_name']
        )
    else:
        fred_timeseries = pd.DataFrame()

    for series in series_desired:
        data = fred.get_series(series, observation_start=observation_start)
        try:
            series_name = SeriesNameDict[series]
        except Exception:
            series_name = series

        if long:
            data_df = pd.DataFrame(
                {
                    'date': data.index,
                    'value': data,
                    'series_id': series,
                    'series_name': series_name,
                }
            )
            data_df.reset_index(drop=True, inplace=True)
            fred_timeseries = pd.concat(
                [fred_timeseries, data_df], axis=0, ignore_index=True
            )
        else:
            data.name = series_name
            fred_timeseries = fred_timeseries.merge(
                data, how="outer", left_index=True, right_index=True
            )
        time.sleep(sleep_seconds)

    return fred_timeseries
