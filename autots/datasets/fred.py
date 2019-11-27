"""
FRED (Federal Reserve Economic Data) Data Import

requires API key from FRED
and pip install fredapi
"""
import pandas as pd

try:
    from fredapi import Fred
except:
    raise ImportError("Package fredapi is required, install with `pip install fredapi`")

def get_fred_data(fredkey: str, SeriesNameDict: dict = {'SeriesID':'SeriesName'}):
    """
    Imports Data from Federal Reserve
    
    args:
        fredkey - an API key from FRED
        
        SeriesNameDict, pairs of FRED Series IDs and Series Names 
            Series id must match Fred IDs, but name can be anything
            if default is use, several default samples are returned
    """
    fred = Fred(api_key=fredkey)
    
    if SeriesNameDict == {'SeriesID':'SeriesName'}:
        SeriesNameDict = {'T10Y2Y':'10 Year Treasury Constant Maturity Minus 2 Year Treasury Constant Maturity', 
                          'DGS10': '10 Year Treasury Constant Maturity Rate',
                          'DCOILWTICO':'Crude Oil West Texas Intermediate Cushing Oklahoma', 
                          'SP500': 'S&P 500', 
                          'DEXUSEU': 'US Euro Foreign Exchange Rate',
                          'DEXCHUS': 'China US Foreign Exchange Rate',
                          'DEXCAUS' : 'Canadian to US Dollar Exchange Rate Daily',
                          'VIXCLS': 'CBOE Volatility Index: VIX',  # this is a more irregular series
                          'T10YIE' : '10 Year Breakeven Inflation Rate',
                          'USEPUINDXD': 'Economic Policy Uncertainty Index for United States' # also very irregular
                          }
    
    series_desired = list(SeriesNameDict.keys())
    
    fred_timeseries = pd.DataFrame(columns = ['date', 'value', 'series_id','series_name'])
    
    for series in series_desired:
        data = fred.get_series(series)
        try:
            series_name = SeriesNameDict[series]
        except Exception:
            series_name = series
        data_df = pd.DataFrame({'date':data.index, 'value':data, 'series_id':series, 'series_name':series_name})
        data_df.reset_index(drop=True, inplace = True)
        fred_timeseries = pd.concat([fred_timeseries, data_df], axis = 0, ignore_index = True)

    return fred_timeseries