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


def load_live_daily(
    long: bool = False,
    observation_start: str = None,
    observation_end: str = None,
    fred_key: str = None,
    fred_series=["DGS10", "T5YIE", "SP500", "DCOILWTICO", "DEXUSEU", "WPU0911"],
    tickers: list = ["MSFT"],
    trends_list: list = ["forecasting", "cycling", "microsoft"],
    trends_geo: str = "US",
    weather_data_types: list = ["AWND", "WSF2", "TAVG", "PRCP"],
    weather_stations: list = ["USW00094846", "USW00014925", "USW00014771"],
    weather_years: int = 5,
    london_air_stations: list = ['CT3', 'SK8'],
    london_air_species: str = "PM25",
    london_air_days: int = 180,
    earthquake_days: int = 180,
    earthquake_min_magnitude: int = 5,
    gsa_key: str = None,  # https://open.gsa.gov/api/dap/
    gov_domain_list=['nasa.gov'],
    gov_domain_limit: int = 600,
    wikipedia_pages: list = ['Microsoft_Office', "List_of_highest-grossing_films"],
    wiki_language: str = "en",
    weather_event_types=["%28Z%29+Winter+Weather", "%28Z%29+Winter+Storm"],
    caiso_query: str = None,
    eia_key: str = None,
    eia_respondents: list = ["MISO", "PJM", "TVA", "US48"],
    timeout: float = 300.05,
    sleep_seconds: int = 2,
    **kwargs,
):
    """Generates a dataframe of data up to the present day. Requires active internet connection.
    Try to be respectful of these free data sources by not calling too much too heavily.
    Pass None instead of specification lists to exclude a data source.

    Args:
        long (bool): whether to return in long format or wide
        observation_start (str): %Y-%m-%d earliest day to retrieve, passed to Fred.get_series and yfinance.history
            note that apis with more restrictions have other default lengths below which ignore this
        observation_end (str):  %Y-%m-%d most recent day to retrieve
        fred_key (str): https://fred.stlouisfed.org/docs/api/api_key.html
        fred_series (list): list of FRED series IDs. This requires fredapi package
        tickers (list): list of stock tickers, requires yfinance pypi package
        trends_list (list): list of search keywords, requires pytrends pypi package. None to skip.
        weather_data_types (list): from NCEI NOAA api data types, GHCN Daily Weather Elements
            PRCP, SNOW, TMAX, TMIN, TAVG, AWND, WSF1, WSF2, WSF5, WSFG
        weather_stations (list): from NCEI NOAA api station ids. Pass empty list to skip.
        london_air_stations (list): londonair.org.uk source station IDs. Pass empty list to skip.
        london_species (str): what measurement to pull from London Air. Not all stations have all metrics.
        earthquake_min_magnitude (int): smallest earthquake magnitude to pull from earthquake.usgs.gov. Set None to skip this.
        gsa_key (str): api key from https://open.gsa.gov/api/dap/
        gov_domain_list (list): dist of government run domains to get traffic data for. Can be very slow, so fewer is better.
            some examples: ['usps.com', 'ncbi.nlm.nih.gov', 'cdc.gov', 'weather.gov', 'irs.gov', "usajobs.gov", "studentaid.gov", 'nasa.gov', "uk.usembassy.gov", "tsunami.gov"]
        gov_domain_limit (int): max number of records. Smaller will be faster. Max is currently 10000.
        wikipedia_pages (list): list of Wikipedia pages, html encoded if needed (underscore for space)
        weather_event_types (list): list of html encoded severe weather event types https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Export-Format.pdf
        caiso_query (str): ENE_SLRS or None, can try others but probably won't work due to other hardcoded params
        timeout (float): used by some queries
        sleep_seconds (int): increasing this may reduce probability of server download failures
    """
    assert sleep_seconds >= 0.5, "sleep_seconds must be >=0.5"

    dataset_lists = []
    if observation_end is None:
        current_date = datetime.datetime.utcnow()
    else:
        current_date = observation_end
    if observation_start is None:
        # should take from observation_end but that's expected as a string
        observation_start = datetime.datetime.utcnow() - datetime.timedelta(
            days=365 * 6
        )
        observation_start = observation_start.strftime("%Y-%m-%d")
    try:
        import requests

        s = requests.Session()
    except Exception as e:
        print(f"requests Session creation failed {repr(e)}")

    try:
        if fred_key is not None and fred_series is not None:
            from fredapi import Fred  # noqa
            from autots.datasets.fred import get_fred_data

            fred_df = get_fred_data(
                fred_key,
                fred_series,
                long=False,
                observation_start=observation_start,
                sleep_seconds=sleep_seconds,
            )
            fred_df.index = fred_df.index.tz_localize(None)
            dataset_lists.append(fred_df)
    except ModuleNotFoundError:
        print("pip install fredapi (and you'll also need an api key)")
    except Exception as e:
        print(f"FRED data failed: {repr(e)}")

    if tickers is not None:
        for ticker in tickers:
            try:
                import yfinance as yf

                msft = yf.Ticker(ticker)
                # get historical market data
                msft_hist = msft.history(start=observation_start)
                msft_hist = msft_hist.rename(
                    columns=lambda x: x.lower().replace(" ", "_")
                )
                msft_hist = msft_hist.rename(columns=lambda x: ticker.lower() + "_" + x)
                try:
                    msft_hist.index = msft_hist.index.tz_localize(None)
                except Exception:
                    pass
                dataset_lists.append(msft_hist)
                time.sleep(sleep_seconds)
            except ModuleNotFoundError:
                print("You need to: pip install yfinance")
            except Exception as e:
                print(f"yfinance data failed: {repr(e)}")

    str_end_time = current_date.strftime("%Y-%m-%d")
    start_date = (current_date - datetime.timedelta(days=360 * weather_years)).strftime(
        "%Y-%m-%d"
    )
    if weather_stations is not None:
        for wstation in weather_stations:
            try:
                wbase = "https://www.ncei.noaa.gov/access/services/data/v1/?dataset=daily-summaries"
                wargs = f"&dataTypes={','.join(weather_data_types)}&stations={wstation}"
                wargs = (
                    wargs
                    + f"&startDate={start_date}&endDate={str_end_time}&boundingBox=90,-180,-90,180&units=standard&format=csv"
                )
                wdf = pd.read_csv(
                    io.StringIO(s.get(wbase + wargs, timeout=timeout).text)
                )
                wdf['DATE'] = pd.to_datetime(wdf['DATE'])
                wdf = wdf.set_index('DATE').drop(columns=['STATION'])
                wdf.rename(columns=lambda x: wstation + "_" + x, inplace=True)
                dataset_lists.append(wdf)
                time.sleep(sleep_seconds)
            except Exception as e:
                print(f"weather data failed: {repr(e)}")

    str_end_time = current_date.strftime("%d-%b-%Y")
    start_date = (current_date - datetime.timedelta(days=london_air_days)).strftime(
        "%d-%b-%Y"
    )
    if london_air_stations is not None:
        for asite in london_air_stations:
            try:
                # abase = "http://api.erg.ic.ac.uk/AirQuality/Data/Site/Wide/"
                # aargs = "SiteCode=CT8/StartDate=2021-07-01/EndDate=2021-07-30/csv"
                abase = 'https://www.londonair.org.uk/london/asp/downloadsite.asp'
                aargs = f"?site={asite}&species1={london_air_species}m&species2=&species3=&species4=&species5=&species6=&start={start_date}&end={str_end_time}&res=6&period=daily&units=ugm3"
                data = s.get(abase + aargs, timeout=timeout).content
                adf = pd.read_csv(io.StringIO(data.decode('utf-8')))
                acol = adf['Site'].iloc[0] + "_" + adf['Species'].iloc[0]
                adf['Datetime'] = pd.to_datetime(adf['ReadingDateTime'], dayfirst=True)
                adf[acol] = adf['Value']
                dataset_lists.append(adf[['Datetime', acol]].set_index("Datetime"))
                time.sleep(sleep_seconds)
                # "/Data/Traffic/Site/SiteCode={SiteCode}/StartDate={StartDate}/EndDate={EndDate}/Json"
            except Exception as e:
                print(f"London Air data failed: {repr(e)}")

    if earthquake_min_magnitude is not None:
        try:
            str_end_time = current_date.strftime("%Y-%m-%d")
            start_date = (
                current_date - datetime.timedelta(days=earthquake_days)
            ).strftime("%Y-%m-%d")
            # is limited to ~1000 rows of data, ie individual earthquakes
            ebase = "https://earthquake.usgs.gov/fdsnws/event/1/query?"
            eargs = f"format=csv&starttime={start_date}&endtime={str_end_time}&minmagnitude={earthquake_min_magnitude}"
            eq = pd.read_csv(ebase + eargs)
            eq["time"] = pd.to_datetime(eq["time"])
            eq["time"] = eq["time"].dt.tz_localize(None)
            eq.set_index("time", inplace=True)
            global_earthquakes = eq.resample("1D").agg(
                {"mag": "mean", "depth": "count"}
            )
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

    if gov_domain_list is not None:
        try:
            # print because this one is slow, and point people at that fact
            if gsa_key is None:
                gsa_key = "DEMO_KEY2"
            # only run 1 if demo_key1
            if "DEMO_KEY" in gsa_key:
                gov_domain_list = gov_domain_list[0:1]
            for domain in gov_domain_list:
                report = "domain"  # site, domain, download, second-level-domain
                url = f"https://api.gsa.gov/analytics/dap/v1.1/domain/{domain}/reports/{report}/data?api_key={gsa_key}&limit={gov_domain_limit}&after={observation_start}"
                data = s.get(url, timeout=timeout)
                gdf = pd.read_json(data.text, orient="records")
                gdf['date'] = pd.to_datetime(gdf['date'])
                # essentially duplicates brought by agency and null agency
                gresult = gdf.groupby('date')['visits'].first()
                gresult.name = domain
                dataset_lists.append(gresult.to_frame())
                time.sleep(sleep_seconds)
        except Exception as e:
            print(f"analytics.gov data failed with {repr(e)}")

    if wikipedia_pages is not None:
        str_start = pd.to_datetime(observation_start).strftime("%Y%m%d00")
        str_end = current_date.strftime("%Y%m%d00")
        headers = {
            'User-Agent': 'AutoTS load_live_daily',
        }
        for page in wikipedia_pages:
            try:
                if page == "all":
                    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/all-projects/all-access/all-agents/daily/{str_start}/{str_end}?maxlag=5"
                else:
                    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{wiki_language}.wikipedia/all-access/all-agents/{page}/daily/{str_start}/{str_end}?maxlag=5"
                data = s.get(url, timeout=timeout, headers=headers)
                data_js = data.json()
                if "items" not in data_js.keys():
                    print(data_js)
                gdf = pd.DataFrame(data_js['items'])
                gdf['date'] = pd.to_datetime(gdf['timestamp'], format="%Y%m%d00")
                gresult = gdf.set_index('date')['views'].fillna(0)
                gresult.name = "wiki_" + str(page)[0:80]
                dataset_lists.append(gresult.to_frame())
                time.sleep(sleep_seconds)
            except Exception as e:
                print(f"Wikipedia api failed with error {repr(e)}")
                time.sleep(10)

    if weather_event_types is not None:
        try:
            for event_type in weather_event_types:
                # appears to have a fixed max of 500 records
                url = f"https://www.ncdc.noaa.gov/stormevents/csv?eventType={event_type}&beginDate_mm=01&beginDate_dd=01&beginDate_yyyy=2000&endDate_mm=09&endDate_dd=30&endDate_yyyy=9999&hailfilter=0.00&tornfilter=2&windfilter=000&sort=DN&statefips=-999%2CALL"
                csv_in = io.StringIO(s.get(url, timeout=timeout).text)
                try:
                    # new in 1.3.0 of pandas
                    df = pd.read_csv(csv_in, low_memory=False, on_bad_lines='skip')
                except Exception:
                    df = pd.read_csv(csv_in, low_memory=False, error_bad_lines=False)
                df['BEGIN_DATE'] = pd.to_datetime(df['BEGIN_DATE'])
                df['END_DATE'] = pd.to_datetime(df['END_DATE'])
                df['day'] = df.apply(
                    lambda row: pd.date_range(
                        row["BEGIN_DATE"], row['END_DATE'], freq='D'
                    ),
                    axis=1,
                )
                df = df.explode('day')
                swresult = df.groupby(["day"])["EVENT_ID"].count()
                swresult.name = "_".join(event_type.split("+")[1:]) + "_Events"
                dataset_lists.append(swresult.to_frame())
                time.sleep(sleep_seconds)
        except Exception as e:
            print(f"Severe Weather data failed with {repr(e)}")

    if trends_list is not None:
        try:
            from pytrends.request import TrendReq

            pytrends = TrendReq(hl="en-US", tz=360)
            pytrends.build_payload(trends_list, geo=trends_geo)
            # pytrends.build_payload(trends_list, timeframe="all")  # 'today 12-m'
            gtrends = pytrends.interest_over_time()
            gtrends.index = gtrends.index.tz_localize(None)
            gtrends.drop(columns="isPartial", inplace=True, errors="ignore")
            dataset_lists.append(gtrends)
        except ImportError:
            print("You need to: pip install pytrends")
        except Exception as e:
            print(f"pytrends data failed: {repr(e)}")

    # this was kinda broken last I checked
    if caiso_query is not None:
        try:
            n_chunks = (364 * weather_years) / 30
            if n_chunks % 30 != 0:
                n_chunks = int(n_chunks) + 1
            energy_df = []
            for x in range(n_chunks):
                try:
                    end_nospace = (
                        current_date - datetime.timedelta(days=30 * x)
                    ).strftime("%Y%m%d")
                    start_nospace = (
                        current_date - datetime.timedelta(days=30 * (x + 1) + 1)
                    ).strftime("%Y%m%d")
                    caiso_url = f"http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname={caiso_query}&version=1&market_run_id=RTM&tac_zone_name=ALL&schedule=Generation&startdatetime={start_nospace}T00:00-0000&enddatetime={end_nospace}T23:00-0000"
                    data = pd.read_csv(caiso_url, compression='zip')
                    data['OPR_DT'] = pd.to_datetime(data['OPR_DT'])
                    data = data[data['OPR_HR'] < 25]
                    energy_df.append(
                        data.groupby(['OPR_DT', 'OPR_HR'])['MW']
                        .mean()
                        .reset_index()
                        .pivot_table(
                            values='MW', index='OPR_DT', columns='OPR_HR', aggfunc='sum'
                        )
                        .rename(columns=lambda x: "CAISO_GENERATION_HR_" + str(x))
                        .sort_index()
                        .bfill()
                    )
                    time.sleep(sleep_seconds + 8)
                except Exception as e:
                    print(f"caiso download failed with error: {repr(e)}")
                    time.sleep(sleep_seconds)
            energy_df = pd.concat(energy_df).sort_index()
            energy_df = energy_df[~energy_df.index.duplicated(keep='last')]
            dataset_lists.append(energy_df)
        except Exception as e:
            print(f"caiso download failed with error: {repr(e)}")

    if eia_key is not None and eia_respondents is not None:
        api_url = 'https://api.eia.gov/v2/electricity/rto/daily-region-data/data/'  # ?api_key={eia-key}
        for respond in eia_respondents:
            try:
                params = {
                    "frequency": "daily",
                    "data": ["value"],
                    "facets": {
                        "type": ["D"],
                        "respondent": [respond],
                        "timezone": ["Eastern"],
                    },
                    "start": None,  # "start": "2018-06-30",
                    "end": None,  # "end": "2023-11-01",
                    "sort": [{"column": "period", "direction": "desc"}],
                    "offset": 0,
                    "length": 5000,
                }

                res = s.get(
                    api_url,
                    params={
                        "api_key": eia_key,
                    },
                    headers={"X-Params": json.dumps(params)},
                )
                eia_df = pd.json_normalize(res.json()['response']['data'])
                eia_df['datetime'] = pd.to_datetime(eia_df['period'])
                eia_df['value'] = eia_df['value'].astype('float')
                eia_df['ID'] = (
                    eia_df['respondent']
                    + "_"
                    + eia_df['type']
                    + "_"
                    + eia_df['timezone']
                )
                temp = eia_df.pivot(columns='ID', index='datetime', values='value')
                dataset_lists.append(temp)
                time.sleep(sleep_seconds)
            except Exception as e:
                print(f"eia download failed with error {repr(e)}")
            try:
                api_url_mix = (
                    "https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/"
                )
                params = {
                    "frequency": "daily",
                    "data": ["value"],
                    "facets": {
                        "respondent": [respond],
                        "timezone": ["Eastern"],
                        "fueltype": [
                            "COL",
                            "NG",
                            "NUC",
                            "SUN",
                            "WAT",
                            "WND",
                        ],
                    },
                    "start": None,
                    "end": None,
                    "sort": [{"column": "period", "direction": "desc"}],
                    "offset": 0,
                    "length": 5000,
                }
                res = s.get(
                    api_url_mix,
                    params={
                        "api_key": eia_key,
                    },
                    headers={"X-Params": json.dumps(params)},
                )
                eia_df = pd.json_normalize(res.json()['response']['data'])
                eia_df['datetime'] = pd.to_datetime(eia_df['period'])
                eia_df['value'] = eia_df['value'].astype('float')
                eia_df['type-name'] = eia_df['type-name'].str.replace(" ", "_")
                eia_df['ID'] = (
                    eia_df['respondent']
                    + "_"
                    + eia_df['type-name']
                    + "_"
                    + eia_df['timezone']
                )
                temp = eia_df.pivot(columns='ID', index='datetime', values='value')
                dataset_lists.append(temp)
                time.sleep(1)
            except Exception as e:
                print(f"eia download failed with error {repr(e)}")

    ### End of data download
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
    s.close()
    df.index.name = "datetime"

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
                scipy.signal.ricker(33, 1), int(np.ceil(size / 33))
            )[:size],
            "wavelet_morlet": np.real(
                np.tile(scipy.signal.morlet2(100, 6.0, 6.0), int(np.ceil(size / 100)))[
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
                        scipy.signal.morlet2(50, 6.0, 6.0), int(np.ceil(size / 50))
                    )[:size]
                    * 10
                ),
                np.real(
                    np.tile(
                        scipy.signal.morlet2(50, 6.0, 0.0), int(np.ceil(size / 50))
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
