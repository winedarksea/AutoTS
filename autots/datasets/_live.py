"""Live data loaders that require an active internet connection.

Moved from _base.py to keep file sizes manageable. The primary entry point
is load_live_daily, which aggregates data from multiple free external APIs
into a single wide-format DataFrame.
"""

import time
import datetime
import io
import math
import sys
import numpy as np
import pandas as pd


def _format_live_download_error(error):
    """Explain browser transport failures without reporting synthetic HTTP 0."""
    error_text = repr(error)
    if sys.platform == "emscripten" and (
        "HTTP/1.1 0" in error_text
        or "HTTP 0" in error_text
        or "status 0" in error_text.lower()
    ):
        return (
            "Browser blocked the request before receiving an HTTP response "
            "(usually a CORS restriction)."
        )
    return error_text


def _eia_query_params(params, prefix=""):
    """Flatten EIA's nested params dict into bracketed query-string tuples.

    EIA's v2 API accepts the same query either as an ``X-Params`` JSON header or
    as bracketed query-string params (e.g. ``facets[respondent][0]=MISO``). The
    query-string form is a "simple" request that avoids a CORS preflight, so it
    works from the browser (Pyodide) while remaining identical for normal Python
    users. ``None`` values are omitted to match the header form, where JSON
    ``null`` is ignored by the API.

    Returns a list of ``(key, value)`` tuples suitable for ``requests``' ``params``
    (a list preserves repeated keys; ``requests`` percent-encodes the brackets,
    which EIA accepts).
    """
    items = []
    if isinstance(params, dict):
        iterator = params.items()
    elif isinstance(params, (list, tuple)):
        iterator = enumerate(params)
    else:
        if params is not None:
            items.append((prefix, params))
        return items
    for key, value in iterator:
        child_prefix = f"{prefix}[{key}]" if prefix else str(key)
        if isinstance(value, (dict, list, tuple)):
            items.extend(_eia_query_params(value, child_prefix))
        elif value is not None:
            items.append((child_prefix, value))
    return items


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
    noaa_cdo_token: str = None,  # https://www.ncdc.noaa.gov/cdo-web/token
    london_air_stations: list = ['KC1', 'BX2'],
    london_air_species: str = "PM25",
    london_air_days: int = 180,
    earthquake_days: int = 180,
    earthquake_min_magnitude: int = 5,
    gsa_key: str = None,  # https://open.gsa.gov/api/dap/
    nasa_api_key: str = "DEMO_KEY",
    gov_domain_list=['nasa.gov'],
    gov_domain_limit: int = 600,
    wikipedia_pages: list = ['Microsoft_Office', "List_of_highest-grossing_films"],
    wiki_language: str = "en",
    weather_event_types=["%28Z%29+Winter+Weather", "%28Z%29+Winter+Storm"],
    caiso_query: str = None,
    eia_key: str = None,
    eia_respondents: list = ["MISO", "PJM", "TVA", "US48"],
    timeout: float = 300.05,
    sleep_seconds: int = 10,
    progress_cb=None,
    status_log=None,
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
        noaa_cdo_token (str): API token from https://www.ncdc.noaa.gov/cdo-web/token (free, required for weather data)
        london_air_stations (list): londonair.org.uk source station IDs. Pass empty list to skip.
        london_species (str): what measurement to pull from London Air. Not all stations have all metrics.
        earthquake_min_magnitude (int): smallest earthquake magnitude to pull from earthquake.usgs.gov. Set None to skip this.
        gsa_key (str): api key from https://open.gsa.gov/api/dap/
        nasa_api_key (str): API key for https://api.nasa.gov/. Set to None to skip NASA DONKI data.
        gov_domain_list (list): dist of government run domains to get traffic data for. Can be very slow, so fewer is better.
            some examples: ['usps.com', 'ncbi.nlm.nih.gov', 'cdc.gov', 'weather.gov', 'irs.gov', "usajobs.gov", "studentaid.gov", 'nasa.gov', "uk.usembassy.gov", "tsunami.gov"]
        gov_domain_limit (int): max number of records. Smaller will be faster. Max is currently 10000.
        wikipedia_pages (list): list of Wikipedia pages, html encoded if needed (underscore for space)
        weather_event_types (list): list of html encoded severe weather event types https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Export-Format.pdf
        caiso_query (str): ENE_SLRS or None, can try others but probably won't work due to other hardcoded params
        timeout (float): used by some queries
        sleep_seconds (int): increasing this may reduce probability of server download failures
        progress_cb (callable): optional, called with a short status string as each source begins.
            Useful for driving a progress bar in a UI.
        status_log (list): optional, if a list is passed it is appended with one dict per attempted
            source: {"source": name, "status": "ok"|"failed", "series": n_columns, "error": repr}.
            Lets callers report which sources succeeded/failed without parsing stdout.
    """
    # TODO: add a proper unittest for this, minimal for each just to test that each API is not broken
    assert sleep_seconds >= 0.5, "sleep_seconds must be >=0.5"

    dataset_lists = []

    # --- per-source progress + status instrumentation (optional) -------------
    # Both default to None, so library/MCP callers see unchanged behavior.
    enabled_sources = []
    if fred_key is not None and fred_series is not None:
        enabled_sources.append("FRED")
    if tickers is not None:
        enabled_sources.append("Stock tickers")
    if weather_stations is not None:
        enabled_sources.append("NOAA weather")
    if london_air_stations is not None:
        enabled_sources.append("London Air")
    if earthquake_min_magnitude is not None:
        enabled_sources.append("Earthquakes")
    if nasa_api_key is not None:
        enabled_sources.append("NASA space weather")
    if gov_domain_list is not None:
        enabled_sources.append("Gov web analytics")
    if wikipedia_pages is not None:
        enabled_sources.append("Wikipedia")
    if weather_event_types is not None:
        enabled_sources.append("Severe weather events")
    if trends_list is not None:
        enabled_sources.append("Google Trends")
    if caiso_query is not None:
        enabled_sources.append("CAISO")
    if eia_key is not None and eia_respondents is not None:
        enabled_sources.append("EIA electricity")
    _total_sources = len(enabled_sources)
    _source_counter = {"i": 0}

    def _start_source(label):
        """Announce a source is starting; return the current dataset_lists length."""
        _source_counter["i"] += 1
        if progress_cb is not None:
            try:
                progress_cb(
                    f"Loading {label} ({_source_counter['i']}/{_total_sources})…"
                )
            except Exception:
                pass
        return len(dataset_lists)

    def _finish_source(label, start_len, error=None):
        """Record per-source outcome into status_log (a no-op if not provided)."""
        if status_log is None:
            return
        added = dataset_lists[start_len:]
        n_series = 0
        for d in added:
            d = d[(d.index >= observation_start_timestamp) & (d.index <= current_date)]
            if isinstance(d, pd.DataFrame):
                n_series += int(d.notna().any(axis=0).sum())
            elif isinstance(d, pd.Series):
                n_series += int(d.notna().any())
        if n_series:
            entry = {"source": label, "status": "ok", "series": int(n_series)}
            if error:
                entry["error"] = error
        else:
            entry = {
                "source": label,
                "status": "failed",
                "error": error or "no usable data returned",
            }
        status_log.append(entry)

    if observation_end is None:
        current_date = pd.Timestamp(datetime.datetime.utcnow())
    else:
        try:
            current_date = pd.Timestamp(observation_end)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "observation_end must be a valid scalar date or datetime."
            ) from exc
        if pd.isna(current_date):
            raise ValueError(
                "observation_end must be a valid scalar date or datetime."
            )
        if current_date.tzinfo is not None:
            current_date = current_date.tz_localize(None)
    if observation_start is None:
        observation_start = current_date - datetime.timedelta(days=365 * 6)
        observation_start = observation_start.strftime("%Y-%m-%d")
    try:
        observation_start_timestamp = pd.Timestamp(observation_start)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "observation_start must be a valid scalar date or datetime."
        ) from exc
    if pd.isna(observation_start_timestamp):
        raise ValueError(
            "observation_start must be a valid scalar date or datetime."
        )
    if observation_start_timestamp.tzinfo is not None:
        observation_start_timestamp = observation_start_timestamp.tz_localize(None)
    if observation_start_timestamp > current_date:
        raise ValueError("observation_start must be on or before observation_end.")
    try:
        import requests

        s = requests.Session()
    except Exception as e:
        raise ValueError(
            f"Live data HTTP transport is unavailable: {e!r}"
        ) from e

    if fred_key is not None and fred_series is not None:
        _blk = _start_source("FRED")
        try:
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
            _finish_source("FRED", _blk)
        except ModuleNotFoundError:
            print("pip install fredapi (and you'll also need an api key)")
            _finish_source("FRED", _blk, error="fredapi not installed")
        except Exception as e:
            error_message = _format_live_download_error(e)
            print(f"FRED data failed: {error_message}")
            _finish_source("FRED", _blk, error=error_message)

    if tickers is not None:
        _blk = _start_source("Stock tickers")
        _ticker_errs = []
        for ticker in tickers:
            try:
                import yfinance as yf

                msft = yf.Ticker(ticker)
                # get historical market data
                msft_hist = msft.history(start=observation_start)
                msft_hist = msft_hist.rename(
                    columns=lambda x: x.lower().replace(" ", "_")
                )
                ticker_lower = ticker.lower()
                msft_hist = msft_hist.rename(columns=lambda x: ticker_lower + "_" + x)
                close_col = f"{ticker_lower}_close"
                if close_col in msft_hist.columns:
                    prev_close = msft_hist[close_col].ffill().shift(1)
                    pct_col = f"{ticker_lower}_close_pct_change"
                    msft_hist[pct_col] = (
                        msft_hist[close_col] - prev_close
                    ) / prev_close
                    direction_col = f"{ticker_lower}_close_direction"
                    delta = msft_hist[close_col] - prev_close
                    direction = np.select(
                        [delta > 0, delta < 0, delta == 0],
                        [1, -1, 0],
                        default=np.nan,
                    )
                    msft_hist[direction_col] = direction
                try:
                    msft_hist.index = msft_hist.index.tz_localize(None)
                except Exception:
                    pass
                dataset_lists.append(msft_hist)
                time.sleep(sleep_seconds)
            except ModuleNotFoundError:
                print("You need to: pip install yfinance")
                _ticker_errs.append("yfinance not installed")
            except Exception as e:
                print(f"yfinance data failed: {repr(e)}")
                _ticker_errs.append(f"{ticker}: {repr(e)}")
        _finish_source(
            "Stock tickers",
            _blk,
            error="; ".join(_ticker_errs) if _ticker_errs else None,
        )

    str_end_time = current_date.strftime("%Y-%m-%d")
    weather_start_date = current_date - datetime.timedelta(days=360 * weather_years)

    if weather_stations is not None:
        _blk = _start_source("NOAA weather")
        _weather_errs = []
        for wstation in weather_stations:
            try:
                # NOAA CDO Web Services v2 API
                # Documentation: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
                # Request token: https://www.ncdc.noaa.gov/cdo-web/token
                if noaa_cdo_token is None:
                    print(
                        f"weather data skipped for {wstation}: noaa_cdo_token required. Get free token at https://www.ncdc.noaa.gov/cdo-web/token"
                    )
                    continue

                # Use v2 API endpoint
                wbase = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
                headers = {'token': noaa_cdo_token}

                # API requires date ranges to be less than 1 year, so we need to chunk requests
                station_data = []

                for dtype in weather_data_types:
                    # Split date range into chunks of less than 1 year (use 360 days to be safe)
                    chunk_size_days = 360
                    all_results = []

                    current_chunk_start = weather_start_date
                    while current_chunk_start < current_date:
                        current_chunk_end = min(
                            current_chunk_start
                            + datetime.timedelta(days=chunk_size_days),
                            current_date,
                        )

                        start_date_str = current_chunk_start.strftime("%Y-%m-%d")
                        end_date_str = current_chunk_end.strftime("%Y-%m-%d")

                        # Paginate within each chunk if needed
                        offset = 1
                        max_requests_per_chunk = 10
                        request_count = 0

                        while request_count < max_requests_per_chunk:
                            params = {
                                'datasetid': 'GHCND',  # Global Historical Climatology Network - Daily
                                'stationid': f'GHCND:{wstation}',
                                'datatypeid': dtype,
                                'startdate': start_date_str,
                                'enddate': end_date_str,
                                'units': 'standard',
                                'limit': 1000,
                                'offset': offset,
                            }

                            response = s.get(
                                wbase, headers=headers, params=params, timeout=timeout
                            )

                            if response.status_code != 200:
                                if (
                                    sys.platform == "emscripten"
                                    and response.status_code == 0
                                ):
                                    error_message = (
                                        "Browser blocked the request before receiving "
                                        "an HTTP response (usually a CORS restriction)."
                                    )
                                else:
                                    error_message = f"HTTP {response.status_code}"
                                if (
                                    offset == 1
                                ):  # Only print error on first request for this chunk
                                    print(
                                        f"weather data failed for {wstation} ({dtype}, "
                                        f"{start_date_str} to {end_date_str}): "
                                        f"{error_message}"
                                    )
                                if (
                                    sys.platform == "emscripten"
                                    and response.status_code == 0
                                ):
                                    raise RuntimeError(error_message)
                                break

                            try:
                                data_json = response.json()
                                if (
                                    'results' not in data_json
                                    or len(data_json['results']) == 0
                                ):
                                    # No more results for this chunk
                                    break

                                results = data_json['results']
                                all_results.extend(results)

                                # If we got fewer than 1000 results, we've reached the end of this chunk
                                if len(results) < 1000:
                                    break

                                # Move to next page
                                offset += len(results)
                                request_count += 1
                                time.sleep(
                                    sleep_seconds * 0.5
                                )  # Shorter sleep within chunk pagination

                            except Exception as parse_error:
                                print(
                                    f"weather data parsing failed for {wstation} ({dtype}): {repr(parse_error)}"
                                )
                                break

                        # Move to next chunk
                        current_chunk_start = current_chunk_end + datetime.timedelta(
                            days=1
                        )
                        time.sleep(sleep_seconds)  # Respect rate limits between chunks

                    # Convert all results to DataFrame
                    if len(all_results) > 0:
                        dtype_df = pd.DataFrame(all_results)
                        dtype_df['date'] = pd.to_datetime(dtype_df['date'])
                        dtype_df = dtype_df.set_index('date')[['value']].rename(
                            columns={'value': f'{wstation}_{dtype}'}
                        )
                        if dtype.upper() == "PRCP":
                            precip_col = f'{wstation}_{dtype}'
                            binary_name = f'{wstation}_PRCP_binary'
                            dtype_df[binary_name] = (
                                dtype_df[precip_col].gt(0).astype(int)
                            )
                        station_data.append(dtype_df)

                # Combine all data types for this station
                if len(station_data) > 0:
                    from functools import reduce

                    wdf = reduce(
                        lambda x, y: pd.merge(
                            x, y, left_index=True, right_index=True, how="outer"
                        ),
                        station_data,
                    )
                    dataset_lists.append(wdf)

            except Exception as e:
                error_message = _format_live_download_error(e)
                print(f"weather data failed for {wstation}: {error_message}")
                _weather_errs.append(f"{wstation}: {error_message}")
                if "usually a CORS restriction" in error_message:
                    break
        if noaa_cdo_token is None:
            _weather_errs.append("noaa_cdo_token required")
        _finish_source(
            "NOAA weather",
            _blk,
            error="; ".join(_weather_errs) if _weather_errs else None,
        )

    str_end_time = current_date.strftime("%d-%b-%Y")
    start_date = (current_date - datetime.timedelta(days=london_air_days)).strftime(
        "%d-%b-%Y"
    )
    if london_air_stations is not None:
        _blk = _start_source("London Air")
        _london_errs = []
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
                _london_errs.append(f"{asite}: {repr(e)}")
        _finish_source(
            "London Air",
            _blk,
            error="; ".join(_london_errs) if _london_errs else None,
        )

    if earthquake_min_magnitude is not None:
        _blk = _start_source("Earthquakes")
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
            _finish_source("Earthquakes", _blk)
        except Exception as e:
            print(f"earthquake data failed: {repr(e)}")
            _finish_source("Earthquakes", _blk, error=repr(e))

    if nasa_api_key is not None:
        _blk = _start_source("NASA space weather")
        try:
            nasa_key = nasa_api_key if nasa_api_key else "DEMO_KEY"
            # convert observation window to NASA-compatible YYYY-MM-DD strings
            nasa_start_dt = pd.to_datetime(observation_start, errors="coerce")
            nasa_end_dt = pd.to_datetime(current_date, errors="coerce")
            if pd.isna(nasa_start_dt) or pd.isna(nasa_end_dt):
                raise ValueError(
                    "Unable to resolve observation window for NASA DONKI request."
                )
            nasa_start_str = nasa_start_dt.strftime("%Y-%m-%d")
            nasa_end_str = nasa_end_dt.strftime("%Y-%m-%d")
            nasa_end_day = nasa_end_dt.normalize()

            def _extract_datetime(record, candidate_keys):
                """Return first valid timestamp from record using candidate keys."""
                if not isinstance(record, dict):
                    return None
                for key in candidate_keys:
                    value = record.get(key)
                    if not value:
                        continue
                    ts = pd.to_datetime(value, errors="coerce")
                    if pd.isna(ts):
                        continue
                    if isinstance(ts, pd.Timestamp):
                        return ts.tz_localize(None)
                return None

            def _fill_after_first_valid(series, end_day, fill_value=0):
                """Fill missing values to zero once data starts, leaving leading NaNs."""
                if series is None or series.empty:
                    return series
                first_valid = series.first_valid_index()
                if first_valid is None:
                    return series
                end_day = pd.Timestamp(end_day).normalize()
                if first_valid > end_day:
                    return series
                new_index = pd.date_range(first_valid, end_day, freq="D")
                filled = series.reindex(new_index).fillna(fill_value)
                filled.index.name = series.index.name
                return filled

            base_url = "https://api.nasa.gov/DONKI/"
            common_params = {
                "startDate": nasa_start_str,
                "endDate": nasa_end_str,
                "api_key": nasa_key,
            }

            # Daily Solar Flare Count
            response = s.get(base_url + "FLR", params=common_params, timeout=timeout)
            if response.status_code == 200:
                flare_json = response.json()
                flare_dates = []
                if isinstance(flare_json, list):
                    for item in flare_json:
                        flare_dt = _extract_datetime(
                            item, ["beginTime", "peakTime", "endTime"]
                        )
                        if flare_dt is not None:
                            flare_dates.append(flare_dt.normalize())
                    if flare_dates:
                        flare_series = (
                            pd.Series(
                                1,
                                index=pd.DatetimeIndex(flare_dates, name="datetime"),
                            )
                            .groupby(level=0)
                            .sum()
                            .rename("nasa_solar_flare_count")
                        )
                        flare_series = _fill_after_first_valid(
                            flare_series, nasa_end_day, fill_value=0
                        )
                        dataset_lists.append(flare_series.to_frame())
            else:
                print(f"NASA FLR request failed with status {response.status_code}")
            time.sleep(sleep_seconds)

            # Daily Geomagnetic Storm Maximum Intensity (Kp Index)
            response = s.get(base_url + "GST", params=common_params, timeout=timeout)
            if response.status_code == 200:
                gst_json = response.json()
                kp_records = []
                if isinstance(gst_json, list):
                    for item in gst_json:
                        kp_entries = item.get("allKpIndex") or []
                        if not isinstance(kp_entries, list):
                            kp_entries = []
                        if not kp_entries and item.get("kpIndex") is not None:
                            kp_entries = [
                                {
                                    "kpIndex": item.get("kpIndex"),
                                    "observedTime": item.get("startTime"),
                                }
                            ]
                        for kp_entry in kp_entries:
                            kp_val = kp_entry.get("kpIndex")
                            obs_time = (
                                kp_entry.get("observedTime")
                                or kp_entry.get("sourceTime")
                                or item.get("startTime")
                            )
                            if kp_val is None or obs_time is None:
                                continue
                            try:
                                kp_float = float(kp_val)
                            except Exception:
                                continue
                            obs_dt = pd.to_datetime(obs_time, errors="coerce")
                            if pd.isna(obs_dt):
                                continue
                            obs_dt = obs_dt.tz_localize(None).normalize()
                            kp_records.append((obs_dt, kp_float))
                    if kp_records:
                        kp_df = pd.DataFrame(kp_records, columns=["datetime", "kp"])
                        kp_series = (
                            kp_df.groupby("datetime")["kp"]
                            .max()
                            .rename("nasa_geomagnetic_kp_max")
                        )
                        kp_series = _fill_after_first_valid(
                            kp_series, nasa_end_day, fill_value=0
                        )
                        dataset_lists.append(kp_series.to_frame())
            else:
                print(f"NASA GST request failed with status {response.status_code}")
            time.sleep(sleep_seconds)

            # Daily Coronal Mass Ejection Count
            response = s.get(base_url + "CME", params=common_params, timeout=timeout)
            if response.status_code == 200:
                cme_json = response.json()
                cme_dates = []
                if isinstance(cme_json, list):
                    for item in cme_json:
                        cme_dt = _extract_datetime(item, ["startTime"])
                        if cme_dt is None:
                            analyses = item.get("cmeAnalyses") or []
                            if isinstance(analyses, list):
                                for analysis in analyses:
                                    cme_dt = _extract_datetime(
                                        analysis, ["time21_5", "time18_5"]
                                    )
                                    if cme_dt is not None:
                                        break
                        if cme_dt is not None:
                            cme_dates.append(cme_dt.normalize())
                    if cme_dates:
                        cme_series = (
                            pd.Series(
                                1,
                                index=pd.DatetimeIndex(cme_dates, name="datetime"),
                            )
                            .groupby(level=0)
                            .sum()
                            .rename("nasa_coronal_mass_ejection_count")
                        )
                        cme_series = _fill_after_first_valid(
                            cme_series, nasa_end_day, fill_value=0
                        )
                        dataset_lists.append(cme_series.to_frame())
            else:
                print(f"NASA CME request failed with status {response.status_code}")
            time.sleep(sleep_seconds)
            _finish_source("NASA space weather", _blk)
        except Exception as e:
            print(f"NASA DONKI download failed with error {repr(e)}")
            _finish_source("NASA space weather", _blk, error=repr(e))

    if gov_domain_list is not None:
        _blk = _start_source("Gov web analytics")
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
                if sys.platform == "emscripten" and not data.text.strip():
                    raise RuntimeError(
                        "Browser blocked the response body (received an empty "
                        "response despite a successful status, usually a CORS "
                        "restriction)."
                    )
                gdf = pd.read_json(io.StringIO(data.text), orient="records")
                gdf['date'] = pd.to_datetime(gdf['date'])
                # essentially duplicates brought by agency and null agency
                gresult = gdf.groupby('date')['visits'].first()
                gresult.name = domain
                dataset_lists.append(gresult.to_frame())
                time.sleep(sleep_seconds)
            _finish_source("Gov web analytics", _blk)
        except Exception as e:
            print(f"analytics.gov data failed with {repr(e)}")
            _finish_source("Gov web analytics", _blk, error=repr(e))

    if wikipedia_pages is not None:
        _blk = _start_source("Wikipedia")
        _wiki_errs = []
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
                _wiki_errs.append(f"{page}: {repr(e)}")
                time.sleep(10)
        _finish_source(
            "Wikipedia", _blk, error="; ".join(_wiki_errs) if _wiki_errs else None
        )

    if weather_event_types is not None:
        _blk = _start_source("Severe weather events")
        try:
            for event_type in weather_event_types:
                # appears to have a fixed max of 500 records
                url = f"https://www.ncdc.noaa.gov/stormevents/csv?eventType={event_type}&beginDate_mm=01&beginDate_dd=01&beginDate_yyyy=2000&endDate_mm=09&endDate_dd=30&endDate_yyyy=9999&hailfilter=0.00&tornfilter=2&windfilter=000&sort=DN&statefips=-999%2CALL"
                csv_in = io.StringIO(s.get(url, timeout=timeout).text)
                try:
                    # on_bad_lines added in pandas 1.3; error_bad_lines removed in 2.0.
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
            _finish_source("Severe weather events", _blk)
        except Exception as e:
            print(f"Severe Weather data failed with {repr(e)}")
            _finish_source("Severe weather events", _blk, error=repr(e))

    if trends_list is not None:
        _blk = _start_source("Google Trends")
        try:
            from pytrends.request import TrendReq

            pytrends = TrendReq(hl="en-US", tz=360)
            pytrends.build_payload(trends_list, geo=trends_geo)
            # pytrends.build_payload(trends_list, timeframe="all")  # 'today 12-m'
            gtrends = pytrends.interest_over_time()
            gtrends.index = gtrends.index.tz_localize(None)
            gtrends.drop(columns="isPartial", inplace=True, errors="ignore")
            dataset_lists.append(gtrends)
            _finish_source("Google Trends", _blk)
        except ImportError:
            print("You need to: pip install pytrends")
            _finish_source("Google Trends", _blk, error="pytrends not installed")
        except Exception as e:
            print(f"pytrends data failed: {repr(e)}")
            _finish_source("Google Trends", _blk, error=repr(e))

    if caiso_query is not None:
        _blk = _start_source("CAISO")
        try:
            # CAISO's OASIS API rejects date ranges over ~28 days with a
            # "Data can be requested for period of 31 days only" error,
            # returned as a 200 OK zip containing an XML error doc (which
            # pd.read_csv silently mis-parses as a 1-row CSV). 27-day chunks
            # (+1 day overlap, as before) stay under that limit.
            chunk_days = 27
            n_chunks = math.ceil((364 * weather_years) / chunk_days)
            energy_df = []
            for x in range(n_chunks):
                try:
                    end_nospace = (
                        current_date - datetime.timedelta(days=chunk_days * x)
                    ).strftime("%Y%m%d")
                    start_nospace = (
                        current_date
                        - datetime.timedelta(days=chunk_days * (x + 1) + 1)
                    ).strftime("%Y%m%d")
                    caiso_url = f"http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname={caiso_query}&version=1&market_run_id=RTM&tac_zone_name=ALL&schedule=Generation&startdatetime={start_nospace}T00:00-0000&enddatetime={end_nospace}T23:00-0000"
                    data = pd.read_csv(caiso_url, compression='zip')
                    if 'OPR_DT' not in data.columns:
                        # CAISO returned an XML error/disclaimer doc instead
                        # of CSV (e.g. date range rejected); pandas parsed it
                        # as a 1-row CSV with the XML declaration as header.
                        raise RuntimeError(
                            f"CAISO returned an unexpected response: {data.columns[0]}"
                        )
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
                    error_message = _format_live_download_error(e)
                    print(f"caiso download failed with error: {error_message}")
                    if sys.platform == "emscripten":
                        # CAISO's OASIS API has no CORS headers, so every
                        # remaining chunk would fail the same way; don't burn
                        # through the rest of the retry/sleep loop for nothing.
                        break
                    time.sleep(sleep_seconds)
            if energy_df:
                energy_df = pd.concat(energy_df).sort_index()
                energy_df = energy_df[~energy_df.index.duplicated(keep='last')]
                dataset_lists.append(energy_df)
                _finish_source("CAISO", _blk)
            else:
                _finish_source("CAISO", _blk, error="no data returned")
        except Exception as e:
            print(f"caiso download failed with error: {repr(e)}")
            _finish_source("CAISO", _blk, error=repr(e))

    if eia_key is not None and eia_respondents is not None:
        _blk = _start_source("EIA electricity")
        _eia_errs = []
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
                    params=[("api_key", eia_key)] + _eia_query_params(params),
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
                _eia_errs.append(f"{respond} (demand): {repr(e)}")
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
                    params=[("api_key", eia_key)] + _eia_query_params(params),
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
                _eia_errs.append(f"{respond} (fuel mix): {repr(e)}")
        _finish_source(
            "EIA electricity",
            _blk,
            error="; ".join(_eia_errs) if _eia_errs else None,
        )

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
    # Some APIs return a named column containing only missing values. Such a
    # column is not a usable time series and causes downstream feature
    # detection to emit nanvar warnings. Also enforce the caller's shared date
    # window after sources with coarse history controls have over-fetched.
    df = df[
        (df.index >= observation_start_timestamp)
        & (df.index <= current_date)
    ]
    df = df.dropna(axis=1, how="all")
    if df.shape[0] < 1 or df.shape[1] < 1:
        raise ValueError("No data successfully downloaded!")
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
