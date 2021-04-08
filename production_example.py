# -*- coding: utf-8 -*-
"""
Uses a number of live public data sources to construct an example production case.
First ~100 lines are just pulling in data.

This is a highly opinionated approach.
Evolve = True allows the timeseries to automatically adapt to changes.
There is a slight risk of it getting caught in suboptimal position however.
It should probably be coupled with some basic data sanity checks.

"""
import os
import datetime
import pandas as pd


forecast_length = 21  # number of days to forecast ahead
fred_key = None  # https://fred.stlouisfed.org/docs/api/api_key.html
initial_training = True  # set this to True on first run, or on reset
evolve = True  # allow time series to progressively evolve on each run, if False, used fixed template
archive_templates = False  # save a copy of the model template with a timestamp
save_location = None  # directory to save templates to. Defaults to working dir

# set max generations based on settings, increase for slower but greater chance of highest accuracy
if initial_training:
    gens = 100
elif evolve:
    gens = 10
else:
    gens = 0

dataset_lists = []

try:
    if fred_key is not None:
        from fredapi import Fred  # noqa
        from autots.datasets.fred import get_fred_data

        fred_series = ["DGS10", "SP500", "T5YIE", "DCOILWTICO", "DEXUSEU"]
        fred_df = get_fred_data(fred_key, fred_series, long=False)
        fred_df.index = fred_df.index.tz_localize(None)
        dataset_lists.append(fred_df)
except ModuleNotFoundError:
    print("pip install fredapi (and you'll also need an api key)")
except Exception as e:
    print(f"FRED data failed: {repr(e)}")

try:
    import yfinance as yf

    ticker = "MSFT"
    msft = yf.Ticker(ticker)
    # get historical market data
    msft_hist = msft.history(period="max")
    msft_hist = msft_hist.rename(columns=lambda x: x.lower().replace(" ", "_"))
    msft_hist = msft_hist.rename(columns=lambda x: ticker.lower() + "_" + x)
    msft_hist.index = msft_hist.index.tz_localize(None)
    dataset_lists.append(msft_hist)
except ModuleNotFoundError:
    print("You need to: pip install yfinance")
except Exception as e:
    print(f"yfinance data failed: {repr(e)}")

try:
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=360)
    kw_list = ["forecasting", "cycling", "cpu", "microsoft"]
    # pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
    pytrends.build_payload(kw_list, timeframe="all")
    gtrends = pytrends.interest_over_time()
    gtrends.index = gtrends.index.tz_localize(None)
    dataset_lists.append(gtrends)
except ModuleNotFoundError:
    print("You need to: pip install pytrends")
except Exception as e:
    print(f"pytrends data failed: {repr(e)}")

try:
    current_date = datetime.datetime.utcnow()
    str_end_time = current_date.strftime("%Y-%m-%d")
    start_date = (current_date - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
    # is limited to 1000 rows of data
    eq = pd.read_csv(
        f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={start_date}&endtime={str_end_time}&minmagnitude=5"
    )
    eq["time"] = pd.to_datetime(eq["time"], infer_datetime_format=True)
    eq["time"] = eq["time"].dt.tz_localize(None)
    eq.set_index("time", inplace=True)
    global_earthquakes = eq.resample("1D").agg({"mag": "mean", "depth": "count"})
    global_earthquakes["mag"] = global_earthquakes["mag"].fillna(5)
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
        lambda x, y: pd.merge(x, y, left_index=True, right_index=True), dataset_lists
    )


start_time = datetime.datetime.now()

# df["datetime"] = pd.to_datetime(df["your_date_column"], infer_datetime_format=True)


from autots import AutoTS

metric_weighting = {
    'smape_weighting': 5,
    'mae_weighting': 0,
    'rmse_weighting': 1,
    'containment_weighting': 0,
    'runtime_weighting': 0,
    'spl_weighting': 1,
    'contour_weighting': 0,
}

model = AutoTS(
    forecast_length=forecast_length,
    frequency="infer",
    prediction_interval=0.8,
    ensemble="simple,distance,horizontal-max",
    model_list="fast_parallel",
    transformer_list="all",
    transformer_max_depth=8,
    max_generations=gens,
    metric_weighting=metric_weighting,
    num_validations=2,
    validation_method="backwards",
    constraint=2,
    # drop_most_recent=2,  # if newest data is incomplete
    n_jobs="auto",
)

template_filename = "autots_forecast_template.csv"
if save_location is not None:
    template_filename = os.path.join(save_location, template_filename)
    

if not initial_training:
    model.import_template(template_filename, method = "only")
model = model.fit(
    df
)

prediction = model.predict()

# Print the details of the best model
print(model)

# point forecasts dataframe
forecasts_df = prediction.forecast

# accuracy of all tried model results
model_results = model.results()

if initial_training or evolve:
    model.export_template(
        template_filename, models="best", n=25, max_per_model_class=5
    )
    if archive_templates:
        arc_file = f"{template_filename.split('.csv')[0]}_{start_time.strftime('%Y%m%d')}.csv"
        model.export_template(
            arc_file, models="best", n=1, max_per_model_class=5
        )
