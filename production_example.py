# -*- coding: utf-8 -*-
"""
Recommended installs: pip install pytrends fredapi yfinance
Uses a number of live public data sources to construct an example production case.

While stock price forecasting is shown here, time series forecasting alone is not a recommended basis for managing investments!

This is a highly opinionated approach.
evolve = True allows the timeseries to automatically adapt to changes.

There is a slight risk of it getting caught in suboptimal position however.
It should probably be coupled with some basic data sanity checks.

cd ./AutoTS
conda activate py38
nohup python production_example.py > /dev/null &
"""
try:  # needs to go first
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception as e:
    print(repr(e))
import json
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # required only for graphs
from autots import AutoTS, load_live_daily, create_regressor

fred_key = None  # https://fred.stlouisfed.org/docs/api/api_key.html
gsa_key = None

forecast_name = "example"
graph = True  # whether to plot graphs
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
frequency = (
    "D"  # "infer" for automatic alignment, but specific offsets are most reliable, 'D' is daily
)
forecast_length = 60  # number of periods to forecast ahead
drop_most_recent = 1  # whether to discard the n most recent records (as incomplete)
num_validations = (
    2  # number of cross validation runs. More is better but slower, usually
)
validation_method = "backwards"  # "similarity", "backwards", "seasonal 364"
n_jobs = "auto"  # or set to number of CPU cores
prediction_interval = (
    0.9  # sets the upper and lower forecast range by probability range. Bigger = wider
)
initial_training = "auto"  # set this to True on first run, or on reset, 'auto' looks for existing template, if found, sets to False.
evolve = True  # allow time series to progressively evolve on each run, if False, uses fixed template
archive_templates = True  # save a copy of the model template used with a timestamp
save_location = None  # "C:/Users/Colin/Downloads"  # directory to save templates to. Defaults to working dir
template_filename = f"autots_forecast_template_{forecast_name}.csv"
forecast_csv_name = None  # f"autots_forecast_{forecast_name}.csv"  # or None, point forecast only is written
model_list = 'fast_parallel'
transformer_list = "fast"  # 'superfast'
transformer_max_depth = 5
models_mode = "default"  # "deep", "regressor"
initial_template = 'random'  # 'random' 'general+random'
preclean = None
{  # preclean option
    "fillna": 'ffill',
    "transformations": {"0": "EWMAFilter"},
    "transformation_params": {
        "0": {"span": 14},
    },
}
back_forecast = False
csv_load = False
start_time = datetime.datetime.now()


if save_location is not None:
    template_filename = os.path.join(save_location, template_filename)
    if forecast_csv_name is not None:
        forecast_csv_name = os.path.join(save_location, forecast_csv_name)

if initial_training == "auto":
    initial_training = not os.path.exists(template_filename)
    if initial_training:
        print("No existing template found.")
    else:
        print("Existing template found.")

# set max generations based on settings, increase for slower but greater chance of highest accuracy
# if include_ensemble is specified in import_templates, ensembles can progressively nest over generations
if initial_training:
    gens = 100
    generation_timeout = 10000  # minutes
    models_to_validate = 0.15
    ensemble = ["horizontal-max", "dist", "simple"]  # , "mosaic", "mosaic-window", 'mlensemble'
elif evolve:
    gens = 500
    generation_timeout = 300  # minutes
    models_to_validate = 0.15
    ensemble = ["horizontal-max"]  # "mosaic", "mosaic-window", "subsample"
else:
    gens = 0
    generation_timeout = 60  # minutes
    models_to_validate = 0.99
    ensemble = ["horizontal-max", "dist", "simple"]  # "mosaic", "mosaic-window",

# only save the very best model if not evolve
if evolve:
    n_export = 50
else:
    n_export = 1  # wouldn't be a bad idea to do > 1, allowing some future adaptability

"""
Begin dataset retrieval
"""
if not csv_load:
    fred_series = [
        "DGS10",
        "T5YIE",
        "SP500",
        "DCOILWTICO",
        "DEXUSEU",
        "BAMLH0A0HYM2",
        "DAAA",
        "DEXUSUK",
        "T10Y2Y",
    ]
    tickers = ["MSFT", "PG"]
    trend_list = ["forecasting", "msft", "p&g"]
    weather_event_types = ["%28Z%29+Winter+Weather", "%28Z%29+Winter+Storm"]
    wikipedia_pages = ['all', 'Microsoft', "Procter_%26_Gamble", "YouTube", "United_States"]
    df = load_live_daily(
        long=False,
        fred_key=fred_key,
        fred_series=fred_series,
        tickers=tickers,
        trends_list=trend_list,
        earthquake_min_magnitude=5,
        weather_years=3,
        london_air_days=700,
        wikipedia_pages=wikipedia_pages,
        gsa_key=gsa_key,
        gov_domain_list=None,  # ['usajobs.gov', 'usps.com', 'weather.gov'],
        gov_domain_limit=700,
        weather_event_types=weather_event_types,
        sleep_seconds=15,
    )
    # be careful of very noisy, large value series mixed into more well-behaved data as they can skew some metrics such that they get most of the attention
    # remove "volume" data as it skews MAE (other solutions are to adjust metric_weighting towards SMAPE, use series `weights`, or pre-scale data)
    df = df[[x for x in df.columns if "_volume" not in x]]
    # remove dividends and stock splits as it skews metrics
    df = df[[x for x in df.columns if "_dividends" not in x]]
    df = df[[x for x in df.columns if "stock_splits" not in x]]
    # scale 'wiki_all' to millions to prevent too much skew of MAE
    if 'wiki_all' in df.columns:
        df['wiki_all_millions'] = df['wiki_all'] / 1000000
        df = df.drop(columns=['wiki_all'])
    
    # manual NaN cleaning where real values are easily approximated, this is the way
    # although if you have 'no good idea' why it is random, auto is best
    # note manual pre-cleaning affects VALIDATION significantly (for better or worse)
    # as NaN times in history are skipped by metrics, but filled values, as added here, are evaluated
    if trend_list is not None:
        for tx in trend_list:
            if tx in df.columns:
                df[tx] = df[tx].interpolate('akima').ffill(limit=30).bfill(limit=30)
    # fill weekends
    if tickers is not None:
        for fx in tickers:
            for suffix in ["_high", "_low", "_open", "_close"]:
                fxs = (fx + suffix).lower()
                if fxs in df.columns:
                    df[fxs] = df[fxs].interpolate('akima')
    if fred_series is not None:
        for fx in fred_series:
            if fx in df.columns:
                df[fx] = df[fx].interpolate('akima')
    if weather_event_types is not None:
        wevnt = [x for x in df.columns if "_Events" in x]
        df[wevnt] = df[wevnt].mask(df[wevnt].notnull().cummax(), df[wevnt].fillna(0))
    # most of the NaN here are just weekends, when financial series aren't collected, ffill of a few steps is fine
    # partial forward fill, no back fill
    df = df.ffill(limit=3)
    
    df = df[df.index.year > 1999]
    # remove any data from the future
    df = df[df.index <= start_time]
    # remove series with no recent data
    df = df.dropna(axis="columns", how="all")
    min_cutoff_date = start_time - datetime.timedelta(days=180)
    most_recent_date = df.notna()[::-1].idxmax()
    drop_cols = most_recent_date[most_recent_date < min_cutoff_date].index.tolist()
    df = df.drop(columns=drop_cols)
    print(
        f"Series with most NaN: {df.head(365).isnull().sum().sort_values(ascending=False).head(5)}"
    )

    # saving this to make it possible to rerun without waiting for download, but remove this in production
    df.to_csv(f"training_data_{forecast_name}.csv")
else:
    df = pd.read_csv(f"training_data_{forecast_name}.csv", index_col=0, parse_dates=[0])

# example future_regressor with some things we can glean from data and datetime index
# note this only accepts `wide` style input dataframes
# and this is optional, not required for the modeling
# also create macro_micro before inclusion
regr_train, regr_fcst = create_regressor(
    df,
    forecast_length=forecast_length,
    frequency=frequency,
    drop_most_recent=drop_most_recent,
    scale=True,
    summarize="auto",
    backfill="bfill",
    fill_na="spline",
    holiday_countries={"US": None},  # requires holidays package
    encode_holiday_type=True,
    # datepart_method="simple_2",
)

# remove the first forecast_length rows (because those are lost in regressor)
df = df.iloc[forecast_length:]
regr_train = regr_train.iloc[forecast_length:]

print("data setup completed, beginning modeling")
"""
Begin modeling
"""

metric_weighting = {
    'smape_weighting': 2,
    'mae_weighting': 2,
    'rmse_weighting': 1.5,
    'made_weighting': 1,
    'mage_weighting': 0,
    'mate_weighting': 0.01,
    'mle_weighting': 0.1,
    'imle_weighting': 0,
    'spl_weighting': 3,
    'dwae_weighting': 1,
    'uwmse_weighting': 1,
    'dwd_weighting': 0.1,
    "oda_weighting": 0.1,
    'runtime_weighting': 0.05,
}

model = AutoTS(
    forecast_length=forecast_length,
    frequency=frequency,
    prediction_interval=prediction_interval,
    ensemble=ensemble,
    model_list=model_list,
    transformer_list=transformer_list,
    transformer_max_depth=transformer_max_depth,
    max_generations=gens,
    metric_weighting=metric_weighting,
    initial_template=initial_template,
    aggfunc="first",
    models_to_validate=models_to_validate,
    model_interrupt=True,
    num_validations=num_validations,
    validation_method=validation_method,
    constraint=None,
    drop_most_recent=drop_most_recent,  # if newest data is incomplete, also remember to increase forecast_length
    preclean=preclean,
    models_mode=models_mode,
    # no_negatives=True,
    # subset=100,
    # prefill_na=0,
    # remove_leading_zeroes=True,
    # current_model_file=f"current_model_{forecast_name}",
    generation_timeout=generation_timeout,
    n_jobs=n_jobs,
    verbose=1,
)

if not initial_training:
    if evolve:
        model.import_template(template_filename, method="addon")
    else:
        # model.import_template(template_filename, method="only")
        model.import_best_model(template_filename)  # include_ensemble=False

if evolve or initial_training:
    model = model.fit(
        df,
        future_regressor=regr_train,
        # weights='mean'
    )
else:
    model.fit_data(df, future_regressor=regr_train)

# save a template of best models
if initial_training or evolve:
    model.export_template(
        template_filename,
        models="best",
        n=n_export,
        max_per_model_class=6,
        include_results=True,
    )
    if archive_templates:
        arc_file = f"{template_filename.split('.csv')[0]}_{start_time.strftime('%Y%m%d%H%M')}.csv"
        model.export_template(arc_file, models="best", n=1)

prediction = model.predict(
    future_regressor=regr_fcst, verbose=2, fail_on_forecast_nan=True
)

# Print the details of the best model
print(model)

"""
Process results
"""

# point forecasts dataframe
forecasts_df = prediction.forecast  # .fillna(0).round(0)
if forecast_csv_name is not None:
    forecasts_df.to_csv(forecast_csv_name)

forecasts_upper_df = prediction.upper_forecast
forecasts_lower_df = prediction.lower_forecast

# accuracy of all tried model results
model_results = model.results()
validation_results = model.results("validation")

print(f"Model failure rate is {model.failure_rate() * 100:.1f}%")
print(f'The following model types failed completely {model.list_failed_model_types()}')
print("Slowest models:")
print(
    model_results[model_results["Ensemble"] < 1]
    .groupby("Model")
    .agg({"TotalRuntimeSeconds": ["mean", "max"]})
    .idxmax()
)

model_parameters = json.loads(model.best_model["ModelParameters"].iloc[0])
 # model.export_template("all_results.csv", models='all')

if graph:
    with plt.style.context("bmh"):
        start_date = 'auto'  # '2021-01-01'

        prediction.plot_grid(model.df_wide_numeric, start_date=start_date)
        plt.show()

        scores = model.best_model_per_series_mape().index.tolist()
        scores = [x for x in scores if x in df.columns]
        worst = scores[0:6]
        prediction.plot_grid(model.df_wide_numeric, start_date=start_date, title="Worst Performing Forecasts", cols=worst)
        plt.show()

        best = scores[-6:]
        prediction.plot_grid(model.df_wide_numeric, start_date=start_date, title="Best Performing Forecasts", cols=best)
        plt.show()

        if model.best_model_name == "Cassandra":
            prediction.model.plot_components(
                prediction, series=None, to_origin_space=True, start_date=start_date
            )
            plt.show()
            prediction.model.plot_trend(
                series=None, start_date=start_date
            )
            plt.show()
    
        ax = model.plot_per_series_mape()
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()
    
        
        if back_forecast:
            model.plot_backforecast()
            plt.show()
        
        ax = model.plot_validations()
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()

        ax = model.plot_validations(subset='best')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()

        ax = model.plot_validations(subset='worst')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()
    
        if model.best_model_ensemble == 2:
            plt.subplots_adjust(bottom=0.5)
            model.plot_horizontal_transformers()
            plt.show()
            model.plot_horizontal_model_count()
            plt.show()
    
            model.plot_horizontal()
            plt.show()
            # plt.savefig("horizontal.png", dpi=300, bbox_inches="tight")
    
            if str(model_parameters["model_name"]).lower() in ["mosaic", "mosaic-window"]:
                mosaic_df = model.mosaic_to_df()
                print(mosaic_df[mosaic_df.columns[0:5]].head(5))

print(f"Completed at system time: {datetime.datetime.now()}")
