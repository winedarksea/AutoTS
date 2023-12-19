"""Informal testing script."""
from time import sleep
import json
import timeit
import os
import platform
import pandas as pd
from autots.datasets import (  # noqa
    load_daily,
    load_hourly,
    load_monthly,
    load_yearly,
    load_weekly,
    load_weekdays,
    load_zeroes,
    load_linear,
    load_sine,
    load_artificial,
)
from autots import AutoTS, create_regressor, model_forecast, __version__  # noqa
from autots.models.base import plot_distributions
import matplotlib.pyplot as plt

print(f"AutoTS version: {__version__}")
# raise ValueError("aaargh!")
use_template = True
save_template = True
force_univariate = False  # long = False
back_forecast = False
graph = True
template_import_method = "addon"  # "only" "addon"
models_to_validate = 0.25  # 0.99 to validate every tried (use with template import)

# this is the template file imported:
template_filename = "template_" + str(platform.node()) + ".csv"
template_filename = "template_categories_1.csv"
name = template_filename.replace('.csv', '').replace("autots_forecast_template_", "")
random_seed = 2023
forecast_length = 90
long = False
# df = load_linear(long=long, shape=(400, 1000), introduce_nan=None)
# df = load_sine(long=long, shape=(400, 1000), start_date="2021-01-01", introduce_random=100).iloc[:, 2:]
# df = load_artificial(long=long, date_start="2018-01-01")
df = load_daily(long=long)
# df.iloc[5, :] = np.nan
interest_series = [
    'wiki_all',
    'wiki_William_Shakespeare',
    'wiki_Periodic_table',
    'wiki_Thanksgiving',
]
if not long and interest_series[0] not in df.columns:
    interest_series = [
        'arima220_outliers',
        'lumpy',
        'out-of-stock',
        "sine_seasonality_monthweek",
        "intermittent_weekly",
        "arima017",
        "old_to_new",
    ]
prediction_interval = 0.9
n_jobs = "auto"
verbose = 2
validation_method = "backwards"  # "similarity"
frequency = "infer"
drop_most_recent = 0
generations = 100
generation_timeout = 5
num_validations = 2  # "auto"
initial_template = "Random"  # "General+Random" 
if use_template:
    initial_training = not os.path.exists(template_filename)
    if initial_training:
        print("No existing template found.")
    else:
        print("Existing template found.")

if force_univariate:
    df = df.iloc[:, 0]

transformer_list = "fast"  # "fast", "all", "superfast"
# transformer_list = ["SeasonalDifference", "Slice", "EWMAFilter", 'MinMaxScaler', "AlignLastValue", "RegressionFilter", "ClipOutliers", "QuantileTransformer", "LevelShiftTransformer", 'AlignLastDiff']
transformer_max_depth = 4
models_mode = "default"  # "default", "regressor", "neuralnets", "gradient_boosting"
model_list = "superfast"
# model_list = "fast"  # fast_parallel, all, fast
# model_list = ["BallTreeMultivariateMotif", "WindowRegression", 'SeasonalityMotif', 'SeasonalNaive']
# model_list = ['PreprocessingRegression', 'MultivariateRegression', 'DatepartRegression', 'WindowRegression']

# only saving with superfast
if model_list == "superfast" and save_template:
    save_template = True
else:
    save_template = False

preclean = None
{
    "fillna": None,
    "transformations": {"0": "LocalLinearTrend"},
    "transformation_params": {
        "0": {
            'rolling_window': 30,
             'n_tails': 0.1,
             'n_future': 0.2,
             'method': 'mean',
             'macro_micro': True
         },
    },
}
ensemble = [
    # "simple",
    # 'mlensemble',
    'horizontal-max',
    # "mosaic-window",
    # 'mosaic-crosshair',
]  # "dist", "subsample", "mosaic-window", "horizontal"
# ensemble = None
metric_weighting = {
    'smape_weighting': 5,
    'mae_weighting': 2,
    'rmse_weighting': 1,
    'made_weighting': 1,
    'mage_weighting': 0,
    'mate_weighting': 1,
    'mle_weighting': 0,  # avoid underestimate
    'imle_weighting': 0,  # avoid overestimate
    'spl_weighting': 3,
    'containment_weighting': 0.1,
    'contour_weighting': 0,
    'runtime_weighting': 0.05,
    'maxe_weighting': 0,
    'oda_weighting': 0,
    'mqae_weighting': 0,
    'uwmse_weighting': 1,
    'wasserstein_weighting': 0,
    'dwd_weighting': 1,
    'smoothness_weighting': -0.5,
}

# metric_weighting = {'ewmae_weighting': 1}
constraint = {
    "constraint_method": "quantile",
    "constraint_regularization": 0.9,
    "upper_constraint": 0.9,
    "lower_constraint": 0.1,
    "bounds": True,
}
if not long:
    if isinstance(df, pd.Series):
        cols = [df.name]
    else:
        cols = df.columns
    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_length + 1, freq=df.index.freq)[1:]
    # sets an extremely high value for the cap, one that should never actually be reached by the data normally
    upper_constraint = pd.DataFrame(9999999999, index=forecast_index, columns=cols)
    # in this case also assuming negatives won't happen so setting a lower constraint of 0
    lower_constraint = pd.DataFrame(0, index=forecast_index, columns=cols)
    # add in your dates you want as definitely 0
    upper_constraint.loc["2022-10-31"] = 0
upper_constraint = 0
lower_constraint = 0
constraint = {
    "constraint_method": "absolute",
    "upper_constraint": upper_constraint,
    "lower_constraint": lower_constraint,
    "bounds": True,
}
constraint = {
    "constraint_method": "stdev_min",
    "upper_constraint": 2.0,
    "lower_constraint": 2.0,
    "bounds": True,
}
constraint = None

model = AutoTS(
    forecast_length=forecast_length,
    frequency=frequency,
    prediction_interval=prediction_interval,
    ensemble=ensemble,
    constraint=constraint,
    max_generations=generations,
    generation_timeout=generation_timeout,
    num_validations=num_validations,
    validation_method=validation_method,
    model_list=model_list,
    transformer_list=transformer_list,
    transformer_max_depth=transformer_max_depth,
    initial_template=initial_template,
    metric_weighting=metric_weighting,
    models_to_validate=models_to_validate,
    max_per_model_class=None,
    model_interrupt=True,
    n_jobs=n_jobs,
    drop_most_recent=drop_most_recent,
    introduce_na=None,
    preclean=preclean,
    # prefill_na=0,
    # subset=2,
    no_negatives=True,
    verbose=verbose,
    models_mode=models_mode,
    random_seed=random_seed,
    # current_model_file=f"current_model_{name}",
)

if not long:
    regr_train, regr_fcst = create_regressor(
        df,
        forecast_length=forecast_length,
        frequency=frequency,
        drop_most_recent=drop_most_recent,
        scale=True,
        summarize="auto",
        backfill="bfill",
        fill_na="pchip",
        holiday_countries=["US"],
        datepart_method="recurring",
        preprocessing_params={
            "fillna": None,
            "transformations": {"0": "LocalLinearTrend"},
            "transformation_params": {
                "0": {
                    'rolling_window': 30,
                     'n_tails': 0.1,
                     'n_future': 0.2,
                     'method': 'mean',
                     'macro_micro': True
                 },
            },
        },
    )
else:
    regr_train = None
    regr_fcst = None

# model = model.import_results('test.pickle')
if use_template:
    if os.path.exists(template_filename):
        model = model.import_template(
            template_filename, method=template_import_method,
            enforce_model_list=False, force_validation=True,
        )
    file2 = "/Users/colincatlin/Downloads/test_import.csv"
    if os.path.exists(file2):
        model = model.import_template(
            file2, method=template_import_method, enforce_model_list=False, force_validation=True,
        )

start_time_for = timeit.default_timer()
model = model.fit(
    df,
    future_regressor=regr_train,
    # weights="inverse_mean",
    # result_file='test.pickle',
    # validation_indexes=[pd.date_range("2001-01-01", "2022-05-02"), pd.date_range("2021-01-01", "2022-02-02"), pd.date_range("2021-01-01", "2022-03-03")],
    date_col="datetime" if long else None,
    value_col="value" if long else None,
    id_col="series_id" if long else None,
)

if save_template:
    model.export_template(
        template_filename,
        models="best",
        n=20,
        max_per_model_class=5,
        include_results=True,
    )
    if False:
        model.export_template(
            "slowest_models_template.csv",
            models="slowest",
            n=10,
            include_results=True,
        )

elapsed_for = timeit.default_timer() - start_time_for

prediction = model.predict(
    future_regressor=regr_fcst, verbose=1, fail_on_forecast_nan=True
)
print(prediction.long_form_results().sample(5))
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
initial_results = model.results()
# validation results
validation_results = model.results("validation")
if long:
    cols = model.df_wide_numeric.columns.tolist()

sleep(5)
print(model)
print(model.validation_test_indexes)
print(f"Model failure rate is {model.failure_rate() * 100:.1f}%")
print(f'The following model types failed completely {model.list_failed_model_types()}')
print("Slowest models:")
runtimes = initial_results[initial_results["Ensemble"] < 1].groupby("Model").agg({
    "TotalRuntimeSeconds": ["mean", "max"],
    "smape": ["median", "min"]
}).rename(columns={
    "median": "median_smape", "min": "min_smape"
})
print(runtimes["TotalRuntimeSeconds"].rename(columns={"mean": "slowest_avg_runtime", "max": "slowest_max_runtime"}).idxmax())
print(runtimes['smape'].idxmin())

### Failure Rate per Transformer type (ignoring ensembles), failure may be due to other model or transformer
failures = []
successes = []
for idx, row in initial_results.iterrows():
    failed = not pd.isnull(row['Exceptions'])
    transforms = list(json.loads(row['TransformationParameters']).get('transformations', {}).values())
    if failed:
        failures = failures + transforms
    else:
        successes = successes + transforms
total = pd.concat([pd.Series(failures).value_counts().rename("failures").to_frame(),pd.Series(successes).value_counts().rename("successes")], axis=1).fillna(0)
total['failure_rate'] = total['failures'] / (total['successes'] + total['failures'])
total.sort_values("failure_rate", ascending=False)['failure_rate'].iloc[0:20].plot(kind='bar', title='Transformers by Failure Rate', color='forestgreen')
plt.show()

if graph:
    start_date = "auto"
    # issues with long and preclean vary 'raw' df choice
    use_df = df if not long else model.df_wide_numeric
    prediction.plot(
        use_df,
        series=cols[0],
        remove_zeroes=False,
        start_date=start_date,
    )
    # plt.savefig("single_forecast2.png", dpi=300, bbox_inches="tight")
    plt.show()
    prediction.plot_grid(use_df, start_date=start_date)
    # plt.savefig("forecast_grid2.png", dpi=300, bbox_inches="tight")
    plt.show()
    scores = model.best_model_per_series_mape().index.tolist()
    scores = [x for x in scores if x in use_df]
    worst = scores[0:6]
    prediction.plot_grid(use_df, start_date=start_date, title="Forecasts of Highest (Worst) Historical MAPE Series", cols=worst)
    plt.show()
    best = scores[-6:]
    prediction.plot_grid(use_df, start_date=start_date, title="Forecasts of Lowest (Best) Historical MAPE Series", cols=best)
    plt.show()
    model.plot_generation_loss()
    plt.show()
    # plt.savefig("improvement_over_generations.png", dpi=300, bbox_inches="tight")

    model.plot_per_series_mape(kind="pie")
    plt.show()

    model.plot_per_series_error()
    plt.show()

    if model.best_model_ensemble == 2:
        model.plot_horizontal_model_count()
        plt.show()

        if back_forecast:
            try:
                model.plot_horizontal_per_generation()
                plt.show()
            except Exception as e:
                print(f"plot horizontal per generation failed with: {repr(e)}")

        plt.show()
        model.plot_horizontal_transformers(method="fillna")
        plt.show()
        model.plot_horizontal_transformers()
        plt.show()
        model.plot_horizontal()
        # plt.savefig(f"horizontal_{name}.png", dpi=300)
        # plt.show()
        if "mosaic" in model.best_model["ModelParameters"].iloc[0].lower():
            mosaic_df = model.mosaic_to_df()
            print(mosaic_df[mosaic_df.columns[0:5]].head(5))

        try:
            prediction.plot_ensemble_runtimes()
            plt.show()
        except Exception as e:
            print(repr(e))

    plt.show()
    if back_forecast:
        model.plot_backforecast(n_splits="auto", start_date="2019-01-01")

    ax = model.plot_validations(use_df, subset='Worst', compare_horizontal=True, include_bounds=False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    ax = model.plot_validations(use_df)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # plt.savefig("validation_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    ax = model.plot_validations(use_df, subset='Best')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # plt.savefig("validation_plot2.png", dpi=300, bbox_inches="tight")
    plt.show()

    ax = model.plot_validations(use_df, subset='Worst')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    ax = model.plot_validations(use_df, subset='Best Score')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    ax = model.plot_validations(use_df, subset='Worst Score')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    val_df = model.retrieve_validation_forecasts()

    try:
        # seaborn plots
        model.plot_metric_corr()
        plt.show()

        f_res = initial_results[(initial_results['Exceptions'].isnull()) & (initial_results["Ensemble"] == 0)]
        plot_distributions(f_res, group_col='Model', y_col='TotalRuntimeSeconds', xlim=0, xlim_right=0.98)
        plt.show()
        # model.metric_corr.loc['wasserstein'].sort_values()
    except Exception as e:
        print(repr(e))

    if True:
        param_impacts_runtime = model.diagnose_params(target="runtime")
        param_impacts_mae = model.diagnose_params(target="mae")
        param_impacts_exception = model.diagnose_params(target="exception")
        param_impacts_smape = model.diagnose_params(target="smape")
        param_impacts = pd.concat([param_impacts_runtime, param_impacts_mae, param_impacts_smape, param_impacts_exception], axis=1).reset_index(drop=False)

df_wide_numeric = model.df_wide_numeric


from autots.models.base import extract_single_transformer
print("Transformers used: " + extract_single_transformer(
    series=df.columns[-1], model_name=model.best_model_name,
    model_parameters=model.best_model_params,
    transformation_params=model.best_model_transformation_params,
))

if not [x for x in interest_series if x in model.df_wide_numeric.columns.tolist()]:
    interest_series = model.df_wide_numeric.columns.tolist()[0:5]
if model.best_model["Ensemble"].iloc[0] == 2:
    interest_models = []
    for x, y in model.best_model_params['series'].items():
        if x in interest_series:
            if isinstance(y, str):
                interest_models.append(y)
            else:
                interest_models.extend(list(y.values()))
            if graph:
                prediction.plot(
                    use_df,
                    series=x,
                    remove_zeroes=False,
                    start_date=start_date,
                )
    interest_models = pd.Series(interest_models).value_counts().head(10)
    print(interest_models)
    print(
        [
            y
            for x, y in model.best_model_params['models'].items()
            if x in interest_models.index.to_list()
        ]
    )
else:
    for x in interest_series:
        if graph:
            prediction.plot(
                use_df,
                series=x,
                remove_zeroes=False,
                start_date=start_date,
                figsize=(16,12),
            )

print("test run complete")

"""
forecasts = model_forecast(
    model_name="UnivariateMotif",
    model_param_dict={'window': 10, "pointed_method":"weighted_mean", "distance_metric": "cosine", "k": 10, "return_result_windows": True},
    model_transform_dict={
        'fillna': 'rolling_mean',
        'transformations': {'0': 'MinMaxScaler', "1": "PCA"},
        'transformation_params': {'0': {}, '1': {"whiten": True}}
    },
    df_train=model.df_wide_numeric,
    forecast_length=forecast_length,
    frequency='infer',
    prediction_interval=prediction_interval,
    no_negatives=False,
    # future_regressor_train=future_regressor_train2d,
    # future_regressor_forecast=future_regressor_forecast2d,
    random_seed=321,
    verbose=1,
    n_jobs="auto",
    return_model=True,
)
result = forecasts.forecast.head(5)
print(result)
print(forecasts.upper_forecast.head(5))
print(forecasts.lower_forecast.head(5))
result_windows = forecasts.model.result_windows
"""

"""
# default save location of files is apparently root
systemd-run --unit=background_cmd_service --remain-after-exit /home/colin/miniconda3/envs/openblas/bin/python /home/colin/AutoTS/test.py
systemd-run --unit=background_cmd_service --remain-after-exit /home/colin/miniconda3/envs/openblas/bin/python /home/colin/AutoTS/local_example.py
journalctl -r -n 10 -u background_cmd_service
journalctl -f -u background_cmd_service
journalctl -b -u background_cmd_service

systemctl stop background_cmd_service
systemctl reset-failed
systemctl kill background_cmd_service

scp colin@192.168.1.122:/home/colin/AutoTS/general_template_colin-1135.csv ./Documents/AutoTS
scp colin@192.168.1.122:/general_template_colin-1135.csv ./Documents/AutoTS


PACKAGE RELEASE
# update version in setup.py, /docs/conf.py, /autots/_init__.py

conda activate env
cd to AutoTS
set PYTHONPATH=%PYTHONPATH%;C:/Users/Colin/Documents/AutoTS
export PYTHONPATH=/users/colincatlin/Documents/AutoTS:$PYTHONPATH

python -m unittest discover ./tests
python -m unittest tests.test_autots.ModelTest.test_models
python -m unittest tests.test_impute.TestImpute.test_impute

pytest tests/ --durations=0

python ./autots/evaluator/benchmark.py > benchmark.txt

cd <project dir>
black ./autots -l 88 -S

mistune==0.8.4 markupsafe==2.0.1 jinja2==2.11.3
https://github.com/sphinx-doc/sphinx/issues/3382
# pip install sphinx==2.4.4
# m2r does not yet work on sphinx 3.0
# pip install m2r2 (replaces old m2r)
cd <project dir>
# delete docs/source and /build (not tutorial or intro.rst)
sphinx-apidoc -f -o docs/source autots
cd ./docs
make html

https://winedarksea.github.io/AutoTS/build/index.html
"""
"""
https://packaging.python.org/tutorials/packaging-projects/

python -m pip install --user --upgrade setuptools wheel
cd /to project directory
# old
python setup.py sdist bdist_wheel
# new
pip install --upgrade build
python -m build
twine upload dist/*
To use this API token:
    Set your username to __token__
    Set your password to the token value, including the pypi- prefix


Merge dev to master on GitHub and create release (include .tar.gz)
Update conda-forge
Update fb third-party (and default)
"""

# Help correlate errors with parameters
"""
test = initial_results[initial_results['TransformationParameters'].str.contains('FastICA')]

cols = ['Model', 'ModelParameters', 'TransformationParameters', 'Exceptions']
if (~initial_results['Exceptions'].isna()).sum() > 0:
    test_corr = error_correlations(
        initial_results[cols], result='corr'
    )  # result='poly corr'

python -m cProfile -o testpy.pstats test.py
gprof2dot -f pstats testpy.pstats | "C:/Program Files (x86)/Graphviz/bin/dot.exe" -Tpng -o test_pstat_output.png
gprof2dot -f pstats testpy.pstats | dot -Tpng -o test_pstat_output.png
"""
