"""Informal testing script."""
from time import sleep
import json
import timeit
import os
import platform
import numpy as np
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
from autots.evaluator.auto_model import generate_score
import matplotlib.pyplot as plt

print(f"AutoTS version: {__version__}")
# raise ValueError("aaargh!")
use_template = True
save_template = True
force_univariate = False  # long = False
back_forecast = False
graph = True
run_param_impacts = False
template_import_method = "addon"  # "only" "addon"
models_to_validate = 0.22  # 0.99 to validate every tried (use with template import)

# this is the template file imported:
template_filename = "template_" + str(platform.node()) + ".csv"
template_filename = "template_categories_1.csv"
name = template_filename.replace('.csv', '').replace("autots_forecast_template_", "")
random_seed = 2025
forecast_length = 90
long = False
# df = load_linear(long=long, shape=(400, 1000), introduce_nan=None)
# df = load_sine(long=long, shape=(400, 1000), start_date="2021-01-01", introduce_random=100).iloc[:, 2:]
# df = load_artificial(long=long, date_start="2018-01-01")
df = load_daily(long=long)
df_p2 = df.tail(forecast_length)
df = df.iloc[:-forecast_length]
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
validation_method = "backwards"  # "backwards"  # "similarity"
frequency = "infer"
drop_most_recent = 0
generations = 10000
generation_timeout = 20
num_validations = 4  # "auto"
initial_template = "General+Random"   # "Random"  # "General+Random" 
if use_template:
    initial_training = not os.path.exists(template_filename)
    if initial_training:
        print("No existing template found.")
    else:
        print("Existing template found.")

if force_univariate:
    df = df.iloc[:, 0]

transformer_list = "no_expanding"  # "fast", "all", "superfast"
# transformer_list = ["SeasonalDifference", 'MinMaxScaler', "AlignLastValue", "ClipOutliers", "QuantileTransformer", "LevelShiftTransformer", 'FIRFilter', 'UpscaleDownscaleTransformer']
transformer_max_depth = 8
models_mode = "default"  # "default", "regressor", "neuralnets", "gradient_boosting"
model_list = "superfast"
# model_list = "fast"  # fast_parallel, all, fast
# model_list = ["BallTreeMultivariateMotif", "WindowRegression", 'SeasonalityMotif', 'SeasonalNaive']
# model_list = ['PreprocessingExperts', 'PreprocessingRegression']
# model_list = ["PreprocessingExperts"]
if "NeuralForecast" in model_list:
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

preclean = None
{
    "fillna": None,
    "transformations": {"0": "RollingMeanTransformer"},
    "transformation_params": {
        "0": {
            'window': 90,
            'fixed': False,
            'center': True,
            'macro_micro': True,
        },
    },
}

# pd.set_option('future.no_silent_downcasting', True)
ensemble = [
    # "simple",
    # 'mlensemble',
    'horizontal-max',
    # "mosaic-window",
    # 'mosaic-crosshair',
    "mosaic-weighted-0-10",
    "mosaic-weighted-profile",
    "mosaic-mae-profile-0-10",  # good
    "mosaic-mae-profile-0-20",  # good
    "mosaic-mae-crosshair-0-10",
    "mosaic-mae-median-0-6",
    "mosaic-mae-median-0-10",
    "mosaic-mae-median-0-20",
    "mosaic-mae-median-0-30",
    "mosaic-mae-median-filtered-0-30",
    "mosiac-mae-0-30",
    "mosaic-weighted-median-0-15",
    "mosaic-weighted-median-0-30",  # popular
    "mosaic-mae-median-profile-0-10",
    # "mosaic-weighted-median",  # not as good on eval loop
    "mosaic-weighted-median-filtered",  # good
    "mosaic-weighted-unpredictability_adjusted-filtered",
    "mosaic-weighted-median-unpredictability_adjusted-filtered",
    "mosaic-weighted-median-unpredictability_adjusted-crosshair_lite-filtered",
    "mosaic-weighted-median-unpredictability_adjusted-crosshair_lite-filtered-horizontal",
    "mosaic-mae-filtered-0-20",
    # "mosaic-mae-unpredictability_adjusted",
    "mosaic-mae-unpredictability_adjusted-0-horizontal",
    "mosaic-weighted-unpredictability_adjusted-0-30",
    "mosaic-spl-unpredictability_adjusted-0-30",
    # "mosaic-weighted-unpredictability_adjusted-median",
    "mosaic-mae-0-horizontal",
    "mosaic-mae-median-0-horizontal",
    "mosaic-mae-median-crosshair_lite-0-30",
    "mosaic-mae-median-crosshair-0-30",
    "mosaic-mae-median-profile-crosshair_lite-horizontal",  # best 11/15
    "mosaic-mae-3-horizontal",
]  # "dist", "subsample", "mosaic-window", "horizontal"
# ensemble = ["simple"]
# ensemble = None

# only saving with superfast
accepted_trans = ["fast", "scalable", "all", "no_expanding"]
superfast_check = model_list == "superfast" and save_template and transformer_list in accepted_trans and preclean is None and ensemble is not None
single_model_check = isinstance(model_list, list) and (len(model_list) == 1) and transformer_list in accepted_trans and preclean is None
if superfast_check or single_model_check:
    save_template = True
else:
    save_template = False

metric_weighting = {
    'smape_weighting': 5,
    'mae_weighting': 2,
    'rmse_weighting': 1,
    'made_weighting': 0.1,
    'mage_weighting': 0,
    'mate_weighting': 0.005,
    'matse_weighting': 0.01,
    'mle_weighting': 0,  # avoid underestimate
    'imle_weighting': 0,  # avoid overestimate
    'spl_weighting': 3,
    'containment_weighting': 0.1,
    'contour_weighting': 0,
    'runtime_weighting': 0.05,
    'maxe_weighting': 0,
    'oda_weighting': 0,
    'mqae_weighting': 0,
    'ewmae_weighting': 0.1,
    'uwmse_weighting': 1,
    'wasserstein_weighting': 0,
    'dwd_weighting': 0.3,
    'smoothness_weighting': -0.1,
}

# metric_weighting = {'ewmae_weighting': 1}

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

constraint = {
    "constraint_method": "stdev_min",
    "constraint_regularization": 0.9,
    "upper_constraint": 2.0,
    "lower_constraint": 2.0,
    "bounds": True,
}
constraint = {"constraints": [
    {
         "constraint_method": "dampening",
         "constraint_value": 0.9,
         "bounds": True,
    },
]}
constraint = None

def custom_metric(A, F, df_train=None, prediction_interval=None):
    submission = F
    objective = A
    abs_err = np.nansum(np.abs(submission - objective))
    err = np.nansum((submission - objective))
    score = abs_err + abs(err)
    epsilon = 1
    big_sum = (
        np.nan_to_num(objective, nan=0.0, posinf=0.0, neginf=0.0).sum().sum()
        + epsilon
    )
    score /= big_sum
    return score

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
    subset=50,
    no_negatives=True,
    verbose=verbose,
    models_mode=models_mode,
    random_seed=random_seed,
    # current_model_file=f"current_model_{name}",
    horizontal_ensemble_validation=True,
    custom_metric=custom_metric,
)
model.traceback = True

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
# model = model.import_results(temp_df)
if use_template:
    if os.path.exists(template_filename):
        model = model.import_template(
            template_filename, method=template_import_method,
            enforce_model_list=True, force_validation=True,
        )
    elif single_model_check:
        file2 = f"/Users/colincatlin/Downloads/{model_list[0]}_reg.csv"
        if os.path.exists(file2):
            model = model.import_template(
                file2, method=template_import_method, enforce_model_list=False, force_validation=True,
            )
            print(f"template for {model_list[0]} imported")
    elif model_list == ['PreprocessingExperts', 'PreprocessingRegression']:
        file2 = "/Users/colincatlin/Downloads/PreprocessingBoth.csv"
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
    if single_model_check:
        file2 = f"/Users/colincatlin/Downloads/{model_list[0]}_reg.csv"
        model.export_template(
            file2,
            models="best", n=10, max_per_model_class=5, include_results=True
        )
    elif model_list == ['PreprocessingExperts', 'PreprocessingRegression']:
        model.export_template(
            "/Users/colincatlin/Downloads/PreprocessingBoth.csv",
            models="best", n=15, max_per_model_class=5, include_results=True
        )
    else:
        model.export_template(
            template_filename,
            models="best",
            n=30,
            max_per_model_class=5,
            include_results=True,
            focus_models=["SeasonalityMotif"],
            min_metrics=['smape', 'mage', 'dwae', 'spl', 'wasserstein', 'dwd', 'rmse', 'ewmae', 'uwmse', 'mqae'],
            max_metrics=['oda'],
        )
    if False:
        model.export_template(
            "slowest_models_template.csv",
            models="slowest",
            n=10,
            include_results=True,
            include_ensemble=False,
        )

elapsed_for = timeit.default_timer() - start_time_for

model.expand_horizontal()

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
# Score Breakdown
model.score_breakdown["sum"] = model.score_breakdown.sum(axis=1)
val_with_score = model.score_breakdown.rename(columns=lambda x: x + "_score").merge(validation_results, left_index=True, right_on='ID')
best_scores = model.score_breakdown[model.score_breakdown.index == model.best_model_id]
which_greater = best_scores > (best_scores.median().median() * 20)
if which_greater.sum().sum() > 0:
    print(f"the following metrics may be out of balance: {best_scores.columns[which_greater.sum() > 0].tolist()}")

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

# minimizing metrics only
# no weights present on metrics
target_metric = "smape"
new_weighting = {
    str(target_metric) + '_weighting': 1,
}
temp_cols = ['ID', 'Model', 'ModelParameters', 'TransformationParameters', 'Ensemble', target_metric]
new_mod = model._return_best_model(new_weighting, template_cols=temp_cols)
new_mod_non = new_mod[1]
new_mod = new_mod[0]
if new_mod['Ensemble'].iloc[0] == 2:
    min_pos = validation_results[validation_results['Ensemble'] == 2][target_metric].min()
    min_pos_non = validation_results[(validation_results['Ensemble'] < 2) & (validation_results['Runs'] > model.num_validations)][target_metric].min()
    chos_pos = new_mod[target_metric].iloc[0]
    print(min_pos)
    print(validation_results[validation_results['Ensemble'] == 2][target_metric].max())
    print(chos_pos)
    print(chos_pos <= min_pos)
    print(np.allclose(chos_pos, min_pos))
    print(np.allclose(new_mod_non[target_metric].iloc[0], min_pos_non))
    print(json.loads(new_mod['ModelParameters'].iloc[0])['model_name'])
    print(json.loads(new_mod['ModelParameters'].iloc[0])['model_metric'])

# test generating a new score
temp = validation_results[validation_results['Runs'] >= num_validations + 1].copy()
new_weighting = {
    'smape_weighting': 4,
    'mae_weighting': 1,
    'mage_weighting': 0.1,
    'spl_weighting': 2,
    'containment_weighting': 0.1,
    'runtime_weighting': 0.001,
    'oda_weighting': 0.1,
    'mqae_weighting': 0.1,
    'uwmse_weighting': 1,
    'wasserstein_weighting': 0.01,
    'dwd_weighting': 1,
}
temp['Score'] = generate_score(temp, metric_weighting=new_weighting)
new_mod = temp.sort_values('Score').iloc[0]
if new_mod['Ensemble'] == 2:
    print(json.loads(new_mod['ModelParameters'])['model_name'])
    print(json.loads(new_mod['ModelParameters'])['model_metric'])
    print(new_mod['smape'])

if graph:
    ### Failure Rate per Transformer type (ignoring ensembles), failure may be due to other model or transformer
    model.plot_failure_rate()
    plt.show()

    model.plot_failure_rate(target="models")
    plt.show()
    # yes, there are duplicates for the same thing...
    model.plot_transformer_failure_rate()
    plt.show()
    model.plot_model_failure_rate()
    plt.show()

    start_date = "auto"
    # issues with long and preclean vary 'raw' df choice
    use_df = pd.concat([df, df_p2]) if not long else model.df_wide_numeric
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
        most_common_models = model.get_top_n_counts(model.best_model_params['series'], n=5)
        print(most_common_models)
        model.get_params_from_id(most_common_models[0][0])

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

            model.plot_mosaic()  # max_rows=df.shape[1])

        try:
            prediction.plot_ensemble_runtimes()
            plt.show()
        except Exception as e:
            print(repr(e))
    else:
        model.plot_chosen_transformer()

    plt.show()
    if back_forecast:
        model.plot_backforecast(n_splits="auto", start_date="2019-01-01")

    # plot a comparison of validation forecasts of several best models by different criteria
    compare_mods = [model.best_model_id, model.best_model_non_horizontal['ID'].iloc[0]]
    spl_min = validation_results[validation_results['Runs'] >= (model.num_validations + 1)].nsmallest(1, columns='spl').iloc[0]['ID']
    compare_mods.append(spl_min)
    mage_min = validation_results[validation_results['Runs'] >= (model.num_validations + 1)].nsmallest(1, columns='mage').iloc[0]['ID']
    compare_mods.append(spl_min)
    ax = model.plot_validations(use_df, models=compare_mods, include_bounds=False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    ax = model.plot_validations(use_df, compare_horizontal=True, include_bounds=False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    ax = model.plot_validations(use_df, subset='Worst', compare_horizontal=True, include_bounds=False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    ax = model.plot_validations(use_df, compare_horizontal=True, include_bounds=False)
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

    ax = model.plot_validations(use_df, subset='agg')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    model.plot_transformer_by_class()
    plt.show()

    model.plot_transformer_by_class(plot_group="Model")
    plt.show()

    model.plot_unpredictability()
    # plt.savefig("uncertainty_plot.png", dpi=300)
    plt.show()
    for column in df.sample(5, axis=1).columns:
        model.plot_unpredictability(series=column)
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

        model.plot_series_corr()
        plt.show()
    except Exception as e:
        print(repr(e))

    if run_param_impacts:
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
    if False:
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

# new_pred = model._predict(model_id="075c17a03f5b4c5f79eca944629bf944")
# new_pred.plot_grid(use_df)
prediction.evaluate(df_p2)
print(prediction.avg_metrics["smape"])

print("test run complete")


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

screen -ls
screen -S python csv_example_dap.py
screen -XS 5900 quit

PACKAGE RELEASE
# update version in setup.py, /docs/conf.py, /autots/_init__.py

conda activate env
cd to AutoTS
set PYTHONPATH=%PYTHONPATH%;C:/Users/Colin/Documents/AutoTS
export PYTHONPATH=/users/colincatlin/Documents/AutoTS:$PYTHONPATH

python -m unittest discover ./tests
python -m unittest tests.test_autots.ModelTest.test_models
python -m unittest tests.test_autots.ModelTest.test_transforms
python -m unittest tests.test_impute.TestImpute.test_impute

pytest tests/ --durations=0

python ./autots/evaluator/benchmark.py > benchmark.txt

cd <project dir>
black ./autots -l 88 -S

mistune==0.8.4 markupsafe==2.0.1 jinja2==2.11.3
https://github.com/sphinx-doc/sphinx/issues/3382
# pip install sphinx==2.4.4
# m2r does not yet work on sphinx 3.0
# pip install m2r2 (replaces old m2r and works on new sphinx)
# pip install sphinxcontrib-googleanalytics
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
