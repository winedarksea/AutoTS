# -*- coding: utf-8 -*-
"""
生产实例

推荐安装： pip install pytrends fredapi yfinance
使用许多实时公共数据源构建示例生产案例。

虽然此处显示了股价预测，但单独的时间序列预测并不是管理投资的推荐基础！

这是一种非常固执己见的方法。
evolution = True 允许时间序列自动适应变化。

然而，它存在陷入次优位置的轻微风险。
它可能应该与一些基本的数据健全性检查相结合。

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
graph = True  # 是否绘制图形
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
frequency = (
    "D"  # “infer”用于自动对齐，但特定偏移量最可靠，“D”是每日
)
forecast_length = 60  #  未来预测的周期数
drop_most_recent = 1  #  是否丢弃最近的n条记录（视为不完整）
num_validations = (
    2  # 交叉验证运行次数。 通常越多越好但速度越慢
)
validation_method = "backwards"  # "similarity", "backwards", "seasonal 364"
n_jobs = "auto"  # 或设置为CPU核心数
prediction_interval = (
    0.9  # 通过概率范围设置预测范围的上限和下限。 更大=更宽 Bigger = wider
)
initial_training = "auto"  # 在第一次运行时将其设置为 True，或者在重置时，'auto' 会查找现有模板，如果找到，则设置为 False。
evolve = True  # 允许时间序列在每次运行中逐步演化，如果为 False，则使用固定模板
archive_templates = True  # 保存使用时间戳的模型模板的副本
save_location = None  # "C:/Users/Colin/Downloads"  # 保存模板的目录。 默认为工作目录
template_filename = f"autots_forecast_template_{forecast_name}.csv"
forecast_csv_name = None  # f"autots_forecast_{forecast_name}.csv" 或 None，仅写入点预测
model_list = "scalable"
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

# 根据设置设置最大代数，增加速度会更慢，但获得最高准确度的机会更大
# 如果在 import_templates 中指定了 include_ensemble，则集成可以逐步嵌套几代
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

# 如果不进化，只保存最好的模型
if evolve:
    n_export = 50
else:
    n_export = 1  # > 1 不是一个坏主意，允许一些未来的适应性

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
    # 小心混合到表现更好的数据中的非常嘈杂的大值序列，因为它们可能会扭曲某些指标，从而获得大部分关注
    # 删除 "volume" 数据，因为它会扭曲 MAE（其他解决方案是将 metric_weighting 调整为 SMAPE、使用系列“权重”或预缩放数据）
    df = df[[x for x in df.columns if "_volume" not in x]]
    # remove dividends and stock splits as it skews metrics
    df = df[[x for x in df.columns if "_dividends" not in x]]
    df = df[[x for x in df.columns if "stock_splits" not in x]]
    # 将“wiki_all”扩展到数百万以防止 MAE 出现太大偏差
    if 'wiki_all' in df.columns:
        df['wiki_all_millions'] = df['wiki_all'] / 1000000
        df = df.drop(columns=['wiki_all'])
    
    # 当真实值容易估计时手动清理NaN是一种方法
    # 尽管如果你对为何它是随机的“没有好主意”，自动处理是最好的
    # 注意手动预清理显著影响验证（无论是好是坏）
    # 因为历史中的NaN时间会被度量标准跳过，但在这里添加的填充值会被评估
    if trend_list is not None:
        for tx in trend_list:
            if tx in df.columns:
                df[tx] = df[tx].interpolate('akima').fillna(method='ffill', limit=30).fillna(method='bfill', limit=30)
    # 填补周末
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
    # 这里的大部分NaN只是周末时的，当时金融系列数据没有被收集，向前填充几步是可以的
    # 部分向前填充，不向后填充
    df = df.fillna(method='ffill', limit=3)
    
    df = df[df.index.year > 1999]
    # 移除任何未来的数据
    df = df[df.index <= start_time]
    # 移除最近没有数据的序列
    df = df.dropna(axis="columns", how="all")
    min_cutoff_date = start_time - datetime.timedelta(days=180)
    most_recent_date = df.notna()[::-1].idxmax()
    drop_cols = most_recent_date[most_recent_date < min_cutoff_date].index.tolist()
    df = df.drop(columns=drop_cols)
    print(
        f"Series with most NaN: {df.head(365).isnull().sum().sort_values(ascending=False).head(5)}"
    )

    # 保存这个以便在不需要等待下载的情况下重新运行，但在生产中移除这个
    df.to_csv(f"training_data_{forecast_name}.csv")
else:
    df = pd.read_csv(f"training_data_{forecast_name}.csv", index_col=0, parse_dates=[0])

# future_regressor 示例，其中包含我们可以从数据和日期时间索引中收集的一些内容
# 请注意，这只接受`wide`样式的输入数据帧
# 这是可选的，建模不需要
# 在包含之前也创建 Macro_micro
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

# 删除前一个 Forecast_length 行（因为这些行在回归器中丢失）
df = df.iloc[forecast_length:]
regr_train = regr_train.iloc[forecast_length:]

print("data setup completed, beginning modeling")
"""
Begin modeling
"""

metric_weighting = {
    'smape_weighting': 1,
    'mae_weighting': 3,
    'rmse_weighting': 2,
    'made_weighting': 1,
    'mage_weighting': 0,
    'mate_weighting': 0.01,
    'mle_weighting': 1,
    'imle_weighting': 0,
    'spl_weighting': 5,
    'dwae_weighting': 1,
    'uwmse_weighting': 1,
    'dwd_weighting': 0.1,
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
    drop_most_recent=drop_most_recent,  # 如果最新数据不完整，也要记得增加forecast_length
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

# 打印最佳模型的详细信息
print(model)

"""
Process results
"""

# 点预测 dataframe
forecasts_df = prediction.forecast  # .fillna(0).round(0)
if forecast_csv_name is not None:
    forecasts_df.to_csv(forecast_csv_name)

forecasts_upper_df = prediction.upper_forecast
forecasts_lower_df = prediction.lower_forecast

# 所有尝试的模型结果的准确性
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
