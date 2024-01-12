# AutoTS

<img src="/img/autots_1280.png" width="400" height="184" title="AutoTS 徽标">

AutoTS 是一个 Python 时间序列包，旨在大规模快速部署高精度预测。

2023 年，AutoTS 在 M6 预测竞赛中获胜，在 12 个月的股市预测中提供了最高绩效的投资决策。

有数十种预测模型可用于`sklearn`风格的`.fit()`和`.predict()`。
其中包括朴素、统计、机器学习和深度学习模型。
此外，在`sklearn`风格的`.fit()`、`.transform()`和`.inverse_transform()`中，还有超过 30 种特定于时间序列的变换。
所有这些功能都直接在 Pandas Dataframes 上运行，无需转换为专有对象。

所有模型都支持预测多元（多个时间序列）输出，并且还支持概率（上限/下限）预测。
大多数模型可以轻松扩展到数万甚至数十万个输入系列。
许多模型还支持传入用户定义的外生回归量。

这些模型均设计用于集成到 AutoML 特征搜索中，该搜索可通过遗传算法自动查找给定数据集的最佳模型、预处理和集成。

水平和马赛克风格的合奏是旗舰合奏类型，允许每个系列接收最准确的模型，同时仍然保持可扩展性。

指标和交叉验证选项的组合、应用子集和加权的能力、回归器生成工具、模拟预测模式、事件风险预测、实时数据集、模板导入和导出、绘图以及数据整形参数的集合使 可用的功能集。

## 目录
* [安装](https://github.com/winedarksea/AutoTS#installation)
* [基本使用](https://github.com/winedarksea/AutoTS#basic-use)
* [速度和大数据提示](https://github.com/winedarksea/AutoTS#tips-for-speed-and-large-data)
* 扩展教程 [GitHub](https://github.com/winedarksea/AutoTS/blob/master/extended_tutorial.md) or [Docs](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html)
* [生产示例](https://github.com/winedarksea/AutoTS/blob/master/production_example.py)

## 安装
```
pip install autots
```
这包括基本模型的依赖项，但某些模型和方法需要[附加包](https://github.com/winedarksea/AutoTS/blob/master/extended_tutorial.md#installation-and-dependency-versioning) 

请注意，还有其他几个项目选择了类似的名称，因此请确保您使用的是正确的 AutoTS 代码、论文和文档。.

##  基本使用

AutoTS 的输入数据预计采用 *长* 或 *宽* 格式：  （*long* or *wide* ）
- *wide* 格式是一个带有`pandas.DatetimeIndex`的`pandas.DataFrame`，每列都是一个不同的series。
- *long* 格式包含三列：
  - Date（最好已经是 pandas 识别的` 日期时间` 格式）。
  - Series ID. 对于单个时间序列，series_id 可以 `= None`。 
  - Value
- 对于 *long* 数据，每个数据的列名称都会作为`date_col`、`id_col`和`value_col`传递给``.fit()`。 *wide* 数据不需要参数。

Lower-level 的函数仅针对 `wide`类型数据而设计。

```python
# 其他载入选项: _hourly, _monthly, _weekly, _yearly, or _live_daily
from autots import AutoTS, load_daily

# 示例数据集可用于*长*导入形状或*宽*导入形状
long = False
df = load_daily(long=long)

model = AutoTS(
    forecast_length=21,
    frequency='infer',
    prediction_interval=0.9,
    ensemble='auto',
    model_list="fast",  # "superfast", "default", "fast_parallel"
    transformer_list="fast",  # "superfast",
    drop_most_recent=1,
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
)
model = model.fit(
    df,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
)

prediction = model.predict()
# 绘制一个样本
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2019-01-01")
# 打印最佳模型的详细信息
print(model)

# 点预测 dataframe
forecasts_df = prediction.forecast
# 预测上限和下限
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# 所有尝试的模型结果的准确性
model_results = model.results()
# 并从交叉验证中汇总
validation_results = model.results("validation")
```

lower-level API，特别是 scikit-learn 风格的大部分time series transformers，也可以独立于 AutoML 框架使用。

查看 [extended_tutorial.md](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html) 以获取更详细的功能指南。

另请查看[production_example.py](https://github.com/winedarksea/AutoTS/blob/master/production_example.py)

## 速度和大数据的提示：
* 使用适当的模型列表，尤其是预定义的列表：
	* `superfast` （简单的朴素模型）和 `fast` （更复杂但仍然更快的模型，针对许多系列进行了优化）
	* `fast_parallel`（`fast`和`parallel`的组合）或`parallel`，假设有许多 CPU 核心可用
		* `n_jobs` 通常与  `='auto'` 非常接近，但根据环境进行必要的调整
	* 使用`from autots.models.model_list import model_lists`查看预定义列表的字典（一些定义供内部使用）
* 使用`subset`参数，当存在许多相似的序列时，`subset=100`通常对成千上万个类似的序列概括得很好。
	* 如果使用`subset`，为序列传递`weights`会使子集选择偏向于优先级更高的序列。
	* 如果受到RAM限制，可以通过在不同批次的数据上运行多个AutoTS实例来分布式处理，首先导入一个预先训练好的模板，作为所有实例的起点。
* 设置`model_interrupt=True`，当按下`KeyboardInterrupt`，即`ctrl+c`时，会跳过当前模型（尽管如果中断发生在生成之间，它会停止整个训练）。
* 使用`.fit()`的`result_file`方法，它会在每一代结束后保存进度 - 这对于长时间的训练非常有帮助。使用`import_results`来恢复进度。
* 虽然转换（Transformations）相当快，但将`transformer_max_depth`设置为较低的数值（例如，2）将提高速度。同时使用`transformer_list`设置为'fast'或'superfast'。
* 查看[这个例子](https://github.com/winedarksea/AutoTS/discussions/76)，了解如何将AutoTS与pandas UDF一起使用。
* 显然，集成模型（Ensembles）预测较慢，因为它们需要运行多个模型，'distance'模型慢2倍，'simple'模型慢3到5倍。
	* 使用`ensemble='horizontal-max'`结合`model_list='no_shared_fast'`可以在有许多CPU核的情况下相对好地扩展，因为每个模型只在所需的序列上运行。
* 减少`num_validations`和`models_to_validate`会减少运行时间，但可能导致模型选择不佳。
* 对于记录数量较多的数据集，上采样（例如，从每日到每月频率预测）如果合适的话，可以缩短训练时间。
	* 这可以通过调整`frequency`和`aggfunc`来完成，但最好在将数据传入AutoTS之前进行。
* 如果NaN已经被填充，则处理会更快。如果不需要寻找最佳NaN填充方法，则在传递给类之前用一种合适的方法填充所有NaN。
* 在`metric_weighting`中将`runtime_weighting`设置为较高的值。这会引导搜索朝向更快的模型，尽管可能会牺牲一些精度。

## 如何贡献：
* 对你觉得文档混乱的地方提供反馈
* 使用AutoTS并且...
	* 通过在GitHub上添加Issues来报告错误和请求功能
	* 发布适合你数据的顶级模型模板（以帮助改进起始模板）
	* 随意推荐你最喜欢的模型的不同搜索网格参数
* 当然，也可以直接在GitHub上对代码库做出贡献。

*也被称为Project CATS（Catlin的自动时间序列），因此有这个logo。*
