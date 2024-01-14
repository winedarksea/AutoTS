# 扩展教程

## 目录
* [一个简单的例子](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id1)
* [验证和交叉验证](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id2)
* [另一个例子](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id3)
* [模型列表](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id4)
* [部署](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#deployment-and-template-import-export)
* [只运行一个模型](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id5)
* [度量](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id6)
* [集成](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#ensembles)
* [安装](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#installation-and-dependency-versioning)
* [注意事项](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#caveats-and-advice)
* [添加回归器](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#adding-regressors-and-other-information)
* [模拟预测](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id8)
* [事件风险预测](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id9)
* [模型](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id10)

### 一个简单的例子
```python
# also: _hourly, _daily, _weekly, or _yearly
from autots.datasets import load_monthly

df_long = load_monthly(long=True)

from autots import AutoTS

model = AutoTS(
    forecast_length=3,
    frequency='infer',
    ensemble='simple',
    max_generations=5,
    num_validations=2,
)
model = model.fit(df_long, date_col='datetime', value_col='value', id_col='series_id')

# Print the description of the best model
print(model)
```

#### 数据导入
接受两种形状/风格的`pandas.DataFrame`数据。
第一种是*long* 数据格式，类似于聚合的销售交易表，包含三个列，分别在`.fit()`中定义为`date_col {pd.Datetime}, value_col {感兴趣的数字或分类数据}，和 id_col {id字符串，如果提供了多个系列}`。
另一种可能是 *wide*格式的数据，其中索引是`pandas.DatetimeIndex`，每一列是一个独立的数据系列。

如果使用水平风格的集成， series_ids/column 名将被强制转换为字符串。

#### 你可以通过一些方式定制这个过程...
提高准确性最简单的方法是增加代数`max_generations=15`。每一代尝试新模型，需要更多时间但提高了准确性。然而，遗传算法的性质意味着每一代的改进并不一致，大量的代数往往只会带来最小的性能提升。

另一种可能提高准确性的方法是设置`ensemble='all'`。集成参数期望一个字符串，例如可以是`'simple,dist'`或`'horizontal'`。由于这意味着存储每个模型的更多细节，因此需要更多的时间和内存。

当你的数据预期始终大于等于0时（如单位销售量），设置`no_negatives=True`是一个方便的参数。这会强制预测值大于或等于0。
类似的功能是`constraint=2.0`。这样做是为了防止预测超出训练数据设定的历史范围。在这个例子中，预测不会被允许超过`max(training data) + 2.0 * st.dev(training data)`,，最小值方面也是如此。`0`的约束将预测限制在历史最小值和最大值之间。

另一个便利功能是`drop_most_recent=1`，指定要删除的最近几个时期的数量。这对于月度数据很有用，因为通常最近一个月的数据是不完整的。
`drop_data_older_than_periods`提供了类似的功能，但删除最旧的数据以加快大型数据集的处理速度。
`remove_leading_zeroes=True`对于前导零表示尚未开始的过程的数据很有用。

在处理许多时间序列时，利用`subset=100`可能会有帮助。子集指定用于测试模型的时间序列的整数数量，对于许多相关时间序列（例如，成千上万的客户销售数据）很有用。通常，在100个相关时间序列上测试出的最佳模型与在成千上万（或更多）系列上测试的非常接近。

子集利用加权，加权较高的系列更有可能被选中。在多个时间序列中使用加权是为了告诉评估器哪些系列最重要。默认情况下，所有系列的权重被假定为1，只有在需要不同于1的值时才需要传入值。
关于加权，需要注意的是，较大的权重=更重要。

可能最容易引起问题的是存在大量NaN/缺失数据。特别是在最近可用数据中有很多缺失数据。
使用适当的交叉验证（特别是如果旧数据中常见NaN而最近数据中不常见，则使用`backwards`）可以帮助。
删除大部分缺失的系列，或使用`prefill_na=0`（或其他值）也可以帮助。

### 需要注意的问题
有一些基本的事情需要警惕，这些常常会导致糟糕的结果：

1. *最近*数据中的坏数据（突然下降或缺失值）是这里最常见的糟糕预测原因。由于许多模型使用最近的数据作为起点，最近数据点中的错误可能会对预测产生过大的影响。
2. 不具代表性的交叉验证样本。模型是基于交叉验证中的表现来选择的。如果验证不能准确代表系列，可能会选择一个不好的模型。选择一个好的方法并尽可能多地进行验证。
3. 不会重复的异常情况。手动去除异常可能比任何自动方法都有效。与此同时，要注意NaN发生模式的变化，因为学习过的FillNA可能不再适用。
4. 人为的历史事件，一个简单的例子是销售促销活动。使用回归器是处理这类事件的最常见方法，对于建模这些类型的事件可能至关重要。

在自动预测之前你不需要做的事情是任何典型的预处理。最好让模型选择过程来选择，因为不同的模型对不同类型的预处理有不同的表现。

### 验证和交叉验证
交叉验证有助于确保最佳模型在时间序列的动态中稳定。
由于需要防止未来数据点的数据泄露，交叉验证在时间序列数据中可能比较棘手。

首先，所有模型最初都在最新的数据片段上进行验证。这是因为最近的数据通常最能接近预测的未来。
在数据量非常小的情况下，可能没有足够的数据进行交叉验证，在这种情况下，`num_validations`可以设置为0。这也可以加速快速测试。
注意，当`num_validations=0`时，仍然会运行*one evaluation*（一次评估） 。它只是不是交叉验证。`num_validations`是除此之外要进行的**cross**(交叉)验证的数量。
一般来说，最安全的方法是尽可能多地进行验证，只要有足够的数据进行验证即可。

这里有一些可用的方法：

**Backwards**（向后）交叉验证是最安全的方法，它从最近的数据开始向后工作。首先取最近的forecast_length样本，然后是下一个最近的forecast_length样本，依此类推。这使得它更适合于较小或快速变化的数据集。

**Even**（平均）交叉验证将数据切分成相等的块。例如，`num_validations=3`将数据分成连续的三等份（减去原始验证样本）。最终的验证结果将包括四部分，三个交叉验证样本的结果以及原始验证样本的结果。

**Seasonal**（季节性）验证提供为`'seasonal n'`，即`'seasonal 364'`。这是对`backwards`验证的变体，如果提供了合适的周期，则提供了所有验证方法中最好的性能。
它像往常一样在最近的数据上进行训练，然后验证是从预测的日期向前`n`个周期。
例如，对于每日数据，预测一个月后，`n=364`，第一个测试可能是在2021年5月，验证在2020年6月和2019年6月，最后的预测则是2021年6月。

**Similarity**（相似性）自动找到与用于预测的最新数据最相似的数据部分。这是最好的通用选择，但目前可能对杂乱的数据敏感。

**Custom**（自定义）允许任何类型的验证。如果使用，.fit()需要传入`validation_indexes` - 一个pd.DatetimeIndex的列表，每个的尾部forecast_length用作测试（应与`num_validations` + 1的长度相同）。

`backwards`、`even`和`seasonal`验证都在最新的数据分割上进行初始评估。`custom`在提供的列表中的第一个索引上进行初始评估，而`similarity`首先作用于最近距离的段落。

从初始验证到交叉验证只选取一部分模型。设置模型数量，如`models_to_validate=10`。
如果提供了0到1之间的浮点数，则视为选择的模型的百分比。
如果你怀疑你最近的数据并不真正代表整体，增加这个参数是个好主意。
然而，将这个值提高到`0.35`（即35%）以上不太可能有太大的好处，因为许多模型参数相似。

虽然NaN值会被处理，但如果任何系列在生成的训练/测试分割中有大量NaN值，模型选择将受到影响。
最常见的情况可能是一些系列有很长的历史，而同一数据集中的其他系列只有非常近期的数据。
在这些情况下，避免使用`even`交叉验证，使用其他验证方法。

### 另一个例子：
这里，我们正在预测明尼苏达州明尼阿波利斯和圣保罗之间94号州际公路的交通。这是一个很好的数据集，用来演示包括外部变量的推荐方式 - 通过将它们作为时间序列并赋予较低的权重来包括。
这里包括了天气数据 - 冬天和道路施工是交通量的主要影响因素，并将与交通量一起预测。这些额外的系列向诸如`RollingRegression`、`VARMAX`和`VECM`等模型提供信息。

这里还可以看到`model_list`的使用。

```python
from autots import AutoTS
from autots.datasets import load_hourly

df_wide = load_hourly(long=False)

# here we care most about traffic volume, all other series assumed to be weight of 1
weights_hourly = {'traffic_volume': 20}

model_list = [
    'LastValueNaive',
    'GLS',
    'ETS',
    'AverageValueNaive',
]

model = AutoTS(
    forecast_length=49,
    frequency='infer',
    prediction_interval=0.95,
    ensemble=['simple', 'horizontal-min'],
    max_generations=5,
    num_validations=2,
    validation_method='seasonal 168',
    model_list=model_list,
	transformer_list='all',
    models_to_validate=0.2,
    drop_most_recent=1,
	n_jobs='auto',
)

model = model.fit(
    df_wide,
    weights=weights_hourly,
)

prediction = model.predict()
forecasts_df = prediction.forecast
# prediction.long_form_results()
```

概率预测适用于所有模型，但在许多情况下只是基于数据的估计而不是模型估计。
```python
upper_forecasts_df = prediction.upper_forecast
lower_forecasts_df = prediction.lower_forecast
```

### 模型列表
默认情况下，大多数可用模型都会被尝试。要使用更有限的模型子集，可以传入一个自定义列表，或更简单地，一个字符串，如`'probabilistic', 'multivariate', 'fast', 'superfast', 或 'all'`。

所有可用模型的表格如下。

在大型多变量系列上，`DynamicFactor`和`VARMAX`可能慢得不切实际。

## 部署和模板导入/导出
请查看[production_example.py](https://github.com/winedarksea/AutoTS/blob/master/production_example.py)

许多模型可以在AutoTS之外相对简单地通过将所选参数放入Statsmodels或其他底层包来反向工程。
在模型训练之后，顶级模型可以导出为`.csv`或`.json`文件，然后在下一次运行时只尝试这些模型。
这允许改善容错能力（不依赖于一个模型，而是依赖于几个可能的模型和底层包），并在时间序列演变时灵活切换模型。
需要注意的一点是，由于AutoTS仍在开发中，模板格式可能会改变，并与未来的包版本不兼容。

```python
# after fitting an AutoTS model
example_filename = "example_export.csv"  # .csv/.json
model.export_template(example_filename, models='best',
					  n=15, max_per_model_class=3)

# on new training
model = AutoTS(forecast_length=forecast_length,
			   frequency='infer', max_generations=0,
			   num_validations=0, verbose=0)
model = model.import_template(example_filename, method='only') # method='add on'
print("Overwrite template is: {}".format(str(model.initial_template)))
```

### 只运行一个模型
虽然上述版本的部署，具有不断演变的模板和每次运行时的交叉验证，是推荐的部署方式，但也可以运行单个固定模型。

来自AutoTS深层内部的这个功能只能接受`wide`（宽）风格的数据（有一个long_to_wide函数可用）。
数据必须已经相对干净 - 全部是数字（或np.nan）。
这将运行集成，因此通常比直接加载模型更推荐。子模型使用sklearn格式。

```python
from autots import load_daily, model_forecast


df = load_daily(long=False)  # long or non-numeric data won't work with this function
df_forecast = model_forecast(
    model_name="AverageValueNaive",
    model_param_dict={'method': 'Mean'},
    model_transform_dict={
        'fillna': 'mean',
        'transformations': {'0': 'DifferencedTransformer'},
        'transformation_params': {'0': {}}
    },
    df_train=df,
    forecast_length=12,
    frequency='infer',
    prediction_interval=0.9,
    no_negatives=False,
    # future_regressor_train=future_regressor_train2d,
    # future_regressor_forecast=future_regressor_forecast2d,
    random_seed=321,
    verbose=0,
    n_jobs="auto",
)
df_forecast.forecast.head(5)
```

AutoTS类的 `model.predict()` 运行模型，该模型由三个存储的属性给出：
```
model.best_model_name,
model.best_model_params,
model.best_model_transformation_params
```
如果你重写这些属性，它将相应地改变预测输出。

### 指标
有许多可用的度量标准，所有这些都结合在一起形成一个评估最佳模型的“得分”。比较模型的“得分”可以通过传递自定义度量权重字典轻松调整。
更高的权重增加了该度量标准的重要性，而0则将该度量标准从考虑中移除。权重必须是大于或等于0的数字。
这种权重不应与系列权重混淆，后者影响任一度量标准对所有系列的平等应用。
```python
metric_weighting = {
	'smape_weighting': 5,
	'mae_weighting': 2,
	'rmse_weighting': 2,
	'made_weighting': 0.5,
	'mage_weighting': 1,
	'mle_weighting': 0,
	'imle_weighting': 0,
	'spl_weighting': 3,
	'containment_weighting': 0,
	'contour_weighting': 1,
	'runtime_weighting': 0.05,
}

model = AutoTS(
	forecast_length=forecast_length,
	frequency='infer',
	metric_weighting=metric_weighting,
)
```
使用多个度量标准的好处有几个原因。首先是为了避免过拟合 - 在多个度量标准上表现良好的模型不太可能过拟合。
其次，预测通常需要满足多重期望。使用综合评分可以平衡
点预测的质量、概率预测的质量、过估计或低估计、视觉拟合度和运行速度。

一些度量标准是缩放的，有些则不是。MAE、RMSE、MAGE、MLE、iMLE是未缩放的，因此在多变量预测中会倾向于对最大规模输入系列的模型性能。

*水平*风格的集成使用`metric_weighting`来选择系列，但只传递了`mae, rmse, made, mle, imle, contour, spl`的值。如果所有这些都是0，则使用mae进行选择。
因此，在使用这些集成时，减少使用`smape`、`containment`和`mage`权重可能更好。在单变量模型中，整体的运行时间不会转换为水平集成内的运行时间。

`sMAPE`是*对称平均绝对百分比损失*，通常是跨多个系列最通用的度量标准，因为它是缩放的。它不擅长处理大量零的预测。

`SPL`是*缩放针球损失*，有时称为*分位数损失*，是优化上下分位数预测准确度的最佳度量标准。

`Containment`度量测试数据落在上下预测之间的百分比，比SPL更易于人类阅读。在其他地方也被称为`coverage_fraction`。

`MLE`和`iMLE`受`mean squared log error`的启发，是*平均对数误差*。它们用于针对MAE对过估计或低估计进行定向，对数(error)用于惩罚较轻（且对异常值敏感性较低）的方向。
`MLE`对预测不足的预测给予更大的惩罚。
`iMLE`是相反的，对过度预测给予更多的惩罚。

`MAGE`是*平均绝对聚合误差*，它衡量预测的汇总误差。在分层/分组预测中，这有助于选择系列，在汇总时最小化过估计或低估计。

`Contour`旨在帮助选择在视觉上看起来与实际相似的模型。因此，它度量了预测和实际都向同一方向变化的点的百分比，无论变化幅度多少，但*不包括*这种差异的幅度。它比MADE更易于人类阅读这些信息。
这与MDA（平均方向准确度）类似，但更快，因为contour将无变化视为正面情况。

`MADE`是*(缩放的)平均绝对差分误差*。与contour类似，它度量预测变化与实际时间步变化的相似程度。Contour度量方向，而MADE度量幅度。当forecast_length=1时，相当于'MAE'。它比contour更适合优化。

contour和MADE度量标准有用，因为它们鼓励'波浪型'预测，即，不是平直线预测。尽管有时平直线朴素或线性预测可能是非常好的模型，但它们对一些经理来说"看起来不够努力"，使用它们倾向于非平直线预测，对许多人来说看起来像是更严肃的模型。

如果某个度量标准在初始结果中完全是NaN，可能是因为保留的数据在实际中完全是NaN。

```python
import matplotlib.pyplot as plt

model = AutoTS().fit(df)
prediction = model.predict()

prediction.plot(
	model.df_wide_numeric,
	series=model.df_wide_numeric.columns[2],
	remove_zeroes=False,
	start_date="2018-09-26",
)
plt.show()

model.plot_per_series_mape(kind="pie")
plt.show()

model.plot_per_series_error()
plt.show()

model.plot_generation_loss()
plt.show()

if model.best_model_ensemble == 2:
	model.plot_horizontal_per_generation()
	plt.show()
	model.plot_horizontal_transformers(method="fillna")
	plt.show()
	model.plot_horizontal_transformers()
	plt.show()
	model.plot_horizontal()
	plt.show()
	if "mosaic" in model.best_model["ModelParameters"].iloc[0].lower():
		mosaic_df = model.mosaic_to_df()
		print(mosaic_df[mosaic_df.columns[0:5]].head(5))

if False:  # slow
	model.plot_backforecast(n_splits="auto", start_date="2019-01-01")
```

### 分层和分组预测
分层和分组是指多变量预测情况下个别系列被聚合的情况。
一个常见的例子是产品销售预测，其中单个产品被预测，然后也聚合以获得对所有产品需求的视图。
聚合结合了个别系列的错误，因此可能导致对整体需求的严重过估计或低估计。
传统上，为了解决这个问题，使用了协调方法，其中顶层和低层预测被平均或以其他方式调整，以产生一个不那么夸张的最终结果。

不幸的是，任何协调方法本质上都是次优的。
在优化预测的实际数据上，个别系列的错误贡献和错误的方向（过估计或低估计）通常是不稳定的，
不仅从预测到预测，而且在每个预测内的时间步之间。因此，协调往往将错误的错误量分配到错误的地方。

这里对这个问题的建议是从一开始就解决问题，利用`MAGE`度量标准进行验证。
这评估了预测的聚合效果，当作为度量权重的一部分使用时，推动模型选择朝向聚合良好的预测。
`MAGE`评估所有存在的系列，因此如果存在非常不同的子组，有时可能需要在单独的运行中对这些组进行建模。
此外，如果分别确定了低估计或过估计是问题，可以使用`MLE`或`iMLE`。

### 集成
集成方法由`ensemble=`参数指定。它可以是列表或逗号分隔的字符串。

`simple`（简单）风格的集成（在模板中标记为'BestN'）是最容易识别的集成形式，是指定模型的简单平均，通常是3或5个模型。
`distance`（距离）风格的集成是两个模型拼接在一起。第一个模型预测预测期的前一部分，第二个模型预测后半部分。模型之间没有重叠。
`simple`和`distance`风格的模型都是在第一组评估数据上构建的，并与其他选定的模型一起进行验证。
这两种方法也可以是递归的，包含集成的集成。当从起始模板导入集成时，可能会发生这种递归集成 - 它们工作得很好，但可能会变得相当慢，因为有很多模型。

`horizontal`（水平）集成是最初创建此软件包的集成类型。
在此模式下，每个系列都有自己的模型。这避免了许多时间序列在一个数据集中时“一刀切”的问题。
为了效率，单变量模型只在需要它们的系列上运行。
`no_shared`列表中未包含的模型可能会使水平集成在规模上非常缓慢 - 因为即使只用于一个系列，它们也必须为每个系列运行。
`horizontal-max`为每个模型选择最佳系列。`horizontal`和`horizontal-min`在尽可能保持准确性的同时，尝试减少选择慢速模型的数量。
一个名为`horizontal_generalization`的功能允许使用`subset`，使这些集成具有容错能力。
然而，如果你看到`no full models available`的消息，这意味着这种泛化可能会失败。通常包括至少一个`superfast`模型或`no_shared`列表中没有的模型可以防止这种情况。
这些集成是基于`mae, rmse, contour, spl`的每系列准确性选择的，按照`metric_weighting`指定的权重加权。
`horizontal`集成可以包含`simple`和`distance`风格集成的递归深度，但不能嵌套`horizontal`集成。

`mosaic`（马赛克）集成是`horizontal`集成的扩展，但对每个系列*和*每个预测周期选择特定模型。
因为这意味着最大模型数量可以是`系列数量 * 预测长度`，显然可能会变得相当慢。
理论上，这种风格的集成提供最高的准确性。
它们更容易过拟合，因此在更稳定的数据上使用更多的验证。
与`horizontal`集成不同，后者只适用于多变量数据集，`mosaic`可以在单个时间序列上运行，只要预测期限> 1。

如果你只关心一个预测点的准确性，但想要进行完整预测期长度的预测，你可以将`mosaic`集成转换为仅针对该预测期的`horizontal`。
```python
import json
from autots.models.ensemble import mosaic_to_horizontal, model_forecast

# assuming model is from AutoTS.fit() with a mosaic as best_model
model_params = mosaic_to_horizontal(model.best_model_params, forecast_period=0)
result = model_forecast(
	model_name="Ensemble",
	model_param_dict=model_params,
	model_transform_dict={},
	df_train=model.df_wide_numeric,
	forecast_length=model.forecast_length,
)
result.forecast
```

## 安装和依赖版本控制
`pip install autots`

某些可选包在Windows上安装时需要[Visual Studio C编译器](https://visualstudio.microsoft.com/visual-cpp-build-tools/)。

在Linux系统上，使用apt-get/yum（而不是pip）安装numpy/pandas可能会安装得更快、更稳定。
Linux还可能需要`sudo apt install build-essential`来安装某些包。

你可以使用`numpy.show_config()`检查你的系统是使用mkl、OpenBLAS还是没有使用。通常建议在安装新包后仔细检查这一点，以确保你没有破坏LINPACK连接。

### 要求：
	Python >= 3.6
	numpy
		>= 1.20（Motif和WindowRegression中的Sliding Window）
	pandas
		>= 1.1.0（prediction.long_form_results()）
		gluonts不兼容1.1、1.2、1.3
	sklearn
		>= 0.23.0（PoissonReg）
		>= 0.24.0（OrdinalEncoder handle_unknown）
		>= 1.0对于受"mse" -> "squared_error"更新影响的模型
		>?（IterativeImputer, HistGradientBoostingRegressor）
	statsmodels
		>= 0.13 ARDL和UECM
	scipy.uniform_filter1d（仅用于mosaic-window集成）
	scipy.stats（异常检测，Kalman）
	scipy.signal（ScipyFilter）
	scipy.spatial.cdist（Motifs）

其中，numpy和pandas是关键的。
没有scikit-learn应该存在有限的功能。
###
	* Sklearn needed for categorical to numeric, some detrends/transformers, horizontal generalization, numerous models, nan_euclidean distance
在没有 statsmodels 的情况下，应该保持完整的功能，尽管可用的模型较少。

Prophet、Greykite和mxnet/GluonTS是一些在某些系统上安装时可能比较挑剔的包。

`pip install autots['additional']`
### 可选的包
	requests
	psutil
	holidays
	prophet
	gluonts (requires mxnet)
	mxnet (mxnet-mkl, mxnet-cu91, mxnet-cu101mkl, etc.)
	tensorflow >= 2.0.0
	lightgbm
	xgboost
	tensorflow-probability
	fredapi
	greykite
	matplotlib
	pytorch-forecasting
	neuralprophet
	scipy
	arch
	
Tensorflow、LightGBM和XGBoost带来了强大的模型，但也是最慢的模型之一。如果速度是一个考虑因素，不安装它们将加快约~Regression风格模型的运行速度。

#### 安装的最安全选择：
使用venv、Anaconda或[Miniforge](https://github.com/conda-forge/miniforge/) ，这里有一些更多的提示 [here](https://syllepsis.live/2022/01/17/setting-up-and-optimizing-python-for-data-science-on-intel-amd-and-arm-including-apple-computers/).
```shell
# # 创建 conda 或 venv 环境
conda create -n timeseries python=3.9 # 创建一个名为timeseries的新conda环境，使用Python 3.9版本
conda activate timeseries  # 激活timeseries环境
# 使用pip安装numpy, scipy, scikit-learn等库，如果已存在则忽略安装 (--exists-action i)
python -m pip install numpy scipy scikit-learn statsmodels lightgbm xgboost numexpr bottleneck yfinance pytrends fredapi --exists-action i

python -m pip install pystan prophet --exists-action i # 安装pystan和prophet，如果失败，可以尝试conda-forge选项，或使用--no-deps参数通过pip安装prophet
python -m pip install tensorflow  # 安装tensorflow
python -m pip install mxnet --no-deps # 安装mxnet，查看mxnet文档了解更多安装选项，也可以尝试使用pip install mxnet --no-deps
python -m pip install gluonts arch # 安装gluonts和arch
python -m pip install holidays-ext pmdarima dill greykite --exists-action i --no-deps
# 安装pytorch
python -m pip install --upgrade numpy pandas --exists-action i   # mxnet喜欢（似乎毫无意义地）安装旧版本的numpy，因此升级numpy和pandas

python -m pip install autots --exists-action i
```

```shell
mamba install scikit-learn pandas statsmodels prophet numexpr bottleneck tqdm holidays lightgbm matplotlib requests xgboost -c conda-forge
pip install mxnet --no-deps
pip install yfinance pytrends fredapi gluonts arch
pip install intel-tensorflow scikit-learn-intelex
mamba install spyder
mamba install autots -c conda-forge
```

```shell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pytorch-forecasting -c conda-forge
pip install neuralprophet
```
GPU支持，仅限Linux。CUDA版本需要与包要求相匹配。
同一会话中运行混合CUDA版本可能导致崩溃。
```shell
nvidia-smi
mamba activate base
mamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 nccl  # install in conda base
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/  # NOT PERMANENT unless add to ./bashrc make sure is for base env, mine /home/colin/mambaforge/lib
mamba create -n gpu python=3.8 scikit-learn pandas statsmodels prophet numexpr bottleneck tqdm holidays lightgbm matplotlib requests -c conda-forge
mamba activate gpu
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install mxnet-cu112 --no-deps
pip install gluonts tensorflow neuralprophet pytorch-lightning pytorch-forecasting
mamba install spyder
```
`mamba` 和 `conda` 命令通常是可以互换的， `conda env remove -n env_name`

#### Intel conda 通道安装（有时更快，也更容易出现错误）
https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html
```shell
# create the environment
mamba create -n aikit37 python=3.7 intel-aikit-modin pandas statsmodels prophet numexpr bottleneck tqdm holidays lightgbm matplotlib requests tensorflow dpctl -c intel
conda config --env --add channels conda-forge
conda config --env --add channels intel
conda config --env --get channels

# install additional packages as desired
python -m pip install mxnet --no-deps
python -m pip install gluonts yfinance pytrends fredapi
mamba update -c intel intel-aikit-modin

python -m pip install autots

# OMP_NUM_THREADS, USE_DAAL4PY_SKLEARN=1
```
#### 自己的安装顺序
```shell
conda create -n autots python=3.10
conda activate autots

conda install scikit-learn pandas statsmodels prophet numexpr bottleneck tqdm holidays lightgbm matplotlib requests xgboost psutil yfinance pytrends fredapi -c conda-forge
pip install tensorflow tensorflow-probability 
pip install mxnet --no-deps  # 本地安装最高1.8，不要在线安装
pip install gluonts arch pystan # pystan 经常出现编译问题，需要安装Visual Studio C++编译器
pip install holidays-ext pmdarima dill greykite --exists-action i --no-deps
# 安装pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pytorch-forecasting -c conda-forge
conda install neuralprophet -c conda-forge
conda install spyder
conda install autots -c conda-forge
# 可选项
pip install intel-tensorflow scikit-learn-intelex


```
### 基准速度
```python
from autots.evaluator.benchmark import Benchmark
bench = Benchmark()
bench.run(n_jobs="auto", times=3)
bench.results
```

## 注意事项和建议

### 神秘的崩溃
通常神秘的崩溃或挂起（那些没有清晰错误信息的）发生在CPU或内存过载时。
`UnivariateRegression`通常最容易出现这些问题，从模型列表中移除它可能会有帮助（默认情况下，由于这个原因它不包含在大多数列表中）。

尝试设置`n_jobs=1`或其他较低的数字，这应该减少负载。也测试一下'超快'的朴素模型，通常它们的资源消耗较低。
GPU加速模型（在Regressions和GluonTS中的Tensorflow）也更容易崩溃，在使用时可能是问题的来源。
如果问题仍然存在，请在GitHub讨论或问题中发帖。

在大量使用多进程之间重启也可以帮助减少未来模型运行中崩溃的风险。

### 系列ID确实需要唯一（或者宽数据中的列名需要全部唯一）
正如所说，如果不是这样，可能会发生一些不该发生的奇怪事情。

另外，如果使用Prophet模型，你不能有任何名为'ds'的系列

### 短期训练历史
多少数据是“太少”取决于数据的季节性和波动性。
最小的训练数据最大程度地影响了进行适当交叉验证的能力。在这种情况下，设置`num_validations=0`。
由于集成基于测试数据集，如果`num_validations=0`，设置`ensemble=None`也是明智的。

### 添加回归器和其他信息
`future_` 回归器，明确这是将来将高度确定的数据。
关于未来的这种数据很少见，一个例子可能是预测销售时，计划每天未来将开放的商店数量。
通常，使用回归器对分离“有机”和“非有机”模式非常有帮助。
“非有机”模式指的是影响结果并可控制的人为业务决策。
一个非常常见的例子是促销和销售活动。
模型可以从过去的促销信息中学习，然后预测输入的计划促销活动的影响。
下面描述的模拟预测是可以并排测试多个促销计划，评估效果的地方。

只有少数模型支持添加回归器，且不是所有模型都能处理多个回归器。
提供回归器的方式是以`wide`风格作为pd.Series/pd.DataFrame，带有DatetimeIndex。

不知道未来？不用担心，模型可以处理相当多的并行时间序列，这是另一种添加信息的方式。
额外的回归器可以通过df_long的额外时间序列来预测。
这里的一些模型可以利用它们提供的额外信息来帮助提高预测质量。
为了防止预测准确性过度考虑这些额外系列，输入系列权重，降低或移除它们的预测准确性考虑。

*回归器的一个例子：*
```python
from autots.datasets import load_monthly
from autots.evaluator.auto_ts import fake_regressor
from autots import AutoTS

long = False
df = load_monthly(long=long)
forecast_length = 14
model = AutoTS(
    forecast_length=forecast_length,
    frequency='infer',
    validation_method="backwards",
    max_generations=2,
)
future_regressor_train2d, future_regressor_forecast2d = fake_regressor(
    df,
    dimensions=4,
    forecast_length=forecast_length,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
    drop_most_recent=model.drop_most_recent,
    aggfunc=model.aggfunc,
    verbose=model.verbose,
)

model = model.fit(
    df,
    future_regressor=future_regressor_train2d,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
)

prediction = model.predict(future_regressor=future_regressor_forecast2d, verbose=0)
forecasts_df = prediction.forecast

print(model)
```

对于这里的低级API中的模型，令人困惑的是，必须指定`regression_type="User"`，同时传递`future_regressor`。为什么？这使得模型搜索可以轻松尝试使用和不使用回归器，因为有时回归器可能弊大于利。

## 模拟预测
模拟预测允许实验不同的潜在未来情景，以检查其对预测的潜在影响。
这是通过在模型的`.fit`中传递`future_regressor`的已知值，然后用`future_regressor`未来值的多种变化运行`.predict`来完成的。
在AutoTS中，默认情况下，当提供了`future_regressor`时，可以利用它的模型都会尝试使用和不使用回归器。
为了强制使用future_regressor进行模拟预测，必须提供以下几个参数：它们是`model_list, models_mode, initial_template`。

```python
from autots.datasets import load_monthly
from autots.evaluator.auto_ts import fake_regressor
from autots import AutoTS

df = load_monthly(long=False)
forecast_length = 14
model = AutoTS(
    forecast_length=forecast_length,
	max_generations=2,
    model_list="regressor",
    models_mode="regressor",
    initial_template="random",
)
# here these are random numbers but in the real world they could be values like weather or store holiday hours
future_regressor_train, future_regressor_forecast = fake_regressor(
    df,
    dimensions=2,
    forecast_length=forecast_length,
    drop_most_recent=model.drop_most_recent,
    aggfunc=model.aggfunc,
    verbose=model.verbose,
)
# another simulation of regressor
future_regressor_forecast_2 = future_regressor_forecast + 10

model = model.fit(
    df,
    future_regressor=future_regressor_train,
)
# first with one version
prediction = model.predict(future_regressor=future_regressor_forecast, verbose=0)
forecasts_df = prediction.forecast

# then with another
prediction_2 = model.predict(future_regressor=future_regressor_forecast_2, verbose=0)
forecasts_df_2 = prediction_2.forecast

print(model)
```
注意，这不一定强制模型对提供的特征赋予很大的价值。
可能需要多次重复运行，直到找到对变量响应满意的模型，
或者尝试使用回归器模型列表的子集，如`['FBProphet', 'GLM', 'ARDL', 'DatepartRegression']`。

## 事件风险预测和异常检测
异常（或离群值）检测是历史性的，事件风险预测是前瞻性的。

事件风险预测
生成一个风险评分（0到1，但通常接近0）表示未来事件超过用户指定的上限或下限的风险。

上限和下限可以是四种类型之一，每个可能不同。
1. None（不为这个方向计算风险评分）
2. 范围[0, 1]内的浮点数，系列的历史分位数（在边缘为历史最小值和最大值）被选为限制。
3. 一个字典{"model_name": x, "model_param_dict": y, "model_transform_dict": z, "prediction_interval": 0.9}来生成预测作为限制
	主要用于像SeasonalNaive这样的简单预测，但可以用于任何AutoTS模型
4. 一个自定义输入的numpy数组，形状为(forecast_length, num_series)

```python
import numpy as np
from autots import (
    load_daily,
    EventRiskForecast,
)
from sklearn.metrics import multilabel_confusion_matrix, classification_report

forecast_length = 6
df_full = load_daily(long=False)
df = df_full[0: (df_full.shape[0] - forecast_length)]
df_test = df[(df.shape[0] - forecast_length):]

upper_limit = 0.95  # --> 95% quantile of historic data
# if using manual array limits, historic limit must be defined separately (if used)
lower_limit = np.ones((forecast_length, df.shape[1]))
historic_lower_limit = np.ones(df.shape)

model = EventRiskForecast(
    df,
    forecast_length=forecast_length,
    upper_limit=upper_limit,
    lower_limit=lower_limit,
)
# .fit() is optional if model_name, model_param_dict, model_transform_dict are already defined (overwrites)
model.fit()
risk_df_upper, risk_df_lower = model.predict()
historic_upper_risk_df, historic_lower_risk_df = model.predict_historic(lower_limit=historic_lower_limit)
model.plot(0)

threshold = 0.1
eval_lower = EventRiskForecast.generate_historic_risk_array(df_test, model.lower_limit_2d, direction="lower")
eval_upper = EventRiskForecast.generate_historic_risk_array(df_test, model.upper_limit_2d, direction="upper")
pred_lower = np.where(model.lower_risk_array > threshold, 1, 0)
pred_upper = np.where(model.upper_risk_array > threshold, 1, 0)
model.plot_eval(df_test, 0)

multilabel_confusion_matrix(eval_upper, pred_upper).sum(axis=0)
print(classification_report(eval_upper, pred_upper, zero_division=1))  # target_names=df.columns
```
通过预测指定的限制可以用来使用一种模型（这里是ARIMA）来判断另一个生产模型界限的风险是否被超过。
这对于可视化特定模型的概率预测的有效性也很有用。

使用预测作为限制也是在历史数据中检测异常的常见方法 - 寻找超出预测期望的数据点。
`forecast_length`影响每个预测步骤的提前程度。更大的值计算更快，更小的值意味着更紧密的准确性（只有最极端的异常值被标记）。
`predict_historic`用于回顾训练数据集。使用`eval_periods`仅查看部分数据。
```python
lower_limit = {
	"model_name": "ARIMA",
	"model_param_dict": {'p': 1, "d": 0, "q": 1},
	"model_transform_dict": {},
	"prediction_interval": 0.9,
}
```

异常检测

有多种方法可用，包括使用`forecast_params`，可以用来分析AutoTS预测模型的历史偏差。

假期检测也可能捕捉到事件或“anti-holidays”（反假期），即需求低的日子。它不会捕捉到通常不会产生显著影响的假期。
```python
from autots.evaluator.anomaly_detector import AnomalyDetector
from autots.datasets import load_live_daily

# internet connection required to load this df
wiki_pages = [
	"Standard_deviation",
	"Christmas",
	"Thanksgiving",
	"all",
]
df = load_live_daily(
	long=False,
	fred_series=None,
	tickers=None,
	trends_list=None,
	earthquake_min_magnitude=None,
	weather_stations=None,
	london_air_stations=None,
	gov_domain_list=None,
	weather_event_types=None,
	wikipedia_pages=wiki_pages,
	sleep_seconds=5,
)

params = AnomalyDetector.get_new_params()
mod = AnomalyDetector(output='multivariate', **params)
mod.detect(df)
mod.plot()
mod.scores # meaning of scores varies by method

# holiday detection, random parameters
holiday_params = HolidayDetector.get_new_params()
mod = HolidayDetector(**holiday_params)
mod.detect(df)
# several outputs are possible, you'll need to subset results from multivariate inputs
full_dates = pd.date_range("2014-01-01", "2024-01-01", freq='D')
prophet_holidays = mod.dates_to_holidays(full_dates, style="prophet")
mod.plot()
```
### 传递参数的一个技巧（对于其他不可用的参数）
这里有很多可用的参数，但并不是特定参数的所有可用选项都实际用于生成的模板。
通常，非常慢的选项被省略了。如果你熟悉某个模型，可以尝试以这种方式手动添加那些参数值来运行...
需要澄清的是，你通常不能以这种方式添加完全新的参数，但你经常可以传递现有参数值的新选择。

1. 使用你想要的模型运行AutoTS并导出模板。
2. 在文本编辑器或Excel中打开模板，手动将参数值更改为你想要的值。
3. 再次运行AutoTS，这次在运行.fit()之前导入模板。
4. 不能保证它会选择给定参数的模型 - 选择是基于验证准确性，但至少它会运行，如果表现良好，它将被纳入该运行中的新模型（这就是遗传算法的工作方式）。

### 分类数据
分类数据被处理了，但处理得很粗糙。例如，优化度量目前不包括任何分类准确性度量。
对于具有有意义顺序的分类数据（例如“低”，“中”，“高”），最好在传入之前由用户编码该数据，
从而正确捕获相对序列（例如'低'=1，'中'=2，'高'=3）。

### 自定义和不寻常的频率
数据必须能强制转换为规则频率。建议将频率指定为pandas文档中的日期偏移量：https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects 
某些模型将支持更有限的频率范围。

## 独立使用Transformers
Transformers只期望数据以`wide`形状并且日期升序。
访问它们最简单的方法是通过[GeneralTransformer](https://winedarksea.github.io/AutoTS/build/html/source/autots.tools.html#autots.tools.transform.GeneralTransformer)。
它接受包含所需 transformers和.wide参数的字符串字典。

Inverse_transforms可能会令人困惑。可能需要Inverse_transforms数据以将预测值恢复到可用空间。
一些Inverse_transforms仅适用于训练期后紧接着的“原始”或“预测”数据。
DifferencedTransformer就是一个例子。
它可以采用训练数据的最后N个值将预测数据带回原始空间，但不适用于与训练期无关的任何未来时期。
一些transformers （主要是平滑过滤器，如`bkfilter`）根本无法inversed (逆转)，但转换后的值接近原始值。

```python
from autots.tools.transform import transformer_dict, DifferencedTransformer
from autots import load_monthly

print(f"Available transformers are: {transformer_dict.keys()}")
df = load_monthly(long=long)

# some transformers tolerate NaN, and some don't...
df = df.fillna(0)

trans = DifferencedTransformer()
df_trans = trans.fit_transform(df)
print(df_trans.tail())

# trans_method is not necessary for most transformers
df_inv_return = trans.inverse_transform(df_trans, trans_method="original")  # forecast for future data
```

### 关于~回归模型的说明
回归模型包括WindowRegression、RollingRegression、UnivariateRegression、MultivariateRegression和DatepartRegression。
它们都是将时间序列转换为传统机器学习和深度学习方法的X和Y的不同方式。
所有这些都来自于相同的潜在模型池，主要是sklearn和tensorflow模型。

* DatepartRegression是X仅为日期特征，Y为该日期的时间序列值的情况。
* WindowRegression取前`n`个数据点作为X来预测序列的未来值或值。
* RollingRegression取所有时间序列及其汇总滚动值在一个巨大的数据框架中作为X。对少量系列效果良好，但不易扩展。
* MultivariateRegression使用上述相同的滚动特征，但一次考虑一个，系列`i`的特征用于预测系列`i`的下一步，模型在所有系列的所有数据上训练。这个模型现在通常被社区称为“全球预测机器学习模型”。
* UnivariateRegression与MultivariateRegression相同，但在每个系列上训练一个独立模型，因此无法从其他系列的模式中学习。在水平集成中表现良好，因为它可以精简到一个系列，对该系列的表现相同。

目前`MultivariateRegression`有一个（较慢的）选项，可以使用具有分位数损失的标准GradientBoostingRegressor进行概率估计，而其他模型使用点到概率估计。

## 模型

| 模型                    | 依赖项       | 可选依赖项               | 概率性 | 多进程 | GPU   | 多变量 | 实验性 | 使用回归器 |
| :-------------          | :----------: | :---------------------: | :-----------  | :-------------- | :---- | :----------: | :----------: | :-----------: |
|  ConstantNaive          |              |                         |               |                 |       |              |              |               |
|  LastValueNaive         |              |                         |               |                 |       |              |              |               |
|  AverageValueNaive      |              |                         |    True       |                 |       |              |              |               |
|  SeasonalNaive          |              |                         |               |                 |       |              |              |               |
|  GLS                    | statsmodels  |                         |               |                 |       | True         |              |               |
|  GLM                    | statsmodels  |                         |               |     joblib      |       |              |              | True          |
| ETS - Exponential Smoothing | statsmodels |                      |               |     joblib      |       |              |              |               |
|  UnobservedComponents   | statsmodels  |                         |    True       |     joblib      |       |              |              | True          |
|  ARIMA                  | statsmodels  |                         |    True       |     joblib      |       |              |              | True          |
|  VARMAX                 | statsmodels  |                         |    True       |                 |       | True         |              |               |
|  DynamicFactor          | statsmodels  |                         |    True       |                 |       | True         |              | True          |
|  DynamicFactorMQ        | statsmodels  |                         |    True       |                 |       | True         |              |               |
|  VECM                   | statsmodels  |                         |               |                 |       | True         |              | True          |
|  VAR                    | statsmodels  |                         |    True       |                 |       | True         |              | True          |
|  Theta                  | statsmodels  |                         |    True       |     joblib      |       |              |              |               |
|  ARDL                   | statsmodels  |                         |    True       |     joblib      |       |              |              | True          |
|  FBProphet              | prophet      |                         |    True       |     joblib      |       |              |              | True          |
|  GluonTS                | gluonts, mxnet |                       |    True       |                 | yes   | True         |              | True          |
|  RollingRegression      | sklearn      | lightgbm, tensorflow    |               |     sklearn     | some  | True         |              | True          |
|  WindowRegression       | sklearn      | lightgbm, tensorflow    |               |     sklearn     | some  | True         |              | True          |
|  DatepartRegression     | sklearn      | lightgbm, tensorflow    |               |     sklearn     | some  |              |              | True          |
|  MultivariateRegression | sklearn      | lightgbm, tensorflow    |    True       |     sklearn     | some  | True         |              | True          |
|  UnivariateRegression   | sklearn      | lightgbm, tensorflow    |               |     sklearn     | some  |              |              | True          |
|  PreprocessingRegression | sklearn     |                         |    False      |                 |       |              |              | True          |
| Univariate/MultivariateMotif | scipy.distance.cdist |            |    True       |     joblib      |       | *            |              |               |
|  SectionalMotif         | scipy.distance.cdist |  sklearn        |    True       |                 |       | True         |              | True          |
|  MetricMotif, SeasonalityMotif |       |                         |    True       |                 |       |              |              |               |
|  BallTreeMultivariateMotif | sklearn, scipy |                    |    True       |                 |       | True         |              |               |
|  NVAR                   |              |                         |    True       |   blas/lapack   |       | True         |              |               |
|  RRVAR, MAR, TMF        |              |                         |               |                 |       | True         |              |               |
|  LATC                   |              |                         |               |                 |       | True         |              |               |
|  NeuralProphet          | neuralprophet |                        |    nyi        |     pytorch     | yes   |              |              | True          |
|  PytorchForecasting     | pytorch-forecasting |                  |    True       |     pytorch     | yes   | True         |              |               |
|  ARCH                   | arch         |                         |    True       |     joblib      |       |              |              | True          |
|  Cassandra              | scipy        |                         |    True       |                 |       | True         |              | True          |
|  KalmanStateSpace       |              |                         |    True       |                 |       |              |              |               |
|  FFT                    |              |                         |    True       |                 |       |              |              |               |
|  TiDE                   | tensorflow   |                         |               |                 | yes   | True         |              |               |
|  NeuralForecast         | NeuralForecast |                       |    True       |                 | yes   | True         | True         | True          |
|  MotifSimulation        | sklearn.metrics.pairwise |             |    True       |     joblib      |       | True         | True         |               |
|  Greykite               | (deprecated) |                         |    True       |     joblib      |       |              | True         |               |
|  TensorflowSTS          | (deprecated) |                         |    True       |                 | yes   | True         | True         |               |
|  TFPRegression          | (deprecated) |                         |    True       |                 | yes   | True         | True         | True          |
|  ComponentAnalysis      | (deprecated) |                         |               |                 |       | True         | True         | _             |

*NYI = 尚未实现*
* 已弃用的模型不会主动维护，但可能会在问题中请求更新