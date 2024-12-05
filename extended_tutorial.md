# Extended Tutorial

## Table of Contents
* [A Simple Example](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id1)
* [Validation and Cross Validation](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id2)
* [Another Example](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id3)
* [Model Lists](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id4)
* [Deployment](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#deployment-and-template-import-export)
* [Running Just One Model](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id5)
* [Metrics](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id6)
* [Ensembles](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#ensembles)
* [Installation](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#installation-and-dependency-versioning)
* [Caveats](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#caveats-and-advice)
* [Adding Regressors](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#adding-regressors-and-other-information)
* [Simulation Forecasting](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id8)
* [Event Risk Forecasting](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id9)
* [Models](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#id10)

### A simple example
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

#### Import of data
There are two shapes/styles of `pandas.DataFrame` which are accepted. 
The first is *long* data, like that out of an aggregated sales-transaction table containing three columns identified to `.fit()` as `date_col {pd.Datetime}, value_col {the numeric or categorical data of interest}, and id_col {id string, if multiple series are provided}`. 
Alternatively, the data may be in a *wide* format where the index is a `pandas.DatetimeIndex`, and each column is a distinct data series. 

If horizontal style ensembles are used, series_ids/column names will be coerced to strings. 

#### You can tailor the process in a few ways...
The simplest way to improve accuracy is to increase the number of generations `max_generations=15`. Each generation tries new models, taking additional time but improving the accuracy. The nature of genetic algorithms, however, means there is no consistent improvement for each generation, and large number of generations will often only result in minimal performance gains.

Another approach that may improve accuracy is to set `ensemble='all'`. Ensemble parameter expects a single string, and can for example be `'simple,dist'`, or `'horizontal'`. As this means storing more details of every model, this takes more time and memory.

A handy parameter for when your data is expected to always be 0 or greater (such as unit sales) is to set `no_negatives=True`. This forces forecasts to be greater than or equal to 0. 
A similar function is `constraint=2.0`. What this does is prevent the forecast from leaving historic bounds set by the training data. In this example, the forecasts would not be allowed to go above `max(training data) + 2.0 * st.dev(training data)`, as well as the reverse on the minimum side. A constraint of `0` would constrain forecasts to historical mins and maxes. 

Another convenience function is `drop_most_recent=1` specifing the number of most recent periods to drop. This can be handy with monthly data, where often the most recent month is incomplete. 
`drop_data_older_than_periods` provides similar functionality but drops the oldest data to speed up the process on large datasets. 
`remove_leading_zeroes=True` is useful for data where leading zeroes represent a process which has not yet started.

When working with many time series, it can be helpful to take advantage of `subset=100`. Subset specifies the interger number of time series to test models on, and can be useful with many related time series (1000's of customer's sales). Usually the best model on a 100 related time series is very close to that tested on many thousands (or more) of series.

Subset takes advantage of weighting, more highly-weighted series are more likely to be selected. Weighting is used with multiple time series to tell the evaluator which series are most important. Series weights are assumed to all be equal to 1, values need only be passed in when a value other than 1 is desired. 
Note for weighting, larger weights = more important.

Probably the most likely thing to cause trouble is having a lot of NaN/missing data. Especially a lot of missing data in the most recent available data. 
Using appropriate cross validation (`backwards` especially if NaN is common in older data but not recent data) can help. 
Dropping series which are mostly missing, or using `prefill_na=0` (or other value) can also help.

### What to Worry About
There are some basic things to beware of that can commonly lead to poor results:

1. Bad data (sudden drops or missing values) in the *most recent* data is the single most common cause of bad forecasts here. As many models use the most recent data as a jumping off point, error in the most recent data points can have an oversized effect on forecasts. Also remove all time series that are entirely NaN or entirely zero.
2. Misrepresentative cross-validation samples. Models are chosen on performance in cross validation. If the validations don't accurately represent the series, a poor model may be chosen. Choose a good method and as many validations as possible. 
3. Anomalies that won't be repeated. Manual anomaly removal can be more effective than any automatic methods. Along with this, beware of a changing pattern of NaN occurrences, as learned FillNA may not longer apply.
4. Artifical historical events, a simple example being sales promotions. Use of regressors is the most common method for dealing with this and may be critical for modeling these types of events. 

What you don't need to do before the automated forecasting is any typical preprocessing. It is best to leave it up to the model selection process to choose, as different models do better with different types of preprocessing. 

One of the most common causes of failures for loading a template on a new dataset is models failing on series that are too short (or essentially all missing). Filter out series that are too new or have been discontinued, before proceeding.

### Validation and Cross Validation
Cross validation helps assure that the optimal model is stable over the dynamics of a time series. 
Cross validation can be tricky in time series data due to the necessity of preventing data leakage from future data points. 

Firstly, all models are initially validated on the most recent piece of data. This is done because the most recent data will generally most closely resemble the forecast future. 
With very small datasets, there may be not be enough data for cross validation, in which case `num_validations` may be set to 0. This can also speed up quick tests. 
Note that when `num_validations=0` *one evaluation* is still run. It's just not cross validation. `num_validations` is the number of **cross** validations to be done in addition. 
In general, the safest approach is to have as many validations as possible, as long as there is sufficient data for it. 

Here are the available methods:

**Backwards** cross validation is the safest method and works backwards from the most recent data. First the most recent forecast_length samples are taken, then the next most recent forecast_length samples, and so on. This makes it more ideal for smaller or fast-changing datasets. 

**Even** cross validation slices the data into equal chunks. For example, `num_validations=3` would split the data into equal, progressive thirds (less the original validation sample). The final validation results would then include four pieces, the results on the three cross validation samples as well as the original validation sample. 

**Seasonal** validation is supplied as `'seasonal n'` ie `'seasonal 364'`. This is a variation on `backwards` validation and offers the best performance of all validation methods if an appropriate period is supplied. 
It trains on the most recent data as usual, then valdations are `n` periods back from the datetime of the forecast would be. 
For example with daily data, forecasting for a month ahead, and `n=364`, the first test might be on May 2021, with validation on June 2020 and June 2019, the final forecast then of June 2021. 

**Similarity** automatically finds the data sections most similar to the most recent data that will be used for prediction. This is the best general purpose choice but currently can be sensitive to messy data.

**Custom** allows validations of any type. If used, .fit() needs `validation_indexes` passed - a list of pd.DatetimeIndex's, tail of forecast_length of each is used as test (which should be of the same length as `num_validations` + 1).

`backwards`, `even` and `seasonal` validation all perform initial evaluation on the most recent split of data. `custom` performs initial evaluation on the first index in the list provided, while `similarity` acts on the closest distance segment first.

Only a subset of models are taken from initial validation to cross validation. The number of models is set such as `models_to_validate=10`. 
If a float in 0 to 1 is provided, it is treated as a % of models to select. 
If you suspect your most recent data is not fairly representative of the whole, it would be a good idea to increase this parameter. 
However, increasing this value above, say, `0.35` (ie 35%) is unlikely to have much benefit, due to the similarity of many model parameters. 

While NaN values are handled, model selection will suffer if any series have large numbers of NaN values in any of the generated train/test splits. 
Most commonly, this may occur where some series have a very long history, while others in the same dataset only have very recent data. 
In these cases, avoid the `even` cross validation and use one of the other validation methods. 

### Another Example:
Here, we are forecasting the traffice along Interstate 94 between Minneapolis and St Paul in Minnesota. This is a great dataset to demonstrate a recommended way of including external variables - by including them as time series with a lower weighting. 
Here weather data is included - winter and road construction being the major influencers for traffic and will be forecast alongside the traffic volume. These additional series carry information to models such as `RollingRegression`, `VARMAX`, and `VECM`. 

Also seen in use here is the `model_list`. 

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

Probabilistic forecasts are *available* for all models, but in many cases are just data-based estimates in lieu of model estimates. 
```python
upper_forecasts_df = prediction.upper_forecast
lower_forecasts_df = prediction.lower_forecast
```

### Model Lists
By default, most available models are tried. For a more limited subset of models, a custom list can be passed in, or more simply, a string, one of `'probabilistic', 'multivariate', 'fast', 'superfast', or 'all'`.

A table of all available models is below.

On large multivariate series, `DynamicFactor` and `VARMAX` can be impractically slow.

## Deployment and Template Import/Export
Take a look at the [production_example.py](https://github.com/winedarksea/AutoTS/blob/master/production_example.py)

Many models can be reverse engineered with (relative) simplicity outside of AutoTS by placing the choosen parameters into Statsmodels or other underlying package. 
Following the model training, the top models can be exported to a `.csv` or `.json` file, then on next run only those models will be tried. 
This allows for improved fault tolerance (by relying not on one, but several possible models and underlying packages), and some flexibility in switching models as the time series evolve. 
One thing to note is that, as AutoTS is still under development, template formats are likely to change and be incompatible with future package versions.

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

### Running Just One Model
While the above version of deployment, with  evolving templates and cross_validation on every run, is the recommended deployment, it is also possible to run a single fixed model. 

Coming from the deeper internals of AutoTS, this function can only take the `wide` style data (there is a long_to_wide function available). 
Data must already be fairly clean - all numerics (or np.nan). 
This will run Ensembles, and as such is generally recommended over loading the models directly. Subsidiary models use the sklearn format.

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

The `model.predict()` of AutoTS class runs the model given by three stored attributes:
```
model.best_model_name,
model.best_model_params,
model.best_model_transformation_params
```
If you overwrite these, it will accordingly change the forecast output.

### Metrics
There are a number of available metrics, all combined together into a 'Score' which evaluates the best model. The 'Score' that compares models can easily be adjusted by passing through custom metric weights dictionary. 
Higher weighting increases the importance of that metric, while 0 removes that metric from consideration. Weights must be numbers greater than or equal to 0.
This weighting is not to be confused with series weighting, which effects how equally any one metric is applied to all the series. 
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
It is best to use several metrics for several reasons. The first is to avoid overfitting - a model that does well on many metrics is less likely to be overfit. 
Secondly, forecasts often have to meet multiple expectations. Using a composite score allows balancing the 
quality of point forecast, quality of probabilistic forecast, overestimation or underestimation, visual fit, and speed of runtime.

Some metrics are scaled and some are not. MAE, RMSE, MAGE, MLE, iMLE are unscaled and accordingly in multivariate forecasting will favor model performance on the largest scale input series. 

*Horizontal* style ensembles use `metric_weighting` for series selection, but only the values passed for `mae, rmse, made, mle, imle, contour, spl`. If all of these are 0, mae is used for selection. 
Accordingly it may be better to reduce the use of`smape`, `containment`, and `mage` weighting when using these ensembles. With univariate models, runtime for overall won't translate to runtime inside a horizontal ensemble. 

`sMAPE` is *Symmetric Mean Absolute Percentage Loss* and is generally the most versatile metric across multiple series as it is scaled. It doesn't handle forecasts with lots of zeroes well. 

`SPL` is *Scaled Pinball Loss*, sometimes called *Quantile Loss*, and is the optimal metric for optimizing upper/lower quantile forecast accuracies.

`Containment` measures the percent of test data that falls between the upper and lower forecasts, and is more human readable than SPL. Also called `coverage_fraction` in other places.

`MLE` and `iMLE` are *Mean Logarithmic Error* inspired by the `mean squared log error`. They are used to target over or underestimation with MAE of the penalized direction and log(error) for the less-penalized (and less outlier sensitive) direction.
`MLE` penalizes an under-predicted forecast greater than an over-predicted forecast. 
`iMLE` is the inverse, and penalizes an over-prediction more.

`MAGE` is *Mean Absolute aGgregate Error* which measures the error of a rollup of the forecasts. This is helpful in hiearchial/grouped forecasts for selecting series that have minimal overestimation or underestimation when summed.

`Contour` is designed to help choose models which when plotted visually appear similar to the actual. As such, it measures the % of points where the forecast and actual both went in the same direction, either both up or both down, but *not* the magnitude of that difference. It is more human-readable than MADE for this information. 
This is similar to but faster than MDA (mean directional accuracy) as contour evaluates no change as a positive case.

`MADE` is *(Scaled) Mean Absolute Differential Error*. Similar to contour, it measures how well similar a forecast changes are to the timestep changes in the actual. Contour measures direction while MADE measures magnitude. Equivalent to 'MAE' when forecast_length=1. It is better for optimization than contour.

The contour and MADE metrics are useful as they encourages 'wavy' forecasts, ie, not flat line forecasts. Although flat line naive or linear forecasts can sometimes be very good models, they "don't look like they are trying hard enough" to some managers, and using them favors non-flat forecasts that (to many) look like a more serious model.

If a metric is entirely NaN in the initial results, likely that holdout was entirely NaN in actuals.

It may be worth viewing something like: `model.score_breakdown[model.score_breakdown.index == model.best_model_id].iloc[0]` to see if any one score is skewing selection. 
Generally you would want the numbers here to follow the balance requested in the `metric_weighting`.

##### Plots
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

### Hierarchial and Grouped Forecasts
Hiearchial and grouping refer to multivariate forecast situations where the individual series are aggregated. 
A common example of this is product sales forecasting, where individual products are forecast and then also aggregated for a view on demand across all products. 
Aggregation combines the errors of individual series, however, potentially resulting in major over- or -under estimation of the overall demand. 
Traditionally to solve this problem, reconciliation is used where a top-level and lower-level forecasts are averaged or otherwise adjusted to produce a less exaggerated final result. 

Unfortunately, any reconciliation approach is inherently sub-optimal. 
On real world data with optimized forecasts, the error contributions of individual series and the direction of the error (over- or under- estimate) are usually unstable, 
not only from forecast to forecast but from timestep to timestep inside each forecast. Thus reconciliation often reassigns the wrong amount of error to the wrong place. 

The suggestion here for this problem is to target the problem from the beginning and utilize the `MAGE` metric across validations. 
This assesses how well the forecasts aggregate, and when used as part of metric_weighting drives model selection towards forecasts that aggregate well. 
`MAGE` assesses all series present, so if very distinct sub-groups are present, it may be sometimes necessary to model those groups in separate runs. 
Additionally, `MLE` or `iMLE` can be used if either underestimation or overestimation respectively has been identified as a problem. 

### Ensembles
Ensemble methods are specified by the `ensemble=` parameter. It can be either a list or a comma-separated string.

`simple` style ensembles (labeled 'BestN' in templates) are the most recognizable form of ensemble and are the simple average of the specified models, here usally 3 or 5 models. 
`distance` style ensembles are two models spliced together. The first model forecasts the first fraction of forecast period, the second model the latter half. There is no overlap of the models. 
Both `simple` and `distance` style models are constructed on the first evaluation set of data, and run through validation along with all other models selected for validation. 
Both of these can also be recursive in depth, containing ensembles of ensembles. This recursive ensembling can happen when ensembles are imported from a starting template - they work just fine, but may get rather slow, having lots of models. 

`horizontal` ensembles are the type of ensembles for which this package was originally created. 
With this, each series gets its own model. This avoids the 'one size does not fit all' problem when many time series are in a dataset. 
In the interest of efficiency, univariate models are only run on the series they are needed for. 
Models not in the `no_shared` list may make horizontal ensembling very slow at scale - as they have to be run for every series, even if they are only used for one. 
`horizontal-max` chooses the best series for every model. `horizontal` and `horizontal-min` attempt to reduce the number of slow models chosen while still maintaining as much accuracy as possble. 
A feature called `horizontal_generalization` allows the use of `subset` and makes these ensembles fault tolerant. 
If you see a message `no full models available`, however, that means this generalization may fail. Including at least one of the `superfast` or a model not in `no_shared` models usually prevents this. 
These ensembles are choosen based on per series accuracy on `mae, rmse, contour, spl`, weighted as specified in `metric_weighting`.
`horizontal` ensembles can contain recursive depths of `simple` and `distance` style ensembles but `horizontal` ensembles cannot be nested. 

`mosaic` enembles are an extension of `horizontal` ensembles, but with a specific model choosen for each series *and* for each forecast period. 
As this means the maximum number of models can be `number of series * forecast_length`, this obviously may get quite slow. 
Theoretically, this style of ensembling offers the highest accuracy. 
They are much more prone to over-fitting, so use this with more validations and more stable data. 
Unlike `horizontal` ensembles, which only work on multivariate datasets, `mosaic` can be run on a single time series with a horizon > 1. 

One thing you can do with `mosaic` ensembles if you only care about the accuracy of one forecast point, but want to run a forecast for the full forecast length, you can convert the mosaic to horizontal for just that forecast period. 
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

## Installation and Dependency Versioning
`pip install autots`

Some optional packages require installing [Visual Studio C compilers](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if on Windows. 

On Linux systems, apt-get/yum (rather than pip) installs of numpy/pandas may install faster/more stable compilations. 
Linux may also require `sudo apt install build-essential` for some packages.

You can check if your system is using mkl, OpenBLAS, or none with `numpy.show_config()`. Generally recommended that you double-check this after installing new packages to make sure you haven't broken the LINPACK connection. 

### Requirements:
	Python >= 3.6
	numpy
		>= 1.20 (Sliding Window in Motif and WindowRegression)
	pandas
		>= 1.1.0 (prediction.long_form_results())
		gluonts incompatible with 1.1, 1.2, 1.3
	sklearn
		>= 0.23.0 (PoissonReg)
		>= 0.24.0 (OrdinalEncoder handle_unknown)
		>= 1.0 for models effected by "mse" -> "squared_error" update
		>? (IterativeImputer, HistGradientBoostingRegressor)
	statsmodels
		>= 0.13 ARDL and UECM
	scipy.uniform_filter1d (for mosaic-window ensemble only)
	scipy.stats (anomaly detection, Kalman)
	scipy.signal (ScipyFilter)
	scipy.spatial.cdist (Motifs)

Of these, numpy and pandas are critical. 
Limited functionality should exist without scikit-learn. 
	* Sklearn needed for categorical to numeric, some detrends/transformers, horizontal generalization, numerous models, nan_euclidean distance
Full functionality should be maintained without statsmodels, albeit with fewer available models. 

Prophet, Greykite, and mxnet/GluonTS are packages which tend to be finicky about installation on some systems.

`pip install autots['additional']`
### Optional Packages
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
	
Tensorflow, LightGBM, and XGBoost bring powerful models, but are also among the slowest. If speed is a concern, not installing them will speed up ~Regression style model runs. 

#### Safest bet for installation:
venv, Anaconda, or [Miniforge](https://github.com/conda-forge/miniforge/) with some more tips [here](https://syllepsis.live/2022/01/17/setting-up-and-optimizing-python-for-data-science-on-intel-amd-and-arm-including-apple-computers/).
```shell
# create a conda or venv environment
conda create -n timeseries python=3.9
conda activate timeseries

python -m pip install numpy scipy scikit-learn statsmodels lightgbm xgboost numexpr bottleneck yfinance pytrends fredapi --exists-action i

python -m pip install pystan prophet --exists-action i  # conda-forge option below works more easily, --no-deps to pip install prophet if this fails
python -m pip install tensorflow
python -m pip install mxnet --no-deps     # check the mxnet documentation for more install options, also try pip install mxnet --no-deps
python -m pip install gluonts arch
python -m pip install holidays-ext pmdarima dill greykite --exists-action i --no-deps
# install pytorch
python -m pip install --upgrade numpy pandas --exists-action i  # mxnet likes to (pointlessly seeming) install old versions of numpy

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
GPU support, Linux only. CUDA versions will need to match package requirements. Mixed CUDA versions may cause crashes if run in same session.
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
`mamba` and `conda` commands are generally interchangeable. `conda env remove -n env_name`

#### Intel conda channel installation (sometime faster, also, more prone to bugs)
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

### Speed Benchmark
```python
from autots.evaluator.benchmark import Benchmark
bench = Benchmark()
bench.run(n_jobs="auto", times=3)
bench.results
```

## Caveats and Advice

### Mysterious crashes
Usually mysterious crashes or hangs (those without clear error messages) occur when the CPU or Memory is overloaded. 
`UnivariateRegression` is usually the most prone to these issues, removing it from the model_list may help (by default it is not included in most lists for this reason). 

Try setting `n_jobs=1` or an otherwise low number, which should reduce the load. Also test the 'superfast' naive models, which are generally low resource consumption. 
GPU-accelerated models (Tensorflow in Regressions and GluonTS) are also more prone to crashes, and may be a source of problems when used. 
If problems persist, post to the GitHub Discussion or Issues. 

Rebooting between heavy uses of multiprocessing can also help reduce the risk of crashing in future model runs.

### Series IDs really need to be unique (or column names need to be all unique in wide data)
Pretty much as it says, if this isn't true, some odd things may happen that shouldn't.

Also if using the Prophet model, you can't have any series named 'ds'

### Short Training History
How much data is 'too little' depends on the seasonality and volatility of the data. 
Minimal training data most greatly impacts the ability to do proper cross validation. Set `num_validations=0` in such cases. 
Since ensembles are based on the test dataset, it would also be wise to set `ensemble=None` if `num_validations=0`.

### Adding regressors and other information
`future_` regressor, to make it clear this is data that will be know with high certainy about the future. 
Such data about the future is rare, one example might be number of stores that will be (planned to be) open each given day in the future when forecast sales. 
Generally using regressors is very helpful for separating 'organic' and 'inorganic' patterns. 
'Inorganic' patterns refers to human business decisions that effect the outcome and can be controlled. 
A very common example of those is promotions and sales events. 
The model can learn from the past promotion information to then anticpate the effects of the input planned promotion events. 
Simulation forecasting, described below, is where multiple promotional plans can be tested side-by-side to evaluate effectiveness. 

Only a handful of models support adding regressors, and not all handle multiple regressors. 
The way to provide regressors is in the `wide` style as a pd.Series/pd.Dataframe with a DatetimeIndex. 

Don't know the future? Don't worry, the models can handle quite a lot of parallel time series, which is another way to add information. 
Additional regressors can be passed through as additional time series to forecast as part of df_long. 
Some models here can utilize the additional information they provide to help improve forecast quality. 
To prevent forecast accuracy for considering these additional series too heavily, input series weights that lower or remove their forecast accuracy from consideration.

*an example of regressors:*
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

For models here in the lower level api, confusingly, regression_type="User" must be specified as well as passing future_regressor. Why? This allows the model search to easily try both with and without the regressor, because sometimes the regressor may do more harm than good.

## Simulation Forecasting
Simulation forecasting allows for experimenting with different potential future scenarios to examine the potential effects on the forecast. 
This is done here by passing known values of a `future_regressor` to model `.fit` and then running `.predict` with multiple variations on the `future_regressor` future values. 
By default in AutoTS, when a `future_regressor` is supplied, models that can utilize it are tried both with and without the regressor. 
To enforce the use of future_regressor for simulation forecasting, a few parameters must be supplied as below. They are: `model_list, models_mode, initial_template`.

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
Note, this does not necessarily force the model to place any great value on the supplied features. 
It may be necessary to rerun multiple times until a model with satisfactory variable response is found, 
or to try with a subset of the regressor model list like `['FBProphet', 'GLM', 'ARDL', 'DatepartRegression']`.

## Event Risk Forecasting and Anomaly Detection
Anomaly (or Outlier) Detection is historic and Event Risk Forecasting is forward looking.

Event Risk Forecasting
Generate a risk score (0 to 1, but usually close to 0) for a future event exceeding user specified upper or lower bounds.

Upper and lower limits can be one of four types, and may each be different.
1. None (no risk score calculated for this direction)
2. Float in range [0, 1] historic quantile of series (which is historic min and max at edges) is chosen as limit.
3. A dictionary of {"model_name": x,  "model_param_dict": y, "model_transform_dict": z, "prediction_interval": 0.9} to generate a forecast as the limits
	Primarily intended for simple forecasts like SeasonalNaive, but can be used with any AutoTS model
4. a custom input numpy array of shape (forecast_length, num_series)

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
A limit specified by a forecast can be used to use one type of model to judge the risk of another production model's bounds (here ARIMA) being exceeded. 
This is also useful for visualizing the effectivness of a particular model's probabilistic forecasts. 

Using forecasts as a limit is also a common method of detecting anomalies in historic data - looking for data points that exceeded what a forecast would have expected. 
Forecast_length effects how far ahead each forecast step is. Larger is faster, smaller means tighter accuracy (only the most extreme outliers are flagged). 
`predict_historic` is used for looking back on the training dataset. Use `eval_periods` to look at only a portion. 
```python
lower_limit = {
	"model_name": "ARIMA",
	"model_param_dict": {'p': 1, "d": 0, "q": 1},
	"model_transform_dict": {},
	"prediction_interval": 0.9,
}
```

Anomaly Detection

Multiple methods are available, including use of `forecast_params` which can be used to analyze the historic deviations of an AutoTS forecasting model.

Holiday detection may also pick up events or 'anti-holidays' ie days of low demand. It won't pick up holidays that don't usually have a significant impact.
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
### A Hack for Passing in Parameters (that aren't otherwise available)
There are a lot of parameters available here, but not always all of the options available for a particular parameter are actually used in generated templates. 
Usually, very slow options are left out. If you are familiar with a model, you can try manualy adding those parameter values in for a run in this way... 
To clarify, you can't usually add in entirely new parameters in this way, but you can often pass in new choices for existing parameter values.

1. Run AutoTS with your desired model and export a template.
2. Open the template in a text editor or Excel and manually change the param values to what you want.
3. Run AutoTS again, this time importing the template before running .fit().
4. There is no guarantee it will choose the model with the given params- choices are made based on validation accuracy, but it will at least run it, and if it does well, it will be incorporated into new models in that run (that's how the genetic algorithms work).


### Categorical Data
Categorical data is handled, but it is handled crudely. For example, optimization metrics do not currently include any categorical accuracy metrics. 
For categorical data that has a meaningful order (ie 'low', 'medium', 'high') it is best for the user to encode that data before passing it in, 
thus properly capturing the relative sequence (ie 'low'=1, 'medium'=2, 'high'=3).

### Custom and Unusual Frequencies
Data must be coercible to a regular frequency. It is recommended the frequency be specified as a datetime offset as per pandas documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects 
Some models will support a more limited range of frequencies. 

## Using the Transformers independently
The transformers expect data only in the `wide` shape with ascending date. 
The simplest way to access them is through the [GeneralTransformer](https://winedarksea.github.io/AutoTS/build/html/source/autots.tools.html#autots.tools.transform.GeneralTransformer). 
This takes dictionaries containing strings of the desired transformers and parameters. 

Inverse_transforms get confusing. It can be necessary to inverse_transform the data to get predictions back to a usable space.
Some inverse_transformer only work on 'original' or 'forecast' data immediately following the training period. 
The DifferencedTransformer is one example. 
It can take the last N value of the training data to bring forecast data back to original space, but will not work for just 'any' future period unconnected to training data. 
Some transformers (mostly the smoothing filters like `bkfilter`) cannot be inversed at all, but transformed values are close to original values. 

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

### Note on ~Regression Models
The Regression models are WindowRegression, RollingRegression, UnivariateRegression, MultivariateRegression, and DatepartRegression. 
They are all different ways of reshaping the time series into X and Y for traditional ML and Deep Learning approaches. 
All draw from the same potential pool of models, mostly sklearn and tensorflow models. 

* DatepartRegression is where X is simply the date features, and Y are the time series values for that date. 
* WindowRegression takes an `n` preceeding data points as X to predict the future value or values of the series. 
* RollingRegression takes all time series and summarized rolling values of those series in one massive dataframe as X. Works well for a small number of series but scales poorly. 
* MultivariateRegression uses the same rolling features as above, but considers them one at a time, features for series `i` are used to predict next step for series `i`, with a model trained on all data from all series. This model is now often called by the community a "global forecasting ML model".
* UnivariateRegression is the same as MultivariateRegression but trains an independent model on each series, thus not capable of learning from the patterns of other series. This performs well in horizontal ensembles as it can be parsed down to one series with the same performance on that series. 

Currently `MultivariateRegression` has the (slower) option to utilize a stock GradientBoostingRegressor with quantile loss for probabilistic estimates, while others utilize point to probabilistic estimates.

## Models

| Model                   | Dependencies | Optional Dependencies   | Probabilistic | Multiprocessing | GPU   | Multivariate | Experimental | Use Regressor |
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
|  DMD                    |              |                         |    True       |                 |       | True         |              |               |
|  BasicLinearModel       |              |                         |    True       |                 |       |              |              | True          |
|  TiDE                   | tensorflow   |                         |               |                 | yes   | True         |              |               |
|  NeuralForecast         | NeuralForecast |                       |    True       |                 | yes   | True         |              | True          |
|  TVVAR                  |              |                         |    True       |                 |       | True         |              | True          |
| BallTreeRegressionMotif | sklearn      |                         |    True       |     joblib      |       | True         |              | True          |
|  MotifSimulation        | sklearn.metrics.pairwise |             |    True       |     joblib      |       | True         | True         |               |
|  Greykite               | (deprecated) |                         |    True       |     joblib      |       |              | True         |               |
|  TensorflowSTS          | (deprecated) |                         |    True       |                 | yes   | True         | True         |               |
|  TFPRegression          | (deprecated) |                         |    True       |                 | yes   | True         | True         | True          |
|  ComponentAnalysis      | (deprecated) |                         |               |                 |       | True         | True         | _             |

*nyi = not yet implemented*
* deprecated models are not actively maintained but updates may be requested in issues
