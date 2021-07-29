## Table of Contents
* [A Simple Example](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#a-simple-example)
* [Validation and Cross Validation](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#validation-and-cross-validation)
* [Another Example](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#another-example)
* [Model Lists](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#model-lists)
* [Deployment](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#deployment-and-template-import-export)
* [Metrics](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#metrics)
* [Installation](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#installation-and-dependency-versioning)
* [Caveats](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#caveats-and-advice)
* [Adding Regressors](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#adding-regressors-and-other-information)
* [Models](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#models)

## Extended Tutorial
There are a number of ways to get a more accurate time series model. AutoTS takes care of a few of these:
1. Pretransforming the data optimally for each model
2. Trying an assortment of different algorithms
3. Trying an assortment of hyperparameters for each algorithm

## Underlying Process
AutoTS works in the following way at present:
* The process begins with data reshaping and basic data handling as needed
* An initial train/test split is generated where the test is the most recent data, of forecast_length
* The initial model template is a combination of transfer learning and randomly generated models. This is tested on the initial train/test
* Models consist of a pre-transformation step (fill na options, outlier removal options, etc), and algorithm (ie ETS) and model paramters (trend, damped, ...)
* The top models (selected by a combination of metrics) are recombined with random mutations for n_generations
* A percentage of the best models from this process go to cross validation, where they are re-assessed on new train/test splits.
* If used, horizontal ensembling uses the validation data to choose the best model for each series.
* The best model or ensemble in validation is selected as best_model and used in the `.predict()` method to generate forecasts.

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

# Print the name of the best model
print(model)
```

#### Import of data
There are two shapes/styles of `pandas.DataFrame` which are accepted. 
The first is *long* data, like that out of an aggregated sales-transaction table containing three columns identified to `.fit()` as `date_col {pd.Datetime}, value_col {the numeric or categorical data of interest}, and id_col {id string, if multiple series are provided}`. 
Alternatively, the data may be in a *wide* format where the index is a `pandas.DatetimeIndex`, and each column is a distinct data series.  

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

### Validation and Cross Validation
Firstly, all models are initially validated on the most recent piece of data. This is done because the most recent data will generally most closely resemble the forecast future. 
With very small datasets, there may be not be enough data for cross validation, in which case `num_validations` may be set to 0. This can also speed up quick tests. 

Cross validation helps assure that the optimal model is stable over the dynamics of a time series. 
Cross validation can be tricky in time series data due to the necessity of preventing data leakage from future data points. 
Here, two methods of cross validation are in place, `'even'` and '`backwards'`.

**Even** cross validation slices the data into equal chunks. For example, `num_validations=3` would split the data into equal, progressive thirds (less the original validation sample). The final validation results would then include four pieces, the results on the three cross validation samples as well as the original validation sample. 

**Backwards** cross validation works backwards from the most recent data. First the most recent forecast_length samples are taken, then the next most recent forecast_length samples, and so on. This makes it more ideal for smaller or fast-changing datasets. 

**Seasonal** validation is supplied as `'seasonal n'` ie `'seasonal 364'`. It trains on the most recent data as usual, then valdations are `n` periods back from the datetime of the forecast would be. 
For example with daily data, forecasting for a month ahead, and `n=364`, the first test might be on May 2020, with validation on June 2019 and June 2018, the final forecast then of June 2020.

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
    ensemble='simple',
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
# model.best_model.to_string()
```

Probabilistic forecasts are *available* for all models, but in many cases are just data-based estimates in lieu of model estimates. 
```python
upper_forecasts_df = prediction.upper_forecast
lower_forecasts_df = prediction.lower_forecast
```

### Model Lists
By default, most available models are tried. For a more limited subset of models, a custom list can be passed in, or more simply, a string, one of `'probabilistic', 'multivariate', 'fast', 'superfast', or 'all'`.

A table of all available models is available further below.

On large multivariate series, `TSFreshRegressor`, `DynamicFactor` and `VARMAX` can be impractically slow.

## Deployment and Template Import/Export
Take a look at the [production_example.py](https://github.com/winedarksea/AutoTS/blob/master/production_example.py)

Many models can be reverse engineered with (relative) simplicity outside of AutoTS by placing the choosen parameters into Statsmodels or other underlying package. 
There are some advantages to deploying within AutoTS using a reduced starting template. Following the model training, the top models can be exported to a `.csv` or `.json` file, then on next run only those models will be tried. 
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

### Metrics
There are a number of available metrics, all combined together into a 'Score' which evaluates the best model. The 'Score' that compares models can easily be adjusted by passing through custom metric weights dictionary. 
Higher weighting increases the importance of that metric, while 0 removes that metric from consideration. Weights should be 0 or positive numbers, and can be floats as well as integers. 
This weighting is not to be confused with series weighting, which effects how equally any one metric is applied to all the series. 
```python
metric_weighting = {
	'smape_weighting' : 10,
	'mae_weighting' : 1,
	'rmse_weighting' : 5,
	'containment_weighting' : 1,
	'runtime_weighting' : 0,
	'spl_weighting': 1,
	'contour_weighting': 0,
}

model = AutoTS(
	forecast_length=forecast_length,
	frequency='infer',
	metric_weighting=metric_weighting,
)
```		
It is wise to usually use several metrics. I often find the best sMAPE model, for example, is only slightly better in sMAPE than the next place model, but that next place model has a much better MAE and RMSE. 
			
**Warning**: weights are not fully balanced 1 - 1 - 1. As such it is usually best to place your favorite metric an order of magnitude or more above the others. 

`sMAPE` is generally the most versatile metric across multiple series, but doesn't handle forecasts with lots of zeroes well. 

`SPL` is *Scaled Pinball Loss* and is the optimal metric for assessing upper/lower quantile forecast accuracies.

`Containment` measures the percent of test data that falls between the upper and lower forecasts, and is more human readable than SPL.

`Contour` is a unique measure. It is designed to help choose models which when plotted visually appear similar to the actual. As such, it measures the % of points where the forecast and actual both went in the same direction, either both up or both down, but *not* the magnitude of that difference. Does not work with forecast_length=1. 

The contour metric is useful as it encourages 'wavy' forecasts, ie, not flat line forecasts. Although flat line naive or linear forecasts can sometimes be very good models, they "don't look like they are trying hard enough" to some managers, and using contour favors non-flat forecasts that (to many) look like a more serious model.

## Installation and Dependency Versioning
`pip install autots`
### Requirements:
	Python >= 3.6
	numpy
	pandas
	sklearn 	>= 0.20.0
				>= 0.23.0 (PoissonReg)
				>= 0.24.0 (OrdinalEncoder handle_unknown)
	statsmodels

Of these, numpy and pandas are critical. 
Limited functionality should exist without scikit-learn. 
	* Sklearn needed for categorical to numeric, some detrends/transformers, horizontal generalization, numerous models
Full functionality should be maintained without statsmodels, albeit with fewer available models. 

Prophet, Greykite, and mxnet/GluonTS are packages which tend to be finicky about installation on some systems.

`pip install autots['additional']`
### Optional Requirements
	holidays
	prophet
	gluonts (requires mxnet)
	mxnet (mxnet-mkl, mxnet-cu91, mxnet-cu101mkl, etc.)
	tensorflow >= 2.0.0
	lightgbm
	xgboost
	psutil
	tensorflow-probability
	fredapi
	greykite

#### Safest bet for installation:
```shell
# create a conda or venv environment
conda create -n openblas python=3.9
conda activate openblas

python -m pip install numpy scipy scikit-learn statsmodels tensorflow lightgbm --exists-action i

python -m pip install pystan prophet --exists-action i  # conda-forge option below works more easily, --no-deps to pip install prophet if this fails
python -m pip install mxnet --exists-action i     # check the mxnet documentation for more install options, also try pip install mxnet --no-deps
python -m pip install gluonts --exists-action i
python -m pip install greykite --exists-action i  # try running again with --no-deps if first try fails. The failing dep is often optional...
python -m pip install --upgrade numpy pandas --exists-action i  # mxnet likes to (pointlessly seeming) install old versions of numpy

python -m pip install autots --exists-action i
```

### Hardware Acceleration with Intel CPU and Nvidia GPU for Ubuntu/Windows
If you are on an Intel CPU, download Anaconda or Miniconda. For AMD/ARM/etc use a venv environment and pip which will use OpenBLAS. 
Intel MKL is included with `anaconda` and offers significant performance gain for Intel CPUs. Use of the Intel conda channel sometimes is necessary. 

(install Visual Studio if on Windows for C compilers)

If you have an Nvidia GPU and plan to use the GPU-accelerated models, download NVIDIA CUDA and CuDNN. 

You can check if your system is using mkl, OpenBLAS, or none with `numpy.show_config()`. Generally recommended that you double-check this after installing new packages to make sure you haven't broken the LINPACK connection. 

On Linux systems, apt-get/yum (rather than pip) installs of numpy/pandas *may* install faster/more stable compilations. 
Linux will also require `sudo apt install build-essential` for some packages.

#### Some conda

```shell
conda create -n timeseries python=3.9
conda activate timeseries

# for simplicity: 
conda install anaconda
# elsewise: 
conda install numpy scipy scikit-learn statsmodels  # -c conda-forge is sometimes a version ahead of main channel

conda install -c conda-forge prophet
pip install mxnet     # check the mxnet documentation for more install options, also try pip install mxnet --no-deps
pip install gluonts
pip install lightgbm tensorflow
conda update anaconda

pip install autots
```
#### Intel conda channel installation (fastest, also, more prone to bugs)
https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html
```shell
# create the environment. Intelpy compatability is often a version or two behind latest py
conda create -n intelpy -c intel python=3.7 intelpython3_full
conda activate intelpy

# install additional packages as desired
conda install -c intel statsmodels lightgbm tensorflow
conda install -c intel tensorflow-probability
python -m pip install mxnet
python -m pip install gluonts
# conda install -c anaconda tornado pystan  # may be necessary for fbprophet
python -m pip install prophet

pip install autots

# also checkout daal4py: https://intelpython.github.io/daal4py/sklearn.html
# pip install intel-tensorflow-avx512  and conda install tensorflow-mkl
# MKL_NUM_THREADS, USE_DAAL4PY_SKLEARN=1
```

## Caveats and Advice

### Series IDs really need to be unique (or column names need to be all unique in wide data)
Pretty much as it says, if this isn't true, some odd things may happen that shouldn't.

Also if using the model Prophet models, you can't have any columns named 'ds'

### Short Training History
How much data is 'too little' depends on the seasonality and volatility of the data. 
Minimal training data most greatly impacts the ability to do proper cross validation. Set `num_validations=0` in such cases. 
Since ensembles are based on the test dataset, it would also be wise to set `ensemble=None` if `num_validations=0`.

### Too much training data.
Too much data is already handled to some extent by the `Slice` Transformer. 
That said, large datasets will be slower and more memory intensive, for high frequency data (say hourly) it can often be advisable to roll that up to a higher level (daily, hourly, etc.). 
Rollup can be accomplished by specifying the frequency = your rollup frequency, and then setting the `agg_func=np.sum` or 'mean' or other appropriate statistic.

### Lots of NaN in data
Various NaN filling techniques are tested in the transformation. Rolling up data to a less-frequent frequency may also help deal with NaNs.

### Adding regressors and other information
`future_` regressor, to make it clear this is data that will be know with high certainy about the future. 
Such data about the future is rare, one example might be number of stores that will be (planned to be) open each given day in the future when forecast sales. 
Only a handful of models support adding regressors, and not all handle multiple regressors. 
The recommended way to provide regressors is as a pd.Series/pd.Dataframe with a DatetimeIndex. 

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
print(f"Was a model choosen that used the regressor? {model.used_regressor_check}")
```

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

## Models

| Model                   | Dependencies | Optional Dependencies   | Probabilistic | Multiprocessing | GPU   | Multivariate | Experimental | Use Regressor |
| :-------------          | :----------: | :---------------------: | :-----------  | :-------------- | :---- | :----------: | :----------: | :-----------: |
|  ZeroesNaive            |              |                         |               |                 |       |              |              |               |
|  LastValueNaive         |              |                         |               |                 |       |              |              |               |
|  AverageValueNaive      |              |                         |    True       |                 |       |              |              |               |
|  SeasonalNaive          |              |                         |               |                 |       |              |              |               |
|  GLS                    | statsmodels  |                         |               |                 |       | True         |              |               |
|  GLM                    | statsmodels  |                         |               |     joblib      |       |              |              | True          |
|  ETS - Exponential Smoothing | statsmodels  |                         |               |     joblib      |       |              |              |               |
|  UnobservedComponents   | statsmodels  |                         |               |     joblib      |       |              |              | True          |
|  ARIMA                  | statsmodels  |                         |    True       |     joblib      |       |              |              | True          |
|  VARMAX                 | statsmodels  |                         |    True       |                 |       | True         |              |               |
|  DynamicFactor          | statsmodels  |                         |    True       |                 |       | True         |              | True          |
|  VECM                   | statsmodels  |                         |               |                 |       | True         |              | True          |
|  VAR                    | statsmodels  |                         |    True       |                 |       | True         |              | True          |
|  FBProphet              | fbprophet    |                         |    True       |     joblib      |       |              |              | True          |
|  GluonTS                | gluonts, mxnet |                       |    True       |                 | yes   | True         |              |               |
|  RollingRegression      | sklearn      | lightgbm, tensorflow    |               |     sklearn     | some  | True         |              | True          |
|  WindowRegression       | sklearn      | lightgbm, tensorflow    |               |     sklearn     | some  | True         |              |               |
|  DatepartRegression     | sklearn      | lightgbm, tensorflow    |               |     sklearn     | some  |              |              | True          |
|  UnivariateRegression   | sklearn      | lightgbm, tensorflow    |               |     sklearn     | some  |              | True         | True          |
|  MotifSimulation        | sklearn.metrics.pairwise |             |    True       |     joblib      |       | True*        | True         |               |
|  TensorflowSTS          | tensorflow_probability   |             |    True       |                 | yes   | True         | True         |               |
|  TFPRegression          | tensorflow_probability   |             |    True       |                 | yes   | True         | True         | True          |
|  ComponentAnalysis      | sklearn      |                         |               |                 |       | True         | True         |               |
|  TSFreshRegressor       | tsfresh, sklearn |                     |               |                 |       |              | True         |               |

