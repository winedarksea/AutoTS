## Extended Tutorial
There are a number of ways to get a more accurate time series model. AutoTS takes care of a few of these:
1. Pretransforming the data optimally for each model
2. Trying an assortment of different algorithms
3. Trying an assortment of hyperparameters for each algorithm

## Underlying Process
AutoTS works in the following way at present:
* It begins by taking long data and converting it to a wide dataframe with DateTimeIndex
* An initial train/test split is generated where the test is the most recent data, of forecast_length
* A random template of models is generated and tested on the initial train/test
* Models consist of a pre-transformation step (fill na options, outlier removal options, etc), and algorithm (ie ETS) and model paramters (trend, damped, etc)
* The top models (selected by a combination of metrics) are recombined with random mutations for n_generations
* A handful of the best models from this process go to cross validation, where they are re-assessed on new train/test splits.
* The best model in validation is selected as best_model and used in the `.predict()` method to generate forecasts.

### A simple example
```
# also: _hourly, _daily, _weekly, or _yearly
from autots.datasets import load_monthly

df_long = load_monthly()

from autots import AutoTS

model = AutoTS(
    forecast_length=3,
    frequency='infer',
    ensemble='simple',
    drop_data_older_than_periods=240,
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

```

#### You can tailor the process in a few ways...
The simplest way to improve accuracy is to increase the number of generations `max_generations=15`. Each generation tries new models, taking additional time but improving the accuracy. The nature of genetic algorithms, however, means there is no consistent improvement for each generation, and large number of generations will often only result in minimal performance gains.

Another approach that may improve accuracy is to set `ensemble='all'`. Ensemble parameter expects a single string, and can for example be `'simple,dist'`, or `'horizontal'`. As this means storing more details of every model, this takes more time and memory.

A handy parameter for when your data is expected to always be 0 or greater (such as unit sales) is to set `no_negatives=True`. This forces forecasts to be greater than or equal to 0. 
A similar function is `constraint=2.0`. What this does is prevent the forecast from leaving historic bounds set by the training data. In this example, the forecasts would not be allowed to go above `max(training data) + 2.0 * st.dev(training data)`, as well as the reverse on the minimum side. A constraint of `0` would constrain forecasts to historical mins and maxes. 

Another convenience function is `drop_most_recent=1` specifing the number of most recent periods to drop. This can be handy with monthly data, where often the most recent month is incomplete. 
`drop_data_older_than_periods` provides similar functionality but drops the oldest data to speed up the process on large datasets. 
`remove_leading_zeroes=True` is useful for data where leading zeroes represent a process which has not yet started.

When working with many time series, it can be helpful to take advantage of `subset=100`. Subset specifies the interger number of time series to test models on, and can be useful with many related time series (1000's of customer's sales). Usually the best model on a 100 related time series is very close to that tested on many thousands (or more) of series. This speeds up the model process in these cases, but does not work with `horizontal` ensemble types.

Subset takes advantage of weighting, more highly-weighted series are more likely to be selected. Weighting is used with multiple time series to tell the evaluator which series are most important. Series weights are assumed to all be equal to 1, values need only be passed in when a value other than 1 is desired. 
Note for weighting, larger weights = more important.

### Validation and Cross Validation
Firstly, all models are initially validated on the most recent piece of data. This is done because the most recent data will generally most closely resemble the forecast future. 
With very small datasets, there may be not be enough data for cross validation, in which case `num_validations` may be set to 0. This can also speed up quick tests. 

Cross validation helps assure that the optimal model is stable over the dynamics of a time series. 
Cross validation can be tricky in time series data due to the necessity of preventing data leakage from future data points. 
Here, two methods of cross validation are in place, `'even'` and '`backwards'`.

**Even** cross validation slices the data into equal chunks. For example, `num_validations=3` would split the data into equal, progressive thirds (less the original validation sample). The final validation results would then include four pieces, the results on the three cross validation samples as well as the original validation sample. 

**Backwards** cross validation works backwards from the most recent data. First the most recent forecast_length samples are taken, then the next most recent forecast_length samples, and so on. This makes it more ideal for smaller or fast-changing datasets. 

**Seasonal** validation is supplied as `'seasonal n'` ie `'seasonal 364'`. It trains on the most recent data as usual, then valdations are `n` periods back from the datetime of the forecast would be. For example with daily data, forecasting for a month ahead, and `n=364`, the first test might be on May 2020, with validation on June 2019 and June 2018, the final forecast then of June 2020.

Only a subset of models are based from initial validation to cross validation. The number of models is set such as `models_to_validate=10`. If a float in 0 to 1 is provided, it is treated as a % of models to select. If you suspect your most recent data is not fairly representative of the whole, it would be a good idea to increase this parameter. 

### A more detailed example:
Here, we are forecasting the traffice along Interstate 94 between Minneapolis and St Paul in Minnesota. This is a great dataset to demonstrate a recommended way of including external variables - by including them as time series with a lower weighting. 
Here weather data is included - winter and road construction being the major influencers for traffic and will be forecast alongside the traffic volume. These additional series carry information to models such as `RollingRegression`, `VARMAX`, and `VECM`. 

Also seen in use here is the `model_list`. 

```
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
    models_to_validate=15,
    drop_most_recent=1,
	n_jobs='auto',
)

model = model.fit(
    df_wide,
    weights=weights_hourly,
)

prediction = model.predict()
forecasts_df = prediction.forecast
# model.best_model.to_string()
```

Probabilistic forecasts are *available* for all models, but in many cases are just data-based estimates in lieu of model estimates, so be careful. 
```
upper_forecasts_df = prediction.upper_forecast
lower_forecasts_df = prediction.lower_forecast
```

### Model Lists
By default, most available models are tried. For a more limited subset of models, a custom list can be passed in, or more simply, a string, one of `'probabilistic', 'multivariate', 'fast', 'superfast', or 'all'`.

A table of all available models is available further below.

On large multivariate series, `TSFreshRegressor`, `DynamicFactor` and `VARMAX` can be impractically slow.

## Deployment and Template Import/Export
Many models can be reverse engineered with (relative) simplicity outside of AutoTS by placing the choosen parameters into Statsmodels or other underlying package. 
There are some advantages to deploying within AutoTS using a reduced starting template. Following the model training, the top models can be exported to a `.csv` or `.json` file, then on next run only those models will be tried. 
This allows for improved fault tolerance (by relying not on one, but several possible models and underlying packages), and some flexibility in switching models as the time series evolve. 
One thing to note is that, as AutoTS is still under development, template formats are likely to change and be incompatible with future package versions.
```
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
```
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

## Installation and Dependency Versioning
`pip install autots`
### Requirements:
	Python >= 3.5
	numpy
	pandas
	sklearn 	>= 0.20.0 (ColumnTransformer)
				>= 0.23.0 (PoissonReg)
	statsmodels

Of these, numpy and pandas are critical. 
Limited functionality should exist without scikit-learn. 
Full functionality should be maintained without statsmodels, albeit with fewer available models. 

`pip install autots['additional']`
### Optional Requirements
	holidays
	fbprophet
	gluonts (requires mxnet)
	mxnet (mxnet-mkl, mxnet-cu91, mxnet-cu101mkl, etc.)
	tensorflow >= 2.0.0
	lightgbm
	xgboost
	psutil

#### Experimental & Dev Requirements
	tensorflow-probability
	fredapi
	tsfresh

### Hardware Acceleration with Intel CPU and Nvidia GPU for Ubuntu/Windows
Download Anaconda or Miniconda.

(install Visual Studio if on Windows for C compilers)

If you have an Nvidia GPU, download NVIDIA CUDA and CuDNN. 
Intel MKL is included with `anaconda` and offers significant performance gain for Intel CPUs.
You can check if your system is using mkl with `numpy.show_config()`. 
If you are on an AMD CPU, you do **not** want to be using MKL, OpenBLAS is better. 
On Linux ARM-based systems, apt-get/yum (rather than pip) installs of numpy/pandas *may* install faster compilations.  
```
conda create -n timeseries python=3.8
conda activate timeseries

# for simplicity: 
conda install anaconda
# elsewise: 
conda install numpy scipy
conda install -c conda-forge scikit-learn
pip install statsmodels

# check the mxnet documentation for various flavors of mxnet available
pip install mxnet
pip install gluonts
conda update anaconda
conda install -c conda-forge fbprophet
pip install tensorflow
pip install tensorflow-probability
```

## Caveats and Advice

### Short Training History
How much data is 'too little' depends on the seasonality and volatility of the data. 
Minimal training data most greatly impacts the ability to do proper cross validation. Set `num_validations=0` in such cases. 
Since ensembles are based on the test dataset, it would also be wise to set `ensemble=None` if `num_validations=0`.

### Too much training data.
Too much data is already handled to some extent by `'context_slicer'` in the transformations, which tests using less training data. 
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

### Categorical Data
Categorical data is handled, but it is handled crudely. For example, optimization metrics do not currently include any categorical accuracy metrics. 
For categorical data that has a meaningful order (ie 'low', 'medium', 'high') it is best for the user to encode that data before passing it in, 
thus properly capturing the relative sequence (ie 'low'=1, 'medium'=2, 'high'=3).

### Custom and Unusual Frequencies
Data must be coercible to a regular frequency. It is recommended the frequency be specified as a datetime offset as per pandas documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects 
Some models will support a more limited range of frequencies. 

#### Tested Frequencies

| Frequency      | Offset Str   | Notes                                        |
| :------------- | :----------: | --------------------------------------------:|
|  Hourly        |     'H'      |                                              |
|  Daily         |     'D'      |                                              |
|  Monthly 01    |     'MS'     |  First day of month                          |
|  Annual        |   'A'/'AS'   | 											   |
|  Weekly        |     'W'      | 											   |

## Models

| Model                   | Dependencies | Optional Dependencies   | Probabilistic | Multiprocessing | GPU   | Multivariate | Experimental |
| :-------------          | :----------: | :---------------------: | :-----------  | :-------------- | :---- | :----------: | :----------: |
|  ZeroesNaive            |              |                         |               |                 |       |              |              |
|  LastValueNaive         |              |                         |               |                 |       |              |              |
|  AverageValueNaive      |              |                         |    True       |                 |       |              |              |
|  SeasonalNaive          |              |                         |               |                 |       |              |              |
|  GLS                    | statsmodels  |                         |               |                 |       | True         |              |
|  GLM                    | statsmodels  |                         |               |     joblib      |       |              |              |
|  ETS                    | statsmodels  |                         |               |     joblib      |       |              |              |
|  UnobservedComponents   | statsmodels  |                         |               |                 |       |              |              |
|  ARIMA                  | statsmodels  |                         |    True       |     joblib      |       |              |              |
|  VARMAX                 | statsmodels  |                         |    True       |                 |       | True         |              |
|  DynamicFactor          | statsmodels  |                         |    True       |                 |       | True         |              |
|  VECM                   | statsmodels  |                         |               |                 |       | True         |              |
|  VAR                    | statsmodels  |                         |    True       |                 |       | True         |              |
|  FBProphet              | fbprophet    |                         |    True       |     joblib      |       |              |              |
|  GluonTS                | gluonts, mxnet |                       |    True       |                 | yes   | True         |              |
|  RollingRegression      | sklearn      | lightgbm, tensorflow    |               |     sklearn     | some  | True         |              |
|  WindowRegression       | sklearn      | lightgbm, tensorflow    |               |     sklearn     | some  | True         |              |
|  MotifSimulation        | sklearn.metrics.pairwise |             |    True       |                 |       | True*        | True         |
|  TensorflowSTS          | tensorflow_probability   |             |    True       |                 | yes   | True         | True         |
|  TFPRegression          | tensorflow_probability   |             |    True       |                 | yes   | True         | True         |
|  ComponentAnalysis      | sklearn      |                         |               |                 |       | True         | True         |
|  TSFreshRegressor       | tsfresh, sklearn |                     |               |                 |       |              | True         |
