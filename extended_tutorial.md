# Extended Tutorial
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
* The top models (selected by a combination of SMAPE, MAE, RMSE) are recombined with random mutations for n_generations
* A handful of the best models from this process go to cross validation, where they are re-assessed on new train/test splits.
* The best model in validation is selected as best_model and used in the .predict() method to generate forecasts.

### A simple example
```
from autots.datasets import load_toy_monthly # also: _daily _yearly or _hourly
df_long = load_toy_monthly()

from autots import AutoTS
model = AutoTS(forecast_length = 3, frequency = 'infer',
			   ensemble = False, drop_data_older_than_periods = 240,
               max_generations = 5, num_validations = 2)
model = model.fit(df_long, date_col = 'datetime', value_col = 'value', id_col = 'series_id')

# Print the name of the best model
print(model)
```

If your data is already wide (one column for each value), to bring to a long format:
```
df_long = df_wide.melt(id_vars = ['datetime_col_name'], var_name = 'series_id', value_name = 'value')
```

#### You can tailor the process in a few ways...
The simplest thing is to increase the number of generations `max_generations = 15`. Each generation tries new models, taking additional time but improving the accuracy. The nature of genetic algorithms, however, means there is no consistent improvement for each generation, and large number of generations will often only result in minimal performance gains.

Another approach that may improve accuracy is to set `ensemble = True`. As this means storing and processing the result of every model, this takes much more time and memory, and *is not recommended with more than a few generations*.

A handy parameter for when your data is expected to always be 0 or greater (such as unit sales) is to set `no_negatives = True`. This forces forecasts to be greater than or equal to 0.

Another convenience function is `drop_most_recent = 1` specifing the number of most recent periods to drop. This can be handy with monthly data, where often the most recent month is incomplete. 
`drop_data_older_than_periods` provides similar functionality but drops the oldest data.

When working with many time series, it can be helpful to take advantage of `subset = 100`. Subset specifies the interger number of time series to test models on, and can be useful with many related time series (1000's of customer's sales). Usually the best model on a 100 related time series is very close to that tested on many thousands (or more) of series. This speeds up the model process in these cases.

Subset takes advantage of weighting, more highly-weighted series are more likely to be selected. Weighting is used with multiple time series to tell the evaluator which series are most important. Series weights are assumed to all be equal to 1, values need only be passed in when a value other than 1 is desired. 
Note for weighting, `weighted = True` and `weights = dict` must both be passed into the model. Larger weights = more important.

### Validation and Cross Validation
Firstly, all models are initially validated on the most recent piece of data. This is done because the most recent data will generally most closely resemble the forecast future. 
With very small datasets, there may be not be enough data for cross validation, in which case `num_validations` may be set to 0. This can also speed up quick tests. 

Cross validation helps assure that the optimal model is stable over the dynamics of a time series. 
Cross validation can be tricky in time series data due to the necessity of preventing data leakage from future data points. 
Here, two methods of cross validation are in place, `'even'` and '`backwards'`.

**Even** cross validation slices the data into equal chunks. For example, `num_validations=3` would split the data into equal, progressive thirds (less the original validation sample). The final validation results would then include four pieces, the results on the three cross validation samples as well as the original validation sample. 

**Backwards** cross validation works backwards from the most recent data. First the most recent forecast_length samples are taken, then the next most recent forecast_length samples, and so on. This makes it more ideal for smaller or fast-changing datasets. 

Only a subset of models are based from initial validation to cross validation. The number of models is set such as `models_to_validate = 10`. If you suspect your most recent data is not fairly representative of the whole, it would be a good idea to increase this parameter. 

### A more detailed example:
Here, we are forecasting the traffice along Interstate 94 between Minneapolis and St Paul in (lovely) Minnesota. This is a great dataset to demonstrate a recommended way of including external variables - by including them as time series with a lower weighting. 
Here weather data is included - winter and road construction being the major influencers for traffic and will be forecast alongside the traffic volume. This carries information to models such as RollingRegression, VARMAX, and VECM. 

Also seen in use here is the `model_list`. By default, most available models are tried. For a more limited subset of models, a custom list can be passed in, or more simply, a string, one of 'probabilistic', 'multivariate', 'fast', 'superfast', or 'all'.

```
from autots.datasets import load_toy_hourly

df_long = load_toy_hourly()

weights_hourly = {'traffic_volume': 20} # all other series assumed to be weight of 1

model_list = ['ZeroesNaive', 'LastValueNaive', 'MedValueNaive', 'GLS',
              'ETS',  'RollingRegression', 'UnobservedComponents', 'VARMAX', 'VECM']

from autots import AutoTS
model = AutoTS(forecast_length = 73, frequency = 'infer',
               prediction_interval = 0.95, ensemble = False, weighted = True,
               max_generations = 5, num_validations = 2, validation_method = 'even',
               model_list = model_list, models_to_validate = 15,
               drop_most_recent = 1)
		
model = model.fit(df_long, date_col = 'datetime', value_col = 'value', id_col = 'series_id', weights = weights_hourly) # and weighted = True

prediction = model.predict()
# point forecasts dataframe
forecasts_df = prediction.forecast
```
Probabilistic forecasts are *available* for all models, but in many cases are just general estimates in lieu of model estimates, so be careful. 
FBProphet, GluonTS, and ARIMA are all sources of probabilistic intervals.
```
upper_forecasts_df = prediction.upper_forecast
lower_forecasts_df = prediction.lower_forecast
```

## Deployment and Template Import/Export
Many models can be reverse engineered with relative simplicity outside of AutoTS by placing the choosen parameters into Statsmodels or other underlying package. 
There are some advantages to deploying within AutoTS using a reduced starting template. Following the model training, the top models can be exported to a .csv or .json file, then on next run only those models will be tried. 
This allows for improved fault tolerance (by relying not on one, but several possible models and underlying packages), and some flexibility in switching models as the time series evolve.
```
# after fitting an AutoTS model
example_filename = "example_export.csv" # .csv/.json
model.export_template(example_filename, models = 'best', n = 15, max_per_model_class = 3)

# on new training
model = AutoTS(forecast_length = forecast_length, frequency = 'infer', max_generations = 0, num_validations = 0, verbose = 0)
model = model.import_template(example_filename, method = 'only') # method = 'add on'
print("Overwrite template is: {}".format(str(model.initial_template)))
```

### Metrics
There are a number of available metrics, all combined together into a 'Score' which evaluates the best model. The 'Score' that compares models can easily be adjusted by passing through custom metric weights dictionary. 
Higher weighting increases the importance of that metric, while 0 removes that metric from consideration. Weights should be 0 or positive numbers, and can be floats as well as integers. 
This weighting is not to be confused with series weighting, which effects how equally any one metric is applied to all the series. 
```
metric_weighting = {'smape_weighting' : 10, 'mae_weighting' : 1, 'rmse_weighting' : 5, 
					'containment_weighting' : 1, 'runtime_weighting' : 0,
					'lower_mae_weighting': 0, 'upper_mae_weighting': 0, 'contour_weighting': 3}

model = AutoTS(forecast_length = forecast_length, frequency = 'infer', metric_weighting = metric_weighting)
```		
It is wise to usually use several metrics. I often find the best sMAPE model, for example, is only slightly better in sMAPE than the next place model, but that next place model has a much better MAE and RMSE. 
			
**Warning**: weights are not (yet) well balanced 1 - 1 - 1. As such it is usually best to place your favorite metric an order of magnitude or more above the others. 

`sMAPE` is generally the most versatile across multiple series, but doesn't handle forecasts with lots of zeroes well. 

`Containment` measures the percent of test data that falls between the upper and lower forecasts. As containment would tend to drive towards massive forecast ranges, `lower_mae` and `upper_mae`, the MAE on the upper and lower forecasts, are available. `Containment` and `upper/lower_mae` counteract each other and help balance the assessement of probabilistic forecasts.

`Contour` is another unique measure. It is designed to help choose models which when plotted visually appear similar to the actual. As such, it measures the % of points where the forecast and actual both went in the same direction, either both up or both down, but *not* the magnitude of that difference.

## Installation and Dependency Versioning
`pip install autots`
#### Requirements:
	numpy
	Python >= 3.5 (typing) >= 3.6 (GluonTS)
	pandas
	sklearn >= 0.20.0 (ColumnTransformer)
	statsmodels
	holidays

`pip install autots['additional']`
#### Optional Requirements
	fbprophet
	fredapi
	tsfresh
	mxnet==1.4.1 (mxnet-mkl, mxnet-cu91, mxnet-cu92mkl, etc.)
	gluonts

If using Anaconda, fbprophet is easier to install with `conda install -c conda-forge fbprophet`

Check out `functional_environments.md` for specific versions tested to work.

## Caveats and Advice

### Short Training History
How much data is 'too little' depends on the seasonality and volatility of the data. 
Minimal training data most greatly impacts the ability to do proper cross validation. Set num_validations = 0 in such cases. 
Since ensembles are based on the test dataset, it would also be wise to set ensemble = False if num_validations = 0.

### Too much training data.
Too much data is already handled to some extent by 'context_slicer' in the transformations, which tests using less training data. 
That said, large datasets will be slower and more memory intensive, for high frequency data (say hourly) it can often be advisable to roll that up to a higher level (daily, hourly, etc.). 
Rollup can be accomplished by specifying the frequency = your rollup frequency, and then setting the agg_func = 'sum' or 'mean' or other appropriate statistic.

### Lots of NaN in data
Various NaN filling techniques are tested in the transformation. Rolling up data to a less-frequent frequency may also help deal with NaNs.

### More than one preord regressor
'Preord' regressor stands for 'Preordained' regressor, to make it clear this is data that will be know with high certainy about the future. 
Such data about the future is rare, one example might be number of stores that will be (planned to be) open each given day in the future when forecast sales. 
Since many algorithms do not handle more than one regressor, only one is handled here. If you would like to use more than one, 
manually select the best variable or use dimensionality reduction to reduce the features to one dimension. 
However, the model can handle quite a lot of parallel time series. Additional regressors can be passed through as additional time series to forecast. 
The regression models here can utilize the information they provide to help improve forecast quality. 
To prevent forecast accuracy for considering these additional series too heavily, input series weights that lower or remove their forecast accuracy from consideration.

### Categorical Data
Categorical data is handled, but it is handled poorly. For example, optimization metrics do not currently include any categorical accuracy metrics. 
For categorical data that has a meaningful order (ie 'low', 'medium', 'high') it is best for the user to encode that data before passing it in, 
thus properly capturing the relative sequence (ie 'low' = 1, 'medium' = 2, 'high' = 3).

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
