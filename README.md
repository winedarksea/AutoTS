# AutoTS

### Project CATS (Catlin Automated Time Series)
(or maybe eventually: Clustered Automated Time Series)
#### Model Selection for Multiple Time Series

Simple package for comparing and predicting with open-source time series implementations.
For other time series needs, check out the package list here: https://github.com/MaxBenChrist/awesome_time_series_in_python

## Basic Use
```
pip install autots
```
This includes dependencies for basic models, but additonal packages are required for some models and methods.

Input data is expected to come in a 'long' format with three columns: Date (ideally already in pd.DateTime format), Value, and Series ID. the column name for each of these is passed to .fit(). For a single time series, series_id can be = None. 

If your data is already wide (one column for each value), to bring to a long format:
```
df_long = df_wide.melt(id_vars = ['datetime_col_name'], var_name = 'series_id', value_name = 'value')
```


```
from autots.datasets import load_toy_monthly # also: _daily _yearly or _hourly
df_long = load_toy_monthly()

from autots import AutoTS
model = AutoTS(forecast_length = 14, frequency = 'infer',
               prediction_interval = 0.9, ensemble = True, weighted = False,
               max_generations = 5, num_validations = 2, validation_method = 'even')
model = model.fit(df_long, date_col = 'datetime', value_col = 'value', id_col = 'series_id')

# Print the name of the best mode
print(model.best_model['Model'].iloc[0])

prediction = model.predict()
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
model_results = model.initial_results.model_results
```


## Underlying Process
AutoTS works in the following way at present:
* It begins by taking long data and converting it to a wide dataframe with DateTimeIndex
* An initial train/test split is generated where the test is the most recent data, of forecast_length
* A random template of models is generated and tested on the initial train/test
	* Models consist of a pre-transformation step (fill na options, outlier removal options, etc), and algorithm (ie ETS) and model paramters (trend, damped, etc)
* The top models (selected by a combination of SMAPE, MAE, RMSE) are recombined with random mutations for n_generations
* A handful of the best models from this process go to cross validation, where they are re-assessed on new train/test splits.
* The best model in validation is selected as best_model and used in the .predict() method to generate forecasts.

# Caveats and Advice

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
#### Requirements
	fbprophet
	fredapi (example datasets)

Check out `functional_environments.md` for specific versions tested to work.


### Short Training History
How much data is 'too little' depends on the seasonality and volatility of the data. 
But less than half a year of daily data or less than two years of monthly data are both going to be tight. 
Minimal training data most greatly impacts the ability to do proper cross validation. Set num_validations = 0 in such cases. 
Since ensembles are based on the test dataset, it would also be wise to set ensemble = False if num_validations = 0.

### Too much training data.
Too much data is already handled to some extent by 'context_slicer' in the transformations, which tests using less training data. 
That said, large datasets will be slower and more memory intensive, for high frequency data (say hourly) it can often be advisable to roll that up to a higher level (daily, hourly, etc.). 
Rollup can be accomplished by specifying the frequency = your rollup frequency, and then setting the agg_func = 'sum' or 'mean' or other appropriate statistic.

### Lots of NaN in data
Various NaN filling techniques are tested in the transformation. Rolling up data to a lower frequency may also help deal with NaNs.

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

### Custom Metrics
Implementing new metrics is rather difficult. However the internal 'Score' that compares models can easily be adjusted by passing through custom metric weights. 
Higher weighting increases the importance of that metric. 
`metric_weighting = {'smape_weighting' : 9, 'mae_weighting' : 1, 'rmse_weighting' : 5, 'containment_weighting' : 1, 'runtime_weighting' : 0.5}` 
sMAPE is generally the most versatile across multiple series, but doesn't handle forecasts with lots of zeroes well. 
Contaiment measures the percent of test data that falls between the upper and lower forecasts. 

### Custom and Unusual Frequencies
Data must be coercible to a regular frequency. It is recommended the frequency be specified as a datetime offset as per pandas documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects 
Some models will support a more limited range of frequencies. 
#### Tested Frequencies
| Frequency      | Offset Str   | Notes                                        |
| :------------- | :----------: | --------------------------------------------:|
|  Hourly        |     'H'      |                                              |
|  Daily         |     'D'      |                                              |
|  Monthly 01    |     'MS'     |  First day of month                          |
|  Annual        |   'A'/'AS'   | Not yet tolerant to unusual month starts     |
