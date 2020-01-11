# AutoTS

### Project CATS (Catlin Automated Time Series)
(or maybe eventually: Clustered Automated Time Series)
#### Model Selection for Multiple Time Series

Simple package for comparing and predicting with open-source time series implementations.
For other time series needs, check out the package list here: https://github.com/MaxBenChrist/awesome_time_series_in_python

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


## Basic Use
Input data is expected to come in a 'long' format with three columns: Date (ideally already in pd.DateTime format), Value, and Series ID. the column name for each of these is passed to .fit(). For a single time series, series_id can be = None. 

```
from autots.datasets import load_toy_daily
df_long = load_toy_daily()

from autots import AutoTS
model = AutoTS(forecast_length = 14, frequency = 'infer',
               prediction_interval = 0.9, ensemble = True, weighted = False,
               max_generations = 5, num_validations = 2, validation_method = 'even')
model = model.fit(df_long, date_col = 'date', value_col = 'value', id_col = 'series_id' )

# Print the name of the best mode
print(model.best_model['Model'].iloc[0])

prediction = model.predict()
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
model_results = model.main_results.model_results
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

## Caveats and Advice

#### Short Training History
How much data is 'too little' depends on the seasonality and volatility of the data. 
But less than half a year of daily data or less than two years of monthly data are both going to be tight. 
Minimal training data most greatly impacts the ability to do proper cross validation. Set num_validations = 0 in such cases. 
Since ensembles are based on the test dataset, it would also be wise to set ensemble = False if num_validations = 0.

#### Too much training data.
Too much data is already handled to some extent by 'context_slicer' in the transformations, which tests using less training data. 
That said, large datasets will be slower and more memory intensive, for high frequency data (say hourly) it can often be advisable to roll that up to a higher level (daily, hourly, etc.). 
Rollup can be accomplished by specifying the frequency = your rollup frequency, and then setting the agg_func = 'sum' or 'mean' or other appropriate statistic.

#### Lots of NaN in data
Various NaN filling techniques are tested in the transformation. Rolling up data to a lower frequency may also help deal with NaNs.

#### More than one preord regressor
'Preord' regressor stands for 'Preordained' regressor, to make it clear this is data that will be know with high certainy about the future. 
Such data about the future is rare, one example might be number of stores that will be (planned to be) open each given day in the future when forecast sales. 
Since many algorithms do not handle more than one regressor, only one is handled here. If you would like to use more than one, 
manually select the best variable or use dimensionality reduction to reduce the features to one dimension. 
However, the model can handle quite a lot of parallel time series. Additional regressors can be passed through as additional time series to forecast. 
The regression models here can utilize the information they provide to help improve forecast quality. 
To prevent forecast accuracy for considering these additional series too heavily, input series weights that lower or remove their forecast accuracy from consideration.

#### Categorical Data
Categorical data is handled, but it is handled poorly. For example, optimization metrics do not currently include any categorical accuracy metrics. 
For categorical data that has a meaningful order (ie 'low', 'medium', 'high') it is best for the user to encode that data before passing it in, 
thus properly capturing the relative sequence (ie 'low' = 1, 'medium' = 2, 'high' = 3).

#### Custom Metrics
Implementing new metrics is rather difficult. However the internal 'Score' that compares models can easily be adjusted by passing through custom metric weights. 
Higher weighting increases the importance of that metric. 
`metric_weighting = {'smape_weighting' : 9, 'mae_weighting' : 1, 'rmse_weighting' : 5, 'containment_weighting' : 1, 'runtime_weighting' : 0.5}` 
sMAPE is generally the most versatile across multiple series, but doesn't handle forecasts with lots of zeroes well. 
Contaiment measures the percent of test data that falls between the upper and lower forecasts. 

## To-Do
* Smaller
	* Recombine best two of each model, if two or more present
	* Duplicates still seem to be occurring in the genetic template runs
	* Inf appearing in MAE and RMSE (possibly all NaN in test)
	* Na Tolerance for test in simple_train_test_split
	* Relative/Absolute Imports and reduce package reloading
	* User regressor to sklearn model regression_type
	* Import/export template
	* ARIMA + Detrend fails
* Things needing testing:
    * Confirm per_series weighting works properly
    * Passing in Start Dates - (Test)
    * Different frequencies
    * Various verbose inputs
    * Test holidays on non-daily data
    * Handle categorical forecasts where forecast leaves known values
* Speed improvements, Profiling, Parallelization, and Distributed options for general greater speed
* Generate list of functional frequences, and improve usability on rarer frequenices
* Warning/handling if lots of NaN in most recent (test) part of data
* Figures: Add option to output figures of train/test + forecast, other performance figures
* Input and Output saved templates as .csv and .json
* 'Check Package' to check if optional model packages are installed
* Pre-clustering on many time series
* If all input are Int, convert floats back to int
* Trim whitespace on string inputs
* Hierachial correction (bottom-up to start with)
* Improved verbosity controls and options. Replace most 'print' with logging.
* Export as simpler code (as TPOT)
* AIC metric, other accuracy metrics
* Analyze and return inaccuracy patterns (most inaccurate periods out, days of week, most inaccurate series)
* Used saved results to resume a search partway through
* Generally improved probabilistic forecasting
* Option to drop series which haven't had a value in last N days
* Option to change which metric is being used for model selections
* Use quantile of training data to provide upper/lower forecast for Last Value Naive (so upper forecast might be 95th percentile largest number)
* More thorough use of setting random seed
* For monthly data account for number of days in month
* Option to run generations until generations no longer see improvement of at least X % over n generations

#### New Ensembles:
	best 3 (unique algorithms not just variations)
	forecast distance 30/30/30
	best per series ensemble
	best point with best probalistic containment
#### New models:
	Seasonal Naive
	Last Value + Drift Naive
	Simple Decomposition forecasting
	GluonTS Models
	Simulations
	Sklearn + TSFresh
	Sklearn + polynomial features
	Sktime
	Ta-lib
	tslearn
	pydlm
	Isotonic regression
	TPOT if it adds multioutput functionality
