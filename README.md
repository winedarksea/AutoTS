# AutoTS

#### Model Selection for Multiple Time Series

Simple package for comparing and predicting with open-source time series implementations.
For other time series needs, check out the package list here: https://github.com/MaxBenChrist/awesome_time_series_in_python

## Features
* Thirteen available models, with thousands of possible hyperparameter configurations
* Finds optimal time series models by genetic programming
* Handles univariate and multivariate/parallel time series
* Point and probabilistic forecasts
* Ability to handle messy data by learning optimal NaN imputation and outlier removal
* Ability to add external known-in-advance regressor
* Allows automatic ensembling of best models
* Multiple cross validation options
* Subsetting and weighting to improve search on many multivariate series
* Option to use one or a combination of SMAPE, RMSE, MAE, and Runtime for model selection
* Ability to upsample data to a custom frequency
* Import and export of templates allowing greater user customization

## Basic Use
```
pip install autots
```
This includes dependencies for basic models, but additonal packages are required for some models and methods.

Input data is expected to come in a 'long' format with three columns: 
* Date (ideally already in pd.DateTime format)
* Value
* Series ID. For a single time series, series_id can be `= None`. 
The column name for each of these is passed to .fit(). 

If your data is already wide (one column for each value), to bring to a long format:
```
df_long = df_wide.melt(id_vars = ['datetime_col_name'], var_name = 'series_id', value_name = 'value')
```


```
from autots.datasets import load_toy_monthly # also: _daily _yearly or _hourly
df_long = load_toy_monthly()

from autots import AutoTS
model = AutoTS(forecast_length = 3, frequency = 'infer',
               prediction_interval = 0.9, ensemble = False, weighted = False,
			   drop_data_older_than_periods = 240,
               max_generations = 5, num_validations = 2, validation_method = 'even')
model = model.fit(df_long, date_col = 'datetime', value_col = 'value', id_col = 'series_id')

# Print the name of the best model
print(model)

prediction = model.predict()
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
model_results = model.initial_results.model_results
# and including cross validation
validation_results = model.validation_results.model_results

```

Check out [extended_tutorial.md](https://github.com/winedarksea/AutoTS/blob/master/functional_environments.md) for a more detailed guide to features!

# How to Contribute:
* Give feedback on where you find the documentation confusing
* Use AutoTS and...
	* Report errors and request features by adding Issues on GitHub
	* Posting the top model templates for your data (to help improve the starting templates)
* And, of course, contributing to the codebase directly on GitHub!


*Also known as Project CATS*
CATS = Catlin's Automated Time Series