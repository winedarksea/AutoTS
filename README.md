# AutoTS
[![PyPI download month](https://img.shields.io/pypi/dm/autots.svg)](https://pypi.python.org/pypi/autots/)
[![PyPI version fury.io](https://badge.fury.io/py/autots.svg)](https://pypi.python.org/pypi/autots/)

<img src="/img/autots_logo.png" width="400" height="184" title="AutoTS Logo">

**Model Selection for Multiple Time Series**

Simple package for comparing and predicting with open-source time series implementations.

For other time series needs, check out the list [here](https://github.com/MaxBenChrist/awesome_time_series_in_python).

## Features
* Twenty available model classes, with tens of thousands of possible hyperparameter configurations
* Finds optimal time series models by genetic programming
* Handles univariate and multivariate/parallel time series
* Point and probabilistic forecasts
* Ability to handle messy data by learning optimal NaN imputation and outlier removal
* Ability to add external known-in-advance regressor
* Allows automatic ensembling of best models
* Multiple cross validation options
* Subsetting and weighting to improve search on many multivariate series
* Option to use one or a combination of metrics for model selection
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

```

# also: _hourly, _daily, _weekly, or _yearly
from autots.datasets import load_monthly 
df_long = load_monthly()

from autots import AutoTS
model = AutoTS(forecast_length=3, frequency='infer',
               prediction_interval=0.9, ensemble='all',
			   model_list='superfast',
               max_generations=5, num_validations=2,
			   validation_method='even')
model = model.fit(df_long, date_col='datetime',
				  value_col='value', id_col='series_id')

# Print the details of the best model
print(model)

prediction = model.predict()
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results
model_results = model.results()
# and aggregated from cross validation
validation_results = model.results("validation")

```

Check out [extended_tutorial.md](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html) for a more detailed guide to features!

## How to Contribute:
* Give feedback on where you find the documentation confusing
* Use AutoTS and...
	* Report errors and request features by adding Issues on GitHub
	* Posting the top model templates for your data (to help improve the starting templates)
	* Feel free to recommend different search grid parameters for your favorite models
* And, of course, contributing to the codebase directly on GitHub!


*Also known as Project CATS (Catlin's Automated Time Series) hence the logo.*
