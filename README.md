# AutoTS

<img src="/img/autots_logo.png" width="400" height="184" title="AutoTS Logo">

**Forecasting Model Selection for Multiple Time Series**

AutoML for forecasting with open-source time series implementations.

For other time series needs, check out the list [here](https://github.com/MaxBenChrist/awesome_time_series_in_python).

## Features
* Finds optimal time series forecasting model and data transformations by genetic programming optimization
* Handles univariate and multivariate/parallel time series
* Point and probabilistic upper/lower bound forecasts for all models
* Over twenty available model classes, with tens of thousands of possible hyperparameter configurations
	* Includes naive, statistical, machine learning, and deep learning models
	* Multiprocessing for univariate models for scalability on multivariate datasets
	* Ability to add external regressors
* Over thirty time series specific data transformations
	* Ability to handle messy data by learning optimal NaN imputation and outlier removal
* Allows automatic ensembling of best models
	* 'horizontal' ensembling on multivariate series - learning the best model for each series
* Multiple cross validation options
	* 'seasonal' validation allows forecasts to be optimized for the season of your forecast period
* Subsetting and weighting to improve speed and relevance of search on large datasets
	* 'constraint' parameter can be used to assure forecasts don't drift beyond historic boundaries
* Option to use one or a combination of metrics for model selection
* Import and export of model templates for deployment and greater user customization

## Installation
```
pip install autots
```
This includes dependencies for basic models, but additonal packages are required for some models and methods.

## Basic Use

Input data is expected to come in either a *long* or a *wide* format:

- The *wide* format is a `pandas.DataFrame` with a `pandas.DatetimeIndex` and each column a distinct series. 
- The *long* format has three columns: 
  - Date (ideally already in pd.DateTime format)
  - Series ID. For a single time series, series_id can be `= None`.
  - Value
- For *long* data, the column name for each of these is passed to .fit() as `date_col`, `id_col`, and `value_col`. No parameters are needed for *wide* data.

```python
# also: _hourly, _daily, _weekly, or _yearly
from autots.datasets import load_monthly

# sample datasets can be used in either of the long or wide import shapes
long = True
df = load_monthly(long=long)

from autots import AutoTS

model = AutoTS(
    forecast_length=3,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,
    model_list="superfast",
	transformer_list="fast",
    max_generations=5,
    num_validations=2,
    validation_method="backwards"
)
model = model.fit(
    df,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
)

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

The lower-level API, in particular the large section of time series transformers in the scikit-learn style, can also be utilized independently from the AutoML framework.

Check out [extended_tutorial.md](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html) for a more detailed guide to features!

## How to Contribute:
* Give feedback on where you find the documentation confusing
* Use AutoTS and...
	* Report errors and request features by adding Issues on GitHub
	* Posting the top model templates for your data (to help improve the starting templates)
	* Feel free to recommend different search grid parameters for your favorite models
* And, of course, contributing to the codebase directly on GitHub!


*Also known as Project CATS (Catlin's Automated Time Series) hence the logo.*
