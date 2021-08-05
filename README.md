# AutoTS

<img src="/img/autots_logo.png" width="400" height="184" title="AutoTS Logo">

**Forecasting Model Selection for Multiple Time Series**

AutoML for forecasting with open-source time series implementations.

For other time series needs, check out the list [here](https://github.com/MaxBenChrist/awesome_time_series_in_python).

## Table of Contents
* [Features](https://github.com/winedarksea/AutoTS#features)
* [Installation](https://github.com/winedarksea/AutoTS#installation)
* [Basic Use](https://github.com/winedarksea/AutoTS#basic-use)
* [Tips for Speed and Large Data](https://github.com/winedarksea/AutoTS#tips-for-speed-and-large-data)
* Extended Tutorial [GitHub](https://github.com/winedarksea/AutoTS/blob/master/extended_tutorial.md) or [Docs](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html)
* [Production Example](https://github.com/winedarksea/AutoTS/blob/master/production_example.py)

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
	* 'seasonal' validation allows forecasts to be optimized for the seasonity of the data
* Subsetting and weighting to improve speed and relevance of search on large datasets
	* 'constraint' parameter can be used to assure forecasts don't drift beyond historic boundaries
* Option to use one or a combination of metrics for model selection
* Import and export of model templates for deployment and greater user customization

## Installation
```
pip install autots
```
This includes dependencies for basic models, but [additonal packages](https://github.com/winedarksea/AutoTS/blob/master/extended_tutorial.md#installation-and-dependency-versioning) are required for some models and methods.

## Basic Use

Input data is expected to come in either a *long* or a *wide* format:

- The *wide* format is a `pandas.DataFrame` with a `pandas.DatetimeIndex` and each column a distinct series. 
- The *long* format has three columns: 
  - Date (ideally already in pd.DateTime format)
  - Series ID. For a single time series, series_id can be `= None`.
  - Value
- For *long* data, the column name for each of these is passed to .fit() as `date_col`, `id_col`, and `value_col`. No parameters are needed for *wide* data.

```python
# also load: _hourly, _monthly, _weekly, _yearly, or _live_daily
from autots import AutoTS, load_daily

# sample datasets can be used in either of the long or wide import shapes
long = False
df = load_daily(long=long)

model = AutoTS(
    forecast_length=21,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,
    model_list="default",
    transformer_list="fast",
    drop_most_recent=1,
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
)
model = model.fit(
    df,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
)

prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2019-01-01")
# Print the details of the best model
print(model)

# point forecasts dataframe
forecasts_df = prediction.forecast
# upper and lower forecasts
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# accuracy of all tried model results
model_results = model.results()
# and aggregated from cross validation
validation_results = model.results("validation")
```

The lower-level API, in particular the large section of time series transformers in the scikit-learn style, can also be utilized independently from the AutoML framework.

Check out [extended_tutorial.md](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html) for a more detailed guide to features!

Also take a look at the [production_example.py](https://github.com/winedarksea/AutoTS/blob/master/production_example.py)


## Tips for Speed and Large Data:
* Use appropriate model lists, especially the predefined lists:
	* `superfast` (simple naive models) and `fast` (more complex but still faster models)
	* `fast_parallel` (a combination of `fast` and `parallel`) or `parallel`, given many CPU cores are available
		* `n_jobs` usually gets pretty close with `='auto'` but adjust as necessary for the environment
	* see a dict of predefined lists (some defined for internal use) with `from autots.models.model_list import model_lists`
* Use the `subset` parameter when there are many similar series, `subset=100` will often generalize well for tens of thousands of similar series.
	* if using `subset`, passing `weights` for series will weight subset selection towards higher priority series.
	* if limited by RAM, it can be easily distributed by running multiple instances of AutoTS on different batches of data, having first imported a template pretrained as a starting point for all.
* Set `model_interrupt=True` which passes over the current model when a `KeyboardInterrupt` ie `crtl+c` is pressed (although if the interrupt falls between generations it will stop the entire training).
* Use the `result_file` method of `.fit()` which will save progress after each generation - helpful to save progress if a long training is being done. Use `import_results` to recover.
* While Transformations are pretty fast, setting `transformer_max_depth` to a lower number (say, 2) will increase speed. Also utilize `transformer_list`.
* Ensembles are obviously slower to predict because they run many models, 'distance' models 2x slower, and 'simple' models 3x-5x slower.
	* `ensemble='horizontal-max'` with `model_list='no_shared_fast'` can scale relatively well given many cpu cores because each model is only run on the series it is needed for.
* Reducing `num_validations` and `models_to_validate` will decrease runtime but may lead to poorer model selections.
* For datasets with many records, upsampling (for example, from daily to monthly frequency forecasts) can reduce training time if appropriate.
	* this can be done by adjusting `frequency` and `aggfunc` but is probably best done before passing data into AutoTS.


## How to Contribute:
* Give feedback on where you find the documentation confusing
* Use AutoTS and...
	* Report errors and request features by adding Issues on GitHub
	* Posting the top model templates for your data (to help improve the starting templates)
	* Feel free to recommend different search grid parameters for your favorite models
* And, of course, contributing to the codebase directly on GitHub!


*Also known as Project CATS (Catlin's Automated Time Series) hence the logo.*
