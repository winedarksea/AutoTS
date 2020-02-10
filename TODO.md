# Basic Tenants
* Ease of Use > Accuracy > Speed
* All models should be probabilistic (upper/lower forecasts)
* All models should be able to handle multiple parallel time series (even if just a `for` loop)
* The expectation is series will largely be consistent in period, or at least up-sampled to regular intervals
* New transformations should be applicable to many datasets and models
* New models need only be sometimes applicable
* Fault tolerance: it is perfectly acceptable for model parameters to fail on some datasets, the higher level API will pass over and use others.

# How to Contribute:
* Give feedback on where you find the documentation confusing
* Use AutoTS and...
	* Report errors and request features by adding Issues on GitHub
	* Posting the top model templates for your data (to help improve the starting templates)
* And, of course, contributing to the codebase directly on GitHub!

## Errors: 
'ValueError: forecast_length is too large, not enough training data, alter min_allowed_train_percent to override' -> daily toy, forecast_length = 30
'Detrend' transformation is still buggy (can't convert to Series)
Possible error where first template model is invalid, 'smape_weighted' doesn't exist error

# To-Do
* Get the sphinx (google style) documentation and readthedocs.io website up
* Improve Readme
* Test 'max_per_model_class' in export template
* Better point to probabilistic (uncertainty of naive last-value forecast) - linear reg of abs error of samples - simulations
* Better X_maker:
	* use feature selection on TSFresh features - autocorrelation lag n, fft/cwt coefficients (abs), abs_energy
	* Holiday not working
	* date part and age/expanding regressors
* Get Tsfresh working with small dataset (short, 2 columns) (check feature importance again)
* Recombine best two of each model parameters, if two or more present
* Inf appearing in MAE and RMSE (possibly all NaN in test)
* Na Tolerance for test in simple_train_test_split
* min_allowed_train_percent into higher-level API
* Relative/Absolute Imports and reduce package reloading
* Weekly sample data (EIA)
* Add a print(AutoTS) that displays fitted model, if given
* Format of Regressor - allow multiple input to at least sklearn models
* 'Age' regressor as an option in addition to User/Holiday in ARIMA, etc.
* Handle categorical forecasts where forecast leaves range of known values, then add to upper/lower forecasts
* Speed improvements, Profiling, Parallelization, and Distributed options for general greater speed
* Improve usability on rarer frequenices (ie monthly data where some series start on 1st, others on 15th, etc.)
* Warning/handling if lots of NaN in most recent (test) part of data
* Figures: Add option to output figures of train/test + forecast, other performance figures
* Pre-clustering on many time series
* If all input are Int, convert floats back to int
* Trim whitespace/case-desensitize on string inputs
* Option to print % improvement of best over last value naive
* Hierachial correction (bottom-up to start with)
* Because I'm a biologist, incorporate more genetics and such. Also as a neuro person, there must be a way to fit networks in...
* Improved verbosity controls and options. Replace most 'print' with logging.
* Export as simpler code (as TPOT)
* set up the lower-level API to be usable as pipelines
	* allow stand-alone pipeline for transformation with format for export data format to other package requirements (use AutoTS just for preprocessing)
* AIC metric, other accuracy metrics
* Analyze and return inaccuracy patterns (most inaccurate periods out, days of week, most inaccurate series)
* Used saved results to resume a search partway through
* Generally improved probabilistic forecasting
* Option to drop series which haven't had a value in last N days
* More thorough use of setting random seed, verbose, n_jobs
* For monthly data account for number of days in month
* add constant to GLM

### Faster Convergence
* Only search useful parameters, highest probability for most likely effective parameters
* 'Expert' starting template to try most likley combinations first
* Recombination of parameters (both transformation and model)
* Remove parameters that are rarely/never useful from get_new_params
* Don't apply transformations to Zeroes naive, possibly other naives
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
	Tensorflow Probability Structural Time Series
	Pytorch Simple LSTM/GRU
	Simulations
	XGBoost (doesn't support multioutput directly)
	Sklearn + TSFresh
	Sktime
	Ta-lib
	tslearn
	GARCH
	pydlm - baysesian dynamic linear
	Isotonic regression
	Survival Analysis
	TPOT if it adds multioutput functionality

#### New Transformations:
	Test variations on 'RollingMean100thN'
