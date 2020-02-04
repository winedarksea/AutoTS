# To-Do
Single time series
Better point to probabilistic (uncertainty of naive last-value forecast) - linear reg of abs error of samples
Better X_maker for Rolling Sklearn
Sklearn Holiday not working
Possible error where first template model is invalid, 'smape_weighted' doesn't exist error
* Recombine best two of each model parameters, if two or more present
* Inf appearing in MAE and RMSE (possibly all NaN in test)
* Na Tolerance for test in simple_train_test_split
* min_allowed_train_percent into higher-level API
* Relative/Absolute Imports and reduce package reloading
* User regressor to sklearn model regression_type (added, needs testing)
* Weekly sample data
* Format of Regressor - allow multiple input to at least sklearn models
* 'Age' regressor as an option in addition to User/Holiday
* Handle categorical forecasts where forecast leaves range of known values, then add to upper/lower forecasts
* Speed improvements, Profiling, Parallelization, and Distributed options for general greater speed
* Improve usability on rarer frequenices
* Warning/handling if lots of NaN in most recent (test) part of data
* Figures: Add option to output figures of train/test + forecast, other performance figures
* Pre-clustering on many time series
* If all input are Int, convert floats back to int
* Trim whitespace on string inputs
* Hierachial correction (bottom-up to start with)
* Improved verbosity controls and options. Replace most 'print' with logging.
* Export as simpler code (as TPOT)
* set up the lower-level API to be usable as pipelines
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
* add constant to GLM

### Faster Convergence
* Only search useful parameters, highest probability for most likely effective parameters
* 'Expert' starting template to try most likley combinations first
* Recombination of parameters (both transformation and model)

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
	pydlm - baysesian dynamic linear
	Isotonic regression
	TPOT if it adds multioutput functionality
