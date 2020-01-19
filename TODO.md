
## To-Do
Recombine best two of each model, if two or more present
Inf appearing in MAE and RMSE (possibly all NaN in test)
Na Tolerance for test in simple_train_test_split
Relative/Absolute Imports and reduce package reloading
User regressor to sklearn model regression_type
Annual, Monthly, Weekly, Hourly sample data
Format of Regressor - allow multiple input to at least sklearn models
'Age' regressor as an option in addition to User/Holiday
Handle categorical forecasts where forecast leaves range of known values

Things needing testing:
    With and without regressor
    With and without weighting
    Passing in Start Dates - (Test)
    Different frequencies
    Various verbose inputs
    Test holidays on non-daily data

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
