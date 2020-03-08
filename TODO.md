# Basic Tenants
* Ease of Use > Accuracy > Speed
* All models should be probabilistic (upper/lower forecasts)
* All models should be able to handle multiple parallel time series (even if just a `for` loop)
* The expectation is series will largely be consistent in period, or at least up-sampled to regular intervals
* New transformations should be applicable to many datasets and models
* New models need only be sometimes applicable
* Fault tolerance: it is perfectly acceptable for model parameters to fail on some datasets, the higher level API will pass over and use others.

# Errors: 
'Detrend' transformation is still buggy (can't convert to Series)
raise AttributeError(("Model String '{}' not recognized").format(model)) -> turn to an allowable exception with a printed warning
Holiday not (always) working

# To-Do
* Get the sphinx (google style) documentation and readthedocs.io website up
* Better point to probabilistic (uncertainty of naive last-value forecast) - linear reg of abs error of samples - simulations
* get_prediction for Statsmodels Statespace models to include confidence interval where possible
	* migrate arima_model to arima.model
* Check how fillna methods handle datasets that are entirely NaN
* Better X_maker:
	* use feature selection on TSFresh features - autocorrelation lag n, fft/cwt coefficients (abs), abs_energy
	* date part and age/expanding regressors
* GluonTS
	* Add support for preord_regressor
	* Modify GluonStart if lots of NaN at start of that series
	* GPU and CPU ctx
* Print methods for prediction/model objects that give simple readme how-to's
* Get Tsfresh working with small dataset (short, 2 columns) (check feature importance again)
* Relative/Absolute Imports and reduce package reloading messages
* Format of Regressor - allow multiple input to at least sklearn models
* 'Age' regressor as an option in addition to User/Holiday in ARIMA, etc.
* Handle categorical forecasts where forecast leaves range of known values, then add to upper/lower forecasts
* Speed improvements, Profiling
* Parallelization, and Distributed options (Dask) for general greater speed
* Improve usability on rarer frequenices (ie monthly data where some series start on 1st, others on 15th, etc.)
* Figures: Add option to output figures of train/test + forecast, other performance figures
* Pre-clustering on many time series
* If all input are Int, convert floats back to int
* Trim whitespace/case-desensitize on string inputs
* Option to print % improvement of best over last value naive
* Hierachial correction (bottom-up to start with)
* Because I'm a biologist, incorporate more genetics and such. Also as a neuro person, there must be a way to fit networks in...
* Improved verbosity controls and options. 
* Replace most 'print' with logging.
* Export as simpler code (as TPOT)
* set up the lower-level API to be usable as pipelines
	* allow stand-alone pipeline for transformation with format for export data format to other package requirements (use AutoTS just for preprocessing)
* AIC metric, other accuracy metrics
	* MAE of upper and lower forecasts, balance with Containment
* Metric to measure if the series follows the same shape (Contour)
	* Potentially % change between n and n-1, compare this % change between forecast and actual
	* One if same direction, 0 otherwise (sum/len)
* Analyze and return inaccuracy patterns (most inaccurate periods out, days of week, most inaccurate series)
* Development tools:
	* Add to Conda distribution as well as pip
	* Continuous integration
	* Code/documentation quality checkers
* Option to drop series which haven't had a value in last N days
* More thorough use of setting random seed, verbose, n_jobs
* For monthly data account for number of days in month
* add constant to GLM
* Ability to automatically add external datasets of parallel time series of global usability (ie from FRED or others)
* Option to import either long or wide data
* Infer column names for df_long to wide based on which is datetime, which is string, and which is numeric

### Faster Convergence / Faster in General
* Only search useful parameters, highest probability for most likely effective parameters
* 'Expert' starting templates to try most likley combinations first
* Recombine best two of each model parameters, if two or more present (plus option to disable this)
* Recombination of transformations
* Remove parameters that are rarely/never useful from get_new_params
* Don't apply transformations to Zeroes naive, possibly other naives
* Option to run generations until generations no longer see improvement of at least X % over n generations
* ignore series of weight 0 in univariate models
* Method to 'unlock' deeper parameter search, 
	* potentially a method = 'deep' to get_new_params used after n generations
	* no unlock, but simply very low-probability deep options in get_new_params
* Exempt or reduce slow models from unnecessary runs, particularly with different transformations
* Numba and Cythion acceleration (metrics might be easy to start with)

#### New Ensembles:
	best 3 (unique algorithms not just variations of same)
	forecast distance 30/30/30
	best per series ensemble ('horizontal ensemble')
	best point with best probalistic containment
#### New models:
	Seasonal Naive
	Last Value + Drift Naive
	Simple Decomposition forecasting
	Statespace variant of ETS which has Confidence Intervals
	Tensorflow Probability Structural Time Series
	Pytorch Simple LSTM/GRU
	Simulations
	XGBoost (doesn't support multioutput directly)
	Sklearn + TSFresh
	Sktime
	Ta-lib
	tslearn
	Multivariate GARCH
	pydlm - baysesian dynamic linear
	Isotonic regression
	Survival Analysis
	MarkovAutoRegression
	Motif discovery, and repeat
	TPOT if it adds multioutput functionality
	https://towardsdatascience.com/pyspark-forecasting-with-pandas-udf-and-fb-prophet-e9d70f86d802
	Compressive Transformer, if they go anywhere

#### New Transformations:
	Test variations on 'RollingMean100thN'
