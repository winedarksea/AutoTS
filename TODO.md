# Basic Tenants
* Ease of Use > Accuracy > Speed
* All models should be probabilistic (upper/lower forecasts)
* All models should be able to handle multiple parallel time series (even if just a `for` loop)
* The expectation is series will largely be consistent in period, or at least up-sampled to regular intervals
* New transformations should be applicable to many datasets and models
* New models need only be sometimes applicable
* Fault tolerance: it is perfectly acceptable for model parameters to fail on some datasets, the higher level API will pass over and use others.

# Errors: 
Holiday not (always) working
\Users\Owner\Documents\Personal\Projects\AutoTS\autots\evaluator\metrics.py:115: RuntimeWarning: invalid value encountered in greater_equal X = X>=0
\Users\Owner\Documents\Personal\Projects\AutoTS\autots\evaluator\metrics.py:72: RuntimeWarning: Mean of empty slice return np.nanmean(mae_result, axis=0)
Models are failing without being captured in model_results

GluonTS to template, best
Select Transformers to include in random, resort in function

Does rolling mean work?
Are transformers included in Try/Except?
Would expect to see all transformers delivery roughly same performance with same model
Bring GeneralTransformer to higher level API.
	wide_to_long and long_to_wide in higher-level API

# To-Do
* Get the sphinx (google style) documentation and readthedocs.io website up
* Better point to probabilistic (uncertainty of naive last-value forecast) - linear reg of abs error of samples - simulations
* get_prediction for Statsmodels Statespace models to include confidence interval where possible
	* migrate arima_model to arima.model
	* uncomp, dynamic factor with uncertainty intervals
* Check how fillna methods handle datasets that are entirely NaN
* add_constant to GLS, GLM
* Better X_maker:
	* add magnitude_1, magnitude2, and so on (new_params have these all the same for models that don't use them)
	* use feature selection on TSFresh features - autocorrelation lag n, fft/cwt coefficients (abs), abs_energy
	* date part and age/expanding regressors
	* moving average +/- moving std deviation
	* Nystroem kernel
* Simple performance:
	* large if collections (ModelMonster, Transformers) with dict lookups
	* replace try/except with if/else in some cases
* GluonTS
	* Add support for preord_regressor
	* Make sure of rolling regression setup
	* Modify GluonStart if lots of NaN at start of that series
	* GPU and CPU ctx
* Print methods for prediction/model objects that give simple readme how-to's
* Relative/Absolute Imports and reduce package reloading messages
* Format of Regressor - allow multiple input to at least sklearn models
	* Miso l filter or similar to reduce to single time series where only on regressor allowed
	* or PCA or other fast approach to reduce dimensions
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
* If model list * max model_per_class is < models to validate or other, raise models_per_clas
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

#### New datasets:
	Second level data that is music (like a radio stream)
	Ecological data

#### New Ensembles:
	best 3 (unique algorithms not just variations of same)
	forecast distance 30/30/30
	best per series ensemble ('horizontal ensemble')
	best point with best probalistic containment
#### New models:
	Simple Decomposition forecasting
	Statespace variant of ETS which has Confidence Intervals
	Tensorflow Probability Structural Time Series
	RollingRegression
		Pytorch and Tensorflow Simple LSTM/GRU
		other sequence models
		Categorical classifier
		RBF kernel SVR
		Clustering then separate models
		PCA or similar -> Univariate Series (Unobserved Components)
	Neural net with just short series as input, Keras time series generator
		Transfer learning (model weights pretrained on other time series)
		Neural net with '2d' output (series * forecast_length)
		At least one for each of:
			Tensorflow/Keras
			Mxnet
			PyTorch
	Simulations
	Motif simulations
	Ta-lib
	tslearn
	Nystroem
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
	Test variations on 'RollingMean100thN', n_bins
		3, 10, 10thN, 100thN
	Simple filter, no inverse (uninverted weighted moving average)
	Weighted moving average
	Symbolic aggregate approximation (SAX) and (PAA) (basically these are just binning)
	PCA
	Simple difference smoothing
	Seasonal means/std/last value differencing
	Cum sum
	Sine regression detrend
	