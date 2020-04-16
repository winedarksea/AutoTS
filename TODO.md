# Basic Tenants
* Ease of Use > Accuracy > Speed
* All models should be probabilistic (upper/lower forecasts)
* All models should be able to handle multiple parallel time series (even if just a `for` loop)
* The expectation is series will largely be consistent in period, or at least up-sampled to regular intervals
* New transformations should be applicable to many datasets and models
* New models need only be sometimes applicable
* Fault tolerance: it is perfectly acceptable for model parameters to fail on some datasets, the higher level API will pass over and use others.


# Errors: 
DynamicFactor holidays 	Exceptions 'numpy.ndarray' object has no attribute 'values'
lower/upper MAE appearing NaN, and then getting a better score
VECM does not recognize exog to predict
kbins not working when it assigns fewer bins than n_bins asked for (use the property transformer.n_bins_ ?)
FastICA 'array must not contain infs or NaNs'
How do fillna methods handle datasets that have entirely NaN series?
Check if any transformation parameters seem to consistently perform poorly, suggesting of problems. Also speed.

Test updated VARMAX, DynamicFactor
Test updated context_slicer
Test updated GluonTS

Add to template: Gluon, Motif
Bring GeneralTransformer to higher level API.
	wide_to_long and long_to_wide in higher-level API
Option to use full traceback in errors in table
Improved genetic recombination:
	Make no new genetic models with just new random transform. Add recombination for rolling regression.
	Issues to handle:
		Approve and unapproved for model recombination
		If not enough (only one example) of that model type
		RollingRegression nested dict
	Randomly select np.random.choice(list(y.keys/items())) some keys from Y, then {**x, **y} all of x, some of y.
	2 best + 2 random model
	2 best from model + 1 best overall + 1 random
	Allow many models at first, but as generations increase, reduce total allowed but up max per model class
### Ignored Errors:
xgboost poisson loss does not accept negatives

# To-Do
* Get the sphinx (google style) documentation and readthedocs.io website up
* Better point to probabilistic (uncertainty of naive last-value forecast) - linear reg of abs error of samples - simulations
	* Data, pct change, find window with % max change pos, and neg then avg. Bound is first point + that percent, roll from that first point and adjust if points cross, variant where all series share knowledge
	* Data, split, normalize, find distribution exponentially weighted to most recent, center around forecast, shared variant
	* Data quantile, recentered around median of forecast.
	* Categorical class probabilities as range for RollingRegression
* get_prediction for Statsmodels Statespace models to include confidence interval where possible
	* migrate arima_model to arima.model
	* uncomp, dynamic factor with uncertainty intervals
* Window regression, 
		shuffle windows, 
		1d or 2d (series * forecast_length) output, 1d or 2d input
		normalize each window
		transfer learning
		At least one for each of:
			Tensorflow/Keras
			Mxnet
			PyTorch
* Better X_maker:
	* 1d and 2d variations
	https://link.springer.com/article/10.1007/s10618-019-00647-x/tables/1
* RollingRegression
	Pytorch and Tensorflow Simple LSTM/GRU
	other sequence models
	Categorical classifier, ComplementNB
	Clustering then separate models
	PCA or similar -> Univariate Series (Unobserved Components)
* Simple performance:
	* large if collections (ModelMonster, Transformers) with dict lookups
	* replace try/except with if/else in some cases
* GluonTS
	* Add support for preord_regressor
	* Make sure of rolling regression setup
	* Modify GluonStart if lots of NaN at start of that series
	* GPU and CPU ctx
* Relative/Absolute Imports and reduce package reloading messages
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
* AIC metric, other accuracy metrics
	* MAE of upper and lower forecasts, balance with Containment
* Analyze and return inaccuracy patterns (most inaccurate periods out, days of week, most inaccurate series)
* Development tools:
	* Add to Conda distribution as well as pip
	* Continuous integration
	* Code/documentation quality checkers
* Ability to automatically add external datasets of parallel time series of global usability (ie from FRED or others)
* Option to import either long or wide data
* Infer column names for df_long to wide based on which is datetime, which is string, and which is numeric

### Faster Convergence / Faster in General
* Only search useful parameters, highest probability for most likely effective parameters
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
* GPU - xgboost, GluontTS

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
	Simulations
	Ta-lib
	tslearn
	Multivariate GARCH
	pydlm - baysesian dynamic linear
	Survival Analysis
	MarkovAutoRegression
	TPOT if it adds multioutput functionality
	https://towardsdatascience.com/pyspark-forecasting-with-pandas-udf-and-fb-prophet-e9d70f86d802
	Compressive Transformer, if they go anywhere

#### New Transformations:
	Weighted moving average
	Symbolic aggregate approximation (SAX) and (PAA) (basically these are just binning)
	Seasonal means/std/last value differencing
		- random row, find other rows closest to it, retrieve indexes, see if indexes have divisible difference between.
	Shared discretization (all series get same binning)
	Ordinal discretization (invertible, all values to bins labeled 1, 2, 3, etc.)
	