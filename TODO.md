# Basic Tenants
* Ease of Use > Accuracy > Speed
* All models should be probabilistic (upper/lower forecasts)
* All models should be able to handle multiple parallel time series (even if just a `for` loop)
* The expectation is series will largely be consistent in period, or at least up-sampled to regular intervals
* New transformations should be applicable to many datasets and models
* New models need only be sometimes applicable
* Fault tolerance: it is perfectly acceptable for model parameters to fail on some datasets, the higher level API will pass over and use others.

# Latest:
* added 'coerce_integer' to GeneralTransformer
* tuned MotifSimulation and GeneralTransformer inputs
* minor improvements to Scoring, Aggregation, Recombination, and export_template

# Errors: 
DynamicFactor holidays 	Exceptions 'numpy.ndarray' object has no attribute 'values'
VECM does not recognize exog to predict
ARIMA with User or Holiday ValueError('Can only compare identically-labeled DataFrame objects',)
Drop Most Recent does not play well logically with added external (future) regressors.
FastICA 'array must not contain infs or NaNs'
How do fillna methods handle datasets that have entirely NaN series?
VAR ValueError('Length of passed values is 4, index implies 9',)
WindowRegression + KerasRNN + 1step + univariate = ValueError('Length mismatch: Expected axis has 54 elements, new values have 9 elements',)
Is Template Eval Error: ValueError('array must not contain infs or NaNs',) related to Point to Probability HISTORIC QUANTILE?
'Fake Date' doesn't work on entirely NaN series - ValueError('Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.',)


### Ignored Errors:
xgboost poisson loss does not accept negatives
GluonTS not accepting quite a lot of frequencies
KerasRNN errors due to parameters not working on all dataset
Tensorflow GPU backend may crash on occasion.

## General Tasks
* test submission
* test whether bottom up significantly overestimates on rollup
	* store level hierarchial
* Profile slow parts of AutoTS on 1,000 series
	* remove slow transformers unless parameter
	* 'fast' option for RandomTransformations generator
* 'Grouping' reconcilliation
* Make sure min per_series gets into validation if doing horizontal
* Post
	* no negatives
	* constraint
	* hierarchial
	* coerce integer

# To-Do
* drop duplicates as function of TemplateEvalObject
* speed up MotifSimulation
* fake date dataset of many series to improve General Template
* better document ensembling
* optimize randomtransform probabilities
* Add to template: Gluon, Motif, WindowRegression
* Convert 'Holiday' regressors into Datepart + Holiday 2d
* best per series to validation template even if poor on score overall
* Bring GeneralTransformer to higher level API.
	* wide_to_long and long_to_wide in higher-level API
* Option to use full traceback in errors in table
* Hierarchial
	* every level must be included in forecasting data
	* provide grouping as dict like weights
		* calculate weights for groups based on series weights
		* per_series metrics
		* subsetting
	* 'mid' level
	* ensembles will not be reconcilled
	* one level. User would have to specify all as based on lowest-level keys if wanted sum-up.
* Better point to probabilistic (uncertainty of naive last-value forecast) 
	* linear reg of abs error of samples - simulations
	* Data, pct change, find window with % max change pos, and neg then avg. Bound is first point + that percent, roll from that first point and adjust if points cross, variant where all series share knowledge
	* Bayesian posterior update of prior
	* variance of k nearest neighbors
	* Data, split, normalize, find distribution exponentially weighted to most recent, center around forecast, shared variant
	* Data quantile, recentered around median of forecast.
	* Categorical class probabilities as range for RollingRegression
* get_forecast for Statsmodels Statespace models to include confidence interval where possible
	* migrate arima_model to arima.model
	* uncomp with uncertainty intervals
* GUI overlay for editing/creating templates, and for running (Flask)
* Window regression
	* transfer learning
* RollingRegression
	* Better X_maker:
		* 1d and 2d variations
		* .cov, .skew, .kurt, .var
		* https://link.springer.com/article/10.1007/s10618-019-00647-x/tables/1
	* Probabilistic:
		https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
	* other sequence models
	* Categorical classifier, ComplementNB
	* PCA or similar -> Univariate Series (Unobserved Components)
* Simple performance:
	* replace try/except with if/else in some cases
* GluonTS
	* Add support for future_regressor
	* Make sure of rolling regression setup
	* Modify GluonStart if lots of NaN at start of that series
	* GPU and CPU ctx
* motif simulation, remove all those for loops
* implement 'borrow' Genetic Recombination for ComponentAnalysis
* Regressor to TensorflowSTS
* Relative/Absolute Imports and reduce package reloading messages
* Replace OrdinalEncoder with non-external code
* 'Age' regressor as an option in addition to User/Holiday in ARIMA, etc.
* Speed improvements, Profiling
* Parallelization, and Distributed options (Dask) for general greater speed
* Improve usability on rarer frequenices (ie monthly data where some series start on 1st, others on 15th, etc.)
* Figures: Add option to output figures of train/test + forecast, other performance figures
* Replace most 'print' with logging.
* Analyze and return inaccuracy patterns (most inaccurate periods out, days of week, most inaccurate series)
* Development tools:
	* Add to Conda (Forge) distribution as well as pip
	* Continuous integration
	* Code/documentation quality checkers
* Ability to automatically add external datasets of parallel time series of global usability (ie from FRED or others)
* make datetime input optional, just allow dataframes of numbers
* Option to import either long or wide data
* Infer column names for df_long to wide based on which is datetime, which is string, and which is numeric

### Links
* https://link.springer.com/article/10.1007/s10618-019-00647-x/tables/1
* https://github.com/gantheory/TPA-LSTM
* https://github.com/huseinzol05/Stock-Prediction-Models/tree/master/deep-learning

### Faster Convergence / Faster in General
* Only search useful parameters, highest probability for most likely effective parameters
* Remove parameters that are rarely/never useful from get_new_params
* Don't apply transformations to Zeroes naive, possibly other naives
* Option to run generations until generations no longer see improvement of at least X % over n generations
* ignore series of weight 0 in univariate models
* Method to 'unlock' deeper parameter search, 
	* potentially a method = 'deep' to get_new_params used after n generations
	* no unlock, but simply very low-probability deep options in get_new_params
* Exempt or reduce slow models from unnecessary runs, particularly with different transformations
* Numba and Cython acceleration (metrics might be easy to start with)
* GPU - xgboost, GluontTS

#### New datasets:
	Second level data that is music (like a radio stream)
	Ecological data

#### New Ensembles:
	Best N combined with Decision Tree

#### New models:
	Croston, SBA, TSB, ADIDA, iMAPA
	Local Linear/Piecewise Regression Model
	Simulations
	Ta-lib
	Pyflux
	tslearn
	GARCH (arch library seems best maintained, none have multivariate)
	pydlm - baysesian dynamic linear
	MarkovAutoRegression
	hmmlearn
	TPOT if it adds multioutput functionality
	https://towardsdatascience.com/pyspark-forecasting-with-pandas-udf-and-fb-prophet-e9d70f86d802
	Compressive Transformer
	Reinforcement Learning

#### New Transformations:
	Sklearn iterative imputer 
	lag and beta to DifferencedTransformer to make it more of an AR process
	Weighted moving average
	Symbolic aggregate approximation (SAX) and (PAA) (basically these are just binning)
	Shared discretization (all series get same shared binning)
	Last Value Centering
	Constraint as a transformation parameter

### New Model Checklist:
	* Add to ModelMonster
	* Add to AutoTS 'all' list
	* all to recombination_approved if so
