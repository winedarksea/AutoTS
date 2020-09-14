# Basic Tenants
* Ease of Use > Accuracy > Speed (with speed more important with 'fast' selections)
* The goal is to be able to run a horizontal ensemble prediction on 1,000 series/hour with a 'fast' selection, 10,000 series/hour with 'very fast'.
* Availability of models which share information among series
* All models should be probabilistic (upper/lower forecasts)
* All models should be able to handle multiple parallel time series
* New transformations should be applicable to many datasets and models
* New models need only be sometimes applicable
* Fault tolerance: it is perfectly acceptable for model parameters to fail on some datasets, the higher level API will pass over and use others.
* Missing data tolerance: large chunks of data can be missing and model will still produce reasonable results (although lower quality than if data is available)

## Assumptions on Data
* Series will largely be consistent in period, or at least up-sampled to regular intervals
* The most recent data will generally be the most important
* Forecasts are desired for the future immediately following the most recent data.

# Latest
* None

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

## Concerns
* resource utilization at scale
	* improve horizontal ensembling efficiency in particular
* structure of General Transformer
* overfitting on first train segment (train progressive subsets?)
* End Users add their own models
* improve starting templates (sorta best to wait until other things ironed out)
* Improve documentation and usability of lower level code
* better metrics, perhaps improve contour
* better summarization of many time series into a few high-information time series as parallel or regressors
* Ability to automatically add useful global information as regressors or parallel series
* Built in GUI or Command Line tools


## To-Do
* Remove SP500 from AutoTS
* Python Black
* Migrate to-do to GitHub issues and project board
	* GitHub badges in Readme, GitHub Actions flake8
* Horizontal improvements
	* don't run univariate models on all series, only on needed series
	* generalizable from run on only a subset
	* handle failure of a lower level model on some series (but not for dist variants)
	* remove 'horizontal' sanity check run, takes too longer
	* allow multiprocessing for each model of horizontal
		* allow multiprocessing to models, pass through num_process + **kwargs
* User friendly:
	* clean up lower level
	* make passing in own models easy
		* clean up base model object
* Make preprocessing templates more flexible...
* Speed:
	* Make Distributable/Multiprocessing
	* Fast window regression only
	* Fast MotifSimulation
* Improve templates
	* 'fake date' dataset with high diversity of series to train on


* Profile slow parts of AutoTS on 1,000 series
	* remove slow transformers unless parameter
	* 'fast' option for RandomTransformations generator
* have subsetting sample for diversity, not just random
* cleanse similar models out first, before horizontal ensembling
* Best3Ensemble
	* Work on model failure
	* Rename from Best3 to BestN
	* Dicts instead of list of DFs
	* Add 'model_count' to parameters

* check models from M5 competition results
* drop duplicates as function of TemplateEvalObject
* optimize randomtransform probabilities
* improve test.py script for actual testing of many features
* Add to template: Gluon, Motif, WindowRegression
* Convert 'Holiday' regressors into Datepart + Holiday 2d
* Option to use full traceback in errors in table
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
* Simple performance:
	* replace try/except with if/else in some cases
* GluonTS
	* Add support for future_regressor
	* Make sure of rolling regression setup
	* Modify GluonStart if lots of NaN at start of that series
	* GPU and CPU ctx
* implement 'borrow' Genetic Recombination for ComponentAnalysis
* Relative/Absolute Imports and reduce package reloading messages
* Replace OrdinalEncoder with non-external code
* 'Age' regressor as an option in addition to User/Holiday in ARIMA, etc.
* Multiprocessing or Distributed options (Dask) for general greater speed
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
* Numba and Cython acceleration
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
	Reinforcement Learning with online forecasting?
	

#### New Transformations:
	Sklearn iterative imputer 
	lag and beta to DifferencedTransformer to make it more of an AR process
	Weighted moving average
	Symbolic aggregate approximation (SAX) and (PAA) (basically these are just binning)
	Shared discretization (all series get same shared binning)
	Last Value Centering
	More sophisticated fillna methods
	Constraint as a transformation parameter

### New Model Checklist:
	* Add to ModelMonster
	* Add to AutoTS 'all' list
	* all to recombination_approved if so
