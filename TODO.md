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
* **breaking change** to model templates: transformers structure change
	* grouping no longer used
* parameter generation for transformers allowing more possible combinations
* transformer_max_depth parameter
* Horizontal Ensembles are now much faster by only running models on the subset of series they apply to
* general starting template improved and updated to new transformer format
* change many np.random to random
	* random.choices further necessitates python 3.6 or greater
* bug fix in Detrend transformer
* bug fix in SeasonalDifference transformer
* SPL bug fix when NaN in test set
* inverse_transform now fills NaN with zero for upper/lower forecasts
* expanded model_list aliases, with dedicated module
* bug fix (creating 0,0 order) and tuning of VARMAX
* Fix export_template bug
* restructuring of some function locations


# Known Errors: 
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
ValueError: percentiles should all be in the interval [0, 1]. Try [-0.00089  0.01089] instead. .sample in Motif Simulation line 729 point_method == 'sample'

### Ignored Errors:
xgboost poisson loss does not accept negatives
GluonTS not accepting quite a lot of frequencies
KerasRNN errors due to parameters not working on all dataset
Tensorflow GPU backend may crash on occasion.

## Concerns
* resource utilization at scale
* overfitting on first train segment (train progressive subsets?)
* End Users add their own models
* Improve documentation and usability of lower level code
* better summarization of many time series into a few high-information time series as parallel or regressors

## To-Do
* Migrate to-do to GitHub issues and project board
* Remove 'horizontal' sanity check run, takes too long (only if metric weights are x)?
* Horizontal and BestN runtime variant, where speed is highly important in model selection
* total runtime for .fit() as attribute (not just manual sum but capture in ModelPrediction)
* allow Index to be other datetime not just DatetimeIndex
* cleanse similar models out first, before horizontal ensembling
* BestNEnsemble Add 5 or more model option
* allow best_model to be specified and entirely bypass the .fit() stage.
* drop duplicates as function of TemplateEvalObject
* improve test.py script for actual testing of many features
* Convert 'Holiday' regressors into Datepart + Holiday 2d
* export and import of results includes all model parameters (but not templates?)
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
	* uncomp with uncertainty intervals
* GUI overlay for editing/creating templates, and for running (Flask)
* Datepart Regression
	* add holiday
* RollingRegression
	* Better X_maker:
		* 1d and 2d variations
		* .cov, .skew, .kurt, .var
		* https://link.springer.com/article/10.1007/s10618-019-00647-x/tables/1
	* Probabilistic:
		https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
* GluonTS
	* Add support for future_regressor
	* Modify GluonStart if lots of NaN at start of that series
	* GPU and CPU ctx
* implement 'borrow' Genetic Recombination for ComponentAnalysis
* Replace OrdinalEncoder with non-external code
* 'Age' regressor as an option in addition to User/Holiday in ARIMA, etc.
* Figures: Add option to output figures of train/test + forecast, other performance figures
* Replace most 'print' with logging.
 * still use verbose: set 0 = error, 1 = info, 2 = debug
* Analyze and return inaccuracy patterns (most inaccurate periods out, days of week, most inaccurate series)
* Ability to automatically add external datasets of parallel time series of global usability (ie from FRED or others)
* make datetime input optional, just allow dataframes of numbers
* Infer column names for df_long to wide based on which is datetime, which is string, and which is numeric
* Option to run generations until generations no longer see improvement of at least X % over n generations
* tune probability of parameters for models and transformers
* ignore series of weight 0 in univariate models

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
	lag and beta to DifferencedTransformer to make it more of an AR process
	Weighted moving average
	Symbolic aggregate approximation (SAX) and (PAA) (basically these are just binning)
	Scipy filter Transformer scipy.signal.lfilter — SciPy v1.5.4 Reference Guide
	Transformer that removes all median IQR values. Selective filters that remove some patterns...

### New Model Checklist:
	* Add to ModelMonster in auto_model.py
	* add to appropriate model_lists: all, recombination_approved if so, no_shared if so
	* add to model table in extended_tutorial.md (most columns here have an equivalent model_list)

## New Transformer Checklist:
	* Make sure that if it modifies the size (more/fewer columns or rows) it returns pd.DataFrame with proper index/columns
	* depth of recombination is?
	* add to "all" transformer_dict
	* add to no_params or external if so
	* add to no_shared if so, in auto_model.py
	* oddities_list for those with forecast/original transform difference
