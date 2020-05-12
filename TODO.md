# Basic Tenants
* Ease of Use > Accuracy > Speed
* All models should be probabilistic (upper/lower forecasts)
* All models should be able to handle multiple parallel time series (even if just a `for` loop)
* The expectation is series will largely be consistent in period, or at least up-sampled to regular intervals
* New transformations should be applicable to many datasets and models
* New models need only be sometimes applicable
* Fault tolerance: it is perfectly acceptable for model parameters to fail on some datasets, the higher level API will pass over and use others.

Latest:
	Changed default for `series_id` so it is no longer required if univariate
	Changed default of `subset` to None.
	Removed `weighted` parameter, now passing weights to .fit() alone is sufficient.
	Fixed a bug where 'One or more series is 90% or more NaN' was printing when it shouldn't
	Fixed (or more accurately, reduced) a bug where multiple initial runs were counting as validation runs.
	Fixed (partially!) bug where validation subsetting was behaving oddly
	Fixed bug where regressor wasn't being passed to validation.
	Made serious efforts to make the code prettier with pylint, still lots to do, however...
	Improved genetic recombination so optimal models should be reached more quickly.
	Improved Point to Probabilistic methods:
		'historic_quantile' more stable quantile-based error ranges
		'inferred normal' Bayesian-inspired method
	Metrics:
		Added Scaled Pinball Loss (SPL)
		Removed upper/lower MAE
	Improved ensembling with new parameter options.
		Recursive ensembling now enabled
	Added a number of new Transformer options
		Multiple new Sklearn-sourced transformers (QuantileTransformer, etc)
		SinTrend
		DifferencedDetrend
		CumSumTransformer
		PctChangeTransformer
		PositiveShift Transformer
		Log
		IntermittentOccurrence
		SeasonalDetrend
	Entirely changed the general transformer to add three levels of transformation.
	GLM
		Error where it apparently won't tolerate any zeroes was compensated for.
		Speed improvement.
	RollingRegression
		Added SVM model
		Added option to tune some model parameters to sklearn
		Fixed Holidays to work
		Added new feature construction parameters
		Added RNNs with Keras
	GluonTS:
		fixed the use of context_length, added more options to that param

	Dynamic Factor added uncertainty from Statsmodels Statespace
	VARMAX added uncertainty from Statsmodels Statespace
		
	New models:
		SeasonalNaive model
		VAR from Statsmodels (faster than VARMAX statespace)
		MotifSimulation
		WindowRegression
		TensorflowSTS
		TFPRegression

# Errors: 
DynamicFactor holidays 	Exceptions 'numpy.ndarray' object has no attribute 'values'
lower/upper MAE appearing NaN, and then getting a better score
VECM does not recognize exog to predict
ARIMA with User or Holiday ValueError('Can only compare identically-labeled DataFrame objects',)
kbins not working when it assigns fewer bins than n_bins asked for (use the property transformer.n_bins_ ?)
Drop Most Recent does not play well logically with added external (preord) regressors.
FastICA 'array must not contain infs or NaNs'
How do fillna methods handle datasets that have entirely NaN series?
Subsetting for validation samples seems to be funky.
VAR ValueError('Length of passed values is 4, index implies 9',)
WindowRegression + KerasRNN + 1step + univariate = ValueError('Length mismatch: Expected axis has 54 elements, new values have 9 elements',)
Is Template Eval Error: ValueError('array must not contain infs or NaNs',) related to Point to Probability HISTORIC QUANTILE?
Categorical forecast leaves bounds: IndexError: index 7 is out of bounds for axis 0 with size 6
	categorical = categorical_transformer.encoder.inverse_transform(df[cat_features].astype(int).values)
	THIS BREAKS PREDICT


### Ignored Errors:
xgboost poisson loss does not accept negatives
GluonTS not accepting quite a lot of frequencies
KerasRNN errors due to parameters not working on all dataset
Tensorflow GPU backend may crash on occasion.

## General Tasks
* simple, dist, (best3)horizontal
* mae, rmse1/rmse2, spl
	* max, top n
	* test on first segment before accepting
* Best 3 horizontal to before validation
* Horizontal but not max in pre stage
* handle subsetting with horizontal
* distance 20/80 horizontal
	* make model count of horizontal more visible
* horizontal ensembling for probabilistic
	* capture SPL per series
* method to test effectiveness across multiple probabilistic intervals.
	* generate base template (based mainly on best point forecast, validated)
	* run for each new prediction intervals (on just most recent)
	* capture containment/SPL from each round.
* overfitting of ensembles
	* after validation then run it on validation again
	* constraint effecting ensembling
	* can do both before and after validations
	* could do no validations and get ensemble, compare with run with more validations
* Fix categorical forecast when out of known values
	* .clip() on max and min for those with categorical
* test whether bottum up significantly overestimates on rollup


# To-Do
* Get the sphinx (google style) documentation and github pages website up
* Add to template: Gluon, Motif, WindowRegression
* Convert 'Holiday' regressors into Datepart + Holiday 2d
* Bring GeneralTransformer to higher level API.
	* wide_to_long and long_to_wide in higher-level API
* Option to use full traceback in errors in table
* Hierarchial
	* every level must be included in forecasting data
	* 'bottom-up' and 'mid' levels
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
* make datetime input optional, just allow dataframes of numbers
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
	* Add support for preord_regressor
	* Make sure of rolling regression setup
	* Modify GluonStart if lots of NaN at start of that series
	* GPU and CPU ctx
* Regressor to TensorflowSTS
* Relative/Absolute Imports and reduce package reloading messages
* 'Age' regressor as an option in addition to User/Holiday in ARIMA, etc.
* Handle categorical forecasts where forecast leaves range of known values, then add to upper/lower forecasts
* Speed improvements, Profiling
* Parallelization, and Distributed options (Dask) for general greater speed
* Improve usability on rarer frequenices (ie monthly data where some series start on 1st, others on 15th, etc.)
* Figures: Add option to output figures of train/test + forecast, other performance figures
* If all input are Int, convert floats back to int
* Hierachial correction (bottom-up to start with)
* Because I'm a biologist, incorporate more genetics and such. Also as a neuro person, there must be a way to fit networks in...
* Improved verbosity controls and options. 
* Replace most 'print' with logging.
* Analyze and return inaccuracy patterns (most inaccurate periods out, days of week, most inaccurate series)
* Development tools:
	* Add to Conda distribution as well as pip
	* Continuous integration
	* Code/documentation quality checkers
* Ability to automatically add external datasets of parallel time series of global usability (ie from FRED or others)
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
	REDUCE OVERFITTING IN MODEL CHOICE
	Other:
		Best SMAPE/MAE for point with Best Containment/UpperMAE/LowerMAE for probabilistic
		Best 10 combined with Decision Tree

#### New models:
	Croston, SBA, TSB, ADIDA, iMAPA
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

#### New Transformations:
	lag and beta to DifferencedTransformer to make it more of an AR process
	Weighted moving average
	Symbolic aggregate approximation (SAX) and (PAA) (basically these are just binning)
	Shared discretization (all series get same shared binning)
	Last Value Centering
	