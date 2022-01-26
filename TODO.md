# Basic Tenants
* Ease of Use > Accuracy > Speed (with speed more important with 'fast' selections)
* Availability of models which share information among series
* All models should be probabilistic (upper/lower forecasts)
* New transformations should be applicable to many datasets and models
* New models need only be sometimes applicable
* Fault tolerance: it is perfectly acceptable for model parameters to fail on some datasets, the higher level API will pass over and use others.
* Missing data tolerance: large chunks of data can be missing and model will still produce reasonable results (although lower quality than if data is available)

## Assumptions on Data
* Series will largely be consistent in period, or at least up-sampled to regular intervals
* The most recent data will generally be the most important
* Forecasts are desired for the future immediately following the most recent data.

# Latest
* fix bug where score was failing to generate if made_weighting > 0 and forecast_length = 1
* made MADE functional on forecast_length=1 if df_train is provided
* SMAPE now in __repr__ of fit AutoTS class
* contour now works on forecast_length = 1
* Added NeuralProphet model
* made the probabilistic modeling of MultivariateRegression a parameter which only occurs when 'deep' mode is active (too slow)
* added more params to pass through to Prophet
* add phi damping to Detrend transformer
* add window slice to Detrend transformer
* added 'simple_2" datepart method
* added package to conda-forge
* preclean method added to AutoTS
* added median and midhinge point_methods to BestN Ensembles
* added additional model selections to 'simple' and 'subsample' ensemble
* switch LightGBM back to MultiOutputRegressor from RegressorChain for speed (with LightGBMRegressorChain replacing)
* removed top-level datasets `requests` dependency
* Added EWMAFilter
* improved NaN in forecast check
* added fail_on_forecast_nan (bool) to AutoTS.predict and model_forecast, if False, can now allow forecasts with NaN to be returned
* fixed bug where Detrend failing with non-datetime index
* improved error handling in Transformers to explicitly reference which failed
* added random.seed() setting in AutoTS which actually seems to standardize the runs
* sped up assembling/concat of horizontal ensembles for large numbers of series
* added polynomial_degree to date_part (and ~Transformer and ~Regression)
* updated infer_frequency and utilized in model base class
* added additional datasets (analytics.gov and severe weather) to load_lively_daily and modified pytrends load
* added rps to metrics (although no plans to build it into evaluation)
* added 'Ridge' and 'GaussianProcessRegressor' as model options for Regressions
* enforcing consistency on inner n_jobs with MultiOutputRegressor
* add DynamicFactorMQ model from Statsmodels
* added plot_per_series_smape and list_failed_model_types to output more run information from AutoTS class

### New Model Checklist:
	* Add to ModelMonster in auto_model.py
	* add to appropriate model_lists: all, recombination_approved if so, no_shared if so
	* add to model table in extended_tutorial.md (most columns here have an equivalent model_list)
	* if model has regressors, make sure it meets Simulation Forecasting needs (method="regressor", fails on no regressor if "User")

## New Transformer Checklist:
	* Make sure that if it modifies the size (more/fewer columns or rows) it returns pd.DataFrame with proper index/columns
	* depth of recombination is?
	* add to "all" transformer_dict
	* add to no_params or external if so
	* add to no_shared if so, in auto_model.py (shared_trans)
	* oddities_list for those with forecast/original transform difference
