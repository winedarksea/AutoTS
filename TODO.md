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
* added 'simple_2" datepart method
* added package to conda-forge
* preclean method added to AutoTS
* added median and midhinge point_methods to BestN Ensembles
* removed datasets requests dependency
* Added EWMAFilter
* improved NaN in forecast check

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
