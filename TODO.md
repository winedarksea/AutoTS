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
* BestN ensembles now support weighting model components
* cluster-based and generate_score_per_series-based 'simple' ensembles
* 'univariate' model_list added
* similarity and custom cross validation now set initial evaluation segment
	* validation_test_indexes and train now include initial eval segment
* 'subsample' ensemble expansion of 'simple'
* added Theta model from statsmodels
* added ARDL model from statsmodels
* expanded UnobservedComponents functionality, although it still fails on some params for unknown reasons
* fixed bug in AutoTS.predict() where it was breaking regressors in some cases
* transition from [] to None as default for no future_regressor
* enforce more extensive failing if regression_type==User and no regressor passed
* fixed regressor handling in DatepartRegression

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
