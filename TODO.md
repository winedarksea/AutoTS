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

# 0.6.22 🇺🇦 🇺🇦 🇺🇦
* added ReconciliationTransformer
* updated cointegration code, replaced Cointegration with CointegrationTransformer
* added mocks for dependency fallbacks
* added variational autoencoder anomaly detection method
* some fixes for breaking changes in dependencies
* adjustment to how custom_metric is scaled so it can work with negatives
* improvements to the calendars, Hindu calendar should be working now
* changes to HistoricValues which hopefully makes it more reliable
* added pMLP and MambaSSM models (painfully slow)
* deleted old models ComponentAnalysis, TFPRegression, TensorflowSTS, Greykite, NeuralProphet. Unlikely you were using these, deprecated for a while
* improvements to model_interrupt
* synthetic data generation and feature extractor new, very much in beta
* updated weather data to CDO v2 (requires API key)
* added NASA solar data to load_live_daily
* created an MCP server for AutoTS
* many bug fixes, tweaks, and added unittests

### Unstable Upstream Pacakges (those that are frequently broken by maintainers)
* Pytorch-Forecasting
* NeuralForecast
* GluonTS

### New Model Checklist:
	* Add to ModelMonster in auto_model.py
	* add to appropriate model_lists: all, recombination_approved if so, no_shared if so
	* add to model table in extended_tutorial.md (most columns here have an equivalent model_list)
	* if model has regressors, make sure it meets Simulation Forecasting needs (method=="regressor", fails on no regressor if "User")
	* if model has result_windows, add to appropriate model_list noting also diff_window_motif_list

## New Transformer Checklist:
	* Make sure that if it modifies the size (more/fewer columns or rows) it returns pd.DataFrame with proper index/columns
	* add to transformer_dict
	* add to trans_dict or have_params or external
	* add to shared_trans if so
	* oddities_list for those with forecast/original transform difference
	* add to docstring of GeneralTransformer
	* add to dictionary by type: filter, scaler, transformer
	* add to test_transform call

## New Metric Checklist:
	* Create function in metrics.py
	* Add to mode base full_metric_evaluation  (benchmark to make sure it is still fast)
	* Add to concat in TemplateWizard (if per_series metrics will be used)
	* Add to concat in TemplateEvalObject (if per_series metrics will be used)
	* Add to generate_score
	* Add to generate_score_per_series (if per_series metrics will be used)
	* Add to validation_aggregation
	* Update test_metrics results
	* metric_weighting in AutoTS, get_new_params, prod example, test
