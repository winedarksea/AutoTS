# 1.0.5 🇺🇦 🇺🇦 🇺🇦
* Feature detection and trend forecasting: Added joint detection of slope changes and level shifts; improved changepoint handling around missing data; added adaptive seasonality fitting, trend damping, slope shrinkage, and wider intervals for possible future trend changes.
* TVA: Added forecast origin anchoring and a configurable internal fit_horizon for factor and torch-free modes. Expanded factor forecasting, covariance estimation, and reconciliation behavior.
* Cassandra: Fixed autoregressive forecasting on either observed levels or detrended residuals. Changed the default multicollinearity filter to off.
* Anomaly detection: Added calibrated Gaussian mixture anomaly scores and improved handling of univariate and multivariate anomaly outputs. (why do LLMs like calibration so much?)
* Time-series utilities and models: Improved changepoint algorithms and probabilistic outputs, two-point frequency inference, short-history handling in transformations
* PWA dependencies: Updated the Rust dependency lockfile.

### New Model Checklist:
* Add to ModelMonster in auto_model.py
* add to appropriate model_lists: all, recombination_approved if so, no_shared if so
* add to model table in extended_tutorial.md (most columns here have an equivalent model_list)
* if model has regressors, make sure it meets Simulation Forecasting needs (method=="regressor", fails on no regressor if "User")
* if model has result_windows, add to appropriate model_list noting also diff_window_motif_list

## New Transformer Checklist:
* Make sure that if it modifies the size (more/fewer columns or rows) it returns pd.DataFrame with proper index/columns
* add to transformer_dict
* add to trans_dict or have_params or external as appropriate
* add to shared_trans if transformer is multivariate (would produce different results if run on each series individually than if run on all together)
* add to oddities_list for those with forecast/original transform difference
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
