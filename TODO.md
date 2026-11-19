# 1.0.6 🇺🇦 🇺🇦 🇺🇦
* **prediction_interval is now central everywhere:** 0.8 means lower_forecast at q0.1 and upper_forecast at q0.9, as most models already did. LastValueNaive, AverageValueNaive, MotifSimulation, NVAR, KalmanStateSpace, GluonTS and every model using historic_quantile intervals (WindowRegression, UnobservedComponents, VECM) were one-sided (q0.2/q0.8) and now give wider bounds at the same setting. SPL (and KalmanStateSpace's internal tuning score) now scores the bounds at the central quantiles, so SPL values are not comparable with earlier runs and search rankings can shift; saved templates remain valid. historic_quantile now actually widens zero-width bounds (the fallback result was previously discarded). score_breakdown column 'contaiment' renamed to 'containment'. Audit: examples/interval_convention_audit.py.

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
