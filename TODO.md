# 1.0.5 🇺🇦 🇺🇦 🇺🇦
* Linear models: replaced inverse/Gram least-squares coefficient fits with direct solves and objective-preserving augmented least-squares fallback. BasicLinearModel now fits finite observations per series and computes normal observation intervals from the training fit, including short histories and trend damping. Cassandra skips redundant condition-number checks while preserving its existing recency weighting and automatic regularization.
* Feature detection and trend forecasting: Added joint detection of slope changes and level shifts; improved changepoint handling around missing data; added adaptive seasonality fitting, trend damping, slope shrinkage, and wider intervals for possible future trend changes.
* TVA: Added forecast origin anchoring and a configurable internal fit_horizon for factor and torch-free modes. Expanded factor forecasting, covariance estimation, and reconciliation behavior.
* Cassandra: Fixed autoregressive forecasting on either observed levels or detrended residuals. Changed the default multicollinearity filter to off.
* Anomaly detection: Added calibrated Gaussian mixture anomaly scores and improved handling of univariate and multivariate anomaly outputs. (why do LLMs like calibration so much?)
* Time-series utilities and models: Improved changepoint algorithms and probabilistic outputs, two-point frequency inference, short-history handling in transformations
* PWA dependencies: Updated the Rust dependency lockfile.
* LevelShiftMagic bug fixes that **change outputs at the defaults**: per-column selection (shift ids were shared across columns, a row-0 group stalled selection, one extra shift was kept), one flag per step (exact ties on integer data flagged both sides), detection windows need max(3, window_size // 3) points so series edges are no longer flagged and the last row can't leak an offset into forecasts, univariate mode combines per-series squared scores so opposite shifts no longer cancel and only individually significant series are corrected. Also: diff_overlap sizing (was ~0), remove_at_shift now sizes from raw data and fills only the removed points.
* LevelShiftMagic: opt-in window_method="linear_gap" (detrended multiscale MOSUM with robust z, bump test spike rejection), plus scales (default [14, 30, 60]), deseasonalize, anomaly_window, tail_scan (default on, linear_gap only), alignment "median"/"linear_robust" and correction_mode="auto" (safe ratio for non-negative series). Its alpha is a robust z threshold (3-6), not the legacy std multiplier. alignment defaults per method (None: "average" legacy, "linear_robust" linear_gap); linear_gap accepts only "linear_robust", "median" or "gap" and raises otherwise. No slower than "overlap" at the defaults; benchmark in examples/level_shift_benchmark.py.
  * Also: tail_scan only fills free max_level_shifts slots; univariate shared scores average over series with a valid score at each date (missing series no longer dilute a shared shift).
* TimeSeriesFeatureDetector (and so TVA): **default level shift params changed** to linear_gap (alpha 3.5, scales [14, 30, 60], linear_robust, tail_scan). The old default (window 364, alpha 1.8, rolling_diff) relied on the LevelShiftMagic bugs and gave ~2.1 validated false shifts per 2-year noise series once they were fixed; the new default measured recall 1.00 at 0.10-0.15.
* TimeSeriesFeatureDetector: new prelim_level_shift (default 'auto' = on for linear_gap params) runs level shift detection before the final seasonality fit, which otherwise absorbs steps near year boundaries.
* Holiday attribution: corrected lunar calendar horizon/start-range dependence; added opt-in Easter rules (`holiday_params={'use_easter_holidays': True}`) and per-series suppression (`global_holiday_anomaly_suppression=None`) with annual holiday peers and an exceptional-outage escape hatch. True retains batch suppression; False disables it. Detector defaults (including those used by TVA) are unchanged: attribution improved, but forecast error and synthetic precision regressed in validation. See examples/holiday_attribution_report.md.

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
