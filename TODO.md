# 1.0.5 🇺🇦 🇺🇦 🇺🇦
* **Saved-template compatibility:** fixes and new defaults below can change forecasts when refitting templates from 1.0.4 or earlier.
* Linear models: replaced inverse/Gram least-squares coefficient fits with direct solves and objective-preserving augmented least-squares fallback. BasicLinearModel now fits finite observations per series and computes normal observation intervals from the training fit, including short histories and trend damping. Cassandra skips redundant condition-number checks.
* Feature detection and trend forecasting: Added joint detection of slope changes and level shifts; improved changepoint handling around missing data; added adaptive seasonality fitting, trend damping, slope shrinkage, and wider intervals for possible future trend changes.
* TVA: Added forecast origin anchoring and a configurable internal fit_horizon for factor and torch-free modes. With fit_horizon=None, the internal horizon is now capped at max(28, T // 4), changing long forecasts in factor and 'none' modes. Expanded factor forecasting, covariance estimation, and reconciliation behavior.
* Cassandra: Fixed ridge penalties to exempt the intercept instead of the first feature, changing explicit lambda and automatic regularization fits; x_scaler=True now fits an intercept. Revised level autoregression changes forecasts from 1.0.4 templates. Fixed autoregressive forecasting on detrended residuals. Changed the default multicollinearity filter to off.
* Anomaly detection: Added calibrated Gaussian mixture anomaly scores and improved handling of univariate and multivariate anomaly outputs. (why do LLMs like calibration so much?)
* Changepoints: CUSUM now uses a robust noise scale when normalized, changing threshold/drift semantics and detections (especially on trends); get_new_params still samples thresholds of 5–30, so existing settings may need retuning. PELT, L1 and L0 now honor min_segment_length; bottom_up treats max_changepoints as a cap instead of a floor; ED-PELT removed its ×5 penalty calibration and can find fewer changepoints. half_yr_spacing now uses elapsed rather than calendar years, changing default spacing in BasicLinearModel and transformers.
* Time-series utilities and transformations: fixed hourly Fourier features to include hours, changing hourly forecasts; FastICA now passes integer column labels (0..k-1) to subsequent transformers instead of series names. infer_frequency now returns None for fewer than three points. Improved probabilistic outputs and short-history handling.
* PWA dependencies: Updated the Rust dependency lockfile.
* LevelShiftMagic bug fixes that **change outputs at the defaults**: per-column selection (shift ids were shared across columns, a row-0 group stalled selection, one extra shift was kept), one flag per step (exact ties on integer data flagged both sides), detection windows need max(3, window_size // 3) points so series edges are no longer flagged and the last row can't leak an offset into forecasts, univariate mode combines per-series squared scores so opposite shifts no longer cancel and only individually significant series are corrected. Also: diff_overlap sizing (was ~0), remove_at_shift now sizes from raw data and fills only the removed points.
* LevelShiftMagic: opt-in window_method="linear_gap" (detrended multiscale MOSUM with robust z, bump test spike rejection), plus scales (default [14, 30, 60]), deseasonalize, anomaly_window, tail_scan (default on, linear_gap only), alignment "median"/"linear_robust" and correction_mode="auto" (safe ratio for non-negative series). Its alpha is a robust z threshold (3-6), not the legacy std multiplier. alignment defaults per method (None: "average" legacy, "linear_robust" linear_gap); linear_gap accepts only "linear_robust", "median" or "gap" and raises otherwise. No slower than "overlap" at the defaults; benchmark in examples/level_shift_benchmark.py.
  * Also: tail_scan only fills free max_level_shifts slots; univariate shared scores average over series with a valid score at each date (missing series no longer dilute a shared shift).
* TimeSeriesFeatureDetector (and so TVA): **default level shift params changed** to linear_gap (alpha 3.5, scales [14, 30, 60], linear_robust, tail_scan). The old default (window 364, alpha 1.8, rolling_diff) relied on the LevelShiftMagic bugs and gave ~2.1 validated false shifts per 2-year noise series once they were fixed; the new default measured recall 1.00 at 0.10-0.15.
* TimeSeriesFeatureDetector: new prelim_level_shift (default 'auto' = on for linear_gap params) runs level shift detection before the final seasonality fit, which otherwise absorbs steps near year boundaries.
* TimeSeriesFeatureDetector compatibility: older templates inherit trend_damping='auto', rather than undamped trends. New detector settings are not serialized into saved templates, so refits use their defaults. general_transformer_params={} now disables the transform instead of selecting the default.
* Holiday attribution: corrected lunar calendar horizon/start-range dependence; added opt-in Easter rules (`holiday_params={'use_easter_holidays': True}`) and per-series suppression (`global_holiday_anomaly_suppression=None`) with annual holiday peers and an exceptional-outage escape hatch. True retains batch suppression; False disables it. Detector defaults (including those used by TVA) are unchanged: attribution improved, but forecast error and synthetic precision regressed in validation. See examples/holiday_attribution_report.md.
* New get_new_params search options and their draw rates: BasicLinearModel linear_model='huber' (2%, not in fast; huber_iterations fixed at 50), ChangepointDetector l1_fused_lasso/l1_total_variation threshold_mode='noise' (20%) with threshold_multiplier, LevelShiftMagic window_method='linear_gap' (~13%), TVAModel origin_anchor (last_value 55%, deseasonalized 20%) and continuation_force='constant' (50%).
* RadiusNeighbors regression: features are standardized and each query's radius is a multiple of its own nearest-neighbor distance (sampled 1.1-5, capped at 50 neighbors), so it is never empty and adapts to series of different scale. At 1.0.4 the absolute radius returned NaN for most predictions on load_daily/load_monthly (RollingRegression failed outright), and the RadiusNeighbors-specific pchip fills in Window/DatepartRegression are removed; the general non-finite guards in the recursive models stay. Saved templates change meaning: radius < 1 now means the single nearest row, and large radii (5, 10) reach the 50-neighbor cap.
* TVAModel/TVA season_length: None (also older templates without the key) now infers the season for inner folds and seasonal-naive baselines instead of always using 7. The origin-anchor window stays 7. Safety-layer folds run at the full horizon when history allows and fall back to fit_horizon only when it does not.
* sklearn models no longer install a process-wide warnings filter at import; the Liblinear convergence warning is silenced only inside LinearSVR fits.

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
