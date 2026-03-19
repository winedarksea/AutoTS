# -*- coding: utf-8 -*-
"""FeatureDetectionOptimizer - Hyperparameter optimization using synthetic data."""

import numpy as np
import pandas as pd
import random
import copy
import json
import time
import warnings


def _get_detector_class():
    """Lazy import to avoid circular dependency."""
    from .detector import TimeSeriesFeatureDetector
    return TimeSeriesFeatureDetector


def _get_loss_class():
    """Lazy import to avoid circular dependency."""
    from .loss import FeatureDetectionLoss
    return FeatureDetectionLoss


class FeatureDetectionOptimizer:
    """
    Optimize TimeSeriesFeatureDetector parameters using synthetic labeled data.

    Uses a genetic-style search with balanced scoring to minimize detection loss.
    """

    def __init__(
        self,
        synthetic_generator,
        loss_calculator=None,
        n_iterations=50,
        random_seed=42,
        starting_params=None,
    ):
        """
        Parameters
        ----------
        synthetic_generator : SyntheticDailyGenerator
            Generator with labeled synthetic data
        loss_calculator : FeatureDetectionLoss, optional
            Custom loss calculator
        n_iterations : int
            Number of random search iterations
        random_seed : int
            Random seed for reproducibility
        starting_params : dict, optional
            Optional detector parameter seed evaluated before random search.
        """
        self.synthetic_generator = synthetic_generator
        self.loss_calculator = loss_calculator or _get_loss_class()()
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        if starting_params is not None and not isinstance(starting_params, dict):
            raise ValueError("starting_params must be a dict or None.")
        self.starting_params = copy.deepcopy(starting_params)

        self.best_params = None
        self.best_loss = float('inf')
        self.best_total_loss = float('inf')
        self.optimization_history = []
        self.baseline_loss = None
        self.history_df = None

    def optimize(self, starting_params=None):
        """
        Run genetic-style optimization to find best detector parameters.

        Parameters
        ----------
        starting_params : dict, optional
            Optional seed parameter configuration. Overrides constructor value
            when provided.

        Returns
        -------
        dict
            Best parameters found
        """
        self.best_params = None
        self.best_loss = float('inf')
        self.best_total_loss = float('inf')
        self.optimization_history = []
        self.baseline_loss = None

        seed = self.starting_params if starting_params is None else starting_params
        if seed is not None and not isinstance(seed, dict):
            raise ValueError("starting_params must be a dict or None.")
        return self._random_search(starting_params=seed)

    def _default_detector_params(self):
        """Return a deep-copied set of default detector parameters."""
        detector = _get_detector_class()()
        return {
            'rough_seasonality_params': copy.deepcopy(
                detector.rough_seasonality_params
            ),
            'seasonality_params': copy.deepcopy(detector.seasonality_params),
            'holiday_params': copy.deepcopy(detector.holiday_params),
            'anomaly_params': copy.deepcopy(detector.anomaly_params),
            'changepoint_params': copy.deepcopy(detector.changepoint_params),
            'level_shift_params': copy.deepcopy(detector.level_shift_params),
            'general_transformer_params': copy.deepcopy(
                detector.general_transformer_params
            ),
            'standardize': detector.standardize,
            'smoothing_window': detector.smoothing_window,
            'extended_anomaly_params': copy.deepcopy(
                detector.extended_anomaly_params
            ),
        }

    def _random_search(self, starting_params=None):
        """Genetic-style optimization with balanced scoring."""
        rng = random.Random(self.random_seed)

        detector_for_sampling = _get_detector_class()()

        baseline_params = self._default_detector_params()
        evaluated_signatures = set()
        try:
            start_time = time.time()
            baseline_loss = self._evaluate_params(baseline_params)
            baseline_runtime = time.time() - start_time

            self.baseline_loss = baseline_loss['total_loss']
            baseline_history_entry = {
                'iteration': 'baseline',
                'params': copy.deepcopy(baseline_params),
                'loss': self.baseline_loss,
                'loss_breakdown': baseline_loss,
                'runtime': baseline_runtime,
            }
            self.optimization_history.append(baseline_history_entry)
            evaluated_signatures.add(self._param_signature(baseline_params))
            print(
                f"Baseline loss = {self.baseline_loss:.4f}, runtime = {baseline_runtime:.2f}s"
            )
        except Exception as e:
            print(f"Warning: Baseline evaluation failed with error: {e}")
            self.baseline_loss = None

        if starting_params is not None:
            starting_signature = self._param_signature(starting_params)
            if starting_signature in evaluated_signatures:
                print("Starting params match baseline; skipping duplicate evaluation.")
            else:
                try:
                    start_time = time.time()
                    starting_loss = self._evaluate_params(starting_params)
                    starting_runtime = time.time() - start_time

                    self.optimization_history.append(
                        {
                            'iteration': 'starting',
                            'params': copy.deepcopy(starting_params),
                            'loss': starting_loss['total_loss'],
                            'loss_breakdown': starting_loss,
                            'runtime': starting_runtime,
                        }
                    )
                    evaluated_signatures.add(starting_signature)
                    print(
                        f"Starting params loss = {starting_loss['total_loss']:.4f}, "
                        f"runtime = {starting_runtime:.2f}s"
                    )
                except Exception as e:
                    print(f"Warning: Starting params evaluation failed with error: {e}")

        successful_iterations = 0
        failed_iterations = 0

        for i in range(self.n_iterations):
            params = None
            attempts = 0
            parent_pool = sorted(
                self.optimization_history,
                key=lambda x: x.get('balanced_loss', x.get('loss', float('inf'))),
            )
            parent_pool = (
                parent_pool[: max(2, min(6, len(parent_pool)))] if parent_pool else []
            )

            # Generate new parameters, avoiding duplicates
            while (
                params is None or self._param_signature(params) in evaluated_signatures
            ):
                attempts += 1
                if parent_pool and rng.random() < 0.7:
                    if len(parent_pool) >= 2:
                        chosen = rng.sample(parent_pool, 2)
                        params = self._crossover_params(
                            chosen[0]['params'], chosen[1]['params'], rng
                        )
                    else:
                        params = copy.deepcopy(parent_pool[0]['params'])
                    if rng.random() < 0.6:
                        params = self._mutate_params(params, detector_for_sampling, rng)
                else:
                    params = detector_for_sampling.get_new_params(method='random')
                if attempts > 8:
                    # Bug fix: ensure we have valid params even if duplicates persist
                    # Force a fresh random sample as last resort
                    params = detector_for_sampling.get_new_params(method='random')
                    break

            if params is None:
                continue

            # Double-check signature (may still be duplicate if max attempts reached)
            signature = self._param_signature(params)
            if signature in evaluated_signatures:
                continue

            try:
                start_time = time.time()
                loss = self._evaluate_params(params)
                runtime = time.time() - start_time

                record = {
                    'iteration': successful_iterations,
                    'params': copy.deepcopy(params),
                    'loss': loss['total_loss'],
                    'loss_breakdown': loss,
                    'runtime': runtime,
                }
                self.optimization_history.append(record)
                evaluated_signatures.add(signature)
                successful_iterations += 1

                # Print progress for every iteration
                if i % 20 == 0 or successful_iterations == 1:
                    print(
                        f"Iteration {i} ({successful_iterations} successful): "
                        f"raw loss = {loss['total_loss']:.4f}, runtime = {runtime:.2f}s"
                    )
            except Exception as e:
                failed_iterations += 1
                if failed_iterations <= 3:
                    print(f"Iteration {i} failed: {str(e)[:100]}")
                continue

        if failed_iterations > 3:
            print(f"... and {failed_iterations - 3} more failures (suppressed)")

        # Calculate runtime statistics
        runtimes = [
            entry.get('runtime')
            for entry in self.optimization_history
            if entry.get('runtime') is not None
        ]
        if runtimes:
            avg_runtime = np.mean(runtimes)
            min_runtime = np.min(runtimes)
            max_runtime = np.max(runtimes)
            total_runtime = np.sum(runtimes)

        print(f"\nOptimization iterations complete!")
        print(f"Successful iterations: {successful_iterations}/{self.n_iterations}")

        # Print runtime statistics
        if runtimes:
            print(f"\nRuntime statistics:")
            print(f"  Total runtime: {total_runtime:.2f}s")
            print(f"  Average runtime per iteration: {avg_runtime:.2f}s")
            print(f"  Min runtime: {min_runtime:.2f}s")
            print(f"  Max runtime: {max_runtime:.2f}s")

        # Now select best model based on properly calculated balanced scores
        print(f"\nCalculating balanced scores and selecting best model...")
        best_params = self._select_best_from_history()

        return best_params

    def _evaluate_params(self, params):
        """Evaluate a parameter configuration."""
        # Create detector with these params
        detector = _get_detector_class()(**params)

        # Fit on synthetic data
        detector.fit(self.synthetic_generator.get_data())

        # Get detected features
        detected_features = detector.get_detected_features(include_components=True)

        # Get true labels
        true_labels = self.synthetic_generator.get_all_labels()
        true_components = self.synthetic_generator.get_components()

        # Calculate loss
        loss = self.loss_calculator.calculate_loss(
            detected_features,
            true_labels,
            true_components=true_components,
            date_index=self.synthetic_generator.date_index,
        )

        return loss

    @staticmethod
    def _param_signature(params):
        """Create a hashable signature for parameter configurations."""
        try:
            canonical = FeatureDetectionOptimizer._signature_safe_value(params)
            return json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        except Exception:
            return repr(params)

    @staticmethod
    def _signature_safe_value(value):
        """Convert potentially non-serializable params into deterministic JSON-safe form."""
        if isinstance(value, dict):
            return {
                str(k): FeatureDetectionOptimizer._signature_safe_value(v)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        if isinstance(value, (list, tuple)):
            return [FeatureDetectionOptimizer._signature_safe_value(v) for v in value]
        if isinstance(value, set):
            converted = [
                FeatureDetectionOptimizer._signature_safe_value(v) for v in value
            ]
            return sorted(converted, key=lambda x: json.dumps(x, sort_keys=True))
        if isinstance(value, np.ndarray):
            return FeatureDetectionOptimizer._signature_safe_value(value.tolist())
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            val = float(value)
            if not np.isfinite(val):
                return str(val)
            return val
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, float) and not np.isfinite(value):
                return str(value)
            return value
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        # Deterministic fallback for custom objects.
        return {'__class__': value.__class__.__name__, '__repr__': str(value)}

    def _crossover_params(self, parent_a, parent_b, rng):
        child = copy.deepcopy(parent_a)
        for key in child.keys():
            if key in parent_b and rng.random() < 0.5:
                child[key] = copy.deepcopy(parent_b[key])
        return child

    def _mutate_params(self, params, sampler, rng):
        mutated = copy.deepcopy(params)
        fresh = sampler.get_new_params(method='random')
        # Only mutate keys that exist in both the current params and fresh sample
        shared_keys = [k for k in mutated.keys() if k in fresh]
        if not shared_keys:
            return mutated
        count = max(1, min(len(shared_keys), 2))
        for key in rng.sample(shared_keys, count):
            mutated[key] = copy.deepcopy(fresh[key])
        return mutated

    def _select_best_from_history(self):
        """
        Post-process optimization history to select best model based on balanced scores.

        Converts history to DataFrame, calculates balanced scores with fixed scalers,
        and selects the model with the best balanced loss.

        Returns
        -------
        dict
            Best parameters based on balanced scoring
        """
        if not self.optimization_history:
            return None

        # Build DataFrame from history
        rows = []
        for entry in self.optimization_history:
            row = {
                'iteration': entry.get('iteration'),
                'loss': entry.get('loss'),
                'runtime': entry.get('runtime'),
            }
            # Add all loss breakdown components
            breakdown = entry.get('loss_breakdown', {})
            for key in self.loss_calculator.weights.keys():
                row[key] = breakdown.get(key, np.nan)
            rows.append(row)

        self.history_df = pd.DataFrame(rows)

        # Calculate scalers based on entire history using a robust lower quantile,
        # which avoids pathological domination from a single tiny metric value.
        scalers = {}
        for key in self.loss_calculator.weights.keys():
            col = self.history_df[key].replace([np.inf, -np.inf], np.nan)
            positive = col[col > 0].dropna()
            if positive.empty:
                scalers[key] = 1.0
            else:
                scale = float(np.nanpercentile(positive, 25))
                if not np.isfinite(scale) or scale <= 1e-6:
                    scale = float(np.nanmedian(positive))
                if np.isfinite(scale) and scale > 1e-6:
                    scalers[key] = scale
                else:
                    scalers[key] = 1.0

        # Calculate balanced loss for each entry
        balanced_losses = []
        for idx, entry in enumerate(self.optimization_history):
            balanced = 0.0
            breakdown = entry.get('loss_breakdown', {})
            for key, weight in self.loss_calculator.weights.items():
                value = breakdown.get(key)
                if value is None or not np.isfinite(value):
                    continue
                balanced += weight * (value / scalers.get(key, 1.0))
            balanced_losses.append(balanced)
            # Store balanced loss back in history entry
            entry['balanced_loss'] = balanced

        self.history_df['balanced_loss'] = balanced_losses

        # Select from the strongest balanced candidates, then require the best raw loss
        # within that pool so we don't prefer configurations with worse total loss.
        balanced_arr = np.asarray(balanced_losses, dtype=float)
        raw_arr = np.asarray(
            [entry.get('loss', np.inf) for entry in self.optimization_history],
            dtype=float,
        )
        valid_idx = np.where(np.isfinite(balanced_arr) & np.isfinite(raw_arr))[0]
        if valid_idx.size == 0:
            best_idx = int(np.nanargmin(balanced_arr))
            candidate_pool_size = 1
        else:
            sorted_balanced = valid_idx[np.argsort(balanced_arr[valid_idx])]
            candidate_pool_size = max(1, min(8, int(np.ceil(sorted_balanced.size * 0.2))))
            top_candidates = sorted_balanced[:candidate_pool_size]
            best_idx = int(top_candidates[np.argmin(raw_arr[top_candidates])])

        best_entry = self.optimization_history[best_idx]

        self.best_params = copy.deepcopy(best_entry['params'])
        self.best_loss = best_entry['balanced_loss']
        self.best_total_loss = best_entry['loss']

        # Find baseline entry for comparison
        baseline_entry = None
        for entry in self.optimization_history:
            if entry.get('iteration') == 'baseline':
                baseline_entry = entry
                break

        if baseline_entry:
            baseline_balanced = baseline_entry.get(
                'balanced_loss', baseline_entry['loss']
            )
            improvement = baseline_balanced - self.best_loss
            improvement_pct = (
                (improvement / baseline_balanced * 100) if baseline_balanced != 0 else 0
            )

            print(f"\n{'='*80}")
            print(f"OPTIMIZATION RESULTS")
            print(f"{'='*80}")
            print(
                f"Baseline balanced loss: {baseline_balanced:.4f} (raw: {baseline_entry['loss']:.4f})"
            )
            print(
                f"Best balanced loss:     {self.best_loss:.4f} (raw: {self.best_total_loss:.4f})"
            )
            print(f"Selection pool size:    {candidate_pool_size}")
            print(f"Improvement:            {improvement:.4f} ({improvement_pct:.2f}%)")
            print(f"Best found at iteration: {best_entry.get('iteration')}")

        return self.best_params

    def get_optimization_summary(self):
        """Return summary of optimization results."""
        summary = {
            'method': 'genetic_search',
            'n_iterations': len(self.optimization_history),
            'best_loss': self.best_loss,
            'baseline_loss': self.baseline_loss,
            'best_total_loss': self.best_total_loss,
            'best_params': copy.deepcopy(self.best_params)
            if self.best_params
            else None,
        }

        if self.optimization_history:
            losses = [
                h.get('balanced_loss', h.get('loss', float('inf')))
                for h in self.optimization_history
            ]
            summary['initial_loss'] = losses[0]
            summary['final_loss'] = losses[-1]
            summary['worst_loss'] = max(losses)
            summary['mean_loss'] = np.mean(losses)
            summary['std_loss'] = np.std(losses)

        component_ranges = {}
        frozen_components = []
        disabled_component_counts = {}
        if self.optimization_history:
            for key in self.loss_calculator.weights.keys():
                values = []
                for entry in self.optimization_history:
                    breakdown = entry.get('loss_breakdown') or {}
                    val = breakdown.get(key)
                    if val is not None and np.isfinite(val):
                        values.append(float(val))
                if values:
                    comp_min = float(np.min(values))
                    comp_max = float(np.max(values))
                    comp_range = comp_max - comp_min
                    component_ranges[key] = {
                        'min': comp_min,
                        'max': comp_max,
                        'range': comp_range,
                    }
                    if comp_range <= 1e-9:
                        frozen_components.append(key)

            for entry in self.optimization_history:
                breakdown = entry.get('loss_breakdown') or {}
                disabled = breakdown.get('disabled_components') or []
                for key in disabled:
                    disabled_component_counts[key] = (
                        disabled_component_counts.get(key, 0) + 1
                    )

        if component_ranges:
            summary['component_ranges'] = component_ranges
        if frozen_components:
            summary['frozen_components'] = sorted(frozen_components)
        if disabled_component_counts:
            summary['disabled_components'] = sorted(disabled_component_counts.keys())
            summary['disabled_component_counts'] = disabled_component_counts

        return summary

    # ------------------------------------------------------------------
    # Changepoint fine-tuning with curriculum Focal Tversky loss
    # ------------------------------------------------------------------

    def fine_tune_changepoints(
        self,
        starting_params,
        n_per_stage=200,
        curriculum_sigmas=None,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_gamma=2.0,
        level_shift_weight=0.5,
        exclude_changepoint_methods=None,
    ):
        """
        Focused fine-tuning pass that freezes every parameter group except
        ``changepoint_params`` and ``level_shift_params``.

        All other parameters (seasonality, anomaly, holiday, etc.) are held fixed
        so the optimizer can zero in on changepoint quality without interference.

        The loss function is a statistical translation of techniques designed for
        neural changepoint training:

        Gaussian Label Smoothing
            Instead of a hard ±tolerance binary label, each true changepoint is
            represented as a Gaussian probability distribution centred on its date
            with standard deviation ``sigma``.  This converts the step-function
            loss landscape into smooth, convex basins and ensures that detections
            that are "close but not exact" receive a constructive gradient signal.

        Focal Tversky Loss (statistical translation)
            The metric used for scoring is the Focal Tversky index with
            ``alpha < beta`` (default 0.3 / 0.7), which heavily penalises false
            negatives over false positives, directly preventing the zero-prediction
            collapse that plagues changepoint tuning.  The focal exponent
            ``gamma=2.0`` concentrates the gradient on partially-matched
            changepoints rather than already-correct ones.

        Curriculum Learning (sigma annealing)
            Three stages with decreasing sigma drive the search from coarse to
            fine sensitivity:
              Stage 1: sigma=14 days — wide window builds initial recall
              Stage 2: sigma=7 days  — medium window matches ±7-day tolerance
              Stage 3: sigma=3.5 days — tight window polishes placement precision

        Parameters
        ----------
        starting_params : dict
            Full detector parameter dict to use as the frozen baseline.
            All keys except ``changepoint_params`` and ``level_shift_params``
            are immutably frozen throughout the run.
        n_per_stage : int
            Number of candidate configurations evaluated per curriculum stage.
        curriculum_sigmas : list of float, optional
            Sigma values (in days) for each curriculum stage.
            Defaults to [14.0, 7.0, 3.5].
        tversky_alpha : float
            FP weight in Tversky denominator (keep < tversky_beta).
        tversky_beta : float
            FN weight in Tversky denominator (keep > tversky_alpha).
        tversky_gamma : float
            Focal exponent applied to (1 - Tversky_index).
        level_shift_weight : float
            Blend weight for level-shift Tversky loss in the final score
            (trend changepoints get 1 - weight).
        exclude_changepoint_methods : list of str, optional
            Changepoint method names to exclude from the search.  Defaults to
            ``['basic']``, which prevents the evenly-spaced pseudo-detector
            from being selected (it cannot be used for analytic purposes).
            Pass an empty list ``[]`` to allow all methods including 'basic'.

        Returns
        -------
        dict
            Best full parameter dict found, with only changepoint/level-shift
            params potentially changed from ``starting_params``.
        """
        if curriculum_sigmas is None:
            curriculum_sigmas = [14.0, 7.0, 3.5]
        if exclude_changepoint_methods is None:
            exclude_changepoint_methods = ['basic']

        rng = random.Random(self.random_seed + 9999)
        detector_for_sampling = _get_detector_class()()

        current_best_params = copy.deepcopy(starting_params)
        current_best_loss = float('inf')
        self.fine_tune_history = []

        print(
            f"\nStarting changepoint fine-tuning "
            f"({len(curriculum_sigmas)} stages × {n_per_stage} iters, "
            f"Tversky α={tversky_alpha} β={tversky_beta} γ={tversky_gamma})"
        )

        for stage_idx, sigma in enumerate(curriculum_sigmas):
            print(
                f"\n--- Stage {stage_idx + 1}/{len(curriculum_sigmas)}: "
                f"sigma={sigma:.1f} days ---"
            )
            stage_best, stage_loss, stage_entries = self._run_changepoint_stage(
                current_best_params,
                sigma,
                n_per_stage,
                detector_for_sampling,
                rng,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                tversky_gamma=tversky_gamma,
                level_shift_weight=level_shift_weight,
                stage_idx=stage_idx,
                exclude_changepoint_methods=exclude_changepoint_methods,
            )
            self.fine_tune_history.extend(stage_entries)

            if stage_loss < current_best_loss:
                current_best_params = stage_best
                current_best_loss = stage_loss
                print(f"  Stage {stage_idx + 1}: improved → loss={stage_loss:.4f}")
            else:
                print(
                    f"  Stage {stage_idx + 1}: no improvement "
                    f"(best so far: {current_best_loss:.4f})"
                )

        print(f"\nFine-tuning complete.  Best Tversky loss: {current_best_loss:.4f}")
        return current_best_params

    def _run_changepoint_stage(
        self,
        frozen_params,
        sigma,
        n_iters,
        detector_for_sampling,
        rng,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_gamma=2.0,
        level_shift_weight=0.5,
        stage_idx=0,
        exclude_changepoint_methods=None,
    ):
        """
        Run one curriculum stage, mutating only changepoint_params and
        level_shift_params while holding everything else frozen.
        """
        from autots.tools.changepoints import ChangepointDetector
        from autots.tools.transform import LevelShiftMagic

        best_params = copy.deepcopy(frozen_params)

        try:
            best_loss = self._evaluate_changepoint_params(
                frozen_params, sigma, tversky_alpha, tversky_beta,
                tversky_gamma, level_shift_weight,
            )
        except Exception:
            best_loss = float('inf')

        history = [{
            'sigma': sigma,
            'iteration': 'stage_start',
            'params': copy.deepcopy(frozen_params),
            'loss': best_loss,
        }]
        evaluated_sigs = {self._param_signature(frozen_params)}

        for i in range(n_iters):
            candidate = copy.deepcopy(best_params)
            if i % 20 == 0:
                print(f"  Stage {stage_idx + 1} iter {i}/{n_iters} (best loss so far: {best_loss:.4f})")

            # Always resample changepoint_params
            fresh_cp = ChangepointDetector.get_new_params(method='random')
            if exclude_changepoint_methods and fresh_cp.get('method') in exclude_changepoint_methods:
                continue
            fresh_cp['aggregate_method'] = 'individual'
            candidate['changepoint_params'] = fresh_cp

            # Resample level_shift_params ~70 % of the time
            if rng.random() < 0.7:
                fresh_ls = LevelShiftMagic.get_new_params(method='random')
                fresh_ls['output'] = 'multivariate'
                candidate['level_shift_params'] = fresh_ls

            sig = self._param_signature(candidate)
            if sig in evaluated_sigs:
                continue
            evaluated_sigs.add(sig)

            try:
                loss = self._evaluate_changepoint_params(
                    candidate, sigma, tversky_alpha, tversky_beta,
                    tversky_gamma, level_shift_weight,
                )
                history.append({
                    'sigma': sigma,
                    'iteration': i,
                    'params': copy.deepcopy(candidate),
                    'loss': loss,
                })
                if loss < best_loss:
                    best_loss = loss
                    best_params = copy.deepcopy(candidate)
                    print(f"    iter {i}: improved → {loss:.4f}")
            except Exception:
                pass

        return best_params, best_loss, history

    def _evaluate_changepoint_params(
        self,
        params,
        sigma,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_gamma=2.0,
        level_shift_weight=0.5,
    ):
        """
        Score a parameter config using only the Focal Tversky changepoint loss.

        Runs a full detector fit (required because changepoints are extracted
        from the decomposed residual series), then extracts trend_changepoints
        and level_shifts per series and computes the Focal Tversky penalty with
        the given Gaussian sigma — bypassing all seasonality, anomaly, and
        holiday components so the gradient is purely changepoint-driven.
        """
        detector = _get_detector_class()(**params)
        detector.fit(self.synthetic_generator.get_data())
        detected_features = detector.get_detected_features(include_components=False)
        true_labels = self.synthetic_generator.get_all_labels()

        series_names = self.loss_calculator._resolve_series_names(
            detected_features, true_labels, None
        )
        if not series_names:
            return float('inf')

        total_cp_loss = 0.0
        total_ls_loss = 0.0
        n_scored = 0

        for name in series_names:
            det_series = self.loss_calculator._extract_detected_series(
                detected_features, name
            )
            true_series = self.loss_calculator._extract_true_series(true_labels, name)

            # Trend changepoints
            det_cp_entries = [
                self.loss_calculator._parse_trend_event(e)
                for e in det_series.get('trend_changepoints', [])
            ]
            true_cp_entries = [
                self.loss_calculator._parse_trend_event(e)
                for e in true_series.get('trend_changepoints', [])
            ]
            cp_loss = self.loss_calculator._focal_tversky_changepoint_penalty(
                det_cp_entries, true_cp_entries,
                sigma=sigma, alpha=tversky_alpha,
                beta=tversky_beta, gamma=tversky_gamma,
            )

            # Level shifts — slightly lighter FN weight (harder to localise exactly)
            det_ls_entries = [
                self.loss_calculator._parse_level_shift_event(e)
                for e in det_series.get('level_shifts', [])
            ]
            true_ls_entries = [
                self.loss_calculator._parse_level_shift_event(e)
                for e in true_series.get('level_shifts', [])
            ]
            ls_beta = max(tversky_beta - 0.1, tversky_alpha + 0.05)
            ls_loss = self.loss_calculator._focal_tversky_changepoint_penalty(
                det_ls_entries, true_ls_entries,
                sigma=sigma, alpha=tversky_alpha,
                beta=ls_beta, gamma=tversky_gamma,
            )

            total_cp_loss += cp_loss
            total_ls_loss += ls_loss
            n_scored += 1

        if n_scored == 0:
            return float('inf')

        avg_cp = total_cp_loss / n_scored
        avg_ls = total_ls_loss / n_scored
        cp_weight = 1.0 - level_shift_weight
        return cp_weight * avg_cp + level_shift_weight * avg_ls

