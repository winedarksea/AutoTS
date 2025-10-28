# Optimizer Fix Summary

## Problem
The `FeatureDetectionOptimizer` was finding better results than baseline during iterations, but still returning baseline parameters as the best result.

## Root Cause
The optimizer was calling `_apply_balanced_scores()` **after every iteration**, which recalculated the balanced loss for **all entries in the history** (including the baseline). Since balanced scoring uses normalization based on the minimum positive value of each metric across all evaluated models, adding new models changed the normalization scalers, which retroactively changed the baseline's balanced score.

This meant:
1. Baseline evaluated → balanced score = X
2. New model evaluated → `_apply_balanced_scores()` called on entire history
3. Baseline's balanced score changes to Y (due to new scalers)
4. Comparison `if current_balanced < self.best_loss` uses the **moving target** of baseline
5. Even better models might not trigger the update because baseline score keeps shifting

## Solution
Refactored the optimizer to:
1. **Remove dynamic tracking** - Don't try to track "best" during the loop
2. **Store all results** - Just collect all evaluation results in `optimization_history`
3. **Post-process at the end** - Calculate balanced scores once after all iterations complete
4. **Select best model** - Use `_select_best_from_history()` utility function

### New Method: `_select_best_from_history()`
```python
def _select_best_from_history(self):
    """
    Post-process optimization history to select best model based on balanced scores.
    
    - Converts history to DataFrame (stored in self.history_df)
    - Calculates balanced scores with fixed scalers based on entire history
    - Selects the model with the best balanced loss
    - Returns best parameters
    """
```

### Changes Made
1. **Added**: `_select_best_from_history()` method
2. **Removed**: `_apply_balanced_scores()` method (no longer needed)
3. **Modified**: `_random_search()` to:
   - Only print raw loss during iterations
   - Call `_select_best_from_history()` at the end
   - Remove dynamic best tracking logic
4. **Added**: `self.history_df` attribute to store results DataFrame
5. **Improved**: Result reporting with clearer baseline comparison

## Benefits
- ✅ Balanced scores are calculated once with consistent normalization
- ✅ Baseline score remains fixed throughout optimization
- ✅ Best model selection is deterministic and reproducible
- ✅ History DataFrame available for analysis (`optimizer.history_df`)
- ✅ Clear reporting of improvement over baseline

## Testing
Created `test_optimizer_fix.py` which:
- Generates small synthetic dataset
- Runs optimizer with 5 iterations
- Verifies history DataFrame is created
- Compares best result to baseline
- Confirms best model selection works correctly

## Example Output
```
Optimization iterations complete!
Successful iterations: 3/5

Calculating balanced scores and selecting best model...

================================================================================
OPTIMIZATION RESULTS
================================================================================
Baseline balanced loss: 8.0220 (raw: 177.5154)
Best balanced loss:     7.8932 (raw: 175.2341)
Improvement:            0.1288 (1.61%)
Best found at iteration: 2
```
