"""Time Series Feature Detection and Optimization package."""

from .detector import TimeSeriesFeatureDetector
from .extended_anomaly import (
    ExtendedAnomalyDetector,
    default_extended_anomaly_params,
)
from .loss import FeatureDetectionLoss, ReconstructionLoss
from .optimizer import FeatureDetectionOptimizer

# Re-export names that were previously importable from the old single-file module
# so that existing code (including unittest.mock.patch paths) continues to work.
from autots.datasets.synthetic import SyntheticDailyGenerator  # noqa: F401

__all__ = [
    "TimeSeriesFeatureDetector",
    "ExtendedAnomalyDetector",
    "default_extended_anomaly_params",
    "FeatureDetectionLoss",
    "ReconstructionLoss",
    "FeatureDetectionOptimizer",
]
