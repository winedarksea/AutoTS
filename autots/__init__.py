"""
Automated Time Series Model Selection for Python

https://github.com/winedarksea/AutoTS
"""

from autots.datasets import (
    load_hourly,
    load_daily,
    load_monthly,
    load_yearly,
    load_weekly,
    load_weekdays,
    load_live_daily,
    load_linear,
    load_artificial,
    load_sine,
)

from autots.evaluator.auto_ts import AutoTS
from autots.evaluator.event_forecasting import EventRiskForecast
from autots.tools.transform import GeneralTransformer, RandomTransform
from autots.tools.shaping import long_to_wide, infer_frequency
from autots.tools.regressor import create_lagged_regressor, create_regressor
from autots.evaluator.auto_model import model_forecast, ModelPrediction
from autots.evaluator.anomaly_detector import AnomalyDetector, HolidayDetector
from autots.models.cassandra import Cassandra


__version__ = '0.6.17'

TransformTS = GeneralTransformer

__all__ = [
    'load_daily',
    'load_monthly',
    'load_yearly',
    'load_hourly',
    'load_weekly',
    'load_weekdays',
    'load_live_daily',
    'load_linear',
    'load_artificial',
    'load_sine',
    'AutoTS',
    'TransformTS',
    'GeneralTransformer',
    'RandomTransform',
    'long_to_wide',
    'model_forecast',
    'create_lagged_regressor',
    'create_regressor',
    'EventRiskForecast',
    'AnomalyDetector',
    'HolidayDetector',
    'Cassandra',
    'infer_frequency',
    'ModelPrediction',
]
