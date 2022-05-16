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
)

from autots.evaluator.auto_ts import AutoTS
from autots.evaluator.event_forecasting import EventRiskForecast
from autots.tools.transform import GeneralTransformer, RandomTransform
from autots.tools.shaping import long_to_wide
from autots.tools.regressor import create_lagged_regressor, create_regressor
from autots.evaluator.auto_model import model_forecast

__version__ = '0.4.1'

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
    'AutoTS',
    'TransformTS',
    'GeneralTransformer',
    'RandomTransform',
    'long_to_wide',
    'model_forecast',
    'create_lagged_regressor',
    'create_regressor',
    'EventRiskForecast',
]
