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
)

from autots.evaluator.auto_ts import AutoTS
from autots.tools.transform import GeneralTransformer, RandomTransform
from autots.tools.shaping import long_to_wide

__version__ = '0.3.3'

TransformTS = GeneralTransformer

__all__ = [
    'load_daily',
    'load_monthly',
    'load_yearly',
    'load_hourly',
    'load_weekly',
    'load_weekdays',
    'AutoTS',
    'TransformTS',
    'GeneralTransformer',
    'RandomTransform',
    'long_to_wide',
]

# import logging
# logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.INFO)
