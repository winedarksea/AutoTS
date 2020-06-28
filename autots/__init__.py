"""
Automated Time Series Model Selection for Python

https://github.com/winedarksea/AutoTS
"""
from autots.datasets import load_hourly
from autots.datasets import load_daily
from autots.datasets import load_monthly
from autots.datasets import load_yearly
from autots.datasets import load_weekly

from autots.evaluator.auto_ts import AutoTS
from autots.tools.transform import GeneralTransformer
from autots.tools.shaping import long_to_wide

__version__ = '0.2.2'


__all__ = ['load_daily','load_monthly', 'load_yearly', 'load_hourly', 'load_weekly',
           'AutoTS', 'GeneralTransformer', 'long_to_wide']

# import logging
# logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.INFO)
