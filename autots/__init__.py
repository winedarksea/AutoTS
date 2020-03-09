"""
Automated Time Series Model Selection for Python

https://github.com/winedarksea/AutoTS
"""
from autots.datasets import load_toy_hourly
from autots.datasets import load_toy_daily
from autots.datasets import load_toy_monthly
from autots.datasets import load_toy_yearly
from autots.datasets import load_toy_weekly

from autots.evaluator.auto_ts import AutoTS

__version__ = '0.1.5'


__all__ = ['load_toy_daily','load_toy_monthly', 'load_toy_yearly', 'load_toy_hourly', 'load_toy_weekly',
           'AutoTS']

# import logging
# logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.INFO)
