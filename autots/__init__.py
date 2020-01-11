"""
Automated Time Series Model Selection for Python

https://github.com/winedarksea/AutoTS
"""
from autots.datasets import load_toy_daily
from autots.evaluator.auto_ts import AutoTS

__version__ = '0.0.3'


__all__ = ['load_toy_daily', 'AutoTS']

# import logging
# logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.INFO)
