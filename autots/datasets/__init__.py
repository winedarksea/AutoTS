"""
Tools for Importing Sample Data
"""
from autots.datasets._base import load_toy_daily
from autots.datasets._base import load_toy_monthly
from autots.datasets._base import load_toy_yearly
from autots.datasets._base import load_toy_hourly

__all__ = ['load_toy_daily', 'load_toy_monthly', 'load_toy_yearly', 'load_toy_hourly']