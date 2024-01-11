"""
用于导入示例数据的工具
"""
from autots.datasets._base import (
    load_daily,
    load_live_daily,
    load_monthly,
    load_yearly,
    load_hourly,
    load_weekly,
    load_weekdays,
    load_zeroes,
    load_linear,
    load_sine,
    load_artificial,
)

__all__ = [
    'load_daily',
    'load_monthly',
    'load_yearly',
    'load_hourly',
    'load_weekly',
    'load_weekdays',
    'load_live_daily',
    'load_zeroes',
    'load_linear',
    'load_sine',
    'load_artificial',
]
