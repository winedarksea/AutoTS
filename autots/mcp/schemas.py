"""MCP Tool definitions for the AutoTS MCP server.

Combines tool schemas from:
  - schemas_data.py     — cache management + data loading/conversion tools
  - schemas_forecast.py — forecasting and prediction manipulation tools
  - schemas_features.py — event risk and feature detection tools
"""

from autots.mcp.schemas_data import DATA_TOOLS
from autots.mcp.schemas_forecast import FORECAST_TOOLS
from autots.mcp.schemas_features import FEATURES_TOOLS

TOOLS = DATA_TOOLS + FORECAST_TOOLS + FEATURES_TOOLS

__all__ = ["TOOLS"]
