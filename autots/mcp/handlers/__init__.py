"""Tool handler registry for the AutoTS MCP server.

TOOL_HANDLERS maps each tool name to its async handler function.
Every handler has the signature: async def handler(arguments: dict, log_progress) -> ...
"""

from autots.mcp.handlers.data import DATA_HANDLERS
from autots.mcp.handlers.forecast import FORECAST_HANDLERS
from autots.mcp.handlers.prediction import PREDICTION_HANDLERS
from autots.mcp.handlers.features import FEATURES_HANDLERS

TOOL_HANDLERS = {
    **DATA_HANDLERS,
    **FORECAST_HANDLERS,
    **PREDICTION_HANDLERS,
    **FEATURES_HANDLERS,
}

__all__ = ["TOOL_HANDLERS"]
