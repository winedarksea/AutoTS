"""
MCP Server for AutoTS Time Series Forecasting

Main entry point and request routing. Delegates to:
  - autots.mcp.schemas    — Tool definitions
  - autots.mcp.prompts    — Prompt & Resource definitions
  - autots.mcp.handlers   — Tool execution logic
  - autots.mcp.cache      — State management

Re-exports of sub-module symbols are provided at the bottom of this file for
backwards compatibility (tests and other code that imports from autots.mcp.server).
"""

import asyncio
import json
import logging
from typing import Any

# Re-export cache utilities (no MCP dependency)
from autots.mcp.cache import (
    CACHE_REGISTRY,
    CACHE_MAX_OBJECTS,
    CACHE_SUMMARY_KEYS,
    _resolve_cache,
    _enforce_cache_limit,
    cache_object,
    get_cached_object,
    list_all_cached_objects,
    clear_cache,
)

# Re-export data utilities (no MCP dependency)
from autots.mcp.data_utils import (
    serialize_timestamps,
    load_to_dataframe,
    dataframe_to_output,
    save_temp_csv,
    build_csv_metadata,
)

# Re-export prompts utilities
from autots.mcp.prompts import read_resource

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent

    import matplotlib

    matplotlib.use('Agg')

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP not available. Install with: pip install autots[mcp]")

logger = logging.getLogger(__name__)


if MCP_AVAILABLE:
    from autots.mcp.schemas import TOOLS
    from autots.mcp.prompts import PROMPTS, get_resources, read_resource, get_prompt
    from autots.mcp.handlers import TOOL_HANDLERS

    app = Server("autots")

    async def _log_progress(message: str) -> None:
        """Send an MCP log notification so clients can show progress."""
        try:
            await app.request_context.session.send_log_message("info", message)
        except Exception:
            pass

    @app.list_tools()
    async def list_tools():
        return TOOLS

    @app.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Dispatch tool calls to the appropriate handler."""
        try:
            handler = TOOL_HANDLERS.get(name)
            if handler is None:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Unknown tool: {name}"}, separators=(',', ':')
                        ),
                    )
                ]
            return await handler(arguments, _log_progress)
        except Exception:
            logger.exception(f"Error in tool {name}")
            raise

    @app.list_resources()
    async def list_resources():
        return get_resources(__file__)

    @app.read_resource()
    async def read_resource_handler(uri: str) -> str:
        return await read_resource(uri)

    @app.list_prompts()
    async def list_prompts():
        return PROMPTS

    @app.get_prompt()
    async def get_prompt_handler(name: str, arguments: dict | None = None):
        return await get_prompt(name, arguments)


def serve():
    """Start the MCP server."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP package not installed. Install with: pip install autots[mcp]"
        )

    logging.basicConfig(level=logging.INFO)

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream, write_stream, app.create_initialization_options()
            )

    asyncio.run(main())


if __name__ == "__main__":
    serve()
