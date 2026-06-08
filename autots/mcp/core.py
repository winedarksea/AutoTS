"""Transport-neutral dispatch for AutoTS tools.

``run_tool`` is the single entry point shared by the MCP server, the Pyodide
worker, and any general-purpose Python API. Handlers return plain,
JSON-serializable data (dicts / lists); transport layers wrap the result as
needed (the MCP server wraps it in ``TextContent`` / ``ImageContent``, the
Pyodide worker serializes it to JSON, etc.).

This module deliberately avoids importing ``mcp`` so it can run in environments
(such as Pyodide) where the MCP SDK is not installed.
"""

import asyncio
import inspect

from autots.mcp.handlers import TOOL_HANDLERS


async def _noop_progress(message):
    """Default progress callback that does nothing."""
    return None


def _ensure_async_progress(progress_cb):
    """Normalize ``progress_cb`` into an awaitable ``async def(message)``.

    Accepts ``None`` (no-op), a sync callable, or an async callable so callers
    can pass whatever is convenient for their transport.
    """
    if progress_cb is None:
        return _noop_progress
    if inspect.iscoroutinefunction(progress_cb):
        return progress_cb

    async def _wrapped(message):
        progress_cb(message)

    return _wrapped


def available_tools():
    """Return the sorted list of registered tool names."""
    return sorted(TOOL_HANDLERS)


async def run_tool(name, arguments=None, progress_cb=None):
    """Dispatch a tool call and return plain, JSON-serializable data.

    Args:
        name: Registered tool name (see :func:`available_tools`).
        arguments: Dict of tool arguments.
        progress_cb: Optional ``None``, sync, or async callable taking a single
            progress-message string.

    Returns:
        The handler's plain-data result (a dict or list). Image-producing tools
        return ``{"image_base64": ..., "mime_type": ...}``. Unknown tool names
        return ``{"error": ...}`` rather than raising.
    """
    arguments = arguments or {}
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {
            "error": f"Unknown tool: {name}",
            "available_tools": available_tools(),
        }
    progress = _ensure_async_progress(progress_cb)
    return await handler(arguments, progress)


def run_tool_sync(name, arguments=None, progress_cb=None):
    """Synchronous wrapper around :func:`run_tool` for non-async callers."""
    return asyncio.run(run_tool(name, arguments, progress_cb))
