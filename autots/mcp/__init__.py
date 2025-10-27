"""
Model Context Protocol (MCP) Server for AutoTS

This package provides an MCP server interface for AutoTS forecasting capabilities,
enabling LLM integration for time series forecasting tasks.
"""

from autots.mcp.server import serve

__all__ = ['serve']
