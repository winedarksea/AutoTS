"""
Allow running autots.mcp as a module: python -m autots.mcp
"""

from autots.mcp.server import serve

if __name__ == "__main__":
    serve()
