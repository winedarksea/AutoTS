# AutoTS MCP Server

Model Context Protocol (MCP) server for AutoTS, enabling LLM integration for time series forecasting. Provides 12 tools covering data loading, forecasting, analysis, and event risk prediction.

## Quick Start

```bash
# Install
pip install autots[mcp]

# Run server
autots-mcp

# Or programmatically
from autots.mcp.server import serve
serve()
```

### VS Code Integration

Add to MCP settings (`.vscode/mcp.json` or user settings):

```json
{
  "mcpServers": {
    "autots": {
      "command": "autots-mcp"
    }
  }
}
```
For local downloads
```json
{
	"servers": {
		"autots": {
			"type": "stdio",
			"command": "python",
			"args": ["-m", "autots.mcp.server"],
			"cwd": "~/Documents/AutoTS"
		}
	},
	"inputs": []
}
```

## Available Tools (12)

**Data (4 tools)**
- `get_sample_data` - Load built-in datasets (daily, hourly, weekly, monthly, yearly, linear, sine, artificial)
- `load_live_daily` - Load live data from FRED, stocks, Google Trends, weather APIs
- `generate_synthetic_data` - Generate synthetic time series with labeled components
- `long_to_wide_converter` - Convert between long/wide formats

**Forecasting (4 tools)**
- `forecast_fast` - **FAST**: Pre-configured profile ensemble using fit_data (no model search)
- `forecast_autots_search` - **MODERATE**: Explainable models only (Cassandra, TVVAR, BasicLinearModel)
- `forecast_custom` - **CUSTOM**: Full AutoTS capabilities with user parameters
- `get_autots_docs` - Documentation for custom parameters

**Analysis (2 tools)**
- `detect_features` - Detect anomalies, holidays, changepoints, and patterns
- `get_cleaned_data` - Clean data (handle missing values, outliers)

**Event Risk (2 tools)**
- `forecast_event_risk_default` - Probability of crossing thresholds (fast)
- `forecast_event_risk_tuning` - Tuned event risk forecasting (slower, more accurate)

## Data Format

Tools use **wide format** by default (datetime index + columns per series):

```json
{
  "datetime": ["2020-01-01", "2020-01-02", "2020-01-03"],
  "series1": [10.5, 11.2, 10.8],
  "series2": [20.1, 21.3, 19.9]
}
```

Long format is also supported with automatic conversion:

```json
{
  "datetime": ["2020-01-01", "2020-01-01"],
  "series_id": ["series1", "series2"],
  "value": [10.5, 20.1]
}
```

## Example Workflows

**Basic Forecast**
```
get_sample_data → forecast_fast
```

**Live Data Analysis**
```
load_live_daily → detect_features → get_cleaned_data → forecast_autots_search
```

**Event Risk**
```
get_sample_data → forecast_event_risk_default (with threshold)
```

**Custom Forecast**
```json
{
  "tool": "forecast_custom",
  "arguments": {
    "data": { ... },
    "forecast_length": 30,
    "autots_params": {
      "ensemble": "simple",
      "model_list": "fast",
      "max_generations": 5
    }
  }
}
```

## Tips

- Start with `forecast_fast` for fastest results
- Use `forecast_autots_search` when you need explainability or have insufficient data
- Event risk tools automatically convert absolute thresholds to quantiles
- All tools are chainable - output from one can be input to another
- Use `generate_synthetic_data` for testing and demos

## Testing

```bash
python -m pytest tests/test_mcp_server.py -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "MCP not available" | `pip install autots[mcp]` |

## Implementation Details

- **Dependencies**: `mcp>=1.0.0` (optional install group)
- **Entry point**: `autots-mcp` command via `pyproject.toml`
- **Architecture**: Async MCP server with stdio transport, tool-based interface
- **Error handling**: Comprehensive try/catch with informative error responses
- **Security**: No code execution, validated inputs, read-only docs access

## License

MIT License (same as AutoTS)
