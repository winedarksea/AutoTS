# AutoTS MCP Server

Model Context Protocol (MCP) server for AutoTS, enabling LLM integration for time series forecasting. Provides **28 tools** covering data loading, forecasting, feature detection, event risk prediction, and post-hoc forecast adjustments.

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

## Available Tools (28)

### Cache Management (2)
- `list_cache` — List all cached objects (predictions, models, data, detectors). Call first to rediscover IDs.
- `clear_cache` — Clear specific objects, an entire cache type, or all caches.

### Data Loading & Preparation (7)
- `load_sample_data` — Load built-in datasets: `daily`, `hourly`, `weekly`, `monthly`, `yearly`, `linear`, `sine`, `artificial`
- `load_live_data` — Load live data from FRED economic API or stock tickers (requires network)
- `generate_synthetic_data` — Generate synthetic daily time series with labeled components for testing
- `load_data_from_file` — Load a CSV from a local file path or URL
- `get_data` — Retrieve cached data as JSON (wide/long) or save as CSV
- `convert_long_to_wide` — Convert long-format data (datetime, series_id, value) to wide format
- `clean_data` — Fill missing values and handle outliers (ffill, mean, median, rolling_mean, linear)

### Forecasting (4)
- `forecast_fast` — **FAST**: Pre-configured mosaic ensemble using `fit_data` (no model search, seconds)
- `forecast_explainable` — **MODERATE**: AutoTS model search restricted to interpretable models (Cassandra, TVVAR, BasicLinearModel)
- `forecast_custom` — **CUSTOM**: Full AutoTS with user-specified parameters or a model template
- `get_autots_docs` — Parameter documentation for `forecast_custom`. Call before `forecast_custom`.

### Forecast Result Tools (6)
- `get_forecast` — Retrieve point/upper/lower forecasts as JSON or CSV. Use `output='all'` for combined long format.
- `plot_forecast` — Plot forecast with history and prediction intervals. Returns base64 PNG.
- `apply_constraints` — Dampen growth, enforce upper/lower bounds, or clip by quantile. Returns new `prediction_id`.
- `apply_adjustments` — Post-hoc adjustments: linear ramp, align-to-last-value, or EWMA smoothing. Returns new `prediction_id`.
- `get_model_params` — Get model name, parameters, and transformation parameters.
- `get_forecast_components` — Decompose forecast into trend/seasonality components (Cassandra and TVVAR models only).

### AutoTS Model Tools (2)
- `get_validation_results` — Cross-validation results and top model rankings from a model search.
- `plot_validation` — Plot cross-validation forecasts. Returns base64 PNG.

### Event Risk (3)
- `forecast_event_risk` — Probability of crossing a threshold over future periods (stockout risk, capacity breach, etc.)
- `get_event_risk_results` — Retrieve event risk probabilities as JSON or CSV.
- `plot_event_risk` — Plot event risk probabilities over the forecast horizon. Returns base64 PNG.

### Feature Detection (4)
- `detect_features` — Detect anomalies, changepoints, level shifts, holidays, and seasonality patterns.
- `get_detected_features` — Query detected features with optional date-range or series filters.
- `plot_features` — Plot detected features overlaid on the time series. Returns base64 PNG.
- `forecast_from_features` — (Experimental) Forecast from decomposed feature-detector components.

## ID-Based Workflow

Tools use a **cache-and-ID pattern**: each operation stores results in a server-side cache and returns an ID. Pass that ID to downstream tools.

```
load_sample_data → data_id
    ↓
forecast_fast(data_id) → prediction_id
    ↓
get_forecast(prediction_id)
plot_forecast(prediction_id)
apply_constraints(prediction_id) → new prediction_id
```

Use `list_cache` at any time to see all live IDs.

## Data Format

Tools use **wide format** by default (datetime index + one column per series):

```json
{
  "datetime": ["2020-01-01", "2020-01-02", "2020-01-03"],
  "series1": [10.5, 11.2, 10.8],
  "series2": [20.1, 21.3, 19.9]
}
```

Long format is also accepted with automatic conversion:

```json
{
  "datetime": ["2020-01-01", "2020-01-01"],
  "series_id": ["series1", "series2"],
  "value": [10.5, 20.1]
}
```

## Example Workflows

**Quickest Forecast**
```
load_sample_data → forecast_fast → get_forecast
```

**Explainable Forecast with Components**
```
load_data_from_file → forecast_explainable → get_forecast_components
                                           → get_validation_results
```

**Live Data Analysis**
```
load_live_data → detect_features → get_detected_features
              → clean_data → forecast_custom → get_forecast
```

**Event Risk**
```
load_sample_data → forecast_event_risk(threshold=0.9) → get_event_risk_results → plot_event_risk
```

**Custom Forecast with Constraints**
```
load_data_from_file → forecast_custom(autots_params={...}) → apply_constraints → apply_adjustments → get_forecast
```

**Custom `forecast_custom` parameters:**
```json
{
  "tool": "forecast_custom",
  "arguments": {
    "data_id": "<data_id>",
    "forecast_length": 30,
    "autots_params": {
      "ensemble": "simple",
      "model_list": "fast",
      "max_generations": 5,
      "num_validations": 2
    }
  }
}
```
Call `get_autots_docs` first to see all available `autots_params` options.

## Tips

- **Start with `forecast_fast`** — fastest results, good baseline
- **Use `forecast_explainable`** when you need interpretability or component decomposition
- **Event risk thresholds**: values in [0, 1] are treated as historical quantiles; outside [0, 1] are absolute thresholds
- **All tools are chainable** — every output ID can be passed as input to the next tool
- **`detect_features` before forecasting** — reveals anomalies and structural breaks that affect model selection
- **`apply_adjustments` with `align_last_value`** — corrects model drift against recent actuals; requires `data_id`
- **Cache persists for the server lifetime** — use `list_cache` to rediscover IDs across conversation turns

## Testing

```bash
python -m pytest tests/test_mcp_server.py -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `"MCP not available"` | `pip install autots[mcp]` |
| `"X_id not found in cache"` | Call `list_cache` to see live IDs; cache is per-process and resets on server restart |
| Forecast takes too long | Use `forecast_fast` or reduce `max_generations` in `forecast_custom` |
| `align_last_value` requires `data_id` | Pass the original `data_id` from the load step alongside the `prediction_id` |

## Implementation Details

- **Dependencies**: `mcp>=1.0.0` (optional install group)
- **Entry point**: `autots-mcp` command via `pyproject.toml`
- **Architecture**: Async MCP server, stdio transport, low-level `Server` API
- **Structured outputs**: Tools with `outputSchema` return `structuredContent` (MCP 1.x) plus auto-generated `content[0].text`
- **Progress notifications**: Long-running tools emit `notifications/message` log events so clients can show progress
- **Cache eviction**: Oldest entries evicted when cache exceeds `AUTOTS_MCP_CACHE_MAX` (default 60). Set via env var.

## License

MIT License (same as AutoTS)
