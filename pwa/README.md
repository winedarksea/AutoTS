# AutoTS PWA

A browser-based, beginner-friendly forecasting app for [AutoTS](../). Upload a
spreadsheet → it auto-cleans and detects the format → see an interactive chart
with detected features → make a fast forecast or run a search → drag sliders to
adjust points → download the result. **Everything runs locally in your browser**
(Python via Pyodide/WASM) — no server, no API key, your data never leaves the
machine.

## Architecture

```
Leptos (Rust→WASM, UI thread)  ──callTool(JSON)──►  Web Worker
  · Material Design 3 UI                              · Pyodide (Python→WASM)
  · SVG charts + accessible tables                    · numpy/pandas/scipy/…
  · slider-based point adjustment                     · autots wheel (micropip)
                                                       · autots.mcp.pyodide_api
```

The seam between Rust and Python is a **language-neutral JS facade**
([js/autots_client.js](js/autots_client.js)) plus a JSON message protocol
([worker/pyodide_worker.js](worker/pyodide_worker.js)). All forecasting logic
lives in Python (`autots/mcp/`), shared with the AutoTS MCP server. The Leptos
view layer is intentionally thin and swappable — a TypeScript + Plotly rewrite
would bind to the same `autotsClient` facade without touching Python.

## Prerequisites

- Rust + the wasm target: `rustup target add wasm32-unknown-unknown`
- [Trunk](https://trunkrs.dev): `cargo install trunk`
- Python (to build the AutoTS wheel): `pip install build`

## Run

```bash
cd pwa
trunk serve --open
trunk serve --port 8082 --open=false
```

The `post_build` hook ([scripts/build_wheel.sh](scripts/build_wheel.sh)) builds
the AutoTS wheel into `dist/autots.whl` (cached under `.wheel/`; pass `--force`
to rebuild after changing Python code). Then open the served page, load a sample
dataset, and forecast.

First load downloads Pyodide + scientific packages from the jsDelivr CDN
(network required). A service worker caches the complete pinned runtime while
it starts. When the header shows **Offline ready**, the in-app Install action
creates an app that can relaunch and forecast without a network connection.

Completed datasets and forecasts are stored as versioned JSON artifacts in
IndexedDB, so they survive tab refreshes. The app keeps up to 25 forecasts,
25 standalone datasets, and 200 MiB using least-recently-used cleanup.

Forecasting runs in a named browser worker (`AutoTS Forecast Worker`) that can
be cancelled from the status area. Cancellation terminates and recreates the
worker, then restores the active dataset from IndexedDB. Browser PWAs cannot
rename the browser-owned operating-system processes shown by Task Manager or
Activity Monitor.

## Smoke test (Python-in-browser, no UI)

```bash
trunk build                      # produces dist/ + dist/autots.whl
python -m http.server -d dist 8088
# open http://localhost:8088/../smoke/pyodide_smoke.html   (serve repo so ../autots.whl resolves)
```

Or serve the `pwa/` directory and open `/smoke/pyodide_smoke.html` after copying
`dist/autots.whl` next to it. The harness runs `smart_load → make_forecast →
search_forecast` and prints pass/fail.

## Notes / limitations

- Heavy models (TensorFlow / Prophet / XGBoost / LightGBM) are unavailable in
  WASM; forecasting uses a Pyodide-safe pure-Python model set
  (`autots.mcp.pyodide_api.PYODIDE_SEARCH_MODELS`) with `n_jobs=1`.
- "Search all night" is best-effort given single-threaded WASM.
- Point adjustment currently uses accessible range sliders (one per forecast
  point); SVG pointer-drag is a planned enhancement. The CSV download always
  reflects the adjusted values.
- Data exchange is JSON today; the worker boundary is designed so Arrow IPC
  (polars ↔ pyarrow) can replace it later without UI changes.
- Spreadsheet upload supports `.xlsx`; legacy `.xls` files must be saved as
  `.xlsx` or exported as CSV.
