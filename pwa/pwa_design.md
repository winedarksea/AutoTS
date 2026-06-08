# Progressive Web App for AutoTS
Main goal: an elegant browser-based progressive web app for forecasting for beginners
Design language: Material Design 3 (skill https://github.com/hamen/material-3-skill/blob/master/skills/material-3/SKILL.md)
Proposed infrastructure: Rust frontend (perhaps Leptos), Pyodide backend.
Proposed data flow: Rust (polars) <-> arrow <-> Python pandas (pyodide)

The basic app is going to consist of data upload. After data upload, the feature detector is run and the user is shown their uploaded data as an interactive line graph including feature detector labels. Users can select making a forecast (which uses the fast feature detector forecast) or a do forecast search button which uses the main AutoTS search. Users should be able to expand and input parameters, but are not shown many by default until they expand the options. Once the forecast is made, users can view it, drag each data point to adjust it, then download the data (reflecting adjustments, if used).

Implemented UI requirements:
	Data preview and download are available as soon as cleaned data is loaded. Feature detection is optional enrichment; detector failures are shown inline and do not suppress the data plot.
	Forecast plots distinguish actuals from forecast values and let users choose how many recent actual periods to display.
	Cached objects are summarized with one row per object, with full metadata available in expandable details.
	Forecast point adjustments are collapsed by default until the adjustment UI is redesigned.
	Loaded data and adjusted forecasts can both be downloaded as wide CSV files.

The long term goal is to include more pages and features in the app, all of the current MCP functions and more. So the design should be extensible.
Another long term goal is more UI based preprocessing, such as users removing anomalies in the UI, with the data updates tracked and passed back to the data used for forecasting.

Goals:
	Make it clear to an LLM how they are to load their data in and get a response.
		As a starting idea this might mean pairing every visual chart component with a screen-reader-friendly or structured data alternative inside the DOM. Include llms.txt. 
	Share/reuse code with the MCP server.py, as much as possible. The MCP code is in beta, so we can change the MCP code as needed to make sure it is most elegantly shared.
		Headless core reused by the PWA, the MCP server, and the MCP UI
			autots/mcp/handlers.py and autots/mcp/schemas_*.py and autots/mcp/cache.py
		Prepare for possibly using an MCP UI in the future:
		https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/
		https://github.com/modelcontextprotocol/ext-apps/tree/main/examples/system-monitor-server
		It looks like the MCP Apps version would be focused on just core functionality, as Buttons: "make forecast", "search for best forecast", "search all night for best forecast"
	It would be nice to have a UI view of the app's cached time series data, so users can review/select/delete as desired.
	The python backend api should be designed to be easily used as a general purpose api for making forecasts.
	Forecasting should be async or otherwise not block the UI frontend or backend.

Upload of input data is likely the biggest challenge from the user's perspective, and so needs to be designed well.
Upload:
	Data load is handled in Python. Rust passes filename or URL (string), the copied text. Python manages the "database" of files and Rust can retrieve, delete, etc by api.
	Sources:
		Copy and paste input that receives tsv, csv
		Upload file option (csv or excel)
		Input link option (example shows from Google Sheets -> Share -> Publish to Web -> CSV)
		Load sample or live daily data (from AutoTS already) (hint to users they can download this as an example to pass to an LLM to reformat their data as needed)
	Documentation describes what long and wide style are
	Users have the option to select "long or wide" and if long, what the column names are for each column. However these default to "auto"
	Here is how auto works:
		The most common 'flaw' in non technical users source data is they have a spreadsheet with a bunch of random other calculations in it, not centered, and empty rows for padding. Our basic cleanup task is to discard any of this. First we delete any rows or columns that are completely empty. Then we discard any that are 95% or more empty (null).
		Once the empty rows and columns are deleted, we can now load the data.
		 If only two columns and we can parse one as a date and one as a value, then assume it is just a single time series (effectively wide)
        We can guess date by what parses as a date (strings with hypens or dashes passed to pandas datetime auto loader), value as what parses as a number, and series id as what parses as a string (only works if dates are common formats, ids are not integer ids). If many columns with numeric ids, and one that is datetime, assume it is wide style data.
