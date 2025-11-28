import pandas as pd

# Read the existing forecast
df_forecast = pd.read_csv('/Users/colincatlin/Documents/AutoTS/test_forecast.csv', parse_dates=True, index_col=0)

# For demonstration, create upper/lower as point +/- 10% (you'd use actual prediction intervals)
# In real usage, the MCP server will have all three from the prediction object

# Convert to long format with forecast_type column
dfs = []

# Point forecast
df_point = df_forecast.copy()
df_point_long = df_point.reset_index().melt(
    id_vars=['index'],
    var_name='series_id',
    value_name='value'
)
df_point_long = df_point_long.rename(columns={'index': 'datetime'})
df_point_long['forecast_type'] = 'point'
dfs.append(df_point_long)

# Upper forecast (example: +10%)
df_upper = df_forecast * 1.1
df_upper_long = df_upper.reset_index().melt(
    id_vars=['index'],
    var_name='series_id',
    value_name='value'
)
df_upper_long = df_upper_long.rename(columns={'index': 'datetime'})
df_upper_long['forecast_type'] = 'upper'
dfs.append(df_upper_long)

# Lower forecast (example: -10%)
df_lower = df_forecast * 0.9
df_lower_long = df_lower.reset_index().melt(
    id_vars=['index'],
    var_name='series_id',
    value_name='value'
)
df_lower_long = df_lower_long.rename(columns={'index': 'datetime'})
df_lower_long['forecast_type'] = 'lower'
dfs.append(df_lower_long)

# Combine
df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all[['datetime', 'series_id', 'forecast_type', 'value']]

# Save
df_all.to_csv('/Users/colincatlin/Documents/AutoTS/test_forecast_all.csv', index=False)
print(f"Saved combined forecast with {len(df_all)} rows")
print(f"Shape: {df_all.shape}")
print(f"\nFirst few rows:")
print(df_all.head(15))
