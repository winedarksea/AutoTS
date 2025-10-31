"""Quick test of pMLP with CNN blocks."""
import numpy as np
import pandas as pd
from autots.models.deepssm import pMLP

# Create test data with seasonal patterns
n_timesteps = 250
n_series = 3
dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')

np.random.seed(42)
# Create data with weekly seasonality that the CNN should capture
time_idx = np.arange(n_timesteps)
weekly_pattern = np.sin(time_idx * 2 * np.pi / 7)
monthly_pattern = np.sin(time_idx * 2 * np.pi / 30)

data = np.zeros((n_timesteps, n_series))
for i in range(n_series):
    data[:, i] = (weekly_pattern * (i + 1) + 
                  monthly_pattern * 0.5 + 
                  time_idx * 0.02 + 
                  np.random.randn(n_timesteps) * 0.3)

df = pd.DataFrame(data, index=dates, columns=['A', 'B', 'C'])

print("Testing pMLP without CNN...")
model_no_cnn = pMLP(
    context_length=30,
    hidden_dims=[64, 32],
    epochs=2,
    batch_size=16,
    verbose=1,
    random_seed=42,
    num_cnn_blocks=0  # No CNN
)
model_no_cnn.fit(df)
pred_no_cnn = model_no_cnn.predict(forecast_length=14)
print(f"✓ No CNN - Forecast shape: {pred_no_cnn.forecast.shape}")

print("\nTesting pMLP with 1 CNN block...")
model_1cnn = pMLP(
    context_length=30,
    hidden_dims=[64, 32],
    epochs=2,
    batch_size=16,
    verbose=1,
    random_seed=42,
    num_cnn_blocks=1  # 1 CNN block
)
model_1cnn.fit(df)
pred_1cnn = model_1cnn.predict(forecast_length=14)
print(f"✓ 1 CNN block - Forecast shape: {pred_1cnn.forecast.shape}")

print("\nTesting pMLP with 2 CNN blocks...")
model_2cnn = pMLP(
    context_length=30,
    hidden_dims=[64, 32],
    epochs=2,
    batch_size=16,
    verbose=1,
    random_seed=42,
    num_cnn_blocks=2  # 2 CNN blocks
)
model_2cnn.fit(df)
pred_2cnn = model_2cnn.predict(forecast_length=14)
print(f"✓ 2 CNN blocks - Forecast shape: {pred_2cnn.forecast.shape}")

print("\nAll tests passed!")
print(f"Fit time - No CNN: {model_no_cnn.fit_runtime}")
print(f"Fit time - 1 CNN: {model_1cnn.fit_runtime}")
print(f"Fit time - 2 CNN: {model_2cnn.fit_runtime}")
