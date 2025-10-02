import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from autots.models.matrix_var import LATC, TMF


def _generate_df(length: int = 24, cols: int = 3):
    index = pd.date_range("2021-01-01", periods=length, freq="D")
    data = np.random.default_rng(42).standard_normal((length, cols))
    return pd.DataFrame(data, index=index, columns=[f"series_{i}" for i in range(cols)])


def test_tmf_forecast_shape_and_finiteness():
    df = _generate_df(length=30, cols=4)
    model = TMF(
        rank=0.5,
        maxiter=5,
        inner_maxiter=2,
        d=1,
        lambda0=0.1,
        rho=1.0,
    )
    model.fit(df)
    forecast = model.predict(5, just_point_forecast=True)
    assert forecast.shape == (5, df.shape[1])
    assert np.isfinite(forecast.to_numpy()).all()


def test_latc_forecast_shape_and_finiteness():
    df = _generate_df(length=28, cols=3)
    df.iloc[::4, 0] = np.nan  # include missing values to exercise imputation path
    model = LATC(
        time_horizon=2,
        seasonality=7,
        time_lags=[1, 2],
        lambda0=0.1,
        learning_rate=0.5,
        theta=1,
        window=14,
        maxiter=5,
    )
    model.fit(df)
    forecast = model.predict(4, just_point_forecast=True)
    assert forecast.shape == (4, df.shape[1])
    assert np.isfinite(forecast.to_numpy()).all()
