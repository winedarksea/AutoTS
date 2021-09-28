import pandas as pd
from autots.tools.impute import FillNA
from autots.tools.shaping import infer_frequency


def create_lagged_regressor(
    df,
    forecast_length: int,
    frequency: str = "infer",
    scale: bool = True,
    summarize: str = None,
    backfill: str = "bfill",
    n_jobs: str = "auto",
    fill_na: str = 'ffill',
):
    """Create a regressor of features lagged by forecast length.
    Useful to some models that don't otherwise use such information.

    It is recommended that the .head(forecast_length) of both regressor_train and the df for training are dropped.
    `df = df.iloc[forecast_length:]`

    Args:
        df (pd.DataFrame): training data
        forecast_length (int): length of forecasts, to shift data by
        frequency (str): the ever necessary frequency for datetime things. Default 'infer'
        scale (bool): if True, use the StandardScaler to standardize the features
        summarize (str): options to summarize the features, if large:
            'pca', 'median', 'mean', 'mean+std'
        backfill (str): method to deal with the NaNs created by shifting
            "bfill"- backfill with last values
            "ETS" -backfill with ETS backwards forecast
            "DatepartRegression" - backfill with DatepartRegression
        fill_na (str): method to prefill NAs in data, same methods as available elsewhere

    Returns:
        regressor_train, regressor_forecast
    """
    model_flag = False
    if frequency == "infer":
        frequency = infer_frequency(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must be a 'wide' dataframe with a pd.DatetimeIndex.")
    if isinstance(summarize, str):
        summarize = summarize.lower()
    if isinstance(backfill, str):
        backfill = backfill.lower()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    dates = df.index
    df_cols = df.columns

    if scale:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), index=dates, columns=df_cols)

    if summarize is None:
        pass
    elif summarize == 'mean':
        df = df.mean(axis=1).to_frame()
    elif summarize == 'median':
        df = df.median(axis=1).to_frame()
    elif summarize == 'pca':
        from sklearn.decomposition import PCA

        df = FillNA(df, method=fill_na)
        df = pd.DataFrame(PCA(n_components='mle').fit_transform(df), index=dates)
    elif summarize == 'mean+std':
        df = pd.concat([df.mean(axis=1).to_frame(), df.std(axis=1).to_frame()], axis=1)
        df.columns = [0, 1]

    df = FillNA(df, method=fill_na)
    regressor_forecast = df.tail(forecast_length)
    # also dates.shift(forecast_length)[-forecast_length:]
    regressor_forecast.index = pd.date_range(
        dates[-1], periods=(forecast_length + 1), freq=frequency
    )[1:]
    regressor_train = df.shift(forecast_length)
    if backfill == "ets":
        model_flag = True
        model_name = "ETS"
        model_param_dict = '{"damped_trend": false, "trend": "additive", "seasonal": null, "seasonal_periods": null}'
    elif backfill == 'datepartregression':
        model_flag = True
        model_name = 'DatepartRegression'
        model_param_dict = '{"regression_model": {"model": "RandomForest", "model_params": {}}, "datepart_method": "recurring", "regression_type": null}'
    else:
        regressor_train = regressor_train.fillna(method="bfill").fillna(method="ffill")

    if model_flag:
        from autots import model_forecast

        df_train = df.iloc[::-1]
        df_train.index = dates
        df_forecast = model_forecast(
            model_name=model_name,
            model_param_dict=model_param_dict,
            model_transform_dict={
                'fillna': 'fake_date',
                'transformations': {'0': 'ClipOutliers'},
                'transformation_params': {'0': {'method': 'clip', 'std_threshold': 3}},
            },
            df_train=df_train,
            forecast_length=forecast_length,
            frequency=frequency,
            random_seed=321,
            verbose=0,
            n_jobs=n_jobs,
        )
        add_on = df_forecast.forecast.iloc[::-1]
        add_on.index = regressor_train.head(forecast_length).index
        regressor_train = pd.concat(
            [add_on, regressor_train.tail(df.shape[0] - forecast_length)]
        )
    return regressor_train, regressor_forecast
