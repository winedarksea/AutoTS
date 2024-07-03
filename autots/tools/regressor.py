import numpy as np
import pandas as pd
from autots.tools.impute import FillNA
from autots.tools.shaping import infer_frequency
from autots.tools.seasonal import date_part
from autots.tools.holiday import holiday_flag
from autots.tools.cointegration import coint_johansen
from autots.evaluator.anomaly_detector import HolidayDetector
from autots.tools.transform import GeneralTransformer


def create_regressor(
    df,
    forecast_length,
    frequency: str = "infer",
    holiday_countries: list = ["US"],
    datepart_method: str = "simple_binarized",
    drop_most_recent: int = 0,
    scale: bool = True,
    summarize: str = "auto",
    backfill: str = "bfill",
    n_jobs: str = "auto",
    fill_na: str = 'ffill',
    aggfunc: str = "first",
    encode_holiday_type=False,
    holiday_detector_params={
        "threshold": 0.8,
        "splash_threshold": None,
        "use_dayofmonth_holidays": True,
        "use_wkdom_holidays": True,
        "use_wkdeom_holidays": False,
        "use_lunar_holidays": False,
        "use_lunar_weekday": False,
        "use_islamic_holidays": False,
        "use_hebrew_holidays": False,
        "output": 'univariate',
        "anomaly_detector_params": {
            "method": "mad",
            "transform_dict": {
                "fillna": None,
                "transformations": {"0": "DifferencedTransformer"},
                "transformation_params": {"0": {}},
            },
            "forecast_params": None,
            "method_params": {"distribution": "gamma", "alpha": 0.05},
        },
    },
    holiday_regr_style: str = "flag",
    preprocessing_params: dict = None,
):
    """Create a regressor from information available in the existing dataset.
    Components: are lagged data, datepart information, and holiday.

    This function has been confusing people. This is NOT necessary for machine learning models, in AutoTS they internally create more elaborate feature sets separately.
    This instead may help some other models (GLM, ARIMA) which accept regressors but won't build a regressor feature set internally.
    And this allows post-hoc customization as needed before input to AutoTS.

    It is recommended that the .head(forecast_length) of both regressor_train and the df for training are dropped.
    `df = df.iloc[forecast_length:]`
    If you don't want the lagged features, set summarize="median" which will only give one column of such, which can then be easily dropped

    Args:
        df (pd.DataFrame): WIDE style dataframe (use long_to_wide if the data isn't already)
            categorical series will be discard for this, if present
        forecast_length (int): time ahead that will be forecast
        frequency (str): those annoying offset codes you have to always use for time series
        holiday_countries (list): list of countries to pull holidays for. Reqs holidays pkg
            also can be a dict of {'country': "subdiv"} to include subdivision (state)
        datepart_method (str): see date_part from seasonal
        scale (bool): if True, use the StandardScaler to standardize the features
        summarize (str): options to summarize the features, if large:
            'pca', 'median', 'mean', 'mean+std', 'feature_agglomeration', 'gaussian_random_projection'
        backfill (str): method to deal with the NaNs created by shifting
            "bfill"- backfill with last values
            "ETS" -backfill with ETS backwards forecast
            "DatepartRegression" - backfill with DatepartRegression
        fill_na (str): method to prefill NAs in data, same methods as available elsewhere
        aggfunc (str): str or func, used if frequency is resampled
        encode_holiday_type (bool): if True, returns column per holiday, ONLY for holidays package country holidays (not Detector)
        holiday_detector_params (dict): passed to HolidayDetector, or None
        holiday_regr_style (str): passed to detector's dates_to_holidays 'flag', 'series_flag', 'impact'
        preprocessing_params (dict): GeneralTransformer params to be applied before regressor creation

    Returns:
        regressor_train, regressor_forecast
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "create_regressor input df must be `wide` style with pd.DatetimeIndex index"
        )
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if drop_most_recent > 0:
        df = df.drop(df.tail(drop_most_recent).index)
    if frequency == "infer":
        frequency = infer_frequency(df)
    else:
        # fill missing dates in index with NaN, resample to freq as necessary
        try:
            if aggfunc == "first":
                df = df.resample(frequency).first()
            else:
                df = df.resample(frequency).apply(aggfunc)
        except Exception:
            df = df.asfreq(frequency, fill_value=None)
    # handle categorical
    # df = df.apply(pd.to_numeric, errors='ignore') # another pointless pandas deprecation
    for col in df.columns:
        try:
            df.loc[:, col] = pd.to_numeric(df[col])
        except ValueError:
            pass
    df = df.select_dtypes(include=np.number)
    # macro_micro
    if preprocessing_params is not None:
        trans = GeneralTransformer(**preprocessing_params)
        df = trans.fit_transform(df)

    # lagged data
    regr_train, regr_fcst = create_lagged_regressor(
        df,
        forecast_length=forecast_length,
        frequency=frequency,
        summarize=summarize,
        scale=scale,  # already done above
        backfill=backfill,
        fill_na=fill_na,
    )
    # datepart
    if datepart_method is not None:
        regr_train = pd.concat(
            [regr_train, date_part(regr_train.index, method=datepart_method)],
            axis=1,
        )
        regr_fcst = pd.concat(
            [regr_fcst, date_part(regr_fcst.index, method=datepart_method)],
            axis=1,
        )
    # holiday (list)
    if holiday_countries is not None:
        if isinstance(holiday_countries, str):
            holiday_countries = holiday_countries.split(",")
        if isinstance(holiday_countries, list):
            holiday_countries = {x: None for x in holiday_countries}

        for holiday_country, holidays_subdiv in holiday_countries.items():
            # create holiday flag for historic regressor
            regr_train = pd.concat(
                [
                    regr_train,
                    holiday_flag(
                        regr_train.index,
                        country=holiday_country,
                        holidays_subdiv=holidays_subdiv,
                        encode_holiday_type=encode_holiday_type,
                    ),
                ],
                axis=1,
            )
            # now do again for future regressor
            regr_fcst = pd.concat(
                [
                    regr_fcst,
                    holiday_flag(
                        regr_fcst.index,
                        country=holiday_country,
                        holidays_subdiv=holidays_subdiv,
                        encode_holiday_type=encode_holiday_type,
                    ),
                ],
                axis=1,
            )
            # now try it for future days
            try:
                holiday_future = holiday_flag(
                    regr_train.index.shift(1, freq=frequency),
                    country=holiday_country,
                    holidays_subdiv=holidays_subdiv,
                )
                holiday_future.index = regr_train.index
                holiday_future_2 = holiday_flag(
                    regr_fcst.index.shift(1, freq=frequency),
                    country=holiday_country,
                    holidays_subdiv=holidays_subdiv,
                )
                holiday_future_2.index = regr_fcst.index
                regr_train[f"holiday_flag_{holiday_country}_future"] = holiday_future
                regr_fcst[f"holiday_flag_{holiday_country}_future"] = holiday_future_2
            except Exception:
                print(
                    f"holiday_future columns failed to add for {holiday_country}, likely due to complex datetime index"
                )
    if holiday_detector_params is not None:
        try:
            mod = HolidayDetector(**holiday_detector_params)
            mod.detect(df)
            train_holidays = mod.dates_to_holidays(
                regr_train.index, style=holiday_regr_style
            )
            fcst_holidays = mod.dates_to_holidays(
                regr_fcst.index, style=holiday_regr_style
            )
            all_cols = train_holidays.columns.union(fcst_holidays.columns)
            regr_train = pd.concat(
                [regr_train, train_holidays.reindex(columns=all_cols).fillna(0)],
                axis=1,
            )
            regr_fcst = pd.concat(
                [regr_fcst, fcst_holidays.reindex(columns=all_cols).fillna(0)],
                axis=1,
            )
        except Exception as e:
            print("HolidayDetector failed with error: " + repr(e)[:180])

    # columns all as strings
    regr_train.columns = [str(xc) for xc in regr_train.columns]
    regr_fcst.columns = [str(xc) for xc in regr_fcst.columns]
    # drop duplicate columns (which come from holidays)
    regr_train = regr_train.loc[:, ~regr_train.columns.duplicated()]
    regr_fcst = regr_fcst.loc[:, ~regr_fcst.columns.duplicated()]
    regr_fcst = regr_fcst.reindex(columns=regr_train.columns, fill_value=0)
    return regr_train, regr_fcst


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
            'pca', 'median', 'mean', 'mean+std', 'feature_agglomeration', 'gaussian_random_projection', "auto"
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
    df_inner = df.copy()

    if scale:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        df_inner = pd.DataFrame(
            scaler.fit_transform(df_inner), index=dates, columns=df_cols
        )

    ag_flag = False
    # these shouldn't care about NaN
    if summarize is None:
        pass
    if summarize == "auto":
        ag_flag = True if df_inner.shape[1] > 10 else False
    elif summarize == 'mean':
        df_inner = df_inner.mean(axis=1).to_frame()
    elif summarize == 'median':
        df_inner = df_inner.median(axis=1).to_frame()
    elif summarize == 'mean+std':
        df_inner = pd.concat(
            [df_inner.mean(axis=1).to_frame(), df_inner.std(axis=1).to_frame()], axis=1
        )
        df_inner.columns = [0, 1]

    df_inner = FillNA(df_inner, method=fill_na)
    # some debate over whether PCA or RandomProjection will result in minor data leakage, if used
    if summarize == 'pca':
        from sklearn.decomposition import PCA

        n_components = "mle" if df_inner.shape[0] > df_inner.shape[1] else None
        df_inner = FillNA(df_inner, method=fill_na)
        df_inner = pd.DataFrame(
            PCA(n_components=n_components).fit_transform(df_inner), index=dates
        )
        ag_flag = True if df_inner.shape[1] > 10 else False
    elif summarize == 'cointegration':
        ev, components_ = coint_johansen(df_inner.values, 0, 1, return_eigenvalues=True)
        df_inner = pd.DataFrame(
            np.matmul(components_, (df_inner.values).T).T,
            index=df_inner.index,
        ).iloc[:, np.flipud(np.argsort(ev))[0:10]]
    elif summarize == "feature_agglomeration" or ag_flag:
        from sklearn.cluster import FeatureAgglomeration

        n_clusters = 10 if ag_flag else 25
        if df_inner.shape[1] > 25:
            df_inner = pd.DataFrame(
                FeatureAgglomeration(n_clusters=n_clusters).fit_transform(df_inner),
                index=dates,
            )
    elif summarize == "gaussian_random_projection":
        from sklearn.random_projection import GaussianRandomProjection

        df_inner = pd.DataFrame(
            GaussianRandomProjection(n_components='auto', eps=0.2).fit_transform(
                df_inner
            ),
            index=dates,
        )

    regressor_forecast = df_inner.tail(forecast_length)
    try:
        regressor_forecast.index = pd.date_range(
            dates[-1], periods=(forecast_length + 1), freq=frequency
        )[1:]
    except Exception:
        raise ValueError(
            "create_regressor doesn't work on data where forecast_length > historical data length"
        )
    regressor_train = df_inner.shift(forecast_length)
    if backfill == "ets":
        model_flag = True
        model_name = "ETS"
        model_param_dict = '{"damped_trend": false, "trend": "additive", "seasonal": null, "seasonal_periods": null}'
    elif backfill == 'datepartregression':
        model_flag = True
        model_name = 'DatepartRegression'
        model_param_dict = '{"regression_model": {"model": "RandomForest", "model_params": {}}, "datepart_method": "recurring", "regression_type": null}'
    else:
        regressor_train = regressor_train.bfill().ffill()

    if model_flag:
        from autots import model_forecast

        df_train = df_inner.iloc[::-1]
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
            [add_on, regressor_train.tail(df_inner.shape[0] - forecast_length)]
        )
    return regressor_train, regressor_forecast
