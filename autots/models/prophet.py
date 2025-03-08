"""
Facebook's Prophet

Since Prophet install can be finicky on Windows, it will be an optional dependency.
"""

import random
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.evaluator.anomaly_detector import HolidayDetector
import logging

# optional packages at the high level
try:
    from joblib import Parallel, delayed

    joblib_present = True
except Exception:
    joblib_present = False


class FBProphet(ModelObject):
    """Facebook's Prophet

    'thou shall count to 3, no more, no less, 3 shall be the number thou shall count, and the number of the counting
    shall be 3. 4 thou shall not count, neither count thou 2, excepting that thou then preceed to 3.' -Python

    For params see: https://facebook.github.io/prophet/docs/diagnostics.html

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holidays
        regression_type (str): type of regression (None, 'User')

    """

    def __init__(
        self,
        name: str = "FBProphet",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday: bool = False,
        regression_type: str = None,
        holiday_country: str = None,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        growth: str = "linear",
        n_changepoints: int = 25,
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = "additive",
        changepoint_range: float = 0.8,
        changepoint_spacing: int = None,
        changepoint_distance_end: int = None,  # optional for other style of changepoints
        seasonality_prior_scale: float = 10.0,
        weekly_seasonality_prior_scale: float = None,
        weekly_seasonality_order: int = 4,
        yearly_seasonality_prior_scale: float = None,
        yearly_seasonality_order: int = None,
        holidays_prior_scale: float = 10.0,
        trend_phi: float = 1,
        phi: float = None,  # needed for Meta import and usage of this
        holidays=None,
        random_seed: int = 2024,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            regression_type=regression_type,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.holiday = holiday
        self.regressor_name = []
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.growth = growth
        self.n_changepoints = n_changepoints
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.changepoint_range = changepoint_range
        self.changepoint_spacing = changepoint_spacing
        self.changepoint_distance_end = changepoint_distance_end
        self.seasonality_prior_scale = seasonality_prior_scale
        self.weekly_seasonality_prior_scale = weekly_seasonality_prior_scale
        self.weekly_seasonality_order = weekly_seasonality_order
        self.yearly_seasonality_prior_scale = yearly_seasonality_prior_scale
        self.yearly_seasonality_order = yearly_seasonality_order
        self.holidays_prior_scale = holidays_prior_scale
        self.trend_phi = trend_phi
        if phi is not None:
            self.trend_phi = phi
        self.holidays = holidays

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.regressor_train = None
        self.dimensionality_reducer = None
        if self.regression_type in ['User', 'user']:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but no future_regressor passed"
                )
            else:
                if future_regressor.ndim > 1:
                    if future_regressor.shape[1] > 1:
                        regr = pd.concat(
                            [df.mean(axis=1).to_frame(), df.std(axis=1).to_frame()],
                            axis=1,
                        )
                        regr.columns = [0, 1]
                    else:
                        regr = future_regressor
                    regr.columns = [
                        str(colr) if colr not in df.columns else str(colr) + "xxxxx"
                        for colr in regr.columns
                    ]
                    self.regressor_train = regr
                    self.regressor_name = regr.columns.tolist()

                else:
                    self.regressor_train = future_regressor.copy()
                    # this is a hack to utilize regressors with a name unlikely to exist
                    random_two = "n9032380gflljWfu8233koWQop3"
                    random_one = "prophet_staging_regressor"
                    self.regressor_name = [
                        random_one if random_one not in df.columns else random_two
                    ]
        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=None,
        just_point_forecast: bool = False,
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        try:  # no idea when they switched
            from prophet import Prophet
        except Exception:
            from fbprophet import Prophet

        # defining in function helps with Joblib it seems
        def seek_the_oracle(
            current_series, args, series, forecast_length, future_regressor, holidays
        ):
            """Prophet for for loop or parallel."""
            current_series = current_series.rename(columns={series: 'y'})
            current_series['ds'] = current_series.index
            # logging needs to be inside the multiprocessing, apparently
            logging.getLogger('fbprophet').disabled = True
            logging.getLogger('fbprophet').setLevel(logging.CRITICAL)
            logging.getLogger('fbprophet.models').setLevel(logging.CRITICAL)
            logging.getLogger('prophet').setLevel(logging.WARNING)
            logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
            if self.weekly_seasonality_prior_scale not in [None, "None"]:
                self.weekly_seasonality = False
            if self.yearly_seasonality_prior_scale not in [None, "None"]:
                self.yearly_seasonality = False
            pargs = {
                "interval_width": args['prediction_interval'],
                "yearly_seasonality": self.yearly_seasonality,
                "weekly_seasonality": self.weekly_seasonality,
                "daily_seasonality": self.daily_seasonality,
                "growth": self.growth,
                "n_changepoints": self.n_changepoints,
                "changepoint_prior_scale": self.changepoint_prior_scale,
                "seasonality_mode": self.seasonality_mode,
                "changepoint_range": self.changepoint_range,
                "seasonality_prior_scale": self.seasonality_prior_scale,
                "holidays_prior_scale": self.holidays_prior_scale,
            }
            if holidays is not None:
                pargs["holidays"] = holidays
            # handle AMFM style changepoint improvements
            # n_changepoints -> changepoint_spacing
            # changepoint_range -> changepoint_distance_end
            if self.changepoint_range > 1 or self.changepoint_distance_end is not None:
                non_null_indices = np.where(current_series["y"].notnull())[0]
                pargs['changepoints'] = get_changepoints(
                    current_series.index[non_null_indices[0]],
                    current_series.index[non_null_indices[-1]],
                    changepoint_spacing=(
                        int(len(current_series.index) / self.n_changepoints)
                        if self.changepoint_spacing is None
                        else int(self.changepoint_spacing)
                    ),
                    changepoint_distance_end=(
                        int(self.changepoint_range)
                        if self.changepoint_distance_end is None
                        else int(self.changepoint_distance_end)
                    ),
                )
                pargs.pop("changepoint_range", None)
                pargs.pop("n_changepoints", None)
            m = Prophet(**pargs)
            # as currently written this customization only works on daily data
            if self.weekly_seasonality_prior_scale not in [None, "None"]:
                m.add_seasonality(
                    name='weekly',
                    period=168 if "H" in self.frequency else 7,
                    fourier_order=int(self.weekly_seasonality_order),
                    prior_scale=self.weekly_seasonality_prior_scale,
                )
            if self.yearly_seasonality_prior_scale not in [None, "None"]:
                if self.yearly_seasonality_order in [None, "None"]:
                    self.yearly_seasonality_order = 12
                if "W" in str(self.frequency).upper():
                    yperiod = 52.18
                elif "M" in str(self.frequency).upper():
                    yperiod = 12
                else:
                    yperiod = 365.25
                m.add_seasonality(
                    name='yearly',
                    period=yperiod,
                    fourier_order=int(self.yearly_seasonality_order),
                    prior_scale=self.yearly_seasonality_prior_scale,
                )
            if isinstance(args['holiday'], pd.DataFrame):
                add_holidays = args['holiday'][args['holiday']['series'] == series]
                if isinstance(m.holidays, pd.DataFrame):
                    if not m.holidays.empty:
                        add_holidays = pd.concat([m.holidays, add_holidays])
                m.holidays = add_holidays
            if args["holiday_country"] not in [None, "None"]:
                if isinstance(args["holiday_country"], list):
                    for hol in args["holiday_country"]:
                        m.add_country_holidays(country_name=hol)
                else:
                    m.add_country_holidays(country_name=args['holiday_country'])
            else:
                pass
                # raise ValueError("`holiday` arg for Prophet not recognized")
            if args['regression_type'] in ['User', 'user']:
                current_series = pd.concat(
                    [current_series, args['regressor_train']], axis=1
                )
                for nme in args['regressor_name']:
                    m.add_regressor(nme)
            m = m.fit(current_series)
            future = m.make_future_dataframe(
                periods=forecast_length, include_history=False
            )
            if args['regression_type'] in ['User', 'user']:
                if future_regressor.ndim > 1:
                    # a = args['dimensionality_reducer'].transform(future_regressor)
                    if future_regressor.shape[1] > 1:
                        ft_regr = (
                            future_regressor.mean(axis=1)
                            .to_frame()
                            .merge(
                                future_regressor.std(axis=1).to_frame(),
                                left_index=True,
                                right_index=True,
                            )
                        )
                    else:
                        ft_regr = future_regressor.copy()
                    ft_regr.columns = args['regressor_train'].columns
                    regr = pd.concat([args['regressor_train'], ft_regr])
                    regr.index.name = 'ds'
                    regr.reset_index(drop=False, inplace=True)
                    future = future.merge(regr, on="ds", how='left')
                else:
                    a = np.append(args['regressor_train'], future_regressor.values)
                    future[args['regressor_name']] = a
            fcst = m.predict(future)
            # fcst = fcst.tail(forecast_length)  # remove the backcast
            if (
                self.trend_phi is not None
                and self.trend_phi != 1
                and forecast_length > 2
            ):
                req_len = fcst.shape[0] - 1
                phi_series = pd.Series(
                    [self.trend_phi] * req_len,
                    index=fcst.index[1:],
                ).pow(range(req_len))

                # adjust all by same margin
                fcst['trend'] = pd.concat(
                    [
                        fcst['trend'].iloc[0:1],
                        fcst['trend'].diff().iloc[1:].mul(phi_series, axis=0),
                    ]
                ).cumsum()
                # for now, not doing the dampening on upper and lower bounds as those might still be good to have the full probability
                forecast = (
                    fcst['trend'] * (1 + fcst["multiplicative_terms"])
                    + fcst['additive_terms']
                )
                upper_forecast = (
                    fcst['trend_upper'] * (1 + fcst["multiplicative_terms_upper"])
                    + fcst['additive_terms_upper']
                )
                lower_forecast = (
                    fcst['trend_lower'] * (1 + fcst["multiplicative_terms_lower"])
                    + fcst['additive_terms_lower']
                )
            else:
                forecast = fcst['yhat']
                lower_forecast = fcst['yhat_lower']
                upper_forecast = fcst['yhat_upper']
            forecast.name = series
            lower_forecast.name = series
            upper_forecast.name = series
            return (forecast, lower_forecast, upper_forecast, fcst)

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        args = {
            'holiday': self.holiday,
            'holiday_country': self.holiday_country,
            'regression_type': self.regression_type,
            'regressor_name': self.regressor_name,
            'regressor_train': self.regressor_train,
            'dimensionality_reducer': self.dimensionality_reducer,
            'prediction_interval': self.prediction_interval,
        }
        if isinstance(self.holiday, dict):
            mod = HolidayDetector(**self.holiday)
            mod.detect(self.df_train)
            args['holiday'] = mod.dates_to_holidays(
                self.df_train.index.union(test_index), style="prophet"
            )

        parallel = True
        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        else:
            if not joblib_present:
                parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            verbs = 0 if self.verbose < 1 else self.verbose - 1
            df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                delayed(seek_the_oracle)(
                    current_series=self.df_train[col].to_frame(),
                    args=args,
                    series=col,
                    forecast_length=forecast_length,
                    future_regressor=future_regressor,
                    holidays=self.holidays,
                )
                for col in cols
            )
        else:
            df_list = []
            for col in cols:
                df_list.append(
                    seek_the_oracle(
                        self.df_train[col].to_frame(),
                        args,
                        col,
                        forecast_length=forecast_length,
                        future_regressor=future_regressor,
                        holidays=self.holidays,
                    )
                )
        complete = list(map(list, zip(*df_list)))
        forecast = pd.concat(complete[0], axis=1)
        forecast.index = test_index
        forecast = forecast[self.column_names]
        lower_forecast = pd.concat(complete[1], axis=1)
        lower_forecast.index = test_index
        lower_forecast = lower_forecast[self.column_names]
        upper_forecast = pd.concat(complete[2], axis=1)
        upper_forecast.index = test_index
        upper_forecast = upper_forecast[self.column_names]
        self.components = complete[3]

        if just_point_forecast:
            return forecast
        else:
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=forecast.index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def process_components(self):
        # process components is a wide style with a multiindex, level 0 = series name
        return pd.concat(self.components, axis=1, keys=self.column_names)

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        holiday_choice = random.choices([True, False, 'auto'], [0.3, 0.4, 0.3])[0]
        if holiday_choice == 'auto':
            holiday_choice = HolidayDetector.get_new_params(method="fast")
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_list = [None, 'User']
            regression_probability = [0.8, 0.2]
            regression_choice = random.choices(regression_list, regression_probability)[
                0
            ]
        yearly_seasonality_order = None
        yearly_seasonality_prior_scale = random.choices(
            [None, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 15, 20, 25, 40],  # default 10
            [0.4, 0.2, 0.05, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05],
        )[0]
        if yearly_seasonality_prior_scale is not None:
            yearly_seasonality_order = random.choices(
                [2, 6, 12, 30], [0.1, 0.2, 0.5, 0.1]
            )[0]
        params = {
            'holiday': holiday_choice,
            'regression_type': regression_choice,
            'changepoint_prior_scale': random.choices(
                [0.0001, 0.001, 0.01, 0.1, 0.05, 0.5, 1, 10, 30, 50],  # 0.05 default
                [0.01, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.05],
            )[0],
            'seasonality_prior_scale': random.choices(
                [0.01, 0.1, 1.0, 10.0, 15, 20, 25, 40],  # default 10
                [0.05, 0.05, 0.05, 0.8, 0.05, 0.05, 0.05, 0.05],
            )[0],
            'holidays_prior_scale': random.choices(
                [0.01, 0.1, 1.0, 10.0, 15, 20, 25, 40],  # default 10
                [0.05, 0.05, 0.05, 0.8, 0.05, 0.05, 0.05, 0.05],
            )[0],
            'seasonality_mode': random.choice(['additive', 'multiplicative']),
            'growth': random.choices(["linear", "flat"], [0.9, 0.1])[0],
            "trend_phi": random.choices(
                [None, 0.98, 0.999, 0.95, 0.8, 0.99], [0.8, 0.1, 0.2, 0.1, 0.1, 0.05]
            )[0],
        }
        way = random.choice(["new", "old"])
        if way == "old":
            params["n_changepoints"] = random.choices(
                [5, 10, 20, 25, 30, 40, 50], [0.05, 0.1, 0.1, 0.9, 0.1, 0.05, 0.05]
            )[0]
            params["changepoint_range"] = random.choices(
                [0.8, 0.85, 0.9, 0.95, 0.98, 30, 60, 180, 360],
                [0.4, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            )[0]
        else:
            params["changepoint_spacing"] = random.choices(
                [10, 20, 180, 30, 40, 50, 60], [0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.9]
            )[0]
            params["changepoint_distance_end"] = random.choices(
                [10, 20, 180, 30, 40, 50, 60], [0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.9]
            )[0]
        way = random.choice(["new", "old"])
        if way == "new":
            params["yearly_seasonality_prior_scale"] = yearly_seasonality_prior_scale
            params["yearly_seasonality_order"] = yearly_seasonality_order
            params["weekly_seasonality_prior_scale"] = random.choices(
                [
                    None,
                    0.0001,
                    0.001,
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    15,
                    20,
                    25,
                    40,
                ],  # default 10
                [0.4, 0.2, 0.05, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05],
            )[0]
        else:
            pass
        return params

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'holiday': self.holiday,
            'regression_type': self.regression_type,
            "growth": self.growth,
            "seasonality_mode": self.seasonality_mode,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "changepoint_spacing": self.changepoint_spacing,
            "changepoint_distance_end": self.changepoint_distance_end,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "yearly_seasonality_prior_scale": self.yearly_seasonality_prior_scale,
            "yearly_seasonality_order": self.yearly_seasonality_order,
            "weekly_seasonality_prior_scale": self.weekly_seasonality_prior_scale,
            "holidays_prior_scale": self.holidays_prior_scale,
            "trend_phi": self.trend_phi,
        }


def get_changepoints(
    training_start_ds,
    training_end_ds,
    changepoint_spacing,
    changepoint_distance_end,
    custom_changepoints="",
):
    """Create the distinct uniform pattern used in Ecosystem Analytics
    This is likely to be replaced in the future. It was likely designed the way
    is was to have exact control over where the last potential changepoint
    could be, which is around the 93% mark of the data, and then setting a
    relatively course, uniform grid going way back in time.
    (sourced from the work of Benn O)
    Args:
        training_start_ds: Pandas datetime or string date of earliest
            historical training date
        training_end_ds: Pandas datetime or string date of last training
            date (used in model fitting)
        changepoint_spacing (int): Number of days between potential
            changepoints
        changepoint_distance_end (int): Number of days from the present into
            the past for which to start the uniform changepoint grid. This will
            also be the interval between potential changepoints
        custom_changepoints (string): comma separated dates in form YYYY-MM-DD. No
            additional quotations are necessary (e.g., "2020-10-12,2020-11-15")
    Returns:
        a pandas Series (dtype: datetime64[ns]) of potential changepoints for Prophet
    """
    last_changepoint = pd.to_datetime(training_end_ds) - datetime.timedelta(
        changepoint_distance_end
    )
    cp_freq = f"-{changepoint_spacing}D"
    df = pd.DataFrame(
        index=pd.date_range(
            start=last_changepoint, end=pd.to_datetime(training_start_ds), freq=cp_freq
        )
    )
    changepoints = df.reset_index().sort_values(["index"])["index"]
    cp_csv = custom_changepoints.replace(" ", "")
    if len(cp_csv) > 0:
        timestamps = [pd.Timestamp(cp_str) for cp_str in cp_csv.split(",")]
        changepoints = pd.concat([changepoints, pd.Series(timestamps)], axis=0)
    changepoints = changepoints.drop_duplicates().sort_values()
    changepoints = changepoints.loc[
        (changepoints > training_start_ds) & (changepoints < training_end_ds)
    ]
    return changepoints


class NeuralProphet(ModelObject):
    """Facebook's Prophet got caught in a net.

    n_jobs is implemented here but it should be set to 1. PyTorch already maxes out cores in all observed cases.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holidays
        regression_type (str): type of regression (None, 'User')

    """

    def __init__(
        self,
        name: str = "NeuralProphet",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday: bool = False,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = None,
        growth: str = "off",
        n_changepoints: int = 10,
        changepoints_range: float = 0.9,
        trend_reg: float = 0,
        trend_reg_threshold: bool = False,
        ar_sparsity: float = None,
        yearly_seasonality: str = "auto",
        weekly_seasonality: str = "auto",
        daily_seasonality: str = "auto",
        seasonality_mode: str = "additive",
        seasonality_reg: float = 0,
        n_lags: int = 0,
        num_hidden_layers: int = 0,
        d_hidden: int = None,
        learning_rate: float = None,
        loss_func: str = "Huber",
        train_speed: int = None,
        normalize: str = "auto",
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            regression_type=regression_type,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.holiday = holiday
        self.regressor_name = []
        self.growth = growth
        self.n_changepoints = n_changepoints
        self.changepoints_range = changepoints_range
        self.trend_reg = trend_reg
        self.trend_reg_threshold = trend_reg_threshold
        self.ar_sparsity = ar_sparsity
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.seasonality_reg = seasonality_reg
        self.n_lags = n_lags
        self.num_hidden_layers = num_hidden_layers
        self.d_hidden = d_hidden
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.train_speed = train_speed
        self.normalize = normalize

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.regressor_train = None
        self.dimensionality_reducer = None
        if self.regression_type in ['User', 'user']:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but no future_regressor passed"
                )
            else:
                if future_regressor.ndim > 1:
                    if future_regressor.shape[1] > 1:
                        regr = pd.concat(
                            [df.mean(axis=1).to_frame(), df.std(axis=1).to_frame()],
                            axis=1,
                        )
                        regr.columns = [0, 1]
                    else:
                        regr = future_regressor
                    regr.columns = [
                        str(colr) if colr not in df.columns else str(colr) + "xxxxx"
                        for colr in regr.columns
                    ]
                    self.regressor_train = regr
                    self.regressor_name = regr.columns.tolist()

                else:
                    self.regressor_train = future_regressor.copy()
                    # this is a hack to utilize regressors with a name unlikely to exist
                    random_two = "n9032380gflljWfu8233koWQop3"
                    random_one = "prophet_staging_regressor"
                    self.regressor_name = [
                        random_one if random_one not in df.columns else random_two
                    ]
        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=None,
        just_point_forecast: bool = False,
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        from neuralprophet import NeuralProphet, set_log_level

        # defining in function helps with Joblib it seems
        def seek_the_oracle(
            current_series, args, series, forecast_length, future_regressor
        ):
            """Prophet for for loop or parallel."""
            current_series = current_series.rename(columns={series: 'y'})
            current_series['ds'] = current_series.index
            try:
                quant_range = (1 - args['prediction_interval']) / 2
                quantiles = [quant_range, 0.5, (1 - quant_range)]
                m = NeuralProphet(
                    quantiles=quantiles,
                    growth=self.growth,
                    n_changepoints=self.n_changepoints,
                    changepoints_range=self.changepoints_range,
                    trend_reg=self.trend_reg,
                    trend_reg_threshold=self.trend_reg_threshold,
                    # ar_sparsity=self.ar_sparsity,
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    seasonality_mode=self.seasonality_mode,
                    seasonality_reg=self.seasonality_reg,
                    n_lags=self.n_lags,
                    n_forecasts=forecast_length,
                    num_hidden_layers=self.num_hidden_layers,
                    d_hidden=self.d_hidden,
                    learning_rate=self.learning_rate,
                    loss_func=self.loss_func,
                    # train_speed=self.train_speed,
                    normalize=self.normalize,
                    collect_metrics=False,
                )
            except Exception:
                m = NeuralProphet(
                    growth=self.growth,
                    n_changepoints=self.n_changepoints,
                    changepoints_range=self.changepoints_range,
                    trend_reg=self.trend_reg,
                    trend_reg_threshold=self.trend_reg_threshold,
                    # ar_sparsity=self.ar_sparsity,
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    seasonality_mode=self.seasonality_mode,
                    seasonality_reg=self.seasonality_reg,
                    n_lags=self.n_lags,
                    n_forecasts=forecast_length,
                    num_hidden_layers=self.num_hidden_layers,
                    d_hidden=self.d_hidden,
                    learning_rate=self.learning_rate,
                    loss_func=self.loss_func,
                    # train_speed=self.train_speed,
                    normalize=self.normalize,
                    collect_metrics=False,
                )
            if args['holiday']:
                m.add_country_holidays(country_name=args['holiday_country'])
            if args['regression_type'] == 'User':
                current_series = pd.concat(
                    [current_series, args['regressor_train']], axis=1
                )
                for nme in args['regressor_name']:
                    m.add_future_regressor(nme)
            m.fit(current_series, freq=args['freq'], minimal=True)
            if args['regression_type'] == 'User':
                if future_regressor.ndim > 1:
                    if future_regressor.shape[1] > 1:
                        ft_regr = (
                            future_regressor.mean(axis=1)
                            .to_frame()
                            .merge(
                                future_regressor.std(axis=1).to_frame(),
                                left_index=True,
                                right_index=True,
                            )
                        )
                    else:
                        ft_regr = future_regressor.copy()
                    ft_regr.columns = args['regressor_train'].columns
                    regr = pd.concat([args['regressor_train'], ft_regr])
                    regr.columns = args['regressor_train'].columns
                    # regr.index.name = 'ds'
                    # regr.reset_index(drop=False, inplace=True)
                    # future = future.merge(regr, on="ds", how='left')
                else:
                    # a = np.append(args['regressor_train'], future_regressor.values)
                    regr = future_regressor
                future = m.make_future_dataframe(
                    current_series, periods=forecast_length, regressors_df=regr
                )
            else:
                future = m.make_future_dataframe(
                    current_series, periods=forecast_length
                )
            fcst = m.predict(future, decompose=False)
            fcst = fcst.tail(forecast_length)  # remove the backcast
            # predicting that someday they will change back to fbprophet format
            if "yhat2" in fcst.columns:
                fcst['yhat1'] = fcst.fillna(0).sum(axis=1, numeric_only=True)
            try:
                forecast = fcst['yhat1']
            except Exception:
                forecast = fcst['yhat']
            forecast.name = series
            # not yet supported, so fill with the NaN column for now if missing
            try:
                lower_forecast = fcst['yhat_lower']
                upper_forecast = fcst['yhat_upper']
            except Exception:
                lower_forecast = fcst['y']
                upper_forecast = fcst['y']
            lower_forecast.name = series
            upper_forecast.name = series
            return (forecast, lower_forecast, upper_forecast)

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        if self.verbose < 0:
            set_log_level("CRITICAL")
        elif self.verbose < 1:
            set_log_level("ERROR")

        args = {
            'freq': self.frequency,
            'holiday': self.holiday,
            'holiday_country': self.holiday_country,
            'regression_type': self.regression_type,
            'regressor_name': self.regressor_name,
            'regressor_train': self.regressor_train,
            'prediction_interval': self.prediction_interval,
        }
        parallel = True
        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        else:
            if not joblib_present:
                parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            verbs = 0 if self.verbose < 1 else self.verbose - 1
            df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                delayed(seek_the_oracle)(
                    current_series=self.df_train[col].to_frame(),
                    args=args,
                    series=col,
                    forecast_length=forecast_length,
                    future_regressor=future_regressor,
                )
                for col in cols
            )
        else:
            df_list = []
            for col in cols:
                df_list.append(
                    seek_the_oracle(
                        self.df_train[col].to_frame(),
                        args,
                        col,
                        forecast_length=forecast_length,
                        future_regressor=future_regressor,
                    )
                )
        complete = list(map(list, zip(*df_list)))
        forecast = pd.concat(complete[0], axis=1)
        forecast.index = test_index
        forecast = forecast[self.column_names]
        lower_forecast = pd.concat(complete[1], axis=1)
        lower_forecast.index = test_index
        lower_forecast = lower_forecast[self.column_names]
        upper_forecast = pd.concat(complete[2], axis=1)
        upper_forecast.index = test_index
        upper_forecast = upper_forecast[self.column_names]
        if lower_forecast.isnull().to_numpy().any():
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )

        if just_point_forecast:
            return forecast
        else:
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=forecast.index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        holiday_choice = random.choices([True, False], [0.5, 0.5])[0]
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_list = [None, 'User']
            regression_probability = [0.8, 0.2]
            regression_choice = random.choices(regression_list, regression_probability)[
                0
            ]
        num_hidden = random.choices([0, 1, 2, 3, 4], [0.5, 0.1, 0.1, 0.1, 0.2])[0]
        if num_hidden > 0:
            d_hidden = random.choices([16, 32, 64], [0.8, 0.1, 0.1])[0]
        else:
            d_hidden = 16
        growth = random.choices(['off', 'linear', 'discontinuous'], [0.8, 0.2, 0.1])[0]
        if growth == "off":
            trend_reg = 0
            trend_reg_threshold = False
        else:
            trend_reg = random.choices([0.1, 0, 1, 10, 100], [0.1, 0.5, 0.1, 0.1, 0.1])[
                0
            ]
            trend_reg_threshold = random.choices([True, False], [0.1, 0.9])[0]

        parameter_dict = {
            'holiday': holiday_choice,
            'regression_type': regression_choice,
            "growth": growth,
            "n_changepoints": random.choice([5, 10, 20, 30]),
            "changepoints_range": random.choice([0.8, 0.9, 0.95]),
            "trend_reg": trend_reg,
            'trend_reg_threshold': trend_reg_threshold,
            # "ar_sparsity": random.choices(
            #    [None, 0.01, 0.03, 0.1], [0.9, 0.1, 0.1, 0.1]
            # )[0],
            "yearly_seasonality": random.choices(["auto", False], [0.1, 0.5])[0],
            "weekly_seasonality": random.choices(["auto", False], [0.1, 0.5])[0],
            "daily_seasonality": random.choices(["auto", False], [0.1, 0.5])[0],
            "seasonality_mode": random.choice(['additive', 'multiplicative']),
            "seasonality_reg": random.choices([0, 0.1, 1, 10], [0.7, 0.1, 0.1, 0.1])[0],
            "n_lags": random.choices([0, 1, 2, 3, 7], [0.8, 0.2, 0.1, 0.05, 0.1])[0],
            "num_hidden_layers": num_hidden,
            'd_hidden': d_hidden,
            "learning_rate": random.choices(
                [None, 1.0, 0.1, 0.01, 0.001], [0.7, 0.2, 0.1, 0.1, 0.1]
            )[0],
            "loss_func": random.choice(['Huber', "MAE", "MSE"]),
            # "train_speed": random.choices(
            #     [None, -1, -2, -3, 1, 2, 3], [0.9, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05]
            # )[0],
            "normalize": random.choices(['off', 'auto', 'soft1'], [0.4, 0.3, 0.3])[0],
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'holiday': self.holiday,
            'regression_type': self.regression_type,
            "growth": self.growth,
            "n_changepoints": self.n_changepoints,
            "changepoints_range": self.changepoints_range,
            "trend_reg": self.trend_reg,
            'trend_reg_threshold': self.trend_reg_threshold,
            # "ar_sparsity": self.ar_sparsity,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "seasonality_mode": self.seasonality_mode,
            "seasonality_reg": self.seasonality_reg,
            "n_lags": self.n_lags,
            "num_hidden_layers": self.num_hidden_layers,
            'd_hidden': self.d_hidden,
            "learning_rate": self.learning_rate,
            "loss_func": self.loss_func,
            # "train_speed": self.train_speed,
            "normalize": self.normalize,
        }
        return parameter_dict
