# -*- coding: utf-8 -*-
"""Greykite."""
import datetime
import random
import pandas as pd
from autots.models.base import ModelObject, PredictionObject

try:
    from greykite.framework.templates.autogen.forecast_config import ForecastConfig
    from greykite.framework.templates.autogen.forecast_config import MetadataParam
    from greykite.framework.templates.forecaster import Forecaster
    from greykite.framework.templates.model_templates import ModelTemplateEnum
    from greykite.framework.templates.autogen.forecast_config import ComputationParam
    from greykite.algo.forecast.silverkite.constants.silverkite_holiday import (
        SilverkiteHoliday,
    )
    from greykite.framework.templates.autogen.forecast_config import (
        ModelComponentsParam,
    )
except Exception:
    _has_greykite = False
else:
    _has_greykite = True


def seek_the_oracle(
    df_index,
    series,
    col,
    forecast_length,
    freq,
    prediction_interval=0.9,
    model_template='silverkite',
    growth=None,
    holiday=True,
    holiday_country="UnitedStates",
    regressors=None,
    verbose=0,
    inner_n_jobs=1,
    **kwargs,
):
    """Internal. For loop or parallel version of Greykite."""
    inner_df = pd.DataFrame(
        {
            'ts': df_index,
            'y': series,
        }
    )
    if regressors is not None:
        inner_regr = regressors.copy()
        new_names = [
            'rrrr' + str(x) if x in inner_df.columns else str(x)
            for x in inner_regr.columns
        ]
        inner_regr.columns = new_names
        inner_regr.index.name = 'ts'
        inner_regr.reset_index(drop=False, inplace=True)
        inner_df = inner_df.merge(inner_regr, left_on='ts', right_on='ts', how='outer')
    metadata = MetadataParam(
        time_col="ts",  # name of the time column ("date" in example above)
        value_col="y",  # name of the value column ("sessions" in example above)
        freq=freq,  # "H" for hourly, "D" for daily, "W" for weekly, etc.
    )
    # INCLUDE forecast_length lagged mean and std of other features!
    model_template = ModelTemplateEnum.SILVERKITE.name
    forecaster = Forecaster()  # Creates forecasts and stores the result
    if regressors is not None:
        model_components = ModelComponentsParam(
            growth=growth, regressors={"regressor_cols": new_names}
        )
    else:
        model_components = ModelComponentsParam(
            growth=growth,  # 'linear', 'quadratic', 'sqrt'
        )
    computation = ComputationParam(n_jobs=inner_n_jobs, verbose=verbose)
    if holiday:  # also 'auto'
        model_components.events = {
            # These holidays as well as their pre/post dates are modeled as individual events.
            "holidays_to_model_separately": SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES,  # all holidays in "holiday_lookup_countries"
            "holiday_lookup_countries": [
                holiday_country
            ],  # only look up holidays in the United States
            "holiday_pre_num_days": 1,  # also mark the 1 days before a holiday as holiday
            "holiday_post_num_days": 1,  # also mark the 1 days after a holiday as holiday
        }
    config = ForecastConfig(
        model_template=model_template,
        forecast_horizon=forecast_length,
        coverage=prediction_interval,
        model_components_param=model_components,
        metadata_param=metadata,
        computation_param=computation,
    )
    result = forecaster.run_forecast_config(  # result is also stored as `forecaster.forecast_result`.
        df=inner_df,
        config=config,
    )
    res_df = result.forecast.df.tail(forecast_length).drop(columns=['actual'])
    res_df['series_id'] = col
    return res_df


class Greykite(ModelObject):
    """Greykite

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holidays
        regression_type (str): type of regression (None, 'User')

    """

    def __init__(
        self,
        name: str = "Greykite",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday: bool = False,
        growth: str = None,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = None,
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
        self.growth = growth

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        if not _has_greykite:
            raise ImportError("Package greykite is required")

        df = self.basic_profile(df)
        self.regressor_train = None

        if self.regression_type == 'User':
            self.regressor_train = future_regressor.copy()

        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=[],
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
        if not _has_greykite:
            raise ImportError("Package greykite is required")
        predictStartTime = datetime.datetime.now()
        regressors = None
        if self.regression_type == 'User':
            regressors = pd.concat([self.regressor_train, future_regressor])

        parallel = True
        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        else:
            try:
                from joblib import Parallel, delayed
            except Exception:
                parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            verbs = 0 if self.verbose < 1 else self.verbose - 1
            df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                delayed(seek_the_oracle)(
                    self.df_train.index,
                    self.df_train[col],
                    col,
                    forecast_length,
                    freq=self.frequency,
                    prediction_interval=self.prediction_interval,
                    growth=dict(growth_term=self.growth),
                    holiday=self.holiday,
                    holiday_country=self.holiday_country,
                    regressors=regressors,
                    inner_n_jobs=self.n_jobs,
                )
                for col in cols
            )
            complete = pd.concat(df_list)
        else:
            df_list = []
            for col in cols:
                df_list.append(
                    seek_the_oracle(
                        self.df_train.index,
                        self.df_train[col],
                        col,
                        forecast_length,
                        freq=self.frequency,
                        prediction_interval=self.prediction_interval,
                        growth=dict(growth_term=self.growth),
                        holiday=self.holiday,
                        holiday_country=self.holiday_country,
                        regressors=regressors,
                        inner_n_jobs=self.n_jobs,
                    )
                )
            complete = pd.concat(df_list)

        forecast = complete.pivot_table(
            values="forecast", index="ts", columns="series_id", aggfunc="sum"
        )

        if just_point_forecast:
            return forecast
        else:
            upper_forecast = complete.pivot_table(
                values="forecast_upper", index="ts", columns="series_id", aggfunc="sum"
            )
            lower_forecast = complete.pivot_table(
                values="forecast_lower", index="ts", columns="series_id", aggfunc="sum"
            )

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
        holiday_choice = random.choices([True, False], [0.2, 0.8])[0]
        regression_list = [None, 'User']
        regression_probability = [0.8, 0.2]
        regression_choice = random.choices(regression_list, regression_probability)[0]
        growth_choice = random.choices(
            [None, 'linear', 'quadratic', 'sqrt'], [0.3, 0.3, 0.1, 0.1]
        )[0]

        parameter_dict = {
            'holiday': holiday_choice,
            'regression_type': regression_choice,
            'growth': growth_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'holiday': self.holiday,
            'regression_type': self.regression_type,
            'growth': self.growth,
        }
        return parameter_dict
