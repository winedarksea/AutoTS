# -*- coding: utf-8 -*-
"""Anomaly Detector
Created on Mon Jul 18 14:19:55 2022

@author: Colin
"""
import random
import numpy as np
import pandas as pd
from autots.tools.anomaly_utils import (
    anomaly_new_params,
    detect_anomalies,
    limits_to_anomalies,
    anomaly_df_to_holidays,
    holiday_new_params,
    dates_to_holidays,
)
from autots.tools.transform import RandomTransform, GeneralTransformer
from autots.evaluator.auto_model import random_model
from autots.evaluator.auto_model import back_forecast


class AnomalyDetector(object):
    def __init__(
        self,
        output="multivariate",
        method="zscore",
        transform_dict={  # also  suggest DifferencedTransformer
            "transformations": {0: "DatepartRegression"},
            "transformation_params": {
                0: {
                    "datepart_method": "simple_3",
                    "regression_model": {
                        "model": "ElasticNet",
                        "model_params": {},
                    },
                }
            },
        },
        forecast_params=None,
        method_params={},
        eval_period=None,
        isolated_only=False,
        n_jobs=1,
    ):
        """Detect anomalies on a historic dataset.
        Note anomaly score patterns vary by method.
        Anomaly flag is standard -1 = anomaly; 1 = regular as per sklearn

        Args:
            output (str): 'multivariate' (each series unique outliers), or 'univariate' (all series together for one outlier flag per timestamp)
            method (str): method choosen, from sklearn, AutoTS, and basic stats. Use `.get_new_params()` to see potential models
            transform_dict (dict): option but helpful, often datepart, differencing, or other standard AutoTS transformer params
            forecast_params (dict): used to backcast and identify 'unforecastable' values, required only for predict_interval method
            method_params (dict): parameters specific to the method, use `.get_new_params()` to see potential models
            eval_periods (int): only use this length tail of data, currently only implemented for forecast_params forecasting if used
            isolated_only (bool): if True, only standalone anomalies reported
            n_jobs (int): multiprocessing jobs, used by some methods

        Methods:
            detect()
            plot()
            get_new_params()
            score_to_anomaly()  # estimate

        Attributes:
            anomalies
            scores
        """
        self.output = output
        self.method = method
        self.transform_dict = transform_dict
        self.forecast_params = forecast_params
        self.method_params = method_params
        self.eval_period = eval_period
        self.isolated_only = isolated_only
        self.n_jobs = n_jobs
        self.anomaly_classifier = None

    def detect(self, df):
        """All will return -1 for anomalies.

        Args:
            df (pd.DataFrame): pandas wide-style data
        Returns:
            pd.DataFrame (classifications, -1 = outlier, 1 = not outlier), pd.DataFrame s(scores)
        """
        self.df = df.copy()
        self.df_anomaly = df.copy()
        if self.transform_dict is not None:
            model = GeneralTransformer(
                verbose=2, **self.transform_dict
            )  # DATEPART, LOG, SMOOTHING, DIFF, CLIP OUTLIERS with high z score
            # the post selecting by columns is for CenterSplit and any similar renames or expansions
            self.df_anomaly = model.fit_transform(self.df_anomaly)[self.df.columns]

        if self.forecast_params is not None:
            backcast = back_forecast(
                self.df_anomaly,
                n_splits=self.method_params.get("n_splits", "auto"),
                forecast_length=self.method_params.get("forecast_length", 4),
                frequency="infer",
                eval_period=self.eval_period,
                prediction_interval=self.method_params.get("prediction_interval", 0.9),
                **self.forecast_params,
            )
            # don't difference for prediction_interval
            if self.method not in ["prediction_interval"]:
                if self.eval_period is not None:
                    self.df_anomaly = (
                        self.df_anomaly.tail(self.eval_period) - backcast.forecast
                    )
                else:
                    self.df_anomaly = self.df_anomaly - backcast.forecast

        if len(self.df_anomaly.columns) != len(df.columns):
            raise ValueError(
                f"anomaly returned a column mismatch from params {self.method_params} and {self.transform_dict}"
            )
        if not all(self.df_anomaly.columns == df.columns):
            self.df_anomaly.columns = df.columns

        if self.method in ["prediction_interval"]:
            self.anomalies, self.scores = limits_to_anomalies(
                self.df_anomaly,
                output=self.output,
                method_params=self.method_params,
                upper_limit=backcast.upper_forecast,
                lower_limit=backcast.lower_forecast,
            )
        else:
            self.anomalies, self.scores = detect_anomalies(
                self.df_anomaly,
                output=self.output,
                method=self.method,
                transform_dict=self.transform_dict,
                method_params=self.method_params,
                eval_period=self.eval_period,
                n_jobs=self.n_jobs,
            )
        if self.isolated_only:
            # replace all anomalies (-1) except those which are isolated (1 before and after)
            mask_minus_one = self.anomalies == -1
            mask_prev_one = self.anomalies.shift(1) == 1
            mask_next_one = self.anomalies.shift(-1) == 1
            mask_replace = mask_minus_one & ~(mask_prev_one & mask_next_one)
            self.anomalies[mask_replace] = 1
        return self.anomalies, self.scores

    def plot(self, series_name=None, title=None, marker_size=None, plot_kwargs={}):
        import matplotlib.pyplot as plt

        if series_name is None:
            series_name = random.choice(self.df.columns)
        if title is None:
            title = series_name[0:50] + f" with {self.method} outliers"
        fig, ax = plt.subplots()
        self.df[series_name].plot(ax=ax, title=title, **plot_kwargs)
        if self.output == "univariate":
            i_anom = self.anomalies.index[self.anomalies.iloc[:, 0] == -1]
        else:
            series_anom = self.anomalies[series_name]
            i_anom = series_anom[series_anom == -1].index
        if len(i_anom) > 0:
            if marker_size is None:
                marker_size = max(20, fig.dpi * 0.45)
            ax.scatter(
                i_anom.tolist(),
                self.df.loc[i_anom, :][series_name],
                c="red",
                s=marker_size,
            )

    def fit(self, df):
        return self.detect(df)

    def fit_anomaly_classifier(self):
        """Fit a model to predict if a score is an anomaly."""
        # Using DecisionTree as it should almost handle nonparametric anomalies
        from sklearn.tree import DecisionTreeClassifier

        scores_flat = self.scores.melt(var_name='series', value_name="value")
        categor = pd.Categorical(scores_flat['series'])
        self.score_categories = categor.categories
        scores_flat['series'] = categor
        scores_flat = pd.concat(
            [pd.get_dummies(scores_flat['series']), scores_flat['value']], axis=1
        )
        anomalies_flat = self.anomalies.melt(var_name='series', value_name="value")
        self.anomaly_classifier = DecisionTreeClassifier(max_depth=None).fit(
            scores_flat, anomalies_flat['value']
        )
        # anomaly_classifier.score(scores_flat, anomalies_flat['value'])

    def score_to_anomaly(self, scores):
        """A DecisionTree model, used as models are nonstandard (and nonparametric)."""
        if self.anomaly_classifier is None:
            self.fit_anomaly_classifier()
        scores.index.name = 'date'
        scores_flat = scores.reset_index(drop=False).melt(
            id_vars="date", var_name='series', value_name="value"
        )
        scores_flat['series'] = pd.Categorical(
            scores_flat['series'], categories=self.score_categories
        )
        res = self.anomaly_classifier.predict(
            pd.concat(
                [
                    pd.get_dummies(scores_flat['series'], dtype=float),
                    scores_flat['value'],
                ],
                axis=1,
            )
        )
        res = pd.concat(
            [scores_flat[['date', "series"]], pd.Series(res, name='value')], axis=1
        ).pivot_table(index='date', columns='series', values="value")
        return res[scores.columns]

    @staticmethod
    def get_new_params(method="random"):
        """Generate random new parameter combinations.

        Args:
            method (str): 'fast', 'deep', 'default', or any of the anomaly method names (ie 'IQR') to specify only that method
        """
        forecast_params = None
        method_choice, method_params, transform_dict = anomaly_new_params(method=method)
        if transform_dict == "random":
            transform_dict = RandomTransform(
                transformer_list='scalable', transformer_max_depth=2
            )
        if method == "fast":
            preforecast = False
        else:
            preforecast = random.choices([True, False], [0.05, 0.95])[0]

        if preforecast or method_choice == "prediction_interval":
            forecast_params = random_model(
                model_list=['LastValueNaive', 'GLS', 'RRVAR', "SeasonalityMotif"],
                model_prob=[0.8, 0.1, 0.05, 0.05],
                transformer_max_depth=5,
                transformer_list="superfast",
                keyword_format=True,
            )
        return {
            "method": method_choice,
            "transform_dict": transform_dict,
            "forecast_params": forecast_params,
            "method_params": method_params,
        }


class HolidayDetector(object):
    def __init__(
        self,
        anomaly_detector_params={},
        threshold=0.8,
        min_occurrences=2,
        splash_threshold=0.65,
        use_dayofmonth_holidays=True,
        use_wkdom_holidays=True,
        use_wkdeom_holidays=True,
        use_lunar_holidays=True,
        use_lunar_weekday=False,
        use_islamic_holidays=False,
        use_hebrew_holidays=False,
        use_hindu_holidays=False,
        output: str = "multivariate",
        n_jobs: int = 1,
    ):
        """Detect anomalies, then mark as holidays (events, festivals, etc) any that reoccur to a calendar.

        Be aware of timezone, especially combining series from multiple time zones. Dates then may not accurately align.
        Can pick up a holiday on the wrong calendar especially for extended holidays (Christmas) and with short (several years is short here) history.
        Holidays on unusual days or weekdays of month (5th Monday of April) may occur
        No multiyear patterns (election year) are detected - would need lots of history

        Args:
            anomaly_detector_params (dict): anomaly detection params passed to detector class
            threshold (float): percent of date occurrences that must be anomalous (0 - 1)
            splash_threshold (float): None, or % required, avg of nearest 2 neighbors to point
            use* (bool): whether to use these calendars for holiday detection
            output (str): "multivariate" or "univariate", for univariate not all dates_to_holidays styles will work

        Methods:
            detect()
            dates_to_holidays()
            plot()
            get_new_params()
        """
        self.anomaly_detector_params = anomaly_detector_params
        self.threshold = threshold
        self.min_occurrences = min_occurrences
        self.splash_threshold = splash_threshold
        self.use_dayofmonth_holidays = use_dayofmonth_holidays
        self.use_wkdom_holidays = use_wkdom_holidays
        self.use_wkdeom_holidays = use_wkdeom_holidays
        self.use_lunar_holidays = use_lunar_holidays
        self.use_lunar_weekday = use_lunar_weekday
        self.use_islamic_holidays = use_islamic_holidays
        self.use_hebrew_holidays = use_hebrew_holidays
        self.use_hindu_holidays = use_hindu_holidays
        self.n_jobs = n_jobs
        self.output = output
        self.anomaly_model = AnomalyDetector(
            output=output, **self.anomaly_detector_params, n_jobs=n_jobs
        )

    def detect(self, df):
        """Run holiday detection. Input wide-style pandas time series."""
        self.anomaly_model.detect(df)
        self.df = df
        self.df_cols = df.columns
        if np.min(self.anomaly_model.anomalies.values) != -1:
            print("No anomalies detected.")
        (
            self.day_holidays,
            self.wkdom_holidays,
            self.wkdeom_holidays,
            self.lunar_holidays,
            self.lunar_weekday,
            self.islamic_holidays,
            self.hebrew_holidays,
            self.hindu_holidays,
        ) = anomaly_df_to_holidays(
            self.anomaly_model.anomalies,
            splash_threshold=self.splash_threshold,
            threshold=self.threshold,
            actuals=df if self.output != "univariate" else None,
            anomaly_scores=(
                self.anomaly_model.scores if self.output != "univariate" else None
            ),
            use_dayofmonth_holidays=self.use_dayofmonth_holidays,
            use_wkdom_holidays=self.use_wkdom_holidays,
            use_wkdeom_holidays=self.use_wkdeom_holidays,
            use_lunar_holidays=self.use_lunar_holidays,
            use_lunar_weekday=self.use_lunar_weekday,
            use_islamic_holidays=self.use_islamic_holidays,
            use_hebrew_holidays=self.use_hebrew_holidays,
            use_hindu_holidays=self.use_hindu_holidays,
        )

    def plot_anomaly(self, kwargs={}):
        self.anomaly_model.plot(**kwargs)

    def plot(
        self,
        series_name=None,
        include_anomalies=True,
        title=None,
        marker_size=None,
        plot_kwargs={},
        series=None,
    ):
        import matplotlib.pyplot as plt

        if series_name is None:
            if series is not None:
                series_name = series
            else:
                series_name = random.choice(self.df.columns)
        if title is None:
            title = (
                series_name[0:50]
                + f" with {self.anomaly_detector_params['method']} holidays"
            )
        fig, ax = plt.subplots()
        self.df[series_name].plot(ax=ax, title=title, **plot_kwargs)
        if marker_size is None:
            marker_size = max(20, fig.dpi * 0.45)
        if include_anomalies:
            # directly copied from above
            if self.anomaly_model.output == "univariate":
                i_anom = self.anomaly_model.anomalies.index[
                    self.anomaly_model.anomalies.iloc[:, 0] == -1
                ]
            else:
                series_anom = self.anomaly_model.anomalies[series_name]
                i_anom = series_anom[series_anom == -1].index
            if len(i_anom) > 0:
                ax.scatter(
                    i_anom.tolist(),
                    self.df.loc[i_anom, :][series_name],
                    c="red",
                    s=marker_size,
                )
        # now the actual holidays
        i_anom = self.dates_to_holidays(self.df.index, style="series_flag")[series_name]
        i_anom = i_anom.index[i_anom == 1]
        if len(i_anom) > 0:
            ax.scatter(
                i_anom.tolist(),
                self.df.loc[i_anom, :][series_name],
                c="green",
                s=marker_size,
            )

    def dates_to_holidays(self, dates, style="flag", holiday_impacts=False):
        """Populate date information for a given pd.DatetimeIndex.

        Args:
            dates (pd.DatetimeIndex): list of dates
            day_holidays (pd.DataFrame): list of month/day holidays. Pass None if not available
            style (str): option for how to return information
                "long" - return date, name, series for all holidays in a long style dataframe
                "impact" - returns dates, series with values of sum of impacts (if given) or joined string of holiday names
                'flag' - return dates, holidays flag, (is not 0-1 but rather sum of input series impacted for that holiday and day)
                'prophet' - return format required for prophet. Will need to be filtered on `series` for multivariate case
                'series_flag' - dates, series 0/1 for if holiday occurred in any calendar
            holiday_impacts (dict): a dict passed to .replace contaning values for holiday_names, or str 'value' or 'anomaly_score'
        """
        return dates_to_holidays(
            dates,
            self.df_cols,
            style=style,
            holiday_impacts=holiday_impacts,
            day_holidays=self.day_holidays,
            wkdom_holidays=self.wkdom_holidays,
            wkdeom_holidays=self.wkdeom_holidays,
            lunar_holidays=self.lunar_holidays,
            lunar_weekday=self.lunar_weekday,
            islamic_holidays=self.islamic_holidays,
            hebrew_holidays=self.hebrew_holidays,
            hindu_holidays=self.hindu_holidays,
        )

    def fit(self, df):
        return self.detect(df)

    @staticmethod
    def get_new_params(method="random"):
        holiday_params = holiday_new_params(method=method)
        holiday_params['anomaly_detector_params'] = AnomalyDetector.get_new_params(
            method=method
        )
        return holiday_params
