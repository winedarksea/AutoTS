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
    fit_anomaly_classifier,
    score_to_anomaly,
)
from autots.tools.transform import RandomTransform, GeneralTransformer
from autots.tools.impute import FillNA
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
        self.anomalies = None
        self.scores = None

    def detect(self, df):
        """Shared anomaly detection routine."""
        self.df = df.copy()
        self.df_anomaly = df.copy()
        if self.transform_dict is not None:
            model = GeneralTransformer(
                verbose=2, **self.transform_dict
            )  # DATEPART, LOG, SMOOTHING, DIFF, CLIP OUTLIERS with high z score
            # the post selecting by columns is for CenterSplit and any similar renames or expansions
            transformed_df = model.fit_transform(self.df_anomaly)
            # Only select columns that exist in both original and transformed data (from expanding transformers)
            common_cols = [col for col in self.df.columns if col in transformed_df.columns]
            if common_cols:
                self.df_anomaly = transformed_df[common_cols]
            else:
                self.df_anomaly = transformed_df

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
            df_for_limits = self.df_anomaly
            if self.eval_period is not None:
                df_for_limits = self.df_anomaly.tail(self.eval_period)
            self.anomalies, self.scores = limits_to_anomalies(
                df_for_limits,
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

    def remove_anomalies(self, df=None, fillna=None):
        """Detect and return a copy of the data with anomalies removed (set NaN or filled).

        Args:
            df (pd.DataFrame, optional): data to run detection on. If None, uses previous `detect` input.
            fillna (str, optional): fill method passed to `autots.tools.impute.FillNA`.
        """
        if df is not None:
            _, _ = self.detect(df)
        elif not hasattr(self, "df"):
            raise ValueError("Call `detect(df)` or provide `df` before removing anomalies.")
        df_clean = self.df.copy()
        df_clean = df_clean[self.anomalies != -1]
        if fillna is not None:
            df_clean = FillNA(df_clean, method=fillna, window=10)
        return df_clean

    def plot(self, series_name=None, title=None, marker_size=None, plot_kwargs={}, start_date=None):
        import matplotlib.pyplot as plt

        if series_name is None:
            series_name = random.choice(self.df.columns)
        if title is None:
            title = series_name[0:50] + f" with {self.method} outliers"
        
        # Filter data by start_date if provided
        df_plot = self.df
        anomalies_plot = self.anomalies
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            df_plot = self.df.loc[self.df.index >= start_date]
            anomalies_plot = self.anomalies.loc[self.anomalies.index >= start_date]
        
        fig, ax = plt.subplots()
        df_plot[series_name].plot(ax=ax, title=title, **plot_kwargs)
        if self.output == "univariate":
            i_anom = anomalies_plot.index[anomalies_plot.iloc[:, 0] == -1]
        else:
            series_anom = anomalies_plot[series_name]
            i_anom = series_anom[series_anom == -1].index
        if len(i_anom) > 0:
            if marker_size is None:
                marker_size = max(20, fig.dpi * 0.45)
            ax.scatter(
                i_anom.tolist(),
                df_plot.loc[i_anom, :][series_name],
                c="red",
                s=marker_size,
            )

    def fit(self, df):
        return self.detect(df)

    def fit_anomaly_classifier(self):
        """Fit a model to predict if a score is an anomaly."""
        self.anomaly_classifier, self.score_categories = fit_anomaly_classifier(
            self.anomalies, self.scores
        )

    def score_to_anomaly(self, scores):
        """A DecisionTree model, used as models are nonstandard (and nonparametric)."""
        if self.anomaly_classifier is None:
            self.fit_anomaly_classifier()
        return score_to_anomaly(scores, self.anomaly_classifier, self.score_categories)

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
            min_occurrences=self.min_occurrences,
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
        return self

    def plot_anomaly(self, kwargs={}):
        # Extract start_date if provided in kwargs to pass to the anomaly detector plot method
        self.anomaly_model.plot(**kwargs)

    def plot(
        self,
        series_name=None,
        include_anomalies=True,
        title=None,
        marker_size=None,
        plot_kwargs={},
        series=None,
        start_date=None,
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
        
        # Filter data by start_date if provided
        df_plot = self.df
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            df_plot = self.df.loc[self.df.index >= start_date]
        
        fig, ax = plt.subplots()
        df_plot[series_name].plot(ax=ax, title=title, **plot_kwargs)
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
            
            # Filter anomalies by start_date if provided
            if start_date is not None:
                i_anom = i_anom[i_anom >= start_date]
            # Ensure anomaly indices exist in filtered dataframe
            i_anom = i_anom[i_anom.isin(df_plot.index)]
            
            if len(i_anom) > 0:
                ax.scatter(
                    i_anom.tolist(),
                    df_plot.loc[i_anom, :][series_name],
                    c="red",
                    s=marker_size,
                )
        # now the actual holidays
        holiday_dates = self.dates_to_holidays(self.df.index, style="series_flag")[series_name]
        i_anom = holiday_dates.index[holiday_dates == 1]
        
        # Filter holidays by start_date if provided
        if start_date is not None:
            i_anom = i_anom[i_anom >= start_date]
        # Ensure holiday indices exist in filtered dataframe
        i_anom = i_anom[i_anom.isin(df_plot.index)]
        
        if len(i_anom) > 0:
            ax.scatter(
                i_anom.tolist(),
                df_plot.loc[i_anom, :][series_name],
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
