# -*- coding: utf-8 -*-
"""Anomaly Detector
Created on Mon Jul 18 14:19:55 2022

@author: Colin
"""
import random
from autots.tools.anomaly_utils import anomaly_new_params, detect_anomalies, limits_to_anomalies
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
        n_jobs=1,
    ):
        """Detect anomalies on a historic dataset.

        Args:
            output (str): 'multivariate' (each series unique outliers), or 'univariate' (all series together for one outlier flag per timestamp)
            method (str): method choosen, from sklearn, AutoTS, and basic stats. Use `.get_new_params()` to see potential models
            transform_dict (dict): option but helpful, often datepart, differencing, or other standard AutoTS transformer params
            forecast_params (dict): used to backcast and identify 'unforecastable' values, required only for predict_interval method
            method_params (dict): parameters specific to the method, use `.get_new_params()` to see potential models
            eval_periods (int): only use this length tail of data, currently only implemented for forecast_params forecasting if used
            n_jobs (int): multiprocessing jobs, used by some methods

        Methods:
            detect()
        """
        self.output = output
        self.method = method
        self.transform_dict = transform_dict
        self.forecast_params = forecast_params
        self.method_params = method_params
        self.eval_period = eval_period
        self.n_jobs = n_jobs

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
                **self.transform_dict
            )  # DATEPART, LOG, SMOOTHING, DIFF, CLIP OUTLIERS with high z score
            self.df_anomaly = model.fit_transform(self.df_anomaly)

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
                    self.df_anomaly = self.df_anomaly.tail(self.eval_period) - backcast.forecast
                else:
                    self.df_anomaly = self.df_anomaly - backcast.forecast

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
        return self.anomalies, self.scores

    def plot(self, series_name=None, title=None, plot_kwargs={}):
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
            ax.scatter(i_anom.tolist(), self.df.loc[i_anom, :][series_name], c="red")

    @staticmethod
    def get_new_params(method="random"):
        forecast_params = None
        method_choice, method_params, transform_dict = anomaly_new_params(method=method)
        if transform_dict == "random":
            transform_dict = RandomTransform(transformer_list='fast', transformer_max_depth=2)
        preforecast = random.choices([True, False], [0.05, 0.95])[0]

        if preforecast or method_choice == "prediction_interval":
            forecast_params = random_model(
                model_list=['LastValueNaive', 'GLS', 'RRVAR'],
                model_prob=[0.8, 0.1, 0.1],
                transformer_max_depth=5,
                keyword_format=True,
            )
        return {
            "method": method_choice,
            "transform_dict": transform_dict,
            "forecast_params": forecast_params,
            "method_params": method_params,
        }
