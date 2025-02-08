"""
Complicated
"""

import random
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.tools.percentile import nan_quantile, trimmed_mean
from autots.tools.transform import (
    GeneralTransformer, RandomTransform,
)
from autots.models.model_list import scalable

class PreprocessingExperts(ModelObject):
    """Regression use the last n values as the basis of training data.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        # regression_type: str = None,
    """

    def __init__(
        self,
        name: str = "PreprocessingExperts",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2023,
        verbose: int = 0,
        model_params: dict = {
            "model_str": 'DatepartRegresion',
            "parameter_dict": {
                'regression_model': {
                    'model': 'ElasticNet',
                    'model_params': {
                        'l1_ratio': 0.1,
                        'fit_intercept': True,
                        'selection': 'cyclic',
                        'max_iter': 1000}},
                'datepart_method': 'expanded_binarized',
                'polynomial_degree': None,
                'holiday_countries_used': False,
                'lags': None,
                'forward_lags': None,
                'regression_type': None
            },
            "transformation_dict": {
                'fillna': 'rolling_mean_24',
                'transformations': {0: 'StandardScaler'},
                'transformation_params': {0: {}}
            },
        },
        transformation_dict=None,
        forecast_length: int = 28,
        point_method: str = "mean",
        n_jobs: int = -1,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.model_params = model_params
        self.transformation_dict = transformation_dict
        self.point_method = point_method
        self.forecast_length = forecast_length

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        self.fit_data(df)
        # from autots.tools.transform import GeneralTransformer  # avoid circular imports

        if self.transformation_dict is None:
            raise ValueError("transformation_dict cannot be None with PreprocessingRegression")

        self.transformer_object = GeneralTransformer(
            n_jobs=self.n_jobs,
            holiday_country=self.holiday_country,
            verbose=self.verbose,
            random_seed=self.random_seed,
            forecast_length=self.forecast_length,
            **self.transformation_dict,
        )
        df = self.transformer_object._first_fit(df)

        from autots.evaluator.auto_model import ModelPrediction  # avoid circular imports

        new_df = df
        self.model = {}
        trans_keys = sorted(list(self.transformer_object.transformations.keys()))
        
        self.model[0] = ModelPrediction(
            forecast_length=self.forecast_length, frequency=self.frequency,
            **self.model_params
        ).fit(new_df, future_regressor=future_regressor)  
        for i in trans_keys:
            trans_name = self.transformer_object.transformations[i]
            # assumes first i is always 0
            if trans_name not in ['Slice']:
                new_df = self.transformer_object._fit_one(new_df, i)
            self.model[int(i) + 1] = ModelPrediction(
                forecast_length=self.forecast_length, frequency=self.frequency,
                **self.model_params
            ).fit(new_df, future_regressor=future_regressor)

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def fit_data(self, df, future_regressor=None):
        df = self.basic_profile(df)
        return self

    def predict(
        self,
        forecast_length: int = None,
        future_regressor=None,
        just_point_forecast: bool = False,
        df=None,
    ):
        """Generate forecast data immediately following dates of .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        if df is not None:
            self.fit_data(df)
        if forecast_length is None:
            forecast_length = self.forecast_length
        if int(forecast_length) > int(self.forecast_length) and not self.one_step:
            print("Regression must be refit to change forecast length!")
        index = self.create_forecast_index(forecast_length=forecast_length)

        df_list = []
        self.saved = {}
        trans_keys = sorted(list(self.transformer_object.transformations.keys()))
        # first one is on no preprocessing
        rfPred = self.model[0].predict(forecast_length, future_regressor=future_regressor)
        self.saved[0] = rfPred.forecast.copy()
        df_list.append(rfPred.forecast)
        for i in trans_keys:
            key = int(i) + 1
            rfPred = self.model[key].predict(forecast_length, future_regressor=future_regressor)
            self.saved[key] = rfPred.forecast.copy()
            # rfPred_upper = self.transformer_object.inverse_transform(rfPred.upper_forecast, start=i)
            # rfPred_lower = self.transformer_object.inverse_transform(rfPred.lower_forecast, start=i)
            rfPred = self.transformer_object.inverse_transform(rfPred.forecast, start=i)
            df_list.append(rfPred)
        # (num_windows, forecast_length, num_series)
        self.result_windows = np.asarray(df_list)

        if self.point_method == "weighted_mean":
            # weighted by later (more preprocessing higher)
            weights = np.arange(len(trans_keys) + 1) + 1
            forecast = np.average(self.result_windows, axis=0, weights=weights)
        elif self.point_method == "mean":
            forecast = np.nanmean(self.result_windows, axis=0)
        elif self.point_method == "median":
            forecast = np.nanmedian(self.result_windows, axis=0)
        elif self.point_method == "midhinge":
            q1 = nan_quantile(self.result_windows, q=0.25, axis=0)
            q2 = nan_quantile(self.result_windows, q=0.75, axis=0)
            forecast = (q1 + q2) / 2
        elif self.point_method == "trimmed_mean_20":
            forecast = trimmed_mean(self.result_windows, percent=0.2, axis=0)
        elif self.point_method == "trimmed_mean_40":
            forecast = trimmed_mean(self.result_windows, percent=0.4, axis=0)
        elif self.point_method == "closest":
            forecast = self.result_windows[-1]
        else:
            raise ValueError(f"point_method {self.point_method} not recognized.")

        df = pd.DataFrame(forecast, index=index, columns=self.column_names)
        pred_int = round((1 - self.prediction_interval) / 2, 5)
        upper_forecast = nan_quantile(self.result_windows, q=(1 - pred_int), axis=0)
        lower_forecast = nan_quantile(self.result_windows, q=pred_int, axis=0)
        lower_forecast = pd.DataFrame(
            lower_forecast, index=index, columns=self.column_names
        )
        upper_forecast = pd.DataFrame(
            upper_forecast, index=index, columns=self.column_names
        )
        if just_point_forecast:
            return df
        else:
            if False:
                # might still be worth using, as the spread of forecasts above is often small
                upper_forecast, lower_forecast = Point_to_Probability(
                    self.last_window,
                    df,
                    prediction_interval=self.prediction_interval,
                    method='inferred_normal',
                )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=df.index,
                forecast_columns=df.columns,
                lower_forecast=lower_forecast,
                forecast=df,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        from autots.evaluator.auto_model import ModelMonster  # avoid circular imports

        model_type = random.choices(list(scalable.keys()), list(scalable.values()))[0]
        model_params =  ModelMonster(model_type).get_new_params(method="default")
        return {
            "point_method": random.choices(
                [
                    "weighted_mean",
                    "mean",
                    "median",
                    "midhinge",
                    'closest',
                    'trimmed_mean_20',
                    'trimmed_mean_40',
                ],
                [0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
            )[0],
            'model_params': {
                "model_str": model_type,
                "parameter_dict": model_params,
                "transformation_dict": RandomTransform(
                    transformer_list="scalable", transformer_max_depth=2
                ),
            },
            'transformation_dict': None,  # assume this passed via AutoTS transformer dict
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            "point_method": self.point_method,
            'model_params': self.model_params,
            'transformation_dict': self.transformation_dict,
        }