# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:28:57 2023

@author: Colin
"""
import json
import sys
import datetime
import random
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.evaluator.auto_model import RandomTemplate, model_forecast
from autots.models.model_list import diff_window_motif_list, all_result_path
from autots.evaluator.event_forecasting import extract_result_windows
from autots.evaluator.validation import (
    validate_num_validations,
    generate_validation_indices,
)
from autots.models.sklearn import (
    retrieve_regressor,
)  # generate_regressor_params, datepart_model_dict
from autots.tools.shaping import simple_train_test_split
from autots.tools.seasonal import date_part


def create_feature(
    df_train,
    models,
    forecast_length,
    future_regressor_train=None,
    future_regressor_forecast=None,
    datepart_method=None,
):
    result_windows = None
    res = []
    # add last value as a feature
    res.append(
        np.repeat(
            df_train.iloc[-1:,].to_numpy(),
            forecast_length,
            axis=0,
        )
    )
    # add averages
    res.append(
        np.repeat(df_train.mean().to_numpy()[np.newaxis, :], forecast_length, axis=0)
    )
    res.append(
        np.repeat(df_train.std().to_numpy()[np.newaxis, :], forecast_length, axis=0)
    )
    for model in models:
        model_name = model['Model']
        if model_name in diff_window_motif_list:
            model_param_dict = {
                **json.loads(model['ModelParameters']),
                **{"return_result_windows": True},
            }
        else:
            model_param_dict = json.loads(model['ModelParameters'])

        forecasts = model_forecast(
            model_name=model['Model'],
            model_param_dict=model_param_dict,
            model_transform_dict=json.loads(model['TransformationParameters']),
            df_train=df_train,  # .iloc[:, 0:1]
            forecast_length=forecast_length,
            frequency='infer',
            prediction_interval=0.9,
            no_negatives=False,
            future_regressor_train=future_regressor_train,
            future_regressor_forecast=future_regressor_forecast,
            random_seed=321,
            verbose=1,
            n_jobs=4,
            return_model=True,
            fail_on_forecast_nan=True,
        )
        res.extend(
            [forecasts.forecast, forecasts.upper_forecast, forecasts.lower_forecast]
        )
        if model_name == 'Cassandra':
            comps = forecasts.model.return_components()
            ncomps = comps.columns.get_level_values(1)
            for cp in set(ncomps):
                res.append(
                    comps.loc[:, comps.columns.get_level_values(1) == cp]
                    .droplevel(1, 1)
                    .reindex(forecasts.forecast.index)
                )
        elif model_name in all_result_path:
            result_windows = extract_result_windows(forecasts, model_name=model_name)
    res = np.stack(res)
    if result_windows is not None:
        res = np.concatenate([res, result_windows], axis=0)
    # add regressor as a feature
    if future_regressor_forecast is not None:
        r = np.repeat(
            future_regressor_forecast.to_numpy().T[:, :, np.newaxis],
            df_train.shape[1],
            axis=2,
        )
        res = np.concatenate([res, r], axis=0)
    # add timestep as a feature
    res = np.concatenate(
        (
            res,
            np.linspace(
                [0] * df_train.shape[1],
                [res.shape[1] - 1] * df_train.shape[1],
                res.shape[1],
            )[np.newaxis, :, :],
        ),
        axis=0,
    )
    # add time series ID as a feature
    res = np.concatenate(
        (
            res,
            np.linspace(
                [0] * forecast_length,
                [res.shape[2] - 1] * forecast_length,
                res.shape[2],
            ).T[np.newaxis, :, :],
        ),
        axis=0,
    )
    if datepart_method not in [None, "None", "none"]:
        date_part_df = date_part(forecasts.forecast.index, method=datepart_method)
        res = np.concatenate(
            (
                res,
                np.repeat(date_part_df.to_numpy()[np.newaxis, :, :], 21, axis=0).T,
            ),
            axis=0,
        )

    sys.stdout.flush()
    return res


"""
X = []
y = []
for val in range(num_validations + 1):
    df_subset = df_wide_numeric.loc[validation_indexes[val]]
    df_train, df_test = simple_train_test_split(
        df_subset,
        forecast_length=forecast_length,
        min_allowed_train_percent=min_allowed_train_percent,
        verbose=verbose,
    )
    try:
        res = create_feature(df_train, future_regressor, models, forecast_length)
        X.append(res.reshape(res.shape[0], -1))
        y.append(df_test.to_numpy().reshape(-1))
    except Exception as e:
        if val < 1:
            raise e
        else:
            print(f"validation round {y} failed with {repr(e)}")

X = np.concatenate(X, axis=1).T
y = np.concatenate(y, axis=0)


regr = retrieve_regressor(
    regression_model={
        "model": 'XGBRegressor',
        "model_params": {
            "base_score":0.5, "booster":'gbtree', "callbacks":[],
            "colsample_bylevel":0.5050297936099943, "colsample_bynode":1,
            "colsample_bytree":0.616466808436318, "early_stopping_rounds":None,
            "enable_categorical":False, "eval_metric":None, "feature_types":None,
            "gamma":0, "gpu_id":-1, "grow_policy":'depthwise', "importance_type":None,
            "interaction_constraints":'', "learning_rate":0.0026632745639476375,
            "max_bin":256, "max_cat_threshold":64, "max_cat_to_onehot":4,
            "max_delta_step":0, "max_depth":10, "max_leaves":0,
            "min_child_weight":0.010458618463525058,
            "monotone_constraints":'()', "n_estimators":1356,
            "num_parallel_tree":1,
        },
    },
    verbose=verbose,
    verbose_bool=False,
    random_seed=2020,
    n_jobs=1,
    multioutput=False,
)
regr.fit(X, y)


X = []
res = create_feature(df_wide_numeric, future_regressor_future, models, forecast_length)
X.append(res.reshape(res.shape[0], -1))
X = np.concatenate(X, axis=1).T
result = pd.DataFrame(
    regr.predict(X).reshape(forecast_length, df_train.shape[1]),
    index=future_regressor_future.index,
    columns=df_wide_numeric.columns,
)
"""


class MLEnsemble(ModelObject):
    """Combine models using an ML model across validations.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
    """

    def __init__(
        self,
        name: str = "MLEnsemble",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        forecast_length: int = 10,
        regression_type: str = None,
        regression_model=None,
        models=[
            {
                'Model': 'Cassandra',
                'ModelParameters': {},
                "TransformationParameters": {},
            },
            {
                'Model': 'MetricMotif',
                'ModelParameters': {},
                "TransformationParameters": {},
            },
            {
                'Model': 'SeasonalityMotif',
                'ModelParameters': {},
                "TransformationParameters": {},
            },
        ],
        num_validations=2,
        validation_method="backwards",
        min_allowed_train_percent=0.5,
        datepart_method="expanded_binarized",
        models_source: str = 'random',
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
            regression_type=regression_type,
        )
        self.forecast_length = forecast_length
        assert num_validations >= 0, f"window {num_validations} must be >= 0"
        self.num_validations = int(num_validations)
        self.validation_method = validation_method
        self.min_allowed_train_percent = min_allowed_train_percent
        self.datepart_method = datepart_method
        if regression_model is None:
            try:
                import xgboost  # noqa

                regression_model = {
                    "model": 'XGBRegressor',
                    "model_params": {
                        "base_score": 0.5,
                        "booster": 'gbtree',
                        "colsample_bylevel": 0.50502979,
                        "colsample_bynode": 1,
                        "colsample_bytree": 0.6164668,
                        "early_stopping_rounds": None,
                        "enable_categorical": False,
                        "eval_metric": None,
                        "feature_types": None,
                        "gamma": 0,
                        "grow_policy": 'depthwise',
                        "importance_type": None,
                        "interaction_constraints": '',
                        "learning_rate": 0.00266327,
                        "max_bin": 256,
                        "max_cat_threshold": 64,
                        "max_cat_to_onehot": 4,
                        "max_delta_step": 0,
                        "max_depth": 10,
                        "max_leaves": 0,
                        "min_child_weight": 0.0104586,
                        "monotone_constraints": '()',
                        "n_estimators": 1356,
                        "num_parallel_tree": 1,
                    },
                }
            except Exception:
                regression_model = {
                    "model": 'RandomForest',
                    "model_params": {
                        "max_features": 0.16322,
                        "max_leaf_nodes": 10,
                        "n_estimators": 5,
                    },
                }
        self.regression_model = regression_model
        self.models = models
        self.models_source = models_source

        stride_size = round(self.forecast_length / 2)
        stride_size = stride_size if stride_size > 0 else 1
        self.similarity_validation_params = {
            "stride_size": stride_size,
            "distance_metric": "nan_euclidean",
            "include_differenced": True,
            "window_size": 30,
        }
        self.seasonal_validation_params = {
            'window_size': 10,
            'distance_metric': 'mae',
            'datepart_method': 'common_fourier_rw',
        }

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            regressor (numpy.Array): additional regressor
        """
        df = self.basic_profile(df)
        self.regressor_train = future_regressor
        if self.regression_type in ["User", "user"] and future_regressor is None:
            raise ValueError("regression_type='User' but no future_regressor passed")
        self.df = df.copy()

        # check how many validations are possible given the length of the data.
        self.num_validations = validate_num_validations(
            self.validation_method,
            self.num_validations,
            df,
            self.forecast_length,
            min_allowed_train_percent=self.min_allowed_train_percent,
            verbose=self.verbose,
        )

        # generate validation indices (so it can fail now, not after all the generations)
        self.validation_indexes = generate_validation_indices(
            self.validation_method,
            self.forecast_length,
            self.num_validations,
            df,
            validation_params=(
                self.similarity_validation_params
                if self.validation_method == "similarity"
                else self.seasonal_validation_params
            ),
            preclean=None,
            verbose=0,
        )
        X = []
        y = []
        for val in range(self.num_validations + 1):
            df_subset = df.loc[self.validation_indexes[val]]
            df_train, df_test = simple_train_test_split(
                df_subset,
                forecast_length=self.forecast_length,
                min_allowed_train_percent=self.min_allowed_train_percent,
                verbose=self.verbose,
            )
            if future_regressor is not None:
                regr_subset_t = future_regressor.reindex(index=df_train.index)
                regr_subset_f = future_regressor.reindex(index=df_test.index)
            else:
                regr_subset_t = None
                regr_subset_f = None
            try:
                self.res = create_feature(
                    df_train,
                    self.models,
                    self.forecast_length,
                    future_regressor_train=regr_subset_t,
                    future_regressor_forecast=regr_subset_f,
                    datepart_method=self.datepart_method,
                )
                X.append(self.res.reshape(self.res.shape[0], -1))
                y.append(df_test.to_numpy().reshape(-1))
            except Exception as e:
                if val < 1:
                    raise e
                else:
                    print(f"validation round {y} failed with {repr(e)}")

        self.X = np.concatenate(X, axis=1).T
        self.y = np.concatenate(y, axis=0)

        self.regr = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=self.verbose,
            verbose_bool=False,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            multioutput=False,
        )
        self.regr.fit(self.X, self.y)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int = None,
        future_regressor=None,
        just_point_forecast=False,
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        if forecast_length != self.forecast_length:
            raise ValueError("forecast_length must be changed in MLEnsemble init")
        test_index = self.create_forecast_index(forecast_length=self.forecast_length)

        X = []
        res = create_feature(
            self.df,
            self.models,
            self.forecast_length,
            future_regressor_train=self.regressor_train,
            future_regressor_forecast=future_regressor,
            datepart_method=self.datepart_method,
        )
        X.append(res.reshape(res.shape[0], -1))
        X = np.concatenate(X, axis=1).T
        forecast = pd.DataFrame(
            self.regr.predict(X).reshape(self.forecast_length, self.df.shape[1]),
            index=test_index,
            columns=self.df.columns,
        )

        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
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
        """Returns dict of new parameters for parameter tuning"""
        if "regressor" in method:
            regression_type_choice = "User"
        else:
            regression_type_choice = None
        # regr_params = generate_regressor_params(model_dict=datepart_model_dict)
        # model3 = random.choice(['Cassandra', 'MetricMotif', 'FBProphet', "SeasonalityMotif", "DatepartRegression"])
        models = RandomTemplate(
            n=3,
            model_list='fast',
            transformer_max_depth=4,
            transformer_list='fast',
        )
        models = models[
            ['Model', 'ModelParameters', 'TransformationParameters']
        ].to_dict(orient='records')

        return {
            "num_validations": random.choices([0, 1, 2], [0.5, 0.3, 0.2])[0],
            "validation_method": random.choices(
                ["backwards", "similarity", "seasonal"], [0.5, 0.3, 0.2]
            )[0],
            "regression_model": None,
            "models": models,
            "datepart_method": random.choices(
                [
                    None,
                    "recurring",
                    "simple",
                    "expanded",
                    "simple_2",
                    "simple_binarized",
                    "expanded_binarized",
                    'common_fourier',
                    'common_fourier_rw',
                ],
                [0.2, 0.2, 0.1, 0.3, 0.3, 0.4, 0.35, 0.45, 0.2],
            )[0],
            "regression_type": regression_type_choice,
            "models_source": 'random',
        }

    def get_params(self):
        """Return dict of current parameters"""
        return {
            "num_validations": self.num_validations,
            "validation_method": self.validation_method,
            "regression_type": self.regression_type,
            "models_source": self.models_source,
            "models": self.models,
            "datepart_method": self.datepart_method,
            "regression_model": self.regression_model,
        }


"""
from flaml import AutoML

# Initialize an AutoML instance
automl = AutoML()
# Specify automl goal and constraint
automl_settings = {
    "time_budget": 120,  # 28800  # in seconds
    "metric": 'mape',
    "task": 'regression',
    "log_file_name": "",
    "early_stop": True,
    "verbose": 1,
    "ensemble": False,
    "n_jobs": 8,
    "free_mem_ratio": 0.2,
    "estimator_list": ['lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'extra_tree'],
}
automl.fit(X_train=X, y_train=y, **automl_settings)
print(automl.model.estimator)
"""

# run models on latest data
