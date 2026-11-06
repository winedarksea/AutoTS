# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:28:57 2023

@author: Colin
"""

import datetime
import random
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.evaluator.auto_model import RandomTemplate
from autots.models.model_list import model_list_to_dict
from autots.models.forecast_correction import (
    ForecastCorrector,
    base_model_intervals,
    forecast_context,
)
from autots.models.mlensemble_features import (
    collect_base_forecasts,
    create_feature,
    default_regression_model,
    _sanitize_feature_matrix,
)
from autots.evaluator.validation import (
    validate_num_validations,
    generate_validation_indices,
)
from autots.models.sklearn import retrieve_regressor
from autots.tools.shaping import simple_train_test_split
from autots.tools.seasonal import random_datepart


class MLEnsemble(ModelObject):
    """Combine models by learning corrections across validation holdouts.

    Base models are refit on inner backtest holdouts to produce hindcasts. The
    default correction methods follow probabilistic bias correction (Guan et
    al. 2026): cheap per-series closed-form corrections that shrink to the
    simple base-model average when hindcasts are scarce (see
    autots.models.forecast_correction). 'regressor' is the original pooled
    ML regressor on stacked features.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        correction_method (str): 'pbc' (debias + persistence averaged), 'debias',
            'persistence', or 'regressor'
        bias_shape (str): 'constant' or 'linear_horizon' bias curve for debias
        bias_shrinkage (float): pseudo-fold count shrinking the bias toward zero
        ridge_lambda (float): persistence ridge strength (0 = least squares)
        interval_method (str): 'base_models' recentres the base models' average
            interval widths, 'inferred_normal' uses Point_to_Probability
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
        correction_method: str = "pbc",
        bias_shape: str = "linear_horizon",
        bias_shrinkage: float = 1.0,
        ridge_lambda: float = 1.0,
        interval_method: str = "base_models",
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
        self.correction_method = correction_method
        self.bias_shape = bias_shape
        self.bias_shrinkage = bias_shrinkage
        self.ridge_lambda = ridge_lambda
        self.interval_method = interval_method
        if regression_model is None and correction_method == "regressor":
            regression_model = default_regression_model()
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

    def _regressor_fold(self, df_train, df_test, regr_train, regr_test):
        res = create_feature(
            df_train,
            self.models,
            self.forecast_length,
            future_regressor_train=regr_train,
            future_regressor_forecast=regr_test,
            datepart_method=self.datepart_method,
            random_seed=self.random_seed,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )
        return res.reshape(res.shape[0], -1), df_test.to_numpy().reshape(-1)

    def _correction_fold(self, df_train, df_test, regr_train, regr_test):
        base_points, _, _, _ = collect_base_forecasts(
            df_train,
            self.models,
            self.forecast_length,
            future_regressor_train=regr_train,
            future_regressor_forecast=regr_test,
            random_seed=self.random_seed,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )
        context = forecast_context(df_train, df_test.index)
        return base_points, df_test.to_numpy(dtype=float), context

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
        use_regressor = self.correction_method == "regressor"
        run_fold = self._regressor_fold if use_regressor else self._correction_fold
        fold_results = []
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
                fold_results.append(
                    run_fold(df_train, df_test, regr_subset_t, regr_subset_f)
                )
            except Exception as e:
                if val < 1:
                    raise e
                else:
                    print(f"validation round {val} failed with {repr(e)}")

        if use_regressor:
            self._fit_regressor(fold_results)
        else:
            base_points, actuals, contexts = zip(*fold_results)
            self.corrector = ForecastCorrector(
                correction_method=self.correction_method,
                bias_shape=self.bias_shape,
                bias_shrinkage=self.bias_shrinkage,
                ridge_lambda=self.ridge_lambda,
            ).fit(np.stack(base_points), np.stack(actuals), list(contexts))
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def _fit_regressor(self, fold_results):
        X, y = zip(*fold_results)
        self.X = _sanitize_feature_matrix(np.concatenate(X, axis=1).T)
        self.y = np.concatenate(y, axis=0).astype(float)
        # Missing/inf actuals (fillna=None, unstable transforms) carry no label; drop rather than impute
        valid_label_mask = np.isfinite(self.y) & (np.abs(self.y) < 1e12)
        if not valid_label_mask.any():
            raise ValueError("MLEnsemble: no finite target values in validation holdouts")
        self.X = self.X[valid_label_mask]
        self.y = self.y[valid_label_mask]

        self.regr = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=self.verbose,
            verbose_bool=False,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            multioutput=False,
        )
        self.regr.fit(self.X, self.y)

    def _predict_regressor(self, future_regressor, test_index):
        res = create_feature(
            self.df,
            self.models,
            self.forecast_length,
            future_regressor_train=self.regressor_train,
            future_regressor_forecast=future_regressor,
            datepart_method=self.datepart_method,
            random_seed=self.random_seed,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )
        X = _sanitize_feature_matrix(res.reshape(res.shape[0], -1).T)
        forecast = pd.DataFrame(
            self.regr.predict(X).reshape(self.forecast_length, self.df.shape[1]),
            index=test_index,
            columns=self.df.columns,
        )
        upper_forecast, lower_forecast = Point_to_Probability(
            self.df,
            forecast,
            method='inferred_normal',
            prediction_interval=self.prediction_interval,
        )
        return forecast, upper_forecast, lower_forecast

    def _predict_corrected(self, future_regressor, test_index):
        base_points, base_upper, base_lower, _ = collect_base_forecasts(
            self.df,
            self.models,
            self.forecast_length,
            future_regressor_train=self.regressor_train,
            future_regressor_forecast=future_regressor,
            prediction_interval=self.prediction_interval,
            random_seed=self.random_seed,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )
        point = self.corrector.predict(
            base_points, forecast_context(self.df, test_index)
        )
        forecast = pd.DataFrame(point, index=test_index, columns=self.df.columns)
        if self.interval_method == "base_models":
            upper, lower = base_model_intervals(
                point, base_points, base_upper, base_lower
            )
            upper_forecast = pd.DataFrame(upper, index=test_index, columns=self.df.columns)
            lower_forecast = pd.DataFrame(lower, index=test_index, columns=self.df.columns)
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df,
                forecast,
                method=self.interval_method,
                prediction_interval=self.prediction_interval,
            )
        return forecast, upper_forecast, lower_forecast

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
        if forecast_length is None:
            forecast_length = self.forecast_length
        if forecast_length != self.forecast_length:
            raise ValueError("forecast_length must be changed in MLEnsemble init")
        test_index = self.create_forecast_index(forecast_length=self.forecast_length)
        if self.correction_method == "regressor":
            forecast, upper_forecast, lower_forecast = self._predict_regressor(
                future_regressor, test_index
            )
        else:
            forecast, upper_forecast, lower_forecast = self._predict_corrected(
                future_regressor, test_index
            )

        if just_point_forecast:
            return forecast
        predict_runtime = datetime.datetime.now() - predictStartTime
        return PredictionObject(
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

    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning"""
        if "regressor" in method:
            regression_type_choice = "User"
        else:
            regression_type_choice = None
        # RandomTemplate cycles its list in order when n < len(list), so sample names first
        model_names, model_weights = model_list_to_dict('fast')
        chosen_names = []
        while len(chosen_names) < 3:
            name = random.choices(model_names, model_weights)[0]
            if name not in chosen_names:
                chosen_names.append(name)
        models = RandomTemplate(
            n=3,
            model_list=chosen_names,
            transformer_max_depth=4,
            transformer_list='fast',
        )
        models = models[
            ['Model', 'ModelParameters', 'TransformationParameters']
        ].to_dict(orient='records')
        correction_method = random.choices(
            ["pbc", "debias", "persistence", "regressor"], [0.5, 0.2, 0.2, 0.1]
        )[0]
        if correction_method == "regressor":
            num_validations = random.choices([0, 1, 2], [0.5, 0.3, 0.2])[0]
        else:
            # more hindcasts is PBC's main lever, but each fold refits every base model
            num_validations = random.choices([0, 1, 2, 3], [0.15, 0.4, 0.3, 0.15])[0]

        return {
            "num_validations": num_validations,
            "validation_method": random.choices(
                ["backwards", "similarity", "seasonal"], [0.5, 0.3, 0.2]
            )[0],
            "regression_model": None,
            "models": models,
            "datepart_method": (
                random_datepart() if correction_method == "regressor" else None
            ),
            "regression_type": regression_type_choice,
            "models_source": 'random',
            "correction_method": correction_method,
            "bias_shape": random.choices(["linear_horizon", "constant"], [0.6, 0.4])[0],
            "bias_shrinkage": random.choices([0.5, 1.0, 2.0], [0.3, 0.4, 0.3])[0],
            "ridge_lambda": random.choices([0.3, 1.0, 3.0, 10.0], [0.2, 0.4, 0.25, 0.15])[0],
            "interval_method": random.choices(
                ["base_models", "inferred_normal"], [0.8, 0.2]
            )[0],
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
            "correction_method": self.correction_method,
            "bias_shape": self.bias_shape,
            "bias_shrinkage": self.bias_shrinkage,
            "ridge_lambda": self.ridge_lambda,
            "interval_method": self.interval_method,
        }
