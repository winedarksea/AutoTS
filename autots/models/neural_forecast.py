"""
Nixtla's NeuralForecast. Be warned, as of writing, their package has commercial restrictions.
"""

import logging
import random
import datetime
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability


class NeuralForecast(ModelObject):
    """See NeuralForecast documentation for details.


    temp['ModelParameters'].str.extract('model": "([a-zA-Z]+)')

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type: str = None,
        model (str or object): string aliases or passed to the models arg of NeuralForecast
        model_args (dict): for all model args that aren't in default list, run get_new_params for default
    """

    def __init__(
        self,
        name: str = "NeuralForecast",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2023,
        verbose: int = 0,
        forecast_length: int = 28,
        regression_type: str = None,
        n_jobs: int = 1,
        model="LSTM",
        loss="MQLoss",
        input_size="2ForecastLength",
        max_steps=100,
        learning_rate=0.001,
        early_stop_patience_steps=-1,
        activation='ReLU',
        scaler_type='robust',
        model_args={},
        point_quantile=None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            regression_type=regression_type,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.model = model
        self.loss = loss
        self.input_size = input_size
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.early_stop_patience_steps = early_stop_patience_steps
        self.activation = activation
        self.scaler_type = scaler_type
        self.model_args = model_args
        self.point_quantile = point_quantile
        self.forecast_length = forecast_length
        self.df_train = None
        self.static_regressor = None

    def fit(
        self,
        df,
        future_regressor=None,
        static_regressor=None,
        regressor_per_series=None,
    ):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        self.basic_profile(df)
        static_cols = None
        if self.regression_type in ["User", "user", True]:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but no future_regressor passed"
                )
            self.static_regressor = static_regressor
            if isinstance(self.static_regressor, pd.DataFrame):
                static_cols = static_regressor.columns.tolist()
            if regressor_per_series is not None:
                if not isinstance(regressor_per_series, dict):
                    raise ValueError("regressor_per_series in incorrect format")

        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import (
            SMAPE,
            MQLoss,
            DistributionLoss,
            MAE,
            HuberLoss,
        )
        from neuralforecast.models import (
            TFT,
            LSTM,
            NHITS,
            MLP,
            NBEATS,
            TimesNet,
            PatchTST,
            DeepAR,
            TiDE,
            FEDformer,
        )

        prediction_interval = self.prediction_interval
        forecast_length = self.forecast_length
        freq = self.frequency

        # Split data and declare panel dataset
        if isinstance(prediction_interval, list):
            levels = prediction_interval
        elif isinstance(prediction_interval, dict):
            levels = list(prediction_interval.values())
        else:
            levels = [int(prediction_interval * 100)]

        logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
        loss = self.loss
        if loss == "MQLoss":
            if self.point_quantile is None:
                loss = MQLoss(level=levels)
            else:
                div = (1 - self.prediction_interval) / 2
                quantiles = [div, 1 - div, self.point_quantile]
                loss = MQLoss(quantiles=quantiles)
        elif loss == "HuberMQLoss":
            from neuralforecast.losses.pytorch import HuberMQLoss

            if self.point_quantile is None:
                loss = HuberMQLoss(level=levels)
            else:
                div = (1 - self.prediction_interval) / 2
                quantiles = [div, 1 - div, self.point_quantile]
                loss = HuberMQLoss(quantiles=quantiles)
        elif loss == "Poisson":
            loss = DistributionLoss(
                distribution='Poisson', level=levels, return_params=False
            )
        elif loss == "Bernoulli":
            loss = DistributionLoss(
                distribution='Bernoulli', level=levels, return_params=False
            )
        elif loss == "StudentT":
            loss = DistributionLoss(
                distribution='StudentT', level=levels, return_params=False
            )
        elif loss == "NegativeBinomial":
            loss = DistributionLoss(
                distribution='NegativeBinomial',
                level=levels,
                total_count=3,
                return_params=False,
            )
        elif loss == "Normal":
            loss = DistributionLoss(
                distribution='Normal', level=levels, return_params=False
            )
        elif loss == "Tweedie":
            loss = DistributionLoss(
                distribution='Tweedie', level=levels, return_params=False, rho=1.5
            )
        elif loss == "MAE":
            self.df_train = df
            loss = MAE()
        elif loss == "HuberLoss":
            self.df_train = df
            loss = HuberLoss()
        elif loss == "SMAPE":
            self.df_train = df
            loss = SMAPE()
        elif isinstance(loss, str):
            raise ValueError(f"loss not recognized: {loss}")
        else:
            # allow custom input
            pass

        str_input = str(self.input_size).lower()
        if "forecastlength" in str_input:
            input_size = forecast_length * int(
                ''.join([x for x in str_input if x.isdigit()])
            )
        else:
            input_size = int(self.input_size)
        self.base_args = {
            "h": forecast_length,
            "input_size": input_size,
            "max_steps": self.max_steps,
            # "num_workers_loader": self.n_jobs,
            "random_seed": self.random_seed,
            "learning_rate": self.learning_rate,
            "loss": loss,
            'scaler_type': self.scaler_type,
            # trying to suppress the stupid file logging lightning does
            "logger": False,
            "log_every_n_steps": 0,
        }
        models = self.model
        model_args = self.model_args
        if self.regression_type in ['User', 'user', True]:
            regr_cols = future_regressor.columns.tolist()
            if regressor_per_series is not None:
                regr_cols + next(iter(regressor_per_series.values())).columns.tolist()
            self.base_args["futr_exog_list"] = regr_cols
            self.base_args['stat_exog_list'] = static_cols

        if isinstance(models, list):
            # User inputs classes directly
            pass
        elif models == 'LSTM':
            models = [LSTM(**{**self.base_args, **model_args})]
        elif models == "NHITS":
            models = [NHITS(**{**self.base_args, **model_args})]
        elif models == "NBEATS":
            models = [NBEATS(**{**self.base_args, **model_args})]
        elif models == "MLP":
            models = [MLP(**{**self.base_args, **model_args})]
        elif models == "TimesNet":
            models = [TimesNet(**{**self.base_args, **model_args})]
        elif models == "TFT":
            models = [TFT(**{**self.base_args, **model_args})]
        elif models == "PatchTST":
            models = [PatchTST(**{**self.base_args, **model_args})]
        elif models == "DeepAR":
            models = [DeepAR(**{**self.base_args, **model_args})]
        elif models == "TiDE":
            models = [TiDE(**{**self.base_args, **model_args})]
        elif models == "FEDformer":
            models = [FEDformer(**{**self.base_args, **model_args})]
        else:
            raise ValueError(f"models not recognized: {models}")

        # model params
        # requires pandas >= 1.5
        silly_format = df.reset_index(names='ds').melt(
            id_vars='ds', value_name='y', var_name='unique_id'
        )
        if self.regression_type in ['User', 'user', True]:
            silly_format = silly_format.merge(
                future_regressor, left_on='ds', right_index=True
            )
            if regressor_per_series is not None:
                full_df = []
                for key, value in regressor_per_series.items():
                    local_copy = value.copy().reindex(df.index)
                    local_copy.index.name = 'ds'
                    local_copy = local_copy.reset_index()
                    local_copy['unique_id'] = str(key)
                    full_df.append(local_copy)
                silly_format = silly_format.merge(
                    pd.concat(full_df), on=['unique_id', 'ds'], how='left'
                ).fillna(0)
        self.nf = NeuralForecast(models=models, freq=freq)
        if self.static_regressor is None:
            self.nf.fit(df=silly_format)
        else:
            another_silly_format = self.static_regressor
            another_silly_format.index.name = "unique_id"
            self.nf.fit(
                df=silly_format, static_df=another_silly_format.reset_index(drop=False)
            )
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length=None,
        future_regressor=None,
        just_point_forecast=False,
        regressor_per_series=None,
    ):
        predictStartTime = datetime.datetime.now()
        if self.regression_type in ['User', 'user', True]:
            index = self.create_forecast_index(forecast_length=self.forecast_length)
            futr_df = pd.concat(
                [
                    pd.Series(col, index=index, name='unique_id')
                    for col in self.column_names
                ]
            )
            futr_df = futr_df.to_frame().merge(
                future_regressor, left_index=True, right_index=True
            )
            futr_df = futr_df.reset_index(names='ds')
            if regressor_per_series is not None:
                full_df = []
                for key, value in regressor_per_series.items():
                    local_copy = value.copy().reindex(index)
                    local_copy.index.name = 'ds'
                    local_copy = local_copy.reset_index()
                    local_copy['unique_id'] = str(key)
                    full_df.append(local_copy)
                futr_df = futr_df.merge(
                    pd.concat(full_df), on=['unique_id', 'ds'], how='left'
                ).fillna(0)
            self.futr_df = futr_df
            long_forecast = self.nf.predict(futr_df=futr_df)
        else:
            long_forecast = self.nf.predict()
        # self.long_forecast = long_forecast
        target_col = [x for x in long_forecast.columns.tolist() if "median" in str(x)]
        if len(target_col) < 1:
            target_col = long_forecast.columns[-1]
        else:
            target_col = target_col[0]
        if self.point_quantile is not None:
            # print(long_forecast.columns)
            target_col = long_forecast.columns[-1]
        forecast = long_forecast.reset_index().pivot_table(
            index='ds', columns='unique_id', values=target_col
        )[self.column_names]

        if just_point_forecast:
            return forecast
        if self.df_train is not None:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )
        else:
            target_col = [x for x in long_forecast.columns if "hi-" in x][0]
            # print(f"upper target col: {target_col}")
            upper_forecast = long_forecast.reset_index().pivot_table(
                index='ds', columns='unique_id', values=target_col
            )[self.column_names]
            target_col = [x for x in long_forecast.columns if "lo-" in x][0]
            # print(f"lower target col {target_col}")
            lower_forecast = long_forecast.reset_index().pivot_table(
                index='ds', columns='unique_id', values=target_col
            )[self.column_names]

        predict_runtime = datetime.datetime.now() - predictStartTime
        prediction = PredictionObject(
            model_name=self.name,
            forecast_length=self.forecast_length,
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
        model_list = [
            'DeepAR',
            'MLP',
            "LSTM",
            "PatchTST",
            "NHITS",
            "TFT",
            "TimesNet",
            "TiDE",
            "FEDformer",
        ]
        if method in model_list:
            models = method
        else:
            models = random.choices(
                model_list, [0.05, 0.4, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
            )[0]
        if "regressor" in method:
            regression_type_choice = "User"
        else:
            regression_type_choice = random.choices([None, "User"], weights=[0.8, 0.2])[
                0
            ]
        if "deep" in method:
            max_steps = random.choices(
                [40, 80, 100, 1000, 5000, 10000, 50000],
                [0.2, 0.2, 0.2, 0.1, 0.05, 0.05, 0.01],
            )[0]
        else:
            max_steps = random.choices(
                [40, 80, 100, 1000, 5000],
                [0.2, 0.2, 0.2, 0.05, 0.03],
            )[0]
        activation = random.choices(
            ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'],
            [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )[0]
        loss = random.choices(
            [
                'MQLoss',
                'HuberMQLoss',
                'Poisson',
                'Bernoulli',
                'NegativeBinomial',
                'Normal',
                'Tweedie',
                'HuberLoss',
                "MAE",
                "SMAPE",
                "StudentT",
            ],
            [0.5, 0.1, 0.1, 0.01, 0.1, 0.1, 0.01, 0.1, 0.1, 0.1, 0.01],
        )[0]
        point_quantile = None
        if loss in ["MQLoss", "HuberMQLoss"]:
            point_quantile = random.choices(
                [None, 0.35, 0.45, 0.55, 0.65, 0.7], [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
            )[0]
        if models == "TFT":
            model_args = {
                "n_head": random.choice([2, 4]),
                "dropout": random.choice([0.05, 0.2, 0.3, 0.6]),
                "batch_size": random.choice([32, 64, 128, 256]),
                "hidden_size": random.choice([4, 64, 128, 256]),
                "windows_batch_size": random.choice([128, 256, 512, 1024]),
            }
        elif models == "PatchTST":
            model_args = {
                "n_heads": random.choice([2, 4, 16, 32]),
                "dropout": random.choice([0.05, 0.2, 0.3, 0.6]),
                "batch_size": random.choice([32, 64, 128, 256]),
                "hidden_size": random.choice([4, 64, 128, 256]),
                "linear_hidden_size": random.choice([4, 64, 128, 256, 512, 1024]),
                "windows_batch_size": random.choice([128, 256, 512, 1024]),
                'encoder_layers': random.choices([1, 2, 3, 4], [0.2, 0.4, 0.6, 0.1])[0],
            }
        elif models == "NHITS":
            model_args = {
                "input_size": random.choice([28, 28 * 2, 28 * 3, 28 * 5]),
                "n_blocks": 5 * [1],
                "mlp_units": 5 * [[512, 512]],
                "n_pool_kernel_size": random.choice(
                    [5 * [1], 5 * [2], 5 * [4], [8, 4, 2, 1, 1]]
                ),
                "n_freq_downsample": random.choice([[8, 4, 2, 1, 1], [1, 1, 1, 1, 1]]),
                "batch_size": random.choice([32, 64, 128, 256]),
                "windows_batch_size": random.choice([128, 256, 512, 1024]),
                "activation": activation,
            }
        elif models == "MLP":
            model_args = {
                'num_layers': random.choices([1, 2, 3, 4], [0.6, 0.4, 0.2, 0.1])[0],
                'hidden_size': random.choice([1024, 512, 2048, 256, 2560, 3072, 4096]),
            }
        elif models == "TiDE":
            model_args = {
                # borrowed mostly from the defaults used for the paper results
                "layernorm": random.choices([True, False], [0.2, 0.8])[0],
                "dropout": random.choices([0.0, 0.3, 0.5, 0.7], [0.3, 0.3, 0.3, 0.05])[
                    0
                ],
                "batch_size": random.choices(
                    [1024, 512, 257, 32], [0.05, 0.4, 0.1, 0.1]
                )[0],
                "hidden_size": random.choices(
                    [1024, 512, 256, 64], [0.1, 0.2, 0.4, 0.1]
                )[0],
                "num_encoder_layers": random.choices([1, 2, 3], [0.85, 0.1, 0.02])[0],
                "num_decoder_layers": random.choices([1, 2, 3], [0.85, 0.1, 0.02])[0],
                "decoder_output_dim": random.choices(
                    [16, 8, 4, 32], [0.4, 0.4, 0.4, 0.1]
                )[0],
                "temporal_decoder_dim": random.choices(
                    [128, 64, 32, 16], [0.1, 0.8, 0.1, 0.1]
                )[0],
            }
        else:
            model_args = {}

        return {
            'model': models,
            'scaler_type': random.choices(
                ["identity", 'robust', 'minmax', 'standard'], [0.5, 0.5, 0.2, 0.2]
            )[0],
            'loss': loss,
            'learning_rate': random.choices(
                [0.001, 0.1, 0.01, 0.0003, 0.00001, 0.000001],
                [0.4, 0.1, 0.1, 0.1, 0.1, 0.02],
            )[0],
            "max_steps": max_steps,
            'input_size': random.choices(
                [10, 28, "2ForecastLength", "3ForecastLength"], [0.2, 0.2, 0.2, 0.2]
            )[0],
            # "early_stop_patience_steps": random.choice([1, 3, 5]),
            "point_quantile": point_quantile,
            "model_args": model_args,
            'regression_type': regression_type_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            # NOTE: `models` (plural) conflicts with AutoTS ensemble setup
            'model': self.model,
            'scaler_type': self.scaler_type,
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            "max_steps": self.max_steps,
            'input_size': self.input_size,
            'point_quantile': self.point_quantile,
            "model_args": self.model_args,
            'regression_type': self.regression_type,
        }


if False:
    # from autots.models.neural_forecast import NeuralForecast
    from autots import load_daily, create_regressor, infer_frequency

    df = load_daily(long=False)
    forecast_length = 28
    frequency = infer_frequency(df)
    regr_train, regr_fcst = create_regressor(
        df,
        forecast_length=forecast_length,
        frequency=frequency,
        drop_most_recent=0,
        scale=True,
        summarize="auto",
        backfill="bfill",
        fill_na="pchip",
        holiday_countries=["US"],
        datepart_method="recurring",
        preprocessing_params={
            "fillna": None,
            "transformations": {"0": "LocalLinearTrend"},
            "transformation_params": {
                "0": {
                    'rolling_window': 30,
                    'n_tails': 0.1,
                    'n_future': 0.2,
                    'method': 'mean',
                    'macro_micro': True,
                },
            },
        },
    )

    params = NeuralForecast().get_new_params()
    print(params)
    mod = NeuralForecast(forecast_length=forecast_length, frequency=frequency, **params)
    mod.fit(df, future_regressor=regr_train)
    prediction = mod.predict(future_regressor=regr_fcst)
    prediction.plot_grid(df)
