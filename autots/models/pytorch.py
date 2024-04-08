# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:32:12 2022

@author: Colin
"""
import random
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.tools.seasonal import date_part, seasonal_int

try:
    # bewarned, pytorch-forecasting has useless error messages
    import torch

    try:
        import lightning.pytorch as pl  # 2.0 way
    except Exception:
        import pytorch_lightning as pl
    try:
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping  # 2.0
        from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
    except Exception:
        from pytorch_lightning.callbacks import EarlyStopping  # , LearningRateMonitor
    from pytorch_forecasting import (
        TimeSeriesDataSet,
        TemporalFusionTransformer,
        DeepAR,
        NHiTS,
        NBeats,
        DecoderMLP,
    )
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_forecasting.data import (
        EncoderNormalizer,
        GroupNormalizer,
        TorchNormalizer,
    )
    from torch.cuda import is_available

    pytorch_present = True
except Exception:
    pytorch_present = False


class PytorchForecasting(ModelObject):
    """pytorch-forecasting for the world's over-obsession of neural nets.

    This is generally going to require more data than most other models.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        model_kwargs (dict): passed to pytorch-forecasting model on creation (for those not already defined above)
        trainer_kwargs (dict): passed to pt lightning Trainer
        callbacks (list): pt lightning callbacks
        quantiles (list): [0.1, 0.5, 0.9] or similar for quantileloss models

    """

    def __init__(
        self,
        name: str = "PytorchForecasting",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2022,
        verbose: int = 0,
        n_jobs: int = 1,
        forecast_length: int = 90,
        max_epochs: int = 100,
        batch_size: int = 128,
        max_encoder_length: int = 12,
        learning_rate: float = 0.03,
        hidden_size: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
        datepart_method: str = "simple",
        add_target_scales: bool = False,
        lags: dict = {},
        target_normalizer: str = "EncoderNormalizer",
        model: str = "TemporalFusionTransformer",
        quantiles: list = [0.01, 0.1, 0.22, 0.36, 0.5, 0.64, 0.78, 0.9, 0.99],
        model_kwargs: dict = {},
        trainer_kwargs: dict = {},
        callbacks: list = None,
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
        self.forecast_length = forecast_length
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_encoder_length = max_encoder_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.datepart_method = datepart_method
        self.add_target_scales = add_target_scales
        self.lags = lags
        self.target_normalizer = target_normalizer
        self.model = model
        self.quantiles = quantiles

        self.model_kwargs = model_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.callbacks = callbacks
        self.range_idx_name = None
        self.date_part_cols = []

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed, wide style data
        """
        if not pytorch_present:
            raise ImportError(
                "pytorch, pytorch lighting, or pytorch-forecasting not present"
            )

        df = self.basic_profile(df)
        pl.seed_everything(self.random_seed)
        df_int = df.copy()

        # over-engineered approach just in case this name is used by a series
        if "range_datetime" not in df_int.columns:
            self.range_idx_name = "range_datetime"
        else:
            while self.range_idx_name is None:
                temp_name = ''.join(
                    random.choice('0123456789ABCDEF') for i in range(14)
                )
                if temp_name not in df_int.columns:
                    self.range_idx_name = temp_name

        # convert wide data back to long for this
        df_int[self.range_idx_name] = pd.RangeIndex(
            start=0, stop=df_int.shape[0], step=1
        )
        self._id_vars = [self.range_idx_name]
        data = df_int.melt(
            id_vars=self._id_vars, var_name='series_id', value_name='value'
        )
        if self.datepart_method is not None:
            dt_merge = df_int[self.range_idx_name].drop_duplicates()
            date_parts = date_part(
                dt_merge.index,
                method=self.datepart_method,
                polynomial_degree=None,
            )
            dt_merge = dt_merge.to_frame().merge(
                date_parts, left_index=True, right_index=True
            )
            self.date_part_cols = date_parts.columns.tolist()
            data = data.merge(dt_merge.reset_index(drop=False), on=self.range_idx_name)

        # define dataset
        val_cutoff = data[self.range_idx_name].max() - self.forecast_length
        self.actual_cutoff = data[self.range_idx_name].max()

        if self.target_normalizer == "TorchNormalizer":
            encoder = TorchNormalizer()
        elif self.target_normalizer == "GroupNormalizer":
            encoder = GroupNormalizer()
        elif self.target_normalizer == "EncoderNormalizer":
            encoder = EncoderNormalizer()
        else:
            encoder = None

        training = TimeSeriesDataSet(
            data,
            time_idx=self.range_idx_name,
            target="value",
            time_varying_unknown_reals=["value"],  # needed for DeepAR
            # weight="weight",
            group_ids=["series_id"],
            # static_categoricals=["series_id"],  # recommeded for DeepAR
            max_encoder_length=self.max_encoder_length,
            min_encoder_length=(
                self.max_encoder_length if self.max_encoder_length < 90 else 7
            ),
            max_prediction_length=self.forecast_length,
            target_normalizer=encoder,
            # static_categoricals=[ ... ],
            # static_reals=[ ... ],
            # time_varying_known_categoricals=self.date_part_cols,
            time_varying_known_reals=self.date_part_cols,
            # time_varying_unknown_categoricals=[ ... ],
            # time_varying_unknown_reals=[ ... ],
            # allow_missing_timesteps=True,
            # scalers={0: RobustScaler()},
            add_target_scales=self.add_target_scales,
            # min_encoder_length=3,
            lags=self.lags,
            add_relative_time_idx=(
                False if self.model in ["NHiTS", "DecoderMLP", "NBeats"] else True
            ),
        )

        # create validation and training dataset
        validation = TimeSeriesDataSet.from_dataset(
            training,
            data,
            min_prediction_idx=val_cutoff + 1,
            predict=True,
            stop_randomization=True,
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=self.batch_size  # , num_workers=self.n_jobs
        )
        train_dataloader = TimeSeriesDataSet.from_parameters(
            training.get_parameters(),
            data[data[self.range_idx_name] <= val_cutoff],
        ).to_dataloader(
            train=True, batch_size=self.batch_size  # , num_workers=self.n_jobs,
        )

        # define trainer with early stopping
        if self.callbacks is None:
            self.callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-5,
                    patience=2,
                    verbose=False,
                    mode="min",
                )
            ]
        try:
            self.callbacks.append(
                ModelCheckpoint(
                    save_last=False,
                    save_top_k=0,
                    every_n_train_steps=0,
                    every_n_epochs=0,
                )
            )
        except Exception as e:
            print(repr(e))

        # lr_logger = LearningRateMonitor()
        if is_available() and "accelerator" not in self.trainer_kwargs.keys():
            self.trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                # num_processes=self.n_jobs,  # don't think this will usually be used
                # gradient_clip_val=0.1,
                # limit_train_batches=30,
                callbacks=self.callbacks,  # lr_logger,
                logger=False,
                accelerator='gpu',
                gradient_clip_val=0.05,  # necessary to be less than 0.1 to prevent a strange bug
                devices=1,
                log_every_n_steps=0,
                **self.trainer_kwargs,
            )
        else:
            self.trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                # num_processes=self.n_jobs,  # don't think this will usually be used
                # gradient_clip_val=0.1,
                # limit_train_batches=30,
                # enable_checkpointing
                callbacks=self.callbacks,  # lr_logger,
                gradient_clip_val=0.05,
                logger=False,
                log_every_n_steps=0,
                **self.trainer_kwargs,
            )

        # create the model
        if self.model in ['TFT', "TemporalFusionTransformer"]:
            self.tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                lstm_layers=self.n_layers,
                # attention_head_size=1,
                # hidden_continuous_size=16,
                output_size=len(self.quantiles),  # must be 1 for non-quantile losses
                loss=QuantileLoss(quantiles=self.quantiles),
                # reduce_on_plateau_patience=4,
                **self.model_kwargs,
            )
        elif self.model == "DeepAR":
            self.tft = DeepAR.from_dataset(
                training,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                log_val_interval=0,
                rnn_layers=self.n_layers,  # said to be important
                # loss=MultivariateNormalDistributionLoss(rank=30),
                **self.model_kwargs,
            )
        elif self.model == "NHiTS":
            self.tft = NHiTS.from_dataset(
                training,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                # n_layers=self.n_layers,  # temporarily broken in 1.0.0, maybe uncomment later
                # weight_decay=1e-2,
                # loss=MQF2DistributionLoss(prediction_length=max_prediction_length),
                # backcast_loss_ratio=0.0,
                output_size=len(self.quantiles),  # must be 1 for non-quantile losses
                loss=QuantileLoss(quantiles=self.quantiles),
                context_length=self.forecast_length * 2,
                **self.model_kwargs,
            )
        elif self.model == "DecoderMLP":
            self.tft = DecoderMLP.from_dataset(
                training,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                # weight_decay=1e-2,
                # loss=MQF2DistributionLoss(prediction_length=max_prediction_length),
                n_hidden_layers=self.n_layers,  # said to be important
                output_size=len(self.quantiles),  # must be 1 for non-quantile losses
                loss=QuantileLoss(quantiles=self.quantiles),
                **self.model_kwargs,
            )
        else:
            self.model = "NBeats"
            self.tft = NBeats.from_dataset(
                training,
                learning_rate=self.learning_rate,
                dropout=self.dropout,
                # weight_decay=1e-2,
                # loss=MQF2DistributionLoss(prediction_length=max_prediction_length),
                # backcast_loss_ratio=0.0,s
                context_length=self.max_encoder_length,
                **self.model_kwargs,
            )

        if self.verbose > 1:
            print(
                f"Number of parameters in pytorch network: {self.tft.size()/1e3:.1f}k"
            )

        # fit the model
        self.trainer.fit(
            self.tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )
        self.encoder_tail = df_int.tail(self.max_encoder_length)

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
                must be equal or lesser to that specified in init
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        if forecast_length is None:
            forecast_length = self.forecast_length

        predictStartTime = datetime.datetime.now()

        test_index = self.create_forecast_index(forecast_length=forecast_length)

        encoder_data = self.encoder_tail.merge(
            pd.DataFrame(
                {
                    self.range_idx_name: pd.RangeIndex(
                        self.actual_cutoff + 1,
                        self.actual_cutoff + forecast_length + 1,
                        step=1,
                    )
                },
                index=test_index,
            ),
            on=self.range_idx_name,
            how='outer',
        ).ffill()
        encoder_data.index = pd.DatetimeIndex(
            np.concatenate([self.encoder_tail.index, test_index], axis=None)
        )
        # combine encoder and decoder data
        self.x_predict = encoder_data.melt(
            id_vars=self._id_vars, var_name='series_id', value_name='value'
        )
        if self.datepart_method is not None:
            dt_merge = encoder_data[self.range_idx_name].drop_duplicates()
            date_parts = date_part(
                dt_merge.index,
                method=self.datepart_method,
                polynomial_degree=None,
            )
            dt_merge = dt_merge.to_frame().merge(
                date_parts, left_index=True, right_index=True
            )
            self.x_predict = self.x_predict.merge(
                dt_merge.reset_index(drop=False), on=self.range_idx_name
            )

        colz = self.column_names
        try:
            predictions_init = self.tft.predict(
                self.x_predict, mode='prediction', return_index=True
            )
            self.predictions = predictions_init[0]
            if not isinstance(self.predictions, torch.Tensor):
                for x in predictions_init:
                    if isinstance(x, torch.Tensor):
                        self.predictions = x
            for x in predictions_init:
                if isinstance(x, pd.DataFrame):
                    if "series_id" in x.columns:
                        colz = x["series_id"]
        except Exception:
            self.predictions, pred_idx = self.tft.predict(
                self.x_predict, mode='prediction', return_index=True
            )
            colz = pred_idx['series_id']
        pred_num = self.predictions.numpy(force=True).T
        predictions_df = pd.DataFrame(pred_num, columns=colz, index=test_index)[
            self.column_names
        ]

        if just_point_forecast:
            return predictions_df
        self.result_windows = self.tft.predict(self.x_predict, mode="quantiles")
        self.result_windows = self.result_windows.transpose(0, 2)
        if self.result_windows.shape[2] > 1:
            c_int = (1.0 - self.prediction_interval) / 2
            # predictions_df = result.quantile(0.5, axis=2).numpy().T
            # taking a quantile of the quantiles given!
            lower_df = pd.DataFrame(
                self.result_windows.quantile(c_int, axis=0).numpy(force=True),
                index=test_index,
                columns=colz,
            )[self.column_names]
            upper_df = pd.DataFrame(
                self.result_windows.quantile(1.0 - c_int, axis=0).numpy(force=True),
                index=test_index,
                columns=colz,
            )[self.column_names]
        else:
            # predictions_df = result.numpy()[:, :, 0].T
            upper_df, lower_df = Point_to_Probability(
                self.encoder_tail,
                predictions_df,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )
        self.result_windows = self.result_windows.numpy(force=True)

        predict_runtime = datetime.datetime.now() - predictStartTime
        prediction = PredictionObject(
            model_name=self.name,
            forecast_length=forecast_length,
            forecast_index=test_index,
            forecast_columns=predictions_df.columns,
            lower_forecast=lower_df,
            forecast=predictions_df,
            upper_forecast=upper_df,
            prediction_interval=self.prediction_interval,
            predict_runtime=predict_runtime,
            fit_runtime=self.fit_runtime,
            model_parameters=self.get_params(),
        )

        return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        parameter_dict = {
            "model": random.choices(
                [
                    "TemporalFusionTransformer",
                    "DecoderMLP",
                    'DeepAR',
                    "NHiTS",
                    "NBeats",
                ],
                [0.3, 0.2, 0.3, 0.2, 0.2],
            )[0],
            "max_encoder_length": random.choice([7, 12, 24, 28, 60, 96, 364]),
            "datepart_method": random.choices(
                [None, "recurring", "simple", "expanded", "simple_2"],
                [0.8, 0.4, 0.3, 0.3, 0.3],
            )[0],
            "add_target_scales": random.choice([True, False]),
            "lags": random.choices(
                [
                    {},
                    {"value": [seasonal_int(very_small=True)]},
                    {"value": [1, seasonal_int(very_small=True)]},
                    {"value": [1]},
                ],
                [0.9, 0.05, 0.05, 0.5],
            )[
                0
            ],  # {"value": [1, 7]}
            "target_normalizer": random.choices(
                ["EncoderNormalizer", "TorchNormalizer", "GroupNormalizer"],
                [0.5, 0.25, 0.25],
            )[0],
            "batch_size": random.choice([32, 64, 128]),
            "max_epochs": random.choice([30, 50, 100]),
            "learning_rate": random.choice([0.1, 3e-2, 0.001, 0.01]),
            "hidden_size": random.choices(
                [8, 16, 32, 30, 64, 128, 256], [0.1, 0.1, 0.25, 0.2, 0.2, 0.1, 0.05]
            )[0],
            "n_layers": random.choices([1, 2, 3, 4], [0.14, 0.4, 0.3, 0.16])[0],
            "dropout": random.choice([0.0, 0.1, 0.2, 0.05]),
            "model_kwargs": {},
            "trainer_kwargs": {},
        }
        if parameter_dict['model'] == "DeepAR":
            parameter_dict["target_normalizer"] = "EncoderNormalizer"
        elif parameter_dict['model'] == "NHiTS":
            parameter_dict["add_target_scales"] = False
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        return {
            "model": self.model,
            "max_encoder_length": self.max_encoder_length,
            "datepart_method": self.datepart_method,
            "add_target_scales": self.add_target_scales,
            "lags": self.lags,
            "target_normalizer": self.target_normalizer,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "model_kwargs": self.model_kwargs,
            "trainer_kwargs": self.trainer_kwargs,
        }
