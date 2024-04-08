"""
GluonTS

Best neuralnet models currently available, released by Amazon, scale well.
Except it is really the only thing I use that runs mxnet, and it takes a while to train these guys...
And MXNet is now sorta-maybe-deprecated? Which is sad because it had excellent CPU-based training speed.

Note that there are routinely package version issues with this and its dependencies.
Stability is not the strong suit of GluonTS.
"""

import logging
import random
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject

try:
    from gluonts.dataset.common import ListDataset

    # GluonTS looooves to move import locations...
    try:
        from gluonts.dataset.field_names import FieldName  # new way
    except Exception:
        from gluonts.transform import FieldName  # old way (0.3.3 and older)
    try:
        try:  # new way, but only with mxnet
            from gluonts.mx.trainer import Trainer
        except Exception:  # old way < 0.5.x
            from gluonts.trainer import Trainer
    except Exception:
        pass
except Exception:  # except ImportError
    _has_gluonts = False
else:
    _has_gluonts = True


class GluonTS(ModelObject):
    """GluonTS based on mxnet.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): Not yet implemented

        gluon_model (str): Model Structure to Use - ['DeepAR', 'NPTS', 'DeepState', 'WaveNet','DeepFactor', 'Transformer','SFF', 'MQCNN', 'DeepVAR', 'GPVAR', 'NBEATS']
        epochs (int): Number of neural network training epochs. Higher generally results in better, then over fit.
        learning_rate (float): Neural net training parameter
        context_length (str): int window, '2ForecastLength', or 'nForecastLength'
        forecast_length (int): Length to forecast. Unlike in other methods, this must be provided *before* fitting model

    """

    def __init__(
        self,
        name: str = "GluonTS",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        gluon_model: str = 'DeepAR',
        epochs: int = 20,
        learning_rate: float = 0.001,
        context_length=10,
        forecast_length: int = 14,
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
        )
        self.gluon_model = gluon_model
        if self.gluon_model in ['NPTS', 'Rotbaum']:
            self.epochs = 20
            self.learning_rate = 0.001
        else:
            self.epochs = epochs
            self.learning_rate = learning_rate
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.multivariate_mods = ['DeepVAR', 'GPVAR']

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        if not _has_gluonts:
            raise ImportError(
                "GluonTS installation is incompatible with AutoTS. The numpy version is sometimes the issue, try 1.23.1 {as of 06-2023}"
            )

        try:
            from mxnet.random import seed as mxnet_seed

            mxnet_seed(self.random_seed)
        except Exception:
            pass

        gluon_freq = str(self.frequency).split('-')[0]
        self.train_index = df.columns
        self.train_columns = df.index

        if gluon_freq in ["MS", "1MS"]:
            gluon_freq = "M"

        if int(self.verbose) > 1:
            print(f"Gluon Frequency is {gluon_freq}")
        if int(self.verbose) < 1:
            try:
                logging.getLogger().disabled = True
                logging.getLogger("mxnet").addFilter(lambda record: False)
            except Exception:
                pass

        if str(self.context_length).replace('.', '').isdigit():
            self.gluon_context_length = int(float(self.context_length))
        elif 'forecastlength' in str(self.context_length).lower():
            len_int = int([x for x in str(self.context_length) if x.isdigit()][0])
            self.gluon_context_length = int(len_int * self.forecast_length)
        else:
            self.gluon_context_length = 20
            self.context_length = '20'
        self.ts_metadata = ts_metadata = {
            'num_series': len(self.train_index),
            'freq': gluon_freq,
            'start_ts': df.index[0],
            'gluon_start': [
                self.train_columns[0] for _ in range(len(self.train_index))
            ],
            'context_length': self.gluon_context_length,
            'forecast_length': self.forecast_length,
        }
        self.fit_data(df, future_regressor=future_regressor)
        npts_flag = False

        pytorch_models = ['PatchTST', 'DeepAR']  # those supporting
        if self.gluon_model in pytorch_models:
            pass
        """
        # this attempts to stop model checkpoing saving spam that is typical of lightning
            try:
                try:
                    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
                except Exception:
                    from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
    
                callbacks = []
                callbacks.append(
                    ModelCheckpoint(
                        save_last=False,
                        save_top_k=0,
                        every_n_train_steps=0,
                        every_n_epochs=0,
                    )
                )
            except Exception as e:
                callbacks = []
                print(repr(e))
        """

        if self.gluon_model == 'DeepAR':
            try:
                try:
                    from gluonts.mx import DeepAREstimator
                except Exception:
                    from gluonts.model.deepar import DeepAREstimator
                estimator = DeepAREstimator(
                    freq=ts_metadata['freq'],
                    context_length=ts_metadata['context_length'],
                    prediction_length=ts_metadata['forecast_length'],
                    trainer=Trainer(
                        epochs=self.epochs, learning_rate=self.learning_rate
                    ),
                )
            except Exception:
                from gluonts.torch import DeepAREstimator

                estimator = DeepAREstimator(
                    freq=ts_metadata['freq'],
                    context_length=ts_metadata['context_length'],
                    prediction_length=ts_metadata['forecast_length'],
                    trainer_kwargs={
                        'logger': False,
                        'log_every_n_steps': 0,
                    },  # , 'callbacks': callbacks
                )

        elif self.gluon_model == 'NPTS':
            try:
                from gluonts.model.npts import NPTSPredictor

                estimator = NPTSPredictor(
                    freq=ts_metadata['freq'],
                    context_length=ts_metadata['context_length'],
                    prediction_length=ts_metadata['forecast_length'],
                )
                npts_flag = True
            except Exception:
                from gluonts.model.npts import NPTSEstimator

                estimator = NPTSEstimator(
                    freq=ts_metadata['freq'],
                    context_length=ts_metadata['context_length'],
                    prediction_length=ts_metadata['forecast_length'],
                )

        elif self.gluon_model == 'MQCNN':
            try:
                from gluonts.mx import MQCNNEstimator
            except Exception:
                from gluonts.model.seq2seq import MQCNNEstimator

            estimator = MQCNNEstimator(
                freq=ts_metadata['freq'],
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )

        elif self.gluon_model == 'SFF':
            try:
                from gluonts.mx import SimpleFeedForwardEstimator
            except Exception:
                from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator

            estimator = SimpleFeedForwardEstimator(
                prediction_length=ts_metadata['forecast_length'],
                context_length=ts_metadata['context_length'],
                # freq=ts_metadata['freq'],
                trainer=Trainer(
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    hybridize=False,
                    num_batches_per_epoch=100,
                ),
            )

        elif self.gluon_model == 'Transformer':
            try:
                from gluonts.mx import TransformerEstimator
            except Exception:
                from gluonts.model.transformer import TransformerEstimator

            estimator = TransformerEstimator(
                prediction_length=ts_metadata['forecast_length'],
                context_length=ts_metadata['context_length'],
                freq=ts_metadata['freq'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )

        elif self.gluon_model == 'DeepState':
            try:
                from gluonts.mx import DeepStateEstimator
            except Exception:
                from gluonts.model.deepstate import DeepStateEstimator

            estimator = DeepStateEstimator(
                prediction_length=ts_metadata['forecast_length'],
                past_length=ts_metadata['context_length'],
                freq=ts_metadata['freq'],
                use_feat_static_cat=False,
                cardinality=[1],
                trainer=Trainer(
                    ctx='cpu', epochs=self.epochs, learning_rate=self.learning_rate
                ),
            )

        elif self.gluon_model == 'DeepFactor':
            try:
                from gluonts.mx import DeepFactorEstimator
            except Exception:
                from gluonts.model.deep_factor import DeepFactorEstimator

            estimator = DeepFactorEstimator(
                freq=ts_metadata['freq'],
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )

        elif self.gluon_model == 'WaveNet':
            # Usually needs more epochs/training iterations than other models do
            try:
                from gluonts.mx import WaveNetEstimator
            except Exception:
                from gluonts.model.wavenet import WaveNetEstimator

            estimator = WaveNetEstimator(
                freq=ts_metadata['freq'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )
        elif self.gluon_model == 'DeepVAR':
            try:
                from gluonts.mx import DeepVAREstimator
            except Exception:
                from gluonts.model.deepvar import DeepVAREstimator

            estimator = DeepVAREstimator(
                target_dim=df.shape[1],
                freq=ts_metadata['freq'],
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )
        elif self.gluon_model == 'GPVAR':
            try:
                from gluonts.mx import GPVAREstimator
            except Exception:
                from gluonts.model.gpvar import GPVAREstimator

            estimator = GPVAREstimator(
                target_dim=df.shape[1],
                freq=ts_metadata['freq'],
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )
        elif self.gluon_model == 'LSTNet':
            try:
                from gluonts.mx import LSTNetEstimator
            except Exception:
                from gluonts.model.lstnet import LSTNetEstimator

            estimator = LSTNetEstimator(
                # freq=ts_metadata['freq'],
                num_series=len(self.train_index),
                skip_size=1,
                ar_window=1,
                channels=2,
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )
        elif self.gluon_model == 'NBEATS':
            try:
                from gluonts.mx import NBEATSEstimator
            except Exception:
                from gluonts.model.n_beats import NBEATSEstimator

            estimator = NBEATSEstimator(
                freq=ts_metadata['freq'],
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )
        elif self.gluon_model == 'Rotbaum':
            try:
                from gluonts.ext.rotbaum import TreeEstimator
            except Exception:
                from gluonts.model.rotbaum import TreeEstimator

            estimator = TreeEstimator(
                freq=ts_metadata['freq'],
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
                # trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )
        elif self.gluon_model == 'DeepRenewalProcess':
            try:
                from gluonts.mx import DeepRenewalProcessEstimator
            except Exception:
                from gluonts.model.renewal import DeepRenewalProcessEstimator

            estimator = DeepRenewalProcessEstimator(
                prediction_length=ts_metadata['forecast_length'],
                context_length=ts_metadata['context_length'],
                num_layers=1,  # original paper used 1 layer, 10 cells
                num_cells=10,
                # freq=ts_metadata['freq'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )
        elif self.gluon_model == 'SelfAttention':
            try:
                from gluonts.nursery.san import SelfAttentionEstimator
            except Exception:
                from gluonts.model.san import SelfAttentionEstimator

            estimator = SelfAttentionEstimator(
                prediction_length=ts_metadata['forecast_length'],
                context_length=ts_metadata['context_length'],
                freq=ts_metadata['freq'],
                trainer=Trainer(
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                ),
            )
        elif self.gluon_model == 'TemporalFusionTransformer':
            try:
                from gluonts.mx import TemporalFusionTransformerEstimator
            except Exception:
                from gluonts.model.tft import TemporalFusionTransformerEstimator

            estimator = TemporalFusionTransformerEstimator(
                prediction_length=ts_metadata['forecast_length'],
                context_length=ts_metadata['context_length'],
                freq=ts_metadata['freq'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )
        elif self.gluon_model == 'DeepTPP':
            try:
                from gluonts.mx import DeepTPPEstimator
            except Exception:
                from gluonts.model.tpp.deeptpp import DeepTPPEstimator

            estimator = DeepTPPEstimator(
                prediction_interval_length=ts_metadata['forecast_length'],
                context_interval_length=ts_metadata['context_length'],
                num_marks=1,  # cardinality
                freq=ts_metadata['freq'],
                trainer=Trainer(
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    hybridize=False,
                ),
            )
        elif self.gluon_model == 'PatchTST':
            from gluonts.torch.model.patch_tst import PatchTSTEstimator

            estimator = PatchTSTEstimator(
                prediction_length=ts_metadata['forecast_length'],
                context_length=ts_metadata['context_length'],
                patch_len=5,
                lr=self.learning_rate,
                trainer_kwargs={'logger': False, 'log_every_n_steps': 0},
            )
        else:
            raise ValueError("'gluon_model' not recognized.")

        if self.gluon_model == 'NPTS' and npts_flag:
            self.GluonPredictor = estimator
        else:
            self.GluonPredictor = estimator.train(self.test_ds)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def fit_data(self, df, future_regressor=None):
        df = self.basic_profile(df)
        gluon_train = df.to_numpy().T
        self.train_index = df.columns
        self.train_columns = df.index
        if self.regression_type == "User":
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but no future_regressor supplied"
                )
        if self.gluon_model in self.multivariate_mods:
            if self.regression_type == "User":
                regr = future_regressor.to_numpy().T
                self.regr_train = regr
                self.test_ds = ListDataset(
                    [
                        {
                            "start": df.index[0],
                            "target": gluon_train,
                            "feat_dynamic_real": regr,
                        }
                    ],
                    freq=self.ts_metadata['freq'],
                    one_dim_target=False,
                )
            else:
                self.test_ds = ListDataset(
                    [{"start": df.index[0], "target": gluon_train}],
                    freq=self.ts_metadata['freq'],
                    one_dim_target=False,
                )
        else:
            if self.regression_type == "User":
                self.gluon_train = gluon_train
                regr = future_regressor.to_numpy().T
                self.regr_train = regr
                self.test_ds = ListDataset(
                    [
                        {
                            FieldName.TARGET: target,
                            FieldName.START: self.ts_metadata['start_ts'],
                            FieldName.FEAT_DYNAMIC_REAL: regr,
                        }
                        for target in gluon_train
                    ],
                    freq=self.ts_metadata['freq'],
                )
            else:
                # use the actual start date, if NaN given (semi-hidden)
                # ts_metadata['gluon_start'] = df.notna().idxmax().tolist()
                self.test_ds = ListDataset(
                    [
                        {FieldName.TARGET: target, FieldName.START: start}
                        for (target, start) in zip(
                            gluon_train, self.ts_metadata['gluon_start']
                        )
                    ],
                    freq=self.ts_metadata['freq'],
                )
        return self

    def predict(
        self,
        forecast_length: int = None,
        future_regressor=[],
        just_point_forecast=False,
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
        if forecast_length is None:
            forecast_length = self.forecast_length
        if int(forecast_length) > int(self.forecast_length):
            raise ValueError("GluonTS must be refit to change forecast length!")
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(
            forecast_length=self.ts_metadata['forecast_length']
        )
        if self.regression_type == "User":
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but no future_regressor supplied"
                )
            regr = future_regressor.to_numpy().T
            regr = np.concatenate([self.regr_train, regr], axis=1)
            self.test_ds = ListDataset(
                [
                    {
                        FieldName.TARGET: target,
                        FieldName.START: self.ts_metadata['start_ts'],
                        FieldName.FEAT_DYNAMIC_REAL: regr,
                    }
                    for target in self.gluon_train
                ],
                freq=self.ts_metadata['freq'],
            )

        gluon_results = self.GluonPredictor.predict(self.test_ds)
        if self.gluon_model in self.multivariate_mods:
            result = list(gluon_results)[0]
            forecast = pd.DataFrame(
                result.quantile(0.5), index=test_index, columns=self.column_names
            )
            upper_forecast = pd.DataFrame(
                result.quantile(self.prediction_interval),
                index=test_index,
                columns=self.column_names,
            )
            lower_forecast = pd.DataFrame(
                result.quantile((1 - self.prediction_interval)),
                index=test_index,
                columns=self.column_names,
            )
        else:
            i = 0
            all_forecast = pd.DataFrame()
            for result in gluon_results:
                current_id = self.train_index[i]
                if isinstance(result.start_date, pd.Period):
                    start_date = test_index[0]
                else:
                    start_date = result.start_date
                rowForecast = pd.DataFrame(
                    {
                        "ForecastDate": pd.date_range(
                            start=start_date,
                            periods=self.ts_metadata['forecast_length'],
                            freq=self.frequency,
                        ),
                        "series_id": current_id,
                        "LowerForecast": (
                            result.quantile((1 - self.prediction_interval))
                        ),
                        "MedianForecast": (result.quantile(0.5)),
                        "UpperForecast": (result.quantile(self.prediction_interval)),
                    }
                )
                all_forecast = pd.concat(
                    [all_forecast, rowForecast], ignore_index=True
                ).reset_index(drop=True)
                i += 1
            if result.start_date != test_index[0] and int(self.verbose) > 0:
                print(
                    f"GluonTS start_date is {result.start_date} vs created index {test_index[0]}"
                )
            forecast = all_forecast.pivot_table(
                values='MedianForecast', index='ForecastDate', columns='series_id'
            )
            forecast.index = test_index
            forecast = forecast[self.column_names]
            lower_forecast = all_forecast.pivot_table(
                values='LowerForecast', index='ForecastDate', columns='series_id'
            )
            lower_forecast.index = test_index
            lower_forecast = lower_forecast[self.column_names]
            upper_forecast = all_forecast.pivot_table(
                values='UpperForecast', index='ForecastDate', columns='series_id'
            )
            upper_forecast.index = test_index
            upper_forecast = upper_forecast[self.column_names]

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

        if "deep" in method:
            gluon_model_choice = random.choices(
                [
                    'DeepAR',
                    'NPTS',
                    'DeepState',
                    'WaveNet',
                    'DeepFactor',
                    'Transformer',
                    'SFF',
                    'MQCNN',
                    'DeepVAR',
                    'GPVAR',
                    'NBEATS',
                    'Rotbaum',
                    'LSTNet',
                    'DeepRenewalProcess',
                    'SelfAttention',
                    'TemporalFusionTransformer',
                    'DeepTPP',
                ],
                [
                    0.1,
                    0.2,
                    0.05,
                    0.1,
                    0.1,
                    0.2,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.05,
                    0.01,
                    0.1,
                    0.1,
                    0.01,
                    0.1,
                    0.01,
                ],
                k=1,
            )[0]
            epochs_choice = random.choices(
                [20, 40, 80, 150, 300, 500], [0.58, 0.35, 0.05, 0.05, 0.05, 0.02]
            )[0]
        else:
            gluon_model_choice = random.choices(
                [
                    'DeepAR',
                    'NPTS',
                    'DeepState',
                    'WaveNet',
                    'DeepFactor',
                    'Transformer',
                    'SFF',
                    'MQCNN',
                    'DeepVAR',
                    'GPVAR',
                    'NBEATS',
                    'Rotbaum',
                    'LSTNet',
                    'DeepRenewalProcess',
                    'SelfAttention',
                    'TemporalFusionTransformer',
                    'DeepTPP',
                    'PatchTST',
                ],
                [
                    0.1,
                    0.2,
                    0.05,
                    0.1,
                    0.1,
                    0.2,
                    0.1,
                    0.1,
                    0.1,
                    0.0,
                    0.05,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                ],
                k=1,
            )[0]
            epochs_choice = random.choices([10, 20, 40, 80], [0.02, 0.58, 0.35, 0.05])[
                0
            ]
        # your base parameters
        context_length_choice = random.choices(
            [5, 10, 30, '1ForecastLength', '2ForecastLength'],
            [0.2, 0.3, 0.1, 0.1, 0.1],
        )[0]
        learning_rate_choice = random.choices(
            [0.01, 0.001, 0.0001, 0.00001], [0.3, 0.6, 0.1, 0.1]
        )[0]
        # NPTS doesn't use these, so just fill a constant
        if gluon_model_choice in ['NPTS', 'Rotbaum']:
            epochs_choice = 20
            learning_rate_choice = 0.001
        # this model being noticeably slower than others at scale
        elif gluon_model_choice == 'GPVAR':
            context_length_choice = random.choice([5, 7, 12])
        if "regressor" in method:
            regression_choice = "User"
        else:
            if gluon_model_choice in self.multivariate_mods:
                regression_choice = None
            else:
                regression_choice = random.choices([None, "User"], [0.8, 0.4])[0]

        return {
            'gluon_model': gluon_model_choice,
            'epochs': epochs_choice,
            'learning_rate': learning_rate_choice,
            'context_length': context_length_choice,
            'regression_type': regression_choice,
            # 'additional_params':
        }

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'gluon_model': self.gluon_model,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'context_length': self.context_length,
            'regression_type': self.regression_type,
        }
        return parameter_dict


"""
model = GluonTS(epochs=5)
model = model.fit(df.ffill().bfill())
prediction = model.predict(forecast_length=14)
prediction.forecast
"""

# to add: model_dim, dropout_rate, act_type, init,
