"""
GluonTS

Excellent models, released by Amazon, scale well.
Except it is really the only thing I use that runs mxnet, and it takes a while to train these guys...
"""
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject

try:
    from gluonts.dataset.common import ListDataset

    try:
        from gluonts.transform import FieldName  # old way (0.3.3 and older)
    except Exception:
        from gluonts.dataset.field_names import FieldName  # new way
    from gluonts.trainer import Trainer
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

        gluon_model (str): Model Structure to Use - ['DeepAR', 'NPTS', 'DeepState', 'WaveNet','DeepFactor', 'Transformer','SFF', 'MQCNN']
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
        )
        self.gluon_model = gluon_model
        if self.gluon_model == 'NPTS':
            self.epochs = 20
            self.learning_rate = 0.001
        else:
            self.epochs = epochs
            self.learning_rate = learning_rate
        self.context_length = context_length
        self.forecast_length = forecast_length

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)

        try:
            from mxnet.random import seed as mxnet_seed

            mxnet_seed(self.random_seed)
        except Exception:
            pass

        gluon_train = df.transpose()
        self.train_index = gluon_train.index

        gluon_freq = str(self.frequency).split('-')[0]
        if gluon_freq in ["MS", "1MS"]:
            gluon_freq = "M"

        if int(self.verbose) > 1:
            print(f"Gluon Frequency is {gluon_freq}")

        if str(self.context_length).replace('.', '').isdigit():
            self.gluon_context_length = int(float(self.context_length))
        elif 'forecastlength' in str(self.context_length).lower():
            len_int = int([x for x in str(self.context_length) if x.isdigit()][0])
            self.gluon_context_length = int(len_int * self.forecast_length)
        else:
            self.gluon_context_length = 2 * self.forecast_length
            self.context_length = '2ForecastLength'
        ts_metadata = {
            'num_series': len(gluon_train.index),
            'freq': gluon_freq,
            'gluon_start': [
                gluon_train.columns[0] for _ in range(len(gluon_train.index))
            ],
            'context_length': self.gluon_context_length,
            'forecast_length': self.forecast_length,
        }
        self.test_ds = ListDataset(
            [
                {FieldName.TARGET: target, FieldName.START: start}
                for (target, start) in zip(
                    gluon_train.values, ts_metadata['gluon_start']
                )
            ],
            freq=ts_metadata['freq'],
        )
        if self.gluon_model == 'DeepAR':
            from gluonts.model.deepar import DeepAREstimator

            estimator = DeepAREstimator(
                freq=ts_metadata['freq'],
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )
        elif self.gluon_model == 'NPTS':
            from gluonts.model.npts import NPTSEstimator

            estimator = NPTSEstimator(
                freq=ts_metadata['freq'],
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
            )

        elif self.gluon_model == 'MQCNN':
            from gluonts.model.seq2seq import MQCNNEstimator

            estimator = MQCNNEstimator(
                freq=ts_metadata['freq'],
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )

        elif self.gluon_model == 'SFF':
            from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator

            estimator = SimpleFeedForwardEstimator(
                prediction_length=ts_metadata['forecast_length'],
                context_length=ts_metadata['context_length'],
                freq=ts_metadata['freq'],
                trainer=Trainer(
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    hybridize=False,
                    num_batches_per_epoch=100,
                ),
            )

        elif self.gluon_model == 'Transformer':
            from gluonts.model.transformer import TransformerEstimator

            estimator = TransformerEstimator(
                prediction_length=ts_metadata['forecast_length'],
                context_length=ts_metadata['context_length'],
                freq=ts_metadata['freq'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )

        elif self.gluon_model == 'DeepState':
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
            from gluonts.model.deep_factor import DeepFactorEstimator

            estimator = DeepFactorEstimator(
                freq=ts_metadata['freq'],
                context_length=ts_metadata['context_length'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )

        elif self.gluon_model == 'WaveNet':
            # Usually needs more epochs/training iterations than other models do
            from gluonts.model.wavenet import WaveNetEstimator

            estimator = WaveNetEstimator(
                freq=ts_metadata['freq'],
                prediction_length=ts_metadata['forecast_length'],
                trainer=Trainer(epochs=self.epochs, learning_rate=self.learning_rate),
            )
        else:
            raise ValueError("'gluon_model' not recognized.")

        self.GluonPredictor = estimator.train(self.test_ds)
        self.ts_metadata = ts_metadata
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=[], just_point_forecast=False
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
        if int(forecast_length) > int(self.forecast_length):
            print("GluonTS must be refit to change forecast length!")
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(
            forecast_length=self.ts_metadata['forecast_length']
        )

        gluon_results = self.GluonPredictor.predict(self.test_ds)
        i = 0
        all_forecast = pd.DataFrame()
        for result in gluon_results:
            current_id = self.train_index[i]
            rowForecast = pd.DataFrame(
                {
                    "ForecastDate": pd.date_range(
                        start=result.start_date,
                        periods=self.ts_metadata['forecast_length'],
                        freq=self.frequency,
                    ),
                    "series_id": current_id,
                    "LowerForecast": (result.quantile((1 - self.prediction_interval))),
                    "MedianForecast": (result.quantile(0.5)),
                    "UpperForecast": (result.quantile(self.prediction_interval)),
                }
            )
            all_forecast = pd.concat(
                [all_forecast, rowForecast], ignore_index=True
            ).reset_index(drop=True)
            i += 1
        forecast = all_forecast.pivot_table(
            values='MedianForecast', index='ForecastDate', columns='series_id'
        )
        forecast = forecast[self.column_names]

        if just_point_forecast:
            return forecast
        else:
            lower_forecast = all_forecast.pivot_table(
                values='LowerForecast', index='ForecastDate', columns='series_id'
            )
            lower_forecast = lower_forecast[self.column_names]
            upper_forecast = all_forecast.pivot_table(
                values='UpperForecast', index='ForecastDate', columns='series_id'
            )
            upper_forecast = upper_forecast[self.column_names]
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=test_index,
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
        gluon_model_choice = np.random.choice(
            a=[
                'DeepAR',
                'NPTS',
                'DeepState',
                'WaveNet',
                'DeepFactor',
                'Transformer',
                'SFF',
                'MQCNN',
            ],
            size=1,
            p=[0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
        ).item()
        if gluon_model_choice == 'NPTS':
            epochs_choice = 20
            learning_rate_choice = 0.001
        else:
            epochs_choice = np.random.choice(
                a=[20, 40, 80, 150], size=1, p=[0.58, 0.35, 0.05, 0.02]
            ).item()
            learning_rate_choice = np.random.choice(
                a=[0.01, 0.001, 0.0001], size=1, p=[0.3, 0.6, 0.1]
            ).item()
        context_length_choice = np.random.choice(
            a=[5, 10, 30, '1ForecastLength', '2ForecastLength'],
            size=1,
            p=[0.2, 0.3, 0.1, 0.1, 0.3],
        ).item()
        return {
            'gluon_model': gluon_model_choice,
            'epochs': epochs_choice,
            'learning_rate': learning_rate_choice,
            'context_length': context_length_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'gluon_model': self.gluon_model,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'context_length': self.context_length,
        }
        return parameter_dict


"""
model = GluonTS(epochs = 5)
model = model.fit(df_wide.fillna(method='ffill').fillna(method='bfill'))
prediction = model.predict(forecast_length = 14)
prediction.forecast
"""
