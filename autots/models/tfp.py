import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject, PredictionObject, seasonal_int

try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow_probability import sts
except Exception:  # except ImportError
    _has_tf = False
else:
    _has_tf = True


class TensorflowSTS(ModelObject):
    """STS from TensorflowProbability.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        regression_type (str): type of regression (None, 'User', or 'Holiday')
    """

    def __init__(
        self,
        name: str = "TensorflowSTS",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        trend: str = 'local',
        seasonal_periods: int = None,
        ar_order: int = None,
        fit_method: str = 'hmc',
        num_steps: int = 200,
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
        self.seasonal_periods = seasonal_periods
        self.ar_order = ar_order
        self.trend = trend
        self.fit_method = fit_method
        self.num_steps = num_steps

    def _build_model(
        self, observed_time_series, trend='local', seasonal_periods=12, ar_order=1
    ):
        if trend == 'semilocal':
            trend = sts.SemiLocalLinearTrend(
                observed_time_series=observed_time_series, name='trend'
            )
        else:
            trend = sts.LocalLinearTrend(
                observed_time_series=observed_time_series, name='trend'
            )
        mods_list = [trend]
        if str(seasonal_periods).isdigit():
            seasonal = tfp.sts.SmoothSeasonal(
                period=seasonal_periods,
                observed_time_series=observed_time_series,
                name='seasonal',
                frequency_multipliers=[1, 2, 3],
            )
            mods_list.append(seasonal)
        if str(ar_order).isdigit():
            autoregressive = sts.Autoregressive(
                order=int(ar_order),
                observed_time_series=observed_time_series,
                name='autoregressive',
            )
            mods_list.append(autoregressive)
        model = sts.Sum(mods_list, observed_time_series=observed_time_series)
        return model

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)

        demand2d = df.transpose().values
        # models expect shape: [..., num_timesteps] or [..., num_timesteps, 1]
        demand_model = self._build_model(
            demand2d,
            trend=self.trend,
            seasonal_periods=self.seasonal_periods,
            ar_order=self.ar_order,
        )

        if self.fit_method == 'variational':
            # Build the variational surrogate posteriors `qs`.
            variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
                model=demand_model
            )
            q_samples = variational_posteriors.sample(int(self.num_steps))

        else:
            self.fit_method = 'hmc'
            q_samples, kernel_results = tfp.sts.fit_with_hmc(
                demand_model,
                demand2d,
                num_results=self.num_steps,
                seed=self.random_seed,
            )
        self.q_samples = q_samples
        self.demand2d = demand2d
        self.demand_model = demand_model

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
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        forecast_dist = tfp.sts.forecast(
            model=self.demand_model,
            observed_time_series=self.demand2d,
            parameter_samples=self.q_samples,
            num_steps_forecast=forecast_length,
            include_observation_noise=True,
        )

        forecast = forecast_dist.mean().numpy()[..., 0]

        forecast = pd.DataFrame(
            forecast, index=self.column_names, columns=test_index
        ).transpose()

        if just_point_forecast:
            return forecast
        else:
            prediction_interval = self.prediction_interval
            # assume follows rules of normal because those are conventional
            from scipy.stats import norm

            # adj = norm.sf(abs(prediction_interval))*2
            p_int = 1 - ((1 - prediction_interval) / 2)
            adj = norm.ppf(p_int)
            forecast_scale = forecast_dist.stddev().numpy()[..., 0]
            upper_forecast = forecast.transpose().values + (forecast_scale * adj)
            lower_forecast = forecast.transpose().values - (forecast_scale * adj)
            lower_forecast = pd.DataFrame(
                lower_forecast, index=self.column_names, columns=test_index
            ).transpose()
            upper_forecast = pd.DataFrame(
                upper_forecast, index=self.column_names, columns=test_index
            ).transpose()
            # alternatively this followed by quantile
            # forecast_samples = self.forecast_dist.sample(10)[..., 0]
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
        fit_method_choice = np.random.choice(['hmc', 'variational'], size=1).item()
        num_steps_choice = np.random.choice([50, 200], size=1).item()
        ar_order_choice = np.random.choice(
            [None, 1, 2, 7], size=1, p=[0.5, 0.3, 0.1, 0.1]
        ).item()
        seasonal_periods_choice = np.random.choice(
            [None, seasonal_int()], size=1
        ).item()
        trend_choice = np.random.choice(['local', 'semilocal'], size=1).item()

        parameter_dict = {
            'fit_method': fit_method_choice,
            'num_steps': num_steps_choice,
            'ar_order': ar_order_choice,
            'seasonal_periods': seasonal_periods_choice,
            'trend': trend_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'fit_method': self.fit_method,
            'num_steps': self.num_steps,
            'ar_order': self.ar_order,
            'seasonal_periods': self.seasonal_periods,
            'trend': self.trend,
        }
        return parameter_dict


class TFPRegressor(object):
    """Wrapper for Tensorflow Keras based RNN.

    Args:
        rnn_type (str): Keras cell type 'GRU' or default 'LSTM'
        kernel_initializer (str): passed to first keras LSTM or GRU layer
        hidden_layer_sizes (tuple): of len 1 or 3 passed to first keras LSTM or GRU layers
        optimizer (str): Passed to keras model.compile
        loss (str): Passed to keras model.compile
        epochs (int): Passed to keras model.fit
        batch_size (int): Passed to keras model.fit
        verbose (int): 0, 1 or 2. Passed to keras model.fit
        random_seed (int): passed to tf.random.set_seed()
    """

    def __init__(
        self,
        kernel_initializer: str = 'lecun_uniform',
        optimizer: str = 'adam',
        loss: str = 'negloglike',
        epochs: int = 50,
        batch_size: int = 32,
        dist: str = 'normal',
        verbose: int = 1,
        random_seed: int = 2020,
    ):
        self.name = 'TFPRegressor'
        verbose += 1
        verbose = 0 if verbose < 0 else verbose
        verbose = 2 if verbose > 2 else verbose
        self.verbose = verbose
        self.random_seed = random_seed
        self.kernel_initializer = kernel_initializer
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss  # negloglike, Huber, mae
        self.dist = dist  # normal, poisson, negbinom

    def fit(self, X, Y):
        """Train the model on dataframes of X and Y."""
        if not _has_tf:
            raise ImportError("TensorflowProbability not installed.")
        tf.keras.backend.clear_session()
        tf.keras.backend.set_floatx('float64')
        tf.random.set_seed(self.random_seed)
        train_X = pd.DataFrame(X).values
        INPUT_SHAPE = (train_X.shape[1],)
        OUTPUT_SHAPE = Y.shape[1]
        train_Y = pd.DataFrame(Y).values
        if self.dist == 'other':
            tfd = tfp.distributions
            tfk = tf.keras
            n_batches = 10
            prior = tfd.Independent(
                tfd.Normal(loc=tf.zeros(OUTPUT_SHAPE, dtype=tf.float64), scale=1.0),
                reinterpreted_batch_ndims=1,
            )
            self.nn_model = tf.keras.Sequential(
                [
                    tfk.layers.InputLayer(input_shape=INPUT_SHAPE, name="input"),
                    tfk.layers.Dense(10, activation="relu", name="dense_1"),
                    tfk.layers.Dense(
                        tfp.layers.MultivariateNormalTriL.params_size(OUTPUT_SHAPE),
                        activation=None,
                        name="distribution_weights",
                    ),
                    tfp.layers.MultivariateNormalTriL(
                        OUTPUT_SHAPE,
                        activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                            prior, weight=1 / n_batches
                        ),
                        name="output",
                    ),
                ],
                name="model",
            )
        elif self.dist == 'poisson':
            self.nn_model = tf.keras.models.Sequential()
            self.nn_model.add(tf.keras.layers.Dense(10, activation="relu"))
            self.nn_model.add(
                tf.keras.layers.Dense(
                    tfp.layers.IndependentPoisson.params_size(OUTPUT_SHAPE + 1)
                )
            )
            self.nn_model.add(
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Normal(
                        loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])
                    )
                )
            )
        else:
            self.dist = 'normal'
            self.nn_model = tf.keras.models.Sequential()
            self.nn_model.add(tf.keras.layers.Dense(OUTPUT_SHAPE + 1))
            self.nn_model.add(
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Normal(
                        loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])
                    )
                )
            )

        if self.loss == 'negloglike':
            self.nn_model.compile(
                optimizer=self.optimizer, loss=lambda y, model: -model.log_prob(y)
            )
        else:
            if self.loss == 'Huber':
                loss = tf.keras.losses.Huber()
            else:
                loss = self.loss
            self.nn_model.compile(optimizer=self.optimizer, loss=loss)

        self.nn_model.fit(
            x=train_X,
            y=train_Y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        """
        nn_model.fit(x=train_X, y=train_Y, epochs=50,
                     batch_size=32)
        """
        return self

    def predict(self, X, conf_int: float = None):
        """Predict on dataframe of X."""
        test = pd.DataFrame(X).values
        if conf_int is None:
            return pd.DataFrame(self.nn_model.predict(test))
        else:
            p_int = 1 - ((1 - conf_int) / 2)
            from scipy.stats import norm

            adj = norm.ppf(p_int)

            yhat = self.nn_model(test)
            mean = self.nn_model.predict(test)
            stddev = yhat.stddev().numpy()
            mean_plus = mean - adj * stddev
            mean_minus = mean + adj * stddev
            return mean, mean_minus, mean_plus


class TFPRegression(ModelObject):
    """Tensorflow Probability regression.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): type of regression (None, 'User')
    """

    def __init__(
        self,
        name: str = "TFPRegression",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 1,
        kernel_initializer: str = 'lecun_uniform',
        optimizer: str = 'adam',
        loss: str = 'negloglike',
        epochs: int = 50,
        batch_size: int = 32,
        dist: str = 'normal',
        regression_type: str = None,
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
        self.verbose = verbose
        self.random_seed = random_seed
        self.kernel_initializer = kernel_initializer
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.optimizer = optimizer
        self.loss = loss  # negloglike, Huber, mae
        self.dist = dist  # normal, poisson, negbinom

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """

        df = self.basic_profile(df)
        if self.regression_type == 'User':
            if (np.array(future_regressor).shape[0]) != (df.shape[0]):
                self.regression_type = None
            else:
                self.future_regressor_train = future_regressor
        self.fit_runtime = datetime.datetime.now() - self.startTime

        from autots.models.sklearn import date_part

        X = date_part(df.index, method='expanded')
        if self.regression_type == 'User':
            # if self.future_regressor_train.ndim == 1:
            #     self.future_regressor_train = np.array(self.future_regressor_train).reshape(-1, 1)
            # X = np.concatenate((X.reshape(-1, 1), self.future_regressor_train), axis=1)
            X = pd.concat(
                [X, pd.DataFrame(self.future_regressor_train).reset_index(drop=True)],
                axis=1,
            )
        y = df
        self.model = TFPRegressor(
            verbose=self.verbose,
            random_seed=self.random_seed,
            kernel_initializer=self.kernel_initializer,
            epochs=self.epochs,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            loss=self.loss,
            dist=self.dist,
        ).fit(X.values, y)
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
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        from autots.models.sklearn import date_part

        Xf = date_part(test_index, method='expanded')
        if self.regression_type == 'User':
            # if future_regressor.ndim == 1:
            #     future_regressor = np.array(future_regressor).reshape(-1, 1)
            # Xf = np.concatenate((Xf.reshape(-1, 1), future_regressor), axis=1)
            Xf = pd.concat(
                [Xf, pd.DataFrame(future_regressor).reset_index(drop=True)], axis=1
            )
        forecast, lower_forecast, upper_forecast = self.model.predict(
            Xf.values, conf_int=self.prediction_interval
        )
        df_forecast = pd.DataFrame(forecast)
        df_forecast.columns = self.column_names
        df_forecast.index = test_index
        if just_point_forecast:
            return df_forecast
        else:
            lower_forecast = pd.DataFrame(
                lower_forecast, index=test_index, columns=self.column_names
            )
            upper_forecast = pd.DataFrame(
                upper_forecast, index=test_index, columns=self.column_names
            )
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=df_forecast.index,
                forecast_columns=df_forecast.columns,
                lower_forecast=lower_forecast,
                forecast=df_forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        init_list = [
            'glorot_uniform',
            'lecun_uniform',
            'glorot_normal',
            'RandomUniform',
            'he_normal',
        ]
        kernel_initializer = np.random.choice(init_list, size=1).item()
        epochs = np.random.choice([15, 50, 150], p=[0.2, 0.7, 0.1], size=1).item()
        batch_size = np.random.choice(
            [8, 16, 32, 72], p=[0.2, 0.2, 0.5, 0.1], size=1
        ).item()
        optimizer = np.random.choice(
            ['adam', 'rmsprop', 'adagrad'], p=[0.4, 0.5, 0.1], size=1
        ).item()
        loss = np.random.choice(
            ['negloglike', 'mae', 'Huber', 'poisson', 'mse', 'mape'],
            p=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
            size=1,
        ).item()
        dist_choice = np.random.choice(
            a=['normal', 'poisson', 'other'], size=1, p=[0.3, 0.3, 0.4]
        ).item()
        regression_type_choice = np.random.choice(
            a=[None, 'User'], size=1, p=[0.8, 0.2]
        ).item()
        return {
            'kernel_initializer': kernel_initializer,
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'loss': loss,  # negloglike, Huber, mae
            'dist': dist_choice,
            'regression_type': regression_type_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'kernel_initializer': self.kernel_initializer,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'dist': self.dist,
            'regression_type': self.regression_type,
        }


"""
temp = TFPRegressor().fit(X, Y)
test = TFPRegression().fit(df)
test.predict(14).forecast

test = TensorflowSTS().fit(df).predict(14)
test.forecast
"""
