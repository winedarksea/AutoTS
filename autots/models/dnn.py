"""Neural Nets."""

import random
import pandas as pd
from autots.tools.shaping import wide_to_3d

try:
    import tensorflow as tf
except Exception:  # except ImportError
    _has_tf = False
else:
    _has_tf = True
"""
X, Y = window_maker(df, forecast_length = 6, shuffle = False,
                    input_dim = 'univariate', window_size = 10,
                    output_dim = '1step')
"""

if _has_tf:

    class ResidualWrapper(tf.keras.Model):
        """From https://www.tensorflow.org/tutorials/structured_data/time_series"""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def call(self, inputs, *args, **kwargs):
            delta = self.model(inputs, *args, **kwargs)

            # The prediction for each time step is the input
            # from the previous time step plus the delta
            # calculated by the model.
            return inputs + delta


class KerasRNN(object):
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
        rnn_type: str = 'LSTM',
        kernel_initializer: str = 'lecun_uniform',
        hidden_layer_sizes: tuple = (32, 32, 32),
        optimizer: str = 'adam',
        loss: str = 'huber',
        epochs: int = 50,
        batch_size: int = 32,
        shape=1,
        verbose: int = 1,
        random_seed: int = 2020,
    ):
        self.name = 'KerasRNN'
        verbose = 0 if verbose < 0 else verbose
        verbose = 2 if verbose > 2 else verbose
        self.verbose = verbose
        self.random_seed = random_seed
        self.kernel_initializer = kernel_initializer
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.hidden_layer_sizes = hidden_layer_sizes
        self.rnn_type = rnn_type
        self.shape = shape

    def fit(self, X, Y):
        """Train the model on dataframes of X and Y."""
        if not _has_tf:
            raise ImportError(
                "Tensorflow not available, install with pip install tensorflow."
            )
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.random_seed)
        train_X = pd.DataFrame(X).to_numpy()
        # target shape is [samples, timesteps, features]
        if self.shape == 1:
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        elif self.shape > 2:
            # needs matching Y shape and predict input
            train_X = wide_to_3d(train_X, self.shape)
        else:
            train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
        INPUT_SHAPE = (train_X.shape[1], train_X.shape[2])
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        OUTPUT_SHAPE = Y.shape[1]
        if self.rnn_type == "E2D2":
            # crudely borrowed from: https://www.analyticsvidhya.com/blog/2020/10/multivariate-multi-step-time-series-forecasting-using-stacked-lstm-sequence-to-sequence-autoencoder-in-tensorflow-2-0-keras/
            encoder_inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
            encoder_l1 = tf.keras.layers.LSTM(
                self.hidden_layer_sizes[0], return_sequences=True, return_state=True
            )
            encoder_outputs1 = encoder_l1(encoder_inputs)
            encoder_states1 = encoder_outputs1[1:]
            encoder_l2 = tf.keras.layers.LSTM(
                self.hidden_layer_sizes[0], return_state=True
            )
            encoder_outputs2 = encoder_l2(encoder_outputs1[0])
            encoder_states2 = encoder_outputs2[1:]
            #
            decoder_inputs = tf.keras.layers.RepeatVector(OUTPUT_SHAPE)(
                encoder_outputs2[0]
            )
            layer_2_shape = self.hidden_layer_sizes
            layer_2_size = (
                layer_2_shape[2] if len(layer_2_shape) >= 3 else layer_2_shape[0]
            )
            #
            decoder_l1 = tf.keras.layers.LSTM(layer_2_size, return_sequences=True)(
                decoder_inputs, initial_state=encoder_states1
            )
            decoder_l2 = tf.keras.layers.LSTM(layer_2_size, return_sequences=True)(
                decoder_l1, initial_state=encoder_states2
            )
            decoder_outputs2 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(OUTPUT_SHAPE)
            )(decoder_l2)
            #
            self.model = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
        if self.rnn_type == "CNN":
            if len(self.hidden_layer_sizes) == 1:
                kernel_size = 10 if INPUT_SHAPE[0] > 10 else INPUT_SHAPE[0]
                self.model = tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv1D(
                            filters=self.hidden_layer_sizes[0],
                            kernel_size=kernel_size,
                            activation='relu',
                            input_shape=INPUT_SHAPE,
                            kernel_initializer=self.kernel_initializer,
                        ),
                        tf.keras.layers.Dense(
                            units=self.hidden_layer_sizes[0], activation='relu'
                        ),
                        tf.keras.layers.Dense(units=OUTPUT_SHAPE),
                    ]
                )
            else:
                # borrowed from https://keras.io/examples/timeseries/timeseries_classification_from_scratch
                input_layer = tf.keras.layers.Input(INPUT_SHAPE)
                layer_shape = self.hidden_layer_sizes
                layer_2_size = (
                    layer_shape[1] if len(layer_shape) >= 2 else layer_shape[0]
                )
                layer_3_size = (
                    layer_shape[2] if len(layer_shape) >= 3 else layer_shape[0]
                )

                conv1 = tf.keras.layers.Conv1D(
                    filters=self.hidden_layer_sizes[0],
                    kernel_size=3,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                )(input_layer)
                conv1 = tf.keras.layers.BatchNormalization()(conv1)
                conv1 = tf.keras.layers.ReLU()(conv1)
                conv2 = tf.keras.layers.Conv1D(
                    filters=layer_2_size, kernel_size=3, padding="same"
                )(conv1)
                conv2 = tf.keras.layers.BatchNormalization()(conv2)
                conv2 = tf.keras.layers.ReLU()(conv2)
                conv3 = tf.keras.layers.Conv1D(
                    filters=layer_3_size, kernel_size=3, padding="same"
                )(conv2)
                conv3 = tf.keras.layers.BatchNormalization()(conv3)
                conv3 = tf.keras.layers.ReLU()(conv3)

                gap = tf.keras.layers.GlobalAveragePooling1D()(conv3)
                output_layer = tf.keras.layers.Dense(OUTPUT_SHAPE)(gap)
                self.model = tf.keras.models.Model(
                    inputs=input_layer, outputs=output_layer
                )
        elif len(self.hidden_layer_sizes) == 3:
            if self.rnn_type == 'GRU':
                self.model = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.GRU(
                            self.hidden_layer_sizes[0],
                            kernel_initializer=self.kernel_initializer,
                            input_shape=INPUT_SHAPE,
                            return_sequences=True,
                        ),
                        tf.keras.layers.GRU(
                            self.hidden_layer_sizes[1], return_sequences=True
                        ),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.GRU(self.hidden_layer_sizes[2]),
                        tf.keras.layers.Dense(OUTPUT_SHAPE),
                    ]
                )
            else:
                self.model = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.LSTM(
                            self.hidden_layer_sizes[0],
                            kernel_initializer=self.kernel_initializer,
                            input_shape=INPUT_SHAPE,
                            return_sequences=True,
                        ),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.LSTM(
                            self.hidden_layer_sizes[1], return_sequences=True
                        ),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.LSTM(self.hidden_layer_sizes[2]),
                        tf.keras.layers.Dense(OUTPUT_SHAPE),
                    ]
                )
        if len(self.hidden_layer_sizes) == 1:
            if self.rnn_type == 'GRU':
                self.model = ResidualWrapper(
                    tf.keras.models.Sequential(
                        [
                            tf.keras.layers.GRU(
                                self.hidden_layer_sizes[0],
                                kernel_initializer=self.kernel_initializer,
                                input_shape=INPUT_SHAPE,
                            ),
                            tf.keras.layers.Dense(10, activation='relu'),
                            tf.keras.layers.Dense(OUTPUT_SHAPE),
                        ]
                    )
                )
            else:
                self.model = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Bidirectional(
                            tf.keras.layers.LSTM(
                                self.hidden_layer_sizes[0],
                                kernel_initializer=self.kernel_initializer,
                                input_shape=INPUT_SHAPE,
                            )
                        ),
                        tf.keras.layers.Dense(32, activation='relu'),
                        tf.keras.layers.Dense(OUTPUT_SHAPE),
                        tf.keras.layers.Lambda(lambda x: x * 100.0),
                    ]
                )

        if str(self.loss).lower() == 'huber':
            loss = tf.keras.losses.Huber()
        else:
            loss = self.loss
        self.model.compile(optimizer=self.optimizer, loss=loss)

        self.model.fit(
            x=train_X,
            y=Y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self

    def predict(self, X):
        """Predict on dataframe of X."""
        if self.shape == 1:
            test = pd.DataFrame(X).to_numpy().reshape((X.shape[0], 1, X.shape[1]))
        else:
            test = pd.DataFrame(X).to_numpy().reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(test)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def transformer_build_model(
    input_shape,
    output_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)
    outputs = tf.keras.layers.Dense(output_shape)(x)
    return tf.keras.Model(inputs, outputs)


class Transformer(object):
    """Wrapper for Tensorflow Keras based Transformer.

    based on: https://keras.io/examples/timeseries/timeseries_transformer_classification/

    Args:

        optimizer (str): Passed to keras model.compile
        loss (str): Passed to keras model.compile
        epochs (int): Passed to keras model.fit
        batch_size (int): Passed to keras model.fit
        verbose (int): 0, 1 or 2. Passed to keras model.fit
        random_seed (int): passed to tf.random.set_seed()
    """

    def __init__(
        self,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
        optimizer: str = 'adam',
        loss: str = 'huber',
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
        random_seed: int = 2020,
    ):
        self.name = 'Transformer'
        verbose = 0 if verbose < 0 else verbose
        verbose = 2 if verbose > 2 else verbose
        self.verbose = verbose
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss

        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.mlp_dropout = mlp_dropout
        self.dropout = dropout

    def fit(self, X, Y):
        """Train the model on dataframes of X and Y."""
        if not _has_tf:
            raise ImportError(
                "Tensorflow not available, install with pip install tensorflow."
            )
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.random_seed)
        train_X = pd.DataFrame(X).to_numpy()
        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
        input_shape = train_X.shape[1:]
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        OUTPUT_SHAPE = Y.shape[1]

        self.model = transformer_build_model(
            input_shape,
            OUTPUT_SHAPE,
            head_size=self.head_size,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            mlp_units=self.mlp_units,
            mlp_dropout=self.mlp_dropout,
            dropout=self.dropout,
        )

        if self.loss == 'Huber':
            loss = tf.keras.losses.Huber()
        else:
            loss = self.loss
        optimizer = self.optimizer
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]

        self.model.fit(
            train_X,
            Y,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        return self

    def predict(self, X):
        """Predict on dataframe of X."""
        test = pd.DataFrame(X).to_numpy().reshape((X.shape[0], X.shape[1], 1))
        return pd.DataFrame(self.model.predict(test))


class ElasticNetwork(object):
    def __init__(
        self,
        size: int = 256,
        l1: float = 0.01,
        l2: float = 0.02,
        feature_subsample_rate: float = None,
        optimizer: str = 'adam',
        loss: str = 'mse',
        epochs: int = 20,
        batch_size: int = 32,
        activation: str = "relu",
        verbose: int = 1,
        random_seed: int = 2024,
    ):
        self.name = 'ElasticNetwork'
        self.verbose = verbose
        self.random_seed = random_seed
        self.size = size
        self.l1 = l1
        self.l2 = l2
        self.feature_subsample_rate = feature_subsample_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.activation = activation

    def fit(self, X, y):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Layer
        from tensorflow.keras.regularizers import L1L2

        # hiding this here as TF is an optional import
        class SubsetDense(Layer):
            def __init__(self, units, input_dim, feature_subsample_rate=0.5, **kwargs):
                super(SubsetDense, self).__init__(**kwargs)
                self.units = units
                self.input_dim = input_dim
                self.feature_subsample_rate = feature_subsample_rate
                self.selected_features_per_unit = []
                self.kernels = []
                self.biases = None

            def build(self, input_shape):
                # Select a subset of the input features for each unit
                num_features = int(self.input_dim * self.feature_subsample_rate)
                for _ in range(self.units):
                    selected_features = random.sample(
                        range(self.input_dim), num_features
                    )
                    self.selected_features_per_unit.append(selected_features)
                    kernel = self.add_weight(
                        shape=(num_features,),
                        initializer='glorot_uniform',
                        name=f'kernel_{len(self.kernels)}',
                    )
                    self.kernels.append(kernel)

                self.biases = self.add_weight(
                    shape=(self.units,), initializer='zeros', name='biases'
                )

            def call(self, inputs):
                outputs = []
                for i in range(self.units):
                    selected_inputs = tf.gather(
                        inputs, self.selected_features_per_unit[i], axis=1
                    )
                    output = (
                        tf.reduce_sum(selected_inputs * self.kernels[i], axis=1)
                        + self.biases[i]
                    )
                    outputs.append(output)
                return tf.stack(outputs, axis=1)

        # Model configuration
        input_dim = X.shape[1]  # Number of input features
        output_dim = y.shape[1]  # Number of outputs

        # Build the model
        if self.feature_subsample_rate is None:
            self.model = Sequential(
                [
                    Dense(
                        self.size,
                        input_dim=input_dim,
                        activation=self.activation,
                        kernel_regularizer=L1L2(l1=self.l1, l2=self.l2),
                    ),  # Example layer
                    Dense(output_dim),  # Output layer
                ]
            )
        else:
            self.model = Sequential(
                [
                    SubsetDense(
                        self.size,
                        input_dim=input_dim,
                        feature_subsample_rate=self.feature_subsample_rate,
                    ),
                    tf.keras.layers.Activation(self.activation),
                    SubsetDense(
                        self.size // 2,
                        input_dim=input_dim,
                        feature_subsample_rate=self.feature_subsample_rate,
                    ),
                    tf.keras.layers.Activation(self.activation),
                    Dense(output_dim),  # Output layer
                ]
            )

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

        return self

    def predict(self, X):
        return self.model.predict(X)
