"""Neural Nets."""
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
except Exception:  # except ImportError
    _has_tf = False
else:
    _has_tf = True
"""
X, Y = window_maker(df, forecast_length = 6, shuffle = False,
                    input_dim = 'multivariate', window_size = 10,
                    output_dim = '1step')
"""


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

    def __init__(self, rnn_type: str = 'LSTM',
                 kernel_initializer: str = 'lecun_uniform',
                 hidden_layer_sizes: tuple = (32, 32, 32),
                 optimizer: str = 'adam', loss: str = 'huber',
                 epochs: int = 50, batch_size: int = 32,
                 verbose: int = 1, random_seed: int = 2020):
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

    def fit(self, X, Y):
        """Train the model on dataframes of X and Y."""
        if not _has_tf:
            raise ImportError("Tensorflow not available, install with pip install tensorflow.")
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.random_seed)
        train_X = pd.DataFrame(X).values
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        INPUT_SHAPE = (train_X.shape[1], train_X.shape[2])
        OUTPUT_SHAPE = Y.shape[1]
        if len(self.hidden_layer_sizes) == 3:
            if self.rnn_type == 'GRU':
                simple_lstm_model = tf.keras.models.Sequential([
                    tf.keras.layers.Conv1D(
                        filters=self.hidden_layer_sizes[0],
                        kernel_size=3, activation='relu',
                        strides=1, padding='causal',
                        kernel_initializer=self.kernel_initializer,
                        input_shape=INPUT_SHAPE),
                    tf.keras.layers.GRU(self.hidden_layer_sizes[1],
                                        return_sequences=True),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.GRU(self.hidden_layer_sizes[2]),
                    tf.keras.layers.Dense(OUTPUT_SHAPE)
                ])
            else:
                simple_lstm_model = tf.keras.models.Sequential([
                    tf.keras.layers.LSTM(
                        self.hidden_layer_sizes[0],
                        kernel_initializer=self.kernel_initializer,
                        input_shape=INPUT_SHAPE,
                        return_sequences=True),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(self.hidden_layer_sizes[1],
                                         return_sequences=True),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(self.hidden_layer_sizes[2]),
                    tf.keras.layers.Dense(OUTPUT_SHAPE)
                ])
        if len(self.hidden_layer_sizes) == 1:
            if self.rnn_type == 'GRU':
                simple_lstm_model = tf.keras.models.Sequential([
                    tf.keras.layers.GRU(
                        self.hidden_layer_sizes[0],
                        kernel_initializer=self.kernel_initializer,
                        input_shape=INPUT_SHAPE),
                    tf.keras.layers.Dense(10, activation='relu'),
                    tf.keras.layers.Dense(OUTPUT_SHAPE)
                ])
            else:
                simple_lstm_model = tf.keras.models.Sequential([
                    tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(
                            self.hidden_layer_sizes[0],
                            kernel_initializer=self.kernel_initializer,
                            input_shape=INPUT_SHAPE)),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(OUTPUT_SHAPE),
                    tf.keras.layers.Lambda(lambda x: x * 100.0)
                ])

        if self.loss == 'Huber':
            loss = tf.keras.losses.Huber()
        else:
            loss = self.loss
        simple_lstm_model.compile(optimizer=self.optimizer, loss=loss)

        simple_lstm_model.fit(x=train_X, y=Y, epochs=self.epochs,
                              batch_size=self.batch_size,
                              verbose=self.verbose)
        self.model = simple_lstm_model
        return self

    def predict(self, X):
        """Predict on dataframe of X."""
        test = pd.DataFrame(X).values.reshape((X.shape[0], 1, X.shape[1]))
        return pd.DataFrame(self.model.predict(test))
