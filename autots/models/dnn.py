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

        "hidden_layer_sizes": np.random.choice(
            [(100,), (25, 15, 25), (50, 25, 50), (25, 50, 25)],
            p=[0.1, 0.5, 0.3, 0.1],
            size=1).item(),
        "max_iter": np.random.choice([250, 500, 1000],
                                     p=[0.89, 0.1, 0.01],
                                     size=1).item(),
        "activation": np.random.choice(['identity', 'logistic',
                                        'tanh', 'relu'],
                                       p=[0.05, 0.05, 0.6, 0.3],
                                       size=1).item(),
        "solver": solver,
        "early_stopping": early_stopping,
        "learning_rate_init"
"""


class KerasRNN(object):
    def __init__(self):
        self.name = 'KerasRNN'

    def fit(self, X, Y):
        train_X = X.values
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        INPUT_SHAPE=(train_X.shape[1], train_X.shape[2])
        OUTPUT_SHAPE = Y.shape[1]

        simple_lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, input_shape=INPUT_SHAPE, return_sequences = True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences = True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(OUTPUT_SHAPE)
        ])
        
        simple_lstm_model.compile(optimizer='adam', loss='mae')
        EVALUATION_INTERVAL = int(X.shape[0]/2)
        EVALUATION_INTERVAL = 200 if EVALUATION_INTERVAL > 200 else EVALUATION_INTERVAL
        EPOCHS = 20
        
        simple_lstm_model.fit(x=train_X, y=Y, epochs=EPOCHS,
                              steps_per_epoch=EVALUATION_INTERVAL)
        self.model = simple_lstm_model
        return self

    def predict(self, X):
        test = X.values.reshape((X.shape[0], 1, X.shape[1]))
        return self.model.predict(test)
"""
LSTM
tf.keras.layers.LSTM(
    units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
    recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False,
    return_state=False, go_backwards=False, stateful=False, time_major=False,
    unroll=False, **kwargs
)

The requirements to use the cuDNN implementation are:

    activation == tanh
    recurrent_activation == sigmoid
    recurrent_dropout == 0
    unroll is False
    use_bias is True
    Inputs are not masked or strictly right padded.

"""

