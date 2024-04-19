"""
models and custom loss functions
Note: reach out to Fahaad to chat more about the choice of loss functions
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation


@tf.keras.utils.register_keras_serializable(package='Custom', name='rmseLoss')
def rmseLoss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


"""
@tf.keras.utils.register_keras_serializable(package='Custom', name='nseLoss')
def nseLoss(y_obs, y_pred):
    numerator = tf.reduce_sum(tf.square(y_pred - y_obs))
    denominator = tf.reduce_sum(tf.square(y_obs - tf.reduce_mean(y_obs)))
    nse_value = 1 - (numerator / denominator)
    # Negative because we want to maximize NSE (i.e., minimize -NSE)
    return -nse_value
"""


@tf.keras.utils.register_keras_serializable(package='Custom', name='nseLoss')
def nseLoss(y_true, y_pred):
    # Replace NaN and infinity values in y_true and y_pred
    y_true = tf.where(tf.math.is_finite(y_true), y_true, 0.0)
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, 0.0)
    mean_observed = tf.reduce_mean(y_true)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    var_observed = tf.reduce_mean(tf.square(y_true - mean_observed))
    # Avoid division by zero
    var_observed = tf.math.maximum(var_observed, tf.keras.backend.epsilon())
    # Compute NSE Loss
    nse_loss_val = mse / var_observed
    return nse_loss_val


@tf.keras.utils.register_keras_serializable()
def rbias(y_true, y_pred):
    """Relative Bias metric."""
    # Replace infinities with NaNs
    y_true = tf.where(tf.math.is_inf(y_true), tf.constant(np.nan, dtype=tf.float32), y_true)
    y_pred = tf.where(tf.math.is_inf(y_pred), tf.constant(np.nan, dtype=tf.float32), y_pred)
    # Remove NaNs from y_true and y_pred
    y_true = tf.boolean_mask(y_true, tf.math.is_finite(y_true))
    y_pred = tf.boolean_mask(y_pred, tf.math.is_finite(y_pred))
    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)
    rbias = (mean_pred - mean_true) / mean_true
    return rbias


# Define KGE metric
@tf.keras.utils.register_keras_serializable()
def kge(y_true, y_pred):
    """Kling-Gupta Efficiency metric."""
    # Replace infinities with NaNs
    y_true = tf.where(tf.math.is_inf(y_true), tf.constant(np.nan, dtype=tf.float32), y_true)
    y_pred = tf.where(tf.math.is_inf(y_pred), tf.constant(np.nan, dtype=tf.float32), y_pred)
    # Remove NaNs from y_true and y_pred
    y_true = tf.boolean_mask(y_true, tf.math.is_finite(y_true))
    y_pred = tf.boolean_mask(y_pred, tf.math.is_finite(y_pred))
    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)
    std_true = K.std(y_true)
    std_pred = K.std(y_pred)
    covar = K.mean((y_true - mean_true) * (y_pred - mean_pred))
    kge = 1 - (covar / (std_true * std_pred))
    return kge


# Define NSE metric
@tf.keras.utils.register_keras_serializable()
def nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency metric."""
    # Replace infinities with NaNs
    y_true = tf.where(tf.math.is_inf(y_true), tf.constant(np.nan, dtype=tf.float32), y_true)
    y_pred = tf.where(tf.math.is_inf(y_pred), tf.constant(np.nan, dtype=tf.float32), y_pred)
    # Remove NaNs from y_true and y_pred
    y_true = tf.boolean_mask(y_true, tf.math.is_finite(y_true))
    y_pred = tf.boolean_mask(y_pred, tf.math.is_finite(y_pred))
    numerator = K.sum(K.square(y_true - y_pred))
    denominator = K.sum(K.square(y_true - K.mean(y_true)))
    nse = 1 - (numerator / denominator)
    return nse


def create_model(no_lookback_days: int, _forecast_days: int, no_of_features):
    WINDOW_SIZE = 20
    model = tf.keras.Sequential()
    model.add(LSTM(WINDOW_SIZE, return_sequences=True, input_shape=(no_lookback_days, no_of_features)))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(WINDOW_SIZE * 2, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(WINDOW_SIZE * 2, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(WINDOW_SIZE, return_sequences=False))
    model.add(Dense(units=_forecast_days))
    model.add(Activation('swish'))
    optzr = optimizers.RMSprop(learning_rate=1e-3, momentum=0.9, centered=False)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optzr)
    return model
