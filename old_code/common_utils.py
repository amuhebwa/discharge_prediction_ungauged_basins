import itertools
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Activation, Bidirectional, Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from columns_utils import *
tf.random.set_seed(1234)
# common variables
parent_dir = "/mnt/nfs/eguide/projects/amuhebwa/RiversPrediction/StreamOrders_FinalExperiments/"
n_steps_in = 270  # lookback window
n_steps_out = 1
stations_chosen = "3"
batch_size = 128
epochs = 200
WINDOW_SIZE = 20
forecast_days = 1


def load_specific_file(fname: str) -> pd.DataFrame:
    temp_df = None
    temp_df = pd.read_csv(fname)
    return temp_df


def load_and_process_current_station(filename: str) -> pd.DataFrame:
    dataset = load_specific_file(filename)
    if dataset is not None:
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        static_df = dataset[static_cols]
        static_df = minmax_scale(static_df.to_numpy(), feature_range=(0, 1), axis=1, copy=True)
        static_df = pd.DataFrame(static_df, columns=static_cols)
        dataset[static_cols] = static_df[static_cols]
        dataset.drop(['Date'], inplace=True, axis=1)
        dataset = dataset[dataset['discharge'].notna()]
    return dataset


def create_dataset_forecast(_dataset, n_steps_in: int, n_steps_out: int):
    X, y = list(), list()
    for i in range(len(_dataset)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(_dataset):
            break
        seq_x, seq_y = _dataset[i:end_ix, :-1], _dataset[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def possible_combinations(stations_arr: np.array(), r: int):
    return list(itertools.combinations(stations_arr, r))


def swish_activation(x, beta=1):
    return K.sigmoid(beta * x) * x


tf.keras.utils.get_custom_objects().update({'swish_activation': Activation(swish_activation)})


def create_model(WINDOW_SIZE: int, no_of_steps: int, no_of_features: int, forecast_days: int) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True), input_shape=(no_of_steps, no_of_features)))
    model.add(Dropout(rate=0.2))
    model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
    model.add(Dropout(rate=0.2))
    model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
    model.add(Dropout(rate=0.2))
    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))
    model.add(Dense(units=forecast_days))
    model.add(Activation(swish_activation, name="swish"))
    optzr = optimizers.RMSprop(learning_rate=0.0001, centered=False)
    model.compile(loss='mse', optimizer=optzr)
    return model
