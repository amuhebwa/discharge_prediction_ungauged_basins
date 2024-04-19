#!/usr/bin/env python3
import code
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(1234)
import pandas as pd
from numpy import array
import tensorflow as tf
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import itertools
from tensorflow.keras import backend as K
import time
import argparse
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Bidirectional
from proper_combination_of_sets import *
from columns_utils import *
from tensorflow.keras import optimizers
import numpy as np

def data_generator(file_path, _batch_size, timesteps):
    data = pd.read_csv(file_path)
    data = data.to_numpy()
    while True:
        # Adjust the range to ensure we don't exceed the array bounds
        for i in range(0, len(data) - timesteps - _batch_size + 1, _batch_size):
            X_batch = []
            y_batch = []
            end = i + _batch_size
            for j in range(i, end):
                X_batch.append(data[j:j+timesteps])
                y_batch.append(data[j+timesteps])
            yield np.array(X_batch), np.array(y_batch)


def split_df(df, validation_split):
    split_point = int(len(df) * (1 - validation_split))
    train_df = df[:split_point]
    validation_df = df[split_point:]
    # reset the index
    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    return train_df, validation_df


def create_dataset_forecast(_dataset, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(_dataset)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(_dataset):
            break
        seq_x, seq_y = _dataset[i:end_ix, :-1], _dataset[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def save_model(_model, _save_path):
    if os.path.exists(_save_path):
        os.remove(_save_path)
    _model.save(_save_path)

def create_temp_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

static_features = ['length', 'sinuosity', 'slope', 'uparea', 'max_width', 'lengthdir', 'strmDrop', 'width_mean', ]


def create_model(no_of_steps, no_of_features, forecast_days):
    WINDOW_SIZE = 20
    model = tf.keras.Sequential()
    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True), input_shape=(no_of_steps, no_of_features)))
    model.add(Dropout(rate=0.2))
    model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
    model.add(Dropout(rate=0.2))
    model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
    model.add(Dropout(rate=0.2))
    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))
    model.add(Dense(units=forecast_days))
    model.add(Activation('swish'))
    optzr = optimizers.RMSprop(learning_rate=0.0001, centered=False)
    model.compile(loss='mse', optimizer=optzr)
    return model

if __name__ == '__main__':
    start_time = time.time()
    base_dir = "/gypsum/eguide/projects/amuhebwa/rivers_ML"
    parser = argparse.ArgumentParser(description='File and Model Parameters')
    parser.add_argument('--ordernumber', required=True)
    parser.add_argument('--set_index', required=True)
    args = parser.parse_args()
    order_number = args.ordernumber
    set_index = args.set_index
    order_number = int(order_number)
    set_index = int(set_index)

    print('order number: ', order_number)

    '''
    which orders do we have ?
    orders: 4, 5, 6, 7, 8
    '''
    # ===========================
    batch_size = 128
    epochs = 100
    WINDOW_SIZE = 20
    lookback_days = 270
    forecast_days = 1
    num_of_features = None
    experiment_name = "distributedExperiment"

    # set up proper columns based on orders 
    current_order_dict = orders_lookuptable.get(str(order_number))
    better_columns_order = current_order_dict.get('better_columns_order')
    columns_to_scale = current_order_dict.get('columns_to_scale')

    current_dataset_name = "order_{}_datasets".format(str(order_number))
    current_order_datasets = orders_dict.get(current_dataset_name)

    heldout_dataset = current_order_datasets[set_index]


    current_order_datasets.remove(heldout_dataset)
    training_dataset = np.unique(list(itertools.chain(*current_order_datasets)))

    """
    Load mean and standard deviations global stats
    """
    stats_path = f"{base_dir}/mean_stdDV_allOrders_stats/{experiment_name}_order_{order_number}_mean_stdDv.csv"
    mackenzie_stats_df = None
    if os.path.exists(stats_path):
        mackenzie_stats_df = pd.read_csv(stats_path)

    # concatennate stations to get the file identifier
    unique_id = '_'.join(heldout_dataset)
    checkpoint_path = f"{base_dir}/checkpoints/{experiment_name}_order_{order_number}_{unique_id}.h5"

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_callbacks = [checkpoint_callback, earlystopping_callback]
    # create a temporary directory to store the datasets
    train_temp_dir = f"{base_dir}/temp_datasets/train_temp_{str(start_time).replace('.', '')}_orders_{order_number}_datasets/"
    validate_temp_dir = f"{base_dir}/temp_datasets/train_temp_{str(start_time).replace('.', '')}_orders_{order_number}_datasets/"
    create_temp_dir(train_temp_dir)
    create_temp_dir(validate_temp_dir)

    for idx, station_comid in enumerate(training_dataset):
        filepath = f"{base_dir}/distributed_complete_dataset/StationId_{station_comid}.csv"
        # check if the file exists
        if os.path.exists(filepath):
            current_dataset = pd.read_csv(filepath)
            current_dataset = current_dataset.drop(columns=['Date'])
            current_dataset = current_dataset[better_columns_order]
            current_dataset = current_dataset[current_dataset['discharge'].notna()]

            current_dataset = current_dataset.fillna(0)

            static_df = current_dataset[static_features]
            # add epsilon to avoid division by zero
            static_df = static_df + 1e-6
            static_df = minmax_scale(static_df.to_numpy(), feature_range=(0, 1), axis=1, copy=True)
            static_df = pd.DataFrame(static_df, columns=static_features)
            current_dataset[static_features] = static_df[static_features]

            current_dataset = current_dataset.fillna(0)
            scalers = {}
            for i, current_column in enumerate(columns_to_scale):
                current_scaler = MinMaxScaler(feature_range=(0, 1))
                scalers['scaler_' + str(current_column)] = current_scaler
                current_dataset[current_column] = (
                    current_scaler.fit_transform(current_dataset[current_column].values.reshape(-1, 1))).ravel()
                del current_scaler
            # split the dataset into train and validation sets
            current_dataset = current_dataset[better_columns_order]
            num_of_features = len(better_columns_order) - 1  # don't count discharge
            train_dataset, validate_dataset = split_df(current_dataset, validation_split=0.3)
            train_file_name = f"{train_temp_dir}/StationId_{station_comid}.csv"
            valid_file_name = f"{validate_temp_dir}/StationId_{station_comid}.csv"
            train_dataset.to_csv(train_file_name, index=False)
            validate_dataset.to_csv(valid_file_name, index=False)

    # create a function that loads the data from the temp directory
    train_dataset_files = sorted([os.path.join(train_temp_dir, f) for f in os.listdir(train_temp_dir) if f.endswith('.csv')])
    validate_dataset_files = sorted([os.path.join(validate_temp_dir, f) for f in os.listdir(validate_temp_dir) if f.endswith('.csv')])


    # create a new model
    model = create_model(lookback_days, num_of_features, forecast_days)
    # if best weights exist, load them
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    for train_path, validation_path in zip(train_dataset_files, validate_dataset_files):
        print(f"Training on {train_path} and validating on {validation_path}")
        train_steps_per_epoch = (len(pd.read_csv(train_path)) - lookback_days) // batch_size
        val_steps_per_epoch = (len(pd.read_csv(validation_path)) - lookback_days) // batch_size

        # create train and validation generators
        train_gen = data_generator(train_path, batch_size, lookback_days)
        val_gen = data_generator(validation_path, batch_size, lookback_days)
        model.fit(
            train_gen, validation_data=val_gen, steps_per_epoch=train_steps_per_epoch, validation_steps=val_steps_per_epoch,
            epochs=epochs, verbose=1, shuffle=False, callbacks=model_callbacks
        )
        # Reset the model states to prevent the model from making continuity assumptions across datasets.
        model.reset_states()
        print("Resetting the model states...")
    # ======================================================================
    # load the best model
    best_model = create_model(lookback_days, num_of_features, forecast_days)
    best_model.load_weights(checkpoint_path)
    final_name2save = f'{base_dir}/trained_models/{experiment_name}_SeqLSTM_order_{order_number}_{unique_id}_model.h5'
    save_model(best_model, final_name2save)
    print(f"Training took {time.time() - start_time} seconds")
    gc.collect()
    tf.keras.backend.clear_session()
