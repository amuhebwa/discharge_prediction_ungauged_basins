#!/usr/bin/env python3
import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(1234)
import pandas as pd
from numpy import array
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.models import load_model
import itertools
from helper_functions import *

import time
import argparse

from proper_combination_of_sets import *
from columns_utils import *
from allModels import *

"""
custom data generator to efficiently load data into memory
"""


def data_generator(x, y, _batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(_batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


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


static_features = ['length', 'sinuosity', 'slope', 'uparea', 'max_width', 'lengthdir', 'strmDrop', 'width_mean', ]

if __name__ == '__main__':
    start_time = time.time()
    base_dir = "/gypsum/eguide/projects/amuhebwa/rivers_ML"
    parser = argparse.ArgumentParser(description='File and Model Parameters')
    parser.add_argument('--ordernumber', required=True)
    args = parser.parse_args()
    order_number = args.ordernumber
    order_number = int(order_number)

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

    current_order_all_stations = np.unique(list(itertools.chain(*current_order_datasets)))
    # get the difference between the two sets
    # training_dataset = list(set(current_order_all_stations) - set(heldout_setofStations))
    trained_models_dir = glob.glob(f"{base_dir}/trained_models/{experiment_name}_SeqLSTM_order_{order_number}_*.h5")
    for model_name in trained_models_dir:
        model = load_model(model_name)
        heldout_dataset = substrings_found = [element for element in current_order_all_stations if element in model_name]
        for test_station_comid in heldout_dataset:
            filepath = f"{base_dir}/distributed_complete_dataset/StationId_{test_station_comid}.csv"
            if os.path.exists(filepath):
                current_dataset = pd.read_csv(filepath)
                current_dataset = current_dataset.drop(columns=['Date'])
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
                complete_dataset = current_dataset[better_columns_order]
                num_of_features = len(better_columns_order) - 1  # don't count discharge
                # split the dataset into train and validation sets
                print(f"SIZE OF DATASET = {complete_dataset.shape}")

            try:
                x_test, y_test = create_dataset_forecast(complete_dataset.to_numpy(), lookback_days, forecast_days)
                predicted = model.predict(x_test)
                predicted_discharge = predicted.reshape(-1, 1)
                observed_discharge = y_test.reshape(-1, 1)
                nonzero_mask = (predicted_discharge >= 0) & (observed_discharge >= 0)
                observed_discharge = observed_discharge[nonzero_mask]
                predicted_discharge = predicted_discharge[nonzero_mask]
                nse = np.round(calculate_NSE(observed_discharge, predicted_discharge), 5)
                kge = np.round(calculate_KGE(observed_discharge, predicted_discharge), 5)
                nrmse = np.round(calculate_NRMSE(observed_discharge, predicted_discharge), 5)
                rbias = np.round(calculate_RBIAS(observed_discharge, predicted_discharge), 5)
                print(f'NRMSE: {nrmse}, RBIAS: {rbias}, NSE: {nse}, KGE: {kge} for station: {test_station_comid}')

            except Exception as e:
                print(f"Error: {e}")
                continue
