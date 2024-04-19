#!/usr/bin/env python3
import argparse
import itertools
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from common_utils import WINDOW_SIZE, forecast_days, batch_size, epochs, parent_dir, n_steps_out
from common_utils import create_dataset_forecast, load_and_process_current_station, create_model

import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from columns_utils import *
from proper_combination_of_sets import *

tf.random.set_seed(1234)

if __name__ == '__main__':
    data_dir = "lumped_complete_dataset"
    parser = argparse.ArgumentParser(description='File and Model Parameters')
    parser.add_argument('--ordernumber', required=True)
    parser.add_argument('--stationschosen', required=True)
    parser.add_argument('--nooffeatures', required=True)
    parser.add_argument('--fileidentifier', required=True)
    parser.add_argument('--lookBackPeriod', required=True)
    args = parser.parse_args()
    order_number = args.ordernumber
    stations_chosen = args.stationschosen
    no_of_features = args.nooffeatures
    file_Identifier = args.fileidentifier
    look_back_period = args.lookBackPeriod
    print('order number: ', order_number)
    print('stations_chosen: ', stations_chosen)
    print('no_of_features: ', no_of_features)
    print('file_Identifier: ', file_Identifier)
    print('Look back time', look_back_period, 'days')
    no_of_steps = int(look_back_period)
    order_number = int(order_number)
    stations_chosen = int(stations_chosen)
    no_of_features = int(no_of_features)

    current_dataset_name = "order_{}_datasets".format(str(order_number))
    current_datasets = orders_dict.get(current_dataset_name)
    current_datasets.reverse()

    executed_stations = list()
    for _, _stations2drop in enumerate(current_datasets):
        t = str(time.time()).replace('.', '_')
        train_datasets = current_datasets.copy()
        train_datasets.pop(0)
        filename2save = '_'.join(_stations2drop)
        final_order_name = "{}_for_order_{}_choose_{}_{}".format(file_Identifier, str(order_number),
                                                                 str(stations_chosen), filename2save)
        print(final_order_name)
        group_training_list = list()
        current_train_datesets = np.unique(list(itertools.chain(*train_datasets)))
        for index, station_id in enumerate(current_train_datesets):
            fname = '{}{}/StationId_{}.csv'.format(parent_dir, data_dir, station_id)
            station_dataset = load_and_process_current_station(fname)
            if station_dataset is not None:
                station_dataset = station_dataset[better_columns_order]
                station_dataset = station_dataset[station_dataset['discharge'] > 2.000]
                decompose_df = station_dataset[columns_to_decompose]
                decompose_df = decompose_df.ewm(span=7, adjust=False).mean()
                station_dataset[columns_to_decompose] = decompose_df[columns_to_decompose]
                station_dataset = station_dataset.fillna(0)
                scalers = {}
                for i, current_column in enumerate(columns_to_scale):
                    current_scaler = MinMaxScaler(feature_range=(0, 1))
                    scalers['scaler_' + str(current_column)] = current_scaler
                    station_dataset[current_column] = (
                        current_scaler.fit_transform(station_dataset[current_column].values.reshape(-1, 1))).ravel()
                    del current_scaler
            group_training_list.append(station_dataset)
            executed_stations.append(station_id)
        complete_dataset = pd.concat(group_training_list, axis=0)

        # since we are only training the model, we don't need train/test split
        df_train = complete_dataset.copy()
        n_steps_in = no_of_steps
        x_train, y_train = create_dataset_forecast(df_train.to_numpy(), n_steps_in, n_steps_out)

        ckpt_path = parent_dir + 'checkpoints/{}_{}.h5'.format(t, final_order_name)
        print('Checkpoint Name: ', ckpt_path)
        print('---CREATING NEW MODEL---')
        model = create_model(WINDOW_SIZE, no_of_steps, no_of_features, forecast_days)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor="val_loss",
                                                                 save_best_only=True, save_weights_only=True, verbose=1)

        earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        callback_list = [checkpoint_callback, earlystopping_callback]

        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callback_list,
                            validation_split=0.3, verbose=1, shuffle=False)
        # we want to load the best model that was saved, once we are done training.
        tf.keras.backend.clear_session()
        model = None
        model = create_model(WINDOW_SIZE, no_of_steps, no_of_features, forecast_days)
        model.load_weights(ckpt_path)
        model.save('{}models/{}LSTM{}_discharge_model.h5'.format(parent_dir, no_of_steps, final_order_name))
        del model
        del t
        tf.keras.backend.clear_session()
