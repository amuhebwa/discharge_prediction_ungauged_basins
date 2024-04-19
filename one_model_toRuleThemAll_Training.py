import argparse
import copy
import gc
import glob
import os
import time

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from allModels import *
from utils import *

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ============================================
_seed = 1234
np.random.seed(_seed)
# tensorflow random seed
tf.random.set_seed(_seed)


# ============================================

def split_df(df, validation_split):
    split_point = int(len(df) * (1 - validation_split))
    train_df = df[:split_point]
    validation_df = df[split_point:]
    # reset the index
    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    return train_df, validation_df


def create_lookup_table(parent_path, dataset_name):
    f = f'{parent_path}/1M2RTA_datasets/{dataset_name}/*.csv'
    files = glob.glob(f)
    data_dict = {}
    for file in files:
        comid = int(file.split('/').pop().split('_').pop().split('.')[0])
        data_dict.update({comid: file})
    return data_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='File Parameters')
    parser.add_argument('--set_index', type=int, required=True)
    parser.add_argument('--set_elements', type=int, required=True)
    parser.add_argument('--orders_to_drop', type=int, required=True)
    return parser.parse_args()


comid_orders_dict = {
    1: [82009553],
    2: [82028537, 82037195, 82042787],
    3: [82014643, 82039999, 82003408, 82009803, 82036989, 82022492, 82042991, 82033417, 82042475, 82034453, 82042176, 82031898],
    4: [82005809, 82022245, 82042445, 82011128, 82041599, 82025133, 82039917, 82008688, 82025158, 82017021, 82039898, 82042777, 82042452, 82026787, 82030103, 82041598, 82014701,
        82042094, 82037880, 82042085, 82037823, 82042102, 82038875, 82011035, 82026945, 82042459, 82025061, 82029883, 82028395, 82028331, 82038962, 82039882, 82017051, 82026827],
    5: [82036808, 82039908, 82031602, 82012264, 82028307, 82041573, 82004876, 82008607, 82014602, 82019471, 82038860, 82036797, 82034349, 82028370, 82041584, 82017050, 82022225,
        82010991, 82028215, 82019559, 82023517, 82025121],
    6: [82026688, 82035594, 82025057, 82029922, 82022185, 82023560, 82037779, 82015719, 82037800, 82025069, 82037835, 82019447, 82037816, 82037818, 82038832, 82040772],
    7: [82029872, 82018092, 82036781, 82028201, 82023524],
    8: [82018084, 82003287, 82025015, 82008543, 82018072]
}


def organize_stations_by_order(curr_orders_list):
    ordered_comids = []
    mackenzie_orders = [8, 7, 6, 5, 4, 3, 2, 1]
    for key in mackenzie_orders:
        # get intersection of two lists
        temp_result = list(set(curr_orders_list).intersection(comid_orders_dict[key]))
        if temp_result:
            print(f"Found intersection for order {key} == {temp_result}")
            ordered_comids += temp_result
    return ordered_comids


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


def preprocess_dataset(dataset_df, stat_features, cols_to_scale, cat_features):
    dataset_df = dataset_df[better_columns_orders]
    dataset_df = dataset_df[dataset_df['discharge'].notna()]
    static_df = dataset_df[stat_features]
    static_df += (np.random.rand(*static_df.shape)) * 0.01  # add a small amount of noise to the data
    static_df = minmax_scale(static_df.to_numpy(), feature_range=(0, 1), axis=1, copy=True)
    static_df = pd.DataFrame(static_df, columns=stat_features)
    dataset_df[stat_features] = static_df[stat_features]
    # convert orders column to categorical
    dataset_df[cat_features] = dataset_df[cat_features].astype('category')
    scalers = dict()
    for i, current_column in enumerate(cols_to_scale):
        current_scaler = MinMaxScaler(feature_range=(0, 1))
        scalers['scaler_' + str(current_column)] = current_scaler
        dataset_df[current_column] = (current_scaler.fit_transform(dataset_df[current_column].values.reshape(-1, 1))).ravel()

    dataset_df = dataset_df[better_columns_orders]
    return dataset_df, scalers


def save_model(_model, _save_path):
    if os.path.exists(_save_path):
        os.remove(_save_path)
    _model.save(_save_path)


"""
custom data generator to efficiently load data into memory
"""


def data_generator(x, y, _batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(_batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    start_time = time.time()
    base_dir = '/gypsum/eguide/projects/amuhebwa/RiversPrediction'
    temp_data_dir = f'{base_dir}/TEMP_DATA_STORAGE'
    complete_dataset_dict = create_lookup_table(base_dir, 'complete_dataset')
    args = parse_args()
    set_index = args.set_index
    set_elements = args.set_elements
    no_of_orders_to_drop = args.orders_to_drop
    # custom_metrics = [kge, rbias]

    """
    TEMP FIX: DROPPING SOME OF THE COLUMNS THAT WE DON'T NEED TO REDUCE COMPUTATIONAL REQUIREMENTS
    """
    columns_to_scale = [col for col in columns_to_scale if not any(drop in col for drop in columns_to_drop)]
    better_columns_orders = [col for col in better_columns_orders if not any(drop in col for drop in columns_to_drop)]

    if no_of_orders_to_drop != 0:
        drop_orders = orders_to_drop[no_of_orders_to_drop]
        better_columns_orders = [b for b in better_columns_orders if not any(a in b for a in drop_orders)]
        static_features = [b for b in static_features if not any(a in b for a in drop_orders)]
    else:
        better_columns_orders = better_columns_orders

    batch_size = 32
    lookback_days, forecast_days = 270, 1
    epochs = 100
    num_of_features = len(better_columns_orders) - 1

    experiment_name = "Orders8To2"

    mackenzie_stats_df = pd.read_csv(f'{base_dir}/1M2RTA_datasets/mackenzie_basin_global_mean_stdDv.csv')

    sets_combinations = sorted(stations_sets_dict[set_elements])
    # reverse sets_combinations
    # sets_combinations = sets_combinations[::-1]
    stations4training = sets_combinations[set_index]

    unique_id = '_'.join([str(s) for s in stations4training])

    '''
    organize stations by order.
    '''
    stations4training = organize_stations_by_order(stations4training)

    checkpoint_path = f'{base_dir}/checkpoints/{experiment_name}_SeqLSTM_setsOf{set_elements}stations_{unique_id}_model.h5'
    # if there's a previous checkpoint, remove it
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    # Define callbacks
    model_callbacks = [EarlyStopping(monitor='val_loss', patience=10), ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)]
    for idx, station_comid in enumerate(stations4training):
        print(f"Training model for station {station_comid} ({idx + 1}/{len(stations4training)})")
        if station_comid in complete_dataset_dict:
            station_filepath = complete_dataset_dict[station_comid]
            current_dataset = pd.read_csv(station_filepath)

            current_dataset = current_dataset[better_columns_orders]
            current_dataset = current_dataset[current_dataset['discharge'].notna()]

            static_df = current_dataset[static_features]
            static_df += (np.random.rand(*static_df.shape)) * 1e-3

            current_dataset[static_features] = static_df[static_features]

            '''
            We won't normalize discharge with global mean and standard deviation
            '''
            temp_columns_to_scale = copy.deepcopy(columns_to_scale)
            temp_columns_to_scale.remove('discharge')
            # Create the MinMaxScaler
            current_scaler = MinMaxScaler(feature_range=(0, 1))
            # Fit and transform the specified column
            current_dataset["discharge"] = current_scaler.fit_transform(current_dataset['discharge'].values.reshape(-1, 1)).ravel()
            # get global mean and standard deviations
            stats_df = mackenzie_stats_df[mackenzie_stats_df['Feature'].isin(temp_columns_to_scale)]
            for _, col2scale in enumerate(temp_columns_to_scale):
                current_mean = stats_df[stats_df['Feature'] == col2scale]['mean'].values[0]
                current_std = stats_df[stats_df['Feature'] == col2scale]['std'].values[0]
                current_dataset.loc[:, col2scale] = (current_dataset[col2scale] - current_mean) / current_std
            complete_dataset = current_dataset[better_columns_orders]

            # split the dataset into train and validation sets
            print(f"SIZE OF DATASET = {complete_dataset.shape}")
            train_dataset, validate_dataset = split_df(complete_dataset, validation_split=0.3)
            x_train, y_train = create_dataset_forecast(train_dataset.to_numpy(), lookback_days, forecast_days)
            x_validate, y_validate = create_dataset_forecast(validate_dataset.to_numpy(), lookback_days, forecast_days)

            # Create data generators for training and validation
            train_gen = data_generator(x_train, y_train, batch_size)
            validate_gen = data_generator(x_validate, y_validate, batch_size)

            # create a new model
            model = create_model(lookback_days, forecast_days, num_of_features)
            # if best weights exist, load them
            if os.path.exists(checkpoint_path):
                model.load_weights(checkpoint_path)
            model.fit(train_gen, validation_data=validate_gen, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=1, callbacks=model_callbacks)

            # ======================================================================
            # Reset the model states to prevent the model from making continuity assumptions across datasets.
            model.reset_states()
            print("Resetting the model states...")
            # ======================================================================
            del complete_dataset, train_dataset, validate_dataset, x_train, y_train, x_validate, y_validate, train_gen, validate_gen
            gc.collect()

    # load the best model
    best_model = create_model(lookback_days, forecast_days, num_of_features)
    best_model.load_weights(checkpoint_path)
    final_name2save = f'trained_models/{experiment_name}_SeqLSTM_setsOf{set_elements}stations_{unique_id}_model.h5'
    save_model(best_model, final_name2save)
    print(f"Training took {time.time() - start_time} seconds")
    gc.collect()
