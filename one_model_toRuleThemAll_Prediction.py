import code
import os
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import tensorflow as tf
# import torch
import random
import re
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model

tf.random.set_seed(1234)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, Activation
from tensorflow.keras import optimizers
from helper_functions import *
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import pandas as pd
import copy

from utils import *
from allModels import *

tf.keras.mixed_precision.set_global_policy('mixed_float16')

_seed = 1234
tf.random.set_seed(_seed)
random.seed(_seed)
np.random.seed(_seed)

"""
@tf.keras.utils.register_keras_serializable(package='Custom', name='nseLoss')
def nseLoss(y_obs, y_pred):
    # Computes the negative Nash-Sutcliffe Efficiency to be used as a loss function
    
    numerator = tf.reduce_sum(tf.square(y_pred - y_obs))
    denominator = tf.reduce_sum(tf.square(y_obs - tf.reduce_mean(y_obs)))
    nse_value = 1 - (numerator / denominator)
    # Negative because we want to maximize NSE (i.e., minimize -NSE)
    return -nse_value

@tf.keras.utils.register_keras_serializable()
def rbias(y_true, y_pred):
    # Relative Bias metric.
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
"""


# Define KGE metric
@tf.keras.utils.register_keras_serializable()
def kge(y_true, y_pred):
    # Kling-Gupta Efficiency metric.

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
    # Nash-Sutcliffe Efficiency metric.

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


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for file parameters.
    Returns:
        argparse.Namespace: Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description='File Parameters')
    parser.add_argument('--set_index', type=int, required=True)
    parser.add_argument('--set_elements', type=int, required=True)
    parser.add_argument('--orders_to_drop', type=int, required=True)
    return parser.parse_args()


def data_generator(x, y, _batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(_batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    base_dir = '/gypsum/eguide/projects/amuhebwa/RiversPrediction'
    complete_dataset_dict = create_lookup_table(base_dir, 'complete_dataset')
    all_stations_comids = list(complete_dataset_dict.keys())
    args = parse_args()
    set_index = args.set_index
    # set_elements = args.set_elements
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
    num_of_features = len(better_columns_orders) - 1
    mackenzie_stats_df = pd.read_csv(f'{base_dir}/1M2RTA_datasets/mackenzie_basin_global_mean_stdDv.csv')

    experiment_name = "Orders8To2"
    model_type = f"{experiment_name}_SeqLSTM_setsOf20stations"

    all_models = sorted(glob.glob(f'{base_dir}/trained_models/{model_type}*.h5'))
    all_models = np.array_split(all_models, 50)
    all_models = [model_arr for model_arr in all_models if len(model_arr) != 0]
    models_list_subset = all_models[set_index]

    all_stations_comids = [str(i) for i in all_stations_comids]

    for model_name in models_list_subset:
        # get the comids of the stations that were used to train the model. maintain their order
        # since we will use these to construct the name of the statistics table
        train_dataset = re.findall(r'\d{8,}', model_name)
        unique_id = '_'.join([str(s) for s in train_dataset])
        # get all the stations that are not in the heldout dataset
        heldout_dataset = [comid for comid in all_stations_comids if comid not in train_dataset]
        # heldout_dataset = all_stations_comids.copy()
        nrmse_list = list()
        rbias_list = list()
        nse_list = list()
        kge_list = list()
        rsquared_list = list()
        stations_ids = list()

        for station_comid in heldout_dataset:
            model = create_model(lookback_days, forecast_days, num_of_features)
            model.load_weights(model_name)
            # model = load_model(model_name)

            station_filepath = complete_dataset_dict[int(station_comid)]
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
            df_test_dataset = current_dataset[better_columns_orders]

            try:
                x_test_ds, y_test_ds = create_dataset_forecast(df_test_dataset.to_numpy(), lookback_days, forecast_days)
                predicted = model.predict(x_test_ds)

                #if experiment_name == "HuberAdadeltaNormalizedDischarge":
                #    discharge_mean = stats_df[stats_df['Feature'] == 'discharge']['mean'].values[0]
                #    discharge_std = stats_df[stats_df['Feature'] == 'discharge']['std'].values[0]
                #    predicted = (predicted * discharge_std) + discharge_mean
                #    y_test_ds = (y_test_ds * discharge_std) + discharge_mean

                predicted_discharge = predicted.reshape(-1, 1)
                observed_discharge = y_test_ds.reshape(-1, 1)

                nonzero_mask = (predicted_discharge >= 0) & (observed_discharge >= 0)
                observed_discharge = observed_discharge[nonzero_mask]
                predicted_discharge = predicted_discharge[nonzero_mask]
                nse = np.round(calculate_NSE(observed_discharge, predicted_discharge), 5)
                kge = np.round(calculate_KGE(observed_discharge, predicted_discharge), 5)
                nrmse = np.round(calculate_NRMSE(observed_discharge, predicted_discharge), 5)
                rbias = np.round(calculate_RBIAS(observed_discharge, predicted_discharge), 5)
                print(f'NRMSE: {nrmse}, RBIAS: {rbias}, NSE: {nse}, KGE: {kge} for station: {station_comid}')
                nrmse_list.append(nrmse)
                rbias_list.append(rbias)
                nse_list.append(nse)
                kge_list.append(kge)
                stations_ids.append(station_comid)
                del model, current_dataset, df_test_dataset, x_test_ds, y_test_ds, predicted, predicted_discharge, observed_discharge
                gc.collect()
            except Exception as e:
                print(f'Error: {e} for station: {station_comid}')
                continue
        results_df = pd.DataFrame()
        results_df['StationID'] = stations_ids
        results_df['Model_Name'] = model_name
        results_df['KGE'] = kge_list
        results_df['NSE'] = nse_list
        results_df['RBIAS'] = rbias_list
        results_df['NRMSE'] = nrmse_list
        name2save = f'{base_dir}/prediction_metrics/{model_type}_chunkId{set_index}.csv'
        results_df.to_csv(name2save, index=False)
        del results_df
        gc.collect()
