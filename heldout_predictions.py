"""
Perform predictions on the heldout set using all the trained models
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import argparse
import os
import pandas
import numpy as np
import glob
import code
import code
import os
import gc  # garbage collector
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation
from keras import backend as K
from tensorflow import keras
from utils import *
from helper_functions import *
from tensorflow.keras import optimizers
import glob
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from tensorflow.keras.utils import Sequence
import hydroeval as he
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras.utils import custom_object_scope
# policy = Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)
tf.random.set_seed(1234)

def swish_activation(x, beta=1):
    return K.sigmoid(beta * x) * x
tf.keras.utils.get_custom_objects().update({'swish_activation': Activation(swish_activation)})



comid_orders_dict = {
    8: [82003287, 82018072, 82018084, 82025015],
    7: [82018092, 82023524, 82028201, 82029872],
    6: [82004055, 82022185, 82023560, 82025057, 82025069, 82026688,82035594, 82037779, 82037835],
    5: [82005753, 82006612, 82008607, 82010991, 82014602, 82019471, 82019559, 82022225, 82023517, 82023555, 82025121, 82028215, 82028370, 82031602, 82034349, 82036808],
    4: [82005809, 82011035, 82011128, 82013426, 82014701, 82022245, 82025061, 82025133, 82025158, 82026787, 82026945, 82028331, 82037880, 82038962, 82039898, 82039917, 82042094, 82042445,82042777],

}
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

def create_lookup_table(base_dir, dataset_name):
    f = f'{base_dir}/1M2RTA_datasets/{dataset_name}/*.csv'
    files = glob.glob(f)
    data_dict = {}
    for file in files:
        comid = int(file.split('/').pop().split('_').pop().split('.')[0])
        data_dict.update({comid: file})
    return data_dict



'''
TO DO
lots of clean-up need to be done on this function
'''
def split_df(df, validation_split):
    split_point = int(len(df) * (1 - validation_split))
    train_df = df[:split_point]
    validation_df = df[split_point:]
    return train_df, validation_df

def load_and_process_current_station(current_station_path: str) -> pd.DataFrame:
    dataset = pd.read_csv(current_station_path)
    if dataset is not None:
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        static_df = dataset[static_features]
        static_df = minmax_scale(static_df.to_numpy(), feature_range=(0, 1), axis=1, copy=True)
        static_df = pd.DataFrame(static_df, columns=static_features)
        dataset[static_features] = static_df[static_features]
        dataset.drop(['Date'], inplace=True, axis=1)
        # dataset = dataset[better_columns_order]
        dataset = dataset[dataset['discharge'].notna()]
        dataset = dataset[dataset['discharge'] > 2.0000]
    return dataset



def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for file parameters.
    Returns:
        argparse.Namespace: Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description='File Parameters')
    parser.add_argument('--chunk_id', type=int, required=True)
    # parser.add_argument('--orders_to_drop', type=int, required=True)
    return parser.parse_args()

class TimeseriesGenerator(Sequence):
    def __init__(self, x_set, y_set, _batch_size, lookback):
        self.x, self.y = x_set, y_set
        self.batch_size = _batch_size
        self.lookback = lookback

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        batch_x = self.x[start_idx:end_idx]
        batch_y = self.y[start_idx:end_idx]

        x = np.zeros((len(batch_x), self.lookback, batch_x.shape[-1]))
        y = np.zeros((len(batch_y),))

        for i in range(len(batch_x)):
            for j in range(self.lookback if start_idx + i - self.lookback >= 0 else 1):
                x[i, -j, :] = self.x[start_idx + i - j, :]

            y[i] = self.y[start_idx + i]

        return np.array(x), np.array(y)

if __name__ == '__main__':
    args = parse_args()
    chunk_index = args.chunk_id
    # no_of_orders_to_drop = args.orders_to_drop
    # Mackenzie is an order 8 river system. If we drop 7 upstream reaches, we are left with only one reach

    base_dir = '/gypsum/eguide/projects/amuhebwa/RiversPrediction'

    model_type = 'LinearLSTM_order_8_trainedOn_6_upstream_orders'
    current_order = 8
    no_of_orders_to_drop = 2

    no_upstream_orders = 8 - no_of_orders_to_drop
    lookback_days, forecast_days = 270, 1
    batch_size = 128
    num_set_elements = 3
    if no_of_orders_to_drop != 0:
        drop_orders = orders_to_drop[no_of_orders_to_drop]
        final_features = [b for b in better_columns_orders if not any(a in b for a in drop_orders)]
        columns_to_decompose = [b for b in columns_to_decompose if not any(a in b for a in drop_orders)]
        static_features = [b for b in static_features if not any(a in b for a in drop_orders)]
    else:
        final_features = better_columns_orders

    all_models = sorted(glob.glob(f'{base_dir}/trained_models/{model_type}*.h5'))
    all_models = np.array_split(all_models, 50)
    models_list_subset = all_models[chunk_index]
    complete_dataset = create_lookup_table(base_dir, 'temp_complete_dataset')
    lookup_keys = list(complete_dataset.keys())
    all_stations_comids = comid_orders_dict.get(current_order)
    all_stations_comids = [str(i) for i in all_stations_comids]
    for model_name in models_list_subset:
        heldout_dataset = [comid for comid in all_stations_comids if comid in model_name]
        num_set_elements = len(heldout_dataset)
        nrmse_list = list()
        rbias_list = list()
        nse_list = list()
        kge_list = list()
        rsquared_list = list()
        stations_ids = list()
        try:
            for current_comid in heldout_dataset:
                print(f'Predicting for station: {current_comid}')
                # load the model
                model = tf.keras.models.load_model(model_name)
                if int(current_comid) in lookup_keys:
                    current_station_file = complete_dataset[int(current_comid)]
                    station_dataset = load_and_process_current_station(current_station_file)
                    station_dataset = station_dataset[better_columns_orders]
                    station_dataset = station_dataset[station_dataset['discharge'] > 2.000]
                    decompose_df = station_dataset[columns_to_decompose]
                    decompose_df = decompose_df.ewm(span=7, adjust=False).mean()
                    station_dataset[columns_to_decompose] = decompose_df[columns_to_decompose]
                    station_dataset = station_dataset.fillna(0)
                    scalers = {}
                    for i, current_column in enumerate(columns_to_scale):
                        current_scaler = MinMaxScaler(feature_range=(0, 1))
                        scalers['scaler_' + str(current_column)] = current_scaler
                        station_dataset[current_column] = (current_scaler.fit_transform(station_dataset[current_column].values.reshape(-1, 1))).ravel()
                        del current_scaler

                    station_dataset = station_dataset[final_features]
                    x_test, y_test = create_dataset_forecast(station_dataset.to_numpy(), lookback_days, forecast_days)
                    predicted = model.predict(x_test)
                    discharge_scaler = scalers.get('scaler_discharge')
                    predicted = discharge_scaler.inverse_transform(predicted)
                    actual = discharge_scaler.inverse_transform(y_test.reshape(-1, 1))
                    temp_results_df = pd.DataFrame()
                    temp_results_df['actual'] =actual.ravel()
                    temp_results_df['predicted'] = predicted.ravel()
                    temp_results_df = temp_results_df[temp_results_df['actual'] > 0.000]
                    # temp_results_df = temp_results_df[temp_results_df['predicted'] >= 0.000]
                    actual = temp_results_df['actual'].values
                    predicted = temp_results_df['predicted'].values

                    try:
                        nrmse = np.round(calculate_NRMSE(actual, predicted), 5)
                        rbias = np.round(calculate_RBIAS(actual, predicted), 5)
                        nse = np.round(calculate_NSE(actual, predicted), 5)
                        kge = np.round(calculate_KGE(actual, predicted), 5)
                        print(f'NRMSE: {nrmse}, RBIAS: {rbias}, NSE: {nse}, KGE: {kge} for station: {current_comid}')

                        nrmse_list.append(nrmse)
                        rbias_list.append(rbias)
                        nse_list.append(nse)
                        kge_list.append(kge)
                        stations_ids.append(current_comid)
                    except Exception as e:
                        print(f'Error: {e}')
                        continue
            results_df = pd.DataFrame()
            results_df['StationID'] = stations_ids
            results_df['Model_Name'] = model_name
            results_df['KGE'] = kge_list
            results_df['NSE'] = nse_list
            results_df['RBIAS'] = rbias_list
            results_df['NRMSE'] = nrmse_list
            name2save = f'{base_dir}/prediction_metrics/{model_type}_chunkId{chunk_index}.csv'
            results_df.to_csv(name2save, index=False)
        except Exception as e:
            print(f'Error: {e}')
            continue
