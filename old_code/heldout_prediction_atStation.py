#!/usr/bin/env python3
import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model
from hydrology_functions import *
from common_utils import *

tf.random.set_seed(1234)


def swish_activation(x, beta=1):
    return K.sigmoid(beta * x) * x


tf.keras.utils.get_custom_objects().update({'swish_activation': Activation(swish_activation)})

if __name__ == "__main__":
    data_dir = "atstation_complete_dataset"
    Identifier = "270LSTMatStation"
    model_collection = glob.glob("{}models/{}*choose_{}_*.h5".format(parent_dir, Identifier, stations_chosen))
    dataset_path = glob.glob("{}{}/*.csv".format(parent_dir, data_dir))
    dataset_dict = {}
    for _, _path in enumerate(dataset_path):
        stationId = _path.split('/').pop().split('_').pop().split('.')[0]
        dataset_dict.update({stationId: _path})

    rrmse_list = list()
    nrmse_list = list()
    rbias_list = list()
    nse_list = list()
    kge_list = list()
    rsquared_list = list()
    stations_classes_list = list()
    stations_ids = list()
    model_names_list = list()

    for _, model_name in enumerate(model_collection):
        model = load_model(model_name)
        temp_arr = model_name.split('/').pop().split('_')
        if temp_arr[0] == Identifier:
            class_name = temp_arr[3]
            _stations = np.asarray([temp_arr[6], temp_arr[7], temp_arr[8]])
            print(temp_arr)
            for _, stationID in enumerate(_stations):
                dataset_name = dataset_dict.get(stationID)
                dataset = pd.read_csv(dataset_name)
                print('Dataset Name: {}'.format(dataset_name))
                static_df = dataset[static_cols]
                static_df = minmax_scale(static_df.to_numpy(), feature_range=(0, 1), axis=1, copy=True)
                static_df = pd.DataFrame(static_df, columns=static_cols)
                dataset[static_cols] = static_df[static_cols]
                dataset.drop(['Date'], inplace=True, axis=1)
                dataset = dataset[dataset['discharge'].notna()]
                dataset = dataset[better_columns_order]
                decompose_df = dataset[columns_to_decompose]
                decompose_df = decompose_df.ewm(span=7, adjust=False).mean()
                dataset[columns_to_decompose] = decompose_df[columns_to_decompose]
                dataset = dataset.fillna(0)

                scalers = {}
                for i, current_column in enumerate(columns_to_scale):
                    current_scaler = MinMaxScaler(feature_range=(0, 1))
                    scalers['scaler_' + str(current_column)] = current_scaler
                    dataset[current_column] = (
                        current_scaler.fit_transform(dataset[current_column].values.reshape(-1, 1))).ravel()
                    del current_scaler

                df_test = dataset.copy()
                x_test, y_test = create_dataset_forecast(df_test.to_numpy(), n_steps_in, n_steps_out)
                predicted = model.predict(x_test)
                discharge_scaler = scalers.get('scaler_discharge')
                actual = discharge_scaler.inverse_transform(y_test.reshape(-1, 1))
                predicted = discharge_scaler.inverse_transform(predicted)
                temp_results_df = pd.DataFrame()
                temp_results_df['actual'] = actual.ravel()
                temp_results_df['predicted'] = predicted.ravel()
                actual = temp_results_df['actual'].values
                predicted = temp_results_df['predicted'].values
                nrmse = np.round(calculate_NRMSE(actual, predicted), 5)
                rbias = np.round(calculate_RBIAS(actual, predicted), 5)
                nse = np.round(calculate_NSE(actual, predicted), 5)
                kge = np.round(calculate_KGE(actual, predicted), 5)
                rsquared = np.round(calculate_RSquared(actual, predicted), 5)
                rel_rmse = np.round(calculate_RRMSE(actual, predicted), 5)
                nrmse_list.append(nrmse)
                rrmse_list.append(rel_rmse)
                rbias_list.append(rbias)
                nse_list.append(nse)
                kge_list.append(kge)
                # rsquared_list.append(rsquared)
                stations_classes_list.append(class_name)
                stations_ids.append(stationID)
                model_names_list.append(model_name)
        tf.keras.backend.clear_session()
        del model

    results_df = pd.DataFrame()
    results_df['StationID'] = stations_ids
    results_df['Model_Name'] = model_names_list
    results_df['RRMSE'] = rrmse_list
    results_df['KGE'] = kge_list
    results_df['NSE'] = nse_list
    results_df['RBIAS'] = rbias_list
    results_df['NRMSE'] = nrmse_list
    results_df['Order'] = stations_classes_list
    results_df.to_csv('{}{}_heldOut_results.csv'.format(parent_dir, Identifier), index=False)
    print('Done')
