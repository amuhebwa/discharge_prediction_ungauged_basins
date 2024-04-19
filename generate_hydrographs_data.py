"""
Create data for plotting hydrographs
"""
import code
import glob

from sklearn.preprocessing import minmax_scale, MinMaxScaler
from tensorflow.keras.models import load_model

from helper_functions import *
from utils import *
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation

def swish_activation(x, beta=1):
    return K.sigmoid(beta * x) * x
tf.keras.utils.get_custom_objects().update({'swish_activation': Activation(swish_activation)})

def create_lookup_table(base_dir, dataset_name):
    f = f'{base_dir}/1M2RTA_datasets/{dataset_name}/*.csv'
    files = glob.glob(f)
    data_dict = {}
    for file in files:
        comid = int(file.split('/').pop().split('_').pop().split('.')[0])
        data_dict.update({comid: file})
    return data_dict


def prepare_dataset(current_dataset):
    current_dataset['Date'] = pd.to_datetime(current_dataset['Date'])
    # convert datetime to cyclic features
    current_dataset['day_sin'] = np.sin(2 * np.pi * current_dataset['Date'].dt.dayofyear / 365.25)
    current_dataset['day_cos'] = np.cos(2 * np.pi * current_dataset['Date'].dt.dayofyear / 365.25)
    current_dataset['month_sin'] = np.sin(2 * np.pi * current_dataset['Date'].dt.month / 12)
    current_dataset['month_cos'] = np.cos(2 * np.pi * current_dataset['Date'].dt.month / 12)
    # then drop the comid and date columns
    current_dataset.drop(['Date'], axis=1, inplace=True)
    # normalize static features
    static_df = current_dataset[static_features]
    static_df = minmax_scale(static_df.to_numpy(), feature_range=(0, 1), axis=1, copy=True)
    static_df = pd.DataFrame(static_df, columns=static_features)
    current_dataset[static_features] = static_df[static_features]
    # convert categorical features to integers
    current_dataset[categorical_features] = current_dataset[categorical_features].astype('int')
    return current_dataset


def scale_dynamic_features(current_dataset, current_scaler):
    scalers = {}
    for i, current_column in enumerate(dynamic_features):
        scalers['scaler_' + str(current_column)] = current_scaler
        current_dataset[current_column] = (
            current_scaler.fit_transform(current_dataset[current_column].values.reshape(-1, 1))).ravel()
    return current_dataset, scalers

def calculate_CI(df, std_part=1.0):
    predicted = df['predicted'].values
    err_std = df.error.std(axis=0)
    err_mean = df.error.mean(axis=0)
    pred_upper = predicted + err_mean + err_std * std_part
    pred_lower = predicted + err_mean - err_std * std_part
    return pred_lower, pred_upper


if __name__ == "__main__":
    n_steps_in, n_steps_out = 210, 1
    num_of_features = len(final_features) - 1
    base_dir = '/gypsum/eguide/projects/amuhebwa/RiversPrediction'
    complete_dataset = create_lookup_table(base_dir, 'complete_dataset')
    model_name = 'lstms'
    fname = f'{base_dir}/1M2RTA_datasets/hydrographs_data/best_average_worst_{model_name}.csv'
    dataset = pd.read_csv(fname)
    df_list = []
    for _, row in dataset.iterrows():
        current_comid = row['StationID']
        model_path = row['Model_Name']
        model = load_model(model_path)
        print(current_comid, model_path)
        if current_comid in complete_dataset:
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            current_df = pd.read_csv(complete_dataset[current_comid])
            current_df['Date'] = pd.to_datetime(current_df['Date'])
            current_df = prepare_dataset(current_df)
            current_df, current_scalars = scale_dynamic_features(current_df, feature_scaler)
            current_df = current_df[final_features]
            '''
            WE(I) NORMALLY DON'T PLOT ALL DAYS IN THE HYDROGRAPHS (the trends won't be that visible)
            for this reason therefore, i will plot the first 1200 days
            '''
            # get the first 1200 days of the dataset
            current_df = current_df.iloc[:1200, :]
            x_test, y_test = create_dataset_forecast(current_df.to_numpy(), n_steps_in, n_steps_out)
            predicted = model.predict(x_test)
            discharge_scaler = current_scalars.get('scaler_discharge')
            actual = discharge_scaler.inverse_transform(y_test.reshape(-1, 1))
            predicted = discharge_scaler.inverse_transform(predicted)
            temp_results_df = pd.DataFrame()
            temp_results_df['actual'] = actual.ravel()
            temp_results_df['predicted'] = predicted.ravel()
            temp_results_df = temp_results_df[temp_results_df['predicted'] >= 0]
            actual = temp_results_df['actual'].values
            predicted = temp_results_df['predicted'].values

            temp_results_df['error'] = temp_results_df['actual'] - temp_results_df['predicted']
            predicted_lower, predicted_upper = calculate_CI(temp_results_df, std_part=1.0)
            temp_results_df['COMID'] = current_comid
            temp_results_df['Model_Name'] = model_path
            temp_results_df['pred_lower'] = predicted_lower
            temp_results_df['pred_upper'] = predicted_upper
            temp_results_df['NSE'] = row['NSE']
            temp_results_df['KGE'] = row['KGE']
            temp_results_df['label'] = row['label']
            df_list.append(temp_results_df)

    final_df = pd.concat(df_list)
    final_df.to_csv(f'{base_dir}/1M2RTA_datasets/hydrographs_data/complete_hydrographs_data_{model_name}.csv', index=False)
