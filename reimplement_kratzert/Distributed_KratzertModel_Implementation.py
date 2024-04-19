"""
Re-implementing Kratzet's model, but on our distributed dataset
Ref:
Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019).
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065
"""
import argparse
import glob
import itertools
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from columns_utils import *
from helper_functions import calculate_NRMSE, calculate_KGE, calculate_NSE, calculate_RBIAS
from kratzert_lstm import LSTM
from proper_combination_of_sets import *

np.random.seed(1234)

static_cols = ['length', 'sinuosity', 'uparea', 'max_width', 'lengthdir', 'strmDrop', 'width_mean', ]


class KratzertModel(nn.Module):
    """
    A model that wraps around LSTM/EA-LSTM with a fully connected layer.
    """

    def __init__(self, input_size_dyn: int, hidden_size: int, initial_forget_bias: int = 5,
                 dropout: float = 0.0, concat_static: bool = False, no_static: bool = False):
        """
        Initialize the model.

        Parameters:
        - input_size_dyn (int): Number of dynamic input features.
        - hidden_size (int): Number of LSTM cells/hidden units.
        - initial_forget_bias (int, optional): Initial forget gate bias. Defaults to 5.
        - dropout (float, optional): Dropout probability in range [0, 1]. Defaults to 0.0.
        - concat_static (bool, optional): If True, uses standard LSTM, else uses EA-LSTM. Defaults to False.
        - no_static (bool, optional): If True, runs standard LSTM. Defaults to False.
        """
        super(KratzertModel, self).__init__()

        # Model attributes
        self.input_size_dyn = input_size_dyn
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        # Model layers
        self.lstm = LSTM(
            input_size=input_size_dyn,
            hidden_size=hidden_size,
            initial_forget_bias=initial_forget_bias
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Parameters:
        - x_d (torch.Tensor): Tensor with dynamic input features of shape [batch, seq_length, n_features]

        Returns:
        - out (torch.Tensor): Network predictions
        - h_n (torch.Tensor): Hidden states of each time step
        - c_n (torch.Tensor): Cell states of each time step
        """
        h_n, c_n = self.lstm(x_d)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)

        return out, h_n, c_n


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, clip_norm=True, clip_value=1.0):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()  # delete the old gradients
        _predictions, _, _ = model(x_batch)
        loss = loss_fn(_predictions.squeeze(), y_batch.squeeze())  # Squeeze both predictions and y_batch
        loss.backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            predictions, _, _ = model(x_batch)
            loss = loss_fn(predictions.squeeze(), y_batch.squeeze())  # Squeeze both predictions and y_batch
            total_loss += loss.item() * len(y_batch)
    return total_loss / len(val_loader.dataset)


def create_lookup_dict(f):
    files = glob.glob(f)
    data_dict = {}
    for file in files:
        comid = file.split('/').pop().split('_').pop().split('.')[0]
        data_dict.update({comid: file})
    return data_dict


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
    return np.array(X), np.array(y)


def process_station_dataset(filepath):
    station_dataset = pd.read_csv(filepath)
    static_df = station_dataset[static_cols]
    static_df = np.log10(static_df + epsilon)
    station_dataset[static_cols] = static_df[static_cols]
    station_dataset = station_dataset[~station_dataset['discharge'].isna()]
    # Fill nans with interpolation
    station_dataset = station_dataset.interpolate(method='linear', limit_direction='both')
    scalers = {}
    for i, current_column in enumerate(columns_to_scale):
        current_scaler = MinMaxScaler(feature_range=(0, 1))
        scalers['scaler_' + str(current_column)] = current_scaler
        station_dataset[current_column] = (current_scaler.fit_transform(station_dataset[current_column].values.reshape(-1, 1))).ravel()
        del current_scaler
    station_dataset = station_dataset[better_columns_order]
    return station_dataset, scalers


if __name__ == "__main__":

    base_dir = '/gypsum/eguide/projects/amuhebwa/rivers_ML/kratzert'
    parser = argparse.ArgumentParser(description='File and Model Parameters')
    parser.add_argument('--ordernumber', required=True)
    parser.add_argument('--set_index', required=True)
    args = parser.parse_args()

    order_number = args.ordernumber
    set_index = args.set_index
    epsilon = 1e-6

    print('order number: ', order_number)

    '''
    which orders do we have ?
    orders: 4, 5, 6, 7, 8
    '''

    order_number = int(order_number)
    set_index = int(set_index)
    discharge_lookup = create_lookup_dict(f'{base_dir}/distributed_complete_dataset/StationId_*.csv')

    # set up proper columns based on orders
    current_order_dict = orders_lookuptable.get(str(order_number))
    better_columns_order = current_order_dict.get('better_columns_order')
    columns_to_scale = current_order_dict.get('columns_to_scale')
    columns_to_decompose = current_order_dict.get('columns_to_decompose')

    current_dataset_name = "order_{}_datasets".format(str(order_number))
    current_datasets = orders_dict.get(current_dataset_name)
    stationsIds = np.unique(list(itertools.chain(*current_datasets)))

    training_set = current_datasets[set_index]
    unique_id = '_'.join([str(s) for s in training_set])

    dataset_list = []
    for idx, station_id in enumerate(training_set):
        file_path = discharge_lookup[station_id]
        current_dataset, current_scalers = process_station_dataset(file_path)
        dataset_list.append(current_dataset)
        del current_dataset

    complete_dataset = pd.concat(dataset_list)

    # ==================================================================================================================
    batch_size = 64
    sequence_length = 270
    forecast_days = 1
    hidden_size = 256
    initial_forget_gate_bias = 5
    learning_rate = 1e-3
    clip_norm = True
    clip_value = 1
    dropout = 0.4
    num_epochs = 50
    epsilon = 1e-6
    no_input_features = complete_dataset.shape[1] - 1
    experimentType = f'Lumped_Order{order_number}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    kratzert_model = KratzertModel(no_input_features, hidden_size, initial_forget_gate_bias, dropout)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        kratzert_model = nn.DataParallel(kratzert_model)

    kratzert_model = kratzert_model.to(device)
    optimizer_kratzert = torch.optim.Adam(kratzert_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    # ==================================================================================================================
    x_train, y_train = create_dataset_forecast(complete_dataset.to_numpy(), sequence_length, forecast_days)
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # We want to free space by deleting some of the variables
    del complete_dataset, x_train, y_train, x_train_tensor, y_train_tensor, train_dataset

    print(f"Training on stations: {training_set}")
    print(f"Order: {order_number} - Number of Features {no_input_features}")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(kratzert_model, train_dataloader, loss_fn, optimizer_kratzert, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        sys.stdout.flush()
        # Release cached GPU memory
        torch.cuda.empty_cache()
    # Save the model
    torch.save(kratzert_model.state_dict(), f'{base_dir}/trained_models/{experimentType}_kratzert_model_{unique_id}.pth')

    # perform inference
    # First set the model to evaluation mode
    kratzert_model.eval()
    test_stations = [s for s in stationsIds if s not in training_set]

    # create arrays to store the results
    kge_arr, nse_arr, rbias_arr, nrmse_arr = [], [], [], []
    for idx, station_id in enumerate(test_stations):
        test_file_path = discharge_lookup[station_id]
        test_dataset, test_scalers = process_station_dataset(test_file_path)
        x_test, y_test = create_dataset_forecast(test_dataset.to_numpy(), sequence_length, forecast_days)

        x_test_tensor = torch.FloatTensor(x_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        # Create a TensorDataset and DataLoader
        test_data = TensorDataset(x_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        predictions_list = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                batch_predictions = kratzert_model(x_batch)
                predictions_list.append(batch_predictions[0].cpu())
        # Concatenate all batch predictions
        predictions = torch.cat(predictions_list).numpy().ravel()
        y_test = y_test.ravel()

        nse = calculate_NSE(y_test, predictions)
        kge = calculate_KGE(y_test, predictions)
        rbias = calculate_RBIAS(y_test, predictions)
        nrmse = calculate_NRMSE(y_test, predictions)

        print(f"Testing on station {idx + 1}/{len(test_stations)}: {station_id} - NSE: {nse:.4f} - KGE: {kge:.4f} - RBIAS: {rbias:.4f} - NRMSE: {nrmse:.4f}")

        kge_arr.append(kge)
        nse_arr.append(nse)
        rbias_arr.append(rbias)
        nrmse_arr.append(nrmse)
    # create a dataframe to store the results
    results_df = pd.DataFrame({'station_id': test_stations, 'KGE': kge_arr, 'NSE': nse_arr, 'RBIAS': rbias_arr, 'NRMSE': nrmse_arr})
    results_df.to_csv(f'{base_dir}/kratzert_prediction_results/{experimentType}_kratzert_model_{unique_id}.csv', index=False)
