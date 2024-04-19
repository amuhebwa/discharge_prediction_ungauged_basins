import code
import glob
import re
import os
import argparse
from kratzert_lstm import LSTM
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from generated_kfold_sets import *
from helper_functions import calculate_NRMSE, calculate_KGE, calculate_NSE, calculate_RBIAS
import sys
np.random.seed(1234)
import gc
better_columns_orders = [
    'order_n-7_AirTemp', 'order_n-7_Albedo',
    'order_n-7_Avg_Skin_Temp', 'order_n-7_PlantCanopyWater',
    'order_n-7_CanopyWaterEvpn', 'order_n-7_DirectEvonBareSoil',
    'order_n-7_Evapotranspn', 'order_n-7_LngWaveRadFlux',
    'order_n-7_NetRadFlux', 'order_n-7_PotEvpnRate',
    'order_n-7_Pressure', 'order_n-7_SpecHmd', 'order_n-7_HeatFlux',
    'order_n-7_Sen.HtFlux', 'order_n-7_LtHeat',
    'order_n-7_StmSurfRunoff', 'order_n-7_BsGndWtrRunoff',
    'order_n-7_SnowMelt', 'order_n-7_TotalPcpRate',
    'order_n-7_RainPcpRate', 'order_n-7_RootZoneSoilMstr',
    'order_n-7_SnowDepthWtrEq', 'order_n-7_DwdShtWvRadFlux',
    'order_n-7_SnowDepth', 'order_n-7_SnowPcpRate',
    'order_n-7_SoilMst10', 'order_n-7_SoilMst40',
    'order_n-7_SoilMst100', 'order_n-7_SoilMst200',
    'order_n-7_SoilTmp10', 'order_n-7_SoilTmp40',
    'order_n-7_SoilTmp100', 'order_n-7_SoilTmp200',
    'order_n-7_NetShtWvRadFlux', 'order_n-7_Tspn', 'order_n-7_WindSpd',

    'order_n-6_AirTemp', 'order_n-6_Albedo',
    'order_n-6_Avg_Skin_Temp', 'order_n-6_PlantCanopyWater',
    'order_n-6_CanopyWaterEvpn', 'order_n-6_DirectEvonBareSoil',
    'order_n-6_Evapotranspn', 'order_n-6_LngWaveRadFlux',
    'order_n-6_NetRadFlux', 'order_n-6_PotEvpnRate',
    'order_n-6_Pressure', 'order_n-6_SpecHmd', 'order_n-6_HeatFlux',
    'order_n-6_Sen.HtFlux', 'order_n-6_LtHeat',
    'order_n-6_StmSurfRunoff', 'order_n-6_BsGndWtrRunoff',
    'order_n-6_SnowMelt', 'order_n-6_TotalPcpRate',
    'order_n-6_RainPcpRate', 'order_n-6_RootZoneSoilMstr',
    'order_n-6_SnowDepthWtrEq', 'order_n-6_DwdShtWvRadFlux',
    'order_n-6_SnowDepth', 'order_n-6_SnowPcpRate',
    'order_n-6_SoilMst10', 'order_n-6_SoilMst40',
    'order_n-6_SoilMst100', 'order_n-6_SoilMst200',
    'order_n-6_SoilTmp10', 'order_n-6_SoilTmp40',
    'order_n-6_SoilTmp100', 'order_n-6_SoilTmp200',
    'order_n-6_NetShtWvRadFlux', 'order_n-6_Tspn', 'order_n-6_WindSpd',

    'order_n-5_AirTemp', 'order_n-5_Albedo', 'order_n-5_Avg_Skin_Temp', 'order_n-5_PlantCanopyWater',
    'order_n-5_CanopyWaterEvpn', 'order_n-5_DirectEvonBareSoil',
    'order_n-5_Evapotranspn', 'order_n-5_LngWaveRadFlux', 'order_n-5_NetRadFlux',
    'order_n-5_PotEvpnRate', 'order_n-5_Pressure', 'order_n-5_SpecHmd', 'order_n-5_HeatFlux',
    'order_n-5_Sen.HtFlux', 'order_n-5_LtHeat', 'order_n-5_StmSurfRunoff',
    'order_n-5_BsGndWtrRunoff', 'order_n-5_SnowMelt', 'order_n-5_TotalPcpRate',
    'order_n-5_RainPcpRate', 'order_n-5_RootZoneSoilMstr', 'order_n-5_SnowDepthWtrEq',
    'order_n-5_DwdShtWvRadFlux', 'order_n-5_SnowDepth', 'order_n-5_SnowPcpRate',
    'order_n-5_SoilMst10', 'order_n-5_SoilMst40', 'order_n-5_SoilMst100',
    'order_n-5_SoilMst200', 'order_n-5_SoilTmp10', 'order_n-5_SoilTmp40',
    'order_n-5_SoilTmp100', 'order_n-5_SoilTmp200', 'order_n-5_NetShtWvRadFlux',
    'order_n-5_Tspn', 'order_n-5_WindSpd',

    'order_n-4_AirTemp', 'order_n-4_Albedo', 'order_n-4_Avg_Skin_Temp', 'order_n-4_PlantCanopyWater',
    'order_n-4_CanopyWaterEvpn',
    'order_n-4_DirectEvonBareSoil', 'order_n-4_Evapotranspn', 'order_n-4_LngWaveRadFlux',
    'order_n-4_NetRadFlux', 'order_n-4_PotEvpnRate', 'order_n-4_Pressure', 'order_n-4_SpecHmd',
    'order_n-4_HeatFlux', 'order_n-4_Sen.HtFlux', 'order_n-4_LtHeat',
    'order_n-4_StmSurfRunoff', 'order_n-4_BsGndWtrRunoff', 'order_n-4_SnowMelt',
    'order_n-4_TotalPcpRate', 'order_n-4_RainPcpRate', 'order_n-4_RootZoneSoilMstr',
    'order_n-4_SnowDepthWtrEq', 'order_n-4_DwdShtWvRadFlux', 'order_n-4_SnowDepth',
    'order_n-4_SnowPcpRate', 'order_n-4_SoilMst10', 'order_n-4_SoilMst40',
    'order_n-4_SoilMst100', 'order_n-4_SoilMst200', 'order_n-4_SoilTmp10',
    'order_n-4_SoilTmp40', 'order_n-4_SoilTmp100', 'order_n-4_SoilTmp200',
    'order_n-4_NetShtWvRadFlux', 'order_n-4_Tspn', 'order_n-4_WindSpd',

    'order_n-3_AirTemp', 'order_n-3_Albedo', 'order_n-3_Avg_Skin_Temp', 'order_n-3_PlantCanopyWater',
    'order_n-3_CanopyWaterEvpn', 'order_n-3_DirectEvonBareSoil',
    'order_n-3_Evapotranspn', 'order_n-3_LngWaveRadFlux', 'order_n-3_NetRadFlux',
    'order_n-3_PotEvpnRate', 'order_n-3_Pressure', 'order_n-3_SpecHmd', 'order_n-3_HeatFlux',
    'order_n-3_Sen.HtFlux', 'order_n-3_LtHeat', 'order_n-3_StmSurfRunoff',
    'order_n-3_BsGndWtrRunoff', 'order_n-3_SnowMelt', 'order_n-3_TotalPcpRate',
    'order_n-3_RainPcpRate', 'order_n-3_RootZoneSoilMstr', 'order_n-3_SnowDepthWtrEq',
    'order_n-3_DwdShtWvRadFlux', 'order_n-3_SnowDepth', 'order_n-3_SnowPcpRate',
    'order_n-3_SoilMst10', 'order_n-3_SoilMst40', 'order_n-3_SoilMst100',
    'order_n-3_SoilMst200', 'order_n-3_SoilTmp10', 'order_n-3_SoilTmp40',
    'order_n-3_SoilTmp100', 'order_n-3_SoilTmp200', 'order_n-3_NetShtWvRadFlux',
    'order_n-3_Tspn', 'order_n-3_WindSpd',

    'order_n-2_AirTemp', 'order_n-2_Albedo', 'order_n-2_Avg_Skin_Temp', 'order_n-2_PlantCanopyWater',
    'order_n-2_CanopyWaterEvpn',
    'order_n-2_DirectEvonBareSoil', 'order_n-2_Evapotranspn', 'order_n-2_LngWaveRadFlux',
    'order_n-2_NetRadFlux', 'order_n-2_PotEvpnRate', 'order_n-2_Pressure', 'order_n-2_SpecHmd',
    'order_n-2_HeatFlux', 'order_n-2_Sen.HtFlux', 'order_n-2_LtHeat',
    'order_n-2_StmSurfRunoff', 'order_n-2_BsGndWtrRunoff', 'order_n-2_SnowMelt',
    'order_n-2_TotalPcpRate', 'order_n-2_RainPcpRate', 'order_n-2_RootZoneSoilMstr',
    'order_n-2_SnowDepthWtrEq', 'order_n-2_DwdShtWvRadFlux', 'order_n-2_SnowDepth',
    'order_n-2_SnowPcpRate', 'order_n-2_SoilMst10', 'order_n-2_SoilMst40',
    'order_n-2_SoilMst100', 'order_n-2_SoilMst200', 'order_n-2_SoilTmp10',
    'order_n-2_SoilTmp40', 'order_n-2_SoilTmp100', 'order_n-2_SoilTmp200',
    'order_n-2_NetShtWvRadFlux', 'order_n-2_Tspn', 'order_n-2_WindSpd',

    'order_n-1_AirTemp', 'order_n-1_Albedo', 'order_n-1_Avg_Skin_Temp', 'order_n-1_PlantCanopyWater',
    'order_n-1_CanopyWaterEvpn', 'order_n-1_DirectEvonBareSoil',
    'order_n-1_Evapotranspn', 'order_n-1_LngWaveRadFlux', 'order_n-1_NetRadFlux',
    'order_n-1_PotEvpnRate', 'order_n-1_Pressure', 'order_n-1_SpecHmd', 'order_n-1_HeatFlux',
    'order_n-1_Sen.HtFlux', 'order_n-1_LtHeat', 'order_n-1_StmSurfRunoff',
    'order_n-1_BsGndWtrRunoff', 'order_n-1_SnowMelt', 'order_n-1_TotalPcpRate',
    'order_n-1_RainPcpRate', 'order_n-1_RootZoneSoilMstr', 'order_n-1_SnowDepthWtrEq',
    'order_n-1_DwdShtWvRadFlux', 'order_n-1_SnowDepth', 'order_n-1_SnowPcpRate',
    'order_n-1_SoilMst10', 'order_n-1_SoilMst40', 'order_n-1_SoilMst100',
    'order_n-1_SoilMst200', 'order_n-1_SoilTmp10', 'order_n-1_SoilTmp40',
    'order_n-1_SoilTmp100', 'order_n-1_SoilTmp200', 'order_n-1_NetShtWvRadFlux',
    'order_n-1_Tspn', 'order_n-1_WindSpd',

    'order_n_AirTemp', 'order_n_Albedo', 'order_n_Avg_Skin_Temp', 'order_n_PlantCanopyWater', 'order_n_CanopyWaterEvpn',
    'order_n_DirectEvonBareSoil', 'order_n_Evapotranspn', 'order_n_LngWaveRadFlux',
    'order_n_NetRadFlux', 'order_n_PotEvpnRate', 'order_n_Pressure', 'order_n_SpecHmd',
    'order_n_HeatFlux', 'order_n_Sen.HtFlux', 'order_n_LtHeat', 'order_n_StmSurfRunoff',
    'order_n_BsGndWtrRunoff', 'order_n_SnowMelt', 'order_n_TotalPcpRate',
    'order_n_RainPcpRate', 'order_n_RootZoneSoilMstr', 'order_n_SnowDepthWtrEq',
    'order_n_DwdShtWvRadFlux', 'order_n_SnowDepth', 'order_n_SnowPcpRate', 'order_n_SoilMst10',
    'order_n_SoilMst40', 'order_n_SoilMst100', 'order_n_SoilMst200', 'order_n_SoilTmp10',
    'order_n_SoilTmp40', 'order_n_SoilTmp100', 'order_n_SoilTmp200', 'order_n_NetShtWvRadFlux',
    'order_n_Tspn', 'order_n_WindSpd',

    'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
    'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure', 'SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat',
    'StmSurfRunoff',
    'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate', 'RootZoneSoilMstr', 'SnowDepthWtrEq',
    'DwdShtWvRadFlux',
    'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40',
    'SoilTmp100', 'SoilTmp200',
    'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI',

    'lengthkm', 'sinuosity', 'slope', 'uparea', 'width_max', 'lengthdir', 'width_mean',
    'width',
    'order',
    'discharge',
]

static_features = [
    'lengthkm', 'sinuosity', 'slope', 'uparea', 'width_max', 'lengthdir', 'width_mean',
]
categorical_features = ['order']

columns_to_scale = [item for item in better_columns_orders if item not in categorical_features]

columns_to_drop = ['CanopyWaterEvpn', 'DirectEvonBareSoil', 'LngWaveRadFlux', 'HeatFlux', 'StmSurfRunoff', 'BsGndWtrRunoff', 'RainPcpRate',
                   'RootZoneSoilMstr', 'DwdShtWvRadFlux', 'SnowDepth', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200']

orders_to_drop = {
    7: ['n-7', 'n-6', 'n-5', 'n-4', 'n-3', 'n-2', 'n-1'],
    6: ['n-7', 'n-6', 'n-5', 'n-4', 'n-3', 'n-2'],
    5: ['n-7', 'n-6', 'n-5', 'n-4', 'n-3'],
    4: ['n-7', 'n-6', 'n-5', 'n-4'],
    3: ['n-7', 'n-6', 'n-5'],
    2: ['n-7', 'n-6'],
    1: ['n-7'],
    0: [],
}

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
        optimizer.zero_grad() # delete the old gradients
        _predictions, _, _ = model(x_batch)
        loss = loss_fn(_predictions.squeeze(), y_batch.squeeze())  # Squeeze both predictions and y_batch
        loss.backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(train_loader.dataset)


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
    static_df = station_dataset[static_features]
    static_df = np.log10(static_df + epsilon)
    station_dataset[static_features] = static_df[static_features]

    station_dataset = station_dataset[~station_dataset['discharge'].isna()]
    # Fill nans with interpolation
    station_dataset = station_dataset.interpolate(method='linear', limit_direction='both')
    scalers = {}
    for i, current_column in enumerate(columns_to_scale):
        current_scaler = MinMaxScaler(feature_range=(0, 1))
        scalers['scaler_' + str(current_column)] = current_scaler
        station_dataset[current_column] = (current_scaler.fit_transform(station_dataset[current_column].values.reshape(-1, 1))).ravel()
        del current_scaler
    station_dataset = station_dataset[better_columns_orders]
    return station_dataset, scalers

def create_lookup_table(parent_path, dataset_name):
    f = f'{parent_path}/1M2RTA_datasets/{dataset_name}/*.csv'
    files = glob.glob(f)
    data_dict = {}
    for file in files:
        comid = int(file.split('/').pop().split('_').pop().split('.')[0])
        data_dict.update({comid: file})
    return data_dict

def parse_args():
    parser = argparse.ArgumentParser(description='File Parameters')
    parser.add_argument('--set_index', type=int, required=True)
    return parser.parse_args()
if __name__=="__main__":
    args = parse_args()
    set_index = int(args.set_index)
    base_dir = '/gypsum/eguide/projects/amuhebwa/rivers_ML/kratzert'

    discharge_lookup = create_lookup_table("/gypsum/eguide/projects/amuhebwa/RiversPrediction", 'complete_dataset')
    stationsIds = list(discharge_lookup.keys())
    batch_size = 128 # 2000
    sequence_length = 270
    forecast_days = 1
    hidden_size = 256
    initial_forget_gate_bias = 5
    learning_rate = 1e-3
    num_days = 270
    clip_norm = True
    clip_value = 1
    dropout = 0.4
    num_epochs = 50
    epsilon = 1e-6
    no_input_features = None
    no_of_orders_to_drop = 0
    experimentType = 'OneM2RTA'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    columns_to_scale = [col for col in columns_to_scale if not any(drop in col for drop in columns_to_drop)]
    better_columns_orders = [col for col in better_columns_orders if not any(drop in col for drop in columns_to_drop)]

    if no_of_orders_to_drop != 0:
        drop_orders = orders_to_drop[no_of_orders_to_drop]
        better_columns_orders = [b for b in better_columns_orders if not any(a in b for a in drop_orders)]
        static_features = [b for b in static_features if not any(a in b for a in drop_orders)]
    else:
        better_columns_orders = better_columns_orders

    no_input_features = len(better_columns_orders)-1

    all_trained_models = glob.glob(f'{base_dir}/trained_models/{experimentType}_kratzert_model_*')
    all_trained_models.sort()
    model_name = all_trained_models[set_index]


    train_stations = re.findall(r'\d{8,}', model_name)

    unique_id = '_'.join([str(s) for s in train_stations])

    train_stations = [int(station) for station in train_stations]
    test_stations = [comid for comid in stationsIds if comid not in train_stations]
    # load pytorch model
    kratzert_model = KratzertModel(no_input_features, hidden_size, initial_forget_gate_bias, dropout)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        kratzert_model = nn.DataParallel(kratzert_model)

    kratzert_model = kratzert_model.to(device)
    optimizer_kratzert = torch.optim.Adam(kratzert_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    if torch.cuda.is_available():
        state_dict = torch.load(model_name)
        # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        kratzert_model.load_state_dict(state_dict)
    else:
        print('Loading model on CPU')

    kratzert_model = kratzert_model.to(device)
    kratzert_model.eval()

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

        print(f"Testing on station {idx+1}/{len(test_stations)}: {station_id} - NSE: {nse:.4f} - KGE: {kge:.4f} - RBIAS: {rbias:.4f} - NRMSE: {nrmse:.4f}")

        kge_arr.append(kge)
        nse_arr.append(nse)
        rbias_arr.append(rbias)
        nrmse_arr.append(nrmse)

        del test_dataset, x_test, y_test, x_test_tensor, predictions
        torch.cuda.empty_cache()
        gc.collect()

    # create a dataframe to store the results
    results_df = pd.DataFrame({'station_id': test_stations, 'KGE': kge_arr, 'NSE': nse_arr, 'RBIAS': rbias_arr, 'NRMSE': nrmse_arr})
    results_df.to_csv(f'{base_dir}/kratzert_prediction_results/{experimentType}_kratzert_model_{unique_id}.csv', index=False)
