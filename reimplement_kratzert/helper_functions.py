import glob
from hydroeval import *


import numpy as np


def calculate_RSquared(actual, predicted):
    corr_mat = np.corrcoef(actual.ravel(), predicted.ravel())
    corrActual_Predicted = corr_mat[0, 1]
    r_squared = corrActual_Predicted ** 2
    return r_squared


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def calculate_error(actual: np.ndarray, predicted: np.ndarray):
    """ calculate error """
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    return _error(actual, predicted)


def calculate_mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    return np.mean(np.square(_error(actual, predicted)))


# Ref: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(calculate_mse(actual, predicted))

"""
def calculate_KGE(actual, predicted):
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    my_kge = evaluator(kge, predicted, actual)
    return my_kge[0][0]
"""
def calculate_KGE(actual, predicted):
    # Convert inputs to numpy arrays and flatten them
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()

    # Handle infinities by replacing them with NaNs
    actual = np.where(np.isinf(actual), np.nan, actual)
    predicted = np.where(np.isinf(predicted), np.nan, predicted)

    # Filter out NaN and inf values
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    # Compute KGE
    my_kge = evaluator(kge, predicted, actual)
    return my_kge[0][0]

# BOUNDED ORIGINAL NSE
def calculate_NSE(actual, predicted):
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    my_nse = evaluator(nse, predicted, actual)
    return my_nse[0]


def calculate_RBIAS(actual, predicted):
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    rbias = np.nanmean((predicted - actual) / np.nanmean(actual))
    return rbias

"""
def calculate_NRMSE(actual: np.ndarray, predicted: np.ndarray):
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    return rmse(actual, predicted) / (actual.max() - actual.min())
"""
def calculate_NRMSE(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]

    # Check if the filtered array is empty
    if actual.size == 0:
        raise ValueError("No valid data points after removing NaN values.")

    return rmse(actual, predicted) / (actual.max() - actual.min())


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


def create_lookup_table(parent_path, dataset_name):
    f = f'{parent_path}/1M2RTA_datasets/{dataset_name}/*.csv'
    files = glob.glob(f)
    data_dict = {}
    for file in files:
        comid = int(file.split('/').pop().split('_').pop().split('.')[0])
        data_dict.update({comid: file})
    return data_dict


def split_df(df, validation_split):
    split_point = int(len(df) * (1 - validation_split))
    train_df = df[:split_point]
    validation_df = df[split_point:]
    return train_df, validation_df
