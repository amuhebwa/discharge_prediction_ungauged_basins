"""
File for calculating all the metrics like KGE, NSE, etc
"""
import numpy as np
import code
from sklearn.metrics import r2_score
from hydroeval import *
from TSErrors import FindErrors

"""
KGE
Ref: https://agrimetsoft.com/calculators/Kling-Gupta%20efficiency 

KGE = 1 - sqrt(
    (cc - 1)^2 
    + 
    ((cd/rd) - 1)^2
    + 
    ((cm/rm) - 1)^2
)
cc -> pearson co-efficient
rm -> average of observed/actual values
cm -> average of forecast/predicted values
rd -> standard deviation of observed values
cd -> standard deviation of forecast values
"""


# https://pypi.org/project/hydroeval/
# https://github.com/ThibHlln/hydroeval/blob/v0.0.2-1/examples/api_usage_example.ipynb
# BOUNDED ORIGINAL KGE: https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf
def calculate_KGE(actual, predicted):
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    my_kge = evaluator(kge, predicted, actual)
    return my_kge[0][0]


# BOUNDED ORIGINAL NSE
def calculate_NSE(actual, predicted):
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    my_nse = evaluator(nse, predicted, actual)
    return my_nse[0]


def calculate_RSquared(actual, predicted):
    corr_mat = np.corrcoef(actual.ravel(), predicted.ravel())
    corrActual_Predicted = corr_mat[0, 1]
    r_squared = corrActual_Predicted ** 2
    return r_squared


# percentage bias
def calculate_RBIAS(actual, predicted):
    rBIAS = np.mean((predicted - actual) / np.mean(actual))
    return rBIAS


# Ref: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def calculate_error(actual: np.ndarray, predicted: np.ndarray):
    """ calculate error """
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    return _error(actual, predicted)


def calculate_mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    return np.mean(np.square(_error(actual, predicted)))


# Ref: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(calculate_mse(actual, predicted))


# Ref: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
def calculate_NRMSE(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())


def calculate_RRMSE(actual: np.ndarray, predicted: np.ndarray):
    all_errors = FindErrors(actual, predicted)
    return all_errors.relative_rmse()
