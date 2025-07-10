import numpy as np
import pandas as pd
from sklearn import metrics

def MSE(y,hat_y):
    """
    Compute mean squared error between actual and predicted values

    :param y: (1D array) true target values
    :param hat_y: (1D array) predicted values
    :return: (float) mean squared error, rounded to 4 decimals
    """
    MSE = np.round(np.sum(np.square(y - hat_y))/len(y),4)
    return MSE

def RMSE(y,hat_y):
    """
    Compute root mean squared error between actual and predicted values

    :param y: (1D array) true target values
    :param hat_y: (1D array) predicted values
    :return: (float) root mean squared error, rounded to 4 decimals
    """
    RMSE = np.round(np.sqrt(np.sum(np.square(y - hat_y))/len(y)),4)
    return RMSE

def MAE(y,hat_y):
    """
    Compute mean absolute error between actual and predicted values

    :param y: (1D array) true target values
    :param hat_y: (1D array) predicted values
    :return: (float) mean absolute error, rounded to 4 decimals
    """
    MAE = np.round(np.sum(np.abs(y - hat_y))/len(y),4)
    return MAE

def MAPE(y,hat_y):
    """
    Compute mean absolute percentage error between actual and predicted values

    :param y: (1D array) true target values
    :param hat_y: (1D array) predicted values
    :return: (float) mean absolute percentage error in %, rounded to 4 decimals
    """
    MAPE = np.round(metrics.mean_absolute_percentage_error(y, hat_y)*100,4)
    return MAPE

def SMAPE(y, hat_y):
    """
    Compute symmetric mean absolute percentage error between actual and predicted values.

    :param y: (1D array) true target values
    :param hat_y: (1D array) predicted values
    :return: (float) symmetric mean absolute percentage error in %, rounded to 4 decimals
    """
    SMAPE = np.round(np.mean( np.abs(hat_y - y) / (np.abs(y) + np.abs(hat_y) + 1e-6)) * 100, 4)
    return SMAPE


def prediction_errors(y,hat_y):
    """
    Compute a set of common prediction error metrics.

    :param y: (1D array) true target values
    :param hat_y: (1D array) predicted values
    :return: (Series) pandas series with values for MSE, RMSE, MAE, MAPE, and SMAPE
    """
    errors_pd = pd.Series({
        "MSE": MSE(y,hat_y),
        "RMSE": RMSE(y,hat_y),
        "MAE": MAE(y,hat_y),
        "MAPE": MAPE(y,hat_y),
        "SMAPE": SMAPE(y,hat_y)
    })
    return errors_pd
