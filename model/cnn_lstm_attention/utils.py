import os
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from library import common_util
import logging
import os
import csv
import sys
import random
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import tensorflow as tf
from library import common_util

def create_data(data, seq_len, input_dim, output_dim, horizon):
    T = data.shape[0]
    pm_data = data[:, -output_dim:].copy()
    input_model = np.zeros(shape=((T - seq_len - horizon), seq_len, input_dim))
    output_model = np.zeros(shape=((T - seq_len - horizon), horizon, output_dim))

    for i in range(T - seq_len - horizon):
        input_model[i, :, :] = data[i:i + seq_len].copy()
        output_model[i, :, :] = pm_data[i + seq_len:i + seq_len + horizon]

    return input_model, output_model


def load_dataset(**kwargs):
    seq_len = kwargs['model'].get('seq_len')
    horizon = kwargs['model'].get('horizon')
    input_dim = kwargs['model'].get('input_dim')
    output_dim = kwargs['model'].get('output_dim')
    dataset_path = kwargs['data'].get('dataset')
    test_size = kwargs['data'].get('test_size')
    valid_size = kwargs['data'].get('valid_size')
    # dataset of taiwan
    # dataset = read_csv(dataset_path, usecols=['AMB_TEMP', 'RH', 'WIND_DIREC', 'WIND_SPEED', 'PM10', 'PM2.5'])
    # features selected by GA
    dataset = read_csv(dataset_path, usecols=['WIND_SPEED', 'TEMP', 'RADIATION', 'PM10', 'PM2.5'])
    # all features Hanoi
    # dataset = read_csv(dataset_path, usecols=['MONTH','DAY','YEAR','HOUR','WIND_SPEED','WIND_DIR','TEMP','RH','BAROMETER','RADIATION','INNER_TEMP','PM10','PM1','PM2.5'])
    train_data2d, valid_data2d, test_data2d = common_util.prepare_train_valid_test(
        data=dataset, test_size=test_size, valid_size=valid_size)

    data = {}
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(train_data2d)
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)
    data['test_data_norm'] = test_data2d_norm.copy()

    input_train, target_train = create_data(train_data2d_norm,
                                            seq_len=seq_len,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            horizon=horizon)
    input_valid, target_valid = create_data(valid_data2d_norm,
                                        seq_len=seq_len,
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        horizon=horizon)

    input_test, target_test = create_data(test_data2d_norm,
                                          seq_len=seq_len,
                                          input_dim=input_dim,
                                          output_dim=output_dim,
                                          horizon=horizon)
                                          
    for cat in ["train", "valid", "test"]:
        x, y = locals()["input_" + cat], locals()["target_" + cat]
        data["input_" + cat] = x
        data["target_" + cat] = y
        print("input_" + cat, x.shape)
        print("target_" + cat, y.shape)

    data['scaler'] = scaler
    return data


def mae(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        # cal mse
        error_mae = mean_absolute_error(test_arr, prediction_arr)
        return error_mae


def cal_error(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        # cal mse
        error_mae = mean_absolute_error(test_arr, prediction_arr)

        # cal rmse
        error_mse = mean_squared_error(test_arr, prediction_arr)
        error_rmse = np.sqrt(error_mse)

        # cal mape
        y_true, y_pred = np.array(test_arr), np.array(prediction_arr)
        error_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        error_list = [error_mae, error_rmse, error_mape]
        print("MAE: %.4f" % (error_mae))
        print("RMSE: %.4f" % (error_rmse))
        print("MAPE: %.4f" % (error_mape))
        return error_list


def save_metrics(error_list, log_dir, alg):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    error_list.insert(0, dt_string)
    with open(log_dir + alg + "_metrics.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(error_list)
