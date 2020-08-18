import os
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from lib import common_util


def create_data(dataset, **kwargs):
    seq_len = kwargs['model'].get('seq_len')
    horizon = kwargs['model'].get('horizon')
    input_dim = kwargs['model'].get('input_dim')
    output_dim = kwargs['model'].get('output_dim')
    T = dataset.shape[0]
    # only take pm10 and pm2.5 to predict
    _input = dataset[:, -input_dim:].copy()
    _target = dataset[:, -output_dim:].copy()
    input_x = np.zeros(shape=((T - seq_len - horizon), seq_len, input_dim))
    output_y = np.zeros(shape=((T - seq_len - horizon), output_dim))

    for i in range(T - seq_len - horizon):
        input_x[i, :, :] = _input[i:i + seq_len].copy()
        output_y[i, :] = _target[i + seq_len].copy()
    return input_x, output_y


def load_dataset(**kwargs):
    data_url = kwargs['data'].get('dataset')
    dataset = read_csv(data_url).to_numpy()
    # scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    # scaler.fit(dataset)
    # dataset = scaler.transform(dataset)

    input_model, output_model = create_data(dataset, **kwargs)

    test_size = kwargs['data'].get('test_size')
    valid_size = kwargs['data'].get('valid_size')

    input_train, input_valid, input_test = common_util.prepare_train_valid_test(
        input_model, test_size=test_size, valid_size=valid_size)
    target_train, target_valid, target_test = common_util.prepare_train_valid_test(
        output_model, test_size=test_size, valid_size=valid_size)

    data = {}
    for cat in ["train", "valid", "test"]:
        x, y = locals()["input_" + cat], locals()["target_" + cat]
        data["input_" + cat] = x
        data["target_" + cat] = y
        print("input_" + cat, x.shape)
        print("target_" +cat, y.shape)
    # data['scaler'] = scaler
    return data