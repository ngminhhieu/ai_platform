import numpy as np
from library import common_util
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

def create_data_prediction(**kwargs):

    data_npz = kwargs['data'].get('dataset')
    seq_len = kwargs['model'].get('seq_len')
    horizon = kwargs['model'].get('horizon')

    time = np.load(data_npz)['time']
    # horizon is in seq_len. the last
    T = len(time)

    map_lon = np.load(data_npz)['map_lon']
    map_lat = np.load(data_npz)['map_lat']
    map_precip = np.load(data_npz)['map_precip']

    gauge_lon = np.load(data_npz)['gauge_lon']
    gauge_lat = np.load(data_npz)['gauge_lat']
    gauge_precip = np.load(data_npz)['gauge_precip']

    raw_precip_gsmap = np.load(data_npz)['raw_precip_gsmap']

    # input is gsmap
    input_model = np.zeros(shape=(T, 160, 120, 1))
    # output is gauge
    output_model = np.zeros(shape=(T, 160, 120, 1))

    for i in range(len(gauge_lat)):
        lat = map_lat[i]
        lon = map_lon[i]
        temp_lat = int(round((23.95 - lat) / 0.1))
        temp_lon = int(round((lon - 100.05) / 0.1))
        for index_lat in range(temp_lat-4, temp_lat+5):
            for index_lon in range(temp_lon-4, temp_lon+5):
                input_model[:, index_lat, index_lon, 0] = map_precip[:, i]
                output_model[:, index_lat, index_lon, 0] = map_precip[:, i]
            
        input_model[:, temp_lat, temp_lon, 0] = map_precip[:, i]
        output_model[:, temp_lat, temp_lon, 0] = gauge_precip[:, i]
    return input_model, output_model


def load_dataset(**kwargs):
    # get preprocessed input and target
    input_conv2d_gsmap, target_conv2d_gsmap = create_data_prediction(**kwargs)

    # get test_size, valid_size from config
    test_size = kwargs['data'].get('test_size')
    valid_size = kwargs['data'].get('valid_size')

    # split data to train_set, valid_set, test_size
    input_train, input_valid, input_test = common_util.prepare_train_valid_test(
        input_conv2d_gsmap, test_size=test_size, valid_size=valid_size)
    target_train, target_valid, target_test = common_util.prepare_train_valid_test(
        target_conv2d_gsmap, test_size=test_size, valid_size=valid_size)
    data = {}
    for cat in ["train", "valid", "test"]:
        x, y = locals()["input_" + cat], locals()["target_" + cat]
        data["input_" + cat] = x
        data["target_" + cat] = y

    return data