import logging
import sys
import os
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import csv

class TrainingTimePerEpoch(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time.time() - self.epoch_time_start)


def get_config_model(**kwargs):
    # init a dictionary to store config
    config_model = {}

    config_model['kwargs'] = kwargs
    config_model['data_kwargs'] = kwargs.get('data')
    config_model['train_kwargs'] = kwargs.get('train')
    config_model['test_kwargs'] = kwargs.get('test')
    config_model['model_kwargs'] = kwargs.get('model')

    # data args
    config_model['dataset'] = config_model['data_kwargs'].get('dataset')
    config_model['test_size'] = config_model['data_kwargs'].get('test_size')
    config_model['valid_size'] = config_model['data_kwargs'].get('valid_size')
    config_model['test_batch_size'] = config_model['data_kwargs'].get(
        'test_batch_size')

    # logging.
    config_model['log_dir'] = _get_log_dir(kwargs)
    log_level = config_model['kwargs'].get('log_level', 'INFO')
    config_model['logger'] = _get_logger(config_model['log_dir'],
                                         __name__,
                                         'info.log',
                                         level=log_level)
    config_model['logger'].info(kwargs)

    # Model's Args
    config_model['type'] = config_model['model_kwargs'].get('type')
    config_model['rnn_units'] = config_model['model_kwargs'].get('rnn_units')
    config_model['seq_len'] = config_model['model_kwargs'].get('seq_len')
    config_model['horizon'] = config_model['model_kwargs'].get('horizon')
    config_model['input_dim'] = config_model['model_kwargs'].get('input_dim')
    config_model['output_dim'] = config_model['model_kwargs'].get('output_dim')
    config_model['rnn_layers'] = config_model['model_kwargs'].get('rnn_layers')
    config_model['activation'] = config_model['model_kwargs'].get('activation')

    # Train's args
    config_model['dropout'] = config_model['train_kwargs'].get('dropout')
    config_model['epochs'] = config_model['train_kwargs'].get('epochs')
    config_model['batch_size'] = config_model['train_kwargs'].get('batch_size')
    config_model['optimizer'] = config_model['train_kwargs'].get('optimizer')
    config_model['loss'] = config_model['train_kwargs'].get('loss')
    config_model['patience'] = config_model['train_kwargs'].get('patience')

    callbacks = []

    checkpoints_callback = ModelCheckpoint(config_model['log_dir'] +
                                           "best_model.hdf5",
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           mode='auto',
                                           save_freq='epoch')
    earlystopping_callback = EarlyStopping(monitor='val_loss',
                                           patience=config_model['patience'],
                                           verbose=1,
                                           mode='auto')
    trainingtime_callback = TrainingTimePerEpoch()

    callbacks.append(checkpoints_callback)
    callbacks.append(earlystopping_callback)
    callbacks.append(trainingtime_callback)
    config_model['callbacks'] = callbacks

    return config_model


def prepare_train_valid_test(data, test_size, valid_size):

    train_len = int(data.shape[0] * (1 - test_size - valid_size))
    valid_len = int(data.shape[0] * valid_size)

    train_set = data[0:train_len]
    valid_set = data[train_len:train_len + valid_len]
    test_set = data[train_len + valid_len:]

    return train_set, valid_set, test_set


def _get_log_dir(kwargs):
    log_dir = kwargs['train'].get('log_dir')
    if log_dir is None:
        run_id = 'default/'
        base_dir = kwargs.get('base_dir')
        log_dir = os.path.join(base_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def _save_model_history(model_history, config_model):
    loss = np.array(model_history.history['loss'])
    val_loss = np.array(model_history.history['val_loss'])
    dump_model_history = pd.DataFrame(
        index=range(loss.size),
        columns=['epoch', 'loss', 'val_loss', 'training_time'])

    dump_model_history['epoch'] = range(loss.size)
    dump_model_history['loss'] = loss
    dump_model_history['val_loss'] = val_loss

    # training time call back at third position
    training_time_callback = config_model['callbacks'][2]
    if training_time_callback.logs is not None:
        dump_model_history['training_time'] = training_time_callback.logs[0] + training_time_callback.logs[1]

    # total training time
    dump_model_history.loc[-1, -1] = np.sum(training_time_callback.logs)

    # save training log to csv
    dump_model_history.to_csv(config_model['log_dir'] + 'training_history.csv',
                              index=False)


def _plot_training_history(model_history, config_model):
    plt.plot(model_history.history['loss'], label='loss')
    plt.plot(model_history.history['val_loss'], label='val_loss')
    plt.savefig(config_model['log_dir'] + 'loss.png')
    plt.legend()
    plt.close()

    plt.plot(model_history.history['val_loss'], label='val_loss')
    plt.savefig(config_model['log_dir'] + 'val_loss.png')
    plt.legend()
    plt.close()


def _get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def mae(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        error_mae = mean_absolute_error(test_arr, prediction_arr)
        return error_mae


def mse(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        error_mse = mean_squared_error(test_arr, prediction_arr)
        return error_mse


def rmse(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        error_rmse = np.sqrt(mean_squared_error(test_arr, prediction_arr))
        return error_rmse


def mape(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        y_true, y_pred = np.array(test_arr), np.array(prediction_arr)
        error_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return error_mape


def cal_error(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        # cal mse
        error_mae = mae(test_arr, prediction_arr)

        # cal rmse
        error_mse = mse(test_arr, prediction_arr)
        error_rmse = rmse(test_arr, prediction_arr)

        # cal mape
        error_mape = mape(test_arr, prediction_arr)
        error_list = [error_mae, error_rmse, error_mape]
        return error_list


def save_metrics(error_list, log_dir, filename):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    error_list = error_list.tolist()
    error_list.insert(0, dt_string)
    with open(log_dir + filename + ".csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(error_list)