from keras.layers import Dense, LSTM, Input, GRU, SimpleRNN
from keras.models import Sequential
import numpy as np
import tensorflow as tf

import sys
from library import common_util
import model.RNN.utils as utils
import os
import yaml
from pandas import read_csv
from keras.utils import plot_model
from keras import backend as K
from keras.losses import mse
from model.layers import *

class Nets():
    def __init__(self, **kwargs):
        self.config_model = common_util.get_config_model(**kwargs)

        # load_data
        self.data = utils.load_dataset(**kwargs)
        self.input_train = self.data['input_train']
        self.input_valid = self.data['input_valid']
        self.input_test = self.data['input_test']
        self.target_train = self.data['target_train']
        self.target_valid = self.data['target_valid']
        self.target_test = self.data['target_test']

        # other configs
        self.log_dir = self.config_model['log_dir']
        self.optimizer = self.config_model['optimizer']
        self.loss = self.config_model['loss']
        self.activation = self.config_model['activation']
        self.batch_size = self.config_model['batch_size']
        self.epochs = self.config_model['epochs']
        self.callbacks = self.config_model['callbacks']
        self.seq_len = self.config_model['seq_len']
        self.horizon = self.config_model['horizon']
        self.input_dim = self.config_model['input_dim']
        self.output_dim = self.config_model['output_dim']
        self.rnn_units = self.config_model['rnn_units']
        self.type = self.config_model['type']
        self.rnn_layers = self.config_model['rnn_layers']
        self.model = self.construct_model()

    def construct_model(self):
        if self.type == 'rnn':
            cell = RNNs(units = self.rnn_units, num_cells= self.rnn_layers, go_backwards=False, dropout=0.,
                 return_sequences=False, return_state=False, name='rnns', l2=0., activation = self.activation)
        elif self.type == 'lstm':
            cell = LSTMs(units=self.rnn_units, num_cells=self.rnn_layers, go_backwards=False, dropout=0.,
                        return_sequences=False, return_state=False, name='lstms', l2=0., activation = self.activation)
        else:
            cell = GRUs(units=self.rnn_units, num_cells=self.rnn_layers, go_backwards=False, dropout=0.,
                        return_sequences=False, return_state=False, name='grus', l2=0., activation = self.activation)

        inputs = Input(shape=(self.seq_len, self.input_dim))
        x = cell(inputs)
        x = Dense(self.output_dim)(x)
        model = tf.keras.Model(inputs = inputs,
                               outputs = x,
                               name = self.type+'s_net')
        model.summary()
        # plot model
        from keras.utils import plot_model
        plot_model(model=model,
                   to_file=self.log_dir + '/'+self.type+'_model.png',
                   show_shapes=True)
        return model

    def train(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['mse', 'mae'])

        training_history = self.model.fit(self.input_train,
                                          self.target_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          callbacks=self.callbacks,
                                          validation_data=(self.input_valid,
                                                           self.target_valid),
                                          shuffle=True,
                                          verbose=1)

        if training_history is not None:
            common_util._plot_training_history(training_history,
                                               self.config_model)
            common_util._save_model_history(training_history,
                                            self.config_model)
            config = dict(self.config_model['kwargs'])

            # create config file in log again
            config_filename = 'config.yaml'
            config['train']['log_dir'] = self.log_dir
            with open(os.path.join(self.log_dir, config_filename), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

    def test(self):
        print("Load model from: {}".format(self.log_dir))
        self.model.load_weights(self.log_dir + 'best_model.tf')
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        input_test = self.input_test
        target_test = self.target_test
        groundtruth = []
        preds = []

        for i in range(len(input_test)):
            yhat = self.model.predict(input_test[i].reshape(1, input_test[i].shape[0], input_test[i].shape[1]))
            preds.append(yhat[0]+0.01)
            groundtruth.append(target_test[i])

        groundtruth = np.array(groundtruth)
        preds = np.array(preds)

        for i in range(groundtruth.shape[1]):
            gt = groundtruth[:, i].copy()
            pd = preds[:, i].copy()
            error_mae, error_rmse, error_mape = common_util.cal_error(gt.reshape(-1), pd.reshape(-1))
            with open(os.path.join(self.log_dir+"test_metric.txt"), 'a') as f:
                f.write('List errors of feature {:d}: MAE = {:.4f} ---- RMSE = {:.4f} ---- MAPE = {:.4f}'.format(i, error_mae, error_rmse, error_mape))
            print('List errors of feature {:d}: MAE = {:.4f} ---- RMSE = {:.4f} ---- MAPE = {:.4f}'.format(i, error_mae, error_rmse, error_mape))

#dataset = read_csv('/home/ad/PycharmProjects/build_models/ML_platform/data/hanoi_data_median.csv').to_numpy()

#with open('/home/ad/PycharmProjects/build_models/ML_platform/config/RNN/rnn.yaml','r') as f:
#    config = yaml.load(f)
#data = utils.load_dataset(**config)
#net = Nets(**config)
#net.train()
#net.train()

