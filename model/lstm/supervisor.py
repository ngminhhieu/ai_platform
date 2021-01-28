from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Sequential
import numpy as np
from library import common_util
import model.lstm.utils as utils
import os
import yaml
from pandas import read_csv
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse


class LSTMSupervisor():
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
        self.model = self.construct_model()

    def construct_model(self):
        model = Sequential()
        model.add(LSTM(self.rnn_units, activation=self.activation, return_sequences=True, input_shape=(self.seq_len, self.input_dim)))
        model.add(LSTM(self.rnn_units, activation=self.activation))
        model.add(Dense(self.output_dim))
        print(model.summary())
        # plot model
        plot_model(model=model,
                   to_file=self.log_dir + '/lstm_model.png',
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
        self.model.load_weights(self.log_dir + 'best_model.hdf5')
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
        
        # finding the best
        min_mae = 99999
        best_preds = np.zeros(shape=(preds.shape[0], 1))
        for i in range(groundtruth.shape[1]):
            gt = groundtruth[:, i].copy()
            pd = preds[:, i].copy()
            mae = common_util.mae(gt, pd)
            if mae < min_mae:
                min_mae = mae
                best_preds = pd.copy()
                only_gt = gt.copy()
        
        common_util.save_metrics(np.array(common_util.cal_error(only_gt.flatten(), best_preds.flatten())), self.log_dir, "list_metrics")
        np.savetxt(self.log_dir + 'groundtruth.csv', only_gt, delimiter=",")
        np.savetxt(self.log_dir + 'preds.csv', best_preds, delimiter=",")

    def plot_result(self):
        from matplotlib import pyplot as plt
        preds = read_csv(self.log_dir + 'preds.csv')
        gt = read_csv(self.log_dir + 'groundtruth.csv')
        preds = preds.to_numpy()
        gt = gt.to_numpy()
        for i in range(preds.shape[1]):
            plt.plot(preds[:, i], label='preds')
            plt.plot(gt[:, i], label='gt')
            plt.legend()
            plt.savefig(self.log_dir + 'result_predict_{}.png'.format(i))
            plt.close()