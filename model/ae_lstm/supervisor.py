from keras.layers import Dense, LSTM, Input, Bidirectional
from keras.models import Sequential
import numpy as np
from library import common_util
import model.ae_lstm.utils as utils
import os
import yaml
from tqdm import tqdm
from keras.utils import plot_model
from keras import backend as K
from keras import optimizers
from keras.losses import mse
import pandas as pd
import matplotlib.pyplot as plt


class AELSTMSupervisor():
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
        self.dropout = self.config_model['dropout']
        self.latent_space = 10
        self.model_ae = self.construct_model_ae()
        self.model_lstm = self.construct_model_lstm()

    def construct_model_ae(self):
        model = Sequential()
        model.add(Dense(self.latent_space, input_shape=(self.seq_len, self.input_dim), activation=self.activation))
        # model.add(Bidirectional(LSTM(self.rnn_units, activation=self.activation, dropout=self.dropout)))
        # model.add(Dense(1, activation=self.activation))
        from keras.utils import plot_model
        plot_model(model=model,
                   to_file=self.log_dir + '/ae_model.png',
                   show_shapes=True)
        return model

    def construct_model_lstm(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(self.rnn_units, input_shape=(self.seq_len, self.latent_space), activation=self.activation, dropout=self.dropout)))
        model.add(Dense(1, activation=self.activation))
        from keras.utils import plot_model
        plot_model(model=model,
                   to_file=self.log_dir + '/lstm_model.png',
                   show_shapes=True)
        return model

    def train(self):
        self.model_ae.compile(optimizer=optimizers.SGD(learning_rate=0.001),
                           loss=self.loss,
                           metrics=['mse', 'mae'])

        self.model_ae.fit(self.input_train,
            self.input_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_data=(self.input_valid,
                            self.input_valid),
            shuffle=True,
            verbose=2)

        outputs = [K.function([self.model_ae.input], [layer.output])([self.input_train]) for layer in self.model_ae.layers]
        outputs_ae = np.array(outputs[0][0])
        outputs = [K.function([self.model_ae.input], [layer.output])([self.input_valid]) for layer in self.model_ae.layers]
        outputs_ae_valid = np.array(outputs[0][0])

        self.model_lstm.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                           loss=self.loss,
                           metrics=['mse', 'mae'])

        training_history = self.model.fit(outputs_ae,
                                          self.target_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          callbacks=self.callbacks,
                                          validation_data=(self.outputs_ae_valid,
                                                           self.target_valid),
                                          shuffle=True,
                                          verbose=2)

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
        scaler = self.data['scaler']
        data_test = self.data['test_data_norm'].copy()
        # this is the meterogical data
        other_features_data = data_test[:, 0:(self.input_dim -
                                              self.output_dim)].copy()
        pm_data = data_test[:, -self.output_dim:].copy()
        T = len(data_test)
        l = self.seq_len
        h = self.horizon
        _pd = np.zeros(shape=(T, self.output_dim), dtype='float32')
        _pd[:l] = pm_data[:l]
        iterator = tqdm(range(0, T - l - h, h))
        for i in iterator:
            if i + l + h > T - h:
                # trimm all zero lines
                # pd = pd[~np.all(pd==0, axis=1)]
                _pd = _pd[~np.all(_pd == 0, axis=1)]
                iterator.close()
                break
            input = np.zeros(shape=(1, l, self.input_dim))
            input[0, :, :] = data_test[i:i + l].copy()
            # yhats = self.model.predict(input)
            outputs_ae = self.model_ae.predict(input)
            yhats = self.model_lstm.predict(outputs_ae)
            _pd[i + l:i + l + h] = yhats

        # rescale metrics
        residual_row = len(other_features_data) - len(_pd)
        if residual_row != 0:
            other_features_data = np.delete(other_features_data,
                                            np.s_[-residual_row:],
                                            axis=0)
        inverse_pred_data = scaler.inverse_transform(
            np.concatenate((other_features_data, _pd), axis=1))
        predicted_data = inverse_pred_data[:, -self.output_dim:]
        inverse_actual_data = scaler.inverse_transform(
            data_test[:predicted_data.shape[0]])
        ground_truth = inverse_actual_data[:, -self.output_dim:]
        np.save(self.log_dir + 'pd', predicted_data)
        np.save(self.log_dir + 'gt', ground_truth)
        # save metrics to log dir
        error_list = utils.cal_error(ground_truth.flatten(),
                                           predicted_data.flatten())
        mae = utils.mae(ground_truth.flatten(), predicted_data.flatten())
        utils.save_metrics(error_list, self.log_dir, "ae_lstm")
        return mae

    def plot_result(self):
        preds = np.load(self.log_dir + 'pd.npy')
        gt = np.load(self.log_dir + 'gt.npy')
        if preds.shape[1] == 1 and gt.shape[1] == 1:
            pd.DataFrame(preds).to_csv(self.log_dir + "prediction_values.csv",
                                       header=['PM2.5'],
                                       index=False)
            pd.DataFrame(gt).to_csv(self.log_dir + "grouthtruth_values.csv",
                                    header=['PM2.5'],
                                    index=False)
        else:
            pd.DataFrame(preds).to_csv(self.log_dir + "prediction_values.csv",
                                       header=['PM10', 'PM2.5'],
                                       index=False)
            pd.DataFrame(gt).to_csv(self.log_dir + "grouthtruth_values.csv",
                                    header=['PM10', 'PM2.5'],
                                    index=False)

        for i in range(preds.shape[1]):
            plt.plot(preds[:, i], label='preds')
            plt.plot(gt[:, i], label='gt')
            plt.legend()
            plt.savefig(self.log_dir +
                        '[result_predict]output_dim_{}.png'.format(str(i + 1)))
            plt.close()