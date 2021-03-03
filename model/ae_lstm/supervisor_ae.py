from tensorflow.keras.layers import Dense, LSTM, Input, Bidirectional
from tensorflow.keras.models import Sequential, Model
import numpy as np
from library import common_util
import model.ae_lstm.utils as utils
import os
import yaml
from tqdm import tqdm
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.losses import mse
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt


class AESupervisor():
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
        self.model = self.construct_model()

        self.timestep = kwargs['model'].get('timestep')

    def construct_model(self):
        model = Sequential()
        model.add(
            Dense(self.latent_space,
                  input_shape=(self.seq_len, self.input_dim),
                  activation=self.activation))
        model.add(Dense(self.input_dim, activation=self.activation))

        plot_model(model=model,
                   to_file=self.log_dir + '/ae_model.png',
                   show_shapes=True)
        return model

    def train(self):
        self.model.compile(optimizer=optimizers.SGD(learning_rate=0.001),
                              loss=self.loss,
                              metrics=['mse', 'mae'])

        training_history = self.model.fit(
            self.input_train,
            self.input_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_data=(self.input_valid, self.input_valid),
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

        outputs = K.function([self.model.input],
                             [self.model.layers[0].output])(
                                 [self.input_train])
        # outputs = [K.function([self.model.input], [layer.output])([self.input_train]) for layer in self.model.layers]
        outputs_ae = np.array(outputs[0])
        outputs = K.function([self.model.input],
                             [self.model.layers[0].output])(
                                 [self.input_valid])
        # outputs = [K.function([self.model.input], [layer.output])([self.input_valid]) for layer in self.model.layers]
        outputs_ae_valid = np.array(outputs[0])
        return outputs_ae, outputs_ae_valid

    def load_weights(self):
        print(self.model.summary())
        self.model.load_weights(self.log_dir + 'best_model.hdf5')
        self.model.compile(optimizer=optimizers.SGD(learning_rate=0.001), loss=self.loss)
        return self.model