from keras.layers import Dense, LSTM, Input, GRU, SimpleRNN, Layer
import tensorflow as tf

#multi cells layer Gru
class GRUs(Layer):
    def __init__(self, units, num_cells = 1, go_backwards =False, dropout = 0.,
                 return_sequences = True, return_state = False, name = None, l2 = 0., activation = 'tanh', *args, **kwargs):
        super(GRUs, self).__init__(name=name)
        self.l2 = None if l2 == 0 else tf.keras.regularizers.L2(l2)
        self.num_cells = num_cells
        self.dim = units
        self.dropout = dropout
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.activation = activation
        self.states = None
        def gruCell():
            return GRU(units, dropout= self.dropout, go_backwards= self.go_backwards,
                                       return_state=True, return_sequences=True, stateful=False,
                                       kernel_regularizer=self.l2, bias_regularizer=self.l2, activation=activation)

        self._layers_name = ['GruCell_' +str(i) for i in range(num_cells)]
        for name in self._layers_name:
            self.__setattr__(name, gruCell())
    def get_config(self):
        config = super(GRUs, self).get_config().copy()
        config.update({
            'num_cells': self.num_cells,
            'dropout': self.dropout,
            'go_backwards': self.go_backwards,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            '_layers_name': self._layers_name,
            'units': self.dim,
            'activation': self.activation
        })
        return config

    def call(self, inputs):
        seq = inputs
        state = None
        for name in self._layers_name:
            cell = self.__getattribute__(name)
            (seq, state) = cell(seq, initial_state=state)
        self.states = state
        if self.return_state:
            if self.return_sequences:
                return [seq, state]
            return [seq[:, -1, :], state]
        if self.return_sequences:
            return seq
        return seq[:, -1, :]

#multi cells layer RNN
class RNNs(Layer):
    def __init__(self, units, num_cells=1, go_backwards=False, dropout=0.,
                 return_sequences=True, return_state=False, name=None, l2=0., activation = 'tanh', *args, **kwargs):
        super(RNNs, self).__init__(name=name)
        self.l2 = None if l2 == 0 else tf.keras.regularizers.L2(l2)
        self.num_cells = num_cells
        self.dim = units
        self.dropout = dropout
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.states = None
        self.activation = activation

        def rnnCell():
            return SimpleRNN(units, dropout=self.dropout, go_backwards=self.go_backwards,
                       return_state=True, return_sequences=True, stateful=False,
                       kernel_regularizer=self.l2, bias_regularizer=self.l2, activation=activation)

        self._layers_name = ['RnnCell_' + str(i) for i in range(num_cells)]
        for name in self._layers_name:
            self.__setattr__(name, rnnCell())

    def get_config(self):
        config = super(RNNs, self).get_config().copy()
        config.update({
            'num_cells': self.num_cells,
            'dropout': self.dropout,
            'go_backwards': self.go_backwards,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            '_layers_name': self._layers_name,
            'units': self.dim,
            'activation': self.activation
        })
        return config

    def call(self, inputs):
        seq = inputs
        state = None
        for name in self._layers_name:
            cell = self.__getattribute__(name)
            (seq, state) = cell(seq, initial_state=state)
        self.states = state
        if self.return_state:
            if self.return_sequences:
                return [seq, state]
            return [seq[:, -1, :], state]
        if self.return_sequences:
            return seq
        return seq[:, -1, :]

#multi cells LSTM
class LSTMs(Layer):
    def __init__(self, units, num_cells=1, go_backwards=False, dropout=0.,
                 return_sequences=True, return_state=False, name=None, l2=0., activation = 'tanh', *args, **kwargs):
        super(LSTMs, self).__init__(name=name)
        self.l2 = None if l2 == 0 else tf.keras.regularizers.L2(l2)
        self.num_cells = num_cells
        self.dim = units
        self.dropout = dropout
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.states = None
        self.activation = activation

        def lstmCell():
            return LSTM(units, dropout=self.dropout, go_backwards=self.go_backwards,
                       return_state=True, return_sequences=True, stateful=False,
                       kernel_regularizer=self.l2, bias_regularizer=self.l2, activation=activation)

        self._layers_name = ['GruCell_' + str(i) for i in range(num_cells)]
        for name in self._layers_name:
            self.__setattr__(name, lstmCell())

    def get_config(self):
        config = super(LSTMs, self).get_config().copy()
        config.update({
            'num_cells': self.num_cells,
            'dropout': self.dropout,
            'go_backwards': self.go_backwards,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            '_layers_name': self._layers_name,
            'units': self.dim,
            'activation': self.activation
        })
        return config

    def call(self, inputs):
        seq = inputs
        state_C, state_H = None, None
        initial_states = None
        if state_H is not None:
            initial_states = [state_H, state_C]
        for name in self._layers_name:
            cell = self.__getattribute__(name)
            if state_H is not None:
                inital_states = [state_H, state_C]
            (seq, state_H, state_C) = cell(seq, initial_state=initial_states)
        self.states = [state_H, state_C]
        if self.return_state:
            if self.return_sequences:
                return [seq, state_H, state_C]
            return [seq[:, -1, :], state_H, state_C]
        if self.return_sequences:
            return seq
        return seq[:, -1, :]