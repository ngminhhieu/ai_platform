# gsmap_adjustment

There are 2 prefix that need to know:

In file config.yaml:

1.type: type of rnn cell, there 3 types available: gru, lstm, rnn ('it's just simple rnn cell')

2.rnn_layers: this is the number of rnn cell types above(default: 1, increase to stack cells)