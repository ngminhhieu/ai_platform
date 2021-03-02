import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
import yaml
import random as rn
from model.cnn.supervisor import Conv2DSupervisor
from model.lstm.supervisor import LSTMSupervisor
from model.ae_lstm.supervisor import AELSTMSupervisor
from model.cnn_lstm_attention.supervisor import Conv1DLSTMAttentionSupervisor
import tensorflow as tf

# allow run multiple command python (sharing GPU)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def checkGPU():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

def seed():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(2)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(1)
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.compat.v1.set_random_seed(1234)


if __name__ == '__main__':
    checkGPU()
    seed()
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='Run mode.')
    parser.add_argument('--model',
                        type=str,
                        help='Select model.')
    args = parser.parse_args()

    # load config for seq2seq model
    if args.config_file != False:
        with open(args.config_file) as f:
            config = yaml.safe_load(f)

    if args.mode == 'train':
        if args.model == 'cnn':
            model = Conv2DSupervisor(**config)
        elif args.model == 'lstm':
            model = LSTMSupervisor(**config)
        elif args.model == 'ae_lstm':
            model = AELSTMSupervisor(**config)
        elif args.model == 'cnn_lstm_attention':
            model = Conv1DLSTMAttentionSupervisor(**config)
        model.train()
    elif args.mode == 'test':
        if args.model == 'cnn':
            model = Conv2DSupervisor(**config)
        elif args.model == 'lstm':
            model = LSTMSupervisor(**config)
        elif args.model == 'ae_lstm':
            model = AELSTMSupervisor(**config)
        elif args.model == 'cnn_lstm_attention':
            model = Conv1DLSTMAttentionSupervisor(**config)
        model.test()
        model.plot_result()
        model.get_inference_time_per_prediction()
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
