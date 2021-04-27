#
# net_generator.py
#
# Generate a NN based on the individual spec from DEAP
#
# Written by Mark Strefford
# (c) 2021 Delirium Digital Ltd
#

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, \
    Flatten, Dense, Conv2DTranspose, Add, Dropout, Input, Layer, Concatenate, add, ZeroPadding2D
from tensorflow.keras.losses import BinaryCrossentropy, MSE
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import initializers
from tensorflow.keras import Model

import sys
sys.path.append('../')
from core.loss import EuclideanLoss
from ga_net.PaddingLayer import PaddingLayer, calculate_max_input_size
from ga_net.NOPLayer import NOPLayer


def net():
