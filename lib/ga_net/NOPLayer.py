#
# NOPLayer.py
#
# Do nothing!
# This is used for NOP layers in the generated architecture as a way to provide variable number of
# layers in the network
#
# Written by Mark Strefford
# (c) 2021 Delirium Digital Ltd
#

import tensorflow as tf
from tensorflow.keras.layers import Layer
from numpy import pad


class NOPLayer(Layer):

    def __init__(self, **kwargs):
        super(NOPLayer, self).__init__(**kwargs)

    def call(self, x):
        return x
