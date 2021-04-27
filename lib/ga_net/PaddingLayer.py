#
# PaddingLayer.py
#
# Pad a 4D tensor (batch, dim1, dim2, dim3) according to a provided 3D array
# Batch size is not padded
#
# Written by Mark Strefford
# (c) 2021 Delirium Digital Ltd
#

import tensorflow as tf
from tensorflow.keras.layers import Layer


def calculate_max_input_size(inbound_connections):
    """
        Take a list of inbound connections and perform the following actions:

        If no inbound connections:
            add input as the inbound connection

        If a single inbound connection,
            return the source layer

        If multiple inbound connections,
            1) Find the maximum shape for all the inbound connections
            2) Add padding layers for each inbound connection to ensure the same shape
            3) Add a concat layer to bring all the inbound connectioons together

        Input:
            inbound_connections: List of inbound layers (indexed into params)

        Returns:
            The last layer
    """
    # Calculate max input shape across inbound layers
    # Works with tensors of shape [batch_size, dim1, dim2, dim3]
    # TODO: Can we ignore dim1?
    max_input_shape = [0, -1, -1, -1]
    for inbound_connection in inbound_connections:
        for k in range(1, 4, 1):
            if max_input_shape[k] < inbound_connection.shape[k]:
                max_input_shape[k] = inbound_connection.shape[k]
    return max_input_shape


class PaddingLayer(Layer):
    """
    Padding Layer designed to add padding to a tensor ahead of a Concat layer
    TODO: Can this be replaced with Zeropadding2D?
    """

    def __init__(self, padding_shape=[0, 0, 0, 0], merge_type='Concatenate', **kwargs):
        super(PaddingLayer, self).__init__(**kwargs)
        # print(f'PaddingLayer.__init__(): padding_shape={padding_shape}')
        self.padding_shape = padding_shape
        self.padding = None
        self.merge_type = merge_type

    # https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding_shape': self.padding_shape,
            'padding': self.padding
        })
        return config

    def _pad(self, x):
        # At first run, calculate padding required by call()
        if not self.padding:

            dim1_p0 = self.padding_shape[1] - x.shape[1]
            dim1_p1 = self.padding_shape[1] - x.shape[1] - dim1_p0

            dim2_p0 = self.padding_shape[2] - x.shape[2]
            dim2_p1 = self.padding_shape[2] - x.shape[2] - dim2_p0

            dim3_p0 = 0
            dim3_p1 = 0

            # if self.merge_type == 'Concatenate':
            #     dim3_p0 = 0
            #     dim3_p1 = 0
            # elif self.merge_type == 'add':
            #     #
            #     dim3_p0 = self.padding_shape[3] - x.shape[3]
            #     dim3_p1 = self.padding_shape[3] - x.shape[3] - dim3_p0
            # else:
            #     exception = f'Merge type {self.merge_type} not supported in PaddingLayer'
            #     raise Exception(exception)

            self.padding = (
                (0, 0),
                (dim1_p0, dim1_p1),
                (dim2_p0, dim2_p1),
                (dim3_p0, dim3_p1)
            )

        return tf.pad(
            x,
            self.padding,
            'constant'
        )

    def call(self, x):
        # print(f'PaddingLayer.call(): x={x}')
        return self._pad(x)
