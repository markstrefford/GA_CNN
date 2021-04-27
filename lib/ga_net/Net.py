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
	x0 = Input(shape=(256, 256, 3), dtype="float32")
	print(x0)
	# *** Params Layer 0, [  2   6   2   1  -8 -10]
	# *** inbound_connections=[0]
	x1 = Conv2DTranspose(64, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_1')(x0)
	print(x1)
	# *** Params Layer 1, [  2   4   2   1 -13 -13]
	# *** inbound_connections=[0]
	x2 = Conv2DTranspose(16, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_2')(x0)
	print(x2)
	# *** Params Layer 2, [  2   6   1   1 -11  -1]
	# *** inbound_connections=[0]
	x3 = Conv2DTranspose(64, (1, 1), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_3')(x0)
	print(x3)
	# *** Params Layer 3, [  2   4   1   1 -11  -3]
	# *** inbound_connections=[0]
	x4 = Conv2DTranspose(16, (1, 1), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_4')(x0)
	print(x4)
	# *** Params Layer 4, [  0   5   2   1   0 -14]
	# *** inbound_connections=[0]
	x5 = DepthwiseConv2D((2, 2), (1, 1), activation='relu', name='DepthwiseConv2D_5')(x0)
	print(x5)
	# *** Params Layer 5, [  0   4   2   1 -10  -4]
	# *** inbound_connections=[0]
	x6 = DepthwiseConv2D((2, 2), (1, 1), activation='relu', name='DepthwiseConv2D_6')(x0)
	print(x6)
	# *** Params Layer 6, [2 6 1 1 0 5]
	# *** inbound_connections=[0, 5]
	max_input_shape_6 = calculate_max_input_size([x0, x5])
	x7_0 = PaddingLayer(padding_shape=max_input_shape_6, merge_type='Concatenate', name='PaddingLayer_x7_0')(x0)
	print(x7_0)
	x7_1 = PaddingLayer(padding_shape=max_input_shape_6, merge_type='Concatenate', name='PaddingLayer_x7_1')(x5)
	print(x7_1)
	x7_2 = Concatenate(name='Concatenate_x7_2')([x7_0, x7_1])
	print(x7_2)
	x7 = Conv2DTranspose(64, (1, 1), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_7')(x7_2)
	print(x7)
	# *** Params Layer 7, [ 1  6  1  1  0 -1]
	# *** inbound_connections=[0]
	x8 = Conv2D(64, (1, 1), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2D_8')(x0)
	print(x8)
	# *** Params Layer 8, [ 2  4  1  1  3 -3]
	# *** inbound_connections=[3]
	x9 = Conv2DTranspose(16, (1, 1), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_9')(x3)
	print(x9)
	# *** Params Layer 9, [  2   6   2   1 -15  -2]
	# *** inbound_connections=[0]
	x10 = Conv2DTranspose(64, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_10')(x0)
	print(x10)
	# *** Params Layer 10, [ 2  4  2  1  5 -9]
	# *** inbound_connections=[5]
	x11 = Conv2DTranspose(16, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_11')(x5)
	print(x11)
	# *** Params Layer 11, [ 2  4  1  1 10 -7]
	# *** inbound_connections=[10]
	x12 = Conv2DTranspose(16, (1, 1), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_12')(x10)
	print(x12)
	# *** Params Layer 12, [ 1  6  1  1 -6 -9]
	# *** inbound_connections=[0]
	x13 = Conv2D(64, (1, 1), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2D_13')(x0)
	print(x13)
	# *** Params Layer 13, [  0   4   2   1   2 -12]
	# *** inbound_connections=[2]
	x14 = DepthwiseConv2D((2, 2), (1, 1), activation='relu', name='DepthwiseConv2D_14')(x2)
	print(x14)
	# *** Params Layer 14, [  0   6   1   1   0 -13]
	# *** inbound_connections=[0]
	x15 = DepthwiseConv2D((1, 1), (1, 1), activation='relu', name='DepthwiseConv2D_15')(x0)
	print(x15)
	# *** Params Layer 15, [ 1  6  1  1  4 -4]
	# *** inbound_connections=[4]
	x16 = Conv2D(64, (1, 1), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2D_16')(x4)
	print(x16)
	# *** Params Layer 16, [ 0  4  2  1 12  6]
	# *** inbound_connections=[12, 6]
	max_input_shape_16 = calculate_max_input_size([x12, x6])
	x17_0 = PaddingLayer(padding_shape=max_input_shape_16, merge_type='Concatenate', name='PaddingLayer_x17_0')(x12)
	print(x17_0)
	x17_1 = PaddingLayer(padding_shape=max_input_shape_16, merge_type='Concatenate', name='PaddingLayer_x17_1')(x6)
	print(x17_1)
	x17_2 = Concatenate(name='Concatenate_x17_2')([x17_0, x17_1])
	print(x17_2)
	x17 = DepthwiseConv2D((2, 2), (1, 1), activation='relu', name='DepthwiseConv2D_17')(x17_2)
	print(x17)
	# *** Params Layer 17, [  1   5   1   1 -12   5]
	# *** inbound_connections=[5]
	x18 = Conv2D(32, (1, 1), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2D_18')(x5)
	print(x18)
	# *** Params Layer 18, [  1   6   2   1   5 -11]
	# *** inbound_connections=[5]
	x19 = Conv2D(64, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2D_19')(x5)
	print(x19)
	# *** Params Layer 19, [ 1  6  2  1  2 -2]
	# *** inbound_connections=[2]
	x20 = Conv2D(64, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2D_20')(x2)
	print(x20)
	# *** Params Layer 20, [ 1  5  2  1 11 15]
	# *** inbound_connections=[11, 15]
	max_input_shape_20 = calculate_max_input_size([x11, x15])
	x21_0 = PaddingLayer(padding_shape=max_input_shape_20, merge_type='Concatenate', name='PaddingLayer_x21_0')(x11)
	print(x21_0)
	x21_1 = PaddingLayer(padding_shape=max_input_shape_20, merge_type='Concatenate', name='PaddingLayer_x21_1')(x15)
	print(x21_1)
	x21_2 = Concatenate(name='Concatenate_x21_2')([x21_0, x21_1])
	print(x21_2)
	x21 = Conv2D(32, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2D_21')(x21_2)
	print(x21)
	# *** Params Layer 21, [ 2  5  1  1  1 10]
	# *** inbound_connections=[1, 10]
	max_input_shape_21 = calculate_max_input_size([x1, x10])
	x22_0 = PaddingLayer(padding_shape=max_input_shape_21, merge_type='Concatenate', name='PaddingLayer_x22_0')(x1)
	print(x22_0)
	x22_1 = PaddingLayer(padding_shape=max_input_shape_21, merge_type='Concatenate', name='PaddingLayer_x22_1')(x10)
	print(x22_1)
	x22_2 = Concatenate(name='Concatenate_x22_2')([x22_0, x22_1])
	print(x22_2)
	x22 = Conv2DTranspose(32, (1, 1), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_22')(x22_2)
	print(x22)
	# *** Params Layer 22, [ 2  4  2  1 -6 -7]
	# *** inbound_connections=[0]
	x23 = Conv2DTranspose(16, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_23')(x0)
	print(x23)
	# *** Params Layer 23, [ 2  6  2  1 21 11]
	# *** inbound_connections=[11, 21]
	max_input_shape_23 = calculate_max_input_size([x11, x21])
	x24_0 = PaddingLayer(padding_shape=max_input_shape_23, merge_type='Concatenate', name='PaddingLayer_x24_0')(x11)
	print(x24_0)
	x24_1 = PaddingLayer(padding_shape=max_input_shape_23, merge_type='Concatenate', name='PaddingLayer_x24_1')(x21)
	print(x24_1)
	x24_2 = Concatenate(name='Concatenate_x24_2')([x24_0, x24_1])
	print(x24_2)
	x24 = Conv2DTranspose(64, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_24')(x24_2)
	print(x24)
	# *** Params Layer 24, [ 0  4  1  1 -6  1]
	# *** inbound_connections=[1]
	x25 = DepthwiseConv2D((1, 1), (1, 1), activation='relu', name='DepthwiseConv2D_25')(x1)
	print(x25)
	# *** Params Layer 25, [  1   4   2   1 -11  10]
	# *** inbound_connections=[10]
	x26 = Conv2D(16, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2D_26')(x10)
	print(x26)
	# *** Params Layer 26, [ 1  4  2  1 21 -7]
	# *** inbound_connections=[21]
	x27 = Conv2D(16, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2D_27')(x21)
	print(x27)
	# *** Params Layer 27, [ 0  6  2  1  7 -6]
	# *** inbound_connections=[7]
	x28 = DepthwiseConv2D((2, 2), (1, 1), activation='relu', name='DepthwiseConv2D_28')(x7)
	print(x28)
	# *** Params Layer 28, [  0   4   1   1  -8 -15]
	# *** inbound_connections=[0]
	x29 = DepthwiseConv2D((1, 1), (1, 1), activation='relu', name='DepthwiseConv2D_29')(x0)
	print(x29)
	# *** Params Layer 29, [  0   5   1   1 -13 -14]
	# *** inbound_connections=[0]
	x30 = DepthwiseConv2D((1, 1), (1, 1), activation='relu', name='DepthwiseConv2D_30')(x0)
	print(x30)
	# *** Params Layer 30, [ 2  5  2  1 15 16]
	# *** inbound_connections=[16, 15]
	max_input_shape_30 = calculate_max_input_size([x16, x15])
	x31_0 = PaddingLayer(padding_shape=max_input_shape_30, merge_type='Concatenate', name='PaddingLayer_x31_0')(x16)
	print(x31_0)
	x31_1 = PaddingLayer(padding_shape=max_input_shape_30, merge_type='Concatenate', name='PaddingLayer_x31_1')(x15)
	print(x31_1)
	x31_2 = Concatenate(name='Concatenate_x31_2')([x31_0, x31_1])
	print(x31_2)
	x31 = Conv2DTranspose(32, (2, 2), (1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Conv2DTranspose_31')(x31_2)
	print(x31)
	# *** Params Layer 31, [  0   5   2   1   2 -11]
	# *** inbound_connections=[2]
	x32 = DepthwiseConv2D((2, 2), (1, 1), activation='relu', name='DepthwiseConv2D_32')(x2)
	print(x32)
	# *** orphans=[8, 9, 13, 14, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30]
	# *** Output Layer: {'name': 'joints_output', 'features': 48, 'activation': 'tanh', 'loss': 'EuclideanLoss'}
	joints_output_flatten = Flatten(name='Flatten_joints_output_flatten')(x31)
	print(joints_output_flatten)
	joints_output = Dense(48, activation='tanh', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Dense_joints_output')(joints_output_flatten)
	print(joints_output)
	# *** Output Layer: {'name': 'joints_vis_output', 'features': 16, 'activation': 'sigmoid', 'loss': '"BinaryCrossentropy"'}
	joints_vis_output_flatten = Flatten(name='Flatten_joints_vis_output_flatten')(x32)
	print(joints_vis_output_flatten)
	joints_vis_output = Dense(16, activation='sigmoid', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(), name='Dense_joints_vis_output')(joints_vis_output_flatten)
	print(joints_vis_output)
	model = Model(inputs=x0, outputs=[joints_output, joints_vis_output, ], name='model_0')
	optimizer = Adam(learning_rate=0.001)
	model.compile(loss=[EuclideanLoss, "BinaryCrossentropy", ], metrics=["accuracy"], optimizer=optimizer)
	return model
	# [  2   6   2   1  -8 -10]
	# [  2   4   2   1 -13 -13]
	# [  2   6   1   1 -11  -1]
	# [  2   4   1   1 -11  -3]
	# [  0   5   2   1   0 -14]
	# [  0   4   2   1 -10  -4]
	# [2 6 1 1 0 5]
	# [ 1  6  1  1  0 -1]
	# [ 2  4  1  1  3 -3]
	# [  2   6   2   1 -15  -2]
	# [ 2  4  2  1  5 -9]
	# [ 2  4  1  1 10 -7]
	# [ 1  6  1  1 -6 -9]
	# [  0   4   2   1   2 -12]
	# [  0   6   1   1   0 -13]
	# [ 1  6  1  1  4 -4]
	# [ 0  4  2  1 12  6]
	# [  1   5   1   1 -12   5]
	# [  1   6   2   1   5 -11]
	# [ 1  6  2  1  2 -2]
	# [ 1  5  2  1 11 15]
	# [ 2  5  1  1  1 10]
	# [ 2  4  2  1 -6 -7]
	# [ 2  6  2  1 21 11]
	# [ 0  4  1  1 -6  1]
	# [  1   4   2   1 -11  10]
	# [ 1  4  2  1 21 -7]
	# [ 0  6  2  1  7 -6]
	# [  0   4   1   1  -8 -15]
	# [  0   5   1   1 -13 -14]
	# [ 2  5  2  1 15 16]
	# [  0   5   2   1   2 -11]
