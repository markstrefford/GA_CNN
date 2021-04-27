# ------------------------------------------------------------------------------
# Written by Mark Strefford, Delirium Digital Limited
#
# Original pytorch code Copyright (c) Microsoft
# Licensed under the MIT License.
# From https://github.com/microsoft/human-pose-estimation.pytorch
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.losses import MSE


def EuclideanLoss(output, target):
    l2_norm = tf.norm(output-target, ord='euclidean')
    return l2_norm


# def JointsMSELoss(output, target, criterion=MSE):  # target_weight):
#     batch_size = output.shape[0]
#     num_joints = output.shape[3]
#     heatmaps_pred = tf.transpose(tf.reshape(output, [batch_size, num_joints, -1]), [1, 0, 2])  # .split(1, 1)
#     heatmaps_gt = tf.transpose(tf.reshape(target, [batch_size, num_joints, -1]), [1, 0, 2])  # .split(1, 1)
#
#     loss = tf.constant(0, dtype=tf.float32)
#
#     for idx in range(num_joints):
#         heatmap_pred = tf.squeeze(heatmaps_pred[idx])
#         heatmap_gt = tf.squeeze(heatmaps_gt[idx])
#         loss = tf.math.add(loss, tf.math.reduce_mean(tf.math.scalar_mul(0.5, criterion(heatmap_pred, heatmap_gt))))
#
#     return loss / num_joints
