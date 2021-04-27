"""
Simplified transforms.py

(c) Delirium Digital Limited 2020
Written by Mark Strefford

"""

import cv2
import numpy as np
import math
# from scipy.spatial.transform import Rotation as R
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

def resize_image(data_numpy, x_scale, y_scale):
    return cv2.resize(data_numpy, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)


def rotate_image(image, angle):
    """
    Rotate image around center by angle (degrees)
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def scale_keypoints(keypoints, x_scale=1, y_scale=1, z_scale=1):
    """
    Scale keypoints based on x and y scales (useful for image crop and resize)
    """
    scaled_keypoints = []
    scale_matrix = np.array([x_scale, y_scale, z_scale])
    for k in keypoints:
        scaled_keypoints.append(k * scale_matrix)
    return np.array(scaled_keypoints)


def center_keypoints(keypoints, center_x, center_y):
    """
    Center keypoints

    TODO: Is this needed?
    """
    centered_keypoints = []
    for keypoint in keypoints:
        kp = [center_x + keypoint[0], center_y - keypoint[1], keypoint[2]]
        centered_keypoints.append(kp)
    return np.array(centered_keypoints)


def offset_keypoints(keypoints, x_offset=0, y_offset=0, z_offset=0):
    keypoints[:, 0] += x_offset
    keypoints[:, 1] += y_offset
    keypoints[:, 2] += z_offset
    return keypoints


def flip_keypoints(keypoints):
    """
    Flipped keypoints horizontally (x = -x)
    """
    for i, k in enumerate(keypoints):
        keypoints[i, 0] = -keypoints[i, 0]
    return keypoints


def rotate(x, y, xo, yo, theta):
    """rotate x,y around xo,yo by theta (rad)"""
    xr = math.cos(theta) * (x - xo) - math.sin(theta) * (y - yo) + xo
    yr = math.sin(theta) * (x - xo) + math.cos(theta) * (y - yo) + yo
    return [xr, yr]


def rotate_keypoints(keypoints, origin, angle):
    """
    Rotate keypoints
    Doesn't impact keypoints_vis as by this point the image is square (TODO: Validate this is true!)
    :param keypoints:
    :param angle: Angle in degrees (to align with rotate_image and opencv angles)
    :param origin: point to rotate around
    :return keypoints:
    """
    rads = -np.deg2rad(angle)
    rotated_keypoints = []
    for keypoint in keypoints:
        kp_r = rotate(keypoint[0], keypoint[1], origin[0], origin[1], rads)
        rotated_keypoints.append(kp_r)
    return rotated_keypoints


def flip_back(output_flipped, matched_parts):
    """
    output_flipped: numpy.ndarray(batch_size, num_keypoints, height, width)
    """
    assert output_flipped.ndim == 4, \
        'output_flipped should be [batch_size, num_keypoints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


# From https://github.com/nsantavas/Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach/blob/2136e93586f0d0eb40518f47d2b9f78c6220ea33/model_train.py#L62
def augment_image(image):
    """
    Augment dataset with random brightness, saturation, contrast, and image quality. Then it is casted to bfloat16 and normalized
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=70, max_jpeg_quality=100)
    image = tf.cast(image, tf.float32)
    image = tf.image.random_brightness(image, max_delta=25 / 255)
    image = tf.image.random_saturation(image, lower=0.3, upper=1.7)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.7)
    image = tf.cast(image, tf.float32)
    return image


# ------------------------------------------------------------------------------
# Code below from https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/utils/transforms.py
#
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0]], joints_vis[pair[1]] = \
            joints_vis[pair[1]], joints_vis[pair[0]].copy()

    _joints_vis = np.transpose(np.array([joints_vis, joints_vis, joints_vis]), axes=[1, 0])
    return joints*_joints_vis, joints_vis


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img