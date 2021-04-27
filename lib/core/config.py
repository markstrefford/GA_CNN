# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
#
# Modified for WatchAR by Mark Strefford, Delirium Digital Limited
# TODO: A lot of legacy code remains... needs tidying/removing!
# ------------------------------------------------------------------------------


import os
import yaml

import numpy as np
from easydict import EasyDict as edict

config = edict()

config.OUTPUT_DIR = ''
config.LOG_DIR = ''
config.DATA_DIR = ''
config.DEVICE = ''
config.GPUS = '0'
config.WORKERS = 4
config.CPU_WORKERS = 1
config.PRINT_FREQ = 20

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = 'watch_ar_mobilenet_v2'
config.MODEL.VERSION = '1'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.ALPHA = 1.0
config.MODEL.PRETRAINED = ''
config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.MODEL.LABEL_IMAGE_SIZE = [512, 512]  # image size used in labeler (need to align image with labels!)
config.MODEL.STYLE = 'tensorflow'
# Legacy - need removing
config.MODEL.NUM_JOINTS = 5
config.MODEL.NUM_FLAGS = 5
config.MODEL.EXTRA = edict()

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.PATH_PREFIX = ''
config.DATASET.DATASET = 'custom'
config.DATASET.TRAIN_SET = 'original-images'
config.DATASET.TEST_SET = 'valid'
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.HYBRID_JOINTS_TYPE = ''
config.DATASET.SELECT_DATA = False
config.DATASET.COLLECTION = 'keypoint_annotations'
config.DATASET.NUM_KEYPOINTS = 5
config.DATASET.NUM_FEATURES_PER_KEYPOINT = 4     # x, y, z, visible
config.DATASET.NUM_FLAGS = 5
config.DATASET.CAMERA_ALIGNED_KEYPOINTS = True   # TODO: Remove as defaults to this...

# training data augmentation
config.DATASET.FLIP = True
config.DATASET.CROP = True
config.DATASET.AUGMENT = True
config.DATASET.SCALE_FACTOR = 0.25
config.DATASET.ROT_FACTOR = 30

# train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 250

config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = ''

config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 32
config.TEST.BATCH_SIZE_CPU = 1
# Test Model Epoch
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True

config.TEST.USE_GT_BBOX = False
# nms
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MODEL_FILE = ''
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0
config.TEST.BREMONT_BBOX_FILE = ''

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = False
config.DEBUG.SAVE_BATCH_IMAGES_GT = False
config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
config.DEBUG.SAVE_HEATMAPS_GT = False
config.DEBUG.SAVE_HEATMAPS_PRED = False


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
        config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
        config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.MODEL.PRETRAINED = os.path.join(
        config.DATA_DIR, config.MODEL.PRETRAINED)


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    # extra = cfg.MODEL.EXTRA
    # if name in ['watch_ar_mobilenet_v2', 'pose_mobilenetv2']:
    # name = '{model}'.format(
    #     model=name)
    full_name = '{height}x{width}_{name}'.format(
        height=cfg.MODEL.IMAGE_SIZE[1],
        width=cfg.MODEL.IMAGE_SIZE[0],
        name=name)
    # else:
    #     raise ValueError('Unknown model: {}'.format(cfg.MODEL))

    return name, full_name


if __name__ == '__main__':
    import sys

    gen_config(sys.argv[1])
