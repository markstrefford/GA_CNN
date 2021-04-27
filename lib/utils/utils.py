# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


import os
import logging
import time
from pathlib import Path
import os

# import torch
# import torch.optim as optim

from tensorflow.keras.optimizers import Adam, SGD
from core.config import get_model_name


# Set deep key, adding intermediary ones as required
def set_kv(d, key, value, sep='_'):
    dd = d
    keys = key.split(sep)
    latest = keys.pop()
    for k in keys:
        dd = dd.setdefault(k, {})
    dd.setdefault(latest, value)


def unflatten_dict(d):
    result = {}
    for k in d.keys():
        v = d[k]
        set_kv(result, k, v)
    return result


def create_logger(cfg, cfg_name, phase='train'):
    # root_output_dir = Path(cfg.OUTPUT_DIR)
    root_output_dir = Path(os.path.join(cfg.DATASET.PATH_PREFIX, cfg.OUTPUT_DIR))
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    time_str = str(time.strftime('%Y-%m-%d-%H-%M'))

    final_output_dir = os.path.join(root_output_dir, dataset, model, cfg_name, time_str)

    print('=> creating {}'.format(final_output_dir))
    Path(final_output_dir).mkdir(parents=True, exist_ok=True)

    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(os.path.join(cfg.DATASET.PATH_PREFIX, cfg.LOG_DIR,
                                            dataset, model, f'{cfg_name}_{time_str}'))
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg):
    optimizer = None
    # TODO: SGD not tested!
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = SGD(
            learning_rate=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = Adam(
            learning_rate=cfg.TRAIN.LR
        )
    return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, loss=False, acc=False, name=""):
        self.loss = loss
        self.acc = acc
        self.name = name
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.history.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def early_stop(self, loss=False, acc=False):
        if self.loss and len(self.history) > 2:
            # Latest loss > Previous loss
            if self.history[-1] >= self.history[-2]:
                # print(f'{self.name}_loss: {self.history[-1]} >= {self.history[-2]}')
                return True
            else:
                return False
        elif self.acc and len(self.history) > 2:
            # Latest acc < Previous acc
            if self.history[-1] <= self.history[-2]:
                # print(f'{self.name}_acc: {self.history[-1]} <= {self.history[-2]}')
                return True
            else:
                return False
        else:
            return False


# Setup average metrics for training
def setup_average_metrics(loss=False, acc=False, names=[]):
    metrics = []
    for name in names:
        metrics.append(AverageMeter(loss=loss, acc=acc, name=name))
    return metrics
