# ------------------------------------------------------------------------------
# Modified from https://github.com/microsoft/human-pose-estimation.pytorch/tree/master/lib/dataset
#
# Original code Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from collections import OrderedDict
import logging
import os
import json_tricks as json
import cv2
import random
import copy

from utils import zipreader
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import numpy as np
from scipy.io import loadmat, savemat

logger = logging.getLogger(__name__)


class MPIIDataset:
    def __init__(self, cfg, batch_size, is_train, sample_size=5000,
                 transform=None, random_order=True, random_seed=None):

        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.batch_size = batch_size

        self.dataset = cfg.DATASET.DATASET
        self.data_path = os.path.join(cfg.DATASET.PATH_PREFIX, 'data', cfg.DATASET.DATASET)
        self.is_train = is_train
        self.image_set = 'train' if self.is_train else 'val'
        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        # self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.transform = transform

        self.random_order = random_order
        self.random_seed = random_seed

        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.sample_size = sample_size
        self.db = self._get_db()
        random.shuffle(self.db)

        # TODO: Is this needed?
        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        # Random selection (MPII has 20k+ samples which take a long time to train)
        # if self.sample_size < len(self.db):
        #     self.db = random.choices(self.db, k=self.sample_size)

        self.num_samples = len(self.db)

        logger.info('=> load {} samples'.format(self.num_samples))

    def shuffle(self):
        if self.random_seed:
            random.seed(a=self.random_seed)
        if self.random_order:
            random.shuffle(self.db)

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(self.data_path,
                                 'annot',
                                 self.image_set+'.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_vis = np.zeros(self.num_joints, dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append({
                'image': os.path.join(self.data_path, image_dir, image_name),
                'center': c,
                'scale': s,
                'joints_3d': joints_3d,
                'joints_vis': joints_vis,
                'filename': '',
                'imgnum': 0,
                })

        return gt_db

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, batch_id):
        images = np.zeros((self.batch_size, self.image_size[1], self.image_size[0], 3), dtype=np.float32)
        joints = np.zeros((self.batch_size, self.num_joints * 3), dtype=np.float32)
        joints_vis = np.zeros((self.batch_size, self.num_joints), dtype=np.float32)
        rotations = []
        image_files = []
        imgnums = []
        centers = []
        scales = []
        scores = []

        for i in range(self.batch_size):
            idx = batch_id * self.batch_size + i
            if idx > self.num_samples:
                break

            db_rec = copy.deepcopy(self.db[idx])

            image_file = db_rec['image']
            filename = db_rec['filename'] if 'filename' in db_rec else ''
            imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

            if self.data_format == 'zip':
                image = zipreader.imread(
                    image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            else:
                image = cv2.imread(
                    image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            if image is None:
                logger.error('=> fail to read {}'.format(image_file))
                raise ValueError('Fail to read {}'.format(image_file))

            _joints = db_rec['joints_3d']
            _joints_vis = db_rec['joints_vis']

            c = db_rec['center']
            s = db_rec['scale']
            score = db_rec['score'] if 'score' in db_rec else 1
            r = 0

            if self.is_train:
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                    if random.random() <= 0.6 else 0

                if self.flip and random.random() <= 0.5:
                    data_numpy = image[:, ::-1, :]
                    _joints, _joints_vis = fliplr_joints(
                        _joints, _joints_vis, image.shape[1], self.flip_pairs)
                    c[0] = image.shape[1] - c[0] - 1

            trans = get_affine_transform(c, s, r, self.image_size)
            image = cv2.warpAffine(
                image,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)

            if self.transform:
                image = self.transform(image)

            for j in range(self.num_joints):
                if _joints_vis[j] > 0.0:
                    _joints[j, 0:2] = affine_transform(_joints[j, 0:2], trans)

            # target, target_weight = self.generate_target(joints, joints_vis)

            # target = torch.from_numpy(target)
            # target_weight = torch.from_numpy(target_weight)

            images[i] = image
            joints[i] = _joints.flatten()
            joints_vis[i] = _joints_vis
            rotations.append(r)
            image_files.append(filename)
            imgnums.append(imgnum)
            centers.append(c)
            scales.append(s)
            scores.append(score)

        images = tf.cast(preprocess_input(images), tf.float32)
        joints = tf.convert_to_tensor(joints, dtype=tf.float32)
        joints_vis = tf.convert_to_tensor(joints_vis, dtype=tf.float32)

        meta = {
            'images': image_files,
            'imgnums': imgnums,
            'joints': joints,
            'joints_vis': joints_vis,
            'centers': centers,
            'scales': scales,
            'rotations': rotations,
            'scores': scores
        }

        return images, joints, joints_vis, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0 * (diff_norm2**2) / (0.2 ** 2 * 2.0 * area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    # def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
    #     # convert 0-based index to 1-based index
    #     preds = preds[:, :, 0:2] + 1.0
    #
    #     if output_dir:
    #         pred_file = os.path.join(output_dir, 'pred.mat')
    #         savemat(pred_file, mdict={'preds': preds})
    #
    #     if 'test' in cfg.DATASET.TEST_SET:
    #         return {'Null': 0.0}, 0.0
    #
    #     SC_BIAS = 0.6
    #     threshold = 0.5
    #
    #     gt_file = os.path.join(cfg.DATASET.ROOT,
    #                            'annot',
    #                            'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
    #     gt_dict = loadmat(gt_file)
    #     dataset_joints = gt_dict['dataset_joints']
    #     jnt_missing = gt_dict['jnt_missing']
    #     pos_gt_src = gt_dict['pos_gt_src']
    #     headboxes_src = gt_dict['headboxes_src']
    #
    #     pos_pred_src = np.transpose(preds, [1, 2, 0])
    #
    #     head = np.where(dataset_joints == 'head')[1][0]
    #     lsho = np.where(dataset_joints == 'lsho')[1][0]
    #     lelb = np.where(dataset_joints == 'lelb')[1][0]
    #     lwri = np.where(dataset_joints == 'lwri')[1][0]
    #     lhip = np.where(dataset_joints == 'lhip')[1][0]
    #     lkne = np.where(dataset_joints == 'lkne')[1][0]
    #     lank = np.where(dataset_joints == 'lank')[1][0]
    #
    #     rsho = np.where(dataset_joints == 'rsho')[1][0]
    #     relb = np.where(dataset_joints == 'relb')[1][0]
    #     rwri = np.where(dataset_joints == 'rwri')[1][0]
    #     rkne = np.where(dataset_joints == 'rkne')[1][0]
    #     rank = np.where(dataset_joints == 'rank')[1][0]
    #     rhip = np.where(dataset_joints == 'rhip')[1][0]
    #
    #     jnt_visible = 1 - jnt_missing
    #     uv_error = pos_pred_src - pos_gt_src
    #     uv_err = np.linalg.norm(uv_error, axis=1)
    #     headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    #     headsizes = np.linalg.norm(headsizes, axis=0)
    #     headsizes *= SC_BIAS
    #     scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    #     scaled_uv_err = np.divide(uv_err, scale)
    #     scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    #     jnt_count = np.sum(jnt_visible, axis=1)
    #     less_than_threshold = np.multiply((scaled_uv_err <= threshold),
    #                                       jnt_visible)
    #     PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)
    #
    #     # save
    #     rng = np.arange(0, 0.5+0.01, 0.01)
    #     pckAll = np.zeros((len(rng), 16))
    #
    #     for r in range(len(rng)):
    #         threshold = rng[r]
    #         less_than_threshold = np.multiply(scaled_uv_err <= threshold,
    #                                           jnt_visible)
    #         pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
    #                                  jnt_count)
    #
    #     PCKh = np.ma.array(PCKh, mask=False)
    #     PCKh.mask[6:8] = True
    #
    #     jnt_count = np.ma.array(jnt_count, mask=False)
    #     jnt_count.mask[6:8] = True
    #     jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)
    #
    #     name_value = [
    #         ('Head', PCKh[head]),
    #         ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
    #         ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
    #         ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
    #         ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
    #         ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
    #         ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
    #         ('Mean', np.sum(PCKh * jnt_ratio)),
    #         ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
    #     ]
    #     name_value = OrderedDict(name_value)
    #
    #     return name_value, name_value['Mean']
