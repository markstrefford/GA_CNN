#
# train.py
#
# Generate a NN based on the individual spec from DEAP
# then train it to determine it's accuracy
#
# Written by Mark Strefford
# (c) 2021 Delirium Digital Ltd
#

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, \
    Flatten, Dense, Conv2DTranspose, Add, Dropout, Input, Layer, ZeroPadding3D
from tensorflow.keras.losses import MSE  # BinaryCrossentropy
from tensorflow.nn import sigmoid_cross_entropy_with_logits
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Model

import os
import numpy as np
import time
import sys
from statistics import mean
from datetime import datetime

from ga_net import Net
from importlib import reload  # https://docs.python.org/3/library/importlib.html#importlib.reload

sys.path.append('../lib/')

from utils.utils import create_logger, get_optimizer, AverageMeter, setup_average_metrics
from utils.vis import save_debug_images
from ga_net.net_generator import NetGenerator


def count_params(weights):
    """Count the total number of scalars composing the weights.
    From https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/utils/layer_utils.py#L94
    Arguments:
        weights: An iterable containing the weights on which to compute params
    Returns:
        The total number of scalars composing the weights
    """
    unique_weights = {id(w): w for w in weights}.values()
    weight_shapes = [w.shape.as_list() for w in unique_weights]
    standardized_weight_shapes = [
        [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
    ]
    return int(sum(np.prod(p) for p in standardized_weight_shapes))


class Trainer:
    """
    Class to generate and train models and return scores back to the Generative Algorithm
    """

    def __init__(self, random_seed, dataset, output_dir,
                 layer_types=None,
                 num_layers=-1, num_chromosomes=12,
                 num_epochs=300, early_stop_epoch=20, activation='relu',
                 merge_type='Concatenate',
                 output_layer_config=None, input_shape=(256, 256, 3),
                 trainable_weights_penalty=12000000,
                 max_trainable_weights=250000000,
                 debug_net_build=False, config=None):

        self.random_seed = random_seed
        self.num_layers = num_layers
        self.num_chromosomes = num_chromosomes
        self.dataset = dataset
        self.output_dir = output_dir
        self.model = None
        self.layer_types = layer_types
        self.num_epochs = num_epochs
        self.early_stop_epoch = early_stop_epoch
        self.config = config
        self.generation = -1  # So first increment starts generations at 0
        self.input_shape = input_shape
        self.output_layer_config = output_layer_config
        self.activation = activation
        self.merge_type = merge_type
        self.debug_net_build = debug_net_build
        self.trainable_weights_penalty = trainable_weights_penalty
        self.params = None
        self.max_trainable_weights = max_trainable_weights
        tf.keras.backend.clear_session()

    def get_accuracy(self, params):
        """
        Build a model and train it
        :param params: Parameters from GA for this model
        :return:
        """
        # Generate network model
        self.params = params
        self.generation += 1  # TODO: Can we get generation ID from DEAP??
        net_generator = NetGenerator(
            params,
            self.generation,
            layer_types=self.layer_types,
            output_dir=self.output_dir,
            num_layers=self.num_layers,
            num_chromosomes=self.num_chromosomes,
            input_shape=self.input_shape,
            output_layer_config=self.output_layer_config,
            merge_type=self.merge_type,
            debug_net_build=self.debug_net_build
        )
        # Import generated model (note this could fail for many reasons!!)
        try:
            print(f'Building net for generation {self.generation}')
            reload(Net)
            model = Net.net()
            print('NN created, training...')
            print(model.summary())
            tf.keras.utils.plot_model(
                model,
                to_file=os.path.join(self.output_dir, f'gen_{self.generation}_model.png'),
                show_shapes=True, show_layer_names=True,
                rankdir='TB', expand_nested=False, dpi=96
            )
        except Exception as e:
            print(f'**** Error {e} when generating network ****')
            tf.keras.backend.clear_session()
            # del model
            return -1000, -1000, -1000, \
                   self.trainable_weights_penalty / 5e8, \
                   self.num_layers - net_generator.num_orphans

        # Determine number of trainable parameters in the model and fail if its over a threshold
        model_num_trainable_params = count_params(model.trainable_weights)
        print(f'train.py: trainable_weights={model_num_trainable_params}')
        if model_num_trainable_params > self.max_trainable_weights:
            print(f'**** Too many trainable weights {model_num_trainable_params} > {self.max_trainable_weights} ****')
            tf.keras.backend.clear_session()
            del model
            return -100, -100, -100, \
                   self.trainable_weights_penalty / model_num_trainable_params, \
                   self.num_layers - net_generator.num_orphans

        # Setup average metrics for training
        # TODO: Configure these programmatically based on output_features
        joints_losses, joints_vis_losses = setup_average_metrics(loss=True, names=['joints', 'joints_vis'])
        joints_acc, joints_vis_acc = setup_average_metrics(acc=True, names=['joints', 'joints_vis'])
        # Setup training loop
        batch_time = AverageMeter()
        data_time = AverageMeter()

        print(f'*** Starting gen {self.generation} training: Running {self.num_epochs} Epochs ***')
        print(f'Generation {self.generation} Start time: {datetime.now()}')

        # Now train the model
        try:
            for epoch in range(self.num_epochs):  # config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH, 1):
                end = time.time()
                e_joints_losses, e_joints_vis_losses = setup_average_metrics(loss=True, names=['joints', 'joints_vis'])
                e_joints_acc, e_joints_vis_acc = setup_average_metrics(acc=True, names=['joints', 'joints_vis'])

                for i, (X, joints, joints_vis, meta) in enumerate(self.dataset):
                    # print(f'_get_item_(): images={X.shape}, joints={joints}, joints_vis={joints_vis}')

                    train_step_metrics = model.train_on_batch(
                        X, [joints, joints_vis],
                        reset_metrics=True, return_dict=True
                    )

                    # measure epoch accuracy and record loss
                    e_joints_losses.update(train_step_metrics['Dense_joints_output_loss'], X.shape[0])
                    e_joints_vis_losses.update(train_step_metrics['Dense_joints_vis_output_loss'], X.shape[0])
                    e_joints_acc.update(train_step_metrics['Dense_joints_output_accuracy'], X.shape[0])
                    e_joints_vis_acc.update(train_step_metrics['Dense_joints_vis_output_accuracy'], X.shape[0])

                    # Predict on training batch to get a view of how accurate it is
                    output = model.predict_on_batch(X)

                    batch_time.update(time.time() - end)
                    end = time.time()

                # TODO: Add parameter for iterim image saving
                # if epoch % 10 == 0:
                #     prefix = os.path.join(self.output_dir, f'gen_{self.generation}_epoch_{epoch}')
                #     _ = save_debug_images(self.config, X, meta, target, output, prefix)

                # Update losses and acc
                joints_losses.update(e_joints_losses.avg)
                joints_vis_losses.update(e_joints_vis_losses.avg)
                joints_acc.update(e_joints_acc.avg)
                joints_vis_acc.update(e_joints_vis_acc.avg)

                # End of epoch statistics
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                      'Speed {speed:.1f} samples/s\t'.format(epoch, self.num_epochs,
                                                             batch_time=batch_time,
                                                             speed=X.shape[0] / batch_time.val,
                                                             data_time=data_time))
                print('Losses (acc): joints {jnts_loss:.5f} ({jnts_acc:.5f}), '
                      'vis {vis_loss:.5f} ({vis_acc:.5f})'.format(jnts_loss=joints_losses.val,
                                                                  jnts_acc=joints_acc.val,
                                                                  vis_loss=joints_vis_losses.val,
                                                                  vis_acc=joints_vis_acc.val))
                print('------------------------------------------------------------------------------')

                if joints_losses.early_stop() + joints_vis_losses.early_stop() + \
                        joints_acc.early_stop() + joints_vis_acc.early_stop() > 3 and \
                        epoch >= self.early_stop_epoch:
                    print(f'Early stopping at epoch {epoch}')
                    break

                # model_save_file = os.path.join(self.output_dir, f'gen_{self.generation}_epoch_{epoch}.h5')
                # self.model.save(model_save_file)

            print(f'Generation {self.generation} End time: {datetime.now()}')
            # Save after training has completed
            prefix = os.path.join(self.output_dir, f'gen_{self.generation}_epoch_{epoch}_final')
            _ = save_debug_images(self.config, X, meta, joints, output, prefix)

            # Save final model
            # TODO: Add parameter for this.
            # final_model_state_file = os.path.join(self.output_dir, f'gen_{self.generation}_final_state.h5')
            # print('saving final model state to {}'.format(final_model_state_file))
            # model.save(final_model_state_file)

        except Exception as e:
            print(f'**** Error {e} when training network ****')
            tf.keras.backend.clear_session()
            del model
            return -100, -100, -100, \
                   model_num_trainable_params / self.trainable_weights_penalty, \
                   -100

        # Clear and delete the current NN to avoid memory leaks!
        tf.keras.backend.clear_session()
        del model

        # Model score weighted according to number of trainable params
        # More params reduces score
        score = mean([joints_acc.avg, joints_vis_acc.avg])

        return joints_acc.avg, joints_vis_acc.avg, \
               self.trainable_weights_penalty / model_num_trainable_params, \
               self.num_layers - (net_generator.num_orphans * 2)  # Penalise for orphans!

    def format_params(self, params):
        """
        Generate the model summary, typically called when all generations have been tested and we want the best output
        :param params:
        :return:
        """
        self.params = params
        self.generation = 'best'
        net_generator = NetGenerator(
            params,
            self.generation,
            output_dir=self.output_dir,
            num_layers=self.num_layers, num_chromosomes=self.num_chromosomes,
            input_shape=self.input_shape,
            output_layer_config=self.output_layer_config,
            debug_net_build=self.debug_net_build
        )
        # Import generated model (note this could fail for many reasons!!)
        print(f'Building best network')
        reload(Net)
        model = Net.net()
        print('NN created, training...')
        print(model.summary())
        tf.keras.utils.plot_model(
            model,
            to_file=os.path.join(self.output_dir, f'gen_{self.generation}_model.png'),
            show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96
        )
        return f'model_params={np.reshape(np.array(params), (self.num_layers, self.num_chromosomes)).astype(np.int32)}'
