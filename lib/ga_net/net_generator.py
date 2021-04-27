#
# net_generator.py
#
# Generate a NN based on the individual spec from DEAP
#
# Written by Mark Strefford
# (c) 2021 Delirium Digital Ltd
#

# TODO: Add ReLU, etc?
# TODO: Is padding and concat better for max 2 layers? Assuming currently we're creating lots of large tensors??!

import os
import shutil
import numpy as np

# TODO: Make this programmatic as currently CWD is the notebooks folder. Will it always be the same?
path_to_ga_net = '../lib/ga_net'
source_net_class_file = os.path.join(path_to_ga_net, 'net_template.py')
net_class_file = os.path.join(path_to_ga_net, 'Net.py')


# def get_max_input_size(inbound_connection_shapes):
#     # Calculate max input shape across inbound layers
#     # Works with tensors of shape [batch_size, dim1, dim2, dim3]
#     max_input_shape = [0, -1, -1, -1]
#     for inbound_connection_shape in inbound_connection_shapes:
#         for k in range(1, 4, 1):
#             if max_input_shape[k] < inbound_connection_shape[k]:
#                 max_input_shape[k] = inbound_connection_shape[k]
#     return max_input_shape


def add_indent(indent=1):
    """
    Add indent to the generated file to avoid python errors
    :param indent: Number of tabs to indent by
    :return: String containing the specified number of tabs
    """
    return '\t' * indent


def get_keys_by_value(dict_of_elements, value_to_find):
    """
    Parse a dict() to get keys of elements with a specified value
    :param dict_of_elements:
    :param value_to_find:
    :return:
    """
    list_of_keys = list()
    list_of_items = dict_of_elements.items()
    for item in list_of_items:
        if item[1] == value_to_find:
            list_of_keys.append(item[0])
    return list_of_keys


class NetGenerator:
    """
    Class to generate a neural network based on the generated config from DEAP
    The model is written to Net.py so that it can be loaded from Trainer
    Note we do this as it's much harder to generate a model in memory using dynamic config (tensors do not
    store naturally in other Python objects and ref()/deref() didn't work well).
    """

    def __init__(self,
                 params, generation,
                 layer_types=None,
                 input_shape=(256, 256, 3), output_layer_config=None,
                 num_layers=-1, num_chromosomes=12,
                 activation='relu',
                 merge_type='Concatenate',
                 stddev=0.01,
                 output_dir=None,
                 merge_orphans=False,
                 debug_net_build=False,
                 config=None):
        """
        Generate a DNN based on the provided parameters
        :param params: Parameters from GA algorithm
        :param generation: Generation number
        :param layer_types: Layer types to use (must align with add_layer() below)
        :param input_shape: Network input image shape
        :param output_layer_config: Config of the output layers for the network
        :param num_layers: Num layers in the network
        :param num_output_layers: Number of output layers (TODO: This can be deduced!)
        :param num_chromosomes: How many chromosomes per layer
        :param activation: activation function (typically relu)
        :param merge_type: Use add or Concatenate (Concatenate uses more memory)
        :param merge_orphans: Merge orphans into final layers (True uses more memory, but provides a complete NN)
        :param stddev: stddev for layer weight initialisation
        :param output_dir: output directory for generated Net.py files
        :param debug_net_build: Debug (TODO: Add more functionality to this!)
        """
        self.num_layers = num_layers
        self.generation = generation
        self.num_output_layers = len(output_layer_config)
        self.num_chromosomes = num_chromosomes
        self.model = None
        self.layer_types = layer_types
        self.input_shape = input_shape
        self.activation = activation
        self.stddev = stddev
        self._params = np.reshape(
            np.array(params), (self.num_layers, self.num_chromosomes)
        ).astype(np.int32)
        self.output_layer_config = output_layer_config
        self.merge_type = merge_type  # TODO: Add to config
        self.merge_orphans = merge_orphans
        self.num_orphans = 0
        self.orphans_layer = []
        self.indent_level = 1
        self.output_dir = output_dir
        self.debug_net_build = debug_net_build
        self.layer_label = 1
        self.layer_used_count = dict()

        print()
        print('*' * 40)
        print(f'Generation {self.generation}: model class={net_class_file}')
        print(f'params={self._params}')

        # Copy template to new file and open for append
        shutil.copyfile(source_net_class_file, net_class_file)
        with open(net_class_file, "a") as self.net_file_object:
            self.add_input_layer()
            self.add_params_layers()
            self.process_orphans()
            self.add_output_layers()
            self.add_model_compile()

            # Write params as a comment for traceability
            for param in self._params:
                self.net_file_object.write(
                    f'{add_indent(self.indent_level)}# {param}\n'
                )

            self.net_file_object.close()

        # Make a copy for later use if needed!
        net_generation_file = os.path.join(self.output_dir, f'Net_{self.generation}.py')
        shutil.copyfile(net_class_file, net_generation_file)

    def add_input_layer(self):
        # Build model from params
        self.add_layer(
            0,
            'Input',
            None,
            layer_params={
                'image_size': self.input_shape
            }
        )

    def add_params_layers(self):
        """
        Process generated params and create each layer of the network
        :return:
        """
        for i in range(self.num_layers):
            sub_layer_id = 0  # For layers in params that need padding and concat added
            # print(f'*** Params Layer {self.params_layer_id}, net Layer {self.net_layer_id}')
            self.net_file_object.write(
                f'{add_indent(self.indent_level)}'
                f'# *** Params Layer {i}, {self._params[i]}\n'
            )

            # Parse params for this layer
            layer_type, filters, kernel, stride, inbound_connections = self.get_layer_params(i)

            self.net_file_object.write(
                f'{add_indent(self.indent_level)}'
                f'# *** inbound_connections={inbound_connections}\n')

            if len(inbound_connections) > 1:
                # Add padding and concat if multiple inbound connections so we have a single inbound
                # for this layer
                inbound_connection = self.add_padding_and_merge_layers(i, inbound_connections)
            else:
                # Use the single inbound connection
                inbound_connection = int(inbound_connections[0])

            self.add_layer(
                self.layer_label,
                layer_type,
                inbound_connection,
                layer_params={
                    'filters': filters,
                    'kernel': kernel,
                    'stride': stride,
                    'activation': self.activation,
                    'stddev': self.stddev
                }
            )
            self.layer_label += 1

    def get_layer_params(self, layer_id):
        """
        Get the parameters for this layer
        :param layer_id: layer id in params
        :return: layer type, number of filters, kernel and stride size, list of processed inbound connections
        """
        layer_params = self._params[layer_id]
        layer_type = self.layer_types[layer_params[0]]
        filters = layer_params[1]
        kernel = layer_params[2]
        stride = layer_params[3]
        inbound_connections = self.process_inbound_connection(layer_id, layer_params[4:])

        return layer_type, filters, kernel, stride, inbound_connections

    def process_inbound_connection(self, layer_id, inbound_connections):
        """
        Generate a list of inbound connections for this layer.
        De-dupe and remove any < 0
        If no valid inbound connections, then set inbound to be input layer (layer 0)
        :param layer_id: layer in params
        :param inbound_connections: list of processed inbound connections
        :return:
        """
        inbound_connections = list(set(inbound_connections))  # De-dupe!
        processed_inbound_connections = []

        # Confirm inbound connections are in range of input (-1) through to current layer - 1
        for i in inbound_connections:
            if 0 <= i < layer_id:
                processed_inbound_connections.append(int(i))
                self.update_layer_used_count(i)

        # If not, then add input to lit
        if not processed_inbound_connections:
            processed_inbound_connections = [0]
            self.update_layer_used_count(0)

        return processed_inbound_connections

    def add_padding_and_merge_layers(self, layer_id, inbound_connections, orphan=None):
        """
        Take a list of inbound connections and perform the following actions:

            1) Find the maximum shape for all the inbound connections
            2) Add padding layers for each inbound connection to ensure the same shape
            3) Add a concat layer to bring all the inbound connectioons together

        Input:
            inbound_connections: List of inbound layers (indexed into params)

        Returns:
            The last layer
        """
        padding_layer_count = 0

        # print(inbound_connections)
        # inbound_layer_names = [f'x{i}' for i in inbound_connections]
        inbound_layer_names = [f'x{i}' if not isinstance(i, str) else i for i in inbound_connections]
        self.net_file_object.write(
            f'{add_indent(indent=self.indent_level)}'
            f'max_input_shape_{layer_id} = calculate_max_input_size([{", ".join(inbound_layer_names)}])\n'
        )

        # TODO: Swap PaddingLayer with Zeropadding2D layer if using concat?
        # Note add needs all 3 dimensions to be the same
        for i in inbound_connections:
            self.add_layer(
                f'x{self.layer_label}_{padding_layer_count}',
                'PaddingLayer',
                i,
                layer_params={'padding_shape': f'max_input_shape_{layer_id}',
                              'merge_type': self.merge_type
                              }
            )
            padding_layer_count += 1

        merge_inputs = [f'x{self.layer_label}_{i}' if not isinstance(i, str) else i for i in range(padding_layer_count)]
        self.add_layer(
            f'x{self.layer_label}_{padding_layer_count}',
            self.merge_type,
            '[' + ', '.join(merge_inputs) + ']'
        )

        return f'x{self.layer_label}_{padding_layer_count}'

    def update_layer_used_count(self, layer_id):
        """
        Keep a track of layers that feed forward to that we can identify orphans later
        :param layer_id: layer id to be updated
        :return:
        """
        if layer_id in self.layer_used_count.keys():
            self.layer_used_count[layer_id] += 1
        else:
            self.layer_used_count[layer_id] = 1

    def process_orphans(self):
        """
        Count number of layers that don't feed forward
        :return:
        """
        # Find layers with no outputs
        # print(f'Number of outputs per layer: {self.layer_used_count}')
        orphan_net_layers = get_keys_by_value(self.layer_used_count, 0)

        orphan_params_layers = [x for x in np.array(orphan_net_layers).flatten().tolist()
                                if x <= self.num_layers - self.num_output_layers]

        print(f'params_layers with no output used: {orphan_params_layers}')
        self.net_file_object.write(
            f'{add_indent(self.indent_level)}'
            f'# *** orphans={orphan_params_layers}\n')
        # orphan_inbound_connection = self.process_inbound_connections(orphan_params_layers)
        # print(f'Orphan inbound connection = {orphan_inbound_connection}')
        self.num_orphans = len(orphan_params_layers)
        if self.merge_orphans and self.num_orphans > 0:  # > 1 ?
            self.orphans_layer = self.add_padding_and_merge_layers('merge_orphans', orphan_params_layers)
        # elif self.num_orphans == 1:
        #     self.orphans_layer = orphan_params_layers[0]
        else:
            self.orphans_layer = []

    def add_output_layers(self):
        """
        Add the output layers to the model based on the provided config
        :return:
        """
        output_layer_inbound_connection = self.num_layers - self.num_output_layers + 1

        for output_layer in self.output_layer_config:
            # print(f'Adding layer with features {output_layer}')

            self.net_file_object.write(
                f'{add_indent(self.indent_level)}'
                f'# *** Output Layer: {output_layer}\n'
            )

            # Merge in orphan layers if there are any
            if self.merge_orphans and self.num_orphans:
                self.layer_label += 1   # Avoid duplicate layer names
                x = self.add_padding_and_merge_layers(
                    f'orphan_merge_{output_layer["name"]}',
                    [self.orphans_layer, output_layer_inbound_connection]
                )
            else:
                x = f'x{output_layer_inbound_connection}'

            self.add_layer(
                f'{output_layer["name"]}_flatten',
                'Flatten',
                x
            )
            self.add_layer(
                # self.net_layer_id,
                output_layer['name'],
                'Dense',
                f'{output_layer["name"]}_flatten',
                layer_params={
                    'units': output_layer['features'],
                    'activation': output_layer['activation'],
                    'stddev': self.stddev
                }
            )
            output_layer_inbound_connection += 1

    def add_layer(self, layer_id, layer_type, input_layer, layer_params=None, name=None):
        """
        Add processing layer to the model. Also handles different layer names and layer labels (x0, etc.)
        :param layer_id: Layer Id in the model
        :param layer_type: Conv2d, etc.
        :param input_layer: Layer feeding forward into this layer
        :param layer_params: Parameters (varies according to the layer type)
        :param name: Layer name
        :return:
        """
        layer_name = f'{layer_type}_{layer_id}' if name is None else name

        if layer_type == 'Input':
            layer = f'Input(shape={layer_params["image_size"]}, dtype="float32")'
        elif layer_type == 'Conv2D':
            layer = f'Conv2D({2 ** layer_params["filters"]}, ' \
                    f'({layer_params["kernel"]}, {layer_params["kernel"]}), ' \
                    f'({layer_params["stride"]}, {layer_params["stride"]}), ' \
                    f'activation=\'{layer_params["activation"]}\', ' \
                    f'kernel_initializer=initializers.RandomNormal(stddev={layer_params["stddev"]}), ' \
                    f'bias_initializer=initializers.Zeros(), ' \
                    f'name=\'{layer_name}\')'
        elif layer_type == 'DepthwiseConv2D':
            layer = f'DepthwiseConv2D(({layer_params["kernel"]}, {layer_params["kernel"]}), ' \
                    f'({layer_params["stride"]}, {layer_params["stride"]}), ' \
                    f'activation=\'{layer_params["activation"]}\', ' \
                    f'name=\'{layer_name}\')'
        elif layer_type == 'Conv2DTranspose':
            layer = f'Conv2DTranspose({2 ** layer_params["filters"]}, ' \
                    f'({layer_params["kernel"]}, {layer_params["kernel"]}), ' \
                    f'({layer_params["stride"]}, {layer_params["stride"]}), ' \
                    f'activation=\'{layer_params["activation"]}\', ' \
                    f'kernel_initializer=initializers.RandomNormal(stddev={layer_params["stddev"]}), ' \
                    f'bias_initializer=initializers.Zeros(), ' \
                    f'name=\'{layer_name}\')'
        elif layer_type == 'PaddingLayer':
            layer = f'PaddingLayer(padding_shape={layer_params["padding_shape"]}, ' \
                    f'merge_type=\'{layer_params["merge_type"]}\', ' \
                    f'name=\'{layer_name}\')'
        elif layer_type == 'Dense':
            layer = f'Dense({layer_params["units"]}, ' \
                    f'activation=\'{layer_params["activation"]}\', ' \
                    f'kernel_initializer=initializers.RandomNormal(stddev={layer_params["stddev"]}), ' \
                    f'bias_initializer=initializers.Zeros(), ' \
                    f'name=\'{layer_name}\')'
        elif layer_type == 'Concatenate' or layer_type == 'Flatten':
            layer = f'{layer_type}(name=\'{layer_name}\')'
        elif layer_type == 'add':
            layer = f'{layer_type}({input_layer}, name=\'{layer_name}\')'
        else:
            layer = f'NOPLayer()'

        # Handle output layers that have defined names
        # TODO: This functionality is also in other parts of the code. Tidy?
        if isinstance(layer_id, int):
            layer_prefix = 'x'
        else:
            layer_prefix = ''

        # Handle input layers
        if isinstance(input_layer, int):
            input_layer = f'x{input_layer}'

        output_line = f'{add_indent(indent=self.indent_level)}' \
                      f'{layer_prefix}{layer_id} = {layer}'
        if layer_type == 'Input' or layer_type == 'add':
            output_line += '\n'
        else:
            output_line += f'({input_layer})\n'

        self.net_file_object.write(output_line)

        # Track how often a processing layer is used to identify orphans
        if layer_type in self.layer_types:
            self.layer_used_count[layer_id] = 0

        if self.debug_net_build:
            debug_line = f'{add_indent(indent=self.indent_level)}print({layer_prefix}{layer_id})\n'
            self.net_file_object.write(debug_line)

    def add_model_compile(self):
        """
        Adds model compile, losses, metrics etc.
        :return:
        """
        # Add model compile, etc.
        output_layer_list = ''
        for i in [output_layer['name'] for output_layer in self.output_layer_config]:
            output_layer_list += f'{i}, '
        self.net_file_object.write(
            f'{add_indent(indent=self.indent_level)}'
            f'model = Model(inputs=x0, '
            f'outputs=[{output_layer_list}], '
            f'name=\'model_{self.generation}\')\n'
        )
        self.net_file_object.write(
            f'{add_indent(indent=self.indent_level)}'
            f'optimizer = Adam(learning_rate=0.001)\n'
        )
        loss_list = ''
        for i in [output_layer['loss'] for output_layer in self.output_layer_config]:
            loss_list += f'{i}, '
        self.net_file_object.write(
            f'{add_indent(indent=self.indent_level)}'
            f'model.compile(loss=[{loss_list}], '
            f'metrics=["accuracy"], '
            f'optimizer=optimizer)\n'
        )
        self.net_file_object.write(f'{add_indent(indent=self.indent_level)}'
                                   f'return model\n')

    def get_params(self):
        """
        Get the parameters for this model
        :return:
        """
        return self._params
