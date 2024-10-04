# Copyright 2022 - 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List, Union, Optional

import nervalibrowwise

from nerva.activation_functions import Activation, NoActivation, parse_activation
from nerva.optimizers import Optimizer, GradientDescent, parse_optimizer
from nerva.weights import WeightInitializer, Xavier, parse_weight_initializer


class Layer(object):
    pass


def print_activation(activation: Activation) -> str:
    if isinstance(activation, NoActivation):
        return 'Linear'
    return str(activation).replace('()', '')


class Dense(Layer):
    # tag::dense_constructor[]
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Activation=NoActivation(),
                 optimizer: Optimizer=GradientDescent(),
                 weight_initializer: WeightInitializer=Xavier(),
                 dropout_rate: float=0
                ):
     # end::dense_constructor[]
        """
        A dense layer.

        :param input_size: the number of inputs of the layer
        :param output_size: the number of outputs of the layer
        :param activation: the activation function
        :param optimizer: the optimizer
        :param weight_initializer: the weight initializer
        :param dropout_rate: the dropout rate
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self.dropout_rate = dropout_rate
        self._layer = None

    def __str__(self):
        return f'Dense(output_size={self.output_size}, activation={self.activation}, optimizer={self.optimizer}, weight_initializer={self.weight_initializer}, dropout={self.dropout_rate})'

    def density_info(self) -> str:
        N = self._layer.W.size
        return f'{N}/{N} (100%)'

    def set_weights_and_bias(self, init: WeightInitializer) -> None:
        self._layer.set_weights_and_bias(str(init))

    # tag::dense_compile[]
    def compile(self, batch_size: int):
        """
        Creates a C++ object for the layer.

        :param batch_size: the batch size
        :return:
        """
        activation = print_activation(self.activation)
        if self.dropout_rate == 0.0:
            layer = nervalibrowwise.make_dense_linear_layer(self.input_size, self.output_size, batch_size, activation, str(self.weight_initializer), str(self.optimizer))
        else:
            layer = nervalibrowwise.make_dense_linear_dropout_layer(self.input_size, self.output_size, batch_size, self.dropout_rate, activation, str(self.weight_initializer), str(self.optimizer))
        self._layer = layer
        return layer
    # end::dense_compile[]

class Sparse(Layer):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 density: float,
                 activation: Activation=NoActivation(),
                 optimizer: Optimizer=GradientDescent(),
                 weight_initializer: WeightInitializer=Xavier()):
        """
        A sparse layer.

        :param input_size: the number of inputs of the layer
        :param output_size: the number of outputs of the layer
        :param density: a hint for the maximum density of the layer. This is a number between 0.0 (fully sparse) and
         1.0 (fully dense). Memory will be reserved to store a matrix with the given density.
        :param activation: the activation function
        :param optimizer: the optimizer
        """
        self.input_size = input_size
        self.output_size = output_size
        self.density = density
        self.activation = activation
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self._layer = None

    def __str__(self):
        return f'Sparse(output_size={self.output_size}, density={self.density}, activation={self.activation}, optimizer={self.optimizer}, weight_initializer={self.weight_initializer})'

    def density_info(self) -> str:
        n, N = self._layer.W.nonzero_count()
        return f'{n}/{N} ({100 * n / N:.3f}%)'

    def compile(self, batch_size: int, dropout_rate: float=0.0):
        """
        Compiles the model into a C++ object

        :param batch_size: the batch size
        :param dropout_rate: the dropout rate
        :return:
        """
        activation = print_activation(self.activation)
        layer = nervalibrowwise.make_sparse_linear_layer(self.input_size, self.output_size, batch_size, self.density, activation, str(self.weight_initializer), str(self.optimizer))
        self._layer = layer
        return layer

    def set_support_random(self, density: float) -> None:
        self._layer.set_support_random(density)

    def set_weights_and_bias(self, weight_initializer: WeightInitializer) -> None:
        self._layer.set_weights_and_bias(str(weight_initializer))

    def initialize_weights(self, weight_initializer: WeightInitializer) -> None:
        self._layer.initialize_weights(str(weight_initializer))

    def weight_count(self):
        return self._layer.weight_count()

    def positive_weight_count(self):
        return self._layer.positive_weight_count()

    def negative_weight_count(self):
        return self._layer.negative_weight_count()

    def prune_magnitude(self, zeta: float) -> int:
        return self._layer.prune_magnitude(zeta)

    def prune_SET(self, zeta: float) -> int:
        return self._layer.prune_SET(zeta)

    def prune_threshold(self, threshold: float) -> int:
        return self._layer.prune_threshold(threshold)

    def grow_random(self, count: int, weight_initializer=Xavier()) -> None:
        self._layer.grow_random(str(weight_initializer), count)


class BatchNormalization(Layer):
    # tag::batchnormalization_constructor[]
    def __init__(self,
                 input_size: int,
                 output_size: Optional[int] = None,
                 optimizer: Optimizer = GradientDescent()
                ):
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer
        if self.output_size is None:
            self.output_size = self.input_size
        assert self.output_size == self.input_size
    # end::batchnormalization_constructor[]

    def compile(self, batch_size: int):
        layer = nervalibrowwise.make_batch_normalization_layer(self.input_size, batch_size, str(self.optimizer))
        self._layer = layer
        return layer

    def __str__(self):
        return f'BatchNormalization(optimizer={self.optimizer})'


def make_linear_layer(input_size: int,
                      output_size: int,
                      density: float,
                      dropout_rate: float,
                      activation: Activation,
                      weight_initializer: WeightInitializer,
                      optimizer: Optimizer
                     ) -> Union[Dense, Sparse]:
    if density == 1.0:
        return Dense(input_size,
                     output_size,
                     activation=activation,
                     optimizer=optimizer,
                     weight_initializer=weight_initializer,
                     dropout_rate=dropout_rate)
    else:
        return Sparse(input_size,
                      output_size,
                      density,
                      activation=activation,
                      optimizer=optimizer,
                      weight_initializer=weight_initializer)


def make_layers(layer_specifications: list[str],
                linear_layer_sizes: list[int],
                linear_layer_densities: list[float],
                linear_layer_dropouts: list[float],
                linear_layer_weights: list[str],
                optimizers: list[str]
               ) -> List[Layer]:

    assert len(linear_layer_densities) == len(linear_layer_dropouts) == len(linear_layer_weights) == len(linear_layer_sizes) - 1
    assert len(optimizers) == len(layer_specifications)

    linear_layer_weights = [parse_weight_initializer(x) for x in linear_layer_weights]
    optimizers = [parse_optimizer(x) for x in optimizers]

    result = []

    linear_layer_index = 0
    optimizer_index = 0
    input_size = linear_layer_sizes[0]

    for spec in layer_specifications:
        if spec == "BatchNormalization":
            output_size = input_size
            optimizer = optimizers[optimizer_index]
            optimizer_index += 1
            blayer = BatchNormalization(input_size, output_size, optimizer)
            result.append(blayer)
        else:  # linear spec
            output_size = linear_layer_sizes[linear_layer_index + 1]
            density = linear_layer_densities[linear_layer_index]
            dropout_rate = linear_layer_dropouts[linear_layer_index]
            activation = parse_activation(spec)
            weights = linear_layer_weights[linear_layer_index]
            optimizer = optimizers[optimizer_index]
            linear_layer_index += 1
            optimizer_index += 1
            llayer = make_linear_layer(input_size, output_size, density, dropout_rate, activation, weights, optimizer)
            result.append(llayer)
        input_size = output_size

    return result
