# Copyright 2022 - 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Optional, List, Union, Tuple

import nervalibrowwise

from nerva.layers import Layer, Sparse
from nerva.weights import WeightInitializer


class MultilayerPerceptron(object):
    def __init__(self, layers: Optional[Union[List[Layer], Tuple[Layer]]]=None):
        self.layers = []
        self._model = None
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Layer):
        self.layers.append(layer)

    def _check_layers(self):
        """
        Checks if the architecture of the layers is OK
        """
        layers = self.layers

        # At least one layer
        if not layers:
            raise RuntimeError('No layers are defined')

    def compile(self, batch_size: int) -> None:
        self._check_layers()

        M = nervalibrowwise.MLP()

        # add layers
        for i, layer in enumerate(self.layers):
            cpp_layer = layer.compile(batch_size)
            M.append_layer(cpp_layer)
        self._model = M

    def feedforward(self, X):
        return self._model.feedforward(X)

    def backpropagate(self, Y, dY):
        return self._model.backpropagate(Y, dY)

    def optimize(self, eta):
        self._model.optimize(eta)

    def renew_dropout_masks(self):
        nervalibrowwise.renew_dropout_masks(self._model)

    def __str__(self):
        layers = ',\n  '.join([str(layer) for layer in self.layers])
        return f'MultilayerPerceptron(\n  {layers}\n)'

    def set_support_random(self):
        for layer in self.layers:
            if isinstance(layer, Sparse):
                layer.set_support_random(layer.density)

    def set_weights_and_bias(self, weight_initializers: List[WeightInitializer]):
        print(f'Initializing weights using {", ".join(str(w) for w in weight_initializers)}')
        self._model.set_weights_and_bias([str(w) for w in weight_initializers])

    def load_weights_and_bias(self, filename: str):
        """
        Loads the weights and biases from a file in .npz format

        The weight matrices are stored using the keys W1, W2, ... and the bias vectors using the keys "b1, b2, ..."
        :param filename: the name of the file
        """
        print(f'Loading weights and bias from {filename}')
        self._model.load_weights_and_bias(filename)

    def save_weights_and_bias(self, filename: str):
        """
        Loads the weights and biases from a file in .npz format

        The weight matrices are stored using the keys W1, W2, ... and the bias vectors using the keys "b1, b2, ..."
        :param filename: the name of the file
        """
        print(f'Saving weights and bias to {filename}')
        self._model.save_weights_and_bias(filename)

    def info(self, msg):
        self._model.info(msg)


def compute_sparse_layer_densities(overall_density: float, layer_sizes: List[int], erk_power_scale: float=1) -> List[float]:
    return nervalibrowwise.compute_sparse_layer_densities(overall_density, layer_sizes, erk_power_scale)


def print_model_info(M: MultilayerPerceptron) -> None:
    """
    Prints detailed information about a multilayer perceptron
    :param M: a multilayer perceptron
    """
    nervalibrowwise.print_model_info(M._model)
