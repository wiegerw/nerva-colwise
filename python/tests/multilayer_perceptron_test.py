# Copyright: Wieger Wesselink 2022 - 2024
#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#
# \file multilayer_perceptron_test.py
# \brief Tests for multilayer perceptrons.

import unittest
from typing import List

from nerva.activation_functions import ReLU, NoActivation
from nerva.layers import Dense, make_layers
from nerva.loss_functions import SoftmaxCrossEntropyLoss
from nerva.multilayer_perceptron import MultilayerPerceptron
from nerva.optimizers import GradientDescent
from nerva.weights import Xavier

import torch

def check_equal_tensors(name1, X1, name2, X2, epsilon=1e-5):
    error = torch.norm(X2 - X1).pow(2).item()
    if error > epsilon:
        print(f'Tensors {name1} and {name2} are not equal.')
        print(f"{name1}: \n{X1}")
        print(f"{name2}: \n{X2}")
        print(f'error: {error}')
        assert error <= epsilon, f"Error {error} exceeds tolerance {epsilon}."
    else:
        print(f'Tensors {name1} and {name2} are equal.')


# tag::construct1[]
def construct_mlp1(sizes: List[int], batch_size: int):

    layer1 = Dense(input_size=sizes[0],
                   output_size=sizes[1],
                   activation=ReLU(),
                   optimizer=GradientDescent(),
                   weight_initializer=Xavier())

    layer2 = Dense(input_size=sizes[1],
                   output_size=sizes[2],
                   activation=ReLU(),
                   optimizer=GradientDescent(),
                   weight_initializer=Xavier())

    layer3 = Dense(input_size=sizes[2],
                   output_size=sizes[3],
                   activation=NoActivation(),
                   optimizer=GradientDescent(),
                   weight_initializer=Xavier())

    M = MultilayerPerceptron()
    M.layers = [layer1, layer2, layer3]
    M.compile(batch_size)  # Initialize the C++ data structures

    return M
# end::construct1[]


# tag::construct2[]
def construct_mlp2(linear_layer_sizes: List[int], batch_size: int):

    layer_specifications = ["ReLU", "ReLU", "Linear"]
    linear_layer_densities = [1.0, 1.0, 1.0]
    linear_layer_dropouts = [0.0, 0.0, 0.0]
    linear_layer_weights = ["Xavier", "Xavier", "Xavier"]
    layer_optimizers = ["GradientDescent", "GradientDescent", "GradientDescent"]
    layers = make_layers(layer_specifications,
                         linear_layer_sizes,
                         linear_layer_densities,
                         linear_layer_dropouts,
                         linear_layer_weights,
                         layer_optimizers)
    M = MultilayerPerceptron()
    M.layers = layers
    M.compile(batch_size)  # Initialize the C++ data structures

    return M
# end::construct2[]


# Example usage in a unittest TestCase
class TestMLPExecution(unittest.TestCase):
    def _test_mlp(self,
                  X: torch.tensor,
                  T: torch.tensor,
                  W1: torch.tensor,
                  b1: torch.tensor,
                  W2: torch.tensor,
                  b2: torch.tensor,
                  W3: torch.tensor,
                  b3: torch.tensor,
                  Y1: torch.tensor,
                  DY1: torch.tensor,
                  Y2: torch.tensor,
                  DY2: torch.tensor,
                  lr: float,
                  sizes: List[int],
                  batch_size: int,
                  construct1=False
                 ):

        M = construct_mlp1(sizes, batch_size) if construct1 else construct_mlp2(sizes, batch_size)

        # Set weights + bias manually
        # tag::layer-access[]
        M.layers[0]._layer.W = W1
        M.layers[0]._layer.b = b1
        M.layers[1]._layer.W = W2
        M.layers[1]._layer.b = b2
        M.layers[2]._layer.W = W3
        M.layers[2]._layer.b = b3
        # end::layer-access[]

        loss = SoftmaxCrossEntropyLoss()
        Y = M.feedforward(X)
        DY = loss.gradient(Y, T) / batch_size   # take the average of the gradients in the batch
    
        check_equal_tensors("Y", Y, "Y1", Y1)
        check_equal_tensors("DY", DY, "DY1", DY1)
    
        M.backpropagate(Y, DY)
        M.optimize(lr)
        Y = M.feedforward(X)
        M.backpropagate(Y, DY)
    
        check_equal_tensors("Y", Y, "Y2", Y2)
        check_equal_tensors("DY", DY, "DY2", DY2)


    def test_mlp1(self):
        X = torch.tensor([
            [0.37454012, 0.95071429],
            [0.73199391, 0.59865850],
            [0.15601864, 0.15599452],
            [0.05808361, 0.86617613],
            [0.60111499, 0.70807260]
        ], dtype=torch.float32)

        T = torch.tensor([
            [0.00000000, 1.00000000, 0.00000000],
            [1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000]
        ], dtype=torch.float32)

        W1 = torch.tensor([
            [0.60928798, 0.12283446],
            [-0.40647557, 0.16275182],
            [-0.51447225, 0.31192824],
            [-0.52783161, -0.25182083],
            [0.17900684, 0.27646673],
            [0.04572495, 0.12897399]
        ], dtype=torch.float32)

        b1 = torch.tensor([[0.38675582, 0.42659640, 0.37051007, -0.11596697, 0.48523235, 0.04033934]], dtype=torch.float32)

        W2 = torch.tensor([
            [0.38505706, -0.13348164, 0.31668609, -0.32666627, -0.27552345, -0.12539147],
            [0.28924564, 0.27090782, 0.15212868, 0.15059654, 0.08706193, 0.06441139],
            [-0.33077568, 0.11333510, -0.19510441, 0.09439045, 0.31387877, 0.26053587],
            [-0.38992766, -0.38738862, -0.18559195, 0.10539542, 0.01223989, -0.13228671]
        ], dtype=torch.float32)

        b2 = torch.tensor([[0.28535178, 0.29878062, 0.18463619, 0.25079638]], dtype=torch.float32)

        W3 = torch.tensor([
            [0.00587273, -0.32241952, 0.30951124, 0.33368146],
            [-0.30861658, 0.04479754, -0.10331273, -0.24567801],
            [-0.02842641, 0.25743872, -0.01068246, 0.43675703]
        ], dtype=torch.float32)

        b3 = torch.tensor([[-0.38727069, 0.00397784, 0.22307467]], dtype=torch.float32)

        Y1 = torch.tensor([
            [-0.57443899, -0.10925630, 0.41034985],
            [-0.56981552, -0.11173054, 0.39652818],
            [-0.53720498, -0.10108448, 0.37866449],
            [-0.56281579, -0.10445327, 0.40684068],
            [-0.57073528, -0.11069612, 0.40081441]
        ], dtype=torch.float32)

        DY1 = torch.tensor([
            [0.03795389, -0.13956583, 0.10161193],
            [-0.16160758, 0.06070010, 0.10090748],
            [0.03963816, -0.13869186, 0.09905367],
            [0.03832504, -0.13938963, 0.10106459],
            [0.03826893, -0.13937680, 0.10110789]
        ], dtype=torch.float32)

        Y2 = torch.tensor([
            [-0.57339072, -0.09924223, 0.40067863],
            [-0.56873512, -0.10201052, 0.38706493],
            [-0.53640091, -0.09185813, 0.36971307],
            [-0.56188405, -0.09455847, 0.39731762],
            [-0.56967103, -0.10088711, 0.39129135]
        ], dtype=torch.float32)

        DY2 = torch.tensor([
            [0.03805648, -0.13885672, 0.10080025],
            [-0.16150525, 0.06138998, 0.10011526],
            [0.03972618, -0.13803604, 0.09830986],
            [0.03842213, -0.13868901, 0.10026687],
            [0.03837107, -0.13868113, 0.10031006]
        ], dtype=torch.float32)

        lr = 0.01
        sizes = [2, 6, 4, 3]
        batch_size = 5

        self._test_mlp(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, batch_size, True)
        self._test_mlp(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, batch_size, False)
