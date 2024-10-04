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
from nerva.loss_functions import SoftmaxCrossEntropyLoss, LossFunction, SquaredErrorLoss, NegativeLogLikelihoodLoss, \
    CrossEntropyLoss, LogisticCrossEntropyLoss
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


class TestLoss(unittest.TestCase):
        # check_equal_tensors("Y", Y, "Y2", Y2)
        # check_equal_tensors("DY", DY, "DY2", DY2)

    def _test_loss(self, name: str, loss: LossFunction, expected_loss: float, Y: torch.tensor, T: torch.tensor):
        print(f"\n=== test_loss {name} ===\n")
        L = loss.value(Y, T)
        self.assertAlmostEqual(expected_loss, L, 5)

    def test_loss(self):
        Y = torch.tensor([
            [0.36742274, 0.35949028, 0.27308698],
            [0.30354068, 0.41444678, 0.28201254],
            [0.34972793, 0.32481684, 0.32545523],
            [0.34815459, 0.44543710, 0.20640831],
            [0.19429503, 0.32073754, 0.48496742],
        ], dtype = torch.float32)

        T = torch.tensor([
            [0.00000000, 1.00000000, 0.00000000],
            [1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.00000000],
            [0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000],
        ], dtype = torch.float32)

        self._test_loss("SquaredErrorLoss", SquaredErrorLoss(), 3.2447052001953125, Y, T)
        self._test_loss("SoftmaxCrossEntropyLoss", SoftmaxCrossEntropyLoss(), 5.419629096984863, Y, T)
        self._test_loss("NegativeLogLikelihoodLoss", NegativeLogLikelihoodLoss(), 5.283669471740723, Y, T)
        self._test_loss("CrossEntropyLoss", CrossEntropyLoss(), 5.283669471740723, Y, T)
        self._test_loss("LogisticCrossEntropyLoss", LogisticCrossEntropyLoss(), 2.666532516479492, Y, T)

