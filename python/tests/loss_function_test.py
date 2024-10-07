# Copyright: Wieger Wesselink 2022 - 2024
#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#
# \file multilayer_perceptron_test.py
# \brief Tests for multilayer perceptrons.

import unittest

import torch

from nervacolwise.loss_functions import SoftmaxCrossEntropyLoss, LossFunction, SquaredErrorLoss, NegativeLogLikelihoodLoss, \
    CrossEntropyLoss, LogisticCrossEntropyLoss


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
    def _test_loss(self, name: str, loss: LossFunction, expected_loss: float, Y: torch.tensor, T: torch.tensor):
        print(f"\n=== test_loss {name} ===\n")
        L = loss.value(Y, T)
        self.assertAlmostEqual(expected_loss, L, 5)


#--- begin generated code ---#
    def test_loss1(self):
        Y = torch.tensor([
            [0.23759169, 0.43770149, 0.20141643, 0.35686849, 0.48552814],
            [0.42272727, 0.28115265, 0.45190243, 0.17944701, 0.26116029],
            [0.33968104, 0.28114586, 0.34668113, 0.46368450, 0.25331157],
        ], dtype = torch.float32)

        T = torch.tensor([
            [1.00000000, 1.00000000, 0.00000000, 0.00000000, 1.00000000],
            [0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000],
        ], dtype = torch.float32)

        self._test_loss("SquaredErrorLoss", SquaredErrorLoss(), 2.6550281475767563, Y, T)
        self._test_loss("SoftmaxCrossEntropyLoss", SoftmaxCrossEntropyLoss(), 5.106889686512423, Y, T)
        self._test_loss("NegativeLogLikelihoodLoss", NegativeLogLikelihoodLoss(), 4.548777728936653, Y, T)
        self._test_loss("CrossEntropyLoss", CrossEntropyLoss(), 4.548777728936653, Y, T)
        self._test_loss("LogisticCrossEntropyLoss", LogisticCrossEntropyLoss(), 2.539463487358204, Y, T)


    def test_loss2(self):
        Y = torch.tensor([
            [0.24335898, 0.21134093, 0.24788846, 0.40312318, 0.43329234],
            [0.40191852, 0.53408849, 0.42021140, 0.24051313, 0.34433141],
            [0.35472250, 0.25457058, 0.33190014, 0.35636369, 0.22237625],
        ], dtype = torch.float32)

        T = torch.tensor([
            [1.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000],
            [0.00000000, 0.00000000, 1.00000000, 1.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000],
        ], dtype = torch.float32)

        self._test_loss("SquaredErrorLoss", SquaredErrorLoss(), 3.6087104890568256, Y, T)
        self._test_loss("SoftmaxCrossEntropyLoss", SoftmaxCrossEntropyLoss(), 5.5889911807479065, Y, T)
        self._test_loss("NegativeLogLikelihoodLoss", NegativeLogLikelihoodLoss(), 5.90971538007391, Y, T)
        self._test_loss("CrossEntropyLoss", CrossEntropyLoss(), 5.909715380073911, Y, T)
        self._test_loss("LogisticCrossEntropyLoss", LogisticCrossEntropyLoss(), 2.7376380548462254, Y, T)


    def test_loss3(self):
        Y = torch.tensor([
            [0.23774258, 0.29687977, 0.43420442, 0.28599538, 0.20014798],
            [0.42741216, 0.43115409, 0.22655227, 0.35224692, 0.43868708],
            [0.33484526, 0.27196615, 0.33924331, 0.36175770, 0.36116494],
        ], dtype = torch.float32)

        T = torch.tensor([
            [0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000],
            [1.00000000, 1.00000000, 1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ], dtype = torch.float32)

        self._test_loss("SquaredErrorLoss", SquaredErrorLoss(), 3.289394384977318, Y, T)
        self._test_loss("SoftmaxCrossEntropyLoss", SoftmaxCrossEntropyLoss(), 5.441938177932827, Y, T)
        self._test_loss("NegativeLogLikelihoodLoss", NegativeLogLikelihoodLoss(), 5.44627595910772, Y, T)
        self._test_loss("CrossEntropyLoss", CrossEntropyLoss(), 5.44627595910772, Y, T)
        self._test_loss("LogisticCrossEntropyLoss", LogisticCrossEntropyLoss(), 2.678127590042374, Y, T)


    def test_loss4(self):
        Y = torch.tensor([
            [0.26787616, 0.26073833, 0.31560020, 0.37231605, 0.49308039],
            [0.35447135, 0.45527664, 0.41003295, 0.17984538, 0.27786731],
            [0.37765249, 0.28398503, 0.27436685, 0.44783858, 0.22905230],
        ], dtype = torch.float32)

        T = torch.tensor([
            [0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000],
            [1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000, 1.00000000, 1.00000000],
        ], dtype = torch.float32)

        self._test_loss("SquaredErrorLoss", SquaredErrorLoss(), 3.521376994732803, Y, T)
        self._test_loss("SoftmaxCrossEntropyLoss", SoftmaxCrossEntropyLoss(), 5.548304798627446, Y, T)
        self._test_loss("NegativeLogLikelihoodLoss", NegativeLogLikelihoodLoss(), 5.726367921857207, Y, T)
        self._test_loss("CrossEntropyLoss", CrossEntropyLoss(), 5.726367921857208, Y, T)
        self._test_loss("LogisticCrossEntropyLoss", LogisticCrossEntropyLoss(), 2.7197402348335156, Y, T)


    def test_loss5(self):
        Y = torch.tensor([
            [0.29207765, 0.38987005, 0.24441444, 0.38397493, 0.29902507],
            [0.40236525, 0.36536339, 0.32191037, 0.35636403, 0.25018760],
            [0.30555710, 0.24476656, 0.43367519, 0.25966104, 0.45078733],
        ], dtype = torch.float32)

        T = torch.tensor([
            [0.00000000, 1.00000000, 1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000],
            [1.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ], dtype = torch.float32)

        self._test_loss("SquaredErrorLoss", SquaredErrorLoss(), 3.2404999669186503, Y, T)
        self._test_loss("SoftmaxCrossEntropyLoss", SoftmaxCrossEntropyLoss(), 5.4240756991825645, Y, T)
        self._test_loss("NegativeLogLikelihoodLoss", NegativeLogLikelihoodLoss(), 5.365012502539291, Y, T)
        self._test_loss("CrossEntropyLoss", CrossEntropyLoss(), 5.365012502539292, Y, T)
        self._test_loss("LogisticCrossEntropyLoss", LogisticCrossEntropyLoss(), 2.6711745146065176, Y, T)



#--- end generated code ---#
