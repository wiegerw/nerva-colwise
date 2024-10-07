#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from nervacolwise.grow import GrowFunction
from nervacolwise.layers import Sparse
from nervacolwise.multilayer_perceptron import MultilayerPerceptron
from nervacolwise.prune import PruneFunction


class RegrowFunction(object):
    """
    Interface for regrowing the sparse layers of a neural network
    """
    def __call__(self, M: MultilayerPerceptron):
        raise NotImplementedError


class PruneGrow(RegrowFunction):
    def __init__(self, prune: PruneFunction, grow: GrowFunction):
        self.prune = prune
        self.grow = grow

    def __call__(self, M: MultilayerPerceptron):
        for layer in M.layers:
            if isinstance(layer, Sparse):
                weight_count = layer.weight_count()
                count = self.prune(layer)
                print(f'regrowing {count}/{weight_count} weights')
                self.grow(layer, count)
