#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from typing import List

from nerva.layers import Sparse
from nerva.multilayer_perceptron import MultilayerPerceptron


def parse_arguments(text: str, name: str, n: int) -> List[str]:
    pattern = name + r'\((.*?)\)'
    m = re.match(pattern, text)
    if not m:
        return []
    result = list(filter(None, m.group(1).split(',')))
    if len(result) != n:
        return []
    return result


class RegrowFunction(object):
    """
    Interface for regrowing the sparse layers of a neural network
    """
    def __call__(self, M: MultilayerPerceptron):
        raise NotImplementedError


class PruneFunction(object):
    """
    Interface for pruning the weights of a sparse layer
    """
    def __call__(self, layer: Sparse):
        raise NotImplementedError


class PruneMagnitude(PruneFunction):
    """
    Prunes a fraction zeta of the weights in the sparse layers. Weights
    are pruned according to their magnitude.
    """
    def __init__(self, zeta):
        self.zeta = zeta

    def __call__(self, layer: Sparse):
        return layer.prune_magnitude(self.zeta)


class PruneThreshold(PruneFunction):
    """
    Prunes weights with magnitude below a given threshold.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, layer: Sparse):
        return layer.prune_threshold(self.threshold)


class PruneSET(PruneFunction):
    """
    Prunes a fraction zeta of the positive and a fraction zeta of the negative
    weights in the sparse layers. Weights are pruned according to their magnitude.
    """
    def __init__(self, zeta):
        self.zeta = zeta

    def __call__(self, layer: Sparse):
        return layer.prune_SET(self.zeta)


def parse_prune_function(strategy: str):
    arguments = parse_arguments(strategy, 'Magnitude', 1)
    if arguments:
        return PruneMagnitude(float(arguments[0]))

    arguments = parse_arguments(strategy, 'Threshold', 1)
    if arguments:
        return PruneThreshold(float(arguments[0]))

    arguments = parse_arguments(strategy, 'SET', 1)
    if arguments:
        return PruneSET(float(arguments[0]))

    raise RuntimeError(f"unknown prune strategy '{strategy}'")
