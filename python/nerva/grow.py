#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from nerva.layers import Sparse
from nerva.weights import WeightInitializer


class GrowFunction(object):
    def __call__(self, layer: Sparse, count: int):
        raise NotImplementedError


class GrowRandom(GrowFunction):
    def __init__(self, init: WeightInitializer):
        self.init = init

    def __call__(self, layer: Sparse, count: int):
        layer.grow_random(count, self.init)


def parse_grow_function(strategy: str, init: WeightInitializer):
    if strategy == 'Random':
        return GrowRandom(init)
    else:
        raise RuntimeError(f"unknown grow strategy '{strategy}'")
