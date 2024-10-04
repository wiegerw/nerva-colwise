#!/bin/bash

# tag::doc[]
../install/bin/mlp \
    --layers="ReLU;ReLU;Linear" \
    --layer-sizes="784;1024;512;10" \
    --layer-weights=Xavier \
    --optimizers="Momentum(0.9);Momentum(0.9);Momentum(0.9)" \
    --loss=SoftmaxCrossEntropy \
    --learning-rate=0.01 \
    --epochs=100 \
    --batch-size=100 \
    --threads=12 \
    --overall-density=0.05 \
    --mnist=../data \
    --seed=123
# end::doc[]
