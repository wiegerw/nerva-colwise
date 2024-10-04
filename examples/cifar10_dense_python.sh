#!/bin/bash

# tag::doc[]
python ../python/tools/mlp.py \
    --layers="ReLU;ReLU;Linear" \
    --layer-sizes="3072;1024;1024;10" \
    --layer-weights=Xavier \
    --optimizers="Nesterov(0.9)" \
    --loss=SoftmaxCrossEntropy \
    --learning-rate=0.01 \
    --epochs=100 \
    --batch-size=100 \
    --threads=12 \
    --cifar10=../data \
    --seed=123
# end::doc[]
