#!/bin/bash

for loss in SquaredError CrossEntropy LogisticCrossEntropy SoftmaxCrossEntropy NegativeLogLikelihood
do 
    echo "======================================="
    echo "       loss = $loss"
    echo "======================================="
    ../install/bin/mlp \
    --layers="ReLU;ReLU;Softmax" \
    --layer-sizes="3072;128;128;10" \
    --layer-weights="Xavier" \
    --densities="1" \
    --optimizers="GradientDescent" \
    --loss=$loss \
    --learning-rate=0.01 \
    --epochs=2 \
    --batch-size=100 \
    --cifar10=../data \
    --threads=12 \
    --verbose \
    --no-shuffle \
    --seed=123
done
