#!/bin/bash

../install/bin/mlp \
    --layers="ReLU;AllReLU(0.5);SReLU(al=0,tl=0,ar=0,tr=1);TReLU(1e-20);Sigmoid;Softmax;LogSoftmax;HyperbolicTangent;BatchNormalization;Linear" \
    --layer-sizes="3072;128;128;128;128;128;128;128;128;10" \
    --layer-weights="Xavier;XavierNormalized;He;Uniform;Xavier;Xavier;Xavier;Xavier;Xavier" \
    --densities="1;1;1;1;0.5;1;1;1;1" \
    --optimizers="GradientDescent;Momentum(0.9);Nesterov(0.9);GradientDescent;GradientDescent;GradientDescent;GradientDescent;GradientDescent;GradientDescent;GradientDescent" \
    --loss=SoftmaxCrossEntropy \
    --learning-rate=0.01 \
    --epochs=1 \
    --batch-size=100 \
    --cifar10=../data \
    --threads=12 \
    --verbose \
    --no-shuffle \
    --seed=123
