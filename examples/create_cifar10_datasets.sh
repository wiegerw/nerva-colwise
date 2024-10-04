#!/bin/bash

# This script can be used to create preprocessed CIFAR-10 datasets.
# Each dataset contains augmented and shuffled data. The datasets
# are stored in files cifar1/epoch000.npz, cifar1/epoch001.npz etc.

# tag::doc[]
python3 ../python/tools/create_cifar10_datasets.py --epochs=100 --seed=1 --outputdir=cifar1 --datadir=../data
# end::doc[]
