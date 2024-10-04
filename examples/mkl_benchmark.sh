#!/bin/bash

# tag::doc[]
../install/bin/mkl --arows=1000 --acols=1000 --brows=1000 --threads=12 --algorithm=sdd --repetitions=3 --densities="0.5,0.2,0.1,0.05"
# end::doc[]
