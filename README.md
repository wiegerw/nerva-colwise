![Nerva Logo](images/nerva-logo.png)
# nerva-colwise
The `nerva-colwise` library is a C++ library for neural networks. This library
is part of the Nerva library collection https://github.com/wiegerw/nerva, that includes
several native Python implementations. Originally the library was developed for experimenting with truly sparse neural networks. But nowadays, the library also aims to provide a transparent and accessible implementation of neural networks.

### Features
The `nerva-colwise` library has the following features:
* It supports common layers, loss functions and activation functions.
* It supports datasets in *column-wise* layout, i.e. each row of the dataset represents a sample. This deviates from frameworks like PyTorch and Tensorflow. Note that there is
  also a repository https://github.com/wiegerw/nerva-rowwise that supports data in *row-wise* layout.
* It supports mini-batches, and all equations (including backpropagation!) are given in matrix form.
* It supports truly sparse layers. The weight matrices of these layers are stored using a sparse matrix representation (CSR).
* It includes Python bindings.
* It has a good performance, which is achieved by using the Intel MKL library for calculating matrix products.

### Limitations
* Only multilayer perceptrons are supported.
* Only the CPU is supported.

### Documentation
The documentation consists of three parts:
* A [C++ manual](https://wiegerw.github.io/nerva-colwise/doc/nerva-cpp.html) that explains the implementation.
* A [Python manual](https://wiegerw.github.io/nerva-colwise/doc/nerva-python.html) that explains the `nerva` Python module.
* A PDF with [mathematical specifications](https://wiegerw.github.io/nerva-rowwise/pdf/nerva-libraries-implementation.pdf) of key components of the Nerva Library.

The following papers about Nerva are available:

[1] *Nerva: a Truly Sparse Implementation of Neural Networks*,  https://arxiv.org/abs/2407.17437. It introduces the library, and describes a number of static sparse training experiments.

[2] *Batch Matrix-form Equations and Implementation
of Multilayer Perceptrons*, https://arxiv.org/abs/TODO. It describes the implementation of the Nerva libraries in great detail.

### Requirements
A C++17 compiler. Due to the dependency on the Intel MKL library, an Intel processor is highly recommended. Intel MKL can technically work on non-Intel processors, but it is unlikely to perform optimally on them.

Compilation has been tested successfully with `g++-11`, `g++-12`, `g++-13`, `clang-18`, `icpx-2024.2` and `Visual Studio 2022`.

### Dependencies
Nerva uses the following third-party libraries.

* doctest (https://github.com/onqtam/doctest)
* FMT (https://github.com/fmtlib/fmt)
* Lyra (https://github.com/bfgroup/Lyra)
* Eigen (https://eigen.tuxfamily.org/)
* pybind11 (https://github.com/pybind/pybind11)
* Intel OneAPI (https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)

# Comparison with other frameworks
To illustrate the most important difference between Nerva and other neural network frameworks, the following characterization comes to mind. For typical neural network frameworks one could argue that ***the specification is the implementation***<sup><a href="#footnote1">1</a></sup>, while for Nerva ***the implementation is the specification***.
By this we mean that if you want to know precisely what a component of a neural network does, in many cases you will have to consult the source code in order to get a definitive answer. In Nerva, the specifications of key components are documented, and the implementation closely matches the specification.

<a id="footnote1"></a>1. Of course, in reality the situation is not this black and white.

# Getting Started

### Installation
C++ users can use `CMake` to install the library in a standard manner. Python users can install the library via `pip`. 
See the [C++ manual](https://wiegerw.github.io/nerva-colwise/doc/nerva-cpp.html) and/or the
[Python manual](https://wiegerw.github.io/nerva-colwise/doc/nerva-python.html) for more details about this.

### Command line interface
An easy way to start using the library is via a command line
called `mlp`. This tool is provided for both the C++ and the Python interface, and it can be used to train a neural network.
An example invocation of the C++ tool is the following:

```sh
../install/bin/mlp \
    --layers="ReLU;ReLU;Linear" \
    --layer-sizes="3072,1024,1024,10" \
    --layer-weights=Xavier \
    --optimizers="Nesterov(0.9)" \
    --loss=SoftmaxCrossEntropy \
    --learning-rate="Constant(0.01)" \
    --epochs=100 \
    --batch-size=100 \
    --threads=12 \
    --overall-density=0.05 \
    --cifar10=../data/cifar-10-batches-bin \
    --seed=123
```
This will train CIFAR10 using a network consisting of two hidden layers of size 1024. The initial weights are generated using Xavier, and in total 5% of the weights in the network is non-zero. The output may look like this:

```
=== Nerva c++ model ===
Sparse(input_size=3072, output_size=1024, density=0.042382877, optimizer=Nesterov(0.90000), activation=ReLU())
Sparse(input_size=1024, output_size=1024, density=0.06357384, optimizer=Nesterov(0.90000), activation=ReLU())
Dense(input_size=1024, output_size=10, optimizer=Nesterov(0.90000), activation=NoActivation())
loss = SoftmaxCrossEntropyLoss()
scheduler = ConstantScheduler(lr=0.01)
layer densities: 133325/3145728 (4.238%), 66662/1048576 (6.357%), 10240/10240 (100%)

epoch   0 lr: 0.01000000  loss: 2.30284437  train accuracy: 0.07904000  test accuracy: 0.08060000 time: 0.00000000s
epoch   1 lr: 0.01000000  loss: 2.14723837  train accuracy: 0.21136000  test accuracy: 0.21320000 time: 2.69463269s
epoch   2 lr: 0.01000000  loss: 1.91454245  train accuracy: 0.29976000  test accuracy: 0.29940000 time: 2.58654317s
epoch   3 lr: 0.01000000  loss: 1.78019225  train accuracy: 0.35416000  test accuracy: 0.35820000 time: 2.70981758s
...
```

# Design philosophy
In the first place, the `nerva-colwise` library is intended to support research with sparse neural networks. For that purpose, it contains algorithms for dynamic sparse training. Another goal of the library is to offer a completely transparent and accessible implementation of neural networks. The `nerva-colwise` library contains explicit formulations of backpropagation, and can thus be used to study in detail of how the execution of neural networks works. Instead, many other frameworks rely on auto differentation, which effectively hides the backpropagation from the user. The implementation of multilayer perceptrons is expressed in a small number of primitive [matrix operations](https://wiegerw.github.io/nerva-colwise/doc/nerva-cpp.html#_matrix_operations). This helps to keep the implementation clean and maintainable. Furthermore, the idea is that a well-structured implementation can serve as the basis for doing performance experiments.

### Other frameworks
There are many popular neural network frameworks available like [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/) and [JAX](https://jax.readthedocs.io/). For those who just want to train a neural network, these frameworks are perfectly adequate, and the Nerva libraries are not meant to compete with those. The Nerva libraries are better suited to be used in a research environment, or for people who want to learn about the details of neural networks.

### Performance
The `nerva-colwise` library uses a pragmatic approach with respect to performance. Our observation is that the performance of neural networks mostly relies on the performance of matrix multiplications, and for that we rely as much as possible on existing library solutions. For the CPU implementation we have opted for the Intel MKL library. But our implementation is modular, and this makes it relatively easy to add implementations based on other matrix libraries.

### Future work
There are two main directions for future work. Firstly, an implementation on GPUs is being considered. It would be very interesting to learn how well dynamic sparse training works on a GPU.
Second, we consider to add more layers, loss functions and activation functions. In particular convolutional layers, pooling layers and transformer layers come to mind.

# Contact
If you have questions, or if you would like to contribute to the Nerva libraries, you can email Wieger Wesselink (j.w.wesselink@tue.nl).
