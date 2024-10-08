#!/usr/bin/env python3

import argparse
import os
from io import StringIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from test_utilities import random_float_matrix, make_target_colwise, make_target_rowwise, print_cpp_matrix_declaration, \
    insert_text_in_file

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pp(name: str, x: torch.Tensor):
    if x.dim() == 1:
        print(f'{name} ({x.shape[0]})\n{x.data}')
    else:
        print(f'{name} ({x.shape[0]}x{x.shape[1]})\n{x.data}')


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        n = len(sizes) - 1
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

    def info(self):
        index = 1
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                pp(f'W{index}', layer.weight.data)
                pp(f'b{index}', layer.bias.data)
                index += 1


def train(M, X, T, optimizer, loss_fn, epochs):
    N, _ = X.shape  # N is the number of training examples

    result = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        Y = M(X)
        Y.retain_grad()
        loss = loss_fn(Y, T)
        loss.backward()
        DY = Y.grad.detach()
        result.append(Y.detach().numpy())
        result.append(DY.detach().numpy())
        optimizer.step()

    return result


def make_testcase(out: StringIO, name: str, sizes: list[int], N: int, rowwise=False):
    """
    Makes a test case for C++.
    :param name: The name of the test case.
    :param sizes: The input and output sizes of the layers.
    :param N: The number of samples in the data set.
    """

    D = sizes[0]   # the input size
    K = sizes[-1]  # the output size
    lr = 0.01      # the learning rate

    X = random_float_matrix(N, D, 0.0, 1.0).astype(np.float32)
    T = make_target_rowwise(N, K).astype(np.float32)  # a one hot encoded target

    X = torch.from_numpy(X)
    T = torch.from_numpy(T)

    device = torch.device("cpu")
    M = MLP(sizes).to(device)
    # M.info()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(M.parameters(), lr=lr)
    epochs = 2

    W1 = M.layers[0].weight.detach().numpy().copy()
    b1 = M.layers[0].bias.detach().numpy().copy()
    W2 = M.layers[1].weight.detach().numpy().copy()
    b2 = M.layers[1].bias.detach().numpy().copy()
    W3 = M.layers[2].weight.detach().numpy().copy()
    b3 = M.layers[2].bias.detach().numpy().copy()
    Y1, DY1, Y2, DY2 = train(M, X, T, optimizer, loss_fn, epochs)
    X = X.detach().numpy()
    T = T.detach().numpy()

    if not rowwise:
        X = X.T
        T = T.T
        Y1 = Y1.T
        DY1 = DY1.T
        Y2 = Y2.T
        DY2 = DY2.T

    out.write(f'TEST_CASE("{name}")\n')
    out.write('{\n')
    print_cpp_matrix_declaration(out, 'X', X)
    print_cpp_matrix_declaration(out, 'T', T)
    print_cpp_matrix_declaration(out, 'W1', W1)
    print_cpp_matrix_declaration(out, 'b1', b1, rowwise=rowwise)
    print_cpp_matrix_declaration(out, 'W2', W2)
    print_cpp_matrix_declaration(out, 'b2', b2, rowwise=rowwise)
    print_cpp_matrix_declaration(out, 'W3', W3)
    print_cpp_matrix_declaration(out, 'b3', b3, rowwise=rowwise)
    print_cpp_matrix_declaration(out, 'Y1', Y1)
    print_cpp_matrix_declaration(out, 'DY1', DY1)
    print_cpp_matrix_declaration(out, 'Y2', Y2)
    print_cpp_matrix_declaration(out, 'DY2', DY2)
    sizes_text = ', '.join(str(x) for x in sizes)
    out.write(f'  scalar lr = {lr};\n')
    out.write(f'  std::vector<long> sizes = {{{sizes_text}}};\n')
    out.write(f'  long N = {N};\n')
    out.write('  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, N);\n')
    out.write('}\n\n')


def main():
    parser = argparse.ArgumentParser(description='Convert and publish analysis files')
    parser.add_argument('--colwise', help='Generate tests for colwise layout', action='store_true')
    args = parser.parse_args()

    np.random.seed(42)
    np.set_printoptions(precision=6)
    torch.set_printoptions(precision=8, edgeitems=3, threshold=5, sci_mode=False, linewidth=160)
    rowwise = not args.colwise

    layer_sizes = [[2, 6, 4, 3], [3, 5, 2, 4], [6, 2, 2, 3]]
    example_counts = [5, 4, 8]

    out = StringIO()
    for i in range(3):
        make_testcase(out, f"test_mlp{i+1}", sizes=layer_sizes[i], N=example_counts[i], rowwise=rowwise)
    text = out.getvalue()
    insert_text_in_file('multilayer_perceptron_test.cpp', text)


if __name__ == '__main__':
    main()
