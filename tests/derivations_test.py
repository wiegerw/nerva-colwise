#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# see also https://docs.sympy.org/latest/modules/matrices/matrices.html

from typing import List
from unittest import TestCase

import sympy as sp
from sympy import Lambda, Matrix, Piecewise

#-------------------------------------#
#           matrix operations
#-------------------------------------#

def is_column_vector(x: Matrix) -> bool:
    m, n = x.shape
    return n == 1


def is_row_vector(x: Matrix) -> bool:
    m, n = x.shape
    return m == 1


def is_square(X: Matrix) -> bool:
    m, n = X.shape
    return m == n


def join_columns(columns: List[Matrix]) -> Matrix:
    assert all(is_column_vector(column) for column in columns)
    return Matrix([x.T for x in columns]).T


def join_rows(rows: List[Matrix]) -> Matrix:
    assert all(is_row_vector(row) for row in rows)
    return Matrix(rows)


def diff(f, X: Matrix) -> Matrix:
    """
    Returns the derivative of a matrix function
    :param f: a real-valued function
    :param X: a matrix
    :return: the derivative of f
    """
    m, n = X.shape
    return Matrix([[sp.diff(f, X[i, j]) for j in range(n)] for i in range(m)])


def substitute(expr, X: Matrix, Y: Matrix):
    assert X.shape == Y.shape
    m, n = X.shape
    substitutions = ((X[i, j], Y[i, j]) for i in range(m) for j in range(n))
    return expr.subs(substitutions)


def jacobian(x: Matrix, y) -> Matrix:
    assert is_column_vector(x) or is_row_vector(x)
    return x.jacobian(y)


def apply(f, x: Matrix) -> Matrix:
    return x.applyfunc(f)


def exp(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.exp(x))


def log(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.log(x))


def sqrt(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.sqrt(x))


def inverse(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: 1 / x)


def inv_sqrt(x: Matrix) -> Matrix:
    return inverse(sqrt(x))


def diag(X: Matrix) -> Matrix:
    assert is_square(X)
    m, n = X.shape
    return Matrix([[X[i, i] for i in range(m)]]).T


def Diag(x: Matrix) -> Matrix:
    assert is_column_vector(x) or is_row_vector(x)
    return sp.diag(*x)


def hadamard(x: Matrix, y: Matrix) -> Matrix:
    assert x.shape == y.shape
    m, n = x.shape
    return Matrix([[x[i, j] * y[i, j] for j in range(n)] for i in range(m)])
    # return matrix_multiply_elementwise(x, y)  ==> this may cause errors:
    #     def mul_elementwise(A, B):
    # >       assert A.domain == B.domain
    # E       AssertionError


def sum_columns(X: Matrix) -> Matrix:
    m, n = X.shape
    columns = [sum(X.col(j)) for j in range(n)]
    return Matrix(columns).T


def sum_rows(X: Matrix) -> Matrix:
    m, n = X.shape
    rows = [sum(X.row(i)) for i in range(m)]
    return Matrix(rows)


def repeat_column(x: Matrix, n: int) -> Matrix:
    assert is_column_vector(x)
    rows, cols = x.shape
    rows = [[x[i, 0]] * n for i in range(rows)]
    return Matrix(rows)


def repeat_row(x: Matrix, n: int) -> Matrix:
    assert is_row_vector(x)
    rows, cols = x.shape
    columns = [[x[0, j]] * n for j in range(cols)]
    return Matrix(columns).T


def rowwise_mean(X: Matrix) -> Matrix:
    m, n = X.shape
    return sum_rows(X) / n


def colwise_mean(X: Matrix) -> Matrix:
    m, n = X.shape
    return sum_columns(X) / m


def identity(n: int) -> Matrix:
    return sp.eye(n)


def ones(m: int, n: int) -> Matrix:
    return sp.ones(m, n)


def sum_elements(X: Matrix):
    m, n = X.shape

    return sum(X[i, j] for i in range(m) for j in range(n))


def matrix(name: str, rows: int, columns: int) -> Matrix:
    return Matrix(sp.symarray(name, (rows, columns), real=True))


def equal_matrices(A: Matrix, B: Matrix, simplify_arguments=False) -> bool:
    m, n = A.shape
    if simplify_arguments:
        A = sp.simplify(A)
        B = sp.simplify(B)
    return A.shape == B.shape and sp.simplify(A - B) == sp.zeros(m, n)


#-------------------------------------#
#           activation functions
#-------------------------------------#

def relu(x):
    # return max(0, x)
    return Piecewise((0, x < 0), (x, True))


def relu_prime(x):
    # return 0 if x < 0 else 1
    return Piecewise((0, x < 0), (1, True))


def leaky_relu(alpha):
    x = sp.symbols('x')
    # fx = max(alpha * x, x)
    fx = Piecewise((alpha * x, x < alpha * x), (x, True))
    return Lambda(x, fx)


def leaky_relu_prime(alpha):
    x = sp.symbols('x')
    # fx = alpha if x < alpha * x else 1
    fx = Piecewise((alpha, x < alpha * x), (1, True))
    return Lambda(x, fx)


def all_relu(alpha):
    x = sp.symbols('x')
    # fx = alpha * x if x < 0 else x
    fx = Piecewise((alpha * x, x < 0), (x, True))
    return Lambda(x, fx)


def all_relu_prime(alpha):
    x = sp.symbols('x')
    # fx = alpha if x < 0 else 1
    fx = Piecewise((alpha, x < 0), (1, True))
    return Lambda(x, fx)

def hyperbolic_tangent(x):
    return sp.tanh(x)


def hyperbolic_tangent_prime(x):
    y = hyperbolic_tangent(x)
    return 1 - y * y


def sigmoid(x):
    return 1 / (1 + sp.exp(-x))


def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)


def srelu(al, tl, ar, tr):
    x = sp.symbols('x')
    return Lambda(x, Piecewise((tl + al * (x - tl), x <= tl), (x, x < tr), (tr + ar * (x - tr), True)))


def srelu_prime(al, tl, ar, tr):
    x = sp.symbols('x')
    return Lambda(x, Piecewise((al, x <= tl), (1, x < tr), (ar, True)))


#-------------------------------------#
#           loss functions
#-------------------------------------#

def squared_error(X: Matrix):
    m, n = X.shape

    def f(x: Matrix) -> float:
        return sp.sqrt(sum(xi * xi for xi in x))

    return sum(f(X.col(j)) for j in range(n))


#-------------------------------------#
#           softmax colwise
#-------------------------------------#

def softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    E = exp(X)
    return hadamard(E, repeat_row(inverse(sum_columns(E)), m))


def softmax_colwise1(X: Matrix) -> Matrix:
    m, n = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return Matrix([softmax(X.col(j)).T for j in range(n)]).T


def stable_softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (1, n), real=True))
    E = exp(X - repeat_row(c, m))
    return hadamard(E, repeat_row(inverse(sum_columns(E)), m))


def softmax_colwise_derivative(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    y = softmax_colwise1(x)
    return Diag(y) - y * y.T


def softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(softmax_colwise1(x), x)


#-------------------------------------#
#           log_softmax colwise
#-------------------------------------#

def log_softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    return X - repeat_row(log(sum_columns(exp(X))), m)


def log_softmax_colwise1(X: Matrix) -> Matrix:
    return log(softmax_colwise(X))


def stable_log_softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (1, n), real=True))
    Y = X - repeat_row(c, m)
    return Y - repeat_row(log(sum_columns(exp(Y))), m)


def log_softmax_colwise_derivative(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    m, n = x.shape
    return sp.eye(m) - repeat_row(softmax_colwise(x).T, m)


def log_softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(log_softmax_colwise(x), x)


#-------------------------------------#
#           softmax rowwise
#-------------------------------------#

def softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    E = exp(X)
    return hadamard(E, repeat_column(inverse(sum_rows(E)), n))


def softmax_rowwise1(X: Matrix) -> Matrix:
    m, n = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return join_rows([softmax(X.row(i)) for i in range(m)])


def softmax_rowwise2(X: Matrix) -> Matrix:
    return softmax_colwise(X.T).T


def stable_softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (m, 1), real=True))
    E = exp(X - repeat_column(c, n))
    return hadamard(E, repeat_column(inverse(sum_rows(E)), n))


def softmax_rowwise_derivative(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    y = softmax_rowwise(x)
    return Diag(y) - y.T * y


def softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(softmax_rowwise(x), x)


def softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return softmax_colwise_derivative(x.T).T


#-------------------------------------#
#           log_softmax rowwise
#-------------------------------------#

def log_softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    return X - repeat_column(log(sum_rows(exp(X))), n)


def log_softmax_rowwise1(X: Matrix) -> Matrix:
    return log(softmax_rowwise(X))


def log_softmax_rowwise2(X: Matrix) -> Matrix:
    return log_softmax_colwise(X.T).T


def stable_log_softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (m, 1), real=True))
    Y = X - repeat_column(c, n)
    return Y - repeat_column(log(sum_rows(exp(Y))), n)


def log_softmax_rowwise_derivative(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    m, n = x.shape
    return sp.eye(n) - repeat_row(softmax_rowwise(x), n)


def log_softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(log_softmax_rowwise(x), x)


def log_softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return log_softmax_colwise_derivative(x.T)


class TestSoftmax(TestCase):
    def test_softmax_colwise(self):
        m = 3
        n = 2
        X = Matrix(sp.symarray('x', (m, n), real=True))

        y1 = softmax_colwise(X)
        y2 = softmax_colwise1(X)
        y3 = stable_softmax_colwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))

        y1 = log_softmax_colwise(X)
        y2 = log_softmax_colwise1(X)
        y3 = stable_log_softmax_colwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))

    def test_softmax_rowwise(self):
        m = 2
        n = 3
        X = Matrix(sp.symarray('x', (m, n), real=True))

        y1 = softmax_rowwise(X)
        y2 = softmax_rowwise1(X)
        y3 = softmax_rowwise2(X)
        y4 = stable_softmax_rowwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y4), sp.zeros(m, n))

        y1 = log_softmax_rowwise(X)
        y2 = log_softmax_rowwise1(X)
        y3 = log_softmax_rowwise2(X)
        y4 = stable_log_softmax_rowwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y4), sp.zeros(m, n))

    def test_softmax_colwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        m, n = x.shape

        y1 = sp.simplify(softmax_colwise_derivative(x))
        y2 = sp.simplify(softmax_colwise_derivative1(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, m))

    def test_log_softmax_colwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        m, n = x.shape

        y1 = sp.simplify(log_softmax_colwise_derivative(x))
        y2 = sp.simplify(log_softmax_colwise_derivative1(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, m))

    def test_softmax_rowwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        m, n = x.shape

        y1 = sp.simplify(softmax_rowwise_derivative(x))
        y2 = sp.simplify(softmax_rowwise_derivative1(x))
        y3 = sp.simplify(softmax_rowwise_derivative2(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(n, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(n, n))

    def test_log_softmax_rowwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        m, n = x.shape

        y1 = sp.simplify(log_softmax_rowwise_derivative(x))
        y2 = sp.simplify(log_softmax_rowwise_derivative1(x))
        y3 = sp.simplify(log_softmax_rowwise_derivative2(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(n, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(n, n))


class TestMatrixOperations(TestCase):
    def test_matrix_operations(self):
        A = Matrix([[1, 2, 3]])
        B = repeat_row(A, 2)
        C = Matrix([[1, 2, 3],
                    [1, 2, 3]])
        self.assertEqual(B, C)


class TestActivationFunctions(TestCase):

    def test_relu(self):
        f = relu
        f1 = relu_prime
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_leaky_relu(self):
        alpha = sp.symbols('alpha', real=True)
        f = leaky_relu(alpha)
        f1 = leaky_relu_prime(alpha)
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_all_relu(self):
        alpha = sp.symbols('alpha', real=True)
        f = all_relu(alpha)
        f1 = all_relu_prime(alpha)
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_hyperbolic_tangent(self):
        f = hyperbolic_tangent
        f1 = hyperbolic_tangent_prime
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_sigmoid(self):
        f = sigmoid
        f1 = sigmoid_prime
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_srelu(self):
        al = sp.symbols('al', real=True)
        tl = sp.symbols('tl', real=True)
        ar = sp.symbols('ar', real=True)
        tr = sp.symbols('tr', real=True)

        f = srelu(al, tl, ar, tr)
        f1 = srelu_prime(al, tl, ar, tr)
        x = sp.symbols('x', real=True)
        self.assertEqual(f1(x), f(x).diff(x))


class TestLinearLayers(TestCase):

    def test_linear_layer_colwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)

        # feedforward
        X = x
        W = w
        Y = W * X + repeat_column(b, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DW = DY * X.T
        Db = sum_rows(DY)
        DX = W.T * DY

        # symbolic differentiation
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_activation_layer_colwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error
        act = hyperbolic_tangent
        act_prime = hyperbolic_tangent_prime

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)

        # feedforward
        X = x
        W = w
        Z = W * X + repeat_column(b, N)
        Y = apply(act, Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(DY, apply(act_prime, Z))
        DW = DZ * X.T
        Db = sum_rows(DZ)
        DX = W.T * DZ

        # symbolic differentiation
        DZ1 = substitute(diff(loss(apply(act, z)), z), z, Z)
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_sigmoid_layer_colwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error
        sigma = sigmoid

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)

        # feedforward
        X = x
        W = w
        Z = W * X + repeat_column(b, N)
        Y = apply(sigma, Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(DY, hadamard(Y, ones(K, N) - Y))
        DW = DZ * X.T
        Db = sum_rows(DZ)
        DX = W.T * DZ

        # symbolic differentiation
        Y_z = apply(sigma, z)
        DZ1 = substitute(diff(loss(Y_z), z), z, Z)
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_linear_layer_rowwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)

        # feedforward
        X = x
        W = w
        Y = X * W.T + repeat_row(b, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DW = DY.T * X
        Db = sum_columns(DY)
        DX = DY * W

        # symbolic differentiation
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_activation_layer_rowwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error
        act = hyperbolic_tangent
        act_prime = hyperbolic_tangent_prime

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)

        # feedforward
        X = x
        W = w
        Z = X * W.T + repeat_row(b, N)
        Y = apply(act, Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(DY, apply(act_prime, Z))
        DW = DZ.T * X
        Db = sum_columns(DZ)
        DX = DZ * W

        # symbolic differentiation
        DZ1 = substitute(diff(loss(apply(act, z)), z), z, Z)
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_sigmoid_layer_rowwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error
        sigma = sigmoid

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)

        # feedforward
        X = x
        W = w
        Z = X * W.T + repeat_row(b, N)
        Y = apply(sigma, Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(DY, hadamard(Y, ones(N, K) - Y))
        DW = DZ.T * X
        Db = sum_columns(DZ)
        DX = DZ * W

        # symbolic differentiation
        Y_z = apply(sigma, z)
        DZ1 = substitute(diff(loss(Y_z), z), z, Z)
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))


class TestSReLULayers(TestCase):

    # test for both colwise and rowwise
    def test_srelu_layer(self):
        N = 2
        K = 2
        loss = sum_elements

        # variables
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        al = sp.symbols('al', real=True)
        tl = sp.symbols('tl', real=True)
        ar = sp.symbols('ar', real=True)
        tr = sp.symbols('tr', real=True)

        act = srelu(al, tl, ar, tr)
        act_prime = srelu_prime(al, tl, ar, tr)

        # feedforward
        Z = z
        Y = apply(act, Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(DY, apply(act_prime, Z))

        Zij = sp.symbols('Zij')
        Al = apply(Lambda(Zij, Piecewise((Zij - tl, Zij <= tl), (0, True))), Z)
        Ar = apply(Lambda(Zij, Piecewise((0, Zij <= tl), (0, Zij < tr), (Zij - tr, True))), Z)
        Tl = apply(Lambda(Zij, Piecewise((1 - al, Zij <= tl), (0, True))), Z)
        Tr = apply(Lambda(Zij, Piecewise((0, Zij <= tl), (0, Zij < tr), (1 - ar, True))), Z)

        Dal = Matrix([[sum_elements(hadamard(DY, Al))]])
        Dar = Matrix([[sum_elements(hadamard(DY, Ar))]])
        Dtl = Matrix([[sum_elements(hadamard(DY, Tl))]])
        Dtr = Matrix([[sum_elements(hadamard(DY, Tr))]])

        # symbolic differentiation
        DZ1 = diff(loss(Y), z)
        Dal1 = diff(loss(Y), Matrix([[al]]))
        Dtl1 = diff(loss(Y), Matrix([[tl]]))
        Dar1 = diff(loss(Y), Matrix([[ar]]))
        Dtr1 = diff(loss(Y), Matrix([[tr]]))

        self.assertTrue(equal_matrices(DZ, DZ1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dal, Dal1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dtl, Dtl1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dar, Dar1, simplify_arguments=False))
        self.assertTrue(equal_matrices(Dtr, Dtr1, simplify_arguments=True))


class TestSoftmaxLayers(TestCase):

    def test_softmax_layer_colwise(self):
        K = 3
        N = 2
        loss = squared_error

        # variables
        y = matrix('y', K, N)
        z = matrix('z', K, N)

        # feedforward
        Z = z
        Y = softmax_colwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(Y, DY - repeat_row(diag(Y.T * DY).T, K))

        # symbolic differentiation
        DZ1 = diff(loss(Y), z)

        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_log_softmax_layer_colwise(self):
        K = 3
        N = 2
        loss = squared_error

        # variables
        y = matrix('y', K, N)
        z = matrix('z', K, N)

        # feedforward
        Z = z
        Y = log_softmax_colwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = DY - hadamard(softmax_colwise(Z), repeat_row(sum_columns(DY), K))

        # symbolic differentiation
        DZ1 = diff(loss(Y), z)

        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_softmax_layer_rowwise(self):
        K = 2
        N = 3
        loss = squared_error

        # variables
        y = matrix('y', K, N)
        z = matrix('z', K, N)

        # feedforward
        Z = z
        Y = softmax_rowwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(Y, DY - repeat_column(diag(DY * Y.T), N))

        # symbolic differentiation
        DZ1 = diff(loss(Y), z)

        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_log_softmax_layer_rowwise(self):
        K = 2
        N = 3
        loss = squared_error

        # variables
        y = matrix('y', K, N)
        z = matrix('z', K, N)

        # feedforward
        Z = z
        Y = log_softmax_rowwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = DY - hadamard(softmax_rowwise(Z), repeat_column(sum_rows(DY), N))

        # symbolic differentiation
        DZ1 = diff(loss(Y), z)

        self.assertTrue(equal_matrices(DZ, DZ1))


class TestBatchNormalizationLayers(TestCase):

    def test_simple_batch_normalization_layer_colwise(self):
        D = 3
        N = 2
        K = D                # K and D are always equal in batch normalization
        loss = sum_elements  # squared_error seems too complicated

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)

        # feedforward
        X = x
        R = X - repeat_column(rowwise_mean(X), N)
        Sigma = diag(R * R.T) / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Y = hadamard(repeat_column(inv_sqrt_Sigma, N), R)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(repeat_column(inv_sqrt_Sigma / N, N),
                      hadamard(Y, repeat_column(-diag(DY * Y.T), N)) + DY * (N * identity(N) - ones(N, N)))

        # symbolic differentiation
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DX, DX1))

    def test_affine_layer_colwise(self):
        D = 3
        N = 2
        K = D
        loss = squared_error

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        beta = matrix('beta', K, 1)
        gamma = matrix('gamma', K, 1)

        # feedforward
        X = x
        Y = hadamard(repeat_column(gamma, N), X) + repeat_column(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(repeat_column(gamma, N), DY)
        Dbeta = sum_rows(DY)
        Dgamma = sum_rows(hadamard(X, DY))

        # symbolic differentiation
        DX1 = diff(loss(Y), x)
        Dbeta1 = diff(loss(Y), beta)
        Dgamma1 = diff(loss(Y), gamma)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))

    def test_batch_normalization_layer_colwise(self):
        D = 3
        N = 2
        K = D                # K and D are always equal in batch normalization
        loss = sum_elements  # squared_error seems too complicated

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        beta = matrix('beta', K, 1)
        gamma = matrix('gamma', K, 1)

        # feedforward
        X = x
        R = X - repeat_column(rowwise_mean(X), N)
        Sigma = diag(R * R.T) / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Z = hadamard(repeat_column(inv_sqrt_Sigma, N), R)
        Y = hadamard(repeat_column(gamma, N), Z) + repeat_column(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(repeat_column(gamma, N), DY)
        Dbeta = sum_rows(DY)
        Dgamma = sum_rows(hadamard(DY, Z))
        DX = hadamard(repeat_column(inv_sqrt_Sigma / N, N),
                      hadamard(Z, repeat_column(-diag(DZ * Z.T), N)) + DZ * (N * identity(N) - ones(N, N)))

        # symbolic differentiation
        DX1 = diff(loss(Y), x)
        Dbeta1 = diff(loss(Y), beta)
        Dgamma1 = diff(loss(Y), gamma)
        Y_z = hadamard(repeat_column(gamma, N), z) + repeat_column(beta, N)
        DZ1 = substitute(diff(loss(Y_z), z), z, Z)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))
        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_simple_batch_normalization_layer_rowwise(self):
        D = 3
        N = 2
        K = D                # K and D are always equal in batch normalization
        loss = sum_elements  # squared_error seems too complicated

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)

        # feedforward
        X = x
        R = (identity(N) - ones(N, N) / N) * X
        Sigma = diag(R.T * R).T / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Y = hadamard(repeat_row(inv_sqrt_Sigma, N), R)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(repeat_row(inv_sqrt_Sigma / N, N),
                      (N * identity(N) - ones(N, N)) * DY - hadamard(Y, repeat_row(diag(Y.T * DY).T, N)))

        # symbolic differentiation
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DX, DX1))

    def test_affine_layer_rowwise(self):
        D = 3
        N = 2
        K = D
        loss = squared_error

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        beta = matrix('beta', 1, K)
        gamma = matrix('gamma', 1, K)

        # feedforward
        X = x
        Y = hadamard(repeat_row(gamma, N), X) + repeat_row(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(repeat_row(gamma, N), DY)
        Dbeta = sum_columns(DY)
        Dgamma = sum_columns(hadamard(X, DY))

        # symbolic differentiation
        DX1 = diff(loss(Y), x)
        Dbeta1 = diff(loss(Y), beta)
        Dgamma1 = diff(loss(Y), gamma)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))

    def test_batch_normalization_layer_rowwise(self):
        D = 3
        N = 2
        K = D                # K and D are always equal in batch normalization
        loss = sum_elements  # squared_error seems too complicated

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        beta = matrix('beta', 1, K)
        gamma = matrix('gamma', 1, K)

        # feedforward
        X = x
        R = (identity(N) - ones(N, N) / N) * X
        Sigma = diag(R.T * R).T / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Z = hadamard(repeat_row(inv_sqrt_Sigma, N), R)
        Y = hadamard(repeat_row(gamma, N), Z) + repeat_row(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(repeat_row(gamma, N), DY)
        Dbeta = sum_columns(DY)
        Dgamma = sum_columns(hadamard(Z, DY))
        DX = hadamard(repeat_row(inv_sqrt_Sigma / N, N),
                      (N * identity(N) - ones(N, N)) * DZ - hadamard(Z, repeat_row(diag(Z.T * DZ).T, N)))

        # symbolic differentiation
        DX1 = diff(loss(Y), x)
        Dbeta1 = diff(loss(Y), beta)
        Dgamma1 = diff(loss(Y), gamma)
        Y_z = hadamard(repeat_row(gamma, N), z) + repeat_row(beta, N)
        DZ1 = substitute(diff(loss(Y_z), z), z, Z)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))
        self.assertTrue(equal_matrices(DZ, DZ1))


class TestLemmas(TestCase):
    def test_lemma_colwise1(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_columns([X.col(j) * X.col(j).T * Y.col(j) for j in range(n)])
        Z2 = hadamard(X, repeat_row(diag(X.T * Y).T, m))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_colwise2(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_columns([repeat_row(X.col(j).T, m) * Y.col(j) for j in range(n)])
        Z2 = repeat_row(diag(X.T * Y).T, m)
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_colwise3(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_columns([repeat_column(X.col(j), m) * Y.col(j) for j in range(n)])
        Z2 = hadamard(X, repeat_row(sum_columns(Y), m))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_rowwise1(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * Y.row(i).T * Y.row(i) for i in range(m)])
        Z2 = hadamard(Y, repeat_column(diag(X * Y.T), n))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_rowwise2(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * repeat_column(Y.row(i).T, n) for i in range(m)])
        Z2 = repeat_column(diag(X * Y.T), n)
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_rowwise3(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * repeat_row(Y.row(i), n) for i in range(m)])
        Z2 = hadamard(Y, repeat_column(sum_rows(X), n))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))


class TestDerivatives(TestCase):
    def test_derivative_gx_x_colwise(self):
        n = 3
        x = Matrix(sp.symbols('x:{}'.format(n)))
        self.assertTrue(is_column_vector(x))

        g = sp.Function('g', real=True)(*x)
        J1 = jacobian(g * x, x)
        J2 = x * jacobian(Matrix([[g]]), x) + g * sp.eye(n)
        self.assertEqual(sp.simplify(J1 - J2), sp.zeros(n, n))

    def test_derivative_gx_x_rowwise(self):
        n = 3
        x = Matrix(sp.symbols('x:{}'.format(n))).T
        self.assertTrue(is_row_vector(x))

        g = sp.Function('g', real=True)(*x)
        J1 = jacobian(g * x, x)
        J2 = x.T * jacobian(Matrix([[g]]), x) + g * sp.eye(n)
        self.assertEqual(sp.simplify(J1 - J2), sp.zeros(n, n))


if __name__ == '__main__':
    import unittest
    unittest.main()
