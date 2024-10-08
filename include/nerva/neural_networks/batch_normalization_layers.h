// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/batch_normalization_layers.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/nerva_timer.h"
#include "nerva/neural_networks/layers.h"
#include "fmt/format.h"
#include <random>

namespace nerva {

struct batch_normalization_layer: public neural_network_layer
{
  using super = neural_network_layer;
  using super::X;
  using super::DX;

  eigen::matrix Z;
  eigen::matrix DZ;
  eigen::matrix gamma;
  eigen::matrix Dgamma;
  eigen::matrix beta;
  eigen::matrix Dbeta;
  eigen::matrix inv_sqrt_Sigma;
  std::shared_ptr<optimizer_function> optimizer;

  explicit batch_normalization_layer(std::size_t D, std::size_t N = 1)
   : super(D, N), Z(D, N), DZ(D, N), gamma(D, 1), Dgamma(D, 1), beta(D, 1), Dbeta(D, 1), inv_sqrt_Sigma(D, 1)
  {
    beta.array() = 0;
    gamma.array() = 1;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("BatchNormalization(input_size={}, output_size={})", Z.rows(), Z.rows());
  }

  void feedforward(eigen::matrix& result) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::inv_sqrt;
    using eigen::column_repeat;
    using eigen::rows_mean;
    auto N = X.cols();

    auto R = (X - column_repeat(rows_mean(X), N)).eval();
    auto Sigma = diag(R * R.transpose()) / N;
    inv_sqrt_Sigma = inv_sqrt(Sigma);
    Z = hadamard(column_repeat(inv_sqrt_Sigma, N), R);
    result = hadamard(column_repeat(gamma, N), Z) + column_repeat(beta, N);
  }

  // tag::timer[]
  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::column_repeat;
    using eigen::rows_sum;
    using eigen::identity;
    using eigen::ones;
    using eigen::inv_sqrt;
    auto N = X.cols();

    NERVA_TIMER_START("batchnorm1")
    DZ = hadamard(column_repeat(gamma, N), DY);
    NERVA_TIMER_STOP("batchnorm1")

    NERVA_TIMER_START("batchnorm2")
    Dbeta = rows_sum(DY);
    NERVA_TIMER_STOP("batchnorm2")

    NERVA_TIMER_START("batchnorm3")
    Dgamma = rows_sum(hadamard(DY, Z));
    NERVA_TIMER_STOP("batchnorm3")

    NERVA_TIMER_START("batchnorm4")
    DX = hadamard(column_repeat(inv_sqrt_Sigma / N, N), hadamard(Z, column_repeat(-diag(DZ * Z.transpose()), N)) + DZ * (N * identity<eigen::matrix>(N) - ones<eigen::matrix>(N, N)));
    NERVA_TIMER_STOP("batchnorm4")
  }
  // end::timer[]

  void optimize(scalar eta) override
  {
    optimizer->update(eta);
  }

  void info(unsigned int layer_index) const override
  {
    std::string i = std::to_string(layer_index);
    std::cout << to_string() << std::endl;
    print_numpy_matrix("beta" + i, beta.transpose());
    print_numpy_matrix("gamma" + i, gamma.transpose());
  }
};

using dense_batch_normalization_layer = batch_normalization_layer;

// batch normalization without an affine transformation
struct simple_batch_normalization_layer: public neural_network_layer
{
  using super = neural_network_layer;
  using super::X;
  using super::DX;

  eigen::matrix inv_sqrt_Sigma;
  std::shared_ptr<optimizer_function> optimizer;

  explicit simple_batch_normalization_layer(std::size_t D, std::size_t N = 1)
    : super(D, N), inv_sqrt_Sigma(D, 1)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    return "SimpleBatchNormalization()";
  }

  void feedforward(eigen::matrix& result) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::inv_sqrt;
    using eigen::column_repeat;
    using eigen::rows_mean;
    auto N = X.cols();

    auto R = (X - column_repeat(rows_mean(X), N)).eval();
    auto Sigma = diag(R * R.transpose()) / N;
    inv_sqrt_Sigma = inv_sqrt(Sigma);
    result = hadamard(column_repeat(inv_sqrt_Sigma, N), R);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::identity;
    using eigen::ones;
    using eigen::inv_sqrt;
    using eigen::column_repeat;
    auto N = X.cols();

    DX = hadamard(column_repeat(inv_sqrt_Sigma / N, N), hadamard(Y, column_repeat(-diag(DY * Y.transpose()), N)) + DY * (N * identity<eigen::matrix>(N) - ones<eigen::matrix>(N, N)));
  }

  void optimize(scalar eta) override
  {
    optimizer->update(eta);
  }
};

using dense_simple_batch_normalization_layer = simple_batch_normalization_layer;

struct affine_layer: public neural_network_layer
{
  using super = neural_network_layer;
  using super::X;
  using super::DX;

  eigen::matrix gamma;
  eigen::matrix Dgamma;
  eigen::matrix beta;
  eigen::matrix Dbeta;
  std::shared_ptr<optimizer_function> optimizer;

  explicit affine_layer(std::size_t D, std::size_t N = 1)
    : super(D, N), gamma(D, 1), Dgamma(D, 1), beta(D, 1), Dbeta(D, 1)
  {
    beta.array() = 0;
    gamma.array() = 1;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return "Affine()";
  }

  void feedforward(eigen::matrix& result) override
  {
    using eigen::hadamard;
    using eigen::column_repeat;
    auto N = X.cols();

    result = hadamard(column_repeat(gamma, N), X) + column_repeat(beta, N);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::column_repeat;
    using eigen::rows_sum;
    auto N = X.cols();

    DX = hadamard(column_repeat(gamma, N), DY);
    Dbeta = rows_sum(DY);
    Dgamma = rows_sum(hadamard(X, DY));
  }

  void optimize(scalar eta) override
  {
    optimizer->update(eta);
  }
};

using dense_affine_layer = affine_layer;

template <typename BatchNormalizationLayer>
void set_batch_normalization_layer_optimizer(BatchNormalizationLayer& layer, const std::string& text)
{
  auto optimizer_beta = parse_optimizer(text, layer.beta, layer.Dbeta);
  auto optimizer_gamma = parse_optimizer(text, layer.gamma, layer.Dgamma);
  layer.optimizer = make_composite_optimizer(optimizer_beta, optimizer_gamma);
}

} // namespace nerva
