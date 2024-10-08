// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/loss_functions.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/activation_functions.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/matrix_operations.h"
#include "nerva/neural_networks/softmax_functions.h"
#include <cmath>
#include <memory>

namespace nerva {

template <typename Vector1, typename Vector2>
auto squared_error_loss_colwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;

  return dot(y - t, y - t);
}

template <typename Vector1, typename Vector2>
auto squared_error_loss_colwise_gradient(const Vector1& y, const Vector2& t)
{
  return 2 * (y - t);
}

template <typename Matrix1, typename Matrix2>
auto Squared_error_loss_colwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;

  return elements_sum(hadamard(Y - T, Y - T));
}

template <typename Matrix1, typename Matrix2>
auto Squared_error_loss_colwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  return 2 * (Y - T);
}

template <typename Vector1, typename Vector2>
auto mean_squared_error_loss_colwise(const Vector1& y, const Vector2& t)
{
  auto K = y.size();
  return squared_error_loss_colwise(y, t) / K;
}

template <typename Vector1, typename Vector2>
auto mean_squared_error_loss_colwise_gradient(const Vector1& y, const Vector2& t)
{
  auto K = y.size();
  return squared_error_loss_colwise_gradient(y, t) / K;
}

template <typename Matrix1, typename Matrix2>
auto Mean_squared_error_loss_colwise(const Matrix1& Y, const Matrix2& T)
{
  auto K = Y.rows();
  auto N = Y.cols();
  return Squared_error_loss_colwise(Y, T) / (K * N);
}

template <typename Matrix1, typename Matrix2>
auto Mean_squared_error_loss_colwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  auto K = Y.rows();
  auto N = Y.cols();
  return Squared_error_loss_colwise_gradient(Y, T) / (K * N);
}

template <typename Vector1, typename Vector2>
auto cross_entropy_loss_colwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;
  using eigen::log;

  return -dot(t, log(y));
}

template <typename Vector1, typename Vector2>
auto cross_entropy_loss_colwise_gradient(const Vector1& y, const Vector2& t)
{
  using eigen::hadamard;
  using eigen::inverse;

  return -hadamard(t, inverse(y));
}

template <typename Matrix1, typename Matrix2>
auto Cross_entropy_loss_colwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;
  using eigen::log;

  return -elements_sum(hadamard(T, log(Y)));
}

template <typename Matrix1, typename Matrix2>
auto Cross_entropy_loss_colwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  using eigen::hadamard;
  using eigen::inverse;

  return -hadamard(T, inverse(Y));
}

template <typename Vector1, typename Vector2>
auto softmax_cross_entropy_loss_colwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;

  return -dot(t, log_softmax_colwise(y));
}

template <typename Vector1, typename Vector2>
auto softmax_cross_entropy_loss_colwise_gradient(const Vector1& y, const Vector2& t)
{
  using eigen::elements_sum;

  return elements_sum(t) * softmax_colwise(y) - t;
}

template <typename Vector1, typename Vector2>
auto softmax_cross_entropy_loss_colwise_gradient_one_hot(const Vector1& y, const Vector2& t)
{
  return softmax_colwise(y) - t;
}

template <typename Matrix1, typename Matrix2>
auto Softmax_cross_entropy_loss_colwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;

  return -elements_sum(hadamard(T, log_softmax_colwise(Y)));
}

template <typename Matrix1, typename Matrix2>
auto Softmax_cross_entropy_loss_colwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  using eigen::columns_sum;
  using eigen::hadamard;
  using eigen::row_repeat;

  auto K = Y.rows();
  return hadamard(softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T;
}

template <typename Matrix1, typename Matrix2>
auto Softmax_cross_entropy_loss_colwise_gradient_one_hot(const Matrix1& Y, const Matrix2& T)
{
  return softmax_colwise(Y) - T;
}

template <typename Vector1, typename Vector2>
auto stable_softmax_cross_entropy_loss_colwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;

  return -dot(t, stable_log_softmax_colwise(y));
}

template <typename Vector1, typename Vector2>
auto stable_softmax_cross_entropy_loss_colwise_gradient(const Vector1& y, const Vector2& t)
{
  using eigen::elements_sum;

  return elements_sum(t) * stable_softmax_colwise(y) - t;
}

template <typename Vector1, typename Vector2>
auto stable_softmax_cross_entropy_loss_colwise_gradient_one_hot(const Vector1& y, const Vector2& t)
{
  return stable_softmax_colwise(y) - t;
}

template <typename Matrix1, typename Matrix2>
auto Stable_softmax_cross_entropy_loss_colwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;

  return -elements_sum(hadamard(T, stable_log_softmax_colwise(Y)));
}

template <typename Matrix1, typename Matrix2>
auto Stable_softmax_cross_entropy_loss_colwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  using eigen::columns_sum;
  using eigen::hadamard;
  using eigen::row_repeat;

  auto K = Y.rows();
  return hadamard(stable_softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T;
}

template <typename Matrix1, typename Matrix2>
auto Stable_softmax_cross_entropy_loss_colwise_gradient_one_hot(const Matrix1& Y, const Matrix2& T)
{
  return stable_softmax_colwise(Y) - T;
}

template <typename Vector1, typename Vector2>
auto logistic_cross_entropy_loss_colwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;
  using eigen::log_sigmoid;

  return -dot(t, log_sigmoid(y));
}

template <typename Vector1, typename Vector2>
auto logistic_cross_entropy_loss_colwise_gradient(const Vector1& y, const Vector2& t)
{
  using eigen::hadamard;

  return hadamard(t, Sigmoid(y)) - t;
}

template <typename Matrix1, typename Matrix2>
auto Logistic_cross_entropy_loss_colwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;
  using eigen::log_sigmoid;

  return -elements_sum(hadamard(T, log_sigmoid(Y)));
}

template <typename Matrix1, typename Matrix2>
auto Logistic_cross_entropy_loss_colwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  using eigen::hadamard;

  return hadamard(T, Sigmoid(Y)) - T;
}

template <typename Vector1, typename Vector2>
auto negative_log_likelihood_loss_colwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;

  return -std::log(dot(y, t));
}

template <typename Vector1, typename Vector2>
auto negative_log_likelihood_loss_colwise_gradient(const Vector1& y, const Vector2& t)
{
  using eigen::dot;

  return (-1 / dot(y, t)) * t;
}

template <typename Matrix1, typename Matrix2>
auto Negative_log_likelihood_loss_colwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::columns_sum;
  using eigen::elements_sum;
  using eigen::hadamard;
  using eigen::log;

  return -elements_sum(log(columns_sum(hadamard(Y, T))));
}

template <typename Matrix1, typename Matrix2>
auto Negative_log_likelihood_loss_colwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  using eigen::columns_sum;
  using eigen::elements_sum;
  using eigen::hadamard;
  using eigen::inverse;
  using eigen::row_repeat;

  auto K = Y.rows();
  return -hadamard(row_repeat(inverse(columns_sum(hadamard(Y, T))), K), T);
}

template <typename Vector1, typename Vector2>
auto squared_error_loss_rowwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;

  return dot(y - t, y - t);
}

template <typename Vector1, typename Vector2>
auto squared_error_loss_rowwise_gradient(const Vector1& y, const Vector2& t)
{
  return 2 * (y - t);
}

template <typename Matrix1, typename Matrix2>
auto Squared_error_loss_rowwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;

  return elements_sum(hadamard(Y - T, Y - T));
}

template <typename Matrix1, typename Matrix2>
auto Squared_error_loss_rowwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  return 2 * (Y - T);
}

template <typename Vector1, typename Vector2>
auto mean_squared_error_loss_rowwise(const Vector1& y, const Vector2& t)
{
  auto K = y.size();
  return squared_error_loss_rowwise(y, t) / K;
}

template <typename Vector1, typename Vector2>
auto mean_squared_error_loss_rowwise_gradient(const Vector1& y, const Vector2& t)
{
  auto K = y.size();
  return squared_error_loss_rowwise_gradient(y, t) / K;
}

template <typename Matrix1, typename Matrix2>
auto Mean_squared_error_loss_rowwise(const Matrix1& Y, const Matrix2& T)
{
  auto K = Y.cols();
  auto N = Y.rows();
  return Squared_error_loss_rowwise(Y, T) / (K * N);
}

template <typename Matrix1, typename Matrix2>
auto Mean_squared_error_loss_rowwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  auto K = Y.cols();
  auto N = Y.rows();
  return Squared_error_loss_rowwise_gradient(Y, T) / (K * N);
}

template <typename Vector1, typename Vector2>
auto cross_entropy_loss_rowwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;
  using eigen::log;

  return -dot(t, log(y));
}

template <typename Vector1, typename Vector2>
auto cross_entropy_loss_rowwise_gradient(const Vector1& y, const Vector2& t)
{
  using eigen::hadamard;
  using eigen::inverse;

  return -hadamard(t, inverse(y));
}

template <typename Matrix1, typename Matrix2>
auto Cross_entropy_loss_rowwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;
  using eigen::log;

  return -elements_sum(hadamard(T, log(Y)));
}

template <typename Matrix1, typename Matrix2>
auto Cross_entropy_loss_rowwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  using eigen::hadamard;
  using eigen::inverse;

  return -hadamard(T, inverse(Y));
}

template <typename Vector1, typename Vector2>
auto softmax_cross_entropy_loss_rowwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;

  return -dot(t, log_softmax_rowwise(y));
}

template <typename Vector1, typename Vector2>
auto softmax_cross_entropy_loss_rowwise_gradient(const Vector1& y, const Vector2& t)
{
  using eigen::elements_sum;

  return softmax_rowwise(y) * elements_sum(t) - t;
}

template <typename Vector1, typename Vector2>
auto softmax_cross_entropy_loss_rowwise_gradient_one_hot(const Vector1& y, const Vector2& t)
{
  return softmax_rowwise(y) - t;
}

template <typename Matrix1, typename Matrix2>
auto Softmax_cross_entropy_loss_rowwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;

  return -elements_sum(hadamard(T, log_softmax_rowwise(Y)));
}

template <typename Matrix1, typename Matrix2>
auto Softmax_cross_entropy_loss_rowwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  using eigen::column_repeat;
  using eigen::hadamard;
  using eigen::rows_sum;

  auto K = Y.cols();
  return hadamard(softmax_rowwise(Y), column_repeat(rows_sum(T), K)) - T;
}

template <typename Matrix1, typename Matrix2>
auto Softmax_cross_entropy_loss_rowwise_gradient_one_hot(const Matrix1& Y, const Matrix2& T)
{
  return softmax_rowwise(Y) - T;
}

template <typename Vector1, typename Vector2>
auto stable_softmax_cross_entropy_loss_rowwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;

  return -dot(t, stable_log_softmax_rowwise(y));
}

template <typename Vector1, typename Vector2>
auto stable_softmax_cross_entropy_loss_rowwise_gradient(const Vector1& y, const Vector2& t)
{
  using eigen::elements_sum;

  return stable_softmax_rowwise(y) * elements_sum(t) - t;
}

template <typename Vector1, typename Vector2>
auto stable_softmax_cross_entropy_loss_rowwise_gradient_one_hot(const Vector1& y, const Vector2& t)
{
  return stable_softmax_rowwise(y) - t;
}

template <typename Matrix1, typename Matrix2>
auto Stable_softmax_cross_entropy_loss_rowwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;

  return -elements_sum(hadamard(T, stable_log_softmax_rowwise(Y)));
}

template <typename Matrix1, typename Matrix2>
auto Stable_softmax_cross_entropy_loss_rowwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  using eigen::column_repeat;
  using eigen::hadamard;
  using eigen::rows_sum;

  auto K = Y.cols();
  return hadamard(stable_softmax_rowwise(Y), column_repeat(rows_sum(T), K)) - T;
}

template <typename Matrix1, typename Matrix2>
auto Stable_softmax_cross_entropy_loss_rowwise_gradient_one_hot(const Matrix1& Y, const Matrix2& T)
{
  return stable_softmax_rowwise(Y) - T;
}

template <typename Vector1, typename Vector2>
auto logistic_cross_entropy_loss_rowwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;
  using eigen::log;

  return -dot(t, log(Sigmoid(y)));
}

template <typename Vector1, typename Vector2>
auto logistic_cross_entropy_loss_rowwise_gradient(const Vector1& y, const Vector2& t)
{
  using eigen::hadamard;

  return hadamard(t, Sigmoid(y)) - t;
}

template <typename Matrix1, typename Matrix2>
auto Logistic_cross_entropy_loss_rowwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;
  using eigen::log;

  return -elements_sum(hadamard(T, log(Sigmoid(Y))));
}

template <typename Matrix1, typename Matrix2>
auto Logistic_cross_entropy_loss_rowwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  using eigen::hadamard;

  return hadamard(T, Sigmoid(Y)) - T;
}

template <typename Vector1, typename Vector2>
auto negative_log_likelihood_loss_rowwise(const Vector1& y, const Vector2& t)
{
  using eigen::dot;

  return -std::log(dot(y, t));
}

template <typename Vector1, typename Vector2>
auto negative_log_likelihood_loss_rowwise_gradient(const Vector1& y, const Vector2& t)
{
  using eigen::dot;

  return (-1 / dot(y, t)) * t;
}

template <typename Matrix1, typename Matrix2>
auto Negative_log_likelihood_loss_rowwise(const Matrix1& Y, const Matrix2& T)
{
  using eigen::elements_sum;
  using eigen::hadamard;
  using eigen::log;
  using eigen::rows_sum;

  return -elements_sum(log(rows_sum(hadamard(Y, T))));
}

template <typename Matrix1, typename Matrix2>
auto Negative_log_likelihood_loss_rowwise_gradient(const Matrix1& Y, const Matrix2& T)
{
  using eigen::column_repeat;
  using eigen::hadamard;
  using eigen::inverse;
  using eigen::rows_sum;

  auto K = Y.cols();
  return -hadamard(column_repeat(inverse(rows_sum(hadamard(Y, T))), K), T);
}

struct loss_function
{
  // tag::doc[]
  /// Calculate the loss for output `Y` and target `T`.
  [[nodiscard]] virtual scalar value(const eigen::matrix& Y, const eigen::matrix& T) const = 0;

  /// Calculate the gradient of the loss for output `Y` and target `T`.
  [[nodiscard]] virtual eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const = 0;
  // end::doc[]

  [[nodiscard]] virtual auto to_string() const -> std::string = 0;

  virtual ~loss_function() = default;
};

struct squared_error_loss: public loss_function
{
  template <typename Target>
  auto operator()(const eigen::vector& y, const Target& t) const -> scalar
  {
    return squared_error_loss_colwise(y, t);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return Squared_error_loss_colwise(Y, T);
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Squared_error_loss_colwise(Y, T);
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Squared_error_loss_colwise_gradient(Y, T);
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "SquaredErrorLoss()";
  }
};

struct cross_entropy_loss: public loss_function
{
  template <typename Target>
  auto operator()(const eigen::vector& y, const Target& t) const -> scalar
  {
    return cross_entropy_loss_colwise(y, t);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return Cross_entropy_loss_colwise(Y, T);
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Cross_entropy_loss_colwise(Y, T);
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Cross_entropy_loss_colwise_gradient(Y, T);
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "CrossEntropyLoss()";
  }
};

struct softmax_cross_entropy_loss: public loss_function
{
  template <typename Target>
  auto operator()(const eigen::vector& y, const Target& t) const -> scalar
  {
    return softmax_cross_entropy_loss_colwise(y, t);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return Softmax_cross_entropy_loss_colwise(Y, T);
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Softmax_cross_entropy_loss_colwise(Y, T);
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Softmax_cross_entropy_loss_colwise_gradient(Y, T);
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "SoftmaxCrossEntropyLoss()";
  }
};

struct logistic_cross_entropy_loss: public loss_function
{
  template <typename Target>
  auto operator()(const eigen::vector& y, const Target& t) const -> scalar
  {
    return logistic_cross_entropy_loss_colwise(y, t);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return Logistic_cross_entropy_loss_colwise(Y, T);
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Logistic_cross_entropy_loss_colwise(Y, T);
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Logistic_cross_entropy_loss_colwise_gradient(Y, T);
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "LogisticCrossEntropyLoss()";
  }
};

struct negative_log_likelihood_loss: public loss_function
{
  template <typename Target>
  auto operator()(const eigen::vector& y, const Target& t) const -> scalar
  {
    return negative_log_likelihood_loss_colwise(y, t);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return Negative_log_likelihood_loss_colwise(Y, T);
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Negative_log_likelihood_loss_colwise(Y, T);
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Negative_log_likelihood_loss_colwise_gradient(Y, T);
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "NegativeLogLikelihoodLoss()";
  }
};

inline
auto parse_loss_function(const std::string& text) -> std::shared_ptr<loss_function>
{
  if (text == "SquaredError")
  {
    return std::make_shared<squared_error_loss>();
  }
  else if (text == "CrossEntropy")
  {
    return std::make_shared<cross_entropy_loss>();
  }
  else if (text == "LogisticCrossEntropy")
  {
    return std::make_shared<logistic_cross_entropy_loss>();
  }
  else if (text == "SoftmaxCrossEntropy")
  {
    return std::make_shared<softmax_cross_entropy_loss>();
  }
  else if (text == "NegativeLogLikelihood")
  {
    return std::make_shared<negative_log_likelihood_loss>();
  }
  else
  {
    throw std::runtime_error("unknown loss function '" + text + "'");
  }
}

} // namespace nerva

