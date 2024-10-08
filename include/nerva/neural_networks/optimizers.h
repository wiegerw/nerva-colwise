// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/optimizers.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/utilities/parse.h"
#include "nerva/utilities/parse_numbers.h"
#include "fmt/format.h"

namespace nerva {

// Generic optimizer_function for dense or sparse matrices.
struct optimizer_function
{
  virtual void update(scalar eta) = 0;

  virtual void clip(scalar epsilon)
  {}

  // Update the support (i.e. the set of nonzero entries). Only applies to sparse matrices.
  virtual void reset_support()
  {}

  [[nodiscard]] virtual auto to_string() const -> std::string = 0;

  virtual ~optimizer_function() = default;
};

template <typename T>
struct gradient_descent_optimizer: public optimizer_function
{
  T& x;
  T& Dx;

  gradient_descent_optimizer(T& x_, T& Dx_)
    : x(x_), Dx(Dx_)
  {}

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "GradientDescent";
  }

  void update(scalar eta) override
  {
    if constexpr (std::is_same<T, mkl::sparse_matrix_csr<scalar>>::value)
    {
      mkl::ss_sum(x, Dx, scalar(1), -eta);
    }
    else
    {
      if (NervaComputation == computation::eigen)
      {
        x -= eta * Dx;
      }
      else
      {
        auto x_view = mkl::make_dense_matrix_view(x);
        auto Dx_view = mkl::make_dense_matrix_view(Dx);
        mkl::cblas_axpy(-eta, Dx_view, x_view);
      }
    }
  }
};

template <typename T>
struct momentum_optimizer: public gradient_descent_optimizer<T>
{
  using super = gradient_descent_optimizer<T>;
  using super::x;
  using super::Dx;
  using super::reset_support;
  static constexpr bool IsSparse = std::is_same_v<T, mkl::sparse_matrix_csr<scalar>>;

  T delta_x;
  scalar mu;

  momentum_optimizer(T& x, T& Dx, scalar mu_)
    : super(x, Dx),
      delta_x(x.rows(), x.cols()),
      mu(mu_)
  {
    if constexpr (IsSparse)
    {
      reset_support();
    }
    else
    {
      delta_x.array() = scalar(0);
    }
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return fmt::format("Momentum({:7.5f})", mu);
  }

  void update(scalar eta) override
  {
    if constexpr (IsSparse)
    {
      mkl::ss_sum(delta_x, Dx, mu, -eta);
      mkl::ss_sum(x, delta_x, scalar(1), scalar(1));
    }
    else
    {
      if (NervaComputation == computation::eigen)
      {
        delta_x = mu * delta_x - eta * Dx;
        x += delta_x;
      }
      else if (NervaComputation == computation::mkl || NervaComputation == computation::blas)
      {
        auto x_view = mkl::make_dense_matrix_view(x);
        auto Dx_view = mkl::make_dense_matrix_view(Dx);
        auto delta_x_view = mkl::make_dense_matrix_view(delta_x);
        mkl::cblas_scal(mu, delta_x_view);
        mkl::cblas_axpy(-eta, Dx_view, delta_x_view);
        mkl::cblas_axpy(scalar(1), delta_x_view, x_view);
      }
      else if (NervaComputation == computation::sycl)
      {
#ifdef NERVA_SYCL
        auto Dx_view = mkl::make_dense_vector_view(Dx);
        auto delta_x_view = mkl::make_dense_vector_view(delta_x);
        assign_axby(mu, -eta, delta_x_view, Dx_view, delta_x_view);
        x += delta_x;
#else
      throw std::runtime_error("computation mode sycl is disabled");
#endif
      }
    }
  }

  void clip(scalar epsilon) override
  {
    if constexpr (IsSparse)
    {
      mkl::clip(delta_x, epsilon);
    }
    else
    {
      eigen::clip(delta_x, epsilon);
    }
  }

  void reset_support() override
  {
    if constexpr (IsSparse)
    {
      delta_x.reset_support(x);
    }
  }
};

template <typename T>
struct nesterov_optimizer: public momentum_optimizer<T>
{
  using super = momentum_optimizer<T>;
  using super::x;
  using super::Dx;
  using super::delta_x;
  using super::mu;
  static constexpr bool IsSparse = std::is_same_v<T, mkl::sparse_matrix_csr<scalar>>;

  void reset_support() override
  {
    if constexpr (IsSparse)
    {
      delta_x.reset_support(x);
    }
  }

  nesterov_optimizer(T& x, T& Dx, scalar mu)
    : super(x, Dx, mu)
  { }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return fmt::format("Nesterov({:7.5f})", mu);
  }

  // Keras equations:
  // 1. velocity = momentum * velocity - learning_rate * g
  // 2. w = w + momentum * velocity - learning_rate * g
  void update(scalar eta) override
  {
    if constexpr (IsSparse)
    {
      mkl::ss_sum(delta_x, Dx, mu, -eta);
      mkl::sss_sum(x, delta_x, Dx, scalar(1), mu, -eta);
    }
    else
    {
      if (NervaComputation == computation::eigen || NervaComputation == computation::mkl)
      {
        delta_x = mu * delta_x - eta * Dx;
        x = x + mu * delta_x - eta * Dx;
      }
      else if (NervaComputation == computation::blas)
      {
        auto x_view = mkl::make_dense_matrix_view(x);
        auto Dx_view = mkl::make_dense_matrix_view(Dx);
        auto delta_x_view = mkl::make_dense_matrix_view(delta_x);
        mkl::cblas_scal(mu, delta_x_view);
        mkl::cblas_axpy(-eta, Dx_view, delta_x_view);
        mkl::cblas_axpy(mu, delta_x_view, x_view);
        mkl::cblas_axpy(-eta, Dx_view, x_view);
      }
      else if (NervaComputation == computation::sycl)
      {
#ifdef NERVA_SYCL
        auto x_view = mkl::make_dense_vector_view(x);
        auto Dx_view = mkl::make_dense_vector_view(Dx);
        auto delta_x_view = mkl::make_dense_vector_view(delta_x);
        assign_axby(mu, -eta, delta_x_view, Dx_view, delta_x_view);
        add_axby(mu, -eta, delta_x_view, Dx_view, x_view);
#else
        throw std::runtime_error("computation mode sycl is disabled");
#endif
      }
    }
  }
};

struct composite_optimizer: public optimizer_function
{
  std::vector<std::shared_ptr<optimizer_function>> optimizers;

  composite_optimizer() = default;

  composite_optimizer(const composite_optimizer& other)
    : optimizers(other.optimizers)
  {}

  composite_optimizer(std::initializer_list<std::shared_ptr<optimizer_function>> items)
    : optimizers(items)
  {}

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return optimizers.front()->to_string();
  }

  void update(scalar eta) override
  {
    for (auto& optimizer: optimizers)
    {
      optimizer->update(eta);
    }
  }

  void clip(scalar epsilon) override
  {
    for (auto& optimizer: optimizers)
    {
      optimizer->clip(epsilon);
    }
  }

  void reset_support() override
  {
    for (auto& optimizer: optimizers)
    {
      optimizer->reset_support();
    }
  }
};

template <typename... Args>
auto make_composite_optimizer(Args&&... args) -> std::shared_ptr<composite_optimizer>
{
  return std::make_shared<composite_optimizer>(std::initializer_list<std::shared_ptr<optimizer_function>>{std::forward<Args>(args)...});
}

template <typename T>
auto parse_optimizer(const std::string& text, T& x, T& Dx) -> std::shared_ptr<optimizer_function>
{
  auto func = utilities::parse_function_call(text);

  if (func.name == "GradientDescent")
  {
    return std::make_shared<gradient_descent_optimizer<T>>(x, Dx);
  }
  else if (func.name == "Momentum")
  {
    scalar mu = func.as_scalar("momentum");
    return std::make_shared<momentum_optimizer<T>>(x, Dx, mu);
  }
  else if (func.name == "Nesterov")
  {
    scalar mu = func.as_scalar("momentum");
    return std::make_shared<nesterov_optimizer<T>>(x, Dx, mu);
  }
  else
  {
    throw std::runtime_error("unknown optimizer '" + text + "'");
  }
}

} // namespace nerva

