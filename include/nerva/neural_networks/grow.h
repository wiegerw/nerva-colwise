// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/grow.h
/// \brief Grow functions for sparse MKL matrices.

#pragma once

#include <algorithm>
#include "fmt/format.h"
#include <random>
#include <stdexcept>
#include <vector>
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/functions.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/neural_networks/settings.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/algorithms.h"

namespace nerva {

/// Assigns a new value to \a count entries of \a A. The entries are outside the support of \a A. The new values generated using the function \a f.
/// \tparam Scalar A number type (float of double)
/// \param A A CSR matrix
/// \param init A weight initializer. The values of added elements will be initialized using \a init.
/// \param count The number of entries that will be added
/// \param rng A random number generator
template <typename Scalar = scalar>
void grow_random(mkl::sparse_matrix_csr<Scalar>& A, const std::shared_ptr<weight_initializer>& init, std::size_t count, std::mt19937& rng)
{
  std::size_t N = A.rows() * A.cols();
  if (A.values().size() + count > N)
  {
    throw std::runtime_error("cannot grow the matrix with " + std::to_string(count) + " elements");
  }

  // Select k random new positions outside the support of A
  std::vector<std::size_t> new_positions = reservoir_sample(count, N - A.values().size(), rng);
  std::sort(new_positions.begin(), new_positions.end());  // TODO: this may become expensive for very large matrices
  auto ni = new_positions.begin();

  std::size_t traversed_elements_count = 0;

  mkl::csr_matrix_builder<Scalar> builder(A.rows(), A.cols(), A.values().size());

  // fills the result matrix until position k
  auto fill_until_index = [&](std::size_t k)
  {
    while (ni != new_positions.end() && *ni + traversed_elements_count < k)
    {
      std::size_t k1 = *ni + traversed_elements_count;
      if (k1 < k)
      {
        std::size_t i1 = k1 / A.cols();
        std::size_t j1 = k1 % A.cols();
        auto value1 = (*init)();
        builder.add_element(i1, j1, value1);
      }
      ++ni;
    }
  };

  mkl::traverse_elements(A, [&](long i, long j, Scalar value)
  {
    std::size_t k = i * A.cols() + j;
    fill_until_index(k);
    if (!std::isnan(value))
    {
      builder.add_element(i, j, value);
    }
    traversed_elements_count++;
  });

  A = builder.result();  // TODO: avoid unnecessary copies
}

// tag::doc[]
struct grow_function
{
  /// Adds `count` elements to the support of matrix `W`
  virtual void operator()(mkl::sparse_matrix_csr<scalar>& W, std::size_t count) const = 0;

  virtual ~grow_function() = default;
};
// end::doc[]

struct grow_random_function: public grow_function
{
  weight_initialization init;
  std::mt19937& rng;

  grow_random_function(weight_initialization init_, std::mt19937& rng_)
    : init(init_), rng(rng_)
  {}

  void operator()(mkl::sparse_matrix_csr<scalar>& W, std::size_t count) const override
  {
    grow_random(W, make_weight_initializer(init, W, rng), count, rng);
  }
};

inline
std::shared_ptr<grow_function> parse_grow_function(const std::string& strategy, weight_initialization init, std::mt19937& rng)
{
  if (strategy == "Random")
  {
    return std::make_shared<grow_random_function>(init, rng);
  }
  throw std::runtime_error(fmt::format("unknown grow strategy {}", strategy));
}

} // namespace nerva

