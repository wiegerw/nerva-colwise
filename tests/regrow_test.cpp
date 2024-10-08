// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file regrow_test.cpp
/// \brief Tests for regrowing of a sparse weight matrix.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/neural_networks/grow_dense.h"
#include "nerva/neural_networks/prune_dense.h"
#include "nerva/neural_networks/regrow.h"
#include <algorithm>

using namespace nerva;

//template <typename Matrix>
//void regrow_threshold(Matrix& W, const std::shared_ptr<weight_initializer>& init, std::size_t count, std::mt19937& rng)
//{
//  using Scalar = typename Matrix::Scalar;
//  std::size_t prune_count = prune_magnitude(W, count, std::numeric_limits<Scalar>::quiet_NaN());
//  grow_random(W, init, prune_count, rng);
//}
//
//template <typename Matrix, typename Scalar = scalar>
//void regrow_interval(Matrix& W, const std::shared_ptr<weight_initializer>& init, std::size_t negative_count, std::size_t positive_count, std::mt19937& rng)
//{
//  auto nan = std::numeric_limits<Scalar>::quiet_NaN();
//  std::size_t negative_prune_count = prune_negative_weights(W, negative_count, nan);
//  std::size_t positive_prune_count = prune_positive_weights(W, positive_count, nan);
//  assert(negative_prune_count == negative_count);
//  assert(positive_prune_count == positive_count);
//  grow_random(W, init, negative_prune_count + positive_prune_count, rng);
//}

TEST_CASE("test_nth_element")
{
  std::vector<int> v;
  std::vector<int>::iterator w;

  v = {-2, -4, -1, -2, -5};
  w = v.begin() + 1;
  std::nth_element(v.begin(), w, v.end(), std::greater<>());
  CHECK_EQ(-2, *w);

  v = {-2, -4, -1, -2, -5};
  w = v.begin() + 2;
  std::nth_element(v.begin(), w, v.end(), std::greater<>());
  CHECK_EQ(-2, *w);

  v = {-2, -4, -1, -2, -5};
  w = v.begin() + 3;
  std::nth_element(v.begin(), w, v.end(), std::greater<>());
  CHECK_EQ(-4, *w);

  v = {1, 5, 8, 4, 3, 6, 7, 4};
  w = v.begin() + 2;
  std::nth_element(v.begin(), w, v.end());
  CHECK_EQ(4, *w);

  v = {1, 5, 8, 4, 3, 6, 7, 4};
  w = v.begin() + 3;
  std::nth_element(v.begin(), w, v.end());
  CHECK_EQ(4, *w);

  v = {1, 5, 8, 4, 3, 6, 7, 4};
  w = v.begin() + 4;
  std::nth_element(v.begin(), w, v.end());
  CHECK_EQ(5, *w);
}

void check_prune_weights(const eigen::matrix& A,
                         std::size_t count,
                         const eigen::matrix& expected,
                         scalar expected_threshold,
                         std::size_t expected_threshold_count)
{
  std::cout << "--- check_prune_count ---\n";
  print_numpy_matrix("A", A);

  scalar threshold;
  unsigned int threshold_count;

  // dense matrices
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_nonzero(), less_magnitude());
  CHECK_EQ(expected_threshold, threshold);
  CHECK_EQ(expected_threshold_count, threshold_count);
  auto A_pruned = A;
  auto prune_count = prune_magnitude(A_pruned, count);
  print_numpy_matrix("A_pruned", A_pruned);
  CHECK_EQ(prune_count, count);
  CHECK_EQ(A_pruned, expected);

  // sparse matrices
  auto A1 = mkl::to_csr(A);
  std::tie(threshold, threshold_count) = detail::nth_element(A1, count - 1, accept_nonzero(), less_magnitude());
  CHECK_EQ(expected_threshold, threshold);
  CHECK_EQ(expected_threshold_count, threshold_count);
  auto A1_pruned = A1;
  prune_count = prune_magnitude(A1_pruned, count);
  print_numpy_matrix("A1_pruned", mkl::to_eigen(A1_pruned));
  CHECK_EQ(prune_count, count);
  CHECK_EQ(mkl::to_eigen(A1_pruned), expected);
}

TEST_CASE("test_prune_weights")
{
  const eigen::matrix A{
    {1,  -1, 0, -2, 0},
    {-2, 5,  0, 0,  0},
    {0,  0,  4, 6,  4},
    {-4, 0,  3, 7,  0},
    {0,  8,  0, 0,  -5}
  };

  eigen::matrix A_pruned_expected {
    {0,  0, 0, 0, 0},
    {0,  5, 0, 0, 0},
    {0,  0, 4, 6, 4},
    {-4, 0, 0, 7, 0},
    {0,  8, 0, 0, -5}
  };

  check_prune_weights(A, 5, A_pruned_expected, 3, 1);
}

TEST_CASE("test_regrow")
{
  std::cout << "--- test1 ---\n";

  const eigen::matrix A {
    {1, -1, 0, -2, 0},
    {-2, 5, 0, 0, 0},
    {0, 0, 4, 6, 4},
    {-4, 0, 3, 7, 0},
    {0, 8, 0, 0, -5}
  };

  eigen::matrix A_pruned_expected {
    {0, 0, 0, 0, 0},
    {0, 5, 0, 0, 0},
    {0, 0, 4, 6, 4},
    {-4, 0, 0, 7, 0},
    {0, 8, 0, 0, -5}
  };

  print_numpy_matrix("A", A);

  std::mt19937 rng{std::random_device{}()};
  auto init = std::make_shared<ten_weight_initializer>(rng);

  long k = 4; // consider the 5 smallest nonzero elements
  scalar threshold;
  unsigned int num_copies;
  std::tie(threshold, num_copies) = detail::nth_element(A, k, accept_nonzero(), less_magnitude());
  CHECK_EQ(3, threshold);
  CHECK_EQ(1, num_copies);

  auto A_pruned = A;
  auto accept = [threshold](scalar x) { return x != 0 && std::fabs(x) <= threshold; };
  auto prune_count = detail::prune(A_pruned, accept);
  print_numpy_matrix("A_pruned", A_pruned);
  CHECK_EQ(A_pruned_expected, A_pruned);
  CHECK_EQ(5, prune_count);

  auto A_grow = A_pruned;
  grow_random(A_grow, init, prune_count, rng);
  print_numpy_matrix("A_grow", A_grow);

  long m = A.rows();
  long n = A.cols();

  // Check if the large weights are unchanged
  for (long i = 0; i < m; i++)
  {
    for (long j = 0; j < n; j++)
    {
      if (std::fabs(A(i, j)) > threshold)
      {
        CHECK_EQ(A(i, j), A_grow(i, j));
      }
    }
  }
  CHECK_EQ((A_grow.array() == 10).count(), prune_count);
  CHECK_EQ((A_grow.array() == 0).count(), (A.array() == 0).count());

  std::size_t zero_count = (A.array() == 0).count();
  std::size_t nonzero_count = (A.array() != 0).count();
  for (std::size_t count = 1; count <= nonzero_count; count++)
  {
    if (count > zero_count)
    {
      continue;
    }
    std::cout << "=== regrow_threshold count = " << count << " ===" << std::endl;
    auto B = A;
    print_numpy_matrix("A", A);
    regrow_magnitude(B, init, count, rng);
    print_numpy_matrix("B", B);
    CHECK_EQ((B.array() == 10).count(), count);
    CHECK_EQ((B.array() == 0).count(), (A.array() == 0).count());
  }
}

TEST_CASE("test2")
{
  std::cout << "--- test2 ---\n";

  const eigen::matrix A {
    { 1, -1, 0, -2,  0},
    {-2,  5, 0,  0,  0},
    { 0,  0, 4,  6,  4},
    {-4,  0, 3,  7,  0},
    { 0,  8, 0,  0, -5}
  };

  eigen::matrix A_pruned_expected {
    {0, 0, 0, 0, 0},
    {0, 5, 0, 0, 0},
    {0, 0, 0, 6, 0},
    {-4, 0, 0, 7, 0},
    {0, 8, 0, 0, -5}
  };

  print_numpy_matrix("A", A);

  auto A_pruned = A;
  float negative_threshold = -2;
  float positive_threshold = 4;
  auto accept = [negative_threshold, positive_threshold](scalar x)
  {
    return (negative_threshold <= x && x < 0) || (0 < x && x <= positive_threshold);
  };
  auto prune_count = detail::prune(A_pruned, accept);
  print_numpy_matrix("A_pruned_expected", A_pruned_expected);
  CHECK_EQ(A_pruned_expected, A_pruned);
  CHECK_EQ(7, prune_count);

  std::mt19937 rng{std::random_device{}()};

  std::size_t max_negative_count = (A.array() < 0).count();
  std::size_t max_positive_count = (A.array() > 0).count();
  std::size_t zero_count = (A.array() == 0).count();
  for (std::size_t negative_count = 1; negative_count <= max_negative_count; negative_count++)
  {
    for (std::size_t positive_count = 1; positive_count <= max_positive_count; positive_count++)
    {
      if (negative_count + positive_count > zero_count)
      {
        continue;
      }
      std::cout << "=== regrow_interval negative_count = " << negative_count << " positive_count = " << positive_count << " ===" << std::endl;
      print_numpy_matrix("A", A);

      // dense matrices
      std::cout << "--- dense ---" << std::endl;
      auto B = A;
      auto init = std::make_shared<ten_weight_initializer>(rng);
      regrow_interval(B, init, negative_count, positive_count, rng);
      print_numpy_matrix("B", B);
      CHECK_EQ((B.array() == 10).count(), negative_count + positive_count);
      CHECK_EQ((B.array() == 0).count(), (A.array() == 0).count());

      // sparse matrices
      std::cout << "--- sparse ---" << std::endl;
      auto B1 = mkl::to_csr(A);
      regrow_interval<mkl::sparse_matrix_csr<scalar>, scalar, false>(B1, init, negative_count, positive_count, rng);
      print_numpy_matrix("B1", mkl::to_eigen(B1));
      auto B2 = mkl::to_eigen(B1);
      CHECK_EQ((B2.array() == 10).count(), negative_count + positive_count);
      CHECK_EQ((B2.array() == 0).count(), (A.array() == 0).count());
    }
  }
}
