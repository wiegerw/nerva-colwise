// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mkl.cpp
/// \brief add your file description here.

#include <lyra/lyra.hpp>
#include "omp.h"
#include "nerva/utilities/command_line_tool.h"
#include "nerva/utilities/stopwatch.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "fmt/format.h"
#include <iostream>
#include <random>
#include <sstream>

using namespace nerva;

enum matrix_layout
{
  column_major = 0,
  row_major = 1
};

inline
std::string layout_string(int layout)
{
  return layout == column_major ? "column-major" : "row-major";
}

inline
float parse_float(const std::string& text)
{
  // Use stold instead of stof to avoid out_of_range errors
  return static_cast<float>(std::stold(text, nullptr));
}

inline
std::vector<float> parse_comma_separated_floats(const std::string& text)
{
  std::vector<float> result;
  for (const std::string& word: utilities::regex_split(text, ","))
  {
    result.push_back(parse_float(word));
  }
  return result;
}

inline
std::string layout_char(int layout)
{
  return layout == column_major ? "C" : "R";
}

template <typename Scalar>
std::string matrix_parameters(const std::string& name, const mkl::sparse_matrix_csr<Scalar>& A)
{
  return fmt::format("density({})={}", name, A.density());
}

template <typename Derived>
std::string matrix_parameters(const std::string& name, const Eigen::MatrixBase<Derived>& A)
{
  constexpr int MatrixLayout = Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor;
  return fmt::format("{}={}", name, layout_string(MatrixLayout));
}

template <typename MatrixA, typename MatrixB, typename MatrixC>
std::string pp(const MatrixA& A, const MatrixB& B, const MatrixC& C)
{
  std::ostringstream out;
  out << matrix_parameters("A", A) << ", " << matrix_parameters("B", B) << ", " << matrix_parameters("C", C);
  return out.str();
}

// A = B * C
template <int MatrixLayoutA = column_major, int MatrixLayoutB = column_major, int MatrixLayoutC = column_major>
void test_ddd_product(long m, long k, long n, int repetitions)
{
  std::cout << "--- testing A = B * C (ddd_product) ---" << std::endl;
  std::cout << fmt::format("A = {:2d}x{:2d} dense  layout={}\n", m, n, layout_string(MatrixLayoutA));
  std::cout << fmt::format("B = {:2d}x{:2d} dense  layout={}\n", m, k, layout_string(MatrixLayoutB));
  std::cout << fmt::format("C = {:2d}x{:2d} dense  layout={}\n\n", k, n, layout_string(MatrixLayoutC));

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  float a = -10;
  float b = 10;

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutA> A(m, n);
  eigen::fill_matrix_random(A, 1.0, a, b, rng);
  mkl::sparse_matrix_csr<float> A1 = mkl::to_csr<float>(A);

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutB> B(m, k);
  eigen::fill_matrix_random(B, 1.0, a, b, rng);

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutC> C(k, n);
  eigen::fill_matrix_random(C, 1.0, a, b, rng);

  utilities::stopwatch watch;

  // dense product
  for (auto i = 0; i < repetitions; ++i)
  {
    watch.reset();
    A = B * C;
    auto seconds = watch.seconds();
    std::cout << fmt::format("{:8.5f}s ddd_product {}\n", seconds, pp(A, B, C));
  }

  std::cout << std::endl;
}

// A = B * C
template <int MatrixLayoutB = column_major, int MatrixLayoutC = column_major>
void test_sdd_product(long m, long k, long n, const std::vector<float>& densities, int repetitions)
{
  std::cout << "--- testing A = B * C (sdd_product) ---" << std::endl;
  std::cout << fmt::format("A = {:2d}x{:2d} sparse\n", m, n);
  std::cout << fmt::format("B = {:2d}x{:2d} dense  layout={}\n", m, k, layout_string(MatrixLayoutB));
  std::cout << fmt::format("C = {:2d}x{:2d} dense  layout={}\n\n", k, n, layout_string(MatrixLayoutC));

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  for (float density: densities)
  {
    std::cout << fmt::format("density(A) = {}\n", density);

    float a = -10;
    float b = 10;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, column_major> A(m, n);
    eigen::fill_matrix_random(A, density, a, b, rng);
    mkl::sparse_matrix_csr<float> A1 = mkl::to_csr<float>(A);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutB> B(m, k);
    eigen::fill_matrix_random(B, 0, a, b, rng);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutC> C(k, n);
    eigen::fill_matrix_random(C, 0, a, b, rng);

    // dense product
    utilities::stopwatch watch;
    for (auto i = 0; i < repetitions; ++i)
    {
      watch.reset();
      A = B * C;
      auto seconds = watch.seconds();
      std::cout << fmt::format("{:8.5f}s ddd_product {}\n", seconds, pp(A, B, C));
    }

    // sdd_product_batch
    for (long batch_size: {5, 10, 100})
    {
      for (auto i = 0; i < repetitions; ++i)
      {
        watch.reset();
        mkl::sdd_product_batch(A1, B, C, batch_size);
        auto seconds = watch.seconds();
        std::cout << fmt::format("{:8.5f}s sdd_product(batchsize={}, {})\n", seconds, batch_size, pp(A1, B, C));
      }
    }

    if (density * m <= 2000)
    {
      for (auto i = 0; i < repetitions; ++i)
      {
        watch.reset();
        mkl::sdd_product_forloop_eigen(A1, B, C);
        auto seconds = watch.seconds();
        std::cout << fmt::format("{:8.5f}s sdd_product_forloop_eigen({})\n", seconds, pp(A1, B, C));
      }

      for (auto i = 0; i < repetitions; ++i)
      {
        watch.reset();
        mkl::sdd_product_forloop_mkl(A1, B, C);
        auto seconds = watch.seconds();
        std::cout << fmt::format("{:8.5f}s sdd_product_forloop_mkl({})\n", seconds, pp(A1, B, C));
      }
    }

    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// A = B * C
template <int MatrixLayoutA = column_major, int MatrixLayoutC = column_major>
void test_dsd_product(long m, long k, long n, const std::vector<float>& densities, int repetitions)
{
  std::cout << "--- testing A = B * C (dsd_product) ---" << std::endl;
  std::cout << fmt::format("A = {:2d}x{:2d} dense  layout={}\n", m, k, layout_string(MatrixLayoutA));
  std::cout << fmt::format("B = {:2d}x{:2d} sparse\n", m, n);
  std::cout << fmt::format("C = {:2d}x{:2d} dense  layout={}\n\n", k, n, layout_string(MatrixLayoutC));

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  for (float density: densities)
  {
    std::cout << fmt::format("density(B) = {}\n", density);

    float a = -10;
    float b = 10;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutA> A(m, n);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, column_major> B(m, k);
    eigen::fill_matrix_random(B, density, a, b, rng);
    mkl::sparse_matrix_csr<float> B1 = mkl::to_csr<float>(B);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutC> C(k, n);
    eigen::fill_matrix_random(C, float(0), a, b, rng);

    utilities::stopwatch watch;
    for (auto i = 0; i < repetitions; ++i)
    {
      watch.reset();
      A = B * C;
      auto seconds = watch.seconds();
      std::cout << fmt::format("{:8.5f}s ddd_product({})\n", seconds, pp(A, B, C));
    }

    for (auto i = 0; i < repetitions; ++i)
    {
      watch.reset();
      mkl::dsd_product(A, B1, C);
      auto seconds = watch.seconds();
      std::cout << fmt::format("{:8.5f}s dsd_product({})\n\n", seconds, pp(A, B1, C));
    }
  }
  std::cout << std::endl;
}

// A = B^T * C
template <int MatrixLayoutA = column_major, int MatrixLayoutC = column_major>
void test_dsd_transpose_product(long m, long k, long n, const std::vector<float>& densities, int repetitions)
{
  std::cout << "--- testing A = B^T * C (dsd_product) ---" << std::endl;
  std::cout << fmt::format("A = {:2d}x{:2d} dense  layout={}\n", m, k, layout_string(MatrixLayoutA));
  std::cout << fmt::format("B = {:2d}x{:2d} sparse\n", m, n);
  std::cout << fmt::format("C = {:2d}x{:2d} dense  layout={}\n\n", k, n, layout_string(MatrixLayoutC));

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  for (float density: densities)
  {
    std::cout << fmt::format("density = {}\n", density);

    float a = -10;
    float b = 10;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutA> A(m, n);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, column_major> B(k, m);
    eigen::fill_matrix_random(B, density, a, b, rng);
    mkl::sparse_matrix_csr<float> B1 = mkl::to_csr<float>(B);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutC> C(k, n);
    eigen::fill_matrix_random(C, 0, a, b, rng);

    utilities::stopwatch watch;

    for (auto i = 0; i < repetitions; ++i)
    {
      watch.reset();
      A = B.transpose() * C;
      auto seconds = watch.seconds();
      std::cout << fmt::format("{:8.5f}s ddd_product({})\n", seconds, pp(A, B, C));
    }

    for (auto i = 0; i < repetitions; ++i)
    {
      watch.reset();
      bool B1_transposed = true;
      mkl::dsd_product(A, B1, C, float(0), float(1), B1_transposed);
      auto seconds = watch.seconds();
      std::cout << fmt::format("{:8.5f}s dsd_product({})\n\n", seconds, pp(A, B1, C));
    }
  }
  std::cout << std::endl;
}

class tool: public command_line_tool
{
  protected:
    int m = 100;
    int k = 100;
    int n = 100;
    float alpha = 1.0;
    float beta = 0.0;
    std::string algorithm;
    int threads = 1;
    int repetitions = 1;
    std::string densities_text = "1.0, 0.5, 0.1, 0.05, 0.01, 0.001";

    void add_options(lyra::cli& cli) override
    {
      cli |= lyra::opt(algorithm, "algorithm")["--algorithm"]["-a"]("The algorithm (sdd, dsd, dsdt)");
      cli |= lyra::opt(m, "m")["--arows"]["-m"]("The number of rows of matrix A");
      cli |= lyra::opt(k, "k")["--acols"]["-k"]("The number of columns of matrix A");
      cli |= lyra::opt(n, "n")["--brows"]["-n"]("The number of rows of matrix B");
      cli |= lyra::opt(threads, "value")["--threads"]("The number of threads.");
      cli |= lyra::opt(densities_text, "value")["--densities"]("The densities that are tested.");
      cli |= lyra::opt(repetitions, "value")["--repetitions"]("The number of repetitions for each test.");
    }

    std::string description() const override
    {
      return "Test the MKL library";
    }

    bool run() override
    {
      if (threads >= 1 && threads <= 16)
      {
        omp_set_num_threads(threads);
      }

      std::vector<float> densities = parse_comma_separated_floats(densities_text);
      if (algorithm == "sdd")
      {
        test_sdd_product<column_major, column_major>(m, k, n, densities, repetitions);
        test_sdd_product<column_major, row_major>(m, k, n, densities, repetitions);
        test_sdd_product<row_major, column_major>(m, k, n, densities, repetitions);
        test_sdd_product<row_major, row_major>(m, k, n, densities, repetitions);
      }
      else if (algorithm == "dsd")
      {
        test_dsd_product<column_major, column_major>(m, k, n, densities, repetitions);
        test_dsd_product<row_major, row_major>(m, k, n, densities, repetitions);
      }
      else if (algorithm == "dsdt")
      {
        test_dsd_transpose_product<column_major, column_major>(m, k, n, densities, repetitions);
        test_dsd_transpose_product<row_major, row_major>(m, k, n, densities, repetitions);
      }
      else if (algorithm == "ddd")
      {
        test_ddd_product<column_major, column_major, column_major>(m, k, n, repetitions);
        test_ddd_product<column_major, column_major, row_major>(m, k, n, repetitions);
        test_ddd_product<column_major, row_major, column_major>(m, k, n, repetitions);
        test_ddd_product<column_major, row_major, row_major>(m, k, n, repetitions);
        test_ddd_product<row_major, column_major, column_major>(m, k, n, repetitions);
        test_ddd_product<row_major, column_major, row_major>(m, k, n, repetitions);
        test_ddd_product<row_major, row_major, column_major>(m, k, n, repetitions);
        test_ddd_product<row_major, row_major, row_major>(m, k, n, repetitions);
      }
      return true;
    }
};

int main(int argc, const char** argv)
{
  return tool().execute(argc, argv);
}
