// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/datasets/cifar10reader.h
/// \brief add your file description here.

#pragma once

#include "nerva/datasets/dataset.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/utilities/logger.h"
#include "nerva/utilities/text_utility.h"
#include <filesystem>
#include <random>
#include <numeric>

namespace nerva::datasets {

/**
 * Reads a slice of the CIFAR-10 dataset
 * @param filename The name of the file containing the slice
 * @param X The matrix in which the examples are stored
 * @param T The matrix in which the corresponding targets are stored
 * @param start The start row of `X` and `T` where the data is stored
 */
inline
void read_cifar10_slice(const std::string& filename, eigen::matrix& X, eigen::matrix& T, long start)
{
  auto to_scalar = [](std::byte x)
      {
        return static_cast<scalar>(std::to_integer<std::uint8_t>(x));
      };

      auto bytes = read_binary_file(filename);
      if (bytes.size() != 30730000)
      {
        throw std::runtime_error("The size of the file " + filename + " is not equal to 3073000");
      }

      NERVA_LOG(log::verbose) << ".";

      for (long j = 0; j < 10000; j++)
      {
        auto first = bytes.begin() + j * 3073;
        auto class_ = std::to_integer<int>(*first++);
        if (class_ > 9)
        {
          throw std::runtime_error("Invalid class " + std::to_string(class_) + " encountered");
        }
        T(class_, start + j) = 1;
        for (long i = 0; i < 3072; i++)
        {
          // store the data as R1 G1 B1 R2 G2 B2 ...
          auto row = 3 * (i % 1024) + (i / 1024);
          X(row, start + j) = to_scalar(*first++);
        }
      }
    }

inline
datasets::dataset load_cifar10_dataset(const std::string& directory, bool normalize=true)
{
  datasets::dataset result;
  result.Xtrain = eigen::matrix(50000, 3072);
  result.Xtest = eigen::matrix(10000, 3072);
  result.Ttrain = eigen::matrix::Zero(50000, 10);
  result.Ttest = eigen::matrix::Zero(10000, 10);

  auto normalize_data = [](eigen::matrix& X)
  {
    X = X.unaryExpr([](scalar t) { return static_cast<scalar>(2) * ((t / static_cast<scalar>(255)) - static_cast<scalar>(0.5)); });
  };

  namespace fs = std::filesystem;
  for (int i = 0; i < 5; i++)
  {
    auto path = fs::path(directory) / fs::path("cifar-10-batches-bin") / fs::path("data_batch_" + std::to_string(i + 1) + ".bin");
    read_cifar10_slice(path.string(), result.Xtrain, result.Ttrain, i * 10000);
  }
  auto path = fs::path(directory) / fs::path("cifar-10-batches-bin") / fs::path("test_batch.bin");
  read_cifar10_slice(path.string(), result.Xtest, result.Ttest, 0);
  NERVA_LOG(log::verbose) << std::endl;

  if (normalize)
  {
    NERVA_LOG(log::verbose) << "Normalizing data" << std::endl;
    normalize_data(result.Xtrain);
    normalize_data(result.Xtest);
  }

  return result;
}

} // namespace nerva::datasets
