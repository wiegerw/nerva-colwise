// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/numpy_eigen.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/eigen.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace nerva::eigen {

namespace py = pybind11;

template <typename Scalar>
bool is_row_major(const pybind11::array_t<Scalar> &x)
{
  return (x.flags() & py::array::c_style) != 0;
}

/// Extracts an Eigen matrix from a NumPy array that is stored in a dictionary.
/// The array will be flattened if needed.
/// @tparam Scalar The number type
/// @tparam MatrixLayout Row major or column major
/// @param data A dictionary
/// @param key The key in the dictionary corresponding to the NumPy array
/// @return The extracted matrix.
template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> extract_matrix(const py::dict& data, const std::string& key)
{
  auto array = data[key.c_str()].cast<py::array_t<Scalar>>();
  auto ndim = array.ndim();

  // Flatten the array if it has more than 2 dimensions
  if (ndim > 2)
  {
    // Flatten all dimensions except for the first one (the batch size)
    auto shape = array.shape();
    long new_dim = std::accumulate(shape + 1, shape + ndim, 1, std::multiplies<long>());
    array = array.reshape({shape[0], new_dim});
  }

  assert(array.ndim() == 2);

  auto shape = array.shape();
  if (is_row_major(array))
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(array.data(), shape[1], shape[0]).transpose();
  }
  else
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(array.data(), shape[1], shape[0]).transpose();
  }
}

template <typename Scalar = scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> extract_column_vector(const py::dict& data, const std::string& key)
{
  const auto& x = data[key.c_str()].cast<py::array_t<Scalar>>();
  return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(x.data(), x.size());
}

template <typename Scalar = scalar>
Eigen::Matrix<Scalar, 1, Eigen::Dynamic> extract_row_vector(const py::dict& data, const std::string& key)
{
  const auto& x = data[key.c_str()].cast<py::array_t<Scalar>>();
  return Eigen::Map<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>(x.data(), x.size());
}

inline
void print_dict(const py::dict& data)
{
  for (const auto& item: data)
  {
    auto key = item.first.cast<std::string>();
    if (key[0] == 'W' || key == "Xtrain" || key == "Xtest")
    {
      print_numpy_matrix(key, eigen::extract_matrix<scalar>(data, key));
    }
    else if (key[0] == 'b')
    {
      print_numpy_vector(key, eigen::extract_column_vector<scalar>(data, key));
    }
    else if (key == "Ttrain" || key == "Ttest")
    {
      print_numpy_vector(key, eigen::extract_column_vector<long>(data, key));
    }
  }
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> to_eigen(const py::array_t<Scalar> &x)
{
  auto shape = x.shape();
  if (is_row_major(x))
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(x.data(), shape[1], shape[0]).transpose();
  }
  else
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(x.data(), shape[1], shape[0]).transpose();
  }
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
std::map<std::string, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>> load_npz(const std::string& filename)
{
  std::cout << "C++: loading data from " << filename << std::endl;
  std::map<std::string, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>> result;
  py::dict data = py::module::import("numpy").attr("load")(filename);

  for (const auto& [key, value]: data)
  {
    result[key.template cast<std::string>()] = to_eigen<Scalar, MatrixLayout>(value.template cast<py::array_t<Scalar>>());
  }
  return result;
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
void save_npz(const std::string& filename, const std::map<std::string, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>>& data)
{
  std::cout << "C++ saving data to " << filename << std::endl;
  py::dict result;
  for (const auto& [key, A]: data)
  {
    std::vector<size_t> shape = {static_cast<unsigned long>(A.rows()), static_cast<unsigned long>(A.cols())};
    if constexpr (MatrixLayout == Eigen::ColMajor)
    {
      py::array_t<Scalar, py::array::f_style> array(shape, A.data());
      result[key.c_str()] = array;
    }
    else
    {
      py::array_t<Scalar, py::array::c_style> array(shape, A.data());
      result[key.c_str()] = array;
    }
  }
  py::module::import("numpy").attr("savez_compressed")(filename, **result);
}

} // namespace nerva::eigen

