// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/sgd_options.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/eigen.h"
#include "nerva/utilities/print.h"
#include <iostream>
#include <random>
#include <string>

namespace nerva {

// options for SGD algorithms
struct sgd_options
{
  std::size_t epochs = 100;
  long batch_size = 1;
  bool shuffle = true;
  scalar regrow_rate = 0.0;
  bool regrow_separate_positive_negative = false; // apply the regrow rate to positive and negative values separately
  bool statistics = true;
  bool debug = false;
  scalar gradient_step = 0;  // if gradient_step > 0 then gradient checks will be done
  scalar clip = 0; // threshold for values that are clipped to 0

  void info() const;
};

inline
std::ostream& operator<<(std::ostream& out, const sgd_options& options)
{
  out << "epochs = " << options.epochs << std::endl;
  out << "batch size = " << options.batch_size << std::endl;
  out << "shuffle = " << std::boolalpha << options.shuffle << std::endl;
  out << "clip = " << options.clip << std::endl;
  if (options.regrow_rate > 0)
  {
    out << "regrow rate = " << options.regrow_rate << std::endl;
    out << "regrow separate positive/negative weights = " << options.regrow_separate_positive_negative << std::endl;
  }
  out << "statistics = " << std::boolalpha << options.statistics << std::endl;
  out << "debug = " << std::boolalpha << options.debug << std::endl;
  return out;
}

inline
void sgd_options::info() const
{
  std::cout << *this;
}

} // namespace nerva

