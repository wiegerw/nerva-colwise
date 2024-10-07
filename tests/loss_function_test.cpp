// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file loss_function_test.cpp
/// \brief Tests for loss functions.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/loss_functions.h"
#include <iostream>
#include <type_traits>

using namespace nerva;

template <typename LossFunction>
void test_loss(const std::string& name, LossFunction loss, scalar expected, const eigen::matrix& Y, const eigen::matrix& T)
{
  std::cout << "\n=== test_loss " << name << " ===" << std::endl;
  scalar L = loss(Y, T);
  scalar epsilon = std::is_same<scalar, double>::value ? scalar(0.000001) : scalar(0.001);
  CHECK(expected == doctest::Approx(L).epsilon(epsilon));
}

//--- begin generated code ---//
TEST_CASE("test_loss1")
{
  eigen::matrix Y {
    {0.23759169, 0.43770149, 0.20141643, 0.35686849, 0.48552814},
    {0.42272727, 0.28115265, 0.45190243, 0.17944701, 0.26116029},
    {0.33968104, 0.28114586, 0.34668113, 0.46368450, 0.25331157},
  };

  eigen::matrix T {
    {1.00000000, 1.00000000, 0.00000000, 0.00000000, 1.00000000},
    {0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 2.6550281475767563, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.106889686512423, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 4.548777728936653, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 4.548777728936653, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.539463487358204, Y, T);
}

TEST_CASE("test_loss2")
{
  eigen::matrix Y {
    {0.24335898, 0.21134093, 0.24788846, 0.40312318, 0.43329234},
    {0.40191852, 0.53408849, 0.42021140, 0.24051313, 0.34433141},
    {0.35472250, 0.25457058, 0.33190014, 0.35636369, 0.22237625},
  };

  eigen::matrix T {
    {1.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000},
    {0.00000000, 0.00000000, 1.00000000, 1.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.6087104890568256, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.5889911807479065, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 5.90971538007391, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 5.909715380073911, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.7376380548462254, Y, T);
}

TEST_CASE("test_loss3")
{
  eigen::matrix Y {
    {0.23774258, 0.29687977, 0.43420442, 0.28599538, 0.20014798},
    {0.42741216, 0.43115409, 0.22655227, 0.35224692, 0.43868708},
    {0.33484526, 0.27196615, 0.33924331, 0.36175770, 0.36116494},
  };

  eigen::matrix T {
    {0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000},
    {1.00000000, 1.00000000, 1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.289394384977318, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.441938177932827, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 5.44627595910772, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 5.44627595910772, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.678127590042374, Y, T);
}

TEST_CASE("test_loss4")
{
  eigen::matrix Y {
    {0.26787616, 0.26073833, 0.31560020, 0.37231605, 0.49308039},
    {0.35447135, 0.45527664, 0.41003295, 0.17984538, 0.27786731},
    {0.37765249, 0.28398503, 0.27436685, 0.44783858, 0.22905230},
  };

  eigen::matrix T {
    {0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000, 1.00000000, 1.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.521376994732803, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.548304798627446, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 5.726367921857207, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 5.726367921857208, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.7197402348335156, Y, T);
}

TEST_CASE("test_loss5")
{
  eigen::matrix Y {
    {0.29207765, 0.38987005, 0.24441444, 0.38397493, 0.29902507},
    {0.40236525, 0.36536339, 0.32191037, 0.35636403, 0.25018760},
    {0.30555710, 0.24476656, 0.43367519, 0.25966104, 0.45078733},
  };

  eigen::matrix T {
    {0.00000000, 1.00000000, 1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.2404999669186503, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.4240756991825645, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 5.365012502539291, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 5.365012502539292, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.6711745146065176, Y, T);
}


//--- end generated code ---//
