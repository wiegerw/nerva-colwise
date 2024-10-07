// Copyright: Wieger Wesselink 2022 - 2024
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file multilayer_perceptron_test.cpp
/// \brief Tests for multilayer perceptrons.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/multilayer_perceptron.h"
#include "nerva/neural_networks/loss_functions.h"
#include "nerva/utilities/string_utility.h"
#include <iostream>

using namespace nerva;

inline
void check_equal_matrices(const std::string& name1, const eigen::matrix& X1, const std::string& name2, const eigen::matrix& X2, scalar epsilon = 1e-5)
{
  scalar error = (X2 - X1).squaredNorm();
  if (error > epsilon)
  {
    CHECK_LE(error, epsilon);
    print_cpp_matrix(name1, X1);
    print_cpp_matrix(name2, X2);
  }
}

// tag::doc[]
void construct_mlp(multilayer_perceptron& M,
                   const eigen::matrix& W1,
                   const eigen::matrix& b1,
                   const eigen::matrix& W2,
                   const eigen::matrix& b2,
                   const eigen::matrix& W3,
                   const eigen::matrix& b3,
                   const std::vector<long>& sizes,
                   long N
                  )
{
  long batch_size = N;

  auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(sizes[0], sizes[1], batch_size);
  M.layers.push_back(layer1);
  auto optimizer_W1 = std::make_shared<gradient_descent_optimizer<eigen::matrix>>(layer1->W, layer1->DW);
  auto optimizer_b1 = std::make_shared<gradient_descent_optimizer<eigen::matrix>>(layer1->b, layer1->Db);
  layer1->optimizer = make_composite_optimizer(optimizer_W1, optimizer_b1);
  layer1->W = W1;
  layer1->b = b1;

  auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(sizes[1], sizes[2], batch_size);
  M.layers.push_back(layer2);
  auto optimizer_W2 = std::make_shared<gradient_descent_optimizer<eigen::matrix>>(layer2->W, layer2->DW);
  auto optimizer_b2 = std::make_shared<gradient_descent_optimizer<eigen::matrix>>(layer2->b, layer2->Db);
  layer2->optimizer = make_composite_optimizer(optimizer_W2, optimizer_b2);
  layer2->W = W2;
  layer2->b = b2;

  auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(sizes[2], sizes[3], batch_size);
  M.layers.push_back(layer3);
  auto optimizer_W3 = std::make_shared<gradient_descent_optimizer<eigen::matrix>>(layer3->W, layer3->DW);
  auto optimizer_b3 = std::make_shared<gradient_descent_optimizer<eigen::matrix>>(layer3->b, layer3->Db);
  layer3->optimizer = make_composite_optimizer(optimizer_W3, optimizer_b3);
  layer3->W = W3;
  layer3->b = b3;
}
// end::doc[]

void test_mlp_execution(const eigen::matrix& X,
                        const eigen::matrix& T,
                        const eigen::matrix& W1,
                        const eigen::matrix& b1,
                        const eigen::matrix& W2,
                        const eigen::matrix& b2,
                        const eigen::matrix& W3,
                        const eigen::matrix& b3,
                        const eigen::matrix& Y1,
                        const eigen::matrix& DY1,
                        const eigen::matrix& Y2,
                        const eigen::matrix& DY2,
                        scalar lr,
                        const std::vector<long>& sizes,
                        long N
                       )
{
  multilayer_perceptron M;
  long K = sizes.back(); // the output size of the MLP
  construct_mlp(M, W1, b1, W2, b2, W3, b3, sizes, N);
  // M.info("M");

  eigen::matrix Y(K, N);
  eigen::matrix DY(K, N);

  softmax_cross_entropy_loss loss;

  M.feedforward(X, Y);
  DY = loss.gradient(Y, T) / N; // take the average of the gradients in the batch

  check_equal_matrices("Y", Y, "Y1", Y1);
  check_equal_matrices("DY", DY, "DY1", DY1);

  M.backpropagate(Y, DY);
  M.optimize(lr);
  M.feedforward(X, Y);
  M.backpropagate(Y, DY);

  check_equal_matrices("Y", Y, "Y2", Y2);
  check_equal_matrices("DY", DY, "DY2", DY2);
}

//--- begin generated code ---//
TEST_CASE("test_mlp1")
{
  eigen::matrix X {
    {0.37454012, 0.73199391, 0.15601864, 0.05808361, 0.60111499},
    {0.95071429, 0.59865850, 0.15599452, 0.86617613, 0.70807260},
  };

  eigen::matrix T {
    {0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000},
    {1.00000000, 0.00000000, 1.00000000, 1.00000000, 1.00000000},
    {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000},
  };

  eigen::matrix W1 {
    {-0.65989369, 0.52053243},
    {-0.47691292, -0.00902988},
    {0.21051954, -0.40127230},
    {0.12944663, -0.24993435},
    {-0.09881101, 0.16878708},
    {-0.15896510, -0.28802213},
  };

  eigen::matrix b1 {
    {-0.34465417},
    {0.41825107},
    {-0.67499119},
    {-0.03056043},
    {0.62188822},
    {0.10260026},
  };

  eigen::matrix W2 {
    {0.32011512, 0.06845723, -0.37301734, 0.21450770, -0.00864073, -0.26720902},
    {-0.06712551, 0.23647705, -0.09944443, -0.14873986, -0.09178439, 0.35508740},
    {0.05559298, -0.15441763, -0.29207242, 0.04468316, -0.29954737, 0.33758661},
    {0.01792780, -0.38909590, -0.28889173, 0.34666714, -0.33454186, 0.13274866},
  };

  eigen::matrix b2 {
    {-0.13568099},
    {-0.37000555},
    {-0.17376846},
    {0.16057922},
  };

  eigen::matrix W3 {
    {-0.38102198, 0.44200462, -0.45120806, -0.44877696},
    {-0.30982375, -0.11014897, 0.10018528, -0.35674077},
    {-0.45877683, 0.14521885, -0.37314588, 0.26365072},
  };

  eigen::matrix b3 {
    {-0.40899998},
    {0.49299562},
    {-0.48446542},
  };

  eigen::matrix Y1 {
    {-0.40899998, -0.40899998, -0.40899998, -0.40899998, -0.40899998},
    {0.49299562, 0.49299562, 0.49299562, 0.49299562, 0.49299562},
    {-0.48446542, -0.48446542, -0.48446542, -0.48446542, -0.48446542},
  };

  eigen::matrix DY1 {
    {0.04553912, -0.15446088, 0.04553912, 0.04553912, 0.04553912},
    {-0.08776809, 0.11223191, -0.08776809, -0.08776809, -0.08776809},
    {0.04222896, 0.04222896, 0.04222896, 0.04222896, 0.04222896},
  };

  eigen::matrix Y2 {
    {-0.40927693, -0.40927693, -0.40927693, -0.40927693, -0.40927693},
    {0.49538404, 0.49538404, 0.49538404, 0.49538404, 0.49538404},
    {-0.48657686, -0.48657686, -0.48657686, -0.48657686, -0.48657686},
  };

  eigen::matrix DY2 {
    {0.04548860, -0.15451141, 0.04548860, 0.04548860, 0.04548860},
    {-0.08759340, 0.11240660, -0.08759340, -0.08759340, -0.08759340},
    {0.04210481, 0.04210481, 0.04210481, 0.04210481, 0.04210481},
  };

  scalar lr = 0.01;
  std::vector<long> sizes = {2, 6, 4, 3};
  long N = 5;
  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, N);
}

TEST_CASE("test_mlp2")
{
  eigen::matrix X {
    {0.00077877, 0.61165315, 0.52477467, 0.97375554},
    {0.99221158, 0.00706631, 0.39986098, 0.23277134},
    {0.61748153, 0.02306242, 0.04666566, 0.09060644},
  };

  eigen::matrix T {
    {0.00000000, 0.00000000, 0.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000, 1.00000000},
    {0.00000000, 0.00000000, 1.00000000, 0.00000000},
  };

  eigen::matrix W1 {
    {0.47408727, 0.33289635, 0.13210465},
    {0.36449283, 0.15735874, 0.22531973},
    {-0.04749243, 0.42560893, 0.21700588},
    {-0.18284874, 0.45094180, 0.03888704},
    {0.16622119, -0.49349993, -0.00275543},
  };

  eigen::matrix b1 {
    {-0.01218865},
    {-0.39734539},
    {0.28914478},
    {0.48664889},
    {0.25556901},
  };

  eigen::matrix W2 {
    {0.06385345, 0.34590760, -0.02349866, -0.13206814, 0.33753499},
    {-0.02190809, -0.23644534, -0.24977244, -0.00085859, 0.39726147},
  };

  eigen::matrix b2 {
    {0.26836538},
    {0.23018752},
  };

  eigen::matrix W3 {
    {-0.20633942, -0.53669602},
    {0.07787751, 0.16051063},
    {-0.16741470, -0.58291954},
    {-0.25485271, -0.60237277},
  };

  eigen::matrix b3 {
    {0.57716340},
    {-0.63520449},
    {-0.42771667},
    {0.47668228},
  };

  eigen::matrix Y1 {
    {0.54165399, 0.34554443, 0.43422574, 0.37457663},
    {-0.62219948, -0.56028354, -0.58832800, -0.56901789},
    {-0.45792001, -0.65946937, -0.56848907, -0.62811804},
    {0.43339550, 0.20858935, 0.31031334, 0.24124867},
  };

  eigen::matrix DY1 {
    {0.09698522, 0.09461612, 0.09578490, 0.09501030},
    {-0.21971340, 0.03824451, 0.03445146, 0.03698050},
    {0.03569407, -0.21536675, 0.03514176, -0.21514171},
    {0.08703411, 0.08250615, -0.16537812, 0.08315092},
  };

  eigen::matrix Y2 {
    {0.53582072, 0.34041077, 0.42896456, 0.36938053},
    {-0.62052947, -0.55903196, -0.58694577, -0.56772107},
    {-0.45602480, -0.65583301, -0.56545067, -0.62464690},
    {0.43037197, 0.20671871, 0.30813509, 0.23927110},
  };

  eigen::matrix DY2 {
    {0.09669518, 0.09430690, 0.09548554, 0.09470256},
    {-0.21957655, 0.03836370, 0.03457271, 0.03710084},
    {0.03586343, -0.21517587, 0.03532390, -0.21495217},
    {0.08701798, 0.08250529, -0.16538215, 0.08314878},
  };

  scalar lr = 0.01;
  std::vector<long> sizes = {3, 5, 2, 4};
  long N = 4;
  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, N);
}

TEST_CASE("test_mlp3")
{
  eigen::matrix X {
    {0.98323089, 0.94220173, 0.68326354, 0.75536144, 0.44975412, 0.52083427, 0.96525532, 0.42340147},
    {0.46676290, 0.56328821, 0.60999668, 0.42515588, 0.39515024, 0.96117204, 0.60703427, 0.39488152},
    {0.85994041, 0.38541651, 0.83319491, 0.20794167, 0.92665887, 0.84453386, 0.27599919, 0.29348817},
    {0.68030757, 0.01596625, 0.17336465, 0.56770033, 0.72727197, 0.74732012, 0.29627350, 0.01407982},
    {0.45049927, 0.23089382, 0.39106062, 0.03131329, 0.32654077, 0.53969210, 0.16526695, 0.19884241},
    {0.01326496, 0.24102546, 0.18223609, 0.84228480, 0.57044399, 0.58675116, 0.01563641, 0.71134198},
  };

  eigen::matrix T {
    {0.00000000, 1.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000},
    {0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000, 1.00000000, 0.00000000},
    {1.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000},
  };

  eigen::matrix W1 {
    {-0.16104594, -0.38876736, 0.40728948, -0.05779317, -0.01024158, 0.07066248},
    {-0.30683780, 0.32566008, -0.17712107, 0.37232774, -0.05356260, 0.15467377},
  };

  eigen::matrix b1 {
    {-0.12809493},
    {-0.13014677},
  };

  eigen::matrix W2 {
    {0.30616304, 0.06215134},
    {-0.39947617, -0.54411167},
  };

  eigen::matrix b2 {
    {0.69337934},
    {-0.05118161},
  };

  eigen::matrix W3 {
    {-0.05247797, 0.14810133},
    {-0.66377264, -0.34096682},
    {-0.32695812, 0.11977519},
  };

  eigen::matrix b3 {
    {0.02374132},
    {0.58407152},
    {-0.07554401},
  };

  eigen::matrix Y1 {
    {-0.01264582, -0.01264582, -0.01264582, -0.01290569, -0.01306202, -0.01334239, -0.01264582, -0.01264582},
    {0.12382528, 0.12382528, 0.12382528, 0.12053823, 0.11856094, 0.11501467, 0.12382528, 0.12382528},
    {-0.30225003, -0.30225003, -0.30225003, -0.30386913, -0.30484310, -0.30658990, -0.30225003, -0.30225003},
  };

  eigen::matrix DY1 {
    {0.04318115, -0.08181885, 0.04318115, -0.08175190, 0.04328839, 0.04336067, 0.04318115, -0.08181885},
    {0.04949518, 0.04949518, 0.04949518, 0.04942208, -0.07562188, 0.04929930, -0.07550482, 0.04949518},
    {-0.09267633, 0.03232368, -0.09267633, 0.03232982, 0.03233349, -0.09265998, 0.03232368, 0.03232368},
  };

  eigen::matrix Y2 {
    {-0.01224686, -0.01224686, -0.01224686, -0.01250593, -0.01265573, -0.01294112, -0.01224686, -0.01224686},
    {0.12128833, 0.12128833, 0.12128833, 0.11799413, 0.11608937, 0.11246035, 0.12128833, 0.12128833},
    {-0.30070713, -0.30070713, -0.30070713, -0.30232328, -0.30325776, -0.30503815, -0.30070713, -0.30070713},
  };

  eigen::matrix DY2 {
    {0.04321853, -0.08178147, 0.04321853, -0.08171443, 0.04332435, 0.04339826, 0.04321853, -0.08178147},
    {0.04939279, 0.04939279, 0.04939279, 0.04931949, -0.07572287, 0.04919642, -0.07560721, 0.04939279},
    {-0.09261131, 0.03238868, -0.09261131, 0.03239493, 0.03239852, -0.09259468, 0.03238868, 0.03238868},
  };

  scalar lr = 0.01;
  std::vector<long> sizes = {6, 2, 2, 3};
  long N = 8;
  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, N);
}


//--- end generated code ---//