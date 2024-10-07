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
TEST_CASE("mlp0")
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
    {0.56110168, 0.66345596},
    {-0.12117522, -0.37362885},
    {0.58880562, -0.04527364},
    {-0.47821110, 0.44252452},
    {0.70591444, -0.14112198},
    {0.67916799, -0.60361999},
  };

  eigen::matrix b1 {
    {0.20102479},
    {0.59402120},
    {0.21540157},
    {0.35205203},
    {0.34316611},
    {0.39400283},
  };

  eigen::matrix W2 {
    {0.08186211, 0.10790163, 0.10656221, 0.05315667, -0.18149444, 0.35163686},
    {-0.19592771, 0.12398510, 0.31654948, -0.07420450, 0.16522692, -0.34778017},
    {0.31287605, -0.02583930, 0.23740244, 0.36834371, -0.06145998, -0.03954251},
    {-0.10137446, -0.21237525, 0.25827262, -0.38412368, -0.11736847, -0.02998422},
  };

  eigen::matrix b2 {
    {0.17123953},
    {-0.11965782},
    {0.35985306},
    {0.39208883},
  };

  eigen::matrix W3 {
    {-0.30466300, 0.16379684, 0.14415038, 0.46986455},
    {0.11790103, -0.04250759, -0.06847537, -0.43189043},
    {-0.36789089, -0.46546835, -0.06826532, -0.03290677},
  };

  eigen::matrix b3 {
    {0.07861263},
    {0.34228545},
    {0.07193470},
  };

  eigen::matrix Y1 {
    {0.15708828, 0.16033420, 0.11323270, 0.12644258, 0.15833774},
    {0.28446376, 0.25479689, 0.28988251, 0.31084442, 0.26541382},
    {-0.10275675, -0.14251150, -0.11241873, -0.09212098, -0.12861101},
  };

  eigen::matrix DY1 {
    {0.06879912, -0.12952240, 0.06686258, 0.06652980, 0.06984292},
    {-0.12185498, 0.07745969, -0.12021869, -0.11999798, -0.12226351},
    {0.05305589, 0.05206273, 0.05335608, 0.05346817, 0.05242061},
  };

  eigen::matrix Y2 {
    {0.14954381, 0.15290099, 0.10718645, 0.11962160, 0.15089402},
    {0.29581159, 0.26589465, 0.29935911, 0.32149324, 0.27656406},
    {-0.10952257, -0.14921829, -0.11815142, -0.09845620, -0.13531879},
  };

  eigen::matrix DY2 {
    {0.06827622, -0.13004242, 0.06644239, 0.06605557, 0.06932384},
    {-0.12096987, 0.07832626, -0.11947980, -0.11916840, -0.12139316},
    {0.05269366, 0.05171615, 0.05303741, 0.05311283, 0.05206933},
  };

  scalar lr = 0.01;
  std::vector<long> sizes = {2, 6, 4, 3};
  long N = 5;
  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, N);
}

TEST_CASE("mlp1")
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
    {-0.39624736, 0.52474105, 0.21831584},
    {-0.38542053, 0.22781183, 0.24430284},
    {-0.48814151, -0.43212008, 0.15193383},
    {-0.42474371, -0.12206060, 0.32209969},
    {0.42924264, 0.54005784, -0.27541694},
  };

  eigen::matrix b1 {
    {0.07732374},
    {0.30383354},
    {0.02999540},
    {-0.43342215},
    {-0.32955059},
  };

  eigen::matrix W2 {
    {-0.01198715, 0.07376724, -0.05684799, -0.18777776, 0.29133716},
    {0.15762907, -0.22121911, 0.35209563, -0.42210799, -0.38685778},
  };

  eigen::matrix b2 {
    {0.43432558},
    {-0.14905037},
  };

  eigen::matrix W3 {
    {-0.09037616, 0.70386237},
    {-0.26921728, 0.64686316},
    {0.34774554, -0.56019980},
    {0.66119587, -0.23288755},
  };

  eigen::matrix b3 {
    {-0.40916497},
    {-0.64210689},
    {-0.35259017},
    {0.22382185},
  };

  eigen::matrix Y1 {
    {-0.45312327, -0.44891989, -0.45228270, -0.45342341},
    {-0.77305222, -0.76053095, -0.77054822, -0.77394629},
    {-0.18344933, -0.19962291, -0.18668362, -0.18229441},
    {0.54542261, 0.51467049, 0.53927302, 0.54761851},
  };

  eigen::matrix DY1 {
    {0.04347773, 0.04435392, 0.04365253, 0.04341538},
    {-0.21842644, 0.03247889, 0.03175326, 0.03150955},
    {0.05693571, -0.19308847, 0.05693214, -0.19306317},
    {0.11801299, 0.11625564, -0.13233793, 0.11813824},
  };

  eigen::matrix Y2 {
    {-0.45506847, -0.45096460, -0.45433164, -0.45551044},
    {-0.77091932, -0.75883305, -0.76874924, -0.77222097},
    {-0.18090963, -0.19661310, -0.18372914, -0.17921844},
    {0.54122663, 0.51152146, 0.53589314, 0.54442573},
  };

  eigen::matrix DY2 {
    {0.04345694, 0.04430398, 0.04360865, 0.04336601},
    {-0.21831259, 0.03256395, 0.03184365, 0.03159395},
    {0.05716429, -0.19286448, 0.05716021, -0.19283350},
    {0.11769136, 0.11599655, -0.13261250, 0.11787353},
  };

  scalar lr = 0.01;
  std::vector<long> sizes = {3, 5, 2, 4};
  long N = 4;
  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, N);
}

TEST_CASE("mlp2")
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
    {-0.32607165, 0.14416048, -0.18457964, 0.37571555, 0.31530383, -0.23843847},
    {0.26372054, 0.36656415, -0.09508932, -0.06380705, -0.21398570, -0.25577575},
  };

  eigen::matrix b1 {
    {-0.01039284},
    {-0.27663937},
  };

  eigen::matrix W2 {
    {-0.23665118, -0.08175746},
    {0.23666702, -0.65622914},
  };

  eigen::matrix b2 {
    {0.18114421},
    {-0.16169840},
  };

  eigen::matrix W3 {
    {-0.12040216, 0.11809724},
    {0.67121518, 0.29201975},
    {0.11411048, -0.23803940},
  };

  eigen::matrix b3 {
    {-0.00158843},
    {-0.40333885},
    {-0.31654432},
  };

  eigen::matrix Y1 {
    {-0.02339859, -0.02310725, -0.02339859, -0.02339859, -0.02339859, -0.02016460, -0.02225747, -0.02339859},
    {-0.28175211, -0.28337622, -0.28175211, -0.28175211, -0.28175211, -0.29978088, -0.28811356, -0.28175211},
    {-0.29587388, -0.29614997, -0.29587388, -0.29587388, -0.29587388, -0.29893887, -0.29695535, -0.29587388},
  };

  eigen::matrix DY1 {
    {0.04933274, -0.07563005, 0.04933274, -0.07566726, 0.04933274, 0.04974561, 0.04947848, -0.07566726},
    {0.03810076, 0.03805654, 0.03810076, 0.03810076, -0.08689924, 0.03761135, -0.08707230, 0.03810076},
    {-0.08743350, 0.03757351, -0.08743350, 0.03756650, 0.03756650, -0.08735696, 0.03759383, 0.03756650},
  };

  eigen::matrix Y2 {
    {-0.02357430, -0.02328351, -0.02357430, -0.02357430, -0.02357430, -0.02033705, -0.02243426, -0.02357430},
    {-0.28248075, -0.28410131, -0.28248075, -0.28248075, -0.28248075, -0.30052209, -0.28883424, -0.28248075},
    {-0.29513836, -0.29541418, -0.29513836, -0.29513836, -0.29513836, -0.29820901, -0.29621974, -0.29513836},
  };

  eigen::matrix DY2 {
    {0.04932753, -0.07563534, 0.04932753, -0.07567246, 0.04932753, 0.04974060, 0.04947305, -0.07567246},
    {0.03807569, 0.03803158, 0.03807569, 0.03807569, -0.08692431, 0.03758618, -0.08709708, 0.03807569},
    {-0.08740322, 0.03760376, -0.08740322, 0.03759678, 0.03759678, -0.08732678, 0.03762402, 0.03759678},
  };

  scalar lr = 0.01;
  std::vector<long> sizes = {6, 2, 2, 3};
  long N = 8;
  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, N);
}


//--- end generated code ---//
