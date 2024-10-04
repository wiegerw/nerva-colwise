// Copyright: Wieger Wesselink 2022 - 2024
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file tools/mlp.cpp
/// \brief add your file description here.

#include "nerva/datasets/cifar10reader.h"
#include "nerva/datasets/dataset.h"
#include "nerva/datasets/mnistreader.h"
#include "nerva/datasets/generate_dataset.h"
#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/learning_rate_schedulers.h"
#include "nerva/neural_networks/loss_functions.h"
#include "nerva/neural_networks/mlp_algorithms.h"
#include "nerva/neural_networks/multilayer_perceptron.h"
#include "nerva/neural_networks/parse_layer.h"
#include "nerva/neural_networks/regrow.h"
#include "nerva/neural_networks/sgd_options.h"
#include "nerva/neural_networks/signal_handling.h"
#include "nerva/neural_networks/training.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/command_line_tool.h"
#include "nerva/utilities/parse_numbers.h"
#include "nerva/utilities/string_utility.h"

#include "omp.h"
#include "fmt/format.h"
#include <algorithm>
#include <iostream>
#include <random>

#ifdef NERVA_ENABLE_PROFILING
#include <valgrind/callgrind.h>
#endif

using namespace nerva;

datasets::dataset dataset;

inline
auto parse_linear_layer_densities(const std::string& densities_text,
                                  double overall_density,
                                  const std::vector<std::size_t>& linear_layer_sizes) -> std::vector<double>
{
  std::vector<std::string> words = utilities::regex_split(utilities::trim_copy(densities_text), ";");
  std::vector<double> densities;
  std::transform(words.begin(), words.end(), std::back_inserter(densities),
    [](const std::string& word){ return std::stold(word); });

  std::size_t n = linear_layer_sizes.size() - 1;  // the number of linear layers

  if (densities.empty())
  {
    if (overall_density == 1)
    {
      densities = std::vector<double>(n, 1);
    }
    else
    {
      densities = compute_sparse_layer_densities(overall_density, linear_layer_sizes);
    }
  }

  if (densities.size() == 1)
  {
    return std::vector<double>(n, densities.front());
  }

  if (densities.size() != n)
  {
    throw std::runtime_error(fmt::format("The number of densities {} does not match with the number of linear layers {}.", densities.size(), n));
  }

  return densities;
}

inline
auto parse_linear_layer_dropouts(const std::string& dropouts_text, std::size_t linear_layer_count) -> std::vector<double>
{
  std::vector<std::string> words = utilities::regex_split(utilities::trim_copy(dropouts_text), ";");
  std::vector<double> dropouts;
  std::transform(words.begin(), words.end(), std::back_inserter(dropouts),
    [](const std::string& word){ return std::stold(word); });

  std::size_t n = linear_layer_count;

  if (dropouts.empty())
  {
    return std::vector<double>(linear_layer_count, 0.0);
  }

  if (dropouts.size() == 1)
  {
    return std::vector<double>(n, dropouts.front());
  }

  if (dropouts.size() != n)
  {
    throw std::runtime_error(fmt::format("The number of dropouts {} does not match with the number of linear layers {}.", dropouts.size(), n));
  }

  return dropouts;
}

inline
auto parse_layer_weights(const std::string& text, std::size_t linear_layer_count) -> std::vector<std::string>
{
  auto n = linear_layer_count;

  std::vector<std::string> words = utilities::regex_split(utilities::trim_copy(text), ";");
  if (words.size() == 1)
  {
    return { n, words.front() };
  }

  if (words.size() != n)
  {
    throw std::runtime_error(fmt::format("The number of weight initializers ({}) does not match with the number of linear layers ({}).", words.size(), n));
  }

  return words;
}

inline
auto parse_optimizers(const std::string& text, std::size_t count) -> std::vector<std::string>
{
  std::vector<std::string> words = utilities::regex_split(utilities::trim_copy(text), ";");
  if (words.empty())
  {
    return {count, "GradientDescent"};
  }
  if (words.size() == 1)
  {
    return {count, words.front()};
  }
  if (words.size() != count)
  {
    throw std::runtime_error(fmt::format("expected {} optimizers instead of {}", count, words.size()));
  }
  return words;
}

inline
void set_optimizers(multilayer_perceptron& M, const std::string& optimizer)
{
  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      set_linear_layer_optimizer(*dlayer, optimizer);
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      set_linear_layer_optimizer(*slayer, optimizer);
    }
  }
}

class sgd_algorithm: public stochastic_gradient_descent_algorithm<datasets::dataset>
{
  protected:
    std::shared_ptr<learning_rate_scheduler> lr_scheduler;
    std::filesystem::path reload_data_directory;
    std::shared_ptr<prune_and_grow> regrow_function;

    using super = stochastic_gradient_descent_algorithm<datasets::dataset>;
    using super::data;
    using super::M;
    using super::rng;
    using super::timer;
    using super::learning_rate;

  public:
    sgd_algorithm(multilayer_perceptron& M,
                  datasets::dataset& data,
                  const sgd_options& options,
                  const std::shared_ptr<loss_function>& loss,
                  scalar learning_rate,
                  const std::shared_ptr<learning_rate_scheduler>& lr_scheduler_,
                  std::mt19937& rng,
                  const std::string& preprocessed_dir_,
                  const std::shared_ptr<prune_function>& prune,
                  const std::shared_ptr<grow_function>& grow
    )
      : super(M, data, options, loss, learning_rate, rng),
        lr_scheduler(lr_scheduler_),
        reload_data_directory(preprocessed_dir_)
    {
      if (prune)
      {
        regrow_function = std::make_shared<prune_and_grow>(prune, grow);
      }
    }

    /// \brief Reloads the dataset if a directory with preprocessed data was specified.
    void reload_data(unsigned int epoch)
    {
      data.load((reload_data_directory / ("epoch" + std::to_string(epoch) + ".npz")).string());
    }

    void on_start_training() override
    {
      if (!reload_data_directory.empty())
      {
        reload_data(0);
      }

      if (lr_scheduler)
      {
        learning_rate = lr_scheduler->operator()(0);
      }
    }

    // tag::event[]
    void on_start_epoch(unsigned int epoch) override
    {
      if (epoch > 0 && !reload_data_directory.empty())
      {
        reload_data(epoch);
      }

      if (lr_scheduler)
      {
        learning_rate = lr_scheduler->operator()(epoch);
      }

      if (epoch > 0)
      {
        renew_dropout_masks(M, rng);
      }

      if (epoch > 0 && regrow_function)
      {
        (*regrow_function)(M);
      }

      if (epoch > 0 && options.clip > 0)
      {
        M.clip(options.clip);
      }
    }
    // end::event[]

    void on_end_epoch(unsigned int epoch) override
    {
      // print_srelu_layers(M);
    }

    void on_start_batch(unsigned int batch_index) override
    {
      check_signal();
    }
};

struct mlp_options: public sgd_options
{
  std::string cifar10;
  std::string mnist;
  std::string dataset;
  std::size_t dataset_size = 2000;
  bool normalize_data = false;
  scalar learning_rate = 0.01;
  std::string learning_rate_scheduler;
  std::string loss_function = "SquaredError";
  std::string architecture;
  std::vector<std::size_t> sizes;
  std::string weights_initialization;
  std::string optimizer = "GradientDescent";
  scalar overall_density = 1;
  std::vector<double> densities;
  std::size_t seed = std::random_device{}();
  int precision = 4;
  int threads = 1;

  void info() const
  {
    std::cout << *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const mlp_options& options)
  {
    out << static_cast<const sgd_options&>(options);
    if (!options.cifar10.empty())
    {
      out << "cifar10 = " << options.cifar10 << std::endl;
    }
    if (!options.dataset.empty())
    {
      out << "dataset = " << options.dataset << std::endl;
      out << "dataset size = " << options.dataset_size << std::endl;
      out << "normalize data = " << std::boolalpha << options.normalize_data << std::endl;
    }
    out << "learning rate = " << options.learning_rate << std::endl;
    if (!options.learning_rate_scheduler.empty())
    {
      out << "learning rate scheduler = " << options.learning_rate_scheduler << std::endl;
    }
    out << "loss function = " << options.loss_function << std::endl;
    out << "architecture = " << options.architecture << std::endl;
    out << "sizes = " << print_list(options.sizes) << std::endl;
    out << "weights initialization = " << options.weights_initialization << std::endl;
    out << "optimizer = " << options.optimizer << std::endl;
    out << "overall density = " << options.overall_density << std::endl;
    out << "densities = " << print_list(options.densities) << std::endl;
    out << "seed = " << options.seed << std::endl;
    out << "precision = " << options.precision << std::endl;
    out << "threads = " << options.threads << std::endl;
    return out;
  }
};

class mlp_tool: public command_line_tool
{
  protected:
    mlp_options options;
    std::string load_weights_file;
    std::string save_weights_file;
    std::string load_dataset_file;
    std::string save_dataset_file;
    std::string linear_layer_sizes_text;
    std::string densities_text;
    std::string dropouts_text;
    std::string layer_specifications_text;
    std::string layer_weights_text = "None";
    std::string computation = "eigen";
    double overall_density = 1;
    std::string preprocessed_dir;  // a directory containing a dataset for every epoch
    bool no_shuffle = false;
    bool no_statistics = false;
    bool info = false;
    std::string timer = "disabled";

    // pruning + growing
    std::string prune_strategy;
    std::string grow_strategy = "Random";
    std::string grow_weights = "Xavier";

    void add_options(lyra::cli& cli) override
    {
      // randomness
      cli |= lyra::opt(options.seed, "value")["--seed"]("A seed value for the random generator.");

      // model parameters
      cli |= lyra::opt(linear_layer_sizes_text, "value")["--layer-sizes"]("A comma separated list of the linear layer sizes");
      cli |= lyra::opt(densities_text, "value")["--densities"]("A comma separated list of sparse layer densities");
      cli |= lyra::opt(dropouts_text, "value")["--dropouts"]("A comma separated list of dropout rates");
      cli |= lyra::opt(overall_density, "value")["--overall-density"]("The overall density level of the sparse layers");
      cli |= lyra::opt(layer_specifications_text, "value")["--layers"]("A semi-colon separated lists of layers. The following layers are supported: "
                                                                  "Linear, ReLU, Sigmoid, Softmax, LogSoftmax, HyperbolicTangent, BatchNormalization, "
                                                                  "AllRelu(<alpha>), TReLU(<epsilon>)");

      // training
      cli |= lyra::opt(options.epochs, "value")["--epochs"]("The number of epochs (default: 100)");
      cli |= lyra::opt(options.batch_size, "value")["--batch-size"]("The batch size of the training algorithm");
      cli |= lyra::opt(no_shuffle)["--no-shuffle"]("Do not shuffle the dataset during training.");
      cli |= lyra::opt(no_statistics)["--no-statistics"]("Do not compute statistics during training.");

      // optimizer
      cli |= lyra::opt(options.optimizer, "value")["--optimizers"]("The optimizer (GradientDescent, Momentum(<mu>), Nesterov(<mu>))");

      // learning rate
      cli |= lyra::opt(options.learning_rate, "value")["--learning-rate"]("The learning rate (default: 0.01)");
      cli |= lyra::opt(options.learning_rate_scheduler, "value")["--learning-rate-scheduler"]("The learning rate scheduler");

      // loss function
      cli |= lyra::opt(options.loss_function, "value")["--loss"]("The loss function (squared-error, cross-entropy, logistic-cross-entropy)");

      // weights
      cli |= lyra::opt(layer_weights_text, "value")["--layer-weights"]("The weight initialization of the layers (default, he, uniform, xavier, normalized_xavier, uniform)");
      cli |= lyra::opt(load_weights_file, "value")["--load-weights"]("Loads the weights and bias from a file in .npz format");
      cli |= lyra::opt(save_weights_file, "value")["--save-weights"]("Saves the weights and bias to a file in .npz format");

      // dataset
      cli |= lyra::opt(options.cifar10, "value")["--cifar10"]("The directory of the CIFAR-10 dataset");
      cli |= lyra::opt(options.mnist, "value")["--mnist"]("The directory of the MNIST dataset");
      cli |= lyra::opt(options.dataset, "value")["--generate-dataset"]("Use a generated dataset (checkerboard, mini)");
      cli |= lyra::opt(options.dataset_size, "value")["--dataset-size"]("The size of the dataset (default: 1000)");
      cli |= lyra::opt(load_dataset_file, "value")["--load-dataset"]("Loads the dataset from a file in .npz format");
      cli |= lyra::opt(save_dataset_file, "value")["--save-dataset"]("Saves the dataset to a file in .npz format");
      cli |= lyra::opt(options.normalize_data)["--normalize"]("Normalize the data");
      cli |= lyra::opt(preprocessed_dir, "value")["--preprocessed"]("A directory containing the files epoch<nnn>.npz");

      // print options
      cli |= lyra::opt(options.precision, "value")["--precision"]("The precision that is used for printing.");
      cli |= lyra::opt(info)["--info"]("print some info about the multilayer_perceptron's");
      cli |= lyra::opt(timer, "-t")["--timer"].choices("disabled", "brief", "full");

      // pruning + growing
      cli |= lyra::opt(prune_strategy, "strategy")["--prune"]("The pruning strategy: Magnitude(<drop_fraction>), SET(<drop_fraction>) or Threshold(<value>)");
      cli |= lyra::opt(grow_strategy, "strategy")["--grow"]("The growing strategy: (default: Random)");
      cli |= lyra::opt(grow_weights, "value")["--grow-weights"]("The weight function used for growing x=Xavier, X=XavierNormalized, ...");

      // miscellaneous
      cli |= lyra::opt(computation, "value")["--computation"]("The computation mode (eigen, mkl, blas)");
      cli |= lyra::opt(options.clip, "value")["--clip"]("A threshold value that is used to set elements to zero");
      cli |= lyra::opt(options.threads, "value")["--threads"]("The number of threads used by Eigen.");
      cli |= lyra::opt(options.gradient_step, "value")["--gradient-step"]("If positive, gradient checks will be done with the given step size");
    }

    auto description() const -> std::string override
    {
      return "A tool for training multilayer perceptrons";
    }

    auto run() -> bool override
    {
      NERVA_LOG(log::verbose) << command_line_call() << "\n\n";

      set_nerva_computation(computation);

      options.debug = is_debug();
      if (no_shuffle)
      {
        options.shuffle = false;
      }
      if (no_statistics)
      {
        options.statistics = false;
      }

      if (timer != "disable")
      {
        nerva_timer.enable();
      }

      if (timer == "full")
      {
        nerva_timer.set_verbose(true);
      }

      if (options.threads >= 1 && options.threads <= 8)
      {
        omp_set_num_threads(options.threads);
      }

      std::mt19937 rng{static_cast<unsigned int>(options.seed)};

      if (!options.cifar10.empty())
      {
        NERVA_LOG(log::verbose) << "Loading dataset CIFAR-10 from folder " << options.cifar10 << '\n';
        dataset = datasets::load_cifar10_dataset(options.cifar10, true);
      }
      else if (!options.mnist.empty())
      {
        NERVA_LOG(log::verbose) << "Loading dataset MNIST from folder " << options.cifar10 << '\n';
        dataset = datasets::load_mnist_dataset(options.mnist, true);
      }
      else if (!options.dataset.empty())
      {
        NERVA_LOG(log::verbose) << "Generating dataset " << options.dataset << '\n';
        dataset = datasets::generate_dataset(options.dataset, options.dataset_size, rng);
      }
      else if (!load_dataset_file.empty())
      {
        dataset.load(load_dataset_file);
      }

      if (!save_dataset_file.empty())
      {
        dataset.save(save_dataset_file);
      }

      auto layer_specifications = parse_layers(layer_specifications_text);
      auto linear_layer_sizes = parse_semicolon_separated_numbers(linear_layer_sizes_text);
      auto linear_layer_count = linear_layer_sizes.size() - 1;
      auto linear_layer_densities = parse_linear_layer_densities(densities_text, overall_density, linear_layer_sizes);
      auto linear_layer_dropouts = parse_linear_layer_dropouts(dropouts_text, linear_layer_count);
      auto linear_layer_weights = parse_layer_weights(layer_weights_text, linear_layer_count);
      auto optimizers = parse_optimizers(options.optimizer, layer_specifications.size());

      // construct the multilayer perceptron M
      multilayer_perceptron M;
      M.layers = make_layers(layer_specifications, linear_layer_sizes, linear_layer_densities, linear_layer_dropouts, linear_layer_weights, optimizers, options.batch_size, rng);

      if (!load_weights_file.empty())
      {
        load_weights_and_bias(M, load_weights_file);
      }

      if (!save_weights_file.empty())
      {
        save_weights_and_bias(M, save_weights_file);
      }

      std::shared_ptr<loss_function> loss = parse_loss_function(options.loss_function);
      std::shared_ptr<learning_rate_scheduler> lr_scheduler = parse_learning_rate_scheduler(options.learning_rate_scheduler);
      std::shared_ptr<prune_function> prune = parse_prune_function(prune_strategy);
      std::shared_ptr<grow_function> grow = parse_grow_function(grow_strategy, parse_weight_initialization(grow_weights), rng);

      if (info)
      {
        dataset.info();
        M.info("before training");
      }

      std::cout << "=== Nerva c++ model ===" << "\n";
      std::cout << M.to_string();
      std::cout << "loss = " << loss->to_string() << "\n";
      if (lr_scheduler)
      {
        std::cout << "learning rate scheduler = " << lr_scheduler->to_string() << "\n";
      }
      std::cout << "layer densities: " << layer_density_info(M) << "\n\n";

      sgd_algorithm algorithm(M, dataset, options, loss, options.learning_rate, lr_scheduler, rng, preprocessed_dir, prune, grow);

#ifdef NERVA_ENABLE_PROFILING
      CALLGRIND_START_INSTRUMENTATION;
#endif

      algorithm.run();

      if (timer == "brief" || timer == "full")
      {
        nerva_timer.print_report();
      }

#ifdef NERVA_ENABLE_PROFILING
      CALLGRIND_STOP_INSTRUMENTATION;
      CALLGRIND_DUMP_STATS;
#endif

      if (info)
      {
        M.info("after training");
      }

      return true;
    }
};

auto main(int argc, const char* argv[]) -> int
{
  pybind11::scoped_interpreter guard{};
  initialize_signal_handling();
  return mlp_tool().execute(argc, argv);
}
