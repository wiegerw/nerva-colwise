#!/usr/bin/env python3

# Copyright 2023 - 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import shlex
import sys
from pathlib import Path
from typing import List

import torch

from nerva.datasets import create_cifar10_augmented_dataloaders, create_cifar10_dataloaders, \
    create_mnist_dataloaders, create_npz_dataloaders, extract_tensors_from_dataloader
from nerva.grow import GrowFunction, parse_grow_function
from nerva.layers import Dense, Sparse, make_layers
from nerva.learning_rate_schedulers import LearningRateScheduler, parse_learning_rate_scheduler
from nerva.loss_functions import LossFunction, parse_loss_function
from nerva.multilayer_perceptron import print_model_info, MultilayerPerceptron
from nerva.prune import PruneFunction, parse_prune_function
from nerva.regrow import PruneGrow
from nerva.training import StochasticGradientDescentAlgorithm, SGDOptions, compute_sparse_layer_densities, \
    to_one_hot, compute_statistics
from nerva.utilities import manual_seed, nerva_timer_enable, pp, set_nerva_computation, nerva_timer_print_report, \
    nerva_timer_set_verbose
from nerva.weights import parse_weight_initializer


class MLPNerva(MultilayerPerceptron):
    """ Nerva Multilayer perceptron
    """
    def __init__(self,
                 linear_layer_sizes: List[int],
                 linear_layer_densities: List[float],
                 optimizers: List[str],
                 linear_layer_weights: List[str],
                 layer_specifications: List[str],
                 linear_layer_dropouts: List[float],
                 loss: LossFunction,
                 learning_rate: float,
                 lr_scheduler: LearningRateScheduler,
                 batch_size: int
                ):
        super().__init__()
        self.layer_sizes = linear_layer_sizes
        self.layer_densities = linear_layer_densities
        self.loss = loss
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.layers = make_layers(layer_specifications, linear_layer_sizes, linear_layer_densities, linear_layer_dropouts, linear_layer_weights, optimizers)
        self.compile(batch_size)

    def __str__(self):
        density_info = [layer.density_info() for layer in self.layers if isinstance(layer, (Dense, Sparse))]
        return f'{super().__str__()}\nloss = {self.loss}\nscheduler = {self.learning_rate}\nlayer densities: {", ".join(density_info)}\n'


def parse_dropouts(text: str, linear_layer_count: int) -> List[float]:
    n = linear_layer_count

    if not text:
        return [0.0] * n

    dropouts = [float(x) for x in text.strip().split(';')]

    if len(dropouts) == 1:
        return [dropouts[0]] * n

    if len(dropouts) != n:
        raise RuntimeError(f'the number of dropouts ({len(dropouts)}) does not match with the number of linear layers ({n})')

    return dropouts


def parse_init_weights(text: str, linear_layer_count: int) -> List[str]:
    words = text.strip().split(';')
    n = linear_layer_count

    if len(words) == 1:
        return [words[0]] * n

    if len(words) != n:
        raise RuntimeError(f'the number of weight initializers ({len(words)}) does not match with the number of linear layers ({n})')

    return words


def parse_optimizers(text: str, layer_count: int) -> List[str]:
    words = text.strip().split(';')
    n = layer_count

    if len(words) == 0:
        return ['GradientDescent'] * n

    if len(words) == 1:
        return [words[0]] * n

    if len(words) != n:
        raise RuntimeError(f'the number of weight initializers ({len(words)}) does not match with the number of linear layers ({n})')

    return words


def make_argument_parser():
    cmdline_parser = argparse.ArgumentParser()

    # randomness
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int)

    # model parameters
    cmdline_parser.add_argument('--layer-sizes', type=str, default='3072,128,64,10', help='A comma separated list of layer sizes, e.g. "3072,128,64,10".')
    cmdline_parser.add_argument('--densities', type=str, help='A comma separated list of layer densities, e.g. "0.05,0.05,1.0".')
    cmdline_parser.add_argument('--overall-density', type=float, help='The overall density of the layers.')
    cmdline_parser.add_argument('--dropouts', help='A comma separated list of dropout rates')
    cmdline_parser.add_argument('--layers', type=str, help='A semi-colon separated lists of layers.')

    # learning rate
    cmdline_parser.add_argument("--learning-rate", type=float, help="The learning rate (default: 0.01)", default=0.01)
    cmdline_parser.add_argument("--learning-rate-scheduler", type=str, help="The learning rate scheduler")

    # loss function
    cmdline_parser.add_argument('--loss', type=str, help='The loss function')

    # training
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)
    cmdline_parser.add_argument("--manual", help="Do not use the DataLoader interface", action="store_true")

    # optimizer
    cmdline_parser.add_argument("--optimizers", type=str, help="The optimizer (GradientDescent, Momentum(<mu>), Nesterov(<mu>))", default="GradientDescent")

    # dataset
    cmdline_parser.add_argument('--dataset', type=str, help='An .npz file containing train and test data')
    cmdline_parser.add_argument('--cifar10', type=str, default='', help='The directory containing the CIFAR-10 dataset')
    cmdline_parser.add_argument('--mnist', type=str, default='', help='The directory containing the MNIST dataset')
    cmdline_parser.add_argument("--augmented", help="use data loaders with augmentation", action="store_true")
    cmdline_parser.add_argument("--preprocessed", help="folder with preprocessed datasets for each epoch")

    # load/save weights
    cmdline_parser.add_argument('--layer-weights', type=str, default='None', help='The initial weights for the layers')
    cmdline_parser.add_argument('--save-weights', type=str, help='Save weights and bias to a file in .npz format')
    cmdline_parser.add_argument('--load-weights', type=str, help='Load weights and bias from a file in .npz format')

    # print options
    cmdline_parser.add_argument("--precision", help="The precision used for printing matrices", type=int, default=8)
    cmdline_parser.add_argument("--edgeitems", help="The edgeitems used for printing matrices", type=int, default=3)
    cmdline_parser.add_argument("--debug", help="print debug information", action="store_true")
    cmdline_parser.add_argument("--info", help="print information about the MLP", action="store_true")

    # pruning + growing (experimental!)
    cmdline_parser.add_argument("--prune", help="The pruning strategy: Magnitude(<rate>), SET(<rate>) or Threshold(<value>)", type=str)
    cmdline_parser.add_argument("--grow", help="The growing strategy: (default: Random)", type=str)
    cmdline_parser.add_argument('--grow-weights', type=str, help='The function used for growing weigths: Xavier, XavierNormalized, He, PyTorch, Zero', default='Xavier')

    # multi-threading
    cmdline_parser.add_argument("--threads", help="The number of threads being used", type=int)

    # timer
    cmdline_parser.add_argument("--timer", choices=["disabled", "brief", "full"], default="disabled", help="Set timer mode: 'disabled', 'brief', or 'full'")

    # computation
    cmdline_parser.add_argument('--computation', type=str, default='eigen', help='The computation mode (eigen, mkl, blas)')
    cmdline_parser.add_argument('--clip', type=float, default=0, help='A threshold value that is used to set elements to zero')

    return cmdline_parser


def check_command_line_arguments(args):
    if args.augmented and args.preprocessed:
        raise RuntimeError('the combination of --augmented and --preprocessed is unsupported')

    if args.densities and args.overall_density:
        raise RuntimeError('the options --densities and --overall-density cannot be used simultaneously')

    if sum([bool(args.dataset), bool(args.cifar10), bool(args.mnist), bool(args.preprocessed)]) != 1:
        raise RuntimeError("Exactly one of the options --dataset, --cifar10, --mnist, or --preprocessed must be set.")


def quote(text):
    if any(char in text for char in '^@%+=:;,./-"\''):
        if not '"' in text:
            return f'"{text}"'
        elif not "'" in text:
            return f"'{text}'"
        else:
            return shlex.quote(text)
    return text


def print_command_line_arguments(args):
    def print_arg(arg):
        words = arg.split('=')
        if len(words) == 1:
            return quote(arg)
        elif len(words) == 2 and not words[1]:
            return ''
        else:
            return f'{words[0]}={quote(words[1])}'

    print("python3 " + " ".join(print_arg(arg) for arg in sys.argv) + '\n')


def initialize_frameworks(args):
    if args.seed:
        manual_seed(args.seed)

    if args.timer != "disabled":
        nerva_timer_enable()

    if args.timer == "full":
        nerva_timer_set_verbose(True)

    torch.set_printoptions(precision=args.precision, edgeitems=args.edgeitems, threshold=5, sci_mode=False, linewidth=160)

    # avoid 'Too many open files' error when using data loaders
    torch.multiprocessing.set_sharing_strategy('file_system')


class SGD(StochasticGradientDescentAlgorithm):
    def __init__(self,
                 M: MultilayerPerceptron,
                 train_loader,
                 test_loader,
                 options: SGDOptions,
                 loss: LossFunction,
                 learning_rate: float,
                 lr_scheduler: LearningRateScheduler,
                 preprocessed_dir: str,
                 prune: PruneFunction,
                 grow: GrowFunction
                 ):
        super().__init__(M, train_loader, test_loader, options, loss, learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.reload_data_directory = preprocessed_dir
        self.regrow = PruneGrow(prune, grow) if prune else None
        self.clip = options.clip

    def reload_data(self, epoch) -> None:
        """
        Reloads the dataset if a directory with preprocessed data was specified.
        """
        path = Path(self.reload_data_directory) / f'epoch{epoch}.npz'
        self.train_loader, self.test_loader = create_npz_dataloaders(str(path), self.options.batch_size)

    def on_start_training(self) -> None:
        if self.reload_data_directory:
            self.reload_data(0)

    # tag::event[]
    def on_start_epoch(self, epoch):
        if epoch > 0 and self.reload_data_directory:
            self.reload_data(epoch)

        if self.lr_scheduler:
            self.learning_rate = self.lr_scheduler(epoch)

        if epoch > 0:
            self.M.renew_dropout_masks()

        if epoch > 0 and self.regrow:
            self.regrow(self.M)

        if epoch > 0 and self.clip > 0:
            self.M._model.clip(self.clip)
    # end::event[]

    # This is faster than using the DataLoader interface
    def run_manual(self):
        M = self.M
        options = self.options
        num_classes = M.layers[-1].output_size

        self.on_start_training()

        dataset = self.train_loader.dataset
        batch_size = len(dataset) // len(self.train_loader)
        Xtrain, Ttrain = extract_tensors_from_dataloader(self.train_loader)
        N = Xtrain.shape[1]  # the number of examples
        I = list(range(N))
        K = N // batch_size  # the number of batches

        compute_statistics(M, self.learning_rate, self.loss, self.train_loader, self.test_loader, 0, 0.0, options.statistics)

        for epoch in range(self.options.epochs):
            self.on_start_epoch(epoch)
            epoch_label = "epoch{}".format(epoch)
            self.timer.start(epoch_label)

            for batch_index in range(K):
                batch = I[batch_index * batch_size: (batch_index + 1) * batch_size]
                X = Xtrain[:, batch]
                T = Ttrain[batch]

                self.on_start_batch(batch_index)
                T = to_one_hot(T, num_classes)
                Y = M.feedforward(X)
                DY = self.loss.gradient(Y, T) / options.batch_size

                if options.debug:
                    print(f'epoch: {epoch} batch: {batch_index}')
                    print_model_info(M)
                    pp("X", X)
                    pp("Y", Y)
                    pp("DY", DY)

                M.backpropagate(Y, DY)
                M.optimize(self.learning_rate)

                self.on_end_batch(batch_index)

            self.timer.stop(epoch_label)
            seconds = self.timer.seconds(epoch_label)
            compute_statistics(M, self.learning_rate, self.loss, self.train_loader, self.test_loader, epoch + 1, seconds, options.statistics)

            self.on_end_epoch(epoch)

        training_time = self.compute_training_time()
        print(f'Total training time for the {options.epochs} epochs: {training_time:.8f}s\n')

        self.on_end_training()


def main():
    cmdline_parser = make_argument_parser()
    args = cmdline_parser.parse_args()
    check_command_line_arguments(args)
    print_command_line_arguments(args)

    initialize_frameworks(args)
    set_nerva_computation(args.computation)

    if args.cifar10:
        if args.augmented:
            train_loader, test_loader = create_cifar10_augmented_dataloaders(args.batch_size, args.batch_size, args.cifar10)
        else:
            train_loader, test_loader = create_cifar10_dataloaders(args.batch_size, args.batch_size, args.cifar10)
    elif args.mnist:
        train_loader, test_loader = create_mnist_dataloaders(args.batch_size, args.batch_size, args.mnist)
    elif args.dataset:
        train_loader, test_loader = create_npz_dataloaders(args.dataset, batch_size=args.batch_size)
    else:
        train_loader, test_loader = None, None

    linear_layer_sizes = [int(s) for s in args.layer_sizes.split(';')]
    linear_layer_count = len(linear_layer_sizes) - 1

    if args.densities:
        linear_layer_densities = list(float(d) for d in args.densities.split(';'))
    elif args.overall_density:
        linear_layer_densities = compute_sparse_layer_densities(args.overall_density, linear_layer_sizes)
    else:
        linear_layer_densities = [1.0] * (len(linear_layer_sizes) - 1)

    layer_specifications = args.layers.split(';')
    linear_layer_weights = parse_init_weights(args.layer_weights, linear_layer_count)
    layer_optimizers = parse_optimizers(args.optimizers, len(layer_specifications))
    linear_layer_dropouts = parse_dropouts(args.dropouts, linear_layer_count)
    loss = parse_loss_function(args.loss)
    lr_scheduler = parse_learning_rate_scheduler(args.learning_rate_scheduler)

    M = MLPNerva(linear_layer_sizes,
                 linear_layer_densities,
                 layer_optimizers,
                 linear_layer_weights,
                 layer_specifications,
                 linear_layer_dropouts,
                 loss,
                 args.learning_rate,
                 lr_scheduler,
                 args.batch_size
                )

    print('=== Nerva python model ===')
    print(M)

    if args.load_weights:
        M.load_weights_and_bias(args.load_weights)

    if args.save_weights:
        M.save_weights_and_bias(args.save_weights)

    if args.info:
        print_model_info(M)

    if args.epochs > 0:
        print('\n=== Training Nerva model ===')
        options = SGDOptions()
        options.epochs = args.epochs
        options.batch_size = args.batch_size
        options.clip = args.clip
        options.shuffle = False
        options.statistics = True
        options.debug = args.debug
        options.gradient_step = 0
        prune = parse_prune_function(args.prune) if args.prune else None
        grow = parse_grow_function(args.grow, parse_weight_initializer(args.grow_weights)) if args.grow else None
        algorithm = SGD(M, train_loader, test_loader, options, M.loss, M.learning_rate, M.lr_scheduler, args.preprocessed, prune, grow)
        if args.manual:
            algorithm.run_manual()
        else:
            algorithm.run()

        if args.timer in ["brief", "full"]:
            nerva_timer_print_report()


if __name__ == '__main__':
    main()
