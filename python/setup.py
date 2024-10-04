# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

import os
import sys

__version__ = "0.3"

define_macros = [('VERSION_INFO', __version__)]
include_dirs = ['../include']
extra_compile_args = []
extra_link_args = []

CMAKE_DEPS_DIR = os.getenv('CMAKE_DEPS_DIR')
if CMAKE_DEPS_DIR:
    CMAKE_DEPS_DIR = Path(CMAKE_DEPS_DIR)
    EIGEN_INCLUDE_DIR = CMAKE_DEPS_DIR / 'eigen-src'
    FMT_INCLUDE_DIR = CMAKE_DEPS_DIR / 'fmt-src' / 'include'
    PYBIND11_INCLUDE_DIR = CMAKE_DEPS_DIR / 'pybind11-src' / 'include'
else:
# Configure Eigen
EIGEN_INCLUDE_DIR = os.getenv('EIGEN_INCLUDE_DIR')
if sys.platform.startswith("win"):
    if not EIGEN_INCLUDE_DIR:
        raise RuntimeError("Eigen library not found. Please set the EIGEN_INCLUDE_DIR environment variable to the path of your Eigen installation.")
    else:    
        EIGEN_INCLUDE_DIR = EIGEN_INCLUDE_DIR or '/usr/include/eigen3'

    # Configure FMT
    FMT_INCLUDE_DIR = os.getenv('FMT_INCLUDE_DIR')
    if not FMT_INCLUDE_DIR:
        raise RuntimeError("FMT library not found. Please set the FMT_INCLUDE_DIR environment variable to the path of your FMT installation.")

    # Configure Pybind11
    PYBIND11_INCLUDE_DIR = os.getenv('PYBIND11_INCLUDE_DIR')
    if sys.platform.startswith("win"):
        if not PYBIND11_INCLUDE_DIR:
            raise RuntimeError("Pybind11 library not found. Please set the PYBIND_INCLUDE_DIR environment variable to the path of your pybind11 installation.")
    else:
        PYBIND11_INCLUDE_DIR = PYBIND11_INCLUDE_DIR or '/usr/include'

print(f'EIGEN_INCLUDE_DIR = {EIGEN_INCLUDE_DIR}')
print(f'FMT_INCLUDE_DIR = {FMT_INCLUDE_DIR}')
print(f'PYBIND11_INCLUDE_DIR = {PYBIND11_INCLUDE_DIR}')

# Configure MKL
ONEAPI_ROOT = os.getenv('ONEAPI_ROOT')
if ONEAPI_ROOT:
MKL_ROOT = f'{ONEAPI_ROOT}/mkl/latest'
else:
    MKL_ROOT = os.getenv('MKL_ROOT')
if not MKL_ROOT:
    raise RuntimeError('Could not detect the MKL library. Please set the ONEAPI_ROOT or the MKL_ROOT environment variable')
if sys.platform.startswith("win") and not ONEAPI_ROOT:
    raise RuntimeError('Could not detect the oneAPI library. Please set the ONEAPI_ROOT environment variable')
MKL_INCLUDE_DIR = f'{MKL_ROOT}/include'
MKL_LIB_DIR = f'{MKL_ROOT}/lib'

define_macros += [('EIGEN_USE_MKL_ALL', 1)]
include_dirs += [EIGEN_INCLUDE_DIR, FMT_INCLUDE_DIR, MKL_INCLUDE_DIR, PYBIND11_INCLUDE_DIR]

extra_compile_args += ['-DMKL_ILP64', '-DFMT_HEADER_ONLY']

if sys.platform.startswith("win"):
    extra_compile_args += ['/wd4244', '/wd4267', '/utf-8']
    extra_link_args += [f'{MKL_LIB_DIR}/mkl_intel_ilp64.lib',
                        f'{MKL_LIB_DIR}/mkl_intel_thread.lib',
                        f'{MKL_LIB_DIR}/mkl_core.lib',
                        f'{ONEAPI_ROOT}/compiler/latest/lib/libiomp5md.lib',
                        f'/LIBPATH:{ONEAPI_ROOT}/compiler/latest/compiler/bin'
                       ]
else:
    extra_compile_args += ['-march=native', '-m64', '-fopenmp']
    extra_link_args += ['-Wl,--start-group',
                        f'{MKL_LIB_DIR}/libmkl_intel_ilp64.a',
                        f'{MKL_LIB_DIR}/libmkl_intel_thread.a',
                        f'{MKL_LIB_DIR}/libmkl_core.a',
                        '-Wl,--end-group',
                        '-liomp5',
                        '-lpthread',
                        '-lm',
                        '-ldl'
                       ]

# We need to use absolute paths, since the src folder is in the parent directory.
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))

ext_modules = [
    Pybind11Extension(
        "nervalibcolwise",
        [
            os.path.join(src_dir, "logger.cpp"),
            os.path.join(src_dir, "python-bindings.cpp"),
            os.path.join(src_dir, "utilities.cpp")
        ],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_dirs,
        cxx_std=20
    ),
]

setup(
    name="nervacolwise",
    version=__version__,
    author="Wieger Wesselink",
    author_email="j.w.wesselink@tue.nl",
    description="C++ library for Neural Networks",
    long_description="",
    ext_modules=ext_modules,
    zip_safe=False,
    packages=['nervacolwise']
)

