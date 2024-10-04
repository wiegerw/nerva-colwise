cmake_minimum_required(VERSION 3.16)
project(nerva_colwise LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package(MKL CONFIG REQUIRED)

# Fetch Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# Fetch FMT
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.0.2
)
FetchContent_MakeAvailable(fmt)
include_directories(${fmt_SOURCE_DIR}/include)

# Fetch Lyra
FetchContent_Declare(
    lyra
    GIT_REPOSITORY https://github.com/bfgroup/Lyra.git
    GIT_TAG 1.6.1
)
FetchContent_MakeAvailable(lyra)
include_directories(${lyra_SOURCE_DIR}/include)

# Fetch pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.5
)
FetchContent_MakeAvailable(pybind11)

# Fetch doctest
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

# This is used for dynamic linking of the MKL libraries
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${MKL_ROOT}/lib")

# Option for disabling timer measurements. This may slightly increase the performance.
option(NERVA_DISABLE_TIMER "Disable timer messages" OFF)
if(NERVA_DISABLE_TIMER)
    add_definitions(-DNERVA_DISABLE_TIMER)
endif()

# Suppress compiler warnings
if(MSVC)
    add_compile_options(/utf-8)
    add_compile_options(/wd4244)
    add_compile_options(/bigobj)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    add_compile_options(-Wno-deprecated-anon-enum-enum-conversion)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-enum-compare)
endif()

include_directories(${Python_INCLUDE_DIRS})
include_directories(include)

add_compile_definitions(EIGEN_USE_MKL_ALL MKL_ILP64 FMT_HEADER_ONLY)

# This is a workaround for a conflict between MKL and Eigen.
add_compile_definitions(EIGEN_COLPIVOTINGHOUSEHOLDERQR_LAPACKE_H)

add_library(nervalib STATIC src/logger.cpp src/utilities.cpp)
target_link_libraries(nervalib PUBLIC MKL::MKL Eigen3::Eigen Python3::Python)

# Note that this only installs the pybind11 module, but not the corresponding Python files.
# See the documentation for instructions on how to install the complete `nerva` module using `pip`.
pybind11_add_module(nervalibcolwise src/python-bindings.cpp src/logger.cpp src/utilities.cpp)
target_link_libraries(nervalibcolwise LINK_PUBLIC MKL::MKL Eigen3::Eigen Python3::Python pybind11::pybind11)
set(PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages" CACHE PATH "Python package install directory")
install(TARGETS nervalibcolwise LIBRARY DESTINATION ${PYTHON_INSTALL_DIR})

enable_testing()
add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(tools)
