project(NERVA_EXAMPLES)

add_executable(subnormal_numbers "subnormal_numbers.cpp")
target_link_libraries(subnormal_numbers PRIVATE Eigen3::Eigen MKL::MKL)
