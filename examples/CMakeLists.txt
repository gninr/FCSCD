cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/external/2dparametricbem/include)
include_directories(/usr/include/eigen3)

add_executable(transmission_bem_test transmission_bem_test.cpp)
add_executable(transmission_fem_test transmission_fem_test.cpp)
add_executable(force_calculation_bem_test force_calculation_bem_test.cpp)

target_link_libraries(transmission_bem_test adj_double_layer double_layer hypersingular parametrizations quadrature single_layer)
target_link_libraries(transmission_fem_test LF::lf.mesh.hybrid2d LF::lf.refinement LF::lf.uscalfe)
target_link_libraries(force_calculation_bem_test adj_double_layer double_layer hypersingular parametrizations quadrature single_layer)
