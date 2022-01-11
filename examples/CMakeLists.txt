cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/external/2dparametricbem/include)
include_directories(/usr/include/eigen3)

add_executable(square_bem square_bem.cpp)
add_executable(square_fem square_fem.cpp)
add_executable(kite_bem kite_bem.cpp)
add_executable(kite_fem kite_fem.cpp)

target_link_libraries(square_bem adj_double_layer double_layer hypersingular parametrizations quadrature single_layer)
target_link_libraries(square_fem LF::lf.mesh.hybrid2d LF::lf.refinement LF::lf.uscalfe)
target_link_libraries(kite_bem adj_double_layer double_layer hypersingular parametrizations quadrature single_layer)
target_link_libraries(kite_fem LF::lf.mesh.hybrid2d LF::lf.refinement LF::lf.uscalfe)