cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(/usr/include/eigen3)

add_executable(geom geom.cpp)
