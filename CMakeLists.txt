cmake_minimum_required(VERSION 3.10)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.307.tar.gz"
    SHA1 "093bd9aab70893cf93470f20d87844938c1546ab"
)

project(fcscd)

hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)

add_subdirectory(examples)
add_subdirectory(external/2dparametricbem)
