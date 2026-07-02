cmake_minimum_required(VERSION 3.10)

project(cpp_ba)

find_package(Ceres REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(
    cpp_ba
    bindings.cpp
    bundle_adjustment.cpp
)

target_link_libraries(
    cpp_ba
    PRIVATE
    Ceres::ceres
)