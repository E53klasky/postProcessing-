cmake_minimum_required(VERSION 3.14)
project(mygrad)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

pybind11_add_module(mygrad cppSrc/gradient.cpp cppSrc/gradient_pybind.cpp cppSrc/gradient2d.cpp)

