cmake_minimum_required(VERSION 3.15)
project(Compress_ADIOS2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED)

add_executable(compress_grid compress_grid.cpp)
target_link_libraries(compress_grid
    PRIVATE
        MPI::MPI_CXX
        adios2::cxx11_mpi
)
