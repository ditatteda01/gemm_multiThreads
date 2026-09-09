cmake_minimum_required(VERSION 3.20)

project(Gemm_multiThread)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gemm_mthreads
    src/main.cpp
    src/gemm.cpp
)

target_include_directories(gemm_mthreads
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(gemm_mthreads
    PRIVATE
        -mcpu=apple-m1
        -Wall
        -O3
)