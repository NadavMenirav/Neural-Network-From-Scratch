cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 99)

include_directories(. external)

add_executable(nn_xor examples/nn_xor.c)