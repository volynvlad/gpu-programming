cmake_minimum_required(VERSION 3.24)

# 2. Name the project and enable both C++ and CUDA languages
project(GPUProgramming LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "89")

# Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(src/common)

add_subdirectory(src/matrix-operations)
add_subdirectory(src/mnist)
