# 1. Require CMake 3.24+ for native GPU architecture detection
cmake_minimum_required(VERSION 3.24)

# 2. Name the project and enable both C++ and CUDA languages
project(CudaVectorAdd LANGUAGES CXX CUDA)

# 3. Target your local GPU automatically (no need to look up sm_XX numbers!)
# If compiling for a different machine, replace "native" with "80;90" (for Ampere/Hopper)
set(CMAKE_CUDA_ARCHITECTURES "89")

# 4. Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 5. Set C++ and CUDA standards (C++17 is standard for modern CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 6. Define your executable and its source files
add_executable(vector_add 
    src/main.cpp
    src/vector_add.cu
)

# 7. Tell CMake where to find your .cuh header files
target_include_directories(vector_add PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 8. (Optional) Enable aggressive optimization in Release mode
target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
