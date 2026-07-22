add_executable(matrix_operations main.cpp)

target_link_libraries(matrix_operations PRIVATE common_lib)

target_compile_options(matrix_operations PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
