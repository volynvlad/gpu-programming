add_executable(mnist main.cpp)

target_link_libraries(mnist PRIVATE common_lib)

target_compile_options(mnist PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
