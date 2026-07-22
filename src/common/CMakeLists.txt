add_library(common_lib STATIC 
    broadcasting.cu
    matrix_mult.cu
    vector_add.cu
)

target_include_directories(common_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

