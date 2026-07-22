#pragma once
#include <cstddef>

// Declare the wrapper function that will call our CUDA kernel
void run_vector_add(float* a, float* b, float* c, size_t n);
