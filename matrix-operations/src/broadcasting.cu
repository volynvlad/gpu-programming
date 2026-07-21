#include "broadcasting.cuh"
#include <cuda_runtime.h>

#define THREADS 256

__global__ void broadcasting_sum(
    float* a, float* b, float* c, float* res,
    size_t shape_a, size_t shape_b, size_t shape_c
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = idx / (shape_c * shape_b);
    size_t column = (idx - row * shape_c * shape_b) / shape_b;
    size_t width = idx % shape_b;
    if (row < shape_a && column < shape_b && width < shape_c) 
        for (int i(0); i < shape_b; i++) 
            for (int j(0); j < shape_c; j++) 
                res[shape_b * shape_c * row + shape_c * i + j] = 
                    a[shape_b * shape_c * row + shape_c * i + j]
                    + b[row * shape_b + j]
                    + c[row];
}

void run_broadcasting(
    float* a, float* b, float* c, float* res,
    size_t shape_a, size_t shape_b, size_t shape_c
) {
    float *d_a, *d_b, *d_c, *d_res;
    
    size_t bytes_a = shape_a * shape_b * shape_c * sizeof(float);
    size_t bytes_b = shape_a * shape_b * sizeof(float);
    size_t bytes_c = shape_a * sizeof(float);

    cudaMalloc(&d_a,   bytes_a);
    cudaMalloc(&d_b,   bytes_b);
    cudaMalloc(&d_c,   bytes_c);
    cudaMalloc(&d_res, bytes_a);

    cudaMemcpy(d_a,   a,   bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,   b,   bytes_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,   c,   bytes_c, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, bytes_a, cudaMemcpyHostToDevice);

    size_t blocks = (shape_a + THREADS - 1) / THREADS;
    broadcasting_sum<<<blocks, THREADS>>>(d_a, d_b, d_c, d_res, shape_a, shape_b, shape_c);

    cudaMemcpy(res, d_res, bytes_a, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_res);
}
