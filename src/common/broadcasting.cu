#include "broadcasting.cuh"
#include <cuda_runtime.h>

#define THREADS 256

__global__ void broadcasting_sum(
    float* a, float* b, float* c, float* res,
    size_t shape_a, size_t shape_b, size_t shape_c
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < shape_a * shape_b * shape_c) {
        size_t row = idx / (shape_b * shape_c);
        size_t column = (idx / shape_c) % shape_b;
        // for res and a don't need to compute the linear offset; just use idx
        res[idx] = a[idx] + b[row * shape_b + column] + c[row];
    }
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

    size_t blocks = (shape_a * shape_b * shape_c + THREADS - 1) / THREADS;
    broadcasting_sum<<<blocks, THREADS>>>(d_a, d_b, d_c, d_res, shape_a, shape_b, shape_c);

    cudaMemcpy(res, d_res, bytes_a, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_res);
}
