#include "matrix_mult.cuh"
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define THREADS 256

__global__ void matmul_elem_kernel(float* a, float* b, float* c, size_t n) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && column < n) {
        float dot_prod = 0.0f;
        for (int i(0); i < n; i++) {
            dot_prod += a[row * n + i] * b[i * n + column];
        }
        c[row * n + column] = dot_prod;
    }
}

__global__ void matmul_elem_kernel_onedim(float* a, float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int column = idx / n;
    int row = idx % n;
    if (row < n && column < n) {
        float dot_prod = 0.0f;
        for (int i(0); i < n; i++) {
            dot_prod += a[row * n + i] * b[i * n + column];
        }
        c[row * n + column] = dot_prod;
    }
}


void run_matmul(float *a, float *b, float *c, size_t n) {
    float *d_a, *d_b, *d_c;
    size_t bytes = n * n * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, bytes, cudaMemcpyHostToDevice);

    int blocks = (n + THREADS - 1) / THREADS;
    matmul_elem_kernel_onedim<<<blocks, THREADS>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
