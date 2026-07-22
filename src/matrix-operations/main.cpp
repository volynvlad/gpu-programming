#include "broadcasting.cuh"
#include "vector_add.cuh"
#include "matrix_mult.cuh"
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

void check_vector_add() {
    size_t N = 1 << 20; // ~1 million elements
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N, 0.0f);

    std::cout << "Running CUDA Vector Addition..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    run_vector_add(a.data(), b.data(), c.data(), N);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end - start;
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Result at index 0: " << c[0] << " (Expected: 3)\n" 
        << "Vector addition " << N << " took:\nIn microseconds (int): " << elapsed_ms.count() << " ms\n"
        << "In microseconds (int): " << elapsed_us.count() << " us\n";
}

void check_matmul() {
    size_t N = 5000;

    std::vector<float> m_a(N * N, 1.0f);
    std::vector<float> m_b(N * N, 2.0f);
    std::vector<float> m_c(N * N, 0.0f);

    std::cout << "Running CUDA Matrix Multiplication..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    run_matmul(m_a.data(), m_b.data(), m_c.data(), N);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end - start;
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Result at row 0 and col 0: " << m_c[0] << " (Expected: 2000)\n"
        << "Matrix Multiplication " << N << " x " << N << " took:\nIn microseconds (int): " << elapsed_ms.count() << " ms\n"
        << "In microseconds (int): " << elapsed_us.count() << " us\n";
}

bool verify_broadcasting(float* a, float* b, float* c, float* res, 
                         int N, int M, int K) {
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < M; y++) {
            for (int z = 0; z < K; z++) {
                int idx_a = x * (M * K) + y * K + z; // a[x][y][z]
                int idx_b = x * M + y;               // b[x][y]
                int idx_c = x;                       // c[x]

                float expected = a[idx_a] + b[idx_b] + c[idx_c];
                
                if (std::fabs(res[idx_a] - expected) > 1e-5f) {
                    std::cerr << "FAIL at [" << x << "][" << y << "][" << z << "]: "
                              << "Expected " << expected << ", got " << res[idx_a] << std::endl;
                    return false;
                }
                // std::cout << "Correct at [" << x << "][" << y << "][" << z << "]: " << "Expected " << expected << " and got " << res[idx_a] << std::endl;
            }
        }
    }
    std::cout << "SUCCESS: All " << (N * M * K) << " elements match expected broadcasting!" << std::endl;
    return true;
}


void check_broadcast() {
    size_t N = 500, M = 1000, K = 200;

    std::vector<float> m_a(N * M * K, 1.0f);
    std::vector<float> m_b(N * M, 10.0f);
    std::vector<float> m_c(N, 100.0f);
    std::vector<float> res(N * M * K, 0.0f);

    std::cout << "Running CUDA Broadcasting sum..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    run_broadcasting(m_a.data(), m_b.data(), m_c.data(), res.data(), N, M, K);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end - start;
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Broadcasting sum " << N << " x " << M << " x " << K << " took:\nIn microseconds (int): " << elapsed_ms.count() << " ms\n"
        << "In microseconds (int): " << elapsed_us.count() << " us\n";
    verify_broadcasting(m_a.data(), m_b.data(), m_c.data(), res.data(), N, M, K);
}


int main() {
    std::cout.imbue(std::locale(""));
    check_matmul();
    // check_broadcast();
    return 0;
}
