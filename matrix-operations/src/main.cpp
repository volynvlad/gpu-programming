#include "vector_add.cuh"
#include "matrix_mult.cuh"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout.imbue(std::locale(""));
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

    N = 5000;

    std::vector<float> m_a(N * N, 1.0f);
    std::vector<float> m_b(N * N, 2.0f);
    std::vector<float> m_c(N * N, 0.0f);

    std::cout << "Running CUDA Matrix Multiplication..." << std::endl;
    start = std::chrono::steady_clock::now();
    run_matmul(m_a.data(), m_b.data(), m_c.data(), N);
    end = std::chrono::steady_clock::now();
    elapsed_ms = end - start;
    elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Result at row 0 and col 0: " << m_c[0] << " (Expected: 2000)\n"
        << "Matrix Multiplication " << N << " x " << N << " took:\nIn microseconds (int): " << elapsed_ms.count() << " ms\n"
        << "In microseconds (int): " << elapsed_us.count() << " us\n";
    return 0;
}
