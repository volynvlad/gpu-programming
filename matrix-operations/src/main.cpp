#include "vector_add.cuh"
#include <iostream>
#include <vector>

int main() {
    size_t N = 1 << 20; // ~1 million elements
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N, 0.0f);

    std::cout << "Running CUDA Vector Addition..." << std::endl;
    run_vector_add(a.data(), b.data(), c.data(), N);
    std::cout << "Result at index 0: " << c[0] << " (Expected: 3)" << std::endl;
    return 0;
}
