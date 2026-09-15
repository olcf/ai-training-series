#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void stress(float *a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float x = a[idx];

    for (int i = 0; i < 1000000; i++) {
        x = x * 1.000001f + 0.000001f;
    }

    a[idx] = x;
}

int main() {
    int N = 1 << 20; // 1,048,576 threads
    float *d_a;
    cudaMalloc(&d_a, N * sizeof(float));

    auto start_time = std::chrono::high_resolution_clock::now();

    while (true) {
        stress<<<1024, 1024>>>(d_a);
        cudaDeviceSynchronize();

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        if (elapsed >= 5.0) break; // stop after 5 seconds
    }

    cudaFree(d_a);
    printf("Finished 5s stress run\n");
    return 0;
}
