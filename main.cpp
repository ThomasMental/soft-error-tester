#include <iostream>
#include <fstream>
#include "cuda_kernels.cuh"

int main() {
    size_t total_size_bytes = static_cast<size_t>(12) * 1024 * 1024 * 1024; // 6GB
    size_t block_size_bytes = 16 * 1024; // 16KB per block
    size_t n_blocks = total_size_bytes / block_size_bytes;
    size_t block_size = 256; // Threads per block
    size_t grid_size = n_blocks;

    float* data;
    cudaMalloc((void**)&data, total_size_bytes);
    cudaMemset(data, 0, total_size_bytes);

    int errors = 0;
    float elapsed_time = 0.0;

    run_kernels(data, block_size_bytes, n_blocks, grid_size, block_size, &errors, &elapsed_time);

    std::cout << "Detected " << errors << " errors." << std::endl;
    std::cout << "Elapsed time: " << elapsed_time / 1000 << " s" << std::endl;

    float* h_data = new float[total_size_bytes / sizeof(float)];
    cudaMemcpy(h_data, data, total_size_bytes, cudaMemcpyDeviceToHost);
    
    std::ofstream outfile;
    outfile.open("data.txt");
    for (size_t i = 0; i < block_size_bytes; ++i) {
        outfile << "Data[" << i << "]: " << h_data[i] << std::endl;
    }
    outfile.close();


    cudaFree(data);
    cudaDeviceReset();

    return 0;
}
