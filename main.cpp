#include <iostream>
#include <fstream>
#include <cstdio> 
#include "cuda_kernels.cuh"

int main(int argc, char** argv) {
    // default value
    int gpu_id = 0; 
    size_t total_size_bytes = static_cast<size_t>(8) * 1024 * 1024 * 1024; // 8GB

    // parse the arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--gpu_id=") == 0) {
            gpu_id = std::stoi(arg.substr(9)); 
        } else if (arg.find("--total_data=") == 0) {
            total_size_bytes = static_cast<size_t>(std::stoll(arg.substr(13))) * 1024 * 1024; 
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    cudaSetDevice(gpu_id);
    std::cout << "Using GPU ID: " << gpu_id << std::endl;
    std::cout << "Using Memory: " << total_size_bytes / (1024 * 1024) << "MB" << std::endl;

    size_t block_size_bytes = 16 * 1024; // 16KB per block
    size_t n_blocks = total_size_bytes / block_size_bytes;
    size_t block_size = 256; // Threads per block   
    size_t grid_size = n_blocks ;

    float* data;
    cudaMalloc((void**)&data, total_size_bytes);
    cudaMemset(data, 0, total_size_bytes);

    int errors = 0;
    float elapsed_time = 0.0f;

    printf("Running Test 1...\n");
    run_kernels(data, block_size_bytes, n_blocks, grid_size, block_size, &errors, &elapsed_time, 1);
    printf("Test 1 Errors: %d, Time Elapsed: %.2f s\n", errors, elapsed_time / 1000.0);

    errors = 0;
    printf("Running Test 2...\n");
    run_kernels(data, block_size_bytes, n_blocks, grid_size, block_size, &errors, &elapsed_time, 2);
    printf("Test 2 Errors: %d, Time Elapsed: %.2f s\n", errors, elapsed_time / 1000.0);

    errors = 0;
    printf("Running Test 3...\n");
    run_kernels(data, block_size_bytes, n_blocks, grid_size, block_size, &errors, &elapsed_time, 3);
    printf("Test 3 Errors: %d, Time Elapsed: %.2f s\n", errors, elapsed_time / 1000.0);

    cudaFree(data);
    cudaDeviceReset();

    return 0;
}
