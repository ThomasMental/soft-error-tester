#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <curand_kernel.h>

__global__ void initialize_to_one(float* data, size_t block_size, size_t n_blocks);
__global__ void rand_write_kernel(float* data, size_t block_size, size_t n_blocks, unsigned long long seed);
__global__ void verify_one_blocks_kernel(float* data, size_t block_size, size_t n_blocks, int* error_count);

void run_kernels(float* data, size_t block_size_bytes, size_t n_blocks, size_t grid_size, size_t block_size, int* host_errors, float* elapsed_time);

#endif
