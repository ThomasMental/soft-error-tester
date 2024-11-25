#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <curand_kernel.h>

// set the value for n elements 
__global__ void simple_write_kernel(float* data, int n_elements, float value);

// read the data and compare to the value
__global__ void simple_read_and_compare_kernel(float* data, int n_elements, float value, int* error_count);

// initialize all the data to 1
__global__ void initialize_to_one(float* data, size_t block_size, size_t n_blocks);

// use seed to write random value to blocks
__global__ void rand_write_kernel(float* data, size_t block_size, size_t n_blocks, unsigned long long seed);

// verify the value of block is 1
__global__ void verify_one_blocks_kernel(float* data, size_t block_size, size_t n_blocks, int* error_count);

// main kernel program
void run_kernels(float* data, size_t block_size_bytes, size_t n_blocks, size_t grid_size, size_t block_size, int* host_errors, float* elapsed_time, int test_type);

#endif
