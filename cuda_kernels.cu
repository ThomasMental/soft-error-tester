#include "cuda_kernels.cuh"
#include "cuda_kernels.cuh"
#include <thread> 
#include <chrono> 
#include <cstdio>

__global__ void simple_write_kernel(float* data, int n_elements, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        data[idx] = value;
    }
}

__global__ void simple_read_and_compare_kernel(float* data, int n_elements, float value, int* error_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        if (data[idx] != value) {
            atomicAdd(error_count, 1);
        }
    }
}

__global__ void initialize_to_one(float* data, size_t block_size, size_t n_blocks) {
    size_t block_idx = blockIdx.x;
    size_t idx = threadIdx.x;
    size_t offset = block_idx * (block_size / sizeof(float));

    if (block_idx % 2 == 1) {  // write 1 to block of odd index
        for (size_t i = idx; i < block_size / sizeof(float); i += blockDim.x) {
            data[offset + i] = 1.0f;
        }
    }
}

__global__ void rand_write_kernel(float* data, size_t block_size, size_t n_blocks, unsigned long long seed) {
    size_t block_idx = blockIdx.x;
    size_t idx = threadIdx.x;
    size_t offset = block_idx * (block_size / sizeof(float));

    if (block_idx % 2 == 0) {  // write random value to block of even index
        curandState state;
        curand_init(seed, idx, 0, &state);
        for (size_t i = idx; i < block_size / sizeof(float); i += blockDim.x) {
            data[offset + i] = curand_uniform(&state);
        }
    }
}

__global__ void verify_one_blocks_kernel(float* data, size_t block_size, size_t n_blocks, int* error_count) {
    size_t block_idx = blockIdx.x;
    size_t idx = threadIdx.x;
    size_t offset = block_idx * (block_size / sizeof(float));

    if (block_idx % 2 == 1) {  // verify the value of blocks of odd index
        for (size_t i = idx; i < block_size / sizeof(float); i += blockDim.x) {
            if (data[offset + i] != 1.0f) {
                atomicAdd(error_count, 1);
            }
        }
    }
}

// Reverse bits
__device__ unsigned int reverseBits(unsigned int n) {
    unsigned int reversed = 0;
    for (int i = 0; i < 32; i++) {
        reversed <<= 1;               // Shift the bits of the result left.
        reversed |= (n & 1);          // Add the least significant bit of n to reversed.
        n >>= 1;                      // Shift the bits of n right.
    }
    return reversed;
}

__global__ void reverse_bits_kernel(float* data, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        // change the type from float* to unsigned int
        unsigned int* bits = reinterpret_cast<unsigned int*>(&data[idx]);
        unsigned int reversedBits = reverseBits(*bits);
        *bits = reversedBits;
    }
}

__global__ void verify_reverse_bits_kernel(float* data, int n_elements, int* error_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        unsigned int* bits = reinterpret_cast<unsigned int*>(&data[idx]);
        unsigned int reversedBits = reverseBits(*bits);
        unsigned int revertedBits = reverseBits(reversedBits);
        if (revertedBits != *bits) {
            atomicAdd(error_count, 1);
        }
    }
}


// Main CUDA kernel used to run all the tests
void run_kernels(float* data, size_t block_size_bytes, size_t n_blocks, size_t grid_size, size_t block_size, int* host_errors, float* elapsed_time, int test_type) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate and initialize error count
    int* error_count;
    cudaMalloc((void**)&error_count, sizeof(int));
    cudaMemset(error_count, 0, sizeof(int));

    // Start timing
    cudaEventRecord(start);

    if (test_type == 1) {
        // Test 1: write the assigned value to all the memory and check 
        size_t total_elements = block_size_bytes * n_blocks;
        size_t grid_size = total_elements / block_size;
        
        for (int j = 0; j < 1000; j++) {
            float test_value = j;
            simple_write_kernel<<<grid_size, block_size>>>(data, total_elements / sizeof(float), test_value);
            cudaDeviceSynchronize();
            simple_read_and_compare_kernel<<<grid_size, block_size>>>(data, total_elements / sizeof(float), test_value, error_count);
            cudaDeviceSynchronize();
        }
    } 
    else if (test_type == 2) {
        // Test 2：write and verify blocks with even and odd index
        initialize_to_one<<<grid_size, block_size>>>(data, block_size_bytes, n_blocks);
        cudaDeviceSynchronize();
        for (int j = 0; j < 1000; j++) {
            rand_write_kernel<<<grid_size, block_size>>>(data, block_size_bytes, n_blocks, time(NULL) + j);
            cudaDeviceSynchronize();
        }
        verify_one_blocks_kernel<<<grid_size, block_size>>>(data, block_size_bytes, n_blocks, error_count);
        cudaDeviceSynchronize();
    }
    else if (test_type == 3) {
        // Test 3: write the value and wait for 10 minutes
        float test_value = 1.0f;
        size_t total_elements = block_size_bytes * n_blocks;
        size_t grid_size = total_elements / block_size;
        simple_write_kernel<<<grid_size, block_size>>>(data, total_elements / sizeof(float), test_value);
        cudaDeviceSynchronize();
        
        printf("Waiting for 10 minutes...\n");
        std::this_thread::sleep_for(std::chrono::minutes(10)); 

        simple_read_and_compare_kernel<<<grid_size, block_size>>>(data, total_elements / sizeof(float), test_value, error_count);
        cudaDeviceSynchronize();
    }
    else if(test_type == 4) {
        for (int j = 0; j < 1000; j++) {
            size_t total_elements = block_size_bytes * n_blocks;
            size_t grid_size = total_elements / block_size;
            reverse_bits_kernel<<<grid_size, block_size>>>(data, total_elements / sizeof(float));
            cudaDeviceSynchronize();
            verify_reverse_bits_kernel<<<grid_size, block_size>>>(data, total_elements / sizeof(float), error_count);
            cudaDeviceSynchronize();
        }
    }
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get elapsed time
    cudaEventElapsedTime(elapsed_time, start, stop);

    // Copy error count back to host
    cudaMemcpy(host_errors, error_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(error_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
