#include "cuda_kernels.cuh"
#include "cuda_kernels.cuh"

__global__ void initialize_to_one(float* data, size_t block_size, size_t n_blocks) {
    size_t block_idx = blockIdx.x;
    size_t idx = threadIdx.x;
    size_t offset = block_idx * (block_size / sizeof(float));

    if (block_idx % 2 == 1) {  // 奇数块填充1
        for (size_t i = idx; i < block_size / sizeof(float); i += blockDim.x) {
            data[offset + i] = 1.0f;
        }
    }
}

__global__ void rand_write_kernel(float* data, size_t block_size, size_t n_blocks, unsigned long long seed) {
    size_t block_idx = blockIdx.x;
    size_t idx = threadIdx.x;
    size_t offset = block_idx * (block_size / sizeof(float));

    if (block_idx % 2 == 0) {  // 偶数块进行随机写入
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

    if (block_idx % 2 == 1) {  // 验证奇数块是否为1
        for (size_t i = idx; i < block_size / sizeof(float); i += blockDim.x) {
            if (data[offset + i] != 1.0f) {
                atomicAdd(error_count, 1);
            }
        }
    }
}


void run_kernels(float* data, size_t block_size_bytes, size_t n_blocks, size_t grid_size, size_t block_size, int* host_errors, float* elapsed_time) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    initialize_to_one<<<grid_size, block_size>>>(data, block_size_bytes, n_blocks);
    cudaDeviceSynchronize();

    for (int j = 0; j < 1000; j++) {
        rand_write_kernel<<<grid_size, block_size>>>(data, block_size_bytes, n_blocks, time(NULL) + j);
        cudaDeviceSynchronize();
    }

    int* error_count;
    cudaMalloc((void**)&error_count, sizeof(int));
    cudaMemset(error_count, 0, sizeof(int));

    verify_one_blocks_kernel<<<grid_size, block_size>>>(data, block_size_bytes, n_blocks, error_count);
    cudaDeviceSynchronize();

    cudaMemcpy(host_errors, error_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Stop timing after work is done
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(elapsed_time, start, stop);

    cudaFree(error_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
