#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <assert.h>

#include "mm.h"
#include "common.h"

using namespace std;

__global__ void matmul_kernel_v0(const __half *A, const __half *B, __half *C, size_t m, size_t n, size_t k, __half alpha, __half beta)
{
    const size_t row_idx = blockIdx.x / n;
    const size_t col_idx = blockIdx.x % n;
    const size_t C_idx = row_idx * n + col_idx;

    const size_t step_per_thread = k / blockDim.x;
    const size_t start_per_thread = threadIdx.x * step_per_thread;
    const size_t end_per_thread = MIN(k, start_per_thread + step_per_thread);

    __half ori_c;
    if (threadIdx.x == 0)
    {
        ori_c = C[C_idx] * beta;
    }

    __syncthreads();

    __half sum_in_thread = 0;
    for (int i = start_per_thread; i < end_per_thread; i++)
    {
        sum_in_thread = sum_in_thread + A[row_idx * k + i] * B[col_idx + n * i];
    }

    using BlockReduce = cub::BlockReduce<__half, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    sum_in_thread = BlockReduce(reduceStore).Sum(sum_in_thread, blockDim.x);

    if (threadIdx.x == 0)
    {
        C[C_idx] = ori_c + alpha * sum_in_thread;
    }
}

void matmul_v0(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle)
{
    assert(handle == nullptr);
    const size_t n_blocks = static_cast<size_t>(m * n);
    const size_t n_threads = static_cast<size_t>(MIN(k, 1024));
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    matmul_kernel_v0<<<n_blocks, n_threads>>>(A, B, C, m, n, k, alpha, beta);
    CUDACHECK(cudaGetLastError());
}
