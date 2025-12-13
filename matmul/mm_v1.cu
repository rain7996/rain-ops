#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <assert.h>

#include "mm.h"
#include "common.h"

using namespace std;

__global__ void matmul_kernel_v1(const __half *A, const __half *B, __half *C, size_t m, size_t n, size_t k, __half alpha, __half beta)
{
    extern __shared__ __half S[];
    __half *SA = S;
    __half *SB = &SA[blockDim.y * k];

    int A_r = blockDim.y * blockIdx.y + threadIdx.y;
    int B_c = blockDim.x * blockIdx.x + threadIdx.x;

    // printf("[block(%d %d %d)  thread (%d %d %d) [A_r, B_c] (%d %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, A_r, B_c);

    if (A_r >= m || B_c >= n)
    {
        return;
    }

    for (int c = threadIdx.x; c < k; c += blockDim.x)
    {
        SA[threadIdx.y * k + c] = A[A_r * k + c];
    }

    for (int r = threadIdx.y; r < k; r += blockDim.y)
    {
        SB[r * blockDim.x + threadIdx.x] = B[r * n + B_c];
    }

    __syncthreads();

    int C_r = A_r;
    int C_c = B_c;
    int C_idx = C_r * n + C_c;

    __half C_v = C[C_idx] * beta;
    float v_sum = 0;
    for (int i = 0; i < k; i++)
    {
        v_sum += __half2float(SA[threadIdx.y * k + i] * SB[i * blockDim.x + threadIdx.x]);
    }

    C[C_idx] = C_v + alpha * __float2half(v_sum);
}

void matmul_v1(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle)
{
    assert(handle == nullptr);
    dim3 block(MM_V1_BLOCK_SIZE, MM_V1_BLOCK_SIZE);
    dim3 grid(
        (static_cast<unsigned int>(n) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(m) + block.y - 1U) / block.y,
        1U);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    unsigned int s_a_size = block.y * static_cast<unsigned int>(k) * sizeof(__half);
    // s_a_size = ALIGN_TO(s_a_size, 32);
    unsigned int s_b_size = block.x * static_cast<unsigned int>(k) * sizeof(__half);
    // s_b_size = ALIGN_TO(s_b_size, 32);

    unsigned int buffer_size = s_a_size + s_b_size;

    cudaFuncSetAttribute(matmul_kernel_v1,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, buffer_size);

    matmul_kernel_v1<<<grid, block, buffer_size>>>(A, B, C, m, n, k, alpha, beta);
    CUDACHECK(cudaGetLastError());
}
