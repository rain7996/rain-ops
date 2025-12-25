#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <assert.h>

#include "mm.h"
#include "common.h"

using namespace std;

#define BLOCK_TILE_SIZE_X 32U
#define BLOCK_TILE_SIZE_Y 32U
#define BLOCK_TILE_SIZE_K 32U
#define NUM_THREADS (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y)

namespace
{

    __device__ void load_data_to_shared_memory(const __half *A, const __half *B,
                                               __half A_shared_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K],
                                               __half B_shared_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X],
                                               size_t block_idx,
                                               size_t thread_linear_idx,
                                               size_t m, size_t n, size_t k)
    {
// load A from DRAM to shared memory
#pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS; load_idx++)
        {
            const size_t SA_r{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
            const size_t SA_c{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};

            const size_t A_r{blockIdx.y * BLOCK_TILE_SIZE_Y + SA_r};
            const size_t A_c{block_idx * BLOCK_TILE_SIZE_K + SA_c};

            __half val;
            if (A_r < m && A_c < k)
            {
                val = A[A_r * k + A_c];
            }
            A_shared_tile[SA_r][SA_c] = val;
        }

#pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) / NUM_THREADS; load_idx++)
        {
            const size_t SB_r{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
            const size_t SB_c{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};

            const size_t B_r{(block_idx * BLOCK_TILE_SIZE_K + SB_r)};
            const size_t B_c{(blockIdx.x * BLOCK_TILE_SIZE_X + SB_c)};

            __half val;
            if (B_r < k && B_c < n)
            {
                val = B[B_r * n + B_c];
            }
            B_shared_tile[SB_r][SB_c] = val;
        }
    }
}

__global__ void matmul_kernel_v2(const __half *A, const __half *B, __half *C, size_t m, size_t n, size_t k, __half alpha, __half beta)
{
    const size_t thread_linear_idx{threadIdx.y * BLOCK_TILE_SIZE_X + threadIdx.x};

    const size_t C_r{blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.y};
    const size_t C_c{blockIdx.x * BLOCK_TILE_SIZE_X + threadIdx.x};

    __shared__ __half A_shared_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ __half B_shared_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    const size_t num_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K};

    float sum_per_thread{0};
    for (size_t block_idx = 0U; block_idx < num_block_tiles; block_idx++)
    {
        load_data_to_shared_memory(A, B, A_shared_tile, B_shared_tile, block_idx, thread_linear_idx, m, n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; k_i++)
        {
            sum_per_thread += __half2float(A_shared_tile[threadIdx.y][k_i] * B_shared_tile[k_i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (C_r < m && C_c < n)
    {
        C[C_r * n + C_c] = alpha * __float2half(sum_per_thread) + beta * C[C_r * n + C_c];
    }
}

void matmul_v2(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle)
{
    assert(handle == nullptr);
    dim3 block(BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U);
    static_assert(BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X % NUM_THREADS == 0U);
    dim3 grid(
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / BLOCK_TILE_SIZE_Y,
        1U);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    matmul_kernel_v2<<<grid, block>>>(A, B, C, m, n, k, alpha, beta);
    CUDACHECK(cudaGetLastError());
}
