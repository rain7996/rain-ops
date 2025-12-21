#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <assert.h>

#include "mm.h"
#include "common.h"

using namespace std;

#define BLOCK_TILE_SIZE_X 64U
#define BLOCK_TILE_SIZE_Y 64U
#define BLOCK_TILE_SIZE_K 8U
#define THREAD_TILE_SIZE_Y 8U
#define NUM_THREADS (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y)

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

// one therad takes account for [THREAD_TILE_SIZE_Y , 1] output elements
__global__ void matmul_kernel_v3(const __half *A, const __half *B, __half *C, size_t m, size_t n, size_t k, __half alpha, __half beta)
{
    const size_t thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    __shared__ __half S_A[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ __half S_B[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_k_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K};

    __half C_results[THREAD_TILE_SIZE_Y] = {static_cast<__half>(0)};

    size_t S_B_c = thread_linear_idx % BLOCK_TILE_SIZE_X;
    size_t PRE_S_A_r = thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y;
    for (size_t b_idx = 0U; b_idx < num_k_block_tiles; b_idx++)
    {
        load_data_to_shared_memory(A, B, S_A, S_B, b_idx, thread_linear_idx, m, n, k);
        __syncthreads();
#pragma unroll
        for (size_t k_i = 0; k_i < BLOCK_TILE_SIZE_K; k_i++)
        {
            __half v_B = S_B[k_i][S_B_c];
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE_SIZE_Y; j++)
            {
                C_results[j] += S_A[PRE_S_A_r + j][k_i] * v_B;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < THREAD_TILE_SIZE_Y; i++)
    {
        size_t C_r = blockIdx.y * BLOCK_TILE_SIZE_Y + (PRE_S_A_r + i);
        size_t C_c = blockIdx.x * BLOCK_TILE_SIZE_X + S_B_c;

        if (C_r < m && C_c < n)
        {
            C[C_r * n + C_c] = alpha * C_results[i] + beta * C[C_r * n + C_c];
        }
    }
}

void matmul_v3(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle)
{
    assert(handle == nullptr);
    dim3 block(NUM_THREADS, 1U, 1U);

    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);
    static_assert(NUM_THREADS % BLOCK_TILE_SIZE_K == 0);
    static_assert(NUM_THREADS % BLOCK_TILE_SIZE_X == 0);

    dim3 grid(
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / BLOCK_TILE_SIZE_Y,
        1U);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    matmul_kernel_v3<<<grid, block>>>(A, B, C, m, n, k, alpha, beta);
    CUDACHECK(cudaGetLastError());
}
