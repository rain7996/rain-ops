#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <assert.h>

#include "mm.h"
#include "common.h"

using namespace std;

#define BLOCK_TILE_SIZE_X 128U
#define BLOCK_TILE_SIZE_Y 128U
#define BLOCK_TILE_SIZE_K 16U
#define THREAD_TILE_SIZE_X 8U
#define THREAD_TILE_SIZE_Y 8U
#define NUM_THREADS (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_X / THREAD_TILE_SIZE_Y)

#define VECTOR_TYPE float4
#define NUM_VECTOR_UNITS (sizeof(VECTOR_TYPE) / sizeof(__half))
#define VECTORIZED_BLOCK_TILE_SIZE_K (BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS)
#define VECTORIZED_BLOCK_TILE_SIZE_X (BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS)

namespace
{

    __device__ void load_data_to_shared_memory(const __half *A, const __half *B,
                                               __half A_shared_tile_trans[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y],
                                               __half B_shared_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X],
                                               size_t block_idx,
                                               size_t thread_linear_idx,
                                               size_t m, size_t n, size_t k)
    {
        // load A from DRAM to shared memory
#pragma unroll
        for (size_t i = 0; i < (BLOCK_TILE_SIZE_Y * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS - 1) / NUM_THREADS; i++)
        {
            const size_t S_A_r = (i * NUM_THREADS + thread_linear_idx) / VECTORIZED_BLOCK_TILE_SIZE_K;
            const size_t S_A_c = (i * NUM_THREADS + thread_linear_idx) % VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS;

            const size_t A_r = blockIdx.y * BLOCK_TILE_SIZE_Y + S_A_r;
            const size_t A_c = block_idx * BLOCK_TILE_SIZE_K + S_A_c;

            float4 A_row_vec_val = {0, 0, 0, 0};
            if (A_r < m && A_c < k)
            {
                A_row_vec_val = *reinterpret_cast<const float4 *>(&A[A_r * k + A_c]);
            }

            __half *A_vec_vals = reinterpret_cast<__half *>(&A_row_vec_val);
            if (A_c + NUM_VECTOR_UNITS > k)
            {
                size_t num_invalid_ele = A_c + NUM_VECTOR_UNITS - k;
                for (size_t t = 0; t < num_invalid_ele; t++)
                {
                    A_vec_vals[NUM_VECTOR_UNITS - t - 1] = 0;
                }
            }

            if (S_A_r < BLOCK_TILE_SIZE_Y && S_A_c < BLOCK_TILE_SIZE_K)
            {
                for (size_t i = 0; i < NUM_VECTOR_UNITS; i++)
                {
                    A_shared_tile_trans[S_A_c + i][S_A_r] = *(A_vec_vals + i);
                }
            }
        }

#pragma unroll
        for (size_t i = 0; i < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X + NUM_THREADS - 1) / NUM_THREADS; i++)
        {
            const size_t S_B_r = (i * NUM_THREADS + thread_linear_idx) / VECTORIZED_BLOCK_TILE_SIZE_X;
            const size_t S_B_c = (i * NUM_THREADS + thread_linear_idx) % VECTORIZED_BLOCK_TILE_SIZE_X * NUM_VECTOR_UNITS;

            const size_t B_r = block_idx * BLOCK_TILE_SIZE_K + S_B_r;
            const size_t B_c = blockIdx.x * BLOCK_TILE_SIZE_X + S_B_c;

            float4 B_row_vec_val = {0, 0, 0, 0};
            if (B_r < k && B_c < n)
            {
                B_row_vec_val = *reinterpret_cast<const float4 *>(&B[B_r * n + B_c]);
            }

            __half *B_vec_vals = reinterpret_cast<__half *>(&B_row_vec_val);
            if (B_c + NUM_VECTOR_UNITS > n)
            {
                size_t num_invalid_ele = B_c + NUM_VECTOR_UNITS - n;
                for (size_t t = 0; t < num_invalid_ele; t++)
                {
                    B_vec_vals[NUM_VECTOR_UNITS - t - 1] = 0;
                }
            }

            if (S_B_r < BLOCK_TILE_SIZE_K && S_B_c < BLOCK_TILE_SIZE_X)
            {
                for (size_t i = 0; i < NUM_VECTOR_UNITS; i++)
                {
                    B_shared_tile[S_B_r][S_B_c + i] = *(B_vec_vals + i);
                }
            }
        }
    }
};

__global__ void matmul_kernel_v4(const __half *A, const __half *B, __half *C, size_t m, size_t n, size_t k, __half alpha, __half beta)
{
    const size_t thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    __shared__ __half S_A_trans[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ __half S_B[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_k_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K};

    __half C_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {static_cast<__half>(0)};

    __half R_A[THREAD_TILE_SIZE_Y] = {static_cast<__half>(0)};
    __half R_B[THREAD_TILE_SIZE_X] = {static_cast<__half>(0)};

    size_t b_r_start = threadIdx.y * THREAD_TILE_SIZE_Y;
    size_t b_c_start = threadIdx.x * THREAD_TILE_SIZE_X;
    for (size_t b_idx = 0U; b_idx < num_k_block_tiles; b_idx++)
    {
        load_data_to_shared_memory(A, B, S_A_trans, S_B, b_idx, thread_linear_idx, m, n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i = 0; k_i < BLOCK_TILE_SIZE_K; k_i++)
        {
            *reinterpret_cast<float4 *>(R_A) = *reinterpret_cast<float4 *>(&S_A_trans[k_i][b_r_start]);
            *reinterpret_cast<float4 *>(R_B) = *reinterpret_cast<float4 *>(&S_B[k_i][b_c_start]);

#pragma unroll
            for (size_t i = 0; i < THREAD_TILE_SIZE_Y; i++)
            {
                for (size_t j = 0; j < THREAD_TILE_SIZE_X; j++)
                {
                    C_results[i][j] += R_A[i] * R_B[j];
                }
            }
        }
        __syncthreads();
    }

    size_t C_r_start = blockIdx.y * BLOCK_TILE_SIZE_Y + b_r_start;
    size_t C_c_start = blockIdx.x * BLOCK_TILE_SIZE_X + b_c_start;
#pragma unroll
    for (size_t i = 0; i < THREAD_TILE_SIZE_Y; i++)
    {
        for (size_t j = 0; j < THREAD_TILE_SIZE_X; j += NUM_VECTOR_UNITS)
        {
            size_t C_r = C_r_start + i;
            size_t C_c = C_c_start + j;

            float4 C_r_vector_vals = *reinterpret_cast<float4 *>(&C[C_r * n + C_c]);

            for (size_t t = 0; t < NUM_VECTOR_UNITS; t++)
            {
                reinterpret_cast<__half *>(&C_r_vector_vals)[t] = alpha * C_results[i][j + t] + beta * reinterpret_cast<__half *>(&C_r_vector_vals)[t];
            }

            if (C_r < m && C_c < n)
            {
                *reinterpret_cast<float4 *>(&C[C_r * n + C_c]) = C_r_vector_vals;
            }
        }
    }
}

void matmul_v4(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle)
{
    assert(handle == nullptr);

    static_assert(sizeof(VECTOR_TYPE) % sizeof(__half) == 0);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_Y * sizeof(__half) % sizeof(VECTOR_TYPE) == 0);
    static_assert(BLOCK_TILE_SIZE_X * sizeof(__half) % sizeof(VECTOR_TYPE) == 0);

    static_assert(THREAD_TILE_SIZE_Y == NUM_VECTOR_UNITS);
    static_assert(THREAD_TILE_SIZE_X == NUM_VECTOR_UNITS);

    // static_assert(BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K % NUM_THREADS == 0);
    // static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X % NUM_THREADS == 0);

    dim3 block((BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y), (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X), 1U);
    dim3 grid(
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / BLOCK_TILE_SIZE_Y,
        1U);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    matmul_kernel_v4<<<grid, block>>>(A, B, C, m, n, k, alpha, beta);
    CUDACHECK(cudaGetLastError());
}