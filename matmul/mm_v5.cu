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

#define VECTOR_TYPE float4
#define NUM_VECTOR_UNITS (sizeof(VECTOR_TYPE) / sizeof(__half))
#define VECTORIZED_BLOCK_TILE_SIZE_K (BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS)
#define VECTORIZED_BLOCK_TILE_SIZE_X (BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS)

#define WARP_TILE_X 32U
#define WARP_TILE_Y 64U
#define NUM_THREADS_PER_WARP_X 4U
#define NUM_THREADS_PER_WARP_Y 8U
#define NUM_WARPS_X (BLOCK_TILE_SIZE_X / WARP_TILE_X)
#define NUM_WARPS_Y (BLOCK_TILE_SIZE_Y / WARP_TILE_Y)
#define NUM_TREHAD_WARP_REPEAT_X (WARP_TILE_X / (NUM_THREADS_PER_WARP_X * THREAD_TILE_SIZE_X))
#define NUM_TREHAD_WARP_REPEAT_Y (WARP_TILE_Y / (NUM_THREADS_PER_WARP_Y * THREAD_TILE_SIZE_Y))

#define NUM_TREHADS_X (NUM_WARPS_X * NUM_THREADS_PER_WARP_X)
#define NUM_TREHADS_Y (NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y)
#define NUM_THREADS (NUM_TREHADS_X * NUM_TREHADS_Y)

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
            __half *A_row_vals = reinterpret_cast<__half *>(&A_row_vec_val);
            if (A_r < m && A_c < k)
            {
                A_row_vec_val = *reinterpret_cast<const float4 *>(&A[A_r * k + A_c]);
                if (A_c + NUM_VECTOR_UNITS > k)
                {
                    size_t num_invalid_ele = A_c + NUM_VECTOR_UNITS - k;
                    for (size_t j = 0; j < num_invalid_ele; j++)
                    {
                        A_row_vals[NUM_VECTOR_UNITS - j - 1] = 0;
                    }
                }
            }

            if (S_A_r < BLOCK_TILE_SIZE_Y && S_A_c < BLOCK_TILE_SIZE_K)
            {
#pragma unroll
                for (size_t j = 0; j < NUM_VECTOR_UNITS; j++)
                {
                    A_shared_tile_trans[S_A_c + j][S_A_r] = A_row_vals[j];
                }
            }
        }

        // load B from DRAM to shared memory
        for (size_t i = 0; i < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X + NUM_THREADS - 1) / NUM_THREADS; i++)
        {
            const size_t S_B_r = (i * NUM_THREADS + thread_linear_idx) / VECTORIZED_BLOCK_TILE_SIZE_X;
            const size_t S_B_c = (i * NUM_THREADS + thread_linear_idx) % VECTORIZED_BLOCK_TILE_SIZE_X * NUM_VECTOR_UNITS;

            const size_t B_r = block_idx * BLOCK_TILE_SIZE_K + S_B_r;
            const size_t B_c = blockIdx.x * BLOCK_TILE_SIZE_X + S_B_c;

            float4 B_row_vec_val = {0, 0, 0, 0};
            __half *B_row_vals = reinterpret_cast<__half *>(&B_row_vec_val);

            if (B_r < k && B_c < n)
            {
                B_row_vec_val = *reinterpret_cast<const float4 *>(&B[B_r * n + B_c]);
                if (B_c + NUM_VECTOR_UNITS > n)
                {
                    size_t num_invalid_ele = B_c + NUM_VECTOR_UNITS - n;
                    for (size_t j = 0; j < num_invalid_ele; j++)
                    {
                        B_row_vals[NUM_VECTOR_UNITS - j - 1] = 0;
                    }
                }
            }

            if (S_B_r < BLOCK_TILE_SIZE_K && S_B_c < BLOCK_TILE_SIZE_X)
            {
                *reinterpret_cast<float4 *>(&B_shared_tile[S_B_r][S_B_c]) = B_row_vec_val;
            }
        }
    };
}

__global__ void matmul_kernel_v5(const __half *A, const __half *B, __half *C, size_t m, size_t n, size_t k, __half alpha, __half beta)
{
    const int thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ __half S_A_trans[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ __half S_B[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    float4 R_A[NUM_TREHAD_WARP_REPEAT_Y][THREAD_TILE_SIZE_Y / NUM_VECTOR_UNITS] = {0};
    float4 R_B[NUM_TREHAD_WARP_REPEAT_X][THREAD_TILE_SIZE_X / NUM_VECTOR_UNITS] = {0};
    float4 C_results[NUM_TREHAD_WARP_REPEAT_Y][NUM_TREHAD_WARP_REPEAT_X][THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X / NUM_VECTOR_UNITS] = {0};

    const size_t num_blocks = (k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K;

    const size_t warp_id = thread_linear_idx / 32;
    const size_t warp_r = warp_id / NUM_WARPS_X;
    const size_t warp_c = warp_id % NUM_WARPS_X;
    const size_t thread_id_in_warp = thread_linear_idx % 32;
    const size_t thread_id_in_warp_r = thread_id_in_warp / NUM_THREADS_PER_WARP_X;
    const size_t thread_id_in_warp_c = thread_id_in_warp % NUM_THREADS_PER_WARP_X;

    const size_t w_r_start_in_block = warp_r * WARP_TILE_Y;
    const size_t w_c_start_in_block = warp_c * WARP_TILE_X;

    for (size_t block_id = 0; block_id < num_blocks; block_id++)
    {
        load_data_to_shared_memory(A, B, S_A_trans, S_B, block_id, thread_linear_idx, m, n, k);
        __syncthreads();

        for (size_t k_i = 0; k_i < BLOCK_TILE_SIZE_K; k_i++)
        {
            for (size_t thread_r_repeat_in_warp = 0; thread_r_repeat_in_warp < NUM_TREHAD_WARP_REPEAT_Y; thread_r_repeat_in_warp++)
            {
                const size_t S_A_r_start = w_r_start_in_block + (thread_r_repeat_in_warp * NUM_THREADS_PER_WARP_Y + thread_id_in_warp_r) * THREAD_TILE_SIZE_Y;
                R_A[thread_r_repeat_in_warp][0] = *reinterpret_cast<float4 *>(&S_A_trans[k_i][S_A_r_start]);
            }

            for (size_t thread_c_repeat_in_warp = 0; thread_c_repeat_in_warp < NUM_TREHAD_WARP_REPEAT_X; thread_c_repeat_in_warp++)
            {

                const size_t S_B_c_start = w_c_start_in_block + (thread_c_repeat_in_warp * NUM_THREADS_PER_WARP_X + thread_id_in_warp_c) * THREAD_TILE_SIZE_X;
                R_B[thread_c_repeat_in_warp][0] = *reinterpret_cast<float4 *>(&S_B[k_i][S_B_c_start]);
            }

            for (size_t thread_r_repeat_in_warp = 0; thread_r_repeat_in_warp < NUM_TREHAD_WARP_REPEAT_Y; thread_r_repeat_in_warp++)
            {
                for (size_t thread_c_repeat_in_warp = 0; thread_c_repeat_in_warp < NUM_TREHAD_WARP_REPEAT_X; thread_c_repeat_in_warp++)
                {
                    for (size_t r = 0; r < THREAD_TILE_SIZE_Y; r++)
                    {
                        for (size_t c = 0; c < THREAD_TILE_SIZE_X; c++)
                        {
                            reinterpret_cast<__half *>(&C_results[thread_r_repeat_in_warp][thread_c_repeat_in_warp][r][0])[c] +=
                                (reinterpret_cast<__half *>(&R_A[thread_r_repeat_in_warp][0])[r]) * (reinterpret_cast<__half *>(&R_B[thread_c_repeat_in_warp][0])[c]);
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    for (size_t thread_r_repeat_in_warp = 0; thread_r_repeat_in_warp < NUM_TREHAD_WARP_REPEAT_Y; thread_r_repeat_in_warp++)
    {
        for (size_t thread_c_repeat_in_warp = 0; thread_c_repeat_in_warp < NUM_TREHAD_WARP_REPEAT_X; thread_c_repeat_in_warp++)
        {
            const size_t C_r_start = blockIdx.y * BLOCK_TILE_SIZE_Y + w_r_start_in_block + (thread_r_repeat_in_warp * NUM_THREADS_PER_WARP_Y + thread_id_in_warp_r) * THREAD_TILE_SIZE_Y;

            const size_t C_c_start = blockIdx.x * BLOCK_TILE_SIZE_X + w_c_start_in_block + (thread_c_repeat_in_warp * NUM_THREADS_PER_WARP_X + thread_id_in_warp_c) * THREAD_TILE_SIZE_X;

            for (size_t r = 0; r < THREAD_TILE_SIZE_Y; r++)
            {
                for (size_t c = 0; c < THREAD_TILE_SIZE_X; c += NUM_VECTOR_UNITS)
                {
                    const size_t C_r = C_r_start + r;
                    const size_t C_c = C_c_start + c;

                    if (C_r < m && C_c < n)
                    {

                        float4 C_vals = {0, 0, 0, 0};

                        if (beta != __float2half(0))
                        {
                            C_vals = *reinterpret_cast<float4 *>(&C[C_r * n + C_c]);
                        }

                        for (size_t i = 0; i < NUM_VECTOR_UNITS; i++)
                        {
                            (reinterpret_cast<__half *>(&C_vals))[i] = alpha * (reinterpret_cast<__half *>(&C_results[thread_r_repeat_in_warp][thread_c_repeat_in_warp][r][0]))[c + i] + beta * (reinterpret_cast<__half *>(&C_vals))[i];
                        }

                        *reinterpret_cast<float4 *>(&C[C_r * n + C_c]) = C_vals;
                    }
                }
            }
        }
    }
}

void matmul_v5(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle)
{
    assert(handle == nullptr);

    static_assert(sizeof(VECTOR_TYPE) % sizeof(__half) == 0);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0);

    static_assert(BLOCK_TILE_SIZE_Y * sizeof(__half) % sizeof(VECTOR_TYPE) == 0);
    static_assert(BLOCK_TILE_SIZE_X * sizeof(__half) % sizeof(VECTOR_TYPE) == 0);

    static_assert(THREAD_TILE_SIZE_Y == NUM_VECTOR_UNITS);
    static_assert(THREAD_TILE_SIZE_X == NUM_VECTOR_UNITS);

    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_X == 0);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_Y == 0);
    static_assert(NUM_THREADS_PER_WARP_X * THREAD_TILE_SIZE_X == WARP_TILE_X);
    static_assert(NUM_THREADS_PER_WARP_Y * THREAD_TILE_SIZE_Y == WARP_TILE_Y);
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    static_assert(WARP_TILE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
    static_assert(WARP_TILE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0);

    static_assert(THREAD_TILE_SIZE_X == NUM_VECTOR_UNITS);
    static_assert(THREAD_TILE_SIZE_Y == NUM_VECTOR_UNITS);

    // static_assert(BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K % NUM_THREADS == 0);
    // static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X % NUM_THREADS == 0);

    dim3 block(NUM_THREADS, 1U, 1U);
    dim3 grid(
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / BLOCK_TILE_SIZE_Y,
        1U);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    matmul_kernel_v5<<<grid, block>>>(A, B, C, m, n, k, alpha, beta);
    CUDACHECK(cudaGetLastError());
}