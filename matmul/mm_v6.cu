#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <assert.h>
#include <mma.h>

#include "mm.h"
#include "common.h"

using namespace std;
using namespace nvcuda;

#define BLOCK_TILE_SIZE_X 128U
#define BLOCK_TILE_SIZE_Y 64U
#define BLOCK_TILE_SIZE_K 16U

#define BLOCK_TILE_SKEW_SIZE_X 8U
#define BLOCK_TILE_SKEW_SIZE_Y 8U

#define WARP_X 32U
#define WARP_Y 64U
#define NUM_WARPS_X (BLOCK_TILE_SIZE_X / WARP_X)
#define NUM_WARPS_Y (BLOCK_TILE_SIZE_Y / WARP_Y)
#define WMMA_TILE_SIZE_X 16U
#define WMMA_TILE_SIZE_Y 16U
#define WMMA_TILE_SIZE_K 16U
#define NUM_THREADS (NUM_WARPS_X * NUM_WARPS_Y * 32)
#define NUM_WMMA_TILES_X (WARP_X / WMMA_TILE_SIZE_X)
#define NUM_WMMA_TILES_Y (WARP_Y / WMMA_TILE_SIZE_Y)
#define NUM_WMMA_TILES_K (BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K)

#define VECTOR_TYPE float4
#define VECTOR_SIZE (sizeof(VECTOR_TYPE))
#define NUM_VECTOR_UNITS (VECTOR_SIZE / sizeof(__half))
#define VECTORIZED_BLOCK_TILE_SIZE_K (BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS)
#define VECTORIZED_BLOCK_TILE_SIZE_X (BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS)

namespace
{

    __device__ void load_data_to_shared_memory(const __half *A, const __half *B,
                                               __half A_shared_tile_trans[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y],
                                               __half B_shared_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
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

__global__ void matmul_kernel_v6(const __half *A, const __half *B, __half *C, size_t m, size_t n, size_t k, __half alpha, __half beta)
{
    size_t thread_linear_idx = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ __half S_A_trans[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y];
    __shared__ __half S_B[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X];

    size_t num_blocks = (k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K;

    wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, __half, wmma::col_major> a_frags[NUM_WMMA_TILES_Y];
    wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, __half, wmma::row_major> b_frags[NUM_WMMA_TILES_X];
    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, __half> acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X];
    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, __half> c_frag;

#pragma unroll
    for (size_t wmma_row_id{0U}; wmma_row_id < NUM_WMMA_TILES_Y; wmma_row_id++)
    {
        for (size_t wmma_col_id{0U}; wmma_col_id < NUM_WMMA_TILES_X; wmma_col_id++)
        {
            wmma::fill_fragment(
                acc_frags[wmma_row_id][wmma_col_id],
                static_cast<__half>(0));
        }
    }

    size_t warp_idx = thread_linear_idx / 32;
    size_t warp_row = warp_idx / NUM_WARPS_X;
    size_t warp_col = warp_idx % NUM_WARPS_X;
    for (size_t bid = 0; bid < num_blocks; bid++)
    {
        load_data_to_shared_memory(A, B, S_A_trans, S_B, bid, thread_linear_idx, m, n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i = 0; k_i < NUM_WMMA_TILES_K; k_i++)
        {
#pragma unroll
            for (size_t wmma_tile_row{0U}; wmma_tile_row < NUM_WMMA_TILES_Y; wmma_tile_row++)
            {
                wmma::load_matrix_sync(a_frags[wmma_tile_row], &S_A_trans[k_i * WMMA_TILE_SIZE_K][warp_row * WARP_Y + wmma_tile_row * WMMA_TILE_SIZE_Y], BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y);
            }
#pragma unroll
            for (size_t wmma_tile_col{0U}; wmma_tile_col < NUM_WMMA_TILES_X; wmma_tile_col++)
            {
                wmma::load_matrix_sync(b_frags[wmma_tile_col], &S_B[k_i * WMMA_TILE_SIZE_K][warp_col * WARP_X + wmma_tile_col * WMMA_TILE_SIZE_X], BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X);
            }

#pragma unroll
            for (size_t wmma_tile_row{0U}; wmma_tile_row < NUM_WMMA_TILES_Y; wmma_tile_row++)
            {
                for (size_t wmma_tile_col{0U}; wmma_tile_col < NUM_WMMA_TILES_X; wmma_tile_col++)
                {
                    wmma::mma_sync(acc_frags[wmma_tile_row][wmma_tile_col], a_frags[wmma_tile_row], b_frags[wmma_tile_col], acc_frags[wmma_tile_row][wmma_tile_col]);
                }
            }
        }
        __syncthreads();
    }

    size_t C_r_start = blockIdx.y * BLOCK_TILE_SIZE_Y + warp_row * WARP_Y;
    size_t C_c_start = blockIdx.x * BLOCK_TILE_SIZE_X + warp_col * WARP_X;
#pragma unroll
    for (size_t wmma_tile_row{0U}; wmma_tile_row < NUM_WMMA_TILES_Y; wmma_tile_row++)
    {
#pragma unroll
        for (size_t wmma_tile_col{0U}; wmma_tile_col < NUM_WMMA_TILES_X; wmma_tile_col++)
        {
            size_t C_r = C_r_start + wmma_tile_row * WMMA_TILE_SIZE_Y;
            size_t C_c = C_c_start + wmma_tile_col * WMMA_TILE_SIZE_X;

            if (C_r < m && C_c < n)
            {
                if (beta != __float2half(0))
                {
                    wmma::load_matrix_sync(c_frag, &C[C_r * n + C_c], n, wmma::mem_row_major);
                }
                else
                {
                    wmma::fill_fragment(c_frag, __float2half(0));
                }

                for (size_t i{0}; i < c_frag.num_elements; i++)
                {
                    c_frag.x[i] = alpha * acc_frags[wmma_tile_row][wmma_tile_col].x[i] + beta * c_frag.x[i];
                }

                wmma::store_matrix_sync(&C[C_r * n + C_c], c_frag, n, wmma::mem_row_major);
            }
        }
    }
}

void matmul_v6(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle)
{
    assert(handle == nullptr);

    static_assert(BLOCK_TILE_SIZE_X % WARP_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_Y == 0U);

    static_assert(BLOCK_TILE_SIZE_K % sizeof(VECTOR_TYPE) == 0);
    static_assert(BLOCK_TILE_SIZE_X % sizeof(VECTOR_TYPE) == 0);

    static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(__half) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y) * sizeof(__half) % sizeof(VECTOR_TYPE) == 0U);

    dim3 block(NUM_THREADS, 1U, 1U);
    dim3 grid(
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / BLOCK_TILE_SIZE_Y,
        1U);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    matmul_kernel_v6<<<grid, block>>>(A, B, C, m, n, k, alpha, beta);
    CUDACHECK(cudaGetLastError());
}