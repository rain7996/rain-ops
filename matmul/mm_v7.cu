/**
 * mm_v7: High-performance HGEMM approaching cuBLAS
 *
 * Key optimizations over mm_v6:
 *  1. mma.sync PTX (m16n8k16) instead of WMMA API — finer register control
 *  2. cp.async for global→shared without register detour
 *  3. Double-buffered shared memory — overlap compute with memory loads
 *  4. Larger block tile (128×256×32) for better data reuse
 *  5. ldmatrix / ldmatrix.trans for efficient shared→register loads
 *  6. FP32 accumulator for precision, convert to FP16 on store
 *
 * Matrix layout: A [M×K] row-major, B [K×N] row-major, C [M×N] row-major
 * Data type: __half (FP16), accumulator: float (FP32)
 *
 * Target arch: sm_80+ (Ampere / Hopper)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <assert.h>
#include <mma.h>

#include "mm.h"
#include "common.h"

// ============================================================================
// Tile configuration
// ============================================================================
static constexpr int BM = 128;
static constexpr int BN = 256;
static constexpr int BK = 32;

static constexpr int WM = 64;
static constexpr int WN = 64;

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;

static constexpr int WARPS_M = BM / WM;   // 2
static constexpr int WARPS_N = BN / WN;   // 4
static constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 8
static constexpr int NUM_THREADS = NUM_WARPS * 32;    // 256

static constexpr int WARP_MMA_M = WM / MMA_M;  // 4
static constexpr int WARP_MMA_N = WN / MMA_N;  // 8
static constexpr int WARP_MMA_K = BK / MMA_K;  // 2

static constexpr int NUM_STAGES = 2;

// Shared memory: pad stride to avoid bank conflicts
// A: [BM][BK], stored row-major, stride = BK + 8
// B: [BK][BN], stored row-major, stride = BN + 8
static constexpr int SMEM_A_STRIDE = BK + 8;        // 40
static constexpr int SMEM_B_STRIDE = BN + 8;        // 264

static constexpr int SMEM_A_SIZE = BM * SMEM_A_STRIDE;
static constexpr int SMEM_B_SIZE = BK * SMEM_B_STRIDE;
static constexpr int SMEM_STAGE_SIZE = SMEM_A_SIZE + SMEM_B_SIZE;

// ============================================================================
// PTX helpers
// ============================================================================

// cp.async: copy 16 bytes (8 halves) from global to shared.
// When pred is false, manually zero-fill the shared memory destination.
__device__ __forceinline__
void cp_async_cg_zfill(void *smem_ptr, const void *gmem_ptr, bool pred) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    if (pred) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_addr), "l"(gmem_ptr)
        );
    } else {
        // Zero-fill 16 bytes = 4 x uint32_t
        uint32_t *dst = reinterpret_cast<uint32_t *>(smem_ptr);
        dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0;
    }
}

__device__ __forceinline__
void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__
void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ldmatrix.x4: load 4 × (m8n8) = 16×16 half matrix from shared memory.
// Each thread provides the address of one 16-byte (8 half) row.
// Returns 4 uint32_t registers.
__device__ __forceinline__
void ldmatrix_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
                 const void *smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

// ldmatrix.x2.trans: load 2 × (m8n8) = 16×8 matrix from shared memory,
// with 8×8 block transposition. Essential for loading row-major B into
// col-major register layout expected by MMA.
__device__ __forceinline__
void ldmatrix_x2_trans(uint32_t &r0, uint32_t &r1, const void *smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
}

// mma.sync m16n8k16: D = A * B + C
// A: 4 × uint32_t (row-major), B: 2 × uint32_t (col-major), C/D: 4 × float
__device__ __forceinline__
void mma_m16n8k16(float *d, const uint32_t *a, const uint32_t *b, const float *c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// ============================================================================
// Global → Shared memory load
// ============================================================================

// Load A tile [BM × BK] from global (row-major M×K) to shared (row-major with stride)
// Each cp.async copies 16 bytes = 8 halves
// Total = BM*BK/8 = 128*32/8 = 512 copies, per thread = 512/256 = 2
__device__ __forceinline__
void load_A_g2s(const __half *A, __half *sA, int M, int K,
                int blk_row, int k_off, int tid) {
    constexpr int VEC = 8;  // halves per cp.async
    constexpr int LOADS_PER_ROW = BK / VEC;         // 4
    constexpr int TOTAL = (BM * BK) / VEC;          // 512
    constexpr int PER_THREAD = TOTAL / NUM_THREADS;  // 2

    #pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        int idx = tid + i * NUM_THREADS;
        int row = idx / LOADS_PER_ROW;
        int col = (idx % LOADS_PER_ROW) * VEC;

        int g_row = blk_row + row;
        int g_col = k_off + col;
        bool valid = (g_row < M) && (g_col + VEC - 1 < K);

        __half *dst = sA + row * SMEM_A_STRIDE + col;
        const __half *src = A + (size_t)g_row * K + g_col;

        cp_async_cg_zfill(dst, src, valid);
    }
}

// Load B tile [BK × BN] from global (row-major K×N) to shared (row-major with stride)
// Total = BK*BN/8 = 32*256/8 = 1024 copies, per thread = 1024/256 = 4
__device__ __forceinline__
void load_B_g2s(const __half *B, __half *sB, int K, int N,
                int k_off, int blk_col, int tid) {
    constexpr int VEC = 8;
    constexpr int LOADS_PER_ROW = BN / VEC;          // 32
    constexpr int TOTAL = (BK * BN) / VEC;           // 1024
    constexpr int PER_THREAD = TOTAL / NUM_THREADS;   // 4

    #pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        int idx = tid + i * NUM_THREADS;
        int row = idx / LOADS_PER_ROW;
        int col = (idx % LOADS_PER_ROW) * VEC;

        int g_row = k_off + row;
        int g_col = blk_col + col;
        bool valid = (g_row < K) && (g_col + VEC - 1 < N);

        __half *dst = sB + row * SMEM_B_STRIDE + col;
        const __half *src = B + (size_t)g_row * N + g_col;

        cp_async_cg_zfill(dst, src, valid);
    }
}

// ============================================================================
// Main kernel
// ============================================================================
__global__ void __launch_bounds__(NUM_THREADS)
matmul_kernel_v7(const __half * __restrict__ A,
                 const __half * __restrict__ B,
                 __half * __restrict__ C,
                 int M, int N, int K) {

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int warp_row = warp_id / WARPS_N;  // 0..1
    const int warp_col = warp_id % WARPS_N;  // 0..3

    const int blk_row = blockIdx.y * BM;
    const int blk_col = blockIdx.x * BN;

    // ------ Shared memory (dynamic, double-buffered) ------
    extern __shared__ __half smem_raw[];
    __half *sA[NUM_STAGES], *sB[NUM_STAGES];
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        sA[s] = smem_raw + s * SMEM_STAGE_SIZE;
        sB[s] = smem_raw + s * SMEM_STAGE_SIZE + SMEM_A_SIZE;
    }

    // ------ Register accumulators (FP32) ------
    float acc[WARP_MMA_M][WARP_MMA_N][4];
    #pragma unroll
    for (int i = 0; i < WARP_MMA_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_MMA_N; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    // ------ Prologue: prefetch into pipeline ------
    const int num_k_blocks = (K + BK - 1) / BK;

    // Stage 0
    load_A_g2s(A, sA[0], M, K, blk_row, 0, tid);
    load_B_g2s(B, sB[0], K, N, 0, blk_col, tid);
    cp_async_commit();

    // Stage 1
    if (num_k_blocks > 1) {
        load_A_g2s(A, sA[1], M, K, blk_row, BK, tid);
        load_B_g2s(B, sB[1], K, N, BK, blk_col, tid);
    }
    cp_async_commit();

    // ------ Main loop ------
    for (int kb = 0; kb < num_k_blocks; kb++) {
        // Wait for current stage data
        cp_async_wait_group<1>();
        __syncthreads();

        const int stage = kb % NUM_STAGES;
        const __half *curA = sA[stage];
        const __half *curB = sB[stage];

        // Prefetch next+2 tile
        const int next = kb + NUM_STAGES;
        if (next < num_k_blocks) {
            load_A_g2s(A, sA[next % NUM_STAGES], M, K, blk_row, next * BK, tid);
            load_B_g2s(B, sB[next % NUM_STAGES], K, N, next * BK, blk_col, tid);
        }
        cp_async_commit();

        // ---- Compute: 2 k-steps (BK/MMA_K = 32/16 = 2) ----
        #pragma unroll
        for (int kk = 0; kk < WARP_MMA_K; kk++) {

            // ---- Load A fragments via ldmatrix.x4 ----
            //
            // ldmatrix.sync.aligned.m8n8.x4 loads 4 sub-matrices of size 8x8
            // from row-major shared memory. The 32 threads are divided into
            // 4 groups of 8:
            //   group g = lane_id / 8   (g = 0,1,2,3)
            //   within-group id = lane_id % 8
            //
            // Sub-matrix layout for a 16x16 tile:
            //   group 0 → rows  0..7,  cols  0..7
            //   group 1 → rows  8..15, cols  0..7
            //   group 2 → rows  0..7,  cols  8..15
            //   group 3 → rows  8..15, cols  8..15
            //
            // Each thread provides the address of 8 contiguous halves (16 bytes)
            // at its row in the sub-matrix.
            //
            // Address for thread t:
            //   row = (lane_id % 8) + (group % 2) * 8
            //       = (lane_id % 8) + ((lane_id / 8) & 1) * 8
            //   col = (group / 2) * 8
            //       = (lane_id / 16) * 8

            uint32_t A_frag[WARP_MMA_M][4];

            #pragma unroll
            for (int mi = 0; mi < WARP_MMA_M; mi++) {
                int a_smem_row = warp_row * WM + mi * MMA_M
                                 + (lane_id % 8) + ((lane_id / 8) & 1) * 8;
                int a_smem_col = kk * MMA_K + (lane_id / 16) * 8;
                const __half *a_ptr = curA + a_smem_row * SMEM_A_STRIDE + a_smem_col;
                ldmatrix_x4(A_frag[mi][0], A_frag[mi][1],
                            A_frag[mi][2], A_frag[mi][3], a_ptr);
            }

            // ---- Load B fragments via ldmatrix.x2.trans ----
            //
            // B is stored row-major in smem: sB[k][n] with stride SMEM_B_STRIDE.
            // MMA expects B in col-major layout.
            // ldmatrix.x2.trans reads 2 × (8×8) row-major blocks and transposes
            // each, producing col-major register layout.
            //
            // For a 16×8 B tile:
            //   group 0 (lane 0..7)   → rows 0..7   of the tile
            //   group 1 (lane 8..15)  → rows 8..15  of the tile
            //   groups 2,3 (lane 16..31) → same as groups 0,1
            //
            // Address for thread t:
            //   row = (lane_id % 8) + ((lane_id / 8) & 1) * 8
            //   col = 0 (tile is only 8 wide)

            uint32_t B_frag[WARP_MMA_N][2];

            #pragma unroll
            for (int nj = 0; nj < WARP_MMA_N; nj++) {
                int b_smem_row = kk * MMA_K
                                 + (lane_id % 8) + ((lane_id / 8) & 1) * 8;
                int b_smem_col = warp_col * WN + nj * MMA_N;
                const __half *b_ptr = curB + b_smem_row * SMEM_B_STRIDE + b_smem_col;
                ldmatrix_x2_trans(B_frag[nj][0], B_frag[nj][1], b_ptr);
            }

            // ---- MMA ----
            #pragma unroll
            for (int mi = 0; mi < WARP_MMA_M; mi++)
                #pragma unroll
                for (int nj = 0; nj < WARP_MMA_N; nj++)
                    mma_m16n8k16(acc[mi][nj], A_frag[mi], B_frag[nj], acc[mi][nj]);
        }

        __syncthreads();
    }

    // Wait for all async copies to complete (might have trailing commit)
    cp_async_wait_group<0>();
    __syncthreads();

    // ------ Epilogue: store C ------
    //
    // mma.m16n8k16 output register layout (D matrix, 16×8):
    //   groupID = lane_id / 4        (0..7)
    //   threadID_in_group = lane_id % 4
    //
    //   d[0] → D[groupID    ][threadID_in_group * 2]
    //   d[1] → D[groupID    ][threadID_in_group * 2 + 1]
    //   d[2] → D[groupID + 8][threadID_in_group * 2]
    //   d[3] → D[groupID + 8][threadID_in_group * 2 + 1]

    const int groupID = lane_id / 4;
    const int tidInGroup = lane_id % 4;

    #pragma unroll
    for (int mi = 0; mi < WARP_MMA_M; mi++) {
        #pragma unroll
        for (int nj = 0; nj < WARP_MMA_N; nj++) {
            int tile_row = blk_row + warp_row * WM + mi * MMA_M;
            int tile_col = blk_col + warp_col * WN + nj * MMA_N;

            // Rows [0..7]: d[0], d[1]
            {
                int r = tile_row + groupID;
                int c = tile_col + tidInGroup * 2;
                if (r < M && c + 1 < N) {
                    __half2 val = __halves2half2(
                        __float2half(acc[mi][nj][0]),
                        __float2half(acc[mi][nj][1]));
                    *reinterpret_cast<__half2 *>(&C[(size_t)r * N + c]) = val;
                } else if (r < M && c < N) {
                    C[(size_t)r * N + c] = __float2half(acc[mi][nj][0]);
                }
            }

            // Rows [8..15]: d[2], d[3]
            {
                int r = tile_row + groupID + 8;
                int c = tile_col + tidInGroup * 2;
                if (r < M && c + 1 < N) {
                    __half2 val = __halves2half2(
                        __float2half(acc[mi][nj][2]),
                        __float2half(acc[mi][nj][3]));
                    *reinterpret_cast<__half2 *>(&C[(size_t)r * N + c]) = val;
                } else if (r < M && c < N) {
                    C[(size_t)r * N + c] = __float2half(acc[mi][nj][2]);
                }
            }
        }
    }
}

// ============================================================================
// Host wrapper
// ============================================================================
void matmul_v7(const __half *A, const __half *B, __half *C,
               int m, int n, int k, cublasHandle_t handle) {
    assert(handle == nullptr);

    dim3 block(NUM_THREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);

    constexpr int smem_bytes = NUM_STAGES * SMEM_STAGE_SIZE * sizeof(__half);

    cudaFuncSetAttribute(matmul_kernel_v7,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);

    matmul_kernel_v7<<<grid, block, smem_bytes>>>(A, B, C, m, n, k);
    CUDACHECK(cudaGetLastError());
}
