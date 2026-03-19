#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <iomanip>
#include <assert.h>

#include "timer.h"
#include "common.h"
#include "mm.h"
#include "manager.h"

using namespace std;

void check()
{
    int shape[3];
    get_single_shape(shape);

    MatManager mm(shape[0], shape[1], shape[2], 1);

    mm.check_correctness(matmul_v0, true, "mm_v0");
    mm.check_correctness(matmul_v1, false, "mm_v1");
    mm.check_correctness(matmul_v2, false, "mm_v2");
    mm.check_correctness(matmul_v3, false, "mm_v3");
    mm.check_correctness(matmul_v4, false, "mm_v4");
    mm.check_correctness(matmul_v5, false, "mm_v5");
    mm.check_correctness(matmul_v6, true, "mm_v6");
    mm.check_correctness(matmul_v7, true, "mm_v7");
}

void benchmark()
{
    vector<vector<int>> shapes = get_shapes();

    int n = shapes.size();
    vector<result> mm_cub_results;
    vector<result> mm_v0_results;
    vector<result> mm_v1_results;
    vector<result> mm_v2_results;
    vector<result> mm_v3_results;
    vector<result> mm_v4_results;
    vector<result> mm_v5_results;
    vector<result> mm_v6_results;
    vector<result> mm_v7_results;

    int loop = 20;

    for (auto shape : shapes)
    {
        MatManager mm(shape[0], shape[1], shape[2], loop);
        mm_cub_results.push_back(mm.benchmark_single_shape(cublas_matmul, true));
        mm_v0_results.push_back(mm.benchmark_single_shape(matmul_v0, false));

        unsigned int v1_buffer_size = MM_V1_BLOCK_SIZE * shape[2] * sizeof(__half) * 2;
        if (v1_buffer_size >= TOTAL_SHARED_MEM_SiZE)
        {
            cerr << "[mm_v1]failed to allocate shared memory, need " << v1_buffer_size << " KB, but " << TOTAL_SHARED_MEM_SiZE << " KB in total" << endl;
            mm_v1_results.push_back(result{
                shape[0],
                shape[1],
                shape[2],
                0,
                0,
            });
        }
        else
        {
            mm_v1_results.push_back(mm.benchmark_single_shape(matmul_v1, false));
        }

        mm_v2_results.push_back(mm.benchmark_single_shape(matmul_v2, false));
        mm_v3_results.push_back(mm.benchmark_single_shape(matmul_v3, false));
        mm_v4_results.push_back(mm.benchmark_single_shape(matmul_v4, false));
        mm_v5_results.push_back(mm.benchmark_single_shape(matmul_v5, false));
        mm_v6_results.push_back(mm.benchmark_single_shape(matmul_v6, false));
        mm_v7_results.push_back(mm.benchmark_single_shape(matmul_v7, false));
    }

    std::cout << std::left
              << std::setw(8) << "m"
              << std::setw(8) << "n"
              << std::setw(8) << "k"
              << std::setw(10) << "cub_t(ms)"
              << std::setw(12) << "cub_tflops"
              << std::setw(12) << "mm_v0_t(ms)"
              << std::setw(12) << "mm_v0_tflops"
              << std::setw(12) << "mm_v1_t(ms)"
              << std::setw(12) << "mm_v1_tflops"
              << std::setw(12) << "mm_v2_t(ms)"
              << std::setw(12) << "mm_v2_tflops"
              << std::setw(12) << "mm_v3_t(ms)"
              << std::setw(12) << "mm_v3_tflops"
              << std::setw(12) << "mm_v4_t(ms)"
              << std::setw(12) << "mm_v4_tflops"
              << std::setw(12) << "mm_v5_t(ms)"
              << std::setw(12) << "mm_v5_tflops"
              << std::setw(12) << "mm_v6_t(ms)"
              << std::setw(12) << "mm_v6_tflops"
              << std::setw(12) << "mm_v7_t(ms)"
              << std::setw(12) << "mm_v7_tflops"
              << '\n';

    for (int i = 0; i < n; i++)
    {
        result &cub_r = mm_cub_results[i];
        result &mm_v0_r = mm_v0_results[i];
        result &mm_v1_r = mm_v1_results[i];
        result &mm_v2_r = mm_v2_results[i];
        result &mm_v3_r = mm_v3_results[i];
        result &mm_v4_r = mm_v4_results[i];
        result &mm_v5_r = mm_v5_results[i];
        result &mm_v6_r = mm_v6_results[i];
        result &mm_v7_r = mm_v7_results[i];

        std::cout
            << std::setw(8) << cub_r.m << std::setw(8) << cub_r.n << std::setw(8) << cub_r.k
            << std::setw(10) << cub_r.t * 1000.0 << std::setw(12) << cub_r.tflops
            << std::setw(12) << mm_v0_r.t * 1000.0 << std::setw(12) << mm_v0_r.tflops
            << std::setw(12) << mm_v1_r.t * 1000.0 << std::setw(12) << mm_v1_r.tflops
            << std::setw(12) << mm_v2_r.t * 1000.0 << std::setw(12) << mm_v2_r.tflops
            << std::setw(12) << mm_v3_r.t * 1000.0 << std::setw(12) << mm_v3_r.tflops
            << std::setw(12) << mm_v4_r.t * 1000.0 << std::setw(12) << mm_v4_r.tflops
            << std::setw(12) << mm_v5_r.t * 1000.0 << std::setw(12) << mm_v5_r.tflops
            << std::setw(12) << mm_v6_r.t * 1000.0 << std::setw(12) << mm_v6_r.tflops
            << std::setw(12) << mm_v7_r.t * 1000.0 << std::setw(12) << mm_v7_r.tflops
            << '\n';
    }
}

int main(int argc, char *argv[])
{
    init_device();
    cout << fixed << setprecision(2);

    check();

    benchmark();

    return 0;
}