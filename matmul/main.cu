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
    mm.check_correctness(matmul_v3, true, "mm_v3");
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

    int loop = 100;

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
    }

    cout << "m\t" << "n\t" << "k\t"
         << "cub_t(ms)\t" << "cub_tflops\t"
         << "mm_v0_t(ms)\t" << "mm_v0_tflops\t"
         << "mm_v1_t(ms)\t" << "mm_v1_tflops\t"
         << "mm_v2_t(ms)\t" << "mm_v2_tflops\t"
         << "mm_v3_t(ms)\t" << "mm_v3_tflops\t"
         << endl;

    for (int i = 0; i < n; i++)
    {
        result &cub_r = mm_cub_results[i];
        result &mm_v0_r = mm_v0_results[i];
        result &mm_v1_r = mm_v1_results[i];
        result &mm_v2_r = mm_v2_results[i];
        result &mm_v3_r = mm_v3_results[i];

        cout << cub_r.m << '\t' << cub_r.n << '\t' << cub_r.k << '\t'
             << cub_r.t * 1000 << "\t\t" << cub_r.tflops << "\t\t"
             << mm_v0_r.t * 1000 << "\t\t" << mm_v0_r.tflops << "\t\t"
             << mm_v1_r.t * 1000 << "\t\t" << mm_v1_r.tflops << "\t\t"
             << mm_v2_r.t * 1000 << "\t\t" << mm_v2_r.tflops << "\t\t"
             << mm_v3_r.t * 1000 << "\t\t" << mm_v3_r.tflops << "\t\t"
             << endl;
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