#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <iomanip>
#include <assert.h>

#include "timer.h"
#include "common.h"
#include "mm.h"
#include "manager.h"

MatManager::MatManager(int m, int n, int k, int loop)
{
    this->m = m;
    this->n = n;
    this->k = k;
    this->loop = loop;

    // malloc pinned memory
    CUDACHECK(cudaMallocHost(reinterpret_cast<void **>(&A), m * k * sizeof(__half)));
    CUDACHECK(cudaMallocHost(reinterpret_cast<void **>(&B), k * n * sizeof(__half)));
    CUDACHECK(cudaMallocHost(reinterpret_cast<void **>(&base_C), m * n * sizeof(__half)));
    CUDACHECK(cudaMallocHost(reinterpret_cast<void **>(&mm_C), m * n * sizeof(__half)));

    INIT_METHOD im = RAND;
    init_mat(A, B, m, n, k, im, 1);
    memset(base_C, 0, m * n * sizeof(__half));
    memset(mm_C, 0, m * n * sizeof(__half));

    CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), static_cast<size_t>(m * k) * sizeof(__half)));
    CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), static_cast<size_t>(n * k) * sizeof(__half)));
    CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), static_cast<size_t>(m * n) * sizeof(__half)));

    CUDACHECK(cudaMemcpy(d_A, A, m * k * sizeof(__half), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, B, n * k * sizeof(__half), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_C, 0, m * n * sizeof(__half)));

    CUBLASCHECK(cublasCreate(&handle));
    CUBLASCHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    CUDACHECK(cudaDeviceSynchronize());
};

MatManager::~MatManager()
{
    CUBLASCHECK(cublasDestroy(handle));
    CUDACHECK(cudaFreeHost(A));
    CUDACHECK(cudaFreeHost(B));
    CUDACHECK(cudaFreeHost(base_C));
    CUDACHECK(cudaFreeHost(mm_C));

    CUDACHECK(cudaFree(reinterpret_cast<void *>(d_A)));
    CUDACHECK(cudaFree(reinterpret_cast<void *>(d_B)));
    CUDACHECK(cudaFree(reinterpret_cast<void *>(d_C)));

    CUDACHECK(cudaDeviceSynchronize());
}

void MatManager::check_correctness(mm_func func, bool need_print, string desc)
{
    if (!base_inited)
    {
        CUDACHECK(cudaMemset(d_C, 0, m * n * sizeof(__half)));
        cublas_matmul(d_A, d_B, d_C, m, n, k, handle);
        CUDACHECK(cudaMemcpy(base_C, d_C, m * n * sizeof(__half), cudaMemcpyDeviceToHost));
        base_inited = true;
        if (need_print)
        {
            print_mat(base_C, m, n, "cublas");
        }
    }

    memset(mm_C, 0, m * n * sizeof(__half));
    CUDACHECK(cudaMemset(d_C, 0, m * n * sizeof(__half)));
    func(d_A, d_B, d_C, m, n, k, nullptr);
    CUDACHECK(cudaMemcpy(mm_C, d_C, m * n * sizeof(__half), cudaMemcpyDeviceToHost));

    mat_diff(base_C, mm_C, m, n);
    CUDACHECK(cudaDeviceSynchronize());

    if (need_print)
    {
        print_mat(mm_C, m, n, desc);
    }
}

result MatManager::benchmark_single_shape(mm_func func, bool isCublas)
{
    const vector<vector<int>> shapes = get_shapes();
    int n_shapes = shapes.size();

    timer tim;
    CUDACHECK(cudaDeviceSynchronize());
    tim.reset();
    for (int i = 0; i < loop; i++)
    {
        if (!isCublas)
        {
            func(d_A, d_B, d_C, m, n, k, nullptr);
        }
        else
        {
            cublas_matmul(d_A, d_B, d_C, m, n, k, handle);
        }
    }
    CUDACHECK(cudaDeviceSynchronize());
    double t = tim.reset();

    double tflops = (double(2.0) * m * n * k) * loop / t / 1e12;
    result ret{
        m,
        n,
        k,
        t,
        tflops,
    };
    return ret;
}
