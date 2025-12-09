#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <iomanip>
#include <assert.h>

#include "timer.h"
#include "baseline.h"
#include "common.h"
#include "mm.h"

using namespace std;

bool check_correctness()
{

    CUDACHECK(cudaDeviceSynchronize());
    int *shape = new int[3];
    get_single_shape(shape);

    int m = shape[0], n = shape[1], k = shape[2];

    __half *A = new __half[m * k];
    __half *B = new __half[k * n];
    __half *C = new __half[m * n];

    INIT_METHOD im = SEQ;
    init_mat(A, B, C, m, n, k, im, 1);
    print_mat(A, m, k, "A");
    print_mat(B, k, n, "B");

    __half *d_A, *d_B, *d_C;
    CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), static_cast<size_t>(m * k) * sizeof(__half)));
    CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), static_cast<size_t>(n * k) * sizeof(__half)));
    CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), static_cast<size_t>(m * n) * sizeof(__half)));

    CUDACHECK(cudaMemcpy(d_A, A, m * k * sizeof(__half), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, B, n * k * sizeof(__half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLASCHECK(cublasCreate(&handle));
    CUBLASCHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    cublas_matmul(d_A, d_B, d_C, m, n, k, handle);
    CUDACHECK(cudaMemcpy(C, d_C, static_cast<size_t>(m * n) * sizeof(__half), cudaMemcpyDeviceToHost));
    print_mat(C, m, n, "cublas_matmul");

    matmul(d_A, d_B, d_C, m, n, k);
    CUDACHECK(cudaMemcpy(C, d_C, static_cast<size_t>(m * n) * sizeof(__half), cudaMemcpyDeviceToHost));
    print_mat(C, m, n, "mm_v0");

    CUBLASCHECK(cublasDestroy(handle));
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] shape;

    CUDACHECK(cudaFree(reinterpret_cast<void *>(d_A)));
    CUDACHECK(cudaFree(reinterpret_cast<void *>(d_B)));
    CUDACHECK(cudaFree(reinterpret_cast<void *>(d_C)));

    CUDACHECK(cudaDeviceSynchronize());

    return true;
}

typedef struct
{
    int m;
    int n;
    int k;
    double t;
    double gflops;
} result;

vector<result> benchmark(MM_ALG alg, int loop, string desc)
{
    const vector<vector<int>> shapes = get_shapes();
    int n_shapes = shapes.size();
    vector<result> ret{};

    cublasHandle_t handle;
    if (alg == CUBLAS)
    {
        CUBLASCHECK(cublasCreate(&handle));
        CUBLASCHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    }

    for (int i = 0; i < n_shapes; i++)
    {
        const vector<int> &shape = shapes[i];
        int m = shape[0], n = shape[1], k = shape[2];

        __half *A = new __half[m * k];
        __half *B = new __half[k * n];
        __half *C = new __half[m * n];

        INIT_METHOD im = RAND;
        init_mat(A, B, C, m, n, k, im);

        __half *d_A, *d_B, *d_C;
        CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), static_cast<size_t>(m * k) * sizeof(__half)));
        CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), static_cast<size_t>(n * k) * sizeof(__half)));
        CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), static_cast<size_t>(m * n) * sizeof(__half)));

        CUDACHECK(cudaMemcpy(d_A, A, m * k * sizeof(__half), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_B, B, n * k * sizeof(__half), cudaMemcpyHostToDevice));

        timer tim;
        CUDACHECK(cudaDeviceSynchronize());
        tim.reset();
        for (int i = 0; i < loop; i++)
        {
            switch (alg)
            {
            case CUBLAS:
                cublas_matmul(d_A, d_B, d_C, m, n, k, handle);
                break;
            case MM_V0:
                matmul(d_A, d_B, d_C, m, n, k);
                break;
            default:
                cerr << "invalid matmul algorithm" << endl;
                exit(-1);
            }
        }
        CUDACHECK(cudaDeviceSynchronize());
        double t = tim.reset();

        double gflops = (double(2.0) * m * n * k) * loop / t / 1e12;
        result single_ret{
            m,
            n,
            k,
            t,
            gflops,
        };

        ret.push_back(single_ret);

        delete[] A;
        delete[] B;
        delete[] C;

        CUDACHECK(cudaFree(reinterpret_cast<void *>(d_A)));
        CUDACHECK(cudaFree(reinterpret_cast<void *>(d_B)));
        CUDACHECK(cudaFree(reinterpret_cast<void *>(d_C)));
    }

    if (alg == CUBLAS)
    {
        CUBLASCHECK(cublasDestroy(handle));
    }
    return ret;
}

int main(int argc, char *argv[])
{
    init_device();
    cout << fixed << setprecision(2);
    timer tim;
    const vector<vector<int>> shapes = get_shapes();
    check_correctness();

    cout << "start warming up" << endl;
    benchmark(CUBLAS, 20, "warm_up");
    cout << "finish warming up" << endl;
    int loop = 20;
    vector<result> cub_results = benchmark(CUBLAS, loop, "cublas_matmul");
    vector<result> mm_v0_results = benchmark(MM_V0, loop, "mm_v0");
    int n_results = shapes.size();
    assert(n_results == cub_results.size());
    assert(n_results == mm_v0_results.size());

    for (int i = 0; i < n_results; i++)
    {
        result &cub_r = cub_results[i];
        result &mm_v0_r = mm_v0_results[i];

        cout << "m\t" << "n\t" << "k\t" << "cub_t(ms)\t" << "cub_tflops\t" << "mm_t(ms)\t" << "mm_v0_gflops" << endl;
        cout << cub_r.m << '\t' << cub_r.n << '\t' << cub_r.k << '\t'
             << cub_r.t * 1000 << "\t\t" << cub_r.gflops << "\t\t"
             << mm_v0_r.t * 1000 << "\t\t" << mm_v0_r.gflops << endl;
    }

    return 0;
}