#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "timer.h"
#include "baseline.h"
#include "common.h"

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

    cublas_matmul(d_A, d_B, d_C, m, n, k);

    CUDACHECK(cudaMemcpy(C, d_C, static_cast<size_t>(m * n) * sizeof(__half), cudaMemcpyDeviceToHost));

    print_mat(C, m, n, "C");

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

int main(int argc, char *argv[])
{
    init_device();
    timer tim;
    const vector<vector<int>> shapes = get_shapes();
    tim.reset();
    check_correctness();
    tim.reset();
    double t = tim.reset() * 1000;
    cout << "elapsed time: " << t << " ms" << endl;

    return 0;
}