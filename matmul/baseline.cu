#include <iostream>
#include <cublas_v2.h>

#include "common.h"
using namespace std;

void cublas_matmul(const __half *d_A, const __half *d_B, __half *d_C, int m, int n, int k)
{
    cublasHandle_t handle;
    CUBLASCHECK(cublasCreate(&handle));
    CUBLASCHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    cudaDataType Atype = CUDA_R_16F;
    cudaDataType Btype = CUDA_R_16F;
    cudaDataType Ctype = CUDA_R_16F;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLASCHECK(cublasSgemmEx(
        handle,
        transb, transa,
        n, m, k,
        &alpha,
        d_B, Btype, n,
        d_A, Atype, k,
        &beta,
        d_C, Ctype, m));

    return;
}