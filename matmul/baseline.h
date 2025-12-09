#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>

void cublas_matmul(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle);