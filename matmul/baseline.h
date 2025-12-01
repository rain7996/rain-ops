#pragma once
#include <cublas_v2.h>

void cublas_matmul(const __half *A, const __half *B, __half *C, int m, int n, int k);