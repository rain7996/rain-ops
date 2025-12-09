#pragma once

#include <cuda_fp16.h>

void matmul(const __half *A, const __half *B, __half *C, int m, int n, int k);