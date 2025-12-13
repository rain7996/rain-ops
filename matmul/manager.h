#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <iomanip>
#include <assert.h>

#include "timer.h"
#include "common.h"
#include "mm.h"

using mm_func = void (*)(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle);

typedef struct
{
    int m;
    int n;
    int k;
    double t;
    double tflops;
} result;

class MatManager
{
private:
    int m, n, k;
    int loop;

    __half *A;
    __half *B;
    __half *base_C;
    __half *mm_C;

    __half *d_A;
    __half *d_B;
    __half *d_C;

    cublasHandle_t handle;
    bool base_inited = false;

public:
    MatManager(int m, int n, int k, int loop);
    ~MatManager();
    void check_correctness(mm_func func, bool need_print, string desc);
    result benchmark_single_shape(mm_func func, bool isCublas);
};