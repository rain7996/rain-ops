#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cuda_fp16.h>

using namespace std;

#define CUDACHECK(call)                                                             \
    do                                                                              \
    {                                                                               \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                     \
        {                                                                           \
            cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl;                                \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

#define CUBLASCHECK(call)                                                             \
    do                                                                                \
    {                                                                                 \
        cublasStatus_t status = call;                                                 \
        if (status != CUBLAS_STATUS_SUCCESS)                                          \
        {                                                                             \
            cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
                 << status << endl;                                                   \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

#define MIN(X, Y) ((X) > (Y) ? (Y) : (X))

#define CEIL_DIV(X, Y) (((X) + (Y) - 1) / (Y))

#define ALIGN_TO(X, Y) ((CEIL_DIV((X), (Y))) * (Y))

#define ABS(X, Y) ((((X) - (Y)) > 0) ? ((X) - (Y)) : ((Y) - (X)))

typedef enum
{
    RAND = 0,
    SEQ = 1,
    FIX = 2
} INIT_METHOD;

void init_device();

const vector<vector<int>> &get_shapes();

void get_single_shape(int *);

__half gen_rand_half();

void init_single_mat(__half *A, int row, int col, INIT_METHOD init_method, __half value);

void init_mat(__half *A, __half *B, int m, int n, int k, INIT_METHOD init_method, __half value = 0);

void print_mat(__half *M, int row, int col, string desc);

void mat_diff(__half *base, __half *target, int m, int n);