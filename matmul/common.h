#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cuda_fp16.h>

using namespace std;

#define CUDACHECK(call)                                                                  \
    do                                                                                   \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess)                                                          \
        {                                                                                \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                           \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

#define CUBLASCHECK(call)                                                                  \
    do                                                                                     \
    {                                                                                      \
        cublasStatus_t status = call;                                                      \
        if (status != CUBLAS_STATUS_SUCCESS)                                               \
        {                                                                                  \
            std::cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << status << std::endl;                                              \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

typedef enum
{
    RAND = 0,
    SEQ = 1,
    FIX = 2
} INIT_METHOD;

typedef enum
{
    CUBLAS = 0,
    MM_V0 = 1,
} MM_ALG;

void init_device();

const vector<vector<int>> &get_shapes();

void get_single_shape(int *);

__half gen_rand_half();

void init_single_mat(__half *A, int row, int col, INIT_METHOD init_method, __half value);

void init_mat(__half *A, __half *B, __half *C, int m, int n, int k, INIT_METHOD init_method, __half value = 0);

void print_mat(__half *M, int row, int col, string desc);