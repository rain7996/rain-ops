#include <iostream>
#include <vector>
#include <random>
#include <string>

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

void init_device();

const vector<vector<int>> &get_shapes();

void get_single_shape(int *);

__half gen_rand_half();

void init_single_mat(__half *A, int row, int col, bool rand, __half value);

void init_mat(__half *A, __half *B, __half *C, int m, int n, int k, bool rand, __half value);

void print_mat(__half *M, int row, int col, string desc);