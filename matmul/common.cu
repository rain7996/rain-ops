#include <cublas.h>
#include <vector>
#include <random>
#include <iostream>
#include <cuda_fp16.h>

#include "common.h"

using namespace std;

namespace
{
    const vector<vector<int>> dimensions = {
        vector<int>{1024, 2048, 1024},
    };
}

void init_device()
{
    CUDACHECK(cudaSetDevice(0));
}

const vector<vector<int>> &get_shapes()
{
    return dimensions;
}

void get_single_shape(int *shape)
{
    shape[0] = 2; // m
    shape[1] = 2; // n
    shape[2] = 3; // k
}

__half gen_rand_half()
{
    static mt19937 rng{random_device{}()};
    static uniform_real_distribution<float> dist(0, 1);
    float num = dist(rng);
    return __float2half(num);
}

void init_single_mat(__half *A, int row, int col, INIT_METHOD init_method, __half value = 0)
{
    __half a = 0;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            switch (init_method)
            {
            case RAND:
                A[i * col + j] = gen_rand_half();
                break;
            case SEQ:
                A[i * col + j] = a;
                break;
            case FIX:
                A[i * col + j] = value;
            default:
                cerr << "invalid init method" << endl;
            }
            a += 1;
        }
    }
}

void init_mat(__half *A, __half *B, __half *C, int m, int n, int k, INIT_METHOD init_method, __half value = 0)
{
    init_single_mat(A, m, k, init_method, value);
    init_single_mat(B, k, n, init_method, value);
    init_single_mat(C, m, n, FIX, 0);
}

void print_mat(__half *M, int row, int col, string desc)
{
    cout << desc << ':' << endl;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            float val = __half2float(M[i * col + j]);
            cout << '\t' << val << ',';
        }
        cout << endl;
    }
}