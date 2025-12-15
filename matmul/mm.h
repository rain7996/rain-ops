#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>

#define SHARED_SIZE_BYTES 192 * 1024

void cublas_matmul(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle);

void matmul_v0(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle);

#define MM_V1_BLOCK_SIZE 16U

void matmul_v1(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle);

void matmul_v2(const __half *A, const __half *B, __half *C, int m, int n, int k, cublasHandle_t handle);