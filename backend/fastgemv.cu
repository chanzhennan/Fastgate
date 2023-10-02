#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "fastgemv.cuh"

void fastgemv(at::Tensor A, at::Tensor B, at::Tensor C){

    int mat_height_ = A.size(0);
    int vec_height_ = B.size(0);

    int block_dim_x = 128;
    int block_dim_y = 4;
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = vec_height_ / block_dim_x;
    assert(num_per_thread >= 8);

    dim3 grid_dim(1, mat_height_ / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);

    gemv_fp16<<<grid_dim, block_dim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        vec_height_, 
        num_per_thread);
}

void fastgemv_int8(at::Tensor A, at::Tensor B, at::Tensor C){

    int mat_height_ = A.size(0);
    int vec_height_ = B.size(0);

    int block_dim_x = 128;
    int block_dim_y = 4;
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = vec_height_ / block_dim_x;
    assert(num_per_thread >= 8);

    dim3 grid_dim(1, mat_height_ / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);

    half zero = __float2half(1.0f);
    half scale = __float2half(0.0f);

    gemv_quantized_int8<<<grid_dim, block_dim>>>(
        reinterpret_cast<int8_t *>(A.data_ptr<int8_t>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        vec_height_, 
        zero, 
        scale,
        num_per_thread);
}

void fastgemv_tuned(at::Tensor A, at::Tensor B, at::Tensor C){

    int mat_height_ = A.size(0);
    int vec_height_ = B.size(0);

    int block_dim_x = 128;
    int block_dim_y = 1;
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = vec_height_ / block_dim_x;
    assert(num_per_thread >= 8);

    dim3 grid_dim(1, mat_height_ / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);

    gemv_fp16_tuned<<<grid_dim, block_dim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        vec_height_, 
        num_per_thread);
}

void fastgemv_extend(at::Tensor A, at::Tensor B, at::Tensor C){

    // A: weight, [N, K]
    // B: vector, [M, K]
    // C: result, [M, N]
    int N = A.size(0);
    int K = B.size(1);
    int M = B.size(0);

    int block_dim_x = 128;
    int block_dim_y = M;
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = K / block_dim_x;
    assert(num_per_thread >= 8);

    dim3 grid_dim(N);
    dim3 block_dim(block_dim_x, block_dim_y);

    gemm_fp16<<<grid_dim, block_dim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        K, N, 
        num_per_thread);
}
