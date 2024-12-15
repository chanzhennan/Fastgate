#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "fastgemv_czn.cuh"

void fastgemv(at::Tensor A, at::Tensor B, at::Tensor C){
    // 输入已经转置:
    // A: [N, K] 矩阵 (原 [K, N] 转置而来)
    // B: [K, 1] 向量 (原 [1, K] 转置而来)
    // C: [N, 1] 结果
    int mat_height_ = A.size(0);  // N
    int vec_height_ = B.size(0);  // K
    int vec_width_ = B.size(1);  // M

    int block_dim_x = 128;
    int block_dim_y = 4;
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = vec_height_ / block_dim_x;  // K/128
    assert(num_per_thread >= 8);

    // 每个block处理4行，总共需要 N/4 个block
    dim3 grid_dim(1, mat_height_ / block_dim_y);  // [1, N/4]
    dim3 block_dim(block_dim_x, block_dim_y);     // [128, 4]


    if(vec_width_ == 1){
    // 现在矩阵是行优先存储，每行是连续的K个元素
    gemv_fp16<<<grid_dim, block_dim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  // [N, K] 矩阵
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),  // [K, 1] 向量
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  // [N, 1] 结果
        vec_height_,  // K
        mat_height_,
        num_per_thread);  // K/128
    }
    else if(vec_width_ == 2){
        gemv_fp16_bs2<<<grid_dim, block_dim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  // [N, K] 矩阵
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),  // [K, 1] 向量
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  // [N, 1] 结果
        vec_height_,  // K
        mat_height_,
        num_per_thread);  // K/128
    }

    
}
