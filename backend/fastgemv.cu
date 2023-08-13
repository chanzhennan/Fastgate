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