
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

// #include "mmabase.cuh"
#include "mma_async.cuh"
// #include "mma_async_stage2.cuh"
// #include "mma_async_stage3.cuh"
// #include "mma_async_stage4.cuh"
// #include "mmanaive.cuh"
#include "fastgemm.h"

void mma(at::Tensor A, at::Tensor B, at::Tensor C) {
    
    int M = A.size(0);  // M
    int K = A.size(1);  // K
    int N = B.size(0);  // N

    // dim3 block(WARP_SIZE);
    // dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
    // mmaNaiveKernel<<<grid, block>>>(reinterpret_cast<half *>(A.data_ptr<at::Half>()),
    //                                 reinterpret_cast<half *>(B.data_ptr<at::Half>()),
    //                                 reinterpret_cast<half *>(C.data_ptr<at::Half>()),
    //                                 M, N, K);


    // static size_t smem_max_size = initMmaBase();
    // dim3 block(THREADS_PER_BLOCK);
    // dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    // mmaBaseKernel<<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(A.data_ptr<at::Half>()),
    //                                 reinterpret_cast<half *>(B.data_ptr<at::Half>()),
    //                                 reinterpret_cast<half *>(C.data_ptr<at::Half>()),
    //                                 M, N, K);

    // Mma/cuBLAS = 195.38% 
    static size_t smem_max_size = initMmaAsync();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    mmaAsyncKernel<<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(A.data_ptr<at::Half>()),
                                   reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                                   reinterpret_cast<half *>(C.data_ptr<at::Half>()),
                                    M, N, K);
  
    // Mma/cuBLAS = 144.11% 
    // static size_t smem_max_size = initMmaAsyncStage2();

    // dim3 block(THREADS_PER_BLOCK);
    // dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    // mmaAsyncStage2Kernel<<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(A.data_ptr<at::Half>()),
    //                                   reinterpret_cast<half *>(B.data_ptr<at::Half>()),
    //                                   reinterpret_cast<half *>(C.data_ptr<at::Half>()),
    //                                   M, N, K);

    // Mma/cuBLAS = 118.88%
    // static size_t smem_max_size = initMmaAsyncStage3();

    // dim3 block(THREADS_PER_BLOCK);
    // dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    // mmaAsyncStage3Kernel<<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(A.data_ptr<at::Half>()),
    //                                   reinterpret_cast<half *>(B.data_ptr<at::Half>()),
    //                                   reinterpret_cast<half *>(C.data_ptr<at::Half>()),
    //  M, N, K);


    // Mma/cuBLAS = 120.88%
    // static size_t smem_max_size = initMmaAsyncStage4();

    // dim3 block(THREADS_PER_BLOCK);
    // dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    // mmaAsyncStage4Kernel<<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(A.data_ptr<at::Half>()),
    //                                   reinterpret_cast<half *>(B.data_ptr<at::Half>()),
    //                                   reinterpret_cast<half *>(C.data_ptr<at::Half>()), M, N, K);

}


