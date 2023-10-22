/*
    A collection of Flat-GEMM with transposed weight (B). @Infinigence.
*/
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "edgemm_tr.cuh"

/*
    Using 16x16x16 tensor core. Not so fast. For the convenience of AMD 
    implementation. By Ke Hong.
*/
void gemm_m8n128k64x4_bt(at::Tensor A, at::Tensor B, at::Tensor C){
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(output column) must be multiple of 128!");
    }

    const int BM = 16, BN = 128, BK = 64;
    dim3 blockDim(256);
    int BX = N / BN;
    int BY = (M + BM - 1) / BM;
    int BZ = 8;

    int tile_num = BX * BY;
    
    C.zero_();

    dim3 gridDim(BX, BY, BZ);

    // about 36.25 KB
    uint smem_a = 2 * BM * (BK + 8);
    uint smem_b = 2 * BN * (BK + 8);
    unsigned int dsmem = (smem_a + smem_b) * sizeof(half);

    gemm_m8n128k64x4_v8_tr<16, 128, 64, 72, 136><<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()), 
        M, N, K);
}

/*
    Using 8x32x16 tensor core. Faster version. By Zehua Wang.
*/
void gemm_m8n32k128x8_bt(at::Tensor A, at::Tensor B, at::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // weight shape: N * K
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    const int BM = 8, BN = 32, BK = 128;
    dim3 blockDim(256);
    dim3 gridDim(N / BN, 1, 2);
    gemm_m8n32k128x8_bz1<BM,BN,BK,BK+8,BN+8><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}

/*
    Using 8x32x16 tensor core. Experimental version. By Ke Hong.
*/
void gemm_m8n64k128x8_bt_exp(at::Tensor A, at::Tensor B, at::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // weight shape: N * K
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    const int BM = 8, BN = 64, BK = 128;
    dim3 blockDim(256);
    dim3 gridDim(N / BN, 1, 1);
    gemm_m8n64k128x8_bz1<BM,BN,BK,BK+8,BN+8><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}

/*
    Using 8x32x16 tensor core. Another faster version. By Zehua Wang. Changed version by Ke Hong.
*/
void gemm_m8n32k256x8_bt(at::Tensor A, at::Tensor B, at::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // weight shape: N * K
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    const int BM = 8, BN = 32, BK = 256;
    dim3 blockDim(256);
    dim3 gridDim(N / BN, 1, 1);
    
    gemm_m8n32k256x8_bz1<BM,BN,BK,BK+8,BN+8><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}


/*
    Using 8x32x16 tensor core. Another faster version. By Zehua Wang.
*/
void gemm_m8n32k256x8_bt_bz2(at::Tensor A, at::Tensor B, at::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // weight shape: N * K
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    const int BM = 8, BN = 32, BK = 256;
    dim3 blockDim(256);
    dim3 gridDim(N / BN, 1, 2);
    
    gemm_m8n32k256x8_bz1<BM,BN,BK,BK+8,BN+8><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}