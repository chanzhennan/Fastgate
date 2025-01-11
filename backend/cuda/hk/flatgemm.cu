#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "flatgemm.cuh"


void flat_gemm_m8n128k64x4_bz1(at::Tensor A, at::Tensor B, at::Tensor C, int bz) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);  // weight shape: N * K
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    // C.zero_();

    const int BM = 8, BN = 128, BK = 64;
    dim3 blockDim(128);
    dim3 gridDim(N / BN, 1, bz);
    // orig: BZ = 2
    gemm_m8n128k64x4_bz1<BM,BN,BK,BK+8,BN+8><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}

void flat_gemm_m8n256k32x8_bz1(at::Tensor A, at::Tensor B, at::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);  // weight shape: N * K
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    C.zero_();

    const int BM = 8, BN = 256, BK = 32;
    dim3 blockDim(256);
    dim3 gridDim(N / BN, 1, 8);
    gemm_m8n256k32x8_bz1<BM,BN,BK,BK+8,BN+8><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}


void flat_gemm_m8n256k64x8(at::Tensor A, at::Tensor B, at::Tensor C, int bz) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(output column) must be multiple of 128!");
    }

    const int BM = 8, BN = 256, BK = 64;
    dim3 blockDim(256);
    int BX = N / BN;
    int BY = (M + BM - 1) / BM;
    int BZ = bz;
    // orig: BZ = 8

    // C.zero_();

    dim3 gridDim(BX, BY, BZ);
    flat_gemm_m8n256k64x8_v3<BM, BN, BK, BK + 8, BN + 8><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}

void flat_gemm_m8n256k32x16(at::Tensor A, at::Tensor B, at::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(output column) must be multiple of 128!");
    }

    const int BM = 8, BN = 256, BK = 32;
    dim3 blockDim(512);
    int BX = N / BN;
    int BY = (M + BM - 1) / BM;
    int BZ = 16;

    C.zero_();

    dim3 gridDim(BX, BY, BZ);
    flat_gemm_m8n256k32x16_db<BM, BN, BK, BK * 2 + 8, BN><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}

void flat_gemm_m8n512k32x16(at::Tensor A, at::Tensor B, at::Tensor C, int bz) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(output column) must be multiple of 128!");
    }

    const int BM = 8, BN = 512, BK = 32;
    dim3 blockDim(512);
    int BX = N / BN;
    int BY = (M + BM - 1) / BM;
    int BZ = bz;

    // C.zero_();

    dim3 gridDim(BX, BY, BZ);
    flat_gemm_m8n512k32x16<BM, BN, BK, BK + 8, BN + 8><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}


void flat_gemm_m8n64k128x8(at::Tensor A, at::Tensor B, at::Tensor C, int bz) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);  // weight shape: N * K
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    const int BM = 8, BN = 64, BK = 128;
    dim3 blockDim(256);
    dim3 gridDim(N / BN, 1, bz);
    // orig: BZ = 1
    flat_gemm_m8n64k128x8<BM,BN,BK,BK+8,BN+8><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}


void flat_gemm_m8n32k256x8(at::Tensor A, at::Tensor B, at::Tensor C, int bz) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);  // weight shape: N * K
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    const int BM = 8, BN = 32, BK = 256;
    dim3 blockDim(256);
    dim3 gridDim(N / BN, 1, bz);
    
    flat_gemm_m8n32k256x8<BM,BN,BK,BK+8,BN+8><<<gridDim, blockDim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}
