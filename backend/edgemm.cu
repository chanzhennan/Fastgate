#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "edgemm.cuh"

void edgemm(at::Tensor A, at::Tensor B, at::Tensor C){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const int BM = 128, BN = 256, BK = 32;
    dim3 blockDim(256);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;

    const int NSPLIT = 4096;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);

    cudaFuncSetAttribute(myHGEMMAlignedV5,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
    
    myHGEMMAlignedV5<<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        M, N, K
        );
}


void edgemm_m8n256k64(at::Tensor A, at::Tensor B, at::Tensor C){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const int BM = 8, BN = 256, BK = 64;
    dim3 blockDim(256);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;

    const int NSPLIT = 4096;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);

    cudaFuncSetAttribute(eed_hgemm_m8n256k64_v3,   
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    // about 76KB for m8n256k64
    // unsigned int dsmem = 2 * (BM * (8 * BK + 8) + BK * (BN + 8)) * sizeof(half);
    unsigned int dsmem = 2 * (BM * (4 * BK + 8) + BK * (BN + 8)) * sizeof(half);
    
    eed_hgemm_m8n256k64_v3<<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        M, N, K
        );
}


void edgemm_m8n128k64(at::Tensor A, at::Tensor B, at::Tensor C){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const int BM = 8, BN = 128, BK = 64;
    dim3 blockDim(128);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;

    const int NSPLIT = 2048;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);

    // cudaFuncSetAttribute(eed_hgemm_m8n256k64_v3,   
    //             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    // about 39 KB for m8n128k64
    unsigned int dsmem = 2 * (BM * (2 * BK + 8) + BK * (BN + 8)) * sizeof(half);
    
    eed_hgemm_m8n128k64_v4<<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        M, N, K
        );
}


void edgemm_m8n128k64x4(at::Tensor A, at::Tensor B, at::Tensor C){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const int BM = 8, BN = 128, BK = 64;
    dim3 blockDim(128);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    int BZ = 8;

    dim3 gridDim(BX, BY, BZ);

    // const int NSPLIT = 2048;
    // int split_num = (N + NSPLIT - 1) / NSPLIT;
    // dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);

    // cudaFuncSetAttribute(eed_hgemm_m8n256k64_v3,   
    //             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    // about 39 KB for m8n128k64
    unsigned int dsmem = 2 * (BM * (2 * BK + 8) + BK * (BN + 8)) * sizeof(half);
    
    eed_hgemm_m8n128k64x4_v7<<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        M, N, K
        );
}

void edgemm_m8n128k64x4_bt(at::Tensor A, at::Tensor B, at::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const int BM = 8, BN = 128, BK = 64;
    dim3 blockDim(128);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    int BZ = 8;

    dim3 gridDim(BX, BY, BZ);

    // 40.25 KB for m8n128k64
    unsigned int dsmem = 2 * (BM * (2 * BK + 8) + BN * (BK + 8)) * sizeof(half);

    eed_hgemm_m8n128k64x4_v7_bt<<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}

void edgemm_m8n128k128(at::Tensor A, at::Tensor B, at::Tensor C){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const int BM = 8, BN = 128, BK = 128;
    dim3 blockDim(128);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;

    const int NSPLIT = 2048;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);

    cudaFuncSetAttribute(eed_hgemm_m8n128k128_v5,   
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    // about 74 KB for m8n128k128
    unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
    
    eed_hgemm_m8n128k128_v5<<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        M, N, K
        );
}


void edgemv_m1n128k64x4(at::Tensor A, at::Tensor B, at::Tensor C){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const int BM = 8, BN = 128, BK = 64;
    dim3 blockDim(128);
    int BX = (N + BN - 1) / BN;
    int BY = 1;
    int BZ = 8;

    dim3 gridDim(BX, BY, BZ);


    // cudaFuncSetAttribute(eed_hgemv_m1n128k64_v6,   
    //             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    // about 39 KB for m8n128k64
    unsigned int dsmem = (2 * (BM * (2 * BK + 8) + BK * (BN + 8)) + 0) * sizeof(half);
    
    eed_hgemv_m1n128k64x4_v6<8><<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        M, N, K
        );
}


void edgemv_m1n256k64x4(at::Tensor A, at::Tensor B, at::Tensor C){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const int BM = 8, BN = 256, BK = 64;
    dim3 blockDim(256);
    int BX = (N + BN - 1) / BN;
    int BY = 1;
    int BZ = 8;

    dim3 gridDim(BX, BY, BZ);


    cudaFuncSetAttribute(eed_hgemv_m1n256k64x4_v8<8>,   
                cudaFuncAttributeMaxDynamicSharedMemorySize, 76032);

    // about 76 KB for m8n256k64
    unsigned int dsmem = (2 * (BM * (4 * BK + 8) + BK * (BN + 8)) + 0) * sizeof(half);
    
    eed_hgemv_m1n256k64x4_v8<8><<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),  
        reinterpret_cast<half *>(B.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),  
        M, N, K
        );
}
