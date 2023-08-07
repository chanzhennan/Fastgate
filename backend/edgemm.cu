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
