#include "hgemm.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>


void hgemm(at::Tensor A, at::Tensor B, at::Tensor C){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // at::Tensor alpha_half = torch::ones({1}, dtype(at::ScalarType::Half));
    // at::Tensor beta_half = torch::zeros({1}, dtype(at::ScalarType::Half));

    cublasHandle_t cublasH = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH);

    // cublasHgemm(
    //       cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
    //       reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()),
    //       // &weight_ptr[mid_weight_id * in_channel * out_channel],
    //       reinterpret_cast<half *>(B.data_ptr<at::Half>()),
    //       N, 
    //       reinterpret_cast<half *>(A.data_ptr<at::Half>()),
    //       K,
    //       reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()),
    //       reinterpret_cast<half *>(C.data_ptr<at::Half>()),
    //       N);

    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_half = __float2half(alpha);
    half beta_half = __float2half(beta);

    cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, M, K,
                    // (const void*)reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()), 
                    (const void*)reinterpret_cast<half *>(&alpha_half),
                    // (const void*)(&alpha), 
                    (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                    CUDA_R_16F, N, 
                    (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                    CUDA_R_16F, K, 
                    // (const void*)reinterpret_cast<half *>(beta_half.data_ptr<at::Half>()), 
                    (const void*)reinterpret_cast<half *>(&beta_half),
                    // (const void*)(&beta), 
                    (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>()), 
                    CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}