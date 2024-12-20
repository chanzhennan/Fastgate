
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "mmanaive.cuh"
#include "fastgemm.h"

void mmanaive(at::Tensor A, at::Tensor B, at::Tensor C) {
    
    // dim3 block(WARP_SIZE);
    // dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

    // mmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}