#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void fastgemm(at::Tensor A, at::Tensor B, at::Tensor C);
void mma(at::Tensor A, at::Tensor B, at::Tensor C) ;
