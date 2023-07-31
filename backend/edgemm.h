#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void edgemm(at::Tensor A, at::Tensor B, at::Tensor C);