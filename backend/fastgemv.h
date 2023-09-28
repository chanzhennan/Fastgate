#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void fastgemv(at::Tensor A, at::Tensor B, at::Tensor C);
void fastgemv_int8(at::Tensor A, at::Tensor B, at::Tensor C);
void fastgemv_tuned(at::Tensor A, at::Tensor B, at::Tensor C);