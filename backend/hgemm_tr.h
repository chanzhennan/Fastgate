#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void hgemm_tr(at::Tensor A, at::Tensor B, at::Tensor C);