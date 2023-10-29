#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void flat_gemm_m8n128k64x4_bz1(at::Tensor A, at::Tensor B, at::Tensor C, int bz);
void flat_gemm_m8n256k32x8_bz1(at::Tensor A, at::Tensor B, at::Tensor C);
void flat_gemm_m8n256k64x8(at::Tensor A, at::Tensor B, at::Tensor C, int bz);
void flat_gemm_m8n256k32x16(at::Tensor A, at::Tensor B, at::Tensor C);
void flat_gemm_m8n512k32x16(at::Tensor A, at::Tensor B, at::Tensor C, int bz);
void flat_gemm_m8n64k128x8(at::Tensor A, at::Tensor B, at::Tensor C, int bz);
void flat_gemm_m8n32k256x8(at::Tensor A, at::Tensor B, at::Tensor C, int bz);