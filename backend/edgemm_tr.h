/*
    A collection of Flat-GEMM with transposed weight (B). @Infinigence.
*/
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void gemm_m8n128k64x4_bt(at::Tensor A, at::Tensor B, at::Tensor C);
void gemm_m8n32k128x8_bt(at::Tensor A, at::Tensor B, at::Tensor C);
void gemm_m8n32k256x8_bt(at::Tensor A, at::Tensor B, at::Tensor C);
void gemm_m8n32k256x8_bt_bz2(at::Tensor A, at::Tensor B, at::Tensor C);
void gemm_m8n64k128x8_bt_exp(at::Tensor A, at::Tensor B, at::Tensor C);