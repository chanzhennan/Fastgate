#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "hgemm.h"
#include "edgemm.h"
#include "fastgemv.h"
#include "edgemm_tr.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("edgemm_m8n128k64x4_tr_amd", &edgemm_m8n128k64x4_tr_amd);
  m.def("edgemm_m8n256k32x8", &edgemm_m8n256k32x8);
  m.def("edgemm_m8n128k64x4_amd", &edgemm_m8n128k64x4_amd);
  m.def("edgemv_m1n256k64x4", &edgemv_m1n256k64x4);
  m.def("edgemm_m8n128k64x4_bt", &edgemm_m8n128k64x4_bt);
  m.def("edgemm_m8n128k64x4", &edgemm_m8n128k64x4);
  m.def("edgemv_m1n128k64x4", &edgemv_m1n128k64x4);
  m.def("edgemm_m8n128k128", &edgemm_m8n128k128);
  m.def("edgemm_m8n128k64", &edgemm_m8n128k64);
  m.def("edgemm_m8n256k64", &edgemm_m8n256k64);
  m.def("edgemm", &edgemm);
  m.def("hgemm", &hgemm);
  m.def("fastgemv", &fastgemv);
  m.def("fastgemv_int8", &fastgemv_int8);
  m.def("fastgemv_tuned", &fastgemv_tuned);
  m.def("fastgemv_extend", &fastgemv_extend);
  /*
    A collection of Flat-GEMM with transposed weight (B):
  */
  m.def("gemm_m8n128k64x4_bt", &gemm_m8n128k64x4_bt);
  m.def("gemm_m8n32k128x8_bt", &gemm_m8n32k128x8_bt);
  m.def("gemm_m8n32k256x8_bt", &gemm_m8n32k256x8_bt);
}
