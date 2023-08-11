#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "hgemm.h"
#include "edgemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("edgemm_m8", &edgemm_m8);
  m.def("edgemm", &edgemm);
  m.def("hgemm", &hgemm);
}
