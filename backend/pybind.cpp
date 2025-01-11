#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#if defined(BUILD_WITH_CUDA)
  #include "cuda/hk/hgemm.h"
  #include "cuda/hk/edgemm.h"
  #include "cuda/hk/flatgemm.h"
  // #include "hk/fastgemv.h"
  #include "cuda/hk/edgemm_tr.h"
  #include "cuda/hk/hgemm_tr.h"
  #include "cuda/czn/fastgemm.h"
#endif

#if defined(BUILD_WITH_KUNLUN)
  #include "kunlun/fused_op.h"
#endif



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
#if defined(BUILD_WITH_CUDA)
  m.def("hgemm", &hgemm);
  // m.def("fastgemv", &fastgemv);
  m.def("fastgemm", &fastgemm);
  m.def("mma", &mma);
#endif

#if defined(BUILD_WITH_KUNLUN)
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
  m.def("rotary_pos_emb", &rotary_pos_emb, "rope forward");
#endif

}
