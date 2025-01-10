#ifndef ROPE_H
#define ROPE_H
#include "xpu/kernel/xtdk.h"
#include "xpu/runtime.h"
#include <torch/extension.h>


// namespace xdnn_pytorch {
// #define PYTORCH_TYPES_AND_ARGS xpukernel::Context *ctx, const Tensor &t, const Tensor &freqs, Tensor &out
// #define PYTORCH_ARGS ctx, t, freqs, out

// #define REGISTER_PYTORCH_2TYPES(tensor0, tensor1, scalar_type0, scalar_type1, func)                                  \
//     if ((tensor0.type == scalar_type0) && (tensor1.type == scalar_type1)) {                                          \
//         return func<ScalarTypeToCPPType<scalar_type0>::type, ScalarTypeToCPPType<scalar_type1>::type>(PYTORCH_ARGS); \
//     }
// template <typename T, typename T2>
void impl_rotary_pos_emb_forward_mine(const torch::Tensor &t, const torch::Tensor &freqs, torch::Tensor &out);
// }
#endif