#include <torch/extension.h>
#include <vector>

#include "xdnn_pytorch/xdnn_pytorch.h"
#include "xdnn_pytorch/xdnn_pytorch_wrapper_check.h"
#include "xdnn_pytorch/xdnn_pytorch_wrapper_dump.h"

torch::Tensor d_sigmoid(torch::Tensor z);
torch::Tensor d_tanh(torch::Tensor z);
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha);
void impl_rotary_pos_emb_forward(const torch::Tensor &t, const torch::Tensor &freqs, torch::Tensor &out);
