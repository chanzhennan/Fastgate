#include <torch/extension.h>
#include <vector>

#include "xdnn_pytorch/xdnn_pytorch.h"
#include "xdnn_pytorch/xdnn_pytorch_wrapper_check.h"
#include "xdnn_pytorch/xdnn_pytorch_wrapper_dump.h"

#include "xpu/xpukernel.h"
#include "kernels/xpu_op.h"
#include "fused_op.h"

std::vector<at::Tensor> lltm_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell) {
    auto X = torch::cat({old_h, input}, /*dim=*/1);

    auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
    auto gates = gate_weights.chunk(3, /*dim=*/1);

    auto input_gate = torch::sigmoid(gates[0]);
    auto output_gate = torch::sigmoid(gates[1]);
    auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

    auto new_cell = old_cell + candidate_cell * input_gate;
    auto new_h = torch::tanh(new_cell) * output_gate;

    return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights};
}

std::vector<torch::Tensor> lltm_backward(
        torch::Tensor grad_h,
        torch::Tensor grad_cell,
        torch::Tensor new_cell,
        torch::Tensor input_gate,
        torch::Tensor output_gate,
        torch::Tensor candidate_cell,
        torch::Tensor X,
        torch::Tensor gate_weights,
        torch::Tensor weights) {
    auto d_output_gate = torch::tanh(new_cell) * grad_h;
    auto d_tanh_new_cell = output_gate * grad_h;
    auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

    auto d_old_cell = d_new_cell;
    auto d_candidate_cell = input_gate * d_new_cell;
    auto d_input_gate = candidate_cell * d_new_cell;

    auto gates = gate_weights.chunk(3, /*dim=*/1);
    d_input_gate *= d_sigmoid(gates[0]);
    d_output_gate *= d_sigmoid(gates[1]);
    d_candidate_cell *= d_elu(gates[2], 1.0);

    auto d_gates = torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

    auto d_weights = d_gates.t().mm(X);
    auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

    auto d_X = d_gates.mm(weights);
    const auto state_size = grad_h.size(1);
    auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
    auto d_input = d_X.slice(/*dim=*/1, state_size);

    return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}

torch::Tensor rotary_pos_emb(
        const torch::Tensor& t,
        const torch::Tensor& freqs,
        torch::Tensor& output) {
    impl_rotary_pos_emb_forward(t, freqs, output);
    return output;
}