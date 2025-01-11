#include "xpu/kernel/xtdk.h"
#include "xpu/runtime.h"
#include <iostream>
#include <torch/extension.h>
#include "xpu_op.h"

__global__ void sigmoid_backward(float* x, float* x_grad, int length);

// sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
torch::Tensor d_sigmoid(torch::Tensor z) {
    TORCH_CHECK(z.dtype() == torch::kFloat32, "d_sigmoid only supports float32");
    if (z.device().is_cuda()) {
        if (!z.is_contiguous()) {
            z = z.contiguous();
        }
        auto x_grad = at::empty_like(z, z.options());
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
        sigmoid_backward<<<8, 64>>>(z.data_ptr<float>(), x_grad.data_ptr<float>(), static_cast<int>(z.numel()));
#pragma GCC diagnostic pop
        return x_grad;
    } else {
        auto s = z.sigmoid();
        return s * (1 - s);
    }
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
    return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
    auto e = z.exp();
    auto mask = (alpha * (e - 1)) < 0;
    return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}