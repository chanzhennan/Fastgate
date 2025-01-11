#include "xdnn_pytorch/xdnn_pytorch.h"
#include "xdnn_pytorch/xdnn_pytorch_wrapper_check.h"
#include "xdnn_pytorch/xdnn_pytorch_wrapper_dump.h"

#include "xpu/xpukernel.h"
#include "xpu/kernel/xtdk.h"
#include "xpu/runtime.h"
#include <torch/extension.h>
#include <iostream>

#include "xpu_op.h"


template <typename T, typename T2>
__attribute__((global)) __attribute__((weak)) void rotary_pos_emb_forward_with_sincos_4dims_small_n_mine(
        const T* t,
        const T2* freqs,
        T* y,
        int t_dim0,
        int t_dim1,
        int t_dim2,
        int t_dim3,
        int t_stride0,
        int t_stride1,
        int t_stride2,
        int t_stride3,
        int freqs_dim0,
        int freqs_dim1,
        int freqs_dim2,
        int freqs_dim3,
        int freqs_stride0,
        int freqs_stride1,
        int freqs_stride2,
        int freqs_stride3,
        int y_stride0,
        int y_stride1,
        int y_stride2,
        int y_stride3);


// namespace xdnn_pytorch {

// template <typename T, typename T2>
void impl_rotary_pos_emb_forward(const torch::Tensor &t, const torch::Tensor &freqs, torch::Tensor &out) {
    // xpukernel::ctx_guard RAII_GUARD(ctx);
    // WRAPPER_DEFINE_MOCK_OF_3TENSORS(ctx, t, freqs, out);
    // WRAPPER_BEGIN_CONTIGUOUS_ATTR_OF_1TENSOR(ctx, freqs);
    float16* t_ptr = reinterpret_cast<float16 *>(t.data_ptr<at::Half>());
    float* freqs_ptr = reinterpret_cast< float *>(freqs.data_ptr<float>());
    float16* out_ptr = reinterpret_cast<float16 *>(out.data_ptr<at::Half>());
    
    int64_t ndim = t.ndimension();
    std::vector<int64_t> t_shape(ndim);
    std::vector<int64_t> freqs_shape(ndim);
    std::vector<int64_t> t_stride(ndim);
    std::vector<int64_t> freqs_stride(ndim);
    std::vector<int64_t> out_stride(ndim);

    for (int i = 0; i < ndim; i++) {
        t_shape[i] = t.sizes()[i];
        t_stride[i] = t.strides()[i];
        freqs_shape[i] = freqs.sizes()[i];
        freqs_stride[i] = freqs.strides()[i];
        out_stride[i] = out.strides()[i];
        // WRAPPER_ASSERT_EQ(ctx, mock_t.sizes[i], mock_out.sizes[i]);
    }

    // int r = xpukernel::NOT_IMPLEMENT;
    // WRAPPER_CHECK_CTX(ctx);
    // WRAPPER_DUMP_PARAM4(ctx, t_ptr, freqs_ptr, out_ptr, ctx->_l3_mgr.get_size());
    // WRAPPER_DUMP_PARAM4(ctx, t_shape, freqs_shape, t_stride, freqs_stride);
    // WRAPPER_DUMP(ctx);
    int freqs_len = 1;
    for (int i = 0; i < t_shape.size(); i++) {
        freqs_len *= freqs_shape[i];
    }

    int t_dim[4], freqs_dim[4], t_strd[4], freqs_strd[4], y_strd[4];
    int t_basic_stride = 1;
    int freqs_basic_stride = 1;
    int y_basic_stride = 1;

    for (int i = 3; i > -1; i--) {
        if (t_shape.size() >= 4 - i) {
            t_dim[i] = t_shape[i];
            t_strd[i] = t_stride[i];
            if (t_stride[i] == 0) {
                t_strd[i] = t_basic_stride;
            }
            freqs_dim[i] = freqs_shape[i];
            freqs_strd[i] = freqs_stride[i];
            if (freqs_stride[i] == 0) {
                freqs_strd[i] = freqs_basic_stride;
            }
            y_strd[i] = out_stride[i];
            if (out_stride[i] == 0) {
                y_strd[i] = y_basic_stride;
            }
        } else {
            t_dim[i] = 1;
            t_strd[i] = 1;
            freqs_dim[i] = 1;
            freqs_strd[i] = 1;
            y_strd[i] = 1;
        }
        t_basic_stride *= t_dim[i];
        freqs_basic_stride *= freqs_dim[i];
        y_basic_stride *= t_dim[i];
    }
    // if (ctx->dev().type() == xpukernel::kXPU3) {
    // auto func = rotary_pos_emb_forward_with_sincos_4dims;
    // if (freqs_dim[3] * sizeof(float16) <= 1024 && freqs_dim[3] % 64 == 0) {
    std::cout << "t_dim: " << t_dim[0] << " " << t_dim[1] << " " << t_dim[2] << " " << t_dim[3] << std::endl;
    std::cout << "t_strd: " << t_strd[0] << " " << t_strd[1] << " " << t_strd[2] << " " << t_strd[3] << std::endl;
    std::cout << "freqs_dim: " << freqs_dim[0] << " " << freqs_dim[1] << " " << freqs_dim[2] << " " << freqs_dim[3] << std::endl;
    std::cout << "freqs_strd: " << freqs_strd[0] << " " << freqs_strd[1] << " " << freqs_strd[2] << " " << freqs_strd[3] << std::endl;
    std::cout << "y_strd: " << y_strd[0] << " " << y_strd[1] << " " << y_strd[2] << " " << y_strd[3] << std::endl;
    auto func = rotary_pos_emb_forward_with_sincos_4dims_small_n_mine<float16, float>;
    // }
    std::cout << "enter3" << std::endl;
    func<<<12, 64>>>(
            t_ptr,
            freqs_ptr,
            out_ptr,
            t_dim[0],
            t_dim[1],
            t_dim[2],
            t_dim[3],
            t_strd[0],
            t_strd[1],
            t_strd[2],
            t_strd[3],
            freqs_dim[0],
            freqs_dim[1],
            freqs_dim[2],
            freqs_dim[3],
            freqs_strd[0],
            freqs_strd[1],
            freqs_strd[2],
            freqs_strd[3],
            y_strd[0],
            y_strd[1],
            y_strd[2],
            y_strd[3]);
        // KERNEL_ASSERT_SUCCESS(ctx, r);
    // } else {
    //     return xpukernel::NOT_IMPLEMENT;
    // }
    std::cout << "exit" << std::endl;
}
