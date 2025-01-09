#include "xdnn_pytorch/xdnn_pytorch.h"
#include "xdnn_pytorch/xdnn_pytorch_wrapper_check.h"
#include "xdnn_pytorch/xdnn_pytorch_wrapper_dump.h"

#include "xpu/refactor/impl/xpukernel_impl.h"
#include "xpu/xpukernel.h"

#define PYTORCH_TYPES_AND_ARGS xpukernel::Context *ctx, const Tensor &t, const Tensor &freqs, Tensor &out

#define KERNEL_ASSERT_SUCCESS(ctx, ret)                 \
    if (!((ret) == XPU_SUCCESS)) {                      \
        _log_kernel_fail(ctx, __FILE__, __LINE__, ret); \
        return xpytorch::xpu::api::RUNTIME_ERROR;       \
    }

namespace xpukernel = xpytorch::xpu::api;

template <typename T, typename T2>
__attribute__((global)) __attribute__((weak)) void rotary_pos_emb_forward_with_sincos_4dims(
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

template <typename T, typename T2>
__attribute__((global)) __attribute__((weak)) void rotary_pos_emb_forward_with_sincos_4dims_small_n(
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

template <typename T, typename T2>
int impl_rotary_pos_emb_forward(PYTORCH_TYPES_AND_ARGS) {
    xpukernel::ctx_guard RAII_GUARD(ctx);
    WRAPPER_DEFINE_MOCK_OF_3TENSORS(ctx, t, freqs, out);
    WRAPPER_BEGIN_CONTIGUOUS_ATTR_OF_1TENSOR(ctx, freqs);
    T* t_ptr = mock_t.ptr<T>();
    T2* freqs_ptr = mock_freqs.ptr<T2>();
    T* out_ptr = mock_out.ptr<T>();

    int64_t ndim = mock_t.sizes.size();
    std::vector<int64_t> t_shape(ndim);
    std::vector<int64_t> freqs_shape(ndim);
    std::vector<int64_t> t_stride(ndim);
    std::vector<int64_t> freqs_stride(ndim);
    std::vector<int64_t> out_stride(ndim);

    for (int i = 0; i < ndim; i++) {
        t_shape[i] = mock_t.sizes[i];
        t_stride[i] = mock_t.strides[i];
        freqs_shape[i] = mock_freqs.sizes[i];
        freqs_stride[i] = mock_freqs.strides[i];
        out_stride[i] = mock_out.strides[i];
        WRAPPER_ASSERT_EQ(ctx, mock_t.sizes[i], mock_out.sizes[i]);
    }
    int r = xpukernel::NOT_IMPLEMENT;
    WRAPPER_CHECK_CTX(ctx);
    WRAPPER_DUMP_PARAM4(ctx, t_ptr, freqs_ptr, out_ptr, ctx->_l3_mgr.get_size());
    WRAPPER_DUMP_PARAM4(ctx, t_shape, freqs_shape, t_stride, freqs_stride);
    WRAPPER_DUMP(ctx);
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
    if (ctx->dev().type() == xpukernel::kXPU3) {
        auto func = rotary_pos_emb_forward_with_sincos_4dims<T, T2>;
        if (freqs_dim[3] * sizeof(T) <= 1024 && freqs_dim[3] % 64 == 0) {
            func = rotary_pos_emb_forward_with_sincos_4dims_small_n<T, T2>;
        }

        r = func<<<ctx->ncluster(), 64>>>(
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
        KERNEL_ASSERT_SUCCESS(ctx, r);
    } else {
        return xpukernel::NOT_IMPLEMENT;
    }
    return xpukernel::SUCCESS;
}