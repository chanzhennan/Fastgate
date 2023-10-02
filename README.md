# exact-efficient-decoding
Operator and system optimization for LLM decoding.

## op

1. 对于小batch size 的GEMM（fp16），调用该库中性能最优的 `edgemm_m8n128k64x4` 函数。
    * input_feat(MxK) * weight(K*N) ---> output_feat(MxN)
    * 需要满足：K 和 N 是128 的倍数

2. 对于小batch size 的GEMM（fp16），权重矩阵转置的情况，请调用后缀为_bt 的函数`edgemm_m8n128k64x4_bt`
    * input_feat(MxK) * weight_t(N*K) ---> output_feat(MxN)
    * 需要满足：K 和 N 是128 的倍数
