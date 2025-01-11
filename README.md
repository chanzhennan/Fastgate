# SPEEDGATE
Operator and system optimization for t2vi inference.

## OPERATIONS

1. 对于小batch size 的GEMM（fp16），调用该库中性能最优的 `edgemm_m8n128k64x4` 函数。
    * input_feat(MxK) * weight(K*N) ---> output_feat(MxN)
    * 需要满足：K 和 N 是128 的倍数

2. 对于小batch size 的GEMM（fp16），权重矩阵转置的情况，请调用后缀为_bt 的函数`edgemm_m8n128k64x4_bt`
    * input_feat(MxK) * weight_t(N*K) ---> output_feat(MxN)
    * 需要满足：K 和 N 是128 的倍数

## COMPILE KUNLUN BACKEND

1. python setup.py install
2. 【待修复】需要把./build/temp.linux-x86_64-cpython-39 下编译的*.xpu --> *.o 文件放到对应的backend/kunlun/kernels/ 目录下
3. python test.py


## ROADMAP
- [✅] AABB Rope @ KUNLUN
- [📝] ABAB Rope @ KUNLUN
- [📝] BS16 GEMM @ CUDA
- [📝] LINT && CICD 
