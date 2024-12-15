file_name=./backend/fastgemv_czn.cu
output_file=fastgemv_czn.o

rm -rf *.o

/usr/local/cuda/bin/nvcc --generate-dependencies-with-compile \
    --dependency-output /localdata/zhennanc/sourceCode/Infinigence/exact-efficient-decoding/build/temp.linux-x86_64-cpython-310/backend/edgemm.o.d \
    -I/opt/conda/envs/hk/lib/python3.10/site-packages/torch/include \
    -I/opt/conda/envs/hk/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
    -I/opt/conda/envs/hk/lib/python3.10/site-packages/torch/include/TH -I/opt/conda/envs/hk/lib/python3.10/site-packages/torch/include/THC \
    -I/usr/local/cuda/include -I/opt/conda/envs/hk/include/python3.10 \
    -keep \
    -c $file_name \
    -o $output_file \
    -D__CUDA_NO_HALF_OPERATORS__ \
    -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    -D__CUDA_NO_HALF2_OPERATORS__ \
    --expt-relaxed-constexpr \
    --compiler-options ''"'"'-fPIC'"'"'' \
    -O3 -v \
    -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=backend -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80 -std=c++17
