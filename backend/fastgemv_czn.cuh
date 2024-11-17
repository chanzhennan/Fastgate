

#ifndef FAST_GEMV_CZN_CUH_
#define FAST_GEMV_CZN_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <cuda_pipeline_primitives.h>

#include "utility.cuh"

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)

#define L2_CACHEHINT(x) ".L2::" #x ""


__device__ __forceinline__ float warpReduceSumFloat(float sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}


template<typename AccessType>
struct CpAsync {
    template<typename T>
    __device__ void operator()(int ptr, const T* __restrict__ src, bool mask = true) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        constexpr int size = sizeof(AccessType);
        if constexpr (size == 16) {
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], %2;\n" 
                :: "r"(ptr), "l"(src), "n"(size));
        }
        else {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" 
                :: "r"(ptr), "l"(src), "n"(size));
        }
#else
        assert(false && "Requires SM80+");
#endif
    }
};

template<typename T>
__device__ __forceinline__ void cp_async(void* dst, const void* src) {
    CpAsync<T> cp;
    unsigned offset = __cvta_generic_to_shared(dst);
    cp(offset, (const T*)src);
}

template<int STAGE = 2>
__global__ void gemv_fp16(half* mat, half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread) {
    float sum = 0;
  // each thread load num_per_thread elements from global
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int start_idx = threadIdx.x;
    __shared__ float4 vec_shared[STAGE][128];
    __shared__ float4 mat_shared[STAGE][4][128];

    unsigned int stage = 0;
    unsigned int next_stage = 1;
    unsigned int j = start_idx;
    unsigned int next_j;


// 只加载一次数据
    if (start_idx < n >> 3) {  // 确保不越界
        if (threadIdx.y == 0) {
            cp_async<float4>(&vec_shared[0][start_idx], 
                           &reinterpret_cast<float4*>(vec)[start_idx]);
        }
        cp_async<float4>(&mat_shared[0][threadIdx.y][start_idx],
                        &reinterpret_cast<float4*>(mat)[row * (n >> 3) + start_idx]);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);


#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    j = start_idx + iter * blockDim.x;
    next_j = start_idx + (iter + 1) * blockDim.x;

    // 加载下一个stage的数据
    if (next_j < n >> 3) {
        if (threadIdx.y == 0) {
            vec_shared[next_stage][start_idx] = reinterpret_cast<float4*>(vec)[next_j];
        }
        mat_shared[next_stage][threadIdx.y][start_idx] = reinterpret_cast<float4*>(mat)[row * (n >> 3) + next_j];
    }


    // 计算当前stage的数据
    if (j < n >> 3) {
      // 这里使用j而不是start_idx，可以让不同线程访问不同的数据
      float4 vec_val = vec_shared[stage][start_idx];
      float4 mat_val = mat_shared[stage][threadIdx.y][start_idx];
      half2* vec_h1 = (half2*)&vec_val.x;
      half2* vec_h2 = (half2*)&vec_val.y;
      half2* vec_h3 = (half2*)&vec_val.z;
      half2* vec_h4 = (half2*)&vec_val.w;
      half2* mat_h1 = (half2*)&mat_val.x;
      half2* mat_h2 = (half2*)&mat_val.y;
      half2* mat_h3 = (half2*)&mat_val.z;
      half2* mat_h4 = (half2*)&mat_val.w;
      sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);
    }
    __syncthreads();
    stage = next_stage;
    next_stage = (next_stage + 1) % STAGE;

  }

  sum = warpReduceSumFloat(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSumFloat(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}
#endif  // FAST_GEMV_CZN_CUH_