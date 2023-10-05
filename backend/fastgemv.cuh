///////////////////////////////////////////////////////////////
// The original version of fastgemv comes from https://github.com/wangsiping97/FastGEMV
// We extend the implementation to M > 1 computation as GEMM
// We make some modifications including larger fetching to improve the performance
///////////////////////////////////////////////////////////////

#ifndef FAST_GEMV_CUH_
#define FAST_GEMV_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utility.cuh"

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)

///////////////////////////// REDUCE SUM //////////////////////////////

__device__ __forceinline__ float warpReduceSum(float sum,
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


__device__ __forceinline__ half warpReduceSum(half sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 1));  // 0-1, 2-3, 4-5, etc.
  return sum;
}


__inline__ __device__ half warpReduceSum2(half val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
  return val;
}


__device__ float block_reduce_sum(float reducing, float *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    const int32_t WPT = blockDim.x * blockDim.y / 32;
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_id = threadIdx.x / 32;

# pragma unroll
    for (int32_t mask = 16; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}


__device__ __forceinline__ half2 warpReduceSum(half2 sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, 1));  // 0-1, 2-3, 4-5, etc.
  return sum;
}


///////////////////////////// NORMAL //////////////////////////////
// thread_per_block = blockDim.x
// blockDim.y <= SHARED_MEM_MAX_ROWS
__global__ void gemv_fp16_tuned(half* mat, half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);
  half2 vec_val[8];
  half2 mat_val[8];

  // half2 temp_sum = {__float2half(0.0f), __float2half(0.0f)};
  float sum = 0.0f;

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 16); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x) << 4;
    if (j >= n) {break;}
    *(float4*)(&vec_val[0]) = *(float4*)(&vec[j + 0]);
    *(float4*)(&vec_val[4]) = *(float4*)(&vec[j + 8]);
    *(float4*)(&mat_val[0]) = *(float4*)(&mat[row * n + j + 0]);
    *(float4*)(&mat_val[4]) = *(float4*)(&mat[row * n + j + 8]);
    sum += __half2float(vec_val[0].x) * __half2float(mat_val[0].x);
    sum += __half2float(vec_val[0].y) * __half2float(mat_val[0].y);
    sum += __half2float(vec_val[1].x) * __half2float(mat_val[1].x);
    sum += __half2float(vec_val[1].y) * __half2float(mat_val[1].y);
    sum += __half2float(vec_val[2].x) * __half2float(mat_val[2].x);
    sum += __half2float(vec_val[2].y) * __half2float(mat_val[2].y);
    sum += __half2float(vec_val[3].x) * __half2float(mat_val[3].x);
    sum += __half2float(vec_val[3].y) * __half2float(mat_val[3].y);
    sum += __half2float(vec_val[4].x) * __half2float(mat_val[4].x);
    sum += __half2float(vec_val[4].y) * __half2float(mat_val[4].y);
    sum += __half2float(vec_val[5].x) * __half2float(mat_val[5].x);
    sum += __half2float(vec_val[5].y) * __half2float(mat_val[5].y);
    sum += __half2float(vec_val[6].x) * __half2float(mat_val[6].x);
    sum += __half2float(vec_val[6].y) * __half2float(mat_val[6].y);
    sum += __half2float(vec_val[7].x) * __half2float(mat_val[7].x);
    sum += __half2float(vec_val[7].y) * __half2float(mat_val[7].y);
  }

  static __shared__ float warpLevelSums[WARP_SIZE];

  sum = block_reduce_sum(sum, warpLevelSums);

  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}


///////////////////////////// extend GEMV for GEMM in FP16 precision //////////////////////////////
// thread_per_block = blockDim.x
// blockDim.y <= SHARED_MEM_MAX_ROWS
__global__ void gemm_fp16(half* mat, __restrict__ half* vec, half* res, unsigned int K,
                          unsigned int N, unsigned int num_per_thread) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned int mid = threadIdx.y;
  unsigned int row = blockIdx.x;
  unsigned int start_idx = threadIdx.x;
  const int32_t lane_id = tid % 32;
  const int32_t warp_id = tid / 32;

  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);
  // half2 vec_val[4];
  // half2 mat_val[4];

  // half2 temp_sum = {__float2half(0.0f), __float2half(0.0f)};
  float sum = 0.0f;

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x);
    if (j >= K >> 3) {break;}
      float4 vec_val = vec4[mid * (K >> 3) + j];
      float4 mat_val = mat4[row * (K >> 3) + j];
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

  static __shared__ float shared_mem[WARP_SIZE];

#pragma unroll
  for (int32_t mask = 16; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  if (lane_id == 0) shared_mem[warp_id] = sum;
  __syncthreads();

  // sum = tid < ((blockDim.x * blockDim.y) >> 5) ? 
  //   shared_mem[tid] : 0.0f;

  // sum = warpReduceSum(sum, (blockDim.x >> 5));

  if ((tid % (blockDim.x >> 5) == 0) && (tid < ((blockDim.x * blockDim.y) >> 5))) {
    sum = shared_mem[tid];
#pragma unroll
    for (int r = 1; r < (blockDim.x >> 5); r++){
      sum += shared_mem[tid + r];
    }
    int store_id = tid / (blockDim.x >> 5);
    res[store_id * N + row] = __float2half(sum);
  }
}


///////////////////////////// QUANTIZED-INT8 //////////////////////////////

__global__ void gemv_quantized_int8(int8_t* mat, half* vec, half* res,
                                    unsigned int n, half scale, half zero_point,
                                    unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  half4* mat4 = reinterpret_cast<half4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = __half2float(zero_point);
  float scale_f = __half2float(scale);

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      half4 mat_val = mat4[row * (n >> 3) + j];
      half2* vec_h1 = (half2*)&vec_val.x;
      half2* vec_h2 = (half2*)&vec_val.y;
      half2* vec_h3 = (half2*)&vec_val.z;
      half2* vec_h4 = (half2*)&vec_val.w;
      int8_2* mat_h1 = (int8_2*)&mat_val.x;
      int8_2* mat_h2 = (int8_2*)&mat_val.y;
      int8_2* mat_h3 = (int8_2*)&mat_val.z;
      int8_2* mat_h4 = (int8_2*)&mat_val.w;
      sum += __half2float(vec_h1->x) *
             (static_cast<float>(mat_h1->x) - zero_point_f);
      sum += __half2float(vec_h1->y) *
             (static_cast<float>(mat_h1->y) - zero_point_f);
      sum += __half2float(vec_h2->x) *
             (static_cast<float>(mat_h2->x) - zero_point_f);
      sum += __half2float(vec_h2->y) *
             (static_cast<float>(mat_h2->y) - zero_point_f);
      sum += __half2float(vec_h3->x) *
             (static_cast<float>(mat_h3->x) - zero_point_f);
      sum += __half2float(vec_h3->y) *
             (static_cast<float>(mat_h3->y) - zero_point_f);
      sum += __half2float(vec_h4->x) *
             (static_cast<float>(mat_h4->x) - zero_point_f);
      sum += __half2float(vec_h4->y) *
             (static_cast<float>(mat_h4->y) - zero_point_f);
    }
  }

  sum *= scale_f;

  sum = warpReduceSum(sum, blockDim.x);

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
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}

///////////////////////////// QUANTIZED-INT4 //////////////////////////////

// based on previous experiments, num_per_thread can >= 16
__global__ void gemv_quantized_int4(uint4_2* mat, half* vec, half* res,
                                    unsigned int n, half scale, half zero_point,
                                    unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  uint4_2_4* mat4 = reinterpret_cast<uint4_2_4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = __half2float(zero_point);
  float scale_f = __half2float(scale);

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 16); iter++) {
    unsigned int j = 2 * (start_idx + iter * blockDim.x);
    if (j < n >> 3) {
      float4 vec_val_1 = vec4[j];  // 8 half
      float4 vec_val_2 = vec4[j + 1];
      half2* vec_h1 = (half2*)&vec_val_1.x;
      half2* vec_h2 = (half2*)&vec_val_1.y;
      half2* vec_h3 = (half2*)&vec_val_1.z;
      half2* vec_h4 = (half2*)&vec_val_1.w;
      half2* vec_h5 = (half2*)&vec_val_2.x;
      half2* vec_h6 = (half2*)&vec_val_2.y;
      half2* vec_h7 = (half2*)&vec_val_2.z;
      half2* vec_h8 = (half2*)&vec_val_2.w;

      uint4_2_4 mat_val_1 = mat4[row * (n >> 3) + j];
      uint4_2_4 mat_val_2 = mat4[row * (n >> 3) + j + 1];
      uint4_2* mat_h1 = (uint4_2*)&mat_val_1.x;
      uint4_2* mat_h2 = (uint4_2*)&mat_val_1.y;
      uint4_2* mat_h3 = (uint4_2*)&mat_val_1.z;
      uint4_2* mat_h4 = (uint4_2*)&mat_val_1.w;
      uint4_2* mat_h5 = (uint4_2*)&mat_val_2.x;
      uint4_2* mat_h6 = (uint4_2*)&mat_val_2.y;
      uint4_2* mat_h7 = (uint4_2*)&mat_val_2.z;
      uint4_2* mat_h8 = (uint4_2*)&mat_val_2.w;

      sum += __half2float(vec_h1->x) *
             (static_cast<float>(mat_h1->getX()) - zero_point_f);
      sum += __half2float(vec_h1->y) *
             (static_cast<float>(mat_h1->getY()) - zero_point_f);
      sum += __half2float(vec_h2->x) *
             (static_cast<float>(mat_h2->getX()) - zero_point_f);
      sum += __half2float(vec_h2->y) *
             (static_cast<float>(mat_h2->getY()) - zero_point_f);
      sum += __half2float(vec_h3->x) *
             (static_cast<float>(mat_h3->getX()) - zero_point_f);
      sum += __half2float(vec_h3->y) *
             (static_cast<float>(mat_h3->getY()) - zero_point_f);
      sum += __half2float(vec_h4->x) *
             (static_cast<float>(mat_h4->getX()) - zero_point_f);
      sum += __half2float(vec_h4->y) *
             (static_cast<float>(mat_h4->getY()) - zero_point_f);
      sum += __half2float(vec_h5->x) *
             (static_cast<float>(mat_h5->getX()) - zero_point_f);
      sum += __half2float(vec_h5->y) *
             (static_cast<float>(mat_h5->getY()) - zero_point_f);
      sum += __half2float(vec_h6->x) *
             (static_cast<float>(mat_h6->getX()) - zero_point_f);
      sum += __half2float(vec_h6->y) *
             (static_cast<float>(mat_h6->getY()) - zero_point_f);
      sum += __half2float(vec_h7->x) *
             (static_cast<float>(mat_h7->getX()) - zero_point_f);
      sum += __half2float(vec_h7->y) *
             (static_cast<float>(mat_h7->getY()) - zero_point_f);
      sum += __half2float(vec_h8->x) *
             (static_cast<float>(mat_h8->getX()) - zero_point_f);
      sum += __half2float(vec_h8->y) *
             (static_cast<float>(mat_h8->getY()) - zero_point_f);
    }
  }

  sum *= scale_f;

  sum = warpReduceSum(sum, blockDim.x);

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
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}

#endif  // FAST_GEMV_CUH_


///////////////////////////// NORMAL //////////////////////////////
// thread_per_block = blockDim.x
// blockDim.y <= SHARED_MEM_MAX_ROWS
__global__ void gemv_fp16(half* mat, half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
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
  }

  sum = warpReduceSum(sum, blockDim.x);

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
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}