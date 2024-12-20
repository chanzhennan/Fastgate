

#ifndef FAST_GEMV_CZN_CUH_
#define FAST_GEMV_CZN_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <cuda_pipeline_primitives.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#include "../hk/utility.cuh"

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 4
#define MAX_THREADS_PER_BLOCK 1024

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

__device__ __forceinline__ float warpReduceSumFloat(float &sum,
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

__inline__ __device__ uint32_t cast_smem_ptr_to_uint(void const* const ptr)
{
    return (uint32_t)__cvta_generic_to_shared(ptr);
}

template<typename AccessType>
struct CpAsync {
    template<typename T>
    __device__ void operator()(int ptr, const T* __restrict__ src, bool mask = true) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        constexpr int size = sizeof(AccessType);
        if constexpr (size == 16) {
            asm volatile(
                "cp.async.cg.shared.global" L2_CACHEHINT(128) " [%0], [%1], %2;\n" ::"r"(ptr), "l"(src), "n"(size));
        }
        else {
            asm volatile(
                "cp.async.ca.shared.global" L2_CACHEHINT(128) " [%0], [%1], %2;\n" ::"r"(ptr), "l"(src), "n"(size));
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

__global__ void gemv_fp16(half* mat, half* vec, half* res, unsigned int k, unsigned int n,
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
    if (j < k >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (k >> 3) + j];
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



// template<int STAGE = 3>
// __global__ void gemv_fp16_bs2(half* mat, half* vec, half* res, unsigned int k, unsigned int n,
//                           unsigned int num_per_thread) {
//     float sum[2] = {0.f, 0.f};
//     unsigned int tid = threadIdx.x;
//     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

//     unsigned int start_idx = threadIdx.x;
//     __shared__ float4 vec_shared[STAGE][2][128];
//     __shared__ float4 mat_shared[STAGE][4][128];

//     if (row >= n) {
//         printf("Warning: Invalid row %d (n=%d)\n", row, n);
//         return;
//     }

//     size_t tmp_j;
//     CpAsync<float4> cp;

//     // 预加载所有pipeline阶段
//     #pragma unroll
//     for (int stage = 0; stage < STAGE; ++stage) {
//         // pipe.producer_acquire();
//         tmp_j = start_idx + stage * blockDim.x;
//         if (tmp_j < k >> 3) {
//           auto vec_uint = cast_smem_ptr_to_uint(&vec_shared[stage][0][start_idx]);
//           auto vec_uint_2 = cast_smem_ptr_to_uint(&vec_shared[stage][1][start_idx]);
//           auto mat_uint = cast_smem_ptr_to_uint(&mat_shared[stage][threadIdx.y][start_idx]);
//           cp(vec_uint, &reinterpret_cast<float4*>(vec)[tmp_j]);
//           cp(vec_uint_2, &reinterpret_cast<float4*>(vec + k)[tmp_j]);
//           cp(mat_uint, &reinterpret_cast<float4*>(mat)[row * (k >> 3) + tmp_j]);
//           __pipeline_commit();
//         }
//     }

   

//     // 主循环处理
//     int stage = 0;
//     #pragma unroll
//     for (size_t iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
//         tmp_j = start_idx + iter * blockDim.x;
//         if (tmp_j >= k >> 3) break;

//         // 等待当前阶段数据就绪
//         __pipeline_wait_prior(STAGE - 1);
//         __syncthreads();

//         // if (tid == 0 && row == 0) {
//         //       float4 vec_val = vec_shared[stage][0][start_idx];
//         //       float4 vec_val_2 = vec_shared[stage][1][start_idx];
//         //       float4 mat_val = mat_shared[stage][threadIdx.y][start_idx];

//         //       half2* vec_h1 = (half2*)&vec_val.x;
//         //       half2* vec_h2 = (half2*)&vec_val_2.x;
//         //       half2* mat_val_1 = (half2*)&mat_val.x;


//         //       printf("vec_shared[%d][0][%d]: %f %f \n", stage, start_idx, __half2float(vec_h1->x), __half2float(vec_h1->y));
//         //       printf("vec_shared_2[%d][1][%d]: %f %f \n", stage, start_idx, __half2float(vec_h2->x), __half2float(vec_h2->y));
//         //       printf("mat_shared[%d][%d][%d]: %f %f \n", stage, threadIdx.y, start_idx, __half2float(mat_val_1->x), __half2float(mat_val_1->y));
//         // } 
//         // 处理当前阶段数据
//         float4 vec_val = vec_shared[stage][0][start_idx];
//         float4 vec_val_2 = vec_shared[stage][1][start_idx];
//         float4 mat_val = mat_shared[stage][threadIdx.y][start_idx];
        
//         half2* vec_h1 = (half2*)&vec_val.x;
//         half2* vec_h2 = (half2*)&vec_val.y;
//         half2* vec_h3 = (half2*)&vec_val.z;
//         half2* vec_h4 = (half2*)&vec_val.w;
        
//         half2* mat_h1 = (half2*)&mat_val.x;
//         half2* mat_h2 = (half2*)&mat_val.y;
//         half2* mat_h3 = (half2*)&mat_val.z;
//         half2* mat_h4 = (half2*)&mat_val.w;
        
//         sum[0] += __half2float(vec_h1->x) * __half2float(mat_h1->x);
//         sum[0] += __half2float(vec_h1->y) * __half2float(mat_h1->y);
//         sum[0] += __half2float(vec_h2->x) * __half2float(mat_h2->x);
//         sum[0] += __half2float(vec_h2->y) * __half2float(mat_h2->y);
//         sum[0] += __half2float(vec_h3->x) * __half2float(mat_h3->x);
//         sum[0] += __half2float(vec_h3->y) * __half2float(mat_h3->y);
//         sum[0] += __half2float(vec_h4->x) * __half2float(mat_h4->x);
//         sum[0] += __half2float(vec_h4->y) * __half2float(mat_h4->y);

//         vec_h1 = (half2*)&vec_val_2.x;
//         vec_h2 = (half2*)&vec_val_2.y;
//         vec_h3 = (half2*)&vec_val_2.z;
//         vec_h4 = (half2*)&vec_val_2.w;

//         sum[1] += __half2float(vec_h1->x) * __half2float(mat_h1->x);
//         sum[1] += __half2float(vec_h1->y) * __half2float(mat_h1->y);
//         sum[1] += __half2float(vec_h2->x) * __half2float(mat_h2->x);
//         sum[1] += __half2float(vec_h2->y) * __half2float(mat_h2->y);
//         sum[1] += __half2float(vec_h3->x) * __half2float(mat_h3->x);
//         sum[1] += __half2float(vec_h3->y) * __half2float(mat_h3->y);
//         sum[1] += __half2float(vec_h4->x) * __half2float(mat_h4->x);
//         sum[1] += __half2float(vec_h4->y) * __half2float(mat_h4->y);


//         // 加载下一批数据
//         tmp_j = start_idx + (iter + STAGE) * blockDim.x;
//         if (tmp_j < k >> 3) {
//           auto vec_uint = cast_smem_ptr_to_uint(&vec_shared[stage][0][start_idx]);
//           auto vec_uint_2 = cast_smem_ptr_to_uint(&vec_shared[stage][1][start_idx]);
//           auto mat_uint = cast_smem_ptr_to_uint(&mat_shared[stage][threadIdx.y][start_idx]);
//           cp(vec_uint, &reinterpret_cast<float4*>(vec)[tmp_j]);
//           cp(vec_uint_2, &reinterpret_cast<float4*>(vec + k)[tmp_j]);
//           cp(mat_uint, &reinterpret_cast<float4*>(mat)[row * (k >> 3) + tmp_j]);
//           __pipeline_commit();
//         }

//         stage = (stage + 1) % STAGE;
//     }

//   sum[0] = warpReduceSumFloat(sum[0], blockDim.x);
//   sum[1] = warpReduceSumFloat(sum[1], blockDim.x);

//   if (blockDim.x <= WARP_SIZE) {
//     if (tid == 0) {
//       res[row] = __float2half(sum[0]);
//       res[row + n] = __float2half(sum[1]);
//     }
//     return;
//   }

//   // Shared mem for partial sums (one per warp in the block)
//   static __shared__ float warpLevelSums[2][SHARED_MEM_MAX_ROWS][WARP_SIZE];
//   const int laneId = threadIdx.x % WARP_SIZE;
//   const int warpId = threadIdx.x / WARP_SIZE;
//   if (laneId == 0) {
//     warpLevelSums[0][threadIdx.y][warpId] = sum[0];
//     warpLevelSums[1][threadIdx.y][warpId] = sum[1];
//   }

//   __syncthreads();
//   // read from shared memory only if that warp existed
//   sum[0] = (threadIdx.x < blockDim.x / WARP_SIZE)
//             ? warpLevelSums[0][threadIdx.y][laneId]
//             : 0.0;
//   sum[1] = (threadIdx.x < blockDim.x / WARP_SIZE)
//             ? warpLevelSums[1][threadIdx.y][laneId]
//             : 0.0;


//   // Final reduce using first warp
//   if (warpId == 0) {
//     sum[0] = warpReduceSumFloat(sum[0], blockDim.x / WARP_SIZE);
//     sum[1] = warpReduceSumFloat(sum[1], blockDim.x / WARP_SIZE);
//   }

//   if (tid == 0) {
//     res[row] = __float2half(sum[0]);
//     res[row + n] = __float2half(sum[1]);
//   }


// }



__global__ void gemv_fp16_bs2(half* mat, half* vec, half* res, unsigned int k, unsigned int n,
                          unsigned int num_per_thread) {
  float sum[2] = {0.f, 0.f};
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);
  float4* vec4_2 = reinterpret_cast<float4*>(vec + k);

  if(row >= n) return;

  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < k >> 3) {
      float4 vec_val = vec4[j];
      float4 vec_val_2 = vec4_2[j];
      float4 mat_val = mat4[row * (k >> 3) + j];
      half2* vec_h1 = (half2*)&vec_val.x;
      half2* vec_h2 = (half2*)&vec_val.y;
      half2* vec_h3 = (half2*)&vec_val.z;
      half2* vec_h4 = (half2*)&vec_val.w;
      half2* mat_h1 = (half2*)&mat_val.x;
      half2* mat_h2 = (half2*)&mat_val.y;
      half2* mat_h3 = (half2*)&mat_val.z;
      half2* mat_h4 = (half2*)&mat_val.w;
      sum[0] += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum[0] += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum[0] += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum[0] += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum[0] += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum[0] += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum[0] += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum[0] += __half2float(vec_h4->y) * __half2float(mat_h4->y);

      vec_h1 = (half2*)&vec_val_2.x;
      vec_h2 = (half2*)&vec_val_2.y;
      vec_h3 = (half2*)&vec_val_2.z;
      vec_h4 = (half2*)&vec_val_2.w;
      sum[1] += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum[1] += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum[1] += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum[1] += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum[1] += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum[1] += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum[1] += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum[1] += __half2float(vec_h4->y) * __half2float(mat_h4->y);

    }
  }

  sum[0] = warpReduceSumFloat(sum[0], blockDim.x);
  sum[1] = warpReduceSumFloat(sum[1], blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum[0]);
      res[row + n] = __float2half(sum[1]);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[2][SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) {
    warpLevelSums[0][threadIdx.y][warpId] = sum[0];
    warpLevelSums[1][threadIdx.y][warpId] = sum[1];
  }

  __syncthreads();
  // read from shared memory only if that warp existed
  sum[0] = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[0][threadIdx.y][laneId]
            : 0.0;
  sum[1] = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[1][threadIdx.y][laneId]
            : 0.0;

  // Final reduce using first warp
  if (warpId == 0) {
    sum[0] = warpReduceSumFloat(sum[0], blockDim.x / WARP_SIZE);
    sum[1] = warpReduceSumFloat(sum[1], blockDim.x / WARP_SIZE);
  }
  if (tid == 0) {
    res[row] = __float2half(sum[0]);
    res[row + n] = __float2half(sum[1]);
  }
}



#endif  // FAST_GEMV_CZN_CUH_
