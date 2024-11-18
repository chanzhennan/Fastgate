

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


__inline__ __device__ void barrier_wait(__mbarrier_t* bar, __mbarrier_token_t token)
{
    bool state = false;
    // 只让一个线程检查状态，其他线程等待
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        while (!state) {
            state = __mbarrier_test_wait(bar, token);
        }
    }
    __syncthreads();  // 同步所有线程
}

template<int STAGE = 3>
__global__ void gemv_fp16(half* mat, half* vec, half* res, unsigned int k, unsigned int n,
                          unsigned int num_per_thread) {
    float sum = 0;
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int start_idx = threadIdx.x;
    __shared__ float4 vec_shared[STAGE][128];
    __shared__ float4 mat_shared[STAGE][4][128];

    if (row >= n) {  // 检查是否超出有效范围
        printf("Warning: Invalid row %d (n=%d)\n", row, n);
        return;
    }

    // Add this near the top of the kernel
    bool should_print = (row == 29 && tid == 0);

    // if (should_print) {
    //     printf("\n=== Basic Thread Information ===\n");
    //     printf("Thread ID (tid): %u\n", tid);
    //     printf("Row: %u\n", row);
    //     printf("Block Dims: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    //     printf("Block Index: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
    //     printf("Thread Index: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    //     printf("Matrix Dimensions - K: %u, N: %u\n", k, n);
    //     printf("Num per thread: %u\n", num_per_thread);
    //     printf("Start Index: %u\n", start_idx);
    //     printf("===========================\n\n");
    // }

    //   // 打印全局内存中前三个stage的数据
    // if (should_print)  {
    //     printf("Global memory data for first 3 stages:\n");
    //     for (int s = 0; s < STAGE; s++) {
    //         size_t curr_j = s * blockDim.x;
    //         if (curr_j < k >> 3) {
    //             float4 vec_tmp = reinterpret_cast<float4*>(vec)[curr_j];
    //             float4 mat_tmp = reinterpret_cast<float4*>(mat)[curr_j];
                
    //             half2* vec_h1 = (half2*)&vec_tmp.x;
    //             half2* vec_h2 = (half2*)&vec_tmp.y;
    //             half2* vec_h3 = (half2*)&vec_tmp.z;
    //             half2* vec_h4 = (half2*)&vec_tmp.w;
                
    //             half2* mat_h1 = (half2*)&mat_tmp.x;
    //             half2* mat_h2 = (half2*)&mat_tmp.y;
    //             half2* mat_h3 = (half2*)&mat_tmp.z;
    //             half2* mat_h4 = (half2*)&mat_tmp.w;
                
    //             printf("tid = %d, Stage %d:\n", tid, s);
    //             printf("Vec[%d]: %f %f %f %f %f %f %f %f\n", 
    //                    curr_j,
    //                    __half2float(vec_h1->x), __half2float(vec_h1->y),
    //                    __half2float(vec_h2->x), __half2float(vec_h2->y),
    //                    __half2float(vec_h3->x), __half2float(vec_h3->y),
    //                    __half2float(vec_h4->x), __half2float(vec_h4->y));
    //             printf("Mat[%d]: %f %f %f %f %f %f %f %f\n", 
    //                    curr_j,
    //                    __half2float(mat_h1->x), __half2float(mat_h1->y),
    //                    __half2float(mat_h2->x), __half2float(mat_h2->y),
    //                    __half2float(mat_h3->x), __half2float(mat_h3->y),
    //                    __half2float(mat_h4->x), __half2float(mat_h4->y));
    //         }
    //     }
    //     printf("\n");
    // }

    // 创建pipeline
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // 预加载所有pipeline阶段
    for (int stage = 0; stage < STAGE; ++stage) {
        pipe.producer_acquire();
        size_t curr_j = start_idx + stage * blockDim.x;
        if (curr_j < k >> 3) {
            if (threadIdx.y == 0) {
                cuda::memcpy_async(&vec_shared[stage][start_idx],
                                 &reinterpret_cast<float4*>(vec)[curr_j],
                                 sizeof(float4),
                                 pipe);
            }
            cuda::memcpy_async(&mat_shared[stage][threadIdx.y][start_idx],
                             &reinterpret_cast<float4*>(mat)[row * (k >> 3) + curr_j],
                             sizeof(float4),
                             pipe);
        }
        pipe.producer_commit();
    }
    __syncthreads();


    // 主循环处理
    int stage = 0;
    for (size_t iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
        size_t curr_j = start_idx + iter * blockDim.x;
        if (curr_j >= k >> 3) break;

        // 等待当前阶段数据就绪
        // cuda::pipeline_consumer_wait_prior<STAGE - 1>(pipe);
        pipe.consumer_wait();
        __syncthreads();
        // if (should_print) {
        //   half2* vec_h = (half2*)&vec_shared[stage][0];
        //   half2* mat_h = (half2*)&mat_shared[stage][0][0];
        //   printf("tid = %d, Stage %d, First few values:\n", tid, stage);
        //   printf("Vec: %f %f %f %f\n", 
        //         __half2float(vec_h[0].x), __half2float(vec_h[0].y),
        //         __half2float(vec_h[1].x), __half2float(vec_h[1].y));
        //   printf("Mat: %f %f %f %f\n", 
        //         __half2float(mat_h[0].x), __half2float(mat_h[0].y),
        //         __half2float(mat_h[1].x), __half2float(mat_h[1].y));
        // }

        // 处理当前阶段数据
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

        __syncthreads();
        pipe.consumer_release();
        __syncthreads();
        // 加载下一批数据
        size_t next_j = start_idx + (iter + STAGE) * blockDim.x;
        if (next_j < k >> 3) {
            pipe.producer_acquire();
            if (threadIdx.y == 0) {
                cuda::memcpy_async(&vec_shared[stage][start_idx],
                                 &reinterpret_cast<float4*>(vec)[next_j],
                                 sizeof(float4),
                                 pipe);
            }
            cuda::memcpy_async(&mat_shared[stage][threadIdx.y][start_idx],
                             &reinterpret_cast<float4*>(mat)[row * (k >> 3) + next_j],
                             sizeof(float4),
                             pipe);
            pipe.producer_commit();
        }

        stage = (stage + 1) % STAGE;
    }
    // if(row == 185) {
    //   printf("row = %d, tid = %d, Final sum: %f\n", row, tid, sum);
    // }

  sum = warpReduceSumFloat(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      // printf("11 row: %d, sum: %f\n", row, sum);
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
    // printf("row: %d, sum: %f\n", row, sum);
    res[row] = __float2half(sum);
  }


}




// __global__ void gemv_fp16(half* mat, half* vec, half* res, unsigned int n,
//                           unsigned int num_per_thread) {
//   float sum = 0;
//   // each thread load num_per_thread elements from global
//   unsigned int tid = threadIdx.x;
//   unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
//   unsigned int start_idx = threadIdx.x;
//   float4* mat4 = reinterpret_cast<float4*>(mat);
//   float4* vec4 = reinterpret_cast<float4*>(vec);

// #pragma unroll
//   for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
//     unsigned int j = start_idx + iter * blockDim.x;
//     if (j < n >> 3) {
//       float4 vec_val = vec4[j];
//       float4 mat_val = mat4[row * (n >> 3) + j];
//       half2* vec_h1 = (half2*)&vec_val.x;
//       half2* vec_h2 = (half2*)&vec_val.y;
//       half2* vec_h3 = (half2*)&vec_val.z;
//       half2* vec_h4 = (half2*)&vec_val.w;
//       half2* mat_h1 = (half2*)&mat_val.x;
//       half2* mat_h2 = (half2*)&mat_val.y;
//       half2* mat_h3 = (half2*)&mat_val.z;
//       half2* mat_h4 = (half2*)&mat_val.w;
//       sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
//       sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
//       sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
//       sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
//       sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
//       sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
//       sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
//       sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);
//     }
//   }

//   sum = warpReduceSumFloat(sum, blockDim.x);

//   if (blockDim.x <= WARP_SIZE) {
//     if (tid == 0) {
//       res[row] = __float2half(sum);
//     }
//     return;
//   }

//   // Shared mem for partial sums (one per warp in the block)
//   static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
//   const int laneId = threadIdx.x % WARP_SIZE;
//   const int warpId = threadIdx.x / WARP_SIZE;
//   if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
//   __syncthreads();
//   // read from shared memory only if that warp existed
//   sum = (threadIdx.x < blockDim.x / WARP_SIZE)
//             ? warpLevelSums[threadIdx.y][laneId]
//             : 0.0;
//   // Final reduce using first warp
//   if (warpId == 0) sum = warpReduceSumFloat(sum, blockDim.x / WARP_SIZE);
//   if (tid == 0) {
//     res[row] = __float2half(sum);
//   }
// }

#endif  // FAST_GEMV_CZN_CUH_
