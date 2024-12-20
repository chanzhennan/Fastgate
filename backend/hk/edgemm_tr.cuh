/*
    A collection of Flat-GEMM with transposed weight (B). @Infinigence.
*/
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <cub/cub.cuh>
#include <mma.h>

#define WARP_SIZE 32
#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

using namespace nvcuda;
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void gemm_m8n32k128x8_bz1(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half s_a[BM * (LDK)];
    __shared__ half s_b[BN * (LDK)];

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    /**
     * 通过位运算获取每个thread对应的索引位置
     * load_a 每个warp访问1*64个元素，通过cp.async指定访问8B即4个half完成
     * load_b 每个warp访问4*256个元素，通过cp.async指定访问16B即8个half完成
    */
    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   每个索引32个一组 共8组
    int load_a_smem_k = (tid & 31) << 2; // 0 ~ 124  | 0 4  8 ... 124(32个数)  循环8组  间隔是4个half 8B
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   每个索引32个一组 共8组
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    #pragma unroll 32
    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        if (load_a_gmem_m < M) {
            asm("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
            asm("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        wmma::load_matrix_sync(frag_a, &s_a[wid * 16], LDK);
        wmma::load_matrix_sync(frag_b, &s_b[wid * 16], LDK);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);


        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
        __syncthreads();
    }

    wmma::store_matrix_sync(&s_b[wid * 32], frag_c, 8 * 32 + 8, wmma::mem_row_major);

    short store_c_m = (tid & 31) >> 2, store_c_n = (tid & 3) << 3;
    int store_c_smem_addr = OFFSET(store_c_m, wid * 32 + store_c_n, 8 * 32 + 8);
    int store_c_gmem_addr = OFFSET(store_c_m, store_c_n + bx * BN, N);

    if (store_c_m < M) {
        #pragma unroll
        for(int i=0; i<4; i++){
            atomicAdd(((half2 *)(&c[store_c_gmem_addr + 2 * i])),
                      *((half2 *)(&s_b[store_c_smem_addr + 2 * i])));
        }
    }
}


/***
 * B转置 A:M*K B:N*K
 * BM=8 BN=32 BK=256
 * LDK = BK + PAD = 264
 * LDN = BN + PAD = 40
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void gemm_m8n32k256x8_bz1(
    const half * __restrict__ a, const half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    // __shared__ half s_a[BM * (LDK)];
    // __shared__ half s_b[BN * (LDK)];
    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    /**
     * 通过位运算获取每个thread对应的索引位置
     * load_a 每个warp访问1*64个元素，通过cp.async指定访问8B即4个half完成
     * load_b 每个warp访问4*256个元素，通过cp.async指定访问16B即8个half完成
    */
    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   每个索引32个一组 共8组
    int load_ab_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248(32个数)  循环8组  间隔是8个half 16B
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   每个索引32个一组 共8组
    // int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    // size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    // size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = __cvta_generic_to_shared(s_a) + OFFSET(load_a_smem_m, load_ab_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = __cvta_generic_to_shared(s_b) + OFFSET(load_b_smem_n, load_ab_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    // int load_a_gmem_m = load_a_smem_m;
    // int load_b_gmem_n = bx * BN + load_b_smem_n;
    // int load_ab_gmem_k = k_start + load_ab_smem_k;
    // int load_b_gmem_k = k_start + load_ab_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_smem_m, (k_start + load_ab_smem_k), K);
    int load_b_gmem_addr = OFFSET((bx * BN + load_b_smem_n), (k_start + load_ab_smem_k), K);

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        if (load_a_smem_m < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
            //     : "r"(load_b_smem_addrs[i]), "l"(ptr_b));
            // ptr_b += K;
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    // wmma::store_matrix_sync(&s_b[wid * 32], frag_c, 8 * 32 + 8, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem[wid * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    // int shmem_c_m = tid >> 5;   // 0, 1, 2, 3, 4, 5, 6, 7
    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(load_a_smem_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(load_a_smem_m, bx * BN + shmem_c_n, N);

    
#pragma unroll
    for (int s = 1; s < 8; s++){
        smem[shmem_c_addr] = __hadd(smem[shmem_c_addr], smem[shmem_c_addr + s * 8 * LDN]);
    }

    __syncthreads();

    // short store_c_m = (tid & 31) >> 2, store_c_n = (tid & 3) << 3;
    // int store_c_smem_addr = OFFSET(store_c_m, wid * 32 + store_c_n, 8 * 32 + 8);
    // int store_c_gmem_addr = OFFSET(store_c_m, store_c_n + bx * BN, N);

    if (load_a_smem_m < M) {
        // #pragma unroll
        // for(int i=0; i<4; i++){
        //     atomicAdd(((half2 *)(&c[gmem_c_addr + 2 * i])),
        //               *((half2 *)(&s_b[shmem_c_addr + 2 * i])));
        // }
        // c[gmem_c_addr] = smem[shmem_c_addr];
        atomicAdd(&c[gmem_c_addr], smem[shmem_c_addr]);
    }
}


/***
 * B转置 A:M*K B:N*K
 * BM=8 BN=256 BK=32
 * LDK = BK + PAD = 40
 * LDN = BN + PAD = 264
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void gemm_m8n64k128x8_bz1(
    const half * __restrict__ a, const half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    // __shared__ half s_a[BM * (LDK)];
    // __shared__ half s_b[BN * (LDK)];
    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    /**
     * 通过位运算获取每个thread对应的索引位置
     * load_a 每个warp访问1*64个元素，通过cp.async指定访问8B即4个half完成
     * load_b 每个warp访问4*256个元素，通过cp.async指定访问16B即8个half完成
    */
    int load_a_smem_m = (tid >> 5);      // 0 ~ 7   
    int load_a_smem_k = (tid & 31) << 2; // 0 ~ 31 x 4 
    int load_b_smem_n = (tid >> 4) << 2; // 0 ~ 64
    int load_b_smem_k = (tid & 15) << 3;  // 0 ~ 15 x 8

    unsigned int w_row = wid / 4;  // 0, 1
    unsigned int w_col = wid % 4;  // 0, 1, 2, 3

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_smem_addr = OFFSET(load_a_smem_m, load_a_smem_k, LDK);
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        if (load_a_gmem_m < M) {
            asm("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
            //     : "r"(load_b_smem_addrs[i]), "l"(ptr_b));
            // ptr_b += K;
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[w_col * 16 + 64 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[w_row * 32 * LDK + w_col * 16 + 64 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    // wmma::store_matrix_sync(&s_b[wid * 32], frag_c, 8 * 32 + 8, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem[w_col * 8 * LDN + w_row * 32], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 5;          // 0, 1, 2, 3, 4, 5, 6, 7
    int shmem_c_n = (tid & 31) << 1;   // 0, 1, 2, 3, 4, ..., 31 x 2
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

#pragma unroll
    for (int s = 1; s < 4; s++){
        *(half2*)(&smem[shmem_c_addr]) = 
            __hadd2(*(half2*)(&smem[shmem_c_addr]), *(half2*)(&smem[shmem_c_addr + s * 8 * LDN]));
    }

    __syncthreads();

    if (shmem_c_m < M) {
        // *(half2*)(&c[gmem_c_addr]) = *(half2*)(&smem[shmem_c_addr]);
        atomicAdd((half2*)(&c[gmem_c_addr]), *(half2*)(&smem[shmem_c_addr]));
    }
}


// BM = 16, BN = 128, BK = 64
// LDK = BK + APAD = 72
// LDN = BN + BPAD = 136
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void gemm_m8n128k64x4_v8_tr(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5; // 0, 1, 2, 3, 4, 5, 6, 7

    // only support dim-M [1, 8]
    if (bx >= N / BN)
        return;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * LDK;
    int s_a_db_offset = BM * LDK;
    int s_b_db_offset = BN * LDK;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 4);       // 0 ~ 15
    int load_a_smem_k = (tid & 15) << 2;  // 0 ~ 60
    int load_b_smem_n = (tid >> 3) << 2;  // 0 ~ 120
    int load_b_smem_k = (tid & 7) << 3;  // 0 ~ 60

    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int B_UNIT = LDK * sizeof(half);
    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     B_UNIT;
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * B_UNIT;
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * B_UNIT;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    // load the first tile of mat_a & mat_b
    {
        if (load_a_gmem_m < M) {
            asm("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(load_a_smem_addr_0),
                  "l"(&a[load_a_gmem_addr]));
        }
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * K]));
    
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < (K / gridDim.z) / BK; bk++) {
        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;

        int loop_offset_a = smem_sel_next * s_a_db_offset * (int)sizeof(half);
        int loop_offset_b = smem_sel_next * s_b_db_offset * (int)sizeof(half);

        // async load the other tile of mat_a & mat_b
        // bk is odd?
        // if (bk % 2 == 0) {
        if (load_a_gmem_m < M) {
            asm("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                    : "r"(load_a_smem_addr_0 + loop_offset_a),
                      "l"(&a[load_a_gmem_addr]));
        }

        // }
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + loop_offset_b), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + loop_offset_b), "l"(&b[load_b_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + loop_offset_b), "l"(&b[load_b_gmem_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + loop_offset_b), "l"(&b[load_b_gmem_addr + 3 * K]));
    
        asm("cp.async.commit_group;\n" ::);  // issue cp.async.wait_group at the end of loop body

        // compute A X B for this bk
        // note that BK / TILE_K = 2
        int s_a_addr = smem_sel * s_a_db_offset;
        int s_b_addr = smem_sel * s_b_db_offset + wid * 16 * LDK;
        
        wmma::load_matrix_sync(frag_a[0], &s_a[s_a_addr], LDK);
        wmma::load_matrix_sync(frag_a[1], &s_a[s_a_addr + 16], LDK);
        wmma::load_matrix_sync(frag_a[2], &s_a[s_a_addr + 32], LDK);
        wmma::load_matrix_sync(frag_a[3], &s_a[s_a_addr + 48], LDK);

        wmma::load_matrix_sync(frag_b[0], &s_b[s_b_addr], LDK);
        wmma::load_matrix_sync(frag_b[1], &s_b[s_b_addr + 16], LDK);
        wmma::load_matrix_sync(frag_b[2], &s_b[s_b_addr + 32], LDK);
        wmma::load_matrix_sync(frag_b[3], &s_b[s_b_addr + 48], LDK);

        wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
        wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);
        wmma::mma_sync(frag_c, frag_a[2], frag_b[2], frag_c);
        wmma::mma_sync(frag_c, frag_a[3], frag_b[3], frag_c);

        asm("cp.async.wait_group 0;\n" ::);

        // it seems compute correctly without this sync.
        // if without this sync, the runtime is reduced by 10us
        __syncthreads();
    }

    int s_a_addr = s_a_db_offset;
    int s_b_addr = s_b_db_offset + wid * 16 * LDK;

    wmma::load_matrix_sync(frag_a[0], &s_a[s_a_addr], LDK);
    wmma::load_matrix_sync(frag_a[1], &s_a[s_a_addr + 16], LDK);
    wmma::load_matrix_sync(frag_a[2], &s_a[s_a_addr + 32], LDK);
    wmma::load_matrix_sync(frag_a[3], &s_a[s_a_addr + 48], LDK);

    wmma::load_matrix_sync(frag_b[0], &s_b[s_b_addr], LDK);
    wmma::load_matrix_sync(frag_b[1], &s_b[s_b_addr + 16], LDK);
    wmma::load_matrix_sync(frag_b[2], &s_b[s_b_addr + 32], LDK);
    wmma::load_matrix_sync(frag_b[3], &s_b[s_b_addr + 48], LDK);

    wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
    wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);
    wmma::mma_sync(frag_c, frag_a[2], frag_b[2], frag_c);
    wmma::mma_sync(frag_c, frag_a[3], frag_b[3], frag_c);

    wmma::store_matrix_sync(&smem[wid * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int load_c_smem_n = (tid & 15) << 3;  // 0 ~ 120
    int load_c_gmem_n = bx * BN + load_c_smem_n;
    int store_c_smem_addr = OFFSET(load_a_smem_m, load_c_smem_n, LDN);
    int store_c_gmem_addr = OFFSET(load_a_gmem_m, load_c_gmem_n, N);

    if (load_a_gmem_m < M) {
        if (gridDim.z > 1) {
            atomicAdd(((half2 *)(&c[store_c_gmem_addr])),
                      *((half2 *)(&smem[store_c_smem_addr])));
            atomicAdd(((half2 *)(&c[store_c_gmem_addr + 2])),
                      *((half2 *)(&smem[store_c_smem_addr + 2])));
            atomicAdd(((half2 *)(&c[store_c_gmem_addr + 4])),
                      *((half2 *)(&smem[store_c_smem_addr + 4])));
            atomicAdd(((half2 *)(&c[store_c_gmem_addr + 6])),
                      *((half2 *)(&smem[store_c_smem_addr + 6])));
        } else {
            *((float4*)(&c[store_c_gmem_addr])) = *((float4*)(&smem[store_c_smem_addr]));
        }
    }
}
