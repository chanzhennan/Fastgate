#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

/***
 * A:M*K B:K*N
 * BM=8 BN=128 BK=64
 * LDK = BK + PAD = 72
 * LDN = BN + PAD = 136
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void gemm_m8n128k64x4_bz1(
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
    __shared__ half smem[(BM * LDK + BK * LDN)];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    /**
     * 通过位运算获取每个thread对应的索引位置
     * load_a 每个warp访问1*64个元素，通过cp.async指定访问8B即4个half完成
     * load_b 每个warp访问4*256个元素，通过cp.async指定访问16B即8个half完成
    */
    int load_a_smem_m = (tid >> 4);      // 0 ~ 7   
    int load_a_smem_k = (tid & 15) << 2; // 0 ~ 15 x 4 
    int load_b_smem_k = (tid >> 4) << 3; // 0 ~ 7 x 8
    int load_b_smem_n = (tid & 15) << 3; // 0 ~ 15 x 8

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[8];
    #pragma unroll
    for(int i=0; i<8; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, LDN) * sizeof(half) + i * (LDN) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    // int load_a_smem_addr = OFFSET(load_a_smem_m, load_a_smem_k, LDK);
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        if (load_a_gmem_m < M) {
            asm("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<8; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * N]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<4; i++){
            wmma::load_matrix_sync(frag_a, &s_a[i * 16], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[i * 16 * LDN + wid * 32], LDN);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }
    
        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
    }

    // wmma::store_matrix_sync(&s_b[wid * 32], frag_c, 8 * 32 + 8, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem[wid * 32], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;          // 0, 1, 2, 3, 4, 5, 6, 7
    int shmem_c_n = (tid & 15) << 3;   // 0 ~ 15 x 8
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    if (shmem_c_m < M) {
        // *(float4*)(&c[gmem_c_addr]) = *(float4*)(&smem[shmem_c_addr]);
        #pragma unroll
        for(int i=0; i<4; i++)
            atomicAdd(((half2 *)(&c[gmem_c_addr + 2 * i])),
                      *((half2 *)(&smem[shmem_c_addr + 2 * i])));
    }
}


/***
 * A:M*K B:K*N
 * BM=8 BN=128 BK=64
 * LDK = BK + PAD = 72
 * LDN = BN + PAD = 136
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void gemm_m8n256k32x8_bz1(
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
    __shared__ half smem[(BM * LDK + BK * LDN)];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    /**
     * 通过位运算获取每个thread对应的索引位置
     * load_a 每个warp访问1*64个元素，通过cp.async指定访问8B即4个half完成
     * load_b 每个warp访问4*256个元素，通过cp.async指定访问16B即8个half完成
    */
    int load_a_smem_m = (tid >> 3);      // 0 ~ 31   
    int load_a_smem_k = (tid & 7) << 2;  // 0 ~ 7 x 4 
    int load_b_smem_k = (tid >> 5) << 2; // 0 ~ 7 x 4
    int load_b_smem_n = (tid & 31) << 3; // 0 ~ 31 x 8

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<8; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, LDN) * sizeof(half) + i * (LDN) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    // int load_a_smem_addr = OFFSET(load_a_smem_m, load_a_smem_k, LDK);
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

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
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * N]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[i * 16], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[i * 16 * LDN + wid * 32], LDN);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }
    
        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
    }

    // wmma::store_matrix_sync(&s_b[wid * 32], frag_c, 8 * 32 + 8, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem[wid * 32], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 5;          // 0, 1, 2, 3, 4, 5, 6, 7
    int shmem_c_n = (tid & 31) << 3;   // 0 ~ 31 x 8
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    if (shmem_c_m < M) {
        // *(float4*)(&c[gmem_c_addr]) = *(float4*)(&smem[gmem_c_addr]);
        #pragma unroll
        for(int i=0; i<4; i++)
            atomicAdd(((half2 *)(&c[gmem_c_addr + 2 * i])),
                      *((half2 *)(&smem[gmem_c_addr + 2 * i])));
    }
}


/***
 * BM=8 BN=256 BK=64
 * LDK = BK + APAD = 72
 * LDN = BN + PAD = 264
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void flat_gemm_m8n256k64x8_v3(
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
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half smem[BM * (LDK) + BK * (LDN)];
    half *s_a = smem;
    half *s_b = smem + BM * (LDK);

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    /**
     * 通过位运算获取每个thread对应的索引位置
     * load_a 每个warp访问1*64个元素，通过cp.async指定访问8B即4个half完成
     * load_b 每个warp访问4*256个元素，通过cp.async指定访问16B即8个half完成
    */
    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   每个索引32个一组 共8组
    int load_a_smem_k = (tid & 31) << 1; // 0 ~ 60    | 0 2  4 ... 60 (32个数)  循环8组  间隔是2个half 8B
    int load_b_smem_k = (tid >> 5) << 3; // 0 ~ 56   | 0 8 16 ... 56   每个索引32个一组 共8组
    int load_b_smem_n = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248(32个数)  循环8组  间隔是8个half 16B

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[8];
    #pragma unroll
    for(int i=0; i<8; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, LDN) * sizeof(half) + i * (LDN) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

    #pragma unroll 32
    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++) {

        if (load_a_gmem_m < M) {
            asm("cp.async.ca.shared.global [%0], [%1], 4;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<8; i++)
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * N]));

        asm("cp.async.commit_group;\n" ::);  // issue cp.async.wait_group at the end of loop body
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(frag_a, &s_a[i * 16], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[i * 16 * (LDN) + wid * 32], LDN);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
    }

    // if(tid==0&&bx==0) printf("%d %f %f %f %f %f %f\n", bz, __half2float(frag_c.x[0]), __half2float(frag_c.x[1]), __half2float(frag_a.x[0]), __half2float(frag_a.x[1]), __half2float(frag_b.x[0]), __half2float(frag_b.x[1]));
    wmma::store_matrix_sync(&smem[wid * 32], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();
    int store_c_smem_addr = OFFSET(load_a_smem_m, load_b_smem_n, LDN);
    int store_c_gmem_addr = OFFSET(load_a_gmem_m, load_b_gmem_n, N);

    if (load_a_gmem_m < M) {
        #pragma unroll
        for(int i=0; i<4; i++)
            atomicAdd(((half2 *)(&c[store_c_gmem_addr + 2 * i])),
                      *((half2 *)(&smem[store_c_smem_addr + 2 * i])));
        // *(float4*)(&c[store_c_gmem_addr]) = *(float4*)(&smem[store_c_smem_addr]);
    }
}


/***
 * BM=8 BN=256 BK=32
 * LDK = 2 * BK + APAD = 72
 * LDN = BN = 256
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void flat_gemm_m8n256k32x16_db(
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
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half smem[BM * (LDK) + 2 * BK * (LDN)];
    half *s_a = smem;
    half *s_b = smem + BM * (LDK);
    int s_b_db_offset = BK * (LDN);

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);       // 0 ~ 7    | 0 1  2 ... 15   每个索引32个一组 共16组
    int load_a_smem_k = (tid & 31) << 1;  // 0 ~ 60   | 0 2  4 ... 62 (32个数)  循环16组  间隔是2个half 4B
    int load_b_smem_k = (tid >> 5) << 1;  // 0 ~ 30   | 0 2  4     30   每个索引32个一组 共16组
    int load_b_smem_n = (tid & 31) << 3;  // 0 ~ 248  | 0 8 16 ... 248(128个数) 循环 4组  间隔是2个half 4B

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);
    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);

    int load_b_smem_addrs[2];
    #pragma unroll
    for(int i=0; i<2; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, LDN) * sizeof(half) + i * (LDN) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

    // load the first tile of mat_a & mat_b
    if (load_a_gmem_m < M && load_a_smem_k < BK) {
        asm("cp.async.ca.shared.global [%0], [%1], 4;\n" :
            : "r"(load_a_smem_addr_0),
                "l"(&a[load_a_gmem_addr]));
    }
    #pragma unroll
    for(int i=0; i<2; i++)
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * N]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    #pragma unroll 32
    for (int bk = 1; bk < (K / gridDim.z) / BK; bk++) {
        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = smem_sel ^ 1;
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        if (load_a_smem_k < BK && load_a_gmem_m < M) {
            asm("cp.async.ca.shared.global [%0], [%1], 4;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<2; i++){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * N]));
        }

        asm("cp.async.commit_group;\n" ::);  // issue cp.async.wait_group at the end of loop body

        wmma::load_matrix_sync(frag_a, &s_a[smem_sel * 32 + (wid >> 3) * 16], LDK);
        wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_db_offset + (wid >> 3) * 16 * (LDN) + (wid & 7) * 32], LDN);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    int smem_sel = ((K / gridDim.z) / BK - 1) & 1;
    wmma::load_matrix_sync(frag_a, &s_a[smem_sel * 32 + (wid >> 3) * 16], LDK);
    wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_db_offset + (wid >> 3) * 16 * (LDN) + (wid & 7) * 32], LDN);
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    // if(tid==0&&bx==0) printf("%d %f %f %f %f %f %f\n", bz, __half2float(frag_c.x[0]), __half2float(frag_c.x[1]), __half2float(frag_a.x[0]), __half2float(frag_a.x[1]), __half2float(frag_b.x[0]), __half2float(frag_b.x[1]));
    wmma::store_matrix_sync(&smem[(wid & 7) * 32 + (wid >> 3) * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    if (load_a_smem_m < M) {
        int store_c_smem_addr = OFFSET(load_a_smem_m + (wid >> 3) * 8, load_b_smem_n, LDN);
        int store_c_gmem_addr = OFFSET(load_a_smem_m, load_b_gmem_n, N);
        #pragma unroll
        for(int i=0; i<4; i++)
            atomicAdd(((half2 *)(&c[store_c_gmem_addr + 2 * i])),
                      *((half2 *)(&smem[store_c_smem_addr + 2 * i])));
    }
}


// /***
//  * BM=8 BN=512 BK=32
//  * LDK = BK + APAD = 40
//  * LDN = BN + BPAD = 520
//  * 访存阻塞更严重了，kernel时间短但性能并不好
// */
// template <int BM, int BN, int BK, int LDK, int LDN>
// __global__ void flat_gemm_m8n512k32x16(
//     half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
//     const int M, const int N, const int K) {
// #if __CUDA_ARCH__ < 800
//     return;
// #endif
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int bz = blockIdx.z;

//     int k_start = K / gridDim.z * bz;

//     int tid = threadIdx.x;
//     int wid = tid >> 5;         // WARP id

//     // bx for N, if bx is out of range, return
//     if (bx >= N / BN)
//         return;

//     __shared__ half smem[BM * (LDK) + BK * (LDN)];
//     half *s_a = smem;
//     half *s_b = smem + BM * (LDK);

//     //                             M   N   K
//     wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
//     wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b;
//     wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

//     wmma::fill_fragment(frag_c, __float2half(0.0f));

//     int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ... 15   每个索引32个一组 共16组
//     int load_a_smem_k = (tid & 31) << 1; // 0 ~ 60   | 0 2  4 ... 62 (32个数)  循环16组  间隔是2个half 4
//     int load_b_smem_k = (tid >> 6) << 2; // 0 ~ 28   | 0 4  8 ... 28     每个索引64个一组 共8组
//     int load_b_smem_n = (tid & 63) << 3; // 0 ~ 504  | 0 8 16 ... 504(64个数)  循环 8组  间隔是8个half 16B

//     // ptx address space conversion
//     size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
//     size_t s_b_base_addr = __cvta_generic_to_shared(s_b);
//     int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);

//     int load_b_smem_addrs[4];
//     #pragma unroll
//     for(int i=0; i<4; i++)
//         load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, LDN) * sizeof(half) + i * (LDN) * sizeof(half);

//     int load_a_gmem_m = load_a_smem_m;
//     int load_a_gmem_k = k_start + load_a_smem_k;
//     int load_b_gmem_n = bx * BN + load_b_smem_n;
//     int load_b_gmem_k = k_start + load_b_smem_k;

//     int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
//     int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

//     #pragma unroll
//     for (int bk = 0; bk < (K / gridDim.z) / BK; bk++) {

//         if (load_a_smem_k < BK && load_a_gmem_m < M) {
//             asm("cp.async.ca.shared.global [%0], [%1], 4;\n" :
//                 : "r"(load_a_smem_addr_0),
//                     "l"(&a[load_a_gmem_addr]));
//         }
//         #pragma unroll              // 一次访问两个16B 32B 16个数
//         for(int i=0; i<4; i++){
//             asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
//                 : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * N]));
//         }

//         asm("cp.async.commit_group;\n" ::);  // issue cp.async.wait_group at the end of loop body
//         asm("cp.async.wait_group 0;\n" ::);
//         __syncthreads();

//         #pragma unroll
//         for (int i = 0; i < 2; i++) {
//             wmma::load_matrix_sync(frag_a, &s_a[i * 16], LDK);
//             wmma::load_matrix_sync(frag_b, &s_b[i * 16 * (LDN) + wid * 32], LDN);
//             wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
//         }

//         load_a_gmem_addr += BK;
//         load_b_gmem_addr += BK * N;
//         __syncthreads();
//     }

//     // if(tid==0&&bx==0) printf("%d %f %f %f %f %f %f\n", bz, __half2float(frag_c.x[0]), __half2float(frag_c.x[1]), __half2float(frag_a.x[0]), __half2float(frag_a.x[1]), __half2float(frag_b.x[0]), __half2float(frag_b.x[1]));
//     wmma::store_matrix_sync(&smem[wid * 32], frag_c, LDN, wmma::mem_row_major);

//     __syncthreads();

//     if (load_a_smem_m < M) {
//         int store_c_smem_addr = OFFSET(load_a_smem_m, load_b_smem_n, LDN);
//         int store_c_gmem_addr = OFFSET(load_a_smem_m, load_b_gmem_n, N);
//         #pragma unroll
//         for(int i=0; i<4; i++)
//             atomicAdd(((half2 *)(&c[store_c_gmem_addr + 2 * i])),
//                       *((half2 *)(&smem[store_c_smem_addr + 2 * i])));
//     }
// }


/***
 * BM=8 BN=512 BK=32
 * LDK = BK + APAD = 40
 * LDN = BN + BPAD = 520
 * 访存阻塞更严重了，kernel时间短但性能并不好
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void flat_gemm_m8n512k32x16(
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
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half smem[BM * (LDK) + BK * (LDN)];
    half *s_a = smem;
    half *s_b = smem + BM * (LDK);

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 127   
    int load_a_smem_k = (tid & 31) << 1; // 0 ~ 31 x 2   
    int load_b_smem_k = (tid >> 6) << 2; // 0 ~ 28   
    int load_b_smem_n = (tid & 63) << 3; // 0 ~ 504  

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);
    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);

    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, LDN) * sizeof(half) + i * (LDN) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

    #pragma unroll
    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++) {

        if (load_a_smem_k < BK && load_a_gmem_m < M) {
            asm("cp.async.ca.shared.global [%0], [%1], 4;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll              // 一次访问两个16B 32B 16个数
        for(int i=0; i<4; i++){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * N]));
        }

        asm("cp.async.commit_group;\n" ::);  // issue cp.async.wait_group at the end of loop body
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(frag_a, &s_a[i * 16], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[i * 16 * (LDN) + wid * 32], LDN);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
        __syncthreads();
    }

    // if(tid==0&&bx==0) printf("%d %f %f %f %f %f %f\n", bz, __half2float(frag_c.x[0]), __half2float(frag_c.x[1]), __half2float(frag_a.x[0]), __half2float(frag_a.x[1]), __half2float(frag_b.x[0]), __half2float(frag_b.x[1]));
    wmma::store_matrix_sync(&smem[wid * 32], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int store_c_smem_addr = OFFSET((tid >> 6), load_b_smem_n, LDN);
    int store_c_gmem_addr = OFFSET((tid >> 6), load_b_gmem_n, N);

    if ((tid >> 6) < M) {
        #pragma unroll
        for(int i=0; i<4; i++)
            atomicAdd(((half2 *)(&c[store_c_gmem_addr + 2 * i])),
                      *((half2 *)(&smem[store_c_smem_addr + 2 * i])));
    }
}


/***
 * A:M*K B:K*N
 * BM=8 BN=256 BK=32
 * LDK = BK + PAD = 40
 * LDN = BN + PAD = 264
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void flat_gemm_m8n64k128x8(
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
    __shared__ half smem[(BM * LDK + BK * LDN)];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    /**
     * 通过位运算获取每个thread对应的索引位置
     * load_a 每个warp访问1*64个元素，通过cp.async指定访问8B即4个half完成
     * load_b 每个warp访问4*256个元素，通过cp.async指定访问16B即8个half完成
    */
    int load_a_smem_m = (tid >> 5);      // 0 ~ 7   
    int load_a_smem_k = (tid & 31) << 2; // 0 ~ 31 x 4 
    int load_b_smem_k = (tid >> 3) << 2; // 0 ~ 31 x 4
    int load_b_smem_n = (tid & 7) << 3;  // 0 ~ 7 x 8

    unsigned int w_row = wid / 4;  // 0, 1
    unsigned int w_col = wid % 4;  // 0, 1, 2, 3

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, LDN) * sizeof(half) + i * (LDN) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_smem_addr = OFFSET(load_a_smem_m, load_a_smem_k, LDK);
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

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
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * N]));
            //     : "r"(load_b_smem_addrs[i]), "l"(ptr_b));
            // ptr_b += K;
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[w_col * 16 + 64 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[w_row * 32 + (w_col * 16 + 64 * i) * LDN], LDN);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
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


/***
 * BM=8 BN=32 BK=256
 * LDK = BK + PAD = 264
 * LDN = BN + PAD = 40
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void flat_gemm_m8n32k256x8(
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
    __shared__ half smem[(BM * LDK + BK * LDN)];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    /**
     * 通过位运算获取每个thread对应的索引位置
     * load_a 每个warp访问1*64个元素，通过cp.async指定访问8B即4个half完成
     * load_b 每个warp访问4*256个元素，通过cp.async指定访问16B即8个half完成
    */
    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   每个索引32个一组 共8组
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248(32个数)  循环8组  间隔是8个half 16B
    int load_b_smem_n = (tid & 3) << 3; // 0 ~ 28   | 0 4  8 ... 28   每个索引32个一组 共8组
    int load_b_smem_k = (tid >> 2) << 2;    // 0 ~ 63 x 4

    int load_a_smem_addr_0 = __cvta_generic_to_shared(s_a) + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = __cvta_generic_to_shared(s_b) + OFFSET(load_b_smem_k, load_b_smem_n, LDN) * sizeof(half) + i * (LDN) * sizeof(half);

    int load_a_gmem_addr = OFFSET(load_a_smem_m, (k_start + load_a_smem_k), K);
    int load_b_gmem_addr = OFFSET((k_start + load_b_smem_k), (bx * BN + load_b_smem_n), N);

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
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * N]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[(wid * 16 + 128 * i) * LDN], LDN);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
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

    if (load_a_smem_m < M) {
        atomicAdd(&c[gmem_c_addr], smem[shmem_c_addr]);
    }
}