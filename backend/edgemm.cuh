#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void myHGEMMAlignedV5(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ >= 800

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    if (bx >= N / BN || by >= M / BM)
        return;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], __float2half(0.0f));
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // wonder if the access block and computation block should change order
        // load A and B from global mem for next bk
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));

        // compute A X B for this bk
        // note that BK / TILE_K = 2
        wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }
#endif
}


__global__ void eed_hgemm_m8n256k32_v1(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ >= 800
    const int BM = 8;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    if (bx >= N / BN || by >= M / BM)
        return;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK * 8 + APAD);
    int s_a_db_offset = BM * (BK * 8 + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    // wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::fill_fragment(frag_c[i][j], __float2half(0.0f));
    //     }
    // }

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);
    int load_a_smem_k = (tid & 31) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK * 8 + APAD) * sizeof(half);
    // int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK * 8 + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    // int comp_c_frag_m = wid &  1;
    // int comp_c_frag_n = wid >> 1;

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // compute A X B for this bk
        // note that BK / TILE_K = 2
        wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 8 * BK + 0], 8 * BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 8 * BK + 16], 8 * BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

        // #pragma unroll
        // for (int i = 0; i < 4; i++) {
        //     #pragma unroll
        //     for (int j = 0; j < 4; j++) {
        //         wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        //         wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        //     }
        // }

        wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
        wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);

        // __syncthreads();

        // wonder if the access block and computation block should change order
        // load A and B from global mem for next bk
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr +     K]));
        if (bk % 8 == 0){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        }
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    // wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    // wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + 7 * BK + 0], 8 * BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + 7 * BK + 16], 8 * BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
    //         wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
    //     }
    // }

    wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
    wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);

    int store_c_gmem_m = by * BM;
    int store_c_gmem_n = bx * BN + wid * 32;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
    //     }
    // }
    wmma::store_matrix_sync(&c[store_c_gmem_addr], frag_c, N, wmma::mem_row_major);
#endif
}


__global__ void eed_hgemm_m8n256k32_v2(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ >= 800

    const int BM = 8;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    if (bx >= N / BN || by >= M / BM)
        return;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK * 8 + APAD);
    int s_a_db_offset = BM * (BK * 8 + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    // wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::fill_fragment(frag_c[i][j], __float2half(0.0f));
    //     }
    // }
 
    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);
    int load_a_smem_k = (tid & 31) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;
    int shift = ((tid & 31) >> 2) & 1;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK * 8 + APAD) * sizeof(half);
    // int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK * 8 + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    // int comp_c_frag_m = wid &  1;
    // int comp_c_frag_n = wid >> 1;

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + shift * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // compute A X B for this bk
        // note that BK / TILE_K = 2
        wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 8 * BK + 0], 8 * BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 8 * BK + 16], 8 * BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

        // #pragma unroll
        // for (int i = 0; i < 4; i++) {
        //     #pragma unroll
        //     for (int j = 0; j < 4; j++) {
        //         wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        //         wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        //     }
        // }

        wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
        wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);

        // __syncthreads();

        // wonder if the access block and computation block should change order
        // load A and B from global mem for next bk
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr +     K]));
        if (bk % 8 == 0){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + shift * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
            // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            //     : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        }
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    // wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    // wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + 7 * BK + 0], 8 * BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + 7 * BK + 16], 8 * BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
    //         wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
    //     }
    // }

    wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
    wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);

    int store_c_gmem_m = by * BM;
    int store_c_gmem_n = bx * BN + wid * 32;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
    //     }
    // }
    wmma::store_matrix_sync(&c[store_c_gmem_addr], frag_c, N, wmma::mem_row_major);
#endif
}


__global__ void eed_hgemm_m8n256k64_v3(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ >= 800

    const int BM = 8;
    const int BN = 256;
    const int BK = 64;

    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    if (bx >= N / BN || by >= M / BM)
        return;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK * 4 + APAD);
    int s_a_db_offset = BM * (BK * 4 + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    // wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[4];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::fill_fragment(frag_c[i][j], __float2half(0.0f));
    //     }
    // }

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);
    int load_a_smem_k = (tid & 31) << 3;
    int load_b_smem_k = (tid >> 5) << 3;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK * 4 + APAD) * sizeof(half);
    // int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK * 8 + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_4 = load_b_smem_addr_0 + 4 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_5 = load_b_smem_addr_0 + 5 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_6 = load_b_smem_addr_0 + 6 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_7 = load_b_smem_addr_0 + 7 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    // int comp_c_frag_m = wid &  1;
    // int comp_c_frag_n = wid >> 1;

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7), "l"(&b[load_b_gmem_addr + 7 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // compute A X B for this bk
        // note that BK / TILE_K = 2
        wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 4 * BK + 0], 4 * BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 4 * BK + 16], 4 * BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[2], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 4 * BK + 32], 4 * BK + APAD);
        wmma::load_matrix_sync(frag_a[3], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 4 * BK + 48], 4 * BK + APAD);

        wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[2], &s_b[smem_sel * s_b_db_offset + 32 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[3], &s_b[smem_sel * s_b_db_offset + 48 * (BN + BPAD) + wid * 32], BN + BPAD);

        // #pragma unroll
        // for (int i = 0; i < 4; i++) {
        //     #pragma unroll
        //     for (int j = 0; j < 4; j++) {
        //         wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        //         wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        //     }
        // }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::mma_sync(frag_c, frag_a[i], frag_b[i], frag_c);
        }

        // __syncthreads();

        // wonder if the access block and computation block should change order
        // load A and B from global mem for next bk
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr +     K]));
        if (bk % 4 == 0){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        }
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 7 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    // wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    // wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + 3 * BK + 0], 4 * BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + 3 * BK + 16], 4 * BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[2], &s_a[smem_sel * s_a_db_offset + 3 * BK + 32], 4 * BK + APAD);
    wmma::load_matrix_sync(frag_a[3], &s_a[smem_sel * s_a_db_offset + 3 * BK + 48], 4 * BK + APAD);

    wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[2], &s_b[smem_sel * s_b_db_offset + 32 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[3], &s_b[smem_sel * s_b_db_offset + 48 * (BN + BPAD) + wid * 32], BN + BPAD);

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
    //         wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
    //     }
    // }

    // wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
    // wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::mma_sync(frag_c, frag_a[i], frag_b[i], frag_c);
    }

    int store_c_gmem_m = by * BM;
    int store_c_gmem_n = bx * BN + wid * 32;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
    //     }
    // }
    wmma::store_matrix_sync(&c[store_c_gmem_addr], frag_c, N, wmma::mem_row_major);
#endif
}


__global__ void eed_hgemm_m8n128k64_v4(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ >= 800

    const int BM = 8;
    const int BN = 128;
    const int BK = 64;

    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    if (bx >= N / BN || by >= M / BM)
        return;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK * 2 + APAD);
    int s_a_db_offset = BM * (BK * 2 + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    // wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[4];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::fill_fragment(frag_c[i][j], __float2half(0.0f));
    //     }
    // }

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 4);
    int load_a_smem_k = (tid & 15) << 3;
    int load_b_smem_k = (tid >> 4) << 3;
    int load_b_smem_n = (tid & 15) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK * 2 + APAD) * sizeof(half);
    // int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK * 8 + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_4 = load_b_smem_addr_0 + 4 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_5 = load_b_smem_addr_0 + 5 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_6 = load_b_smem_addr_0 + 6 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_7 = load_b_smem_addr_0 + 7 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    // int comp_c_frag_m = wid &  1;
    // int comp_c_frag_n = wid >> 1;

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7), "l"(&b[load_b_gmem_addr + 7 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // compute A X B for this bk
        // note that BK / TILE_K = 2
        wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 0], 2 * BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 16], 2 * BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[2], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 32], 2 * BK + APAD);
        wmma::load_matrix_sync(frag_a[3], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 48], 2 * BK + APAD);

        wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[2], &s_b[smem_sel * s_b_db_offset + 32 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[3], &s_b[smem_sel * s_b_db_offset + 48 * (BN + BPAD) + wid * 32], BN + BPAD);

        // #pragma unroll
        // for (int i = 0; i < 4; i++) {
        //     #pragma unroll
        //     for (int j = 0; j < 4; j++) {
        //         wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        //         wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        //     }
        // }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::mma_sync(frag_c, frag_a[i], frag_b[i], frag_c);
        }

        // __syncthreads();

        // wonder if the access block and computation block should change order
        // load A and B from global mem for next bk
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr +     K]));
        if (bk % 2 == 0){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        }
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 7 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    // wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    // wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + 1 * BK + 0], 2 * BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + 1 * BK + 16], 2 * BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[2], &s_a[smem_sel * s_a_db_offset + 1 * BK + 32], 2 * BK + APAD);
    wmma::load_matrix_sync(frag_a[3], &s_a[smem_sel * s_a_db_offset + 1 * BK + 48], 2 * BK + APAD);

    wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[2], &s_b[smem_sel * s_b_db_offset + 32 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[3], &s_b[smem_sel * s_b_db_offset + 48 * (BN + BPAD) + wid * 32], BN + BPAD);

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
    //         wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
    //     }
    // }

    // wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
    // wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::mma_sync(frag_c, frag_a[i], frag_b[i], frag_c);
    }

    int store_c_gmem_m = by * BM;
    int store_c_gmem_n = bx * BN + wid * 32;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
    //     }
    // }
    wmma::store_matrix_sync(&c[store_c_gmem_addr], frag_c, N, wmma::mem_row_major);
#endif
}


__global__ void eed_hgemm_m8n128k128_v5(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ >= 800

    const int BM = 8;
    const int BN = 128;
    const int BK = 128;

    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    if (bx >= N / BN || by >= M / BM)
        return;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    // wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[8];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[8];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::fill_fragment(frag_c[i][j], __float2half(0.0f));
    //     }
    // }

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 4);
    int load_a_smem_k = (tid & 15) << 3;
    int load_b_smem_k = (tid >> 4) << 4;
    int load_b_smem_n = (tid & 15) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    // int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK * 8 + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_4 = load_b_smem_addr_0 + 4 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_5 = load_b_smem_addr_0 + 5 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_6 = load_b_smem_addr_0 + 6 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_7 = load_b_smem_addr_0 + 7 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_8 = load_b_smem_addr_0 + 8 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_9 = load_b_smem_addr_0 + 9 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_10 = load_b_smem_addr_0 + 10 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_11 = load_b_smem_addr_0 + 11 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_12 = load_b_smem_addr_0 + 12 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_13 = load_b_smem_addr_0 + 13 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_14 = load_b_smem_addr_0 + 14 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_15 = load_b_smem_addr_0 + 15 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    // int comp_c_frag_m = wid &  1;
    // int comp_c_frag_n = wid >> 1;

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7), "l"(&b[load_b_gmem_addr + 7 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_8), "l"(&b[load_b_gmem_addr + 8 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_9), "l"(&b[load_b_gmem_addr + 9 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_10), "l"(&b[load_b_gmem_addr + 10 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_11), "l"(&b[load_b_gmem_addr + 11 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_12), "l"(&b[load_b_gmem_addr + 12 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_13), "l"(&b[load_b_gmem_addr + 13 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_14), "l"(&b[load_b_gmem_addr + 14 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_15), "l"(&b[load_b_gmem_addr + 15 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // compute A X B for this bk
        // note that BK / TILE_K = 2
        wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + 0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[2], &s_a[smem_sel * s_a_db_offset + 32], BK + APAD);
        wmma::load_matrix_sync(frag_a[3], &s_a[smem_sel * s_a_db_offset + 48], BK + APAD);
        wmma::load_matrix_sync(frag_a[4], &s_a[smem_sel * s_a_db_offset + 64], BK + APAD);
        wmma::load_matrix_sync(frag_a[5], &s_a[smem_sel * s_a_db_offset + 80], BK + APAD);
        wmma::load_matrix_sync(frag_a[6], &s_a[smem_sel * s_a_db_offset + 96], BK + APAD);
        wmma::load_matrix_sync(frag_a[7], &s_a[smem_sel * s_a_db_offset + 112], BK + APAD);

        wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[2], &s_b[smem_sel * s_b_db_offset + 32 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[3], &s_b[smem_sel * s_b_db_offset + 48 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[4], &s_b[smem_sel * s_b_db_offset + 64 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[5], &s_b[smem_sel * s_b_db_offset + 80 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[6], &s_b[smem_sel * s_b_db_offset + 96 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[7], &s_b[smem_sel * s_b_db_offset + 112 * (BN + BPAD) + wid * 32], BN + BPAD);

        // #pragma unroll
        // for (int i = 0; i < 4; i++) {
        //     #pragma unroll
        //     for (int j = 0; j < 4; j++) {
        //         wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        //         wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        //     }
        // }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            wmma::mma_sync(frag_c, frag_a[i], frag_b[i], frag_c);
        }

        // __syncthreads();

        // wonder if the access block and computation block should change order
        // load A and B from global mem for next bk
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr +     K]));
        // if (bk % 2 == 0){
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));  
        // }
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 7 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_8 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 8 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_9 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 9 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_10 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 10 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_11 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 11 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_12 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 12 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_13 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 13 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_14 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 14 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_15 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 15 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    // wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    // wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + 0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    // wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    // wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[2], &s_a[smem_sel * s_a_db_offset + 32], BK + APAD);
    wmma::load_matrix_sync(frag_a[3], &s_a[smem_sel * s_a_db_offset + 48], BK + APAD);
    wmma::load_matrix_sync(frag_a[4], &s_a[smem_sel * s_a_db_offset + 64], BK + APAD);
    wmma::load_matrix_sync(frag_a[5], &s_a[smem_sel * s_a_db_offset + 80], BK + APAD);
    wmma::load_matrix_sync(frag_a[6], &s_a[smem_sel * s_a_db_offset + 96], BK + APAD);
    wmma::load_matrix_sync(frag_a[7], &s_a[smem_sel * s_a_db_offset + 112], BK + APAD);

    wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    // wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[2], &s_b[smem_sel * s_b_db_offset + 32 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[3], &s_b[smem_sel * s_b_db_offset + 48 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[4], &s_b[smem_sel * s_b_db_offset + 64 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[5], &s_b[smem_sel * s_b_db_offset + 80 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[6], &s_b[smem_sel * s_b_db_offset + 96 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[7], &s_b[smem_sel * s_b_db_offset + 112 * (BN + BPAD) + wid * 32], BN + BPAD);

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
    //         wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
    //     }
    // }

    // wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
    // wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        wmma::mma_sync(frag_c, frag_a[i], frag_b[i], frag_c);
    }

    int store_c_gmem_m = by * BM;
    int store_c_gmem_n = bx * BN + wid * 32;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
    //     }
    // }
    wmma::store_matrix_sync(&c[store_c_gmem_addr], frag_c, N, wmma::mem_row_major);
#endif
}


__global__ void eed_hgemm_m8n128k64x4_v7(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ >= 800

    const int BM = 8;
    const int BN = 128;
    const int BK = 64;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int k_start = K / 4 * bz;
    int k_end = K / 4 * (bz + 1);
    int tid = threadIdx.x;
    int wid = tid >> 5;

    if (bx >= N / BN || by >= M / BM)
        return;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK * 2 + APAD);
    int s_a_db_offset = BM * (BK * 2 + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[4];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 4);
    int load_a_smem_k = (tid & 15) << 3;
    int load_b_smem_k = (tid >> 4) << 3;
    int load_b_smem_n = (tid & 15) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK * 2 + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_4 = load_b_smem_addr_0 + 4 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_5 = load_b_smem_addr_0 + 5 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_6 = load_b_smem_addr_0 + 6 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_7 = load_b_smem_addr_0 + 7 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7), "l"(&b[load_b_gmem_addr + 7 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < (K / 4) / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // compute A X B for this bk
        // note that BK / TILE_K = 2
        wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 0], 2 * BK + APAD);
        wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 16], 2 * BK + APAD);
        wmma::load_matrix_sync(frag_a[2], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 32], 2 * BK + APAD);
        wmma::load_matrix_sync(frag_a[3], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 48], 2 * BK + APAD);

        wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[2], &s_b[smem_sel * s_b_db_offset + 32 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[3], &s_b[smem_sel * s_b_db_offset + 48 * (BN + BPAD) + wid * 32], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::mma_sync(frag_c, frag_a[i], frag_b[i], frag_c);
        }

        if (bk % 2 == 0){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        }
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 7 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + 1 * BK + 0], 2 * BK + APAD);
    wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + 1 * BK + 16], 2 * BK + APAD);
    wmma::load_matrix_sync(frag_a[2], &s_a[smem_sel * s_a_db_offset + 1 * BK + 32], 2 * BK + APAD);
    wmma::load_matrix_sync(frag_a[3], &s_a[smem_sel * s_a_db_offset + 1 * BK + 48], 2 * BK + APAD);

    wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[2], &s_b[smem_sel * s_b_db_offset + 32 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[3], &s_b[smem_sel * s_b_db_offset + 48 * (BN + BPAD) + wid * 32], BN + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::mma_sync(frag_c, frag_a[i], frag_b[i], frag_c);
    }

    // int store_c_gmem_m = by * BM;
    // int store_c_gmem_n = bx * BN + wid * 32;
    // int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    // wmma::store_matrix_sync(&c[store_c_gmem_addr], frag_c, N, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem[wid * 32], frag_c, BN + BPAD, wmma::mem_row_major);

    __syncthreads();

    int store_c_smem_addr = OFFSET(load_a_smem_m, load_b_smem_n, BN + BPAD);
    int store_c_gmem_addr = OFFSET(load_a_gmem_m, load_b_gmem_n, N);

    atomicAdd(((half2*)(&c[store_c_gmem_addr    ])), *((half2*)(&smem[store_c_smem_addr    ])));
    atomicAdd(((half2*)(&c[store_c_gmem_addr + 2])), *((half2*)(&smem[store_c_smem_addr + 2])));
    atomicAdd(((half2*)(&c[store_c_gmem_addr + 4])), *((half2*)(&smem[store_c_smem_addr + 4])));
    atomicAdd(((half2*)(&c[store_c_gmem_addr + 6])), *((half2*)(&smem[store_c_smem_addr + 6])));
#endif
}

template <int SPLITK>
__global__ void eed_hgemv_m1n128k64x4_v6(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ >= 800

    const int BM = 8;
    const int BN = 128;
    const int BK = 64;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int k_start = K / SPLITK * bz;
    int k_end = K / SPLITK * (bz + 1);
    int tid = threadIdx.x;
    int wid = tid >> 5;

    if (bx >= N / BN)
        return;

    const int APAD = 8;
    const int BPAD = 8;

    half pads[8] = {__float2half(0.0f)};

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK * 2 + APAD);
    int s_a_db_offset = BM * (BK * 2 + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[4];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 4);
    int load_a_smem_k = (tid & 15) << 3;
    int load_b_smem_k = (tid >> 4) << 3;
    int load_b_smem_n = (tid & 15) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK * 2 + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_4 = load_b_smem_addr_0 + 4 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_5 = load_b_smem_addr_0 + 5 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_6 = load_b_smem_addr_0 + 6 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_7 = load_b_smem_addr_0 + 7 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_smem_addr = OFFSET(load_a_smem_m, load_a_smem_k, BK * 2 + APAD);
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

    *((float4*)(s_a + load_a_smem_addr + 0 * s_a_db_offset)) = (load_a_smem_m == 0) ? 
        *((float4*)(&a[load_a_gmem_addr])) : 
        *((float4*)(&pads[0]));

    *((float4*)(s_a + load_a_smem_addr + 1 * s_a_db_offset)) = (load_a_smem_m == 0) ? 
        *((float4*)(&a[load_a_gmem_addr])) : 
        *((float4*)(&pads[0]));

    {
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7), "l"(&b[load_b_gmem_addr + 7 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < (K / SPLITK) / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // compute A X B for this bk
        // note that BK / TILE_K = 2
        wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 0 ], 2 * BK + APAD);
        wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 16], 2 * BK + APAD);
        wmma::load_matrix_sync(frag_a[2], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 32], 2 * BK + APAD);
        wmma::load_matrix_sync(frag_a[3], &s_a[smem_sel * s_a_db_offset + (bk - 1) % 2 * BK + 48], 2 * BK + APAD);

        wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[2], &s_b[smem_sel * s_b_db_offset + 32 * (BN + BPAD) + wid * 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[3], &s_b[smem_sel * s_b_db_offset + 48 * (BN + BPAD) + wid * 32], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::mma_sync(frag_c, frag_a[i], frag_b[i], frag_c);
        }

        if (bk % 2 == 0 && load_a_smem_m == 0){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + 0 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0 + 1 * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        }
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 7 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    wmma::load_matrix_sync(frag_a[0], &s_a[smem_sel * s_a_db_offset + 1 * BK + 0], 2 * BK + APAD);
    wmma::load_matrix_sync(frag_a[1], &s_a[smem_sel * s_a_db_offset + 1 * BK + 16], 2 * BK + APAD);
    wmma::load_matrix_sync(frag_a[2], &s_a[smem_sel * s_a_db_offset + 1 * BK + 32], 2 * BK + APAD);
    wmma::load_matrix_sync(frag_a[3], &s_a[smem_sel * s_a_db_offset + 1 * BK + 48], 2 * BK + APAD);

    wmma::load_matrix_sync(frag_b[0], &s_b[smem_sel * s_b_db_offset +                    wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[2], &s_b[smem_sel * s_b_db_offset + 32 * (BN + BPAD) + wid * 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[3], &s_b[smem_sel * s_b_db_offset + 48 * (BN + BPAD) + wid * 32], BN + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::mma_sync(frag_c, frag_a[i], frag_b[i], frag_c);
    }

    wmma::store_matrix_sync(&smem[wid * 32], frag_c, BN + BPAD, wmma::mem_row_major);

    __syncthreads();

    int store_c_smem_m = 0;
    int store_c_smem_n = tid;
    int store_c_gmem_m = by * BM + store_c_smem_m;
    int store_c_gmem_n = bx * BN + store_c_smem_n;
    // int store_c_smem_addr = OFFSET(store_c_smem_m, store_c_smem_n, BN + BPAD);
    // int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

    atomicAdd(&c[store_c_gmem_n], smem[store_c_smem_n]);
    // atomicAdd(((half2*)(&c[store_c_gmem_addr + 2])), *((half2*)(&smem[store_c_smem_addr + 2])));
    // atomicAdd(((half2*)(&c[store_c_gmem_addr + 4])), *((half2*)(&smem[store_c_smem_addr + 4])));
    // atomicAdd(((half2*)(&c[store_c_gmem_addr + 6])), *((half2*)(&smem[store_c_smem_addr + 6])));
#endif
}
