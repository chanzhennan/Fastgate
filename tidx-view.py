
# for tid in range(256):
#     load_a_smem_m = (tid >> 2) << 1
#     load_a_smem_k = (tid &  3) << 3
#     load_b_smem_k = (tid >> 5) << 2
#     load_b_smem_n = (tid & 31) << 3
#     print("tidx: %d, a_m: %d, a_k: %d, b_k: %d, b_n: %d" % (
#         tid, load_a_smem_m, load_a_smem_k, load_b_smem_k, load_b_smem_n))

# # for bk in range(2):
# #     smem_sel = (bk & 1) ^ 1
# #     smem_sel_next = ((bk - 1) & 1) ^ 1
# #     print("bk: %d, sel: %d, sel next: %d" % (
# #         bk, smem_sel, smem_sel_next
# #     ))

for wid in range(8):
    comp_c_frag_m = wid &  1
    comp_c_frag_n = wid >> 1
    print("wid: %d, frag_m: %d, frag_n: %d" % (
        wid, comp_c_frag_m, comp_c_frag_n
    ))