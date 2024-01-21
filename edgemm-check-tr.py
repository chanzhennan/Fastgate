import torch
import time
import argparse
from eed.backend import edgemm, edgemm_m8n128k64, edgemm_m8n256k64, edgemm_m8n128k128, edgemv_m1n128k64x4, edgemm_m8n128k64x4
from eed.backend import edgemm_m8n256k32x8
from eed.backend import fastgemv, fastgemv_tuned, fastgemv_extend
from eed.backend import hgemm_tr
from eed.backend import gemm_m8n32k256x8_bt, gemm_m8n32k128x8_bt, gemm_m8n64k128x8_bt_exp, gemm_m8n32k256x8_bt_bz2

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=int, default=8)
parser.add_argument('--k', type=int, default=4096)
parser.add_argument('--n', type=int, default=4096)
args = parser.parse_args()

M = args.m
K = args.k
N = args.n
tc_func = gemm_m8n32k256x8_bt_bz2
cc_func = fastgemv_extend
split = 1

A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
B = torch.empty((K, N), dtype=torch.float16, device="cuda").normal_(mean=1., std=0.5)
B[:, -4:-2] = B[:, -4:-2] / 100
C_ = torch.empty((M, N), dtype=torch.float16, device='cuda')
C_c = torch.empty((M, N), dtype=torch.float16, device='cuda')
C = torch.empty((M, N), dtype=torch.float16, device='cuda')
C_f = torch.empty((M, N), dtype=torch.float16, device='cuda')

A_T = A.transpose(0, 1).contiguous()
B_T = B.transpose(0, 1).contiguous()
C_T = C.transpose(0, 1).contiguous()

torch.matmul(A, B, out=C_)
# print(C_)

torch.cuda.cudart().cudaProfilerStart()
hgemm_tr(A, B_T, C_c)
torch.cuda.cudart().cudaProfilerStop()
# print(C_c)

all_close = torch.allclose(C_, C_c, rtol=1e-0, atol=1e-2)
# print("cuBLAS verse torch: ", all_close)
# print((C_c - C_).max().item())

torch.cuda.cudart().cudaProfilerStart()
tc_func(A, B_T, C, split)
torch.cuda.cudart().cudaProfilerStop()
# print(C)

all_close = torch.allclose(C_, C, rtol=1e-0, atol=1e-2)
print("edgemm verse torch: ", all_close)
print((C_ - C).max().item())
idx = torch.argmax((C_ - C))
row_idx = idx // N 
col_idx = idx % N
print(C_[row_idx, col_idx].item(), C[row_idx, col_idx].item())


# torch.cuda.cudart().cudaProfilerStart()
# cc_func(B_T, A, C_f)
# torch.cuda.cudart().cudaProfilerStop()
# # print(C_f)

# all_close = torch.allclose(C_f, C_, rtol=1e-0, atol=1e-2)
# print("fastgemv verse torch: ", all_close)

torch_dur = 0
for _ in range(10):
    torch.matmul(A, B, out=C_)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    torch.matmul(A, B, out=C_)
    torch.cuda.synchronize()
    ed = time.time()
    torch_dur += (ed - st) * 10

cublas_dur = 0
for _ in range(10):
    hgemm_tr(A, B_T, C_c)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    hgemm_tr(A, B_T, C_c)
    torch.cuda.synchronize()
    ed = time.time()
    cublas_dur += (ed - st) * 10

edgemm_dur = 0
for _ in range(10):
    tc_func(A, B_T, C, split)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    tc_func(A, B_T, C, split)
    torch.cuda.synchronize()
    ed = time.time()
    edgemm_dur += (ed - st) * 10

fastgemv_dur = 0
# for _ in range(10):
#     cc_func(B_T, A, C_f)

# for _ in range(100):
#     torch.cuda.synchronize()
#     st = time.time()
#     cc_func(B_T, A, C_f)
#     torch.cuda.synchronize()
#     ed = time.time()
#     fastgemv_dur += (ed - st) * 10

# print('torch - cublas - edgemm - fastgemv: %.4f ms - %.4f ms - %.4f ms - %.4f ms' % 
#     (torch_dur, cublas_dur, edgemm_dur, fastgemv_dur))

print('%.4f %.4f %.4f %.4f' % (torch_dur, cublas_dur, edgemm_dur, fastgemv_dur))