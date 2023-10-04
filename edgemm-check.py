import torch
import time
from eed.backend import edgemm, edgemm_m8n128k64, edgemm_m8n256k64, edgemm_m8n128k128, edgemv_m1n128k64x4, edgemm_m8n128k64x4, edgemv_m1n256k64x4
from eed.backend import edgemm_m8n128k64x4_bt
from eed.backend import fastgemv, fastgemv_tuned, fastgemv_extend

M = 1
K = 4096
N = 4096

tc_func = edgemv_m1n128k64x4
cc_func = fastgemv_tuned

A = torch.rand((M, K), dtype=torch.float16, device='cuda')
B = torch.rand((K, N), dtype=torch.float16, device='cuda')
B[:, -4:-2] = B[:, -4:-2] / 100
C_ = torch.zeros((M, N), dtype=torch.float16, device='cuda')
C = torch.zeros((M, N), dtype=torch.float16, device='cuda')
C_f = torch.zeros((M, N), dtype=torch.float16, device='cuda')

A_T = A.transpose(0, 1).contiguous()
B_T = B.transpose(0, 1).contiguous()
C_T = C.transpose(0, 1).contiguous()

torch.matmul(A, B, out=C_)

print(C_)

tc_func(A, B, C)
print(C)

all_close = torch.allclose(C_, C, rtol=1e-1, atol=1e-2)
print("edgemm verse torch: ", all_close)

cc_func(B_T, A, C_f)
print(C_f)

all_close = torch.allclose(C_f, C_, rtol=1e-1, atol=1e-2)
print("fastgemv verse torch: ", all_close)

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

edgemm_dur = 0
for _ in range(10):
    tc_func(A, B, C)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    tc_func(A, B, C)
    torch.cuda.synchronize()
    ed = time.time()
    edgemm_dur += (ed - st) * 10

fastgemv_dur = 0
for _ in range(10):
    cc_func(B_T, A, C_f)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    cc_func(B_T, A, C_f)
    torch.cuda.synchronize()
    ed = time.time()
    fastgemv_dur += (ed - st) * 10

print('torch - edgemm - fastgemv: %.4f ms - %.4f ms - %.4f ms' % (torch_dur, edgemm_dur, fastgemv_dur))