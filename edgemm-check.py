import torch
import time
from eed.backend import edgemm, edgemm_m8n128k64, edgemm_m8n256k64, edgemm_m8n128k128, edgemv_m1n128k64x4, edgemm_m8n128k64x4

M = 1
K = 12288 * 4
N = 12288

func = edgemv_m1n128k64x4

A = torch.rand((M, K), dtype=torch.float16, device='cuda')
B = torch.rand((K, N), dtype=torch.float16, device='cuda')
B[:, -4:-2] = B[:, -4:-2] / 100
C_ = torch.zeros((M, N), dtype=torch.float16, device='cuda')
C = torch.zeros((M, N), dtype=torch.float16, device='cuda')

torch.matmul(A, B, out=C_)

print(C_)

func(A, B, C)
print(C)

all_close = torch.allclose(C_, C, rtol=1e-1, atol=1e-2)
print("edgemm verse torch: ", all_close)

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
    func(A, B, C)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    func(A, B, C)
    torch.cuda.synchronize()
    ed = time.time()
    edgemm_dur += (ed - st) * 10

print('torch - edgemm: %.4f ms - %.4f ms' % (torch_dur, edgemm_dur))