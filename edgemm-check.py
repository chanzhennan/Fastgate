import torch
import time
from eed.backend import edgemm, edgemm_m8n128k64, edgemm_m8n256k64, edgemm_m8n128k128

M = 8
K = 6144 * 4
N = 6144

func = edgemm_m8n128k128

A = torch.rand((M, K), dtype=torch.float16, device='cuda')
B = torch.rand((K, N), dtype=torch.float16, device='cuda')
B[:, -4:-2] = B[:, -4:-2] / 100

torch_output = torch.matmul(A, B)

print(torch_output)

C = torch.zeros((M, N), dtype=torch.float16, device='cuda')

func(A, B, C)
print(C)

all_close = torch.allclose(torch_output, C, rtol=1e-1, atol=1e-2)
print("edgemm verse torch: ", all_close)

torch_dur = 0
for _ in range(10):
    torch_output = torch.matmul(A, B)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    torch_output = torch.matmul(A, B)
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