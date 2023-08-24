import torch
import time
from eed.backend import hgemm, fastgemv, fastgemv_int8

M = 1
K = 4096
N = 1280

A = torch.rand((M, K), dtype=torch.float16, device='cuda')
B = torch.randint(0, 10, (K, N), dtype=torch.int8, device='cuda')
C = torch.zeros((M, N), dtype=torch.float16, device='cuda')

A_T = A.transpose(0, 1).contiguous()
B_T = B.transpose(0, 1).contiguous()
C_T = C.transpose(0, 1).contiguous()

fastgemv_int8(B_T, A_T, C_T)

fastgemv_dur = 0
for _ in range(10):
    fastgemv_int8(B_T, A_T, C_T)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    _ = torch.empty((N, M), dtype=torch.float16, device='cuda')
    fastgemv_int8(B_T, A_T, C_T)
    torch.cuda.synchronize()
    ed = time.time()
    fastgemv_dur += (ed - st) * 10

print('dur: %.4f ms' % fastgemv_dur)