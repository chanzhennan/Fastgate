import torch
import time
from hgemm.backend import hgemm

A = torch.rand((1, 6144), dtype=torch.float16, device='cuda')
B = torch.rand((6144, 6144 * 4), dtype=torch.float16, device='cuda')

torch.cuda.cudart().cudaProfilerStart()  
torch_output = torch.matmul(A, B)
torch.cuda.cudart().cudaProfilerStop()  

print(torch_output)

C = torch.zeros((1, 6144 * 4), dtype=torch.float16, device='cuda')
torch.cuda.cudart().cudaProfilerStart()  
hgemm(A, B, C)
torch.cuda.cudart().cudaProfilerStop()  

print(C)

all_close = torch.allclose(torch_output, C, rtol=1e-2, atol=1e-4)
print(all_close)

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

cublas_dur = 0
for _ in range(10):
    hgemm(A, B, C)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    hgemm(A, B, C)
    torch.cuda.synchronize()
    ed = time.time()
    cublas_dur += (ed - st) * 10

print('torch - cublas: %.4f ms - %.4f ms' % (torch_dur, cublas_dur))

