import torch
import time
from eed.backend import hgemm
from triton_mm import matmul as triton_matmul

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
print("cublas verse torch: ", all_close)

torch.cuda.cudart().cudaProfilerStart() 
triton_output = triton_matmul(A, B)
torch.cuda.cudart().cudaProfilerStop()

print(triton_output)

all_close = torch.allclose(torch_output, triton_output, rtol=1e-2, atol=1e-4)
print("triton verse torch: ", all_close)

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

triton_dur = 0
for _ in range(10):
    triton_output = triton_matmul(A, B)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    triton_output = triton_matmul(A, B)
    torch.cuda.synchronize()
    ed = time.time()
    triton_dur += (ed - st) * 10

print('torch - cublas - triton: %.4f ms - %.4f ms - %.4f ms' % (torch_dur, cublas_dur, triton_dur))

