import torch
import time
from eed.backend import hgemm, fastgemv
from triton_mm import matmul as triton_matmul

M = 1
K = 1024 * 4
N = 1024

A = torch.rand((M, K), dtype=torch.float16, device='cuda')
B = torch.rand((K, N), dtype=torch.float16, device='cuda')
C = torch.zeros((M, N), dtype=torch.float16, device='cuda')

A_T = A.transpose(0, 1).contiguous()
B_T = B.transpose(0, 1).contiguous()
C_T = C.transpose(0, 1).contiguous()

# MM from torch
torch.cuda.cudart().cudaProfilerStart()  
torch_output = torch.matmul(A, B)
torch.cuda.cudart().cudaProfilerStop()  

print(torch_output)

# hgemm from cuBLAS
torch.cuda.cudart().cudaProfilerStart()  
hgemm(A, B, C)
torch.cuda.cudart().cudaProfilerStop()  

print(C)

all_close = torch.allclose(torch_output, C, rtol=1e-2, atol=1e-4)
print("cublas verse torch: ", all_close)

# MM from triton (https://github.com/openai/triton)
torch.cuda.cudart().cudaProfilerStart() 
triton_output = triton_matmul(A, B)
torch.cuda.cudart().cudaProfilerStop()

print(triton_output)

all_close = torch.allclose(torch_output, triton_output, rtol=1e-2, atol=1e-4)
print("triton verse torch: ", all_close)

# MV from fast_gemv (https://github.com/wangsiping97/FastGEMV)
torch.cuda.cudart().cudaProfilerStart() 
fastgemv(B_T, A_T, C_T)
torch.cuda.cudart().cudaProfilerStop()

all_close = torch.allclose(torch_output, C_T.transpose(0, 1), rtol=1e-2, atol=1e-4)
print("fastgemv verse torch: ", all_close)

###########################################
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

###########################################
mv_dur = 0
for _ in range(10):
    torch_output = torch.matmul(A, B)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    _ = torch.mv(B_T, A_T.reshape((-1)))
    torch.cuda.synchronize()
    ed = time.time()
    mv_dur += (ed - st) * 10

###########################################
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

###########################################
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

###########################################
fastgemv_dur = 0
for _ in range(10):
    hgemm(A, B, C)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    fastgemv(B_T, A_T, C_T)
    torch.cuda.synchronize()
    ed = time.time()
    fastgemv_dur += (ed - st) * 10


print('torch[mm] - cublas - triton - torch[mv] - fastgemv: %.4f ms - %.4f ms - %.4f ms - %.4f ms - %.4f ms' 
      % (torch_dur, cublas_dur, triton_dur, mv_dur, fastgemv_dur))

