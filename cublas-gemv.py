import torch
import time
import argparse
from eed.backend import hgemm

parser = argparse.ArgumentParser()
parser.add_argument('--M', type=int, default=1)
parser.add_argument('--K', type=int, default=4096)
parser.add_argument('--N', type=int, default=4096)
args = parser.parse_args()

A = torch.rand((1, args.K), dtype=torch.float16, device='cuda')
A_ = torch.rand((args.M, args.K), dtype=torch.float16, device='cuda')
B = torch.rand((args.K, args.N), dtype=torch.float16, device='cuda')

# torch.cuda.cudart().cudaProfilerStart()  
# torch_output = torch.matmul(A, B)
# torch.cuda.cudart().cudaProfilerStop()  

# print(torch_output)

C = torch.zeros((1, args.N), dtype=torch.float16, device='cuda')
C_ = torch.zeros((args.M, args.N), dtype=torch.float16, device='cuda')

torch.cuda.cudart().cudaProfilerStart()  
hgemm(A, B, C)
hgemm(A_, B, C_)
torch.cuda.cudart().cudaProfilerStop()  

C_T = C_[0, :].reshape((1, -1))

print(C)
print(C_T)

all_close = torch.allclose(C, C_T, rtol=1e-2, atol=1e-4)
print(all_close)

gemv_dur = 0
for _ in range(10):
    hgemm(A, B, C)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    hgemm(A, B, C)
    torch.cuda.synchronize()
    ed = time.time()
    gemv_dur += (ed - st) * 10

gemm_dur = 0
for _ in range(10):
    hgemm(A_, B, C_)

for _ in range(100):
    torch.cuda.synchronize()
    st = time.time()
    hgemm(A_, B, C_)
    torch.cuda.synchronize()
    ed = time.time()
    gemm_dur += (ed - st) * 10

print('gemv - gemm: %.4f ms - %.4f ms' % (gemv_dur, gemm_dur))

