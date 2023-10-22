import torch
import time

from eed.backend import gemm_m8n32k256x8_bt, gemm_m8n32k128x8_bt, gemm_m8n64k128x8_bt_exp

M = 2
K = 4096
N = 4096 * 4

A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=1., std=0.5)
C = torch.empty((M, N), dtype=torch.float16, device="cuda").normal_(mean=1., std=0.5)

torch.cuda.cudaProfilerStart()
gemm_m8n32k128x8_bt(A, B, C)
gemm_m8n64k128x8_bt_exp(A, B, C)
torch.cuda.cudaProfilerStop()