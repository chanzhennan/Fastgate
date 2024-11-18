import torch
import time
from eed.backend import hgemm, fastgemv
from triton_mm import matmul as triton_matmul

# Define test configurations
configs = [
    # llama2-7b
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    # # # llama2-13b
    (1, 5120, 5120),
    (1, 5120, 13824),
    (1, 13824, 5120),
    # # # qwen-14b / baichuan-7b 
    (1, 5120, 5120),
    (1, 5120, 13696),
    # (1, 13696, 5120),
    # # # llama2-70b
    # (1, 8192, 8192),
    # (1, 8192, 28672),
    # (1, 28672, 8192),
]

def run_benchmark(M, K, N):
    print(f"\nTesting dimensions: M={M}, K={K}, N={N}")
    
    A = torch.ones((M, K), dtype=torch.float16, device='cuda')
    B = torch.ones((K, N), dtype=torch.float16, device='cuda')
    C = torch.zeros((M, N), dtype=torch.float16, device='cuda')

    A_T = A.transpose(0, 1).contiguous()
    B_T = B.transpose(0, 1).contiguous()
    C_T = C.transpose(0, 1).contiguous()

    # MM from torch
    torch.cuda.cudart().cudaProfilerStart()  
    torch_output = torch.matmul(A, B)
    torch.cuda.cudart().cudaProfilerStop()  

    # MV from fast_gemv
    torch.cuda.cudart().cudaProfilerStart() 
    fastgemv(B_T, A_T, C_T)
    torch.cuda.cudart().cudaProfilerStop()

    all_close = torch.allclose(torch_output, C_T.transpose(0, 1), rtol=1e-2, atol=1e-4)
    # abs_diff = torch.abs(torch_output - C_T.transpose(0, 1))
    # rel_diff = abs_diff / (torch.abs(torch_output) + 1e-7)  # 添加小值避免除零
    
    # max_abs_diff = torch.max(abs_diff).item()
    # max_rel_diff = torch.max(rel_diff).item()
    # max_abs_diff_idx = torch.argmax(abs_diff)
    
    # print(f"Maximum absolute difference: {max_abs_diff}")
    # print(f"Maximum relative difference: {max_rel_diff}")
    # print(f"At index {max_abs_diff_idx}:")
    # print(f"torch_output: {torch_output.flatten()[max_abs_diff_idx]}")
    # print(f"fastgemv output: {C_T.transpose(0, 1).flatten()[max_abs_diff_idx]}")
    
    print("fastgemv verse torch allclose: ", all_close)
    # print("\n\n\n\n\n\n\n\n")

    # ###########################################
    # Timing tests
    torch_dur = 0
    for _ in range(10):  # warmup
        torch_output = torch.matmul(A, B)

    for _ in range(100):
        torch.cuda.synchronize()
        st = time.time()
        torch_output = torch.matmul(A, B)
        torch.cuda.synchronize()
        ed = time.time()
        torch_dur += (ed - st) * 10

    # ###########################################
    fastgemv_dur = 0
    for _ in range(10):  # warmup
        fastgemv(B_T, A_T, C_T)

    for _ in range(100):
        torch.cuda.synchronize()
        st = time.time()
        _ = torch.empty((N, M), dtype=torch.float16, device='cuda')
        fastgemv(B_T, A_T, C_T)
        torch.cuda.synchronize()
        ed = time.time()
        fastgemv_dur += (ed - st) * 10

    print('torch[mm] - fastgemv: %.4f ms - %.4f ms' 
          % (torch_dur, fastgemv_dur))

    _ = torch.empty((N, M), dtype=torch.float16, device='cuda')
    fastgemv(B_T, A_T, C_T)
    torch.cuda.synchronize()

# Run benchmarks for all configurations
for M, K, N in configs:
    run_benchmark(M, K, N)
