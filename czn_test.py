import torch
import time
from eed.backend import hgemm, fastgemv

def benchmark_shapes(shapes):
    results = []
    
    for M, K, N in shapes:
        print(f"\nTesting shape: M={M}, K={K}, N={N}")
        
        # Initialize tensors
        A = torch.rand((M, K), dtype=torch.float16, device='cuda')
        B = torch.rand((K, N), dtype=torch.float16, device='cuda')
        C = torch.zeros((M, N), dtype=torch.float16, device='cuda')
        
        A_T = A.transpose(0, 1).contiguous()
        B_T = B.transpose(0, 1).contiguous()
        C_T = C.transpose(0, 1).contiguous()

        # Warmup
        for _ in range(10):
            torch_output = torch.matmul(A, B)
            hgemm(A, B, C)
            fastgemv(B_T, A_T, C_T)

        # Benchmark torch.matmul
        torch_dur = 0
        for _ in range(10):
            torch.cuda.synchronize()
            st = time.time()
            torch_output = torch.matmul(A, B)
            torch.cuda.synchronize()
            ed = time.time()
            torch_dur += (ed - st) * 10

        # Benchmark cuBLAS
        cublas_dur = 0
        for _ in range(10):
            torch.cuda.synchronize()
            st = time.time()
            _ = torch.empty((M, N), dtype=torch.float16, device='cuda')
            hgemm(A, B, C)
            torch.cuda.synchronize()
            ed = time.time()
            cublas_dur += (ed - st) * 10

        # Benchmark FastGEMV
        fastgemv_dur = 0
        for _ in range(10):
            torch.cuda.synchronize()
            st = time.time()
            _ = torch.empty((N, M), dtype=torch.float16, device='cuda')
            fastgemv(B_T, A_T, C_T)
            torch.cuda.synchronize()
            ed = time.time()
            fastgemv_dur += (ed - st) * 10

        # Verify results
        all_close_cublas = torch.allclose(torch_output, C, rtol=1e-2, atol=1e-4)

        if not all_close_cublas:
            print(f"cuBLAS is not correct for shape {M}x{K}x{N}")
            max_diff = torch.max(torch.abs(torch_output - C))
            print(f"Max diff: {max_diff}")


        all_close_fastgemv = torch.allclose(torch_output, C_T.transpose(0, 1), rtol=1e-2, atol=1e-4)
        all_close_fastgemv_cublas = torch.allclose(C, C_T.transpose(0, 1), rtol=1e-2, atol=1e-4)

        results.append({
            'shape': (M, K, N),
            'torch_time': torch_dur,
            'cublas_time': cublas_dur,
            'fastgemv_time': fastgemv_dur,
            'fastgemv_cublas_ratio': fastgemv_dur / cublas_dur,
            'cublas_correct': all_close_cublas,
            'fastgemv_correct': all_close_fastgemv,
            'fastgemv_cublas_correct': all_close_fastgemv_cublas
        })

        print(f'Shape {M}x{K}x{N}:')
        print(f'torch[mm] - cublas - fastgemv: {torch_dur:.4f} ms - {cublas_dur:.4f} ms - {fastgemv_dur:.4f} ms')
        print(f'Correctness - cublas: {all_close_cublas}, fastgemv: {all_close_fastgemv}, fastgemv_cublas: {all_close_fastgemv_cublas}')

        
        # print(f'fastgemv_cublas_ratio: {fastgemv_dur / cublas_dur:.4f}')
    return results

# Define shapes to test
shapes_to_test = [
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    # llama2-13b
    (1, 5120, 5120),
    (1, 5120, 13824),
    (1, 13824, 5120),
    # qwen-14b / baichuan-7b 
    (1, 5120, 5120),
    (1, 5120, 13696),
    (1, 13696, 5120),
    # llama2-70b
    (1, 8192, 8192),
    (1, 8192, 28672),
    (1, 28672, 8192),
    #bs 2
    (2, 4096, 4096),
    (2, 4096, 11008),
    (2, 11008, 4096),
    # llama2-13b
    (2, 5120, 5120),
    (2, 5120, 13824),
    (2, 13824, 5120),
    # qwen-14b / baichuan-7b 
    (2, 5120, 5120),
    (2, 5120, 13696),
    (2, 13696, 5120),
    # llama2-70b
    (2, 8192, 8192),
    (2, 8192, 28672),
    (2, 28672, 8192),

]

# Run benchmarks
results = benchmark_shapes(shapes_to_test)

# Print summary
print("\nSummary:")
print("Shape (M,K,N) | Torch (ms) | cuBLAS (ms) | FastGEMV (ms) | FastGEMV_cublas_ratio | cuBLAS correct | FastGEMV correct | FastGEMV_cublas correct")
print("-" * 100)
for r in results:
    shape = r['shape']
    print(f"{shape} | {r['torch_time']:.4f} | {r['cublas_time']:.4f} | {r['fastgemv_time']:.4f} | {r['fastgemv_cublas_ratio']:.4f} | {r['cublas_correct']} | {r['fastgemv_correct']} | {r['fastgemv_cublas_correct']}")
