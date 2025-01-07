import torch
import time
from eed.backend import hgemm, fastgemm, mma

def benchmark_shapes(shapes):
    results = []
    

    for M, K, N in shapes:
        print(f"\nTesting shape: M={M}, K={K}, N={N}")
        
        # Initialize tensors
        A = torch.rand((M, K), dtype=torch.float16, device='cuda')
        B = torch.rand((K, N), dtype=torch.float16, device='cuda')
        C = torch.zeros((M, N), dtype=torch.float16, device='cuda')
        C_mma = torch.zeros((M, N), dtype=torch.float16, device='cuda')
        
        _A = A
        B_T = B.transpose(0, 1).contiguous()
        _C = torch.zeros((M, N), dtype=torch.float16, device='cuda')

        # Warmup
        for _ in range(10):
            torch_output = torch.matmul(A, B)
            hgemm(A, B, C)
            fastgemm(B_T, _A, _C)
            mma(A, B_T, C_mma)
            

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

        # Benchmark FastGEMM
        fastgemm_dur = 0
        for _ in range(10):
            torch.cuda.synchronize()
            st = time.time()
            _ = torch.empty((N, M), dtype=torch.float16, device='cuda')
            fastgemm(B_T, _A, _C)
            torch.cuda.synchronize()
            ed = time.time()
            fastgemm_dur += (ed - st) * 10

        # Benchmark mma
        mma_dur = 0
        for _ in range(10):
            torch.cuda.synchronize()
            st = time.time()
            _ = torch.empty((N, M), dtype=torch.float16, device='cuda')
            mma(A, B_T, C_mma)
            torch.cuda.synchronize()
            ed = time.time()
            mma_dur += (ed - st) * 10

        # Verify results
        all_close_cublas_torch = torch.allclose(torch_output, C, rtol=1e-2, atol=1e-4)
        all_close_fastgemm_torch = torch.allclose(torch_output, _C, rtol=1e-2, atol=1e-4)
        all_close_fastgemm_cublas = torch.allclose(C, _C, rtol=1e-2, atol=1e-4)
        all_close_mma_torch = torch.allclose(torch_output, C_mma, rtol=1e-2, atol=1e-4)

        # import pdb; pdb.set_trace() 

        results.append({
            'shape': (M, K, N),
            'torch_time': torch_dur,
            'cublas_time': cublas_dur,
            'fastgemm_time': fastgemm_dur,
            'mma_time': mma_dur,
            'fastgemm_cublas_ratio': f'{(fastgemm_dur / cublas_dur * 100):.2f}%',
            'mma_cublas_ratio': f'{(mma_dur / cublas_dur * 100):.2f}%',
            'cublas_torch_correct': all_close_cublas_torch,
            'fastgemm_torch_correct': all_close_fastgemm_torch,
            'fastgemm_cublas_correct': all_close_fastgemm_cublas,
            'mma_torch_correct': all_close_mma_torch
        })

        if not all_close_mma_torch:
            print(f"cuBLAS is not correct for shape {M}x{K}x{N}")
            max_diff_torch_cublas = torch.max(torch.abs(torch_output - C))
            max_diff_torch_mma = torch.max(torch.abs(torch_output - C_mma))
            print(f"Max diff (torch_output - C): {max_diff_torch_cublas}, (torch_output - C_mma): {max_diff_torch_mma}")
            print(f"torch_output: {torch_output[0, :10]}")
            print(f"C: {C[0, :10]}")
            print(f"C_T: {_C[0, :10]}")
            print(f"C_mma: {C_mma[0, :10]}")

        print(f'Shape {M}x{K}x{N}:')
        print(f'torch[mm] - cublas - fastgemm - mma: {torch_dur:.4f} ms - {cublas_dur:.4f} ms - {fastgemm_dur:.4f} ms - {mma_dur:.4f} ms')
        print(f'Correctness - cublas: {all_close_cublas_torch}, fastgemm: {all_close_fastgemm_torch}, fastgemm_cublas: {all_close_fastgemm_cublas}, mma: {all_close_mma_torch}')

        
    return results

# Define shapes to test
# (M, N, K)
shapes_to_test = [
    # (1, 4096, 4096),
    # (1, 4096, 11008),
    # (1, 11008, 4096),
    # # llama2-13b
    # (1, 5120, 5120),
    # (1, 5120, 13824),
    # (1, 13824, 5120),
    # # qwen-14b / baichuan-7b 
    # (1, 5120, 5120),
    # (1, 5120, 13696),
    # (1, 13696, 5120),
    # # llama2-70b
    # (1, 8192, 8192),
    # (1, 8192, 28672),
    # (1, 28672, 8192),
    #bs 2
    (512, 1024, 2048),
    # (2, 4096, 4096),
    # (2, 4096, 11008),
    # (2, 11008, 4096),
    # # # llama2-13b
    # (2, 5120, 5120),
    # (2, 5120, 13824),
    # (2, 13824, 5120),
    # # qwen-14b / baichuan-7b 
    # (2, 5120, 5120),
    # (2, 5120, 13696),
    # (2, 13696, 5120),
    # # llama2-70b
    # (2, 8192, 8192),
    # (2, 8192, 28672),
    # (2, 28672, 8192),

]

# Run benchmarks
results = benchmark_shapes(shapes_to_test)

# Print summary
print("\nSummary:")
print(f"{'Shape (M,K,N)':<15} | {'Torch (ms)':<10} | {'cuBLAS (ms)':<11} | {'FastGEMm (ms)':<13} | {'MMA (ms)':<8} | {'Mma/cuBLAS':<15} | {'cuBLAS_Torch':<15} | {'FastGEMm_Torch':<15} | {'FastGEMm_cuBLAS':<15} | {'MMA_Torch':<8} |")
print("-" * 155)

for r in results:
    shape = f"({r['shape'][0]},{r['shape'][1]},{r['shape'][2]})"
    print(f"{shape:<15} | {r['torch_time']:>10.4f} | {r['cublas_time']:>11.4f} | {r['fastgemm_time']:>13.4f} | {r['mma_time']:>8.4f} | {r['mma_cublas_ratio']:>15} | {str(r['cublas_torch_correct']):>15} | {str(r['fastgemm_torch_correct']):>15} | {str(r['fastgemm_cublas_correct']):>15} | {str(r['mma_torch_correct']):>8} |")