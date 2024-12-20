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
        
        _A = A
        B_T = B.transpose(0, 1).contiguous()
        _C = torch.zeros((M, N), dtype=torch.float16, device='cuda')

        # Warmup
        for _ in range(10):
            torch_output = torch.matmul(A, B)
            hgemm(A, B, C)
            fastgemv(B_T, _A, _C)

        # import pdb; pdb.set_trace()
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
            fastgemv(B_T, _A, _C)
            torch.cuda.synchronize()
            ed = time.time()
            fastgemv_dur += (ed - st) * 10

        # Verify results
        all_close_cublas_torch = torch.allclose(torch_output, C, rtol=1e-2, atol=1e-4)
        all_close_fastgemv_torch = torch.allclose(torch_output, _C, rtol=1e-2, atol=1e-4)
        all_close_fastgemv_cublas = torch.allclose(C, _C, rtol=1e-2, atol=1e-4)

        results.append({
            'shape': (M, K, N),
            'torch_time': torch_dur,
            'cublas_time': cublas_dur,
            'fastgemv_time': fastgemv_dur,
            'fastgemv_cublas_ratio': f'{(fastgemv_dur / cublas_dur * 100):.2f}%',
            'cublas_torch_correct': all_close_cublas_torch,
            'fastgemv_torch_correct': all_close_fastgemv_torch,
            'fastgemv_cublas_correct': all_close_fastgemv_cublas
        })

        if not all_close_fastgemv_torch:
            print(f"cuBLAS is not correct for shape {M}x{K}x{N}")
            max_diff = torch.max(torch.abs(torch_output - C))
            print(f"Max diff: {max_diff}")
            print(f"torch_output: {torch_output[0, :10]}")
            print(f"C: {C[0, :10]}")
            print(f"C_T: {_C[0, :10]}")

        print(f'Shape {M}x{K}x{N}:')
        print(f'torch[mm] - cublas - fastgemv: {torch_dur:.4f} ms - {cublas_dur:.4f} ms - {fastgemv_dur:.4f} ms')
        print(f'Correctness - cublas: {all_close_cublas_torch}, fastgemv: {all_close_fastgemv_torch}, fastgemv_cublas: {all_close_fastgemv_cublas}')

        
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
    # # llama2-13b
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
print(f"{'Shape (M,K,N)':<15} | {'Torch (ms)':<10} | {'cuBLAS (ms)':<11} | {'FastGEMV (ms)':<13} | {'FastGEMV/cuBLAS':<15} | {'cuBLAS_Torch':<15} | {'FastGEMV_Torch':<15} | {'FastGEMV_cuBLAS':<15}")
print("-" * 130)

for r in results:
    shape = f"({r['shape'][0]},{r['shape'][1]},{r['shape'][2]})"
    print(f"{shape:<15} | {r['torch_time']:>10.4f} | {r['cublas_time']:>11.4f} | {r['fastgemv_time']:>13.4f} | {r['fastgemv_cublas_ratio']:>15} | {str(r['cublas_torch_correct']):>15} | {str(r['fastgemv_torch_correct']):>15} | {str(r['fastgemv_cublas_correct']):>15}")