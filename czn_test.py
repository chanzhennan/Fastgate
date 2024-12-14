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
        for _ in range(100):
            torch.cuda.synchronize()
            st = time.time()
            torch_output = torch.matmul(A, B)
            torch.cuda.synchronize()
            ed = time.time()
            torch_dur += (ed - st) * 10

        # Benchmark cuBLAS
        cublas_dur = 0
        for _ in range(100):
            torch.cuda.synchronize()
            st = time.time()
            _ = torch.empty((M, N), dtype=torch.float16, device='cuda')
            hgemm(A, B, C)
            torch.cuda.synchronize()
            ed = time.time()
            cublas_dur += (ed - st) * 10

        # Benchmark FastGEMV
        fastgemv_dur = 0
        for _ in range(100):
            torch.cuda.synchronize()
            st = time.time()
            _ = torch.empty((N, M), dtype=torch.float16, device='cuda')
            fastgemv(B_T, A_T, C_T)
            torch.cuda.synchronize()
            ed = time.time()
            fastgemv_dur += (ed - st) * 10

        # Verify results
        all_close_cublas = torch.allclose(torch_output, C, rtol=1e-2, atol=1e-4)
        all_close_fastgemv = torch.allclose(torch_output, C_T.transpose(0, 1), rtol=1e-2, atol=1e-4)

        results.append({
            'shape': (M, K, N),
            'torch_time': torch_dur,
            'cublas_time': cublas_dur,
            'fastgemv_time': fastgemv_dur,
            'cublas_correct': all_close_cublas,
            'fastgemv_correct': all_close_fastgemv
        })

        print(f'Shape {M}x{K}x{N}:')
        print(f'torch[mm] - cublas - fastgemv: {torch_dur:.4f} ms - {cublas_dur:.4f} ms - {fastgemv_dur:.4f} ms')
        print(f'Correctness - cublas: {all_close_cublas}, fastgemv: {all_close_fastgemv}')

    return results

# Define shapes to test
shapes_to_test = [
    (1, 4096, 4096),    # Original shape
    (1, 8192, 8192),    # Larger square
]

# Run benchmarks
results = benchmark_shapes(shapes_to_test)

# Print summary
print("\nSummary:")
print("Shape (M,K,N) | Torch (ms) | cuBLAS (ms) | FastGEMV (ms) | cuBLAS correct | FastGEMV correct")
print("-" * 80)
for r in results:
    shape = r['shape']
    print(f"{shape} | {r['torch_time']:.4f} | {r['cublas_time']:.4f} | {r['fastgemv_time']:.4f} | {r['cublas_correct']} | {r['fastgemv_correct']}")