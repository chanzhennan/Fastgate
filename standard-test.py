import torch
import time
from eed.backend import edgemm, edgemm_m8n128k64, edgemm_m8n256k64, edgemm_m8n128k128, edgemv_m1n128k64x4, edgemm_m8n128k64x4, edgemv_m1n256k64x4
from eed.backend import edgemm_m8n128k64x4_bt, edgemm_m8n128k64x8, edgemm_m8n32k128x8_bt, edgemm_m8n32k256x8_bt
from eed.backend import edgemm_m8n256k32x8, edgemm_m8n256k64x8, edgemm_m8n512k32x16, edgemm_m8n256k32x16, edgemm_m8n64k128x8
from eed.backend import edgemm_m8n32k128x8_bt_tuning, edgemm_m8n32k256x8_bt_tuning, edgemm_m8n64k128x8_bt_tuning
from eed.backend import fastgemv_tuned, fastgemv, hgemm_warp2, edgemm_mix, fastgemv_ex, hgemm
from triton_mm import matmul

hidden = 4096
hidden_ = 5120
interm = 11008
interm_ = 13824

Configs = [
    # # M, K, N

    # (1, hidden, hidden),
    # (1, hidden, hidden*3),
    # (1, hidden, interm),
    # (1, hidden, interm*2),
    # (1, interm, hidden),
    
    # (2, hidden, hidden),
    # (2, hidden, interm),
    # (2, hidden, hidden*3),
    # (2, hidden, hidden*4),
    # (2, interm, hidden),
    # (2, hidden*4, hidden),
    # (2, hidden_, hidden_),
    # (2, hidden_, interm_),
    # (2, hidden_, hidden_*3),
    # (2, hidden_, hidden_*4),
    # (2, interm_, hidden_),
    # (2, hidden_*4, hidden_),

    # (4, hidden, hidden),
    # (4, hidden, interm),
    # (4, hidden, hidden*3),
    # (4, hidden, hidden*4),
    # (4, interm, hidden),
    # (4, hidden*4, hidden),
    # (4, hidden_, hidden_),
    # (4, hidden_, interm_),
    # (4, hidden_, hidden_*3),  
    # (4, hidden_, hidden_*4),
    # (4, interm_, hidden_),
    # (4, hidden_*4, hidden_),
    
    (8, hidden, hidden),
    (8, hidden, interm),
    (8, hidden, hidden*3),
    (8, hidden, hidden*4),
    (8, interm, hidden),
    (8, hidden*4, hidden),
    (8, hidden_, hidden_),
    (8, hidden_, interm_),
    (8, hidden_, hidden_*3),
    (8, hidden_, hidden_*4),
    (8, interm_, hidden_),
    (8, hidden_*4, hidden_),
]



Funcs = [
    # edgemm, 
    # edgemm_m8n128k64,
    # edgemm_m8n256k64,
    # edgemm_m8n128k128,
    # edgemv_m1n128k64x4,
    # edgemv_m1n256k64x4,
    # edgemm_m8n128k64x4,
    # edgemm_m8n128k64x8,          # 之前的v8
    # edgemm_m8n256k32x8,          # 大部分情况最快
    # edgemm_mix,
    # edgemm_m8n256k64x8,        # K = interm最快
    # edgemm_m8n512k32x16,
    # edgemm_m8n256k32x16
    # edgemm_m8n64k128x8,
    # hgemm_warp2
    # hgemm,
    edgemm_m8n32k128x8_bt,
    edgemm_m8n32k256x8_bt,
    # fastgemv_ex
    # edgemm_m8n128k64x4_bt
]

all_transe_func = [fastgemv, fastgemv_tuned]
b_transe_func = [edgemm_m8n32k256x8_bt, edgemm_m8n32k128x8_bt, fastgemv_ex, hgemm]
# tuning_func = [edgemm_m8n32k128x8_bt_tuning, edgemm_m8n32k256x8_bt_tuning, edgemm_m8n64k128x8_bt_tuning]
tuning_func = [edgemm_m8n64k128x8_bt_tuning]

def dispatch_func(func, A, B, C1, A_T, B_T, C_T):
    if func in all_transe_func:
        func(B_T, A_T, C_T)
    elif func in b_transe_func:
        func(A, B_T, C1)
    elif type(func) == torch.nn.Linear:
        C1 = func(A)
    else:
        func(A, B, C1)

# 输入A是正常的 B是转置的 C是正常的
def dispatch_func_btrans(func, A, B, C, bz=1):
    if func in all_transe_func:
        A_T = A.transpose(0, 1).contiguous()
        C_T = C.transpose(0, 1).contiguous()
        func(B, A_T, C_T)
    elif func in b_transe_func:
        func(A, B, C)
    elif func in tuning_func:
        func(A, B, C, bz)
    elif type(func) == torch.nn.Linear:
        C = func(A)
    else:
        B_T = B.transpose(0, 1).contiguous()
        func(A, B_T, C)

warmup = 10
iters = 100
torch.manual_seed(10)

def Linear_test():
    def speed_test(func, warmup, iters, A, B, C, bz=1):
        duration = 0
        for _ in range(warmup):
            dispatch_func_btrans(func, A, B, C, bz)

        for _ in range(iters):
            torch.cuda.synchronize()
            st = time.time()
            dispatch_func_btrans(func, A, B, C, bz)
            torch.cuda.synchronize()
            ed = time.time()
            duration += (ed - st) * 10
        return duration

    for f in tuning_func:
        for M, K, N in Configs:
            A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=1.0)
            B = torch.empty((K, N), dtype=torch.float16, device="cuda").normal_(mean=1., std=1.0)
            B[:, -4:-2] = B[:, -4:-2] / 100
            C_ = torch.zeros((M, N), dtype=torch.float16, device='cuda')
            C = torch.zeros((M, N), dtype=torch.float16, device='cuda')
            C1 = torch.zeros((M, N), dtype=torch.float16, device='cuda')
            A_T = A.transpose(0, 1).contiguous()
            B_T = B.transpose(0, 1).contiguous()
            C_T = C1.transpose(0, 1).contiguous()
            torch_mm = torch.nn.Linear(K, N, bias=False, dtype=torch.float16, device='cuda')
            torch_mm.weight = torch.nn.Parameter(B_T)
            torch.matmul(A, B, out=C_)
            hgemm(A,B_T,C)
            cublas_acc = C[abs(C_- C).le(100)].shape[0] / (C.shape[0] * C.shape[1])
            ours_acc = []

            ############################# test not tuning func #########################
            # for f in Funcs:
            #     C1 = torch.zeros((M, N), dtype=torch.float16, device='cuda')
            #     dispatch_func(f, A, B, C1, A_T, B_T, C_T)
            #     acc = C1[abs(C_- C1).le(100)].shape[0] / (C1.shape[0] * C1.shape[1])
            #     ours_acc.append(acc)

            C1 = torch.zeros((M, N), dtype=torch.float16, device='cuda')
            dispatch_func_btrans(f, A, B_T, C1, 1)
            acc = C1[abs(C_- C1).le(100)].shape[0] / (C1.shape[0] * C1.shape[1])
            ours_acc.append(acc)

            torch_dur = speed_test(torch_mm, warmup, iters, A, B_T, C)
            cublas_dur = speed_test(hgemm, warmup, iters, A, B_T, C)
            ours_dur = []
            # for f in Funcs:
            #     ours_dur.append(speed_test(f, warmup, iters, A, B_T, C))
            

            for bz in [1,2]:
                ours_dur.append(speed_test(f, warmup, iters, A, B_T, C, bz))
            
            
            print(M,K,N,'%.4f' %(torch_dur),'%.4f' %(cublas_dur), end=' ')
            for dur in ours_dur:
                print('%.4f' %(dur), end=' ')
            print(cublas_acc, end=' ')
            for acc in ours_acc:
                print(acc, end=' ')
            print()



def iter_func():
    for func in Funcs:
        for M, K, N in Configs:
                

                A = torch.rand((M, K), dtype=torch.float16, device='cuda')
                B = torch.rand((K, N), dtype=torch.float16, device='cuda')
                B[:, -4:-2] = B[:, -4:-2] / 100
                C_ = torch.zeros((M, N), dtype=torch.float16, device='cuda')
                C = torch.zeros((M, N), dtype=torch.float16, device='cuda')
                C2 = torch.zeros((M, N), dtype=torch.float16, device='cuda')

                A_T = A.transpose(0, 1).contiguous()
                B_T = B.transpose(0, 1).contiguous()
                C_T = C.transpose(0, 1).contiguous()

                torch.matmul(A, B, out=C_)

                # print(C_)

                func(A, B, C)
                torch.cuda.synchronize()
                func(A, B, C2)
                torch.cuda.synchronize()
                print(C)
                print(C2)
                print(torch.allclose(C2, C, rtol=1e-1, atol=1e-2))
                
                # fastgemv(B_T, A_T, C_T)
                # C = C_T.transpose(0, 1).contiguous()
                # print(C)

                all_close = torch.allclose(C_, C, rtol=1e-1, atol=1e-2)
                print("func verse torch: ", all_close)
                # if not all_close:
                #     continue

                torch_dur = 0
                for _ in range(warmup):
                    torch.matmul(A, B, out=C_)

                for _ in range(iters):
                    torch.cuda.synchronize()
                    st = time.time()
                    torch.matmul(A, B, out=C_)
                    torch.cuda.synchronize()
                    ed = time.time()
                    torch_dur += (ed - st) * 10

                edgemm_dur = 0
                for _ in range(warmup):
                    func(A, B, C)

                for _ in range(iters):
                    torch.cuda.synchronize()
                    st = time.time()
                    func(A, B, C)
                    torch.cuda.synchronize()
                    ed = time.time()
                    edgemm_dur += (ed - st) * 10

                fastgemv_dur = 0
                # for _ in range(warmup):
                #     fastgemv(B_T, A_T, C_T)

                # for _ in range(iters):
                #     torch.cuda.synchronize()
                #     st = time.time()
                #     fastgemv(B_T, A_T, C_T)
                #     torch.cuda.synchronize()
                #     ed = time.time()
                #     fastgemv_dur += (ed - st) * 10

                triton_dur = 0
                # for _ in range(warmup):
                #     matmul(A, B)

                # for _ in range(iters):
                #     torch.cuda.synchronize()
                #     st = time.time()
                #     matmul(A, B)
                #     torch.cuda.synchronize()
                #     ed = time.time()
                #     triton_dur += (ed - st) * 10

                # print('M- K - N - func: %d - %d - %d - %s\t' % (M, K, N, func.__name__), \
                #       'torch - edgemm - triton - fastgemv: %.4f ms - %.4f ms - %.4f ms - %.4f ms' % (torch_dur, edgemm_dur, triton_dur, fastgemv_dur))
                print(M,K,N,'%.4f' %(torch_dur),'%.4f' %(edgemm_dur))
                # print('M - K - N - func: %d - %d - %d - %s\t' % (M, K, N, func.__name__), \
                #       'torch - edgemm - t_sync - e_sync: %.4f ms - %.4f ms - %.4f ms - %.4f ms' % (torch_dur, torch_sync, edgemm_dur, edgemm_sync))


def iter_conf():
    func1 = Funcs[0]
    func2 = Funcs[1]
    for M, K, N in Configs:
        # A = torch.random((M, K), dtype=torch.float16, device='cuda')
        # B = torch.random((K, N), dtype=torch.float16, device='cuda')
        A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=1.0)
        B = torch.empty((K, N), dtype=torch.float16, device="cuda").normal_(mean=1., std=1.0)
        # A = A*10
        # B = B*10
        B[:, -4:-2] = B[:, -4:-2] / 100
        C_ = torch.zeros((M, N), dtype=torch.float16, device='cuda')

        C = torch.zeros((M, N), dtype=torch.float16, device='cuda')
        C1 = torch.zeros((M, N), dtype=torch.float16, device='cuda')

        A_T = A.transpose(0, 1).contiguous()
        B_T = B.transpose(0, 1).contiguous()
        C_T = C1.transpose(0, 1).contiguous()

        torch.matmul(A, B, out=C_)
        torch.cuda.synchronize()

        # func1(A,B,C)
        dispatch_func(func1, A,B,C,A_T,B_T,C_T)
        torch.cuda.synchronize()

        dispatch_func(func2, A,B,C1,A_T,B_T,C_T)
        torch.cuda.synchronize()
        
        acc_1 = C[abs(C_- C).le(100)].shape[0] / (C.shape[0] * C.shape[1])
        
        acc_2 = C1[abs(C_-C1).le(100)].shape[0] / (C1.shape[0] * C1.shape[1])
        # print(C)

        # all_close = torch.allclose(C_, C, rtol=1e-1, atol=1e-2)
        # print("func verse torch: ", all_close)
        # if not all_close:
        #     continue

        torch_dur = 0
        for _ in range(warmup):
            torch.matmul(A, B, out=C_)

        for _ in range(iters):
            torch.cuda.synchronize()
            st = time.time()
            torch.matmul(A, B, out=C_)
            torch.cuda.synchronize()
            ed = time.time()
            torch_dur += (ed - st) * 10

        f1_dur = 0
        for _ in range(warmup):
            dispatch_func(func1, A,B,C,A_T,B_T,C_T)

        for _ in range(iters):
            torch.cuda.synchronize()
            st = time.time()
            dispatch_func(func1, A,B,C,A_T,B_T,C_T)
            torch.cuda.synchronize()
            ed = time.time()
            f1_dur += (ed - st) * 10

        f2_dur = 0
        for _ in range(warmup):
            dispatch_func(func2, A,B,C1,A_T,B_T,C_T)
                
        for _ in range(iters):
            torch.cuda.synchronize()
            st = time.time()
            dispatch_func(func2, A,B,C1,A_T,B_T,C_T)
            torch.cuda.synchronize()
            ed = time.time()
            f2_dur += (ed - st) * 10

        print(M,K,N,'%.4f' %(torch_dur),'%.4f' %(f1_dur), '%.4f' %(f2_dur), acc_1, acc_2)
        # print(M,K,N,'%.4f' %(torch_dur),'%.4f' %(f1_dur), '%.4f' %(f2_dur))

Linear_test()
# iter_conf()
# iter_func()