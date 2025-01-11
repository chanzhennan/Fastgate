import torch
import speedgate  # 假设你的C++模块名是speedgate

def test_rotary_pos_emb():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 准备测试数据
    batch_size = 2
    seq_len = 8
    num_heads = 4
    head_dim = 32
    
    # 创建输入tensor
    t = torch.randn(batch_size, seq_len, num_heads, head_dim).half().cuda()
    
    # 创建频率tensor
    freqs = torch.randn(seq_len, head_dim // 2).cuda()  # 通常是head_dim的一半
    
    # 创建输出tensor
    output = torch.empty_like(t).cuda()
    
    # 调用C++实现
    result = speedgate.rotary_pos_emb(t, freqs, output)
    
    # 打印形状和值
    print("Shapes:")
    print(f"Input shape: {t.shape}")
    print(f"Freqs shape: {freqs.shape}")
    print(f"Output shape: {result.shape}")
    
    print("\nFirst few values:")
    print(f"Input first values:\n{t[0, 0, 0, :5]}")
    print(f"Output first values:\n{result[0, 0, 0, :5]}")
    
    # 检查输出是否有效
    print("\nChecks:")
    print(f"Has NaN: {torch.isnan(result).any()}")
    print(f"Has Inf: {torch.isinf(result).any()}")
    print(f"Max value: {result.abs().max()}")
    print(f"Min value: {result.abs().min()}")

if __name__ == "__main__":
    test_rotary_pos_emb()