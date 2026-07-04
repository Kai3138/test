import torch
from mamba_ssm import Mamba2

# 设定统一的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x.shape: torch.Size([32, 87, 768])
batch, length, dim = 32, 87, 768
x = torch.randn(batch, length, dim).to(device)  # 将输入数据移到 device 上
# 初始化模型并移到 device 上
model = Mamba2(
    d_model=dim,   # Model dimension d_model
    d_state=64,    # SSM state expansion factor, typically 64 or 128
    d_conv=4,      # Local convolution width
    expand=2       # Block expansion factor
).to(device)

y = model(x)

print("x.shape:", x.shape)
print("y.shape:", y.shape)
assert y.shape == x.shape  # 检查输出的形状是否与输入匹配
