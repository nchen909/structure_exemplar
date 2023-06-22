import torch
from thop import profile
from torchvision.models import resnet50

# 首先创建一个模型
model = resnet50()

# 创建一个随机输入
input = torch.randn(1, 3, 224, 224)

# 使用profile函数来计算FLOPs
macs, params = profile(model, inputs=(input,input ))

# 将FLOPs转换为TFLOPs
tflops = macs / (1024 ** 4)
print('TFLOPs:', tflops)
