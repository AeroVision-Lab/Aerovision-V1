# training/scripts/simple_forward.py
"""最简单的模型前向传播示例"""

import torch
import timm

# 1. 创建模型
# pretrained=True 表示使用在 ImageNet 上预训练的权重
model = timm.create_model("convnext_base", pretrained=True)
print(f"模型类型: {type(model).__name__}")

# 2. 创建一个假的输入图片
# 形状: [batch_size, channels, height, width]
# 这里是 1 张 224x224 的 RGB 图片
x = torch.randn(1, 3, 224, 224)
print(f"输入形状: {x.shape}")

# 3. 前向传播
model.eval()  # 设置为评估模式
with torch.no_grad():  # 不计算梯度（推理时）
    y = model(x)

print(f"输出形状: {y.shape}")  # [1, 1000] - ImageNet 有 1000 个类别

# 4. 获取预测类别
pred_class = y.argmax(dim=1).item()
print(f"预测类别索引: {pred_class}")

# 5. 查看概率分布
probs = torch.softmax(y, dim=1)
top5_probs, top5_indices = probs.topk(5)
print(f"Top-5 预测:")
for prob, idx in zip(top5_probs[0], top5_indices[0]):
    print(f"  类别 {idx.item()}: {prob.item():.4f}")