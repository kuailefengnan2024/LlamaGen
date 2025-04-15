"""
该脚本用于将PyTorch Lightning格式的模型权重转换为标准PyTorch格式。
在LlamaGen项目中，此脚本的主要作用是：
1. 处理从PyTorch Lightning框架训练的向量量化(VQ)模型
2. 转换权重格式，使其与标准PyTorch兼容
3. 简化模型权重的加载过程，移除不必要的Lightning特定结构

此转换对于项目中使用预训练VQ模型进行图像编码和生成至关重要，
确保模型可以在不依赖PyTorch Lightning的环境中正常使用。
"""

import os
import torch

MODEL_PATH = 'pretrained_models'
pt_lightnings = [
    'vqgan_imagenet_f16_1024/ckpts/last.ckpt',
    'vqgan_imagenet_f16_16384/ckpts/last.ckpt',
    'vq-f8-n256/model.ckpt',
    'vq-f8/model.ckpt',
]
pts = [
    'vqgan_imagenet_f16_1024/ckpts/last.pth',
    'vqgan_imagenet_f16_16384/ckpts/last.pth',
    'vq-f8-n256/model.pth',
    'vq-f8/model.pth',
]

for pt_l, pt in zip(pt_lightnings, pts):
    pt_l_weight = torch.load(os.path.join(MODEL_PATH, pt_l), map_location='cpu')
    pt_weight = {
        'state_dict': pt_l_weight['state_dict']
    }
    pt_path = os.path.join(MODEL_PATH, pt)
    torch.save(pt_weight, pt_path)
    print(f'saving to {pt_path}')
