"""
Pexels图像数据集加载模块。
此脚本提供了加载Pexels图像数据集的功能，使用了torchvision的ImageFolder类。
简单封装了对Pexels图像数据的访问，方便统一数据加载接口。

训练用途:
- 高质量图像生成模型
- 风格迁移
- 图像增强
- 真实场景渲染

数据集特色:
- 包含高分辨率、高质量的摄影作品
- 覆盖多种场景、主题和视觉风格
- 专业级别的照片质量，适合训练生成高质量图像的模型
- 自然光照和构图，适合学习真实世界的视觉特征
"""
from torchvision.datasets import ImageFolder

def build_pexels(args, transform):
    return ImageFolder(args.data_path, transform=transform)