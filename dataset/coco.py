"""
COCO数据集加载模块。
此脚本提供了加载和处理COCO数据集的功能。
包含SingleFolderDataset类，用于从单个文件夹加载图像数据。
主要用于简单的图像数据处理和模型训练。

训练用途:
- 目标检测
- 实例分割
- 关键点检测
- 场景理解
- 图像生成与增强

数据集特色:
- 包含33万张图像，超过200万个实例标注
- 涵盖80个常见目标类别
- 提供复杂场景中的多目标图像
- 包含丰富的上下文信息和自然场景
- 适合训练需要理解场景结构的生成模型
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class SingleFolderDataset(Dataset):
    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, file_name) for file_name in os.listdir(directory)
                            if os.path.isfile(os.path.join(directory, file_name))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)


def build_coco(args, transform):
    return SingleFolderDataset(args.data_path, transform=transform)