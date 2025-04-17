"""
OpenImages数据集加载模块。
此脚本提供了从OpenImages数据集加载和处理图像数据的功能。
主要包含DatasetJson类，用于从json文件中读取图像路径并加载图像数据。

训练用途:
- 目标检测与识别
- 视觉属性分类
- 大规模图像生成模型训练

数据集特色:
- 包含约900万张图像
- 带有丰富的图像级标签和对象级标注
- 多样化的视觉内容，覆盖广泛的场景和物体类别
- 适合训练需要大量多样化数据的生成模型
"""
import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DatasetJson(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        json_path = os.path.join(data_path, 'image_paths.json')
        assert os.path.exists(json_path), f"please first run: python3 tools/openimage_json.py"
        with open(json_path, 'r') as f:
            self.image_paths = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')
    
    def getdata(self, idx):
        image_path = self.image_paths[idx]
        image_path_full = os.path.join(self.data_path, image_path)
        image = Image.open(image_path_full).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)


def build_openimage(args, transform):
    return DatasetJson(args.data_path, transform=transform)
