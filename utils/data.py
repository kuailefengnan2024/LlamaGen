"""
该模块提供数据处理和转换的辅助函数。
在LlamaGen项目中，此模块主要负责：
1. 处理图像数据，如居中裁剪等操作
2. 确保输入图像符合模型所需的尺寸和格式
3. 提供跨数据集的一致预处理方法

这些函数确保模型训练和推理过程中使用的图像数据经过适当处理，
保持一致的质量和标准化格式，从而提高模型性能。
"""

import numpy as np
from PIL import Image

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])