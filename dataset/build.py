"""
数据集构建模块。
此脚本是数据集加载的统一入口点，负责根据配置构建不同类型的数据集。
支持多种数据集类型：
- ImageNet（原始图像和代码形式）
- COCO
- OpenImage
- Pexels
- T2I（文本到图像，包括图像、代码等多种形式）
提供了统一的接口，方便模型训练和评估过程中的数据加载。

用途:
- 统一不同数据集的加载接口
- 简化模型训练流程
- 支持多种数据集类型的切换
- 实现数据集配置的灵活管理

特色:
- 模块化设计，易于扩展新数据集
- 基于命令行参数的数据集选择
- 支持图像和文本图像混合数据集
- 与训练框架无缝集成
- 提供标准化的数据接口，减少重复代码
"""
from dataset.imagenet import build_imagenet, build_imagenet_code
from dataset.coco import build_coco
from dataset.openimage import build_openimage
from dataset.pexels import build_pexels
from dataset.t2i import build_t2i, build_t2i_code, build_t2i_image


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    if args.dataset == 'pexels':
        return build_pexels(args, **kwargs)
    if args.dataset == 't2i_image':
        return build_t2i_image(args, **kwargs)
    if args.dataset == 't2i':
        return build_t2i(args, **kwargs)
    if args.dataset == 't2i_code':
        return build_t2i_code(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')