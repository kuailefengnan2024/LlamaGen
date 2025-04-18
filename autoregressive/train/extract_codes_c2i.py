"""
类别到图像训练前处理模块：autoregressive/train/extract_codes_c2i.py
用途：提取并存储用于训练类别到图像模型的图像编码
功能：
1. 使用矢量量化（VQ）模型将图像转换为离散编码
2. 支持分布式处理，加速大规模数据集的处理
3. 提供数据增强选项，包括翻转和十裁剪（ten-crop）
4. 将类别标签和图像编码成对存储，保持对应关系
5. 支持调试模式，便于开发和测试

该模块是LlamaGen项目中类别到图像（Class-to-Image）训练流程的预处理步骤，
通过将图像转换为离散编码并与类别标签配对，为后续自回归模型训练准备数据，
提高训练效率并支持多种数据增强策略。
"""

# 修改自:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import os

from utils.distributed import init_distributed_mode
from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_model import VQ_models


#################################################################################
#                                  训练循环                                       #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # 设置DDP:
    if not args.debug:
        init_distributed_mode(args)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = 'cuda'
        rank = 0
    
    # 设置特征文件夹:
    if args.debug or rank == 0:
        os.makedirs(args.code_path, exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_codes'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_labels'), exist_ok=True)

    # 创建并加载模型
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # 设置数据:
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size), # 这是一个PIL图像的元组
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # 返回一个4D张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        crop_size = args.image_size 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    dataset = build_dataset(args, transform=transform)
    if not args.debug:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=1, # 重要！
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    total = 0
    for x, y in loader:
        x = x.to(device)
        if args.ten_crop:
            x_all = x.flatten(0, 1)
            num_aug = 10
        else:
            x_flip = torch.flip(x, dims=[-1])
            x_all = torch.cat([x, x_flip])
            num_aug = 2
        y = y.to(device)
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(x_all)
        codes = indices.reshape(x.shape[0], num_aug, -1)

        x = codes.detach().cpu().numpy()    # (1, num_aug, args.image_size//16 * args.image_size//16)
        train_steps = rank + total
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_codes/{train_steps}.npy', x)

        y = y.detach().cpu().numpy()    # (1,)
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_labels/{train_steps}.npy', y)
        if not args.debug:
            total += dist.get_world_size()
        else:
            total += 1
        print(total)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="vq模型的检查点路径")
    parser.add_argument("--codebook-size", type=int, default=16384, help="向量量化的码本大小")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="向量量化的码本维度")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--ten-crop", action='store_true', help="是否使用随机裁剪")
    parser.add_argument("--crop-range", type=float, default=1.1, help="中心裁剪的扩展范围")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
