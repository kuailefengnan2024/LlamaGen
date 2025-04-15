"""
该模块实现指数移动平均(EMA)功能，用于模型训练中的权重平均。
在LlamaGen项目中，此模块主要负责：
1. 在训练过程中维护模型参数的平滑版本，减少参数值的波动
2. 提高模型最终性能和生成图像的质量
3. 控制模型参数是否需要梯度更新

EMA是稳定生成模型训练的重要技术，通过对模型参数进行时间上的平均，
可以得到更稳定、泛化能力更强的模型，特别适用于图像生成任务。
"""

import torch
from collections import OrderedDict

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag