"""
该模块提供项目中使用的日志记录功能。
在LlamaGen项目中，此模块主要负责：
1. 创建统一格式的日志记录器，输出到控制台和文件
2. 针对分布式训练环境进行优化，只在主进程上记录日志
3. 为项目的各个组件提供一致的日志记录接口

有效的日志记录对于监控模型训练过程、调试问题和记录实验结果至关重要，
此模块确保日志信息的清晰可见和持久保存。
"""

import logging
import torch.distributed as dist

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger