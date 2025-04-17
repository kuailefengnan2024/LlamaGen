# 自回归模型击败扩散模型：🦙 Llama用于可扩展图像生成


<div align="center">

[![demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online_Demo-blue)](https://huggingface.co/spaces/FoundationVision/LlamaGen)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.06525-b31b1b.svg)](https://arxiv.org/abs/2406.06525)&nbsp;
[![project page](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://peizesun.github.io/llamagen/)&nbsp;

</div>


<p align="center">
<img src="assets/teaser.jpg" width=95%>
<p>



本仓库包含预训练模型权重和训练/采样PyTorch(torch>=2.1.0)代码，使用于

> [**自回归模型击败扩散模型：Llama用于可扩展图像生成**](https://arxiv.org/abs/2406.06525)<br>
> [Peize Sun](https://peizesun.github.io/), [Yi Jiang](https://enjoyyi.github.io/), [Shoufa Chen](https://www.shoufachen.com/), [Shilong Zhang](https://jshilong.github.io/), [Bingyue Peng](), [Ping Luo](http://luoping.me/), [Zehuan Yuan](https://shallowyuan.github.io/)
> <br>香港大学, 字节跳动<br>

您可以在[![项目页面](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://peizesun.github.io/llamagen/)上找到更多可视化内容。

## 🔥 更新
- [2024.06.28] 用于文本条件图像生成的图像分词器和AR模型已发布！快来试试！
- [2024.06.15] 所有从100M到3B参数的模型都支持vLLM！
- [2024.06.11] 用于类别条件图像生成的图像分词器和AR模型已发布！
- [2024.06.11] 代码和演示已发布！

## 🌿 介绍
我们介绍了LlamaGen，一种新的图像生成模型家族，将大型语言模型的原始"下一个令牌预测"范式应用于视觉生成领域。它是对没有视觉信号归纳偏差的普通自回归模型（例如Llama）在适当扩展时能否实现最先进图像生成性能的肯定回答。我们重新审视了图像分词器的设计空间、图像生成模型的可扩展性属性及其训练数据质量。

在本仓库中，我们发布了：
* 两种下采样比例为16和8的图像分词器。
* 七种从100M到3B参数的类别条件生成模型。
* 两种700M参数的文本条件生成模型。
* 在[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/FoundationVision/LlamaGen)上的在线演示，用于运行预训练模型。
* 支持vLLM服务框架，实现300% - 400%的加速。

## 🦄 ImageNet上的类别条件图像生成
### VQ-VAE模型
方法 | 参数 | 令牌 | rFID (256x256) | 权重
--- |:---:|:---:|:---:|:---:
vq_ds16_c2i | 72M | 16x16 | 2.19 | [vq_ds16_c2i.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt) 
vq_ds16_c2i | 72M | 24x24 | 0.94 | 上述
vq_ds16_c2i | 72M | 32x32 | 0.70 | 上述
vq_ds8_c2i  | 70M | 32x32 | 0.59 | [vq_ds8_c2i.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds8_c2i.pt)

### AR模型
方法 | 参数 | 训练 | 令牌 | FID (256x256) | 权重 
--- |:---:|:---:|:---:|:---:|:---:|
LlamaGen-B   | 111M | DDP | 16x16 | 5.46 | [c2i_B_256.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_B_256.pt)
LlamaGen-B   | 111M | DDP | 24x24 | 6.09 | [c2i_B_384.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_B_384.pt)
LlamaGen-L   | 343M | DDP | 16x16 | 3.80 | [c2i_L_256.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_L_256.pt)
LlamaGen-L   | 343M | DDP | 24x24 | 3.07 | [c2i_L_384.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_L_384.pt)
LlamaGen-XL  | 775M | DDP | 24x24 | 2.62 | [c2i_X_384L.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_XL_384.pt)
LlamaGen-XXL | 1.4B | FSDP | 24x24 | 2.34 | [c2i_XXL_384.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_XXL_384.pt)
LlamaGen-3B  | 3.1B | FSDP | 24x24 | 2.18 | [c2i_3B_384.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_3B_384.pt)


### 演示
请下载模型，将其放在文件夹`./pretrained_models`中，并运行
```
python3 autoregressive/sample/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_L_384.pt --gpt-model GPT-L --image-size 384
# 或者
python3 autoregressive/sample/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_XXL_384.pt --gpt-model GPT-XXL --from-fsdp --image-size 384
```
生成的图像将保存为`sample_c2i.png`。

### Gradio演示 <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>

您可以使用我们的在线gradio演示 [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/FoundationVision/LlamaGen) 或在本地运行gradio：
```bash
python app.py
```


## 🚀 文本条件图像生成
### VQ-VAE模型
方法 | 参数 | 令牌 | 数据 | 权重
--- |:---:|:---:|:---:|:---:
vq_ds16_t2i | 72M | 16x16 | LAION COCO (50M) + 内部数据 (10M) | [vq_ds16_t2i.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt)

### AR模型
方法 | 参数 | 令牌 | 数据 | 权重 
--- |:---:|:---:|:---:|:---:
LlamaGen-XL  | 775M | 16x16 | LAION COCO (50M) | [t2i_XL_stage1_256.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/t2i_XL_stage1_256.pt)
LlamaGen-XL  | 775M | 32x32 | 内部数据 (10M) | [t2i_XL_stage2_512.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/t2i_XL_stage2_512.pt)

### 演示
在运行演示之前，请参考[语言readme](language/README.md)安装所需的软件包和语言模型。  

请下载模型，将其放在文件夹`./pretrained_models`中，并运行
```
python3 autoregressive/sample/sample_t2i.py --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt --gpt-ckpt ./pretrained_models/t2i_XL_stage1_256.pt --gpt-model GPT-XL --image-size 256
# 或者
python3 autoregressive/sample/sample_t2i.py --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt --gpt-ckpt ./pretrained_models/t2i_XL_stage2_512.pt --gpt-model GPT-XL --image-size 512
```
生成的图像将保存为`sample_t2i.png`。

### 本地Gradio演示



## ⚡ 服务
我们使用服务框架[vLLM](https://github.com/vllm-project/vllm)以实现更高的吞吐量。请参考[服务readme](autoregressive/serve/README.md)安装所需的软件包。  
```
python3 autoregressive/serve/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_XXL_384.pt --gpt-model GPT-XXL --from-fsdp --image-size 384
```
生成的图像将保存为`sample_c2i_vllm.png`。


## 入门指南
请参阅[入门指南](GETTING_STARTED.md)以获取安装、训练和评估信息。


## 许可证
本项目的大部分内容在MIT许可证下授权。项目的部分内容在相应文件中详细说明的引用项目的单独许可证下提供。


## BibTeX
```bibtex
@article{sun2024autoregressive,
  title={Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation},
  author={Sun, Peize and Jiang, Yi and Chen, Shoufa and Zhang, Shilong and Peng, Bingyue and Luo, Ping and Yuan, Zehuan},
  journal={arXiv preprint arXiv:2406.06525},
  year={2024}
}
```
