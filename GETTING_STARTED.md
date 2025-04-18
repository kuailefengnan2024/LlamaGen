## 入门指南
### 要求
- Linux，Python版本≥ 3.7
- PyTorch版本≥ 2.1
- A100 GPU

### 训练VQVAE模型
```
bash scripts/tokenizer/train_vq.sh --cloud-save-path /path/to/cloud_disk --data-path /path/to/imagenet/train --image-size 256 --vq-model VQ-16
```


### 预提取训练图像的离散代码
```
bash scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --data-path /path/to/imagenet/train --code-path /path/to/imagenet_code_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 384
```
和/或
``` 
bash scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --data-path /path/to/imagenet/train --code-path /path/to/imagenet_code_c2i_flip_ten_crop_105 --ten-crop --crop-range 1.05 --image-size 384
```


### 使用DDP训练AR模型
在运行之前，请更改`.sh`中的`nnodes, nproc_per_node, node_rank, master_addr, master_port`
```
bash scripts/autoregressive/train_c2i.sh --cloud-save-path /path/to/cloud_disk --code-path /path/to/imagenet_code_c2i_flip_ten_crop --image-size 384 --gpt-model GPT-B

bash scripts/autoregressive/train_c2i.sh --cloud-save-path /path/to/cloud_disk --code-path /path/to/imagenet_code_c2i_flip_ten_crop --image-size 384 --gpt-model GPT-L

bash scripts/autoregressive/train_c2i.sh --cloud-save-path /path/to/cloud_disk --code-path /path/to/imagenet_code_c2i_flip_ten_crop --image-size 384 --gpt-model GPT-XL
```


### 使用FSDP训练AR模型
在运行之前，请更改`.sh`中的`nnodes, nproc_per_node, node_rank, master_addr, master_port`
```
bash scripts/autoregressive/train_c2i_fsdp.sh --cloud-save-path /path/to/cloud_disk --code-path /path/to/imagenet_code_c2i_flip_ten_crop --image-size 384 --gpt-model GPT-XXL

bash scripts/autoregressive/train_c2i_fsdp.sh --cloud-save-path /path/to/cloud_disk --code-path /path/to/imagenet_code_c2i_flip_ten_crop --image-size 384 --gpt-model GPT-3B
```


### 采样
```
bash scripts/autoregressive/sample_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_B.pt --gpt-model GPT-B --image-size 384 --image-size-eval 256 --cfg-scale 2.0

bash scripts/autoregressive/sample_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_L.pt --gpt-model GPT-L --image-size 384 --image-size-eval 256 --cfg-scale 2.0

bash scripts/autoregressive/sample_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_XL.pt --gpt-model GPT-XL --image-size 384 --image-size-eval 256 --cfg-scale 1.75

bash scripts/autoregressive/sample_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_XXL.pt --gpt-model GPT-XXL --from-fsdp --image-size 384 --image-size-eval 256 --cfg-scale 1.75

bash scripts/autoregressive/sample_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_3B.pt --gpt-model GPT-3B --from-fsdp --image-size 384 --image-size-eval 256 --cfg-scale 1.65
```


### 评估
在评估之前，请参考[评估readme](evaluations/c2i/README.md)安装所需的软件包。
```
python3 evaluations/c2i/evaluator.py VIRTUAL_imagenet256_labeled.npz samples/GPT-B-c2i_B-size-384-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0.npz
```