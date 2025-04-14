import time
import argparse
import torch
from torchvision.utils import save_image

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.serve.gpt_model import GPT_models
from autoregressive.serve.llm import LLM 
from vllm import SamplingParams


def main(args):
    # 设置PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建并加载模型
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # 用于条件模型的标签（可以随意更改）:
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    latent_size = args.image_size // args.downsample_size
    qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]
    prompt_token_ids = [[cind] for cind in class_labels]
    if args.cfg_scale > 1.0:
        prompt_token_ids.extend([[args.num_classes] for _ in range(len(prompt_token_ids))])
    # 创建一个LLM.
    llm = LLM(
        args=args, 
        model='autoregressive/serve/fake_json/{}.json'.format(args.gpt_model), 
        gpu_memory_utilization=0.9, 
        skip_tokenizer_init=True)
    print(f"gpt model is loaded")

    # 创建采样参数对象.
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, 
        max_tokens=latent_size ** 2)

    # 从提示生成文本。输出是RequestOutput对象的列表
    # 包含提示、生成的文本和其他信息。
    t1 = time.time()
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False)
    sampling_time = time.time() - t1
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.") 

    # 解码为图像
    index_sample = torch.tensor([output.outputs[0].token_ids for output in outputs], device=device)
    if args.cfg_scale > 1.0:
        index_sample = index_sample[:len(class_labels)]
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape) # 输出值在[-1, 1]之间
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # 保存并显示图像:
    save_image(samples, "sample_{}_vllm.png".format(args.gpt_type), nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_{args.gpt_type}_vllm.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, required=True, help="gpt模型的检查点路径")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="类别条件或文本条件")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="条件输入的最大标记数")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="vq模型的检查点路径")
    parser.add_argument("--codebook-size", type=int, default=16384, help="向量量化的码本大小")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="向量量化的码本维度")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000,help="采样使用的top-k值")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样使用的温度值")
    parser.add_argument("--top-p", type=float, default=1.0, help="采样使用的top-p值")
    args = parser.parse_args()
    main(args)
