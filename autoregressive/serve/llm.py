# Modified from:
#   vLLM:    https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py
from typing import List, Optional, Union
import argparse

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
# from vllm.engine.llm_engine import LLMEngine
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter

from autoregressive.serve.llm_engine import LLMEngine


class LLM:
    """一个用于根据给定提示和采样参数生成文本的LLM。

    这个类包括一个分词器、一个语言模型（可能分布在多个GPU上）以及为中间状态（即KV缓存）
    分配的GPU内存空间。给定一批提示和采样参数，该类使用智能批处理机制和高效的内存管理
    从模型生成文本。

    注意：此类旨在用于离线推理。对于在线服务，请使用`AsyncLLMEngine`类。
    注意：有关参数的完整列表，请参见`EngineArgs`。

    参数：
        model: HuggingFace Transformers模型的名称或路径。
        tokenizer: HuggingFace Transformers分词器的名称或路径。
        tokenizer_mode: 分词器模式。"auto"将在可用时使用快速分词器，"slow"将始终使用慢速分词器。
        skip_tokenizer_init: 如果为真，跳过分词器和解分词器的初始化。
            预期输入中的prompt_token_ids有效，prompt为None。
        trust_remote_code: 下载模型和分词器时信任远程代码（例如来自HuggingFace）。
        tensor_parallel_size: 用于使用张量并行的分布式执行的GPU数量。
        dtype: 模型权重和激活的数据类型。目前，我们支持`float32`、`float16`和`bfloat16`。
            如果为`auto`，我们使用模型配置文件中指定的`torch_dtype`属性。
            但是，如果配置中的`torch_dtype`是`float32`，我们将使用`float16`代替。
        quantization: 用于量化模型权重的方法。目前，我们支持"awq"、"gptq"、"squeezellm"和"fp8"（实验性）。
            如果为None，我们首先检查模型配置文件中的`quantization_config`属性。如果那也是None，
            我们假设模型权重未量化，并使用`dtype`确定权重的数据类型。
        revision: 要使用的特定模型版本。可以是分支名称、标签名称或提交ID。
        tokenizer_revision: 要使用的特定分词器版本。可以是分支名称、标签名称或提交ID。
        seed: 初始化用于采样的随机数生成器的种子。
        gpu_memory_utilization: 为模型权重、激活和KV缓存保留的GPU内存比率（0到1之间）。
            较高的值将增加KV缓存大小，从而提高模型的吞吐量。但是，如果该值太高，
            可能会导致内存不足（OOM）错误。
        swap_space: 每个GPU用作交换空间的CPU内存大小（GiB）。当请求的`best_of`采样参数
            大于1时，这可以用于临时存储请求的状态。如果所有请求都有`best_of=1`，
            可以安全地将此设置为0。否则，太小的值可能会导致内存不足（OOM）错误。
        enforce_eager: 是否强制执行急切模式。如果为True，我们将禁用CUDA图并始终以急切模式执行模型。
            如果为False，我们将在混合模式下使用CUDA图和急切执行。
        max_context_len_to_capture: CUDA图覆盖的最大上下文长度。当序列的上下文长度大于此值时，
            我们回退到急切模式。
        disable_custom_all_reduce: 参见ParallelConfig
    """

    def __init__(
        self,
        args: argparse.ArgumentParser,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS, args=args)
        self.request_counter = Counter()

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> List[RequestOutput]:
        """为输入提示生成补全。

        注意：此类自动批处理给定的提示，考虑内存限制。为了获得最佳性能，
        将所有提示放入单个列表中，并将其传递给此方法。

        参数：
            prompts: 要为其生成补全的提示列表。
            sampling_params: 文本生成的采样参数。如果为None，我们使用默认的采样参数。
                当它是单个值时，它应用于每个提示。
                当它是一个列表时，该列表必须与提示具有相同的长度，并且它与提示一一配对。
            prompt_token_ids: 提示的标记ID列表。如果为None，我们使用分词器将提示转换为标记ID。
            use_tqdm: 是否使用tqdm显示进度条。
            lora_request: 用于生成的LoRA请求（如果有）。
            multi_modal_data: 多模态数据。

        返回：
            包含按输入提示顺序生成的补全的`RequestOutput`对象列表。
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if self.llm_engine.model_config.skip_tokenizer_init \
            and prompts is not None:
            raise ValueError("prompts must be None if skip_tokenizer_init "
                             "is True")
        if isinstance(prompts, str):
            # 将单个提示转换为列表。
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")

        if prompts is not None:
            num_requests = len(prompts)
        else:
            assert prompt_token_ids is not None
            num_requests = len(prompt_token_ids)

        if sampling_params is None:
            # 使用默认采样参数。
            sampling_params = SamplingParams()

        elif isinstance(sampling_params,
                        list) and len(sampling_params) != num_requests:
            raise ValueError("The lengths of prompts and sampling_params "
                             "must be the same.")
        if multi_modal_data:
            multi_modal_data.data = multi_modal_data.data.to(torch.float16)

        # Add requests to the engine.
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[i]
            self._add_request(
                prompt,
                sampling_params[i]
                if isinstance(sampling_params, list) else sampling_params,
                token_ids,
                lora_request=lora_request,
                # Get ith image while maintaining the batch dim.
                multi_modal_data=MultiModalData(
                    type=multi_modal_data.type,
                    data=multi_modal_data.data[i].unsqueeze(0))
                if multi_modal_data else None,
            )
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids,
                                    lora_request=lora_request,
                                    multi_modal_data=multi_modal_data)


    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=f"Generation Speed: {0:.2f} toks/s",
            )
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        total_toks += (sum(
                            len(stp.token_ids) for stp in output.outputs))
                        spd = total_toks / pbar.format_dict["elapsed"]
                        pbar.postfix = f"Generation Speed: {spd:.2f} toks/s"
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
