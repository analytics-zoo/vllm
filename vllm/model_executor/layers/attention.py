"""Multi-head attention."""
from typing import List, Optional
from functools import lru_cache
import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.utils import is_hip, is_xpu

if not is_xpu():
    from vllm.model_executor.layers.triton_kernel.prefix_prefill import (
        context_attention_fwd)

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttention(nn.Module):
    """MHA/MQA/GQA layer with PagedAttention.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Reshape and store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention using either
        xformers or the PagedAttention custom op.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                input_metadata.slot_mapping.flatten(),
            )

        if input_metadata.is_prompt:
            # Prompt run.
            # import pdb
            # pdb.set_trace()
            if self.num_kv_heads != self.num_heads:
                # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
                # project the key and value tensors to the desired number of
                # heads.
                # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                query = query.view(query.shape[0], self.num_kv_heads,
                                   self.num_queries_per_kv, query.shape[-1])
                key = key[:, :,
                          None, :].expand(key.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          key.shape[-1])
                value = value[:, :, None, :].expand(value.shape[0],
                                                    self.num_kv_heads,
                                                    self.num_queries_per_kv,
                                                    value.shape[-1])
            # normal attention
            if (key_cache is None or value_cache is None
                    or input_metadata.block_tables.numel() == 0):
                # Set attention bias if not provided. This typically happens at
                # the very attention layer of every iteration.
                # FIXME(woosuk): This is a hack.
                if input_metadata.attn_bias is None:
                    if self.alibi_slopes is None:
                        attn_bias = BlockDiagonalCausalMask.from_seqlens(
                            [seq_len] * batch_size)
                        # attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        #     [seq_len])
                        if self.sliding_window is not None:
                            attn_bias = attn_bias.make_local_attention(
                                self.sliding_window)
                        input_metadata.attn_bias = attn_bias
                    else:
                        input_metadata.attn_bias = _make_alibi_bias(
                            self.alibi_slopes, self.num_kv_heads, batch_size,
                            seq_len, query.dtype)

                # TODO(woosuk): Too many view operations. Let's try to reduce
                # them in the future for code readability.
                if self.alibi_slopes is None:
                    query = query.unsqueeze(0)
                    key = key.unsqueeze(0)
                    value = value.unsqueeze(0)
                else:
                    query = query.unflatten(0, (batch_size, seq_len))
                    key = key.unflatten(0, (batch_size, seq_len))
                    value = value.unflatten(0, (batch_size, seq_len))

                if is_xpu():
                    new_shape = (seq_len * batch_size + 8 - 1) & 0xFFFFFFF8
                    # print(input_metadata.attn_bias.shape, flush=True)
                    attn_mask = input_metadata.attn_bias.materialize(
                        (1, new_shape, new_shape),
                        # (1, 4, 4),
                        # (1, 32, 4),
                        dtype=query.dtype,
                        device=query.device).contiguous()
                    # print(attn_mask.shape)
                    # TODO: repeat kv
                    # print(f"We are here at scaled dot product with different shapes:"
                    #       f"query:{query.shape}, key:{key.shape}, value:{value.shape},"
                    #       f"attn_mask:{attn_mask.shape}"
                    #       f"with q.dtype: {query.dtype}"
                    #       f"with k.dtype: {key.dtype}"
                    #       f"with v.dtype: {value.dtype}"
                    #       f"with attn.dtype: {attn_mask.dtype}")

                    # original_shape = query.shape
                    # import pdb
                    # pdb.set_trace()
                    # new_query = query.reshape(query.shape[0], query.shape[1], query.shape[2] * query.shape[3], query.shape[4])
                    # new_key = key.reshape(key.shape[0], key.shape[1], key.shape[2] * key.shape[3], key.shape[4])
                    # new_value = value.reshape(value.shape[0], value.shape[1], value.shape[2] * value.shape[3], value.shape[4])

                    out = torch.nn.functional.scaled_dot_product_attention(
                        query.movedim(1, query.dim() - 2),
                        key.movedim(1, query.dim() - 2),
                        value.movedim(1, value.dim() - 2),
                        attn_mask,
                        0.0).movedim(query.dim() - 2, 1).contiguous()

                    # out = out.view(query.shape)

                else:
                    out = xops.memory_efficient_attention_forward(
                        query,
                        key,
                        value,
                        attn_bias=input_metadata.attn_bias,
                        p=0.0,
                        scale=self.scale,
                        op=xops.fmha.
                        MemoryEfficientAttentionFlashAttentionOp[0] if
                        (is_hip()) else None,
                    )
                output = out.view_as(query).to(query.dtype)
            else:
                # prefix-enabled attention
                output = torch.empty_like(query)
                context_attention_fwd(
                    query,
                    key,
                    value,
                    output,
                    key_cache,
                    value_cache,
                    input_metadata.block_tables,  # [BS, max_block_per_request]
                    input_metadata.start_loc,
                    input_metadata.prompt_lens,
                    input_metadata.context_lens,
                    input_metadata.max_seq_len,
                    getattr(self, "alibi_slopes", None),
                )

        else:
            # Decoding run.
            output = _paged_attention(
                query,
                key_cache,
                value_cache,
                input_metadata,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> LowerTriangularMaskWithTensorBias:
    bias = torch.arange(seq_len, dtype=dtype)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    attn_bias = LowerTriangularMaskWithTensorBias(bias)
    return attn_bias


def _paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    input_metadata: InputMetadata,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> torch.Tensor:
    output = torch.empty_like(query)

    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (
        (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
        _PARTITION_SIZE)
    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    # TODO(woosuk): Tune this heuristic.
    # For context len > 8192, use V2 kernel to avoid shared memory shortage.
    use_v1 = input_metadata.max_context_len <= 8192 and (
        max_num_partitions == 1 or num_seqs * num_heads > 512)
    if use_v1:
        # Run PagedAttention V1.
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
        )
    return output



class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        if _use_flash_attn():
            from vllm.model_executor.layers.attention.backends.flash_attn import FlashAttentionBackend  # noqa: E501
            self.backend = FlashAttentionBackend(num_heads, head_size, scale,
                                                 num_kv_heads, alibi_slopes,
                                                 sliding_window)
        elif is_xpu():
            from vllm.model_executor.layers.attention.backends.torch_sdpa import TorchSDPABackend  # noqa: E501
            self.backend = TorchSDPABackend(num_heads, head_size, scale,
                                            num_kv_heads, alibi_slopes,
                                            sliding_window)
        else:
            from vllm.model_executor.layers.attention.backends.xformers import XFormersBackend  # noqa: E501
            self.backend = XFormersBackend(num_heads, head_size, scale,
                                           num_kv_heads, alibi_slopes,
                                           sliding_window)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        return self.backend.forward(query, key, value, key_cache, value_cache,
                                    input_metadata)


@lru_cache(maxsize=1)
def _use_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info("flash_attn is not found. Using xformers backend.")
        return False

    if is_hip():
        # AMD GPUs.
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("flash_attn is not supported on Turing or older GPUs. "
                    "Using xformers backend.")
        return False
    if torch.get_default_dtype() not in (torch.float16, torch.bfloat16):
        logger.info(
            "flash_attn only supports torch.float16 or torch.bfloat16. "
            "Using xformers backend.")
        return False

    logger.info("Using flash_attn backend.")
    return True
