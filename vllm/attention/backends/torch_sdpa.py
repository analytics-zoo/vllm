"""Attention layer with torch scaled_dot_product_attention and PagedAttention."""
import importlib
from typing import List, Optional, Tuple, Type, Dict
from dataclasses import dataclass

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.utils import is_xpu


class TorchSDPABackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["TorchSDPABackendImpl"]:
        return TorchSDPABackendImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "TorchSDPAMetadata":
        return TorchSDPAMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class TorchSDPAMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for TorchSDPABackend.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    slot_mapping: torch.Tensor
    prompt_lens: Optional[List[int]]
    prompt_lens_tensor: Optional[torch.Tensor]
    num_prompt_tokens: int
    num_generation_tokens: int

    max_subquery_len: Optional[int] = None
    max_prompt_len: Optional[int] = None
    subquery_start_loc: Optional[torch.Tensor] = None
    seq_start_loc: Optional[torch.Tensor] = None
    use_cuda_graph: bool = False

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[torch.Tensor] = None


class TorchSDPABackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            assert len(alibi_slopes) == num_heads
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)
        self.fuse_batch = is_xpu()

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: TorchSDPAMetadata,
    ) -> torch.Tensor:
        """Forward pass with torch SDPA and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                attn_metadata.kv_cache_dtype)

        if attn_metadata.is_prompt:
            if kv_cache is None or attn_metadata.block_tables.numel() == 0:
                if self.num_kv_heads != self.num_heads:
                    key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
                    value = value.repeat_interleave(self.num_queries_per_kv,
                                                    dim=1)

                if attn_metadata.attn_bias is None:
                    if self.alibi_slopes is not None:
                        att_masks = _make_alibi_bias(
                            self.alibi_slopes, query.dtype,
                            attn_metadata.prompt_lens)  # type: ignore
                    elif self.sliding_window is not None:
                        att_masks = _make_sliding_window_bias(
                            attn_metadata.prompt_lens, self.sliding_window,
                            query.dtype)  # type: ignore
                    else:
                        att_masks = [None] * len(attn_metadata.prompt_lens)
                    attn_metadata.attn_bias = att_masks

                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)
                query = query.movedim(1, query.dim() - 2)
                key = key.movedim(1, key.dim() - 2)
                value = value.movedim(1, value.dim() - 2)

                if self.fuse_batch:
                    mask = _make_attention_mask(attn_metadata.attn_bias,
                                                attn_metadata.prompt_lens,
                                                sum(attn_metadata.prompt_lens),
                                                query.dtype).to(query.device)
                    out = torch.nn.functional.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=mask,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=self.scale).movedim(query.dim() - 2, 1).contiguous()
                else:
                    start = 0
                    out = torch.empty(
                        (1, num_tokens, self.num_heads, self.head_size),
                        dtype=query.dtype,
                        device=query.device)
                    for prompt_len, mask in zip(attn_metadata.prompt_lens,
                                                attn_metadata.attn_bias):
                        end = start + prompt_len
                        sub_out = torch.nn.functional.scaled_dot_product_attention(
                            query[:, :, start:end, :],
                            key[:, :, start:end, :],
                            value[:, :, start:end, :],
                            attn_mask=mask,
                            dropout_p=0.0,
                            is_causal=not self.need_mask,
                            scale=self.scale).movedim(query.dim() - 2, 1)
                        out[:, start:end, :, :] = sub_out
                        start = end

                output = out.view_as(query).to(query.dtype)
            else:
                # prefix-enabled attention
                raise RuntimeError(
                    "Torch SDPA backend doesn't support prefix decoding.")

        else:
            # Decoding run.
            output = PagedAttention.forward_decode(
                query,
                key_cache,
                value_cache,
                attn_metadata.block_tables,
                attn_metadata.context_lens,
                attn_metadata.max_context_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)


def _make_attention_mask(
    att_bias: List[torch.Tensor],
    prompt_lens: List[int],
    prompt_token_num: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert att_bias[0].dim() == 3
    assert len(att_bias) == len(prompt_lens)
    head_size, _, _ = att_bias[0].size()
    mask = torch.empty(head_size,
                       prompt_token_num,
                       prompt_token_num,
                       dtype=dtype)
    mask.fill_(-torch.inf)
    start = 0
    for prompt_len, sub_mask in zip(prompt_lens, att_bias):
        end = start + prompt_len
        mask[:, start:end, start:end] = sub_mask
        start += prompt_len
    return mask

def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
    prompt_lens: List[int],
) -> List[torch.Tensor]:
    attn_biases = []
    for prompt_len in prompt_lens:
        bias = torch.arange(prompt_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(prompt_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        bias = bias[None, :] - bias[:, None]

        num_heads = alibi_slopes.shape[0]
        bias = bias[None, :].expand(num_heads, prompt_len, prompt_len)
        bias.mul_(alibi_slopes[:, None, None])
        inf_mask = torch.empty(
            (1, prompt_len, prompt_len),
            dtype=bias.dtype).fill_(-torch.inf).triu_(diagonal=1)
        attn_biases.append((bias + inf_mask).to(dtype))

    return attn_biases


def _make_sliding_window_bias(
    prompt_lens: List[int],
    window_size: Optional[int],
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    attn_biases = []
    for prompt_len in prompt_lens:
        tensor = torch.full(
            (1, prompt_len, prompt_len),
            dtype=dtype,
            fill_value=1,
        )
        shift = 0
        mask = torch.tril(tensor, diagonal=shift).to(dtype)  # type: ignore
        if window_size is not None:
            mask = torch.triu(mask, diagonal=shift - window_size + 1)
        mask = torch.log(mask)
        attn_biases.append(mask.to(dtype))

    return attn_biases
