""" Attention layer with torch scaled_dot_product_attention
    and PagedAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm._ipex_ops import ipex_ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)

_PARTITION_SIZE = 512


class IpexAttnBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "ipex-attn"

    @staticmethod
    def get_impl_cls() -> Type["IpexAttnBackendImpl"]:
        return IpexAttnBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["IpexAttnMetadata"]:
        return IpexAttnMetadata

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
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class IpexAttnMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for IpexAttnBackend.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    slot_mapping: torch.Tensor
    seq_lens: Optional[List[int]]
    seqlen_q: Optional[torch.Tensor]
    max_seqlen: Optional[int]

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[torch.Tensor]] = None

    @property
    def prefill_metadata(self) -> Optional["IpexAttnMetadata"]:
        # Currently chunked prefill is not supported
        if self.num_decode_tokens == 0:
            assert self.num_prefills > 0
            return self

        return None

    @property
    def decode_metadata(self) -> Optional["IpexAttnMetadata"]:
        # Currently chunked prefill is not supported
        if self.num_prefills > 0:
            assert self.num_decode_tokens == 0
            return None

        return self


from torch.nn.functional import scaled_dot_product_attention

def _make_attention_mask(
    att_bias: List[torch.Tensor],
    seq_lens: List[int],
    prompt_token_num: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert att_bias[0].dim() == 3
    assert len(att_bias) == len(seq_lens)
    head_size, _, _ = att_bias[0].size()
    mask = torch.empty(head_size,
                       prompt_token_num,
                       prompt_token_num,
                       dtype=dtype)
    mask.fill_(-torch.inf)
    start = 0
    for prompt_len, sub_mask in zip(seq_lens, att_bias):
        end = start + prompt_len
        mask[:, start:end, start:end] = sub_mask
        start += prompt_len
    return mask


class IpexAttnBackendImpl(AttentionImpl[IpexAttnMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert blocksparse_params is None, ValueError(
            "Torch SPDA does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                "IPEX backend does not support FP8 KV cache. "
                "Please use xFormers backend instead.")

    def split_kv_cache(
        self,
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 1
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: IpexAttnMetadata,  # type: ignore
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with IPEX varlen_attention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert k_scale == 1.0 and v_scale == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "IpexAttnBackendImpl")
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = self.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
            # ipex_ops.reshape_and_cache(
            #     key,
            #     value,
            #     key_cache,
            #     value_cache,
            #     attn_metadata.slot_mapping.flatten(),
            #     self.kv_cache_dtype,
            #     k_scale,
            #     v_scale,
            # )

        if attn_metadata.is_prompt:
            assert attn_metadata.seq_lens is not None
            if (kv_cache is None or attn_metadata.block_tables.numel() == 0):
                if self.num_kv_heads != self.num_heads:
                    key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
                    value = value.repeat_interleave(self.num_queries_per_kv,
                                                    dim=1)

                if attn_metadata.attn_bias is None:
                    if self.alibi_slopes is not None:
                        att_masks = _make_alibi_bias(
                            self.alibi_slopes, query.dtype,
                            attn_metadata.seq_lens)  # type: ignore
                    elif self.sliding_window is not None:
                        att_masks = _make_sliding_window_bias(
                            attn_metadata.seq_lens, self.sliding_window,
                            query.dtype)  # type: ignore
                    else:
                        att_masks = _make_sliding_window_bias(
                            attn_metadata.seq_lens, None, dtype=query.dtype)
                    attn_metadata.attn_bias = att_masks

                # output = torch.empty(
                #     (num_tokens, self.num_heads, self.head_size),
                #     dtype=query.dtype,
                #     device=query.device)
                # ipex_ops.varlen_attention(query,
                #                           key,
                #                           value,
                #                           output,
                #                           attn_metadata.seqlen_q,
                #                           attn_metadata.seqlen_q,
                #                           attn_metadata.max_seqlen,
                #                           attn_metadata.max_seqlen,
                #                           pdropout=0.0,
                #                           softmax_scale=self.scale,
                #                           zero_tensors=False,
                #                           is_causal=True,
                #                           return_softmax=False,
                #                           gen_=None)

                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)
                query = query.movedim(1, query.dim() - 2)
                key = key.movedim(1, key.dim() - 2)
                value = value.movedim(1, value.dim() - 2)

                mask = _make_attention_mask(attn_metadata.attn_bias,
                                            attn_metadata.seq_lens,
                                            sum(attn_metadata.seq_lens),
                                            query.dtype).to(query.device)
                out = scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.scale).movedim(query.dim() - 2,
                                                1).contiguous()
                output = out.view_as(query).to(query.dtype)
            else:
                # prefix-enabled attention
                raise RuntimeError(
                    "IPEX backend doesn't support prefix decoding.")

        else:
            # Decoding run.
            max_seq_len = attn_metadata.max_decode_seq_len
            output = torch.empty_like(query)
            block_size = value_cache.shape[3]
            num_seqs, num_heads, head_size = query.shape
            max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                                  _PARTITION_SIZE)
            # NOTE(woosuk): We use a simple heuristic to decide whether to use
            # PagedAttention V1 or V2. If the number of partitions is 1, we use
            # V1 to avoid the overhead of reduction. Also, if the number of
            # sequences or heads is large, we use V1 since there is enough work
            # to parallelize.
            # TODO(woosuk): Tune this heuristic.
            # For context len > 8192, use V2 kernel to avoid shared memory
            # shortage.
            use_v1 = (max_seq_len <= 8192 and
                      (max_num_partitions == 1 or num_seqs * num_heads > 512))
            if use_v1:
                # Run PagedAttention V1.
                ipex_ops.paged_attention_v1(
                    output,
                    query,
                    key_cache,
                    value_cache,
                    self.num_kv_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens_tensor,
                    block_size,
                    max_seq_len,
                    self.alibi_slopes,
                    self.kv_cache_dtype,
                    k_scale,
                    v_scale,
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
                ipex_ops.paged_attention_v2(
                    output,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    self.num_kv_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens_tensor,
                    block_size,
                    max_seq_len,
                    self.alibi_slopes,
                    self.kv_cache_dtype,
                    k_scale,
                    v_scale,
                )

            # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> List[torch.Tensor]:
    attn_biases = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype, device=alibi_slopes.device)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        bias = bias[None, :] - bias[:, None]

        num_heads = alibi_slopes.shape[0]
        bias = bias[None, :].repeat((num_heads, 1, 1))
        bias.mul_(alibi_slopes[:, None, None])
        inf_mask = torch.empty(
            (1, seq_len, seq_len),
            dtype=bias.dtype,
            device=alibi_slopes.device).fill_(-torch.inf).triu_(diagonal=1)
        attn_biases.append((bias + inf_mask).to(dtype))

    return attn_biases


def _make_sliding_window_bias(
    seq_lens: List[int],
    window_size: Optional[int],
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    attn_biases = []
    for seq_len in seq_lens:
        tensor = torch.full(
            (1, seq_len, seq_len),
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
