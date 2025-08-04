# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    logger.warning("Import error msg: %s", e.msg)

from vllm.utils import direct_register_custom_op
import vllm._C.ops


def ipex_fp8_gemm(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    d: Optional[torch.Tensor],
    dtype: torch.dtype,
    a_scale_inv: Optional[torch.Tensor],
    b_scale_inv: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    acc: bool,
) -> torch.Tensor:
    return torch.ops.torch_ipex.fp8_gemm2(a, trans_a, b, trans_b, d, dtype,
                                          a_scale_inv, b_scale_inv, bias, acc)


def ipex_fp8_gemm_fake_(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    d: Optional[torch.Tensor],
    dtype: torch.dtype,
    a_scale_inv: Optional[torch.Tensor],
    b_scale_inv: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    acc: bool,
) -> torch.Tensor:
    dim_0 = a.size(1) if trans_a else a.size(0)
    dim_1 = b.size(0) if trans_b else b.size(1)
    return torch.zeros((dim_0, dim_1), device=a.device, dtype=a.dtype)


direct_register_custom_op(op_name="fp8_gemm",
                          op_func=ipex_fp8_gemm,
                          mutates_args=[],
                          fake_impl=ipex_fp8_gemm_fake_,
                          dispatch_key="XPU")


class ipex_ops:

    @staticmethod
    def _reshape_activation_tensor(
            x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num = x.size(0)
        d = x.size(1) // 2
        x = x.reshape(num, 2, d)
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = x1.reshape(num, d)
        x2 = x2.reshape(num, d)
        return x1, x2

    @staticmethod
    def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.silu_and_mul(x, out)

    @staticmethod
    def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.gelu_and_mul(x, out)

    @staticmethod
    def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.gelu_and_mul(x, out)

    @staticmethod
    def gelu_fast(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    @staticmethod
    def gelu_new(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    @staticmethod
    def gelu_quick(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.gelu_quick(x, out)

    @staticmethod
    def paged_attention_v1(
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None:
        assert kv_cache_dtype == "auto"
        num_heads = out.size(1)
        num_queries_per_tokens = num_heads // num_kv_heads
        ipex.llm.modules.PagedAttention.single_query_kv_attention(
            out,
            query.contiguous(),
            key_cache.view_as(value_cache),
            value_cache,
            num_queries_per_tokens,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )

    @staticmethod
    def paged_attention_v2(
        out: torch.Tensor,
        exp_sum: torch.Tensor,
        max_logits: torch.Tensor,
        tmp_out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None:
        assert kv_cache_dtype == "auto"
        num_heads = out.size(1)
        num_queries_per_tokens = num_heads // num_kv_heads
        ipex.llm.modules.PagedAttention.single_query_kv_attention(
            out,
            query.contiguous(),
            key_cache.view_as(value_cache),
            value_cache,
            num_queries_per_tokens,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )

    @staticmethod
    def rotary_embedding(
        positions: torch.Tensor,  # [batch_size, seq_len]
        query: torch.Tensor,  # [batch_size, seq_len, num_heads*head_size]
        key: torch.Tensor,  # [batch_size, seq_len, num_kv_heads*head_size]
        head_size: int,
        cos_sin_cache: torch.Tensor,  # [cos_sin_dim, rot_dim]
        is_neox: bool,
    ) -> None:
        # rot_dim = cos_sin_cache.size(1)
        # ipex.llm.functional.rotary_embedding_batched(positions, query, key,
        #                                              head_size, cos_sin_cache,
        #                                              is_neox, rot_dim)
        import vllm._C.ops
        vllm._C.ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)

    @staticmethod
    def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                                 key: torch.Tensor, head_size: int,
                                 cos_sin_cache: torch.Tensor, is_neox: bool,
                                 rot_dim: int,
                                 cos_sin_cache_offsets: torch.Tensor) -> None:
        ipex.llm.functional.rotary_embedding_batched(positions, query, key,
                                                     head_size, cos_sin_cache,
                                                     is_neox, rot_dim,
                                                     cos_sin_cache_offsets)

    @staticmethod
    def rms_norm(input: torch.Tensor, weight: torch.Tensor,
                 epsilon: float) -> torch.Tensor:
        # return ipex.llm.functional.rms_norm(input, weight, epsilon)
        tmp = torch.empty_like(input)
        vllm._C.ops.rms_norm(tmp, input, weight, epsilon)
        return tmp

    @staticmethod
    def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                           weight: torch.Tensor, epsilon: float) -> None:
        # tmp = ipex.llm.functional.add_rms_norm(residual, input, weight, None,
        #                                        epsilon, True)
        # input.copy_(tmp)
        vllm._C.ops.fused_add_rms_norm(input, residual, weight, epsilon)

    @staticmethod
    def varlen_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        seqlen_q: torch.Tensor,
        seqlen_k: torch.Tensor,
        alibi_slopes: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        pdropout: float,
        softmax_scale: float,
        zero_tensors: bool,
        is_causal: bool,
        return_softmax: bool,
        gen_: torch.Generator,
        window_size_left: float,
        window_size_right: float,
        logits_soft_cap: float,
    ) -> None:
        if ipex.__version__.endswith("cpu"):
            if logits_soft_cap != 0.0:
                raise ValueError("IPEX CPU does not support logits_soft_cap")
            assert alibi_slopes is None
            assert window_size_left < 0 and window_size_right < 0
            ipex.llm.functional.varlen_attention(query.contiguous(),
                                                 key.contiguous(),
                                                 value.contiguous(), out,
                                                 seqlen_q.int(),
                                                 seqlen_k.int(), max_seqlen_q,
                                                 max_seqlen_k, pdropout,
                                                 softmax_scale, zero_tensors,
                                                 is_causal, return_softmax,
                                                 gen_)
        else:  # XPU build
            ipex.llm.functional.varlen_attention(
                query.contiguous(), key.contiguous(), value.contiguous(), out,
                seqlen_q.int(), seqlen_k.int(), alibi_slopes, max_seqlen_q,
                max_seqlen_k, pdropout, softmax_scale, zero_tensors, is_causal,
                return_softmax, gen_, window_size_left, window_size_right,
                logits_soft_cap)

    @staticmethod
    def reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        assert kv_cache_dtype == "auto"
        ipex.llm.modules.PagedAttention.reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping)

    @staticmethod
    def reshape_and_cache_flash(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype,
            k_scale, v_scale)

    @staticmethod
    def chunked_prefill(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seq_used_k: Optional[torch.Tensor],
        block_table: torch.Tensor,
        alibi_slopes: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        p_dropout: float,
        softmax_scale: float,
        zero_tensors: bool,
        window_size_left: int,
        window_size_right: int,
        is_causal: bool,
        return_softmax: bool,
        gen_: Optional[torch.Generator],
        kv_cache_dtype: str,
    ):
        return ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
            output,
            query.contiguous(),
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            is_causal,
            block_table,
            alibi_slopes,
            kv_cache_dtype=kv_cache_dtype,  # "fp8"
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            k_scale=1.0,
            v_scale=1.0,
        )

    @staticmethod
    def copy_blocks(key_caches: list[torch.Tensor],
                    value_caches: list[torch.Tensor],
                    block_mapping: torch.Tensor) -> None:
        torch.xpu.copy_blocks(  # type: ignore
            key_caches,
            value_caches,
            block_mapping,
        )

    @staticmethod
    def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                    block_mapping: torch.Tensor) -> None:
        torch.xpu.swap_blocks(src, dst, block_mapping)  # type: ignore

    @staticmethod
    def bgmv_shrink(inputs: torch.Tensor,
                    lora_a_weights: torch.Tensor,
                    output_tensor: torch.Tensor,
                    lora_indices_tensor: torch.Tensor,
                    scaling: float = 1.0) -> None:
        ipex.llm.functional.bgmv_shrink(inputs, lora_a_weights, output_tensor,
                                        lora_indices_tensor, scaling)

    @staticmethod
    def bgmv_expand(inputs: torch.Tensor,
                    lora_b_weights: torch.Tensor,
                    output_tensor: torch.Tensor,
                    lora_indices_tensor: torch.Tensor,
                    add_inputs: bool = True) -> None:
        ipex.llm.functional.bgmv_expand(inputs, lora_b_weights, output_tensor,
                                        lora_indices_tensor, add_inputs)

    @staticmethod
    def bgmv_expand_slice(inputs: torch.Tensor,
                          lora_b_weights: torch.Tensor,
                          output_tensor: torch.Tensor,
                          lora_indices_tensor: torch.Tensor,
                          slice_offset: int,
                          slice_size: int,
                          add_inputs: bool = True) -> None:
        ipex.llm.functional.bgmv_expand_slice(inputs, lora_b_weights,
                                              output_tensor,
                                              lora_indices_tensor,
                                              slice_offset, slice_size,
                                              add_inputs)

    @staticmethod
    def sgmv_shrink(inputs: torch.Tensor,
                    lora_a_weights: torch.Tensor,
                    output_tensor: torch.Tensor,
                    b_seq_start_loc: torch.Tensor,
                    seq_len_tensor: torch.Tensor,
                    lora_indices_tensor: torch.Tensor,
                    batches: int,
                    max_seq_length: int,
                    token_nums: int,
                    scaling: float = 1.0) -> None:
        assert inputs.size(0) == token_nums
        ipex.llm.functional.sgmv_shrink(inputs, lora_a_weights, output_tensor,
                                        b_seq_start_loc, seq_len_tensor,
                                        lora_indices_tensor, batches,
                                        max_seq_length, scaling)

    @staticmethod
    def sgmv_expand(inputs: torch.Tensor,
                    lora_b_weights: torch.Tensor,
                    output_tensor: torch.Tensor,
                    b_seq_start_loc: torch.Tensor,
                    seq_len_tensor: torch.Tensor,
                    lora_indices_tensor: torch.Tensor,
                    batches: int,
                    max_seq_length: int,
                    token_nums: int,
                    add_inputs: bool = False) -> None:
        assert inputs.size(0) == token_nums
        ipex.llm.functional.sgmv_expand(inputs, lora_b_weights, output_tensor,
                                        b_seq_start_loc, seq_len_tensor,
                                        lora_indices_tensor, batches,
                                        max_seq_length, add_inputs)

    @staticmethod
    def sgmv_expand_slice(inputs: torch.Tensor,
                          lora_b_weights: torch.Tensor,
                          output_tensor: torch.Tensor,
                          b_seq_start_loc: torch.Tensor,
                          seq_len_tensor: torch.Tensor,
                          lora_indices_tensor: torch.Tensor,
                          batches: int,
                          max_seq_length: int,
                          token_nums: int,
                          slice_offset: int,
                          slice_size: int,
                          add_inputs: bool = False) -> None:
        assert inputs.size(0) == token_nums
        ipex.llm.functional.sgmv_expand_slice(inputs, lora_b_weights,
                                              output_tensor, b_seq_start_loc,
                                              seq_len_tensor,
                                              lora_indices_tensor, batches,
                                              max_seq_length, slice_offset,
                                              slice_size, add_inputs)

    # @staticmethod
    # def lora_expand(inputs: torch.Tensor,
    #                 lora_b_weights: List[torch.Tensor],
    #                 output_tensor: torch.Tensor,
    #                 token_lora_mapping: torch.Tensor,
    #                 token_indices_sorted_by_lora_ids: torch.Tensor,
    #                 num_tokens_per_lora: torch.Tensor,
    #                 lora_token_start_loc: torch.Tensor,
    #                 lora_ids: torch.Tensor,
    #                 offset_start: int = 0,
    #                 add_inputs: bool = False) -> None:
    #     ipex.llm.functional.lora_expand(inputs, lora_b_weights,
    #                                     output_tensor, token_lora_mapping,
    #                                     token_indices_sorted_by_lora_ids,
    #                                     num_tokens_per_lora, num_tokens_per_lora,
    #                                     lora_token_start_loc, lora_ids,
    #                                     offset_start, add_inputs)

    # @staticmethod
    # def lora_shrink(inputs: torch.Tensor,
    #                 lora_a_weights: List[torch.Tensor],
    #                 output_tensor: torch.Tensor,
    #                 token_lora_mapping: torch.Tensor,
    #                 token_indices_sorted_by_lora_ids: torch.Tensor,
    #                 num_tokens_per_lora: torch.Tensor,
    #                 lora_token_start_loc: torch.Tensor,
    #                 lora_ids: torch.Tensor,
    #                 scaling: float) -> None:
    #     ipex.llm.functional.lora_shrink(inputs, lora_a_weights,
    #                                     output_tensor, token_lora_mapping,
    #                                     token_indices_sorted_by_lora_ids,
    #                                     num_tokens_per_lora, num_tokens_per_lora,
    #                                     lora_token_start_loc, lora_ids,
    #                                     scaling)
