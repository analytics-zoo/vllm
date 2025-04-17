# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_dp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils import direct_register_custom_op

fused_experts = None  # type: ignore
fused_moe_pallas = None  # type: ignore
if current_platform.is_xpu():
    from .moe_pallas import fused_moe_xpu
else:
    fused_moe_xpu = None  # type: ignore
logger = init_logger(__name__)

from ipex_llm.ggml.quantize import ggml_tensor_qtype, gguf_mixed_qtype
import ipex_llm.ggml.model.llama.llama_cpp as ggml
from ipex_llm.transformers.low_bit_linear import LowBitLinear, FP4Params, \
        FP16Linear, BF16Linear, ggml_convert_qtype, ggml_int4_convert_fp32

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.gguf import GGUFUninitializedParameter
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm.model_executor.layers.activation import SiluAndMul


import gguf
from gguf import GGMLQuantizationType as WeightType
import ipex_llm.ggml.model.llama.llama_cpp as ggml
from ipex_llm.ggml.quantize import ggml_tensor_qtype

from ipex_llm.transformers.low_bit_linear import MatMulLowBit
import xe_linear
import xe_batch

from vllm._ipex_ops import ipex_ops
import vllm._C.ops

# @CustomOp.register("unquantized_fused_moe")
class IPEXLLMFusedMoEMethod(FusedMoEMethodBase):
    """MoE method without quantization."""

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # w1: [num_experts, intermediate_size * 2, hidden_size]
        # w2: [num_experts, hidden_size, intermediate_size]
        self.num_experts = layer.w13_weight.data.shape[0]
        self.intermediate_size = layer.w2_weight.data.shape[2]
        self.hidden_size = layer.w13_weight.data.shape[2]

        # import pdb
        # pdb.set_trace()

        local_rank = os.environ["LOCAL_RANK"]
        self.device = torch.device(f"xpu:{local_rank}")
        lowbit = os.getenv("IPEX_LLM_LOWBIT", "sym_int4")
        qtype = ggml_tensor_qtype[lowbit]
        self.qtype = qtype

        w13_params = []
        for i in range(self.num_experts):
            cur_params = FP4Params(data=layer.w13_weight.data[i,:,:],
                                    requires_grad=False,
                                    quantized=False,
                                    _shape=None,
                                    convert_shape_only=False,
                                    qtype=self.qtype).to(self.device)
            w13_params.append(cur_params)
        layer._parameters['w13_weight'] = None
        self.qw1_weight = w13_params
        

        w2_params = []
        for i in range(self.num_experts):
            cur_params = FP4Params(data=layer.w2_weight.data[i,:,:],
                                    requires_grad=False,
                                    quantized=False,
                                    _shape=None,
                                    convert_shape_only=False,
                                    qtype=self.qtype).to(self.device)
            w2_params.append(cur_params)
        layer._parameters['w2_weight'] = None
        self.qw2_weight = w2_params

        w1 = self.qw1_weight
        self.w1_addrs = [expert.data_ptr() for expert in w1]
        self.w1_addrs = torch.tensor(self.w1_addrs, device=self.device, dtype=torch.uint64)
        w2 = self.qw2_weight
        self.w2_addrs = [expert.data_ptr() for expert in w2]
        self.w2_addrs = torch.tensor(self.w2_addrs, device=self.device, dtype=torch.uint64)
        logger.warning_once("model processed.")


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        return self.forward_xpu(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input)


    def forward_xpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        **kwargs,
    ):
        return self.fused_moe_xpu(hidden_states=x,
                                w1=layer.w13_weight,
                                w2=layer.w2_weight,
                                topk=top_k,
                                gating_output=router_logits,
                                global_num_experts=global_num_experts,
                                expert_map=expert_map,
                                renormalize=renormalize)


    def fused_moe_xpu(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        global_num_experts,
        expert_map,
        renormalize: bool,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [*, hidden_size]
            w1: [num_experts, intermediate_size * 2, hidden_size]
            w2: [num_experts, hidden_size, intermediate_size]
            gating_output: [*, num_experts]
        """
        orig_shape = hidden_states.shape
        hidden_size = hidden_states.shape[-1]
        num_tokens = hidden_states.shape[:-1].numel()
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        qtype = self.qtype

        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_states = hidden_states.view(num_tokens, hidden_size)
        gating_output = gating_output.view(num_tokens, global_num_experts)
        topk_weights, topk_indices = F.softmax(gating_output, dim=-1, dtype=torch.float).topk(topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)
        if expert_map is not None:
            expert_map = expert_map.to(device=device)
            topk_indices = expert_map[topk_indices]

        # w1: [num_experts, intermediate_size * 2, hidden_size]
        # [num_tokens, hidden_size] -> [num_experts, intermediate_size * 2] 

        # topk_indices: [bsz * seq_len, num_selected_experts]
        # hidden_states: [bsz * seq_len, hidden_size]
        # tmp: [bsz * seq_len, intermediate_size * 2]

        w1 = self.qw1_weight
        w2 = self.qw2_weight
        
        # tmp = vllm._C.ops.moe_forward(hidden_states, topk_indices, self.w1_addrs, hidden_size, intermediate_size, expert_size, intermediate_size * 2, qtype)


        topk_indices = topk_indices.flatten()
        topk_argsort_indices = topk_indices.argsort()
        topk_argsort_revert_indices = topk_argsort_indices.argsort()
        token_indices = torch.arange(num_tokens, device=device).repeat_interleave(topk)
        token_indices = token_indices[topk_argsort_indices]
        group_sizes = custom_histogram(topk_indices.to(torch.int32), 0, num_experts - 1)
        
        x = hidden_states[token_indices]
        
        x = custom_gmm(x, w1, group_sizes, intermediate_size * 2)
        # x = F.silu(x[..., :intermediate_size]) * x[..., intermediate_size:]

        output = torch.zeros((x.shape[0], intermediate_size), device=x.device, dtype=x.dtype)
        ipex_ops.silu_and_mul(output, x)
        x = output

        x = custom_gmm(x, w2, group_sizes, hidden_size)
        x = x[topk_argsort_revert_indices].reshape(-1, topk, hidden_size)

        x = x * topk_weights.unsqueeze_(dim=-1)
        x = x.sum(dim=-2)
        x = x.reshape(orig_shape)
        return x


def custom_histogram(indices, min, max):
    bin_counts = torch.histc(indices, bins=max - min + 1, min=min, max=max).to(torch.int32)
    return bin_counts

def custom_gmm(x, w, group_sizes, out_len):
    result = torch.zeros(
            (x.shape[0], out_len),
            dtype=x.dtype,
            device=x.device
        )
    start = 0
    i = 0
    
    lowbit = os.getenv("IPEX_LLM_LOWBIT", "sym_int4")
    qtype = ggml_tensor_qtype[lowbit]
    for end_index in group_sizes.tolist():
        if end_index > 0:
            end = start + end_index

            # result[start:end] = torch.matmul(x[start:end], w[i])
            
            # cur_x = x[start:end].contiguous()
            # cur_w = xe_linear.dequant(cur_x, w[i].contiguous(), qtype)
            # result[start:end] = torch.matmul(cur_x, cur_w.T)

            cur_x = x[start:end].contiguous()
            cur_w = w[i]
            # print(cur_x.shape, " ", cur_w.shape, " ", out_len)
            cur_res = xe_linear.forward_new(cur_x, cur_w, qtype, out_len)
            # print(cur_res.shape)
            result[start:end] = cur_res

            start = end
        i += 1
    return result
