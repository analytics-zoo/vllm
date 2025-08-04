# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple
# from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
# from vllm.model_executor.layers.quantization.ipex_quant import MIN_IPEX_VERSION

from vllm.envs import VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT, VLLM_QUANTIZE_Q40_LIB
import ctypes

MIN_IPEX_VERSION = "2.5.0"
class SymInt4Config(QuantizationConfig):
    """SYM_INT4 quantization config class which uses IPEX kernel behind the scene...
    The weight will be quantized according to GPTQ setups...
    """
    def __init__(
        self,
    ) -> None:
        super().__init__()


    @classmethod
    def get_name(cls) -> str:
        return "sym_int4"


    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]


    @classmethod
    def get_min_capability(cls) -> int:
        # TODO: check if this will affect things...
        # May need to check platform xpu
        return -1


    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SymInt4Config":
        return cls()

    @classmethod
    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        if isinstance(layer, LinearBase):
            return SymInt4LinearMethod(self)
        else:
            return None



class SymInt4LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: SymInt4Config):
        self.quant_config = quant_config
        # Initialize the quant_config
        try:
            self.clib = ctypes.CDLL(VLLM_QUANTIZE_Q40_LIB)
        except OSError as e:
            raise RuntimeError(f"Failed to load required quantization lib at {VLLM_QUANTIZE_Q40_LIB}: {e}")
        self.clib.quantize_q4_0_to_qweight_and_scale.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.clib.quantize_q4_0_to_qweight_and_scale.restype = ctypes.c_size_t

    def ggml_quantize_tensor(self, weight: torch.Tensor, out_qweight: torch.Tensor, out_scale:torch.Tensor, out_features: int, in_features: int):
        # Convert src to float *
        # Currently, only handles dimension = 2
        assert(weight.dim()==2)
        assert out_qweight.shape == (out_features, in_features // 8)
        assert out_scale.shape == (out_features, in_features // 64)

        assert weight.dtype == torch.float32
        assert out_qweight.dtype == torch.int32
        assert out_scale.dtype == torch.float16

        assert(out_qweight.is_contiguous())
        assert(out_scale.is_contiguous())
        src = weight.data.data_ptr()
        src = ctypes.cast(src, ctypes.POINTER(ctypes.c_float))

        qweight = out_qweight.data.data_ptr()
        qweight = ctypes.cast(qweight, ctypes.POINTER(ctypes.c_int32))

        scale = out_scale.data.data_ptr()
        scale = ctypes.cast(scale, ctypes.POINTER(ctypes.c_uint16))
        self.clib.quantize_q4_0_to_qweight_and_scale(src, qweight, scale, out_features, in_features)
        out_qweight = out_qweight.transpose(0,1).contiguous()
        out_scale = out_scale.transpose(0,1).contiguous()
        return out_qweight, out_scale


    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight_dtype = params_dtype
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=weight_dtype,
            device="cpu" if VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT else None),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)


    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # The same with the GPTQ's linear method by IPEX
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size, ))


    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight.float()
        out_features = layer.weight.shape[0]
        in_features = layer.weight.shape[1]

        qweight = torch.zeros((out_features, in_features // 8), dtype=torch.int32, device=layer.weight.device)
        scale = torch.zeros((out_features, in_features // 64), dtype=torch.float16, device=layer.weight.device)
        qweight, scale = self.ggml_quantize_tensor(weight, qweight, scale, out_features, in_features)
        
        qweight = qweight.to("xpu")
        scale = scale.to("xpu")

        # Use qweight to replace weight...
        layer.weight = Parameter(qweight, requires_grad=False)
        # qweight_scale
        layer.weight_scale = Parameter(scale, requires_grad=False)
        # layer.input_scale = None
        try:
            import intel_extension_for_pytorch as ipex
            if ipex.__version__ < MIN_IPEX_VERSION:
                raise ImportError(
                    "intel_extension_for_pytorch version is "
                    "wrong. Please install "
                    f"intel_extension_for_pytorch>={MIN_IPEX_VERSION}.")
        except ImportError as err:
            raise ImportError(
                "Please install "
                f"intel_extension_for_pytorch>={MIN_IPEX_VERSION} via "
                f"`pip install intel_extension_for_pytorch>={MIN_IPEX_VERSION}`"
                " to use IPEX-AWQ linear method.") from err
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=64,
        )
        layer.ipex_output_size = layer.weight.shape[-1]
        g_idx = None
        layer.ipex_qlinear = ipex.llm.quantization.woq_linear. \
            IPEXWeightOnlyQuantizedLinear.from_weight(
            layer.weight,     # weight should be on xpu...
            layer.weight_scale,
            torch.tensor([8], device=layer.weight.device, dtype=torch.int8),
            layer.weight.size(0),
            layer.ipex_output_size,
            qconfig=qconfig,
            g_idx=g_idx,
            bias=None,
            group_size=64,
            # For GPTQ layout
            quant_method=0
        )
