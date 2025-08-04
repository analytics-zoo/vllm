# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import torch
import time

from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.awq import (AWQLinearMethod,
                                                         is_layer_skipped_awq)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
from vllm.model_executor.parameter import (ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization import get_quantization_config


MIN_IPEX_VERSION = "2.7.0"
ACTIVATION_SCHEMES = ["static", "dynamic"]

class IPEXConfig(QuantizationConfig):
    """INT8 quantization config class using IPEX for the CPU/XPU backend,
    including AWQ, GPTQ.
    """

    IPEX_QUANT_METHOD_MAP = {
        "awq": 1,
        "gptq": 0,
    }

    def __init__(
        self,
        method: str,
        weight_bits: int,
        group_size: int,
        modules_to_not_convert: Optional[list[str]] = None,
        desc_act: Optional[bool] = None,
        lm_head_quantized: Optional[bool] = None,
        is_checkpoint_fp8_serialized: bool = False,
    ) -> None:
        super().__init__()
        self.method = method
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.modules_to_not_convert = modules_to_not_convert or []
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = 32 // self.weight_bits
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if self.method not in ["awq", "gptq", "auto-round", "fp8"]:
            raise ValueError(f"IPEX quantization supports [awq, gptq, auto-round, fp8], "
                             f"but got {self.method}.")
        if is_checkpoint_fp8_serialized:
            self.quant_method = "fp8"
            print("Detected fp8 checkpoint. Please note that the "
                   "format is experimental and subject to change.")
        self.activation_scheme = "dynamic"

    def __repr__(self) -> str:
        return (f"IPEXConfig(method={self.method},"
                f"weight_bits={self.weight_bits}, "
                f"group_size={self.group_size})")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "ipex"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IPEXConfig":
        method = cls.get_from_keys(config, ["quant_method"]).lower()
        if method == "awq":
            weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
            group_size = cls.get_from_keys(config,
                                           ["q_group_size", "group_size"])
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None)
            return cls(method, weight_bits, group_size, modules_to_not_convert,
                       False, False)
        # otherwise for gptq
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        data_type = cls.get_from_keys_or(config, ["data_type"],
                                      default="int4")
        is_checkpoint_fp8_serialized = ("fp8" in data_type)

        desc_act = cls.get_from_keys_or(config, ["desc_act"], default=False)
        return cls(method, weight_bits, group_size, [], desc_act,
                   lm_head_quantized, is_checkpoint_fp8_serialized)

    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[QuantizationMethods]:
        if not current_platform.is_cpu() and not current_platform.is_xpu():
            return None

        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        if quant_method in ["awq", "gptq", "auto-round", "fp8"]:
            return cls.get_name()

        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["LinearMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.method == "awq":
                if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                    return UnquantizedLinearMethod()
                return IPEXAWQLinearMethod(self)
            if self.method == "gptq":
                return IPEXGPTQLinearMethod(self)
            if self.method == "auto-round" or self.method == "fp8":
                return IPEXAutoRoundLinearMethod(self)
        return None

class IPEXAutoRoundLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
            self,
            layer: torch.nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
    ):
        # maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        # WEIGHT
        weight_dtype = (torch.float8_e5m2
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)

        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=weight_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes),
                                 dtype=torch.float32),
                weight_loader=weight_loader,
            )
            scale[:] = torch.finfo(torch.float32).min
            set_weight_attrs(scale, {"scale_type": "weight_scale"})
            set_weight_attrs(scale, {"needs_scalar_to_array": True})
            layer.register_parameter("weight_scale", scale)
            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                scale = PerTensorScaleParameter(data=torch.empty(
                    1, dtype=torch.float32),
                    weight_loader=weight_loader)

                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "input_scale"})
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = layer.weight.data
        scale = layer.weight_scale.data

        output = torch.ops.vllm.fp8_gemm(x, False, weight, True, None, x.dtype, None, scale, bias, False)
        return output

class IPEXGPTQLinearMethod(GPTQLinearMethod):
    """GPTQ linear method using IPEX for the CPU/XPU backend.
    """

    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        bias = layer.bias if not layer.skip_bias_add else None

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
        # Using the compute dtype (lowp_mode) as INT8 to leverage instructions
        # with better performance.
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK

        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=self.quant_config.group_size,
        )
        layer.ipex_output_size = layer.qweight.shape[-1]
        g_idx = layer.g_idx if self.quant_config.desc_act else None
        layer.ipex_qlinear = ipex.llm.quantization.woq_linear. \
            IPEXWeightOnlyQuantizedLinear.from_weight(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            layer.qweight.size(0),
            layer.ipex_output_size,
            qconfig=qconfig,
            g_idx=g_idx,
            bias=bias,
            group_size=self.quant_config.group_size,
            quant_method=IPEXConfig.IPEX_QUANT_METHOD_MAP["gptq"]
        )

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size, ))


class IPEXAWQLinearMethod(AWQLinearMethod):
    """AWQ linear method using IPEX for the CPU/XPU backend.
    """

    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer=layer)

        bias = layer.bias if not layer.skip_bias_add else None

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

        # Using the compute dtype (lowp_mode) as INT8 to leverage instructions
        # with better performance.
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH

        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=self.quant_config.group_size,
        )

        layer.ipex_output_size = layer.qweight.size(
            1) * self.quant_config.pack_factor
        layer.ipex_qlinear = ipex.llm.quantization.woq_linear. \
            IPEXWeightOnlyQuantizedLinear.from_weight(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            layer.qweight.size(0),
            layer.ipex_output_size,
            qconfig=qconfig,
            bias=bias,
            group_size=self.quant_config.group_size,
            quant_method=IPEXConfig.IPEX_QUANT_METHOD_MAP["awq"]  # type: ignore
        )

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size, ))
