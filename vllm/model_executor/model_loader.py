"""Utilities for selecting and loading models."""
import contextlib
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import DeviceConfig, ModelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.weight_utils import (get_quant_config,
                                              initialize_dummy_weights)
from vllm.utils import is_xpu


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(model_config: ModelConfig,
              device_config: DeviceConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    # Get the (maybe quantized) linear method.
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config.quantization,
                                        model_config.model,
                                        model_config.hf_config,
                                        model_config.download_dir)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")
        linear_method = quant_config.get_linear_method()
    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device("cpu"):
            model = model_class(model_config.hf_config, linear_method)
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format, model_config.revision)
    model = model.eval()

    if is_xpu():
        import intel_extension_for_pytorch as ipex
        # model = ipex.optimize(model)
        from bigdl.llm import optimize_model
        # print(model)
        # input("pause")
        optimize_model(model)
        # print("optimized ***********************************")
        # print(model)
        model = model.to(device=device_config.device, dtype=model_config.dtype)
        # import gc
        # gc.collect()
        # torch.xpu.empty_cache()
        # input("pause")
    return model
