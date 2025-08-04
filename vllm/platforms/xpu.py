# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional

import os
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils import DEFAULT_MAX_NUM_BATCHED_TOKENS

from .interface import DeviceCapability, Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None

logger = init_logger(__name__)

def device_id_to_physical_device_id(device_id: int) -> int:
    if "ZE_AFFINITY_MASK" in os.environ:
        device_ids = os.environ["ZE_AFFINITY_MASK"].split(",")
        if device_ids == [""]:
            msg = (
                "ZE_AFFINITY_MASK is set to empty string, which means"
                " GPU support is disabled. If you are using ray, please unset"
                " the environment variable `ZE_AFFINITY_MASK` inside the"
                " worker/actor. "
                "Check https://github.com/vllm-project/vllm/issues/8402 for"
                " more information.")
            raise RuntimeError(msg)
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    # Intel XPU's device key is "GPU" for Ray.
    # see https://github.com/ray-project/ray/blob/6a5eb5865eeb9ccf058a79b44f107e327e360673/python/ray/_private/accelerators/intel_gpu.py#L20 # noqa: E501
    ray_device_key: str = "GPU"
    device_control_env_var: str = "ZE_AFFINITY_MASK"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        if selected_backend not in [_Backend.IPEX, _Backend.IPEX_V1]:
            logger.warning_once(
                f"Cannot use {selected_backend} backend on XPU.")
        use_v1 = envs.VLLM_USE_V1
        if use_v1:
            if selected_backend == _Backend.IPEX:
                logger.warning_once("For v1 on XPU, should use "
                                    "IPEX_V1 attention backend.")
            logger.info_once("Using IPEX_V1 attention backend.")
            return "vllm.v1.attention.backends.ipex_attn.IPEXAttentionBackend"
        else:
            if selected_backend == _Backend.IPEX:
                logger.warning_once("For v0 on XPU, should use "
                                    "IPEX attention backend.")
            logger.info_once("Using IPEX attention backend.")
            return "vllm.attention.backends.ipex_attn.IpexAttnBackend"

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> Optional[DeviceCapability]:
        # capacity format differs from cuda's and will cause unexpected
        # failure, so use None directly
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.xpu.get_device_name(device_id)

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.xpu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @classmethod
    def get_piecewise_backend_cls(cls) -> str:
        return "vllm.compilation.cuda_piecewise_backend.CUDAPiecewiseBackend"  # noqa

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            if envs.VLLM_USE_V1:
                cache_config.block_size = 64
            else:
                cache_config.block_size = 16

        # check and update model config
        model_config = vllm_config.model_config
        if not model_config.enforce_eager:
            logger.warning(
                "CUDA graph is not supported on XPU, fallback to the eager "
                "mode.")
            model_config.enforce_eager = True

        if vllm_config.device_config is not None:
            assert vllm_config.device_config.device_type == "xpu"

        # check and update parallel config
        parallel_config = vllm_config.parallel_config
        if vllm_config.speculative_config:
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = \
                    "vllm.v1.worker.xpu_worker.XPUWorker"
            else:
                raise NotImplementedError(
                    "XPU v0 does not support speculative decoding")
        else:
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls =\
                    "vllm.v1.worker.xpu_worker.XPUWorker"
            else:
                parallel_config.worker_cls = "vllm.worker.xpu_worker.XPUWorker"

        if parallel_config.distributed_executor_backend is None:
            if parallel_config.world_size > 1:
                parallel_config.distributed_executor_backend = "ray"
            else:
                parallel_config.distributed_executor_backend = "uni"
        elif parallel_config.distributed_executor_backend == "mp":
            # FIXME(kunshang):
            # spawn needs calling `if __name__ == '__main__':``
            # fork is not supported for xpu start new process.
            logger.warning(
                "Please use spawn as start method if you want to use mp.")
        elif parallel_config.distributed_executor_backend != "ray" and \
                parallel_config.distributed_executor_backend != "uni":
            logger.warning(
                "%s is not supported on XPU, fallback to ray distributed"
                " executor backend.",
                parallel_config.distributed_executor_backend)
            parallel_config.distributed_executor_backend = "ray"

        if vllm_config.model_config and vllm_config.model_config.use_mla:
            logger.info(
                "MLA is enabled on a non-GPU platform; forcing chunked "
                "prefill and prefix caching to be disabled.")
            vllm_config.scheduler_config.enable_chunked_prefill = False
            vllm_config.scheduler_config.chunked_prefill_enabled = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.scheduler_config.max_model_len,
                DEFAULT_MAX_NUM_BATCHED_TOKENS)

    @classmethod
    def is_pin_memory_available(cls):
        return True

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.xpu.reset_peak_memory_stats(device)
        return torch.xpu.max_memory_allocated(device)

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if envs.VLLM_XPU_FP8_DTYPE == "e4m3":
            return torch.float8_e4m3fn
        else:
            return torch.float8_e5m2

    @classmethod
    def is_data_center_gpu(cls) -> bool:
        device_name = cls.get_device_name().lower()
        return device_name.count("data center gpu") > 0

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.xpu_communicator.XpuCommunicator"  # noqa

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        return True

    @classmethod
    def device_count(cls) -> int:
        return torch.xpu.device_count()
