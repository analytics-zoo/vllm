from abc import abstractmethod
from typing import Any, Dict, Optional, Set, Tuple

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.executor.utils import check_block_size_valid
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput
from vllm.engine.local_worker_utils import (LocalWorkerVllm, ResultHandler,
                                            WorkerMonitor)
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
import torch

from functools import partial

logger = init_logger(__name__)
import os

def get_xpu_device_type(x):
    if x.device.type != "xpu":
        return x.device.type
    name = torch.xpu.get_device_name(x.device.index)
    if name.startswith("Intel(R) Arc(TM) A"):
        return "arc"
    elif name.startswith("Intel(R) Arc(TM)"):
        return "mtl"
    elif name.startswith("Intel(R) Data Center GPU Flex"):
        return "flex"
    elif name.startswith("Intel(R) Data Center GPU Max"):
        return "pvc"
    else:
        return "others"
"""
To create a new worker, we probably needs to do the following:
1. Create the executor, the executor should manage workers...
2. It is used to execute the model on devices
Need to implement the following methods:
add_lora method
remove_lora method
list_loras method
check_health method
"""

def _create_worker(*args, **kwargs):
    # Import within worker process to avoid CUDA init issues
    from vllm.worker.worker import Worker
    return Worker(*args, **kwargs)

class SingleNodeXpuExecutor(ExecutorBase):
    """Python multiprocessing-based multi-GPU executor"""
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        print(f"Invoked into singlenodexpuexecutor")
        # TODO: change here
        import intel_extension_for_pytorch as ipex
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config

        self._init_executor()

    def _init_executor(self) -> None:
        # Create the parallel GPU workers.
        self._init_workers()

        # Profile the memory usage and initialize the cache.
        self._init_cache()


    # TODO: implement multi-card self-selection to select cards
    def _init_workers(self):
        # TODO: fix the CUDA issues
        world_size = self.parallel_config.tensor_parallel_size

        # Set CUDA_VISIBLE_DEVICES for the driver, inherited by workers
        # if "CUDA_VISIBLE_DEVICES" not in os.environ:
        #     set_cuda_visible_devices(range(world_size))

        # TODO: enable device count using torch.xpu api
        # from torch.cuda import device_count
        # assert world_size <= device_count(), (
        #     "please set tensor_parallel_size to less than max local gpu count")

        # Get a distributed_init_method
        # FIXME: we probably want to do something with the proxy?
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())

        if world_size == 1:
            self.workers = []
        else:
            result_handler = ResultHandler()
            self.workers = [
                LocalWorkerVllm(
                    result_handler,
                    partial(
                        _create_worker,
                        self.model_config,
                        self.parallel_config,
                        self.scheduler_config,
                        self.device_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        lora_config=self.lora_config,
                        kv_cache_dtype=self.cache_config.cache_dtype,
                    )) for rank in range(1, world_size)
            ]

            for worker in self.workers:
                worker.start()

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self._init_driver_worker_and_model(0, 0, distributed_init_method)


    def _init_driver_worker_and_model(self, rank: int, local_rank: int,
                                      distributed_init_method: str):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        # Initialize the driver worker with the Worker class.
        self.driver_worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )

        # self._run_workers("init_device")
        self._run_workers("init_model",
                          cupy_port=get_open_port()
                          if not self.model_config.enforce_eager else None)
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

    def _init_cache(self) -> None:
        # TODO: fix this _init_cache
        """Profiles the memory usage and initializes the KV cache.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.
        More details can be found in the
        :meth:`~vllm.worker.worker.Worker.profile_num_available_blocks` method
        from class :class:`~vllm.worker.Worker`.

        Afterwards, as there may be multiple workers,
        we take the minimum number of blocks across all workers
        to ensure this can be applied to all of them.

        Finally, the engine will initialize the KV cache
        with the calculated number of blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
            cache_dtype=self.cache_config.cache_dtype,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        # if self.cache_config.forced_num_gpu_blocks is not None:
        #     forced_num_gpu_blocks = self.cache_config.forced_num_gpu_blocks
        #     logger.info(f"Replacing profiled {num_gpu_blocks=} with "
        #                 f"{forced_num_gpu_blocks=}")
        #     num_gpu_blocks = forced_num_gpu_blocks

        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        check_block_size_valid(num_gpu_blocks, self.cache_config.block_size,
                               self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)
        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        self._run_workers("warm_up_model")


    def execute_model(self, *args, **kwargs) -> SamplerOutput:
        all_outputs = self._run_workers("execute_model",
                                        driver_args=args,
                                        driver_kwargs=kwargs)

        # Only the driver worker returns the sampling results.
        return all_outputs[0]

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        if not self.worker_monitor.is_alive():
            raise RuntimeError("Worker processes are not running")

    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        # Start the workers first.
        worker_outputs = [
            worker.execute_method(method, *args, **kwargs)
            for worker in self.workers
        ]

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        # Start the driver worker after all the ray workers.
        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*driver_args,
                                                    **driver_kwargs)

        # Get the results of the workers.
        return [driver_worker_output
                ] + [output.get() for output in worker_outputs]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "add_lora",
            lora_request=lora_request,
        )

    def list_loras(self) -> Set[int]:
        return self._run_workers("list_loras")


    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "remove_lora",
            lora_id=lora_id,
        )