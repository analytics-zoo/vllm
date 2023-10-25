"""Try start the AsyncLLMEngine"""

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
import pytest
import asyncio

# This is the model to load for workers
MODEL_PATH = "YOUR_MODEL_PATH"
"""
1. Test to start a AsyncLLMEngine, to ensure that all goes well before start serving.
"""


@pytest.mark.asyncio
async def test_model_execution():
    # Let's build an engine_args
    engine_args = AsyncEngineArgs(model=MODEL_PATH,
                                  tokenizer=MODEL_PATH,
                                  tokenizer_mode='auto',
                                  trust_remote_code=False,
                                  download_dir=None,
                                  load_format='auto',
                                  dtype='auto',
                                  seed=0,
                                  max_model_len=None,
                                  worker_use_ray=False,
                                  pipeline_parallel_size=1,
                                  tensor_parallel_size=1,
                                  block_size=16,
                                  swap_space=16,
                                  gpu_memory_utilization=0.9,
                                  max_num_batched_tokens=None,
                                  max_num_seqs=256,
                                  disable_log_stats=False,
                                  revision=None,
                                  tokenizer_revision=None,
                                  quantization=None,
                                  engine_use_ray=False,
                                  disable_log_requests=True,
                                  max_log_len=None)
    # Start the engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    engine.start_background_loop()
    await asyncio.sleep(5)
