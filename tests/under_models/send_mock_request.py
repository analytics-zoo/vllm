"""Try sending a mocked request to the underlying model execute stage"""

from vllm.sequence import SequenceGroupMetadata
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData
from vllm.outputs import RequestOutput
from vllm.engine.arg_utils import AsyncEngineArgs
import pytest

from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, Type,
                    Union)
from functools import partial
import asyncio

# This is the model to load for workers
MODEL_PATH="/models/vicuna-7b/"


"""
1. Prepare a faked sequencegroup meta data
2. Start a mocked AsyncLLMEngine, and modify its step_async function
3. invoke the step_async function manually
"""

class UglyAsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def step_async(self) -> List[RequestOutput]:
        sampling_para = SamplingParams(n=2, best_of=5, temperature=0.8, top_p=0.95, max_tokens=7)
        seq_data = {}
        seq_data[0] = SequenceData(prompt_token_ids=[1, 3087, 8970, 338, 263])
        request_id = "cmpl-7bef75eaa4394a3d895b5508dd5f69f6"

        seq_group_meta_data = SequenceGroupMetadata(request_id=request_id, is_prompt=True, seq_data=seq_data, sampling_params=sampling_para, block_tables={})
        seq_group_meta_data_lists = [seq_group_meta_data]

        output = await self._run_workers_async(
            "execute_model",
            seq_group_metadata_list=seq_group_meta_data_lists,
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
        )
        print(output)

        # TODO: change this to real one
        return RequestOutput(request_id=request_id, prompt="", prompt_token_ids=[1, 3087, 8970, 338, 263], outputs=[], finished=False)

    async def step_async_multiple(self) -> List[RequestOutput]:
        seq_group_metadata_lists = []
        request_id_0= "cmpl-81e2b9767b5b47bca7e649482698d385"
        seq_data_0 = {0: SequenceData(prompt_token_ids=[1, 3087, 8970, 338, 263])}
        sampling_params_0 = SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=0.0, top_p=1.0, top_k=-1, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], ignore_eos=False, max_tokens=7, logprobs=None, skip_special_tokens=True)

        seq_group_metadata_lists.append(SequenceGroupMetadata(request_id_0, True, seq_data_0, sampling_params_0, {}))

        request_id_1 = "cmpl-81e2b9767b5b47bca7e649482698d385"
        seq_data_1 = {1: SequenceData(prompt_token_ids=[1, 3087, 8970, 338, 263])}
        sampling_params_1 = SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=0.0, top_p=1.0, top_k=-1, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], ignore_eos=False, max_tokens=7, logprobs=None, skip_special_tokens=True)

        seq_group_metadata_lists.append(SequenceGroupMetadata(request_id_1, True, seq_data_1, sampling_params_1, {}))

        output = await self._run_workers_async(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_lists,
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
        )

        # TODO: change this to real one
        return RequestOutput(request_id=request_id_0, prompt="", prompt_token_ids=[1, 3087, 8970, 338, 263], outputs=[], finished=False)


    async def _run_workers_async(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = await asyncio.gather(*all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output

setattr(AsyncLLMEngine, "_engine_class", UglyAsyncLLMEngine)


@pytest.mark.asyncio
async def test_model_execution():
    # Let's build an engine_args    
    engine_args = AsyncEngineArgs(model='/models/vicuna-7b/', tokenizer='/models/vicuna-7b/', tokenizer_mode='auto', trust_remote_code=False, download_dir=None, load_format='dummy', dtype='auto', seed=0, max_model_len=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, block_size=16, swap_space=16, gpu_memory_utilization=0.9, max_num_batched_tokens=None, max_num_seqs=256, disable_log_stats=False, revision=None, tokenizer_revision=None, quantization=None, engine_use_ray=False, disable_log_requests=True, max_log_len=None)
    # Start the engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    engine.start_background_loop()
    await asyncio.sleep(2)
    await engine.engine.step_async()
    # Now let's try something difficult
    await engine.engine.step_async_multiple()




