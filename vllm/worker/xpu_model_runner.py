# SPDX-License-Identifier: Apache-2.0

import dataclasses
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple,
                    Type, TypeVar)

import torch
import torch.nn as nn

from vllm.attention import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadataCache
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap,
                             MultiModalRegistry)
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import (
    DeviceMemoryProfiler, GiB_bytes, make_tensor_with_pad, PyObjectCache,
    is_pin_memory_available, flatten_2d_lists)
from vllm.worker.model_runner import AttentionMetadata, SamplingMetadata
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

LORA_WARMUP_RANK = 8

logger = init_logger(__name__)

_PAD_SLOT_ID = -1

TModelInputForXPU = TypeVar('TModelInputForXPU', bound="ModelInputForXPU")


@dataclass(frozen=True)
class ModelInputForXPU(ModelRunnerInputBase):
    """
    Used by the NeuronModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    virtual_engine: Optional[int] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    lora_mapping: Optional["LoRAMapping"] = None
    lora_requests: Optional[Set[LoRARequest]] = None
    async_callback: Optional[Callable] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)

        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForXPU],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForXPU:
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


@dataclass(frozen=True)
class ModelInputForXPUWithSamplingMetadata(ModelInputForXPU):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None
    is_prompt: Optional[bool] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForXPUWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class ModelInputForXPUBuilder(ModelRunnerInputBuilderBase[ModelInputForXPU]):
    class InterDataForSeqGroup:
        """Intermediate data for the current sequence group."""

        def simple_reinit(self):
            self.input_tokens[0].clear()  # type: ignore
            self.input_positions[0].clear()  # type: ignore
            self.token_types[0].clear()  # type: ignore
            self.mrope_input_positions = None  # type: ignore
            self.seq_lens[0] = 0  # type: ignore
            self.orig_seq_lens[0] = 0  # type: ignore
            self.query_lens[0] = 0  # type: ignore
            self.context_lens[0] = 0  # type: ignore
            self.curr_sliding_window_blocks[0] = 0  # type: ignore
            self.lora_index_mapping.clear()  # type: ignore
            self.lora_prompt_mapping.clear()  # type: ignore
            self.lora_requests.clear()  # type: ignore
            self.prompt_adapter_index_mapping.clear()  # type: ignore
            self.prompt_adapter_prompt_mapping.clear()  # type: ignore

        def __init__(
            self,
            *,
            # From sequence group metadata.
            request_id: str,
            seq_ids: List[int],
            is_prompt: bool,
            block_tables: Optional[Dict[int, List[int]]],
            computed_block_nums: List[int],
            n_seqs: int = 0,

            # Input tokens and positions.
            input_tokens: Optional[List[List[int]]] = None,
            input_positions: Optional[List[List[int]]] = None,
            token_types: Optional[List[List[int]]] = None,
            mrope_input_positions: Optional[List[List[List[int]]]] = None,

            # The sequence length (may be capped to the sliding window).
            seq_lens: Optional[List[int]] = None,
            # The original sequence length (before applying sliding window).
            # This is used to compute slot mapping.
            orig_seq_lens: Optional[List[int]] = None,
            # The query length.
            query_lens: Optional[List[int]] = None,
            # The number of tokens that are already computed.
            context_lens: Optional[List[int]] = None,
            # The current sliding window block.
            curr_sliding_window_blocks: Optional[List[int]] = None,

            # LoRA inputs.
            lora_index_mapping: Optional[List[List[int]]] = None,
            lora_prompt_mapping: Optional[List[List[int]]] = None,
            lora_requests: Optional[Set[LoRARequest]] = None,

            # Prompt adapter inputs.
            prompt_adapter_index_mapping: Optional[List[int]] = None,
            prompt_adapter_prompt_mapping: Optional[List[int]] = None,
            prompt_adapter_request: Optional[PromptAdapterRequest] = None,

            # Multi-modal inputs.
            multi_modal_kwargs: Optional[MultiModalKwargs] = None,
            multi_modal_placeholder_maps: Optional[Dict[
                str, MultiModalPlaceholderMap]] = None,

            # Whether the prefix cache is hit (prefill only).
            prefix_cache_hit: bool = False,
            reinit: bool = False,
            reinit_use_defaults: bool = False,
            encoder_seq_len: int = 0,
        ):
            if reinit:
                assert len(self.seq_ids) == len(seq_ids)  # type: ignore
                for i, seq_id in enumerate(seq_ids):
                    self.seq_ids[i] = seq_id  # type: ignore
            else:
                self.seq_ids = seq_ids

            self.request_id = request_id
            self.is_prompt = is_prompt
            self.block_tables = block_tables
            self.computed_block_nums = computed_block_nums
            self.n_seqs = n_seqs
            self.encoder_seq_len = encoder_seq_len

            if reinit:
                if len(self.seq_ids) == 1 and reinit_use_defaults:
                    self.simple_reinit()
                else:
                    if input_tokens:
                        self.input_tokens = input_tokens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.input_tokens[seq_id].clear()

                    if input_positions:
                        self.input_positions = input_positions
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.input_positions[seq_id].clear()

                    if token_types:
                        self.token_types = token_types
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.token_types[seq_id].clear()

                    self.mrope_input_positions = None

                    if seq_lens:
                        self.seq_lens = seq_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.seq_lens[seq_id] = 0

                    if orig_seq_lens:
                        self.orig_seq_lens = orig_seq_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.orig_seq_lens[seq_id] = 0

                    if query_lens:
                        self.query_lens = query_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.query_lens[seq_id] = 0

                    if context_lens:
                        self.context_lens = context_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.context_lens[seq_id] = 0

                    if curr_sliding_window_blocks:
                        self.curr_sliding_window_blocks = \
                            curr_sliding_window_blocks
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.curr_sliding_window_blocks[seq_id] = 0

                    if lora_index_mapping:
                        self.lora_index_mapping = lora_index_mapping
                    else:
                        self.lora_index_mapping.clear()

                    if lora_prompt_mapping:
                        self.lora_prompt_mapping = lora_prompt_mapping
                    else:
                        self.lora_prompt_mapping.clear()

                    if lora_requests:
                        self.lora_requests = lora_requests
                    else:
                        self.lora_requests.clear()

                    if prompt_adapter_index_mapping:
                        self.prompt_adapter_index_mapping = \
                            prompt_adapter_index_mapping
                    else:
                        self.prompt_adapter_index_mapping.clear()

                    if prompt_adapter_prompt_mapping:
                        self.prompt_adapter_prompt_mapping = \
                            prompt_adapter_prompt_mapping
                    else:
                        self.prompt_adapter_prompt_mapping.clear()

            else:
                self.input_tokens = input_tokens or []
                self.input_positions = input_positions or []
                self.token_types = token_types or []
                self.mrope_input_positions = mrope_input_positions or None
                self.seq_lens = seq_lens or []
                self.orig_seq_lens = orig_seq_lens or []
                self.query_lens = query_lens or []
                self.context_lens = context_lens or []
                self.curr_sliding_window_blocks = \
                    curr_sliding_window_blocks or []

                self.lora_index_mapping = lora_index_mapping or []
                self.lora_prompt_mapping = lora_prompt_mapping or []
                self.lora_requests = lora_requests or set()

                self.prompt_adapter_index_mapping = (
                    prompt_adapter_index_mapping or [])
                self.prompt_adapter_prompt_mapping = (
                    prompt_adapter_prompt_mapping or [])

            self.prompt_adapter_request = prompt_adapter_request
            self.multi_modal_kwargs = multi_modal_kwargs
            self.multi_modal_placeholder_maps = multi_modal_placeholder_maps
            self.prefix_cache_hit = prefix_cache_hit

            self.n_seqs = len(self.seq_ids)

            if not reinit:
                self.__post_init__()

        def __post_init__(self):
            self.n_seqs = len(self.seq_ids)

            self.input_tokens = [[] for _ in range(self.n_seqs)]
            self.input_positions = [[] for _ in range(self.n_seqs)]
            self.token_types = [[] for _ in range(self.n_seqs)]
            self.mrope_input_positions = None
            self.seq_lens = [0] * self.n_seqs
            self.orig_seq_lens = [0] * self.n_seqs
            self.query_lens = [0] * self.n_seqs
            self.context_lens = [0] * self.n_seqs
            self.curr_sliding_window_blocks = [0] * self.n_seqs

            self.lora_index_mapping = []
            self.lora_prompt_mapping = []

    def gen_inter_data_builder(self, num_seqs: int):
        return lambda: ModelInputForXPUBuilder.InterDataForSeqGroup(
            request_id="",
            seq_ids=[0] * num_seqs,
            is_prompt=True,
            block_tables=None,
            computed_block_nums=[])

    def init_cached_inter_data(self, *args, **kwargs):
        assert len(args) == 0
        assert "seq_ids" in kwargs
        seq_ids = kwargs["seq_ids"]
        num_seqs = len(seq_ids)

        # The inter-data cache is per model_runner
        inter_data_cache = self.runner.inter_data_cache
        if num_seqs not in inter_data_cache:
            inter_data_cache[num_seqs] = PyObjectCache(
                self.gen_inter_data_builder(num_seqs))

        obj = inter_data_cache[num_seqs].get_object()
        obj.__init__(*args, **kwargs)
        return obj

    def reset_cached_inter_data(self):
        for cache in self.runner.inter_data_cache.values():
            cache.reset()

    def __init__(self,
                 runner: "XPUModelRunner",
                 finished_requests_ids: Optional[List[str]] = None) -> None:
        super().__init__()
        self.runner = runner
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.device = self.runner.device
        self.enable_lora = self.runner.lora_config is not None
        self.scheduler_config = self.runner.scheduler_config
        self.chunked_prefill_enabled = (
            self.scheduler_config is not None
            and self.scheduler_config.chunked_prefill_enabled)

    def prepare(self,
                finished_requests_ids: Optional[List[str]] = None) -> None:
        self.seq_group_metadata_list: List[SequenceGroupMetadata] = []
        self.decode_only = True
        # Intermediate data (data in CPU before going to GPU) for
        # the current sequence group.
        self.inter_data_list: List[
            ModelInputForXPUBuilder.InterDataForSeqGroup] = []

    def _compute_lens(self, inter_data: InterDataForSeqGroup, seq_idx: int,
                      seq_group_metadata: SequenceGroupMetadata):
        """Compute context length, sequence length and tokens
        for the given sequence data.
        """
        seq_data = seq_group_metadata.seq_data[inter_data.seq_ids[seq_idx]]
        token_chunk_size = seq_group_metadata.token_chunk_size

        # Compute context length (the number of tokens that are
        # already computed) and sequence length (total number of tokens).

        seq_len = seq_data.get_len()
        if inter_data.is_prompt:
            context_len = seq_data.get_num_computed_tokens()
            seq_len = min(seq_len, context_len + token_chunk_size)
        elif self.runner.scheduler_config.is_multi_step or \
            self.runner.model_config.is_encoder_decoder:
            context_len = seq_len - 1
        else:
            context_len = seq_data.get_num_computed_tokens()

        # Compute tokens.
        tokens = seq_data.get_token_ids()[context_len:seq_len]
        token_types = seq_group_metadata.token_type_ids

        inter_data.seq_lens[seq_idx] = seq_len
        inter_data.orig_seq_lens[seq_idx] = seq_len
        inter_data.context_lens[seq_idx] = context_len
        inter_data.input_tokens[seq_idx].extend(tokens)
        inter_data.input_positions[seq_idx].extend(range(context_len, seq_len))
        inter_data.token_types[seq_idx].extend(
            token_types if token_types else [])
        inter_data.query_lens[seq_idx] = seq_len - context_len

        if seq_data.mrope_position_delta is not None:
            if inter_data.mrope_input_positions is None:
                inter_data.mrope_input_positions = [None] * inter_data.n_seqs

            inter_data.mrope_input_positions[
                seq_idx] = MRotaryEmbedding.get_next_input_positions(
                    seq_data.mrope_position_delta,
                    context_len,
                    seq_len,
                )

    def _compute_lora_input(self, inter_data: InterDataForSeqGroup,
                            seq_idx: int,
                            seq_group_metadata: SequenceGroupMetadata):
        """If LoRA is enabled, compute LoRA index and prompt mapping."""
        if not self.enable_lora:
            return

        lora_id = seq_group_metadata.lora_int_id
        if lora_id > 0:
            inter_data.lora_requests.add(seq_group_metadata.lora_request)
        query_len = inter_data.query_lens[seq_idx]
        inter_data.lora_index_mapping.append([lora_id] * query_len)
        sampling_params = seq_group_metadata.sampling_params
        if sampling_params and sampling_params.prompt_logprobs is not None:
            inter_data.lora_prompt_mapping.append([lora_id] * query_len)
        elif not self.chunked_prefill_enabled or seq_group_metadata.do_sample:
            inter_data.lora_prompt_mapping.append([lora_id])
        else:
            inter_data.lora_prompt_mapping.append([])

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        seq_ids = seq_group_metadata.seq_data.keys()
        n_seqs = len(seq_ids)
        is_prompt = seq_group_metadata.is_prompt

        if is_prompt:
            assert n_seqs == 1
            self.decode_only = False

        encoder_seq_len = 0

        if self.runner.model_config.is_encoder_decoder:
            encoder_seq_len = seq_group_metadata.encoder_seq_data.get_len()

        inter_data = self.init_cached_inter_data(
            request_id=seq_group_metadata.request_id,
            seq_ids=seq_ids,
            is_prompt=is_prompt,
            block_tables=seq_group_metadata.block_tables,
            computed_block_nums=seq_group_metadata.computed_block_nums,
            reinit=True,
            reinit_use_defaults=True,
            encoder_seq_len=encoder_seq_len)

        self.inter_data_list.append(inter_data)

        for seq_idx in range(n_seqs):
            self._compute_lens(inter_data, seq_idx, seq_group_metadata)
            self._compute_lora_input(inter_data, seq_idx, seq_group_metadata)

        self.seq_group_metadata_list.append(seq_group_metadata)

    def build(self) -> ModelInputForXPU:
        # LoRA data.
        lora_requests = set()
        lora_mapping = None
        if self.enable_lora:
            lora_requests = set(r for data in self.inter_data_list
                                for r in data.lora_requests)
            lora_index_mapping = flatten_2d_lists([
                flatten_2d_lists(inter_data.lora_index_mapping)
                for inter_data in self.inter_data_list
            ])
            lora_prompt_mapping = flatten_2d_lists([
                flatten_2d_lists(inter_data.lora_prompt_mapping)
                for inter_data in self.inter_data_list
            ])

            lora_mapping = LoRAMapping(
                **dict(index_mapping=lora_index_mapping,
                       prompt_mapping=lora_prompt_mapping,
                       is_prefill=not self.decode_only))

        is_prompt = self.seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, attn_metadata, seq_lens,
             multi_modal_kwargs) = self._prepare_prompt(
                 self.seq_group_metadata_list)
        else:
            (input_tokens, input_positions,
             attn_metadata) = self._prepare_decode(
                 self.seq_group_metadata_list)
            seq_lens = None
            multi_modal_kwargs = None

        return self.model_input_cls(
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
            seq_lens=seq_lens,
            query_lens=seq_lens,
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
        )

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, List[int],
               BatchedTensorInputs]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        multi_modal_kwargs_list: List[MultiModalKwargs] = []
        multi_modal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            computed_len = seq_data.get_num_computed_tokens()
            seq_len = len(prompt_tokens)

            seq_lens.append(seq_len)  # Prompt token num
            input_tokens.extend(prompt_tokens)  # Token ids

            # Token position ids
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            positions_range = range(computed_len, seq_len)
            input_positions.extend(list(positions_range))

            if seq_group_metadata.multi_modal_data:
                # NOTE: mm_kwargs only includes the subset of multi-modal items
                # that intersect with the current prefill positions.
                mm_kwargs, placeholder_maps = MultiModalPlaceholderMap \
                    .from_seq_group(seq_group_metadata, positions_range)

                multi_modal_kwargs_list.append(mm_kwargs)

                for modality, placeholder_map in placeholder_maps.items():
                    multi_modal_placeholder_maps[modality].extend(
                        placeholder_map)

            if seq_group_metadata.block_tables is None or seq_group_metadata.block_tables[seq_id] is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, seq_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, seq_len - self.sliding_window)

            for i in range(computed_len, seq_len):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i //
                                           self.block_size]  # type: ignore
                block_offset = i % self.block_size  # type: ignore
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        num_prompt_tokens = len(input_tokens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)  # type: ignore
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            multi_modal_placeholder_maps.items()
        }

        max_seqlen = max(seq_lens)
        tmp = [0]
        tmp.extend(seq_lens)
        seqlen = torch.tensor(tmp)
        seqlen_q = torch.cumsum(seqlen, dim=0).to(device=self.device)
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)
        # if "reranker" in self.runner.model_config.model.lower():
        #     seq_lens_tensor = torch.tensor(seq_lens,
        #                                    dtype=torch.int,
        #                                    device=self.device)
        # else:
        #     seq_lens_tensor = torch.tensor([])

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=False,
            seq_lens=seq_lens,
            seqlen_q=seqlen_q,
            max_seqlen=max_seqlen,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=0,
            num_prefills=len(seq_lens),
            num_prefill_tokens=num_prompt_tokens,
            num_decode_tokens=0,
            block_tables=torch.tensor([], device=self.device, dtype=torch.int),
        )

        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        return (input_tokens, input_positions, attn_metadata, seq_lens,
                multi_modal_kwargs)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        block_tables: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        max_decode_seq_len = max(seq_lens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)

        block_tables = make_tensor_with_pad(
            block_tables,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            seq_lens=seq_lens,
            seqlen_q=torch.tensor([]),
            max_seqlen=0,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=max_decode_seq_len,
            num_prefill_tokens=0,
            num_decode_tokens=len(input_tokens),
            num_prefills=0,
            block_tables=block_tables,
        )
        return (
            input_tokens,
            input_positions,
            attn_metadata,
        )

# Add XPUModelRUnnerBase to support embedding models
class XPUModelRunnerBase(ModelRunnerBase[TModelInputForXPU]):

    _model_input_cls: Type[TModelInputForXPU]
    _builder_cls: Type[ModelInputForXPUBuilder]

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):

        ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        model_config = self.model_config
        cache_config = self.cache_config
        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size

        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        )

        # Multi-modal data support
        self.input_registry = input_registry
        self.mm_registry = mm_registry

        # Lazy initialization.
        self.model: nn.Module  # Set after init_Model
        self.sampler = get_sampler()
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None

        # Used to cache python objects
        self.inter_data_cache: Dict[int, PyObjectCache] = {}

        self.sampling_metadata_cache: SamplingMetadataCache = \
              SamplingMetadataCache() \
                if self.parallel_config.pipeline_parallel_size == 1 else None

        self.builder = self._builder_cls(weakref.proxy(self))

    def load_model(self) -> None:
        with DeviceMemoryProfiler() as m:
            self.model = get_model(vllm_config=self.vllm_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GiB",
                    self.model_memory_usage / GiB_bytes)

        if self.lora_config:
            assert supports_lora(self.model), "Model does not support LoRA"
            assert not supports_multimodal(
                self.model
            ), "To be tested: Multi-modal model with LoRA settings."

            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.vocab_size,
                self.lora_config,
                self.device,
                self.model.embedding_modules,
                self.model.embedding_padding_modules,
                max_position_embeddings=self.model.config.
                max_position_embeddings,
            )
            self.model = self.lora_manager.create_lora_manager(self.model)

    def get_model(self) -> nn.Module:
        return self.model

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for multi-modal encoding, which
        # needs to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
            self.model_config)
        if max_mm_tokens > 0:
            max_num_seqs_orig = max_num_seqs
            max_num_seqs = min(max_num_seqs,
                               max_num_batched_tokens // max_mm_tokens)
            if max_num_seqs < 1:
                expr = (f"min({max_num_seqs_orig}, "
                        f"{max_num_batched_tokens} // {max_mm_tokens})")
                logger.warning(
                    "Computed max_num_seqs (%s) to be less than 1. "
                    "Setting it to the minimum value of 1.", expr)
                max_num_seqs = 1

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            dummy_data = self.input_registry \
                .dummy_data_for_profiling(self.model_config,
                                          seq_len,
                                          self.mm_registry)

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: dummy_data.seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=dummy_data.multi_modal_data,
                multi_modal_placeholders=dummy_data.multi_modal_placeholders)
            seqs.append(seq)

        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=batch_size,
                dtype=self.model_config.dtype,
                device=self.device)
        self.execute_model(model_input, None, intermediate_tensors)
        torch.xpu.synchronize()
        return

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForXPUWithSamplingMetadata:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        """
        builder = self.builder
        builder.prepare(finished_requests_ids)
        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)

        self.builder.reset_cached_inter_data()

        return builder.build()  # type: ignore


class XPUModelRunner(XPUModelRunnerBase[ModelInputForXPUWithSamplingMetadata]):

    _model_input_cls: Type[ModelInputForXPUWithSamplingMetadata] = (
        ModelInputForXPUWithSamplingMetadata)
    _builder_cls: Type[ModelInputForXPUBuilder] = ModelInputForXPUBuilder

    def make_model_input_from_broadcasted_tensor_dict(
            self,
            tensor_dict: Dict[str,
                              Any]) -> ModelInputForXPUWithSamplingMetadata:
        return (
            ModelInputForXPUWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            ))

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForXPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        # Sampling metadata is only required for the final pp group
        generators = self.get_generators(finished_requests_ids)
        if get_pp_group().is_last_rank:
            # Sampling metadata is only required for the final pp group
            generators = self.get_generators(finished_requests_ids)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, model_input.seq_lens,
                model_input.query_lens, self.device, self.pin_memory,
                generators, self.sampling_metadata_cache)
        else:
            sampling_metadata = None
        is_prompt = (seq_group_metadata_list[0].is_prompt
             if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt,
                                   virtual_engine=virtual_engine)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForXPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "XPUModelRunner does not support multi-step execution.")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        model_executable = self.model
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start_time = time.time()
        with set_forward_context(model_input.attn_metadata, self.vllm_config,
                                 model_input.virtual_engine):
            hidden_or_intermediate_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                intermediate_tensors=intermediate_tensors,
                **MultiModalKwargs.as_kwargs(
                    model_input.multi_modal_kwargs or {},
                    dtype=self.model_config.dtype,
                    device=self.device,
                ),
            )
        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end_time = time.time()

        # Compute the logits.
        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.sampler(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time
                and output is not None):
            model_forward_time = (model_forward_end_time -
                                  model_forward_start_time)
            # If there are multiple workers, we are still tracking the latency
            # from the start time of the driver worker to the end time of the
            # driver worker. The model forward time will then end up covering
            # the communication time as well.
            output.model_forward_time = model_forward_time

        return [output]

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_adapter(lora_request)
