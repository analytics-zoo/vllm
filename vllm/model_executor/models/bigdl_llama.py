import torch
from torch import nn

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, LlamaConfig
from typing import Optional, Tuple, List, Type, Dict

from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
from vllm.model_executor.quantization_utils import QuantizationConfig
from vllm.sequence import SamplerOutput, SequenceOutputs
import math

import pdb

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    # if repetition_penalty > 1.0:
    #     processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list

class BigDLLlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        # pdb.set_trace()
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(config._name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = self.model.config.torch_dtype

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def forward(
        self, seq_group_meta_data_lists, kv_cache: Optional = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        kv_cache_0 = self.model.config.num_hidden_layers
        kv_cache_1 = 2
        bigdl_kv_cache = [[torch.tensor([], device=self.device, dtype = self.dtype) for _ in range(kv_cache_1)] for _ in range(kv_cache_0)]
        seq_len = len(seq_group_meta_data_lists)
        # for i in range(seq_len):
        #     if kv_cache.get(i) is None:
        #         kv_cache[i] = bigdl_kv_cache[:]

        bigdl_input_ids = []
        bigdl_position_ids = []
        cur_seq_ids = []
        bigdl_sampling_params = {}
        
        all_decoding = True
        for seq_group_meta_data in seq_group_meta_data_lists:
            req_id = seq_group_meta_data.request_id
            all_decoding = all_decoding and (not seq_group_meta_data.is_prompt)
            seq_ids = list(seq_group_meta_data.seq_data.keys())
            seq_id = seq_ids[0]
            print(seq_id)
            cur_seq_ids.append(seq_id)
            seq_data = seq_group_meta_data.seq_data[seq_id]
            
            cur_seq_input_ids = seq_data.get_token_ids()
            bigdl_input_ids.append(cur_seq_input_ids)
            
            bigdl_sampling_params[seq_id] = seq_group_meta_data.sampling_params

            context_len = seq_data.get_len()
            bigdl_position_ids.append(range(context_len))

        if all_decoding: 
            # pdb.set_trace()
            for seq_group_meta_data in seq_group_meta_data_lists:
                seq_ids = list(seq_group_meta_data.seq_data.keys())
                seq_id = seq_ids[0]
                for i in range(kv_cache_0):
                    for j in range(kv_cache_1):
                        bigdl_kv_cache[i][j] = torch.cat((bigdl_kv_cache[i][j], kv_cache[seq_id][i][j]), dim=0).to(dtype = self.dtype)
            
        bigdl_input_ids = torch.tensor(bigdl_input_ids, device=self.device)
        bigdl_position_ids = torch.tensor(bigdl_position_ids, device=self.device)
        if all_decoding:
            kwargs = {
                        "input_ids": bigdl_input_ids,
                        "position_ids": bigdl_position_ids,
                        "past_key_values": bigdl_kv_cache,
                        "use_cache": True,
                        "return_dict": True,
                    }
        else:
            kwargs = {
                        "input_ids": bigdl_input_ids,
                        "position_ids": bigdl_position_ids,
                        "past_key_values": None,
                        "use_cache": True,
                        "return_dict": True,
                    }
        # kwargs["position_ids"] = position_ids
        # pdb.set_trace()
        outputs = self.model.forward(**kwargs)
        index = 0
        bigdl_output = []
        for seq_id in cur_seq_ids:
            cur_sampling_params = bigdl_sampling_params[seq_id]
            logits_processor = prepare_logits_processor(
                cur_sampling_params.temperature, 1,
                cur_sampling_params.top_p, cur_sampling_params.top_k
            )
    
            last_token_logits = logits_processor(None, outputs.logits[index:index+1, -1, :])[0]
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]

            logprobs = math.log(probs[tokens[0]])
            seq_output = SequenceOutputs(
                parent_seq_id = seq_id,
                output_token = tokens[0],
                logprobs = {tokens[0]: logprobs}
            )
            bigdl_output.append([seq_output])
            if kv_cache.get(seq_id) is None:
                kv_cache[seq_id] = [[[] for _ in range(kv_cache_1)] for _ in range(kv_cache_0)]
            for i in range(kv_cache_0):
                for j in range(kv_cache_1):
                    kv_cache[seq_id][i][j] = outputs.past_key_values[i][j][index].unsqueeze(0).to(device=self.device,dtype = self.dtype)
            index = index + 1

            #pdb.set_trace()

        return bigdl_output

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        pass
