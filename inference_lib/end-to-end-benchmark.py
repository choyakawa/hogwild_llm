import datetime
import json
from pathlib import Path

import numpy as np
import transformers

from typing import Optional, Tuple, Unpack, Callable, Dict, Any, List, Union

from transformers import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig
from torch import nn
from dataclasses import dataclass
import time
import torch
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

torch.ops.load_library("libhogatt.so")


@torch.compiler.disable
def hogwild_sdpa(queries: torch.Tensor, locations: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor],
                 scale: float, fragment_lengths=None, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty((queries.size(1), queries.size(2), queries.size(3), values[0].size(3)), dtype=queries.dtype,
                          device=queries.device)
    if fragment_lengths is None:
        fragment_lengths = torch.tensor([k.size(2) for k in keys], dtype=torch.int32, device=queries.device)
    keys = [k.contiguous() for k in keys]
    values = [v.contiguous() for v in values]
    torch.ops.libhogatt.hogwild_sdpa(out, scale, locations, queries.contiguous(), fragment_lengths, keys, values)
    return out


def hogwild_rope(queries: torch.Tensor, cosines: torch.Tensor, sines: torch.Tensor, out=None):
    if out is None:
        out = torch.empty((cosines.size(0), queries.size(0), queries.size(1), queries.size(2), queries.size(3)),
                          dtype=queries.dtype, device=queries.device)
    torch.ops.libhogatt.hogwild_rope(out, queries, cosines, sines)
    return out


def hogwild_fused(queries: torch.Tensor, locations: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor],
                  scale: float, fragment_lengths, cosines: torch.Tensor, sines: torch.Tensor, *, rotated_queries, out):
    torch.ops.libhogatt.hogwild_fused(out, rotated_queries, scale, locations, queries.contiguous(), fragment_lengths, keys, values, cosines, sines)
    return out


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot_emb = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat((x_rot_emb, x_pass), dim=-1).to(x.dtype)


@dataclass
class InternalCacheMeta:
    cos: list[torch.Tensor] | list[None] | torch.Tensor
    sin: list[torch.Tensor] | list[None] | torch.Tensor
    loc: list[torch.Tensor] | list[None] | torch.Tensor
    cs: Cache = None


@dataclass
class CacheStructure:
    keys: list[torch.Tensor]  # keys of the fragment
    values: list[torch.Tensor]  # values of this fragment
    frags: torch.Tensor  # fragment lengths
    cos: torch.Tensor  # cosines to apply to query
    sin: torch.Tensor  # sines to apply to query
    loc: torch.Tensor  # relative location


class HogwildCache(Cache):
    def __init__(
            self,
            cache_structure: List[List[Cache]],
            model: LlamaModel,
            write_to: Optional[List[Cache]] = None,
            input_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            override_length: Optional[int] = None,

    ):
        self.model = model
        self.cache_structure = cache_structure
        self.write_to = write_to if write_to else [cl[-1] for cl in cache_structure]
        self.cosines = []
        self.sines = []
        self.locations = []
        self.segments = []
        self.frags = []
        self.queries_buffer = None
        self.att_buffer = None

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.cache_structure[0][-1].get_seq_length()

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> CacheStructure:
        # update the worker caches
        assert key_states.shape[0] == len(self.cache_structure)

        # assume each batch index corresponds to one worker
        # TODO handle cases where not all workers are active
        for w in range(key_states.shape[0]):
            self.write_to[w].update(key_states[w:w + 1, ...], value_states[w:w + 1, ...], layer_idx, cache_kwargs)

        for cs in self.segments:
            cs.key_cache[layer_idx] = cs.key_cache[layer_idx].contiguous()
            cs.value_cache[layer_idx] = cs.value_cache[layer_idx].contiguous()

        if layer_idx == 0:
            mapping: Dict[int, InternalCacheMeta] = {}
            workers = len(self.cache_structure)

            for cs in self.cache_structure[0]:
                mapping[id(cs)] = InternalCacheMeta(
                    cos=[None] * workers, sin=[None] * workers, loc=[None] * workers, cs=cs)

            # and construct the info we need to actually run attention
            for w in range(key_states.shape[0]):
                pos = 0
                for cs in reversed(self.cache_structure[w]):
                    pos += cs.get_seq_length(layer_idx)
                    # at this point, pos already includes the newly-added tokens
                    # so, in order to match the right query position, we need to subtract the number of
                    # tokens currently being added
                    pos_t = torch.arange(pos - key_states.shape[2], pos, device=key_states.device, dtype=torch.int32)
                    mapping[id(cs)].loc[w] = pos_t

            # rearrange
            locations = []
            segments = []
            for entry in mapping.values():
                locations += entry.loc
                segments.append(entry.cs)

            locations = torch.stack(locations, dim=0)
            cosines, sines = self.model.rotary_emb(key_states, locations)
            self.cosines = cosines.reshape(len(segments), workers, locations.shape[1], cosines.shape[2]).to(torch.float)
            self.sines = sines.reshape(len(segments), workers, locations.shape[1], cosines.shape[2]).to(torch.float)
            self.locations = locations.reshape(len(segments), workers, locations.shape[1])
            self.segments = segments
            self.frags = torch.tensor([cs.get_seq_length(layer_idx) for cs in self.segments], dtype=torch.int32, device=self.cosines.device)
            # for some reason, having an explicit graph break is *essential* for good performance
            torch._dynamo.graph_break()
        keys = []
        vals = []
        for cs in self.segments:
            keys.append(cs.key_cache[layer_idx].contiguous())
            vals.append(cs.value_cache[layer_idx].contiguous())
        return CacheStructure(keys=keys, values=vals, cos=self.cosines, sin=self.sines, loc=self.locations, frags=self.frags)

    def get_queries_buffer(self, queries, layer_idx):
        if layer_idx == 0:
            self.queries_buffer = torch.empty((self.cosines.size(0), queries.size(0), queries.size(1),
                                               queries.size(2), queries.size(3)),
                                              dtype=queries.dtype, device=queries.device)
        return self.queries_buffer

    def get_att_buffer(self, r_queries, layer_idx):
        if layer_idx == 0:
            self.att_buffer = torch.empty((r_queries.size(1), r_queries.size(2), r_queries.size(3),
                                           self.cache_structure[0][0].value_cache[layer_idx].size(3)),
                                          dtype=r_queries.dtype, device=r_queries.device)
        return self.att_buffer


class LlamaAttentionMod(nn.Module):
    """Modified attention layer adapted to HogwildCache.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    # @torch.compile()
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[HogwildCache] = None,
            cache_position: torch.LongTensor = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # print("FORWARD")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # queries will be rotated for individual segments, so nothing to do here
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        cache: CacheStructure = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # concatenate queries in sequence length dimension
        # expand query per fragment
        rq=past_key_value.get_queries_buffer(query_states, layer_idx=self.layer_idx)
        attn_output = hogwild_fused(
            query_states,
            cache.loc,
            cache.keys,
            cache.values,
            self.scaling,
            cache.frags,
            cache.cos, cache.sin,
            rotated_queries=rq,
            out=past_key_value.get_att_buffer(rq, layer_idx=self.layer_idx),
        ).transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


########################################################################################################################

MODEL_NAME = "Qwen/QwQ-32B-AWQ"  # for 48GB gpus, use "Qwen/QwQ-32B-AWQ" instead
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device)

torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 128

for l in model.model.layers:
    old = l.self_attn
    l.self_attn = LlamaAttentionMod(model.model.config, l.self_attn.layer_idx)
    l.self_attn.k_proj = old.k_proj
    l.self_attn.v_proj = old.v_proj
    l.self_attn.q_proj = old.q_proj
    l.self_attn.o_proj = old.o_proj

parallelism_prompt_common = """
I will collaborate this problem with another. We refer to each other as Alice and Bob. We are assistants.

We will reason together and try to collaborate. I will take into account what the other assistant is doing and try to help them.

We will write our solutions concurrently. I will write my own thoughts at the bottom, and see the other's thoughts above.

I will not repeat the copy assistant's thoughts: I can already see them above.

The other assistant will continue writing their thoughts above while I am writing mine. They will add more text every time I check.

Since we both write our thoughts in parallel, I will initially see only partial (unfinished) thoughts of the other assistant.
I will use these partial thoughts to decide how best to help the other assistant without doing the same work twice.

When reasoning, we will give each other tasks to coordinate (e.g. if Alice writes: Bob, please do this, then Bob should take this into account).

Before doing anything, I will check the other assistant's workspace. If they have already done that or are currently doing it, I don't need to do that again. If so, I will stop (e.g. 'Wait, this is already done') and pivot to a different task.
""".strip()

worker_headers = ["\n\n# Alice workspace\n\n", "\n\n# Bob workspace\n\n"]
prompt_split = " <the assistant will continue here>\n\n"

forbidden_token_ix = [tokenizer.vocab[x] for x in ("#",)]
for x in tokenizer.special_tokens_map.values():
    forbidden_token_ix.extend([tokenizer.vocab[x]] if isinstance(x, str) else map(tokenizer.vocab.get, x))
tokenizer_kwargs = dict(add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')

problem = """Calculate x - x^2 + x^3 for x = 5,6,7,8. Alice must return all 4 answers in \\boxed{ }."""

raw_model = model.model
model = torch.compile(model)

parser = argparse.ArgumentParser()
parser.add_argument("--workers", default=1, type=int)
parser.add_argument("--profile", default=0, type=int)
parser.add_argument("--prefill", default=1, type=int)
parser.add_argument("--steps", default=200, type=int)

args = parser.parse_args()

results = {
    "model": MODEL_NAME,
    "attention": "v12",
    "workers": args.workers,
    "prefix": [],
    "duration": []
}
result_file = Path(f"benchmark-{MODEL_NAME.replace('/', '--').lower()}-w{args.workers}-{datetime.datetime.now()}.json")

prompt_full_input = tokenizer.apply_chat_template(
    [dict(role='user', content=problem)], tokenize=False, add_generation_prompt=True
) + "\n\n" + parallelism_prompt_common * args.prefill

worker_prompts = [
    f"""{worker_headers[0]}I am Alice. Let's solve this together, Bob. Here's how we should collaborate: I'll handle calculating the values for x = 156 and 157""",
    f"""{worker_headers[1]}I am Bob. Let's solve this together, Alice."""
]

cache_input, cache_split, cache_w1, cache_w2 = (DynamicCache() for _ in range(4))
cm = HogwildCache(cache_structure=[
    [cache_input, cache_w2, cache_split, cache_w1],
    [cache_input, cache_w1, cache_split, cache_w2],
], write_to=[cache_w1, cache_w2], model=raw_model)

# pre-fill common parts
with torch.inference_mode():
    model(**tokenizer(prompt_full_input, **tokenizer_kwargs).to(device),
          use_cache=True,
          past_key_values=HogwildCache([[cache_input]], model=raw_model))  # <-- write to common prompt
    model(**tokenizer(prompt_split, **tokenizer_kwargs).to(device),
          use_cache=True,
          past_key_values=HogwildCache([[cache_split]], model=raw_model))  # <-- write to common separator

last = time.perf_counter()

all_step_times = []

# generate tokens in parallel with each worker
next_inputs = tokenizer(worker_prompts, **tokenizer_kwargs).to(device)
tokens_by_worker = tokenizer(worker_prompts, add_special_tokens=False)["input_ids"]
for inference_step in range(args.steps):  # <-- change max tokens here
    with torch.inference_mode():
        if inference_step > 5 and inference_step < 10 and args.profile:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                logits = model(**next_inputs, past_key_values=cm).logits[..., -1, :]
            print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))
            prof.export_chrome_trace(f"step-new{inference_step}.json")
        else:
            logits = model(**next_inputs, past_key_values=cm).logits[..., -1, :]

        new_tokens = logits.argmax(-1)  # <-- greedy generation
        next_inputs = dict(input_ids=new_tokens.view(-1, 1))

    torch.cuda.synchronize()
    if inference_step > 5:
        duration = float(time.perf_counter() - last)
        results["duration"].append(duration)
        results["prefix"].append(int(cache_input.get_seq_length()) + int(cache_w1.get_seq_length()) + int(cache_w2.get_seq_length()) + int(cache_split.get_seq_length()))
        all_step_times.append(duration)
    last = time.perf_counter()
    for worker_tokens, new_token in zip(tokens_by_worker, new_tokens.tolist()):
        worker_tokens.append(new_token)
print(f"Prefix length: {cache_input.get_seq_length()}, duration: {np.array(all_step_times).mean()}")
result_file.write_text(json.dumps(results, indent=2))
