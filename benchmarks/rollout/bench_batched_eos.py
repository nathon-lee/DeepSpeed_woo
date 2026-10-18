# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compare per-request and batched EOS checks on steady continuous CUDA decoding."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig


class MeasuredRollout(HybridEngineRollout):

    def __init__(self, model, eos_token_id, graph, legacy, steps):
        super().__init__(SimpleNamespace(module=model),
                         SimpleNamespace(pad_token_id=0, eos_token_id=eos_token_id),
                         cfg=HybridEngineRolloutConfig(use_graph_capture=graph))
        self.legacy = legacy
        self.steps = steps
        self.step = 0
        self.decode_ms = None

    def check_eos(self, active_ids, next_tokens):
        if not self.legacy:
            return super()._continuous_finished_ids(active_ids, next_tokens)
        finished = []
        for request_id in active_ids:
            token = next_tokens[request_id]
            if self.tokenizer.eos_token_id is not None:
                eos_ids = torch.as_tensor(self.tokenizer.eos_token_id, dtype=token.dtype,
                                          device=token.device).flatten()
                if (token == eos_ids).any().item():
                    finished.append(request_id)
        return finished

    def _continuous_finished_ids(self, active_ids, next_tokens):
        finished = self.check_eos(active_ids, next_tokens)
        self.step += 1
        # Exclude prefill and capture. Synchronize only at the decode window boundaries.
        if self.step == 1:
            get_accelerator().synchronize()
            self.decode_start = time.perf_counter()
        elif self.step == self.steps:
            get_accelerator().synchronize()
            self.decode_ms = (time.perf_counter() - self.decode_start) * 1000
        return finished

    def measure(self, request, sampling, reference):
        self.step = 0
        self.decode_ms = None
        get_accelerator().synchronize()
        start = time.perf_counter()
        output = self.generate(request, sampling)
        get_accelerator().synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert torch.equal(output.input_ids, reference)
        assert torch.equal(output.attention_mask, torch.ones_like(reference))
        assert output.response_start_idx.tolist() == [request.prompt_ids.shape[1]] * reference.shape[0]
        assert self.step == self.steps and self.decode_ms is not None
        return {"total_ms": elapsed_ms, "decode_ms": self.decode_ms}


def profile_eos(rollout, batch_size, device):
    tokens = torch.full((batch_size, 1), 7, dtype=torch.long, device=device)
    active_ids = tuple(range(batch_size))
    next_tokens = dict(zip(active_ids, tokens.split(1)))
    rollout.check_eos(active_ids, next_tokens)
    get_accelerator().synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        assert rollout.check_eos(active_ids, next_tokens) == []
    events = profile.events()
    return {
        "scalar_reads": sum(event.name == "aten::_local_scalar_dense" for event in events),
        "d2h_copies": sum("Memcpy DtoH" in event.name for event in events),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_sizes + [args.prompt_len, args.iterations]) < 1 or args.max_new_tokens < 2:
        parser.error("sizes and iterations must be positive; max-new-tokens must be at least 2")
    accelerator = get_accelerator()
    if accelerator.device_name() != "cuda" or not accelerator.is_available():
        parser.error("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(1234)
    config = LlamaConfig(vocab_size=128,
                         hidden_size=64,
                         intermediate_size=128,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         max_position_embeddings=args.prompt_len + args.max_new_tokens,
                         bos_token_id=1,
                         eos_token_id=None,
                         pad_token_id=0)
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.float16).eval()
    model.config._attn_implementation = "eager"
    properties = torch.cuda.get_device_properties(device)  #ignore-cuda
    report = {
        "hardware": {
            "gpu": properties.name,
            "memory_bytes": properties.total_memory
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__
        },
        "model": config.to_dict(),
        "workload": {
            "batch_sizes":
            args.batch_sizes,
            "prompt_len":
            args.prompt_len,
            "max_new_tokens":
            args.max_new_tokens,
            "iterations":
            args.iterations,
            "seed":
            1234,
            "dtype":
            "float16",
            "configured_eos_ids": [128, 129],
            "note":
            "Synthetic random Llama; non-emittable EOS IDs keep the active batch fixed. "
            "Decode window spans first-to-last EOS-check completion, excluding prefill/capture, "
            "including scheduling and token selection. One warmup per mode; no weight updates.",
        },
        "results": [],
    }
    with torch.no_grad():
        for batch_size in args.batch_sizes:
            prompts = torch.randint(4, config.vocab_size, (batch_size, args.prompt_len), device=device)
            request = RolloutRequest(prompts, torch.ones_like(prompts))
            sampling = SamplingConfig(max_new_tokens=args.max_new_tokens,
                                      temperature=0,
                                      continuous_batch_size=batch_size)
            reference = model.generate(input_ids=prompts,
                                       attention_mask=request.prompt_attention_mask,
                                       max_new_tokens=args.max_new_tokens,
                                       do_sample=False,
                                       eos_token_id=None,
                                       pad_token_id=0)
            for graph in (False, True):
                for eos_ids in ([128, 129], None):
                    rollouts = [
                        MeasuredRollout(model, eos_ids, graph, legacy, args.max_new_tokens) for legacy in (True, False)
                    ]
                    for rollout in rollouts:
                        rollout.measure(request, sampling, reference)
                    measurements = [[], []]
                    # Alternate order to reduce systematic before/after ordering bias.
                    for iteration in range(args.iterations):
                        for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                            measurements[index].append(rollouts[index].measure(request, sampling, reference))
                    for rollout, calls in zip(rollouts, measurements):
                        eos_profile = profile_eos(rollout, batch_size, device)
                        expected_copies = (batch_size if rollout.legacy else 1) if eos_ids is not None else 0
                        assert eos_profile["d2h_copies"] == expected_copies
                        assert eos_profile["scalar_reads"] == (expected_copies if rollout.legacy else 0)
                        report["results"].append({
                            "batch_size":
                            batch_size,
                            "graph":
                            graph,
                            "eos_enabled":
                            eos_ids is not None,
                            "implementation":
                            "per_request" if rollout.legacy else "batched",
                            "calls":
                            calls,
                            "median_total_ms":
                            statistics.median(call["total_ms"] for call in calls),
                            "median_decode_ms":
                            statistics.median(call["decode_ms"] for call in calls),
                            "decode_steps":
                            args.max_new_tokens - 1,
                            "eos_profile":
                            eos_profile,
                        })
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for result in report["results"]:
        print(f"batch={result['batch_size']} graph={result['graph']} eos={result['eos_enabled']} "
              f"{result['implementation']}: decode={result['median_decode_ms']:.3f} ms, "
              f"total={result['median_total_ms']:.3f} ms, EOS profile={result['eos_profile']}")


if __name__ == "__main__":
    main()
