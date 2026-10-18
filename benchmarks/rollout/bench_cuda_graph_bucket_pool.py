# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Measure alternating prompt buckets with eager, single-state, and pooled decode."""

import argparse
import gc
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


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-widths", nargs="+", type=positive_int, default=[32, 64, 31])
    parser.add_argument("--buckets", nargs="+", type=positive_int, default=[32, 64])
    parser.add_argument("--pool-size", type=positive_int, default=2)
    parser.add_argument("--requests", type=positive_int, default=8)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--max-new-tokens", type=positive_int, default=32)
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.buckets != sorted(set(args.buckets)) or max(args.prompt_widths) > args.buckets[-1]:
        parser.error("--buckets must be strictly increasing and cover all prompt widths")
    if args.max_new_tokens < 2 or args.requests <= args.batch_size:
        parser.error("use at least two generated tokens and more requests than active slots")
    return args


def measure_cycle(rollout, requests, references, sampling):
    accelerator = get_accelerator()
    calls = []
    for request, reference in zip(requests, references):
        accelerator.synchronize()
        start = time.perf_counter()
        output = rollout.generate(request, sampling)
        accelerator.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        torch.testing.assert_close(output.input_ids.cpu(), reference, rtol=0, atol=0)
        width = request.prompt_ids.shape[1]
        assert torch.equal(output.attention_mask[:, :width], request.prompt_attention_mask)
        assert output.attention_mask[:, width:].eq(1).all()
        assert output.response_start_idx.eq(width).all()
        stats = rollout.get_last_continuous_stats()
        calls.append({
            "prompt_width": width,
            "latency_ms": elapsed_ms,
            "graph_captures": stats.get("graph_captures", 0),
            "graph_pool_size": stats.get("graph_pool_size", 0),
        })
    return calls


def summarize(calls, tokens_per_call):
    total_ms = sum(call["latency_ms"] for call in calls)
    return {
        "calls": calls,
        "median_ms": statistics.median(call["latency_ms"] for call in calls),
        "graph_captures": sum(call["graph_captures"] for call in calls),
        "tokens_per_second": len(calls) * tokens_per_call / (total_ms / 1000.0),
    }


@torch.no_grad()
def run_benchmark(args):
    accelerator = get_accelerator()
    if accelerator.device_name() != "cuda" or not accelerator.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    accelerator.set_device(0)
    torch.manual_seed(args.seed)
    device = torch.device(accelerator.current_device_name())
    config = LlamaConfig(vocab_size=256,
                         hidden_size=128,
                         intermediate_size=256,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         max_position_embeddings=(max(args.buckets) + args.max_new_tokens) * args.requests,
                         pad_token_id=0,
                         eos_token_id=None,
                         attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.float16).eval()
    requests = []
    references = []
    for width in args.prompt_widths:
        ids = torch.randint(1, config.vocab_size, (args.requests, width), device=device)
        request = RolloutRequest(ids, torch.ones_like(ids))
        requests.append(request)
        references.append(
            model.generate(ids,
                           attention_mask=request.prompt_attention_mask,
                           max_new_tokens=args.max_new_tokens,
                           do_sample=False,
                           eos_token_id=None,
                           pad_token_id=0).cpu())

    sampling = SamplingConfig(max_new_tokens=args.max_new_tokens, temperature=0, continuous_batch_size=args.batch_size)
    results = {}
    for mode, pool_size in (("eager", 1), ("single_state", 1), ("pooled", args.pool_size)):
        gc.collect()
        accelerator.empty_cache()
        accelerator.reset_peak_memory_stats()
        rollout = HybridEngineRollout(
            SimpleNamespace(module=model),
            SimpleNamespace(pad_token_id=0, eos_token_id=None),
            HybridEngineRolloutConfig(use_graph_capture=mode != "eager",
                                      continuous_graph_pool_size=pool_size,
                                      continuous_prompt_buckets=tuple(args.buckets)),
        )
        first_cycle = measure_cycle(rollout, requests, references, sampling)
        steady = []
        for _ in range(args.iterations):
            steady.extend(measure_cycle(rollout, requests, references, sampling))
        tokens = args.requests * args.max_new_tokens
        results[mode] = {
            "first_cycle": summarize(first_cycle, tokens),
            "steady_state": summarize(steady, tokens),
            "allocated_bytes": accelerator.memory_allocated(),
            "reserved_bytes": accelerator.memory_reserved(),
            "peak_allocated_bytes": accelerator.max_memory_allocated(),
            "peak_reserved_bytes": accelerator.max_memory_reserved(),
        }
        del rollout

    return {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),  #ignore-cuda: accelerator has no GPU model-name API
            "total_memory_bytes": accelerator.total_memory(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "workload": {
            "model": "random tiny Llama (no download)",
            "model_config": config.to_dict(),
            "dtype": "float16",
            "seed": args.seed,
            "requests": args.requests,
            "batch_size": args.batch_size,
            "prompt_widths": args.prompt_widths,
            "buckets": args.buckets,
            "pool_size": args.pool_size,
            "max_new_tokens": args.max_new_tokens,
            "iterations": args.iterations,
            "eos_disabled": True,
            "profiling_enabled": False,
        },
        "results": results,
        "outputs_match": True,
    }


if __name__ == "__main__":
    args = parse_args()
    serialized = json.dumps(run_benchmark(args), indent=2)
    if args.output is not None:
        args.output.write_text(serialized + "\n")
    print(serialized)
