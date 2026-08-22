# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Benchmark stage-level profiling for HybridEngine-backed OPSD rollout.

The benchmark uses synthetic token IDs so prompt lengths are exact and model
tokenization does not become part of the measured rollout. Run it with one
accelerator process; ZeRO-3 and multi-rank measurements are intentionally left
for a later benchmark stage.
"""

import argparse
import itertools
import json
import math
import os
import statistics
import sys
from pathlib import Path

# Running a nested script does not automatically put the repository root on
# sys.path. Prefer the checkout under test over an older installed DeepSpeed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig


_TIMING_FIELDS = ("prompt_expansion_ms", "generation_ms", "post_processing_ms", "total_ms",
                  "tokens_per_second")
_OPTIONAL_TIMING_FIELDS = ("cache_retake_ms", "model_generation_ms", "prefill_ms", "decode_ms", "cache_release_ms",
                           "workspace_release_ms",
                           "gc_collect_ms", "empty_cache_ms", "memory_allocated_before_release_mb",
                           "memory_reserved_before_release_mb", "memory_allocated_after_release_mb",
                           "memory_reserved_after_release_mb")


def _percentile(values, percentile):
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _summarize(profiles):
    summary = {}
    fields = _TIMING_FIELDS + tuple(field for field in _OPTIONAL_TIMING_FIELDS if field in profiles[0])
    for field in fields:
        values = [profile[field] for profile in profiles]
        summary[field] = {
            "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": _percentile(values, 0.95),
        }
    return summary


def _validate_args(args):
    positive_values = [*args.batch_sizes, *args.samples_per_prompt, *args.prompt_lengths, *args.response_lengths]
    positive_values.extend([args.warmup, args.iterations])
    if any(value <= 0 for value in positive_values):
        raise ValueError("Batch sizes, sequence lengths, warmup, and iterations must all be positive")
    if args.temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top-p must be in the interval (0, 1]")
    if args.use_graph_capture and args.temperature > 0.0:
        raise ValueError("CUDA graph capture currently supports greedy decoding only")
    if args.torch_profile_output:
        matrix_dimensions = (args.batch_sizes, args.samples_per_prompt, args.prompt_lengths, args.response_lengths)
        if any(len(values) != 1 for values in matrix_dimensions):
            raise ValueError("Torch profiling requires exactly one value for each benchmark matrix dimension")


def _load_model_and_tokenizer(model_name, dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    return model.to(device), tokenizer


def _build_engine(model, args):
    use_bf16 = args.dtype == "bf16"
    max_effective_batch_size = max(batch_size * samples_per_prompt for batch_size, samples_per_prompt in
                                   itertools.product(args.batch_sizes, args.samples_per_prompt))
    ds_config = {
        "train_batch_size": max_effective_batch_size,
        "train_micro_batch_size_per_gpu": max_effective_batch_size,
        "fp16": {
            "enabled": not use_bf16,
        },
        "bf16": {
            "enabled": use_bf16,
        },
        "zero_optimization": {
            "stage": 0,
        },
        "hybrid_engine": {
            "enabled": True,
            "max_out_tokens": max(args.prompt_lengths) + max(args.response_lengths),
            "release_inference_cache": args.release_inference_cache,
            "enable_cuda_graph": args.use_graph_capture,
        },
    }
    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    if not hasattr(engine, "_generate"):
        raise RuntimeError("The model architecture did not create HybridEngine inference containers")
    engine.eval()
    return engine


def _make_request(model, batch_size, prompt_length, device):
    vocab_size = model.config.vocab_size
    first_token_id = min(3, vocab_size - 1)
    prompt_ids = torch.randint(first_token_id, vocab_size, (batch_size, prompt_length), device=device)
    prompt_attention_mask = torch.ones_like(prompt_ids)
    return RolloutRequest(prompt_ids=prompt_ids, prompt_attention_mask=prompt_attention_mask)


def _profile_rollout(rollout, request, sampling, output_path):
    activities = [torch.profiler.ProfilerActivity.CPU]
    sort_by = "self_cpu_time_total"
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_by = "self_cuda_time_total"

    trace_path = Path(output_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
    ) as profiler:
        rollout.generate(request, sampling)

    profiler.export_chrome_trace(str(trace_path))
    summary = profiler.key_averages().table(sort_by=sort_by, row_limit=30)
    summary_path = trace_path.with_suffix(".summary.txt")
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote profiler trace to {trace_path}")
    print(f"Wrote profiler summary to {summary_path}")


def _run_case(rollout, model, args, batch_size, samples_per_prompt, prompt_length, response_length, device):
    request = _make_request(model, batch_size, prompt_length, device)
    sampling = SamplingConfig(
        max_new_tokens=response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        n_samples_per_prompt=samples_per_prompt,
    )

    for _ in range(args.warmup):
        rollout.generate(request, sampling)

    accelerator = get_accelerator()
    if args.torch_profile_output:
        _profile_rollout(rollout, request, sampling, args.torch_profile_output)
    accelerator.reset_peak_memory_stats()
    profiles = []
    for _ in range(args.iterations):
        rollout.generate(request, sampling)
        profiles.append(dict(rollout.get_last_profile()))

    decode_graphs = getattr(rollout.engine, "_decode_graphs", None)
    captured_positions = decode_graphs.captured_positions if decode_graphs is not None else 0
    return {
        "batch_size": batch_size,
        "samples_per_prompt": samples_per_prompt,
        "prompt_length": prompt_length,
        "requested_response_length": response_length,
        "returned_response_length": profiles[-1]["response_length"],
        "cuda_graph_captured_positions": captured_positions,
        "peak_memory_mb": accelerator.max_memory_allocated() / (1024**2),
        "summary": _summarize(profiles),
        "profiles": profiles,
    }


def _ordered_case_specs(args):
    requested = list(
        itertools.product(args.batch_sizes, args.samples_per_prompt, args.prompt_lengths, args.response_lengths))
    execution_order = sorted(range(len(requested)),
                             key=lambda index: requested[index][0] * requested[index][1],
                             reverse=True)
    return requested, execution_order


def _run(args):
    _validate_args(args)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError("This initial benchmark supports exactly one accelerator process")

    torch.manual_seed(args.seed)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    accelerator = get_accelerator()
    accelerator.set_device(local_rank)
    device = accelerator.device_name(local_rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    try:
        model, tokenizer = _load_model_and_tokenizer(args.model, dtype, device)
        engine = _build_engine(model, args)
        enable_generation_phase_profiling = (args.enable_generation_phase_profiling
                                             and not args.torch_profile_output)
        rollout_config = HybridEngineRolloutConfig(
            use_graph_capture=args.use_graph_capture,
            enable_profiling=True,
            enable_generation_phase_profiling=enable_generation_phase_profiling,
        )
        rollout = HybridEngineRollout(engine, tokenizer, rollout_config)

        case_specs, execution_order = _ordered_case_specs(args)
        cases = [None] * len(case_specs)
        # HybridEngine sizes its inference workspace on the first forward. Run
        # the largest effective batch first while preserving requested output order.
        for case_index in execution_order:
            batch_size, samples_per_prompt, prompt_length, response_length = case_specs[case_index]
            cases[case_index] = _run_case(rollout, engine.module, args, batch_size, samples_per_prompt, prompt_length,
                                          response_length, device)

        result = {
            "model": args.model,
            "dtype": args.dtype,
            "device": device,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "use_graph_capture": args.use_graph_capture,
            "generation_phase_profiling": enable_generation_phase_profiling,
            "torch_profile_output": args.torch_profile_output,
            "release_inference_cache": args.release_inference_cache,
            "cases": cases,
        }
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote benchmark results to {output_path}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark HybridEngine rollout stage profiling")
    parser.add_argument("--model", default="facebook/opt-6.7b", help="HuggingFace model ID or local model path")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--samples-per-prompt", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--response-lengths", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--release-inference-cache", action="store_true")
    parser.add_argument("--use-graph-capture", action="store_true", help="Use CUDA graph capture for greedy decode")
    parser.add_argument("--no-generation-phase-profiling",
                        action="store_false",
                        dest="enable_generation_phase_profiling",
                        help="Disable per-forward prefill and decode profiling")
    parser.add_argument("--torch-profile-output", help="Write a one-iteration PyTorch profiler Chrome trace")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="opsd_rollout_profile.json")
    return parser.parse_args(argv)


def main():
    _run(_parse_args())


if __name__ == "__main__":
    main()
