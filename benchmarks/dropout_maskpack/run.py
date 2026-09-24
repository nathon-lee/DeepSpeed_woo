#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Reproducible dropout mask-pack correctness and CUDA-event benchmark driver.

Correctness is intentionally run in one process per source tree.  Run this
program once for the baseline and once for the candidate, then use --compare.
The benchmark mode loads both source trees in one process and alternates them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List

import torch
from torch.utils.cpp_extension import load


SIZES = (1 << 20, 8 << 20, 64 << 20)
DIMS = (1024, 4096)
RATIO = 0.1
DTYPES = (torch.float16, torch.float32)
WARMUP = 30
GROUPS = 60
CALLS = 50


def commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def tool_version(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        return "missing"
    try:
        return subprocess.check_output([executable, "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def environment(root: Path) -> Dict[str, object]:
    result: Dict[str, object] = {
        "commit": commit(root),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "compiler": tool_version(os.environ.get("CXX", "c++")),
        "nvcc": tool_version("nvcc"),
    }
    if torch.cuda.is_available():
        result["device"] = torch.cuda.get_device_name()
        result["capability"] = list(torch.cuda.get_device_capability())
    else:
        result["device"] = None
        result["capability"] = None
    return result


def sha256_tensor(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def input_tensor(numel: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
    # Integer arithmetic makes inputs identical in independent processes.
    values = torch.arange(numel, device="cuda", dtype=torch.float32)
    values = values.remainder(257.0).sub(128.0).div(127.0)
    return values.to(dtype=dtype).reshape(numel // dim, dim)


def wrapper_source() -> str:
    return r'''
#include <torch/extension.h>
#include "context.h"
#include "custom_cuda_layers.h"

namespace {

template <typename T>
std::tuple<torch::Tensor, torch::Tensor> forward_impl(const torch::Tensor& input,
                                                       double ratio,
                                                       uint64_t seed) {
    auto output = torch::empty_like(input);
    auto mask = torch::empty({input.numel()}, input.options().dtype(torch::kUInt8));
    TrainingContext::Instance().SetSeed(seed);
    launch_dropout<T>(output.data_ptr<T>(),
                      input.data_ptr<T>(),
                      mask.data_ptr<uint8_t>(),
                      static_cast<int>(input.numel()),
                      static_cast<int>(input.size(-1)),
                      static_cast<float>(ratio),
                      TrainingContext::Instance().GetCurrentStream(),
                      false);
    return {output, mask};
}

template <typename T>
void launch_impl(const torch::Tensor& output,
                 const torch::Tensor& input,
                 const torch::Tensor& mask,
                 double ratio,
                 bool backward) {
    launch_dropout<T>(output.data_ptr<T>(),
                      input.data_ptr<T>(),
                      mask.data_ptr<uint8_t>(),
                      static_cast<int>(input.numel()),
                      static_cast<int>(input.size(-1)),
                      static_cast<float>(ratio),
                      TrainingContext::Instance().GetCurrentStream(),
                      backward);
}

template <typename T>
std::tuple<torch::Tensor, torch::Tensor> forward_checked(const torch::Tensor& input,
                                                          double ratio,
                                                          uint64_t seed) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA");
    TORCH_CHECK(input.numel() % 4 == 0, "dropout launcher requires numel divisible by four");
    return forward_impl<T>(input, ratio, seed);
}

template <typename T>
void launch_checked(const torch::Tensor& output,
                    const torch::Tensor& input,
                    const torch::Tensor& mask,
                    double ratio,
                    bool backward) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA");
    TORCH_CHECK(output.is_cuda() && output.is_contiguous(), "output must be contiguous CUDA");
    TORCH_CHECK(mask.is_cuda() && mask.is_contiguous(), "mask must be contiguous CUDA");
    TORCH_CHECK(input.numel() % 4 == 0, "dropout launcher requires numel divisible by four");
    launch_impl<T>(output, input, mask, ratio, backward);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_fp16", &forward_checked<at::Half>);
    m.def("forward_fp32", &forward_checked<float>);
    m.def("launch_fp16", &launch_checked<at::Half>);
    m.def("launch_fp32", &launch_checked<float>);
}
'''


def load_dropout(root: Path, tag: str):
    build_dir = Path(tempfile.mkdtemp(prefix=f"ds_dropout_{tag}_"))
    binding = build_dir / "binding.cpp"
    binding.write_text(wrapper_source())
    name = f"ds_dropout_{tag}_{commit(root)[:12]}"
    return load(
        name=name,
        sources=[str(binding), str(root / "csrc/transformer/dropout_kernels.cu")],
        extra_include_paths=[str(root / "csrc/includes")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcurand", "-lcublas"],
        with_cuda=True,
        build_directory=str(build_dir),
        verbose=False,
    )


def percentile(values: Iterable[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def dtype_name(dtype: torch.dtype) -> str:
    return "fp16" if dtype == torch.float16 else "fp32"


def correctness(root: Path, output: Path) -> Dict[str, object]:
    module = load_dropout(root, "correctness")
    cases: List[Dict[str, object]] = []
    for dtype in DTYPES:
        for numel in SIZES:
            for dim in DIMS:
                if numel % dim:
                    continue
                input_value = input_tensor(numel, dim, dtype)
                forward = getattr(module, f"forward_{dtype_name(dtype)}")
                output_value, mask = forward(input_value, RATIO, 123)
                expected = input_value.float() * mask.float().reshape_as(input_value) * (1.0 / (1.0 - RATIO))
                forward_error = (output_value.float() - expected).abs()
                backward_value = torch.empty_like(input_value)
                getattr(module, f"launch_{dtype_name(dtype)}")(
                    backward_value, input_value, mask, RATIO, True
                )
                backward_error = (backward_value.float() - expected).abs()
                cases.append(
                    {
                        "dtype": dtype_name(dtype),
                        "numel": numel,
                        "dim": dim,
                        "ratio": RATIO,
                        "mask_sha256": sha256_tensor(mask),
                        "output_sha256": sha256_tensor(output_value),
                        "max_abs_forward": float(forward_error.max().item()),
                        "max_abs_backward": float(backward_error.max().item()),
                        "finite_forward": bool(torch.isfinite(output_value).all().item()),
                        "finite_backward": bool(torch.isfinite(backward_value).all().item()),
                    }
                )
    result = {
        "status": "complete",
        "mode": "correctness",
        "environment": environment(root),
        "alignment": "numel divisible by four; non-four tail is not exercised",
        "cases": cases,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def timed_call(module, dtype_name_value: str, output, input_value, mask, ratio: float) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    getattr(module, f"launch_{dtype_name_value}")(output, input_value, mask, ratio, False)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop))


def benchmark(baseline_root: Path, candidate_root: Path, output: Path, csv_output: Path) -> Dict[str, object]:
    baseline = load_dropout(baseline_root, "baseline")
    candidate = load_dropout(candidate_root, "candidate")
    rows: List[Dict[str, object]] = []
    for dtype in DTYPES:
        dtype_value = dtype_name(dtype)
        for numel in SIZES:
            for dim in DIMS:
                if numel % dim:
                    continue
                input_value = input_tensor(numel, dim, dtype)
                baseline_output = torch.empty_like(input_value)
                baseline_mask = torch.empty(numel, device="cuda", dtype=torch.uint8)
                candidate_output = torch.empty_like(input_value)
                candidate_mask = torch.empty(numel, device="cuda", dtype=torch.uint8)
                for module, out_value, mask_value in (
                    (baseline, baseline_output, baseline_mask),
                    (candidate, candidate_output, candidate_mask),
                ):
                    for _ in range(WARMUP):
                        getattr(module, f"launch_{dtype_value}")(
                            out_value, input_value, mask_value, RATIO, False
                        )
                    torch.cuda.synchronize()
                baseline_groups: List[float] = []
                candidate_groups: List[float] = []
                for group in range(GROUPS):
                    if group % 2 == 0:
                        first = (baseline, baseline_output, baseline_mask, baseline_groups)
                        second = (candidate, candidate_output, candidate_mask, candidate_groups)
                    else:
                        first = (candidate, candidate_output, candidate_mask, candidate_groups)
                        second = (baseline, baseline_output, baseline_mask, baseline_groups)
                    for module, out_value, mask_value, groups in (first, second):
                        samples = []
                        for _ in range(CALLS):
                            samples.append(
                                timed_call(module, dtype_value, out_value, input_value, mask_value, RATIO)
                            )
                        groups.append(statistics.median(samples))
                baseline_ms = percentile(baseline_groups, 0.5)
                candidate_ms = percentile(candidate_groups, 0.5)
                row = {
                    "dtype": dtype_value,
                    "numel": numel,
                    "dim": dim,
                    "baseline_median_ms": baseline_ms,
                    "baseline_p10_ms": percentile(baseline_groups, 0.1),
                    "baseline_p90_ms": percentile(baseline_groups, 0.9),
                    "candidate_median_ms": candidate_ms,
                    "candidate_p10_ms": percentile(candidate_groups, 0.1),
                    "candidate_p90_ms": percentile(candidate_groups, 0.9),
                    "speedup_percent": (baseline_ms / candidate_ms - 1.0) * 100.0,
                    "groups": GROUPS,
                    "calls_per_group": CALLS,
                }
                rows.append(row)
    result = {
        "status": "complete",
        "mode": "benchmark",
        "environment_baseline": environment(baseline_root),
        "environment_candidate": environment(candidate_root),
        "warmup": WARMUP,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def blocked(root: Path, mode: str, output: Path, reason: str) -> None:
    result = {"status": "blocked", "mode": mode, "reason": reason, "environment": environment(root)}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def compare(first: Path, second: Path) -> int:
    left = json.loads(first.read_text())
    right = json.loads(second.read_text())
    if left.get("status") != "complete" or right.get("status") != "complete":
        print("comparison blocked: both correctness runs must be complete")
        return 2
    left_cases = {(x["dtype"], x["numel"], x["dim"]): x for x in left["cases"]}
    right_cases = {(x["dtype"], x["numel"], x["dim"]): x for x in right["cases"]}
    mismatches = []
    for key in sorted(left_cases):
        for field in ("mask_sha256", "output_sha256"):
            if left_cases[key][field] != right_cases[key][field]:
                mismatches.append((key, field, left_cases[key][field], right_cases[key][field]))
    if mismatches:
        print(json.dumps({"status": "mismatch", "mismatches": mismatches}, indent=2))
        return 1
    print(json.dumps({"status": "match", "cases": len(left_cases)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("correctness", "benchmark", "compare"), required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    if args.mode == "compare":
        if not args.first or not args.second:
            parser.error("--first and --second are required for compare")
        return compare(args.first, args.second)

    root = args.source_root or args.candidate_root or Path.cwd()
    if not args.output:
        parser.error("--output is required for correctness and benchmark")
    if not torch.cuda.is_available():
        blocked(root, args.mode, args.output, "CUDA GPU unavailable: nvidia-smi/nvcc or a CUDA device is missing")
        if args.csv_output:
            args.csv_output.write_text("status,reason\nblocked,CUDA GPU unavailable\n")
        print(f"BLOCKED: CUDA GPU unavailable; wrote {args.output}")
        return 2

    if args.mode == "correctness":
        correctness(root, args.output)
    else:
        if not args.baseline_root or not args.candidate_root or not args.csv_output:
            parser.error("--baseline-root, --candidate-root and --csv-output are required for benchmark")
        benchmark(args.baseline_root, args.candidate_root, args.output, args.csv_output)
    print(f"COMPLETE: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
