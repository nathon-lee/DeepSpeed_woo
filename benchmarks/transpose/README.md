# Positive-only transpose validation

`benchmark.py` builds baseline commit `0b987c0e99f546e7867f0c6af4c8336d7d1e2eca` and candidate commit `e90e6aef4a17a08446dd362aeed54042387bd531` as separate CUDA extensions. It records CUDA-event timings, exact output checks, SASS hashes, GPU state, and every group in a timestamped output directory. Use `--reverse` for the candidate-first dynamic-loader check.

```bash
CUDA_VISIBLE_DEVICES=<idle-h200> python3 benchmarks/transpose/benchmark.py \\
  --baseline /path/to/baseline \\
  --candidate /path/to/candidate \\
  --output /path/to/results-UTC
```

The launcher requires one explicitly selected GPU. The current validation host has no `nvidia-smi`, `nvcc`, or CUDA device, so its output is marked `blocked` and contains no performance claim.
