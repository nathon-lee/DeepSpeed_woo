# Dropout mask-pack validation

`run.py` builds the real `csrc/transformer/dropout_kernels.cu` through a small PyTorch CUDA extension. Correctness runs must be launched in separate processes for the two source trees; `--compare` checks mask and FP16 output SHA-256 values. Benchmark mode loads both source trees in one process, warms each kernel 30 times, alternates them, and records 60 groups of 50 CUDA-event timings.

From two clean worktrees (one at `0b987c0e99f546e7867f0c6af4c8336d7d1e2eca`, one at the candidate commit), run:

```bash
python3 benchmarks/dropout_maskpack/run.py --mode correctness \\
  --source-root /path/to/baseline --output baseline_correctness.json
python3 benchmarks/dropout_maskpack/run.py --mode correctness \\
  --source-root /path/to/candidate --output candidate_correctness.json
python3 benchmarks/dropout_maskpack/run.py --mode compare \\
  --first baseline_correctness.json --second candidate_correctness.json
python3 benchmarks/dropout_maskpack/run.py --mode benchmark \\
  --baseline-root /path/to/baseline --candidate-root /path/to/candidate \\
  --output benchmark.json --csv-output benchmark.csv
```

The launcher requires element counts divisible by four, so the recorded cases do not exercise a non-four tail. The script records the commit, PyTorch/CUDA/device information, hashes, finite checks, forward/backward errors, and percentile timings. If CUDA or a GPU is unavailable it writes a machine-readable `status: blocked` result and exits with status 2.
