# DeepSpeed Benchmarks

If you are looking for DeepSpeed benchmarks, please see the following resources:

1. [Communication Benchmarking Suite](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/communication)
2. [Inference Benchmarks](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/inference)

## Continuous rollout CUDA Graph bucket pool

Run on a CUDA GPU with a Transformers version compatible with `DeepSpeedStaticCache`
(for example 4.57.6; 5.20.1 requires the separate cache query-offset fix):

```bash
python -m benchmarks.rollout.bench_cuda_graph_bucket_pool \
    --prompt-widths 32 64 31 --buckets 32 64 --pool-size 2 \
    --requests 8 --batch-size 4 --max-new-tokens 32 \
    --iterations 10 --output rollout-bucket-pool.json
```

This uses the first-call benchmark's synchronized end-to-end timing and native
Hugging Face `generate()` oracle, on a randomly initialized tiny FP16 Llama with
eager attention and EOS disabled (no model download). It compares eager rollout,
one retained graph, and a two-state LRU pool. Both graph modes normalize the same
bucket layout: widths 32 and 31 share bucket 32, and width 64 uses bucket 64.
The first cycle reports each A / B / A call separately; steady state repeats that
cycle and includes recaptures caused by eviction, rather than hiding them.
Every measured call checks tokens, attention masks, and response offsets.

JSON includes all call latencies, capture counts, throughput, current and peak
allocated/reserved GPU bytes, and hardware/software/model/workload metadata.
Memory peaks cover the complete first and steady-state phases for each mode;
reserved memory includes allocator caching, not just live graph tensors.
These synthetic measurements are not pretrained-model performance claims.
Training-update invalidation and EOS/refill parity are covered by the CUDA
integration tests in `tests/unit/v1/test_hybrid_engine_rollout_v1.py`.

Measured on 2026-10-18 with the command above: NVIDIA RTX A4500 (20 GiB),
PyTorch 2.9.1+cu128 / CUDA 12.8, Transformers 4.57.6. Latencies are synchronized
end-to-end milliseconds; steady values are medians over 30 calls.

| Mode | First A | First B | Return to A (width 31) | Steady median | Steady captures | Peak allocated / reserved (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager | 405.156 | 406.020 | 402.987 | 389.234 | 0 | 10.125 / 24 |
| One state | 66.912 | 63.892 | 65.066 | 64.224 | 20 | 204.438 / 510 |
| Two-state pool | 67.477 | 63.378 | 51.707 | 51.146 | 0 | 221.511 / 550 |

First-cycle graph captures were `[1, 1, 1]` with one state and `[1, 1, 0]`
with two. Pooling reduced this workload's steady median by 20.4%, with
17.1 MiB more peak allocated memory and 40 MiB more peak reserved memory.
All measured outputs matched native generation. Library/model initialization
and native reference generation preceded measurement; rollout graphs were not
prewarmed. Modes ran sequentially as eager, one state, then pooled. These are
single-run synthetic observations, not a general performance guarantee.
