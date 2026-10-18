# DeepSpeed Benchmarks

If you are looking for DeepSpeed benchmarks, please see the following resources:

1. [Communication Benchmarking Suite](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/communication)
2. [Inference Benchmarks](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/inference)

## Continuous rollout batched EOS checks

Continuous greedy rollouts batch EOS detection in scheduler request order, including
tokens from newly prefetched requests and surviving decode rows. Eager and CUDA graph
rollouts transfer one boolean vector per scheduler step instead of reading a scalar
per request. With `eos_token_id=None` or an empty EOS list, no EOS flags are transferred.
Other scheduler/cache synchronization is unchanged.

Compare the previous per-request check with the batched implementation on CUDA:

```bash
python -m benchmarks.rollout.bench_batched_eos \
    --batch-sizes 1 4 16 --prompt-len 32 --max-new-tokens 32 \
    --iterations 10 --output rollout-batched-eos.json
```

The offline benchmark uses a random two-layer FP16 Llama with eager attention and
checks every call against Hugging Face greedy generation, including returned masks
and response offsets. Configured non-emittable EOS IDs hold batch sizes constant;
retirement/refill behavior is covered separately by rollout integration tests.
Both implementations also run without EOS configuration. Each mode warms up once,
then the measurement order alternates. Graph capture is excluded from measured calls.
The synchronized decode window excludes prefill and spans the first-to-last EOS-check
completion, including scheduling and token selection, not just model forwards.
The JSON contains individual calls, median decode and total latency, hardware/software
and model metadata, and profiler counts of scalar reads and directional D2H copies
for one EOS check. These synthetic results do not establish production-model speedups
or elimination of all decode synchronization.

Use a Transformers version compatible with the underlying continuous-rollout cache.
The original implementation requires the separate cache query-offset compatibility
fix to run Llama on Transformers 5.20.1; Transformers 4.57.6 does not require that fix.
