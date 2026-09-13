```markdown
# [RFC] Training-aware paged KV cache for OPSD rollouts

## Summary

OPSD already has the two initial building blocks required for efficient rollout
generation:

- [#8296]([https://github.com/deepspeedai/DeepSpeed/pull/8296](https://github.com/deepspeedai/DeepSpeed/pull/8296)) adds shared prompt
  prefill for multiple response samples from the same prompt.
- [#8368]([https://github.com/deepspeedai/DeepSpeed/pull/8368](https://github.com/deepspeedai/DeepSpeed/pull/8368)) adds experimental
  continuous batching with request retirement, refill, per-row write positions,
  and `StaticCache` row management.

The next step is to replace the current contiguous KV-cache layout used by the
continuous path with an opt-in paged cache that avoids unnecessary KV movement
and provides a foundation for prompt reuse and rollout branching.

The target is a training-aware KV cache designed for OPSD, where cache reuse
must preserve rollout and training correctness rather than only improve serving
latency.

## Motivation

The current `DeepSpeedStaticCache` is a useful fixed-shape and CUDA-Graph
baseline. It preallocates contiguous tensors with a layout similar to:

```text
[batch, kv_heads, max_cache_len, head_dim]
```

This layout creates several costs for continuous rollouts:

- memory is reserved for the maximum batch and sequence length;
- request retirement can require moving active KV rows;
- left trimming can move KV contents across all layers;
- variable-length requests are represented through shared physical storage;
- cache ownership is tied to rows rather than independently managed blocks;
- a general prefix lookup and eviction policy is not available.

OPSD has an additional correctness requirement. Any cache optimization must
preserve:

- generated tokens;
- student logits;
- teacher logits;
- response masks;
- KL or JSD loss;
- reward values;
- the resulting training behavior.

## Goals

This RFC proposes an opt-in local paged KV cache for OPSD rollouts.

The initial implementation should:

1. avoid full-row KV movement during request retirement and refill;
2. support variable prompt and response lengths;
3. maintain a logical-to-physical block table per request;
4. reuse prompt KV across multiple response samples;
5. preserve independent decode state for every response branch;
6. provide explicit cache ownership and lifecycle management;
7. validate all optimizations against a sequential eager oracle.

The existing `DeepSpeedStaticCache` path must remain available for fixed-batch
and CUDA-Graph generation.

## Proposed design

```text
OPSD rollout
    |
    v
KVCacheManager
    ├── Physical block pool
    ├── Free-block queue
    ├── Request block tables
    ├── Reference counting
    └── Optional prefix index
    |
    v
Attention cache interface
```

The initial internal interface should be small:

```python
allocate(request_id, num_tokens)
append(request_id, num_tokens)
get_block_table(request_id)
free(request_id)
reset()
```

The public rollout API should remain:

```python
generate(request, sampling)
```

The paged implementation should be selected through an opt-in configuration.

## Phase 0: Baseline and profiling

Measure the existing implementations before changing the physical layout.

Compare:

1. sequential eager generation;
2. static batching;
3. continuous batching with `DeepSpeedStaticCache`;
4. shared-prefill generation from #8296.

The benchmark should report:

- prefill latency;
- decode latency;
- end-to-end rollout latency;
- useful-token throughput;
- peak allocated memory;
- reserved and used KV capacity;
- bytes moved by compaction and trimming;
- temporary prefill-cache memory;
- request retirement and refill counts;
- token agreement with sequential eager generation.

The workload should include:

- `n_samples_per_prompt = 1, 4, 8`;
- repeated and non-repeated prompts;
- variable prompt lengths;
- variable response lengths;
- staggered EOS;
- request retirement and refill.

## Phase 1: Local paged KV cache

Implement an opt-in local physical block allocator.

Required behavior:

- preallocate a pool of fixed-size KV blocks;
- allocate blocks as requests grow;
- maintain a logical-to-physical block table for every request;
- release blocks when requests retire;
- support variable prompt and response lengths;
- support staggered EOS and pending-request refill;
- prevent stale KV from being observed after block reuse;
- avoid full-row compaction during normal request retirement;
- avoid `trim_left()` during normal request retirement.

The first phase does not require prefix reuse, host offload, distributed storage,
or CUDA Graph support for dynamic block allocation.

## Phase 2: Shared prompt blocks and rollout branches

Extend the paged cache to represent the OPSD sampling pattern:

```text
prompt prefill once
        |
        +── response branch 0
        +── response branch 1
        +── response branch 2
```

Prompt blocks should be shared while response branches remain independent.

Add:

- reference counting for shared blocks;
- shared prompt blocks across response samples;
- independent response block tables;
- copy-on-write when a branch appends to a shared block;
- comparison with the existing `use_shared_prefill` implementation.

The optimized and unoptimized paths must produce identical tokens and response
masks under deterministic generation.

## Phase 3: Prefix reuse

Add optional prefix reuse within a compatible rollout and policy version.

The first implementation may use hash-based block lookup. A radix-tree index
can be evaluated later for workloads with many variable-length shared prefixes.

A prefix entry should contain:

- physical block IDs;
- prefix identity;
- reference count;
- last-access information;
- cache namespace;
- model and policy identity.

The cache identity must include all inputs that affect KV computation, including:

- model identity;
- tokenizer identity;
- adapter or LoRA identity;
- position-encoding configuration;
- prompt token IDs;
- cache namespace.

Persistent reuse across policy updates must be disabled unless compatibility is
explicitly established.

## Phase 4: Training lifecycle

Make cache lifetime and invalidation explicit at the OPSD rollout boundary.

Retained cache entries must be invalidated or moved to a new namespace after:

- a student policy update;
- checkpoint restore;
- rollback;
- adapter or LoRA change;
- tokenizer change;
- position-encoding change;
- model replacement.

Teacher and student cache entries must use separate namespaces if both models
use cached KV.

This phase must verify that reuse preserves:

- generated tokens;
- student and teacher logits;
- response masks;
- KL or JSD loss;
- reward values;
- representative short-run OPSD training metrics.

## Phase 5: Tiered and distributed cache

Evaluate tiered and distributed storage only after the local paged cache has
been validated.

Possible follow-up work includes:

```text
GPU HBM -> CPU/DRAM -> SSD/NVMe -> remote KV store
```

Potential capabilities include:

- asynchronous GPU-to-host KV movement;
- prefill/decode disaggregation;
- cross-worker KV reuse;
- topology-aware transfer;
- replication and eviction policies;
- optional Mooncake integration.

These capabilities are follow-up work and are not required for the initial
paged-cache implementation.

## Correctness contract

The sequential eager path is the reference oracle.

For every optimization, validation should include:

- exact token comparison for deterministic greedy generation;
- logit comparison with an explicitly documented tolerance;
- response-mask comparison;
- KL or JSD loss comparison;
- reward comparison where applicable;
- a representative OPSD training-step comparison.

The test matrix should cover:

- one and multiple samples per prompt;
- repeated and non-repeated prompts;
- variable prompt lengths;
- variable response lengths;
- staggered EOS;
- request retirement and refill;
- shared prompt blocks;
- copy-on-write response branches;
- physical block reuse;
- policy or checkpoint changes.

## Acceptance criteria

### Initial paged-cache implementation

- Existing `DeepSpeedStaticCache` behavior is unchanged.
- The paged cache is opt-in.
- The existing `generate(request, sampling)` API is preserved.
- Variable prompt and response lengths are supported.
- Staggered EOS and request refill are supported.
- Normal request retirement does not require full-row KV compaction.
- Reused blocks cannot expose stale KV.
- Paged-cache output matches the sequential eager token oracle.
- Benchmarks report memory use, KV movement, latency, and throughput.

### Shared prompt and branch support

- Multiple response samples share prompt KV.
- Response branches maintain independent decode state.
- Copy-on-write preserves token and mask correctness.
- Shared and non-shared paths produce equivalent outputs.
- Prompt prefill work is reduced for repeated prompts.

### Training-aware lifecycle

- Retained cache entries are invalidated or namespaced across policy updates.
- Teacher and student cache namespaces cannot alias.
- Cache reuse preserves logits, KL or JSD loss, reward, and declared training
  metrics.

## Non-goals of the initial implementation

The first implementation does not require:

- speculative decoding;
- INT8, FP8, or FP4 KV-cache quantization;
- a new attention kernel;
- RDMA or SSD deployment;
- cross-node KV storage;
- CUDA Graph support for dynamic paged allocation;
- replacing the existing shared-prefill implementation.

## Relevant implementation

- `deepspeed/utils/static_cache.py`
- `deepspeed/runtime/rollout/hybrid_engine_rollout.py`
- `deepspeed/runtime/rollout/continuous_batching.py`

## Related work

- [#8296: Share prompt prefill across rollout samples]([https://github.com/deepspeedai/DeepSpeed/pull/8296](https://github.com/deepspeedai/DeepSpeed/pull/8296))
- [#8368: Add continuous batching generation prototype]([https://github.com/deepspeedai/DeepSpeed/pull/8368](https://github.com/deepspeedai/DeepSpeed/pull/8368))

## References

- [vLLM KV cache manager]([https://docs.vllm.ai/en/stable/api/vllm/v1/core/kv_cache_manager/](https://docs.vllm.ai/en/stable/api/vllm/v1/core/kv_cache_manager/))
- [https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
- [SGLang RadixAttention]([https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention](https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention))
- [Mooncake distributed KV cache]([https://github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake))
- [PagedAttention]([https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180))
```
