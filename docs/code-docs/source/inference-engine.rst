Inference API
=============

:func:`deepspeed.init_inference` returns an *inference engine*
of type :class:`InferenceEngine`.

.. code-block:: python

    for step, batch in enumerate(data_loader):
        #forward() method
        loss = engine(batch)

Forward Propagation
-------------------
.. autofunction:: deepspeed.InferenceEngine.forward

HybridEngine Rollout Profiling
------------------------------

``HybridEngineRollout`` can record synchronized stage timings for a rollout.
Profiling is disabled by default because synchronization changes execution
behavior and adds overhead. Enable it through ``HybridEngineRolloutConfig``::

    from deepspeed.runtime.rollout.hybrid_engine_rollout import (
        HybridEngineRollout,
        HybridEngineRolloutConfig,
    )

    rollout = HybridEngineRollout(
        engine,
        tokenizer,
        cfg=HybridEngineRolloutConfig(enable_profiling=True),
    )
    output = rollout.generate(request, sampling)
    profile = rollout.get_last_profile()

The profile contains synchronized times for prompt expansion, generation,
post-processing, and the complete rollout. Times are reported in milliseconds.
When the underlying engine exposes HybridEngine cache instrumentation, the
profile also includes ``cache_retake_ms``, ``model_generation_ms``, and
``cache_release_ms``. These values isolate workspace acquisition, model
generation, and cache release from the rollout-level timings.
When phase profiling is enabled, ``prefill_ms`` and ``decode_ms`` split
``model_generation_ms`` into the initial prompt forward and subsequent token
forwards. Use ``--no-generation-phase-profiling`` in the OPSD benchmark to
omit those per-forward synchronization points when comparing end-to-end
performance.
For released caches, ``workspace_release_ms``, ``gc_collect_ms``, and
``empty_cache_ms`` further break down cache cleanup. The profile also records
allocated and reserved memory immediately before and after cache release.
``num_generated_tokens`` counts all returned response positions across the
expanded batch, including padding positions. ``tokens_per_second`` divides
that count by the end-to-end rollout time. The profile also records the input
batch size, samples per prompt, prompt length, and returned response length.
The OPSD benchmark accepts ``--use-graph-capture`` to compare greedy decode
against its default HuggingFace generation path. Each benchmark case records
``cuda_graph_captured_positions`` so a capture failure and eager fallback are
visible in the output.
Use ``--torch-profile-output TRACE.json`` with a single benchmark case to
capture one additional rollout after warmup. The profiler trace is excluded
from benchmark summary statistics, automatically disables generation phase
hooks, and writes the top operator table alongside the trace as
``TRACE.summary.txt``.
