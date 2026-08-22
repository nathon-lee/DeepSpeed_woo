# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine backed by DeepSpeed's hybrid engine.

Generation delegates to HuggingFace ``generate``. When graph capture is
enabled, the response length is pinned so HybridEngine can safely replay its
native per-position decode graphs.
"""

import time
from dataclasses import dataclass
from numbers import Real

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig


@dataclass
class HybridEngineRolloutConfig:
    """Configuration for HybridEngineRollout."""
    use_graph_capture: bool = False
    enable_profiling: bool = False


class HybridEngineRollout(RolloutEngine):
    """Rollout engine using DeepSpeed hybrid engine.

    Args:
        engine: DeepSpeed engine wrapping the model.
        tokenizer: HuggingFace tokenizer (must have pad_token_id or eos_token_id).
        cfg: Optional HybridEngineRolloutConfig.
    """

    def __init__(self, engine, tokenizer, cfg=None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.use_graph_capture = getattr(cfg, 'use_graph_capture', False) if cfg else False
        self.enable_profiling = getattr(cfg, 'enable_profiling', False) if cfg else False
        self._last_profile = None

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        device = request.prompt_ids.device
        B = request.prompt_ids.shape[0]
        n = sampling.n_samples_per_prompt
        total = B * n
        prompt_len = request.prompt_ids.shape[1]
        max_new_tokens = sampling.max_new_tokens
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        module = self.engine.module

        if self.enable_profiling:
            accelerator = get_accelerator()
            accelerator.synchronize()
            profile_start = time.perf_counter()

        # Expand prompts for n samples per prompt
        if n > 1:
            prompt_ids = request.prompt_ids.repeat_interleave(n, dim=0)
            prompt_attn = request.prompt_attention_mask.repeat_interleave(n, dim=0)
        else:
            prompt_ids = request.prompt_ids
            prompt_attn = request.prompt_attention_mask

        if self.enable_profiling:
            accelerator.synchronize()
            expansion_end = time.perf_counter()

        is_greedy = sampling.temperature <= 0.0

        temperature = max(sampling.temperature, 1e-8)
        do_sample = not is_greedy
        generation_kwargs = {
            "attention_mask": prompt_attn,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "top_p": sampling.top_p if do_sample else 1.0,
            "pad_token_id": pad_token_id,
        }
        if self.use_graph_capture:
            generation_kwargs["min_new_tokens"] = max_new_tokens
        self.engine._profile_generation_phases = self.enable_profiling
        try:
            output_ids = module.generate(prompt_ids, **generation_kwargs)
        finally:
            self.engine._profile_generation_phases = False

        if self.enable_profiling:
            accelerator.synchronize()
            generation_end = time.perf_counter()

        # Build attention mask: pad positions (both left padding from prompt
        # and right padding from EOS / shorter sequences) are 0.
        response_start = prompt_len
        attention_mask = (output_ids != pad_token_id).long()
        for i in range(total):
            prompt_valid = request.prompt_attention_mask[i // n if B > 1 else 0]
            attention_mask[i, :prompt_len] = prompt_valid

        rollout_batch = RolloutBatch(
            input_ids=output_ids,
            attention_mask=attention_mask,
            response_start_idx=torch.full((total, ), response_start, dtype=torch.long, device=device),
        )

        if self.enable_profiling:
            accelerator.synchronize()
            post_processing_end = time.perf_counter()
            prompt_expansion_ms = (expansion_end - profile_start) * 1000.0
            generation_ms = (generation_end - expansion_end) * 1000.0
            post_processing_ms = (post_processing_end - generation_end) * 1000.0
            total_ms = (post_processing_end - profile_start) * 1000.0
            response_length = int(output_ids.shape[1] - prompt_len)
            num_generated_tokens = int(output_ids.shape[0] * response_length)
            tokens_per_second = 0.0
            if total_ms > 0.0:
                tokens_per_second = num_generated_tokens / (total_ms / 1000.0)
            self._last_profile = {
                "prompt_expansion_ms": prompt_expansion_ms,
                "generation_ms": generation_ms,
                "post_processing_ms": post_processing_ms,
                "total_ms": total_ms,
                "num_generated_tokens": num_generated_tokens,
                "tokens_per_second": tokens_per_second,
                "batch_size": B,
                "num_samples_per_prompt": n,
                "prompt_length": prompt_len,
                "response_length": response_length,
            }
            for source_name, profile_name in (
                    ("_cache_retake_latency", "cache_retake_ms"),
                    ("_model_generation_latency", "model_generation_ms"),
                    ("_prefill_latency", "prefill_ms"),
                    ("_decode_latency", "decode_ms"),
                    ("_cache_release_latency", "cache_release_ms"),
                    ("_workspace_release_latency", "workspace_release_ms"),
                    ("_gc_collect_latency", "gc_collect_ms"),
                    ("_empty_cache_latency", "empty_cache_ms")):
                value = getattr(self.engine, source_name, None)
                if isinstance(value, Real):
                    self._last_profile[profile_name] = value * 1000.0
            memory_before = getattr(self.engine, "_memory_before_release", (0, 0))
            memory_after = getattr(self.engine, "_memory_after_release", (0, 0))
            if (isinstance(memory_before, tuple) and isinstance(memory_after, tuple)
                    and all(isinstance(value, Real) for value in (*memory_before, *memory_after))
                    and (memory_before != (0, 0) or memory_after != (0, 0))):
                self._last_profile.update({
                    "memory_allocated_before_release_mb": memory_before[0] / (1024**2),
                    "memory_reserved_before_release_mb": memory_before[1] / (1024**2),
                    "memory_allocated_after_release_mb": memory_after[0] / (1024**2),
                    "memory_reserved_after_release_mb": memory_after[1] / (1024**2),
                })

        return rollout_batch

    def get_last_profile(self):
        """Return the most recent profiling snapshot for this rollout instance."""
        return self._last_profile

    @staticmethod
    def _sample_top_p(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
        """Sample from logits with temperature and nucleus (top-p) filtering."""
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = (cumulative_probs - torch.softmax(sorted_logits, dim=-1)) >= top_p
            sorted_logits[mask] = -float('inf')
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            tokens = sorted_indices.gather(1, sampled)
        else:
            probs = torch.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs, 1)
        return tokens

    def sync_weights(self, step: int) -> None:  # noqa: ARG002
        """No-op: hybrid engine reads model weights live."""
        return None
