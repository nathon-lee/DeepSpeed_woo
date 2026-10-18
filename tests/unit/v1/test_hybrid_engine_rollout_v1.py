# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Accelerator-backed v1 HybridEngineRollout tests."""

import os
from types import SimpleNamespace

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig


def test_continuous_generation_profile_on_accelerator():
    accelerator = get_accelerator()
    if not accelerator.is_available():
        pytest.skip("An accelerator is required for asynchronous profiling coverage")

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            # StaticCache on current Transformers main reads this before choosing layer types.
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            _, values = past_key_values.update(states, states, layer_idx=0, cache_kwargs=kwargs)
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16), device=input_ids.device)
            logits[..., 7] = 1
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    device = torch.device(accelerator.device_name())
    model = CacheClassModel().to(device)
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(enable_profiling=True),
    )
    request = RolloutRequest(
        torch.tensor([[1, 2, 3], [1, 2, 4]], device=device),
        torch.ones((2, 3), dtype=torch.long, device=device),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))

    profile = rollout.get_last_profile()
    assert output.input_ids[:, 3:].cpu().tolist() == [[7, 7], [7, 7]]
    assert profile["num_prefill_forwards"] == 2
    assert profile["num_decode_forwards"] == 2
    assert profile["num_generated_tokens"] == 4
    assert profile["active_batch_size"] == 1
    assert profile["continuous_batch_size"] == 1
    assert profile["total_ms"] >= profile["generation_ms"]


@pytest.mark.parametrize("buckets", [None, (4, 6)])
def test_continuous_graph_generation_refills_a_fixed_slot_on_cuda(buckets):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for CUDA graph rollout coverage")

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            _, values = past_key_values.update(states, states, layer_idx=0, cache_kwargs=kwargs)
            cache_sums = values[:, 0].sum(dim=(1, 2)).long()
            if input_ids.shape[1] > 1:
                next_tokens = torch.where(cache_sums == 6, 2, 7)
            else:
                next_tokens = torch.where((cache_sums == 14) | (cache_sums == 16), 2, 8)
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16), device=input_ids.device)
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    device = torch.device(accelerator.device_name())
    graph_rollout = HybridEngineRollout(
        SimpleNamespace(module=CacheClassModel().to(device)),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(use_graph_capture=True,
                                      continuous_graph_pool_size=2,
                                      continuous_prompt_buckets=buckets),
    )
    eager_rollout = HybridEngineRollout(
        SimpleNamespace(module=CacheClassModel().to(device)),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
    )
    request = RolloutRequest(
        torch.tensor([[1, 2, 3], [1, 2, 4], [3, 3, 3]], device=device),
        torch.ones((3, 3), dtype=torch.long, device=device),
    )

    sampling = SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=2)
    for width, captures in ([(3, 1)] if buckets is None else [(3, 1), (5, 1), (4, 0), (3, 0)]):
        padded_request = RolloutRequest(
            torch.nn.functional.pad(request.prompt_ids, (width - 3, 0)),
            torch.nn.functional.pad(request.prompt_attention_mask, (width - 3, 0)),
        )
        eager_output = eager_rollout.generate(padded_request, sampling)
        graph_output = graph_rollout.generate(padded_request, sampling)

        assert torch.equal(graph_output.input_ids, eager_output.input_ids)
        assert torch.equal(graph_output.attention_mask, eager_output.attention_mask)
        assert torch.equal(graph_output.response_start_idx, eager_output.response_start_idx)
        assert graph_output.input_ids[:, width:].cpu().tolist() == [[2, 0], [7, 2], [7, 2]]
        assert graph_rollout.get_last_continuous_stats()["graph_captures"] == captures
    stats = graph_rollout.get_last_continuous_stats()
    assert stats["cache_capacity"] == (5 if buckets is None else 6)
    assert stats["peak_cache_length"] == (4 if buckets is None else 5)
    assert stats["decode_steps"] == 2
    assert stats["trim_count"] == 0
    assert stats["trim_frequency"] == 0.0


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_continuous_graph_buckets_match_native_llama_after_training(attention):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for CUDA graph rollout coverage")

    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1234)
    device = torch.device(accelerator.current_device_name())
    config = LlamaConfig(vocab_size=64,
                         hidden_size=32,
                         intermediate_size=64,
                         num_hidden_layers=2,
                         num_attention_heads=2,
                         num_key_value_heads=1,
                         max_position_embeddings=64,
                         pad_token_id=0,
                         eos_token_id=None,
                         attention_dropout=0.0)
    config._attn_implementation = attention
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.float16).eval()
    model.generation_config.repetition_penalty = 1.2
    engine = SimpleNamespace(module=model)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
    eager = HybridEngineRollout(engine, tokenizer)
    graph = HybridEngineRollout(
        engine,
        tokenizer,
        HybridEngineRolloutConfig(use_graph_capture=True,
                                  continuous_graph_pool_size=2,
                                  continuous_prompt_buckets=(4, 6, 8)),
    )
    sampling = SamplingConfig(max_new_tokens=4, temperature=0, continuous_batch_size=2)
    for width, captures in [(3, 1), (5, 1), (4, 0), (3, 0), (7, 1), (5, 1)]:
        prompt_ids = torch.randint(1, config.vocab_size, (4, width), device=device)
        prompt_ids[1, :2] = 0
        request = RolloutRequest(prompt_ids, (prompt_ids != 0).long())
        with torch.no_grad():
            expected = model.generate(prompt_ids,
                                      attention_mask=request.prompt_attention_mask,
                                      max_new_tokens=4,
                                      do_sample=False,
                                      eos_token_id=None,
                                      pad_token_id=0)
        for rollout in (eager, graph):
            output = rollout.generate(request, sampling)
            assert torch.equal(output.input_ids, expected)
            assert torch.equal(output.attention_mask[:, :width], request.prompt_attention_mask)
            assert output.attention_mask[:, width:].eq(1).all()
            assert output.response_start_idx.eq(width).all()
        assert graph.get_last_continuous_stats()["graph_captures"] == captures
        assert graph.get_last_continuous_stats()["graph_pool_size"] <= 2

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    model(prompt_ids, attention_mask=request.prompt_attention_mask, labels=prompt_ids, use_cache=False).loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    # Both retained buckets must recapture, not just the first one after SGD.
    for width in (5, 7):
        prompt_ids = torch.ones((4, width), dtype=torch.long, device=device)
        request = RolloutRequest(prompt_ids, torch.ones_like(prompt_ids))
        expected = eager.generate(request, sampling)
        actual = graph.generate(request, sampling)
        assert torch.equal(actual.input_ids, expected.input_ids)
        assert torch.equal(actual.attention_mask, expected.attention_mask)
        assert torch.equal(actual.response_start_idx, expected.response_start_idx)
        assert graph.get_last_continuous_stats()["graph_captures"] == 1

    model.config.max_position_embeddings = 10
    with pytest.raises(ValueError, match="model maximum position"):
        graph.generate(request, sampling)


def test_continuous_graph_matches_pretrained_model_when_enabled():
    model_name = os.getenv("DEEPSPEED_PRETRAINED_ROLLOUT_MODEL")
    if not model_name:
        pytest.skip("set DEEPSPEED_PRETRAINED_ROLLOUT_MODEL to run pretrained rollout coverage")

    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for pretrained CUDA graph rollout coverage")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(accelerator.device_name())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eager_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
    graph_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
    graph_model.load_state_dict(eager_model.state_dict())

    sampling = SamplingConfig(max_new_tokens=32, temperature=0, continuous_batch_size=1)
    eager_rollout = HybridEngineRollout(SimpleNamespace(module=eager_model), tokenizer)
    graph_rollout = HybridEngineRollout(
        SimpleNamespace(module=graph_model),
        tokenizer,
        cfg=HybridEngineRolloutConfig(use_graph_capture=True),
    )
    original_create_graph = accelerator.create_graph
    capture_count = [0]

    def counted_create_graph():
        capture_count[0] += 1
        return original_create_graph()

    accelerator.create_graph = counted_create_graph
    try:
        prompts = (
            "CUDA Graph replay avoids repeated Python launch overhead during autoregressive decoding.",
            "DeepSpeed continuous batching keeps physical cache slots stable across request refills.",
        )
        for seed, prompt in zip((1234, 1235), prompts):
            torch.manual_seed(seed)
            encoded = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
            request = RolloutRequest(encoded.input_ids.to(device), encoded.attention_mask.to(device))
            eager_output = eager_rollout.generate(request, sampling)
            graph_output = graph_rollout.generate(request, sampling)

            assert torch.equal(graph_output.input_ids, eager_output.input_ids)
            assert torch.equal(graph_output.attention_mask, eager_output.attention_mask)
    finally:
        accelerator.create_graph = original_create_graph

    assert capture_count[0] == 1
