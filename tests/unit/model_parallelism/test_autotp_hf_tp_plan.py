# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

from deepspeed.module_inject.autotp_config import PartitionType
from deepspeed.module_inject.hf_tp_plan import hf_model_tp_plan_to_config
from deepspeed.runtime.tensor_parallel.config import TPTrainingConfig


class _MockColwiseParallel:
    pass


class _MockRowwiseParallel:
    pass


class _MockPackedColwiseParallel:

    def __init__(self):
        self.partition = "colwise"
        self.num_splits = 3
        self.partition_dim = 0


class _MockUnevenPackedColwiseParallel:

    def __init__(self):
        self.partition = "columnwise"
        self.partition_dim = 0
        self.sub_param_sizes = [8, 4, 4]


class _MockReplicatedPlan:

    def __init__(self):
        self.partition = "replicated"


class DummyHFModel:

    def __init__(self, tp_plan=None, model_type="llama"):
        self.base_model_tp_plan = tp_plan
        self.config = SimpleNamespace(model_type=model_type)


def test_hf_tp_plan_to_config_builds_layer_specs():
    model = DummyHFModel(tp_plan={
        "model.layers.0.self_attn.q_proj.weight": _MockColwiseParallel(),
        "model.layers.0.self_attn.o_proj.weight": _MockRowwiseParallel(),
        "model.layers.0.self_attn.q_proj.bias": _MockColwiseParallel(),
    })

    config = hf_model_tp_plan_to_config(model, tp_size=2)

    assert config is not None
    assert config.tp_size == 2
    assert len(config.layer_specs) == 2
    q_spec = config.find_matching_spec("model.layers.0.self_attn.q_proj.weight", "llama")
    o_spec = config.find_matching_spec("model.layers.0.self_attn.o_proj.weight", "llama")
    assert q_spec.partition_type == PartitionType.COLUMN
    assert o_spec.partition_type == PartitionType.ROW


def test_hf_tp_plan_merges_with_user_partition_config():
    model = DummyHFModel(tp_plan={"model.layers.0.self_attn.o_proj.weight": _MockRowwiseParallel()})
    tp_config = TPTrainingConfig(autotp_size=4,
                                 partition_config={
                                     "use_default_specs": True,
                                     "layer_specs": [{
                                         "patterns": [r".*\\.mlp\\.up_proj\\.weight$"],
                                         "partition_type": "column",
                                     }],
                                 })

    config = tp_config.get_partition_config_object(model=model)

    assert config is not None
    assert config.tp_size == 4
    assert config.find_matching_spec("model.layers.0.self_attn.o_proj.weight", "llama").partition_type == PartitionType.ROW
    assert config.find_matching_spec("model.layers.0.mlp.up_proj.weight", "llama").partition_type == PartitionType.COLUMN


def test_hf_tp_plan_ignores_unknown_plan_kinds():
    model = DummyHFModel(tp_plan={"model.layers.0.self_attn.q_proj.weight": object()})

    config = hf_model_tp_plan_to_config(model, tp_size=2)

    assert config is None


def test_hf_tp_plan_extracts_equal_sized_packed_shape():
    model = DummyHFModel(tp_plan={"model.layers.0.self_attn.qkv_proj.weight": _MockPackedColwiseParallel()})

    config = hf_model_tp_plan_to_config(model, tp_size=2)

    assert config is not None
    spec = config.find_matching_spec("model.layers.0.self_attn.qkv_proj.weight", "llama")
    assert spec.partition_type == PartitionType.COLUMN
    assert spec.partition_dim == 0
    assert spec.shape == (3, -1)


def test_hf_tp_plan_extracts_uneven_subparam_shape():
    model = DummyHFModel(tp_plan={"model.layers.0.self_attn.qkv_proj.weight": _MockUnevenPackedColwiseParallel()})

    config = hf_model_tp_plan_to_config(model, tp_size=2)

    assert config is not None
    spec = config.find_matching_spec("model.layers.0.self_attn.qkv_proj.weight", "llama")
    assert spec.partition_type == PartitionType.COLUMN
    assert spec.partition_dim == 0
    assert spec.shape == ((8, 4, 4), -1)


def test_hf_tp_plan_can_mark_layers_as_skip():
    model = DummyHFModel(tp_plan={"model.layers.0.mlp.gate.weight": _MockReplicatedPlan()})

    config = hf_model_tp_plan_to_config(model, tp_size=2)

    assert config is not None
    spec = config.find_matching_spec("model.layers.0.mlp.gate.weight", "llama")
    assert spec.partition_type == PartitionType.SKIP
