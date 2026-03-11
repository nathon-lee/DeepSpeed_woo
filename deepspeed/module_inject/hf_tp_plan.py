# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""Helpers for deriving AutoTP configuration from HuggingFace tp_plan metadata.

This module intentionally keeps the conversion logic lightweight and tolerant of
different HuggingFace plan object shapes. It normalizes a model's tp-plan-like
metadata into ``TPLayerSpec`` entries that can be consumed by DeepSpeed's
config-driven AutoTP path.

The current adapter focuses on common partition metadata:
- partition direction (row / column / skip)
- optional partition dimension
- optional packed or sub-parameter shapes

Unknown or unsupported plan entries are ignored rather than treated as errors.
This keeps the integration usable across models that only partially expose the
metadata DeepSpeed needs.
"""

from typing import Any, Dict, Iterable, Optional, Sequence

from .autotp_config import AutoTPConfig, PartitionType, TPLayerSpec


_ROW_PLAN_KEYWORDS = (
    "rowwise",
    "row-wise",
    "row_parallel",
    "row-parallel",
    "replicate_on_output",
)

_COLUMN_PLAN_KEYWORDS = (
    "colwise",
    "columnwise",
    "col-wise",
    "column-wise",
    "col_parallel",
    "column_parallel",
    "col-parallel",
    "column-parallel",
)

_SKIP_PLAN_KEYWORDS = (
    "replicated",
    "replicate",
    "copy",
    "skip",
    "none",
)


def _normalize_tp_plan_entries(tp_plan: Any) -> Dict[str, Any]:
    """Return a normalized mapping of parameter name to plan descriptor."""
    if tp_plan is None:
        return {}

    if isinstance(tp_plan, dict):
        return dict(tp_plan)

    if isinstance(tp_plan, (list, tuple)):
        normalized = {}
        for item in tp_plan:
            if isinstance(item, tuple) and len(item) == 2:
                normalized[item[0]] = item[1]
        return normalized

    return {}


def _extract_model_type(model: Any) -> Optional[str]:
    """Best-effort extraction of the HuggingFace ``model_type`` string."""
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if model_type:
            return str(model_type).lower()
    return None


def _normalize_patterns(param_name: str) -> Iterable[str]:
    """Generate anchored regexes for matching a concrete parameter name."""
    escaped = str(param_name).replace(".", r"\.")
    yield rf".*\.{escaped}$"
    yield rf"^{escaped}$"


def _get_plan_value(plan_spec: Any, *names: str) -> Any:
    """Read the first matching field from a dict-like or attribute-based plan object."""
    if plan_spec is None:
        return None

    if isinstance(plan_spec, dict):
        for name in names:
            if name in plan_spec:
                return plan_spec[name]
        return None

    for name in names:
        if hasattr(plan_spec, name):
            return getattr(plan_spec, name)
    return None


def _normalize_shape(shape: Any):
    """Convert list-based shapes into tuples expected by ``TPLayerSpec``."""
    if shape is None:
        return None
    if isinstance(shape, tuple):
        return tuple(_normalize_shape(item) if isinstance(item, list) else item for item in shape)
    if isinstance(shape, list):
        return tuple(_normalize_shape(item) if isinstance(item, (list, tuple)) else item for item in shape)
    return shape


def _coerce_partition_dim(plan_spec: Any) -> Optional[int]:
    """Extract an integer partition dimension if the plan provides one."""
    partition_dim = _get_plan_value(plan_spec, "partition_dim", "dim", "shard_dim", "tp_dim")
    if partition_dim is None:
        return None
    try:
        return int(partition_dim)
    except (TypeError, ValueError):
        return None


def _coerce_shape(plan_spec: Any):
    """Read an explicit logical/view shape from the plan when available."""
    shape = _get_plan_value(plan_spec, "shape", "logical_shape", "view_shape", "partition_shape", "tp_shape")
    return _normalize_shape(shape)


def _coerce_num_splits(plan_spec: Any) -> Optional[int]:
    """Read equal-sized packing metadata such as QKV split count."""
    num_splits = _get_plan_value(plan_spec, "num_splits", "split_count", "num_partitions", "packed")
    if isinstance(num_splits, bool):
        return 2 if num_splits else None
    try:
        return int(num_splits) if num_splits is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_subparam_sizes(plan_spec: Any):
    """Read uneven packed sub-parameter sizes when the plan exposes them."""
    sub_param_sizes = _get_plan_value(plan_spec, "sub_param_sizes", "split_sizes", "sizes", "chunks")
    if not isinstance(sub_param_sizes, Sequence) or isinstance(sub_param_sizes, (str, bytes)):
        return None
    normalized = []
    for item in sub_param_sizes:
        try:
            normalized.append(int(item))
        except (TypeError, ValueError):
            return None
    return tuple(normalized) if normalized else None


def _build_shape_from_plan(plan_spec: Any, partition_type: PartitionType):
    """Infer a ``TPLayerSpec.shape`` from explicit or packed tp-plan metadata."""
    explicit_shape = _coerce_shape(plan_spec)
    if explicit_shape is not None:
        return explicit_shape

    partition_dim = _coerce_partition_dim(plan_spec)
    sub_param_sizes = _coerce_subparam_sizes(plan_spec)
    if sub_param_sizes is not None:
        if partition_dim is None:
            partition_dim = 0 if partition_type == PartitionType.COLUMN else 1
        if partition_dim == 0:
            return (tuple(sub_param_sizes), -1)
        if partition_dim == 1:
            return (-1, tuple(sub_param_sizes))

    num_splits = _coerce_num_splits(plan_spec)
    if num_splits is not None and num_splits > 1:
        if partition_dim is None:
            partition_dim = 0 if partition_type == PartitionType.COLUMN else 1
        if partition_dim == 0:
            return (num_splits, -1)
        if partition_dim == 1:
            return (-1, num_splits)

    return None


def _coerce_partition_type(plan_spec: Any) -> Optional[PartitionType]:
    """Map HuggingFace plan descriptors onto DeepSpeed partition semantics."""
    if plan_spec is None:
        return None

    skip_flag = _get_plan_value(plan_spec, "skip", "replicated", "is_replicated", "replicate")
    if skip_flag is True:
        return PartitionType.SKIP

    if isinstance(plan_spec, str):
        spec_name = plan_spec.lower()
    else:
        spec_name = getattr(plan_spec, "style", None) or getattr(plan_spec, "type", None) or getattr(
            plan_spec, "partition", None)
        if spec_name is None:
            spec_name = plan_spec.__class__.__name__
        spec_name = str(spec_name).lower()

    if any(keyword in spec_name for keyword in _SKIP_PLAN_KEYWORDS):
        return PartitionType.SKIP

    if any(keyword in spec_name for keyword in _ROW_PLAN_KEYWORDS):
        return PartitionType.ROW
    if any(keyword in spec_name for keyword in _COLUMN_PLAN_KEYWORDS):
        return PartitionType.COLUMN
    return None


def hf_model_tp_plan_to_config(model: Any, *, tp_size: int = 1) -> Optional[AutoTPConfig]:
    """Build an ``AutoTPConfig`` from a HuggingFace model's tp-plan metadata.

    The adapter looks for ``base_model_tp_plan`` first and then ``_tp_plan``.
    Entries are interpreted conservatively:

    - only ``*.weight`` parameters are converted into layer specs
    - unknown plan kinds are skipped
    - optional shape and partition-dimension metadata is preserved when present

    Returns ``None`` when no usable tp-plan entries are found.
    """

    tp_plan = getattr(model, "base_model_tp_plan", None)
    if tp_plan is None:
        tp_plan = getattr(model, "_tp_plan", None)

    entries = _normalize_tp_plan_entries(tp_plan)
    if not entries:
        return None

    model_type = _extract_model_type(model)
    layer_specs = []
    seen_param_names = set()

    for param_name, plan_spec in entries.items():
        partition_type = _coerce_partition_type(plan_spec)
        if partition_type is None:
            continue

        if not isinstance(param_name, str) or not param_name.endswith(".weight"):
            continue
        if param_name in seen_param_names:
            continue

        partition_dim = _coerce_partition_dim(plan_spec)
        shape = _build_shape_from_plan(plan_spec, partition_type)

        layer_specs.append(
            TPLayerSpec(patterns=list(_normalize_patterns(param_name)),
                        partition_type=partition_type,
                        shape=shape,
                        partition_dim=partition_dim,
                        model_types=[model_type] if model_type else None))
        seen_param_names.add(param_name)

    if not layer_specs:
        return None

    return AutoTPConfig(tp_size=tp_size, layer_specs=layer_specs, use_default_specs=False)
