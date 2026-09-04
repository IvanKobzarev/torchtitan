# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING

from torchtitan.experiments.graph_trainer.graph_pp.pipeline import graph_pipeline_llm
from torchtitan.models.common.config_utils import MoeBackend
from torchtitan.models.deepseek_v3 import model_registry as deepseek_v3_model_registry
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraphTrainerDeepSeekV3Model
from .parallelize import parallelize_deepseekv3

if TYPE_CHECKING:
    from torchtitan.models.common.dist_moe import DistMoeBackendConfig


def _parallelize_fn(model, *, compile_config, **kwargs):
    if compile_config.enable_autoparallel:
        from .parallelize_autoparallel import parallelize_autoparallel_deepseekv3

        return parallelize_autoparallel_deepseekv3(
            model, compile_config=compile_config, **kwargs
        )
    return parallelize_deepseekv3(model, compile_config=compile_config, **kwargs)


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_backend: MoeBackend = "standard",
    moe_comm_backend: str = "standard",
    dist_moe: DistMoeBackendConfig | None = None,
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    """Build a GraphTrainer DeepSeek V3 model specification.

    Args:
        flavor: Registered DeepSeek V3 model flavor.
        attn_backend: Attention implementation, including test-only SDPA.
        moe_backend: Routed-expert implementation.
        moe_comm_backend: Communication backend for standard routed experts.
        dist_moe: Advanced DistMoE execution policy.
        non_blocking_capacity_factor: Capacity for non-blocking standard EP.
        converters: Optional model-config conversions applied before wrapping
            the GraphTrainer model type.

    Returns:
        GraphTrainer model spec retaining the base backend lifecycle hooks.
    """
    base_spec = deepseek_v3_model_registry(
        flavor,
        attn_backend="flex" if attn_backend == "sdpa" else attn_backend,
        moe_backend=moe_backend,
        moe_comm_backend=moe_comm_backend,
        dist_moe=dist_moe,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        converters=converters,
    )
    base = base_spec.model
    if attn_backend == "sdpa":
        from torchtitan.models.common.attention import ScaledDotProductAttention

        for layer in base.layers:
            layer.attention.inner_attention = ScaledDotProductAttention.Config()
    config = GraphTrainerDeepSeekV3Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return replace(
        base_spec,
        name="graph_trainer/deepseek_v3",
        model=config,
        parallelize_fn=_parallelize_fn,
        pipelining_fn=graph_pipeline_llm,
    )
