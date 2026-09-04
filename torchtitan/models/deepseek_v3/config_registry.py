# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from typing import Literal, TYPE_CHECKING

import torch

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.quantization import (
    Float8GroupedExpertsConverter,
    Float8LinearConverter,
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import VarlenAttention
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.router_gate import (
    RouterGateLinearConverter,
    RouterGemmConfig,
)
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.trainer import Trainer

from . import model_registry
from .model import Attention

if TYPE_CHECKING:
    from torchtitan.models.common.dist_moe import DistMoeBackendConfig


_DIST_MOE_VARLEN_MAX_NUM_DOCUMENTS = 512


def _mxfp8_dist_moe_backend(
    *,
    max_routing_imbalance_factor: float = 1.0,
    device_memory_budget_bytes: int | Literal["maximum_useful"] | None = None,
    vmm_host_scratch_imbalance_factor: float | Literal["auto"] | None = "auto",
) -> "DistMoeBackendConfig":
    """Build the production MXFP8 E4M3 DistMoE policy."""
    # pyrefly: ignore [missing-import]
    from dist_moe import BlockScaledFormat, DistMoeBlockScaledConfig

    from torchtitan.models.common.dist_moe import DistMoeBackendConfig

    return DistMoeBackendConfig(
        max_routing_imbalance_factor=max_routing_imbalance_factor,
        device_memory_budget_bytes=device_memory_budget_bytes,
        vmm_host_scratch_imbalance_factor=vmm_host_scratch_imbalance_factor,
        blockscaled=DistMoeBlockScaledConfig(
            format=BlockScaledFormat.MXFP8_E4M3,
            fast_math=True,
            pipeline="staged",
        ),
    )


def _mxfp8_dense_converters() -> list[ModelConfigConverter.Config]:
    """Return MXFP8 converters for the non-routed DeepSeek GEMMs."""
    return [
        MXFP8LinearConverter.Config(
            model_compile_enabled=False,
            fqns=["attention", "shared_experts", "feed_forward", "lm_head"],
        )
    ]


def _dist_moe_converters(*, quantize_dense: bool) -> list[ModelConfigConverter.Config]:
    """Return the router and optional dense-MXFP8 converters for DistMoE."""
    converters: list[ModelConfigConverter.Config] = [
        RouterGateLinearConverter.Config(
            forward=RouterGemmConfig(
                input_dtype=torch.bfloat16,
                compute_mode="bf16",
                output_dtype=torch.float32,
            ),
            backward=RouterGemmConfig(
                input_dtype=torch.float32,
                compute_mode="tf32",
                output_dtype=torch.float32,
            ),
        )
    ]
    if quantize_dense:
        converters.extend(_mxfp8_dense_converters())
    return converters


def _apply_dist_moe_mixed_precision(config: Trainer.Config) -> None:
    """Apply the DistMoE mixed-precision and optimizer-state policy."""
    config.training.dtype = "float32"
    config.training.mixed_precision_param = "bfloat16"
    config.training.mixed_precision_reduce = "bfloat16"
    config.optimizer.implementation = "fused_opt_states_bf16"


def attention_batch_size(config: Trainer.Config) -> int:
    """Return the number of fixed-length rows in one local microbatch."""
    num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    max_context_length = config.training.max_context_length
    if num_tokens % max_context_length:
        raise ValueError(
            "num_tokens_per_microbatch_per_dp_rank must divide evenly by "
            "max_context_length for fixed-row attention"
        )
    return num_tokens // max_context_length


def _set_varlen_max_num_documents(
    config: Trainer.Config,
    max_num_documents: int = _DIST_MOE_VARLEN_MAX_NUM_DOCUMENTS,
) -> None:
    """Set the fixed document capacity on every varlen attention layer."""
    assert config.model_spec is not None
    for _, attention, _, _ in config.model_spec.model.traverse(VarlenAttention.Config):
        assert isinstance(attention, VarlenAttention.Config)
        attention.max_num_documents = max_num_documents


def enable_mlperf_packing(config: Trainer.Config) -> None:
    """Use continuous positions so every packed row is one causal document."""
    dataloader = config.dataloader
    if not isinstance(dataloader, GrainDataLoader.Config) or not isinstance(
        dataloader.dataset, ConcatThenSplitPackingConfig
    ):
        raise ValueError("MLPerf packing requires Grain ConcatThenSplit packing")
    dataloader.dataset = replace(
        dataloader.dataset,
        mask_document_boundaries=False,
    )
    max_num_documents = attention_batch_size(config)
    assert config.model_spec is not None
    for _, attention, _, _ in config.model_spec.model.traverse(VarlenAttention.Config):
        assert isinstance(attention, VarlenAttention.Config)
        attention.max_num_documents = max_num_documents
        attention.single_document_rows = True


def _enable_dist_moe(
    config: Trainer.Config,
    flavor: str,
    backend: "DistMoeBackendConfig",
    *,
    quantize_dense: bool = False,
) -> Trainer.Config:
    """Configure one standard Trainer DistMoE experiment."""
    config.training.disable_cuda_graphs = False
    config.compile.enable = False
    config.model_spec = model_registry(
        flavor,
        attn_backend="varlen",
        moe_backend="dist_moe",
        dist_moe=backend,
        converters=_dist_moe_converters(quantize_dense=quantize_dense),
    )
    _set_varlen_max_num_documents(config)
    enable_fused_mla(config)
    _apply_dist_moe_mixed_precision(config)
    return config


def enable_fused_swiglu(config: Trainer.Config) -> None:
    # fused_swiglu.py registers two overrides (dense FeedForward + MoE grouped
    # experts); activate both by naming each factory.
    for override in (
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
    ):
        assert override not in config.override.imports
        config.override.imports.append(override)


def fused_mla_query_projection(config: Trainer.Config) -> str:
    """Return the query projection mutated by fused Q RoPE.

    Args:
        config: DeepSeek-V3 trainer configuration.

    Returns:
        Module FQN of the query projection mutated by fused Q RoPE.
    """
    assert config.model_spec is not None
    q_lora_ranks = set()
    for _, attention, _, _ in config.model_spec.model.traverse(Attention.Config):
        assert isinstance(attention, Attention.Config)
        q_lora_ranks.add(attention.q_lora_rank)
    if len(q_lora_ranks) != 1:
        raise ValueError("Fused MLA requires one query projection layout")
    return "attention.wq" if q_lora_ranks == {0} else "attention.wq_b"


def enable_fused_mla(config: Trainer.Config) -> str:
    """Enable fused MLA assembly for a DeepSeek-V3 training recipe.

    Args:
        config: Trainer configuration updated in place.

    Returns:
        Module FQN of the query projection mutated by fused Q RoPE.
    """
    override = "torchtitan.overrides.fused_mla.fused_mla"
    assert override not in config.override.imports
    config.override.imports.append(override)
    q_projection = fused_mla_query_projection(config)
    if isinstance(config.activation_checkpoint, SelectiveAC.Config):
        # Q RoPE mutates the projection result, so SAC must recompute rather
        # than cache that GEMM's output for backward replay.
        if (
            q_projection
            not in config.activation_checkpoint.force_recompute_mm_shapes_by_fqns
        ):
            config.activation_checkpoint.force_recompute_mm_shapes_by_fqns.append(
                q_projection
            )
    return q_projection


def deepseek_v3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * 2048,
            max_context_length=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def deepseek_v3_debugmodel_mtp() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel", num_mtp_layers=1)
    config.loss = MTPLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec),
    )
    return config


def deepseek_v3_debugmodel_mxfp8() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    # Quantize the MoE expert grouped GEMMs to MXFP8, plus the dense Linear
    # layers in attention, the shared experts, and the dense-layer feed-forward.
    # fqns is an include-list (substring match), so the MoE router gate
    # (moe.router.gate) and lm_head are left in bf16.
    # pad_multiple=128 is required by the CuTeDSL quantization kernel
    # on sm_100 (e.g. B200)
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention", "shared_experts", "feed_forward"],
            ),
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                pad_multiple=128,
            ),
        ],
    )
    return config


def deepseek_v3_debugmodel_hybridep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def deepseek_v3_debugmodel_minimal_async_ep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    return config


def deepseek_v3_16b() -> Trainer.Config:
    model_spec = model_registry("16B", attn_backend="flex")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=default_adamw(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_16b_varlen() -> Trainer.Config:
    """Build DSV3 16B with fixed-shape FA4 varlen attention.

    Returns:
        No-compile configuration compatible with full-step CUDA graph replay.
    """
    config = deepseek_v3_16b()
    config.model_spec = model_registry("16B", attn_backend="varlen")
    _set_varlen_max_num_documents(config)
    config.training.disable_cuda_graphs = False
    config.compile.enable = False
    return config


def deepseek_v3_16b_dist_moe_bf16() -> Trainer.Config:
    """Build DSV3 16B with BF16-compute DistMoE and full CUDA graphs.

    Returns:
        Standard Trainer configuration with FP32 persistent state and BF16
        FSDP compute parameters.
    """
    from torchtitan.models.common.dist_moe import DistMoeBackendConfig

    return _enable_dist_moe(
        deepseek_v3_16b_varlen(),
        "16B",
        DistMoeBackendConfig(),
    )


def deepseek_v3_16b_dist_moe_mxfp8() -> Trainer.Config:
    """Build DSV3 16B with MXFP8 DistMoE and dense GEMMs.

    Returns:
        Standard Trainer MXFP8 configuration with FP32 persistent state and
        BF16 FSDP inputs to weight quantization.
    """
    return _enable_dist_moe(
        deepseek_v3_16b_varlen(),
        "16B",
        _mxfp8_dist_moe_backend(),
        quantize_dense=True,
    )


def deepseek_v3_16b_hybridep() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    config.training.disable_cuda_graphs = False
    return config


def deepseek_v3_16b_minimal_async_ep() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    config.training.disable_cuda_graphs = False
    return config


def deepseek_v3_671b() -> Trainer.Config:
    model_spec = model_registry(
        "671B",
        attn_backend="flex",
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=default_adamw(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
            steps=10000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_671b_varlen() -> Trainer.Config:
    """Build DSV3 671B with fixed-shape FA4 varlen attention.

    Returns:
        No-compile configuration compatible with full-step CUDA graph replay.
    """
    config = deepseek_v3_671b()
    config.model_spec = model_registry("671B", attn_backend="varlen")
    _set_varlen_max_num_documents(config)
    config.training.disable_cuda_graphs = False
    config.compile.enable = False
    return config


def deepseek_v3_671b_dist_moe_bf16() -> Trainer.Config:
    """Build DSV3 671B with BF16-compute DistMoE and full CUDA graphs.

    Returns:
        Standard Trainer configuration with FP32 persistent state and BF16
        FSDP compute parameters.
    """
    from torchtitan.models.common.dist_moe import DistMoeBackendConfig

    return _enable_dist_moe(
        deepseek_v3_671b_varlen(),
        "671B",
        DistMoeBackendConfig(),
    )


def deepseek_v3_671b_dist_moe_mxfp8() -> Trainer.Config:
    """Build DSV3 671B with MXFP8 DistMoE and dense GEMMs.

    Returns:
        Standard Trainer MXFP8 configuration with FP32 persistent state and
        BF16 FSDP inputs to weight quantization.
    """
    return _enable_dist_moe(
        deepseek_v3_671b_varlen(),
        "671B",
        _mxfp8_dist_moe_backend(),
        quantize_dense=True,
    )


def deepseek_v3_671b_float8() -> Trainer.Config:
    config = deepseek_v3_671b()
    # Quantize the dense Linear layers and the MoE expert grouped GEMMs to
    # float8 (fp8). This requires torchao and is only supported on NVIDIA SM89+
    # or AMD MI300+; on other backends (e.g. Intel XPU) the converter raises at
    # build time, so use the plain deepseek_v3_671b config there.
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "671B",
        attn_backend="flex",
        converters=[
            Float8LinearConverter.Config(
                filter_fqns=["lm_head", "router.gate"],
                model_compile_enabled=model_compile_enabled,
            ),
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled
            ),
        ],
    )
    return config
