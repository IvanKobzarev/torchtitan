# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
import torch.nn as nn

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.utils import get_spmd_context
from torchtitan.experiments.graph_trainer.configs import (
    EpOverlapConfig,
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.trainer import Trainer


class _SingleRankParallelDims(SimpleNamespace):
    """Provide the parallel-topology interface used by single-rank tests."""

    def get_optional_mesh(self, mesh_name: str) -> None:
        """Return no optional mesh for a single-rank test topology.

        Args:
            mesh_name: Requested optional mesh name.

        Returns:
            ``None`` because focused unit tests do not construct device meshes.
        """
        del mesh_name
        return None


def build_minimal_trainer(
    model: nn.Module,
    model_config,
    trainer_cls: type[Trainer],
    *,
    activation_checkpoint_mode: str = "none",
    compile_enable_passes: bool = True,
    compile_passes: list[str] | None = None,
    compile_ep_overlap_enabled: bool = False,
    compile_ep_overlap_chunk_dim: str = "batch",
    compile_ep_overlap_chunk_strategy: str = "graph",
    compile_ep_overlap_module_fqn: str = "layers.*",
    compile_ep_overlap_disable_early_grad_accumulation: bool = False,
    compile_inductor_compilation: str = "regional",
    compile_disable_passes: list[str] | None = None,
    compile_numerics_changing_optim: bool = False,
    tokenizer=None,
    fsdp_reshard_after_forward: str = "default",
    parallel_dims=None,
) -> Trainer:
    """Build the minimal Trainer state needed for test steps.

    Args:
        model: Model executed by the trainer.
        model_config: Configuration used to build ``model``.
        trainer_cls: Trainer implementation to instantiate without ``__init__``.
        activation_checkpoint_mode: Activation-checkpoint policy name.
        compile_enable_passes: Whether GraphTrainer applies graph passes.
        compile_passes: Additional graph pass names.
        compile_ep_overlap_enabled: Whether EP overlap is enabled.
        compile_ep_overlap_chunk_dim: Dimension used for EP overlap chunks.
        compile_ep_overlap_chunk_strategy: Eager or graph chunking strategy.
        compile_ep_overlap_module_fqn: Module pattern chunked for EP overlap.
        compile_ep_overlap_disable_early_grad_accumulation: Whether to defer
            parameter-gradient accumulation until after chunk recombination.
        compile_inductor_compilation: GraphTrainer Inductor compilation mode.
        compile_disable_passes: Graph pass names disabled for the test.
        compile_numerics_changing_optim: Whether numerics-changing passes run.
        tokenizer: Optional tokenizer used to prepare model inputs.
        fsdp_reshard_after_forward: FSDP parameter reshard policy.
        parallel_dims: Optional real parallel topology for distributed tests.

    Returns:
        A minimally initialized trainer suitable for focused tests.
    """
    trainer = object.__new__(trainer_cls)
    trainer.model_parts = [model]
    loss_config = CrossEntropyLoss.Config()
    trainer.loss_fn = loss_config.build()
    trainer.parallel_dims = (
        _SingleRankParallelDims(pp_enabled=False, cp_enabled=False)
        if parallel_dims is None
        else parallel_dims
    )
    trainer.train_context = get_spmd_context()
    trainer.fwd_bwd_fn = trainer._forward_backward_body
    trainer.model_config = model_config
    trainer.device = torch.device("cuda")
    trainer.tokenizer = tokenizer
    trainer.ntokens_seen = 0

    if trainer_cls is GraphTrainer:
        trainer.config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                enable=True,
                mode="aot_fx_trace",
                enable_passes=compile_enable_passes,
                passes=[] if compile_passes is None else list(compile_passes),
                disable_passes=(
                    []
                    if compile_disable_passes is None
                    else list(compile_disable_passes)
                ),
                inductor_compilation=compile_inductor_compilation,
                numerics_changing_optim=compile_numerics_changing_optim,
                ep_overlap=EpOverlapConfig(
                    enabled=compile_ep_overlap_enabled,
                    chunk_dim=compile_ep_overlap_chunk_dim,
                    strategy=compile_ep_overlap_chunk_strategy,
                    module_fqn=compile_ep_overlap_module_fqn,
                    disable_early_grad_accumulation=(
                        compile_ep_overlap_disable_early_grad_accumulation
                    ),
                ),
            ),
            loss=loss_config,
            model_spec=SimpleNamespace(model=model_config),
            activation_checkpoint={
                "none": None,
                "selective": SelectiveAC.Config(),
                "full": FullAC.Config(),
            }[activation_checkpoint_mode],
            parallelism=SimpleNamespace(
                pipeline_parallel_degree=1,
                fsdp_reshard_after_forward=fsdp_reshard_after_forward,
                spmd_backend="partial_dtensor",
            ),
        )
        trainer._fwd_bwd_step_module = None
        trainer._traced_step = None
    else:
        trainer.config = SimpleNamespace(
            parallelism=SimpleNamespace(spmd_backend="partial_dtensor"),
        )

    return trainer
