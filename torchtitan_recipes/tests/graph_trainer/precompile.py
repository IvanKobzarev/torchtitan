# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer precompile integration-test configurations."""

import os

from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_debugmodel,
    graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8,
)
from torchtitan_recipes.tests.graph_trainer.llama3 import (
    graph_trainer_llama3_debugmodel_sdpa,
)


def llama3_precompile_fsdp_tp() -> GraphTrainer.Config:
    config = graph_trainer_llama3_debugmodel_sdpa()
    config.compile.precompile_artifact_dir = os.environ[
        "TORCHTITAN_PRECOMPILE_ARTIFACT_DIR"
    ]
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 4
    return config


def deepseek_v3_precompile_fsdp_tp_ep() -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v3_debugmodel()
    config.compile.precompile_artifact_dir = os.environ[
        "TORCHTITAN_PRECOMPILE_ARTIFACT_DIR"
    ]
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    return config


def deepseek_v3_dist_moe_precompile_fsdp2_ep2_pp2() -> GraphTrainer.Config:
    """Exercise rank-local GraphPP precompile with DistMoE and 32 microbatches."""
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime

    config = graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8(seq_len=128)
    runtime_config = config.dist_moe
    assert isinstance(runtime_config, DistMoeRuntime.Config)
    runtime_config.scratch_capacity_factor = 2.0
    config.comm.backend = "real_pp_fake_spmd"
    config.compile.precompile_artifact_dir = os.environ[
        "TORCHTITAN_PRECOMPILE_ARTIFACT_DIR"
    ]
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 32
    config.training.num_tokens_per_microbatch_per_dp_rank = 128
    config.training.max_context_length = 128
    config.training.steps = 1
    config.training.disable_cuda_graphs = True
    config.debug.seed = 42
    config.debug.deterministic = True
    config.checkpointer = None
    return config
