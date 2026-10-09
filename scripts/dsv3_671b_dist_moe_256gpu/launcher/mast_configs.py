# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expose and specialize recipes archived in the pinned TorchTitan runtime."""

import os

from scripts.dsv3_671b_dist_moe_256gpu import mast_configs as _mast_configs
from scripts.dsv3_671b_dist_moe_256gpu.mast_configs import *  # noqa: F401, F403


def _configure_single_step_pp2_profile(config):
    """Record one final-geometry PP2 step without a Kineto warmup cycle.

    A 120-microbatch PP2 step can fill Kineto's default activity buffer before
    a multi-step warmup finishes. The launcher pairs this recipe with a larger
    profiling-only buffer through ``TORCHTITAN_KINETO_MAX_GPU_BUFFER_SIZE_MB``.
    """
    config.profiler.profile_freq = 41
    config.profiler.profiler_warmup = 0
    config.profiler.profiler_active = 1
    config.profiler.profiler_repeat = 1
    return config


def deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_single_step_profile():
    """Profile one eager final-geometry PP2 MTP1 training step."""
    config = (
        _mast_configs.deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile()
    )
    return _configure_single_step_pp2_profile(config)


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_single_step_profile():
    """Profile one GraphTrainer final-geometry PP2 MTP1 training step."""
    config = (
        _mast_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile()
    )
    return _configure_single_step_pp2_profile(config)


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_full_lookahead_performance():
    """Measure GraphTrainer PP2 MTP1 with the complete residency lookahead."""
    config = (
        _mast_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance()
    )
    config.parallelism.pp_num_unshard_lookahead_factor = "full"
    return config


def _configure_precompile(config):
    """Load the artifact embedded in the immutable launcher package."""
    config.compile.precompile_artifact_dir = os.environ[
        "TORCHTITAN_PRECOMPILE_ARTIFACT_DIR"
    ]
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_pp1_64gpu_numerics():
    """Verify PP1 precompile geometry on 64 GB300s before full scale."""
    config = (
        _mast_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_numerics()
    )
    config.parallelism.data_parallel_shard_degree = 64
    config.training.num_tokens_per_train_step = 4096 * 64 * 16
    config.training.steps = 10
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_pp1_precompile_64gpu_numerics():
    """Load the PP1 artifact on the matched 64-GB300 geometry."""
    config = (
        graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_pp1_64gpu_numerics()
    )
    return _configure_precompile(config)


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_pp1_precompile_256gpu_performance():
    """Profile precompiled PP1 MTP1 at the final 256-GPU geometry."""
    config = (
        _mast_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance()
    )
    return _configure_precompile(config)


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_pp1_precompile_256gpu_numerics():
    """Record full-precision metrics for precompiled PP1 at full scale."""
    config = (
        _mast_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_numerics()
    )
    return _configure_precompile(config)


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_pp2_64gpu_numerics():
    """Verify the final PP2 topology on 64 GB300s before full scale."""
    config = (
        _mast_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_numerics()
    )
    config.parallelism.data_parallel_shard_degree = 32
    config.parallelism.expert_parallel_degree = 32
    config.training.num_tokens_per_train_step = 4096 * 32 * 120
    config.training.steps = 10
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_pp2_precompile_64gpu_numerics():
    """Load the PP2 bundles on the matched 64-GB300 geometry."""
    config = (
        graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_pp2_64gpu_numerics()
    )
    return _configure_precompile(config)


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_pp2_precompile_256gpu_performance():
    """Profile precompiled PP2/VPP8 MTP1 at the final 256-GPU geometry."""
    config = (
        _mast_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance()
    )
    return _configure_precompile(config)


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_pp2_precompile_256gpu_numerics():
    """Record full-precision metrics for precompiled PP2 at full scale."""
    config = (
        _mast_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_numerics()
    )
    return _configure_precompile(config)
