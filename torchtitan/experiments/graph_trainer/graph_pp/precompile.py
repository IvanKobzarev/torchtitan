# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serialization for rank-local GraphPP precompile artifacts."""

from __future__ import annotations

import functools
import logging
import pickle
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.experiments.graph_trainer.precompile import (
    _execution_plan_signature,
    _map_precompiled_graph_device,
    _register_coor_ops,
    _validate_config_fingerprint,
    compute_config_fingerprint,
    ConfigFingerprint,
    graph_config_signature,
    RuntimeInputSchema,
)
from torchtitan.experiments.graph_trainer.storage import StorageAdapter


if TYPE_CHECKING:
    from torch.distributed.pipelining.schedules import _PipelineScheduleRuntime

    from torchtitan.distributed import ParallelismContext
    from torchtitan.experiments.graph_trainer.graph_builder import GraphExecutionPlan
    from torchtitan.experiments.graph_trainer.graph_pp.pp_graph_builder import (
        GraphTrainerStageGraphs,
    )
    from torchtitan.experiments.graph_trainer.graph_pp.stage import GraphPipelineStage
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


logger = logging.getLogger(__name__)

_GRAPH_PP_FORMAT_VERSION = 2
_GRAPH_PP_MODULE_NAMES = (
    "fw",
    "full_bw_repeat",
    "full_bw_first",
    "bw_di",
    "bw_dw_repeat",
    "bw_dw_first",
    "unshard_launch",
    "unshard_wait",
    "reduce_grad",
)


def graph_pp_rank_artifact_key(pp_rank: int) -> str:
    """Return the flat storage key for one pipeline-parallel rank."""
    return f"graph_pp_pp{pp_rank}"


def graph_pp_stage_config_fingerprint(
    stage: GraphPipelineStage,
    *,
    config: GraphTrainer.Config,
    parallelism_context: ParallelismContext,
    execution_plan: GraphExecutionPlan,
    schedule_name: str,
    num_microbatches: int,
    outer_cudagraphs_enabled: bool,
) -> ConfigFingerprint:
    """Fingerprint one stage and the complete GraphPP graph contract."""
    graph_signature = (
        *graph_config_signature(config),
        (
            "graph_pp",
            (
                ("format_version", _GRAPH_PP_FORMAT_VERSION),
                ("schedule", schedule_name),
                ("num_microbatches", num_microbatches),
                ("stage_index", stage.stage_index),
                ("is_first", stage.is_first),
                ("is_last", stage.is_last),
                ("execution_plan", _execution_plan_signature(execution_plan)),
            ),
        ),
    )
    return compute_config_fingerprint(
        stage.submod,
        config.compile,
        parallelism_context,
        graph_signature=graph_signature,
        outer_cudagraphs_enabled=outer_cudagraphs_enabled,
    )


def ensure_graph_pp_schedule_precompilable(
    schedule: _PipelineScheduleRuntime,
) -> None:
    """Reject schedules whose runtime-only multiplex graphs are not serialized."""
    from torchtitan.experiments.graph_trainer.graph_pp import stage_builder

    overlap_pairs = stage_builder._required_multiplex_pairs(schedule)
    if overlap_pairs:
        raise NotImplementedError(
            "GraphPP precompile does not support schedules with OVERLAP_F_B "
            f"actions; required stage pairs are {sorted(overlap_pairs)}."
        )


def _serialize_graph_module(graph_module: fx.GraphModule) -> bytes:
    from torch.fx._graph_pickler import GraphPickler, Options

    from torchtitan.experiments.graph_trainer.inductor_passes import (
        _node_metadata_key_filter_distributed,
    )

    return GraphPickler.dumps(
        graph_module,
        Options(
            ops_filter=None,
            node_metadata_key_filter=_node_metadata_key_filter_distributed,
        ),
    )


def _deserialize_graph_module(
    data: bytes,
    *,
    device: torch.device,
) -> fx.GraphModule:
    from torch._subclasses import FakeTensorMode
    from torch.fx._graph_pickler import GraphPickler
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    _register_coor_ops()
    fake_mode = FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv(),
    )
    graph_module = GraphPickler.loads(
        data,
        fake_mode,
        map_location=functools.partial(
            _map_precompiled_graph_device,
            runtime_device=device,
        ),
    )
    graph_module.recompile()
    return graph_module


@dataclass(frozen=True)
class PrecompiledGraphPPStageArtifact:
    """Serialized graph modules and calling convention for one local stage."""

    stage_index: int
    serialized_modules: dict[str, bytes]
    meta: Any
    compiled: bool
    full_bw_grad_accumulator_indices: tuple[int, ...]
    bw_dw_grad_accumulator_indices: tuple[int, ...]
    config_fingerprint: ConfigFingerprint
    runtime_input_schema: RuntimeInputSchema

    @classmethod
    def from_stage(
        cls,
        stage: GraphPipelineStage,
        *,
        config_fingerprint: ConfigFingerprint,
        runtime_input_schema: RuntimeInputSchema,
    ) -> PrecompiledGraphPPStageArtifact:
        from torchtitan.experiments.graph_trainer.graph_pp.pp_graph_builder import (
            GraphTrainerStageGraphs,
        )

        if not isinstance(stage.graphs, GraphTrainerStageGraphs):
            raise ValueError(
                "GraphPP precompile requires built GraphTrainer stage graphs for "
                f"stage {stage.stage_index}."
            )
        if not stage.graphs.compiled:
            raise ValueError(
                "GraphPP precompile requires compiled or boxed stage graphs for "
                f"stage {stage.stage_index}."
            )
        serialized_modules = {
            name: _serialize_graph_module(graph_module)
            for name in _GRAPH_PP_MODULE_NAMES
            if (graph_module := getattr(stage.graphs.modules, name)) is not None
        }
        return cls(
            stage_index=stage.stage_index,
            serialized_modules=serialized_modules,
            meta=stage.graphs.meta,
            compiled=stage.graphs.compiled,
            full_bw_grad_accumulator_indices=(
                stage.graphs.full_bw_grad_accumulator_indices
            ),
            bw_dw_grad_accumulator_indices=(
                stage.graphs.bw_dw_grad_accumulator_indices
            ),
            config_fingerprint=config_fingerprint,
            runtime_input_schema=runtime_input_schema,
        )

    def to_stage_graphs(
        self,
        *,
        runtime_meshes: list[DeviceMesh],
        device: torch.device,
    ) -> GraphTrainerStageGraphs:
        from torchtitan.experiments.graph_trainer.graph_pp.pp_graph_builder import (
            _StageGraphModules,
            GraphTrainerStageGraphs,
        )

        modules = {
            name: (
                _deserialize_graph_module(data, device=device)
                if (data := self.serialized_modules.get(name)) is not None
                else None
            )
            for name in _GRAPH_PP_MODULE_NAMES
        }
        if modules["fw"] is None or modules["full_bw_repeat"] is None:
            raise ValueError(
                "GraphPP precompile artifact for stage "
                f"{self.stage_index} is missing fw or full_bw_repeat."
            )
        return GraphTrainerStageGraphs(
            modules=_StageGraphModules(**modules),
            meta=self.meta,
            compiled=self.compiled,
            full_bw_grad_accumulator_indices=(self.full_bw_grad_accumulator_indices),
            bw_dw_grad_accumulator_indices=self.bw_dw_grad_accumulator_indices,
            runtime_meshes=tuple(runtime_meshes),
        )


@dataclass(frozen=True)
class PrecompiledGraphPPRankArtifact:
    """All stage graphs owned by one pipeline-parallel rank."""

    format_version: int
    pp_rank: int
    schedule_name: str
    num_microbatches: int
    num_runtime_mesh_inputs: int
    execution_plan_signature: tuple[tuple[str, Any], ...]
    stages: tuple[PrecompiledGraphPPStageArtifact, ...]

    @classmethod
    def from_stages(
        cls,
        stages: list[GraphPipelineStage],
        *,
        pp_rank: int,
        schedule_name: str,
        num_microbatches: int,
        num_runtime_mesh_inputs: int,
        execution_plan: GraphExecutionPlan,
        config_fingerprints: dict[int, ConfigFingerprint],
        runtime_input_schemas: dict[int, RuntimeInputSchema],
    ) -> PrecompiledGraphPPRankArtifact:
        stage_indices = tuple(stage.stage_index for stage in stages)
        if len(stage_indices) != len(set(stage_indices)):
            raise ValueError(
                f"GraphPP precompile stage indices must be unique: {stage_indices}"
            )
        expected_stage_indices = set(stage_indices)
        for name, values in (
            ("config fingerprints", config_fingerprints),
            ("runtime input schemas", runtime_input_schemas),
        ):
            if set(values) != expected_stage_indices:
                raise ValueError(
                    f"GraphPP precompile {name} must cover exactly the local "
                    f"stages: expected {sorted(expected_stage_indices)}, "
                    f"got {sorted(values)}"
                )
        return cls(
            format_version=_GRAPH_PP_FORMAT_VERSION,
            pp_rank=pp_rank,
            schedule_name=schedule_name,
            num_microbatches=num_microbatches,
            num_runtime_mesh_inputs=num_runtime_mesh_inputs,
            execution_plan_signature=_execution_plan_signature(execution_plan),
            stages=tuple(
                PrecompiledGraphPPStageArtifact.from_stage(
                    stage,
                    config_fingerprint=config_fingerprints[stage.stage_index],
                    runtime_input_schema=runtime_input_schemas[stage.stage_index],
                )
                for stage in stages
            ),
        )


def save_graph_pp_rank_artifact(
    stages: list[GraphPipelineStage],
    storage: StorageAdapter,
    *,
    pp_rank: int,
    schedule_name: str,
    num_microbatches: int,
    num_runtime_mesh_inputs: int,
    execution_plan: GraphExecutionPlan,
    config_fingerprints: dict[int, ConfigFingerprint],
    runtime_input_schemas: dict[int, RuntimeInputSchema],
) -> str:
    """Save one trusted, rank-local GraphPP artifact."""
    artifact = PrecompiledGraphPPRankArtifact.from_stages(
        stages,
        pp_rank=pp_rank,
        schedule_name=schedule_name,
        num_microbatches=num_microbatches,
        num_runtime_mesh_inputs=num_runtime_mesh_inputs,
        execution_plan=execution_plan,
        config_fingerprints=config_fingerprints,
        runtime_input_schemas=runtime_input_schemas,
    )
    data = pickle.dumps(artifact)
    path = storage.save(graph_pp_rank_artifact_key(pp_rank), data)
    logger.info(
        "GraphPP precompile artifact saved: pp_rank=%s, stages=%s, "
        "size=%s bytes, path=%s",
        pp_rank,
        [stage.stage_index for stage in artifact.stages],
        len(data),
        path,
    )
    return path


def load_graph_pp_rank_artifact(
    storage: StorageAdapter,
    *,
    pp_rank: int,
    schedule_name: str,
    num_microbatches: int,
    runtime_meshes: list[DeviceMesh],
    execution_plan: GraphExecutionPlan,
    expected_stage_indices: tuple[int, ...],
    expected_config_fingerprints: dict[int, ConfigFingerprint],
    expected_runtime_input_schemas: dict[int, RuntimeInputSchema],
    stage_devices: dict[int, torch.device],
) -> dict[int, GraphTrainerStageGraphs]:
    """Load and validate a trusted rank-local GraphPP artifact."""
    data = storage.load(graph_pp_rank_artifact_key(pp_rank))
    artifact: PrecompiledGraphPPRankArtifact = pickle.loads(data)
    if getattr(artifact, "format_version", 0) != _GRAPH_PP_FORMAT_VERSION:
        raise ValueError(
            "GraphPP precompile artifact format mismatch: "
            f"expected {_GRAPH_PP_FORMAT_VERSION}, "
            f"got {getattr(artifact, 'format_version', 0)}."
        )
    expected_rank_contract = (
        pp_rank,
        schedule_name,
        num_microbatches,
        len(runtime_meshes),
        _execution_plan_signature(execution_plan),
    )
    artifact_rank_contract = (
        artifact.pp_rank,
        artifact.schedule_name,
        artifact.num_microbatches,
        artifact.num_runtime_mesh_inputs,
        artifact.execution_plan_signature,
    )
    if artifact_rank_contract != expected_rank_contract:
        raise ValueError(
            "GraphPP precompile rank contract mismatch: "
            f"artifact={artifact_rank_contract}, runtime={expected_rank_contract}"
        )
    artifact_stage_indices = tuple(stage.stage_index for stage in artifact.stages)
    if len(artifact_stage_indices) != len(set(artifact_stage_indices)):
        raise ValueError(
            "GraphPP precompile artifact contains duplicate stage indices: "
            f"{artifact_stage_indices}"
        )
    if artifact_stage_indices != expected_stage_indices:
        raise ValueError(
            "GraphPP precompile stage assignment mismatch: "
            f"artifact={artifact_stage_indices}, runtime={expected_stage_indices}"
        )
    if set(expected_config_fingerprints) != set(expected_stage_indices):
        raise ValueError(
            "Expected GraphPP config fingerprints must cover exactly the local "
            f"stages: {expected_stage_indices}"
        )
    if set(expected_runtime_input_schemas) != set(expected_stage_indices):
        raise ValueError(
            "Expected GraphPP runtime schemas must cover exactly the local "
            f"stages: {expected_stage_indices}"
        )
    if set(stage_devices) != set(expected_stage_indices):
        raise ValueError(
            "GraphPP stage devices must cover exactly the local stages: "
            f"{expected_stage_indices}"
        )

    stage_graphs = {}
    for stage in artifact.stages:
        _validate_config_fingerprint(
            stage.config_fingerprint,
            expected_config_fingerprints[stage.stage_index],
        )
        expected_schema = expected_runtime_input_schemas[stage.stage_index]
        if stage.runtime_input_schema != expected_schema:
            raise ValueError(
                "GraphPP precompile runtime input schema mismatch for stage "
                f"{stage.stage_index}: artifact={stage.runtime_input_schema}, "
                f"runtime={expected_schema}"
            )
        stage_graphs[stage.stage_index] = stage.to_stage_graphs(
            runtime_meshes=runtime_meshes,
            device=stage_devices[stage.stage_index],
        )
    logger.info(
        "GraphPP precompile artifact loaded: pp_rank=%s, stages=%s",
        pp_rank,
        list(stage_graphs),
    )
    return stage_graphs
