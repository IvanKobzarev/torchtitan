# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from torchtitan.experiments.graph_trainer.common_utils import ensure_boxed_graph_module
from torchtitan.experiments.graph_trainer.graph_builder import GraphExecutionPlan
from torchtitan.experiments.graph_trainer.graph_builder_utils import (
    _execute_graph_module,
)
from torchtitan.experiments.graph_trainer.graph_pp.pp_graph_builder import (
    _ReduceGradRuntimeInputSource,
    _StageGraphModules,
    GraphTrainerStageGraphs,
)
from torchtitan.experiments.graph_trainer.graph_pp.precompile import (
    _deserialize_graph_module,
    _serialize_graph_module,
    graph_pp_rank_artifact_key,
    graph_pp_stage_config_fingerprint,
    load_graph_pp_rank_artifact,
    PrecompiledGraphPPRankArtifact,
    PrecompiledGraphPPStageArtifact,
    save_graph_pp_rank_artifact,
)
from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter


def _make_boxed_graph_module():
    return ensure_boxed_graph_module(torch.fx.symbolic_trace(lambda value: value + 1))


def _make_stage(stage_index: int):
    graph_module = _make_boxed_graph_module()
    stage_graphs = GraphTrainerStageGraphs(
        modules=_StageGraphModules(
            fw=graph_module,
            full_bw_repeat=graph_module,
            full_bw_first=graph_module,
            bw_di=graph_module,
            bw_dw_repeat=graph_module,
            bw_dw_first=graph_module,
            unshard_launch=None,
            unshard_wait=None,
            reduce_grad=None,
        ),
        meta=SimpleNamespace(
            num_fw_param_inputs=0,
            fwd_input_names=("value",),
            fwd_flat_input_indices=(0,),
            unshard_runtime_mesh_indices=(),
            reduce_grad_runtime_input_sources=(
                _ReduceGradRuntimeInputSource(
                    runtime_mesh_index=None,
                    saved_for_backward_index=0,
                ),
            ),
            num_saved_for_backward=1,
        ),
        compiled=True,
        full_bw_grad_accumulator_indices=(0,),
        bw_dw_grad_accumulator_indices=(1,),
    )
    return SimpleNamespace(
        stage_index=stage_index,
        device=torch.device("cpu"),
        graphs=stage_graphs,
    )


def _make_plan() -> GraphExecutionPlan:
    return GraphExecutionPlan(
        pp_enabled=True,
        num_microbatches=32,
        unshard="schedule",
        reduce_grad="schedule",
        fuse_wgrad_accumulation=True,
    )


class TestGraphPPPrecompileArtifact(unittest.TestCase):
    def test_stage_load_binds_bucketed_all_gather_runtime_rank(self) -> None:
        from torch._inductor.fx_passes import bucketing as _bucketing  # noqa: F401

        from torchtitan.experiments.graph_trainer.precompile import _register_coor_ops

        _register_coor_ops()
        graph = torch.fx.Graph()
        mesh = graph.placeholder("mesh")
        local_input = graph.placeholder("local_input")
        bucket = graph.call_function(
            torch.ops.bucketing._pre_bucket_all_gather.default,
            args=([local_input], 4, torch.bfloat16, [1], 0),
        )
        local_bucket = graph.call_function(
            torch.ops.aten.slice.Tensor,
            args=(bucket, 0, 0, 8),
        )
        submesh = graph.call_function(
            torch.ops.device_mesh._get_submesh.default,
            args=(mesh, [0]),
        )
        group_name = graph.call_function(
            torch.ops._dtensor.mesh_get_process_group.default,
            args=(submesh, 0),
        )
        gathered = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor_out.default,
            args=(local_bucket, 4, group_name),
            kwargs={"out": bucket},
        )
        graph.output(gathered)

        loaded = _deserialize_graph_module(
            _serialize_graph_module(torch.fx.GraphModule({}, graph)),
            device=torch.device("cpu"),
        )

        (loaded_bucket,) = loaded.graph.find_nodes(
            op="call_function",
            target=torch.ops.bucketing._pre_bucket_all_gather.default,
        )
        self.assertEqual(loaded_bucket.args[4].op, "call_method")
        self.assertEqual(loaded_bucket.args[4].target, "rank")
        loaded_group_name = loaded_bucket.args[4].args[0]
        self.assertEqual(
            loaded_group_name.target,
            torch.ops._dtensor.mesh_get_process_group.default,
        )

    def test_stage_fingerprint_uses_target_outer_cudagraph_state(self) -> None:
        stage = SimpleNamespace(
            stage_index=0,
            is_first=True,
            is_last=False,
            submod=torch.nn.Linear(2, 2),
        )
        with (
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_pp.precompile."
                "graph_config_signature",
                return_value=(),
            ),
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_pp.precompile."
                "compute_config_fingerprint",
                return_value="fingerprint",
            ) as compute,
        ):
            fingerprint = graph_pp_stage_config_fingerprint(
                stage,
                config=SimpleNamespace(compile=object()),
                parallelism_context=object(),
                execution_plan=_make_plan(),
                schedule_name="Interleaved1F1B",
                num_microbatches=32,
                outer_cudagraphs_enabled=True,
            )

        self.assertEqual(fingerprint, "fingerprint")
        self.assertTrue(compute.call_args.kwargs["outer_cudagraphs_enabled"])

    def test_rank_bundle_roundtrip_preserves_all_local_stages(self) -> None:
        stage_indices = tuple(range(0, 16, 2))
        stages = [_make_stage(stage_index) for stage_index in stage_indices]
        fingerprints = {
            stage_index: f"fingerprint-{stage_index}" for stage_index in stage_indices
        }
        schemas = {
            stage_index: (f"schema-{stage_index}", ()) for stage_index in stage_indices
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            save_graph_pp_rank_artifact(
                stages,
                storage,
                pp_rank=0,
                schedule_name="Interleaved1F1B",
                num_microbatches=32,
                num_runtime_mesh_inputs=0,
                execution_plan=_make_plan(),
                config_fingerprints=fingerprints,
                runtime_input_schemas=schemas,
            )
            loaded = load_graph_pp_rank_artifact(
                storage,
                pp_rank=0,
                schedule_name="Interleaved1F1B",
                num_microbatches=32,
                runtime_meshes=[],
                execution_plan=_make_plan(),
                expected_stage_indices=stage_indices,
                expected_config_fingerprints=fingerprints,
                expected_runtime_input_schemas=schemas,
                stage_devices={
                    stage_index: torch.device("cpu") for stage_index in stage_indices
                },
            )

        self.assertEqual(tuple(loaded), stage_indices)
        for stage_index, graphs in loaded.items():
            self.assertTrue(graphs.compiled)
            self.assertEqual(graphs.full_bw_grad_accumulator_indices, (0,))
            self.assertEqual(graphs.bw_dw_grad_accumulator_indices, (1,))
            self.assertIsNotNone(graphs.modules.full_bw_first)
            self.assertIsNotNone(graphs.modules.bw_di)
            self.assertIsNotNone(graphs.modules.bw_dw_repeat)
            self.assertIsNotNone(graphs.modules.bw_dw_first)
            args = [torch.tensor(stage_index)]
            (output,) = _execute_graph_module(graphs.modules.fw, args)
            torch.testing.assert_close(output, torch.tensor(stage_index + 1))
            self.assertEqual(args, [])

    def test_load_rejects_stale_stage_runtime_schema(self) -> None:
        stage = _make_stage(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            save_graph_pp_rank_artifact(
                [stage],
                storage,
                pp_rank=1,
                schedule_name="Interleaved1F1B",
                num_microbatches=32,
                num_runtime_mesh_inputs=0,
                execution_plan=_make_plan(),
                config_fingerprints={1: "fingerprint"},
                runtime_input_schemas={1: ("artifact-schema", ())},
            )
            with self.assertRaisesRegex(ValueError, "runtime input schema mismatch"):
                load_graph_pp_rank_artifact(
                    storage,
                    pp_rank=1,
                    schedule_name="Interleaved1F1B",
                    num_microbatches=32,
                    runtime_meshes=[],
                    execution_plan=_make_plan(),
                    expected_stage_indices=(1,),
                    expected_config_fingerprints={1: "fingerprint"},
                    expected_runtime_input_schemas={1: ("runtime-schema", ())},
                    stage_devices={1: torch.device("cpu")},
                )

    def test_load_rejects_previous_metadata_format(self) -> None:
        stage = _make_stage(0)
        artifact = PrecompiledGraphPPRankArtifact.from_stages(
            [stage],
            pp_rank=0,
            schedule_name="Interleaved1F1B",
            num_microbatches=32,
            num_runtime_mesh_inputs=0,
            execution_plan=_make_plan(),
            config_fingerprints={0: "fingerprint"},
            runtime_input_schemas={0: ("schema", ())},
        )
        artifact = dataclasses.replace(artifact, format_version=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save(graph_pp_rank_artifact_key(0), pickle.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "format mismatch"):
                load_graph_pp_rank_artifact(
                    storage,
                    pp_rank=0,
                    schedule_name="Interleaved1F1B",
                    num_microbatches=32,
                    runtime_meshes=[],
                    execution_plan=_make_plan(),
                    expected_stage_indices=(0,),
                    expected_config_fingerprints={0: "fingerprint"},
                    expected_runtime_input_schemas={0: ("schema", ())},
                    stage_devices={0: torch.device("cpu")},
                )

    def test_load_rejects_wrong_rank_even_if_storage_key_matches(self) -> None:
        stage = _make_stage(1)
        artifact = PrecompiledGraphPPRankArtifact.from_stages(
            [stage],
            pp_rank=0,
            schedule_name="Interleaved1F1B",
            num_microbatches=32,
            num_runtime_mesh_inputs=0,
            execution_plan=_make_plan(),
            config_fingerprints={1: "fingerprint"},
            runtime_input_schemas={1: ("schema", ())},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save(graph_pp_rank_artifact_key(1), pickle.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "rank contract mismatch"):
                load_graph_pp_rank_artifact(
                    storage,
                    pp_rank=1,
                    schedule_name="Interleaved1F1B",
                    num_microbatches=32,
                    runtime_meshes=[],
                    execution_plan=_make_plan(),
                    expected_stage_indices=(1,),
                    expected_config_fingerprints={1: "fingerprint"},
                    expected_runtime_input_schemas={1: ("schema", ())},
                    stage_devices={1: torch.device("cpu")},
                )

    def test_stage_load_maps_graph_state_to_runtime_device(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        stage = _make_stage(0)
        with FakeTensorMode():
            gpu_constant = torch.ones(2, device="cuda:0")
        stage.graphs.modules.fw.register_buffer("gpu_constant", gpu_constant)
        stage.graphs.modules.fw.register_buffer("cpu_constant", torch.ones(2))

        artifact = PrecompiledGraphPPStageArtifact.from_stage(
            stage,
            config_fingerprint="fingerprint",
            runtime_input_schema=("schema", ()),
        )
        loaded = artifact.to_stage_graphs(
            runtime_meshes=[],
            device=torch.device("cuda:1"),
        )

        self.assertEqual(
            loaded.modules.fw.gpu_constant.device,
            torch.device("cuda:1"),
        )
        self.assertEqual(
            loaded.modules.fw.cpu_constant.device,
            torch.device("cpu"),
        )


if __name__ == "__main__":
    unittest.main()
