# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import operator
import types
import unittest
import warnings
from dataclasses import dataclass, replace
from typing import Any
from unittest import mock

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from dist_moe import BlockScaledFormat, DistMoeBlockScaledConfig
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.quantization import MXFP8LinearConverter
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import _configure_fsdp_modules
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    model_registry as dsv3_model_registry,
)
from torchtitan.experiments.graph_trainer.deferred_fsdp import (
    bind_deferred_fsdp_graph,
    build_deferred_fsdp_graph,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    deduplicate_fsdp_unshard_chains_pass,
    get_fsdp_param_module_order,
    joint_transformer_block_bucketing_reordering_pass,
    materialize_fsdp_bucket_outputs_pass,
    preserve_fsdp_unshard_output_boundaries_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    _PACKED_FSDP_UNSHARD_PARAM,
    find_fsdp_reduce_grad_boundary,
    find_fsdp_reduce_grad_input,
    find_fsdp_unshard_output,
    find_fsdp_unshard_save_node,
    find_fsdp_unshard_save_nodes,
)
from torchtitan.experiments.graph_trainer.gradient_accumulation import (
    GraphGradientState,
)
from torchtitan.experiments.graph_trainer.graph_pp import (
    partition_joint_graph,
    split_backward_fsdp_collectives,
    split_di_dw_graph,
    split_fsdp_unshard_collectives,
)
from torchtitan.experiments.graph_trainer.graph_pp.partition import GraphMeta
from torchtitan.experiments.graph_trainer.graph_pp.utils import flatten_graph_values
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    canonicalize_graph_pass,
    eliminate_dead_code_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    disable_active_parametrization,
)
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.dist_moe import (
    cleanup_dist_moe,
    DistMoeBackendConfig,
    setup_dist_moe,
)
from torchtitan.trainer import Trainer


@dataclass(frozen=True, slots=True)
class _Dsv3MoeBlockTrace:
    traced: TracedResult
    flat_inputs: list[Any]
    output_grad: torch.Tensor
    num_param_grad_values: int
    num_flat_param_values: int


class _DistMoePreparedWeightConsumer(torch.nn.Module):
    def __init__(self, experts: torch.nn.Module) -> None:
        super().__init__()
        self.experts = experts

    def forward(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        expert_ids: torch.Tensor,
        local_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.experts(x, scores, expert_ids, local_tokens)


@contextlib.contextmanager
def _stable_flex_attention_compile_config():
    original_configs = FlexAttention.inductor_configs
    original_compiled_flex_attn = FlexAttention._compiled_flex_attn
    FlexAttention.inductor_configs = {
        **original_configs,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    FlexAttention._compiled_flex_attn = torch.compile(
        flex_attention,
        options=FlexAttention.inductor_configs,
    )
    try:
        yield
    finally:
        FlexAttention.inductor_configs = original_configs
        FlexAttention._compiled_flex_attn = original_compiled_flex_attn


def _trace_dsv3_moe_block_stage(
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    include_input_grad: bool = True,
    fsdp_mesh: Any | None = None,
    quantize_dense: bool = False,
    annotate_modules: bool = False,
) -> _Dsv3MoeBlockTrace:
    """Trace a real DeepSeek V3 MoE decoder block as a GraphPP stage.

    This is intentionally CUDA-only: FlexAttention backward is not supported on
    CPU, and these pass tests need to exercise the real BlockMask tracing path
    rather than a maskless SDPA shortcut. The unit test enables EP sharding
    metadata and traces the first MoE layer. When ``fsdp_mesh`` is supplied the
    same block is wrapped with the graph trainer's simple-FSDP path so the
    partition pass is tested against the collective shapes that later passes
    consume. True EP numerics are covered by the distributed GraphPP DSV3
    loss-compare tests.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("DeepSeek V3 MoE block tracing requires CUDA")

    maybe_register_blockmask_pytree_node()
    torch.manual_seed(0)

    with _stable_flex_attention_compile_config():
        converters = (
            [
                MXFP8LinearConverter.Config(
                    model_compile_enabled=True,
                    fqns=["attention", "shared_experts", "feed_forward"],
                )
            ]
            if quantize_dense
            else None
        )
        model_spec = dsv3_model_registry(
            "debugmodel",
            attn_backend="flex",
            converters=converters,
        )
        model_config = model_spec.model
        runtime_config = Trainer.Config(
            model_spec=model_spec,
            training=TrainingConfig(
                num_tokens_per_microbatch_per_dp_rank=batch_size * seq_len,
                max_context_length=seq_len,
                steps=1,
            ),
            parallelism=ParallelismConfig(expert_parallel_degree=2),
            checkpoint=CheckpointManager.Config(initial_load_model_only=False),
            debug=DebugConfig(seed=0, deterministic=True),
        )
        model_config.update_from_config(config=runtime_config)
        moe_layer_config = model_config.layers[1]
        if moe_layer_config.moe is None:
            raise AssertionError("DeepSeek V3 MoE layer must contain an MoE block")

        with torch.device("meta"):
            model = model_config.build()
        model.to_empty(device="cuda")
        with torch.no_grad():
            model.init_states(buffer_device=None)
        model._apply(
            lambda tensor: tensor.to(dtype=torch.bfloat16)
            if tensor.is_floating_point()
            else tensor
        )
        model.train()

        block = model.layers["1"]
        if not block.moe_enabled:
            raise AssertionError("DeepSeek V3 debug layer 1 must be a MoE layer")
        if annotate_modules:
            annotate_module_fqns(block)
        if fsdp_mesh is not None:
            _configure_fsdp_modules(block)
            block = data_parallel(
                block,
                device_mesh=fsdp_mesh,
                mode="fully_shard",
            )

        num_tokens = batch_size * seq_len
        x = torch.randn(
            num_tokens,
            model_config.dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=include_input_grad,
        )
        positions = torch.arange(seq_len, device="cuda").repeat(batch_size)
        attention_masks = model.get_attention_masks(positions)
        output_grad = torch.randn_like(x)

        def stage_step(
            x: torch.Tensor,
            positions: torch.Tensor,
            attention_masks: Any,
            output_grad: torch.Tensor,
        ):
            out = block(x, attention_masks, positions)
            params = [
                p
                for _, p in block.named_parameters(remove_duplicate=False)
                if p.requires_grad
            ]
            grad_targets = [*params, x] if include_input_grad else params
            grads = torch.autograd.grad(
                out,
                grad_targets,
                grad_outputs=output_grad,
            )
            return [out, *grads]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="flex_attention called without torch.compile",
                category=UserWarning,
            )
            traced = minimal_fx_tracer(stage_step, module=block)(
                x,
                positions,
                attention_masks,
                output_grad,
            )

        user_flat_inputs, _ = pytree.tree_flatten(
            ((x, positions, attention_masks, output_grad), {})
        )
        state_flat_inputs, _ = pytree.tree_flatten(extract_module_state(block))
        flat_inputs = flatten_graph_values([*state_flat_inputs, *user_flat_inputs])
        if len(flat_inputs) != len(traced.example_inputs):
            raise AssertionError(
                "Real flat inputs must match traced flat input count: "
                f"{len(flat_inputs)} != {len(traced.example_inputs)}"
            )

        state_params = [p for _, p in block.named_parameters(remove_duplicate=False)]
        grad_params = [p for p in state_params if p.requires_grad]
        return _Dsv3MoeBlockTrace(
            traced=traced,
            flat_inputs=flat_inputs,
            output_grad=output_grad,
            num_param_grad_values=len(flatten_graph_values(grad_params)),
            num_flat_param_values=len(flatten_graph_values(state_params)),
        )


def _boxed_run(gm: fx.GraphModule, args: list[Any]):
    return fx.Interpreter(gm).boxed_run(args)


def _backward_args_from_partition(
    meta: GraphMeta,
    fw_outputs: tuple[Any, ...],
    backward_grad_inputs: tuple[Any, ...],
) -> list[Any]:
    saved_by_name = dict(
        zip(
            meta.saved_for_backward_names,
            fw_outputs[
                meta.num_fwd_user_outputs : meta.num_fwd_user_outputs
                + meta.num_saved_for_backward
            ],
            strict=True,
        )
    )
    backward_grad_by_name = dict(
        zip(
            meta.backward_grad_input_names,
            (backward_grad_inputs[index] for index in meta.backward_grad_input_indices),
            strict=True,
        )
    )
    return [
        saved_by_name[name] if name in saved_by_name else backward_grad_by_name[name]
        for name in meta.bwd_input_names
    ]


def _assert_tensor_sequence_equal(
    test_case: unittest.TestCase,
    actual_values: tuple[Any, ...],
    expected_values: tuple[Any, ...],
) -> None:
    test_case.assertEqual(len(actual_values), len(expected_values))
    for actual, expected in zip(actual_values, expected_values, strict=True):
        if actual is None or expected is None:
            test_case.assertIs(actual, expected)
        elif not isinstance(actual, torch.Tensor) or not isinstance(
            expected, torch.Tensor
        ):
            test_case.assertEqual(actual, expected)
        else:
            test_case.assertTrue(torch.equal(actual, expected))


class GraphPPPartitionTest(unittest.TestCase):
    def test_real_dsv3_moe_block_partition_matches_joint_graph(self) -> None:
        traced_block = _trace_dsv3_moe_block_stage()

        fw_module, bw_module, meta = partition_joint_graph(
            traced_block.traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced_block.traced.example_inputs) - 1,),
        )

        joint_outputs = traced_block.traced.gm(*traced_block.flat_inputs)
        fw_args = [
            traced_block.flat_inputs[index] for index in meta.fwd_flat_input_indices
        ]
        fw_outputs = _boxed_run(fw_module, list(fw_args))

        self.assertTrue(torch.equal(fw_outputs[0], joint_outputs[0]))
        self.assertEqual(meta.num_backward_grad_inputs, 1)
        self.assertEqual(
            meta.num_bwd_outputs,
            traced_block.num_param_grad_values + 1,
        )
        self.assertGreater(meta.num_saved_for_backward, 0)
        self.assertEqual(
            len(fw_outputs),
            meta.num_fwd_user_outputs
            + meta.num_saved_for_backward
            + meta.num_fwd_side_effect_outputs,
        )

        bw_args = _backward_args_from_partition(
            meta,
            fw_outputs,
            (traced_block.output_grad,),
        )
        bw_outputs = _boxed_run(bw_module, list(bw_args))
        _assert_tensor_sequence_equal(self, bw_outputs, joint_outputs[1:])

    def test_partition_saves_backward_passthrough_placeholders(self) -> None:
        def stage_step(
            x: torch.Tensor,
            dtensor_layout_metadata: torch.Tensor,
            output_grad: torch.Tensor,
        ):
            out = x.sin()
            (grad_x,) = torch.autograd.grad(
                out,
                x,
                grad_outputs=output_grad,
            )
            return [out, grad_x, dtensor_layout_metadata]

        x = torch.randn(2, 4, requires_grad=True)
        dtensor_layout_metadata = torch.arange(2)
        output_grad = torch.randn(2, 4)
        traced = minimal_fx_tracer(stage_step)(x, dtensor_layout_metadata, output_grad)

        fw_module, bw_module, meta = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced.example_inputs) - 1,),
        )

        flat_inputs = [x, dtensor_layout_metadata, output_grad]
        fw_args = [flat_inputs[index] for index in meta.fwd_flat_input_indices]
        fw_outputs = _boxed_run(fw_module, fw_args)

        self.assertIn("arg1_1", meta.saved_for_backward_names)
        self.assertNotIn("arg2_1", meta.fwd_input_names)
        self.assertEqual(
            meta.bwd_input_names,
            (*meta.saved_for_backward_names, *meta.backward_grad_input_names),
        )

        bw_args = _backward_args_from_partition(meta, fw_outputs, (output_grad,))
        bw_outputs = _boxed_run(bw_module, bw_args)
        joint_outputs = traced.gm(*flat_inputs)
        _assert_tensor_sequence_equal(self, bw_outputs, joint_outputs[1:])

    def test_invalid_backward_only_input_indices_raise(self) -> None:
        def stage_step(x: torch.Tensor, output_grad: torch.Tensor):
            out = x.sin()
            (grad_x,) = torch.autograd.grad(out, x, grad_outputs=output_grad)
            return [out, grad_x]

        x = torch.randn(2, 4, requires_grad=True)
        output_grad = torch.randn(2, 4)
        traced = minimal_fx_tracer(stage_step)(x, output_grad)

        with self.assertRaisesRegex(ValueError, "must be unique"):
            partition_joint_graph(
                traced,
                num_fwd_outputs=1,
                backward_only_input_indices=(1, 1),
            )

        with self.assertRaisesRegex(ValueError, "must reference traced graph"):
            partition_joint_graph(
                traced,
                num_fwd_outputs=1,
                backward_only_input_indices=(len(traced.example_inputs),),
            )

    def test_backward_only_input_required_by_forward_raises(self) -> None:
        def stage_step(x: torch.Tensor, output_grads_from_next: torch.Tensor):
            out = x + output_grads_from_next
            (grad_x,) = torch.autograd.grad(out.sum(), x)
            return [out, grad_x]

        x = torch.randn(2, 4, requires_grad=True)
        output_grads_from_next = torch.randn(2, 4)
        traced = minimal_fx_tracer(stage_step)(x, output_grads_from_next)

        with self.assertRaisesRegex(
            ValueError,
            "Forward graph outputs require backward-only inputs",
        ):
            partition_joint_graph(
                traced,
                num_fwd_outputs=1,
                backward_only_input_indices=(1,),
            )

    def test_forward_mutation_of_backward_only_input_raises(self) -> None:
        def stage_step(x: torch.Tensor, output_grads_from_next: torch.Tensor):
            output_grads_from_next.add_(1.0)
            out = x.sin()
            (grad_x,) = torch.autograd.grad(
                out,
                x,
                grad_outputs=torch.ones_like(out),
            )
            return [out, grad_x]

        x = torch.randn(2, 4, requires_grad=True)
        output_grads_from_next = torch.randn(2, 4)
        traced = minimal_fx_tracer(stage_step)(x, output_grads_from_next)

        with self.assertRaisesRegex(
            ValueError,
            "Forward mutation cannot target a backward-only input",
        ):
            partition_joint_graph(
                traced,
                num_fwd_outputs=1,
                backward_only_input_indices=(1,),
            )


class _GraphPPDsv3FSDPTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return max(1, min(torch.cuda.device_count(), 2))

    def _setup(self) -> None:
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            spmd_backend="partial_dtensor",
        )


class GraphPPPartitionFSDPTest(_GraphPPDsv3FSDPTest):
    def test_real_dsv3_moe_block_fsdp_partition_matches_joint_graph(self) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real FSDP collective trace requires 2 GPUs")

        self._setup()
        traced_block = _trace_dsv3_moe_block_stage(
            fsdp_mesh=self.parallel_dims.get_mesh("fsdp")
        )

        fw_module, bw_module, meta = partition_joint_graph(
            traced_block.traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced_block.traced.example_inputs) - 1,),
        )

        joint_outputs = traced_block.traced.gm(*traced_block.flat_inputs)
        fw_args = [
            traced_block.flat_inputs[index] for index in meta.fwd_flat_input_indices
        ]
        fw_outputs = _boxed_run(fw_module, list(fw_args))
        bw_args = _backward_args_from_partition(
            meta,
            fw_outputs,
            (traced_block.output_grad,),
        )
        bw_outputs = _boxed_run(bw_module, list(bw_args))

        self.assertTrue(torch.equal(fw_outputs[0], joint_outputs[0]))
        _assert_tensor_sequence_equal(self, bw_outputs, joint_outputs[1:])


class GraphPPSplitDiDwTest(unittest.TestCase):
    def test_real_dsv3_moe_block_split_reconstructs_backward(self) -> None:
        traced_block = _trace_dsv3_moe_block_stage()
        fw_module, bw_module, meta = partition_joint_graph(
            traced_block.traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced_block.traced.example_inputs) - 1,),
        )
        split = split_di_dw_graph(
            bw_module,
            num_param_grads=traced_block.num_param_grad_values,
        )

        self.assertIsNotNone(split)
        if split is None:
            self.fail("Expected dI/dW split for decoder block with input grad")
        self.assertEqual(split.num_input_grads, 1)
        self.assertGreater(len(split.bw_dw_input_names), 0)

        fw_args = [
            traced_block.flat_inputs[index] for index in meta.fwd_flat_input_indices
        ]
        fw_outputs = _boxed_run(fw_module, list(fw_args))
        bw_args = _backward_args_from_partition(
            meta,
            fw_outputs,
            (traced_block.output_grad,),
        )
        full_bw_outputs = _boxed_run(bw_module, list(bw_args))

        di_outputs = _boxed_run(split.bw_di_module, list(bw_args))
        input_grads_to_prev = di_outputs[: split.num_input_grads]
        dw_live_ins = di_outputs[split.num_input_grads :]
        dw_outputs = _boxed_run(split.bw_dw_module, list(dw_live_ins))

        _assert_tensor_sequence_equal(
            self,
            input_grads_to_prev,
            full_bw_outputs[traced_block.num_param_grad_values :],
        )
        _assert_tensor_sequence_equal(
            self,
            dw_outputs,
            full_bw_outputs[: traced_block.num_param_grad_values],
        )

    def test_real_dsv3_moe_block_without_input_grad_skips_split(self) -> None:
        traced_block = _trace_dsv3_moe_block_stage(include_input_grad=False)
        _, bw_module, _ = partition_joint_graph(
            traced_block.traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced_block.traced.example_inputs) - 1,),
        )

        split = split_di_dw_graph(
            bw_module,
            num_param_grads=traced_block.num_param_grad_values,
        )

        self.assertIsNone(split)


_FAKE_PG = "graph_pp_test_pg"


def _call_targets(gm: fx.GraphModule) -> set[object]:
    return {node.target for node in gm.graph.nodes if node.op == "call_function"}


def _call_target_count(gm: fx.GraphModule, target: object) -> int:
    return sum(
        node.op == "call_function" and node.target == target for node in gm.graph.nodes
    )


def _pre_bucket_reduce_scatter_count(gm: fx.GraphModule) -> int:
    names = {
        "bucketing::_pre_bucket_reduce_scatter",
        "bucketing::_pre_bucket_reduce_scatter_chunk_cat",
    }
    return sum(
        node.op == "call_function"
        and getattr(getattr(node.target, "_schema", None), "name", None) in names
        for node in gm.graph.nodes
    )


def _mxfp8_weight_prepare_count(gm: fx.GraphModule) -> int:
    """Count dense MXFP8 preparation of unsharded weights."""
    targets = {
        torch.ops.torchao.triton_to_mxfp8_32x32_swizzle_dim0_and_dim1.default,
        torch.ops.torchao.triton_to_mxfp8_32x32_swizzle_dim0_and_dim1_out.default,
    }
    return sum(
        node.op == "call_function" and node.target in targets for node in gm.graph.nodes
    )


def _dist_moe_mxfp8_weight_prepare_count(gm: fx.GraphModule) -> int:
    """Count DistMoE MXFP8 preparation of unsharded expert weights."""
    return sum(
        node.op == "call_function"
        and node.target == torch.ops.dist_moe.prepare_mxfp8_weight.default
        for node in gm.graph.nodes
    )


def _placeholder_names(gm: fx.GraphModule) -> tuple[str, ...]:
    return tuple(node.name for node in gm.graph.find_nodes(op="placeholder"))


def _make_graph_module(graph: fx.Graph) -> fx.GraphModule:
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_forward_graph_with_unshard_and_replicated_param() -> fx.GraphModule:
    graph = fx.Graph()
    sharded_param = graph.placeholder("sharded_param")
    replicated_param = graph.placeholder("replicated_param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(sharded_param, 1, _FAKE_PG),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    split = graph.call_function(torch.ops.aten.split.Tensor, args=(wait, 2, 0))
    left = graph.call_function(operator.getitem, args=(split, 0))
    right = graph.call_function(operator.getitem, args=(split, 1))
    unsharded_param = graph.call_function(
        torch.ops.aten.cat.default,
        args=([left, right], 0),
    )
    prepared_param = graph.call_function(
        torch.ops.aten.neg.default,
        args=(unsharded_param,),
    )
    duplicate_all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(sharded_param, 1, _FAKE_PG),
    )
    duplicate_wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(duplicate_all_gather,),
    )
    duplicate_split = graph.call_function(
        torch.ops.aten.split.Tensor,
        args=(duplicate_wait, 2, 0),
    )
    duplicate_left = graph.call_function(operator.getitem, args=(duplicate_split, 0))
    duplicate_right = graph.call_function(operator.getitem, args=(duplicate_split, 1))
    duplicate_unsharded_param = graph.call_function(
        torch.ops.aten.cat.default,
        args=([duplicate_left, duplicate_right], 0),
    )
    duplicate_prepared_param = graph.call_function(
        torch.ops.aten.neg.default,
        args=(duplicate_unsharded_param,),
    )
    sharded_param_uses = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(prepared_param, duplicate_prepared_param),
    )
    params = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(sharded_param_uses, replicated_param),
    )
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(params, x))
    graph.output((out,))
    return _make_graph_module(graph)


def _make_forward_graph_without_fsdp() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(param, x))
    graph.output((out,))
    return _make_graph_module(graph)


def _make_joint_graph_with_raf_lifecycles() -> fx.GraphModule:
    """Build distinct forward/backward unshard and preparation chains."""
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    output_grad = graph.placeholder("output_grad")

    def unshard_and_prepare() -> fx.Node:
        """Append one synthetic FSDP unshard and weight-preparation chain."""
        all_gather = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(param, 1, _FAKE_PG),
        )
        wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(all_gather,),
        )
        return graph.call_function(torch.ops.aten.neg.default, args=(wait,))

    forward_weight = unshard_and_prepare()
    forward = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(forward_weight, x),
    )
    backward_weight = unshard_and_prepare()
    param_grad = graph.call_function(
        torch.ops.aten.mul.Tensor,
        args=(backward_weight, output_grad),
    )
    graph.output((forward, param_grad))
    return _make_graph_module(graph)


def _make_joint_graph_with_distinct_weight_preparations() -> fx.GraphModule:
    """Build duplicate unshards followed by incompatible preparation layouts."""
    graph = fx.Graph()
    param = graph.placeholder("param")

    def unshard() -> fx.Node:
        all_gather = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(param, 1, _FAKE_PG),
        )
        wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(all_gather,),
        )
        return graph.call_function(
            torch.ops.aten.view.default,
            args=(wait, [-1, 4]),
        )

    forward_weight = graph.call_function(
        torch.ops.aten.neg.default,
        args=(unshard(),),
    )
    forward_output = graph.call_function(
        torch.ops.aten.relu.default,
        args=(forward_weight,),
    )
    backward_preparation = graph.call_function(
        torch.ops.aten.max.dim,
        args=(unshard(), 1),
    )
    backward_weight = graph.call_function(
        operator.getitem,
        args=(backward_preparation, 0),
    )
    graph.output((forward_output, backward_weight))
    return _make_graph_module(graph)


def _make_bucketed_forward_graph(split_target: object) -> fx.GraphModule:
    """Build a two-parameter bucketed FSDP unshard graph."""
    graph = fx.Graph()
    first_param = graph.placeholder("first_param")
    second_param = graph.placeholder("second_param")
    first_input = graph.call_function(
        torch.ops.aten.view.default,
        args=(first_param, [-1]),
    )
    pre_bucket = graph.call_function(
        torch.ops.bucketing._pre_bucket_all_gather.default,
        args=([first_input, second_param], 2, torch.float32, [6, 6], 0),
    )
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor_out.default,
        args=(pre_bucket, 2, _FAKE_PG),
        kwargs={"out": pre_bucket},
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    reshaped = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(wait, [2, 8]),
    )
    split = graph.call_function(split_target, args=(reshaped, [4, 4], 1))
    first = graph.call_function(operator.getitem, args=(split, 0))
    second = graph.call_function(operator.getitem, args=(split, 1))
    graph.output((first, second))
    return _make_graph_module(graph)


def _make_bucketed_forward_out_split_graph() -> fx.GraphModule:
    """Build the mutating fused-unpack shape emitted by FSDP bucketing."""
    graph = fx.Graph()
    first_param = graph.placeholder("first_param")
    second_param = graph.placeholder("second_param")
    first_input = graph.call_function(
        torch.ops.aten.view.default,
        args=(first_param, [-1]),
    )
    pre_bucket = graph.call_function(
        torch.ops.bucketing._pre_bucket_all_gather.default,
        args=([first_input, second_param], 2, torch.float32, [6, 6], 0),
    )
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor_out.default,
        args=(pre_bucket, 2, _FAKE_PG),
        kwargs={"out": pre_bucket},
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    reshaped = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(wait, [2, 8]),
    )
    first_storage = graph.call_function(
        torch.ops.aten.empty.memory_format,
        args=([8],),
        kwargs={"dtype": torch.float32, "device": torch.device("cpu")},
    )
    second_storage = graph.call_function(
        torch.ops.aten.empty.memory_format,
        args=([8],),
        kwargs={"dtype": torch.float32, "device": torch.device("cpu")},
    )
    first_out = graph.call_function(
        torch.ops.aten.view.default,
        args=(first_storage, [2, 4]),
    )
    first_out = graph.call_function(
        torch.ops.aten.view.dtype,
        args=(first_out, torch.float32),
    )
    second_out = graph.call_function(
        torch.ops.aten.view.default,
        args=(second_storage, [2, 4]),
    )
    second_out = graph.call_function(
        torch.ops.aten.view.dtype,
        args=(second_out, torch.float32),
    )
    graph.call_function(
        torch.ops.fsdp.split_with_sizes_copy.default,
        args=(reshaped, [4, 4]),
        kwargs={"dim": 1, "out": [first_out, second_out]},
    )
    first = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(first_storage, [8]),
    )
    second = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(second_storage, [8]),
    )
    graph.output((first, second))
    return _make_graph_module(graph)


def _make_prepared_out_split_graph() -> fx.GraphModule:
    """Build a tuple-valued preparation after fused FSDP unpack."""
    graph = fx.Graph()
    param = graph.placeholder("param")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(param, 2, _FAKE_PG),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    gathered = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(wait, [2, 4]),
    )
    storage = graph.call_function(
        torch.ops.aten.empty.memory_format,
        args=([8],),
        kwargs={"dtype": torch.float32, "device": torch.device("cpu")},
    )
    out = graph.call_function(
        torch.ops.aten.view.default,
        args=(storage, [2, 4]),
    )
    graph.call_function(
        torch.ops.fsdp.split_with_sizes_copy.default,
        args=(gathered, [4]),
        kwargs={"dim": 1, "out": [out]},
    )
    prepared = graph.call_function(
        torch.ops.aten.max.dim,
        args=(out, 0),
    )
    prepared.meta[_PACKED_FSDP_UNSHARD_PARAM] = param.name
    values = graph.call_function(operator.getitem, args=(prepared, 0))
    indices = graph.call_function(operator.getitem, args=(prepared, 1))
    indices = graph.call_function(
        torch.ops.aten._to_copy.default,
        args=(indices,),
        kwargs={"dtype": torch.float32},
    )
    result = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(values, indices),
    )
    graph.output((result,))
    return _make_graph_module(graph)


class _FakeCollectiveInterpreter(fx.Interpreter):
    def call_function(self, target, args, kwargs):
        if target == torch.ops._c10d_functional.all_gather_into_tensor.default:
            return args[0].repeat(args[1])
        if target == torch.ops._c10d_functional.reduce_scatter_tensor.default:
            if args[2] != 1:
                raise ValueError("Test interpreter supports group size one only")
            return args[0]
        if target == torch.ops._c10d_functional.wait_tensor.default:
            return args[0]
        return super().call_function(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        module = self.fetch_attr(target)
        if isinstance(module, fx.GraphModule):
            return _FakeCollectiveInterpreter(module).run(*args, **kwargs)
        return super().call_module(target, args, kwargs)


def _make_forward_graph_without_wait() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(param, 1, _FAKE_PG),
    )
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(all_gather, x))
    graph.output((out,))
    return _make_graph_module(graph)


def _make_backward_graph_with_reduce_grad_epilogues() -> fx.GraphModule:
    graph = fx.Graph()
    fsdp_grad = graph.placeholder("fsdp_grad")
    ddp_grad = graph.placeholder("ddp_grad")
    input_grad = graph.placeholder("input_grad")
    cast = graph.call_function(
        torch.ops.aten._to_copy.default,
        args=(fsdp_grad,),
        kwargs={"dtype": torch.float32},
    )
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(cast, "sum", 1, _FAKE_PG),
    )
    reduce_scatter_wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    all_reduce = graph.call_function(
        torch.ops._c10d_functional.all_reduce.default,
        args=(ddp_grad, "sum", _FAKE_PG),
    )
    all_reduce_wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_reduce,),
    )
    graph.output((reduce_scatter_wait, all_reduce_wait, None, input_grad))
    return _make_graph_module(graph)


def _make_bucketed_backward_graph() -> fx.GraphModule:
    graph = fx.Graph()
    first_grad = graph.placeholder("first_grad")
    second_grad = graph.placeholder("second_grad")
    pre_bucket = graph.call_function(
        torch.ops.bucketing._pre_bucket_reduce_scatter.default,
        args=([first_grad, second_grad], 2),
    )
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(pre_bucket, "sum", 2, _FAKE_PG),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    split = graph.call_function(
        torch.ops.aten.split_with_sizes.default,
        args=(wait, [3, 5], 0),
    )
    first = graph.call_function(operator.getitem, args=(split, 0))
    second = graph.call_function(operator.getitem, args=(split, 1))
    first = graph.call_function(torch.ops.aten.reshape.default, args=(first, [3]))
    second = graph.call_function(torch.ops.aten.reshape.default, args=(second, [5]))
    graph.output((first, second))
    return _make_graph_module(graph)


def _make_reordered_bucketed_backward_graph() -> fx.GraphModule:
    graph = fx.Graph()
    first_grad = graph.placeholder("first_grad")
    second_grad = graph.placeholder("second_grad")
    first_grad.meta["val"] = torch.zeros(2, 3)
    second_grad.meta["val"] = torch.zeros(3, 5)
    first_alias = graph.call_function(
        torch.ops.aten.alias.default,
        args=(first_grad,),
    )
    first_alias.meta["val"] = first_grad.meta["val"]
    pre_bucket = graph.call_function(
        torch.ops.bucketing._pre_bucket_reduce_scatter.default,
        args=([second_grad, first_alias], 2, [0]),
    )
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(pre_bucket, "sum", 2, _FAKE_PG),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    split = graph.call_function(
        torch.ops.aten.split_with_sizes.default,
        args=(wait, [10, 3], 0),
    )
    second = graph.call_function(operator.getitem, args=(split, 0))
    first = graph.call_function(operator.getitem, args=(split, 1))
    first = graph.call_function(torch.ops.aten.alias.default, args=(first,))
    graph.output((first, second))
    return _make_graph_module(graph)


def _make_bucketed_deferred_trace() -> TracedResult:
    param0_value = torch.tensor([2.0, -1.0])
    param1_value = torch.tensor([0.5, -2.0, 3.0])
    buffer0_value = torch.zeros_like(param0_value)
    buffer1_value = torch.zeros_like(param1_value)
    input0_value = torch.tensor([1.0, 3.0])
    input1_value = torch.tensor([-1.0, 2.0, 4.0])

    graph = fx.Graph()

    def with_value(node: fx.Node, value: Any) -> fx.Node:
        node.meta["val"] = value
        return node

    param0 = with_value(graph.placeholder("param0"), param0_value)
    param1 = with_value(graph.placeholder("param1"), param1_value)
    with_value(graph.placeholder("buffer0"), buffer0_value)
    with_value(graph.placeholder("buffer1"), buffer1_value)
    input0 = with_value(graph.placeholder("input0"), input0_value)
    input1 = with_value(graph.placeholder("input1"), input1_value)

    all_gather = with_value(
        graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(param0, 1, _FAKE_PG),
        ),
        param0_value,
    )
    unsharded_param0 = with_value(
        graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(all_gather,),
        ),
        param0_value,
    )
    grad0_value = param0_value * input0_value
    grad1_value = param1_value * input1_value
    grad0 = with_value(
        graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(unsharded_param0, input0),
        ),
        grad0_value,
    )
    grad1 = with_value(
        graph.call_function(torch.ops.aten.mul.Tensor, args=(param1, input1)),
        grad1_value,
    )
    grad1_alias = with_value(
        graph.call_function(torch.ops.aten.alias.default, args=(grad1,)),
        grad1_value,
    )
    loss0 = with_value(
        graph.call_function(torch.ops.aten.sum.default, args=(input0,)),
        input0_value.sum(),
    )
    loss1 = with_value(
        graph.call_function(torch.ops.aten.sum.default, args=(input1,)),
        input1_value.sum(),
    )
    loss = with_value(
        graph.call_function(torch.ops.aten.add.Tensor, args=(loss0, loss1)),
        input0_value.sum() + input1_value.sum(),
    )

    packed_value = torch.cat((grad1_value, grad0_value))
    pre_bucket = with_value(
        graph.call_function(
            torch.ops.bucketing._pre_bucket_reduce_scatter.default,
            args=([grad1_alias, grad0], 1),
        ),
        packed_value,
    )
    reduce_scatter = with_value(
        graph.call_function(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            args=(pre_bucket, "sum", 1, _FAKE_PG),
        ),
        packed_value,
    )
    waited = with_value(
        graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(reduce_scatter,),
        ),
        packed_value,
    )
    split = with_value(
        graph.call_function(
            torch.ops.aten.split_with_sizes.default,
            args=(waited, [3, 2], 0),
        ),
        (grad1_value, grad0_value),
    )
    grad1_output = with_value(
        graph.call_function(operator.getitem, args=(split, 0)),
        grad1_value,
    )
    grad0_output = with_value(
        graph.call_function(operator.getitem, args=(split, 1)),
        grad0_value,
    )
    grad0_output = with_value(
        graph.call_function(torch.ops.aten.alias.default, args=(grad0_output,)),
        grad0_value,
    )
    grad0_output.meta["graph_state_outputs"] = (("param0", 0),)
    grad1_output.meta["graph_state_outputs"] = (("param1", 0),)
    graph.output((loss, grad0_output, grad1_output))
    gm = _make_graph_module(graph)

    _, user_inputs_spec = pytree.tree_flatten(((input0_value, input1_value), {}))
    _, output_spec = pytree.tree_flatten((loss.meta["val"], grad0_value, grad1_value))
    return TracedResult(
        gm=gm,
        example_inputs=(
            param0_value,
            param1_value,
            buffer0_value,
            buffer1_value,
            input0_value,
            input1_value,
        ),
        num_flat_inputs=6,
        input_subclass_layouts={},
        user_inputs_spec=user_inputs_spec,
        tensor_input_indices=list(range(6)),
        num_flat_outputs=3,
        output_subclass_layouts={},
        output_spec=output_spec,
        state_fqns=["param0", "param1"],
        graph_state_fqns=["param0", "param1"],
        graph_state_input_indices=((2,), (3,)),
        graph_state_output_indices=(1, 2),
        grad_sink_active=False,
    )


def _add_bucketed_all_gather_to_deferred_trace(
    traced: TracedResult,
) -> TracedResult:
    all_gather = traced.gm.graph.find_nodes(
        op="call_function",
        target=torch.ops._c10d_functional.all_gather_into_tensor.default,
    )[0]
    sharded_parameter = all_gather.args[0]
    with traced.gm.graph.inserting_before(all_gather):
        pre_bucket = traced.gm.graph.call_function(
            torch.ops.bucketing._pre_bucket_all_gather.default,
            args=([sharded_parameter], 1, torch.float32, [6], 0),
        )
    pre_bucket.meta["val"] = all_gather.meta["val"]
    all_gather.target = torch.ops._c10d_functional.all_gather_into_tensor_out.default
    all_gather.args = (pre_bucket, 1, _FAKE_PG)
    all_gather.kwargs = {"out": pre_bucket}
    wait = next(iter(all_gather.users))
    wait_users = tuple(wait.users)
    with traced.gm.graph.inserting_before(all_gather):
        unsharded_parameter = traced.gm.graph.call_function(
            torch.ops.aten.empty.memory_format,
            args=([2],),
            kwargs={"dtype": torch.float32},
        )
    unsharded_parameter.meta["val"] = all_gather.meta["val"]
    with traced.gm.graph.inserting_after(wait):
        split = traced.gm.graph.call_function(
            torch.ops.fsdp.split_with_sizes_copy.default,
            args=(wait, [2], 0),
            kwargs={"out": [unsharded_parameter]},
        )
    split.meta["val"] = None
    for user in wait_users:
        user.replace_input_with(wait, unsharded_parameter)
    traced.gm.graph.lint()
    traced.gm.recompile()
    return traced


def _make_backward_graph_without_fsdp() -> fx.GraphModule:
    graph = fx.Graph()
    grad = graph.placeholder("grad")
    out = graph.call_function(torch.ops.aten.neg.default, args=(grad,))
    graph.output((out,))
    return _make_graph_module(graph)


class GraphPPFSDPCollectiveSplitTest(unittest.TestCase):
    def test_prepared_boundary_survives_mutating_bucket_unpack(self) -> None:
        gm = _make_prepared_out_split_graph()
        split = split_fsdp_unshard_collectives(
            gm,
            num_params=1,
            input_names=("param",),
            flat_input_indices=(0,),
        )
        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected prepared FSDP unshard to be extracted")

        unpack = torch.ops.fsdp.split_with_sizes_copy.default
        prepare = torch.ops.aten.max.dim
        self.assertIn(unpack, _call_targets(split.unshard_module))
        self.assertIn(prepare, _call_targets(split.unshard_module))
        self.assertNotIn(unpack, _call_targets(split.compute_module))
        self.assertNotIn(prepare, _call_targets(split.compute_module))

        param = torch.arange(4, dtype=torch.float32)
        expected = _FakeCollectiveInterpreter(gm).run(param)
        unsharded = _FakeCollectiveInterpreter(split.unshard_module).run(param)
        actual = _FakeCollectiveInterpreter(split.compute_module).run(*unsharded)
        _assert_tensor_sequence_equal(self, actual, expected)

    def test_bucketed_reduce_scatter_finds_shared_pre_bucket_input(self) -> None:
        gm = _make_bucketed_backward_graph()
        pre_bucket = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.bucketing._pre_bucket_reduce_scatter.default,
        )[0]
        outputs = gm.graph.find_nodes(op="output")[0].args[0]

        self.assertIsNone(find_fsdp_reduce_grad_input(outputs[0]))
        self.assertIsNone(find_fsdp_reduce_grad_input(outputs[1]))
        self.assertIs(
            find_fsdp_reduce_grad_input(
                outputs[0],
                allow_bucket_fanout=True,
            ),
            pre_bucket,
        )
        self.assertIs(
            find_fsdp_reduce_grad_input(
                outputs[1],
                allow_bucket_fanout=True,
            ),
            pre_bucket,
        )

    def test_deferred_bucketed_reduce_scatter_maps_exact_logical_inputs(
        self,
    ) -> None:
        gm = _make_reordered_bucketed_backward_graph()
        first_grad, second_grad = gm.graph.find_nodes(op="placeholder")
        first_alias = next(
            node
            for node in gm.graph.find_nodes(
                op="call_function",
                target=torch.ops.aten.alias.default,
            )
            if node.args[0] is first_grad
        )
        pre_bucket = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.bucketing._pre_bucket_reduce_scatter.default,
        )[0]
        first_output, second_output = gm.graph.find_nodes(op="output")[0].args[0]

        first_boundary = find_fsdp_reduce_grad_boundary(first_output)
        second_boundary = find_fsdp_reduce_grad_boundary(second_output)
        self.assertIsNotNone(first_boundary)
        self.assertIsNotNone(second_boundary)
        if first_boundary is None or second_boundary is None:
            self.fail("Expected both bucket outputs to have reduction boundaries")
        self.assertIs(first_boundary.accumulation_input, first_alias)
        self.assertIs(second_boundary.accumulation_input, second_grad)
        self.assertIs(first_boundary.collective_input, pre_bucket)
        self.assertIs(second_boundary.collective_input, pre_bucket)

    def test_deferred_bucketed_reduce_scatter_validates_split_order(self) -> None:
        gm = _make_reordered_bucketed_backward_graph()
        split = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.split_with_sizes.default,
        )[0]
        split.args = (split.args[0], [3, 10], 0)
        first_output = gm.graph.find_nodes(op="output")[0].args[0][0]

        with self.assertRaisesRegex(ValueError, "do not match"):
            find_fsdp_reduce_grad_boundary(first_output)

    def test_deferred_bucketed_reduce_scatter_validates_layout_contract(
        self,
    ) -> None:
        for field in (
            "split_dim",
            "group_size",
            "unwrapped_indices",
            "packed",
            "legacy_packed",
        ):
            with self.subTest(field=field):
                gm = _make_reordered_bucketed_backward_graph()
                pre_bucket = gm.graph.find_nodes(
                    op="call_function",
                    target=torch.ops.bucketing._pre_bucket_reduce_scatter.default,
                )[0]
                reduce_scatter = gm.graph.find_nodes(
                    op="call_function",
                    target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
                )[0]
                split = gm.graph.find_nodes(
                    op="call_function",
                    target=torch.ops.aten.split_with_sizes.default,
                )[0]
                if field == "split_dim":
                    split.args = (split.args[0], split.args[1], 1)
                    expected = "dimension zero"
                elif field == "group_size":
                    reduce_scatter.args = (
                        reduce_scatter.args[0],
                        reduce_scatter.args[1],
                        4,
                        reduce_scatter.args[3],
                    )
                    expected = "Group sizes"
                elif field == "unwrapped_indices":
                    pre_bucket.args = (pre_bucket.args[0], 2, [2])
                    expected = "Invalid unwrapped input indices"
                elif field == "packed":
                    pre_bucket.args = (pre_bucket.args[0], 2, [])
                    expected = "is not divisible"
                else:
                    pre_bucket.args = (pre_bucket.args[0], 2)
                    expected = "is not divisible"
                first_output = gm.graph.find_nodes(op="output")[0].args[0][0]

                with self.assertRaisesRegex(ValueError, expected):
                    find_fsdp_reduce_grad_boundary(first_output)

    def test_deferred_bucketed_hsdp_supports_post_rs_all_reduce(self) -> None:
        gm = _make_reordered_bucketed_backward_graph()
        output = gm.graph.find_nodes(op="output")[0]
        first_output, second_output = output.args[0]
        with gm.graph.inserting_before(output):
            all_reduce = gm.graph.call_function(
                torch.ops._c10d_functional.all_reduce.default,
                args=(first_output, "sum", _FAKE_PG),
            )
            waited = gm.graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(all_reduce,),
            )
        output.args = ((waited, second_output),)
        gm.graph.lint()
        gm.recompile()

        boundary = find_fsdp_reduce_grad_boundary(waited)

        self.assertIsNotNone(boundary)
        if boundary is None:
            self.fail("Expected a bucketed HSDP reduction boundary")
        first_grad = gm.graph.find_nodes(op="placeholder")[0]
        first_alias = next(
            node
            for node in gm.graph.find_nodes(
                op="call_function",
                target=torch.ops.aten.alias.default,
            )
            if node.args[0] is first_grad
        )
        self.assertIs(boundary.accumulation_input, first_alias)

    def test_deferred_bucketed_hsdp_rejects_pre_rs_all_reduce(self) -> None:
        gm = _make_reordered_bucketed_backward_graph()
        _, second_grad = gm.graph.find_nodes(op="placeholder")
        pre_bucket = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.bucketing._pre_bucket_reduce_scatter.default,
        )[0]
        with gm.graph.inserting_before(pre_bucket):
            all_reduce = gm.graph.call_function(
                torch.ops._c10d_functional.all_reduce.default,
                args=(second_grad, "sum", _FAKE_PG),
            )
            all_reduce.meta["val"] = second_grad.meta["val"]
            waited = gm.graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(all_reduce,),
            )
            waited.meta["val"] = second_grad.meta["val"]
        first_alias = pre_bucket.args[0][1]
        pre_bucket.args = ([waited, first_alias], 2, [0])
        gm.graph.lint()
        gm.recompile()
        second_output = gm.graph.find_nodes(op="output")[0].args[0][1]

        with self.assertRaisesRegex(
            NotImplementedError,
            "reduce-scatter to be the first gradient collective",
        ):
            find_fsdp_reduce_grad_boundary(second_output)

    def test_deferred_bucketed_reduce_scatter_rejects_input_fanout(self) -> None:
        gm = _make_reordered_bucketed_backward_graph()
        _, second_grad = gm.graph.find_nodes(op="placeholder")
        output = gm.graph.find_nodes(op="output")[0]
        with gm.graph.inserting_before(output):
            gm.graph.call_function(torch.ops.aten.neg.default, args=(second_grad,))
        gm.graph.lint()
        gm.recompile()
        second_output = output.args[0][1]

        with self.assertRaisesRegex(NotImplementedError, "sole user"):
            find_fsdp_reduce_grad_boundary(second_output)

    def test_deferred_bucketed_reduce_scatter_rejects_pre_bucket_fanout(
        self,
    ) -> None:
        gm = _make_reordered_bucketed_backward_graph()
        pre_bucket = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.bucketing._pre_bucket_reduce_scatter.default,
        )[0]
        output = gm.graph.find_nodes(op="output")[0]
        with gm.graph.inserting_before(output):
            gm.graph.call_function(torch.ops.aten.neg.default, args=(pre_bucket,))
        gm.graph.lint()
        gm.recompile()
        first_output = output.args[0][0]

        with self.assertRaisesRegex(NotImplementedError, "pre-bucket"):
            find_fsdp_reduce_grad_boundary(first_output)

    def test_deferred_bucketed_reduce_scatter_accumulates_before_packing(
        self,
    ) -> None:
        traced = _make_bucketed_deferred_trace()
        deferred = build_deferred_fsdp_graph(
            traced,
            num_flat_parameters=2,
            num_microbatches=3,
            compile_config=GraphTrainerCompileConfig(
                enable_passes=False,
                inductor_compilation="none",
                disable_passes=["cudagraph_pass"],
            ),
            enable_cudagraph=False,
        )
        reduce_scatter = torch.ops._c10d_functional.reduce_scatter_tensor.default
        for graph_name in ("first", "middle"):
            graph = getattr(deferred.gm, graph_name)
            self.assertEqual(_pre_bucket_reduce_scatter_count(graph), 0)
            self.assertEqual(_call_target_count(graph, reduce_scatter), 0)
        self.assertEqual(_pre_bucket_reduce_scatter_count(deferred.gm.final), 1)
        self.assertEqual(_call_target_count(deferred.gm.final, reduce_scatter), 1)
        self.assertEqual(deferred.num_reduce_scatters, 1)

        param0, param1, buffer0, buffer1, _, _ = traced.example_inputs
        microbatches = (
            (
                torch.tensor([1.0, 3.0]),
                torch.tensor([-1.0, 2.0, 4.0]),
            ),
            (
                torch.tensor([-2.0, 1.0]),
                torch.tensor([3.0, 0.5, -1.0]),
            ),
            (
                torch.tensor([4.0, -3.0]),
                torch.tensor([2.0, -2.0, 0.25]),
            ),
        )
        expected_grad0 = sum(param0 * inputs[0] for inputs in microbatches)
        expected_grad1 = sum(param1 * inputs[1] for inputs in microbatches)
        expected_loss = sum(
            input0.sum() + input1.sum() for input0, input1 in microbatches
        )
        flat_inputs = [param0, param1, buffer0, buffer1]
        for inputs in microbatches:
            flat_inputs.extend(inputs)

        outputs = _FakeCollectiveInterpreter(deferred.gm).run(*flat_inputs)

        self.assertEqual(len(outputs), 1)
        torch.testing.assert_close(outputs[0], expected_loss)
        torch.testing.assert_close(buffer0, expected_grad0)
        torch.testing.assert_close(buffer1, expected_grad1)

    def test_deferred_symmetric_memory_rewrites_extracted_children(self) -> None:
        traced = _add_bucketed_all_gather_to_deferred_trace(
            _make_bucketed_deferred_trace()
        )
        process_group = object()
        with (
            mock.patch(
                "torchtitan.experiments.graph_trainer.fsdp_passes.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "torchtitan.experiments.graph_trainer.fsdp_passes.dist.get_backend",
                return_value=torch.distributed.Backend.NCCL,
            ),
            mock.patch(
                "torchtitan.experiments.graph_trainer.fsdp_passes.dist.get_rank",
                return_value=0,
            ),
            mock.patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value=process_group,
            ),
            mock.patch(
                "torchtitan.experiments.graph_trainer.fsdp_passes.symm_mem.get_backend",
                return_value="NCCL",
            ),
            mock.patch(
                "torchtitan.experiments.graph_trainer.fsdp_passes.dist.barrier"
            ) as barrier,
        ):
            deferred = build_deferred_fsdp_graph(
                traced,
                num_flat_parameters=2,
                num_microbatches=3,
                compile_config=GraphTrainerCompileConfig(
                    enable_passes=False,
                    inductor_compilation="full",
                    disable_passes=["cudagraph_pass"],
                ),
                enable_cudagraph=False,
                fsdp_symm_mem_policy="widest",
            )

        symm_all_gather = torch.ops.bucketing._pre_bucket_all_gather_symm_mem.default
        symm_reduce_scatter = (
            torch.ops.bucketing._pre_bucket_reduce_scatter_symm_mem.default
        )
        self.assertEqual(_call_target_count(deferred.gm.first, symm_all_gather), 1)
        self.assertEqual(_call_target_count(deferred.gm.middle, symm_all_gather), 0)
        self.assertEqual(_call_target_count(deferred.gm.final, symm_all_gather), 0)
        self.assertEqual(
            _call_target_count(deferred.gm.first, symm_reduce_scatter),
            0,
        )
        self.assertEqual(
            _call_target_count(deferred.gm.middle, symm_reduce_scatter),
            0,
        )
        self.assertEqual(
            _call_target_count(deferred.gm.final, symm_reduce_scatter),
            1,
        )
        self.assertEqual(deferred.num_all_gathers, 1)
        self.assertEqual(deferred.num_reduce_scatters, 1)
        barrier.assert_called_once_with(group=process_group, device_ids=[0])

    def test_deferred_symmetric_memory_preallocates_for_eager_children(self) -> None:
        for compilation, expected in (
            ("none", True),
            ("regional", True),
            ("full", False),
        ):
            with (
                self.subTest(compilation=compilation),
                mock.patch(
                    "torchtitan.experiments.graph_trainer.deferred_fsdp."
                    "enable_fsdp_symmetric_memory_for_graphs",
                    return_value=(0, 0),
                ) as enable_symmetric_memory,
            ):
                build_deferred_fsdp_graph(
                    _make_bucketed_deferred_trace(),
                    num_flat_parameters=2,
                    num_microbatches=3,
                    compile_config=GraphTrainerCompileConfig(
                        enable_passes=False,
                        inductor_compilation=compilation,
                        disable_passes=["cudagraph_pass"],
                    ),
                    enable_cudagraph=False,
                    fsdp_symm_mem_policy="widest",
                )
                self.assertEqual(
                    enable_symmetric_memory.call_args.kwargs["preallocate"],
                    expected,
                )

    def test_raf_policy_selects_per_microbatch_or_stage_lifetime(self) -> None:
        """RAF=true keeps both lifecycles; RAF=false extracts one prepared value."""
        all_gather = torch.ops._c10d_functional.all_gather_into_tensor.default
        prepare = torch.ops.aten.neg.default

        raf_true_joint = _make_joint_graph_with_raf_lifecycles()
        raf_true_fw, raf_true_bw, _ = partition_joint_graph(
            types.SimpleNamespace(gm=raf_true_joint),
            num_fwd_outputs=1,
            backward_only_input_indices=(2,),
        )
        for graph in (raf_true_fw, raf_true_bw):
            self.assertEqual(_call_target_count(graph, all_gather), 1)
            self.assertEqual(_call_target_count(graph, prepare), 1)

        raf_false_joint = _make_joint_graph_with_raf_lifecycles()
        deduplicate_fsdp_unshard_chains_pass(raf_false_joint)
        split = split_fsdp_unshard_collectives(
            raf_false_joint,
            num_params=1,
            input_names=("param", "x", "output_grad"),
            flat_input_indices=(0, 1, 2),
        )
        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected RAF=false to extract one FSDP unshard")
        raf_false_fw, raf_false_bw, _ = partition_joint_graph(
            types.SimpleNamespace(gm=split.compute_module),
            num_fwd_outputs=1,
            backward_only_input_indices=(2,),
            flat_input_indices=split.compute_flat_input_indices,
        )
        self.assertEqual(_call_target_count(split.unshard_module, all_gather), 1)
        self.assertEqual(_call_target_count(split.unshard_module, prepare), 1)
        for graph in (raf_false_fw, raf_false_bw):
            self.assertEqual(_call_target_count(graph, all_gather), 0)
            self.assertEqual(_call_target_count(graph, prepare), 0)

    def test_dedup_preserves_distinct_post_all_gather_preparations(self) -> None:
        """Dedup shares BF16 reconstruction without merging format layouts."""
        gm = _make_joint_graph_with_distinct_weight_preparations()
        deduplicate_fsdp_unshard_chains_pass(gm)

        self.assertEqual(
            _call_target_count(
                gm,
                torch.ops._c10d_functional.all_gather_into_tensor.default,
            ),
            1,
        )
        self.assertEqual(_call_target_count(gm, torch.ops.aten.neg.default), 1)
        self.assertEqual(_call_target_count(gm, torch.ops.aten.max.dim), 1)

        split = split_fsdp_unshard_collectives(
            gm,
            num_params=1,
            input_names=("param",),
            flat_input_indices=(0,),
        )
        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected the shared FSDP reconstruction to be extracted")
        self.assertIn(
            torch.ops.aten.neg.default,
            _call_targets(split.unshard_module),
        )
        self.assertIn(
            torch.ops.aten.max.dim,
            _call_targets(split.unshard_module),
        )
        self.assertNotIn(
            torch.ops.aten.neg.default,
            _call_targets(split.compute_module),
        )
        self.assertNotIn(
            torch.ops.aten.max.dim,
            _call_targets(split.compute_module),
        )

    def test_bucketed_unshard_matches_view_and_copy_splits(self) -> None:
        """GraphPP maps each bucket input to either split implementation."""
        for split_target in (
            torch.ops.aten.split_with_sizes.default,
            torch.ops.aten.split_with_sizes_copy.default,
        ):
            with self.subTest(split_target=split_target):
                gm = _make_bucketed_forward_graph(split_target)
                first_param, second_param = gm.graph.find_nodes(op="placeholder")
                split = gm.graph.find_nodes(
                    op="call_function",
                    target=split_target,
                )[0]

                self.assertIs(
                    find_fsdp_unshard_output(first_param),
                    next(user for user in split.users if user.args[1] == 0),
                )
                self.assertIs(
                    find_fsdp_unshard_output(second_param),
                    next(user for user in split.users if user.args[1] == 1),
                )

    def test_bucketed_unshard_matches_mutating_out_split(self) -> None:
        """GraphPP maps inputs through FSDP's fused out-buffer unpack."""
        gm = _make_bucketed_forward_out_split_graph()
        first_param, second_param = gm.graph.find_nodes(op="placeholder")
        empty_outputs = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.empty.memory_format,
        )

        self.assertIs(find_fsdp_unshard_output(first_param), empty_outputs[0])
        self.assertIs(find_fsdp_unshard_output(second_param), empty_outputs[1])

        split = split_fsdp_unshard_collectives(
            gm,
            num_params=2,
            input_names=("first_param", "second_param"),
            flat_input_indices=(0, 1),
        )
        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected fused FSDP unpack to be extracted")
        self.assertIn(
            torch.ops.fsdp.split_with_sizes_copy.default,
            _call_targets(split.unshard_module),
        )
        self.assertNotIn(
            torch.ops.fsdp.split_with_sizes_copy.default,
            _call_targets(split.compute_module),
        )

    def test_bucketed_unshard_rejects_malformed_out_split(self) -> None:
        """Fused unpack outputs must map one-to-one to bucket inputs."""
        gm = _make_bucketed_forward_out_split_graph()
        first_param = gm.graph.find_nodes(op="placeholder")[0]
        out_split = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.fsdp.split_with_sizes_copy.default,
        )[0]
        out = out_split.kwargs["out"]
        out_split.kwargs = {**out_split.kwargs, "out": out[:1]}

        with self.assertRaisesRegex(ValueError, "Expected 2 outputs"):
            find_fsdp_unshard_output(first_param)

        gm = _make_bucketed_forward_out_split_graph()
        first_param = gm.graph.find_nodes(op="placeholder")[0]
        out_split = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.fsdp.split_with_sizes_copy.default,
        )[0]
        out = out_split.kwargs["out"]
        out_split.kwargs = {**out_split.kwargs, "out": [out[0], out[0]]}

        with self.assertRaisesRegex(ValueError, "distinct output storage"):
            find_fsdp_unshard_output(first_param)

        gm = _make_bucketed_forward_out_split_graph()
        first_param, second_param = gm.graph.find_nodes(op="placeholder")
        out_split = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.fsdp.split_with_sizes_copy.default,
        )[0]
        out = out_split.kwargs["out"]
        out_split.kwargs = {
            **out_split.kwargs,
            "out": [first_param, out[1]],
        }

        with self.assertRaisesRegex(ValueError, "fresh output storage"):
            find_fsdp_unshard_output(second_param)

    def test_forward_pattern_matches_reshard_force_save_pattern(self) -> None:
        gm = _make_forward_graph_with_unshard_and_replicated_param()
        deduplicate_fsdp_unshard_chains_pass(gm)
        sharded_param = gm.graph.find_nodes(op="placeholder")[0]
        save_nodes = find_fsdp_unshard_save_nodes(sharded_param)

        self.assertEqual(len(save_nodes), 1)
        self.assertIs(find_fsdp_unshard_output(sharded_param), save_nodes[0])
        self.assertIs(find_fsdp_unshard_save_node(sharded_param), save_nodes[0])

    def test_forward_split_extracts_unshard_and_replicated_params(self) -> None:
        gm = _make_forward_graph_with_unshard_and_replicated_param()
        deduplicate_fsdp_unshard_chains_pass(gm)

        split = split_fsdp_unshard_collectives(
            gm,
            num_params=2,
            input_names=("sharded_param", "replicated_param", "x"),
            flat_input_indices=(0, 1, 2),
        )

        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected forward FSDP split to extract an unshard graph")
        self.assertEqual(
            _placeholder_names(split.unshard_module),
            ("sharded_param", "replicated_param"),
        )
        self.assertEqual(split.unshard_flat_param_indices, (0, 1))
        self.assertEqual(split.compute_flat_input_indices, (0, 1, 2))
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(split.unshard_module),
        )
        self.assertIn(torch.ops.aten.cat.default, _call_targets(split.unshard_module))
        self.assertIn(torch.ops.aten.neg.default, _call_targets(split.unshard_module))
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(split.compute_module),
        )
        self.assertNotIn(
            torch.ops.aten.neg.default,
            _call_targets(split.compute_module),
        )

    def test_forward_split_no_fsdp_is_noop(self) -> None:
        gm = _make_forward_graph_without_fsdp()
        split = split_fsdp_unshard_collectives(
            gm,
            num_params=1,
            input_names=("param", "x"),
            flat_input_indices=(0, 1),
        )

        self.assertIsNone(split.unshard_module)
        self.assertIs(split.compute_module, gm)
        self.assertEqual(_placeholder_names(split.compute_module), ("param", "x"))
        self.assertEqual(split.compute_flat_input_indices, (0, 1))

    def test_forward_split_requires_wait_after_all_gather(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected wait_tensor"):
            split_fsdp_unshard_collectives(
                _make_forward_graph_without_wait(),
                num_params=1,
                input_names=("param", "x"),
                flat_input_indices=(0, 1),
            )

    def test_backward_split_extracts_reduce_grad_epilogues(self) -> None:
        split = split_backward_fsdp_collectives(
            _make_backward_graph_with_reduce_grad_epilogues(),
            num_param_grads=3,
        )

        self.assertIsNotNone(split.reduce_grad_module)
        if split.reduce_grad_module is None:
            self.fail("Expected backward FSDP split to extract reduce-grad graph")
        bw_no_fsdp_targets = _call_targets(split.bw_no_fsdp_module)
        reduce_grad_targets = _call_targets(split.reduce_grad_module)
        self.assertIn(torch.ops.aten._to_copy.default, bw_no_fsdp_targets)
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            bw_no_fsdp_targets,
        )
        self.assertNotIn(
            torch.ops._c10d_functional.all_reduce.default,
            bw_no_fsdp_targets,
        )
        self.assertNotIn(torch.ops.aten._to_copy.default, reduce_grad_targets)
        self.assertIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            reduce_grad_targets,
        )
        self.assertIn(
            torch.ops._c10d_functional.all_reduce.default,
            reduce_grad_targets,
        )
        self.assertEqual(
            split.reduce_grad_input_names,
            split.bw_no_fsdp_output_names[:2],
        )
        self.assertEqual(len(split.bw_no_fsdp_output_names), 4)
        self.assertEqual(split.bw_no_fsdp_output_names[-1], "input_grad")

    def test_backward_split_no_fsdp_is_noop_and_validates_grad_count(self) -> None:
        gm = _make_backward_graph_without_fsdp()
        split = split_backward_fsdp_collectives(gm, num_param_grads=1)

        self.assertIsNone(split.reduce_grad_module)
        self.assertIs(split.bw_no_fsdp_module, gm)

        with self.assertRaisesRegex(ValueError, "num_param_grads cannot exceed"):
            split_backward_fsdp_collectives(gm, num_param_grads=2)


class GraphPPFSDPCollectiveSplitDsv3Test(_GraphPPDsv3FSDPTest):
    def test_real_dist_moe_mxfp8_bucket_then_split_hoists_weight_prepare(
        self,
    ) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real FSDP collective trace requires 2 GPUs")

        self._setup()
        self.addCleanup(cleanup_dist_moe)
        model_spec = dsv3_model_registry(
            "debugmodel",
            moe_backend="dist_moe",
            dist_moe=DistMoeBackendConfig(
                vmm_host_scratch_imbalance_factor=None,
                blockscaled=DistMoeBlockScaledConfig(
                    format=BlockScaledFormat.MXFP8_E4M3,
                ),
            ),
        )
        runtime_config = Trainer.Config(
            model_spec=model_spec,
            training=TrainingConfig(
                num_tokens_per_microbatch_per_dp_rank=128,
                num_tokens_per_train_step=128 * self.world_size,
                max_context_length=128,
                steps=1,
                mixed_precision_param="bfloat16",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=-1,
                expert_parallel_degree=1,
            ),
            checkpoint=CheckpointManager.Config(initial_load_model_only=False),
            debug=DebugConfig(seed=0, deterministic=True),
        )
        model_spec.model.update_from_config(config=runtime_config)
        moe_config = model_spec.model.layers[1].moe
        if moe_config is None:
            self.fail("DeepSeek V3 debug layer 1 must contain an MoE block")
        with torch.device("meta"):
            experts = moe_config.routed_experts.build()
        model = _DistMoePreparedWeightConsumer(experts)
        _configure_fsdp_modules(model)
        annotate_module_fqns(model)
        model = data_parallel(
            model,
            device_mesh=self.parallel_dims.get_mesh("fsdp"),
            mode="fully_shard",
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
        )
        setup_dist_moe(
            config=runtime_config,
            model_parts=[model],
            parallel_dims=self.parallel_dims,
            device=torch.device("cuda"),
        )
        model.to_empty(device="cuda")
        with torch.no_grad(), disable_active_parametrization():
            experts.init_states(buffer_device=None)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        gradient_state = GraphGradientState.create(model, [optimizer])
        x = torch.randn(
            runtime_config.training.num_tokens_per_microbatch_per_dp_rank,
            experts.hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        scores = torch.full(
            (
                runtime_config.training.num_tokens_per_microbatch_per_dp_rank,
                experts.top_k,
            ),
            1.0 / experts.top_k,
            device="cuda",
            dtype=torch.float32,
        )
        expert_ids = torch.arange(experts.top_k, device="cuda").view(1, -1)
        expert_ids = expert_ids.expand(
            runtime_config.training.num_tokens_per_microbatch_per_dp_rank,
            -1,
        )
        local_tokens = torch.zeros(
            experts.num_experts,
            device="cuda",
            dtype=torch.int64,
        )
        labels = (scores, expert_ids, local_tokens)
        global_valid_tokens = torch.ones((), device="cuda")

        def stage_step(
            x: torch.Tensor,
            labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            global_valid_tokens: torch.Tensor,
            extra_kwargs: dict[str, Any],
        ):
            del global_valid_tokens, extra_kwargs
            scores, expert_ids, local_tokens = labels
            output = model(x, scores, expert_ids, local_tokens)
            parameters = [
                parameter
                for _, parameter in model.named_parameters(remove_duplicate=False)
                if parameter.requires_grad
            ]
            loss = output.float().sum()
            gradients = torch.autograd.grad(
                loss,
                parameters,
            )
            return [loss, *gradients]

        traced = minimal_fx_tracer(
            stage_step,
            module=model,
            graph_state=gradient_state.graph_state,
            graph_state_output_indices=tuple(
                range(1, 1 + len(gradient_state.parameters))
            ),
        )(
            x,
            labels,
            global_valid_tokens,
            {},
        )

        gm = traced.gm
        eliminate_dead_code_pass(gm, traced.example_inputs)
        canonicalize_graph_pass(gm, traced.example_inputs)
        deduplicate_fsdp_unshard_chains_pass(gm, traced.example_inputs)
        preserve_fsdp_unshard_output_boundaries_pass(
            gm,
            num_model_state_tensor_inputs=traced.num_model_state_tensor_inputs,
        )
        gm = joint_transformer_block_bucketing_reordering_pass(
            gm,
            traced.example_inputs,
            module_bucket_plans=["experts"],
            fsdp_param_module_order=get_fsdp_param_module_order(traced.state_fqns),
        )
        gm = materialize_fsdp_bucket_outputs_pass(
            gm,
            traced.example_inputs,
            module_fqn_patterns=("experts",),
        )
        traced.gm = gm

        input_names = _placeholder_names(gm)
        num_flat_param_values = len(
            flatten_graph_values(
                [
                    parameter
                    for _, parameter in model.named_parameters(remove_duplicate=False)
                ]
            )
        )
        unshard_split = split_fsdp_unshard_collectives(
            gm,
            num_params=num_flat_param_values,
            input_names=input_names,
            flat_input_indices=tuple(range(len(input_names))),
        )
        self.assertIsNotNone(unshard_split.unshard_module)
        if unshard_split.unshard_module is None:
            self.fail("Expected bucketed DistMoE FSDP trace to contain an unshard")

        self.assertEqual(
            _dist_moe_mxfp8_weight_prepare_count(unshard_split.unshard_module),
            2,
        )
        self.assertEqual(
            _dist_moe_mxfp8_weight_prepare_count(unshard_split.compute_module),
            0,
        )
        baseline_deferred = build_deferred_fsdp_graph(
            traced,
            num_flat_parameters=num_flat_param_values,
            num_microbatches=3,
            compile_config=GraphTrainerCompileConfig(
                enable_passes=True,
                inductor_compilation="none",
                numerics_changing_optim=False,
                disable_passes=["cudagraph_pass"],
            ),
            enable_cudagraph=False,
        )
        deferred = build_deferred_fsdp_graph(
            traced,
            num_flat_parameters=num_flat_param_values,
            num_microbatches=3,
            compile_config=GraphTrainerCompileConfig(
                enable_passes=True,
                inductor_compilation="none",
                numerics_changing_optim=True,
                disable_passes=["cudagraph_pass"],
            ),
            enable_cudagraph=False,
        )
        cudagraph_deferred = build_deferred_fsdp_graph(
            traced,
            num_flat_parameters=num_flat_param_values,
            num_microbatches=3,
            compile_config=GraphTrainerCompileConfig(
                enable_passes=True,
                inductor_compilation="none",
                numerics_changing_optim=True,
                require_cudagraph=True,
            ),
            enable_cudagraph=True,
        )
        num_pre_buckets = _pre_bucket_reduce_scatter_count(traced.gm)
        self.assertGreater(num_pre_buckets, 0)
        self.assertEqual(_pre_bucket_reduce_scatter_count(deferred.gm.first), 0)
        self.assertEqual(_pre_bucket_reduce_scatter_count(deferred.gm.middle), 0)
        self.assertEqual(
            _pre_bucket_reduce_scatter_count(deferred.gm.final),
            num_pre_buckets,
        )
        self.assertEqual(
            _dist_moe_mxfp8_weight_prepare_count(deferred.gm.first),
            2,
        )
        self.assertEqual(
            _dist_moe_mxfp8_weight_prepare_count(deferred.gm.middle),
            0,
        )
        self.assertEqual(
            _dist_moe_mxfp8_weight_prepare_count(deferred.gm.final),
            0,
        )
        ordinary_backward = torch.ops.dist_moe.block_scaled_backward.default
        accumulating_backward = (
            torch.ops.dist_moe.block_scaled_backward_accumulate.default
        )
        self.assertEqual(_call_target_count(deferred.gm.first, ordinary_backward), 1)
        self.assertEqual(
            _call_target_count(deferred.gm.first, accumulating_backward),
            0,
        )
        for child in (deferred.gm.middle, deferred.gm.final):
            self.assertEqual(
                _call_target_count(child, ordinary_backward),
                0,
                str(child.graph),
            )
            self.assertEqual(_call_target_count(child, accumulating_backward), 1)

        baseline_run = bind_deferred_fsdp_graph(
            baseline_deferred,
            module=model,
            gradient_state=gradient_state,
        )
        fused_run = bind_deferred_fsdp_graph(
            deferred,
            module=model,
            gradient_state=gradient_state,
        )
        cudagraph_run = bind_deferred_fsdp_graph(
            cudagraph_deferred,
            module=model,
            gradient_state=gradient_state,
        )
        microbatches = [(x, labels, {}) for _ in range(3)]
        reference_loss = baseline_run(microbatches, global_valid_tokens)
        reference_gradients = [
            buffer.to_local().clone() for buffer in gradient_state.buffers
        ]
        for buffer in gradient_state.buffers:
            buffer.zero_()
        actual_loss = fused_run(microbatches, global_valid_tokens)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_loss, reference_loss)
        for actual, expected in zip(
            gradient_state.buffers,
            reference_gradients,
            strict=True,
        ):
            torch.testing.assert_close(
                actual.to_local(),
                expected,
                rtol=2e-2,
                atol=2e-2,
            )
        for _ in range(3):
            for buffer in gradient_state.buffers:
                buffer.zero_()
            cudagraph_loss = cudagraph_run(microbatches, global_valid_tokens)
            torch.cuda.synchronize()
            torch.testing.assert_close(cudagraph_loss, reference_loss)
            for actual, expected in zip(
                gradient_state.buffers,
                reference_gradients,
                strict=True,
            ):
                torch.testing.assert_close(
                    actual.to_local(),
                    expected,
                    rtol=2e-2,
                    atol=2e-2,
                )
        self.assertIsNotNone(cudagraph_deferred.gm.forward._cudagraph)
        cudagraph_deferred.gm.forward.teardown()

    def test_real_mxfp8_dsv3_bucket_then_split_hoists_weight_prepare(self) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real FSDP collective trace requires 2 GPUs")

        self._setup()
        traced_block = _trace_dsv3_moe_block_stage(
            batch_size=1,
            fsdp_mesh=self.parallel_dims.get_mesh("fsdp"),
            quantize_dense=True,
            annotate_modules=True,
        )
        gm = traced_block.traced.gm
        deduplicate_fsdp_unshard_chains_pass(
            gm,
            traced_block.traced.example_inputs,
        )
        preserve_fsdp_unshard_output_boundaries_pass(
            gm,
            num_model_state_tensor_inputs=(
                traced_block.traced.num_model_state_tensor_inputs
            ),
        )
        gm = joint_transformer_block_bucketing_reordering_pass(
            gm,
            traced_block.traced.example_inputs,
            module_bucket_plans=[["attention", "attention_norm", "ffn_norm", "moe"]],
            fsdp_param_module_order=get_fsdp_param_module_order(
                traced_block.traced.state_fqns
            ),
        )

        input_names = _placeholder_names(gm)
        unshard_split = split_fsdp_unshard_collectives(
            gm,
            num_params=traced_block.num_flat_param_values,
            input_names=input_names,
            flat_input_indices=tuple(range(len(input_names))),
        )
        self.assertIsNotNone(unshard_split.unshard_module)
        if unshard_split.unshard_module is None:
            self.fail("Expected bucketed DSV3 FSDP trace to contain an unshard")

        self.assertGreater(
            _mxfp8_weight_prepare_count(unshard_split.unshard_module),
            0,
        )
        self.assertEqual(
            _mxfp8_weight_prepare_count(unshard_split.compute_module),
            0,
        )
        pre_bucket = torch.ops.bucketing._pre_bucket_all_gather.default
        self.assertIn(
            pre_bucket,
            _call_targets(unshard_split.unshard_module),
        )
        self.assertNotIn(
            pre_bucket,
            _call_targets(unshard_split.compute_module),
        )

        expected_outputs = _boxed_run(gm, list(traced_block.flat_inputs))
        unshard_args = [
            traced_block.flat_inputs[index]
            for index in unshard_split.unshard_flat_param_indices
        ]
        unsharded_params = _boxed_run(
            unshard_split.unshard_module,
            unshard_args,
        )
        materialized_flat_inputs = list(traced_block.flat_inputs)
        for flat_index, value in zip(
            unshard_split.unshard_flat_param_indices,
            unsharded_params,
            strict=True,
        ):
            materialized_flat_inputs[flat_index] = value
        compute_args = [
            materialized_flat_inputs[index]
            for index in unshard_split.compute_flat_input_indices
        ]
        actual_outputs = _boxed_run(
            unshard_split.compute_module,
            compute_args,
        )
        _assert_tensor_sequence_equal(self, actual_outputs, expected_outputs)

    def test_real_mxfp8_dsv3_fsdp_split_reconstructs_graphs(self) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real FSDP collective trace requires 2 GPUs")

        self._setup()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        traced_block = _trace_dsv3_moe_block_stage(
            fsdp_mesh=fsdp_mesh,
            quantize_dense=True,
        )
        deduplicate_fsdp_unshard_chains_pass(
            traced_block.traced.gm,
            traced_block.traced.example_inputs,
        )
        backward_only_index = len(traced_block.traced.example_inputs) - 1
        fw_module, bw_module, meta = partition_joint_graph(
            traced_block.traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(backward_only_index,),
        )

        joint_input_names = _placeholder_names(traced_block.traced.gm)
        unshard_split = split_fsdp_unshard_collectives(
            traced_block.traced.gm,
            num_params=traced_block.num_flat_param_values,
            input_names=joint_input_names,
            flat_input_indices=tuple(range(len(joint_input_names))),
        )
        split_traced = replace(
            traced_block.traced,
            gm=unshard_split.compute_module,
        )
        split_backward_only_index = unshard_split.compute_flat_input_indices.index(
            backward_only_index
        )
        split_fw_module, split_bw_module, split_meta = partition_joint_graph(
            split_traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(split_backward_only_index,),
            flat_input_indices=unshard_split.compute_flat_input_indices,
        )
        bw_split = split_backward_fsdp_collectives(
            split_bw_module,
            num_param_grads=traced_block.num_param_grad_values,
        )

        self.assertIsNotNone(unshard_split.unshard_module)
        self.assertIsNotNone(bw_split.reduce_grad_module)
        if unshard_split.unshard_module is None or bw_split.reduce_grad_module is None:
            self.fail("Expected real DSV3 FSDP trace to contain split collectives")
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(split_fw_module),
        )
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(split_bw_module),
        )
        self.assertGreater(_mxfp8_weight_prepare_count(unshard_split.unshard_module), 0)
        self.assertEqual(_mxfp8_weight_prepare_count(split_fw_module), 0)
        self.assertEqual(_mxfp8_weight_prepare_count(split_bw_module), 0)
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _call_targets(bw_split.bw_no_fsdp_module),
        )

        fw_args = [
            traced_block.flat_inputs[index] for index in meta.fwd_flat_input_indices
        ]
        fw_outputs = _boxed_run(fw_module, list(fw_args))
        unshard_args = [
            traced_block.flat_inputs[index]
            for index in unshard_split.unshard_flat_param_indices
        ]
        unsharded_params = _boxed_run(unshard_split.unshard_module, unshard_args)
        materialized_flat_inputs = list(traced_block.flat_inputs)
        for flat_index, value in zip(
            unshard_split.unshard_flat_param_indices,
            unsharded_params,
            strict=True,
        ):
            materialized_flat_inputs[flat_index] = value
        split_fw_args = [
            materialized_flat_inputs[index]
            for index in split_meta.fwd_flat_input_indices
        ]
        split_fw_outputs = _boxed_run(
            split_fw_module,
            list(split_fw_args),
        )
        _assert_tensor_sequence_equal(self, split_fw_outputs[:1], fw_outputs[:1])

        bw_args = _backward_args_from_partition(
            meta,
            fw_outputs,
            (traced_block.output_grad,),
        )
        bw_outputs = _boxed_run(bw_module, list(bw_args))
        split_bw_args = _backward_args_from_partition(
            split_meta,
            split_fw_outputs,
            (traced_block.output_grad,),
        )
        bw_no_fsdp_outputs = _boxed_run(
            bw_split.bw_no_fsdp_module,
            list(split_bw_args),
        )
        grad_values_by_name = dict(
            zip(
                bw_split.bw_no_fsdp_output_names[: traced_block.num_param_grad_values],
                bw_no_fsdp_outputs[: traced_block.num_param_grad_values],
                strict=True,
            )
        )
        reduce_grad_args = [
            grad_values_by_name[name] for name in bw_split.reduce_grad_input_names
        ]
        reduced_grads = _boxed_run(bw_split.reduce_grad_module, reduce_grad_args)
        split_bw_outputs = (
            *reduced_grads,
            *bw_no_fsdp_outputs[traced_block.num_param_grad_values :],
        )
        _assert_tensor_sequence_equal(self, split_bw_outputs, bw_outputs)


if __name__ == "__main__":
    unittest.main()
