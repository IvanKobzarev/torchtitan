# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Any

import dist_moe._blockscaled  # noqa: F401
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import TestCase

from torchtitan.experiments.graph_trainer.performance_passes import (
    annotate_rmsnorm_for_regional_inductor_pass,
)
from torchtitan.experiments.graph_trainer.wgrad_accumulation import (
    fuse_deferred_wgrad_accumulation_pass,
)


class TestAnnotateRMSNormForRegionalInductorPass(TestCase):
    """Unit tests for annotate_rmsnorm_for_regional_inductor_pass."""

    def _build_rmsnorm_gm(self, node_specs):
        """Build a GraphModule with fused RMSNorm ops and other ops.

        Args:
            node_specs: List of op targets. For ``_fused_rms_norm`` and
                ``_fused_rms_norm_backward`` nodes, ``getitem`` users are
                automatically appended (mirroring the real traced graph
                structure).
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        last = x

        _FUSED_TARGETS = {
            torch.ops.aten._fused_rms_norm.default,
            torch.ops.aten._fused_rms_norm_backward.default,
        }

        for target in node_specs:
            if target in _FUSED_TARGETS:
                # Mimic the real graph: fused op returns a tuple,
                # followed by getitem nodes extracting elements.
                if target == torch.ops.aten._fused_rms_norm.default:
                    fused = graph.call_function(target, args=(last, [256], w, 1e-5))
                else:
                    fused = graph.call_function(
                        target, args=(last, w, last, [256], 1e-5)
                    )
                gi0 = graph.call_function(operator.getitem, args=(fused, 0))
                gi1 = graph.call_function(operator.getitem, args=(fused, 1))
                last = gi0
            else:
                last = graph.call_function(target, args=(last,))

        graph.output(last)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def _count_tagged_nodes(self, gm):
        """Count nodes that have compile_with_inductor in their custom metadata."""
        count = 0
        for node in gm.graph.nodes:
            custom = node.meta.get("custom", {})
            if "compile_with_inductor" in custom:
                count += 1
        return count

    def test_tags_fused_rmsnorm_and_getitems(self):
        """_fused_rms_norm nodes and their getitem users are tagged."""
        gm = self._build_rmsnorm_gm(
            [
                torch.ops.aten._fused_rms_norm.default,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.add.Tensor,
            ]
        )

        annotate_rmsnorm_for_regional_inductor_pass(gm)

        # 1 fused node + 2 getitem users = 3 tagged nodes
        self.assertEqual(self._count_tagged_nodes(gm), 3)

    def test_does_not_tag_non_rmsnorm_nodes(self):
        """Nodes that are not _fused_rms_norm targets are not tagged."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        n1 = graph.call_function(torch.ops.aten.mul.Tensor, args=(x, x))
        n2 = graph.call_function(torch.ops.aten.add.Tensor, args=(n1, x))
        graph.output(n2)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        annotate_rmsnorm_for_regional_inductor_pass(gm)

        self.assertEqual(self._count_tagged_nodes(gm), 0)

    def test_fwd_and_bwd_both_tagged(self):
        """Forward and backward fused norms and their getitems are all tagged."""
        gm = self._build_rmsnorm_gm(
            [
                torch.ops.aten._fused_rms_norm.default,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten._fused_rms_norm_backward.default,
            ]
        )

        annotate_rmsnorm_for_regional_inductor_pass(gm)

        # 2 fused nodes + 2*2 getitem users = 6 tagged nodes
        self.assertEqual(self._count_tagged_nodes(gm), 6)

    def test_custom_compile_config_propagated(self):
        """A custom compile config is wrapped under inductor_configs."""
        gm = self._build_rmsnorm_gm([torch.ops.aten._fused_rms_norm.default])

        config = {"max_autotune": True, "coordinate_descent_tuning": True}
        annotate_rmsnorm_for_regional_inductor_pass(gm, rmsnorm_compile_config=config)

        for node in gm.graph.nodes:
            annotation = node.meta.get("custom", {}).get("compile_with_inductor")
            if annotation is not None:
                self.assertEqual(annotation["inductor_configs"], config)


class TestFuseDeferredWgradAccumulationPass(TestCase):
    def _fake_cuda_tensor(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        stride: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        with FakeTensorMode():
            if stride is not None:
                return torch.empty_strided(
                    shape,
                    stride,
                    device="cuda",
                    dtype=dtype,
                )
            return torch.empty(shape, device="cuda", dtype=dtype)

    def _placeholder(
        self,
        graph: torch.fx.Graph,
        name: str,
        value: torch.Tensor,
    ) -> torch.fx.Node:
        node = graph.placeholder(name)
        node.meta["val"] = value
        return node

    def _build_graph(
        self,
        *,
        marker: str | None = "deferred_fsdp_final_gradient",
        final_user: bool = True,
        view_chain: bool = False,
        producer_fanout: bool = False,
        boundary_fanout: bool = False,
        accumulator_fanout: bool = False,
        accumulator_dtype: torch.dtype = torch.bfloat16,
        accumulator_stride: tuple[int, ...] | None = None,
        out_dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
        scale_result: bool = False,
        alpha: float = 1.0,
        use_fast_accum: bool = True,
    ) -> tuple[torch.fx.GraphModule, dict[str, torch.fx.Node]]:
        graph = torch.fx.Graph()
        producer_shape = (2, 3)
        boundary_shape = (6,) if view_chain else producer_shape
        accumulator = self._placeholder(
            graph,
            "accumulator",
            self._fake_cuda_tensor(
                boundary_shape,
                dtype=accumulator_dtype,
                stride=accumulator_stride,
            ),
        )
        lhs = self._placeholder(
            graph,
            "lhs",
            self._fake_cuda_tensor((2, 4), dtype=torch.float8_e4m3fn),
        )
        rhs = self._placeholder(
            graph,
            "rhs",
            self._fake_cuda_tensor((4, 3), dtype=torch.float8_e4m3fn),
        )
        lhs_scale = self._placeholder(
            graph,
            "lhs_scale",
            self._fake_cuda_tensor((1,), dtype=torch.float8_e8m0fnu),
        )
        rhs_scale = self._placeholder(
            graph,
            "rhs_scale",
            self._fake_cuda_tensor((1,), dtype=torch.float8_e8m0fnu),
        )
        bias_node = (
            self._placeholder(
                graph,
                "bias",
                self._fake_cuda_tensor((3,), dtype=torch.bfloat16),
            )
            if bias
            else None
        )
        scale_result_node = (
            self._placeholder(
                graph,
                "scale_result",
                self._fake_cuda_tensor((1,), dtype=torch.float32),
            )
            if scale_result
            else None
        )
        producer = graph.call_function(
            torch.ops.aten._scaled_mm.default,
            args=(lhs, rhs, lhs_scale, rhs_scale),
            kwargs={
                "bias": bias_node,
                "scale_result": scale_result_node,
                "out_dtype": out_dtype,
                "use_fast_accum": use_fast_accum,
            },
        )
        producer.meta.update(
            {
                "autograd_backward": True,
                "custom": {"module_fqn": "layers.0.feed_forward"},
                "val": self._fake_cuda_tensor(producer_shape, dtype=out_dtype),
            }
        )
        boundary = producer
        view = None
        alias = None
        if view_chain:
            view = graph.call_function(
                torch.ops.aten.view.default,
                args=(producer, list(boundary_shape)),
            )
            view.meta["val"] = self._fake_cuda_tensor(
                boundary_shape,
                dtype=out_dtype,
            )
            alias = graph.call_function(torch.ops.aten.alias.default, args=(view,))
            alias.meta["val"] = view.meta["val"]
            boundary = alias

        extra_users = []
        if producer_fanout:
            extra_users.append(
                graph.call_function(torch.ops.aten.neg.default, args=(producer,))
            )
        if boundary_fanout:
            extra_users.append(
                graph.call_function(torch.ops.aten.neg.default, args=(boundary,))
            )
        if accumulator_fanout:
            extra_users.append(
                graph.call_function(torch.ops.aten.neg.default, args=(accumulator,))
            )

        sink = graph.call_function(
            torch.ops.aten.add_.Tensor,
            args=(accumulator, boundary),
            kwargs={} if alpha == 1 else {"alpha": alpha},
        )
        sink.meta["val"] = accumulator.meta["val"]
        if marker is not None:
            sink.meta[marker] = True
        loss = self._placeholder(
            graph,
            "loss",
            self._fake_cuda_tensor((), dtype=torch.float32),
        )
        result = (
            graph.call_function(torch.ops.aten.neg.default, args=(sink,))
            if final_user
            else loss
        )
        graph.output((result, *extra_users))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        gm.graph.lint()
        return gm, {
            "accumulator": accumulator,
            "producer": producer,
            "boundary": boundary,
            "sink": sink,
            "view": view,
            "alias": alias,
            "result": result,
        }

    def _nodes_by_target(
        self,
        gm: torch.fx.GraphModule,
        target: Any,
    ) -> list[torch.fx.Node]:
        return [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == target
        ]

    def _dist_moe_targets(self, kind: str) -> tuple[Any, Any]:
        backward = getattr(torch.ops.dist_moe, f"{kind}_backward").default
        accumulate = getattr(
            torch.ops.dist_moe,
            f"{kind}_backward_accumulate",
        ).default
        return backward, accumulate

    def _build_dist_moe_graph(
        self,
        kind: str,
        *,
        missing_marker_index: int | None = None,
        fanout_index: int | None = None,
    ) -> tuple[torch.fx.GraphModule, dict[str, Any]]:
        graph = torch.fx.Graph()
        accumulator_shapes = ((24,), (40,))
        accumulators = tuple(
            self._placeholder(
                graph,
                f"accumulator_{index}",
                self._fake_cuda_tensor(shape, dtype=torch.bfloat16),
            )
            for index, shape in enumerate(accumulator_shapes)
        )
        grad_output = self._placeholder(
            graph,
            "grad_output",
            self._fake_cuda_tensor((8, 4), dtype=torch.bfloat16),
        )
        w13 = self._placeholder(
            graph,
            "w13",
            self._fake_cuda_tensor((2, 3, 4), dtype=torch.bfloat16),
        )
        w2 = self._placeholder(
            graph,
            "w2",
            self._fake_cuda_tensor((2, 4, 5), dtype=torch.bfloat16),
        )
        topk_scores = self._placeholder(
            graph,
            "topk_scores",
            self._fake_cuda_tensor((8, 2), dtype=torch.float32),
        )
        state = self._placeholder(
            graph,
            "state",
            self._fake_cuda_tensor((8,), dtype=torch.int64),
        )
        backward_target, accumulate_target = self._dist_moe_targets(kind)
        backward_args: tuple[Any, ...] = (
            grad_output,
            w13,
            w2,
            topk_scores,
            [state],
            7,
        )
        if kind == "block_scaled":
            topk_ids = self._placeholder(
                graph,
                "topk_ids",
                self._fake_cuda_tensor((8, 2), dtype=torch.int64),
            )
            backward_args = (
                grad_output,
                w13,
                w2,
                topk_ids,
                topk_scores,
                [state],
                7,
            )
        backward = graph.call_function(backward_target, args=backward_args)
        output_values = (
            self._fake_cuda_tensor((8, 4), dtype=torch.bfloat16),
            self._fake_cuda_tensor((8, 2), dtype=torch.float32),
            self._fake_cuda_tensor((2, 3, 4), dtype=torch.bfloat16),
            self._fake_cuda_tensor((2, 4, 5), dtype=torch.bfloat16),
        )
        backward.meta["val"] = output_values
        getitems = tuple(
            graph.call_function(operator.getitem, args=(backward, index))
            for index in range(4)
        )
        for getitem, value in zip(getitems, output_values, strict=True):
            getitem.meta["val"] = value

        sinks = []
        consumers = []
        extra_users = []
        for pair_index, output_index in enumerate((2, 3)):
            getitem = getitems[output_index]
            boundary = graph.call_function(
                torch.ops.aten.view.default,
                args=(getitem, list(accumulator_shapes[pair_index])),
            )
            boundary.meta["val"] = accumulators[pair_index].meta["val"]
            if fanout_index == output_index:
                extra_users.append(
                    graph.call_function(torch.ops.aten.neg.default, args=(getitem,))
                )
            sink = graph.call_function(
                torch.ops.aten.add_.Tensor,
                args=(accumulators[pair_index], boundary),
            )
            sink.meta["val"] = accumulators[pair_index].meta["val"]
            if missing_marker_index != output_index:
                sink.meta["deferred_fsdp_final_gradient"] = True
            sinks.append(sink)
            consumers.append(
                graph.call_function(torch.ops.aten.neg.default, args=(sink,))
            )
        graph.output((*getitems[:2], *consumers, *extra_users))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        gm.graph.lint()
        return gm, {
            "accumulate_target": accumulate_target,
            "accumulators": accumulators,
            "backward": backward,
            "consumers": tuple(consumers),
            "getitems": getitems,
            "original_args": backward_args,
            "output_values": output_values,
            "sinks": tuple(sinks),
        }

    def _assert_unfused(self, gm: torch.fx.GraphModule) -> None:
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten._scaled_mm.default)),
            1,
        )
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten._scaled_addmm_.default)),
            0,
        )
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten.add_.Tensor)),
            1,
        )

    def test_final_fuses_view_chain_and_preserves_users_and_metadata(self) -> None:
        gm, original = self._build_graph(view_chain=True, use_fast_accum=True)

        result = fuse_deferred_wgrad_accumulation_pass(gm)

        self.assertIs(result, gm)
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten._scaled_mm.default)),
            0,
        )
        fused = self._nodes_by_target(
            gm,
            torch.ops.aten._scaled_addmm_.default,
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten.add_.Tensor)),
            0,
        )
        fused_node = fused[0]
        accumulator_view = fused_node.args[0]
        self.assertIsInstance(accumulator_view, torch.fx.Node)
        assert isinstance(accumulator_view, torch.fx.Node)
        self.assertEqual(accumulator_view.target, torch.ops.aten.view.default)
        self.assertIs(accumulator_view.args[0], original["accumulator"])
        self.assertEqual(accumulator_view.args[1], [2, 3])
        self.assertTrue(fused_node.kwargs["use_fast_accum"])
        self.assertTrue(fused_node.meta["deferred_fsdp_fused_wgrad_accumulation"])
        self.assertTrue(fused_node.meta["autograd_backward"])
        self.assertEqual(
            fused_node.meta["custom"],
            {"module_fqn": "layers.0.feed_forward"},
        )
        self.assertIs(original["result"].args[0], original["boundary"])
        self.assertIs(original["view"].args[0], fused_node)
        self.assertIn(original["view"], gm.graph.nodes)
        self.assertIn(original["alias"], gm.graph.nodes)
        gm.graph.lint()

    def test_middle_fusion_keeps_unused_mutation_alive(self) -> None:
        gm, _ = self._build_graph(
            marker="deferred_fsdp_gradient_accumulation",
            final_user=False,
            view_chain=True,
            use_fast_accum=False,
        )

        fuse_deferred_wgrad_accumulation_pass(gm)
        gm.graph.eliminate_dead_code()
        gm.recompile()

        fused = self._nodes_by_target(
            gm,
            torch.ops.aten._scaled_addmm_.default,
        )
        self.assertEqual(len(fused), 1)
        self.assertFalse(fused[0].kwargs["use_fast_accum"])
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten.add_.Tensor)),
            0,
        )
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten.alias.default)),
            0,
        )
        gm.graph.lint()

    def test_direct_middle_fusion_keeps_unused_mutation_alive(self) -> None:
        gm, original = self._build_graph(
            marker="deferred_fsdp_gradient_accumulation",
            final_user=False,
        )

        fuse_deferred_wgrad_accumulation_pass(gm)
        gm.graph.eliminate_dead_code()
        gm.recompile()

        self.assertEqual(
            len(
                self._nodes_by_target(
                    gm,
                    torch.ops.aten._scaled_addmm_.default,
                )
            ),
            1,
        )
        self.assertNotIn(original["producer"], gm.graph.nodes)
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten.add_.Tensor)),
            0,
        )
        gm.graph.lint()

    def test_direct_producer_fusion_preserves_downstream_dependency(self) -> None:
        gm, original = self._build_graph()

        fuse_deferred_wgrad_accumulation_pass(gm)

        fused = self._nodes_by_target(
            gm,
            torch.ops.aten._scaled_addmm_.default,
        )
        self.assertEqual(len(fused), 1)
        self.assertNotIn(original["producer"], gm.graph.nodes)
        self.assertIs(original["result"].args[0], fused[0])
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten.add_.Tensor)),
            0,
        )
        gm.graph.lint()

    def test_unmarked_accumulation_is_not_fused(self) -> None:
        gm, _ = self._build_graph(marker=None)

        fuse_deferred_wgrad_accumulation_pass(gm)

        self._assert_unfused(gm)

    def test_alias_or_producer_fanout_is_not_fused(self) -> None:
        for fanout in ("producer", "boundary", "accumulator"):
            with self.subTest(fanout=fanout):
                gm, _ = self._build_graph(
                    view_chain=True,
                    producer_fanout=fanout == "producer",
                    boundary_fanout=fanout == "boundary",
                    accumulator_fanout=fanout == "accumulator",
                )

                fuse_deferred_wgrad_accumulation_pass(gm)

                self._assert_unfused(gm)

    def test_incompatible_tensor_metadata_is_not_fused(self) -> None:
        cases = (
            {"accumulator_dtype": torch.float32},
            {"accumulator_stride": (1, 2)},
            {"out_dtype": torch.float32},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                gm, _ = self._build_graph(**overrides)

                fuse_deferred_wgrad_accumulation_pass(gm)

                self._assert_unfused(gm)

    def test_boundary_layout_mismatch_is_not_fused(self) -> None:
        for shape, stride in (((3, 2), None), ((2, 3), (1, 2))):
            with self.subTest(shape=shape, stride=stride):
                gm, original = self._build_graph()
                original["boundary"].meta["val"] = self._fake_cuda_tensor(
                    shape,
                    dtype=torch.bfloat16,
                    stride=stride,
                )

                fuse_deferred_wgrad_accumulation_pass(gm)

                self._assert_unfused(gm)

    def test_scaled_mm_options_and_non_unit_alpha_are_not_fused(self) -> None:
        cases = (
            {"bias": True},
            {"scale_result": True},
            {"alpha": 2.0},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                gm, _ = self._build_graph(**overrides)

                fuse_deferred_wgrad_accumulation_pass(gm)

                self._assert_unfused(gm)

    def test_dist_moe_fuses_paired_wgrads_and_preserves_other_outputs(self) -> None:
        for kind in ("bf16", "block_scaled"):
            with self.subTest(kind=kind):
                gm, original = self._build_dist_moe_graph(kind)

                fuse_deferred_wgrad_accumulation_pass(gm)

                backward = original["backward"]
                accumulators = original["accumulators"]
                self.assertEqual(backward.target, original["accumulate_target"])
                self.assertEqual(backward.args[:2], accumulators)
                self.assertEqual(backward.args[2:], original["original_args"])
                self.assertEqual(len(backward.meta["val"]), 2)
                for actual, expected in zip(
                    backward.meta["val"],
                    original["output_values"][:2],
                    strict=True,
                ):
                    self.assertIs(actual, expected)
                self.assertEqual(backward.meta["original_aten"], backward.target)
                self.assertTrue(backward.meta["deferred_fsdp_fused_wgrad_accumulation"])
                self.assertEqual(
                    len(self._nodes_by_target(gm, torch.ops.aten.add_.Tensor)),
                    0,
                )
                remaining_getitems = self._nodes_by_target(gm, operator.getitem)
                self.assertEqual(remaining_getitems, list(original["getitems"][:2]))
                for index, getitem in enumerate(remaining_getitems):
                    self.assertIs(getitem.args[0], backward)
                    self.assertEqual(getitem.args[1], index)
                for consumer, accumulator in zip(
                    original["consumers"],
                    accumulators,
                    strict=True,
                ):
                    self.assertIs(consumer.args[0], accumulator)
                gm.graph.lint()

    def test_dist_moe_requires_both_safe_marked_wgrads(self) -> None:
        for kind in ("bf16", "block_scaled"):
            for unsafe, overrides in (
                ("partial", {"missing_marker_index": 3}),
                ("fanout", {"fanout_index": 2}),
            ):
                with self.subTest(kind=kind, unsafe=unsafe):
                    gm, original = self._build_dist_moe_graph(kind, **overrides)

                    fuse_deferred_wgrad_accumulation_pass(gm)

                    self.assertIs(
                        original["backward"].target,
                        self._dist_moe_targets(kind)[0],
                    )
                    self.assertEqual(
                        original["backward"].args,
                        original["original_args"],
                    )
                    self.assertEqual(
                        len(self._nodes_by_target(gm, torch.ops.aten.add_.Tensor)),
                        2,
                    )
                    self.assertEqual(
                        len(self._nodes_by_target(gm, operator.getitem)),
                        4,
                    )
                    gm.graph.lint()

    def test_dist_moe_requires_complete_output_metadata(self) -> None:
        gm, original = self._build_dist_moe_graph("block_scaled")
        original["backward"].meta.pop("val")

        fuse_deferred_wgrad_accumulation_pass(gm)

        self.assertIs(
            original["backward"].target,
            self._dist_moe_targets("block_scaled")[0],
        )
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten.add_.Tensor)),
            2,
        )
        gm.graph.lint()

    def test_dist_moe_reshape_requires_contiguous_source(self) -> None:
        gm, original = self._build_dist_moe_graph("block_scaled")
        w13_boundary = original["sinks"][0].args[1]
        assert isinstance(w13_boundary, torch.fx.Node)
        w13_boundary.target = torch.ops.aten.reshape.default

        fuse_deferred_wgrad_accumulation_pass(gm)

        self.assertIs(
            original["backward"].target,
            original["accumulate_target"],
        )

        gm, original = self._build_dist_moe_graph("block_scaled")
        w13_boundary = original["sinks"][0].args[1]
        assert isinstance(w13_boundary, torch.fx.Node)
        w13_boundary.target = torch.ops.aten.reshape.default
        original["getitems"][2].meta["val"] = self._fake_cuda_tensor(
            (2, 3, 4),
            dtype=torch.bfloat16,
            stride=(12, 1, 3),
        )

        fuse_deferred_wgrad_accumulation_pass(gm)

        self.assertIs(
            original["backward"].target,
            self._dist_moe_targets("block_scaled")[0],
        )
        gm.graph.lint()


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
