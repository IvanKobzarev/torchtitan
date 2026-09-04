# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import operator
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch.fx._lazy_graph_module import _make_graph_module

from torchtitan.experiments.graph_trainer.cudagraph import (
    cudagraph_pass,
    insert_kernel_annotations_pass,
    is_cudagraph_compatible,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    enable_fsdp_symmetric_memory_for_graphs,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_reduce_grad_boundary,
    find_fsdp_unshard_outputs,
    is_all_gather_into_tensor,
    is_all_reduce,
    is_reduce_scatter_tensor,
)
from torchtitan.experiments.graph_trainer.gradient_accumulation import (
    _graph_state_leaf_offsets,
    _insert_graph_gradient_sink,
    _validate_device_mesh_leaf,
    _validate_gradient_leaf_mapping,
    GraphGradientState,
    validate_graph_gradient_output_mapping,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_fsdp_collectives import (
    split_fsdp_unshard_collectives,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    allow_fx_graph_extraction_of_side_effectful_ops,
    example_inputs_from_placeholders,
    graph_outputs,
    placeholder_names,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _bound_static_values,
    _flat_input_binding,
    _flat_tensor_ranges,
    _flatten_bound_state,
    _unwrap_subclasses,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    final_inductor_compile_passes,
)
from torchtitan.experiments.graph_trainer.wgrad_accumulation import (
    fuse_deferred_wgrad_accumulation_pass,
)


@dataclass(frozen=True, slots=True)
class _BoundarySpec:
    output_flat_index: int
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, slots=True)
class DeferredFSDPGraph:
    """A whole-optimizer-step minimal-FX graph with one FSDP gradient sync."""

    gm: fx.GraphModule
    traced_result: TracedResult
    example_inputs: tuple[Any, ...]
    num_microbatches: int
    num_static_inputs: int
    tensor_input_indices: tuple[int, ...]
    num_all_gathers: int
    num_reduce_scatters: int
    num_all_reduces: int

    @property
    def num_gradient_collectives(self) -> int:
        return self.num_reduce_scatters + self.num_all_reduces


def _gradient_and_loss_flat_indices(
    traced_result: TracedResult,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ranges = _flat_tensor_ranges(
        traced_result.num_flat_outputs,
        traced_result.output_subclass_layouts,
    )
    gradient_indices = tuple(
        index
        for output_index in traced_result.graph_state_output_indices
        for index in ranges[output_index]
    )
    gradient_index_set = set(gradient_indices)
    loss_indices = tuple(
        index
        for index in range(sum(len(indices) for indices in ranges))
        if index not in gradient_index_set
    )
    if len(loss_indices) != 1:
        raise ValueError(
            "Deferred FSDP gradient sync requires one flat loss output, got "
            f"{len(loss_indices)}"
        )
    return gradient_indices, loss_indices


def _boundary_signature(node: fx.Node, output_flat_index: int) -> _BoundarySpec:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or not (
        value.is_floating_point() or value.is_complex()
    ):
        raise ValueError(
            "Deferred FSDP gradient boundaries must be floating-point tensors; "
            f"output {output_flat_index} has {value!r}"
        )
    if value.layout != torch.strided:
        raise NotImplementedError(
            "Deferred FSDP gradient sync requires strided pre-reduction "
            f"gradients, got {value.layout}"
        )
    return _BoundarySpec(
        output_flat_index=output_flat_index,
        shape=value.shape,
        stride=value.stride(),
        dtype=value.dtype,
        device=value.device,
    )


def _gradient_boundaries(
    gm: fx.GraphModule,
    traced_result: TracedResult,
) -> tuple[list[fx.Node], tuple[_BoundarySpec, ...], int]:
    outputs = graph_outputs(gm.graph)
    gradient_indices, _ = _gradient_and_loss_flat_indices(traced_result)
    boundaries: list[fx.Node] = []
    specs: list[_BoundarySpec] = []
    seen: set[fx.Node] = set()
    collective_boundaries: set[fx.Node] = set()
    for output_index in gradient_indices:
        output = outputs[output_index]
        if not isinstance(output, fx.Node):
            continue
        reduce_grad_boundary = find_fsdp_reduce_grad_boundary(output)
        if reduce_grad_boundary is not None:
            boundary = reduce_grad_boundary.accumulation_input
            collective_boundaries.add(reduce_grad_boundary.collective_input)
        else:
            boundary = output
        value = boundary.meta.get("val")
        if not isinstance(value, torch.Tensor) or not (
            value.is_floating_point() or value.is_complex()
        ):
            continue
        if boundary in seen:
            continue
        seen.add(boundary)
        boundaries.append(boundary)
        specs.append(_boundary_signature(boundary, output_index))
    if not boundaries:
        raise ValueError("Deferred FSDP gradient sync found no gradient tensors")
    return boundaries, tuple(specs), len(collective_boundaries)


def _assert_matching_boundaries(
    expected: tuple[_BoundarySpec, ...],
    actual: tuple[_BoundarySpec, ...],
) -> None:
    expected_metadata = tuple(
        (
            spec.output_flat_index,
            spec.shape,
            spec.stride,
            spec.dtype,
            spec.device,
        )
        for spec in expected
    )
    actual_metadata = tuple(
        (
            spec.output_flat_index,
            spec.shape,
            spec.stride,
            spec.dtype,
            spec.device,
        )
        for spec in actual
    )
    if expected_metadata != actual_metadata:
        raise ValueError(
            "Deferred FSDP gradient boundaries changed after unshard "
            f"extraction: {expected_metadata!r} != {actual_metadata!r}"
        )


def _extract_graph(
    gm: fx.GraphModule,
    selected_outputs: list[fx.Node],
    *,
    name: str,
) -> fx.GraphModule:
    placeholders = gm.graph.find_nodes(op="placeholder")
    with allow_fx_graph_extraction_of_side_effectful_ops(
        {
            torch.ops._c10d_functional.wait_tensor,
            torch.ops._c10d_functional.wait_tensor.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            torch.ops._c10d_functional.all_reduce.default,
        }
    ):
        graph = _extract_graph_with_inputs_outputs(
            gm.graph,
            placeholders,
            selected_outputs,
            [None] * len(selected_outputs),
            name,
            ignore_must_be_in_fw_bw=True,
        )
    return _make_graph_module(gm, graph)


def _unsharded_parameter_outputs(
    gm: fx.GraphModule,
    *,
    num_flat_parameters: int,
) -> list[fx.Node]:
    placeholders = gm.graph.find_nodes(op="placeholder")
    outputs: list[fx.Node] = []
    found_unshard = False
    for placeholder in placeholders[:num_flat_parameters]:
        unsharded = find_fsdp_unshard_outputs(placeholder)
        if not unsharded:
            outputs.append(placeholder)
            continue
        if len(unsharded) != 1:
            raise ValueError(
                "Deferred FSDP gradient sync expects one canonical unshard "
                f"output for {placeholder.name}, got {len(unsharded)}"
            )
        found_unshard = True
        outputs.append(unsharded[0])
    if not found_unshard:
        raise ValueError(
            "Deferred FSDP gradient sync requires a traced FSDP all-gather"
        )
    return outputs


def _loss_outputs(
    gm: fx.GraphModule,
    traced_result: TracedResult,
) -> list[fx.Node]:
    outputs = graph_outputs(gm.graph)
    _, loss_indices = _gradient_and_loss_flat_indices(traced_result)
    losses = [outputs[index] for index in loss_indices]
    if any(not isinstance(loss, fx.Node) for loss in losses):
        raise ValueError("Deferred FSDP gradient sync requires a tensor loss")
    return list(losses)


def _build_first_microbatch_graph(
    gm: fx.GraphModule,
    traced_result: TracedResult,
    *,
    num_flat_parameters: int,
) -> tuple[fx.GraphModule, tuple[_BoundarySpec, ...], int]:
    work = copy.deepcopy(gm)
    boundaries, specs, num_collective_boundaries = _gradient_boundaries(
        work, traced_result
    )
    unsharded = _unsharded_parameter_outputs(
        work,
        num_flat_parameters=num_flat_parameters,
    )
    first = _extract_graph(
        work,
        [*_loss_outputs(work, traced_result), *boundaries, *unsharded],
        name="deferred_fsdp_first",
    )
    return first, specs, num_collective_boundaries


def _append_accumulator_placeholders(
    gm: fx.GraphModule,
    boundaries: list[fx.Node],
) -> list[fx.Node]:
    first_non_placeholder = next(
        node for node in gm.graph.nodes if node.op != "placeholder"
    )
    accumulators = []
    with gm.graph.inserting_before(first_non_placeholder):
        for index, boundary in enumerate(boundaries):
            accumulator = gm.graph.placeholder(f"deferred_grad_{index}")
            accumulator.meta = copy.copy(boundary.meta)
            accumulators.append(accumulator)
    return accumulators


def _build_middle_microbatch_graph(
    gm: fx.GraphModule,
    traced_result: TracedResult,
    expected_specs: tuple[_BoundarySpec, ...],
) -> tuple[fx.GraphModule, int]:
    work = copy.deepcopy(gm)
    boundaries, specs, num_collective_boundaries = _gradient_boundaries(
        work, traced_result
    )
    _assert_matching_boundaries(expected_specs, specs)
    middle = _extract_graph(
        work,
        [*_loss_outputs(work, traced_result), *boundaries],
        name="deferred_fsdp_middle",
    )
    outputs = graph_outputs(middle.graph)
    loss_count = len(_loss_outputs(middle, traced_result))
    middle_boundaries = list(outputs[loss_count:])
    if any(not isinstance(boundary, fx.Node) for boundary in middle_boundaries):
        raise ValueError("Deferred FSDP middle graph lost a gradient boundary")
    typed_boundaries = [
        boundary for boundary in middle_boundaries if isinstance(boundary, fx.Node)
    ]
    accumulators = _append_accumulator_placeholders(middle, typed_boundaries)
    for boundary, accumulator in zip(typed_boundaries, accumulators, strict=True):
        with middle.graph.inserting_after(boundary):
            sink = middle.graph.call_function(
                torch.ops.aten.add_.Tensor,
                args=(accumulator, boundary),
            )
        sink.meta = copy.copy(boundary.meta)
        sink.meta["deferred_fsdp_gradient_accumulation"] = True
    output = middle.graph.find_nodes(op="output")[0]
    output.args = (tuple(outputs[:loss_count]),)
    middle.graph.lint()
    middle.recompile()
    return middle, num_collective_boundaries


def _compute_placeholder_by_original_index(
    gm: fx.GraphModule,
    compute_flat_input_indices: tuple[int, ...],
) -> dict[int, fx.Node]:
    placeholders = gm.graph.find_nodes(op="placeholder")
    if len(placeholders) < len(compute_flat_input_indices):
        raise ValueError(
            "Deferred FSDP compute input metadata has fewer placeholders than "
            f"expected: {len(placeholders)} < {len(compute_flat_input_indices)}"
        )
    return dict(
        zip(
            compute_flat_input_indices,
            placeholders[: len(compute_flat_input_indices)],
            strict=True,
        )
    )


def _insert_final_gradient_sinks(
    gm: fx.GraphModule,
    traced_result: TracedResult,
    *,
    compute_flat_input_indices: tuple[int, ...],
) -> None:
    outputs = graph_outputs(gm.graph)
    output_ranges = _flat_tensor_ranges(
        traced_result.num_flat_outputs,
        traced_result.output_subclass_layouts,
    )
    placeholder_by_input = _compute_placeholder_by_original_index(
        gm, compute_flat_input_indices
    )
    for state_index, (fqn, buffer_indices, gradient_output_index) in enumerate(
        zip(
            traced_result.graph_state_fqns,
            traced_result.graph_state_input_indices,
            traced_result.graph_state_output_indices,
            strict=True,
        )
    ):
        buffer_logical_index = len(traced_result.state_fqns) + state_index
        leaf_offsets, device_mesh_offset = _graph_state_leaf_offsets(
            fqn,
            traced_result.input_subclass_layouts.get(buffer_logical_index),
            traced_result.output_subclass_layouts.get(gradient_output_index),
        )
        gradient_indices = output_ranges[gradient_output_index]
        if device_mesh_offset is not None:
            _validate_device_mesh_leaf(
                fqn,
                placeholder_by_input[buffer_indices[device_mesh_offset]],
                outputs[gradient_indices[device_mesh_offset]],
            )
        for leaf_offset in leaf_offsets:
            buffer = placeholder_by_input[buffer_indices[leaf_offset]]
            gradient = outputs[gradient_indices[leaf_offset]]
            if not isinstance(gradient, fx.Node):
                raise ValueError(f"Gradient output for {fqn!r} is not a tensor")
            _validate_gradient_leaf_mapping(fqn, leaf_offset, gradient)
            with gm.graph.inserting_after(gradient):
                _insert_graph_gradient_sink(
                    gm,
                    fqn=fqn,
                    buffer=buffer,
                    gradient=gradient,
                )


def _build_final_microbatch_graph(
    gm: fx.GraphModule,
    traced_result: TracedResult,
    expected_specs: tuple[_BoundarySpec, ...],
    *,
    compute_flat_input_indices: tuple[int, ...],
) -> tuple[fx.GraphModule, int]:
    final = copy.deepcopy(gm)
    boundaries, specs, num_collective_boundaries = _gradient_boundaries(
        final, traced_result
    )
    _assert_matching_boundaries(expected_specs, specs)
    accumulators = _append_accumulator_placeholders(final, boundaries)
    for boundary, accumulator in zip(boundaries, accumulators, strict=True):
        users = tuple(boundary.users)
        with final.graph.inserting_after(boundary):
            combined = final.graph.call_function(
                torch.ops.aten.add_.Tensor,
                args=(accumulator, boundary),
            )
        _copy_placeholder_meta(accumulator, combined)
        combined.meta["deferred_fsdp_final_gradient"] = True
        for user in users:
            user.replace_input_with(boundary, combined)

    _insert_final_gradient_sinks(
        final,
        traced_result,
        compute_flat_input_indices=compute_flat_input_indices,
    )
    output = final.graph.find_nodes(op="output")[0]
    output.args = (tuple(_loss_outputs(final, traced_result)),)
    final.graph.lint()
    final.recompile()
    return final, num_collective_boundaries


def _compile_child(
    gm: fx.GraphModule,
    *,
    compile_config: Any,
) -> fx.GraphModule:
    if (
        compile_config.enable
        and compile_config.enable_passes
        and compile_config.inductor_compilation != "none"
    ):
        gm = apply_graph_passes(
            gm,
            example_inputs_from_placeholders(gm),
            final_inductor_compile_passes(compile_config),
            compile_config=compile_config,
        )
    return gm


def _copy_placeholder_meta(source: fx.Node, target: fx.Node) -> None:
    target.meta = copy.copy(source.meta)
    for key in ("custom", "unbacked_bindings"):
        if isinstance(value := target.meta.get(key), dict):
            target.meta[key] = copy.copy(value)


def _call_module_outputs(
    graph: fx.Graph,
    target: str,
    args: list[fx.Node],
    output_values: tuple[Any, ...],
) -> list[fx.Node]:
    call = graph.call_module(target, args=tuple(args))
    call.meta["val"] = tuple(
        value.meta.get("val") if isinstance(value, fx.Node) else value
        for value in output_values
    )
    outputs = []
    insertion_point = call
    for index, value in enumerate(output_values):
        with graph.inserting_after(insertion_point):
            item = graph.call_function(operator.getitem, args=(call, index))
        if isinstance(value, fx.Node):
            _copy_placeholder_meta(value, item)
        outputs.append(item)
        insertion_point = item
    return outputs


def _outer_placeholders(
    graph: fx.Graph,
    traced_result: TracedResult,
    *,
    num_microbatches: int,
) -> tuple[list[list[fx.Node]], tuple[Any, ...]]:
    original_placeholders = traced_result.gm.graph.find_nodes(op="placeholder")
    static_count = traced_result.num_static_inputs
    static_nodes = []
    example_inputs: list[Any] = []
    for index, source in enumerate(original_placeholders[:static_count]):
        node = graph.placeholder(f"static_{index}_{source.name}")
        _copy_placeholder_meta(source, node)
        static_nodes.append(node)
        example_inputs.append(traced_result.example_inputs[index])

    calls = []
    for microbatch in range(num_microbatches):
        call_nodes = list(static_nodes)
        for index, source in enumerate(original_placeholders[static_count:]):
            node = graph.placeholder(f"mb{microbatch}_{index}_{source.name}")
            _copy_placeholder_meta(source, node)
            call_nodes.append(node)
            example_inputs.append(traced_result.example_inputs[static_count + index])
        calls.append(call_nodes)
    return calls, tuple(example_inputs)


def _reuse_call_args(
    original_args: list[fx.Node],
    unsharded_params: list[fx.Node],
    *,
    num_flat_parameters: int,
    compute_flat_input_indices: tuple[int, ...],
) -> list[fx.Node]:
    result = []
    for original_index in compute_flat_input_indices:
        if original_index < num_flat_parameters:
            result.append(unsharded_params[original_index])
        else:
            result.append(original_args[original_index])
    return result


def _build_outer_graph(
    traced_result: TracedResult,
    *,
    first: fx.GraphModule,
    middle: fx.GraphModule,
    final: fx.GraphModule,
    num_microbatches: int,
    num_boundaries: int,
    num_flat_parameters: int,
    compute_flat_input_indices: tuple[int, ...],
) -> tuple[fx.GraphModule, tuple[Any, ...]]:
    graph = fx.Graph()
    call_args, example_inputs = _outer_placeholders(
        graph,
        traced_result,
        num_microbatches=num_microbatches,
    )
    first_values = graph_outputs(first.graph)
    first_outputs = _call_module_outputs(
        graph,
        "first",
        call_args[0],
        first_values,
    )
    loss = first_outputs[0]
    accumulators = first_outputs[1 : 1 + num_boundaries]
    unsharded_params = first_outputs[1 + num_boundaries :]
    if len(unsharded_params) != num_flat_parameters:
        raise ValueError(
            "Deferred FSDP first graph returned the wrong number of unsharded "
            f"parameters: {len(unsharded_params)} != {num_flat_parameters}"
        )

    for microbatch in range(1, num_microbatches - 1):
        middle_args = _reuse_call_args(
            call_args[microbatch],
            unsharded_params,
            num_flat_parameters=num_flat_parameters,
            compute_flat_input_indices=compute_flat_input_indices,
        )
        middle_outputs = _call_module_outputs(
            graph,
            "middle",
            [*middle_args, *accumulators],
            graph_outputs(middle.graph),
        )
        with graph.inserting_after(middle_outputs[0]):
            next_loss = graph.call_function(
                torch.ops.aten.add.Tensor,
                args=(loss, middle_outputs[0]),
            )
        _copy_placeholder_meta(loss, next_loss)
        loss = next_loss

    final_args = _reuse_call_args(
        call_args[-1],
        unsharded_params,
        num_flat_parameters=num_flat_parameters,
        compute_flat_input_indices=compute_flat_input_indices,
    )
    final_outputs = _call_module_outputs(
        graph,
        "final",
        [*final_args, *accumulators],
        graph_outputs(final.graph),
    )
    with graph.inserting_after(final_outputs[0]):
        loss = graph.call_function(
            torch.ops.aten.add.Tensor,
            args=(loss, final_outputs[0]),
        )
    _copy_placeholder_meta(final_outputs[0], loss)
    graph.output((loss,))

    root = nn.Module()
    root.add_module("first", first)
    root.add_module("middle", middle)
    root.add_module("final", final)
    outer = fx.GraphModule(root, graph, "DeferredFSDPAccumulation")
    outer.graph.lint()
    outer.recompile()
    return outer, example_inputs


def build_deferred_fsdp_graph(
    traced_result: TracedResult,
    *,
    num_flat_parameters: int,
    num_microbatches: int,
    compile_config: Any,
    enable_cudagraph: bool,
    fsdp_symm_mem_policy: str | None = None,
) -> DeferredFSDPGraph:
    """Build one minimal-FX program for an SPMD accumulation step."""
    if num_microbatches < 2:
        raise ValueError("Deferred FSDP gradient sync requires at least two batches")
    validate_graph_gradient_output_mapping(traced_result.gm, traced_result)
    input_names = placeholder_names(traced_result.gm)
    split = split_fsdp_unshard_collectives(
        traced_result.gm,
        num_params=num_flat_parameters,
        input_names=input_names,
        flat_input_indices=tuple(range(len(input_names))),
    )
    if split.unshard_module is None:
        raise ValueError("Deferred FSDP gradient sync found no FSDP all-gather")

    first, boundary_specs, num_collective_boundaries = _build_first_microbatch_graph(
        traced_result.gm,
        traced_result,
        num_flat_parameters=num_flat_parameters,
    )
    if num_collective_boundaries == 0:
        raise ValueError("Deferred FSDP gradient sync found no gradient reduction")
    middle, middle_collective_boundaries = _build_middle_microbatch_graph(
        split.compute_module,
        traced_result,
        boundary_specs,
    )
    final, final_collective_boundaries = _build_final_microbatch_graph(
        split.compute_module,
        traced_result,
        boundary_specs,
        compute_flat_input_indices=split.compute_flat_input_indices,
    )
    if not (
        num_collective_boundaries
        == middle_collective_boundaries
        == final_collective_boundaries
    ):
        raise ValueError(
            "Deferred FSDP microbatch graphs found different gradient "
            "collective-boundary counts: "
            f"{num_collective_boundaries}, {middle_collective_boundaries}, "
            f"{final_collective_boundaries}"
        )

    if (
        compile_config.enable_passes
        and compile_config.numerics_changing_optim
        and "fuse_deferred_wgrad_accumulation_pass" not in compile_config.disable_passes
    ):
        fuse_deferred_wgrad_accumulation_pass(middle)
        fuse_deferred_wgrad_accumulation_pass(final)

    first_all_gathers = sum(
        is_all_gather_into_tensor(node) for node in first.graph.nodes
    )
    middle_all_gathers = sum(
        is_all_gather_into_tensor(node) for node in middle.graph.nodes
    )
    final_all_gathers = sum(
        is_all_gather_into_tensor(node) for node in final.graph.nodes
    )
    if middle_all_gathers != final_all_gathers:
        raise ValueError(
            "Deferred FSDP reused graphs have different non-parameter "
            "all-gather counts: "
            f"{middle_all_gathers} != {final_all_gathers}"
        )
    num_all_gathers = first_all_gathers - middle_all_gathers
    if num_all_gathers <= 0:
        raise ValueError(
            "Deferred FSDP first graph did not retain parameter all-gather"
        )

    first_reductions = (
        sum(is_reduce_scatter_tensor(node) for node in first.graph.nodes),
        sum(is_all_reduce(node) for node in first.graph.nodes),
    )
    middle_reductions = (
        sum(is_reduce_scatter_tensor(node) for node in middle.graph.nodes),
        sum(is_all_reduce(node) for node in middle.graph.nodes),
    )
    final_reductions = (
        sum(is_reduce_scatter_tensor(node) for node in final.graph.nodes),
        sum(is_all_reduce(node) for node in final.graph.nodes),
    )
    if first_reductions != middle_reductions:
        raise ValueError(
            "Deferred FSDP first and reused non-final graphs have different "
            "non-gradient reduction counts: "
            f"{first_reductions} != {middle_reductions}"
        )
    num_reduce_scatters = final_reductions[0] - middle_reductions[0]
    num_all_reduces = final_reductions[1] - middle_reductions[1]
    if num_reduce_scatters < 0 or num_all_reduces < 0:
        raise ValueError(
            "Deferred FSDP final graph removed reduction collectives: "
            f"middle={middle_reductions}, final={final_reductions}"
        )
    num_gradient_collectives = num_reduce_scatters + num_all_reduces
    if num_gradient_collectives < num_collective_boundaries:
        raise ValueError(
            "Deferred FSDP final graph restored "
            f"{num_gradient_collectives} gradient collectives for "
            f"{num_collective_boundaries} deferred reduction boundaries"
        )

    if fsdp_symm_mem_policy is not None:
        enable_fsdp_symmetric_memory_for_graphs(
            (first, middle, final),
            selection_graph=traced_result.gm,
            policy=fsdp_symm_mem_policy,
            preallocate=compile_config.inductor_compilation != "full",
        )

    children = {
        name: _compile_child(
            child,
            compile_config=compile_config,
        )
        for name, child in (
            ("first", first),
            ("middle", middle),
            ("final", final),
        )
    }
    outer, example_inputs = _build_outer_graph(
        traced_result,
        first=children["first"],
        middle=children["middle"],
        final=children["final"],
        num_microbatches=num_microbatches,
        num_boundaries=len(boundary_specs),
        num_flat_parameters=num_flat_parameters,
        compute_flat_input_indices=split.compute_flat_input_indices,
    )
    cudagraph_compatible = is_cudagraph_compatible(outer) and all(
        is_cudagraph_compatible(child) for child in children.values()
    )
    outer.meta["cudagraph_compatible"] = cudagraph_compatible
    want_annotations = (
        enable_cudagraph
        and cudagraph_compatible
        and compile_config.inductor_compilation != "full"
        and "insert_kernel_annotations_pass" not in compile_config.disable_passes
    )
    if want_annotations:
        for child in children.values():
            insert_kernel_annotations_pass(child)
    tensor_input_indices = tuple(
        index
        for index, value in enumerate(example_inputs)
        if isinstance(value, torch.Tensor)
    )
    if enable_cudagraph:
        outer = cudagraph_pass(
            outer,
            example_inputs,
            static_input_indices=list(range(traced_result.num_static_inputs)),
            tensor_input_indices=list(tensor_input_indices),
            require=compile_config.require_cudagraph,
            annotate_kernels=False,
        )
    return DeferredFSDPGraph(
        gm=outer,
        traced_result=traced_result,
        example_inputs=example_inputs,
        num_microbatches=num_microbatches,
        num_static_inputs=traced_result.num_static_inputs,
        tensor_input_indices=tensor_input_indices,
        num_all_gathers=num_all_gathers,
        num_reduce_scatters=num_reduce_scatters,
        num_all_reduces=num_all_reduces,
    )


class BoundDeferredFSDPRunner:
    """Bind stable model and optimizer-gradient state to a deferred graph."""

    def __init__(
        self,
        deferred: DeferredFSDPGraph,
        *,
        module: nn.Module,
        gradient_state: GraphGradientState,
    ) -> None:
        self._deferred = deferred
        self._module = module
        self._gradient_state = gradient_state
        static_values = _bound_static_values(
            deferred.traced_result,
            module,
            gradient_state.graph_state,
        )
        self._static_values = static_values
        self._flat_static_inputs = _flatten_bound_state(
            deferred.traced_result,
            static_values,
        )
        self._flat_static_bindings = tuple(
            _flat_input_binding(value) for value in self._flat_static_inputs
        )

    def validate_state(self) -> None:
        parameters = tuple(
            parameter
            for _, parameter in self._module.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        )
        self._gradient_state.validate_parameters(parameters)
        self._gradient_state.validate_bindings()
        static_values = _bound_static_values(
            self._deferred.traced_result,
            self._module,
            self._gradient_state.graph_state,
        )
        flat_static_inputs = _flatten_bound_state(
            self._deferred.traced_result,
            static_values,
        )
        if any(
            actual is not expected
            for actual, expected in zip(static_values, self._static_values, strict=True)
        ) or any(
            actual is not expected
            for actual, expected in zip(
                flat_static_inputs, self._flat_static_inputs, strict=True
            )
        ):
            raise RuntimeError("Deferred FSDP bound state objects changed")
        if tuple(_flat_input_binding(value) for value in flat_static_inputs) != (
            self._flat_static_bindings
        ):
            raise RuntimeError("Deferred FSDP bound state storage changed")

    def __call__(
        self,
        microbatches: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if len(microbatches) != self._deferred.num_microbatches:
            raise ValueError(
                "Deferred FSDP graph expected "
                f"{self._deferred.num_microbatches} microbatches, got "
                f"{len(microbatches)}"
            )
        flat_inputs: list[Any] = list(self._flat_static_inputs)
        for inputs, labels, extra_kwargs in microbatches:
            user_values, spec = pytree.tree_flatten(
                ((inputs, labels, global_valid_tokens, extra_kwargs), {})
            )
            if spec != self._deferred.traced_result.user_inputs_spec:
                raise ValueError(
                    "Deferred FSDP runtime input structure changed after tracing"
                )
            flat_user_inputs, _ = _unwrap_subclasses(user_values)
            flat_inputs.extend(flat_user_inputs)
        with torch.no_grad():
            outputs = self._deferred.gm(*flat_inputs)
        if not isinstance(outputs, tuple) or len(outputs) != 1:
            raise RuntimeError(
                "Deferred FSDP graph must return exactly one loss tensor"
            )
        loss = outputs[0]
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("Deferred FSDP graph returned a non-tensor loss")
        return loss


def bind_deferred_fsdp_graph(
    deferred: DeferredFSDPGraph,
    *,
    module: nn.Module,
    gradient_state: GraphGradientState,
) -> BoundDeferredFSDPRunner:
    return BoundDeferredFSDPRunner(
        deferred,
        module=module,
        gradient_state=gradient_state,
    )
