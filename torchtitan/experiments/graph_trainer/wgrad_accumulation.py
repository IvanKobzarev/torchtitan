# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse deferred gradient accumulation into compatible WGrad producers."""

from __future__ import annotations

import copy
import math
import operator
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx

from torchtitan.tools.logging import logger


_DEFERRED_ACCUMULATION_MARKERS = (
    "deferred_fsdp_gradient_accumulation",
    "deferred_fsdp_final_gradient",
)
_VIEW_TARGETS = frozenset(
    {
        torch.ops.aten.alias.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    }
)
_DIST_MOE_ACCUMULATION_TARGETS = {
    "dist_moe.block_scaled_backward.default": "block_scaled_backward_accumulate",
    "dist_moe.bf16_backward.default": "bf16_backward_accumulate",
}


@dataclass(frozen=True, slots=True)
class _DistMoeWgradSink:
    sink: fx.Node
    accumulator: fx.Node
    boundary: fx.Node
    getitem: fx.Node
    view_chain: tuple[fx.Node, ...]


def _tensor_meta(node: fx.Node) -> torch.Tensor | None:
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _static_numel(tensor: torch.Tensor) -> int | None:
    if not all(isinstance(dim, int) for dim in tensor.shape):
        return None
    return math.prod(tensor.shape)


def _node_argument(
    node: fx.Node,
    name: str,
    position: int,
    default: Any,
) -> Any:
    if name in node.kwargs:
        return node.kwargs[name]
    if position < len(node.args):
        return node.args[position]
    return default


def _is_marked_accumulation(node: fx.Node) -> bool:
    return node.target == torch.ops.aten.add_.Tensor and any(
        node.meta.get(marker) for marker in _DEFERRED_ACCUMULATION_MARKERS
    )


def _sole_user(node: fx.Node, expected: fx.Node) -> bool:
    return len(node.users) == 1 and expected in node.users


def _is_storage_alias(node: fx.Node) -> bool:
    if node.target in _VIEW_TARGETS:
        return True
    if node.target != torch.ops.aten.reshape.default:
        return False
    if not node.args or not isinstance(node.args[0], fx.Node):
        return False
    source_value = _tensor_meta(node.args[0])
    result_value = _tensor_meta(node)
    if source_value is None or result_value is None:
        return False
    return (
        source_value.is_contiguous()
        and result_value.is_contiguous()
        and source_value.dtype == result_value.dtype
        and source_value.device == result_value.device
        and _static_numel(source_value) == _static_numel(result_value)
    )


def _source_through_views(
    boundary: fx.Node,
    sink: fx.Node,
) -> tuple[fx.Node, tuple[fx.Node, ...]] | None:
    chain: list[fx.Node] = []
    current = boundary
    expected_user = sink
    while _is_storage_alias(current):
        if not _sole_user(current, expected_user):
            return None
        if not current.args or not isinstance(current.args[0], fx.Node):
            return None
        chain.append(current)
        expected_user = current
        current = current.args[0]

    if not _sole_user(current, expected_user):
        return None
    return current, tuple(chain)


def _is_compatible_bf16_accumulator(
    accumulator: fx.Node,
    boundary: fx.Node,
) -> bool:
    accumulator_value = _tensor_meta(accumulator)
    boundary_value = _tensor_meta(boundary)
    if accumulator_value is None or boundary_value is None:
        return False
    accumulator_numel = _static_numel(accumulator_value)
    boundary_numel = _static_numel(boundary_value)
    accumulator_stride = accumulator_value.stride()
    boundary_stride = boundary_value.stride()
    if (
        accumulator_numel is None
        or boundary_numel is None
        or not all(
            isinstance(stride, int)
            for stride in (*accumulator_stride, *boundary_stride)
        )
    ):
        return False
    return (
        accumulator_value.dtype == torch.bfloat16
        and boundary_value.dtype == torch.bfloat16
        and accumulator_value.device.type == "cuda"
        and boundary_value.device == accumulator_value.device
        and accumulator_value.is_contiguous()
        and accumulator_value.shape == boundary_value.shape
        and accumulator_stride == boundary_stride
        and accumulator_numel == boundary_numel
    )


def _eligible_scaled_mm(
    producer: fx.Node,
    boundary: fx.Node,
    accumulator: fx.Node,
) -> tuple[tuple[Any, Any, Any, Any], bool] | None:
    if accumulator.op != "placeholder":
        return None
    if len(producer.args) < 4:
        return None

    bias = _node_argument(producer, "bias", 4, None)
    scale_result = _node_argument(producer, "scale_result", 5, None)
    out_dtype = _node_argument(producer, "out_dtype", 6, None)
    use_fast_accum = _node_argument(producer, "use_fast_accum", 7, False)
    if (
        bias is not None
        or scale_result is not None
        or out_dtype != torch.bfloat16
        or not isinstance(use_fast_accum, bool)
    ):
        return None

    producer_value = _tensor_meta(producer)
    if producer_value is None or not _is_compatible_bf16_accumulator(
        accumulator, boundary
    ):
        return None
    accumulator_value = _tensor_meta(accumulator)
    boundary_value = _tensor_meta(boundary)
    assert accumulator_value is not None
    assert boundary_value is not None
    if (
        producer_value.dtype != torch.bfloat16
        or producer_value.device != accumulator_value.device
        or producer_value.dim() != 2
    ):
        return None

    if _static_numel(producer_value) != _static_numel(boundary_value):
        return None

    return (
        (
            producer.args[0],
            producer.args[1],
            producer.args[2],
            producer.args[3],
        ),
        use_fast_accum,
    )


def _fuse_scaled_mm_sink(gm: fx.GraphModule, sink: fx.Node) -> bool:
    if len(sink.args) < 2:
        return False
    accumulator, boundary = sink.args[:2]
    if not isinstance(accumulator, fx.Node) or not isinstance(boundary, fx.Node):
        return False
    alpha = _node_argument(sink, "alpha", 2, 1)
    if alpha != 1 or not _sole_user(accumulator, sink):
        return False

    source = _source_through_views(boundary, sink)
    if source is None:
        return False
    producer, view_chain = source
    if producer.target != torch.ops.aten._scaled_mm.default:
        return False
    eligible = _eligible_scaled_mm(producer, boundary, accumulator)
    if eligible is None:
        return False
    operands, use_fast_accum = eligible
    producer_value = _tensor_meta(producer)
    assert producer_value is not None

    with gm.graph.inserting_before(producer):
        accumulator_view = gm.graph.call_function(
            torch.ops.aten.view.default,
            args=(accumulator, list(producer_value.shape)),
        )
        fused = gm.graph.call_function(
            torch.ops.aten._scaled_addmm_.default,
            args=(accumulator_view, *operands),
            kwargs={"use_fast_accum": use_fast_accum},
        )
    accumulator_view.meta = copy.copy(producer.meta)
    accumulator_view.meta["original_aten"] = torch.ops.aten.view.default
    fused.meta = copy.copy(producer.meta)
    fused.meta["original_aten"] = torch.ops.aten._scaled_addmm_.default
    fused.meta["deferred_fsdp_fused_wgrad_accumulation"] = True

    sink_replacement = boundary if view_chain else fused
    producer.replace_all_uses_with(fused)
    sink_users = tuple(sink.users)
    sink.replace_all_uses_with(sink_replacement)
    gm.graph.erase_node(sink)
    gm.graph.erase_node(producer)

    if not sink_users:
        for view in view_chain:
            if not view.users:
                gm.graph.erase_node(view)
    return True


def _eligible_dist_moe_sink(sink: fx.Node) -> _DistMoeWgradSink | None:
    if len(sink.args) < 2 or _node_argument(sink, "alpha", 2, 1) != 1:
        return None
    accumulator, boundary = sink.args[:2]
    if not isinstance(accumulator, fx.Node) or not isinstance(boundary, fx.Node):
        return None
    if accumulator.op != "placeholder" or not _sole_user(accumulator, sink):
        return None

    source = _source_through_views(boundary, sink)
    if source is None:
        return None
    getitem, view_chain = source
    if (
        getitem.target is not operator.getitem
        or len(getitem.args) < 2
        or getitem.args[1] not in (2, 3)
        or not isinstance(getitem.args[0], fx.Node)
    ):
        return None
    backward = getitem.args[0]
    if str(backward.target) not in _DIST_MOE_ACCUMULATION_TARGETS:
        return None

    if not _is_compatible_bf16_accumulator(accumulator, boundary):
        return None
    return _DistMoeWgradSink(
        sink=sink,
        accumulator=accumulator,
        boundary=boundary,
        getitem=getitem,
        view_chain=view_chain,
    )


def _dist_moe_accumulation_target(backward: fx.Node) -> Any | None:
    op_name = _DIST_MOE_ACCUMULATION_TARGETS.get(str(backward.target))
    if op_name is None:
        return None
    try:
        return getattr(torch.ops.dist_moe, op_name).default
    except AttributeError:
        return None


def _has_only_expected_wgrad_getitems(
    backward: fx.Node,
    matches: dict[int, _DistMoeWgradSink],
) -> bool:
    for user in backward.users:
        if user.target is not operator.getitem or len(user.args) < 2:
            return False
        index = user.args[1]
        if index in (2, 3) and matches.get(index) is not None:
            if user is not matches[index].getitem:
                return False
    return True


def _erase_dist_moe_wgrad_sink(
    gm: fx.GraphModule,
    match: _DistMoeWgradSink,
) -> None:
    match.sink.replace_all_uses_with(match.accumulator)
    gm.graph.erase_node(match.sink)
    for view in match.view_chain:
        if not view.users:
            gm.graph.erase_node(view)
    gm.graph.erase_node(match.getitem)


def _fuse_dist_moe_backward(
    gm: fx.GraphModule,
    backward: fx.Node,
    matches: dict[int, _DistMoeWgradSink],
) -> bool:
    if set(matches) != {2, 3} or not _has_only_expected_wgrad_getitems(
        backward, matches
    ):
        return False
    target = _dist_moe_accumulation_target(backward)
    if target is None:
        return False
    values = backward.meta.get("val")
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        return False

    backward.target = target
    backward.args = (
        matches[2].accumulator,
        matches[3].accumulator,
        *backward.args,
    )
    backward.meta["val"] = tuple(values[:2])
    backward.meta["original_aten"] = target
    backward.meta["deferred_fsdp_fused_wgrad_accumulation"] = True
    _erase_dist_moe_wgrad_sink(gm, matches[2])
    _erase_dist_moe_wgrad_sink(gm, matches[3])
    return True


def _fuse_dist_moe_sinks(gm: fx.GraphModule) -> int:
    matches_by_backward: dict[fx.Node, dict[int, _DistMoeWgradSink]] = {}
    ambiguous_backwards: set[fx.Node] = set()
    for sink in tuple(gm.graph.nodes):
        if not _is_marked_accumulation(sink):
            continue
        match = _eligible_dist_moe_sink(sink)
        if match is None:
            continue
        backward = match.getitem.args[0]
        assert isinstance(backward, fx.Node)
        index = match.getitem.args[1]
        assert isinstance(index, int)
        matches = matches_by_backward.setdefault(backward, {})
        if index in matches:
            ambiguous_backwards.add(backward)
        else:
            matches[index] = match

    return sum(
        _fuse_dist_moe_backward(gm, backward, matches)
        for backward, matches in matches_by_backward.items()
        if backward not in ambiguous_backwards
    )


def fuse_deferred_wgrad_accumulation_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
) -> fx.GraphModule:
    """Fuse marked deferred BF16 accumulation into compatible WGrad ops.

    The first microbatch still creates each gradient. Reused middle and final
    graphs update the stable deferred-gradient input directly. Unsupported
    producers retain the explicit ``add_`` accumulation sink.
    """
    del example_inputs
    num_scaled_mm_fused = 0
    for node in tuple(gm.graph.nodes):
        if _is_marked_accumulation(node) and _fuse_scaled_mm_sink(gm, node):
            num_scaled_mm_fused += 1
    num_dist_moe_fused = _fuse_dist_moe_sinks(gm)

    if num_scaled_mm_fused or num_dist_moe_fused:
        gm.graph.lint()
        gm.recompile()
        logger.info(
            "Fused deferred WGrad accumulation into %d scaled GEMMs and %d "
            "DistMoE backwards",
            num_scaled_mm_fused,
            num_dist_moe_fused,
        )
    return gm
