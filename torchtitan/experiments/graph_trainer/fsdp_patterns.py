# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Pattern helpers for GraphPP's current simple-FSDP collective traces.

The matchers intentionally follow c10d functional traces produced by FSDP2:

    param_shard -> all_gather -> wait -> view*/split-cat -> compute
    local_grad -> cast/view* -> reduce_scatter -> wait -> param_grad
    local_grad -> cast/view* -> all_reduce -> wait -> param_grad

They are structural helpers for today's trace shape. A future upstream FSDP or
torch.pipelining annotation should replace this with explicit collective-region
metadata instead of broader pattern matching.
"""

import math
import operator
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx


_PACKED_FSDP_UNSHARD_PARAM = "packed_fsdp_unshard_param"


@dataclass(frozen=True, slots=True)
class FSDPReduceGradBoundary:
    """Logical accumulation input and its shared collective boundary."""

    accumulation_input: fx.Node
    collective_input: fx.Node


def is_wait_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def is_all_gather_into_tensor(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in {
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        torch.ops._c10d_functional.all_gather_into_tensor_out.default,
    }


_BUCKETED_ALL_GATHER_SPLITS = {
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.split_with_sizes_copy.default,
}
_BUCKETED_REDUCE_SCATTER_SPLITS = {
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.split_with_sizes_copy.default,
}
_GUARANTEED_STORAGE_ALIAS_OPS = {
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.alias.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
    torch.ops.aten.view.dtype,
}
_FRESH_STORAGE_ALLOCATION_OPS = {
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.empty_strided.default,
}
_FSDP_INPUT_PATH_OPS = {
    torch.ops.aten._to_copy.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.view.default,
    torch.ops.aten.view.dtype,
}


def _is_bucketed_all_gather_input(node: fx.Node) -> bool:
    """Check whether a node packs inputs for one bucketed all-gather.

    Args:
        node: FX node to inspect.

    Returns:
        ``True`` when the node is the bucketing input custom op.
    """
    schema = getattr(node.target, "_schema", None)
    return node.op == "call_function" and getattr(schema, "name", None) == (
        "bucketing::_pre_bucket_all_gather"
    )


def _is_bucketed_all_gather_out_split(node: fx.Node) -> bool:
    """Check for FSDP's mutating all-gather output unpack operation."""
    schema = getattr(node.target, "_schema", None)
    return (
        node.op == "call_function"
        and getattr(schema, "name", None)
        in {
            "aten::split_with_sizes_copy",
            "fsdp::split_with_sizes_copy",
        }
        and getattr(schema, "is_mutable", False)
        and any(
            getattr(argument, "name", None) == "out"
            for argument in getattr(schema, "arguments", ())
        )
    )


def is_fsdp_all_gather_output_split(node: fx.Node) -> bool:
    """Check whether a node splits one waited FSDP all-gather output.

    Args:
        node: FX node to inspect.

    Returns:
        ``True`` when the node is an all-gather output split.
    """
    if node.op != "call_function" or node.target not in _BUCKETED_ALL_GATHER_SPLITS:
        return False

    current = node.args[0]
    while isinstance(current, fx.Node):
        if is_wait_tensor(current):
            launch = current.args[0]
            return isinstance(launch, fx.Node) and is_all_gather_into_tensor(launch)
        if len(current.all_input_nodes) != 1:
            return False
        current = current.all_input_nodes[0]
    return False


def _find_bucket_split(pre_bucket: fx.Node) -> fx.Node:
    """Find the parameter split paired with one pre-bucket node.

    Args:
        pre_bucket: Input-packing node produced by FSDP bucketing.

    Returns:
        The view or copy split that reconstructs individual parameters.

    Raises:
        ValueError: If the bucketed all-gather structure is malformed.
    """
    launches = [user for user in pre_bucket.users if is_all_gather_into_tensor(user)]
    if len(launches) != 1:
        raise ValueError(
            "Expected one all-gather launch for FSDP pre-bucket node "
            f"{pre_bucket.name}, got {len(launches)}"
        )
    launch = launches[0]
    if len(launch.users) != 1:
        raise ValueError(
            f"Expected one wait user for bucketed all-gather {launch.name}, "
            f"got {len(launch.users)}"
        )
    current = next(iter(launch.users))
    if not is_wait_tensor(current):
        raise ValueError(
            f"Expected wait_tensor after bucketed all-gather {launch.name}, "
            f"got {current.name}"
        )

    while True:
        data_users = [user for user in current.users if current in user.all_input_nodes]
        if len(data_users) != 1:
            break
        current = data_users[0]
        if current.target in _BUCKETED_ALL_GATHER_SPLITS:
            return current
        if _is_bucketed_all_gather_out_split(current):
            return current
        if current.op != "call_function" or current.target not in _FSDP_INPUT_PATH_OPS:
            break
    raise ValueError(f"Expected a split after bucketed FSDP all-gather {launch.name}")


def _find_out_split_storage_roots(
    split: fx.Node,
    *,
    expected_outputs: int,
) -> tuple[fx.Node, ...]:
    """Return distinct storage roots mutated by an FSDP out-buffer split."""
    out = split.kwargs.get("out")
    if not isinstance(out, (list, tuple)):
        raise ValueError(f"Expected output list for FSDP split {split.name}")
    if len(out) != expected_outputs:
        raise ValueError(
            f"Expected {expected_outputs} outputs for FSDP split "
            f"{split.name}, got {len(out)}"
        )
    if not all(isinstance(output, fx.Node) for output in out):
        raise ValueError(f"Expected tensor outputs for FSDP split {split.name}")

    roots = []
    for output in out:
        current = output
        while current.target in _GUARANTEED_STORAGE_ALIAS_OPS:
            if (
                len(current.all_input_nodes) != 1
                or not current.args
                or not isinstance(current.args[0], fx.Node)
            ):
                raise ValueError(
                    f"Expected tensor input for FSDP split output {current.name}"
                )
            current = current.args[0]
        if current.target not in _FRESH_STORAGE_ALLOCATION_OPS:
            raise ValueError(
                f"Expected fresh output storage for FSDP split {split.name}, "
                f"got {current.name}"
            )
        roots.append(current)
    node_order = {node: index for index, node in enumerate(split.graph.nodes)}
    if any(node_order[root] >= node_order[split] for root in roots):
        raise ValueError(
            f"Expected FSDP split {split.name} output storage to be allocated "
            "before the split"
        )
    if len(set(roots)) != len(roots):
        raise ValueError(
            f"Expected distinct output storage for FSDP split {split.name}"
        )
    return tuple(roots)


def _find_bucketed_unshard_outputs(param_placeholder: fx.Node) -> list[fx.Node]:
    """Find bucket split outputs corresponding to one parameter placeholder.

    Args:
        param_placeholder: Flat FSDP parameter input to the joint graph.

    Returns:
        Split outputs for every forward or backward unshard of the parameter.

    Raises:
        ValueError: If bucket input or output indices are ambiguous.
    """
    outputs: list[fx.Node] = []
    worklist = [(user, param_placeholder) for user in param_placeholder.users]
    visited: set[tuple[fx.Node, fx.Node]] = set()
    while worklist:
        node, input_node = worklist.pop()
        if (node, input_node) in visited:
            continue
        visited.add((node, input_node))
        if _is_bucketed_all_gather_input(node):
            bucket_inputs = node.args[0]
            if not isinstance(bucket_inputs, (list, tuple)):
                raise ValueError(
                    f"Expected tensor list for FSDP pre-bucket node {node.name}"
                )
            input_indices = [
                index
                for index, bucket_input in enumerate(bucket_inputs)
                if bucket_input is input_node
            ]
            if len(input_indices) != 1:
                raise ValueError(
                    f"Expected {input_node.name} once in FSDP pre-bucket "
                    f"{node.name}, got {len(input_indices)} entries"
                )
            split = _find_bucket_split(node)
            index = input_indices[0]
            if _is_bucketed_all_gather_out_split(split):
                storage_roots = _find_out_split_storage_roots(
                    split,
                    expected_outputs=len(bucket_inputs),
                )
                outputs.append(storage_roots[index])
            else:
                split_outputs = [
                    user
                    for user in split.users
                    if user.target == operator.getitem and user.args[1] == index
                ]
                if len(split_outputs) != 1:
                    raise ValueError(
                        f"Expected split output {index} for FSDP pre-bucket "
                        f"{node.name}, got {len(split_outputs)}"
                    )
                outputs.append(split_outputs[0])
            continue
        if (
            node.op == "call_function"
            and node.target in _FSDP_INPUT_PATH_OPS
            and len(node.all_input_nodes) == 1
        ):
            worklist.extend((user, node) for user in node.users)
    return outputs


def is_reduce_scatter_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
    )


def is_all_reduce(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.all_reduce.default
    )


def is_reduce_grad_collective(node: fx.Node) -> bool:
    return is_reduce_scatter_tensor(node) or is_all_reduce(node)


def _find_last_all_gather_in_chain(start_node: fx.Node) -> fx.Node | None:
    """Find the final all-gather in a linear FSDP unshard launch chain."""
    node = start_node
    last_all_gather = None
    while True:
        if is_all_gather_into_tensor(node):
            last_all_gather = node
        if len(node.users) != 1:
            break
        user = next(iter(node.users))
        if len(user.all_input_nodes) > 1:
            break
        node = user
    return last_all_gather


def _find_last_user_in_wait_chain(wait_node: fx.Node) -> fx.Node:
    """Find the last FSDP unshard node before the value enters real compute.

    The traced FSDP unshard has a mostly linear shape:

        flat_param -> ... -> all_gather -> wait -> view* -> compute

    Some models reshape the gathered flat buffer through a split/cat fanout:

        wait -> split -> getitem_0 --+
                      -> getitem_1 --+-> cat -> view* -> compute

    In both cases the FSDP region ends before the first consumer with multiple
    FX inputs. The split/getitem/cat fanout is still part of reconstructing the
    unsharded parameter value, so it is included in the chain.
    """
    node = wait_node
    while True:
        if len(node.users) != 1:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.split.Tensor
                and all(
                    user.op == "call_function"
                    and user.target == operator.getitem
                    and len(user.users) == 1
                    for user in node.users
                )
            ):
                getitem_users = [next(iter(user.users)) for user in node.users]
                potential_cat = getitem_users[0]
                if all(user == potential_cat for user in getitem_users) and (
                    potential_cat.op == "call_function"
                    and potential_cat.target == torch.ops.aten.cat.default
                ):
                    node = potential_cat
                    continue
            break

        user = next(iter(node.users))
        if len(user.all_input_nodes) > 1:
            break
        node = user
    return node


def _find_last_fsdp_reconstruction_user(wait_node: fx.Node) -> fx.Node:
    """Find the last node that only reconstructs the gathered parameter.

    Args:
        wait_node: Wait node for one FSDP all-gather.

    Returns:
        The final view-like FSDP reconstruction node before format-specific
        weight preparation or model compute begins.
    """
    node = wait_node
    while True:
        if len(node.users) != 1:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.split.Tensor
                and all(
                    user.op == "call_function"
                    and user.target == operator.getitem
                    and len(user.users) == 1
                    for user in node.users
                )
            ):
                getitem_users = [next(iter(user.users)) for user in node.users]
                potential_cat = getitem_users[0]
                if all(user == potential_cat for user in getitem_users) and (
                    potential_cat.op == "call_function"
                    and potential_cat.target == torch.ops.aten.cat.default
                ):
                    node = potential_cat
                    continue
            break

        user = next(iter(node.users))
        if (
            user.op != "call_function"
            or user.target not in _FSDP_INPUT_PATH_OPS
            or len(user.all_input_nodes) != 1
        ):
            break
        node = user
    return node


def _find_last_non_view_node_in_chain(node: fx.Node) -> fx.Node:
    """Return the value GraphPP should pass across the unshard boundary."""
    result = node
    while hasattr(result.target, "is_view") and result.target.is_view:
        if len(result.all_input_nodes) != 1:
            raise ValueError(f"View node {result.name} should have exactly one input")
        result = result.all_input_nodes[0]
    return result


def _unshard_output_from_all_gather(last_all_gather: fx.Node) -> fx.Node:
    if len(last_all_gather.users) != 1:
        raise ValueError(
            f"Expected one wait_tensor user for all_gather node {last_all_gather.name}, "
            f"got {len(last_all_gather.users)}"
        )
    wait_node = next(iter(last_all_gather.users))
    if not is_wait_tensor(wait_node):
        raise ValueError(
            f"Expected wait_tensor after all_gather node {last_all_gather.name}, "
            f"got {wait_node.name}"
        )

    wait_chain_user = _find_last_user_in_wait_chain(wait_node)
    return _find_last_non_view_node_in_chain(wait_chain_user)


def _unshard_reconstruction_output_from_all_gather(
    last_all_gather: fx.Node,
) -> fx.Node:
    """Return the high-precision FSDP reconstruction boundary.

    Args:
        last_all_gather: Final all-gather launch in one FSDP unshard chain.

    Returns:
        The reconstructed parameter before format-specific preparation.

    Raises:
        ValueError: If the all-gather is not followed by exactly one wait.
    """
    if len(last_all_gather.users) != 1:
        raise ValueError(
            f"Expected one wait_tensor user for all_gather node "
            f"{last_all_gather.name}, got {len(last_all_gather.users)}"
        )
    wait_node = next(iter(last_all_gather.users))
    if not is_wait_tensor(wait_node):
        raise ValueError(
            f"Expected wait_tensor after all_gather node {last_all_gather.name}, "
            f"got {wait_node.name}"
        )
    reconstruction_user = _find_last_fsdp_reconstruction_user(wait_node)
    return _find_last_non_view_node_in_chain(reconstruction_user)


def find_fsdp_unshard_outputs(param_placeholder: fx.Node) -> tuple[fx.Node, ...]:
    """Return all FSDP unshard outputs launched from one flat parameter input.

    Most parameters have one linear all-gather chain. Some real traces read the
    same parametrized value more than once, which produces multiple equivalent
    all-gather/wait chains from the same placeholder.
    ``deduplicate_fsdp_unshard_chains_pass`` canonicalizes those duplicate
    chains before downstream FSDP passes rely on a single unsharded value.
    """
    packed_outputs = tuple(
        node
        for node in param_placeholder.graph.nodes
        if node.meta.get(_PACKED_FSDP_UNSHARD_PARAM) == param_placeholder.name
    )
    if packed_outputs:
        return packed_outputs

    bucketed_outputs = _find_bucketed_unshard_outputs(param_placeholder)
    if bucketed_outputs:
        return tuple(bucketed_outputs)

    last_all_gather = _find_last_all_gather_in_chain(param_placeholder)
    if last_all_gather is not None:
        return (_unshard_output_from_all_gather(last_all_gather),)

    outputs: list[fx.Node] = []
    seen: set[fx.Node] = set()
    for user in param_placeholder.users:
        if len(user.all_input_nodes) > 1:
            continue
        last_all_gather = _find_last_all_gather_in_chain(user)
        if last_all_gather is None:
            continue
        output = _unshard_output_from_all_gather(last_all_gather)
        if output not in seen:
            seen.add(output)
            outputs.append(output)
    return tuple(outputs)


def find_fsdp_unshard_reconstruction_outputs(
    param_placeholder: fx.Node,
) -> tuple[fx.Node, ...]:
    """Return high-precision reconstruction outputs for each unshard chain.

    Unlike :func:`find_fsdp_unshard_outputs`, this matcher never crosses into
    format-specific post-all-gather preparation. It is used when distinct
    preparations, such as row- and column-layout MXFP8 quantization, share one
    FSDP parameter and must retain their separate semantics.

    Args:
        param_placeholder: Flat FSDP parameter input to the joint graph.

    Returns:
        One reconstructed high-precision parameter node per unshard chain.
    """
    bucketed_outputs = _find_bucketed_unshard_outputs(param_placeholder)
    if bucketed_outputs:
        return tuple(bucketed_outputs)

    outputs: list[fx.Node] = []
    seen: set[fx.Node] = set()
    starts = [param_placeholder, *param_placeholder.users]
    for start in starts:
        last_all_gather = _find_last_all_gather_in_chain(start)
        if last_all_gather is None:
            continue
        output = _unshard_reconstruction_output_from_all_gather(last_all_gather)
        if output not in seen:
            seen.add(output)
            outputs.append(output)
    return tuple(outputs)


def find_fsdp_unshard_output(param_placeholder: fx.Node) -> fx.Node | None:
    """Return the extracted unshard output for one flat parameter input.

    GraphPP and the SAC force-save policy must agree on this node. It is the
    same value AutoParallel saves for ``reshard_after_forward=False``: the last
    non-view node in the FSDP all-gather/wait reconstruction chain. Parameters
    without an all-gather are replicated or otherwise already local, so callers
    should keep the original placeholder as that parameter's unsharded value.

    TODO(sanketpurandare): requires upstream change: FSDP trace/passes should
    annotate unshard collective regions for downstream graph extraction.
    """
    outputs = find_fsdp_unshard_outputs(param_placeholder)
    if not outputs:
        return None
    return outputs[0]


def find_fsdp_unshard_save_node(param_placeholder: fx.Node) -> fx.Node | None:
    return find_fsdp_unshard_output(param_placeholder)


def find_fsdp_unshard_save_nodes(param_placeholder: fx.Node) -> tuple[fx.Node, ...]:
    """Return all FSDP unshard values that SAC must save for one parameter."""
    return find_fsdp_unshard_outputs(param_placeholder)


def find_fsdp_reduce_grad_input(
    param_grad_output: Any,
    *,
    allow_bucket_fanout: bool = False,
) -> fx.Node | None:
    """Return the split point before an FSDP reduce-grad epilogue.

    The backward FSDP/DDP/HSDP tail is traced as a chain ending in the synced
    grad output:

        local_grad -> cast/view* -> reduce_scatter -> wait -> sharded_grad
        local_grad -> cast/view* -> all_reduce -> wait -> replicated_grad
        local_grad -> cast/view* -> all_reduce -> wait -> reduce_scatter
          -> wait -> grad
        local_grads -> pre_bucket -> reduce_scatter -> wait -> split
          -> getitem/view* -> grads

    GraphPP splits at the input to the earliest grad-sync collective in that
    suffix. The cast remains in ``bw_no_fsdp`` so microbatch accumulation
    happens in FSDP's reduce dtype, and ``reduce_grad`` contains only the
    scheduled collective epilogue. Values that are not FX nodes, such as
    ``None`` parameter-grad slots, are not collective outputs and are preserved
    by the caller. ``allow_bucket_fanout`` is reserved for callers that dedupe
    shared packed bucket inputs before accumulating them; GraphPP's per-parameter
    accumulation must retain the default behavior.

    TODO(sanketpurandare): requires upstream change: FSDP trace/passes should
    annotate reduce-grad collective regions for downstream graph extraction.
    """
    if not isinstance(param_grad_output, fx.Node):
        return None

    node = param_grad_output
    reduce_grad_input = None
    while isinstance(node, fx.Node) and len(node.all_input_nodes) == 1:
        input_node = node.all_input_nodes[0]
        is_bucket_split_output = (
            node.op == "call_function"
            and node.target == operator.getitem
            and input_node.op == "call_function"
            and input_node.target in _BUCKETED_REDUCE_SCATTER_SPLITS
        )
        if len(input_node.users) > 1 and not (
            allow_bucket_fanout and is_bucket_split_output
        ):
            break
        previous_node = node
        node = input_node
        if is_reduce_grad_collective(previous_node):
            reduce_grad_input = node
    return reduce_grad_input


def _is_bucketed_reduce_scatter_input(node: fx.Node) -> bool:
    schema = getattr(node.target, "_schema", None)
    return node.op == "call_function" and getattr(schema, "name", None) in {
        "bucketing::_pre_bucket_reduce_scatter",
        "bucketing::_pre_bucket_reduce_scatter_chunk_cat",
    }


def _bucket_split_output(
    node: fx.Node,
) -> tuple[fx.Node, int] | None:
    if (
        node.op != "call_function"
        or node.target != operator.getitem
        or len(node.all_input_nodes) != 1
    ):
        return None
    split = node.all_input_nodes[0]
    if (
        split.op != "call_function"
        or split.target not in _BUCKETED_REDUCE_SCATTER_SPLITS
    ):
        return None
    if len(node.args) < 2 or type(node.args[1]) is not int:
        raise ValueError(f"Expected a static output index for {split.name}")
    return split, node.args[1]


def _bucket_output_numel(
    bucket_input: fx.Node,
    group_size: int,
    *,
    is_unwrapped: bool,
) -> int | None:
    """Return one rank's packed segment for a logical or prepacked input."""
    value = bucket_input.meta.get("val")
    if not isinstance(value, torch.Tensor) or len(value.shape) == 0:
        return None
    shape = tuple(value.shape)
    if not all(type(size) is int for size in shape):
        return None
    if is_unwrapped is False:
        if shape[0] % group_size != 0:
            raise ValueError(
                f"Packed bucket input {bucket_input.name} is not divisible by "
                f"group size {group_size}"
            )
        return math.prod(shape) // group_size
    return ((shape[0] + group_size - 1) // group_size) * math.prod(shape[1:])


def _unwrapped_bucket_input_indices(
    pre_bucket: fx.Node,
    num_inputs: int,
) -> set[int]:
    schema_name = getattr(getattr(pre_bucket.target, "_schema", None), "name", None)
    if schema_name == "bucketing::_pre_bucket_reduce_scatter_chunk_cat":
        return set(range(num_inputs))
    if len(pre_bucket.args) <= 2 and "unwrapped_input_indices" not in (
        pre_bucket.kwargs
    ):
        return set()
    raw_indices = (
        pre_bucket.args[2]
        if len(pre_bucket.args) > 2
        else pre_bucket.kwargs["unwrapped_input_indices"]
    )
    if raw_indices is None:
        return set()
    if not isinstance(raw_indices, (list, tuple)) or not all(
        type(index) is int and 0 <= index < num_inputs for index in raw_indices
    ):
        raise ValueError(f"Invalid unwrapped input indices for {pre_bucket.name}")
    if len(set(raw_indices)) != len(raw_indices):
        raise ValueError(f"Duplicate unwrapped input indices for {pre_bucket.name}")
    return set(raw_indices)


def _bucket_inputs(
    pre_bucket: fx.Node,
    split: fx.Node,
    reduce_scatter: fx.Node,
) -> tuple[fx.Node, ...]:
    raw_inputs = pre_bucket.args[0] if pre_bucket.args else None
    if not isinstance(raw_inputs, (list, tuple)) or not all(
        isinstance(bucket_input, fx.Node) for bucket_input in raw_inputs
    ):
        raise ValueError(
            "Expected tensor list for reduce-scatter pre-bucket node "
            f"{pre_bucket.name}"
        )
    bucket_inputs = tuple(raw_inputs)
    group_size = (
        pre_bucket.args[1]
        if len(pre_bucket.args) > 1
        else pre_bucket.kwargs.get("group_size")
    )
    if type(group_size) is not int or group_size <= 0:
        raise ValueError(f"Expected a positive static group size for {pre_bucket.name}")
    reduce_scatter_group_size = (
        reduce_scatter.args[2]
        if len(reduce_scatter.args) > 2
        else reduce_scatter.kwargs.get("group_size")
    )
    if (
        type(reduce_scatter_group_size) is not int
        or reduce_scatter_group_size != group_size
    ):
        raise ValueError(
            f"Group sizes of {pre_bucket.name} and {reduce_scatter.name} "
            f"do not match: {group_size} != {reduce_scatter_group_size}"
        )
    unwrapped_indices = _unwrapped_bucket_input_indices(
        pre_bucket,
        len(bucket_inputs),
    )

    split_dim = split.args[2] if len(split.args) > 2 else split.kwargs.get("dim", 0)
    if type(split_dim) is not int or split_dim != 0:
        raise ValueError(
            f"Expected {split.name} to split reduce-scatter output on dimension zero"
        )
    split_sizes = split.args[1] if len(split.args) > 1 else None
    if not isinstance(split_sizes, (list, tuple)):
        raise NotImplementedError(
            "Deferred bucketed FSDP gradient sync requires explicit "
            "reduce-scatter output split sizes"
        )
    if len(split_sizes) != len(bucket_inputs):
        raise ValueError(
            f"Expected one output of {split.name} per input to "
            f"{pre_bucket.name}, got {len(split_sizes)} outputs and "
            f"{len(bucket_inputs)} inputs"
        )
    expected_sizes = tuple(
        _bucket_output_numel(
            bucket_input,
            group_size,
            is_unwrapped=index in unwrapped_indices,
        )
        for index, bucket_input in enumerate(bucket_inputs)
    )
    if any(size is None for size in expected_sizes) or not all(
        type(size) is int for size in split_sizes
    ):
        raise NotImplementedError(
            "Deferred bucketed FSDP gradient sync requires static bucket "
            "input shapes and output split sizes"
        )
    if tuple(split_sizes) != expected_sizes:
        raise ValueError(
            f"Output sizes of {split.name} do not match the tensor-list "
            f"inputs to {pre_bucket.name}: {tuple(split_sizes)} != "
            f"{expected_sizes}"
        )
    return bucket_inputs


def _has_upstream_reduce_grad_collective(node: fx.Node) -> bool:
    while isinstance(node, fx.Node):
        if is_reduce_grad_collective(node):
            return True
        if len(node.all_input_nodes) != 1:
            return False
        input_node = node.all_input_nodes[0]
        if len(input_node.users) != 1:
            return False
        node = input_node
    return False


def _find_bucketed_reduce_grad_boundary(
    param_grad_output: Any,
) -> FSDPReduceGradBoundary | None:
    if not isinstance(param_grad_output, fx.Node):
        return None

    node = param_grad_output
    split_node: fx.Node | None = None
    split_output_index: int | None = None
    while isinstance(node, fx.Node):
        if is_reduce_scatter_tensor(node):
            if split_node is None or split_output_index is None:
                return None
            pre_bucket = node.args[0] if node.args else None
            if not isinstance(pre_bucket, fx.Node) or not (
                _is_bucketed_reduce_scatter_input(pre_bucket)
            ):
                raise ValueError(
                    "Expected bucketed reduce-scatter input before " f"{node.name}"
                )
            if set(pre_bucket.users) != {node}:
                raise NotImplementedError(
                    "Deferred bucketed FSDP gradient sync requires the "
                    "pre-bucket to have its reduce-scatter as its sole user"
                )
            bucket_inputs = _bucket_inputs(pre_bucket, split_node, node)
            if not -len(bucket_inputs) <= split_output_index < len(bucket_inputs):
                raise ValueError(
                    f"Bucket output index {split_output_index} is outside the "
                    f"{len(bucket_inputs)} inputs to {pre_bucket.name}"
                )
            bucket_input = bucket_inputs[split_output_index]
            if set(bucket_input.users) != {pre_bucket}:
                raise NotImplementedError(
                    "Deferred bucketed FSDP gradient sync requires each "
                    "logical bucket input to have the pre-bucket as its sole user"
                )
            if _has_upstream_reduce_grad_collective(bucket_input):
                raise NotImplementedError(
                    "Deferred bucketed FSDP gradient sync requires the "
                    "reduce-scatter to be the first gradient collective"
                )
            return FSDPReduceGradBoundary(bucket_input, pre_bucket)

        if is_all_reduce(node):
            all_reduce_input = node.args[0] if node.args else None
            if not isinstance(all_reduce_input, fx.Node):
                return None
            node = all_reduce_input
            continue

        if len(node.all_input_nodes) != 1:
            if split_node is not None:
                raise ValueError(
                    "Bucketed reduce-scatter output does not have a linear "
                    f"path from {split_node.name} to its collective"
                )
            return None
        input_node = node.all_input_nodes[0]
        bucket_split_output = _bucket_split_output(node)
        if bucket_split_output is not None:
            if split_node is not None:
                raise ValueError("Nested bucketed reduce-scatter output splits")
            split_node, split_output_index = bucket_split_output
        elif len(input_node.users) > 1:
            if split_node is not None:
                raise ValueError(
                    "Bucketed reduce-scatter output has an unexpected fanout "
                    f"at {input_node.name}"
                )
            return None
        node = input_node

    if split_node is not None:
        raise ValueError(
            f"Could not find reduce-scatter for bucket output {split_node.name}"
        )
    return None


def find_fsdp_reduce_grad_boundary(
    param_grad_output: Any,
) -> FSDPReduceGradBoundary | None:
    """Return the logical value to accumulate before one reduce-grad tail.

    Bucketed reduce-scatter outputs share one packed collective input. Each
    output maps by split index to the exact tensor-list element packed for that
    output, while ``collective_input`` remains shared across the whole bucket.
    Bucketed HSDP is supported when reduce-scatter precedes all-reduce; an
    all-reduce before the packed reduce-scatter is rejected because it would
    otherwise remain in every non-final microbatch graph.
    GraphPP continues to use :func:`find_fsdp_reduce_grad_input`, whose default
    single-boundary behavior is unchanged.
    """
    bucketed = _find_bucketed_reduce_grad_boundary(param_grad_output)
    if bucketed is not None:
        return bucketed
    accumulation_input = find_fsdp_reduce_grad_input(param_grad_output)
    if accumulation_input is None:
        return None
    return FSDPReduceGradBoundary(accumulation_input, accumulation_input)
