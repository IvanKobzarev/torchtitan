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

SimpleFSDP traces annotate the unshard construction with its parameter FQN.
The structural match runs before collective bucketing and uses that provenance
to stop before real compute. Its annotations keep the discovered parameter
boundary available to later graph passes after the original all-gather and
wait nodes have been replaced.
"""

import operator
from typing import Any

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.simple_fsdp import FSDP_PARAM_FQNS_META


_FSDP_UNSHARD_OUTPUT_PARAM_NAMES = "fsdp_unshard_output_param_names"
_FSDP_UNSHARD_CONSUMER_INPUTS = "fsdp_unshard_consumer_inputs"


def is_wait_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def is_all_gather_into_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


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


def _fsdp_param_fqns(node: fx.Node) -> tuple[str, ...]:
    return node.meta.get("custom", {}).get(FSDP_PARAM_FQNS_META, ())


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

    Trace-time parameter metadata defines the region when available. The
    structural fallback ends before the first consumer with multiple FX inputs.
    The split/getitem/cat fanout is included in both cases.
    """
    param_fqns = _fsdp_param_fqns(wait_node)
    node = wait_node
    while True:
        users = tuple(node.users)
        if param_fqns:
            users = tuple(
                user
                for user in users
                if not user.meta.get("autograd_backward", False)
                and _fsdp_param_fqns(user) == param_fqns
            )

        if len(users) != 1:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.split.Tensor
                and users
                and all(
                    user.op == "call_function"
                    and user.target == operator.getitem
                    and len(user.users) == 1
                    for user in users
                )
            ):
                getitem_users = [next(iter(user.users)) for user in users]
                potential_cat = getitem_users[0]
                if all(user == potential_cat for user in getitem_users) and (
                    potential_cat.op == "call_function"
                    and potential_cat.target == torch.ops.aten.cat.default
                    and (
                        not param_fqns or _fsdp_param_fqns(potential_cat) == param_fqns
                    )
                ):
                    node = potential_cat
                    continue
            break

        user = users[0]
        if not param_fqns and len(user.all_input_nodes) > 1:
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

    all_gather_fqns = _fsdp_param_fqns(last_all_gather)
    wait_fqns = _fsdp_param_fqns(wait_node)
    if (all_gather_fqns or wait_fqns) and all_gather_fqns != wait_fqns:
        raise ValueError(
            "FSDP trace metadata does not match between all-gather "
            f"{last_all_gather.name} {all_gather_fqns} and wait "
            f"{wait_node.name} {wait_fqns}"
        )

    wait_chain_user = _find_last_user_in_wait_chain(wait_node)
    output = _find_last_non_view_node_in_chain(wait_chain_user)
    if all_gather_fqns and _fsdp_param_fqns(output) != all_gather_fqns:
        raise ValueError(
            f"FSDP unshard output {output.name} does not carry parameter "
            f"metadata {all_gather_fqns}"
        )
    return output


def _find_fsdp_unshard_outputs_structural(
    param_placeholder: fx.Node,
) -> tuple[fx.Node, ...]:
    """Match FSDP unshard outputs in the original, unbucketed graph."""
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


def annotate_fsdp_unshard_outputs(gm: fx.GraphModule) -> None:
    """Preserve FSDP parameter boundaries across collective bucketing.

    Bucketing replaces each original all-gather and wait with a shared bucket
    plus reconstructed parameter values. Record both the selected unshard
    output and its consumer input edges before that rewrite. The output marker
    preserves subclass operations such as weight quantization when they
    survive bucketing. The consumer edge recovers the replacement value when
    bucketing removes the original wait output.
    """
    for placeholder in gm.graph.find_nodes(op="placeholder"):
        for output in _find_fsdp_unshard_outputs_structural(placeholder):
            output_param_names = tuple(
                output.meta.get(_FSDP_UNSHARD_OUTPUT_PARAM_NAMES, ())
            )
            if placeholder.name not in output_param_names:
                output.meta[_FSDP_UNSHARD_OUTPUT_PARAM_NAMES] = (
                    *output_param_names,
                    placeholder.name,
                )

            for consumer in output.users:
                input_indices = tuple(
                    index
                    for index, input_node in enumerate(consumer.all_input_nodes)
                    if input_node is output
                )
                if not input_indices:
                    continue
                consumer_inputs = dict(
                    consumer.meta.get(_FSDP_UNSHARD_CONSUMER_INPUTS, {})
                )
                consumer_inputs[placeholder.name] = input_indices
                consumer.meta[_FSDP_UNSHARD_CONSUMER_INPUTS] = consumer_inputs


def _find_annotated_fsdp_unshard_outputs(
    param_placeholder: fx.Node,
) -> tuple[fx.Node, ...]:
    graph = param_placeholder.graph
    marked_outputs = tuple(
        node
        for node in graph.nodes
        if param_placeholder.name in node.meta.get(_FSDP_UNSHARD_OUTPUT_PARAM_NAMES, ())
    )
    if marked_outputs:
        return marked_outputs

    outputs: list[fx.Node] = []
    seen: set[fx.Node] = set()
    for consumer in graph.nodes:
        consumer_inputs = consumer.meta.get(_FSDP_UNSHARD_CONSUMER_INPUTS, {})
        input_indices = consumer_inputs.get(param_placeholder.name, ())
        current_inputs = consumer.all_input_nodes
        for input_index in input_indices:
            if input_index >= len(current_inputs):
                raise ValueError(
                    f"FSDP unshard consumer {consumer.name} lost input "
                    f"{input_index} for parameter {param_placeholder.name}"
                )
            output = current_inputs[input_index]
            if output not in seen:
                seen.add(output)
                outputs.append(output)
    return tuple(outputs)


def find_fsdp_unshard_outputs(param_placeholder: fx.Node) -> tuple[fx.Node, ...]:
    """Return all FSDP unshard outputs launched from one flat parameter input.

    Most parameters have one linear all-gather chain. Some real traces read the
    same parametrized value more than once, which produces multiple equivalent
    all-gather/wait chains from the same placeholder.
    ``deduplicate_fsdp_unshard_chains_pass`` canonicalizes those duplicate
    chains and annotates their boundaries before downstream FSDP passes rely on
    a single unsharded value. The annotation remains valid after FSDP bucketing
    replaces the original collective and wait nodes.
    """
    annotated_outputs = _find_annotated_fsdp_unshard_outputs(param_placeholder)
    if annotated_outputs:
        return annotated_outputs
    return _find_fsdp_unshard_outputs_structural(param_placeholder)


def find_fsdp_unshard_output(param_placeholder: fx.Node) -> fx.Node | None:
    """Return the extracted unshard output for one flat parameter input.

    GraphPP and the SAC force-save policy must agree on this node. It is the
    same value AutoParallel saves for ``reshard_after_forward=False``: the last
    non-view node in the FSDP all-gather/wait reconstruction chain. Parameters
    without an all-gather are replicated or otherwise already local, so callers
    should keep the original placeholder as that parameter's unsharded value.

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


def find_fsdp_reduce_grad_input(param_grad_output: Any) -> fx.Node | None:
    """Return the split point before an FSDP reduce-grad epilogue.

    The backward FSDP/DDP/HSDP tail is traced as a unary chain ending in the
    synced grad output:

        local_grad -> cast/view* -> reduce_scatter -> wait -> sharded_grad
        local_grad -> cast/view* -> all_reduce -> wait -> replicated_grad
        local_grad -> cast/view* -> all_reduce -> wait -> reduce_scatter
          -> wait -> grad

    GraphPP splits at the input to the earliest grad-sync collective in that
    suffix. The cast remains in ``bw_no_fsdp`` so microbatch accumulation
    happens in FSDP's reduce dtype, and ``reduce_grad`` contains only the
    scheduled collective epilogue. Values that are not FX nodes, such as
    ``None`` parameter-grad slots, are not collective outputs and are preserved
    by the caller.

    TODO(sanketpurandare): requires upstream change: FSDP trace/passes should
    annotate reduce-grad collective regions for downstream graph extraction.
    """
    if not isinstance(param_grad_output, fx.Node):
        return None

    node = param_grad_output
    reduce_grad_input = None
    while isinstance(node, fx.Node) and len(node.all_input_nodes) == 1:
        input_node = node.all_input_nodes[0]
        if len(input_node.users) > 1:
            break
        previous_node = node
        node = input_node
        if is_reduce_grad_collective(previous_node):
            reduce_grad_input = node
    return reduce_grad_input
