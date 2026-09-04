# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FSDP-specific compiler passes for graph_trainer.

These passes operate on graphs containing SimpleFSDP all-gather/reduce-scatter
collectives.  They are no-ops when the graph contains no FSDP collectives.
"""

from __future__ import annotations

import heapq
import operator
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch._inductor.fx_passes.bucketing import (
    _recompute_changed_user_metadata,
    BucketMode,
    enable_symmetric_memory_for_fsdp_buckets,
    FSDPSymmetricMemoryAllocation,
    is_all_gather_into_tensor as is_all_gather,
    is_wait_tensor,
)

try:
    from torch._inductor.fx_passes.overlap_manual_scheduling import _move_overlap_nodes
except ImportError:
    _move_overlap_nodes = None
from torch._inductor.fx_passes.overlap_manual_scheduling import (
    manual_overlap_bucketing,
    ManualOverlapPreservingBucketer,
    ManualOverlapScheduler,
)
from torch._inductor.fx_passes.overlap_scheduling import (
    is_compute_node,
    schedule_overlap_bucketing,
)
from torch.utils._ordered_set import OrderedSet

from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node,
    _MODULE_FQN,
    matches_module_fqn_pattern,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    _PACKED_FSDP_UNSHARD_PARAM,
    find_fsdp_unshard_outputs,
    find_fsdp_unshard_reconstruction_outputs,
    is_fsdp_all_gather_output_split,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    capture_graph_state_output_metadata,
    restore_graph_state_output_metadata,
)
from torchtitan.tools.logging import logger


_FSDP_BUCKET_META = "fsdp_bucket"


def configure_fsdp_symmetric_memory_backend() -> None:
    """Select NCCL before any GraphTrainer model runtime can allocate symm mem."""
    backend = symm_mem.get_backend(torch.device("cuda"))
    if backend == "NCCL":
        return
    try:
        symm_mem.set_backend("NCCL")
    except RuntimeError as error:
        raise RuntimeError(
            "GraphTrainer FSDP symmetric memory could not select the NCCL "
            f"backend before model setup; the current CUDA backend is {backend!r}. "
            "Symmetric-memory backends cannot change after the first allocation."
        ) from error


def _bucketed_fsdp_all_gather_widths(gm: fx.GraphModule) -> set[int]:
    widths: set[int] = set()
    target = torch.ops.bucketing._pre_bucket_all_gather.default
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != target:
            continue
        group_size = node.args[1]
        if type(group_size) is not int or group_size <= 0:
            raise ValueError(
                "FSDP symmetric memory requires positive static all-gather "
                f"group sizes, got {group_size!r}"
            )
        widths.add(group_size)
    return widths


def _selected_fsdp_symmetric_memory_widths(
    gm: fx.GraphModule,
    policy: str,
) -> set[int]:
    if policy not in {"all", "widest"}:
        raise ValueError(
            "fsdp_symm_mem_policy must be 'all' or 'widest', " f"got {policy!r}"
        )
    widths = _bucketed_fsdp_all_gather_widths(gm)
    if policy == "all" or not widths:
        return widths
    return {max(widths)}


def _resolve_symmetric_memory_group(
    group_name: str,
    group: object,
) -> dist.ProcessGroup:
    if isinstance(group, str):
        return dist.distributed_c10d._resolve_process_group(group_name)
    if isinstance(group, dist.ProcessGroup):
        return group
    raise TypeError(
        "FSDP symmetric-memory group must be a name or ProcessGroup, "
        f"got {type(group).__name__}"
    )


def _initialize_fsdp_symmetric_memory_groups(
    selected_groups: dict[str, object],
) -> None:
    resolved_groups = []
    for group_name, group in selected_groups.items():
        process_group = _resolve_symmetric_memory_group(group_name, group)
        backend = dist.get_backend(process_group)
        if backend != dist.Backend.NCCL:
            raise ValueError(
                "FSDP symmetric memory requires NCCL process groups, got "
                f"{backend!r} for {group_name!r}"
            )
        resolved_groups.append(process_group)

    if not resolved_groups:
        return
    backend = symm_mem.get_backend(torch.device("cuda"))
    if backend != "NCCL":
        raise RuntimeError(
            "GraphTrainer FSDP symmetric memory requires the NCCL backend to "
            f"be selected before model setup, got {backend!r}"
        )
    device_id = torch.cuda.current_device()
    for process_group in resolved_groups:
        dist.barrier(group=process_group, device_ids=[device_id])


def _rewrite_fsdp_symmetric_memory_graphs(
    graph_modules: Iterable[fx.GraphModule],
    group_sizes: set[int],
    *,
    preallocate: bool,
    max_reduce_scatter_input_buffers: int,
) -> tuple[
    int,
    int,
    dict[str, object],
    Counter[str],
    list[FSDPSymmetricMemoryAllocation],
]:
    num_all_gathers = 0
    num_reduce_scatters = 0
    selected_groups: dict[str, object] = {}
    planned_bytes_by_role: Counter[str] = Counter()
    preallocated_buffers: list[FSDPSymmetricMemoryAllocation] = []
    for gm in graph_modules:
        result = enable_symmetric_memory_for_fsdp_buckets(
            gm,
            group_sizes,
            preallocate=preallocate,
            max_reduce_scatter_input_buffers=max_reduce_scatter_input_buffers,
        )
        num_all_gathers += result.num_all_gathers
        num_reduce_scatters += result.num_reduce_scatters
        planned_bytes_by_role.update(dict(result.planned_bytes))
        preallocated_buffers.extend(result.preallocated_buffers)
        for group_name, group in result.selected_groups:
            selected_groups.setdefault(group_name, group)
    return (
        num_all_gathers,
        num_reduce_scatters,
        selected_groups,
        planned_bytes_by_role,
        preallocated_buffers,
    )


def _rendezvous_fsdp_symmetric_memory_buffers(
    allocations: Iterable[FSDPSymmetricMemoryAllocation],
) -> Counter[str]:
    allocated_bytes_by_role: Counter[str] = Counter()
    for allocation in allocations:
        process_group = _resolve_symmetric_memory_group(
            allocation.group_name,
            allocation.group,
        )
        symm_mem.rendezvous(allocation.tensor, group=process_group)
        allocated_bytes_by_role[allocation.role] += (
            allocation.tensor.numel() * allocation.tensor.element_size()
        )
    return allocated_bytes_by_role


def enable_fsdp_symmetric_memory_for_graphs(
    graph_modules: Iterable[fx.GraphModule],
    *,
    selection_graph: fx.GraphModule,
    policy: str,
    preallocate: bool = False,
    max_reduce_scatter_input_buffers: int = 1,
) -> tuple[int, int]:
    """Rewrite selected FSDP buckets and initialize their real process groups."""
    group_sizes = _selected_fsdp_symmetric_memory_widths(selection_graph, policy)
    if not group_sizes:
        warnings.warn(
            "FSDP symmetric memory found no eligible bucketed all-gathers; "
            "the FSDP degree may be one or the bucketing pass may be disabled",
            stacklevel=2,
        )
        return 0, 0
    if not dist.is_initialized():
        raise RuntimeError("FSDP symmetric memory requires distributed initialization")
    if dist.get_backend() == dist.Backend.FAKE:
        warnings.warn(
            "FSDP symmetric memory is disabled for the fake process-group backend",
            stacklevel=2,
        )
        return 0, 0

    (
        num_all_gathers,
        num_reduce_scatters,
        selected_groups,
        planned_bytes_by_role,
        preallocated_buffers,
    ) = _rewrite_fsdp_symmetric_memory_graphs(
        graph_modules,
        group_sizes,
        preallocate=preallocate,
        max_reduce_scatter_input_buffers=max_reduce_scatter_input_buffers,
    )
    if not selected_groups:
        warnings.warn(
            "FSDP symmetric memory found no concrete positive-size buckets "
            f"at the selected group sizes {sorted(group_sizes)}",
            stacklevel=2,
        )
        return num_all_gathers, num_reduce_scatters
    _initialize_fsdp_symmetric_memory_groups(selected_groups)
    preallocated_bytes_by_role = _rendezvous_fsdp_symmetric_memory_buffers(
        preallocated_buffers
    )
    logger.info(
        "Enabled FSDP symmetric memory for %d all-gather and %d "
        "reduce-scatter buckets at group sizes %s; planned bytes by role %s; "
        "preallocated bytes by role %s",
        num_all_gathers,
        num_reduce_scatters,
        sorted(group_sizes),
        dict(planned_bytes_by_role),
        dict(preallocated_bytes_by_role),
    )
    return num_all_gathers, num_reduce_scatters


def enable_fsdp_symmetric_memory_pass(
    gm: fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    policy: str,
    preallocate: bool,
) -> fx.GraphModule:
    """Apply FSDP symmetric memory after scheduling and before compilation."""
    del example_inputs
    enable_fsdp_symmetric_memory_for_graphs(
        (gm,),
        selection_graph=gm,
        policy=policy,
        preallocate=preallocate,
    )
    return gm


def _chain_nodes_to_placeholder(
    output: fx.Node,
    placeholder: fx.Node,
) -> set[fx.Node]:
    """Return graph nodes in one FSDP unshard chain, excluding the placeholder."""
    chain_nodes: set[fx.Node] = set()
    queue = [output]
    while queue:
        node = queue.pop()
        if node is placeholder or node in chain_nodes:
            continue
        chain_nodes.add(node)
        queue.extend(node.all_input_nodes)
    return chain_nodes


def _fsdp_chain_signature(value: Any, placeholder: fx.Node) -> Any:
    """Build a structural signature for one unshard/preparation result.

    Args:
        value: FX value to describe.
        placeholder: Parameter placeholder shared by candidate chains.

    Returns:
        A recursively comparable signature that ignores FX node identity while
        retaining operators, literal arguments, and other input identities.
    """
    memo: dict[fx.Node, Any] = {}

    def signature(item: Any) -> Any:
        if isinstance(item, fx.Node):
            if item is placeholder:
                return ("parameter",)
            if item in memo:
                return memo[item]
            if item.op == "placeholder":
                result = ("placeholder", item.name)
            else:
                result = (
                    item.op,
                    item.target,
                    signature(item.args),
                    signature(item.kwargs),
                )
            memo[item] = result
            return result
        if isinstance(item, tuple):
            return ("tuple", tuple(signature(element) for element in item))
        if isinstance(item, list):
            return ("list", tuple(signature(element) for element in item))
        if isinstance(item, dict):
            return (
                "dict",
                tuple((key, signature(element)) for key, element in item.items()),
            )
        try:
            hash(item)
        except TypeError:
            return ("literal", repr(item))
        return ("literal", item)

    return signature(value)


def _pack_distinct_unshard_outputs(
    placeholder: fx.Node,
    outputs: tuple[fx.Node, ...],
) -> None:
    """Pack heterogeneous preparation results into one parameter value.

    Args:
        placeholder: Flat FSDP parameter input shared by the outputs.
        outputs: Distinct post-all-gather preparation results.
    """
    region_nodes = set().union(
        *(_chain_nodes_to_placeholder(output, placeholder) for output in outputs)
    )
    external_users = {
        output: tuple(user for user in output.users if user not in region_nodes)
        for output in outputs
    }
    live_outputs = tuple(output for output in outputs if external_users[output])
    if len(live_outputs) <= 1:
        return

    positions = {node: index for index, node in enumerate(placeholder.graph.nodes)}
    last_output = max(live_outputs, key=positions.__getitem__)
    with placeholder.graph.inserting_after(last_output):
        packed = placeholder.graph.call_function(tuple, args=(live_outputs,))
    packed.meta[_PACKED_FSDP_UNSHARD_PARAM] = placeholder.name

    insertion_point = packed
    for index, output in enumerate(live_outputs):
        with placeholder.graph.inserting_after(insertion_point):
            unpacked = placeholder.graph.call_function(
                operator.getitem,
                args=(packed, index),
            )
        insertion_point = unpacked
        for user in external_users[output]:
            user.replace_input_with(output, unpacked)


def deduplicate_fsdp_unshard_chains_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Canonicalize duplicate SimpleFSDP unshard chains per flat parameter.

    A traced parametrized module can read the same FSDP parameter more than
    once. Equivalent preparation chains are deduplicated directly. Distinct
    preparations share the high-precision reconstruction and are packed into
    one per-parameter value so GraphPP can hoist every preparation without
    conflating their layouts.
    """
    del example_inputs

    removable_nodes: set[fx.Node] = set()
    num_duplicate_chains = 0
    for placeholder in gm.graph.find_nodes(op="placeholder"):
        unshard_outputs = find_fsdp_unshard_outputs(placeholder)
        if len(unshard_outputs) <= 1:
            continue
        preparation_groups: dict[Any, list[fx.Node]] = defaultdict(list)
        for output in unshard_outputs:
            preparation_groups[_fsdp_chain_signature(output, placeholder)].append(
                output
            )
        canonical_preparations: dict[fx.Node, fx.Node] = {}
        for group in preparation_groups.values():
            canonical_output = group[0]
            canonical_preparations[canonical_output] = canonical_output
            for duplicate_output in group[1:]:
                removable_nodes.update(
                    _chain_nodes_to_placeholder(duplicate_output, placeholder)
                )
                duplicate_output.replace_all_uses_with(canonical_output)
                canonical_preparations[duplicate_output] = canonical_output
                num_duplicate_chains += 1
        if len(preparation_groups) == 1:
            continue

        reconstruction_outputs = find_fsdp_unshard_reconstruction_outputs(placeholder)
        if len(reconstruction_outputs) != len(unshard_outputs):
            raise ValueError(
                "FSDP preparation and reconstruction chain counts differ for "
                f"{placeholder.name}: {len(unshard_outputs)} preparation "
                f"outputs but {len(reconstruction_outputs)} reconstructions"
            )
        canonical_reconstruction = reconstruction_outputs[0]
        replacements: dict[fx.Node, fx.Node] = {}
        for duplicate_output in reconstruction_outputs[1:]:
            removable_nodes.update(
                _chain_nodes_to_placeholder(duplicate_output, placeholder)
            )
            duplicate_output.replace_all_uses_with(canonical_reconstruction)
            replacements[duplicate_output] = canonical_reconstruction
            num_duplicate_chains += 1
        distinct_outputs = tuple(
            dict.fromkeys(
                replacements.get(
                    canonical_preparations[output], canonical_preparations[output]
                )
                for output in unshard_outputs
            )
        )
        _pack_distinct_unshard_outputs(placeholder, distinct_outputs)

    if num_duplicate_chains == 0:
        return gm

    # Heterogeneous forward/backward preparations are packed at one boundary,
    # so their pure producer chains must move before the earliest compute user.
    _stable_topological_sort(gm.graph, {})

    def _is_impure_for_fsdp_dedup(node: fx.Node) -> bool:
        if node in removable_nodes:
            return False
        return node.is_impure()

    gm.graph.eliminate_dead_code(is_impure_node=_is_impure_for_fsdp_dedup)
    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Canonicalized %d duplicate FSDP unshard chain(s)",
        num_duplicate_chains,
    )
    return gm


def preserve_fsdp_unshard_output_boundaries_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    num_model_state_tensor_inputs: int,
) -> torch.fx.GraphModule:
    """Preserve canonical post-all-gather outputs through FSDP bucketing.

    This pass must run after the final unshard deduplication and immediately
    before bucketing. Bucketing rewrites the all-gather reconstruction, so the
    structural matcher would otherwise fall back to the reconstructed
    high-precision parameter instead of a format-specific prepared value.
    """
    del example_inputs

    placeholders = gm.graph.find_nodes(op="placeholder")
    if not 0 <= num_model_state_tensor_inputs <= len(placeholders):
        raise ValueError(
            "num_model_state_tensor_inputs must be between zero and the graph's "
            f"{len(placeholders)} placeholders, got "
            f"{num_model_state_tensor_inputs}"
        )

    num_boundaries = 0
    for placeholder in placeholders[:num_model_state_tensor_inputs]:
        unshard_outputs = find_fsdp_unshard_outputs(placeholder)
        if not unshard_outputs:
            continue
        if len(unshard_outputs) != 1:
            raise ValueError(
                "FSDP unshard boundary preservation expects one canonical "
                f"output for {placeholder.name}, got {len(unshard_outputs)}. "
                "Run deduplicate_fsdp_unshard_chains_pass first."
            )
        unshard_outputs[0].meta[_PACKED_FSDP_UNSHARD_PARAM] = placeholder.name
        num_boundaries += 1

    logger.info("Preserved %d FSDP unshard output boundary(s)", num_boundaries)
    return gm


def is_wait_tensor_from_fsdp(node: torch.fx.Node) -> bool:
    """
    Returns True if the node is a wait_tensor node that is the result of an all_gather
    that can be arbitrarily prefetched, i.e., if all its recursive inputs are
    single-input operators that leads to a graph input.
    """
    if is_wait_tensor(node) and is_all_gather(node.args[0]):
        n: torch.fx.Node = node.all_input_nodes[0]
        while len(n.all_input_nodes) == 1:
            if n.all_input_nodes[0].op == "placeholder":
                return True
            n = n.all_input_nodes[0]
    return False


# Maps an FSDP group_name to an extra group_name created by this pass.
# Each NCCL PG gets its own CUDA stream, so the extra PG is what enables
# AG/RS overlap in backward.
_EXTRA_FSDP_PG_REGISTRY: dict[str, str] = {}


def _reorder_overlap_nodes(
    graph: fx.Graph,
    overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    bucketed_node_types: dict[fx.Node, str],
) -> None:
    if _move_overlap_nodes is None:
        # TODO(ivankobzarev): Remove this fallback once the PyTorch nightly wheel
        # includes _move_overlap_nodes.
        _stable_topological_sort(graph, overlap_deps)
    else:
        _move_overlap_nodes(graph, overlap_deps, bucketed_node_types)


def _get_or_create_extra_pg(
    source_pg_name: str,
    registry: dict[str, str],
    *,
    group_desc: str,
    high_priority: bool = False,
) -> str:
    import torch.distributed as dist

    if source_pg_name in registry:
        return registry[source_pg_name]

    source_pg = dist.distributed_c10d._resolve_process_group(source_pg_name)
    ranks = dist.get_process_group_ranks(source_pg)
    pg_options = (
        dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        if high_priority and hasattr(dist, "ProcessGroupNCCL")
        else None
    )
    extra_pg = dist.new_group(
        ranks=ranks,
        backend="nccl" if pg_options is not None else None,
        pg_options=pg_options,
        group_desc=group_desc,
        use_local_synchronization=True,
    )
    registry[source_pg_name] = extra_pg.group_name
    logger.info(
        f"Created extra {group_desc} PG (source: {source_pg_name}, "
        f"extra: {extra_pg.group_name}, high_priority={high_priority})"
    )
    return extra_pg.group_name


def _get_or_create_extra_fsdp_pg(source_pg_name: str) -> str:
    """Return an extra FSDP PG with the same ranks and a distinct NCCL stream."""
    return _get_or_create_extra_pg(
        source_pg_name,
        _EXTRA_FSDP_PG_REGISTRY,
        group_desc="fsdp_extra",
    )


def _is_backward_fsdp_all_gather_wait(node: fx.Node) -> bool:
    if not (
        is_wait_tensor(node)
        and isinstance(node.args[0], fx.Node)
        and is_all_gather(node.args[0])
    ):
        return False
    all_gather = node.args[0]
    bucket_meta = _read_fsdp_bucket_meta(all_gather) or _read_fsdp_bucket_meta(node)
    if bucket_meta is not None:
        _plan_fqns, direction = bucket_meta
        return direction == "bwd"
    if not is_wait_tensor_from_fsdp(node):
        return False
    return any(
        _is_backward_or_recomputed_node(candidate)
        for candidate in (all_gather, node, *node.users)
    )


def reassign_collective_pgs_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Reassign backward FSDP all-gathers to dedicated NCCL process groups.

    Each PG runs on its own CUDA stream, so moving backward all-gathers to an
    extra PG with the same ranks lets them overlap reduce-scatters left on the
    original PG. Forward all-gathers stay on the original PG because they do
    not participate in that overlap. No-op without targeted collectives. Run
    after bucketing, whose direction metadata distinguishes surviving forward
    and backward parameter all-gathers exactly.
    """
    source_pg_names: OrderedSet[str] = OrderedSet()
    backward_all_gathers: OrderedSet[fx.Node] = OrderedSet()
    for node in gm.graph.nodes:
        if _is_backward_fsdp_all_gather_wait(node):
            ag_node = node.args[0]
            source_pg_names.add(ag_node.args[2])
            backward_all_gathers.add(ag_node)

    if not source_pg_names:
        logger.info(
            "FSDP AG/RS overlap found no backward all-gathers; "
            "forward all-gathers remain on their source process groups"
        )
        return gm

    pg_mapping: dict[str, str] = {
        pg: _get_or_create_extra_fsdp_pg(pg) for pg in source_pg_names
    }
    for all_gather in backward_all_gathers:
        source_pg_name = all_gather.args[2]
        all_gather.args = (
            all_gather.args[0],
            all_gather.args[1],
            pg_mapping[source_pg_name],
        )
    for source, target in pg_mapping.items():
        logger.info(
            f"Rewrote backward all-gather node(s) from PG {source} to PG {target}"
        )
    gm.recompile()
    return gm


def autobucketing_reordering_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple | None = None
) -> torch.fx.GraphModule:
    """
    Apply autobucketing and reordering optimization.

    This pass applies schedule_overlap_bucketing with collective_bucketing enabled
    to optimize comm/compute overlap patterns in the graph.
    """
    schedule_overlap_bucketing(gm, collective_bucketing=True)
    gm.recompile()
    return gm


def transformer_block_bucketing_reordering_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    fsdp_manual_buckets,
) -> torch.fx.GraphModule:
    """
    Apply aten-level manual bucketing and reordering optimization.
    """
    manual_overlap_bucketing(
        gm, module_bucket_plans=fsdp_manual_buckets, insert_overlap_deps=False
    )
    gm.recompile()
    return gm


def get_fsdp_param_module_order(state_fqns: list[str]) -> dict[str, int]:
    """Return module order matching FSDP2's first-seen parameter order."""
    order: dict[str, int] = {}
    for fqn in state_fqns:
        if "." not in fqn:
            continue
        module_fqn = fqn.rsplit(".", 1)[0]
        order.setdefault(module_fqn, len(order))
    return order


class FSDPParamOrderBucketer(ManualOverlapPreservingBucketer):
    """Pack FSDP buckets in Eager FSDP2 parameter order."""

    def __init__(
        self,
        *args: Any,
        fsdp_param_module_order: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fsdp_param_module_order = fsdp_param_module_order or {}

    def _param_order_key(self, node: fx.Node) -> tuple[int, int]:
        module_fqn = node.meta.get("custom", {}).get(_MODULE_FQN)
        param_idx = self.fsdp_param_module_order.get(module_fqn, len(self.node_idx))
        return (param_idx, self.node_idx[node])

    def _bucket_group(self, coll_nodes: list[fx.Node]) -> None:
        if self.fsdp_param_module_order:
            coll_nodes = sorted(coll_nodes, key=self._param_order_key)
        return super()._bucket_group(coll_nodes)


class JointManualOverlapScheduler(ManualOverlapScheduler):
    """Manual overlap scheduler for joint forward+backward graphs.

    For the aot_fx_trace path we trace a joint forward+backward graph and
    want to bucket + reorder both directions in a single pass over the
    graph. This subclass of :class:`ManualOverlapScheduler` produces the
    same bucketing and prefetch pattern as invoking the upstream
    ``manual_overlap_bucketing`` twice (once per direction).

    Overrides :meth:`_manual_bucket_collectives` to split each module's
    collectives by direction before handing them to the bucketer.

    Overrides :meth:`_manual_reorder_graph` to track per-direction state
    so a single reversed walk emits correct AG prefetch edges for both
    forward and backward regions.

    The caller supplies:

    * ``module_stack_fn`` — must return a non-empty module stack for both
      forward and backward nodes belonging to ``module_bucket_plans``
      (i.e. do not filter by direction; that is this class's job).
    * ``is_backward_fn`` — returns ``True`` for nodes that should be
      treated as backward, including SAC-recomputed forward ops that are
      emitted into the backward section.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        module_bucket_plans: list[list[str] | str],
        insert_overlap_deps: bool,
        *,
        is_backward_fn: Callable[[fx.Node], bool],
        module_stack_fn: Callable[[fx.Node], list[tuple[str, type[Any]]]],
        bucket_mode: BucketMode | None = None,
        fsdp_param_module_order: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            gm,
            module_bucket_plans,
            insert_overlap_deps,
            module_stack_fn=module_stack_fn,
            bucket_mode=bucket_mode,
        )
        self._is_backward_fn = is_backward_fn
        effective_bucket_mode = self.bucketer.bucket_mode
        self.bucketer = FSDPParamOrderBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            scheduled=OrderedSet(self.graph.nodes),
            bucket_mode=effective_bucket_mode,
            fsdp_param_module_order=fsdp_param_module_order,
        )

    def _manual_bucket_collectives(self) -> None:
        """Bucket per module, splitting by direction to keep fwd/bwd buckets disjoint."""
        self._obtain_nodes_in_subgraph()
        for bucket_plan, nodes in zip(
            self.module_bucket_plans, self.nodes_in_subgraph, strict=True
        ):
            bucket_fqns = _bucket_plan_fqns(bucket_plan)
            fwd_nodes = [n for n in nodes if not self._is_backward_fn(n)]
            bwd_nodes = [n for n in nodes if self._is_backward_fn(n)]
            if fwd_nodes:
                pre_nodes = set(self.graph.nodes)
                self.bucketer.manual_bucket_collectives(nodes=fwd_nodes)
                self._annotate_new_bucket_nodes(pre_nodes, bucket_fqns, "fwd")
            if bwd_nodes:
                pre_nodes = set(self.graph.nodes)
                self.bucketer.manual_bucket_collectives(nodes=bwd_nodes)
                self._annotate_new_bucket_nodes(pre_nodes, bucket_fqns, "bwd")

        if _move_overlap_nodes is None:
            _stable_topological_sort(self.graph, {})

        self.graph.lint()
        self.nodes = list(self.graph.nodes)
        self.in_degree = Counter(user for node in self.nodes for user in node.users)

    def _annotate_new_bucket_nodes(
        self,
        pre_nodes: set[fx.Node],
        bucket_fqns: tuple[str, ...],
        direction: str,
    ) -> None:
        # The upstream bucketer preserves sample ``custom`` metadata for
        # readability, but bucket ownership is the bucketing plan itself. Store
        # that provenance once, at bucket creation time, so later scheduling
        # passes do not need to rediscover it by walking arbitrary users.
        for node in self.graph.nodes:
            if node in pre_nodes or node.op != "call_function":
                continue
            node.meta[_FSDP_BUCKET_META] = {
                "plan_fqns": bucket_fqns,
                "direction": direction,
            }
            if direction == "bwd":
                node.meta["autograd_backward"] = True

    def _manual_reorder_graph(self) -> None:
        """Reorder pass with separate fwd/bwd buffers so AG pairing never
        crosses the fwd/bwd boundary. RS pairing is unchanged — RSs only
        occur in backward and are already direction-scoped.
        """
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        self._schedule_rs_prefetch(overlap_deps)
        self._schedule_ag_prefetch(overlap_deps)

        total_deps = sum(len(v) for v in overlap_deps.values())
        logger.info(
            "FSDP reorder: %d overlap deps across %d target nodes",
            total_deps,
            len(overlap_deps),
        )
        _reorder_overlap_nodes(
            self.graph, overlap_deps, self.bucketer.bucketed_node_types
        )
        self.graph.lint()

        if self.insert_overlap_deps:
            from torch._inductor.fx_passes.control_dependencies import (
                preserve_node_ordering,
            )

            preserve_node_ordering(self.graph, overlap_deps)

    def _schedule_rs_prefetch(
        self,
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    ) -> None:
        """Top-down scheduling loop that emits RS prefetch edges.

        RSs only occur in backward, so no direction tracking is needed.
        Populates ``self.scheduled`` in topological order for the
        subsequent reversed walk.
        """
        delayed_rs_wait_nodes: list[fx.Node] = []
        current_rs_start_nodes: list[fx.Node] = []

        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.on_path_ready = []
        self.scheduled = OrderedSet()
        for node in self.nodes:
            if self.in_degree[node] == 0:
                self._add_to_ready_queue(node)

        while self.on_path_ready:
            _, node = heapq.heappop(self.on_path_ready)
            node_type = self.bucketer.bucketed_node_types.get(node, "")

            if node in self.scheduled:
                continue

            if node_type == "bucketed_reduce_scatter":
                current_rs_start_nodes.append(node)
            elif node_type == "bucketed_reduce_scatter_wait":
                if current_rs_start_nodes:
                    for delayed in delayed_rs_wait_nodes:
                        for rs_start in current_rs_start_nodes:
                            overlap_deps[delayed].add(rs_start)
                    delayed_rs_wait_nodes.clear()
                    current_rs_start_nodes.clear()
                delayed_rs_wait_nodes.append(node)

            self._schedule(node)

    def _schedule_ag_prefetch(
        self,
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    ) -> None:
        """Reversed walk that emits per-direction AG prefetch edges.

        Uses separate fwd/bwd buffers so AG pairing never crosses the
        fwd/bwd boundary. Consumes ``self.scheduled`` produced by
        :meth:`_emit_rs_prefetch`.
        """
        self.scheduled = OrderedSet(reversed(list(self.scheduled)))

        bwd_scope: OrderedSet[fx.Node] = OrderedSet()
        fwd_scope: OrderedSet[fx.Node] = OrderedSet()
        for sublist in self.nodes_in_subgraph:
            for n in sublist:
                if self._is_backward_fn(n):
                    bwd_scope.add(n)
                else:
                    fwd_scope.add(n)

        bwd_picked: list[fx.Node] = []
        fwd_picked: list[fx.Node] = []
        bwd_last_compute: fx.Node | None = None
        fwd_last_compute: fx.Node | None = None

        for node in self.scheduled:
            node_type = self.bucketer.bucketed_node_types.get(node, "")
            is_bwd = self._is_backward_fn(node)
            picked = bwd_picked if is_bwd else fwd_picked

            if node_type == "bucketed_all_gather":
                picked.append(node)
                continue

            if node_type == "bucketed_all_gather_wait":
                if picked:
                    for ag in picked:
                        overlap_deps[self.bucketer.node_to_wait_map[node]].add(ag)
                picked.clear()

            if is_compute_node(node):
                # Track per-direction last_compute so orphan bwd all-gathers
                # attach to a bwd-region compute and orphan fwd all-gathers
                # attach to a fwd-region compute.
                if is_bwd and node in bwd_scope:
                    bwd_last_compute = node
                elif not is_bwd and node in fwd_scope:
                    fwd_last_compute = node

        # Trailing block, applied once per direction. Attaches any orphan
        # AG starts (those whose wait was not matched during the reversed
        # walk of their direction) to the last compute in that direction,
        # unless they are already an ancestor of it.
        self._apply_trailing_block(bwd_picked, bwd_last_compute, overlap_deps)
        self._apply_trailing_block(fwd_picked, fwd_last_compute, overlap_deps)

    def _apply_trailing_block(
        self,
        picked: list[fx.Node],
        last_compute: fx.Node | None,
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    ) -> None:
        if last_compute is None or not picked:
            return
        ancestors = self.node_ancestors
        # TODO(ivankobzarev): remove OrderedSet fallback after nightly picks up BitsetAncestors
        if hasattr(ancestors, "is_ancestor"):
            blocked = any(ancestors.is_ancestor(ag, last_compute) for ag in picked)
        else:
            blocked = bool(OrderedSet(picked) & OrderedSet(ancestors[last_compute]))
        if blocked:
            return
        for ag in picked:
            overlap_deps[last_compute].add(ag)


def joint_transformer_block_bucketing_reordering_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    module_bucket_plans: list[list[str] | str],
    insert_overlap_deps: bool = False,
    bucket_mode: BucketMode | None = None,
    fsdp_param_module_order: dict[str, int] | None = None,
) -> torch.fx.GraphModule:
    """Run joint-graph manual bucketing and reordering.

    Joint-graph equivalent of
    ``torch._inductor.fx_passes.overlap_manual_scheduling.manual_overlap_bucketing``.
    Buckets forward all-gathers, backward all-gathers, and backward reduce-scatters
    of each module into separate buckets per transformer block and emits prefetching.

    Run ``reassign_collective_pgs_pass`` after this pass so its authoritative
    bucket-direction metadata selects only backward all-gathers for dedicated
    process groups and streams.

    Args:
        gm: joint forward+backward graph module.
        example_inputs: unused, required by the pass interface.
        module_bucket_plans: list of module FQNs (or lists of FQNs); each
            entry defines one bucketing scope whose collectives should be
            merged into a single bucket per direction per collective type.
        insert_overlap_deps: if ``True``, insert explicit control deps via
            ``preserve_node_ordering`` after the topological sort.
        bucket_mode: bucket mode forwarded to the underlying bucketer;
            defaults to ``"custom_ops"`` via the parent class.
        fsdp_param_module_order: module order derived from traced parameter
            FQNs, used to pack FSDP buckets like Eager FSDP2.
    """

    def _stack_fn(node: torch.fx.Node) -> list[tuple[str, type]]:
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN)
        if not fqn:
            return []
        return [(fqn, torch.nn.Module)]

    graph_state_output_metadata = capture_graph_state_output_metadata(gm)
    scheduler = JointManualOverlapScheduler(
        gm,
        module_bucket_plans,
        insert_overlap_deps,
        is_backward_fn=_is_backward_node,
        module_stack_fn=_stack_fn,
        bucket_mode=bucket_mode,
        fsdp_param_module_order=fsdp_param_module_order,
    )
    # TODO: Produce compact per-parameter all-gather outputs directly instead
    # of materializing bucket slices. Joint forward-backward CUDA graphs can
    # otherwise retain every compact weight copy until its backward use.
    overlapped_gm = scheduler.run()
    restore_graph_state_output_metadata(
        overlapped_gm,
        graph_state_output_metadata,
    )
    overlapped_gm.recompile()
    return overlapped_gm


# --- EP-aware FSDP dense-region scheduling ---


@dataclass(frozen=True)
class _FsdpComm:
    launch: fx.Node
    wait: fx.Node
    kind: str
    direction: str
    plan_fqns: tuple[str, ...]
    logical_index: int | None


def _bucket_plan_fqns(bucket: list[str] | str) -> tuple[str, ...]:
    return (bucket,) if isinstance(bucket, str) else tuple(bucket)


def _layer_id_from_fqn(fqn: str) -> int | None:
    """Extract layer index from ``layers.N...`` FQN prefix."""
    if not fqn.startswith("layers."):
        return None
    parts = fqn.split(".", 2)
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def get_transformer_block_bucket_counts(
    module_bucket_plans: list[list[str] | str],
    *,
    n_layers: int,
) -> dict[int, int]:
    """Count how many bucket-plan entries belong to each transformer block."""
    bucket_counts = {layer_id: 0 for layer_id in range(n_layers)}
    for bucket in module_bucket_plans:
        fqns = [bucket] if isinstance(bucket, str) else bucket
        layer_ids = {
            layer_id
            for fqn in fqns
            if (layer_id := _layer_id_from_fqn(fqn)) is not None
        }
        if not layer_ids:
            continue
        if len(layer_ids) != 1:
            raise ValueError(
                "Transformer block bucket plans must not span multiple layers, "
                f"got bucket {bucket!r} with layer ids {sorted(layer_ids)}."
            )
        layer_id = next(iter(layer_ids))
        if layer_id not in bucket_counts:
            raise ValueError(
                f"Transformer block bucket plan references layer {layer_id}, "
                f"but n_layers={n_layers}."
            )
        bucket_counts[layer_id] += 1
    missing = [layer_id for layer_id, count in bucket_counts.items() if count == 0]
    if missing:
        raise ValueError(
            "Transformer block bucket plan is missing layers "
            f"{missing} for n_layers={n_layers}."
        )
    return bucket_counts


def _is_moe_layer_dense_fqn(fqn: str) -> bool:
    """Return whether an MoE-layer FQN belongs to the dense attention region."""
    parts = fqn.split(".")
    if len(parts) < 3:
        return False
    return parts[2] in {"attention", "attention_norm"}


def _is_top_level_param_fqn(fqn: str) -> bool:
    return fqn in {"norm", "lm_head"} or fqn.startswith(("norm.", "lm_head."))


def _fsdp_bucket_logical_index(
    plan_fqns: tuple[str, ...],
    *,
    n_layers: int,
) -> int | None:
    """Map bucket provenance to the logical model slot it owns.

    ``0..N-1`` are transformer blocks. ``N`` is the post-transformer
    norm/lm_head bucket. Embedding buckets return ``None`` because the dense
    scheduler intentionally leaves the bottom edge to the upstream bucketer.
    """
    layer_ids = {
        layer_id
        for fqn in plan_fqns
        if (layer_id := _layer_id_from_fqn(fqn)) is not None
    }
    if len(layer_ids) > 1:
        raise ValueError(
            "FSDP dense-region scheduling cannot place a bucket spanning "
            f"multiple transformer blocks: {plan_fqns!r}"
        )
    if layer_ids:
        layer_id = next(iter(layer_ids))
        if layer_id >= n_layers:
            raise ValueError(
                f"FSDP bucket references layer {layer_id}, but n_layers={n_layers}."
            )
        return layer_id
    if any(_is_top_level_param_fqn(fqn) for fqn in plan_fqns):
        return n_layers
    return None


def _is_dense_region_target_node(node: fx.Node) -> bool:
    """Return whether ``node`` is useful dense compute, not FSDP plumbing.

    Dense-region scheduling uses these nodes as insertion anchors.  FSDP waits
    and parameter-unpack view/copy nodes are part of getting parameters ready;
    choosing them as anchors hoists future AG/RS launches out of the actual
    compute window and can make them interfere with MoE token exchange.
    """
    if node.op != "call_function":
        return False
    if is_wait_tensor(node) or is_all_gather(node):
        return False
    if node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default:
        return False
    namespace = getattr(node.target, "namespace", None)
    if namespace in {"_c10d_functional", "c10d_functional", "c10d"}:
        return False
    if namespace == "bucketing":
        return False

    non_compute_targets = {
        operator.getitem,
        torch.ops.aten._to_copy.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.split_with_sizes_copy.default,
        torch.ops.aten.sym_size.int,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.view.default,
        torch.ops.aten.view.dtype,
    }
    if node.target in non_compute_targets:
        return False

    dense_compute_targets = {
        torch.ops.aten._fused_rms_norm.default,
        torch.ops.aten._fused_rms_norm_backward.default,
        torch.ops.aten.relu.default,
    }
    return is_compute_node(node) or node.target in dense_compute_targets


def _is_backward_or_recomputed_node(node: fx.Node) -> bool:
    """Return whether ``node`` executes in the backward section."""
    return _is_backward_node(node) or "_recomputed" in node.name


def _build_layer_dense_regions(
    gm: torch.fx.GraphModule,
    n_layers: int,
    moe_layer_ids: frozenset[int],
) -> dict[int, dict[str, list[fx.Node]]]:
    """Build per-layer dense-region node lists for forward and backward.

    Dense region = the attention portion of a transformer block. For MoE
    layers, generic ``layers.N`` boundary nodes, ffn_norm, and moe nodes are
    deliberately excluded so FSDP launches do not overlap with MoE token
    exchange. For dense-only layers (no MoE), the entire block is dense.
    """
    regions: dict[int, dict[str, list[fx.Node]]] = {
        i: {"fwd_dense": [], "bwd_dense": []} for i in range(n_layers)
    }
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN)
        if not fqn:
            continue
        layer_id = _layer_id_from_fqn(fqn)
        if layer_id is None or layer_id not in regions:
            continue
        if layer_id in moe_layer_ids and not _is_moe_layer_dense_fqn(fqn):
            continue
        if not _is_dense_region_target_node(node):
            continue
        key = "bwd_dense" if _is_backward_node(node) else "fwd_dense"
        regions[layer_id][key].append(node)
    return regions


def _build_top_dense_regions(gm: torch.fx.GraphModule) -> dict[str, list[fx.Node]]:
    """Build dense compute regions for norm/lm_head outside transformer blocks."""
    regions: dict[str, list[fx.Node]] = {"fwd_dense": [], "bwd_dense": []}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN)
        if not fqn or not _is_top_level_param_fqn(fqn):
            continue
        if not _is_dense_region_target_node(node):
            continue
        key = "bwd_dense" if _is_backward_or_recomputed_node(node) else "fwd_dense"
        regions[key].append(node)
    return regions


def _read_fsdp_bucket_meta(node: fx.Node) -> tuple[tuple[str, ...], str] | None:
    meta = node.meta.get(_FSDP_BUCKET_META)
    if not isinstance(meta, dict):
        return None
    plan_fqns = meta.get("plan_fqns")
    direction = meta.get("direction")
    if not isinstance(plan_fqns, tuple) or direction not in {"fwd", "bwd"}:
        return None
    return plan_fqns, direction


_FSDP_PARAMETER_RECONSTRUCTION_OPS = {
    operator.getitem,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.alias.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.view.default,
    torch.ops.aten.view.dtype,
}


def _bucket_outputs_are_compact_at_use(split_outputs: list[fx.Node]) -> bool:
    saw_terminal_use = False
    work = list(split_outputs)
    visited: set[fx.Node] = set()
    while work:
        node = work.pop()
        if node in visited:
            continue
        visited.add(node)

        reconstruction_users = []
        has_terminal_use = False
        for user in node.users:
            if (
                user.op == "call_function"
                and user.target in _FSDP_PARAMETER_RECONSTRUCTION_OPS
            ):
                reconstruction_users.append(user)
            else:
                has_terminal_use = True

        if has_terminal_use:
            saw_terminal_use = True
            value = node.meta.get("val")
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"FSDP parameter reconstruction node {node.name} is "
                    "missing tensor metadata"
                )
            if not value.is_contiguous():
                return False
        work.extend(reconstruction_users)

    if not saw_terminal_use:
        names = ", ".join(output.name for output in split_outputs)
        raise RuntimeError(f"FSDP bucket outputs have no terminal use: {names}")
    return True


def materialize_fsdp_bucket_outputs_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    module_fqn_patterns: tuple[str, ...],
) -> torch.fx.GraphModule:
    """Materialize selected non-compact FSDP all-gather bucket outputs."""
    del example_inputs
    if not module_fqn_patterns:
        return gm

    num_materialized = 0
    for node in gm.graph.nodes:
        if node.target not in {
            torch.ops.aten.split_with_sizes.default,
            torch.ops.aten.split_with_sizes_copy.default,
        }:
            continue
        bucket_meta = _read_fsdp_bucket_meta(node)
        if bucket_meta is None:
            continue
        plan_fqns, _direction = bucket_meta
        if not any(
            matches_module_fqn_pattern(pattern, fqn)
            for pattern in module_fqn_patterns
            for fqn in plan_fqns
        ):
            continue
        if not is_fsdp_all_gather_output_split(node):
            continue

        split_outputs = [
            user
            for user in node.users
            if user.op == "call_function" and user.target == operator.getitem
        ]
        if len(split_outputs) != len(node.users):
            raise RuntimeError(f"Selected FSDP split {node.name} has non-getitem users")
        output_values = [output.meta.get("val") for output in split_outputs]
        if not output_values or not all(
            isinstance(value, torch.Tensor) for value in output_values
        ):
            raise RuntimeError(
                f"Selected FSDP split {node.name} is missing tensor metadata"
            )
        if _bucket_outputs_are_compact_at_use(split_outputs):
            continue
        if node.target == torch.ops.aten.split_with_sizes_copy.default:
            raise RuntimeError(
                f"Copy split {node.name} still has non-compact output metadata"
            )

        node.target = torch.ops.aten.split_with_sizes_copy.default
        _recompute_changed_user_metadata([node])
        if not _bucket_outputs_are_compact_at_use(split_outputs):
            raise RuntimeError(
                f"Materialized FSDP split {node.name} still reaches a "
                "non-compact parameter use"
            )
        num_materialized += 1

    if num_materialized:
        gm.graph.lint()
        gm.recompile()
        logger.info(
            "Materialized %d compact FSDP all-gather bucket split(s)",
            num_materialized,
        )
    return gm


def _collect_bucketed_fsdp_comms(
    gm: torch.fx.GraphModule,
    *,
    n_layers: int,
) -> tuple[dict[int, dict[str, list[tuple[fx.Node, fx.Node]]]], list[_FsdpComm]]:
    """Collect bucketed FSDP AG and RS (launch, wait) pairs per layer and direction.

    Returns ``({layer_id: {"fwd_ag": [...], "bwd_ag": [...], "bwd_rs": [...]}},
    all_comms)``. The per-layer view is kept for validation and the historical
    transformer-block placement rules; ``all_comms`` retains non-transformer
    bucket provenance so edge buckets can be placed without user-walking
    heuristics.
    """
    all_comms: list[_FsdpComm] = []
    order = {node: i for i, node in enumerate(gm.graph.nodes)}
    first_backward_pos = min(
        (order[node] for node in gm.graph.nodes if _is_backward_node(node)),
        default=None,
    )

    def _infer_fqn_and_direction(
        launch: fx.Node, wait: fx.Node
    ) -> tuple[str | None, bool]:
        def _nearest_from(starts: Iterable[fx.Node]) -> tuple[str | None, bool]:
            frontier = list(starts)
            visited: set[fx.Node] = set()
            while frontier:
                next_frontier: list[fx.Node] = []
                candidates: list[tuple[str, bool]] = []
                for n in frontier:
                    if n in visited or n.op == "output":
                        continue
                    visited.add(n)
                    candidate = n.meta.get("custom", {}).get(_MODULE_FQN)
                    if candidate:
                        candidates.append(
                            (candidate, _is_backward_or_recomputed_node(n))
                        )
                    next_frontier.extend(n.users)
                if candidates:
                    fqn = candidates[0][0]
                    return fqn, any(is_bwd for _, is_bwd in candidates)
                frontier = next_frontier
            return None, False

        # The wait use site is the semantic owner of a bucketed AG.  Do not
        # classify direction by arbitrarily walking to transitive backward
        # users; a forward parameter use naturally feeds backward later.
        fqn, is_bwd = _nearest_from(wait.users)
        if fqn is not None:
            if first_backward_pos is not None and order[wait] >= first_backward_pos:
                is_bwd = True
            return fqn, is_bwd
        fqn = wait.meta.get("custom", {}).get(_MODULE_FQN)
        if fqn is not None:
            is_bwd = _is_backward_or_recomputed_node(wait)
            if first_backward_pos is not None and order[wait] >= first_backward_pos:
                is_bwd = True
            return fqn, is_bwd
        fqn = launch.meta.get("custom", {}).get(_MODULE_FQN)
        if fqn is not None:
            is_bwd = _is_backward_or_recomputed_node(launch)
            if first_backward_pos is not None and order[wait] >= first_backward_pos:
                is_bwd = True
            return fqn, is_bwd
        return None, False

    comms: dict[int, dict[str, list[tuple[fx.Node, fx.Node]]]] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        is_ag = is_all_gather(node)
        is_rs = node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
        if not is_ag and not is_rs:
            continue
        wait_nodes = [user for user in node.users if is_wait_tensor(user)]
        if not wait_nodes:
            continue
        wait = wait_nodes[0]
        bucket_meta = _read_fsdp_bucket_meta(node) or _read_fsdp_bucket_meta(wait)
        if bucket_meta is not None:
            plan_fqns, direction = bucket_meta
        else:
            # Fallback for tests and legacy graphs. Bucketed launches created
            # by this stack should carry _FSDP_BUCKET_META.
            fqn, is_bwd = _infer_fqn_and_direction(node, wait)
            if not fqn:
                continue
            plan_fqns = (fqn,)
            direction = "bwd" if is_bwd else "fwd"

        logical_index = _fsdp_bucket_logical_index(plan_fqns, n_layers=n_layers)
        kind = "ag" if is_ag else "rs"
        all_comms.append(
            _FsdpComm(
                launch=node,
                wait=wait,
                kind=kind,
                direction=direction,
                plan_fqns=plan_fqns,
                logical_index=logical_index,
            )
        )

        if logical_index is None or logical_index >= n_layers:
            continue
        layer_id = logical_index
        comms.setdefault(layer_id, {"fwd_ag": [], "bwd_ag": [], "bwd_rs": []})
        if kind == "ag" and direction == "fwd":
            comms[layer_id]["fwd_ag"].append((node, wait))
        elif kind == "ag" and direction == "bwd":
            comms[layer_id]["bwd_ag"].append((node, wait))
        elif kind == "rs" and direction == "bwd":
            comms[layer_id]["bwd_rs"].append((node, wait))
    logger.info(
        "Collected FSDP comms for dense-region scheduling: fwd_ag=%d bwd_ag=%d bwd_rs=%d",
        sum(len(layer_comms["fwd_ag"]) for layer_comms in comms.values()),
        sum(len(layer_comms["bwd_ag"]) for layer_comms in comms.values()),
        sum(len(layer_comms["bwd_rs"]) for layer_comms in comms.values()),
    )
    return comms, all_comms


def _has_waited_fsdp_comm_launches(gm: torch.fx.GraphModule) -> bool:
    """Return whether the graph contains any waited FSDP AG/RS launch."""
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if not (
            is_all_gather(node)
            or node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            continue
        if any(is_wait_tensor(user) for user in node.users):
            return True
    return False


def _validate_transformer_block_bucket_counts(
    comms: dict[int, dict[str, list[tuple[fx.Node, fx.Node]]]],
    *,
    n_layers: int,
    expected_bucket_counts: dict[int, int],
) -> None:
    """Validate required buckets while allowing saved forward unshards.

    Forward all-gathers and backward reduce-scatters are required for every
    planned bucket. A backward all-gather exists only when activation remat
    needs to reconstruct that bucket, so its count may range from zero to the
    planned count independently for each layer.
    """
    missing_expected = [
        layer_id
        for layer_id in range(n_layers)
        if layer_id not in expected_bucket_counts
    ]
    extra_expected = [
        layer_id
        for layer_id in expected_bucket_counts
        if layer_id not in range(n_layers)
    ]
    errors: list[str] = []
    if missing_expected:
        errors.append(f"missing expected counts for layers {missing_expected}")
    if extra_expected:
        errors.append(f"unexpected expected-count layers {sorted(extra_expected)}")

    empty_layer_comms: dict[str, list[tuple[fx.Node, fx.Node]]] = {
        "fwd_ag": [],
        "bwd_ag": [],
        "bwd_rs": [],
    }
    for layer_id in range(n_layers):
        expected = expected_bucket_counts.get(layer_id)
        if expected is None:
            continue
        layer_comms = comms.get(layer_id, empty_layer_comms)
        actual = {
            "fwd_ag": len(layer_comms["fwd_ag"]),
            "bwd_ag": len(layer_comms["bwd_ag"]),
            "bwd_rs": len(layer_comms["bwd_rs"]),
        }
        required_kinds = ("fwd_ag", "bwd_rs")
        mismatches = {
            kind: actual[kind] for kind in required_kinds if actual[kind] != expected
        }
        if actual["bwd_ag"] > expected:
            mismatches["bwd_ag"] = actual["bwd_ag"]
        if mismatches:
            errors.append(
                f"layer {layer_id}: expected {expected} forward all-gather and "
                f"backward reduce-scatter buckets, and at most {expected} "
                "backward all-gather buckets; "
                f"got fwd_ag={actual['fwd_ag']} bwd_ag={actual['bwd_ag']} "
                f"bwd_rs={actual['bwd_rs']}"
            )

    if errors:
        raise ValueError(
            "FSDP dense-region scheduling requires complete transformer-block "
            "forward all-gather and backward reduce-scatter buckets. Backward "
            "all-gather buckets may be absent when their forward values are saved:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def _call_mutates_node(call: fx.Node, value: fx.Node) -> bool:
    target = call.target
    if call.op != "call_function" or not isinstance(target, torch._ops.OpOverload):
        return False
    for index, schema_arg in enumerate(target._schema.arguments):
        if schema_arg.alias_info is None or not schema_arg.alias_info.is_write:
            continue
        if index < len(call.args):
            argument = call.args[index]
        elif schema_arg.name in call.kwargs:
            argument = call.kwargs[schema_arg.name]
        else:
            continue
        leaves, _ = pytree.tree_flatten(argument)
        if any(leaf is value for leaf in leaves):
            return True
    return False


def _find_mutable_alias_use(
    value: fx.Node,
    *,
    order: dict[fx.Node, int],
    start_pos: int,
    end_pos: int,
    ignored_calls: set[fx.Node],
) -> fx.Node | None:
    aliases = [value]
    visited: set[fx.Node] = set()
    while aliases:
        alias = aliases.pop()
        if alias in visited:
            continue
        visited.add(alias)
        for user in alias.users:
            if (
                user not in ignored_calls
                and start_pos <= order[user] < end_pos
                and _call_mutates_node(user, alias)
            ):
                return user
            target = user.target
            if user.op == "call_function" and (
                target is operator.getitem
                or target is torch.ops.aten._unsafe_view.default
                or (isinstance(target, torch._ops.OpOverload) and target.is_view)
            ):
                aliases.append(user)
    return None


def _plan_static_all_gather_hoist(
    launch: fx.Node,
    target: fx.Node,
    wait: fx.Node,
    *,
    order: dict[fx.Node, int],
    model_state_placeholders: set[fx.Node],
    infra_chain: set[fx.Node],
) -> tuple[list[fx.Node] | None, str | None]:
    if order[wait] <= order[target]:
        return None, f"wait {wait.name} does not follow target {target.name}"

    target_pos = order[target]
    ancestors: set[fx.Node] = {launch}
    roots: set[fx.Node] = set()
    getattrs: set[fx.Node] = set()
    launch_data = (launch.args[0], launch.kwargs.get("out"))
    data_leaves, _ = pytree.tree_flatten(launch_data)
    work = [leaf for leaf in data_leaves if isinstance(leaf, fx.Node)]
    while work:
        node = work.pop()
        if node in ancestors:
            continue
        if node.op == "placeholder":
            roots.add(node)
            continue
        if node.op == "get_attr":
            getattrs.add(node)
            continue
        ancestors.add(node)
        work.extend(node.all_input_nodes)

    late_non_data_inputs = [
        node
        for node in launch.all_input_nodes
        if node not in ancestors and order[node] >= target_pos
    ]
    if late_non_data_inputs:
        names = ", ".join(sorted(node.name for node in late_non_data_inputs))
        return None, f"all-gather launch has late non-tensor inputs: {names}"
    late_getattrs = [node for node in getattrs if order[node] >= target_pos]
    if late_getattrs:
        names = ", ".join(sorted(node.name for node in late_getattrs))
        return None, f"all-gather preparation has late get_attr inputs: {names}"

    launch_pos = order[launch]
    interval_start = min(launch_pos, target_pos)
    interval_end = max(launch_pos, target_pos)
    mutated_values = [
        (node, mutation)
        for node in ancestors | roots | getattrs
        if (
            mutation := _find_mutable_alias_use(
                node,
                order=order,
                start_pos=interval_start,
                end_pos=interval_end,
                ignored_calls={launch},
            )
        )
        is not None
    ]
    if mutated_values:
        details = ", ".join(
            f"{value.name} by {mutation.name}"
            for value, mutation in sorted(
                mutated_values, key=lambda pair: order[pair[1]]
            )
        )
        return None, f"all-gather preparation crosses mutations: {details}"

    to_hoist = [node for node in ancestors if order[node] >= target_pos]
    external_prep = [node for node in to_hoist if node not in infra_chain]
    if external_prep:
        invalid_roots = roots - model_state_placeholders
        if invalid_roots:
            names = ", ".join(sorted(node.name for node in invalid_roots))
            return None, f"all-gather preparation depends on non-model inputs: {names}"
    unsafe = [node for node in to_hoist if node is not launch and node.is_impure()]
    if unsafe:
        names = ", ".join(sorted(node.name for node in unsafe))
        return None, f"all-gather preparation has effectful nodes: {names}"
    if target in ancestors:
        return None, f"target {target.name} is an all-gather dependency"
    return sorted(to_hoist, key=order.__getitem__), None


def schedule_fsdp_comms_to_dense_regions_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    moe_layer_ids: frozenset[int],
    n_layers: int,
    num_model_state_tensor_inputs: int = 0,
    transformer_bucket_counts_by_layer: dict[int, int] | None = None,
    strict: bool = False,
) -> torch.fx.GraphModule:
    """Schedule bucketed FSDP comms to overlap with dense (attention) regions.

    Physically moves AG/RS launch nodes in the FX graph so that Inductor
    emits them at the desired positions. Inductor's
    ``decide_global_ordering_of_comms`` chains collectives in FX graph
    order, and the stable topological sort preserves FX order as a
    tiebreaker, so FX graph order IS the execution order for comms.

    Forward (execution order i-1 -> i):
      AG(i) launch at start of dense_fwd(i-1), for i > 0.
      AG(i) wait stays before layer i's first param use (unchanged).
      AG(0) is left to the original bucketing schedule.

    Backward (execution order i+1 -> i -> i-1):
      AG(i) launch at start of dense_bwd(i+1), for i < N-1.
      AG(i) wait stays before layer i's first param use (unchanged).
      AG(N-1) launch at start of top-level backward dense compute when present.
      RS(i) launch in dense_bwd(i-1), for i > 0, after its gradient input is ready.
      RS(0) is left to the original bucketing schedule.
      RS waits and pure output-unpack users are sunk to the graph tail.

    Embedding and standalone loss buckets are left to the upstream bucketer. Their
    reduce-scatter waits are sunk only when their users are output-only. The
    top-level norm/lm_head bucket is treated as the edge after transformer block
    N-1.
    """
    del example_inputs
    regions = _build_layer_dense_regions(gm, n_layers, moe_layer_ids)
    top_regions = _build_top_dense_regions(gm)
    has_fsdp_comms = _has_waited_fsdp_comm_launches(gm)
    comms, all_comms = _collect_bucketed_fsdp_comms(gm, n_layers=n_layers)

    if not has_fsdp_comms:
        return gm

    if transformer_bucket_counts_by_layer is not None:
        _validate_transformer_block_bucket_counts(
            comms,
            n_layers=n_layers,
            expected_bucket_counts=transformer_bucket_counts_by_layer,
        )

    order = {node: i for i, node in enumerate(gm.graph.nodes)}
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if not 0 <= num_model_state_tensor_inputs <= len(placeholders):
        raise ValueError(
            "num_model_state_tensor_inputs must be between zero and the graph's "
            f"{len(placeholders)} placeholders, got {num_model_state_tensor_inputs}"
        )
    model_state_placeholders = set(placeholders[:num_model_state_tensor_inputs])

    _FSDP_INFRA_OPS = {
        torch.ops.bucketing._pre_bucket_all_gather.default,
        torch.ops.bucketing._pre_bucket_reduce_scatter.default,
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        torch.ops._c10d_functional.all_gather_into_tensor_out.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops.aten.constant_pad_nd.default,
        torch.ops.aten.slice.Tensor,
    }

    _FSDP_WAIT_OUTPUT_OPS = {
        operator.getitem,
        torch.ops.aten.alias.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.split_with_sizes_copy.default,
        torch.ops.aten.view.default,
        torch.ops.aten.view.dtype,
    }

    wait_closures_sunk = 0

    def _is_allowed_wait_tail_node(
        node: fx.Node,
        closure: set[fx.Node],
        wait: fx.Node,
    ) -> tuple[bool, str | None]:
        if node.target in _FSDP_WAIT_OUTPUT_OPS:
            return True, None
        if node.target is not torch.ops.aten.add_.Tensor:
            return False, f"{wait.name} has non-output user {node.name} ({node.target})"

        mutated_arg = node.args[0] if node.args else None
        if not isinstance(mutated_arg, fx.Node):
            return False, f"{node.name} has non-node mutated input"

        # Chunked loss can accumulate multiple reduce-scattered grad shards via
        # an in-place add chain before returning the detached final shard. That
        # chain is safe to sink only when nothing else reads the accumulator.
        external_users = [
            user
            for user in mutated_arg.users
            if user is not node and user not in closure and user.op != "output"
        ]
        if external_users:
            users = ", ".join(user.name for user in external_users[:3])
            return (
                False,
                f"{node.name} mutates {mutated_arg.name}, which has "
                f"non-tail users: {users}",
            )
        return True, None

    def _collect_launch_chain(launch: fx.Node) -> list[fx.Node]:
        """Collect the FSDP infrastructure chain rooted at ``launch``."""
        chain: list[fx.Node] = []
        work = [launch]
        visited: set[fx.Node] = set()
        while work:
            node = work.pop()
            if node in visited or node.op in ("placeholder", "get_attr"):
                continue
            if node.target not in _FSDP_INFRA_OPS:
                continue
            visited.add(node)
            chain.append(node)
            work.extend(node.all_input_nodes)
        return chain

    def _find_dense_target(
        launch: fx.Node, dense_region: list[fx.Node]
    ) -> fx.Node | None:
        """Pick the first dense node after the launch chain's external deps."""
        chain = set(_collect_launch_chain(launch))
        if not chain:
            return None
        min_pos = 0
        for node in chain:
            for inp in node.all_input_nodes:
                if inp not in chain:
                    min_pos = max(min_pos, order.get(inp, -1) + 1)
        for target in dense_region:
            if order[target] >= min_pos:
                return target
        return None

    def _sink_output_only_wait_closure(wait: fx.Node) -> tuple[bool, str | None]:
        """Sink an RS wait and tail-only output/grad-accum users."""
        closure: set[fx.Node] = set()
        work = [wait]
        while work:
            n = work.pop()
            if n in closure:
                continue
            if n is not wait:
                allowed, reason = _is_allowed_wait_tail_node(n, closure, wait)
                if not allowed:
                    return False, reason
            closure.add(n)
            for user in n.users:
                if user.op == "output" or user in closure:
                    continue
                work.append(user)
        output = next(node for node in gm.graph.nodes if node.op == "output")
        for n in sorted(closure, key=lambda node: order.get(node, 0)):
            output.prepend(n)
        return True, None

    def _move_chain_before(launch: fx.Node, target: fx.Node) -> tuple[bool, list[str]]:
        """Move an AG/RS launch and its FSDP infrastructure before ``target``."""
        target_pos = order[target]
        chain = _collect_launch_chain(launch)
        movable = set(chain)
        reasons: list[str] = []
        changed = True
        while changed:
            changed = False
            for node in list(movable):
                for inp in node.all_input_nodes:
                    if inp not in movable and order.get(inp, -1) >= target_pos:
                        reasons.append(
                            f"{node.name} is pinned before {target.name}: "
                            f"input {inp.name} occurs at {order.get(inp, -1)}, "
                            f"target at {target_pos}"
                        )
                        movable.discard(node)
                        changed = True
                        break
                if node not in movable or order.get(node, -1) >= target_pos:
                    continue
                for user in node.users:
                    if user not in movable and order.get(user, -1) < target_pos:
                        reasons.append(
                            f"{node.name} is pinned before {target.name}: "
                            f"user {user.name} occurs at {order.get(user, -1)}, "
                            f"target at {target_pos}"
                        )
                        movable.discard(node)
                        changed = True
                        break

        to_move = sorted(movable, key=order.__getitem__)
        if not to_move:
            return False, reasons
        for node in to_move:
            target.prepend(node)
        return True, []

    moved = 0
    moves: list[tuple[fx.Node, fx.Node, fx.Node, str, list[fx.Node]]] = []
    blockers: list[str] = []

    def _add_move(
        launch: fx.Node,
        wait: fx.Node,
        dense_region: list[fx.Node],
        description: str,
        *,
        allow_static_ag_hoist: bool,
    ) -> None:
        if not dense_region:
            blockers.append(f"{description}: no dense target region")
            return
        if allow_static_ag_hoist:
            infra_chain = set(_collect_launch_chain(launch))
            for candidate in dense_region:
                to_hoist, reason = _plan_static_all_gather_hoist(
                    launch,
                    candidate,
                    wait,
                    order=order,
                    model_state_placeholders=model_state_placeholders,
                    infra_chain=infra_chain,
                )
                if to_hoist is not None:
                    moves.append((launch, wait, candidate, description, to_hoist))
                    return
        else:
            target = _find_dense_target(launch, dense_region)
            if target is not None:
                moves.append((launch, wait, target, description, []))
                return
            reason = "launch inputs are after dense region"
        blockers.append(
            f"{description}: {reason or 'no safe all-gather preparation hoist'}"
        )

    for layer_id, layer_comms in sorted(comms.items()):
        for ag_launch, _ag_wait in layer_comms["fwd_ag"]:
            prev = layer_id - 1
            if prev >= 0 and regions[prev]["fwd_dense"]:
                _add_move(
                    ag_launch,
                    _ag_wait,
                    regions[prev]["fwd_dense"],
                    f"fwd AG layer {layer_id} -> dense_fwd layer {prev}",
                    allow_static_ag_hoist=True,
                )
        for ag_launch, _ag_wait in layer_comms["bwd_ag"]:
            next_layer = layer_id + 1
            if next_layer < n_layers and regions[next_layer]["bwd_dense"]:
                _add_move(
                    ag_launch,
                    _ag_wait,
                    regions[next_layer]["bwd_dense"],
                    f"bwd AG layer {layer_id} -> dense_bwd layer {next_layer}",
                    allow_static_ag_hoist=True,
                )
        for rs_launch, _rs_wait in layer_comms["bwd_rs"]:
            prev = layer_id - 1
            if prev >= 0 and regions[prev]["bwd_dense"]:
                _add_move(
                    rs_launch,
                    _rs_wait,
                    regions[prev]["bwd_dense"],
                    f"bwd RS layer {layer_id} -> dense_bwd layer {prev}",
                    allow_static_ag_hoist=False,
                )

    for comm in all_comms:
        if comm.logical_index != n_layers:
            continue
        if comm.kind == "ag" and comm.direction == "fwd" and n_layers > 0:
            _add_move(
                comm.launch,
                comm.wait,
                regions[n_layers - 1]["fwd_dense"],
                "fwd AG top-level bucket -> dense_fwd last transformer block",
                allow_static_ag_hoist=True,
            )
        elif comm.kind == "rs" and comm.direction == "bwd" and n_layers > 0:
            _add_move(
                comm.launch,
                comm.wait,
                regions[n_layers - 1]["bwd_dense"],
                "bwd RS top-level bucket -> dense_bwd last transformer block",
                allow_static_ag_hoist=False,
            )

    if n_layers > 0 and top_regions["bwd_dense"]:
        for ag_launch, _ag_wait in comms.get(n_layers - 1, {}).get("bwd_ag", []):
            _add_move(
                ag_launch,
                _ag_wait,
                top_regions["bwd_dense"],
                "bwd AG last transformer block -> top-level backward dense",
                allow_static_ag_hoist=True,
            )

    if blockers and strict:
        raise ValueError(
            "Could not schedule all interior FSDP comm launches to dense regions:\n"
            + "\n".join(f"- {blocker}" for blocker in blockers)
        )

    for comm in all_comms:
        if comm.kind != "rs":
            continue
        moved_wait, reason = _sink_output_only_wait_closure(comm.wait)
        wait_closures_sunk += int(moved_wait)
        if not moved_wait and strict and comm.logical_index is not None:
            blockers.append(
                f"RS wait sink for {comm.plan_fqns!r} {comm.wait.name}: {reason}"
            )

    if blockers and strict:
        raise ValueError(
            "Could not finish FSDP dense-region scheduling:\n"
            + "\n".join(f"- {blocker}" for blocker in blockers)
        )

    failed_moves: list[str] = []
    for launch, _wait, target, description, to_hoist in moves:
        order = {node: i for i, node in enumerate(gm.graph.nodes)}
        if to_hoist:
            target_pos = order[target]
            pending = [node for node in to_hoist if order[node] >= target_pos]
            for node in sorted(pending, key=order.__getitem__):
                target.prepend(node)
            moved += 1
            continue
        moved_chain, reasons = _move_chain_before(launch, target)
        if moved_chain:
            moved += 1
        elif strict:
            reason = (
                "; ".join(reasons[:3]) if reasons else "no movable launch chain nodes"
            )
            failed_moves.append(f"{description}: {reason}")

    if failed_moves:
        raise ValueError(
            "Could not move FSDP comm launches to selected dense targets:\n"
            + "\n".join(f"- {move}" for move in failed_moves)
        )

    if moved or wait_closures_sunk:
        gm.graph.lint()
        gm.recompile()
    logger.info("schedule_fsdp_comms_to_dense_regions: moved %d comm chains", moved)
    return gm
