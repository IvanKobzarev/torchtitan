# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inductor compilation passes for GraphTrainer.

``regional_inductor_pass`` compiles only explicitly tagged regions.
``full_inductor_compilation_pass`` uses ``standalone_compile`` once for the
whole graph, without regional discovery and extraction overhead.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass
from typing import Literal

import torch
from torch.fx.passes.regional_inductor import (
    _disable_remat_for_regional_subcompile,
    _dummy_wrapper,
    regional_inductor,
)

from torchtitan.experiments.graph_trainer.common_utils import (
    set_graph_module_boxed_codegen,
)


logger = logging.getLogger(__name__)


def _ops_filter_with_distributed(name: str) -> bool:
    """Ops filter that allows distributed collective ops for serialization.

    The default GraphPickler ops filter only allows aten and fbgemm ops.
    SimpleFSDP uses _c10d_functional collectives that must also be
    allowed for the graph to serialize correctly.  The device_mesh ops
    (e.g. _get_submesh) appear in the backward graph when DTensor
    reconstructs submeshes from tracked ancestor meshes.
    """
    return name.startswith(
        (
            "torch.ops.aten",
            "torch.ops.fbgemm",
            "torch.ops._c10d_functional",
            "torch.ops._dtensor",
            "torch.ops.device_mesh",
            "torch.ops.bucketing",
        )
    )


def _node_metadata_key_filter_distributed(key: str) -> bool:
    """Metadata key filter for regional_inductor with distributed ops.

    Distributed ops (e.g. _get_submesh, mesh_get_process_group) produce
    opaque values (DeviceMesh, ProcessGroup) in node.meta["val"] and
    node.meta["eager_input_vals"] that cannot be pickled.  We strip
    both — they are not needed at runtime.
    """
    if key in ("val", "eager_input_vals"):
        return False
    return key not in ["source_fn_stack", "nn_module_stack", "fwd_source_fn_stack"]


def _get_fake_mode_from_gm(gm: torch.fx.GraphModule):
    from torch._subclasses.fake_tensor import FakeTensor

    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            val = node.meta["val"]
            if isinstance(val, FakeTensor):
                return val.fake_mode
    return None


def _wrap_compiled_artifact_as_graph_module(
    root: torch.fx.GraphModule,
    compiled_fn,
    used_placeholder_indices: list[int],
    output_spec: torch.utils._pytree.TreeSpec,
    output_leaf_meta: list[dict],
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    placeholders = []
    output_meta = None
    for node in root.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(graph.node_copy(node))
        elif node.op == "output":
            output_meta = node.meta.copy()

    call_args = tuple(placeholders[i] for i in used_placeholder_indices)
    call = graph.call_function(_dummy_wrapper(compiled_fn), args=call_args)
    if output_meta:
        call.meta = output_meta

    flat_outputs = []
    for index, leaf_meta in enumerate(output_leaf_meta):
        leaf = graph.call_function(operator.getitem, args=(call, index))
        leaf.meta = leaf_meta
        flat_outputs.append(leaf)

    output_iterator = iter(flat_outputs)

    def reconstruct_output(spec):
        if spec.is_leaf():
            return next(output_iterator)
        children = [reconstruct_output(child) for child in spec.children()]
        if torch.utils._pytree.is_namedtuple_class(spec.context):
            return graph.call_function(spec.context, args=tuple(children))
        if torch.utils._pytree.is_structseq_class(spec.type):
            return graph.call_function(spec.type, args=(tuple(children),))
        return torch.utils._pytree.SUPPORTED_NODES[spec.type].unflatten_fn(
            children, spec.context
        )

    graph.output(reconstruct_output(output_spec))
    graph.lint()

    wrapped = torch.fx.GraphModule(root, graph)
    wrapped.meta.update(root.meta)
    return wrapped


@dataclass(frozen=True)
class _StandaloneCompileGraph:
    gm: torch.fx.GraphModule
    example_inputs: tuple
    used_placeholder_indices: list[int]
    output_spec: torch.utils._pytree.TreeSpec
    output_leaf_meta: list[dict]


def _copy_graph_for_standalone_compile(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> _StandaloneCompileGraph:
    original_placeholders = list(gm.graph.find_nodes(op="placeholder"))
    used_placeholder_indices = [
        i for i, node in enumerate(original_placeholders) if len(node.users) > 0
    ]
    used_placeholders = set(original_placeholders[i] for i in used_placeholder_indices)

    graph = torch.fx.Graph()
    env = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node not in used_placeholders:
                continue
            env[node] = graph.node_copy(node)
        else:
            env[node] = graph.node_copy(node, lambda n: env[n])
    graph.lint()

    copied_gm = torch.fx.GraphModule(gm, graph)
    copied_gm.meta.update(gm.meta)
    output_node = copied_gm.graph.output_node()

    flat_outputs = []
    output_leaf_meta = []

    def output_leaf(value):
        flat_outputs.append(value)
        output_leaf_meta.append(
            value.meta.copy() if isinstance(value, torch.fx.Node) else {}
        )
        return object()

    def extract_path(value, example_value, path):
        for key in path:
            if isinstance(
                key,
                (
                    torch.utils._pytree.SequenceKey,
                    torch.utils._pytree.MappingKey,
                ),
            ):
                index = key.idx if hasattr(key, "idx") else key.key
                target = operator.getitem
            elif isinstance(key, torch.utils._pytree.GetAttrKey):
                index = key.name
                target = getattr
            else:
                raise ValueError(
                    "standalone Inductor output normalization does not support "
                    f"pytree key {key!r}"
                )
            with copied_gm.graph.inserting_before(output_node):
                value = copied_gm.graph.call_function(target, args=(value, index))
            example_value = key.get(example_value)
            value.meta["val"] = example_value
        return value

    def flatten_output(value):
        if isinstance(value, torch.fx.Node):
            if "val" in value.meta:
                example_value = value.meta["val"]
                path_leaves, example_spec = torch.utils._pytree.tree_flatten_with_path(
                    example_value
                )
                if not example_spec.is_leaf():
                    structure_leaves = [
                        output_leaf(extract_path(value, example_value, path))
                        for path, _ in path_leaves
                    ]
                    return torch.utils._pytree.tree_unflatten(
                        structure_leaves, example_spec
                    )
            elif (
                isinstance(value.target, torch._ops.OpOverload)
                and len(value.target._schema.returns) > 1
            ):
                raise ValueError(
                    f"multi-output node {value.name!r} requires val metadata before "
                    "standalone Inductor compilation"
                )
            return output_leaf(value)

        children, value_spec = torch.utils._pytree.tree_flatten(value)
        if value_spec.is_leaf():
            return output_leaf(value)
        return torch.utils._pytree.tree_unflatten(
            [flatten_output(child) for child in children], value_spec
        )

    output_structure = flatten_output(output_node.args[0])
    _, output_spec = torch.utils._pytree.tree_flatten(output_structure)
    output_node.args = (tuple(flat_outputs),)
    copied_gm.graph.lint()
    copied_gm.recompile()
    used_example_inputs = tuple(example_inputs[i] for i in used_placeholder_indices)
    return _StandaloneCompileGraph(
        gm=copied_gm,
        example_inputs=used_example_inputs,
        used_placeholder_indices=used_placeholder_indices,
        output_spec=output_spec,
        output_leaf_meta=output_leaf_meta,
    )


def standalone_inductor_compilation_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    artifact_mode: Literal["runtime", "serializable"],
    boxed_codegen: bool = False,
    inductor_configs=None,
) -> torch.fx.GraphModule:
    """Compile a joint GraphTrainer graph with one Inductor invocation.

    GraphTrainer has already traced and scheduled the joint forward/backward
    graph. Runtime compilation therefore needs a directly callable cache
    artifact, while offline precompilation needs a serializable bundled AOT
    artifact.

    Args:
        gm: The graph module to compile.
        example_inputs: Example inputs corresponding to the graph placeholders.
        artifact_mode: ``"runtime"`` returns an online callable;
            ``"serializable"`` produces an artifact for precompile save/load.
        boxed_codegen: Whether the returned graph uses boxed calling convention.
        inductor_configs: Optional Inductor config overrides.
    """
    import torch._inductor.config as ic

    if artifact_mode not in ("runtime", "serializable"):
        raise ValueError(
            "artifact_mode must be 'runtime' or 'serializable', "
            f"got {artifact_mode!r}"
        )

    fake_mode = _get_fake_mode_from_gm(gm)
    tracing_ctx = torch._guards.TracingContext(fake_mode)
    full_inductor_configs = {
        "reorder_for_peak_memory": True,
        **dict(inductor_configs or {}),
    }

    compile_graph = _copy_graph_for_standalone_compile(gm, example_inputs)

    with (
        torch._guards.tracing(tracing_ctx),
        ic.patch(full_inductor_configs),
        _disable_remat_for_regional_subcompile(),
        torch.no_grad(),
    ):
        compiled_fn = torch._inductor.standalone_compile(
            compile_graph.gm,
            compile_graph.example_inputs,
            dynamic_shapes="from_tracing_context",
            aot=artifact_mode == "serializable",
            donate_graph_module=False,
        )

    return set_graph_module_boxed_codegen(
        _wrap_compiled_artifact_as_graph_module(
            gm,
            compiled_fn,
            compile_graph.used_placeholder_indices,
            compile_graph.output_spec,
            compile_graph.output_leaf_meta,
        ),
        boxed=boxed_codegen,
    )


def regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    serializable: bool = False,
    boxed_codegen: bool = False,
) -> torch.fx.GraphModule:
    """Compile tagged graph regions with ``regional_inductor``.

    Scans the graph for nodes whose ``node.meta["custom"]`` contains a
    ``compile_with_inductor`` key and compiles those regions with
    TorchInductor.  Nodes without this tag are left unchanged.  If no
    nodes are tagged the pass is a no-op.

    Inductor is configured for bitwise-equal numerics so that the
    compiled regions match eager execution exactly.

    Args:
        gm: The graph module to compile.
        example_inputs: Example inputs for shape propagation.
        serializable: When True (precompile mode), sets
            ``force_autograd_cache`` so that ``regional_inductor`` wraps
            its output in ``RegionalOutputCode``, and overrides the ops
            filter to allow distributed collective ops.
        boxed_codegen: When True, compile the returned FX graph with boxed
            calling convention so its mutable runtime arg list is cleared after
            placeholder extraction.
    """
    import torch._inductor.config as ic

    if serializable and boxed_codegen:
        raise ValueError(
            "regional_inductor_pass cannot use boxed_codegen with "
            "serializable=True because precompile returns RegionalOutputCode, "
            "not a normal FX GraphModule callable."
        )

    # Recommended by inductor team -- enable as needed:
    # ic.emulate_precision_casts = True
    # ic.eager_numerics.disable_ftz = True
    # ic.eager_numerics.use_pytorch_libdevice = True
    # ic.fallback_random = True

    # regional_inductor calls standalone_compile with
    # dynamic_shapes="from_tracing_context", which requires an active
    # TracingContext with a FakeTensorMode.  When this pass is called
    # outside torch.compile (e.g. after make_fx tracing in graph_trainer),
    # no TracingContext exists, so we create one from the graph's fake
    # tensor metadata.
    fake_mode = _get_fake_mode_from_gm(gm)
    tracing_ctx = torch._guards.TracingContext(fake_mode)

    if serializable:
        with (
            torch._guards.tracing(tracing_ctx),
            ic.patch({"eager_numerics.division_rounding": True}),
            torch._functorch.config.patch("force_autograd_cache", True),
        ):
            result = regional_inductor(gm, example_inputs)
        from torch._inductor.output_code import RegionalOutputCode

        # Override the ops filter after compilation so that
        # serialization (which happens later) allows distributed
        # collective ops like _c10d_functional through GraphPickler.
        if isinstance(result, RegionalOutputCode):
            result._ops_filter = _ops_filter_with_distributed
            result._node_metadata_key_filter = _node_metadata_key_filter_distributed
        else:
            logger.warning(
                "regional_inductor with serializable=True did not produce "
                "RegionalOutputCode; distributed ops may not serialize correctly."
            )
        return result

    with (
        torch._guards.tracing(tracing_ctx),
        ic.patch({"eager_numerics.division_rounding": True}),
    ):
        gm = regional_inductor(gm, example_inputs)

    return set_graph_module_boxed_codegen(gm, boxed=boxed_codegen)


def annotate_flex_attention_for_regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    flex_compile_config: dict | None,
    mask_compile_config: dict | None = None,
) -> torch.fx.GraphModule:
    """Tag flex attention HOPs with compile_with_inductor for regional_inductor.

    Annotates three sets of nodes so that regional_inductor correctly
    scoops and compiles flex attention regions:
    1. The HOP node itself (flex_attention / flex_attention_backward)
    2. The get_attr nodes referencing score_mod / mask_mod submodules.
    3. All nodes inside those submodule graphs.

    Args:
        gm: The graph module to annotate.
        example_inputs: Example inputs (unused, required by pass interface).
        flex_compile_config: Inductor config dict for flex attention HOP
            nodes and their get_attr submodule references. When provided,
            wrapped as ``{"inductor_configs": flex_compile_config}``.
            When None, nodes are tagged with an empty annotation.
        mask_compile_config: Inductor config dict for nodes inside mask_mod
            subgraphs. When provided, wrapped as
            ``{"inductor_configs": mask_compile_config}``.
            When None, nodes are tagged with an empty annotation.
    """
    flex_compile_annotation: dict = (
        {"inductor_configs": flex_compile_config}
        if flex_compile_config is not None
        else {}
    )
    mask_compile_annotation: dict = (
        {"inductor_configs": mask_compile_config}
        if mask_compile_config is not None
        else {}
    )

    for node in gm.graph.nodes:
        if node.target not in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            continue
        node.meta.setdefault("custom", {})[
            "compile_with_inductor"
        ] = flex_compile_annotation
        for inp in node.all_input_nodes:
            if inp.op != "get_attr":
                continue
            submod = getattr(gm, inp.target, None)
            if not isinstance(submod, torch.fx.GraphModule):
                continue
            inp.meta.setdefault("custom", {})[
                "compile_with_inductor"
            ] = flex_compile_annotation

            # Following are the nodes in mask_mod subgraph
            for sub_node in submod.graph.nodes:
                sub_node.meta.setdefault("custom", {})[
                    "compile_with_inductor"
                ] = mask_compile_annotation
    return gm


def _migrate_cpu_get_attrs_to_cuda(gm: torch.fx.GraphModule) -> None:
    """Move CPU constant tensor get_attrs to CUDA so CUDA graph capture works."""
    from torch.fx.graph_module import _assign_attr, _get_attr

    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.find_nodes(op="get_attr"):
            attr = _get_attr(module, node.target)
            if isinstance(attr, torch.Tensor) and attr.device.type == "cpu":
                _assign_attr(attr.cuda(), module, node.target)


def full_inductor_compilation_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    boxed_codegen: bool = False,
    inductor_configs=None,
) -> torch.fx.GraphModule:
    """Compile the whole graph with one direct ``standalone_compile`` call.

    The graph already represents the intended GraphTrainer compilation unit,
    so routing it through regional discovery would copy and extract the same
    full graph before invoking ``standalone_compile``. The direct path keeps
    the same compiler entry point and configuration without that redundant
    transformation.

    The collapse hides CUDA-graph-incompatible ops (unpinned D2H copies,
    sm<10 ``_grouped_mm``) inside the opaque ``standalone_compile_inner``
    node, so the later :func:`is_cuda_graph_compatible` scan can't see
    them. Snapshot the verdict on the pre-collapse gm and stash it on
    the result so the downstream scan can honor it.

    Must be the **terminal** pass — no FX-graph-level passes (e.g.
    ``custom_codegen_pass``, ``insert_kernel_annotations_pass``) can
    run after this because the FX graph is no longer authoritative.

    Args:
        gm: The graph module to compile.
        example_inputs: Example inputs for shape propagation.
        boxed_codegen: When True, the returned FX graph uses boxed calling
            convention and clears its mutable runtime arg list.
        inductor_configs: Optional Inductor config overrides for the full
            compiled region.
    """
    from torchtitan.experiments.graph_trainer.cuda_graph import is_cuda_graph_compatible

    pre_collapse_cuda_graph_compatible = is_cuda_graph_compatible(
        gm, skip_flex_attention_check=True
    )

    _migrate_cpu_get_attrs_to_cuda(gm)
    result = standalone_inductor_compilation_pass(
        gm,
        example_inputs,
        artifact_mode="runtime",
        boxed_codegen=boxed_codegen,
        inductor_configs={
            "eager_numerics.division_rounding": True,
            **dict(inductor_configs or {}),
        },
    )

    # Carry the pre-collapse CUDA graph verdict forward via gm.meta. The
    # collapse is information-destroying; this is how downstream passes
    # know whether the artifact contains hidden CUDA-graph-incompatible ops.
    result.meta["cuda_graph_compatible"] = pre_collapse_cuda_graph_compatible
    return result
