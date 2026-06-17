# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inductor compilation passes for graph_trainer.

Regional and full Inductor compilation, plus FlexAttention annotation for
regional_inductor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, CustomSchedulerPass
from torch.fx.passes.regional_inductor import regional_inductor

from torchtitan.tools.logging import logger


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


def regional_inductor_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple, *, serializable: bool = False
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
    """
    import torch._inductor.config as ic
    from torch._subclasses.fake_tensor import FakeTensor

    def _get_fake_mode_from_gm(gm: torch.fx.GraphModule):
        """Extract the FakeTensorMode from a graph module's placeholder metadata."""
        for node in gm.graph.nodes:
            if node.op == "placeholder" and "val" in node.meta:
                val = node.meta["val"]
                if isinstance(val, FakeTensor):
                    return val.fake_mode
        return None

    # Ensure inductor produces bitwise-equal numerics vs eager.
    ic.eager_numerics.division_rounding = True
    # Recommended by inductor team — uncomment as needed:
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

    with torch._guards.tracing(tracing_ctx):
        gm = regional_inductor(gm, example_inputs)

    # regional_inductor may switch to boxed calling convention; reset to
    # default so the graph can be called with positional args as usual.
    gm.graph.set_codegen(torch.fx.graph.CodeGen())
    gm.recompile()
    return gm


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
        for inp in list(node.all_input_nodes):
            if inp.op != "get_attr":
                continue
            submod = getattr(gm, inp.target, None)
            if not isinstance(submod, torch.fx.GraphModule):
                continue
            with gm.graph.inserting_before(node):
                cloned_inp = gm.graph.get_attr(inp.target)
            cloned_inp.meta = dict(inp.meta)
            node.replace_input_with(inp, cloned_inp)
            cloned_inp.meta.setdefault("custom", {})[
                "compile_with_inductor"
            ] = flex_compile_annotation

            # Following are the nodes in mask_mod subgraph
            for sub_node in submod.graph.nodes:
                sub_node.meta.setdefault("custom", {})[
                    "compile_with_inductor"
                ] = mask_compile_annotation
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def _migrate_cpu_get_attrs_to_cuda(gm: torch.fx.GraphModule) -> None:
    """Move CPU constant tensor get_attrs to CUDA so cudagraph capture works."""
    from torch.fx.graph_module import _assign_attr, _get_attr

    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.find_nodes(op="get_attr"):
            attr = _get_attr(module, node.target)
            if isinstance(attr, torch.Tensor) and attr.device.type == "cpu":
                _assign_attr(attr.cuda(), module, node.target)


class _FullInductorFuseRegionPass(CustomGraphPass):
    def __init__(self, strategy: str) -> None:
        if strategy != "blocks_contiguous":
            raise ValueError(
                f"Unknown full-Inductor fuse-region strategy: {strategy!r}"
            )
        self.strategy = strategy

    def uuid(self) -> object:
        return ("graph_trainer_full_inductor_fuse_region", self.strategy, 6)

    def __call__(self, graph: torch.fx.Graph) -> None:
        from torch._inductor.fx_passes.control_dependencies import mark_fuse_region
        from torch._logging import trace_structured

        groups: list[tuple[str, list[torch.fx.Node]]] = []

        skipped = not self._is_outer_flex_graph(graph)
        if not skipped:
            current_key: str | None = None
            current_nodes: list[torch.fx.Node] = []

            def flush() -> None:
                nonlocal current_key, current_nodes
                if current_key is not None and len(current_nodes) > 1:
                    groups.append((current_key, current_nodes))
                current_key = None
                current_nodes = []

            for node in list(graph.nodes):
                key = self._region_key(node)
                if key is None:
                    flush()
                    continue
                if key != current_key:
                    flush()
                    current_key = key
                current_nodes.append(node)
            flush()

            for idx, (key, nodes) in enumerate(groups):
                if key.startswith("txt_unemb_chunk_"):
                    mark_fuse_region(graph, nodes, fuse_region_id=f"{key}_island_{idx}")
                else:
                    mark_fuse_region(graph, nodes)

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "graph_trainer_full_inductor_fuse_regions",
                "encoding": "string",
            },
            payload_fn=lambda: "\n".join(
                [
                    f"strategy={self.strategy}",
                    "unique_islands=True",
                    f"skipped={skipped}",
                    f"marked_groups={len(groups)}",
                    *[
                        f"{idx}\t{key}\t{len(nodes)}\t{nodes[0].name}\t{nodes[-1].name}"
                        for idx, (key, nodes) in enumerate(groups)
                    ],
                ]
            )
            + "\n",
        )

    @staticmethod
    def _is_outer_flex_graph(graph: torch.fx.Graph) -> bool:
        return any(
            node.op == "call_function"
            and node.target
            in {
                torch.ops.higher_order.flex_attention,
                torch.ops.higher_order.flex_attention_backward,
            }
            for node in graph.nodes
        )

    def _custom_region_key(self, node: torch.fx.Node) -> str | None:
        if node.op in ("placeholder", "output", "get_attr"):
            return None
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            return None
        if self._has_graph_module_arg(node):
            return None
        custom = node.meta.get("custom")
        if not isinstance(custom, dict):
            return None
        custom_region = custom.get("full_inductor_fuse_region")
        if isinstance(custom_region, str):
            return custom_region
        return None

    def _region_key(self, node: torch.fx.Node) -> str | None:
        if node.op in ("placeholder", "output", "get_attr"):
            return None
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            return None
        if self._has_graph_module_arg(node):
            return None
        custom_region = self._custom_region_key(node)
        if custom_region is not None:
            return custom_region
        custom = node.meta.get("custom")
        if not isinstance(custom, dict):
            return None
        module_fqn = custom.get("module_fqn")
        if not isinstance(module_fqn, str):
            return None
        parts = module_fqn.split(".")
        if module_fqn.startswith("blocks."):
            if len(parts) < 2 or not parts[1].isdigit():
                return None
            return f"blocks.{parts[1]}"
        return None

    @staticmethod
    def _has_graph_module_arg(node: torch.fx.Node) -> bool:
        gm = node.graph.owning_module
        if gm is None:
            return False
        return any(
            inp.op == "get_attr"
            and isinstance(inp.target, str)
            and isinstance(getattr(gm, inp.target, None), torch.fx.GraphModule)
            for inp in node.all_input_nodes
        )


class _FullInductorFuseRegionOrderPass(CustomSchedulerPass):
    def uuid(self) -> object:
        return ("graph_trainer_full_inductor_fuse_region_order", 4)

    def __call__(self, nodes: list) -> list:
        from torch._inductor.dependencies import WeakDep
        from torch._inductor.fx_passes.control_dependencies import FUSE_REGION
        from torch._logging import trace_structured

        chunk_nodes: dict[int, list] = {}
        for node in nodes:
            region = self._chunk_region(node, FUSE_REGION)
            match = re.fullmatch(r"txt_unemb_chunk_(\d+)", region or "")
            if match is None:
                continue
            chunk_nodes.setdefault(int(match.group(1)), []).append(node)

        chunk_buffers = {
            chunk: self._buffer_names(chunk_nodes[chunk]) for chunk in chunk_nodes
        }
        future_chunk_buffers: dict[int, set[str]] = {}
        future_buffers: set[str] = set()
        for chunk in sorted(chunk_nodes, reverse=True):
            future_chunk_buffers[chunk] = set(future_buffers)
            future_buffers.update(chunk_buffers[chunk])

        drain_nodes = {
            chunk: self._drain_nodes(
                nodes,
                chunk_nodes[chunk],
                chunk_buffers[chunk],
                future_chunk_buffers[chunk],
                FUSE_REGION,
            )
            for chunk in chunk_nodes
        }
        drain_buffers = {
            chunk: self._first_buffer_names(drain_nodes[chunk]) or chunk_buffers[chunk]
            for chunk in chunk_nodes
        }

        deps_added = 0
        prev_order_buffers: list[str] = []
        for chunk in sorted(chunk_nodes):
            current = chunk_nodes[chunk]
            if prev_order_buffers:
                for node in current:
                    mutating_buf = self._first_buffer_name(node)
                    if mutating_buf is None:
                        continue
                    for dep_name in prev_order_buffers:
                        dep = WeakDep(dep_name, mutating_buf=mutating_buf, is_fake=True)
                        self._add_fake_dep(node, dep)
                        deps_added += 1
            prev_order_buffers = drain_buffers[chunk]

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "graph_trainer_full_inductor_fuse_region_order",
                "encoding": "string",
            },
            payload_fn=lambda: "\n".join(
                [
                    f"chunks={sorted(chunk_nodes)}",
                    f"deps_added={deps_added}",
                    *[
                        f"{chunk}\t{len(chunk_nodes[chunk])}\t"
                        f"buffers={len(chunk_buffers[chunk])}\t"
                        f"drain_nodes={len(drain_nodes[chunk])}\t"
                        f"drain_buffers={len(drain_buffers[chunk])}\t"
                        f"{chunk_nodes[chunk][0].get_name()}\t"
                        f"{chunk_nodes[chunk][-1].get_name()}"
                        for chunk in sorted(chunk_nodes)
                    ],
                ]
            )
            + "\n",
        )
        return nodes

    @classmethod
    def _chunk_region(cls, node, annotation_key: str) -> str | None:
        region = cls._annotation(node, annotation_key)
        if region is None:
            return None
        match = re.match(r"(txt_unemb_chunk_\d+)_island_\d+$", region)
        if match is None:
            return None
        return match.group(1)

    @staticmethod
    def _annotation(node, annotation_key: str) -> str | None:
        region: str | None = None
        seen_region = False
        for snode in node.get_nodes():
            op = snode.node
            if op is None or not hasattr(op, "annotations"):
                continue
            op_region = op.annotations.get(annotation_key)
            if op_region is not None and not isinstance(op_region, str):
                raise AssertionError(
                    f"expected {annotation_key} to be str, got {op_region}"
                )
            if seen_region and region != op_region:
                raise AssertionError(
                    f"expected one {annotation_key} per scheduler node, "
                    f"got {region} and {op_region}"
                )
            region = op_region
            seen_region = True
        return region

    @staticmethod
    def _first_buffer_name(node) -> str | None:
        for name in node.get_buffer_names():
            return name
        return None

    @staticmethod
    def _add_fake_dep(node, dep) -> None:
        try:
            node.add_fake_dep(dep)
        except NotImplementedError:
            node.set_read_writes(node.read_writes.with_read(dep))
            node.unmet_dependencies.add(dep)

    @classmethod
    def _buffer_names(cls, nodes: list) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            for name in node.get_buffer_names():
                if name not in seen:
                    names.append(name)
                    seen.add(name)
        return names

    @classmethod
    def _first_buffer_names(cls, nodes: list) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            name = cls._first_buffer_name(node)
            if name is not None and name not in seen:
                names.append(name)
                seen.add(name)
        return names

    @staticmethod
    def _read_names(node) -> set[str]:
        return {dep.name for dep in node.read_writes.reads}

    @classmethod
    def _drain_nodes(
        cls,
        nodes: list,
        chunk: list,
        chunk_buffers: list[str],
        future_chunk_buffers: set[str],
        annotation_key: str,
    ) -> list:
        chunk_set = set(chunk)
        chunk_buffer_set = set(chunk_buffers)
        drain_nodes = []
        for node in nodes:
            if node in chunk_set:
                continue
            region = cls._chunk_region(node, annotation_key)
            if re.fullmatch(r"txt_unemb_chunk_(\d+)", region or "") is not None:
                continue
            read_names = cls._read_names(node)
            if chunk_buffer_set.isdisjoint(read_names):
                continue
            if not future_chunk_buffers.isdisjoint(read_names):
                continue
            drain_nodes.append(node)
        return drain_nodes


@dataclass(frozen=True)
class ActivationCheckpointingPassConfig:
    mode: str = "eager"
    layer_prefix: str = "layers"
    min_cut_policy: str = "none"
    max_peak_increase_gb: float | None = None
    memory_estimator: str = "approximate"
    save_scope: str = "min_cut"
    save_final_layer_output: bool = True
    relax_relaxable_must_saves: bool = False
    allow_allowed_saves: bool = False


def activation_checkpointing_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    ac_config: ActivationCheckpointingPassConfig = ActivationCheckpointingPassConfig(),
) -> torch.fx.GraphModule:
    """Apply tag-based AC and optional min-cut before compilation.

    This is the generic pass used by graph_trainer-adjacent callers that do not
    have a full ``GraphTrainer.Config`` but still want the same save-layer-inputs,
    min-cut, and selective-remat implementation.
    """
    if ac_config.mode in ("eager", "default", None):
        if ac_config.min_cut_policy != "none":
            raise ValueError("min-cut AC requires mode='save_layer_inputs'")
        return gm
    if ac_config.mode != "save_layer_inputs":
        raise ValueError(f"Unknown AC mode: {ac_config.mode!r}")
    if ac_config.min_cut_policy not in ("none", "min_cut_peak_aware"):
        raise ValueError(f"Unknown min-cut policy: {ac_config.min_cut_policy!r}")

    from torchtitan.experiments.graph_trainer.common_utils import (
        apply_save_layer_inputs_ac,
    )
    from torchtitan.experiments.graph_trainer.min_cut_ac import (
        ac_allow_allowed_saves,
        ac_relax_relaxable_must_saves,
        min_cut_ac_pass,
    )
    from torchtitan.experiments.graph_trainer.selective_activation_remat import (
        selective_activation_remat_pass,
    )

    n_before = len(list(gm.graph.nodes))
    apply_save_layer_inputs_ac(
        gm,
        layer_prefix=ac_config.layer_prefix,
        save_final_layer_output=ac_config.save_final_layer_output,
    )

    if ac_config.min_cut_policy == "min_cut_peak_aware":
        min_cut_ac_pass(
            gm,
            example_inputs,
            max_peak_increase_gb=ac_config.max_peak_increase_gb,
            memory_estimator=ac_config.memory_estimator,
            save_scope=ac_config.save_scope,
            relax_relaxable_must_saves=ac_config.relax_relaxable_must_saves,
            allow_allowed_saves=ac_config.allow_allowed_saves,
        )
    else:
        if ac_config.relax_relaxable_must_saves:
            ac_relax_relaxable_must_saves(gm, example_inputs)
        if ac_config.allow_allowed_saves:
            ac_allow_allowed_saves(gm, example_inputs)

    gm = selective_activation_remat_pass(gm, example_inputs)
    logger.info(
        "activation_checkpointing_pass: mode=%s min_cut_policy=%s "
        "max_peak_increase_gb=%s memory_estimator=%s save_scope=%s "
        "save_final_layer_output=%s relax_relaxable_must_saves=%s "
        "allow_allowed_saves=%s nodes=%d->%d",
        ac_config.mode,
        ac_config.min_cut_policy,
        ac_config.max_peak_increase_gb,
        ac_config.memory_estimator,
        ac_config.save_scope,
        ac_config.save_final_layer_output,
        ac_config.relax_relaxable_must_saves,
        ac_config.allow_allowed_saves,
        n_before,
        len(list(gm.graph.nodes)),
    )
    return gm


def full_inductor_compilation_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    ac_config: ActivationCheckpointingPassConfig | None = None,
    inductor_configs: dict | None = None,
    fuse_region_strategy: str | None = "blocks_contiguous",
) -> torch.fx.GraphModule:
    """Apply full Inductor compilation by tagging every node and delegating
    to :func:`regional_inductor_pass`.

    Marks every non-placeholder/output node with the ``compile_with_inductor``
    custom metadata key so ``regional_inductor`` scoops the entire graph as
    one compiled region. This reuses the regional path (which goes through
    ``standalone_compile`` and gets c10d functionalization, PG unboxing,
    decompositions, and caching for free) instead of duplicating that prep
    around a direct ``compile_fx_inner`` call.

    The collapse hides cudagraph-incompatible ops (unpinned D2H copies,
    sm<10 ``_grouped_mm``) inside the opaque ``standalone_compile_inner``
    node, so the later :func:`is_cudagraph_compatible` scan can't see
    them. Snapshot the verdict on the pre-collapse gm and stash it on
    the result so the downstream scan can honor it.

    Must be the **terminal** pass — no FX-graph-level passes (e.g.
    ``custom_codegen_pass``, ``insert_kernel_annotations_pass``) can
    run after this because the FX graph is no longer authoritative.
    """
    import torch._inductor.config as ic

    from torchtitan.experiments.graph_trainer.cudagraph import is_cudagraph_compatible

    if ac_config is not None:
        gm = activation_checkpointing_pass(gm, example_inputs, ac_config=ac_config)

    pre_collapse_cudagraph_compatible = is_cudagraph_compatible(gm)

    full_inductor_configs = {
        "reorder_for_peak_memory": True,
        "size_threshold_for_succ_based_strategy": 1,
        **(inductor_configs or {}),
    }
    if fuse_region_strategy is not None:
        fuse_region_pass = _FullInductorFuseRegionPass(fuse_region_strategy)
        custom_post_pass = full_inductor_configs.get("post_grad_custom_post_pass")
        if custom_post_pass is None:
            full_inductor_configs["post_grad_custom_post_pass"] = fuse_region_pass
        elif isinstance(custom_post_pass, (list, tuple)):
            full_inductor_configs["post_grad_custom_post_pass"] = [
                *custom_post_pass,
                fuse_region_pass,
            ]
        else:
            full_inductor_configs["post_grad_custom_post_pass"] = [
                custom_post_pass,
                fuse_region_pass,
            ]
        full_inductor_configs.setdefault(
            "_post_fusion_custom_pass",
            _FullInductorFuseRegionOrderPass(),
        )
    _migrate_cpu_get_attrs_to_cuda(gm)
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            node.meta.setdefault("custom", {})["compile_with_inductor"] = {
                "inductor_configs": full_inductor_configs
            }
    # AOT autograd (via ``standalone_compile``) reorders the gm and breaks
    # fwd/bwd interleaving, blowing up the baseline schedule. Re-enable
    # Inductor's reorder pass (disabled globally in ``compile.py``) to fix.
    with ic.patch(full_inductor_configs):
        result = regional_inductor_pass(gm, example_inputs)

    # Carry the pre-collapse cudagraph verdict forward via gm.meta. The
    # collapse is information-destroying; this is how downstream passes
    # know whether the artifact contains hidden cudagraph-incompatible ops.
    result.meta["cudagraph_compatible"] = pre_collapse_cudagraph_compatible
    return result
