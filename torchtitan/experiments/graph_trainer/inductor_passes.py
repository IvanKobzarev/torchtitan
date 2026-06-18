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


FULL_INDUCTOR_FUSE_REGION_PHASE = "full_inductor_fuse_region_phase"
FULL_INDUCTOR_GRAPH_MODULE_INPUT = "full_inductor_graph_module_input"
FULL_INDUCTOR_UNREGIONED_MIN_VOCAB = 100_000


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


def _clone_full_inductor_graph_module_inputs(gm: torch.fx.GraphModule) -> None:
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in list(module.graph.nodes):
            custom = node.meta.get("custom")
            compile_region = (
                custom.get("full_inductor_compile_region")
                if isinstance(custom, dict)
                else None
            )
            for inp in list(node.all_input_nodes):
                if inp.op != "get_attr" or not isinstance(inp.target, str):
                    continue
                submod = getattr(module, inp.target, None)
                if not isinstance(submod, torch.fx.GraphModule):
                    continue
                with module.graph.inserting_before(node):
                    cloned_inp = module.graph.get_attr(inp.target)
                cloned_inp.meta = dict(inp.meta)
                inp_custom = cloned_inp.meta.get("custom")
                cloned_custom = dict(inp_custom) if isinstance(inp_custom, dict) else {}
                cloned_inp.meta["custom"] = cloned_custom
                cloned_custom[FULL_INDUCTOR_GRAPH_MODULE_INPUT] = True
                if isinstance(compile_region, str):
                    cloned_custom["full_inductor_compile_region"] = compile_region
                node.replace_input_with(inp, cloned_inp)
        module.graph.eliminate_dead_code()
        module.recompile()


class _FullInductorFuseRegionPass(CustomGraphPass):
    def __init__(self, strategy: str) -> None:
        strategy_parts = strategy.split(",")
        if strategy_parts[0] != "blocks_contiguous":
            raise ValueError(
                f"Unknown full-Inductor fuse-region strategy: {strategy!r}"
            )
        self.strategy = strategy
        self.base_strategy = strategy_parts[0]
        self.disabled = False
        self.include_regions: set[str] = set()
        self.exclude_regions: set[str] = set()
        self.include_prefixes: tuple[str, ...] = ()
        self.exclude_prefixes: tuple[str, ...] = ()
        self.share_prefixes: tuple[str, ...] = ()
        self.merge_txt_unemb_phases = False
        self.include_unregioned = False
        self.order_excluded_regions = False
        for option in strategy_parts[1:]:
            if option == "disable":
                self.disabled = True
            elif option == "merge_txt_unemb_phases":
                self.merge_txt_unemb_phases = True
            elif option == "include_unregioned":
                self.include_unregioned = True
            elif option == "order_excluded_regions":
                self.order_excluded_regions = True
            elif option.startswith("include="):
                self.include_regions.update(self._split_option_values(option))
            elif option.startswith("exclude="):
                self.exclude_regions.update(self._split_option_values(option))
            elif option.startswith("include_prefix="):
                self.include_prefixes += self._split_option_values(option)
            elif option.startswith("exclude_prefix="):
                self.exclude_prefixes += self._split_option_values(option)
            elif option.startswith("share_prefix="):
                self.share_prefixes += self._split_option_values(option)
            else:
                raise ValueError(
                    f"Unknown full-Inductor fuse-region strategy option: {option!r}"
                )

    def uuid(self) -> object:
        return ("graph_trainer_full_inductor_fuse_region", self.strategy, 12)

    def __call__(self, graph: torch.fx.Graph) -> None:
        from torch._inductor.fx_passes.control_dependencies import (
            control_deps,
            FUSE_REGION,
            mark_fuse_region,
        )
        from torch._logging import trace_structured

        include_unregioned = (
            self.include_unregioned
            and self._graph_has_txt_unemb_region(graph)
            and self._graph_has_large_vocab_tensor(graph)
        )
        groups: list[tuple[str, list[torch.fx.Node]]] = []
        current_key: str | None = None
        current_nodes: list[torch.fx.Node] = []

        def flush() -> None:
            nonlocal current_key, current_nodes
            if current_key is not None and current_nodes:
                groups.append((current_key, current_nodes))
            current_key = None
            current_nodes = []

        for node in list(graph.nodes):
            key = self._region_key(node, include_unregioned)
            if key is not None and not self._region_enabled(key):
                key = None
            if key is None:
                flush()
                continue
            if key != current_key:
                flush()
                current_key = key
            current_nodes.append(node)
        flush()

        for idx, (key, nodes) in enumerate(groups):
            before = set(graph.nodes)
            mark_fuse_region(
                graph,
                nodes,
                fuse_region_id=self._fuse_region_id(key),
            )
            for node in graph.nodes:
                if (
                    node not in before
                    and node.op == "call_function"
                    and node.target is control_deps
                    and node.kwargs.get(FUSE_REGION) is True
                ):
                    node.name = self._region_node_name(key, idx)
                    break

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "graph_trainer_full_inductor_fuse_regions",
                "encoding": "string",
            },
            payload_fn=lambda: "\n".join(
                [
                    f"strategy={self.base_strategy}",
                    f"options={self._options_summary()}",
                    "unique_islands=True",
                    f"marked_groups={len(groups)}",
                    *[
                        f"{idx}\t{key}\t{len(nodes)}\t{nodes[0].name}\t{nodes[-1].name}"
                        for idx, (key, nodes) in enumerate(groups)
                    ],
                ]
            )
            + "\n",
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
            return self._region_key_with_phase(node, custom_region)
        compile_value = custom.get("compile_with_inductor")
        if isinstance(compile_value, dict):
            compiled_region = compile_value.get("full_inductor_fuse_region")
            if isinstance(compiled_region, str):
                return self._region_key_with_phase(node, compiled_region)
        return None

    def _region_key_with_phase(self, node: torch.fx.Node, region: str) -> str:
        if re.fullmatch(r"txt_unemb_chunk_\d+", region) is None:
            return region
        phase = "bwd" if node.meta.get("autograd_backward") is True else "fwd"
        custom = node.meta.get("custom")
        if isinstance(custom, dict):
            custom[FULL_INDUCTOR_FUSE_REGION_PHASE] = phase
        if self.merge_txt_unemb_phases:
            return region
        return f"{region}_{phase}"

    def _region_key(self, node: torch.fx.Node, include_unregioned: bool) -> str | None:
        region = self._custom_region_key(node)
        if region is None:
            region = self._inferred_txt_unemb_region_key(node)
        if region is not None or not include_unregioned:
            return region
        return self._unregioned_region_key(node)

    def _inferred_txt_unemb_region_key(self, node: torch.fx.Node) -> str | None:
        if node.op != "call_function":
            return None
        from torch._inductor import inductor_prims

        if node.target is not inductor_prims.prepare_softmax_online:
            return None
        regions = {
            region
            for inp in node.all_input_nodes
            if (region := self._txt_unemb_region_base(inp)) is not None
        }
        if len(regions) != 1:
            return None
        region = next(iter(regions))
        custom = node.meta.get("custom")
        if not isinstance(custom, dict):
            custom = {}
            node.meta["custom"] = custom
        custom["full_inductor_fuse_region"] = region
        return self._region_key_with_phase(node, region)

    @staticmethod
    def _txt_unemb_region_base(node: torch.fx.Node) -> str | None:
        custom = node.meta.get("custom")
        if not isinstance(custom, dict):
            return None
        region = custom.get("full_inductor_fuse_region")
        if not isinstance(region, str):
            compile_value = custom.get("compile_with_inductor")
            if isinstance(compile_value, dict):
                region = compile_value.get("full_inductor_fuse_region")
        if not isinstance(region, str):
            return None
        match = re.fullmatch(r"(txt_unemb_chunk_\d+)(?:_(?:fwd|bwd))?", region)
        return None if match is None else match.group(1)

    def _unregioned_region_key(self, node: torch.fx.Node) -> str | None:
        if node.op in ("placeholder", "output", "get_attr"):
            return None
        if node.op != "call_function":
            return None
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            return None
        namespace = getattr(node.target, "namespace", None)
        if namespace not in ("aten", "prims"):
            return None
        if not self._has_tensor_meta(node):
            return None
        if self._has_graph_module_arg(node):
            return None
        phase = "bwd" if node.meta.get("autograd_backward") is True else "fwd"
        return f"unregioned_{phase}"

    @staticmethod
    def _graph_has_txt_unemb_region(graph: torch.fx.Graph) -> bool:
        for node in graph.nodes:
            custom = node.meta.get("custom")
            if not isinstance(custom, dict):
                continue
            region = custom.get("full_inductor_fuse_region")
            if isinstance(region, str) and region.startswith("txt_unemb_chunk_"):
                return True
            compile_value = custom.get("compile_with_inductor")
            if isinstance(compile_value, dict):
                region = compile_value.get("full_inductor_fuse_region")
                if isinstance(region, str) and region.startswith("txt_unemb_chunk_"):
                    return True
        return False

    @staticmethod
    def _graph_has_large_vocab_tensor(graph: torch.fx.Graph) -> bool:
        for node in graph.nodes:
            val = node.meta.get("val")
            if val is None:
                continue
            leaves = (
                (val,)
                if isinstance(val, torch.Tensor)
                else torch.utils._pytree.tree_leaves(val)
            )
            for leaf in leaves:
                if not isinstance(leaf, torch.Tensor):
                    continue
                for dim in leaf.shape:
                    if (
                        isinstance(dim, int)
                        and dim >= FULL_INDUCTOR_UNREGIONED_MIN_VOCAB
                    ):
                        return True
        return False

    @staticmethod
    def _has_tensor_meta(node: torch.fx.Node) -> bool:
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            return True
        if val is None:
            return False
        leaves = torch.utils._pytree.tree_leaves(val)
        return any(isinstance(leaf, torch.Tensor) for leaf in leaves)

    @staticmethod
    def _split_option_values(option: str) -> tuple[str, ...]:
        return tuple(value for value in option.split("=", 1)[1].split("|") if value)

    def _region_enabled(self, key: str) -> bool:
        if self.disabled:
            return False
        if (
            (self.include_regions or self.include_prefixes)
            and key not in self.include_regions
            and not key.startswith(self.include_prefixes)
        ):
            return False
        return key not in self.exclude_regions and not key.startswith(
            self.exclude_prefixes
        )

    def _options_summary(self) -> str:
        return repr(
            {
                "disabled": self.disabled,
                "include": sorted(self.include_regions),
                "exclude": sorted(self.exclude_regions),
                "include_prefix": self.include_prefixes,
                "exclude_prefix": self.exclude_prefixes,
                "share_prefix": self.share_prefixes,
                "merge_txt_unemb_phases": self.merge_txt_unemb_phases,
                "include_unregioned": self.include_unregioned,
                "order_excluded_regions": self.order_excluded_regions,
            }
        )

    def _fuse_region_id(self, key: str) -> str | None:
        if self.share_prefixes and key.startswith(self.share_prefixes):
            return key
        return None

    @staticmethod
    def _region_node_name(key: str, idx: int) -> str:
        name = re.sub(r"\W+", "_", key).strip("_")
        if not name or name[0].isdigit():
            name = f"region_{name}"
        return f"{name}_island_{idx}"

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
    def __init__(self, strategy: str | None = None) -> None:
        self.strategy = strategy
        self.region_filter = None
        if strategy is not None:
            region_filter = _FullInductorFuseRegionPass(strategy)
            if not region_filter.order_excluded_regions:
                self.region_filter = region_filter

    def uuid(self) -> object:
        return ("graph_trainer_full_inductor_fuse_region_order", self.strategy, 9)

    def __call__(self, nodes: list) -> list:
        from torch._inductor.dependencies import WeakDep
        from torch._inductor.fx_passes.control_dependencies import FUSE_REGION
        from torch._logging import trace_structured

        chunk_nodes: dict[int, list] = {}
        phase_nodes: dict[tuple[int, str], list] = {}
        for node in nodes:
            region = self._chunk_region(node, FUSE_REGION)
            match = re.fullmatch(r"txt_unemb_chunk_(\d+)(?:_(fwd|bwd))?", region or "")
            if match is None:
                continue
            chunk = int(match.group(1))
            chunk_nodes.setdefault(chunk, []).append(node)
            phase = self._phase(node) or match.group(2)
            if phase is not None:
                phase_nodes.setdefault((chunk, phase), []).append(node)

        large_vocab_chunks = {
            chunk
            for chunk, chunk_group in chunk_nodes.items()
            if any(self._node_has_large_vocab_origin(node) for node in chunk_group)
        }
        chunk_nodes = {
            chunk: chunk_group
            for chunk, chunk_group in chunk_nodes.items()
            if chunk in large_vocab_chunks
        }
        phase_nodes = {
            key: phase_group
            for key, phase_group in phase_nodes.items()
            if key[0] in large_vocab_chunks
        }

        chunk_buffers = {
            chunk: self._buffer_names(chunk_nodes[chunk]) for chunk in chunk_nodes
        }
        future_chunk_buffers: dict[int, set[str]] = {}
        future_buffers: set[str] = set()
        for chunk in sorted(chunk_nodes, reverse=True):
            future_chunk_buffers[chunk] = set(future_buffers)
            future_buffers.update(chunk_buffers[chunk])

        drain_nodes = {
            chunk: [
                *self._external_drain_nodes(
                    nodes,
                    chunk_nodes[chunk],
                    chunk_buffers[chunk],
                    future_chunk_buffers[chunk],
                    FUSE_REGION,
                ),
                *self._internal_drain_nodes(
                    phase_nodes.get((chunk, "bwd"), []),
                    chunk_buffers[chunk],
                    future_chunk_buffers[chunk],
                ),
            ]
            for chunk in chunk_nodes
        }
        drain_buffers = {
            chunk: self._first_buffer_names(drain_nodes[chunk]) or chunk_buffers[chunk]
            for chunk in chunk_nodes
        }

        buffer_producers = self._buffer_producers(nodes)
        deps_by_node = self._dependency_predecessors(nodes, buffer_producers)

        deps_added = 0
        deps_skipped_cycle = 0
        phase_deps_added = 0
        phase_deps_skipped_cycle = 0
        prev_order_buffers: list[str] = []
        for chunk in sorted(chunk_nodes):
            current = chunk_nodes[chunk]
            if prev_order_buffers:
                for node in current:
                    mutating_buf = self._first_buffer_name(node)
                    if mutating_buf is None:
                        continue
                    for dep_name in prev_order_buffers:
                        producer = buffer_producers.get(dep_name)
                        if producer is not None and self._has_dependency_path(
                            producer, node, deps_by_node
                        ):
                            deps_skipped_cycle += 1
                            continue
                        dep = WeakDep(dep_name, mutating_buf=mutating_buf, is_fake=True)
                        self._add_fake_dep(node, dep)
                        if producer is not None:
                            deps_by_node.setdefault(node, set()).add(producer)
                        deps_added += 1
            prev_order_buffers = drain_buffers[chunk]

        for chunk in sorted(chunk_nodes):
            fwd_nodes = phase_nodes.get((chunk, "fwd"), [])
            bwd_nodes = phase_nodes.get((chunk, "bwd"), [])
            fwd_order_buffers = self._first_buffer_names(fwd_nodes)
            if not fwd_order_buffers or not bwd_nodes:
                continue
            for node in bwd_nodes:
                mutating_buf = self._first_buffer_name(node)
                if mutating_buf is None:
                    continue
                for dep_name in fwd_order_buffers:
                    producer = buffer_producers.get(dep_name)
                    if producer is not None and self._has_dependency_path(
                        producer, node, deps_by_node
                    ):
                        phase_deps_skipped_cycle += 1
                        continue
                    dep = WeakDep(dep_name, mutating_buf=mutating_buf, is_fake=True)
                    self._add_fake_dep(node, dep)
                    if producer is not None:
                        deps_by_node.setdefault(node, set()).add(producer)
                    phase_deps_added += 1

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "graph_trainer_full_inductor_fuse_region_order",
                "encoding": "string",
            },
            payload_fn=lambda: "\n".join(
                [
                    f"chunks={sorted(chunk_nodes)}",
                    f"large_vocab_chunks={sorted(large_vocab_chunks)}",
                    f"deps_added={deps_added}",
                    f"deps_skipped_cycle={deps_skipped_cycle}",
                    f"phase_deps_added={phase_deps_added}",
                    f"phase_deps_skipped_cycle={phase_deps_skipped_cycle}",
                    *[
                        f"{chunk}\t{len(chunk_nodes[chunk])}\t"
                        f"fwd_nodes={len(phase_nodes.get((chunk, 'fwd'), []))}\t"
                        f"bwd_nodes={len(phase_nodes.get((chunk, 'bwd'), []))}\t"
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

    def _chunk_region(self, node, annotation_key: str) -> str | None:
        region = self._annotation(node, annotation_key)
        if region is None:
            region = self._origin_chunk_region(node)
        if region is None:
            return None
        region = self._normalize_chunk_region(region)
        if region is None or not self._region_enabled(region):
            return None
        return region

    @staticmethod
    def _normalize_chunk_region(region: str) -> str | None:
        match = re.match(r"(txt_unemb_chunk_\d+(?:_(?:fwd|bwd))?)_island_\d+$", region)
        if match is not None:
            return match.group(1)
        match = re.fullmatch(r"txt_unemb_chunk_\d+(?:_(?:fwd|bwd))?", region)
        if match is None:
            return None
        return region

    def _region_enabled(self, region: str) -> bool:
        if self.region_filter is None:
            return True
        return self.region_filter._region_enabled(region)

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

    @classmethod
    def _phase(cls, node) -> str | None:
        phases: set[str] = set()
        for origin in cls._origins(node):
            meta = getattr(origin, "meta", None)
            if not isinstance(meta, dict):
                continue
            if meta.get("autograd_backward") is True:
                phases.add("bwd")
            custom = meta.get("custom")
            if not isinstance(custom, dict):
                continue
            phase = custom.get(FULL_INDUCTOR_FUSE_REGION_PHASE)
            if phase is None:
                continue
            if phase not in ("fwd", "bwd"):
                raise AssertionError(
                    f"expected fuse-region phase to be fwd/bwd, got {phase!r}"
                )
            phases.add(phase)
        if "bwd" in phases:
            return "bwd"
        if "fwd" in phases:
            return "fwd"
        return None

    @classmethod
    def _origin_chunk_region(cls, node) -> str | None:
        regions: set[str] = set()
        for origin in cls._origins(node):
            meta = getattr(origin, "meta", None)
            if not isinstance(meta, dict):
                continue
            custom = meta.get("custom")
            if not isinstance(custom, dict):
                continue
            region = custom.get("full_inductor_fuse_region")
            if not isinstance(region, str):
                compile_value = custom.get("compile_with_inductor")
                if isinstance(compile_value, dict):
                    region = compile_value.get("full_inductor_fuse_region")
            if isinstance(region, str) and re.fullmatch(
                r"txt_unemb_chunk_\d+(?:_(?:fwd|bwd))?", region
            ):
                regions.add(region)
        if len(regions) > 1:
            return None
        return next(iter(regions), None)

    @classmethod
    def _node_has_large_vocab_origin(cls, node) -> bool:
        for origin in cls._origins(node):
            meta = getattr(origin, "meta", None)
            if not isinstance(meta, dict):
                continue
            val = meta.get("val")
            if val is None:
                continue
            leaves = (
                (val,)
                if isinstance(val, torch.Tensor)
                else torch.utils._pytree.tree_leaves(val)
            )
            for leaf in leaves:
                if not isinstance(leaf, torch.Tensor):
                    continue
                for dim in leaf.shape:
                    if (
                        isinstance(dim, int)
                        and dim >= FULL_INDUCTOR_UNREGIONED_MIN_VOCAB
                    ):
                        return True
        return False

    @staticmethod
    def _origins(node) -> list[object]:
        origins: list[object] = []
        for snode in node.get_nodes():
            op = snode.node
            if op is None:
                continue
            op_origins = op.get_origins() if hasattr(op, "get_origins") else None
            if op_origins is not None:
                origins.extend(op_origins)
            origin = op.get_origin_node() if hasattr(op, "get_origin_node") else None
            if origin is not None:
                origins.append(origin)
        return origins

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
    def _buffer_producers(cls, nodes: list) -> dict[str, object]:
        producers = {}
        for node in nodes:
            for name in node.get_buffer_names():
                producers[name] = node
        return producers

    @classmethod
    def _dependency_predecessors(
        cls, nodes: list, buffer_producers: dict[str, object]
    ) -> dict[object, set[object]]:
        deps_by_node = {}
        for node in nodes:
            deps = set()
            for dep_name in cls._read_names(node):
                producer = buffer_producers.get(dep_name)
                if producer is not None:
                    deps.add(producer)
            deps_by_node[node] = deps
        return deps_by_node

    @staticmethod
    def _has_dependency_path(src, dst, deps_by_node: dict[object, set[object]]) -> bool:
        stack = list(deps_by_node.get(src, ()))
        seen = set()
        while stack:
            node = stack.pop()
            if node is dst:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(deps_by_node.get(node, ()))
        return False

    def _external_drain_nodes(
        self,
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
            region = self._chunk_region(node, annotation_key)
            if (
                re.fullmatch(r"txt_unemb_chunk_(\d+)(?:_(?:fwd|bwd))?", region or "")
                is not None
            ):
                continue
            read_names = self._read_names(node)
            if chunk_buffer_set.isdisjoint(read_names):
                continue
            if not future_chunk_buffers.isdisjoint(read_names):
                continue
            drain_nodes.append(node)
        return drain_nodes

    @classmethod
    def _internal_drain_nodes(
        cls,
        bwd_nodes: list,
        chunk_buffers: list[str],
        future_chunk_buffers: set[str],
    ) -> list:
        chunk_buffer_set = set(chunk_buffers)
        drain_nodes = []
        for node in bwd_nodes:
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
    min_broad_candidate_gb: float | None = None
    max_broad_candidate_gb: float | None = None
    save_final_layer_output: bool = True
    relax_relaxable_must_saves: bool = False
    allow_allowed_saves: bool = False
    allow_unsaveable_recomputes: bool = False


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
        ac_allow_unsaveable_recomputes,
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
            min_broad_candidate_gb=ac_config.min_broad_candidate_gb,
            max_broad_candidate_gb=ac_config.max_broad_candidate_gb,
            relax_relaxable_must_saves=ac_config.relax_relaxable_must_saves,
            allow_allowed_saves=ac_config.allow_allowed_saves,
            allow_unsaveable_recomputes=ac_config.allow_unsaveable_recomputes,
        )
    else:
        if ac_config.relax_relaxable_must_saves:
            ac_relax_relaxable_must_saves(gm, example_inputs)
        if ac_config.allow_allowed_saves:
            ac_allow_allowed_saves(gm, example_inputs)
        if ac_config.allow_unsaveable_recomputes:
            ac_allow_unsaveable_recomputes(gm, example_inputs)

    gm = selective_activation_remat_pass(gm, example_inputs)
    logger.info(
        "activation_checkpointing_pass: mode=%s min_cut_policy=%s "
        "max_peak_increase_gb=%s memory_estimator=%s save_scope=%s "
        "min_broad_candidate_gb=%s max_broad_candidate_gb=%s "
        "save_final_layer_output=%s "
        "relax_relaxable_must_saves=%s allow_allowed_saves=%s "
        "allow_unsaveable_recomputes=%s nodes=%d->%d",
        ac_config.mode,
        ac_config.min_cut_policy,
        ac_config.max_peak_increase_gb,
        ac_config.memory_estimator,
        ac_config.save_scope,
        ac_config.min_broad_candidate_gb,
        ac_config.max_broad_candidate_gb,
        ac_config.save_final_layer_output,
        ac_config.relax_relaxable_must_saves,
        ac_config.allow_allowed_saves,
        ac_config.allow_unsaveable_recomputes,
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
        "allow_buffer_reuse_across_fuse_regions": False,
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
            _FullInductorFuseRegionOrderPass(fuse_region_strategy),
        )
    _migrate_cpu_get_attrs_to_cuda(gm)
    _clone_full_inductor_graph_module_inputs(gm)
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            compile_annotation = {"inductor_configs": full_inductor_configs}
            custom = node.meta.get("custom")
            if (
                node.op == "get_attr"
                and isinstance(node.target, str)
                and isinstance(getattr(module, node.target, None), torch.fx.GraphModule)
                and not (
                    isinstance(custom, dict)
                    and custom.get(FULL_INDUCTOR_GRAPH_MODULE_INPUT) is True
                )
            ):
                continue
            if isinstance(custom, dict):
                fuse_region = custom.get("full_inductor_fuse_region")
                if isinstance(fuse_region, str):
                    compile_annotation["full_inductor_fuse_region"] = fuse_region
                compile_region = custom.get("full_inductor_compile_region")
                if isinstance(compile_region, str):
                    compile_annotation["inductor_region"] = compile_region
            node.meta.setdefault("custom", {})["compile_with_inductor"] = {
                **compile_annotation,
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
