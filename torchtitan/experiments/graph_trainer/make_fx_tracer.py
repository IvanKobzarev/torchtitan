# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._guards import tracing, TracingContext
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from torchtitan.experiments.graph_trainer.dynamic_shapes import (
    _fakeify_input,
    _insert_runtime_asserts as _insert_runtime_asserts_pass,
    _wrapper_subclass_has_marked_dynamic_dims,
)

# Tensors and make_fx-safe primitives are allowed as pytree leaves in args.
# Everything else (callables, custom objects) should be registered as pytree
# nodes/constants or captured in fn's closure.
_ALLOWED_LEAF_TYPES = (torch.Tensor, int, float, bool, str, type(None))


@contextmanager
def _skip_nested_compile() -> Generator[None, None, None]:
    """Tell dynamo to skip torch.compile calls encountered during make_fx tracing.

    make_fx cannot trace through torch.compile'd functions (e.g. compiled
    flex_attention in FlexAttention). Setting error_on_nested_fx_trace
    to False makes dynamo silently inline the wrapped function instead of
    raising, so make_fx traces the underlying ops normally.
    """
    prev = torch._dynamo.config.error_on_nested_fx_trace
    torch._dynamo.config.error_on_nested_fx_trace = False
    try:
        yield
    finally:
        torch._dynamo.config.error_on_nested_fx_trace = prev


@dataclass
class SubclassMeta:
    cls: type
    attrs: list[str]
    ctx: Any
    inner_metas: dict[str, tuple[int, Any]]
    outer_size: torch.Size
    outer_stride: tuple[int, ...]


@dataclass
class SubclassLayout:
    num_tensors: int
    meta: SubclassMeta | None


def _unwrap_subclass(t: torch.Tensor) -> tuple[list[torch.Tensor], SubclassMeta | None]:
    if not is_traceable_wrapper_subclass(t):
        return [t], None
    attrs, ctx = t.__tensor_flatten__()
    all_inner = []
    inner_metas = {}
    for attr in attrs:
        inner_t = getattr(t, attr)
        tensors, meta = _unwrap_subclass(inner_t)
        all_inner.extend(tensors)
        inner_metas[attr] = (len(tensors), meta)
    meta = SubclassMeta(
        cls=type(t),
        attrs=attrs,
        ctx=ctx,
        inner_metas=inner_metas,
        outer_size=t.size(),
        outer_stride=t.stride(),
    )
    return all_inner, meta


def _wrap_to_subclass(
    plain_tensors: list[torch.Tensor],
    meta: SubclassMeta,
) -> torch.Tensor:
    inner_dict = {}
    idx = 0
    for attr in meta.attrs:
        num_inner, inner_meta = meta.inner_metas[attr]
        inner_tensors = plain_tensors[idx : idx + num_inner]
        idx += num_inner
        if inner_meta is None:
            inner_dict[attr] = inner_tensors[0]
        else:
            inner_dict[attr] = _wrap_to_subclass(list(inner_tensors), inner_meta)

    return meta.cls.__tensor_unflatten__(
        inner_dict,
        meta.ctx,
        meta.outer_size,
        meta.outer_stride,
    )


def _unwrap_subclasses(
    args: list,
) -> tuple[list, dict[int, SubclassLayout]]:
    """Unwrap tensor subclasses into plain tensors.

    Returns the flattened plain tensors and a dict mapping original arg index
    to its SubclassLayout.  Plain tensors have no entry.
    """
    flat: list = []
    layouts: dict[int, SubclassLayout] = {}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and is_traceable_wrapper_subclass(arg):
            inner_tensors, meta = _unwrap_subclass(arg)
            layouts[i] = SubclassLayout(len(inner_tensors), meta)
            flat.extend(inner_tensors)
        else:
            flat.append(arg)
    return flat, layouts


def _wrap_subclasses(
    flat_tensors: tuple | list,
    num_args: int,
    layouts: dict[int, SubclassLayout],
) -> list:
    """Rewrap plain tensors back into their original subclass types.

    Positions not in ``layouts`` are plain tensors (taken one-to-one).
    """
    wrapped = []
    idx = 0
    for i in range(num_args):
        if i in layouts:
            layout = layouts[i]
            tensors = flat_tensors[idx : idx + layout.num_tensors]
            idx += layout.num_tensors
            wrapped.append(_wrap_to_subclass(list(tensors), layout.meta))
        else:
            wrapped.append(flat_tensors[idx])
            idx += 1
    return wrapped


def _copy_fwd_metadata_to_bw_nodes(fx_g: torch.fx.GraphModule) -> None:
    """Copy forward metadata to backward nodes across all nested FX subgraphs.

    Uses a two-pass approach over all submodule graphs (including HOP subgraphs
    like score_mod/mask_mod). Pass 1 collects forward nodes by seq_nr; pass 2
    copies custom/nn_module_stack/stack_trace from the matching forward node to
    each backward node. Backward nodes are identified by the autograd engine's
    ``autograd_backward`` tag on ``node.meta``.
    """

    def _is_backward(node: torch.fx.Node) -> bool:
        return node.meta.get("autograd_backward", False)

    seq_nr_to_fwd_node: dict[int, torch.fx.Node] = {}

    for submod in fx_g.modules():
        if not isinstance(submod, torch.fx.GraphModule):
            continue
        for node in submod.graph.nodes:
            if (
                node.op not in ("call_function", "get_attr")
                or "seq_nr" not in node.meta
                or _is_backward(node)
            ):
                continue
            seq_nr = node.meta["seq_nr"]
            if seq_nr not in seq_nr_to_fwd_node:
                seq_nr_to_fwd_node[seq_nr] = node

    for submod in fx_g.modules():
        if not isinstance(submod, torch.fx.GraphModule):
            continue
        for node in submod.graph.nodes:
            if (
                node.op not in ("call_function", "get_attr")
                or "seq_nr" not in node.meta
                or not _is_backward(node)
            ):
                continue
            fwd_node = seq_nr_to_fwd_node.get(node.meta["seq_nr"])
            if fwd_node is None or fwd_node is node:
                continue

            custom = fwd_node.meta.get("custom")
            if custom:
                node.meta.setdefault("custom", {}).update(copy.deepcopy(custom))
            nn_module_stack = fwd_node.meta.get("nn_module_stack")
            if nn_module_stack is not None:
                node.meta["nn_module_stack"] = nn_module_stack.copy()
            stack_trace = fwd_node.meta.get("stack_trace")
            if stack_trace is not None:
                node.meta["stack_trace"] = stack_trace


def extract_module_state(mod: nn.Module) -> dict[str, torch.Tensor]:
    """Return a merged dict of the module's named parameters and buffers."""
    return {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }


def extract_train_state(
    module: nn.Module | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return ``(model_state, optim_state)`` for ``minimal_fx_tracer``.

    Both are dicts (empty when the corresponding object is ``None``) and are
    sampled from the live module/optimizer, so callers can reuse this helper
    to refresh state at runtime.
    """
    model_state = extract_module_state(module) if module is not None else {}
    optim_state = optimizer.state_dict() if optimizer is not None else {}
    return model_state, optim_state


def _check_optimizer_has_module(
    module: nn.Module | None,
    optimizer: "torch.optim.Optimizer | None",
) -> None:
    """Optimizer parameters align with module.named_parameters() by order, so
    requiring a module guarantees that alignment."""
    if optimizer is not None and module is None:
        raise ValueError(
            "minimal_fx_tracer: when 'optimizer' is provided, 'module' must also "
            "be provided so optimizer parameters align with the module's parameters."
        )


@contextlib.contextmanager
def _reparametrize_train_state(
    module: nn.Module | None,
    optimizer: "torch.optim.Optimizer | None",
    model_state: dict[str, torch.Tensor],
    optim_state: dict[str, Any],
):
    """Reparametrize module and optimizer with explicit tensor state for tracing."""
    with contextlib.ExitStack() as stack:
        if optimizer is not None:
            # swap_in pairs values positionally in optimizer.param_groups flat
            # order, which differs from named_parameters() order for bucketed
            # param_groups. Walk param_groups and resolve names by id() against
            # the originals; must run before _reparametrize_module rebinds them.
            id_to_name = {
                id(p): n for n, p in module.named_parameters(remove_duplicate=False)
            }
            params_for_optim = {
                id_to_name[id(p)]: model_state[id_to_name[id(p)]]
                for group in optimizer.param_groups
                for p in group["params"]
            }
            stack.enter_context(
                torch.optim.swap_in_optimizer_params_and_state(
                    optimizer, params_for_optim, optim_state
                )
            )
        if module is not None:
            stack.enter_context(stateless._reparametrize_module(module, model_state))
        yield


@dataclass
class TracedResult:
    """Execution metadata returned by :func:`minimal_fx_tracer`.

    Attributes:
        gm: The traced FX graph as a pure function of flat tensors.
        example_inputs: Trace-time fake flat inputs used by downstream graph passes.
        num_flat_inputs: Number of flat graph inputs before subclass unwrapping.
        input_subclass_layouts: Subclass unwrap/rewrap metadata for inputs.
        user_inputs_spec: Trace-time pytree spec for ``(args, kwargs)``.
        num_flat_outputs: Number of flat graph outputs before subclass rewrapping.
        output_subclass_layouts: Subclass unwrap/rewrap metadata for outputs.
        output_spec: Original output pytree spec used during reconstruction.
        state_fqns: Trace-time module parameter/buffer FQNs.
        graph_state_fqns: Names of trainer-owned tensor state threaded through
            the graph separately from model and optimizer state.
        graph_state_input_indices: Flat graph-input indices for each logical
            graph-state tensor.
        graph_state_output_indices: Logical graph-output index whose tensor
            leaves contribute to each graph-state tensor. Empty for state that
            is not an accumulated gradient destination.
        grad_sink_active: Whether the graph writes parameter gradients into
            graph state and no longer returns them.
        num_optimizer_state_inputs: Number of logical optimizer-state inputs.
    """

    gm: torch.fx.GraphModule

    # input related
    example_inputs: tuple[Any, ...]
    num_flat_inputs: int
    input_subclass_layouts: dict[int, SubclassLayout]
    user_inputs_spec: pytree.TreeSpec
    tensor_input_indices: list[int]

    # output related
    num_flat_outputs: int
    output_subclass_layouts: dict[int, SubclassLayout]
    output_spec: pytree.TreeSpec

    # state related
    state_fqns: list[str]
    graph_state_fqns: list[str]
    graph_state_input_indices: tuple[tuple[int, ...], ...]
    graph_state_output_indices: tuple[int, ...]
    grad_sink_active: bool
    num_optimizer_state_inputs: int = 0

    @property
    def num_model_state_tensor_inputs(self) -> int:
        """Number of leading graph inputs produced from model state."""
        return sum(
            self.input_subclass_layouts[i].num_tensors
            if i in self.input_subclass_layouts
            else 1
            for i in range(len(self.state_fqns))
        )

    @property
    def num_static_inputs(self) -> int:
        """Number of leading graph inputs with stable tensor addresses.

        Parameters and buffers (the state entries) have fixed addresses across
        training steps. Each may expand to multiple plain tensors after
        subclass unwrapping (e.g. DTensor -> inner tensors).

        TODO: graph_trainer does not trace optimizers yet, so optimizer state
        tensors are not counted here. When optimizer tracing is enabled, they
        should be included since their addresses are also stable across steps,
        avoiding cudagraph re-copying them every step.
        """
        num_state = len(self.state_fqns) + len(self.graph_state_fqns)
        return sum(
            self.input_subclass_layouts[i].num_tensors
            if i in self.input_subclass_layouts
            else 1
            for i in range(num_state)
        )


_GRAPH_STATE_OUTPUT_META = "graph_state_outputs"


def _flat_tensor_ranges(
    num_values: int,
    layouts: dict[int, SubclassLayout],
) -> tuple[tuple[int, ...], ...]:
    ranges = []
    flat_index = 0
    for logical_index in range(num_values):
        num_tensors = (
            layouts[logical_index].num_tensors if logical_index in layouts else 1
        )
        ranges.append(tuple(range(flat_index, flat_index + num_tensors)))
        flat_index += num_tensors
    return tuple(ranges)


def capture_graph_state_output_metadata(
    gm: torch.fx.GraphModule,
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Capture gradient-state tags by flat output position."""
    output = next((node for node in gm.graph.nodes if node.op == "output"), None)
    if output is None:
        raise ValueError("Traced graph has no output node")
    return tuple(
        tuple(leaf.meta.get(_GRAPH_STATE_OUTPUT_META, ()))
        if isinstance(leaf, torch.fx.Node)
        else ()
        for leaf in pytree.tree_leaves(output.args[0])
    )


def restore_graph_state_output_metadata(
    gm: torch.fx.GraphModule,
    output_metadata: Sequence[tuple[tuple[str, int], ...]],
) -> None:
    """Restore tags after a pass that rebuilds output-producing nodes by position."""
    output = next((node for node in gm.graph.nodes if node.op == "output"), None)
    if output is None:
        raise ValueError("Transformed graph has no output node")
    output_leaves = pytree.tree_leaves(output.args[0])
    if len(output_leaves) != len(output_metadata):
        raise ValueError(
            "Graph pass changed the number of outputs while preserving gradient "
            "state metadata"
        )
    for leaf, mappings in zip(output_leaves, output_metadata, strict=True):
        if not mappings:
            continue
        if not isinstance(leaf, torch.fx.Node):
            raise ValueError("Graph pass replaced a gradient output with a non-tensor")
        leaf.meta[_GRAPH_STATE_OUTPUT_META] = mappings


def _tag_graph_state_outputs(
    gm: torch.fx.GraphModule,
    graph_state_fqns: Sequence[str],
    graph_state_output_indices: Sequence[int],
    num_flat_outputs: int,
    output_subclass_layouts: dict[int, SubclassLayout],
) -> None:
    if len(set(graph_state_output_indices)) != len(graph_state_output_indices):
        raise ValueError("graph_state_output_indices must be unique")

    output_ranges = _flat_tensor_ranges(num_flat_outputs, output_subclass_layouts)
    output = next((node for node in gm.graph.nodes if node.op == "output"), None)
    if output is None:
        raise ValueError("Traced graph has no output node")
    output_leaves = pytree.tree_leaves(output.args[0])
    expected_num_leaves = sum(len(indices) for indices in output_ranges)
    if len(output_leaves) != expected_num_leaves:
        raise ValueError(
            "Traced graph output metadata does not match its FX output: "
            f"expected {expected_num_leaves} tensor leaves, got "
            f"{len(output_leaves)}"
        )

    for fqn, output_index in zip(
        graph_state_fqns,
        graph_state_output_indices,
        strict=True,
    ):
        if output_index < 0 or output_index >= len(output_ranges):
            raise ValueError(
                f"Graph-state output index {output_index} for {fqn!r} is outside "
                f"the {len(output_ranges)} traced outputs"
            )
        for leaf_index, flat_index in enumerate(output_ranges[output_index]):
            leaf = output_leaves[flat_index]
            if not isinstance(leaf, torch.fx.Node):
                raise ValueError(
                    f"Graph-state output {output_index} for {fqn!r} is not a tensor"
                )
            mappings = list(leaf.meta.get(_GRAPH_STATE_OUTPUT_META, ()))
            mappings.append((fqn, leaf_index))
            leaf.meta[_GRAPH_STATE_OUTPUT_META] = tuple(mappings)

    gm.graph.lint()
    gm.recompile()


def minimal_fx_tracer(
    fn: Callable,
    module: nn.Module | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
    *,
    graph_state: dict[str, torch.Tensor] | None = None,
    graph_state_output_indices: Sequence[int] = (),
    prepare_inputs: Callable[[tuple[Any, ...], dict[str, Any]], None] | None = None,
    prepare_call_inputs: Callable[
        [tuple[Any, ...], dict[str, Any]],
        tuple[tuple[Any, ...], dict[str, Any]] | None,
    ]
    | None = None,
    record_stack_traces: bool = True,
    _insert_runtime_asserts: bool = False,
) -> Callable[..., TracedResult]:
    """Return a tracer that captures ``fn`` with implicit module/optimizer state.

    The returned callable takes the user-facing ``*args`` and ``**kwargs`` for
    ``fn``; module parameters/buffers and optimizer state are extracted from the
    live objects and threaded through the graph as static inputs::

        # Stateless function: no module, no optimizer.
        traced = minimal_fx_tracer(fn)(*args, **kwargs)

        # Module-only: parameters/buffers extracted from `module`.
        traced = minimal_fx_tracer(fn, module=model)(*args, **kwargs)

        # Module + optimizer: optimizer state must already be initialized
        # before tracing.
        traced = minimal_fx_tracer(fn, module=model, optimizer=opt)(*args, **kwargs)

    ``fn`` should reference ``module`` and ``optimizer`` from its enclosing
    closure — passing them explicitly through ``args``/``kwargs`` is invalid
    because ``nn.Module`` and ``Optimizer`` instances are not pytree-able.
    ``graph_state`` supplies additional named tensors as explicit, stable graph
    inputs. The traced function does not receive them directly; graph passes may
    consume their placeholders to add stateful operations.
    ``graph_state_output_indices`` explicitly maps each graph-state tensor to a
    logical output whose leaves will be accumulated into it.

    The trace-time ``args`` and ``kwargs`` must satisfy these constraints:

    - all pytree leaves must be tensors or make_fx-safe primitives
      (``int``, ``float``, ``bool``, ``str``, ``None``)
    - there must be no ``nn.Module`` instances in ``args`` or ``kwargs``

    Tensor subclasses (for example ``DTensor``) are recursively unwrapped into
    plain tensors for tracing, and the layouts needed to rewrap them are stored
    in the returned :class:`TracedResult`.

    ``_insert_runtime_asserts`` opts into materializing the ShapeEnv's deferred
    runtime asserts (from ``mark_unbacked()`` bounds and ``torch._check()``
    calls) into the graph as ``_assert_scalar`` nodes. Off by default because
    cudagraph capture does not evaluate these nodes, and downstream graph
    passes generally don't need them.

    ``record_stack_traces`` controls whether make_fx records Python stack traces
    in node metadata. It defaults to on to preserve the existing debugging
    behavior.
    """
    _check_optimizer_has_module(module, optimizer)

    def _trace_with_args(*args: Any, **kwargs: Any) -> TracedResult:
        if prepare_inputs is not None:
            prepare_inputs(args, kwargs)

        model_state, optim_state = extract_train_state(module, optimizer)
        state_fqns = list(model_state.keys())
        graph_state_t = graph_state or {}
        graph_state_fqns = list(graph_state_t.keys())
        graph_output_indices = tuple(graph_state_output_indices)
        if graph_output_indices and len(graph_output_indices) != len(graph_state_fqns):
            raise ValueError(
                "minimal_fx_tracer requires one graph_state_output_indices entry "
                "per graph_state tensor"
            )

        model_state_flat, model_state_spec = pytree.tree_flatten(model_state)
        graph_state_flat, graph_state_spec = pytree.tree_flatten(graph_state_t)
        optim_state_flat, optim_state_spec = pytree.tree_flatten(optim_state)
        num_model_state_inputs = len(model_state_flat)
        num_graph_state_inputs = len(graph_state_flat)
        num_optim_state_inputs = len(optim_state_flat)

        if any(not isinstance(value, torch.Tensor) for value in graph_state_flat):
            raise ValueError("minimal_fx_tracer graph_state values must be tensors")

        user_inputs_flat, user_inputs_spec = pytree.tree_flatten((args, kwargs))

        # Validate leaves.
        for leaf in [
            *model_state_flat,
            *graph_state_flat,
            *optim_state_flat,
            *user_inputs_flat,
        ]:
            if isinstance(leaf, nn.Module):
                raise ValueError(
                    "minimal_fx_tracer requires explicit tensor state, not nn.Module "
                    "instances. Capture nn.Modules in fn's closure or pass them "
                    "via the 'module' kwarg."
                )
            if not isinstance(leaf, _ALLOWED_LEAF_TYPES):
                raise ValueError(
                    "minimal_fx_tracer requires all pytree leaves in state/args to "
                    f"be tensors or primitives (int/float/bool/str), got "
                    f"{type(leaf).__name__}. Non-primitive values should either be "
                    "registered as pytree nodes (register_pytree_node) or constants "
                    f"(pytree.register_constant), or captured in fn's closure."
                )

        # Graph state follows model state so both form one static input prefix.
        full_args = (
            list(model_state_flat)
            + list(graph_state_flat)
            + list(optim_state_flat)
            + list(user_inputs_flat)
        )
        num_full_args = len(full_args)
        for arg in full_args:
            if not isinstance(arg, torch.Tensor):
                continue
            if _wrapper_subclass_has_marked_dynamic_dims(arg):
                raise ValueError(
                    "minimal_fx_tracer only supports marked dynamic dims on plain "
                    "tensor inputs; wrapper subclasses such as DTensor are not "
                    "supported"
                )
        unwrapped_args, input_layouts = _unwrap_subclasses(full_args)
        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=torch.fx.experimental.symbolic_shapes.ShapeEnv(),
        )
        fake_args = tuple(
            _fakeify_input(fake_mode, a, input_name=f"input_{i}")
            if isinstance(a, torch.Tensor)
            else a
            for i, a in enumerate(unwrapped_args)
        )

        graph_state_input_indices: list[tuple[int, ...]] = []
        flat_index = 0
        graph_state_start = num_model_state_inputs
        graph_state_end = graph_state_start + num_graph_state_inputs
        for logical_index in range(num_full_args):
            num_tensors = (
                input_layouts[logical_index].num_tensors
                if logical_index in input_layouts
                else 1
            )
            indices = tuple(range(flat_index, flat_index + num_tensors))
            if graph_state_start <= logical_index < graph_state_end:
                graph_state_input_indices.append(indices)
            flat_index += num_tensors

        output_layouts: dict[int, SubclassLayout] = {}
        num_flat_outputs: int = 0
        output_spec: pytree.TreeSpec | None = None

        def fn_with_subclass_handling(*plain_args: Any) -> list:
            nonlocal output_layouts, output_spec, num_flat_outputs
            output_layouts = {}

            wrapped = _wrap_subclasses(plain_args, num_full_args, input_layouts)
            model_state_end = num_model_state_inputs
            graph_state_end = model_state_end + num_graph_state_inputs
            optim_state_end = graph_state_end + num_optim_state_inputs
            model_state_t = pytree.tree_unflatten(
                list(wrapped[:model_state_end]), model_state_spec
            )
            # Graph-state inputs are consumed by post-trace graph transforms.
            # Reconstruct them here to preserve their pytree input contract.
            _ = pytree.tree_unflatten(
                list(wrapped[model_state_end:graph_state_end]), graph_state_spec
            )
            optim_state_t = pytree.tree_unflatten(
                list(wrapped[graph_state_end:optim_state_end]), optim_state_spec
            )
            user_flat = wrapped[optim_state_end:]
            user_args, user_kwargs = pytree.tree_unflatten(
                list(user_flat), user_inputs_spec
            )
            if prepare_call_inputs is not None:
                prepared = prepare_call_inputs(user_args, user_kwargs)
                if prepared is not None:
                    user_args, user_kwargs = prepared

            with _reparametrize_train_state(
                module, optimizer, model_state_t, optim_state_t
            ), torch.compiler._patch_engine_backward():
                result = fn(*user_args, **user_kwargs)

            flat_outs, output_spec = pytree.tree_flatten(result)
            num_flat_outputs = len(flat_outs)
            unwrapped_outs, output_layouts = _unwrap_subclasses(flat_outs)
            return unwrapped_outs

        ctx = TracingContext(fake_mode)
        # preserve_node_meta propagates fx.traceback.annotate metadata to traced nodes
        # Disable autograd multithreading so that backward tracing
        # runs on the calling thread. Without this, the C++ autograd
        # engine dispatches backward to a worker thread that has a
        # fresh contextvars.Context, making the compile_on_one_rank
        # ContextVar invisible and causing _sym_get_coordinate to
        # bake rank 0's concrete coordinates into the backward graph.
        # TODO: Move set_multithreading_enabled(False) to global init.
        # Forcing backward onto the main CPU thread is a good default
        # for both tracing and runtime, not just the tracing path.
        # _skip_nested_compile lets the current make_fx trace inline through
        # torch.compile'd FlexAttention kernels instead of erroring.
        # _non_strict_tracing_context is required by _patch_autograd_grad() and
        # marks this make_fx pass as the non-strict tracing flow, distinct from
        # other make_fx-based entry points such as non-strict export.
        with (
            fake_mode,
            tracing(ctx),
            preserve_node_meta(),
            _skip_nested_compile(),
            torch.autograd.set_multithreading_enabled(False),
            torch.compiler._non_strict_tracing_context(),
        ):
            traced = make_fx(
                fn_with_subclass_handling,
                record_stack_traces=record_stack_traces,
                record_module_stack=False,  # don't need nn_module_stack for now
            )(*fake_args)

        # Copy forward annotations to backward nodes.
        _copy_fwd_metadata_to_bw_nodes(traced)

        if _insert_runtime_asserts:
            _insert_runtime_asserts_pass(traced, fake_mode)

        assert output_spec is not None
        if graph_output_indices:
            _tag_graph_state_outputs(
                traced,
                graph_state_fqns,
                graph_output_indices,
                num_flat_outputs,
                output_layouts,
            )
        return TracedResult(
            gm=traced,
            example_inputs=fake_args,
            num_flat_inputs=num_full_args,
            input_subclass_layouts=input_layouts,
            user_inputs_spec=user_inputs_spec,
            tensor_input_indices=[
                i for i, x in enumerate(fake_args) if isinstance(x, torch.Tensor)
            ],
            num_flat_outputs=num_flat_outputs,
            output_subclass_layouts=output_layouts,
            output_spec=output_spec,
            state_fqns=state_fqns,
            graph_state_fqns=graph_state_fqns,
            graph_state_input_indices=tuple(graph_state_input_indices),
            graph_state_output_indices=graph_output_indices,
            grad_sink_active=False,
            num_optimizer_state_inputs=num_optim_state_inputs,
        )

    return _trace_with_args


def run_traced(
    traced_result: TracedResult,
    *,
    module: nn.Module | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
    graph_state: dict[str, torch.Tensor] | None = None,
    _validate_runtime: bool = False,
    interpreter_cls: type | None = None,
) -> Callable[..., Any]:
    """Return a runner that executes a traced graph against live module/optimizer state.

    The returned callable takes user-facing args and kwargs::

        traced = minimal_fx_tracer(fn, module=model, optimizer=opt)(*args, **kwargs)
        outputs = run_traced(traced, module=model, optimizer=opt)(*args, **kwargs)

    Mirrors :func:`minimal_fx_tracer`'s state extraction: parameters/buffers
    are sampled from ``module``, ``graph_state`` supplies separately owned
    persistent tensors, and optimizer state is sampled from
    ``optimizer.state_dict()``. Runs under ``torch.no_grad()`` because the graph
    already contains explicit backward ops (from ``torch.autograd.grad`` traced
    by make_fx). Without this, PyTorch would build a redundant autograd graph on
    top, keeping all forward intermediates alive via ``grad_fn`` references.

    With ``_validate_runtime=True``, runtime module parameter/buffer FQNs must match
    trace time and runtime ``(args, kwargs)`` must flatten to the same pytree
    spec as trace time; any mismatch raises. Off by default to keep the
    per-step path overhead-free; the caller must pass kwargs in trace-time
    order.

    If ``interpreter_cls`` is provided, the traced graph is executed via that
    FX interpreter instead of called directly; used by activation tracing.
    """
    _check_optimizer_has_module(module, optimizer)

    def _run(*args: Any, **kwargs: Any) -> Any:
        model_state, optim_state = extract_train_state(module, optimizer)
        if _validate_runtime and list(model_state.keys()) != traced_result.state_fqns:
            raise ValueError(
                "module has different parameter/buffer names than during tracing.\n"
                f"  Traced: {traced_result.state_fqns}\n"
                f"  Got:    {list(model_state.keys())}"
            )
        graph_state_t = graph_state or {}
        if list(graph_state_t.keys()) != traced_result.graph_state_fqns:
            raise ValueError(
                "graph state has different names than during tracing.\n"
                f"  Traced: {traced_result.graph_state_fqns}\n"
                f"  Got:    {list(graph_state_t.keys())}"
            )
        model_state_flat, _ = pytree.tree_flatten(model_state)
        graph_state_flat, _ = pytree.tree_flatten(graph_state_t)
        optim_state_flat, _ = pytree.tree_flatten(optim_state)

        user_inputs_flat, runtime_spec = pytree.tree_flatten((args, kwargs))
        # TODO: pytree's dict flatten preserves insertion order, so kwargs in a
        # different order than trace produce a different spec even though they
        # describe the same logical inputs. If pytree sorted dict keys (or
        # provided a canonicalizing flatten), this check could match valid
        # reordered calls without needing an explicit reorder step here.
        if _validate_runtime and runtime_spec != traced_result.user_inputs_spec:
            raise ValueError(
                f"input spec mismatch: runtime {runtime_spec} != "
                f"trace-time {traced_result.user_inputs_spec}"
            )
        if any(
            isinstance(leaf, nn.Module)
            for leaf in [
                *model_state_flat,
                *graph_state_flat,
                *optim_state_flat,
                *user_inputs_flat,
            ]
        ):
            raise ValueError(
                "run_traced requires explicit tensor state, not nn.Module instances. "
                "Capture nn.Modules in fn's closure or pass them via the 'module' kwarg."
            )
        all_args = (
            list(model_state_flat)
            + list(graph_state_flat)
            + list(optim_state_flat)
            + list(user_inputs_flat)
        )
        flat_inputs, _ = _unwrap_subclasses(all_args)

        with torch.no_grad():
            if interpreter_cls is not None:
                flat_outputs = interpreter_cls(traced_result.gm).run(*flat_inputs)
            else:
                flat_outputs = traced_result.gm(*flat_inputs)
        wrapped = _wrap_subclasses(
            flat_outputs,
            traced_result.num_flat_outputs,
            traced_result.output_subclass_layouts,
        )
        return pytree.tree_unflatten(wrapped, traced_result.output_spec)

    return _run


def _bound_static_values(
    traced_result: TracedResult,
    module: nn.Module | None,
    graph_state: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, ...]:
    model_state, _ = extract_train_state(module)
    if list(model_state) != traced_result.state_fqns:
        raise ValueError(
            "module has different parameter/buffer names than during tracing.\n"
            f"  Traced: {traced_result.state_fqns}\n"
            f"  Got:    {list(model_state)}"
        )
    graph_state_t = graph_state or {}
    if list(graph_state_t) != traced_result.graph_state_fqns:
        raise ValueError(
            "graph state has different names than during tracing.\n"
            f"  Traced: {traced_result.graph_state_fqns}\n"
            f"  Got:    {list(graph_state_t)}"
        )
    return tuple([*model_state.values(), *graph_state_t.values()])


def _flat_input_metadata(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, torch.Tensor):
        return (type(value),)
    stride = value.stride() if value.layout == torch.strided else None
    storage_offset = value.storage_offset() if value.layout == torch.strided else None
    return (
        value.dtype,
        value.device,
        value.layout,
        value.size(),
        stride,
        storage_offset,
        value.requires_grad,
    )


def _flat_input_binding(value: Any) -> tuple[Any, ...]:
    storage_ptr = None
    if (
        isinstance(value, torch.Tensor)
        and not isinstance(value, FakeTensor)
        and value.device.type != "meta"
        and value.layout == torch.strided
    ):
        storage_ptr = value.untyped_storage().data_ptr()
    return (*_flat_input_metadata(value), storage_ptr)


def _flatten_bound_state(
    traced_result: TracedResult,
    static_values: tuple[torch.Tensor, ...],
) -> tuple[Any, ...]:
    flat_inputs, layouts = _unwrap_subclasses(list(static_values))
    expected_layouts = {
        index: layout
        for index, layout in traced_result.input_subclass_layouts.items()
        if index < len(static_values)
    }
    if layouts != expected_layouts:
        raise ValueError(
            "bound state has a different tensor-subclass layout than during tracing"
        )
    if traced_result.example_inputs:
        expected_inputs = traced_result.example_inputs[
            : traced_result.num_static_inputs
        ]
    else:
        # Serialized artifacts may omit placeholder values. Their configuration
        # fingerprint validates model shapes, dtypes, and parallel dimensions.
        expected_inputs = tuple(
            node.meta.get("val")
            for node in traced_result.gm.graph.nodes
            if node.op == "placeholder"
        )[: traced_result.num_static_inputs]
    if traced_result.example_inputs and len(expected_inputs) != len(flat_inputs):
        raise ValueError(
            "bound state has a different flattened arity than during tracing"
        )
    metadata_available = len(expected_inputs) == len(flat_inputs) and all(
        expected is not None for expected in expected_inputs
    )
    if metadata_available and any(
        _flat_input_metadata(actual) != _flat_input_metadata(expected)
        for actual, expected in zip(flat_inputs, expected_inputs, strict=True)
    ):
        raise ValueError(
            "bound state has different tensor metadata than during tracing"
        )
    return tuple(flat_inputs)


class BoundTracedRunner:
    """Run a traced graph against lifetime-bound module and graph state."""

    def __init__(
        self,
        traced_result: TracedResult,
        *,
        module: nn.Module | None,
        graph_state: dict[str, torch.Tensor] | None,
        validate_runtime: bool,
        interpreter_cls: type | None,
    ) -> None:
        if traced_result.num_optimizer_state_inputs != 0:
            raise ValueError("bind_traced does not support traced optimizer state")
        self._traced_result = traced_result
        self._static_values = _bound_static_values(traced_result, module, graph_state)
        self._flat_static_inputs = _flatten_bound_state(
            traced_result, self._static_values
        )
        self._flat_static_bindings = tuple(
            _flat_input_binding(value) for value in self._flat_static_inputs
        )
        self._validate_runtime = validate_runtime
        self._interpreter_cls = interpreter_cls

    def validate_state(
        self,
        *,
        module: nn.Module | None,
        graph_state: dict[str, torch.Tensor] | None,
    ) -> None:
        """Fail if bound state objects, subclass leaves, or storage changed."""
        try:
            static_values = _bound_static_values(
                self._traced_result, module, graph_state
            )
            flat_static_inputs = _flatten_bound_state(
                self._traced_result, static_values
            )
        except ValueError as error:
            raise RuntimeError("bound traced state changed after binding") from error
        if any(
            actual is not expected
            for actual, expected in zip(static_values, self._static_values, strict=True)
        ) or any(
            actual is not expected
            for actual, expected in zip(
                flat_static_inputs, self._flat_static_inputs, strict=True
            )
        ):
            raise RuntimeError("bound traced state objects changed after binding")
        if tuple(_flat_input_binding(value) for value in flat_static_inputs) != (
            self._flat_static_bindings
        ):
            raise RuntimeError("bound traced state storage changed after binding")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        user_inputs_flat, runtime_spec = pytree.tree_flatten((args, kwargs))
        if self._validate_runtime and runtime_spec != (
            self._traced_result.user_inputs_spec
        ):
            raise ValueError(
                f"input spec mismatch: runtime {runtime_spec} != "
                f"trace-time {self._traced_result.user_inputs_spec}"
            )
        if any(isinstance(leaf, nn.Module) for leaf in user_inputs_flat):
            raise ValueError(
                "bind_traced requires explicit tensor inputs, not nn.Module "
                "instances. Capture nn.Modules in fn's closure or bind them "
                "through the 'module' kwarg."
            )
        flat_user_inputs, _ = _unwrap_subclasses(user_inputs_flat)
        flat_inputs = [*self._flat_static_inputs, *flat_user_inputs]

        with torch.no_grad():
            if self._interpreter_cls is not None:
                flat_outputs = self._interpreter_cls(self._traced_result.gm).run(
                    *flat_inputs
                )
            else:
                flat_outputs = self._traced_result.gm(*flat_inputs)
        wrapped = _wrap_subclasses(
            flat_outputs,
            self._traced_result.num_flat_outputs,
            self._traced_result.output_subclass_layouts,
        )
        return pytree.tree_unflatten(wrapped, self._traced_result.output_spec)


def bind_traced(
    traced_result: TracedResult,
    *,
    module: nn.Module | None = None,
    graph_state: dict[str, torch.Tensor] | None = None,
    _validate_runtime: bool = False,
    interpreter_cls: type | None = None,
) -> BoundTracedRunner:
    """Bind stable module and graph state to a traced graph runner.

    Optimizer updates remain visible because the binding retains references to
    live tensor storage. Call :meth:`BoundTracedRunner.validate_state` at state
    mutation boundaries, and create a new trace/binding after replacement.
    """
    return BoundTracedRunner(
        traced_result,
        module=module,
        graph_state=graph_state,
        validate_runtime=_validate_runtime,
        interpreter_cls=interpreter_cls,
    )
