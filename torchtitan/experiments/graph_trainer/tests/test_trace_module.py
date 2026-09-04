# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import weakref
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from torch._inductor.fx_passes.bucketing import (
    _register_fsdp_symmetric_buffer,
    has_preallocated_fsdp_symmetric_memory_buffers,
    release_preallocated_fsdp_symmetric_memory_buffers,
)
from torch.distributed.tensor import DTensor
from torch.optim import swap_in_optimizer_params_and_state
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    _maybe_materialize_grad_for_param_layout,
    accumulate_param_grads_,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.cudagraph import (
    cudagraph_pass,
    CUDAGraphWrapper,
)
from torchtitan.experiments.graph_trainer.deferred_fsdp import (
    bind_deferred_fsdp_graph,
    build_deferred_fsdp_graph,
)
from torchtitan.experiments.graph_trainer.gradient_accumulation import (
    _accumulation_leaf_offsets,
    _validate_device_mesh_leaf,
    finalize_graph_gradient_accumulation,
    GraphGradientState,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import flatten_graph_values
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _copy_fwd_metadata_to_bw_nodes,
    bind_traced,
    extract_module_state,
    minimal_fx_tracer,
    run_traced,
    SubclassLayout,
    SubclassMeta,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    annotate_flex_attention_for_regional_inductor_pass,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.trainer import Trainer


def get_loss(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        reduction="sum",
    )


def make_train_step(model, loss_fn):
    """Return a plain function that closes over ``model`` for module-based tracing."""

    def train_step(*args):
        *fwd_args, labels = args
        logits = model(*fwd_args)
        loss = loss_fn(logits, labels)
        params = list(model.parameters())
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)

    return train_step


def create_model(config_cls, model_config, device="cuda", dtype=torch.float32):
    model = config_cls(model_config)
    model.to(device=device, dtype=dtype)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device(device))
    return model


def _apply_regional_inductor(traced_result):
    """Apply regional_inductor to compile annotated HOP regions in the traced graph."""
    from torch.fx.graph import CodeGen
    from torch.fx.passes.regional_inductor import regional_inductor

    from torchtitan.models.common.attention import FlexAttention

    annotate_flex_attention_for_regional_inductor_pass(
        traced_result.gm,
        flex_compile_config=FlexAttention.inductor_configs,
    )

    fake_inputs = _graph_placeholder_fake_inputs(traced_result.gm)
    fake_mode = _graph_fake_mode(fake_inputs)
    with torch._guards.tracing(torch._guards.TracingContext(fake_mode)):
        traced_result.gm = regional_inductor(traced_result.gm)

    traced_result.gm.graph.set_codegen(CodeGen())
    traced_result.gm.recompile()


def _graph_placeholder_fake_inputs(gm):
    fake_inputs = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val")
        if val is None:
            raise RuntimeError(f"Missing placeholder meta val for {node}")
        fake_inputs.append(val)
    return fake_inputs


def _graph_fake_mode(fake_inputs):
    return next(
        (
            val.fake_mode
            for val in fake_inputs
            if isinstance(val, torch.Tensor) and hasattr(val, "fake_mode")
        ),
        None,
    )


class TestGraphTrainerSymmetricMemoryTeardown(unittest.TestCase):
    def _identity_graph(self):
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        graph.output(value)
        return torch.fx.GraphModule(nn.Module(), graph)

    def _trainer_with_static_slabs(self):
        normal = self._identity_graph()
        child = self._identity_graph()
        with patch.object(
            torch._C._distributed_c10d._SymmetricMemory,
            "empty_strided_p2p",
            side_effect=lambda *args, **kwargs: torch.empty(4),
        ):
            _register_fsdp_symmetric_buffer(normal, 4, torch.empty(1), 0)
            _register_fsdp_symmetric_buffer(child, 4, torch.empty(1), 0)

        root = nn.Module()
        root.add_module("child", child)
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        result = graph.call_module("child", args=(value,))
        graph.output(result)
        deferred = torch.fx.GraphModule(root, graph)

        trainer = object.__new__(GraphTrainer)
        trainer._pinned_pool_ctx = None
        trainer._traced_step = SimpleNamespace(gm=normal)
        trainer._bound_traced_step = object()
        trainer._deferred_fsdp_graph = SimpleNamespace(gm=deferred)
        trainer._bound_deferred_fsdp_graph = object()
        return trainer, normal, deferred

    def test_close_releases_normal_and_deferred_slabs_before_base_close(self):
        trainer, normal, deferred = self._trainer_with_static_slabs()
        buffer_refs = [
            weakref.ref(buffer)
            for gm in (normal, deferred)
            for _, buffer in gm.named_buffers()
        ]
        events = []

        def release(gm):
            events.append("release")
            return release_preallocated_fsdp_symmetric_memory_buffers(gm)

        with (
            patch(
                "torchtitan.experiments.graph_trainer.trainer.cudagraph_teardown",
                side_effect=lambda: events.append("cudagraph_teardown"),
            ),
            patch.object(
                torch.cuda,
                "synchronize",
                side_effect=lambda: events.append("synchronize"),
            ),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(
                torch.distributed,
                "get_backend",
                return_value=torch.distributed.Backend.NCCL,
            ),
            patch.object(
                torch.distributed,
                "barrier",
                side_effect=lambda **kwargs: events.append("barrier"),
            ),
            patch(
                "torchtitan.experiments.graph_trainer.trainer."
                "release_preallocated_fsdp_symmetric_memory_buffers",
                side_effect=release,
            ),
            patch.object(
                Trainer,
                "close",
                autospec=True,
                side_effect=lambda _: events.append("base_close"),
            ),
        ):
            trainer.close()
            self.assertEqual(
                events,
                [
                    "cudagraph_teardown",
                    "synchronize",
                    "barrier",
                    "release",
                    "release",
                    "synchronize",
                    "barrier",
                    "base_close",
                ],
            )
            events.clear()
            trainer.close()
            self.assertEqual(events, ["cudagraph_teardown", "base_close"])

        self.assertFalse(has_preallocated_fsdp_symmetric_memory_buffers(normal))
        self.assertFalse(has_preallocated_fsdp_symmetric_memory_buffers(deferred))
        self.assertTrue(all(buffer_ref() is None for buffer_ref in buffer_refs))
        self.assertIsNone(trainer._traced_step)
        self.assertIsNone(trainer._bound_traced_step)
        self.assertIsNone(trainer._deferred_fsdp_graph)
        self.assertIsNone(trainer._bound_deferred_fsdp_graph)

    def test_release_skips_barriers_without_real_process_group(self):
        for initialized, backend in (
            (False, None),
            (True, torch.distributed.Backend.FAKE),
        ):
            with self.subTest(initialized=initialized, backend=backend):
                trainer, normal, _ = self._trainer_with_static_slabs()
                with (
                    patch.object(
                        torch.distributed,
                        "is_initialized",
                        return_value=initialized,
                    ),
                    patch.object(
                        torch.distributed,
                        "get_backend",
                        return_value=backend,
                    ),
                    patch.object(torch.distributed, "barrier") as barrier,
                    patch.object(torch.cuda, "synchronize") as synchronize,
                ):
                    trainer._release_fsdp_symmetric_memory_buffers()
                barrier.assert_not_called()
                synchronize.assert_not_called()
                self.assertFalse(has_preallocated_fsdp_symmetric_memory_buffers(normal))

    def test_release_skips_barriers_without_static_slabs(self):
        trainer = object.__new__(GraphTrainer)
        trainer._traced_step = SimpleNamespace(gm=self._identity_graph())
        trainer._deferred_fsdp_graph = None
        with (
            patch.object(torch.distributed, "is_initialized") as initialized,
            patch.object(torch.distributed, "barrier") as barrier,
            patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            trainer._release_fsdp_symmetric_memory_buffers()
        initialized.assert_not_called()
        barrier.assert_not_called()
        synchronize.assert_not_called()


class SimpleMLP(nn.Module):
    def __init__(self, dim=64, hidden=128, vocab_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(self.embed(x))))


class _TraceableWrapper(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem):
        wrapper = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
        )
        wrapper.elem = elem
        return wrapper

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise RuntimeError("Test wrapper should be rejected before tracing")

    def __tensor_flatten__(self):
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return _TraceableWrapper(inner_tensors["elem"])


class TestBoundTracedRunner(unittest.TestCase):
    def test_observes_in_place_parameter_updates_without_resampling_state(self):
        class CountingLinear(nn.Linear):
            def __init__(self):
                super().__init__(3, 2, dtype=torch.float64)
                self.named_parameters_calls = 0

            def named_parameters(self, *args, **kwargs):
                self.named_parameters_calls += 1
                return super().named_parameters(*args, **kwargs)

        torch.manual_seed(42)
        model = CountingLinear()
        inputs = torch.randn(4, 3, dtype=torch.float64)

        def forward(value):
            return model(value)

        traced = minimal_fx_tracer(forward, module=model)(inputs)
        run = bind_traced(traced, module=model)
        calls_after_bind = model.named_parameters_calls
        initial_output = run(inputs)

        with torch.no_grad():
            model.weight.add_(0.25)
            model.bias.sub_(0.5)

        expected = model(inputs)
        actual = run(inputs)

        self.assertFalse(torch.equal(initial_output, expected))
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(calls_after_bind, model.named_parameters_calls)

    def test_rejects_incompatible_module_at_bind_time(self):
        model = nn.Linear(3, 2)
        inputs = torch.randn(4, 3)

        def forward(value):
            return model(value)

        traced = minimal_fx_tracer(forward, module=model)(inputs)
        incompatible_model = nn.Sequential(nn.Linear(3, 2))

        with self.assertRaisesRegex(ValueError, "parameter/buffer names"):
            bind_traced(traced, module=incompatible_model)

    def test_validation_rejects_parameter_and_storage_replacement(self):
        model = nn.Linear(3, 2)
        inputs = torch.randn(4, 3)

        def forward(value):
            return model(value)

        traced = minimal_fx_tracer(forward, module=model)(inputs)
        run = bind_traced(traced, module=model)
        original_weight = model.weight
        model.weight = nn.Parameter(model.weight.detach().clone())
        with self.assertRaisesRegex(RuntimeError, "state objects changed"):
            run.validate_state(module=model, graph_state=None)

        model.weight = original_weight
        run = bind_traced(traced, module=model)
        with torch.no_grad():
            model.weight.set_(model.weight.detach().clone())
        with self.assertRaisesRegex(RuntimeError, "state storage changed"):
            run.validate_state(module=model, graph_state=None)

    def test_validation_allows_in_place_buffer_updates(self):
        class BufferedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.ones(3))

            def forward(self, value):
                return value + self.offset

        model = BufferedModule()
        inputs = torch.randn(4, 3)
        traced = minimal_fx_tracer(model.forward, module=model)(inputs)
        run = bind_traced(traced, module=model)

        model.offset.add_(2)
        run.validate_state(module=model, graph_state=None)
        self.assertTrue(torch.equal(model(inputs), run(inputs)))

        model.offset = model.offset.clone()
        with self.assertRaisesRegex(RuntimeError, "state objects changed"):
            run.validate_state(module=model, graph_state=None)

    def test_validation_rejects_graph_state_replacement(self):
        graph_state = {"accumulator": torch.zeros(3)}
        inputs = torch.randn(4, 3)

        def forward(value):
            return value.sin()

        traced = minimal_fx_tracer(forward, graph_state=graph_state)(inputs)
        run = bind_traced(traced, graph_state=graph_state)
        graph_state["accumulator"] = torch.ones(3)

        with self.assertRaisesRegex(RuntimeError, "state objects changed"):
            run.validate_state(module=None, graph_state=graph_state)

    def test_rejects_traced_optimizer_state(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        inputs = torch.randn(4, 3)
        model(inputs).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)

        def forward(value):
            return model(value)

        traced = minimal_fx_tracer(
            forward,
            module=model,
            optimizer=optimizer,
        )(inputs)

        with self.assertRaisesRegex(ValueError, "optimizer state"):
            bind_traced(traced, module=model)


class TestGraphGradientAccumulation(unittest.TestCase):
    class _OptimizerCollection:
        def __init__(self, optimizer):
            self.optimizers = [optimizer]

        def __iter__(self):
            return iter(self.optimizers)

        def zero_grad(self, *, set_to_none):
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=set_to_none)

        def step(self):
            for optimizer in self.optimizers:
                optimizer.step()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_graph_accumulates_each_call_exactly_once(self):
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        model_ref = nn.Linear(3, 2, device=device)
        model_test = deepcopy(model_ref)
        optimizer = torch.optim.SGD(model_test.parameters(), lr=0.1)
        gradient_state = GraphGradientState.create(model_test, [optimizer])

        def train_step(inputs, targets):
            loss = torch.nn.functional.mse_loss(
                model_test(inputs),
                targets,
                reduction="sum",
            )
            grads = torch.autograd.grad(loss, tuple(model_test.parameters()))
            return [loss, *grads]

        microbatches = [
            (
                torch.randn(4, 3, device=device),
                torch.randn(4, 2, device=device),
            )
            for _ in range(3)
        ]
        traced = minimal_fx_tracer(
            train_step,
            module=model_test,
            graph_state=gradient_state.graph_state,
            graph_state_output_indices=(1, 2),
        )(*microbatches[0])
        traced.gm = finalize_graph_gradient_accumulation(
            traced.gm,
            traced_result=traced,
        )
        traced.gm = cudagraph_pass(
            traced.gm,
            traced.example_inputs,
            static_input_indices=list(range(traced.num_static_inputs)),
            tensor_input_indices=traced.tensor_input_indices,
            require=True,
        )
        self.assertIsInstance(traced.gm.forward, CUDAGraphWrapper)
        run = bind_traced(
            traced,
            module=model_test,
            graph_state=gradient_state.graph_state,
        )

        for inputs, targets in microbatches:
            loss_ref = torch.nn.functional.mse_loss(
                model_ref(inputs),
                targets,
                reduction="sum",
            )
            loss_ref.backward()
            outputs = run(inputs, targets)
            torch.cuda.synchronize()

            self.assertEqual(len(outputs), 1)
            torch.testing.assert_close(outputs[0], loss_ref)
            for parameter_ref, buffer in zip(
                model_ref.parameters(), gradient_state.buffers, strict=True
            ):
                torch.testing.assert_close(buffer, parameter_ref.grad)

        traced.gm.forward.teardown()

    def test_terminal_sink_accumulates_and_preserves_optimizer_grad_buffers(self):
        torch.manual_seed(42)
        model_ref = nn.Linear(3, 2, dtype=torch.float64)
        model_test = deepcopy(model_ref)
        optimizer_ref = torch.optim.SGD(model_ref.parameters(), lr=0.05)
        optimizer_test = torch.optim.SGD(model_test.parameters(), lr=0.05)
        gradient_state = GraphGradientState.create(model_test, [optimizer_test])

        def train_step(inputs, targets):
            predictions = model_test(inputs)
            loss = torch.nn.functional.mse_loss(
                predictions,
                targets,
                reduction="sum",
            )
            grads = torch.autograd.grad(loss, tuple(model_test.parameters()))
            return [loss, *grads]

        microbatches = [
            (
                torch.tensor(
                    [[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]],
                    dtype=torch.float64,
                ),
                torch.tensor([[0.5, -1.0], [2.0, 0.25]], dtype=torch.float64),
            ),
            (
                torch.tensor(
                    [[-0.5, 1.0, 2.0], [1.25, -0.75, 0.5]],
                    dtype=torch.float64,
                ),
                torch.tensor([[-0.25, 1.5], [0.75, -2.0]], dtype=torch.float64),
            ),
            (
                torch.tensor(
                    [[2.0, 0.5, -1.5], [-1.0, 0.75, 1.25]],
                    dtype=torch.float64,
                ),
                torch.tensor([[1.0, 0.0], [-0.5, 2.5]], dtype=torch.float64),
            ),
            (
                torch.tensor(
                    [[0.75, -1.25, 1.0], [-2.0, 0.5, 0.25]],
                    dtype=torch.float64,
                ),
                torch.tensor([[1.25, -0.75], [0.5, 1.0]], dtype=torch.float64),
            ),
        ]
        traced = minimal_fx_tracer(
            train_step,
            module=model_test,
            graph_state=gradient_state.graph_state,
            graph_state_output_indices=tuple(range(1, len(gradient_state.buffers) + 1)),
        )(*microbatches[0])
        traced.gm = finalize_graph_gradient_accumulation(
            traced.gm,
            traced_result=traced,
        )
        run = bind_traced(
            traced,
            module=model_test,
            graph_state=gradient_state.graph_state,
        )

        parameters_test = tuple(model_test.parameters())
        grad_ids = tuple(id(parameter.grad) for parameter in parameters_test)
        grad_data_ptrs = tuple(
            parameter.grad.data_ptr() for parameter in parameters_test
        )
        self.assertTrue(traced.grad_sink_active)
        self.assertTrue(
            all(torch.count_nonzero(buffer) == 0 for buffer in gradient_state.buffers)
        )
        self.assertEqual(
            len(
                [
                    node
                    for node in traced.gm.graph.nodes
                    if "graph_gradient_fqn" in node.meta
                ]
            ),
            len(parameters_test),
        )

        for inputs, targets in microbatches[:3]:
            loss_ref = torch.nn.functional.mse_loss(
                model_ref(inputs),
                targets,
                reduction="sum",
            )
            loss_ref.backward()
            outputs = run(inputs, targets)

            self.assertEqual(len(outputs), 1)
            torch.testing.assert_close(outputs[0], loss_ref)
            for parameter_ref, parameter_test, buffer in zip(
                model_ref.parameters(),
                parameters_test,
                gradient_state.buffers,
                strict=True,
            ):
                self.assertIs(parameter_test.grad, buffer)
                torch.testing.assert_close(parameter_test.grad, parameter_ref.grad)
            self.assertEqual(
                tuple(id(parameter.grad) for parameter in parameters_test),
                grad_ids,
            )
            self.assertEqual(
                tuple(parameter.grad.data_ptr() for parameter in parameters_test),
                grad_data_ptrs,
            )

        optimizer_ref.step()
        optimizer_test.step()
        for parameter_ref, parameter_test in zip(
            model_ref.parameters(), model_test.parameters(), strict=True
        ):
            torch.testing.assert_close(parameter_test, parameter_ref)

        optimizer_ref.zero_grad(set_to_none=False)
        optimizer_test.zero_grad(set_to_none=False)
        self.assertTrue(
            all(torch.count_nonzero(buffer) == 0 for buffer in gradient_state.buffers)
        )
        self.assertEqual(
            tuple(id(parameter.grad) for parameter in parameters_test),
            grad_ids,
        )
        self.assertEqual(
            tuple(parameter.grad.data_ptr() for parameter in parameters_test),
            grad_data_ptrs,
        )

        inputs, targets = microbatches[3]
        loss_ref = torch.nn.functional.mse_loss(
            model_ref(inputs),
            targets,
            reduction="sum",
        )
        loss_ref.backward()
        outputs = run(inputs, targets)
        self.assertEqual(len(outputs), 1)
        torch.testing.assert_close(outputs[0], loss_ref)
        for parameter_ref, parameter_test, buffer in zip(
            model_ref.parameters(),
            parameters_test,
            gradient_state.buffers,
            strict=True,
        ):
            self.assertIs(parameter_test.grad, buffer)
            torch.testing.assert_close(parameter_test.grad, parameter_ref.grad)
        self.assertEqual(
            tuple(parameter.grad.data_ptr() for parameter in parameters_test),
            grad_data_ptrs,
        )

        optimizer_ref.step()
        optimizer_test.step()
        for parameter_ref, parameter_test in zip(
            model_ref.parameters(), model_test.parameters(), strict=True
        ):
            torch.testing.assert_close(parameter_test, parameter_ref)

    def test_trainer_accumulates_three_microbatches_across_optimizer_steps(self):
        torch.manual_seed(42)
        model_ref = nn.Linear(3, 2, dtype=torch.float64)
        model_test = deepcopy(model_ref)
        optimizer_ref = torch.optim.SGD(model_ref.parameters(), lr=0.05)
        optimizer_test = torch.optim.SGD(model_test.parameters(), lr=0.05)
        optimizers = self._OptimizerCollection(optimizer_test)

        trainer = object.__new__(GraphTrainer)
        trainer.config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                enable_graph_gradient_accumulation=True,
                enable_passes=False,
                inductor_compilation="none",
            ),
            training=SimpleNamespace(
                disable_cuda_graphs=True,
                max_norm=float("inf"),
            ),
        )
        trainer.parallel_dims = SimpleNamespace(
            pp_enabled=False,
            dp_enabled=False,
            ep_enabled=False,
            get_optional_mesh=lambda _axis: None,
        )
        trainer.model_parts = [model_test]
        trainer.optimizers = optimizers
        trainer.lr_schedulers = SimpleNamespace(
            get_metrics=lambda: {},
            step=lambda: None,
        )
        trainer.metrics_processor = SimpleNamespace(should_log=lambda _step: False)
        trainer.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
        trainer.device = torch.device("cpu")
        trainer.step = 1
        trainer.gradient_accumulation_steps = 3
        trainer.num_pp_microbatches = 1
        trainer._traced_step = None
        trainer._bound_traced_step = None
        trainer._trainable_params = None
        trainer._graph_gradient_state = None
        trainer.loss_fn = lambda prediction, target, global_valid_tokens: (
            torch.nn.functional.mse_loss(prediction, target, reduction="sum")
            / global_valid_tokens
        )
        trainer.train_context = nullcontext
        trainer.post_dataloading_process = lambda input_dict, labels: (
            input_dict["input"],
            labels,
            {},
        )

        grad_ids = None
        grad_data_ptrs = None
        for _ in range(2):
            microbatches = [
                (
                    {"input": torch.randn(4, 3, dtype=torch.float64)},
                    torch.randn(4, 2, dtype=torch.float64),
                )
                for _ in range(3)
            ]
            optimizer_ref.zero_grad(set_to_none=False)
            global_valid_tokens = sum(labels.numel() for _, labels in microbatches)
            for input_dict, labels in microbatches:
                loss_ref = torch.nn.functional.mse_loss(
                    model_ref(input_dict["input"]),
                    labels,
                    reduction="sum",
                )
                (loss_ref / global_valid_tokens).backward()
            optimizer_ref.step()

            trainer.train_step(iter(microbatches))

            self.assertIsNotNone(trainer._graph_gradient_state)
            self.assertIsNotNone(trainer._traced_step)
            assert trainer._graph_gradient_state is not None
            assert trainer._traced_step is not None
            self.assertTrue(trainer._traced_step.grad_sink_active)
            for parameter_ref, parameter_test in zip(
                model_ref.parameters(), model_test.parameters(), strict=True
            ):
                torch.testing.assert_close(parameter_test.grad, parameter_ref.grad)
                torch.testing.assert_close(parameter_test, parameter_ref)

            current_grad_ids = tuple(
                id(parameter.grad) for parameter in model_test.parameters()
            )
            current_grad_data_ptrs = tuple(
                parameter.grad.data_ptr() for parameter in model_test.parameters()
            )
            if grad_ids is None:
                grad_ids = current_grad_ids
                grad_data_ptrs = current_grad_data_ptrs
            else:
                self.assertEqual(current_grad_ids, grad_ids)
                self.assertEqual(current_grad_data_ptrs, grad_data_ptrs)

        self.assertEqual(
            len(
                [
                    node
                    for node in trainer._traced_step.gm.graph.nodes
                    if "graph_gradient_fqn" in node.meta
                ]
            ),
            len(tuple(model_test.parameters())),
        )

    def test_graph_gradient_state_rejects_tied_parameters(self):
        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, 2))
                self.tied_weight = self.weight

        model = TiedModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with self.assertRaisesRegex(ValueError, "tied parameter"):
            GraphGradientState.create(model, [optimizer])

    def test_graph_gradient_state_binding_failure_is_atomic(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        parameters = tuple(model.parameters())
        parameters[1].grad = torch.ones_like(parameters[1])

        with self.assertRaisesRegex(RuntimeError, "already has a gradient"):
            GraphGradientState.create(model, [optimizer])

        self.assertIsNone(parameters[0].grad)

    def test_graph_gradient_state_rejects_inplace_wgrad_modules_before_binding(self):
        class InplaceWgradModel(nn.Module):
            def __init__(self, *, nested_policy):
                super().__init__()
                self.projection = nn.Linear(3, 2)
                if nested_policy:
                    self.projection._runtime_policy = SimpleNamespace(
                        inplace_wgrad_accum=True
                    )
                else:
                    self.projection.inplace_wgrad_accum = True

        for nested_policy in (False, True):
            with self.subTest(nested_policy=nested_policy):
                model = InplaceWgradModel(nested_policy=nested_policy)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

                with self.assertRaisesRegex(
                    ValueError,
                    "projection.*enables inplace_wgrad_accum",
                ):
                    GraphGradientState.create(model, [optimizer])

                self.assertTrue(
                    all(parameter.grad is None for parameter in model.parameters())
                )

    def test_graph_gradient_state_rejects_shared_parameter_storage(self):
        class AliasedModel(nn.Module):
            def __init__(self):
                super().__init__()
                storage = torch.ones(8)
                self.first = nn.Parameter(storage[:4])
                self.second = nn.Parameter(storage[4:])

        model = AliasedModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with self.assertRaisesRegex(ValueError, "sharing storage"):
            GraphGradientState.create(model, [optimizer])

    def test_graph_gradient_state_rejects_other_wrapper_subclasses(self):
        parameter = _TraceableWrapper(torch.ones(2, 2, requires_grad=True))

        class WrappedModel(nn.Module):
            def named_parameters(self, *args, **kwargs):
                return iter((("weight", parameter),))

        optimizer = SimpleNamespace(param_groups=[{"params": [parameter]}])

        with self.assertRaisesRegex(
            NotImplementedError,
            "only supports plain tensors and DTensor parameters",
        ):
            GraphGradientState.create(WrappedModel(), [optimizer])

    def test_terminal_sink_rejects_nested_dtensor_local_wrapper(self):
        nested_meta = SubclassMeta(
            cls=_TraceableWrapper,
            attrs=["elem"],
            ctx=None,
            inner_metas={"elem": (1, None)},
            outer_size=torch.Size((2, 2)),
            outer_stride=(2, 1),
        )
        dtensor_meta = SubclassMeta(
            cls=DTensor,
            attrs=["_local_tensor", "device_mesh"],
            ctx=None,
            inner_metas={
                "_local_tensor": (1, nested_meta),
                "device_mesh": (1, None),
            },
            outer_size=torch.Size((2, 2)),
            outer_stride=(2, 1),
        )
        layout = SubclassLayout(num_tensors=2, meta=dtensor_meta)

        with self.assertRaisesRegex(
            NotImplementedError,
            "requires plain DTensor local tensors",
        ):
            _accumulation_leaf_offsets("weight", layout, layout)

    def test_terminal_sink_allows_only_dtensor_stride_mismatch(self):
        from torch.distributed.tensor import Replicate, Shard
        from torch.distributed.tensor._dtensor_spec import TensorMeta

        shape = torch.Size((16, 2, 8))

        def layout(
            *,
            stride,
            placements=(Shard(0),),
            dtype=torch.float32,
            shard_order=("shard-order",),
            requires_grad=False,
        ):
            return SubclassLayout(
                num_tensors=2,
                meta=SubclassMeta(
                    cls=DTensor,
                    attrs=["_local_tensor", "device_mesh"],
                    ctx=(
                        placements,
                        TensorMeta(shape=shape, stride=stride, dtype=dtype),
                        shard_order,
                        requires_grad,
                    ),
                    inner_metas={
                        "_local_tensor": (1, None),
                        "device_mesh": (1, None),
                    },
                    outer_size=shape,
                    outer_stride=stride,
                ),
            )

        buffer_layout = layout(stride=(16, 8, 1))
        gradient_layout = layout(stride=(2, 1, 32))
        self.assertEqual(
            _accumulation_leaf_offsets(
                "weight",
                buffer_layout,
                gradient_layout,
            ),
            (0,),
        )

        mismatches = {
            "placement": layout(
                stride=(2, 1, 32),
                placements=(Replicate(),),
            ),
            "dtype": layout(stride=(2, 1, 32), dtype=torch.bfloat16),
            "shard order": layout(
                stride=(2, 1, 32),
                shard_order=("different-order",),
            ),
            "requires grad": layout(stride=(2, 1, 32), requires_grad=True),
        }
        for mismatch, mismatched_layout in mismatches.items():
            with self.subTest(mismatch=mismatch), self.assertRaisesRegex(
                ValueError,
                "metadata does not match",
            ):
                _accumulation_leaf_offsets(
                    "weight",
                    buffer_layout,
                    mismatched_layout,
                )

    def test_graph_gradient_state_rejects_parameter_storage_replacement(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        gradient_state = GraphGradientState.create(model, [optimizer])
        parameters = tuple(model.parameters())

        with torch.no_grad():
            parameters[0].set_(parameters[0].detach().clone())

        with self.assertRaisesRegex(RuntimeError, "parameter storage"):
            gradient_state.validate_parameters(parameters)

    def test_graph_gradient_state_rejects_graph_state_replacement(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        gradient_state = GraphGradientState.create(model, [optimizer])
        gradient_state.graph_state["weight"] = torch.zeros_like(model.weight)

        with self.assertRaisesRegex(RuntimeError, "mapping value was replaced"):
            gradient_state.validate_bindings()

    def test_trainer_zero_grad_revalidates_optimizer_membership(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizers = self._OptimizerCollection(optimizer)
        gradient_state = GraphGradientState.create(model, optimizers)
        for buffer in gradient_state.buffers:
            buffer.fill_(1)
        buffer_ids = tuple(id(buffer) for buffer in gradient_state.buffers)

        trainer = object.__new__(GraphTrainer)
        trainer._graph_gradient_state = gradient_state
        trainer._graph_gradient_state_prepared_for_step = False
        trainer.model_parts = [model]
        trainer.optimizers = optimizers
        trainer._zero_grad()

        self.assertEqual(
            tuple(id(parameter.grad) for parameter in model.parameters()),
            buffer_ids,
        )
        self.assertTrue(
            all(torch.count_nonzero(buffer) == 0 for buffer in gradient_state.buffers)
        )

        optimizer.param_groups[0]["params"].pop()
        with self.assertRaisesRegex(ValueError, "do not match"):
            trainer._zero_grad()

    def test_trainer_rejects_unsupported_execution_modes(self):
        cases = (
            ("jit", False, "compile.mode='aot_fx_trace'"),
            ("aot_fx_trace", True, "pipeline parallelism"),
        )
        for mode, pp_enabled, error in cases:
            with self.subTest(mode=mode, pp_enabled=pp_enabled):
                trainer = object.__new__(GraphTrainer)
                trainer.config = SimpleNamespace(
                    compile=GraphTrainerCompileConfig(
                        mode=mode,
                        enable_graph_gradient_accumulation=True,
                    )
                )
                trainer.parallel_dims = SimpleNamespace(pp_enabled=pp_enabled)

                with self.assertRaisesRegex(ValueError, error):
                    trainer._validate_graph_gradient_accumulation_config()

    def test_deferred_fsdp_rejects_contradictory_cuda_graph_config(self):
        trainer = object.__new__(GraphTrainer)
        trainer.config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                mode="aot_fx_trace",
                enable_graph_gradient_accumulation=True,
                enable_deferred_fsdp_gradient_sync=True,
                require_cudagraph=True,
            ),
            parallelism=SimpleNamespace(fsdp_reshard_after_forward="never"),
            training=SimpleNamespace(disable_cuda_graphs=True),
        )
        trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
        trainer.gradient_accumulation_steps = 2

        with self.assertRaisesRegex(ValueError, "CUDA graphs are disabled"):
            trainer._validate_graph_gradient_accumulation_config()

    def test_fsdp_symmetric_memory_rejects_unsupported_graph_modes(self):
        cases = (
            ({"mode": "jit"}, False, "compile.mode='aot_fx_trace'"),
            ({"enable_passes": False}, False, "compile.enable_passes"),
            (
                {"precompile_artifact_dir": "/tmp/precompiled"},
                False,
                "precompiled artifacts",
            ),
            ({}, True, "SPMD only"),
        )
        for compile_overrides, pp_enabled, error in cases:
            with self.subTest(
                compile_overrides=compile_overrides,
                pp_enabled=pp_enabled,
            ):
                compile_kwargs = {
                    "mode": "aot_fx_trace",
                    "enable_passes": True,
                }
                compile_kwargs.update(compile_overrides)
                compile_config = GraphTrainerCompileConfig(**compile_kwargs)
                trainer = object.__new__(GraphTrainer)
                trainer.config = SimpleNamespace(
                    compile=compile_config,
                    parallelism=SimpleNamespace(
                        enable_fsdp_symm_mem=True,
                        fsdp_symm_mem_policy="widest",
                    ),
                )
                trainer.parallel_dims = SimpleNamespace(pp_enabled=pp_enabled)

                with self.assertRaisesRegex(ValueError, error):
                    trainer._validate_fsdp_symmetric_memory_config()

    def test_fsdp_symmetric_memory_rejects_unknown_policy(self):
        trainer = object.__new__(GraphTrainer)
        trainer.config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(),
            parallelism=SimpleNamespace(
                enable_fsdp_symm_mem=True,
                fsdp_symm_mem_policy="dense",
            ),
        )
        trainer.parallel_dims = SimpleNamespace(pp_enabled=False)

        with self.assertRaisesRegex(ValueError, "must be 'all' or 'widest'"):
            trainer._validate_fsdp_symmetric_memory_config()

    def test_fsdp_symmetric_memory_selects_backend_before_pg_init(self):
        trainer = object.__new__(GraphTrainer)
        trainer.config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(),
            parallelism=SimpleNamespace(
                enable_fsdp_symm_mem=True,
                fsdp_symm_mem_policy="widest",
            ),
        )
        parallel_dims = SimpleNamespace(pp_enabled=False)
        events = []

        with (
            patch.object(
                Trainer,
                "init_distributed",
                side_effect=lambda: events.append("distributed") or parallel_dims,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.trainer.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.trainer.torch.distributed.get_backend",
                return_value=torch.distributed.Backend.NCCL,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.trainer.configure_fsdp_symmetric_memory_backend",
                side_effect=lambda: events.append("backend"),
            ),
        ):
            result = trainer.init_distributed()

        self.assertIs(result, parallel_dims)
        self.assertEqual(events, ["backend", "distributed"])

    def test_fsdp_symmetric_memory_does_not_select_backend_for_fake_pg(self):
        trainer = object.__new__(GraphTrainer)
        trainer.config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(),
            parallelism=SimpleNamespace(
                enable_fsdp_symm_mem=True,
                fsdp_symm_mem_policy="widest",
            ),
        )
        parallel_dims = SimpleNamespace(pp_enabled=False)

        with (
            patch.object(
                Trainer,
                "init_distributed",
                return_value=parallel_dims,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.trainer.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.trainer.torch.distributed.get_backend",
                return_value=torch.distributed.Backend.FAKE,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.trainer.configure_fsdp_symmetric_memory_backend"
            ) as configure_backend,
        ):
            trainer.init_distributed()

        configure_backend.assert_not_called()

    def test_fsdp_symmetric_memory_skips_early_backend_for_fake_comm_modes(self):
        for comm_mode in (
            "fake_backend",
            "local_tensor",
            "real_pp_fake_spmd_backend",
        ):
            with self.subTest(comm_mode=comm_mode):
                trainer = object.__new__(GraphTrainer)
                trainer.config = SimpleNamespace(
                    comm=SimpleNamespace(mode=comm_mode),
                    compile=GraphTrainerCompileConfig(),
                    parallelism=SimpleNamespace(
                        enable_fsdp_symm_mem=True,
                        fsdp_symm_mem_policy="widest",
                    ),
                )
                parallel_dims = SimpleNamespace(pp_enabled=False)

                with (
                    patch.object(
                        Trainer,
                        "init_distributed",
                        return_value=parallel_dims,
                    ),
                    patch(
                        "torchtitan.experiments.graph_trainer.trainer."
                        "torch.distributed.is_initialized",
                        return_value=False,
                    ),
                    patch(
                        "torchtitan.experiments.graph_trainer.trainer."
                        "configure_fsdp_symmetric_memory_backend"
                    ) as configure_backend,
                ):
                    trainer.init_distributed()

                configure_backend.assert_not_called()

    def test_disabled_fsdp_symmetric_memory_does_not_touch_backend(self):
        trainer = object.__new__(GraphTrainer)
        trainer.config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(),
            parallelism=SimpleNamespace(
                enable_fsdp_symm_mem=False,
                fsdp_symm_mem_policy="widest",
            ),
        )
        parallel_dims = SimpleNamespace(pp_enabled=False)

        with (
            patch.object(
                Trainer,
                "init_distributed",
                return_value=parallel_dims,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.trainer.torch.distributed.get_backend"
            ) as get_backend,
            patch(
                "torchtitan.experiments.graph_trainer.trainer.configure_fsdp_symmetric_memory_backend"
            ) as configure_backend,
        ):
            trainer.init_distributed()

        get_backend.assert_not_called()
        configure_backend.assert_not_called()

    def test_terminal_sink_rejects_reordered_gradient_outputs(self):
        model = nn.Linear(3, 2, dtype=torch.float64)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        gradient_state = GraphGradientState.create(model, [optimizer])

        def train_step(inputs, targets):
            loss = torch.nn.functional.mse_loss(
                model(inputs),
                targets,
                reduction="sum",
            )
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            return [loss, *grads]

        inputs = torch.randn(2, 3, dtype=torch.float64)
        targets = torch.randn(2, 2, dtype=torch.float64)
        traced = minimal_fx_tracer(
            train_step,
            module=model,
            graph_state=gradient_state.graph_state,
            graph_state_output_indices=(1, 2),
        )(inputs, targets)
        output = next(node for node in traced.gm.graph.nodes if node.op == "output")
        output_values = list(output.args[0])
        output_values[1], output_values[2] = output_values[2], output_values[1]
        output.args = (output_values,)

        with self.assertRaisesRegex(ValueError, "gradient-output mapping"):
            finalize_graph_gradient_accumulation(
                traced.gm,
                traced_result=traced,
            )

    def test_terminal_sink_rejects_incompatible_gradient_buffer(self):
        model = nn.Linear(3, 2, dtype=torch.float64)

        def train_step(inputs, targets):
            loss = torch.nn.functional.mse_loss(
                model(inputs),
                targets,
                reduction="sum",
            )
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            return [loss, *grads]

        graph_state = {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in model.named_parameters()
        }
        traced = minimal_fx_tracer(
            train_step,
            module=model,
            graph_state=graph_state,
            graph_state_output_indices=(1, 2),
        )(
            torch.randn(2, 3, dtype=torch.float64),
            torch.randn(2, 2, dtype=torch.float64),
        )

        with self.assertRaisesRegex(ValueError, "shape, dtype, and device"):
            finalize_graph_gradient_accumulation(
                traced.gm,
                traced_result=traced,
            )

    def test_terminal_sink_does_not_alias_mutable_gradient_metadata(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        gradient_state = GraphGradientState.create(model, [optimizer])

        def train_step(inputs, targets):
            loss = torch.nn.functional.mse_loss(
                model(inputs),
                targets,
                reduction="sum",
            )
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            return [loss, *grads]

        traced = minimal_fx_tracer(
            train_step,
            module=model,
            graph_state=gradient_state.graph_state,
            graph_state_output_indices=(1, 2),
        )(torch.randn(2, 3), torch.randn(2, 2))
        output = next(node for node in traced.gm.graph.nodes if node.op == "output")
        gradient = output.args[0][1]
        gradient.meta["custom"] = {"owner": "gradient"}
        gradient.meta["unbacked_bindings"] = {"symbol": "gradient"}

        finalize_graph_gradient_accumulation(traced.gm, traced_result=traced)
        sink = next(
            node
            for node in traced.gm.graph.nodes
            if node.meta.get("graph_gradient_fqn") == "weight"
        )
        gradient.meta["custom"]["late_annotation"] = True
        gradient.meta["unbacked_bindings"]["late_symbol"] = True

        self.assertNotIn("late_annotation", sink.meta["custom"])
        self.assertNotIn("late_symbol", sink.meta["unbacked_bindings"])


class TestMinimalFXTracerDynamicShapes(unittest.TestCase):
    def _trace_mark_dynamic_value_range(self, *, min_value=None, max_value=None):
        from torch._dynamo import mark_dynamic

        def forward(x):
            return x.sin()

        kwargs = {}
        if min_value is not None:
            kwargs["min"] = min_value
        if max_value is not None:
            kwargs["max"] = max_value

        x = torch.randn(4, 4)
        mark_dynamic(x, 0, **kwargs)

        traced = minimal_fx_tracer(forward)(x)
        fake_x = next(
            node.meta["val"]
            for node in traced.gm.graph.nodes
            if node.op == "placeholder"
        )
        sym = fake_x.shape[0].node.expr
        return fake_x.shape[0].node.shape_env.var_to_range[sym]

    def test_fakeify_input_copies_only_shape_annotations(self):
        from torch._dynamo import mark_dynamic
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from torchtitan.experiments.graph_trainer.dynamic_shapes import _fakeify_input

        x = torch.randn(2, 4)
        mark_dynamic(x, 0)
        x._graph_trainer_unrelated_state = "must not be copied"

        fake_mode = FakeTensorMode(shape_env=ShapeEnv(), static_shapes=False)
        with fake_mode:
            fake_x = _fakeify_input(fake_mode, x, input_name="x")

        self.assertTrue(hasattr(fake_x, "_dynamo_dynamic_indices"))
        self.assertTrue(hasattr(fake_x, "_dynamo_dynamic_range"))
        self.assertFalse(hasattr(fake_x, "_graph_trainer_unrelated_state"))

    def test_mark_dynamic_min_max_preserves_range(self):
        value_range = self._trace_mark_dynamic_value_range(min_value=2, max_value=8)

        self.assertEqual(value_range.lower, 2)
        self.assertEqual(value_range.upper, 8)

    def test_mark_dynamic_one_sided_ranges_preserve_bounds(self):
        from torch.utils._sympy.numbers import int_oo

        # ShapeEnv tightens dynamic tensor sizes to exclude 0/1, so a max-only
        # user range is observed as [2, max] after fakeification.
        cases = (
            ("min_only", {"min_value": 2}, 2, int_oo),
            ("max_only", {"max_value": 8}, 2, 8),
        )
        for name, kwargs, expected_lower, expected_upper in cases:
            with self.subTest(name=name):
                value_range = self._trace_mark_dynamic_value_range(**kwargs)

                self.assertEqual(value_range.lower, expected_lower)
                self.assertEqual(value_range.upper, expected_upper)

    def test_mark_dynamic_wrapper_subclass_rejected(self):
        from torch._dynamo import mark_dynamic

        wrapper = _TraceableWrapper(torch.randn(2, 4))
        mark_dynamic(wrapper, 0)

        def forward(x):
            return x

        with self.assertRaisesRegex(
            ValueError,
            "only supports marked dynamic dims on plain tensor inputs",
        ):
            minimal_fx_tracer(forward)(wrapper)

    def test_nested_wrapper_subclass_marked_inner_rejected(self):
        from torch._dynamo import mark_dynamic
        from torch._dynamo.decorators import mark_unbacked

        def forward(x):
            return x

        for marker in (mark_dynamic, mark_unbacked):
            with self.subTest(marker=marker.__name__):
                inner = torch.randn(2, 4)
                marker(inner, 0)
                wrapper = _TraceableWrapper(_TraceableWrapper(inner))

                with self.assertRaisesRegex(
                    ValueError,
                    "only supports marked dynamic dims on plain tensor inputs",
                ):
                    minimal_fx_tracer(forward)(wrapper)

    def test_mark_dynamic_token_dim_with_rope(self):
        from torch._dynamo import mark_dynamic

        from torchtitan.models.common.rope import (
            _reshape_for_broadcast,
            ComplexRoPE,
            CosSinRoPE,
        )

        def forward(x, xq, xk, freqs_cis, rope_cache, positions):
            complex_cache = _reshape_for_broadcast(
                freqs_cis, (*x.shape[:-1], x.shape[-1] // 2), positions
            )
            single, _ = ComplexRoPE.apply_rotary_emb(x, x, complex_cache)
            cos_sin_cache = _reshape_for_broadcast(rope_cache, xq.shape, positions)
            q, k = CosSinRoPE.apply_rotary_emb(xq, xk, cos_sin_cache)
            return single + q + k

        num_tokens, heads, head_dim = 8, 1, 8
        position_cases = {
            "none": None,
            "flat": torch.arange(num_tokens),
        }

        for name, positions in position_cases.items():
            with self.subTest(positions=name):
                x = torch.randn(num_tokens, heads, head_dim)
                xq = torch.randn(num_tokens, heads, head_dim)
                xk = torch.randn(num_tokens, heads, head_dim)
                freqs = torch.randn(num_tokens * 2, head_dim // 2)
                freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
                rope_cache = torch.randn(num_tokens * 2, head_dim * 2)

                for tensor in (x, xq, xk):
                    mark_dynamic(tensor, 0)
                if positions is not None:
                    mark_dynamic(positions, 0)

                traced = minimal_fx_tracer(forward)(
                    x, xq, xk, freqs_cis, rope_cache, positions
                )
                self.assertTrue(
                    torch.equal(
                        forward(x, xq, xk, freqs_cis, rope_cache, positions),
                        run_traced(traced)(x, xq, xk, freqs_cis, rope_cache, positions),
                    )
                )

    def test_mark_unbacked_positions_token_dim_with_rope(self):
        from torch._dynamo.decorators import mark_unbacked

        from torchtitan.models.common.rope import _reshape_for_broadcast, CosSinRoPE

        def forward(xq, xk, rope_cache, positions):
            cos_sin_cache = _reshape_for_broadcast(rope_cache, xq.shape, positions)
            q, k = CosSinRoPE.apply_rotary_emb(xq, xk, cos_sin_cache)
            return q + k

        num_tokens, heads, head_dim = 16, 1, 8
        xq = torch.randn(num_tokens, heads, head_dim)
        xk = torch.randn(num_tokens, heads, head_dim)
        rope_cache = torch.randn(num_tokens * 2, head_dim * 2)
        positions = torch.arange(num_tokens)
        for tensor in (xq, xk, positions):
            mark_unbacked(
                tensor,
                0,
                hint_override=num_tokens,
                min=1,
                max=num_tokens,
                shape_id="tokens",
            )

        traced = minimal_fx_tracer(forward)(xq, xk, rope_cache, positions)

        self.assertTrue(
            torch.equal(
                forward(xq, xk, rope_cache, positions),
                run_traced(traced)(xq, xk, rope_cache, positions),
            )
        )

    def test_rope_symbolic_positions_compile_fullgraph(self):
        from torchtitan.models.common.rope import _reshape_for_broadcast

        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def forward(xq, rope_cache, positions):
            return _reshape_for_broadcast(rope_cache, xq.shape, positions)

        num_tokens, head_dim = 5, 8
        xq = torch.randn(num_tokens, 1, head_dim)
        rope_cache = torch.randn(num_tokens * 2, head_dim * 2)
        positions = torch.arange(num_tokens)

        self.assertTrue(
            torch.equal(
                forward(xq, rope_cache, positions),
                _reshape_for_broadcast(rope_cache, xq.shape, positions),
            )
        )

    def test_maybe_materialize_grad_for_param_layout_restores_param_strides(self):
        param = torch.empty_strided((2, 3), (1, 2))
        grad = torch.arange(6.0).reshape(2, 3)

        materialized = _maybe_materialize_grad_for_param_layout(param, grad)

        self.assertEqual(materialized.stride(), param.stride())
        self.assertTrue(torch.equal(materialized, grad))
        self.assertIs(
            _maybe_materialize_grad_for_param_layout(param, materialized),
            materialized,
        )

    def test_accumulate_param_grads_owns_first_graph_output(self):
        param = nn.Parameter(torch.zeros(2))
        graph_grad = torch.tensor([1.0, 2.0])

        accumulate_param_grads_([param], [graph_grad], outputs_are_replay_owned=True)
        graph_grad.fill_(3.0)

        self.assertTrue(torch.equal(param.grad, torch.tensor([1.0, 2.0])))
        accumulate_param_grads_([param], [graph_grad], outputs_are_replay_owned=True)
        self.assertTrue(torch.equal(param.grad, torch.tensor([4.0, 5.0])))

    def test_mark_unbacked_mixed_with_static_input_replay(self):
        from torch._dynamo.decorators import mark_unbacked

        def forward(dynamic_x, static_y):
            return dynamic_x.cos() + static_y.sin()

        dynamic_x = torch.randn(2, 4)
        static_y = torch.randn(2, 4)
        mark_unbacked(dynamic_x, 0)

        traced = minimal_fx_tracer(forward)(dynamic_x, static_y)
        dynamic_x_other = torch.randn(3, 4)
        static_y_other = torch.randn(3, 4)

        self.assertTrue(
            torch.equal(
                forward(dynamic_x, static_y),
                run_traced(traced)(dynamic_x, static_y),
            )
        )
        self.assertTrue(
            torch.equal(
                forward(dynamic_x_other, static_y_other),
                run_traced(traced)(dynamic_x_other, static_y_other),
            )
        )

    def test_mark_unbacked_shape_branch_rejected(self):
        from torch._dynamo.decorators import mark_unbacked
        from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

        def forward(x):
            if x.shape[0] > 100:
                return x.cos()
            return x.sin()

        x = torch.randn(4, 4)
        mark_unbacked(x, 0)

        with self.assertRaisesRegex(
            GuardOnDataDependentSymNode,
            "Could not guard on data-dependent expression",
        ):
            minimal_fx_tracer(forward)(x)

    def test_mark_unbacked_min_max_preserves_unbacked_placeholder_dim(self):
        from torch._dynamo.decorators import mark_unbacked
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        def forward(x):
            if x.size(0) >= 2 and x.size(0) <= 5:
                return x.sin()
            return x.cos()

        x = torch.randn(3, 4)
        mark_unbacked(x, 0, min=2, max=5)

        traced = minimal_fx_tracer(forward)(x)
        fake_x = next(
            node.meta["val"]
            for node in traced.gm.graph.nodes
            if node.op == "placeholder"
        )
        x_min = torch.randn(2, 4)
        x_max = torch.randn(5, 4)

        self.assertIsInstance(fake_x.size(0), torch.SymInt)
        self.assertTrue(free_unbacked_symbols(fake_x.size(0)))
        self.assertEqual(fake_x.size(1), 4)
        self.assertTrue(torch.equal(forward(x_min), run_traced(traced)(x_min)))
        self.assertTrue(torch.equal(forward(x_max), run_traced(traced)(x_max)))

    def test_mark_unbacked_input_symbol_is_not_pending_fresh(self):
        from torch._dynamo.decorators import mark_unbacked

        def forward(x):
            return x.sin()

        x = torch.randn(3, 4)
        mark_unbacked(x, 0, min=2, max=5)

        traced = minimal_fx_tracer(forward)(x)
        fake_x = next(
            node.meta["val"]
            for node in traced.gm.graph.nodes
            if node.op == "placeholder"
        )
        shape_env = fake_x.shape[0].node.shape_env

        self.assertEqual(shape_env.pending_fresh_unbacked_symbols, [])
        self.assertEqual(shape_env.ignorable_fresh_unbacked_symbols, [])

    def test_mark_unbacked_preserves_unbacked_placeholder_dim(self):
        from torch._dynamo.decorators import mark_unbacked
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        def forward(x):
            return x.sin()

        x = torch.randn(2, 4)
        mark_unbacked(x, 0)

        traced = minimal_fx_tracer(forward)(x)
        fake_x = next(
            node.meta["val"]
            for node in traced.gm.graph.nodes
            if node.op == "placeholder"
        )
        x_other = torch.randn(3, 4)

        self.assertIsInstance(fake_x.size(0), torch.SymInt)
        self.assertTrue(free_unbacked_symbols(fake_x.size(0)))
        self.assertEqual(fake_x.size(1), 4)
        self.assertTrue(torch.equal(forward(x), run_traced(traced)(x)))
        self.assertTrue(
            torch.equal(
                forward(x_other),
                run_traced(traced)(x_other),
            )
        )

    def test_mark_unbacked_multiple_inputs_replay(self):
        from torch._dynamo.decorators import mark_unbacked

        def forward(x, y):
            return x.sin() + y.cos()

        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        mark_unbacked(x, 0)
        mark_unbacked(y, 0)

        traced = minimal_fx_tracer(forward)(x, y)
        x_other = torch.randn(3, 4)
        y_other = torch.randn(3, 4)

        self.assertTrue(torch.equal(forward(x, y), run_traced(traced)(x, y)))
        self.assertTrue(
            torch.equal(
                forward(x_other, y_other),
                run_traced(traced)(x_other, y_other),
            )
        )

    def test_data_dependent_check_emits_runtime_asserts(self):
        """torch._check on a data-dependent .item() symbol becomes _assert_scalar nodes."""

        def forward(x, n):
            v = n.item()
            torch._check(v > 0)
            torch._check(v < 100)
            return x.sin().sum() + v

        x = torch.randn(8, 4)
        n = torch.tensor([5])

        traced = minimal_fx_tracer(forward, _insert_runtime_asserts=True)(x, n)
        assert_count = sum(
            1
            for node in traced.gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.aten._assert_scalar.default
        )
        # Both inline (gt/lt) and bound-style (>= 1, <= 99) asserts are emitted.
        self.assertEqual(assert_count, 4)

    def test_no_runtime_asserts_when_no_constraints(self):
        """Tracing without data-dependent _check produces no _assert_scalar nodes."""
        from torch._dynamo.decorators import mark_unbacked

        def forward(x):
            return x.sin()

        x = torch.randn(4, 4)
        mark_unbacked(x, 0)

        traced = minimal_fx_tracer(forward)(x)
        assert_count = sum(
            1
            for node in traced.gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.aten._assert_scalar.default
        )
        self.assertEqual(assert_count, 0)

    def test_mark_unbacked_shape_id_multiple_inputs_replay(self):
        from torch._dynamo.decorators import mark_unbacked

        def forward(x, y):
            if x.size(0) == y.size(0):
                return x.sin() + y.cos()
            return x.cos() + y.sin()

        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        mark_unbacked(x, 0, shape_id="batch")
        mark_unbacked(y, 0, shape_id="batch")

        traced = minimal_fx_tracer(forward)(x, y)
        x_other = torch.randn(3, 4)
        y_other = torch.randn(3, 4)

        self.assertTrue(
            torch.equal(
                forward(x_other, y_other),
                run_traced(traced)(x_other, y_other),
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceModule(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32
    BATCH_SIZE = 2
    SEQ_LEN = 128
    NUM_STEPS = 5
    LR = 1e-3

    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _make_mlp(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(
            0, 256, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        labels = torch.randint(
            0, 256, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        return model, tokens, labels, get_loss

    def test_mlp_forward(self):
        model, tokens, labels, loss_fn = self._make_mlp()

        def forward(tokens):
            return model(tokens)

        traced = minimal_fx_tracer(forward, module=model)(tokens)
        out_eager = model(tokens)
        wrapped = run_traced(traced, module=model)(tokens)
        self.assertTrue(torch.equal(out_eager, wrapped))

    def test_mlp_train_step(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_test = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())

        train_step = make_train_step(model_ref, loss_fn)
        traced = minimal_fx_tracer(train_step, module=model_ref)(tokens, labels)

        logits_ref = model_ref(tokens)
        loss_ref = loss_fn(logits_ref, labels)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        wrapped = run_traced(traced, module=model_test)(tokens, labels)
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(torch.equal(loss_ref, loss_tr))
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr, gt))

    def test_chunked_loss_train_step(self):
        D, V, num_chunks = 32, 257, 4
        lm_head_ref = nn.Linear(D, V, bias=False).to(
            device=self.DEVICE, dtype=self.DTYPE
        )
        lm_head_test = nn.Linear(D, V, bias=False).to(
            device=self.DEVICE, dtype=self.DTYPE
        )
        lm_head_test.load_state_dict(lm_head_ref.state_dict())
        num_tokens = self.BATCH_SIZE * self.SEQ_LEN
        hidden_states = torch.randn(
            num_tokens,
            D,
            device=self.DEVICE,
            dtype=self.DTYPE,
            requires_grad=True,
        )
        labels = torch.randint(0, V, (num_tokens,), device=self.DEVICE)

        def train_step(lm_head, hidden_states, labels):
            loss_fn = ChunkedLossWrapperWithParamGrads(
                ChunkedLossWrapperWithParamGrads.Config(num_chunks=num_chunks)
            )
            loss_fn.set_lm_head(lm_head)
            loss, _ = loss_fn(hidden_states, labels)
            grads = torch.autograd.grad(loss, [hidden_states, *lm_head.parameters()])
            return [loss, *grads]

        eager_out = train_step(lm_head_ref, hidden_states, labels)

        def train_step_closure(hidden_states, labels):
            return train_step(lm_head_test, hidden_states, labels)

        traced = minimal_fx_tracer(train_step_closure, module=lm_head_test)(
            hidden_states, labels
        )
        replay_out = run_traced(traced, module=lm_head_test)(hidden_states, labels)

        for ref, tr in zip(eager_out, replay_out, strict=True):
            self.assertTrue(torch.equal(ref, tr))

    def test_mlp_multistep_bitwise(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_test = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())

        train_step = make_train_step(model_ref, loss_fn)
        traced = minimal_fx_tracer(train_step, module=model_ref)(tokens, labels)

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=self.LR)
        opt_copy = torch.optim.Adam(model_test.parameters(), lr=self.LR)

        for step in range(1, self.NUM_STEPS + 1):
            logits_ref = model_ref(tokens)
            loss_ref = loss_fn(logits_ref, labels)
            loss_ref.backward()
            grads_ref = [p.grad.clone() for p in model_ref.parameters()]
            opt_ref.step()
            opt_ref.zero_grad()

            wrapped = run_traced(traced, module=model_test)(tokens, labels)
            loss_tr = wrapped[0]
            grads_tr = wrapped[1:]
            for p, g in zip(model_test.parameters(), grads_tr, strict=True):
                p.grad = g
            opt_copy.step()
            opt_copy.zero_grad()

            self.assertTrue(
                torch.equal(loss_ref, loss_tr), f"Step {step}: loss mismatch"
            )
            for gr, gt in zip(grads_ref, grads_tr, strict=True):
                self.assertTrue(torch.equal(gr, gt), f"Step {step}: grad mismatch")

    def test_non_tensor_leaf_raises(self):
        """Passing a callable leaf in args raises (should be in closure instead)."""

        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        def fn(x, loss_fn):
            return loss_fn(model(x))

        with self.assertRaises(ValueError, msg="all pytree leaves"):
            minimal_fx_tracer(fn, module=model)(tokens, lambda x: x.sum())

    def test_mismatched_module_raises_when_validation_enabled(self):
        """Opt-in module FQN validation catches execution with the wrong module."""
        model, tokens, labels, loss_fn = self._make_mlp()

        def forward(tokens):
            return model(tokens)

        traced = minimal_fx_tracer(forward, module=model)(tokens)

        different_model = nn.Sequential(
            nn.Embedding(256, 64),
            nn.Linear(64, 256),
        ).to(device=self.DEVICE, dtype=self.DTYPE)

        with self.assertRaises(ValueError, msg="different parameter/buffer names"):
            run_traced(traced, module=different_model, _validate_runtime=True)(tokens)

    def test_optimizer_passed_without_module_raises(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        opt = torch.optim.Adam(model.parameters())
        with self.assertRaises(ValueError, msg="optimizer"):
            minimal_fx_tracer(lambda: None, optimizer=opt)

    def test_kwargs_roundtrip(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        scale = torch.tensor(2.0, device=self.DEVICE)

        def forward(state, tokens, *, scale):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens) * scale

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens, scale=scale)
        out_ref = forward(state, tokens, scale=scale)
        out_traced = run_traced(traced)(state, tokens, scale=scale)
        self.assertTrue(torch.equal(out_ref, out_traced))

    def test_kwargs_runtime_reorder_raises(self):
        """Runtime kwargs in different order produce a different spec; with
        ``_validate_runtime=True``, this raises."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        a = torch.tensor(2.0, device=self.DEVICE)
        b = torch.tensor(3.0, device=self.DEVICE)

        def forward(state, tokens, *, a, b):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens) * a + b

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens, a=a, b=b)
        with self.assertRaisesRegex(ValueError, "input spec mismatch"):
            run_traced(traced, _validate_runtime=True)(state, tokens, b=b, a=a)

    def test_kwargs_unknown_kwarg_raises(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        scale = torch.tensor(2.0, device=self.DEVICE)

        def forward(state, tokens, *, scale):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens) * scale

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens, scale=scale)
        with self.assertRaisesRegex(ValueError, "input spec mismatch"):
            run_traced(traced, _validate_runtime=True)(state, tokens, factor=scale)

    def test_kwargs_default_omitted_bakes_constant(self):
        """fn with a default kwarg, not passed at trace: default is baked in.
        Runtime must also omit it (passing it would change the spec)."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        scale = torch.tensor(3.0, device=self.DEVICE)

        def forward(state, tokens, *, scale=2.0):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens) * scale

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens)
        out_default = run_traced(traced)(state, tokens)
        out_ref = forward(state, tokens)
        self.assertTrue(torch.equal(out_ref, out_default))

        with self.assertRaisesRegex(ValueError, "input spec mismatch"):
            run_traced(traced, _validate_runtime=True)(state, tokens, scale=scale)

    def test_kwargs_var_keyword_missing_key_raises(self):
        """fn with **opts: missing a kwarg at runtime changes the kwargs spec."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        a = torch.tensor(2.0, device=self.DEVICE)
        b = torch.tensor(5.0, device=self.DEVICE)

        def forward(state, tokens, **opts):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens) * opts["a"] + opts["b"]

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens, a=a, b=b)
        out_ref = forward(state, tokens, a=a, b=b)
        out_traced = run_traced(traced)(state, tokens, a=a, b=b)
        self.assertTrue(torch.equal(out_ref, out_traced))

        with self.assertRaisesRegex(ValueError, "input spec mismatch"):
            run_traced(traced, _validate_runtime=True)(state, tokens, a=a)

    def test_flex_attention_block_mask_mark_unbacked(self):
        from torch._dynamo.decorators import mark_unbacked
        from torch.nn.attention.flex_attention import (
            AuxRequest,
            BlockMask,
            flex_attention,
        )

        maybe_register_blockmask_pytree_node()

        def make_mask_fn(attn_regions, document_ids):
            def mask_mod(b, h, q_idx, kv_idx):
                return (
                    (q_idx >= kv_idx)
                    & (attn_regions[q_idx] == attn_regions[kv_idx])
                    & (document_ids[q_idx] == document_ids[kv_idx])
                )

            return mask_mod

        def make_mask(ntoks=256, block_size=128):
            nblocks = (ntoks + block_size - 1) // block_size
            width = 2

            kv_indices = (
                torch.arange(width, dtype=torch.int32, device=self.DEVICE)
                .expand(nblocks, width)
                .clone()
            )
            full_kv_indices = kv_indices.clone()
            q_indices = kv_indices.clone()
            full_q_indices = kv_indices.clone()
            kv_num_blocks = torch.full(
                (nblocks,), width, dtype=torch.int32, device=self.DEVICE
            )
            full_kv_num_blocks = kv_num_blocks.clone()
            q_num_blocks = kv_num_blocks.clone()
            full_q_num_blocks = kv_num_blocks.clone()

            mark_unbacked(kv_indices, 1)
            mark_unbacked(full_kv_indices, 1)
            mark_unbacked(q_indices, 1)
            mark_unbacked(full_q_indices, 1)

            attn_regions = torch.arange(ntoks, dtype=torch.int32, device=self.DEVICE)
            document_ids = torch.zeros(ntoks, dtype=torch.int32, device=self.DEVICE)
            return BlockMask(
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                full_kv_num_blocks=full_kv_num_blocks,
                full_kv_indices=full_kv_indices,
                q_num_blocks=q_num_blocks,
                q_indices=q_indices,
                full_q_num_blocks=full_q_num_blocks,
                full_q_indices=full_q_indices,
                BLOCK_SIZE=(block_size, block_size),
                mask_mod=make_mask_fn(attn_regions, document_ids),
                seq_lengths=(ntoks, ntoks),
            )

        q = torch.randn(1, 2, 256, 32, device=self.DEVICE)
        k = torch.randn(1, 2, 256, 32, device=self.DEVICE)
        v = torch.randn(1, 2, 256, 32, device=self.DEVICE)
        mask = make_mask()
        cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)

        def forward(q, k, v, block_mask):
            out, aux = cflex(
                q,
                k,
                v,
                block_mask=block_mask,
                return_aux=AuxRequest(max_scores=True),
            )
            return out.sum().detach(), aux.max_scores.max().detach()

        minimal_fx_tracer(forward)(q, k, v, mask)

    def test_module_in_args_raises(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        other_model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        def forward(other, tokens):
            return other(tokens)

        with self.assertRaises(ValueError, msg="nn.Module"):
            minimal_fx_tracer(forward, module=model)(other_model, tokens)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestReparametrizeOptimizer(unittest.TestCase):
    """Verify swap_in_optimizer_params_and_state works with a torchtitan OptimizersContainer.

    OptimizersContainer is itself an Optimizer subclass, but it delegates
    ``step``/``state``/``state_dict`` to inner ``torch.optim.Adam``/``AdamW``
    instances. ``OptimizersContainer.state_dict()`` returns a DCP-flattened
    (FQN-keyed) dict, so the reparametrize helper consumes the inner
    optimizer's raw ``state_dict()`` (packed-int-id format) instead.
    """

    DEVICE = "cuda"
    DTYPE = torch.float32

    def test_titan_optimizers_container(self):
        from torchtitan.components.optimizer import (
            OptimizersContainer,
            ParamGroupConfig,
        )

        torch.manual_seed(0)
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        container = OptimizersContainer(
            OptimizersContainer.Config(
                param_groups=[
                    ParamGroupConfig(
                        pattern=r".*",
                        optimizer_name="AdamW",
                        optimizer_kwargs={"lr": 1e-3},
                    )
                ],
                implementation="for-loop",
            ),
            model_parts=[model],
        )
        inner = container.optimizers[0]

        # Initialize Adam's lazy per-parameter state.
        x = torch.randint(0, 256, (2, 16), device=self.DEVICE)
        loss = model(x).sum()
        loss.backward()
        container.step()
        container.zero_grad(set_to_none=True)

        # Snapshot the originals so we can verify perfect restoration.
        optim_state_dict = inner.state_dict()
        original_param_ids = [[id(p) for p in g["params"]] for g in inner.param_groups]
        original_state_keys = list(inner.state.keys())
        original_state_id = id(inner.state)

        # Rebind to fake parameter tensors (zeros to make the swap obvious).
        params_dict = dict(model.named_parameters(remove_duplicate=False))
        fake_params = {name: torch.zeros_like(p) for name, p in params_dict.items()}

        with swap_in_optimizer_params_and_state(inner, fake_params, optim_state_dict):
            # The live optimizer now points at the fake tensors.
            rebound = [p for g in inner.param_groups for p in g["params"]]
            for fake, rebound_p in zip(fake_params.values(), rebound, strict=True):
                self.assertIs(fake, rebound_p)

            # Per-param state is keyed by the rebound tensors and shares the
            # original tensor values (so in-place ops would propagate).
            for fake in fake_params.values():
                self.assertIn(fake, inner.state)
            for name, fake in fake_params.items():
                # Match against the original state_dict via positional
                # alignment in the optimizer's first (only) param group.
                idx = list(fake_params).index(name)
                packed_id = optim_state_dict["param_groups"][0]["params"][idx]
                expected_state = optim_state_dict["state"][packed_id]
                self.assertEqual(
                    set(inner.state[fake].keys()), set(expected_state.keys())
                )
                for k, v in expected_state.items():
                    if isinstance(v, torch.Tensor):
                        self.assertIs(inner.state[fake][k], v)

        # After the context the live optimizer is fully restored.
        self.assertEqual(id(inner.state), original_state_id)
        self.assertEqual(list(inner.state.keys()), original_state_keys)
        for orig_ids, group in zip(original_param_ids, inner.param_groups, strict=True):
            self.assertEqual([id(p) for p in group["params"]], orig_ids)

    def test_minimal_fx_tracer_with_bucketed_optimizer(self):
        torch.manual_seed(0)
        module = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 7))
        weights = [p for n, p in module.named_parameters() if n.endswith("weight")]
        biases = [p for n, p in module.named_parameters() if n.endswith("bias")]
        optimizer = torch.optim.AdamW(
            [{"params": weights, "lr": 0.1}, {"params": biases, "lr": 0.01}]
        )

        x = torch.randn(2, 3)
        module(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        def train_step(x):
            optimizer.zero_grad(set_to_none=True)
            module(x).sum().backward()
            optimizer.step()
            return torch.stack([p.detach().sum() for p in module.parameters()])

        traced = minimal_fx_tracer(train_step, module=module, optimizer=optimizer)(x)

        model_sd = deepcopy(module.state_dict())
        optim_sd = deepcopy(optimizer.state_dict())

        eager_out = train_step(x)
        eager_params = [p.detach().clone() for p in module.parameters()]

        module.load_state_dict(model_sd)
        optimizer.load_state_dict(optim_sd)
        traced_out = run_traced(traced, module=module, optimizer=optimizer)(x)
        traced_params = [p.detach().clone() for p in module.parameters()]

        self.assertTrue(torch.equal(eager_out, traced_out))
        for ep, tp in zip(eager_params, traced_params, strict=True):
            self.assertTrue(torch.equal(ep, tp))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceDTensor(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32

    def setUp(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:12357",
                world_size=1,
                rank=0,
            )
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        import torch.distributed as dist

        torch.use_deterministic_algorithms(False)
        if dist.is_initialized():
            dist.destroy_process_group()

    def _distribute_params(self, model, mesh):
        from torch.distributed._tensor import distribute_tensor, Replicate

        for name, param in list(model.named_parameters()):
            dt = distribute_tensor(param, mesh, [Replicate()])
            param_parts = name.split(".")
            mod = model
            for part in param_parts[:-1]:
                mod = getattr(mod, part)
            setattr(mod, param_parts[-1], nn.Parameter(dt))

    def test_dtensor_forward(self):
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.DEVICE, (1,))

        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        self._distribute_params(model, mesh)

        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        tokens_dt = DTensor.from_local(tokens, mesh, [Replicate()])

        def forward(tokens):
            return model(tokens)

        traced = minimal_fx_tracer(forward, module=model)(tokens_dt)
        has_subclass = any(
            layout.meta is not None for layout in traced.input_subclass_layouts.values()
        )
        self.assertTrue(has_subclass)

        out_eager = model(tokens_dt)
        wrapped = run_traced(traced, module=model)(tokens_dt)
        self.assertTrue(torch.equal(out_eager.full_tensor(), wrapped.full_tensor()))

    def test_dtensor_mark_unbacked_rejected(self):
        from torch._dynamo.decorators import mark_unbacked
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.DEVICE, (1,))
        tokens = torch.randn(2, 32, device=self.DEVICE)
        tokens_dt = DTensor.from_local(tokens, mesh, [Replicate()])
        mark_unbacked(tokens_dt, 0)

        def forward(tokens):
            return tokens

        with self.assertRaisesRegex(
            ValueError,
            "only supports marked dynamic dims on plain tensor inputs",
        ):
            minimal_fx_tracer(forward)(tokens_dt)

    def test_dtensor_train_step(self):
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.DEVICE, (1,))

        model_ref = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())

        self._distribute_params(model_ref, mesh)
        self._distribute_params(model_test, mesh)

        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        tokens_dt = DTensor.from_local(tokens, mesh, [Replicate()])
        labels_dt = DTensor.from_local(labels, mesh, [Replicate()])

        train_step = make_train_step(model_ref, get_loss)
        traced = minimal_fx_tracer(train_step, module=model_ref)(tokens_dt, labels_dt)

        logits_ref = model_ref(tokens_dt)
        loss_ref = get_loss(logits_ref, labels_dt)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        wrapped = run_traced(traced, module=model_test)(tokens_dt, labels_dt)
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(torch.equal(loss_ref.full_tensor(), loss_tr.full_tensor()))
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr.full_tensor(), gt.full_tensor()))

    def test_full_inductor_pass_on_collective(self):
        # ``make_fx`` traces ``dist.*`` collectives as raw ``c10d.{op}_``
        # inplace ops with a torchbind ``ProcessGroup`` baked in as a graph
        # attr. ``full_inductor_compilation_pass`` must functionalize the
        # collective and unbox the PG before ``compile_fx_inner`` — otherwise
        # the cache key calls ``__eq__`` on the torchbind and crashes.
        import torch.distributed as dist

        from torchtitan.experiments.graph_trainer.inductor_passes import (
            full_inductor_compilation_pass,
        )

        def f(_state, t):
            t = t.clone()
            dist.all_reduce(t)
            return t + 1

        traced = minimal_fx_tracer(f)({}, torch.ones(4, device=self.DEVICE))
        compiled_gm = full_inductor_compilation_pass(traced.gm, traced.example_inputs)

        real_input = torch.ones(4, device=self.DEVICE)
        expected = f({}, real_input.clone())
        actual = compiled_gm(real_input.clone())
        if isinstance(actual, (list, tuple)):
            actual = actual[0]
        torch.testing.assert_close(actual, expected)

    def test_full_inductor_pass_migrates_cpu_attrs(self):
        from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_pass
        from torchtitan.experiments.graph_trainer.inductor_passes import (
            full_inductor_compilation_pass,
        )

        def f(_state, x):
            pad = torch.tensor(-1, dtype=torch.int64)
            fill = torch.tensor(0, dtype=torch.bfloat16)
            scale = torch.tensor(1.0, dtype=torch.float32)
            return x + pad.to(x.dtype) + fill.to(x.dtype) + scale.to(x.dtype)

        traced = minimal_fx_tracer(f)(
            {}, torch.zeros(4, dtype=torch.float32, device=self.DEVICE)
        )

        cpu_attr_names = [
            n.target
            for n in traced.gm.graph.find_nodes(op="get_attr")
            if isinstance(getattr(traced.gm, n.target, None), torch.Tensor)
            and getattr(traced.gm, n.target).device.type == "cpu"
        ]

        gm = full_inductor_compilation_pass(traced.gm, traced.example_inputs)

        for name in cpu_attr_names:
            attr = getattr(traced.gm, name, None)
            self.assertIsInstance(attr, torch.Tensor)
            self.assertEqual(
                attr.device.type,
                "cuda",
                f"{name} should have been migrated to CUDA",
            )

        gm = cudagraph_pass(gm, traced.example_inputs)
        real_x = torch.zeros(4, dtype=torch.float32, device=self.DEVICE)
        expected = f({}, real_x.clone())
        for _ in range(3):
            actual = gm(real_x.clone())
            if isinstance(actual, (list, tuple)):
                actual = actual[0]
            torch.testing.assert_close(actual, expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMetadataPropagation(unittest.TestCase):
    """Tests for _copy_fwd_metadata_to_bw_nodes."""

    DEVICE = "cuda"
    DTYPE = torch.float32

    def setUp(self):
        torch.manual_seed(42)

    def test_backward_nodes_have_seq_nr(self):
        """Verify that backward FX nodes get seq_nr metadata via patched autograd.grad."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        train_step = make_train_step(model, get_loss)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        traced = minimal_fx_tracer(train_step, module=model)(tokens, labels)

        # Collect seq_nr values from all call_function nodes
        seq_nrs = []
        for node in traced.gm.graph.nodes:
            if node.op == "call_function" and "seq_nr" in node.meta:
                seq_nrs.append(node.meta["seq_nr"])

        # There should be seq_nr values present (both fwd and bwd nodes)
        self.assertGreater(len(seq_nrs), 0, "No seq_nr metadata found on any node")

        # There should be duplicate seq_nrs (fwd and bwd nodes sharing seq_nr)
        counts = Counter(seq_nrs)
        shared = [nr for nr, cnt in counts.items() if cnt > 1]
        self.assertGreater(
            len(shared),
            0,
            "Expected some seq_nr values shared between fwd and bwd nodes",
        )

    def test_copy_fwd_metadata_propagates_custom(self):
        """Verify _copy_fwd_metadata_to_bw_nodes copies custom metadata to bwd nodes."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)

        train_step = make_train_step(model, get_loss)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        traced = minimal_fx_tracer(train_step, module=model)(tokens, labels)
        gm = traced.gm

        # Manually set custom metadata on the first fwd node for each seq_nr
        # to test that _copy_fwd_metadata_to_bw_nodes works
        seq_nr_first: dict[int, torch.fx.Node] = {}
        for node in gm.graph.nodes:
            if node.op == "call_function" and "seq_nr" in node.meta:
                seq_nr = node.meta["seq_nr"]
                if seq_nr not in seq_nr_first:
                    seq_nr_first[seq_nr] = node
                    node.meta["custom"] = {"test_key": "test_value"}

        # Run the copy pass again
        _copy_fwd_metadata_to_bw_nodes(gm)

        def is_backward(node: torch.fx.Node) -> bool:
            return node.meta.get("autograd_backward", False)

        # Check that bwd nodes with shared seq_nr got the custom metadata
        for node in gm.graph.nodes:
            if node.op != "call_function" or "seq_nr" not in node.meta:
                continue
            seq_nr = node.meta["seq_nr"]
            if node is not seq_nr_first.get(seq_nr) and is_backward(node):
                # This is a backward node
                custom = node.meta.get("custom")
                self.assertIsNotNone(
                    custom,
                    f"Backward node {node.name} with seq_nr={seq_nr} missing custom metadata",
                )
                self.assertEqual(custom.get("test_key"), "test_value")

    def test_copy_fwd_metadata_uses_backward_tagging(self):
        graph = torch.fx.Graph()
        fwd = graph.call_function(torch.ops.aten.add.Tensor, args=(1, 2))
        fwd.meta["seq_nr"] = 7
        fwd.meta["custom"] = {"test_key": "test_value"}
        bwd = graph.call_function(torch.ops.aten.mul.Tensor, args=(fwd, 3))
        bwd.meta["seq_nr"] = 7
        bwd.meta["autograd_backward"] = True
        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        _copy_fwd_metadata_to_bw_nodes(gm)

        self.assertEqual(bwd.meta["custom"].get("test_key"), "test_value")

    def test_backward_nodes_have_stack_trace(self):
        """Verify that backward nodes get stack_trace from their forward counterpart."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        train_step = make_train_step(model, get_loss)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        traced = minimal_fx_tracer(train_step, module=model)(tokens, labels)

        # Find backward nodes: nodes sharing a seq_nr with an earlier (forward) node
        seq_nr_first: dict[int, torch.fx.Node] = {}
        bwd_nodes_missing_stack_trace = []
        num_checked = 0
        for node in traced.gm.graph.nodes:
            if node.op != "call_function" or "seq_nr" not in node.meta:
                continue
            seq_nr = node.meta["seq_nr"]
            if seq_nr not in seq_nr_first:
                seq_nr_first[seq_nr] = node
            else:
                # This is a backward node
                fwd_node = seq_nr_first[seq_nr]
                if not fwd_node.stack_trace:
                    continue
                num_checked += 1
                if not node.stack_trace:
                    bwd_nodes_missing_stack_trace.append((node.name, seq_nr))

        self.assertGreater(
            num_checked,
            0,
            "Expected at least one backward node with a forward stack_trace",
        )
        self.assertEqual(
            bwd_nodes_missing_stack_trace,
            [],
            f"Backward nodes missing stack_trace: {bwd_nodes_missing_stack_trace}",
        )


# Large head_dims (qwen3 head_dim=128, deepseek qk_head_dim=192) run in bf16:
# the FlexAttention Triton kernel's fp32 shared-memory footprint exceeds the
# H100 default limit (~99KB) -> "InductorError: out of resource:
# triton_tem_fused_flex_attention". bf16 halves the smem so the kernel fits.
# SDPA never hit this; it only surfaced once flex became the default LM backend.
# llama3 (head_dim 16) is small enough to stay in fp32.


def _disable_flex_autotune():
    """Disable FlexAttention max_autotune; returns the originals to restore.

    max_autotune searches flex block sizes that exceed the H100 shared-memory
    limit for larger head_dims (qwen3 head_dim=128, deepseek qk_head_dim=192),
    raising ``InductorError: out of resource: triton_tem_fused_flex_attention``.
    Disabling it falls back to the default block config (which fits) and keeps
    the eager vs regional-inductor kernels consistent. Mirrors
    ``test_bitwise_deterministic.setUp``.
    """
    from torch.nn.attention.flex_attention import flex_attention

    from torchtitan.models.common.attention import FlexAttention

    orig = (FlexAttention.inductor_configs, FlexAttention._compiled_flex_attn)
    FlexAttention.inductor_configs = {
        **FlexAttention.inductor_configs,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    FlexAttention._compiled_flex_attn = torch.compile(
        flex_attention, options=FlexAttention.inductor_configs
    )
    return orig


def _restore_flex_autotune(orig):
    from torchtitan.models.common.attention import FlexAttention

    FlexAttention.inductor_configs, FlexAttention._compiled_flex_attn = orig


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceModels(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32
    BATCH_SIZE = 2
    SEQ_LEN = 128
    NUM_STEPS = 5
    LR = 1e-3

    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        self._flex_orig = _disable_flex_autotune()

    def tearDown(self):
        _restore_flex_autotune(self._flex_orig)
        torch.use_deterministic_algorithms(False)

    def _run_bitwise_test(
        self,
        model_ref,
        model_test,
        fwd_args,
        labels,
        check_collective_ops=False,
        use_regional_inductor=False,
        num_steps=5,
        lr=1e-3,
    ):
        train_step = make_train_step(model_ref, get_loss)

        maybe_register_blockmask_pytree_node()
        traced: TracedResult = minimal_fx_tracer(train_step, module=model_ref)(
            *fwd_args, labels
        )

        if check_collective_ops:
            ag = sum(
                1
                for n in traced.gm.graph.nodes
                if "all_gather_into_tensor" in str(n.target)
            )
            rs = sum(
                1
                for n in traced.gm.graph.nodes
                if "reduce_scatter_tensor" in str(n.target)
            )
            self.assertTrue(
                ag > 0 and rs > 0,
                f"Expected collective ops in FSDP graph (ag={ag}, rs={rs})",
            )

        if use_regional_inductor:
            _apply_regional_inductor(traced)

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=lr)
        opt_copy = torch.optim.Adam(model_test.parameters(), lr=lr)

        for step in range(1, num_steps + 1):
            logits_ref = model_ref(*fwd_args)
            loss_ref = get_loss(logits_ref, labels)
            loss_ref.backward()
            grads_ref = [p.grad.clone() for p in model_ref.parameters()]
            opt_ref.step()
            opt_ref.zero_grad()

            wrapped = run_traced(traced, module=model_test)(*fwd_args, labels)
            loss_tr = wrapped[0]
            grads_tr = wrapped[1:]
            for p, g in zip(model_test.parameters(), grads_tr, strict=True):
                p.grad = g
            opt_copy.step()
            opt_copy.zero_grad()

            self.assertTrue(
                torch.equal(loss_ref, loss_tr), f"Step {step}: loss mismatch"
            )
            for gr, gt in zip(grads_ref, grads_tr, strict=True):
                self.assertTrue(torch.equal(gr, gt), f"Step {step}: grad mismatch")

    def _run_model_test(
        self,
        config_cls,
        model_config,
        use_attn_masks=False,
        use_regional_inductor=False,
        dtype=None,
    ):
        dtype = dtype or self.DTYPE
        vocab_size = model_config.vocab_size
        model_ref = create_model(config_cls, model_config, self.DEVICE, dtype)
        model_test = create_model(config_cls, model_config, self.DEVICE, dtype)
        model_test.load_state_dict(model_ref.state_dict())
        num_tokens = self.BATCH_SIZE * self.SEQ_LEN
        tokens = torch.randint(0, vocab_size, (num_tokens,), device=self.DEVICE)
        labels = torch.randint(0, vocab_size, (num_tokens,), device=self.DEVICE)

        fwd_args = (tokens,)
        if use_attn_masks:
            from torchtitan.models.common.attention import (
                create_attention_mask,
                get_causal_mask_mod,
            )

            attn_masks = create_attention_mask(
                get_causal_mask_mod(), 1, None, num_tokens, num_tokens
            )
            # Decoder.forward is (tokens, positions, attention_masks). Pass
            # explicit sequential positions (make_fx can't trace a None
            # placeholder) so the BlockMask lands in the attention_masks slot.
            positions = torch.arange(num_tokens, device=self.DEVICE)
            fwd_args = (tokens, positions, attn_masks)

        self._run_bitwise_test(
            model_ref,
            model_test,
            fwd_args,
            labels,
            use_regional_inductor=use_regional_inductor,
            num_steps=self.NUM_STEPS,
            lr=self.LR,
        )

    def test_llama3(self):
        from torchtitan.models.llama3 import llama3_configs, Llama3Model

        config = llama3_configs["debugmodel"](attn_backend="flex")
        self._run_model_test(
            Llama3Model, config, use_attn_masks=True, use_regional_inductor=True
        )

    def test_qwen3(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        config = qwen3_configs["debugmodel"](attn_backend="flex")
        self._run_model_test(
            Qwen3Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    def test_qwen3_moe(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        config = qwen3_configs["debugmodel_moe"](attn_backend="flex")
        self._run_model_test(
            Qwen3Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    def test_deepseek_v3(self):
        from torchtitan.models.deepseek_v3 import deepseekv3_configs
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        config = deepseekv3_configs["debugmodel"](
            attn_backend="flex", moe_comm_backend="standard"
        )
        self._run_model_test(
            DeepSeekV3Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    def test_deepseek_v3_flex_attention(self):
        """Tests if we can propagate fwd node metadata reliably through backward.
        Annotates FlexAttention.forward via annotate_fn before
        tracing so compile_with_inductor flows into the graph naturally.
        """
        from torch.fx.traceback import annotate_fn
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            create_attention_mask,
            FlexAttention,
            get_causal_mask_mod,
            get_document_mask_mod,
        )
        from torchtitan.models.common.linear import Linear
        from torchtitan.models.common.nn_modules import RMSNorm
        from torchtitan.models.common.rope import ComplexRoPE
        from torchtitan.models.deepseek_v3.model import Attention as DSAttention

        dim = 64
        n_heads = 4
        rope_dim = 16
        seq_len = 64
        vocab_size = 128

        # Build a tiny model: embedding -> MLA flex attention -> projection
        class TinyFlexMLA(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                kv_lora_rank = 32
                qk_nope_head_dim = 16
                v_head_dim = 16
                qk_head_dim = qk_nope_head_dim + rope_dim
                self.attn = DSAttention(
                    DSAttention.Config(
                        n_heads=n_heads,
                        dim=dim,
                        q_lora_rank=0,
                        kv_lora_rank=kv_lora_rank,
                        qk_nope_head_dim=qk_nope_head_dim,
                        qk_rope_head_dim=rope_dim,
                        v_head_dim=v_head_dim,
                        rope=ComplexRoPE.Config(
                            dim=rope_dim,
                            max_context_length=seq_len,
                            scaling="none",
                        ),
                        q_norm=RMSNorm.Config(normalized_shape=1),
                        kv_norm=RMSNorm.Config(normalized_shape=kv_lora_rank),
                        inner_attention=FlexAttention.Config(),
                        wq=Linear.Config(
                            in_features=dim,
                            out_features=n_heads * qk_head_dim,
                        ),
                        wkv_a=Linear.Config(
                            in_features=dim,
                            out_features=kv_lora_rank + rope_dim,
                        ),
                        wkv_b=Linear.Config(
                            in_features=kv_lora_rank,
                            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
                        ),
                        wo=Linear.Config(
                            in_features=n_heads * v_head_dim,
                            out_features=dim,
                        ),
                    ),
                )
                self.proj = nn.Linear(dim, vocab_size)

            def init_states(self, buffer_device=None):
                self.attn.rope._init_self_buffers(
                    buffer_device=buffer_device or torch.device("cuda")
                )

            def forward(self, tokens, block_mask):
                x = self.embed(tokens)
                x = self.attn(x, block_mask)
                return self.proj(x)

        model = TinyFlexMLA().to(device=self.DEVICE, dtype=self.DTYPE)
        with torch.no_grad():
            model.init_states(buffer_device=torch.device(self.DEVICE))

        tokens = torch.randint(0, vocab_size, (seq_len,), device=self.DEVICE)
        labels = torch.randint(0, vocab_size, (seq_len,), device=self.DEVICE)
        # Build positions that reset to 0 every 16 tokens (document boundaries)
        positions = torch.arange(seq_len, device=self.DEVICE) % 16
        block_mask = create_attention_mask(
            and_masks(get_causal_mask_mod(), get_document_mask_mod(positions)),
            B=1,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        )

        # Annotate FlexAttention.forward so compile_with_inductor flows into
        # the traced graph. Restore the original after tracing.
        orig_forward = FlexAttention.forward
        FlexAttention.forward = annotate_fn(
            {
                "compile_with_inductor": {
                    "inductor_configs": FlexAttention.inductor_configs
                }
            }
        )(FlexAttention.forward)
        try:
            train_step = make_train_step(model, get_loss)
            maybe_register_blockmask_pytree_node()
            traced = minimal_fx_tracer(train_step, module=model)(
                tokens, block_mask, labels
            )
        finally:
            FlexAttention.forward = orig_forward

        # Verify flex attention HOPs got the annotation
        for node in traced.gm.graph.nodes:
            if node.target in {
                torch.ops.higher_order.flex_attention,
                torch.ops.higher_order.flex_attention_backward,
            }:
                custom = node.meta.get("custom", {})
                self.assertIn(
                    "compile_with_inductor",
                    custom,
                    f"{node.name} missing compile_with_inductor annotation",
                )

    # TODO: Fix scatter() dtype mismatch — scatter_add expects self.dtype == src.dtype
    # but GptOss produces mismatched dtypes during tracing.
    @unittest.skip("scatter(): Expected self.dtype to be equal to src.dtype")
    def test_gpt_oss(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )
        from torchtitan.models.gpt_oss import gptoss_configs
        from torchtitan.models.gpt_oss.model import GptOssModel

        config = gptoss_configs["debugmodel"](
            moe_comm_backend="standard", attn_backend="flex"
        )
        vocab_size = config.vocab_size
        model_ref = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        model_test = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())
        num_tokens = self.BATCH_SIZE * self.SEQ_LEN
        tokens = torch.randint(0, vocab_size, (num_tokens,), device=self.DEVICE)
        labels = torch.randint(0, vocab_size, (num_tokens,), device=self.DEVICE)
        causal = get_causal_mask_mod()
        sw_size = config.layers[0].attention.sliding_window_size
        basic_mask = create_attention_mask(causal, 1, None, num_tokens, num_tokens)
        sliding_window_mask = create_attention_mask(
            and_masks(causal, get_sliding_window_mask_mod(sw_size)),
            1,
            None,
            num_tokens,
            num_tokens,
        )
        attn_masks = {
            "basic_mask": basic_mask,
            "sliding_window_mask": sliding_window_mask,
        }
        self._run_bitwise_test(
            model_ref,
            model_test,
            (tokens, attn_masks),
            labels,
            use_regional_inductor=True,
            num_steps=self.NUM_STEPS,
            lr=self.LR,
        )

    def test_flex_attention_annotations(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.experiments.graph_trainer.common_utils import (
            annotate_module_fqns,
        )
        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )
        from torchtitan.models.gpt_oss import gptoss_configs
        from torchtitan.models.gpt_oss.model import GptOssModel

        config = gptoss_configs["debugmodel"](
            moe_comm_backend="standard", attn_backend="flex"
        )
        model = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        annotate_module_fqns(model)

        num_tokens = self.BATCH_SIZE * self.SEQ_LEN
        tokens = torch.randint(0, config.vocab_size, (num_tokens,), device=self.DEVICE)
        causal = get_causal_mask_mod()
        sw_size = config.layers[0].attention.sliding_window_size
        basic_mask = create_attention_mask(causal, 1, None, num_tokens, num_tokens)
        sliding_window_mask = create_attention_mask(
            and_masks(causal, get_sliding_window_mask_mod(sw_size)),
            1,
            None,
            num_tokens,
            num_tokens,
        )
        attn_masks = {
            "basic_mask": basic_mask,
            "sliding_window_mask": sliding_window_mask,
        }
        maybe_register_blockmask_pytree_node()

        def forward(tokens, attn_masks):
            return model(tokens, attention_masks=attn_masks)

        traced = minimal_fx_tracer(forward, module=model)(tokens, attn_masks)

        flex_nodes = [
            n
            for n in traced.gm.graph.nodes
            if "flex_attention" in str(n.target) and "backward" not in str(n.target)
        ]
        self.assertGreater(len(flex_nodes), 0, "No FlexAttentionHOP nodes found")

        from torchtitan.models.common.attention import FlexAttention

        annotate_flex_attention_for_regional_inductor_pass(
            traced.gm,
            flex_compile_config=FlexAttention.inductor_configs,
        )

        for node in flex_nodes:
            custom = node.meta.get("custom", {})
            self.assertIn(
                "compile_with_inductor",
                custom,
                f"{node.name} missing compile_with_inductor annotation",
            )


class TestTraceFSDP(FSDPTest):
    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 4)

    def _setup(self):
        from torchtitan.distributed import ParallelDims

        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            spmd_backend="partial_dtensor",
        )

    def test_graph_gradient_accumulation_preserves_fsdp_layout(self):
        from torch.distributed.tensor import DTensor

        from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel

        torch.manual_seed(42)
        self._setup()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        model_ref = nn.Linear(8, 4, device="cuda")
        model_test = nn.Linear(8, 4, device="cuda")
        model_test.load_state_dict(model_ref.state_dict())
        model_ref = data_parallel(model_ref, device_mesh=fsdp_mesh, mode="fully_shard")
        model_test = data_parallel(
            model_test,
            device_mesh=fsdp_mesh,
            mode="fully_shard",
        )
        optimizer_ref = torch.optim.SGD(model_ref.parameters(), lr=0.1)
        optimizer_test = torch.optim.SGD(model_test.parameters(), lr=0.1)
        gradient_state = GraphGradientState.create(model_test, [optimizer_test])

        def train_step(inputs, targets):
            loss = torch.nn.functional.mse_loss(
                model_test(inputs),
                targets,
                reduction="sum",
            )
            grads = torch.autograd.grad(loss, tuple(model_test.parameters()))
            return [loss, *grads]

        microbatches = [
            (
                torch.randn(3, 8, device="cuda"),
                torch.randn(3, 4, device="cuda"),
            )
            for _ in range(2)
        ]
        traced = minimal_fx_tracer(
            train_step,
            module=model_test,
            graph_state=gradient_state.graph_state,
            graph_state_output_indices=tuple(range(1, len(gradient_state.buffers) + 1)),
        )(*microbatches[0])
        traced.gm = finalize_graph_gradient_accumulation(
            traced.gm,
            traced_result=traced,
        )
        run = run_traced(
            traced,
            module=model_test,
            graph_state=gradient_state.graph_state,
        )

        for inputs, targets in microbatches:
            loss_ref = torch.nn.functional.mse_loss(
                model_ref(inputs),
                targets,
                reduction="sum",
            )
            loss_ref.backward()
            outputs = run(inputs, targets)

            self.assertEqual(len(outputs), 1)
            torch.testing.assert_close(outputs[0], loss_ref)
            for parameter_ref, parameter_test, buffer in zip(
                model_ref.parameters(),
                model_test.parameters(),
                gradient_state.buffers,
                strict=True,
            ):
                self.assertIs(parameter_test.grad, buffer)
                self.assertIsInstance(buffer, DTensor)
                self.assertEqual(buffer.placements, parameter_test.placements)
                torch.testing.assert_close(
                    buffer.to_local(),
                    parameter_ref.grad.to_local(),
                )

        optimizer_ref.step()
        optimizer_test.step()
        for parameter_ref, parameter_test in zip(
            model_ref.parameters(), model_test.parameters(), strict=True
        ):
            torch.testing.assert_close(
                parameter_test.to_local(),
                parameter_ref.to_local(),
            )

    def _run_deferred_fsdp_gradient_sync_case(
        self,
        *,
        enable_cudagraph: bool,
    ) -> None:
        from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel

        torch.manual_seed(42)
        self._setup()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        model_ref = data_parallel(
            nn.Linear(8, 4, device="cuda"),
            device_mesh=fsdp_mesh,
            mode="fully_shard",
        )
        model_test = data_parallel(
            nn.Linear(8, 4, device="cuda"),
            device_mesh=fsdp_mesh,
            mode="fully_shard",
        )
        model_test.load_state_dict(model_ref.state_dict())
        optimizer_ref = torch.optim.SGD(model_ref.parameters(), lr=0.1)
        optimizer_test = torch.optim.SGD(model_test.parameters(), lr=0.1)
        gradient_state = GraphGradientState.create(model_test, [optimizer_test])

        def train_step(inputs, targets, global_valid_tokens, extra_kwargs):
            del extra_kwargs
            loss = (
                torch.nn.functional.mse_loss(
                    model_test(inputs),
                    targets,
                    reduction="sum",
                )
                / global_valid_tokens
            )
            grads = torch.autograd.grad(loss, tuple(model_test.parameters()))
            return [loss, *grads]

        microbatch_steps = [
            [
                (
                    torch.randn(3, 8, device="cuda"),
                    torch.randn(3, 4, device="cuda"),
                    {},
                )
                for _ in range(3)
            ]
            for _ in range(3)
        ]
        microbatches = microbatch_steps[0]
        global_valid_tokens = torch.tensor(
            sum(target.numel() for _, target, _ in microbatches),
            device="cuda",
        )
        traced = minimal_fx_tracer(
            train_step,
            module=model_test,
            graph_state=gradient_state.graph_state,
            graph_state_output_indices=(1, 2),
        )(
            *microbatches[0][:2],
            global_valid_tokens,
            microbatches[0][2],
        )
        num_flat_parameters = len(flatten_graph_values(list(model_test.parameters())))
        deferred = build_deferred_fsdp_graph(
            traced,
            num_flat_parameters=num_flat_parameters,
            num_microbatches=len(microbatches),
            compile_config=GraphTrainerCompileConfig(
                enable_passes=False,
                inductor_compilation="none",
                disable_passes=[] if enable_cudagraph else ["cudagraph_pass"],
                require_cudagraph=enable_cudagraph,
            ),
            enable_cudagraph=enable_cudagraph,
        )
        run = bind_deferred_fsdp_graph(
            deferred,
            module=model_test,
            gradient_state=gradient_state,
        )

        for microbatches in microbatch_steps:
            optimizer_ref.zero_grad(set_to_none=False)
            optimizer_test.zero_grad(set_to_none=False)
            loss_ref = torch.zeros((), device="cuda")
            for inputs, targets, _ in microbatches:
                loss = (
                    torch.nn.functional.mse_loss(
                        model_ref(inputs),
                        targets,
                        reduction="sum",
                    )
                    / global_valid_tokens
                )
                loss.backward()
                loss_ref += loss.detach()
            loss_test = run(microbatches, global_valid_tokens)
            torch.cuda.synchronize()

            torch.testing.assert_close(loss_test, loss_ref)
            for parameter_ref, parameter_test in zip(
                model_ref.parameters(), model_test.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    parameter_test.grad.to_local(),
                    parameter_ref.grad.to_local(),
                )
            optimizer_ref.step()
            optimizer_test.step()
            for parameter_ref, parameter_test in zip(
                model_ref.parameters(), model_test.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    parameter_test.to_local(),
                    parameter_ref.to_local(),
                )
        self.assertGreater(deferred.num_all_gathers, 0)
        self.assertGreater(deferred.num_gradient_collectives, 0)
        if enable_cudagraph:
            self.assertIsInstance(deferred.gm.forward, CUDAGraphWrapper)
            deferred.gm.forward.teardown()

    def test_deferred_fsdp_gradient_sync_matches_per_microbatch_sync(self):
        self._run_deferred_fsdp_gradient_sync_case(enable_cudagraph=False)

    def test_deferred_fsdp_gradient_sync_cuda_graph_replays_exactly_once(self):
        self._run_deferred_fsdp_gradient_sync_case(enable_cudagraph=True)

    def test_mxfp8_fsdp_gradient_state_uses_plain_local_leaves(self):
        import torchtitan.components.quantization.mx as mx

        from torchtitan.experiments.graph_trainer.simple_fsdp import (
            data_parallel,
            MixedPrecisionPolicy,
        )

        if mx.MXFP8Linear is None:
            raise unittest.SkipTest("MXFP8 dependencies are unavailable")

        self._setup()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        model = mx.MXFP8Linear(
            mx.MXFP8Linear.Config(in_features=64, out_features=64, bias=False)
        ).to(device="cuda", dtype=torch.bfloat16)
        model.configure_fsdp()
        model = data_parallel(
            model,
            device_mesh=fsdp_mesh,
            mode="fully_shard",
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
        )
        parameter = next(model.parameters())
        self.assertIsInstance(parameter, DTensor)
        self.assertIsInstance(parameter.to_local(), mx._MXFP8FSDPWeight)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        gradient_state = GraphGradientState.create(model, [optimizer])
        buffer = gradient_state.buffers[0]
        self.assertIsInstance(buffer, DTensor)
        self.assertIs(type(buffer.to_local()), torch.Tensor)
        self.assertEqual(buffer.placements, parameter.placements)
        self.assertIs(parameter.grad, buffer)

        class SyntheticMXFP8Use(torch.autograd.Function):
            @staticmethod
            def forward(ctx, weight):
                ctx.weight_shape = weight.shape
                return torch.ones((), dtype=weight.dtype, device=weight.device)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.new_ones(ctx.weight_shape) * grad_output

        def train_step():
            traced_parameter = next(model.parameters())
            loss = SyntheticMXFP8Use.apply(model.weight)
            gradient = torch.autograd.grad(loss, (traced_parameter,))[0]
            return [loss, gradient]

        traced = minimal_fx_tracer(
            train_step,
            module=model,
            graph_state=gradient_state.graph_state,
            graph_state_output_indices=(1,),
        )()
        gradient_layout = traced.output_subclass_layouts[1]
        assert gradient_layout.meta is not None
        self.assertIs(gradient_layout.meta.cls, DTensor)
        num_local_leaves, local_meta = gradient_layout.meta.inner_metas["_local_tensor"]
        self.assertEqual(num_local_leaves, 1)
        self.assertIsNone(local_meta)

        finalize_graph_gradient_accumulation(
            traced.gm,
            traced_result=traced,
        )
        sinks = [
            node
            for node in traced.gm.graph.nodes
            if node.meta.get("graph_gradient_fqn") == "weight"
        ]
        self.assertTrue(traced.grad_sink_active)
        self.assertEqual(len(sinks), 1)

    def _run_deferred_mxfp8_wgrad_fusion_case(
        self,
        *,
        enable_cudagraph: bool,
    ) -> None:
        import torchtitan.components.quantization.mx as mx

        from torchtitan.experiments.graph_trainer.fsdp_passes import (
            deduplicate_fsdp_unshard_chains_pass,
        )
        from torchtitan.experiments.graph_trainer.simple_fsdp import (
            data_parallel,
            MixedPrecisionPolicy,
        )

        if mx.MXFP8Linear is None:
            raise unittest.SkipTest("MXFP8 dependencies are unavailable")
        if torch.cuda.get_device_capability() < (10, 0):
            raise unittest.SkipTest("MXFP8 WGrad accumulation requires SM100")

        torch.manual_seed(42)
        self._setup()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        model_ref = mx.MXFP8Linear(
            mx.MXFP8Linear.Config(
                in_features=64,
                out_features=64,
                bias=False,
            )
        ).to(device="cuda", dtype=torch.bfloat16)
        model_test = deepcopy(model_ref)
        for model in (model_ref, model_test):
            model.configure_fsdp()
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )
        model_ref = data_parallel(
            model_ref,
            device_mesh=fsdp_mesh,
            mode="fully_shard",
            mp_policy=mp_policy,
        )
        model_test = data_parallel(
            model_test,
            device_mesh=fsdp_mesh,
            mode="fully_shard",
            mp_policy=mp_policy,
        )
        optimizer_ref = torch.optim.SGD(model_ref.parameters(), lr=0.1)
        optimizer_test = torch.optim.SGD(model_test.parameters(), lr=0.1)
        gradient_state_ref = GraphGradientState.create(model_ref, [optimizer_ref])
        gradient_state_test = GraphGradientState.create(model_test, [optimizer_test])
        microbatches = [
            (
                torch.randn(32, 64, device="cuda", dtype=torch.bfloat16),
                torch.randn(32, 64, device="cuda", dtype=torch.bfloat16),
                {},
            )
            for _ in range(3)
        ]
        global_valid_tokens = torch.tensor(
            sum(target.numel() for _, target, _ in microbatches),
            device="cuda",
        )

        def build(
            model: nn.Module,
            gradient_state: GraphGradientState,
            *,
            fuse_wgrad: bool,
        ):
            def train_step(inputs, targets, valid_tokens, extra_kwargs):
                del extra_kwargs
                loss = (
                    torch.nn.functional.mse_loss(
                        model(inputs).float(),
                        targets.float(),
                        reduction="sum",
                    )
                    / valid_tokens
                )
                gradients = torch.autograd.grad(loss, tuple(model.parameters()))
                return [loss, *gradients]

            traced = minimal_fx_tracer(
                train_step,
                module=model,
                graph_state=gradient_state.graph_state,
                graph_state_output_indices=(1,),
            )(
                *microbatches[0][:2],
                global_valid_tokens,
                microbatches[0][2],
            )
            deduplicate_fsdp_unshard_chains_pass(traced.gm, traced.example_inputs)
            deferred = build_deferred_fsdp_graph(
                traced,
                num_flat_parameters=len(flatten_graph_values(list(model.parameters()))),
                num_microbatches=len(microbatches),
                compile_config=GraphTrainerCompileConfig(
                    enable_passes=True,
                    inductor_compilation="none",
                    numerics_changing_optim=fuse_wgrad,
                    disable_passes=([] if enable_cudagraph else ["cudagraph_pass"]),
                    require_cudagraph=enable_cudagraph,
                ),
                enable_cudagraph=enable_cudagraph,
            )
            return deferred, bind_deferred_fsdp_graph(
                deferred,
                module=model,
                gradient_state=gradient_state,
            )

        deferred_ref, run_ref = build(
            model_ref,
            gradient_state_ref,
            fuse_wgrad=False,
        )
        deferred_test, run_test = build(
            model_test,
            gradient_state_test,
            fuse_wgrad=True,
        )
        scaled_mm = torch.ops.aten._scaled_mm.default
        scaled_addmm = torch.ops.aten._scaled_addmm_.default
        self.assertGreater(
            sum(
                node.target == scaled_mm for node in deferred_test.gm.first.graph.nodes
            ),
            0,
        )
        self.assertEqual(
            sum(
                node.target == scaled_addmm
                for node in deferred_test.gm.first.graph.nodes
            ),
            0,
        )
        for name in ("middle", "final"):
            reference_child = getattr(deferred_ref.gm, name)
            fused_child = getattr(deferred_test.gm, name)
            self.assertEqual(
                sum(
                    node.target == scaled_addmm for node in reference_child.graph.nodes
                ),
                0,
            )
            self.assertEqual(
                sum(node.target == scaled_addmm for node in fused_child.graph.nodes),
                1,
            )

        num_steps = 3 if enable_cudagraph else 2
        for _ in range(num_steps):
            optimizer_ref.zero_grad(set_to_none=False)
            optimizer_test.zero_grad(set_to_none=False)
            loss_ref = run_ref(microbatches, global_valid_tokens)
            loss_test = run_test(microbatches, global_valid_tokens)
            torch.cuda.synchronize()
            torch.testing.assert_close(loss_test, loss_ref)
            for parameter_ref, parameter_test in zip(
                model_ref.parameters(), model_test.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    parameter_test.grad.to_local(),
                    parameter_ref.grad.to_local(),
                    rtol=2e-2,
                    atol=2e-2,
                )
            optimizer_ref.step()
            optimizer_test.step()
            for parameter_ref, parameter_test in zip(
                model_ref.parameters(), model_test.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    parameter_test.to_local(),
                    parameter_ref.to_local(),
                    rtol=2e-2,
                    atol=2e-2,
                )
        if enable_cudagraph:
            self.assertIsNotNone(deferred_test.gm.forward._cudagraph)
            deferred_ref.gm.forward.teardown()
            deferred_test.gm.forward.teardown()

    def test_deferred_mxfp8_wgrad_fusion_runs_without_cuda_graph(self) -> None:
        self._run_deferred_mxfp8_wgrad_fusion_case(enable_cudagraph=False)

    def test_deferred_mxfp8_wgrad_fusion_runs_with_cuda_graph(self) -> None:
        self._run_deferred_mxfp8_wgrad_fusion_case(enable_cudagraph=True)

    def test_fused_wgrad_stride_accumulates_in_graph(self):
        from torchtitan.experiments.graph_trainer.simple_fsdp import (
            data_parallel,
            MixedPrecisionPolicy,
        )

        class FusedProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.w13 = nn.Parameter(torch.randn(64, 2, 32, device="cuda"))

            def forward(self, x):
                return torch.einsum("...d,hgd->...hg", x, self.w13)

        torch.manual_seed(42)
        self._setup()
        model = data_parallel(
            FusedProjection(),
            device_mesh=self.parallel_dims.get_mesh("fsdp"),
            mode="fully_shard",
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        gradient_state = GraphGradientState.create(model, [optimizer])

        def train_step(inputs):
            loss = model(inputs).float().sum()
            gradient = torch.autograd.grad(loss, tuple(model.parameters()))[0]
            return [loss, gradient]

        microbatches = [
            torch.randn(4, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2)
        ]
        traced = minimal_fx_tracer(
            train_step,
            module=model,
            graph_state=gradient_state.graph_state,
            graph_state_output_indices=(1,),
        )(microbatches[0])
        buffer_meta = traced.input_subclass_layouts[len(traced.state_fqns)].meta
        gradient_meta = traced.output_subclass_layouts[1].meta
        assert buffer_meta is not None
        assert gradient_meta is not None
        self.assertEqual(buffer_meta.outer_size, torch.Size((64, 2, 32)))
        self.assertEqual(gradient_meta.outer_size, buffer_meta.outer_size)
        self.assertNotEqual(gradient_meta.outer_stride, buffer_meta.outer_stride)

        traced.gm = finalize_graph_gradient_accumulation(
            traced.gm,
            traced_result=traced,
        )
        run = bind_traced(
            traced,
            module=model,
            graph_state=gradient_state.graph_state,
        )
        run.validate_state(
            module=model,
            graph_state=gradient_state.graph_state,
        )
        expected_gradient = torch.zeros_like(gradient_state.buffers[0].to_local())
        for inputs in microbatches:
            expected_loss, gradient = train_step(inputs)
            expected_gradient.add_(gradient.to_local())
            outputs = run(inputs)
            self.assertEqual(len(outputs), 1)
            torch.testing.assert_close(outputs[0], expected_loss)
            self.assertTrue(
                torch.equal(
                    gradient_state.buffers[0].to_local(),
                    expected_gradient,
                )
            )

    def test_terminal_sink_validates_dtensor_device_mesh(self):
        from torch.distributed.device_mesh import DeviceMesh

        self._setup()
        mesh = self.parallel_dims.get_mesh("fsdp")
        equal_mesh = DeviceMesh(
            mesh.device_type,
            mesh.mesh.clone(),
            mesh_dim_names=mesh.mesh_dim_names,
            _init_backend=False,
        )
        different_mesh = DeviceMesh(
            mesh.device_type,
            mesh.mesh.clone(),
            mesh_dim_names=("different",),
            _init_backend=False,
        )
        graph = torch.fx.Graph()
        buffer = graph.placeholder("buffer_mesh")
        gradient = graph.placeholder("gradient_mesh")
        buffer.meta["val"] = mesh
        gradient.meta["val"] = equal_mesh
        _validate_device_mesh_leaf("weight", buffer, gradient)

        gradient.meta["val"] = different_mesh
        with self.assertRaisesRegex(ValueError, "device mesh does not match"):
            _validate_device_mesh_leaf("weight", buffer, gradient)

        gradient.meta.pop("val")
        with self.assertRaisesRegex(ValueError, "device mesh does not match"):
            _validate_device_mesh_leaf("weight", buffer, gradient)

    def _run_fsdp_model_test(
        self,
        config_cls,
        model_config,
        use_attn_masks=False,
        attn_masks=None,
        use_regional_inductor=False,
        dtype=torch.float32,
    ):
        from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        self._setup()
        # FSDPTest.setUp runs in the parent, so disable flex max_autotune here
        # (in the child process) to keep flex kernels within the H100 shared
        # memory limit. No restore needed: each rank is a fresh subprocess.
        _disable_flex_autotune()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")

        model_ref = create_model(config_cls, model_config, "cuda", dtype)
        model_test = create_model(config_cls, model_config, "cuda", dtype)
        model_test.load_state_dict(model_ref.state_dict())
        model_ref = data_parallel(model_ref, device_mesh=fsdp_mesh, mode="fully_shard")
        model_test = data_parallel(
            model_test, device_mesh=fsdp_mesh, mode="fully_shard"
        )

        vocab_size = model_config.vocab_size
        seq_len = 128
        num_tokens = 2 * seq_len
        tokens = torch.randint(0, vocab_size, (num_tokens,), device="cuda")
        labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda")
        # Decoder.forward is (tokens, positions, attention_masks). Pass explicit
        # sequential positions (make_fx can't trace a None placeholder) so the
        # BlockMask lands in the attention_masks slot.
        positions = torch.arange(num_tokens, device="cuda")

        if attn_masks is not None:
            fwd_args = (tokens, positions, attn_masks)
        elif use_attn_masks:
            from torchtitan.models.common.attention import (
                create_attention_mask,
                get_causal_mask_mod,
            )

            attn_masks = create_attention_mask(
                get_causal_mask_mod(), 1, None, num_tokens, num_tokens
            )
            fwd_args = (tokens, positions, attn_masks)
        else:
            fwd_args = (tokens,)

        train_step = make_train_step(model_ref, get_loss)

        maybe_register_blockmask_pytree_node()
        traced = minimal_fx_tracer(train_step, module=model_ref)(*fwd_args, labels)

        ag = sum(
            1
            for n in traced.gm.graph.nodes
            if "all_gather_into_tensor" in str(n.target)
        )
        rs = sum(
            1 for n in traced.gm.graph.nodes if "reduce_scatter_tensor" in str(n.target)
        )
        self.assertTrue(
            ag > 0 and rs > 0,
            f"Expected collective ops in FSDP graph (ag={ag}, rs={rs})",
        )

        if use_regional_inductor:
            _apply_regional_inductor(traced)

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=1e-3)
        opt_copy = torch.optim.Adam(model_test.parameters(), lr=1e-3)

        for step in range(1, 6):
            logits_ref = model_ref(*fwd_args)
            loss_ref = get_loss(logits_ref, labels)
            loss_ref.backward()
            grads_ref = [p.grad.clone() for p in model_ref.parameters()]
            opt_ref.step()
            opt_ref.zero_grad()

            wrapped = run_traced(traced, module=model_test)(*fwd_args, labels)
            loss_tr = wrapped[0]
            grads_tr = wrapped[1:]
            for p, g in zip(model_test.parameters(), grads_tr, strict=True):
                p.grad = g
            opt_copy.step()
            opt_copy.zero_grad()

            self.assertTrue(
                torch.equal(loss_ref, loss_tr), f"Step {step}: loss mismatch"
            )
            for gr, gt in zip(grads_ref, grads_tr, strict=True):
                self.assertTrue(torch.equal(gr, gt), f"Step {step}: grad mismatch")

    def test_chunked_lm_head_uses_one_bf16_gradient_reduction(self):
        self._run_chunked_lm_head_gradient_reduction(torch.bfloat16)

    def test_chunked_lm_head_uses_one_fp32_gradient_reduction(self):
        self._run_chunked_lm_head_gradient_reduction(torch.float32)

    def _run_chunked_lm_head_gradient_reduction(
        self,
        reduce_dtype: torch.dtype,
    ) -> None:
        from torch.distributed.fsdp import (
            fully_shard,
            MixedPrecisionPolicy as FSDPMixedPrecisionPolicy,
        )

        from torchtitan.components.loss import ChunkedLossWrapper
        from torchtitan.distributed.fsdp import disable_fsdp_gradient_division
        from torchtitan.experiments.graph_trainer.simple_fsdp import (
            data_parallel,
            MixedPrecisionPolicy,
        )

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self._setup()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        dim, vocab_size, num_chunks = 32, 64, 8

        reference_head = nn.Linear(dim, vocab_size, bias=False, device="cuda")
        traced_head = nn.Linear(dim, vocab_size, bias=False, device="cuda")
        traced_head.load_state_dict(reference_head.state_dict())
        fully_shard(
            reference_head,
            mesh=fsdp_mesh,
            mp_policy=FSDPMixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_dtype,
            ),
        )
        disable_fsdp_gradient_division(reference_head)
        traced_head = data_parallel(
            traced_head,
            device_mesh=fsdp_mesh,
            mode="fully_shard",
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_dtype,
            ),
        )

        hidden_states = torch.randn(
            2,
            16,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab_size, (2, 16), device="cuda")

        reference_hidden = hidden_states.detach().clone().requires_grad_(True)
        reference_loss_fn = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(num_chunks=num_chunks)
        )
        reference_loss_fn.set_lm_head(reference_head)
        reference_loss, _ = reference_loss_fn(reference_hidden, labels)
        reference_loss.backward()
        reference_outputs = (
            reference_loss,
            reference_hidden.grad,
            next(reference_head.parameters()).grad,
        )

        def train_step(hidden_states, labels):
            loss_fn = ChunkedLossWrapperWithParamGrads(
                ChunkedLossWrapperWithParamGrads.Config(num_chunks=num_chunks)
            )
            loss_fn.set_weight_gradient_reduce_dtype(reduce_dtype)
            loss_fn.set_lm_head(traced_head)
            loss, _ = loss_fn(hidden_states, labels)
            hidden_grad, weight_grad = torch.autograd.grad(
                loss,
                (hidden_states, *traced_head.parameters()),
            )
            return loss, hidden_grad, weight_grad

        eager_outputs = train_step(hidden_states, labels)
        for reference, eager in zip(reference_outputs, eager_outputs, strict=True):
            self.assertTrue(torch.equal(reference, eager))

        traced = minimal_fx_tracer(train_step, module=traced_head)(
            hidden_states,
            labels,
        )
        all_gathers = [
            node
            for node in traced.gm.graph.nodes
            if node.target is torch.ops._c10d_functional.all_gather_into_tensor.default
        ]
        reduce_scatters = [
            node
            for node in traced.gm.graph.nodes
            if node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
        ]
        self.assertEqual(len(all_gathers), 1)
        self.assertEqual(len(reduce_scatters), 1)
        reduce_scatter_input = reduce_scatters[0].args[0]
        self.assertIsInstance(reduce_scatter_input, torch.fx.Node)
        self.assertEqual(reduce_scatter_input.meta["val"].dtype, reduce_dtype)
        sharded_weight = next(traced_head.parameters())
        self.assertEqual(eager_outputs[2].dtype, sharded_weight.dtype)
        self.assertEqual(eager_outputs[2].placements, sharded_weight.placements)

        replay_outputs = run_traced(traced, module=traced_head)(
            hidden_states,
            labels,
        )
        for eager, replay in zip(eager_outputs, replay_outputs, strict=True):
            self.assertTrue(torch.equal(eager, replay))

    def test_llama3_fsdp(self):
        from torchtitan.models.llama3 import llama3_configs, Llama3Model

        config = llama3_configs["debugmodel"](attn_backend="flex")
        self._run_fsdp_model_test(
            Llama3Model, config, use_attn_masks=True, use_regional_inductor=True
        )

    def test_qwen3_fsdp(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        config = qwen3_configs["debugmodel"](attn_backend="flex")
        self._run_fsdp_model_test(
            Qwen3Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    def test_deepseek_v3_fsdp(self):
        from torchtitan.models.deepseek_v3 import deepseekv3_configs
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        config = deepseekv3_configs["debugmodel"](
            attn_backend="flex", moe_comm_backend="standard"
        )
        self._run_fsdp_model_test(
            DeepSeekV3Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    # TODO: Fix scatter() dtype mismatch — same root cause as TestTraceModels.test_gpt_oss.
    @unittest.skip("scatter(): Expected self.dtype to be equal to src.dtype")
    def test_gpt_oss_fsdp(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )
        from torchtitan.models.gpt_oss import gptoss_configs
        from torchtitan.models.gpt_oss.model import GptOssModel

        config = gptoss_configs["debugmodel"](
            moe_comm_backend="standard", attn_backend="flex"
        )
        seq_len = 128
        num_tokens = 2 * seq_len
        causal = get_causal_mask_mod()
        sw_size = config.layers[0].attention.sliding_window_size
        basic_mask = create_attention_mask(causal, 1, None, num_tokens, num_tokens)
        sliding_window_mask = create_attention_mask(
            and_masks(causal, get_sliding_window_mask_mod(sw_size)),
            1,
            None,
            num_tokens,
            num_tokens,
        )
        attn_masks = {
            "basic_mask": basic_mask,
            "sliding_window_mask": sliding_window_mask,
        }
        self._run_fsdp_model_test(
            GptOssModel,
            config,
            attn_masks=attn_masks,
            use_regional_inductor=True,
        )


# TODO: Re-enable after graph_trainer adopts spmd_types; partial_dtensor does
# not apply the CP placements declared in ShardingConfig.
@unittest.skip("Context Parallel is not supported by graph_trainer")
@unittest.skipIf(torch.cuda.device_count() < 2, "CP trace test requires 2 GPUs")
class TestTraceContextParallel(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _trace_llama3_step_code(
        self,
        *,
        dp_shard_degree: int,
        context_parallel_degree: int,
    ) -> dict[str, object]:
        import os
        import tempfile

        import torch.distributed as dist

        from torchtitan.experiments.graph_trainer.llama3.config_registry import (
            graph_trainer_llama3_debugmodel_sdpa,
        )
        from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

        old_local_rank = os.environ.get("LOCAL_RANK")
        os.environ["LOCAL_RANK"] = str(dist.get_rank() % torch.cuda.device_count())

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        trainer = None
        try:
            with tempfile.TemporaryDirectory() as dump_folder:
                config = graph_trainer_llama3_debugmodel_sdpa()
                config.dump_folder = dump_folder
                config.training.max_context_length = 128
                config.training.num_tokens_per_microbatch_per_dp_rank = 2 * 128
                config.training.steps = 1
                config.parallelism.data_parallel_replicate_degree = 1
                config.parallelism.data_parallel_shard_degree = dp_shard_degree
                config.parallelism.context_parallel_degree = context_parallel_degree
                config.parallelism.tensor_parallel_degree = 1
                config.activation_checkpoint = None
                config.compile.enable = False
                config.compile.enable_passes = False
                config.debug.enable_structured_logging = False
                config.model_spec.model.layers = config.model_spec.model.layers[:1]

                trainer = GraphTrainer(config)
                num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
                tokens = torch.randint(
                    0,
                    trainer.model_config.vocab_size,
                    (num_tokens,),
                    device=trainer.device,
                )
                labels = torch.randint(
                    0,
                    trainer.model_config.vocab_size,
                    (num_tokens,),
                    device=trainer.device,
                )
                # The dataloader always supplies per-document positions, which
                # drive RoPE (SDPA itself is maskless and uses is_causal).
                positions = (
                    torch.arange(
                        num_tokens,
                        device=trainer.device,
                        dtype=torch.int32,
                    )
                    % config.training.max_context_length
                )
                trainer.forward_backward_step(
                    input_dict={"input": tokens, "positions": positions},
                    labels=labels,
                    global_valid_tokens=torch.tensor(
                        labels.numel(), device=trainer.device
                    ),
                )
                assert trainer._traced_step is not None
                code_lines = trainer._traced_step.gm.graph.python_code(
                    "self"
                ).src.splitlines()
                sdpa_line = next(
                    (
                        idx
                        for idx, line in enumerate(code_lines)
                        if "scaled_dot_product" in line
                    ),
                    None,
                )
                self.assertIsNotNone(
                    sdpa_line,
                    "Expected SDPA in generated code:\n" + "\n".join(code_lines),
                )
                assert sdpa_line is not None
                all_gather_pg_names_before_sdpa = []
                for node in trainer._traced_step.gm.graph.nodes:
                    if "scaled_dot_product" in str(node.target):
                        break
                    if "all_gather_into_tensor" in str(node.target):
                        all_gather_pg_names_before_sdpa.append(node.args[2])

                cp_pg_name = (
                    trainer.parallel_dims.get_mesh("cp").get_group().group_name
                    if trainer.parallel_dims.cp_enabled
                    else None
                )
                fsdp_pg_name = (
                    trainer.parallel_dims.get_mesh("fsdp").get_group().group_name
                )
                code = trainer._traced_step.gm.graph.python_code("self").src
                trainer.close()
                trainer = None
                return {
                    "code": code,
                    "all_gather_pg_names_before_sdpa": (
                        all_gather_pg_names_before_sdpa
                    ),
                    "cp_pg_name": cp_pg_name,
                    "fsdp_pg_name": fsdp_pg_name,
                }
        finally:
            if trainer is not None:
                trainer.close()
            if old_local_rank is None:
                os.environ.pop("LOCAL_RANK", None)
            else:
                os.environ["LOCAL_RANK"] = old_local_rank

    # Pinned to the SDPA backend: this validates CP all_gather-before-SDPA
    # codegen, which requires the scaled_dot_product op. The default
    # FlexAttention backend has no SDPA op and flex + CP is unsupported anyway
    # (torch's _create_cp_block_mask requires seq_len divisible by 2 *
    # BLOCK_SIZE, here 128 < 256; see the aot_fx_trace_llama3_fsdp_tp_cp
    # integration flavor). SDPA has native CP support and emits the SDPA op.
    def test_llama3_cp_only_codegen_all_gather_before_sdpa(self):
        cp_trace = self._trace_llama3_step_code(
            dp_shard_degree=1,
            context_parallel_degree=2,
        )
        # Verify AG along CP PG exists before SDPA
        self.assertIn(
            cp_trace["cp_pg_name"],
            cp_trace["all_gather_pg_names_before_sdpa"],
            "Expected CP all_gather on the CP mesh before SDPA. "
            f"CP pg={cp_trace['cp_pg_name']}, "
            f"FSDP pg={cp_trace['fsdp_pg_name']}, "
            "pre-SDPA all_gather pgs="
            f"{cp_trace['all_gather_pg_names_before_sdpa']}.\n"
            f"Generated code:\n{cp_trace['code']}",
        )


class TestAutogradGradVsBackwardFSDP(FSDPTest):
    """Verify autograd.grad() and loss.backward() have identical peak memory with FSDP."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 4)

    def test_peak_memory_identical_fsdp(self):
        from torchtitan.distributed import ParallelDims
        from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
        from torchtitan.models.llama3 import llama3_configs, Llama3Model

        config = llama3_configs["debugmodel"](attn_backend="flex")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)

        try:
            parallel_dims = ParallelDims(
                dp_shard=-1,
                dp_replicate=1,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                world_size=self.world_size,
                spmd_backend="partial_dtensor",
            )
            fsdp_mesh = parallel_dims.get_mesh("fsdp")

            model_backward = create_model(Llama3Model, config, "cuda", torch.bfloat16)
            model_grad = create_model(Llama3Model, config, "cuda", torch.bfloat16)
            model_grad.load_state_dict(model_backward.state_dict())
            model_backward = data_parallel(
                model_backward, device_mesh=fsdp_mesh, mode="fully_shard"
            )
            model_grad = data_parallel(
                model_grad, device_mesh=fsdp_mesh, mode="fully_shard"
            )

            num_tokens = 2 * 128
            tokens = torch.randint(0, config.vocab_size, (num_tokens,), device="cuda")
            labels = torch.randint(0, config.vocab_size, (num_tokens,), device="cuda")

            from torchtitan.models.common.attention import (
                create_attention_mask,
                get_causal_mask_mod,
            )

            attention_masks = create_attention_mask(
                get_causal_mask_mod(), 1, None, num_tokens, num_tokens
            )

            def run_backward(model):
                logits = model(tokens, attention_masks=attention_masks)
                loss = get_loss(logits, labels)
                loss.backward()

            def run_grad(model):
                logits = model(tokens, attention_masks=attention_masks)
                loss = get_loss(logits, labels)
                params = [p for p in model.parameters() if p.requires_grad]
                grads = torch.autograd.grad(loss, params)
                for p, g in zip(params, grads, strict=True):
                    p.grad = g

            # Warmup
            run_backward(model_backward)
            model_backward.zero_grad()
            run_grad(model_grad)
            model_grad.zero_grad()
            torch.cuda.empty_cache()

            # Measure backward()
            torch.cuda.reset_peak_memory_stats()
            run_backward(model_backward)
            peak_backward = torch.cuda.max_memory_allocated()
            model_backward.zero_grad()
            torch.cuda.empty_cache()

            # Measure autograd.grad()
            torch.cuda.reset_peak_memory_stats()
            run_grad(model_grad)
            peak_grad = torch.cuda.max_memory_allocated()
            model_grad.zero_grad()
            torch.cuda.empty_cache()

            self.assertEqual(
                peak_backward,
                peak_grad,
                f"Peak memory differs: backward()={peak_backward / 1e9:.2f} GB "
                f"vs autograd.grad()={peak_grad / 1e9:.2f} GB",
            )
        finally:
            torch.use_deterministic_algorithms(prev_deterministic)


if __name__ == "__main__":
    unittest.main()
