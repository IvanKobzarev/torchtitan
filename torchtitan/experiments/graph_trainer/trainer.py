# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch._inductor.fx_passes.bucketing import (
    has_preallocated_fsdp_symmetric_memory_buffers,
    release_preallocated_fsdp_symmetric_memory_buffers,
)
from torch.distributed.tensor import DTensor

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    accumulate_param_grads_,
    compute_annotated_loss,
    get_transformer_block_buckets,
    log_timer,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    trace_input_preparer_keys,
)
from torchtitan.experiments.graph_trainer.cudagraph import (
    cudagraph_teardown,
    CUDAGraphWrapper,
)
from torchtitan.experiments.graph_trainer.deferred_fsdp import (
    bind_deferred_fsdp_graph,
    BoundDeferredFSDPRunner,
    build_deferred_fsdp_graph,
    DeferredFSDPGraph,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    configure_fsdp_symmetric_memory_backend,
    deduplicate_fsdp_unshard_chains_pass,
)
from torchtitan.experiments.graph_trainer.gradient_accumulation import (
    finalize_graph_gradient_accumulation,
    GraphGradientState,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import flatten_graph_values
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    bind_traced,
    BoundTracedRunner,
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    validate_memory_policy_config,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    canonicalize_graph_pass,
    compile_time_passes,
    construct_default_graph_passes,
    eliminate_dead_code_pass,
)
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    POST_INIT_HOOKS,
    PRE_TRAIN_STEP_HOOKS,
    TRACE_CALL_INPUT_PREPARERS,
    TRACE_INPUT_PREPARERS,
)
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


_FAKE_SPMD_COMM_MODES = {
    "fake_backend",
    "local_tensor",
    "real_pp_fake_spmd_backend",
}


def _maybe_apply_numa_binding(device_index: int, device_type: str) -> None:
    """Pin this process to the NUMA node of its GPU for local memory bandwidth.

    On multi-NUMA machines (e.g. GB200 NVLink-C2C), pinned-memory allocations
    that land on the GPU's local NUMA node get ~350 GB/s D2H bandwidth vs
    ~120 GB/s cross-NUMA. Must run before any pinned memory is allocated.
    """
    if device_type != "cuda":
        return
    from torch.numa.binding import (
        _maybe_apply_numa_binding_to_current_process,
        AffinityMode,
        NumaOptions,
    )

    _maybe_apply_numa_binding_to_current_process(
        device_index=device_index,
        numa_options=NumaOptions(
            affinity_mode=AffinityMode.NODE,
            should_fall_back_if_binding_fails=True,
        ),
    )
    logger.info("NUMA binding applied for GPU %d", device_index)


def make_fwd_bwd_step(model, loss_fn):
    """Return a plain function that traces the entire fwd+loss+bwd step.

    ``model`` and ``loss_fn`` are captured in the closure so neither shows up
    as a graph input. Pass ``model`` through ``minimal_fx_tracer(fn, module=model)``
    to thread its parameters/buffers as static graph inputs.
    """

    def fwd_bwd_step(inputs, labels, global_valid_tokens, extra_kwargs):
        pred = model(inputs, **extra_kwargs)
        # The loss function is not a submodule of the model, so
        # annotate_module_fqns won't tag it. Annotate it here so that
        # downstream passes (bucketing, SAC, kernel annotations) can
        # attribute loss nodes in the traced graph.
        loss = compute_annotated_loss(
            loss_fn,
            pred,
            labels,
            {"global_valid_tokens": global_valid_tokens},
        )
        params = [
            p
            for _, p in model.named_parameters(remove_duplicate=False)
            if p.requires_grad
        ]
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)

    return fwd_bwd_step


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def __init__(self, config):
        super().__init__(config)

        if isinstance(self.loss_fn, ChunkedLossWrapperWithParamGrads):
            self.loss_fn.set_weight_gradient_reduce_dtype(
                TORCH_DTYPE_MAP[config.training.mixed_precision_reduce]
            )

        validate_memory_policy_config(self.config.compile)

        _maybe_apply_numa_binding(self.device.index, self.device.type)

        # Lazy state for aot_fx_trace mode
        self._traced_step: TracedResult | None = None
        self._bound_traced_step: BoundTracedRunner | None = None
        self._trainable_params: tuple[torch.Tensor, ...] | None = None
        self._graph_gradient_state: GraphGradientState | None = None
        self._deferred_fsdp_graph: DeferredFSDPGraph | None = None
        self._bound_deferred_fsdp_graph: BoundDeferredFSDPRunner | None = None
        self._validate_graph_gradient_accumulation_config()

        if self.config.compile.memory_policy == "sac_and_offload":
            from torch._functorch._activation_offloading.offload_ops import (
                pinned_memory_pool,
            )

            self._pinned_pool_ctx = pinned_memory_pool()
            self._pinned_pool_ctx.__enter__()
        else:
            self._pinned_pool_ctx = None

        # Run post-init hook for the active pass pipeline
        POST_INIT_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(self)

    def init_distributed(self):
        comm_mode = getattr(getattr(self.config, "comm", None), "mode", "default")
        if (
            self.config.parallelism.enable_fsdp_symm_mem
            and comm_mode not in _FAKE_SPMD_COMM_MODES
            and (
                not torch.distributed.is_initialized()
                or torch.distributed.get_backend() != torch.distributed.Backend.FAKE
            )
        ):
            configure_fsdp_symmetric_memory_backend()
        parallel_dims = super().init_distributed()
        self._validate_fsdp_symmetric_memory_config(parallel_dims=parallel_dims)
        return parallel_dims

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
        labels: torch.Tensor | list[torch.Tensor],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_graph_gradient_accumulation_config()
        if self.parallel_dims.pp_enabled or self.config.compile.mode != "aot_fx_trace":
            return super().forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )

        assert isinstance(input_dict, dict)
        assert isinstance(labels, torch.Tensor)
        assert len(self.model_parts) == 1
        model = self.model_parts[0]

        inputs, labels, extra_kwargs = self.post_dataloading_process(input_dict, labels)
        params = self._get_trainable_parameters(model)
        return self._make_fx_forward_backward_step(
            model,
            inputs,
            labels,
            global_valid_tokens,
            params,
            extra_kwargs,
        )

    def _load_precompiled_fx_trace(self, model: nn.Module) -> None:
        """Load a precompiled aot_fx_trace artifact from disk."""
        from torchtitan.experiments.graph_trainer.precompile import (
            _FX_TRACE_ARTIFACT_KEY,
            compute_config_fingerprint,
            precompile_fx_trace_load,
        )
        from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter

        compile_config = self.config.compile
        storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)

        if not storage.exists(_FX_TRACE_ARTIFACT_KEY):
            raise ValueError(
                f"Precompiled fx_trace artifact not found at "
                f"'{compile_config.precompile_artifact_dir}/{_FX_TRACE_ARTIFACT_KEY}'. "
                f"Run precompile_main with --compile.mode aot_fx_trace first."
            )

        config_fingerprint = compute_config_fingerprint(
            model,
            compile_config,
            self.parallel_dims,
            mixed_precision_reduce=self.config.training.mixed_precision_reduce,
        )

        self._traced_step = precompile_fx_trace_load(
            storage,
            expected_fingerprint=config_fingerprint,
        )

    def _make_fx_forward_backward_step(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        params: tuple[torch.Tensor, ...],
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        gradient_state = self._graph_gradient_state
        if self.config.compile.enable_graph_gradient_accumulation:
            gradient_state = self._ensure_graph_gradient_state(model)
        if self._traced_step is None:
            maybe_register_blockmask_pytree_node()
            if self.config.compile.precompile_artifact_dir:
                self._load_precompiled_fx_trace(model)
            else:
                fwd_bwd_fn = make_fwd_bwd_step(model, self.loss_fn)
                with self.train_context(), log_timer("minimal_fx_tracer"):
                    self._traced_step = minimal_fx_tracer(
                        fwd_bwd_fn,
                        module=model,
                        graph_state=(
                            gradient_state.graph_state
                            if gradient_state is not None
                            else None
                        ),
                        graph_state_output_indices=(
                            tuple(range(1, len(params) + 1))
                            if gradient_state is not None
                            else ()
                        ),
                        prepare_inputs=self._prepare_trace_inputs,
                        prepare_call_inputs=self._prepare_trace_call_inputs,
                    )(
                        inputs,
                        labels,
                        global_valid_tokens,
                        extra_kwargs,
                    )

            if self.config.compile.enable_passes:
                pipeline_fn = PASS_PIPELINE_REGISTRY.get(
                    self.config.compile.pass_pipeline
                )
                if pipeline_fn is None:
                    fsdp_bucket_plan = get_transformer_block_buckets(
                        model,
                        parallel_dims=self.parallel_dims,
                        chunked_loss_enabled=isinstance(
                            self.config.loss, ChunkedLossWrapper.Config
                        ),
                    )
                    passes = construct_default_graph_passes(
                        self._traced_step,
                        self.config,
                        fsdp_bucket_plan=fsdp_bucket_plan,
                        parallel_dims=self.parallel_dims,
                    )
                else:
                    passes = pipeline_fn(
                        self._traced_step,
                        self.config,
                        parallel_dims=self.parallel_dims,
                    )

                self._traced_step.gm = apply_graph_passes(
                    self._traced_step.gm,
                    self._traced_step.example_inputs,
                    passes,
                    compile_config=self.config.compile,
                    traced_result=self._traced_step,
                )
            elif gradient_state is not None:
                self._traced_step.gm = finalize_graph_gradient_accumulation(
                    self._traced_step.gm,
                    traced_result=self._traced_step,
                )
        if self._bound_traced_step is None:
            self._bound_traced_step = bind_traced(
                self._traced_step,
                module=model,
                graph_state=(
                    gradient_state.graph_state if gradient_state is not None else None
                ),
            )
        with self.train_context():
            outputs = self._bound_traced_step(
                inputs,
                labels,
                global_valid_tokens,
                extra_kwargs,
            )
        if gradient_state is None:
            loss = outputs[0]
            grads = outputs[1:]
            accumulate_param_grads_(
                params,
                grads,
                outputs_are_replay_owned=isinstance(
                    self._traced_step.gm.forward, CUDAGraphWrapper
                ),
            )
            return loss

        if not self._traced_step.grad_sink_active:
            raise RuntimeError(
                "GraphTrainer traced execution did not install the mandatory "
                "in-graph gradient sink"
            )
        if len(outputs) != 1:
            raise RuntimeError(
                "GraphTrainer in-graph gradient accumulation expected a "
                f"loss-only output, got {len(outputs)} outputs"
            )
        return outputs[0]

    def _ensure_graph_gradient_state(
        self,
        model: nn.Module,
    ) -> GraphGradientState:
        self._validate_graph_gradient_accumulation_config()
        if self._graph_gradient_state is None:
            self._graph_gradient_state = GraphGradientState.create(
                model,
                self.optimizers,
            )
            self._trainable_params = self._graph_gradient_state.parameters
        return self._graph_gradient_state

    def _get_trainable_parameters(
        self,
        model: nn.Module,
    ) -> tuple[torch.Tensor, ...]:
        if self._trainable_params is None:
            # remove_duplicate=False preserves duplicate parameter entries from
            # weight tying (for example, shared embedding/output weights).
            self._trainable_params = tuple(
                parameter
                for _, parameter in model.named_parameters(remove_duplicate=False)
                if parameter.requires_grad
            )
        return self._trainable_params

    def _validate_fsdp_symmetric_memory_config(self, *, parallel_dims=None) -> None:
        if parallel_dims is None:
            parallel_dims = self.parallel_dims
        parallelism = getattr(self.config, "parallelism", None)
        enable_fsdp_symm_mem = parallelism is not None and getattr(
            parallelism,
            "enable_fsdp_symm_mem",
            False,
        )
        if enable_fsdp_symm_mem:
            policy = parallelism.fsdp_symm_mem_policy
            if policy not in {"all", "widest"}:
                raise ValueError(
                    "parallelism.fsdp_symm_mem_policy must be 'all' or "
                    f"'widest', got {policy!r}"
                )
            if self.config.compile.mode != "aot_fx_trace":
                raise ValueError(
                    "GraphTrainer FSDP symmetric memory requires "
                    "compile.mode='aot_fx_trace'"
                )
            if not self.config.compile.enable_passes:
                raise ValueError(
                    "GraphTrainer FSDP symmetric memory requires "
                    "compile.enable_passes"
                )
            if parallel_dims.pp_enabled:
                raise ValueError(
                    "GraphTrainer FSDP symmetric memory currently supports " "SPMD only"
                )
            if self.config.compile.precompile_artifact_dir:
                raise ValueError(
                    "GraphTrainer FSDP symmetric memory does not yet support "
                    "precompiled artifacts"
                )
            if self.config.compile.pass_pipeline in PASS_PIPELINE_REGISTRY:
                raise ValueError(
                    "GraphTrainer FSDP symmetric memory does not yet support "
                    "custom pass pipelines"
                )

    def _validate_graph_gradient_accumulation_config(self) -> None:
        self._validate_fsdp_symmetric_memory_config()
        if (
            self.config.compile.enable_deferred_fsdp_gradient_sync
            and not self.config.compile.enable_graph_gradient_accumulation
        ):
            raise ValueError(
                "Deferred FSDP gradient sync requires "
                "compile.enable_graph_gradient_accumulation"
            )
        if not self.config.compile.enable_graph_gradient_accumulation:
            return
        if self.config.compile.mode != "aot_fx_trace":
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation requires "
                "compile.mode='aot_fx_trace'"
            )
        if self.parallel_dims.pp_enabled:
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation does not yet "
                "support pipeline parallelism"
            )
        if self.config.compile.inductor_compilation == "full":
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation does not yet "
                "support full Inductor compilation"
            )
        if self.config.compile.precompile_artifact_dir:
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation does not yet "
                "support precompiled artifacts"
            )
        if self.config.compile.pass_pipeline in PASS_PIPELINE_REGISTRY:
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation does not yet "
                "support custom pass pipelines"
            )

        if not self.config.compile.enable_deferred_fsdp_gradient_sync:
            return
        if self.gradient_accumulation_steps < 2:
            raise ValueError(
                "Deferred FSDP gradient sync requires at least two gradient "
                "accumulation microbatches"
            )
        if self.config.parallelism.fsdp_reshard_after_forward != "never":
            raise ValueError(
                "Deferred FSDP gradient sync requires "
                "parallelism.fsdp_reshard_after_forward='never'"
            )
        if self.config.compile.memory_policy == "sac_and_offload":
            raise ValueError(
                "Deferred FSDP gradient sync does not yet support activation " "offload"
            )
        if self.config.compile.require_cudagraph and (
            self.config.training.disable_cuda_graphs
            or "cudagraph_pass" in self.config.compile.disable_passes
        ):
            raise ValueError(
                "Deferred FSDP gradient sync cannot require CUDA graphs while "
                "CUDA graphs are disabled"
            )

    def _trace_deferred_fsdp_graph(
        self,
        model: nn.Module,
        example_microbatch: tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
        global_valid_tokens: torch.Tensor,
    ) -> None:
        gradient_state = self._ensure_graph_gradient_state(model)
        inputs, labels, extra_kwargs = example_microbatch
        fwd_bwd_fn = make_fwd_bwd_step(model, self.loss_fn)
        with self.train_context(), log_timer("minimal_fx_tracer"):
            traced = minimal_fx_tracer(
                fwd_bwd_fn,
                module=model,
                graph_state=gradient_state.graph_state,
                graph_state_output_indices=tuple(
                    range(1, len(gradient_state.buffers) + 1)
                ),
                prepare_inputs=self._prepare_trace_inputs,
                prepare_call_inputs=self._prepare_trace_call_inputs,
            )(
                inputs,
                labels,
                global_valid_tokens,
                extra_kwargs,
            )

        if self.config.compile.enable_passes:
            fsdp_bucket_plan = get_transformer_block_buckets(
                model,
                parallel_dims=self.parallel_dims,
                chunked_loss_enabled=isinstance(
                    self.config.loss, ChunkedLossWrapper.Config
                ),
            )
            passes = compile_time_passes(
                traced,
                self.config,
                fsdp_bucket_plan=fsdp_bucket_plan,
                parallel_dims=self.parallel_dims,
                include_inductor=False,
                deduplicate_fsdp_before_bucketing=True,
                include_gradient_sink=False,
                include_fsdp_symmetric_memory=False,
            )
        else:
            passes = [
                eliminate_dead_code_pass,
                canonicalize_graph_pass,
                deduplicate_fsdp_unshard_chains_pass,
            ]
        traced.gm = apply_graph_passes(
            traced.gm,
            traced.example_inputs,
            passes,
            compile_config=self.config.compile,
            traced_result=traced,
        )
        num_flat_parameters = len(
            flatten_graph_values(
                [
                    parameter
                    for _, parameter in model.named_parameters(remove_duplicate=False)
                ]
            )
        )
        enable_cudagraph = (
            not self.config.training.disable_cuda_graphs
            and "cudagraph_pass" not in self.config.compile.disable_passes
        )
        self._deferred_fsdp_graph = build_deferred_fsdp_graph(
            traced,
            num_flat_parameters=num_flat_parameters,
            num_microbatches=self.gradient_accumulation_steps,
            compile_config=self.config.compile,
            enable_cudagraph=enable_cudagraph,
            fsdp_symm_mem_policy=(
                self.config.parallelism.fsdp_symm_mem_policy
                if self.config.parallelism.enable_fsdp_symm_mem
                else None
            ),
        )
        logger.info(
            "Built deferred FSDP graph for %d microbatches with %d "
            "first-microbatch all-gathers, %d final gradient "
            "reduce-scatters, and %d final gradient all-reduces",
            self._deferred_fsdp_graph.num_microbatches,
            self._deferred_fsdp_graph.num_all_gathers,
            self._deferred_fsdp_graph.num_reduce_scatters,
            self._deferred_fsdp_graph.num_all_reduces,
        )
        self._bound_deferred_fsdp_graph = bind_deferred_fsdp_graph(
            self._deferred_fsdp_graph,
            module=model,
            gradient_state=gradient_state,
        )
        self._traced_step = traced

    def _run_microbatch_groups(
        self,
        microbatch_groups: list[list[tuple[dict[str, torch.Tensor], torch.Tensor]]],
        global_valid_tokens: torch.Tensor,
        *,
        should_log: bool,
    ) -> torch.Tensor | None:
        if not self.config.compile.enable_deferred_fsdp_gradient_sync:
            return super()._run_microbatch_groups(
                microbatch_groups,
                global_valid_tokens,
                should_log=should_log,
            )

        prepared = []
        for microbatches in microbatch_groups:
            if len(microbatches) != 1:
                raise ValueError(
                    "Deferred FSDP gradient sync supports SPMD microbatches only"
                )
            input_dict, labels = microbatches[0]
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(self.device)
            prepared.append(
                self.post_dataloading_process(
                    input_dict,
                    labels.to(self.device),
                )
            )

        assert len(self.model_parts) == 1
        if self._bound_deferred_fsdp_graph is None:
            self._trace_deferred_fsdp_graph(
                self.model_parts[0],
                prepared[0],
                global_valid_tokens,
            )
        assert self._bound_deferred_fsdp_graph is not None
        with self.train_context():
            loss = self._bound_deferred_fsdp_graph(prepared, global_valid_tokens)
        detached_loss = loss.detach()
        local_loss = (
            detached_loss.to_local()
            if isinstance(detached_loss, DTensor)
            else detached_loss
        )
        self._loss_is_finite_for_step = torch.isfinite(local_loss).all().to(torch.int32)
        return detached_loss if should_log else None

    def _zero_grad(self) -> None:
        if self._graph_gradient_state is None:
            super()._zero_grad()
            return
        self._graph_gradient_state.validate_optimizers(self.optimizers)
        self.optimizers.zero_grad(set_to_none=False)
        self._graph_gradient_state.validate_bindings()

    def _prepare_trace_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        for pass_name in trace_input_preparer_keys(self.config.compile):
            prepare = TRACE_INPUT_PREPARERS.get(pass_name)
            if prepare is not None:
                prepare(self.config.compile, args, kwargs)

    def _prepare_trace_call_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        for pass_name in trace_input_preparer_keys(self.config.compile):
            prepare = TRACE_CALL_INPUT_PREPARERS.get(pass_name)
            if prepare is not None:
                prepared = prepare(self.config.compile, args, kwargs)
                if prepared is not None:
                    args, kwargs = prepared
        return args, kwargs

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        PRE_TRAIN_STEP_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(
            self
        )
        self._validate_graph_gradient_accumulation_config()
        if self.config.compile.enable_graph_gradient_accumulation:
            assert len(self.model_parts) == 1
            self._ensure_graph_gradient_state(self.model_parts[0])
        if self._bound_traced_step is not None:
            assert len(self.model_parts) == 1
            gradient_state = self._graph_gradient_state
            self._bound_traced_step.validate_state(
                module=self.model_parts[0],
                graph_state=(
                    gradient_state.graph_state if gradient_state is not None else None
                ),
            )
        if (
            bound_deferred_fsdp_graph := getattr(
                self, "_bound_deferred_fsdp_graph", None
            )
        ) is not None:
            bound_deferred_fsdp_graph.validate_state()
        super().train_step(data_iterator)

    def close(self) -> None:
        if self._pinned_pool_ctx is not None:
            self._pinned_pool_ctx.__exit__(None, None, None)
            self._pinned_pool_ctx = None

        # See Note [explicit cudagraph teardown] in cudagraph.py
        cudagraph_teardown()
        self._release_fsdp_symmetric_memory_buffers()
        super().close()

    def _release_fsdp_symmetric_memory_buffers(self) -> None:
        graph_modules = []
        if self._traced_step is not None:
            graph_modules.append(self._traced_step.gm)
        if self._deferred_fsdp_graph is not None:
            graph_modules.append(self._deferred_fsdp_graph.gm)
        if not any(
            has_preallocated_fsdp_symmetric_memory_buffers(gm) for gm in graph_modules
        ):
            return

        synchronize_ranks = (
            torch.distributed.is_initialized()
            and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        )
        if synchronize_ranks:
            torch.cuda.synchronize()
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

        num_released = sum(
            release_preallocated_fsdp_symmetric_memory_buffers(gm)
            for gm in graph_modules
        )
        self._bound_traced_step = None
        self._bound_deferred_fsdp_graph = None
        self._traced_step = None
        self._deferred_fsdp_graph = None

        if synchronize_ranks:
            torch.cuda.synchronize()
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        logger.info(
            "Released %d preallocated FSDP symmetric-memory slabs", num_released
        )
