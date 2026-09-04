# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""First-class TorchTitan backend for the standalone CuTe DistMoE package."""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, ClassVar, Literal, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.distributed as dist

# pyrefly: ignore [missing-import]
from dist_moe import (
    BlockScaledFormat,
    create_context,
    dist_moe as run_dist_moe,
    DistMoeBlockScaledConfig,
    DistMoeConfig as KernelConfig,
    DistMoeContext,
    DistMoeExecutionOptions,
    DistMoePreparedWeight,
    DistMoeVmmConfig,
    DistMoeVmmPrefetch,
    plan_dist_moe_memory,
    prefetch_dist_moe_vmm,
    prepare_blockscaled_weight,
)
from torch.distributed.pipelining._schedule_visualizer import get_schedule_ops
from torch.distributed.pipelining.schedules import _Action, _ComputationType
from torch.distributed.tensor import DTensor

from torchtitan.distributed._prepared_weight import _FSDPPreparedWeight
from torchtitan.distributed.parallel_dims import ParallelDims, SpmdLayout
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger

from ._fused_weights import make_fused_gate_up_init

if TYPE_CHECKING:
    from torchtitan.trainer import Trainer

__all__ = [
    "cleanup_dist_moe",
    "dist_moe_config",
    "DistMoeBackendConfig",
    "DistMoeRoutedExperts",
    "setup_dist_moe",
]


@dataclass(kw_only=True, slots=True)
class DistMoeBackendConfig:
    """Advanced execution policy for the DistMoE backend.

    Args:
        max_routing_imbalance_factor: Receive imbalance retained in HBM.
        device_memory_budget_bytes: Exact activation and scratch HBM budget,
            ``"maximum_useful"`` to retain every planned activation, or
            ``None`` for the minimum all-recompute budget.
        num_microbatch_stacks: Concurrent activation stacks in the planner,
            or ``"auto"`` to derive the exact rank-local requirement from the
            pipeline schedule.
        activation_slot_granularity: Whether pipeline lifetimes span a whole
            microbatch or one virtual-stage/microbatch pair.
        vmm_host_scratch_imbalance_factor: Total imbalance covered with
            host-backed VMM scratch, ``None`` to disable VMM, or ``"auto"``
            to use the default capacity.
        num_sms: Optional SM count for each CuTe launch.
        kernel_config: Optional explicit BF16 grouped-GEMM schedule.
        blockscaled: Optional MXFP8/NVFP4 compute and kernel policy.
        weight_gather: Parameter communication format for prepared weights.
        wgrad_dtype: ``"bfloat16"`` or ``"float32"`` gradient destination.
        inplace_wgrad_accum: Fuse serialized eager WGRAD contributions into
            standard parameter gradients.
    """

    max_routing_imbalance_factor: float = 1.0
    device_memory_budget_bytes: int | Literal["maximum_useful"] | None = None
    num_microbatch_stacks: int | Literal["auto"] = "auto"
    activation_slot_granularity: Literal[
        "microbatch", "stage_microbatch"
    ] = "stage_microbatch"
    vmm_host_scratch_imbalance_factor: float | Literal["auto"] | None = "auto"
    num_sms: int | None = None
    kernel_config: str | None = None
    blockscaled: DistMoeBlockScaledConfig | None = None
    weight_gather: Literal["bf16", "mxfp8"] = "bf16"
    wgrad_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    inplace_wgrad_accum: bool = False

    def __post_init__(self) -> None:
        """Validate the activation-stack policy.

        Raises:
            ValueError: If an explicit stack count is not positive.
        """
        stacks = self.num_microbatch_stacks
        if stacks != "auto" and (
            isinstance(stacks, bool) or not isinstance(stacks, int) or stacks <= 0
        ):
            raise ValueError("num_microbatch_stacks must be 'auto' or positive")
        if self.activation_slot_granularity not in (
            "microbatch",
            "stage_microbatch",
        ):
            raise ValueError(
                "activation_slot_granularity must be 'microbatch' or "
                "'stage_microbatch'"
            )
        if self.weight_gather != "bf16":
            raise ValueError(
                "MXFP8 parameter all-gather is not implemented; "
                "weight_gather must be 'bf16'"
            )
        budget = self.device_memory_budget_bytes
        if (
            budget != "maximum_useful"
            and budget is not None
            and (isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0)
        ):
            raise ValueError(
                "device_memory_budget_bytes must be 'maximum_useful', None, "
                "or a positive integer"
            )


@dataclass(frozen=True, slots=True)
class _MicrobatchSlotPlan:
    """Immutable activation-stack assignment for one pipeline rank.

    Args:
        num_slots: Exact peak number of simultaneously live microbatches.
        slot_by_microbatch: Slot selected for each raw pipeline microbatch ID.
    """

    num_slots: int
    slot_by_microbatch: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _StageMicrobatchSlotPlan:
    """Immutable slot assignment for local stage/microbatch pairs.

    Args:
        num_slots: Exact peak number of simultaneously live pairs.
        slots_by_stage: Slot mapping for every microbatch of each supplied
            local stage, in the same order as the stage-index input.
    """

    num_slots: int
    slots_by_stage: tuple[tuple[int, ...], ...]


def _assign_microbatch_slots(
    actions: Sequence[_Action | None],
    num_microbatches: int,
) -> _MicrobatchSlotPlan:
    """Color rank-local microbatch lifetimes with the minimum number of slots.

    A lifetime starts at the first local forward and ends after the final local
    full backward. These lifetimes are intervals in schedule order, so assigning
    the lowest free slot is optimal and avoids the collisions possible with raw
    ``microbatch_id % num_slots`` mapping.

    Args:
        actions: Compute actions for one physical pipeline rank.
        num_microbatches: Total number of pipeline microbatches in the step.

    Returns:
        Exact rank-local slot count and static microbatch-to-slot mapping.

    Raises:
        ValueError: If the schedule is incomplete or uses unsupported split or
            overlapped backward actions.
    """
    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be positive")
    expected = set(range(num_microbatches))
    forwards: set[int] = set()
    remaining_backwards: Counter[int] = Counter()
    unsupported = {
        _ComputationType.BACKWARD_INPUT,
        _ComputationType.BACKWARD_WEIGHT,
        _ComputationType.OVERLAP_F_B,
    }
    for action in actions:
        if action is None:
            continue
        computation = action.computation_type
        if computation in unsupported:
            raise ValueError(
                "DistMoE activation slots require a whole-backward, "
                f"non-overlapped pipeline schedule; found {computation}"
            )
        if computation not in (
            _ComputationType.FORWARD,
            _ComputationType.FULL_BACKWARD,
        ):
            continue
        microbatch_id = action.microbatch_index
        if microbatch_id is None or microbatch_id not in expected:
            raise ValueError(
                "pipeline action has an invalid microbatch index: " f"{microbatch_id!r}"
            )
        if computation == _ComputationType.FORWARD:
            forwards.add(microbatch_id)
        else:
            remaining_backwards[microbatch_id] += 1
    if forwards != expected or set(remaining_backwards) != expected:
        raise ValueError(
            "pipeline schedule must contain forward and full-backward actions "
            "for every microbatch"
        )

    free_slots: list[int] = []
    live_slots: dict[int, int] = {}
    slot_by_microbatch = [-1] * num_microbatches
    next_slot = 0
    peak_live = 0
    for action in actions:
        if action is None:
            continue
        computation = action.computation_type
        microbatch_id = action.microbatch_index
        if microbatch_id is None or computation not in (
            _ComputationType.FORWARD,
            _ComputationType.FULL_BACKWARD,
        ):
            continue
        if (
            computation == _ComputationType.FORWARD
            and slot_by_microbatch[microbatch_id] < 0
        ):
            slot = heapq.heappop(free_slots) if free_slots else next_slot
            if slot == next_slot:
                next_slot += 1
            live_slots[microbatch_id] = slot
            slot_by_microbatch[microbatch_id] = slot
            peak_live = max(peak_live, len(live_slots))

        elif (
            computation == _ComputationType.FORWARD and microbatch_id not in live_slots
        ):
            raise ValueError(
                f"pipeline microbatch {microbatch_id} resumed after release"
            )
        elif computation == _ComputationType.FULL_BACKWARD:
            slot = live_slots.get(microbatch_id)
            if slot is None:
                raise ValueError(
                    "pipeline microbatch backward released an inactive slot: "
                    f"{microbatch_id}"
                )
            remaining_backwards[microbatch_id] -= 1
            if remaining_backwards[microbatch_id] == 0:
                del live_slots[microbatch_id]
                heapq.heappush(free_slots, slot)

    if live_slots or any(slot < 0 for slot in slot_by_microbatch):
        raise ValueError("pipeline schedule left incomplete microbatch lifetimes")
    if next_slot != peak_live:
        raise AssertionError("interval coloring did not match peak liveness")
    return _MicrobatchSlotPlan(next_slot, tuple(slot_by_microbatch))


def _pipeline_microbatch_slots(
    *,
    schedule: str,
    pp_degree: int,
    pp_rank: int,
    num_microbatches: int,
    num_stages_per_rank: int,
) -> _MicrobatchSlotPlan:
    """Derive the activation-slot plan for one configured pipeline rank.

    Args:
        schedule: Registered PyTorch pipeline schedule name.
        pp_degree: Number of physical pipeline ranks.
        pp_rank: Current physical pipeline rank.
        num_microbatches: Number of microbatches in one local training batch.
        num_stages_per_rank: Number of virtual stages resident on each rank.

    Returns:
        Exact static activation-slot plan for ``pp_rank``.

    Raises:
        ValueError: If ``pp_rank`` is outside the generated schedule.
    """
    schedule_ops = get_schedule_ops(
        schedule=schedule,
        pp_degree=pp_degree,
        num_microbatches=num_microbatches,
        num_stages_per_rank=num_stages_per_rank,
        with_comms=False,
    )
    if not 0 <= pp_rank < len(schedule_ops):
        raise ValueError(f"invalid pipeline rank {pp_rank} for degree {pp_degree}")
    return _assign_microbatch_slots(schedule_ops[pp_rank], num_microbatches)


def _assign_stage_microbatch_slots(
    actions: Sequence[_Action | None],
    num_microbatches: int,
    stage_indices: Sequence[int],
) -> _StageMicrobatchSlotPlan:
    """Color each local stage/microbatch lifetime independently.

    Args:
        actions: Compute actions for one physical pipeline rank.
        num_microbatches: Total pipeline microbatches in the step.
        stage_indices: Global indices of local stages that contain DistMoE.

    Returns:
        Exact slot count and one microbatch mapping per supplied stage.
    """
    stage_positions = {stage: index for index, stage in enumerate(stage_indices)}
    if len(stage_positions) != len(stage_indices):
        raise ValueError("stage_indices must be unique")
    remapped: list[_Action | None] = []
    for action in actions:
        if action is None or action.stage_index not in stage_positions:
            remapped.append(None)
            continue
        microbatch_id = action.microbatch_index
        if microbatch_id is None:
            raise ValueError("pipeline compute action is missing a microbatch index")
        lifetime = (
            stage_positions[action.stage_index] * num_microbatches + microbatch_id
        )
        remapped.append(_Action(0, action.computation_type, lifetime))
    flat = _assign_microbatch_slots(
        remapped,
        num_microbatches=len(stage_indices) * num_microbatches,
    )
    return _StageMicrobatchSlotPlan(
        num_slots=flat.num_slots,
        slots_by_stage=tuple(
            flat.slot_by_microbatch[
                stage * num_microbatches : (stage + 1) * num_microbatches
            ]
            for stage in range(len(stage_indices))
        ),
    )


def _pipeline_stage_microbatch_slots(
    *,
    schedule: str,
    pp_degree: int,
    pp_rank: int,
    num_microbatches: int,
    num_stages_per_rank: int,
    stage_indices: Sequence[int],
) -> _StageMicrobatchSlotPlan:
    """Derive activation slots for local stage/microbatch lifetimes.

    Args:
        schedule: Registered PyTorch pipeline schedule name.
        pp_degree: Number of physical pipeline ranks.
        pp_rank: Current physical pipeline rank.
        num_microbatches: Number of pipeline microbatches in one step.
        num_stages_per_rank: Number of virtual stages resident on each rank.
        stage_indices: Global indices of local stages containing DistMoE.

    Returns:
        Exact static stage/microbatch slot plan for ``pp_rank``.
    """
    schedule_ops = get_schedule_ops(
        schedule=schedule,
        pp_degree=pp_degree,
        num_microbatches=num_microbatches,
        num_stages_per_rank=num_stages_per_rank,
        with_comms=False,
    )
    if not 0 <= pp_rank < len(schedule_ops):
        raise ValueError(f"invalid pipeline rank {pp_rank} for degree {pp_degree}")
    return _assign_stage_microbatch_slots(
        schedule_ops[pp_rank],
        num_microbatches,
        stage_indices,
    )


@dataclass(eq=False)
class _DistMoeRuntime:
    """Shared annex context and optional VMM prefetch for local MoE layers.

    Args:
        config: Fully derived annex execution configuration.
        group: Expert-parallel process group.
        prefetch: Optional asynchronously prepared VMM mapping.
        slots_by_stage: Slot selected for each local stage and microbatch.
        moe_layers_by_stage: MoE-layer depth encoded for each local stage.
        context: Materialized annex context, initially absent.
    """

    config: KernelConfig
    group: dist.ProcessGroup
    prefetch: DistMoeVmmPrefetch | None
    slots_by_stage: tuple[tuple[int, ...], ...] = ((0,),)
    moe_layers_by_stage: tuple[int, ...] = (1,)
    context: DistMoeContext | None = None

    def initialize(self, device: torch.device) -> DistMoeContext:
        """Create the shared context once and consume any VMM prefetch.

        Args:
            device: CUDA device on which the local model part executes.

        Returns:
            The initialized shared DistMoE context.
        """
        if self.context is None:
            self.context = create_context(
                group=self.group,
                config=self.config,
                device=device,
                prefetched_vmm=self.prefetch,
            )
            self.prefetch = None
        return self.context

    def close(self) -> None:
        """Release the shared context after captured execution has stopped."""
        context = self.context
        if context is not None:
            context.close()
            self.context = None

    def select_microbatch(self, microbatch_id: int, *, stage_index: int = 0) -> None:
        """Select the precomputed activation slot for one stage action.

        The lookup is immutable and CPU-only. The resulting slot is written to
        the context's stable device scalar, which remains safe to read from a
        CUDA graph when the write is stream ordered before replay.

        Args:
            microbatch_id: Raw pipeline microbatch index.
            stage_index: Position of the local virtual stage.

        Raises:
            IndexError: If the schedule supplies an unknown microbatch index.
            RuntimeError: If the runtime has not been initialized.
        """
        if not 0 <= stage_index < len(self.slots_by_stage):
            raise IndexError(f"unknown local pipeline stage {stage_index}")
        slots = self.slots_by_stage[stage_index]
        if not 0 <= microbatch_id < len(slots):
            raise IndexError(f"unknown pipeline microbatch ID {microbatch_id}")
        context = self.context
        if context is None:
            raise RuntimeError("DistMoE runtime is not initialized")
        context.select_activation_slot(
            slots[microbatch_id],
            self.moe_layers_by_stage[stage_index],
        )


_ACTIVE_RUNTIMES: list[_DistMoeRuntime] = []


class _DistMoeFSDPWeight(_FSDPPreparedWeight):
    """BF16 expert parameter carrying unshard-lifetime prepared operands."""

    def __init__(
        self,
        tensor: torch.Tensor,
        config: DistMoeBlockScaledConfig,
        *,
        gate_up: bool,
        prepared: DistMoePreparedWeight | None = None,
        **logical_metadata: Any,
    ) -> None:
        super().__init__(tensor, prepared, **logical_metadata)
        self._config = config
        self._gate_up = gate_up
        if prepared is not None:
            self._fprop_data = prepared.fprop_data
            self._fprop_scale = prepared.fprop_scale
            self._dgrad_data = prepared.dgrad_data
            self._dgrad_scale = prepared.dgrad_scale
            self._global_scale = prepared.global_scale
            self._quantization_workspace = prepared._quantization_workspace

    def _same_metadata(self, other: _FSDPPreparedWeight) -> bool:
        """Return whether two wrappers have identical DistMoE metadata.

        Args:
            other: Prepared-weight wrapper participating in the same operation.

        Returns:
            Whether both wrappers share configuration and prepared state.
        """
        return (
            isinstance(other, _DistMoeFSDPWeight)
            and super()._same_metadata(other)
            and self._config == other._config
            and self._gate_up == other._gate_up
        )

    def _new(
        self,
        tensor: torch.Tensor,
        prepared: DistMoePreparedWeight | None,
        **logical_metadata: Any,
    ) -> _DistMoeFSDPWeight:
        """Construct a DistMoE wrapper for one FSDP lifecycle state.

        Args:
            tensor: BF16 storage or prepared-state metadata anchor.
            prepared: Optional FPROP and DGRAD operands.
            **logical_metadata: Logical tensor metadata overrides.

        Returns:
            DistMoE FSDP weight wrapper.
        """
        return _DistMoeFSDPWeight(
            tensor,
            self._config,
            gate_up=self._gate_up,
            prepared=prepared,
            **logical_metadata,
        )

    def __tensor_flatten__(self):
        """Expose the storage owned by the current FSDP lifecycle.

        Returns:
            Inner tensor names and DistMoE preparation metadata.
        """
        if self._tensor is not None:
            return ["_tensor"], ("sharded", self._config, self._gate_up, self.dtype)
        names = ["_fprop_data", "_fprop_scale"]
        optional = (
            ("_dgrad_data", self._dgrad_data),
            ("_dgrad_scale", self._dgrad_scale),
            ("_global_scale", self._global_scale),
            ("_quantization_workspace", self._quantization_workspace),
        )
        names.extend(name for name, value in optional if value is not None)
        return names, (
            "prepared",
            self._config,
            self._gate_up,
            self.dtype,
            tuple(value is not None for _, value in optional),
        )

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        """Restore a sharded wrapper without prepared operands.

        Args:
            inner_tensors: Serialized high-precision storage.
            metadata: DistMoE configuration and gate-axis flag.
            outer_size: Serialized outer shape.
            outer_stride: Serialized outer stride.

        Returns:
            Sharded DistMoE FSDP weight.
        """
        state, config, gate_up, dtype, *rest = metadata
        if state == "sharded":
            return _DistMoeFSDPWeight(
                inner_tensors["_tensor"],
                config,
                gate_up=gate_up,
            )
        (present,) = rest
        optional_names = (
            "_dgrad_data",
            "_dgrad_scale",
            "_global_scale",
            "_quantization_workspace",
        )
        optional = {
            name: inner_tensors[name] if is_present else None
            for name, is_present in zip(optional_names, present, strict=True)
        }
        anchor = inner_tensors["_fprop_data"]
        prepared = DistMoePreparedWeight(
            source=anchor.new_empty((0,), dtype=dtype),
            format=config.format,
            fprop_data=anchor,
            fprop_scale=inner_tensors["_fprop_scale"],
            dgrad_data=optional["_dgrad_data"],
            dgrad_scale=optional["_dgrad_scale"],
            global_scale=optional["_global_scale"],
            _quantization_workspace=optional["_quantization_workspace"],
        )
        return _DistMoeFSDPWeight(
            anchor,
            config,
            gate_up=gate_up,
            prepared=prepared,
            _logical_size=outer_size,
            _logical_stride=outer_stride,
            _logical_dtype=dtype,
            _logical_device=anchor.device,
        )

    def _compute_view(self, weight: torch.Tensor) -> torch.Tensor:
        """Return the grouped weight shape consumed by DistMoE.

        Args:
            weight: Four- or three-dimensional expert parameter.

        Returns:
            Three-dimensional grouped expert weight.
        """
        return weight.flatten(1, 2) if self._gate_up else weight

    def _prepare(
        self,
        weight: torch.Tensor,
        out: DistMoePreparedWeight | None = None,
    ) -> DistMoePreparedWeight:
        """Allocate or refill DistMoE MXFP8 operands.

        Args:
            weight: Unsharded BF16 expert weight.
            out: Optional caller-owned prepared storage.

        Returns:
            Prepared DistMoE weight state.
        """
        compute_weight = self._compute_view(weight)
        refill = None if out is None else replace(out, source=compute_weight)
        prepared = prepare_blockscaled_weight(
            compute_weight,
            self._config,
            out=refill,
        )
        return replace(
            prepared,
            source=prepared.fprop_data.new_empty((0,), dtype=weight.dtype),
        )

    def _prepared_tensors(
        self,
        prepared: DistMoePreparedWeight,
    ) -> tuple[torch.Tensor, ...]:
        """Return DistMoE tensors owned with the unsharded parameter.

        Args:
            prepared: Prepared DistMoE weight state.

        Returns:
            Independently allocated quantized data, scales, and workspace.
        """
        return prepared.storage_tensors()


class DistMoeRoutedExperts(RoutedExperts):
    """RoutedExperts-compatible adapter around the standalone DistMoE op.

    The physical gate/up weight is ``[E, 2, F, D]``. Flattening its two middle
    dimensions produces the annex's required non-interleaved ``[E, 2F, D]``
    view without an allocation or transpose.

    Args:
        config: Stock expert shape plus DistMoE runtime policy.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        """Stock expert shape plus a DistMoE execution policy.

        Args:
            backend: Expert-facing DistMoE runtime and memory policy.
        """

        supports_cuda_graphs: ClassVar[bool] = True
        backend: DistMoeBackendConfig = field(default_factory=DistMoeBackendConfig)

        def build(self, **kwargs) -> DistMoeRoutedExperts:
            """Build after model policy has populated stock sharding metadata.

            Args:
                **kwargs: Constructor overrides forwarded by the module protocol.

            Returns:
                DistMoE routed experts with fused initialization and sharding.
            """
            config = replace(
                self,
                param_init=_fuse_param_init(self.inner_experts.param_init),
                sharding_config=_fuse_sharding(
                    self.sharding_config,
                    self.inner_experts.sharding_config,
                ),
            )
            return Module.Config.build(config, **kwargs)

    def __init__(self, config: Config):
        Module.__init__(self)
        experts = config.inner_experts
        self.num_experts = experts.num_experts
        self.hidden_dim = experts.dim
        self.intermediate_dim = experts.hidden_dim
        self.top_k = config.token_dispatcher.top_k
        self.w13 = torch.nn.Parameter(
            torch.empty(
                experts.num_experts,
                2,
                experts.hidden_dim,
                experts.dim,
            )
        )
        self.w2_EDF = torch.nn.Parameter(
            torch.empty(experts.num_experts, experts.dim, experts.hidden_dim)
        )
        self._runtime_policy = config.backend
        self._execution_options = DistMoeExecutionOptions(
            inplace_wgrad_accum=config.backend.inplace_wgrad_accum
        )
        self._runtime: _DistMoeRuntime | None = None
        self._ep_group: dist.ProcessGroup | None = None
        self._sp_size = 1

        self.register_state_dict_post_hook(type(self)._split_fused_state_on_save)
        self.register_load_state_dict_pre_hook(type(self)._merge_fused_state_on_load)

    @property
    def sp_size(self) -> int:
        """Return the sequence-parallel degree wired during parallelization."""
        return self._sp_size

    def expert_parameters_module(self) -> torch.nn.Module:
        """Return this fused module as the expert data-parallel target.

        Returns:
            This module, which directly owns the fused expert parameters.
        """
        return self

    @staticmethod
    def _wrap_fsdp_weight(
        weight: torch.Tensor,
        config: DistMoeBlockScaledConfig,
        *,
        gate_up: bool,
    ) -> torch.Tensor:
        """Wrap one local BF16 parameter before FSDP sharding.

        Args:
            weight: Plain tensor or DTensor parameter.
            config: MXFP8 DistMoE execution policy.
            gate_up: Whether the parameter has a separate gate axis.

        Returns:
            Weight whose local tensor implements the FSDP preparation hooks.
        """
        if isinstance(weight, DTensor):
            local = weight.to_local()
            if isinstance(local, _DistMoeFSDPWeight):
                return weight
            wrapped = _DistMoeFSDPWeight(local, config, gate_up=gate_up)
            return DTensor.from_local(
                wrapped,
                weight.device_mesh,
                weight.placements,
                run_check=False,
                shape=weight.shape,
                stride=weight.stride(),
            )
        if isinstance(weight, _DistMoeFSDPWeight):
            return weight
        return _DistMoeFSDPWeight(weight, config, gate_up=gate_up)

    def configure_fsdp(self) -> None:
        """Tie prepared MXFP8 weights to each FSDP unshard lifetime."""
        blockscaled = self._runtime_policy.blockscaled
        if blockscaled is None or blockscaled.format != BlockScaledFormat.MXFP8_E4M3:
            return
        torch.utils.swap_tensors(
            self.w13,
            torch.nn.Parameter(
                self._wrap_fsdp_weight(self.w13, blockscaled, gate_up=True),
                requires_grad=self.w13.requires_grad,
            ),
        )
        torch.utils.swap_tensors(
            self.w2_EDF,
            torch.nn.Parameter(
                self._wrap_fsdp_weight(self.w2_EDF, blockscaled, gate_up=False),
                requires_grad=self.w2_EDF.requires_grad,
            ),
        )

    def synchronize(self) -> None:
        """Complete the routed path before its output is consumed.

        The annex records dispatch, compute, combine, and its stream waits in
        one autograd operation, so it has no deferred TorchTitan-side wait.
        """

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        """Shard weights and record the EP process group used by the annex.

        Args:
            parallel_dims: TorchTitan mesh topology for this model part.
        """
        Module.parallelize(self, parallel_dims)
        ep_mesh = parallel_dims.get_optional_mesh(
            "ep",
            include_singleton_axes=True,
        )
        self._ep_group = ep_mesh.get_group() if ep_mesh is not None else None
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        self._sp_size = tp_mesh.size() if tp_mesh is not None else 1

    def _init_self_buffers(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        """Materialize the shared communication and activation context.

        Args:
            buffer_device: Explicit CUDA buffer device used with CPU-offloaded
                parameter initialization.

        Raises:
            RuntimeError: If the runtime setup hook did not bind a context.
        """
        if self._runtime is None:
            if self.w13.device.type == "cpu" and buffer_device is None:
                # Seed-checkpoint creation initializes model weights on CPU and
                # never executes the CUDA-only routed path.
                return
            raise RuntimeError("DistMoE backend runtime was not initialized")
        device = buffer_device or self.w13.device
        self._runtime.initialize(torch.device(device))

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused distributed dispatch, experts, and combine.

        Args:
            x_TD: Local input activations with shape ``[T, D]``.
            topk_scores_TK: Router weights with shape ``[T, K]``.
            topk_expert_ids_TK: Global expert IDs with shape ``[T, K]``.
            num_local_tokens_per_expert_E: Stock routing metadata, unused
                because the annex constructs distributed metadata internally.

        Returns:
            Local combined expert output with shape ``[T, D]``.

        Raises:
            RuntimeError: If state initialization did not create the context.
        """
        del num_local_tokens_per_expert_E
        if self._runtime is None or self._runtime.context is None:
            raise RuntimeError("DistMoE context is not initialized")

        w13 = self.w13.to_local() if isinstance(self.w13, DTensor) else self.w13
        w2 = self.w2_EDF.to_local() if isinstance(self.w2_EDF, DTensor) else self.w2_EDF
        w13_compute = w13.flatten(1, 2)
        w13_prepared = (
            w13.prepared_state() if isinstance(w13, _DistMoeFSDPWeight) else None
        )
        w2_prepared = (
            w2.prepared_state() if isinstance(w2, _DistMoeFSDPWeight) else None
        )
        w13_arg = (
            w13_compute
            if w13_prepared is None
            else replace(w13_prepared, source=w13_compute)
        )
        w2_arg = w2 if w2_prepared is None else replace(w2_prepared, source=w2)
        options = self._execution_options
        if options.inplace_wgrad_accum:
            options = replace(
                options,
                wgrad_parameter_owners=(self.w13, self.w2_EDF),
            )
        output = run_dist_moe(
            x_TD.contiguous(),
            topk_expert_ids_TK.contiguous(),
            topk_scores_TK.contiguous(),
            w13_arg,
            w2_arg,
            self._runtime.context,
            options=options,
        )
        return output

    @staticmethod
    def _split_fused_state_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Emit fused parameters under stock inner-expert keys.

        Args:
            module: Module owning the state-dict hook.
            state_dict: Mutable model state dictionary.
            prefix: State-dict prefix for this module.
            local_metadata: PyTorch state-dict metadata for this module.
        """
        w13 = state_dict.pop(f"{prefix}w13")
        w2 = state_dict.pop(f"{prefix}w2_EDF")
        stock_prefix = f"{prefix}inner_experts."
        state_dict[f"{stock_prefix}w1_EFD"] = w13[:, 0].contiguous()
        state_dict[f"{stock_prefix}w2_EDF"] = w2
        state_dict[f"{stock_prefix}w3_EFD"] = w13[:, 1].contiguous()

    @staticmethod
    def _merge_fused_state_on_load(module, state_dict, prefix, *args) -> None:
        """Merge stock inner-expert keys into fused parameters.

        Args:
            module: Module owning the load hook.
            state_dict: Mutable model state dictionary.
            prefix: State-dict prefix for this module.
            *args: Remaining PyTorch load-hook arguments.
        """
        stock_prefix = f"{prefix}inner_experts."
        w1_key, w3_key = f"{stock_prefix}w1_EFD", f"{stock_prefix}w3_EFD"
        if w1_key in state_dict and w3_key in state_dict:
            state_dict[f"{prefix}w13"] = torch.stack(
                [state_dict.pop(w1_key), state_dict.pop(w3_key)],
                dim=1,
            )
        w2_key = f"{stock_prefix}w2_EDF"
        if w2_key in state_dict:
            state_dict[f"{prefix}w2_EDF"] = state_dict.pop(w2_key)

    def _optimizer_state_dict_post_hook(
        self,
        state_dict: dict[str, Any],
        prefix: str,
    ) -> None:
        """Split fused optimizer state into stock grouped-expert keys.

        Args:
            state_dict: Mutable flat optimizer state dictionary.
            prefix: Canonical module FQN followed by a dot, or an empty string.
        """
        self._split_flat_optimizer_keys(state_dict, prefix)

    def _optimizer_load_state_dict_pre_hook(
        self,
        state_dict: dict[str, Any],
        prefix: str,
    ) -> None:
        """Merge stock optimizer state into the live fused parameter key.

        Args:
            state_dict: Mutable flat optimizer state dictionary.
            prefix: Canonical module FQN followed by a dot, or an empty string.
        """
        self._merge_flat_optimizer_keys(state_dict, prefix)

    def _split_flat_optimizer_keys(
        self,
        state_dict: dict[str, Any],
        prefix: str,
    ) -> None:
        """Rewrite flat ``w13`` state and parameter-group entries in place.

        Args:
            state_dict: Mutable flat optimizer state dictionary.
            prefix: Canonical module FQN followed by a dot, or an empty string.
        """
        for section in ("state", "param_groups"):
            fused_prefix = f"{section}.{prefix}w13."
            stock_prefix = f"{section}.{prefix}inner_experts."
            for key in [key for key in state_dict if key.startswith(fused_prefix)]:
                suffix = key[len(fused_prefix) :]
                value = state_dict.pop(key)
                if isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(
                    self.w13.shape
                ):
                    w1_value = value[:, 0].contiguous()
                    w3_value = value[:, 1].contiguous()
                else:
                    # Scalar optimizer metadata (for example Adam's step) and
                    # parameter-group options apply identically to both halves.
                    w1_value = w3_value = value
                state_dict[f"{stock_prefix}w1_EFD.{suffix}"] = w1_value
                state_dict[f"{stock_prefix}w3_EFD.{suffix}"] = w3_value

            w2_prefix = f"{section}.{prefix}w2_EDF."
            for key in [key for key in state_dict if key.startswith(w2_prefix)]:
                suffix = key[len(w2_prefix) :]
                state_dict[f"{stock_prefix}w2_EDF.{suffix}"] = state_dict.pop(key)

    def _merge_flat_optimizer_keys(
        self,
        state_dict: dict[str, Any],
        prefix: str,
    ) -> None:
        """Rewrite flat stock state and parameter-group entries in place.

        Args:
            state_dict: Mutable flat optimizer state dictionary.
            prefix: Canonical module FQN followed by a dot, or an empty string.

        Raises:
            KeyError: If a gate optimizer key has no matching up-projection key.
            ValueError: If duplicated scalar or parameter-group metadata differs.
        """
        stock_shape = (self.w13.shape[0], *self.w13.shape[2:])
        for section in ("state", "param_groups"):
            stock_prefix = f"{section}.{prefix}inner_experts."
            w1_prefix = f"{stock_prefix}w1_EFD."
            for w1_key in [key for key in state_dict if key.startswith(w1_prefix)]:
                suffix = w1_key[len(w1_prefix) :]
                w3_key = f"{stock_prefix}w3_EFD.{suffix}"
                if w3_key not in state_dict:
                    raise KeyError(f"Missing matching optimizer key {w3_key!r}")
                w1_value = state_dict.pop(w1_key)
                w3_value = state_dict.pop(w3_key)
                if (
                    isinstance(w1_value, torch.Tensor)
                    and isinstance(w3_value, torch.Tensor)
                    and tuple(w1_value.shape) == stock_shape
                    and tuple(w3_value.shape) == stock_shape
                ):
                    fused_value = torch.stack([w1_value, w3_value], dim=1)
                else:
                    if not _state_values_equal(w1_value, w3_value):
                        raise ValueError(
                            f"Optimizer metadata differs between {w1_key!r} and {w3_key!r}"
                        )
                    fused_value = w1_value
                state_dict[f"{section}.{prefix}w13.{suffix}"] = fused_value

            w2_prefix = f"{stock_prefix}w2_EDF."
            for key in [key for key in state_dict if key.startswith(w2_prefix)]:
                suffix = key[len(w2_prefix) :]
                state_dict[f"{section}.{prefix}w2_EDF.{suffix}"] = state_dict.pop(key)


def _state_values_equal(left: Any, right: Any) -> bool:
    """Return whether duplicated optimizer metadata has matching values.

    Args:
        left: Gate-projection optimizer metadata.
        right: Up-projection optimizer metadata.

    Returns:
        ``True`` when both values can represent one fused optimizer entry.
    """
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    return left == right


def _fuse_param_init(param_init: dict | None) -> dict | None:
    """Map stock gate/up initializers onto the non-interleaved fused weight.

    Args:
        param_init: Stock parameter-name to initializer mapping.

    Returns:
        Mapping with ``w1_EFD`` and ``w3_EFD`` replaced by ``w13``.
    """
    if param_init is None:
        return None
    w1_init = param_init.get("w1_EFD")
    w3_init = param_init.get("w3_EFD")
    fused = {
        key: value
        for key, value in param_init.items()
        if key not in ("w1_EFD", "w3_EFD")
    }
    if w1_init is not None and w3_init is not None:
        fused["w13"] = make_fused_gate_up_init(w1_init, w3_init, gate_up_axis=1)
    return fused or None


def _insert_gate_axis(layout: SpmdLayout) -> SpmdLayout:
    """Shift sharded tensor dimensions after inserting axis 1 into a weight.

    Args:
        layout: Stock ``[E, F, D]`` parameter layout.

    Returns:
        Equivalent layout for a ``[E, 2, F, D]`` parameter.
    """
    axis_types = {
        axis: spmd.S(axis_type.dim + 1)
        if isinstance(axis_type, spmd.Shard) and axis_type.dim >= 1
        else axis_type
        for axis, axis_type in layout.axis_types.items()
    }
    partition_spec = layout.partition_spec
    if partition_spec is not None:
        partition_spec = (*partition_spec[:1], None, *partition_spec[1:])
    return SpmdLayout(axis_types, partition_spec=partition_spec)


def _fuse_sharding(
    routed: ShardingConfig | None,
    inner: ShardingConfig | None,
) -> ShardingConfig | None:
    """Move stock inner state layouts onto the fused routed module.

    Args:
        routed: Routed-expert activation and local-map sharding configuration.
        inner: Stock inner-expert weight sharding configuration.

    Returns:
        Sharding configuration containing ``w13`` and ``w2_EDF``.

    Raises:
        ValueError: If gate and up projections use different layouts.
    """
    if inner is None:
        return routed
    state = dict(inner.state_shardings)
    w1_layout = state.pop("w1_EFD")
    w3_layout = state.pop("w3_EFD")
    if w1_layout != w3_layout:
        raise ValueError("w1_EFD and w3_EFD must use identical sharding")
    state["w13"] = _insert_gate_axis(w1_layout)
    if routed is None:
        return ShardingConfig(state_shardings=state)
    return replace(routed, state_shardings=state)


def _kernel_config(
    module: DistMoeRoutedExperts,
    *,
    num_tokens: int,
    num_moe_layers: int,
    num_microbatch_stacks: int | None = None,
) -> KernelConfig:
    """Derive an annex configuration from one TorchTitan expert module.

    Args:
        module: Local routed-expert module carrying expert policy.
        num_tokens: Local input tokens processed by one invocation.
        num_moe_layers: Local MoE layers sharing the activation arena.
        num_microbatch_stacks: Minimum stack count derived from the pipeline
            schedule, or ``None`` outside pipeline parallelism.

    Returns:
        Valid standalone annex configuration.

    Raises:
        ValueError: If the requested weight-gradient dtype or stack count is
            unsupported.
    """
    policy = module._runtime_policy
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if policy.wgrad_dtype not in dtype_by_name:
        raise ValueError("wgrad_dtype must be 'bfloat16' or 'float32'")
    if policy.blockscaled is not None and policy.kernel_config is not None:
        raise ValueError("kernel_config is specific to BF16 execution")
    vmm_factor = policy.vmm_host_scratch_imbalance_factor
    if vmm_factor == "auto":
        vmm_factor = 16.0
    if vmm_factor is not None and not isinstance(vmm_factor, (int, float)):
        raise TypeError(
            "vmm_host_scratch_imbalance_factor must be 'auto', a number, or None"
        )
    required_stacks = num_microbatch_stacks or 1
    requested_stacks = policy.num_microbatch_stacks
    if requested_stacks == "auto":
        effective_stacks = required_stacks
    else:
        effective_stacks = requested_stacks
        if effective_stacks < required_stacks:
            raise ValueError(
                "num_microbatch_stacks is smaller than the pipeline schedule "
                f"requires: configured={effective_stacks}, required={required_stacks}"
            )
    vmm = (
        None
        if vmm_factor is None
        else DistMoeVmmConfig(host_scratch_imbalance_factor=float(vmm_factor))
    )
    assert isinstance(effective_stacks, int)
    device_memory_budget = policy.device_memory_budget_bytes
    return KernelConfig(
        max_num_tokens=num_tokens,
        hidden_dim=module.hidden_dim,
        intermediate_dim=module.intermediate_dim,
        top_k=module.top_k,
        num_experts=module.num_experts,
        num_moe_layers=num_moe_layers,
        max_routing_imbalance_factor=policy.max_routing_imbalance_factor,
        device_memory_budget_bytes=(
            None if device_memory_budget == "maximum_useful" else device_memory_budget
        ),
        num_microbatch_stacks=effective_stacks,
        vmm=vmm,
        num_sms=policy.num_sms,
        kernel_config=policy.kernel_config,
        blockscaled=policy.blockscaled,
        wgrad_dtype=dtype_by_name[policy.wgrad_dtype],
    )


def _resolve_device_memory_budget(
    config: KernelConfig,
    requested_budget: int | Literal["maximum_useful"] | None,
    *,
    ep_size: int,
) -> KernelConfig:
    """Resolve a symbolic device budget after the EP topology is known.

    Args:
        config: Derived annex configuration using the minimum budget.
        requested_budget: Exact bytes, ``"maximum_useful"``, or ``None``.
        ep_size: Expert-parallel group size used by the memory planner.

    Returns:
        Configuration with the final numeric device budget.
    """
    if requested_budget != "maximum_useful":
        return config
    plan = plan_dist_moe_memory(config, ep_size=ep_size)
    return replace(
        config,
        device_memory_budget_bytes=plan.maximum_useful_device_budget_bytes,
    )


def setup_dist_moe(
    *,
    config: Trainer.Config,
    model_parts: list[torch.nn.Module],
    parallel_dims: ParallelDims,
    device: torch.device,
) -> None:
    """Bind local DistMoE layers to one prefetched annex runtime.

    Args:
        config: Fully updated trainer configuration.
        model_parts: Parallelized model parts resident on this pipeline rank.
        parallel_dims: Runtime mesh topology.
        device: CUDA device for the local rank.

    Raises:
        ValueError: If the current training shape or parallel mode is unsupported.
        RuntimeError: If the modules were not wired to one EP process group.
    """
    modules_by_part: list[list[DistMoeRoutedExperts]] = []
    modules: list[DistMoeRoutedExperts] = []
    seen: set[int] = set()
    for part in model_parts:
        part_modules: list[DistMoeRoutedExperts] = []
        for module in part.modules():
            if isinstance(module, DistMoeRoutedExperts) and id(module) not in seen:
                seen.add(id(module))
                modules.append(module)
                part_modules.append(module)
        modules_by_part.append(part_modules)
    if not modules or config.checkpoint.create_seed_checkpoint:
        return
    accumulation_modes = {
        module._runtime_policy.inplace_wgrad_accum for module in modules
    }
    if len(accumulation_modes) != 1:
        raise ValueError("All local DistMoE layers must share inplace_wgrad_accum")
    inplace_wgrad_accum = accumulation_modes.pop()
    if inplace_wgrad_accum:
        if modules[0]._runtime_policy.wgrad_dtype != "bfloat16":
            raise ValueError("inplace_wgrad_accum requires BF16 parameter gradients")
        batch_mesh = parallel_dims.get_optional_mesh(
            "batch",
            include_singleton_axes=True,
        )
        batch_degree = 1 if batch_mesh is None else batch_mesh.size()
        num_tokens_per_iteration = (
            config.training.num_tokens_per_microbatch_per_dp_rank
            * config.parallelism.num_pp_microbatches
            * batch_degree
        )
        num_tokens_per_train_step = config.training.num_tokens_per_train_step
        if num_tokens_per_train_step < 0:
            num_tokens_per_train_step = num_tokens_per_iteration
        if num_tokens_per_train_step != num_tokens_per_iteration:
            raise ValueError(
                "inplace_wgrad_accum does not support outer gradient accumulation"
            )
    if config.training.mixed_precision_param != "bfloat16":
        raise ValueError("DistMoE requires training.mixed_precision_param='bfloat16'")
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
        raise ValueError("DistMoE requires a Blackwell SM100 CUDA device")
    granularity = modules[0]._runtime_policy.activation_slot_granularity
    if any(
        module._runtime_policy.activation_slot_granularity != granularity
        for module in modules[1:]
    ):
        raise ValueError("All local DistMoE layers must share one slot granularity")
    slots_by_stage: tuple[tuple[int, ...], ...] = tuple((0,) for _ in model_parts)
    moe_layers_by_stage = tuple(len(part_modules) for part_modules in modules_by_part)
    planner_num_moe_layers = len(modules)
    required_slots = 1
    num_pipeline_microbatches = 1
    pp_rank = 0
    if parallel_dims.pp > 1:
        if config.parallelism.pipeline_parallel_schedule != "Interleaved1F1B":
            raise ValueError(
                "DistMoE pipeline parallelism currently requires the "
                "whole-backward Interleaved1F1B schedule"
            )
        if config.parallelism.pipeline_parallel_schedule_csv:
            raise ValueError("DistMoE does not support custom pipeline schedule CSV")
        num_pipeline_microbatches = config.parallelism.num_pp_microbatches
        pp_mesh = parallel_dims.get_optional_mesh(
            "pp",
            include_singleton_axes=True,
        )
        if pp_mesh is None:
            raise RuntimeError("pipeline-parallel mesh is unavailable")
        pp_rank = pp_mesh.get_local_rank()
        from torchtitan.distributed.pipeline_parallel import (
            _get_pp_rank_to_stage_indices_mapping,
        )

        local_stage_indices = _get_pp_rank_to_stage_indices_mapping(
            pp_rank,
            parallel_dims.pp,
            config.parallelism.pipeline_parallel_schedule,
            parallel_dims.pp * len(model_parts),
        )
        if granularity == "stage_microbatch":
            moe_stage_positions = tuple(
                index for index, count in enumerate(moe_layers_by_stage) if count > 0
            )
            stage_plan = _pipeline_stage_microbatch_slots(
                schedule=config.parallelism.pipeline_parallel_schedule,
                pp_degree=parallel_dims.pp,
                pp_rank=pp_rank,
                num_microbatches=num_pipeline_microbatches,
                num_stages_per_rank=len(model_parts),
                stage_indices=tuple(
                    local_stage_indices[index] for index in moe_stage_positions
                ),
            )
            slots = [tuple(0 for _ in range(num_pipeline_microbatches))] * len(
                model_parts
            )
            for position, stage_slots in zip(
                moe_stage_positions,
                stage_plan.slots_by_stage,
                strict=True,
            ):
                slots[position] = stage_slots
            slots_by_stage = tuple(slots)
            planner_num_moe_layers = max(moe_layers_by_stage)
            required_slots = stage_plan.num_slots
        else:
            microbatch_plan = _pipeline_microbatch_slots(
                schedule=config.parallelism.pipeline_parallel_schedule,
                pp_degree=parallel_dims.pp,
                pp_rank=pp_rank,
                num_microbatches=num_pipeline_microbatches,
                num_stages_per_rank=len(model_parts),
            )
            slots_by_stage = tuple(
                microbatch_plan.slot_by_microbatch for _ in model_parts
            )
            moe_layers_by_stage = tuple(
                len(modules) if count else 0 for count in moe_layers_by_stage
            )
            required_slots = microbatch_plan.num_slots

    group = modules[0]._ep_group
    if group is None:
        ep_mesh = parallel_dims.get_optional_mesh(
            "ep",
            include_singleton_axes=True,
        )
        if ep_mesh is None:
            raise RuntimeError("DistMoE requires an expert-parallel mesh")
        group = ep_mesh.get_group()
    if any(module._ep_group not in (None, group) for module in modules[1:]):
        raise RuntimeError("DistMoE layers must share one expert-parallel group")

    total_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    shard_degree = parallel_dims.cp * modules[0].sp_size
    if total_tokens % shard_degree:
        raise ValueError(
            "num_tokens_per_microbatch_per_dp_rank must divide evenly across "
            "CP and SP"
        )
    num_tokens = total_tokens // shard_degree
    kernel_config = _kernel_config(
        modules[0],
        num_tokens=num_tokens,
        num_moe_layers=planner_num_moe_layers,
        num_microbatch_stacks=required_slots,
    )
    requested_device_budget = modules[0]._runtime_policy.device_memory_budget_bytes
    if any(
        module._runtime_policy.device_memory_budget_bytes != requested_device_budget
        for module in modules[1:]
    ):
        raise ValueError("All local DistMoE layers must share one device budget")
    for module in modules[1:]:
        if (
            _kernel_config(
                module,
                num_tokens=num_tokens,
                num_moe_layers=planner_num_moe_layers,
                num_microbatch_stacks=required_slots,
            )
            != kernel_config
        ):
            raise ValueError("All local DistMoE layers must share one configuration")

    ep_size = dist.get_world_size(group)
    kernel_config = _resolve_device_memory_budget(
        kernel_config,
        requested_device_budget,
        ep_size=ep_size,
    )
    plan = plan_dist_moe_memory(kernel_config, ep_size=ep_size)
    prefetch = None
    if plan.uses_host_scratch:
        prefetch = prefetch_dist_moe_vmm(
            config=kernel_config,
            ep_size=ep_size,
            device=device,
        )
    runtime = _DistMoeRuntime(
        config=kernel_config,
        group=group,
        prefetch=prefetch,
        slots_by_stage=slots_by_stage,
        moe_layers_by_stage=moe_layers_by_stage,
    )
    _ACTIVE_RUNTIMES.append(runtime)
    for module in modules:
        module._runtime = runtime
    if parallel_dims.pp > 1:
        logger.info(
            "DistMoE PP activation slots: rank=%d granularity=%s "
            "microbatches=%d stage_depths=%s required=%d configured=%d "
            "mappings=%s",
            pp_rank,
            granularity,
            num_pipeline_microbatches,
            moe_layers_by_stage,
            required_slots,
            kernel_config.num_microbatch_stacks,
            slots_by_stage,
        )
        from torchtitan.distributed.pipeline_parallel import (
            _register_pipeline_microbatch_callback,
        )

        for stage_index, (part, part_modules) in enumerate(
            zip(model_parts, modules_by_part, strict=True)
        ):
            if part_modules:
                _register_pipeline_microbatch_callback(
                    part,
                    partial(runtime.select_microbatch, stage_index=stage_index),
                )


def cleanup_dist_moe() -> None:
    """Release every context created by the active DistMoE model backend."""
    while _ACTIVE_RUNTIMES:
        _ACTIVE_RUNTIMES.pop().close()


def dist_moe_config(
    cfg: RoutedExperts.Config,
    *,
    backend: DistMoeBackendConfig | None = None,
) -> DistMoeRoutedExperts.Config:
    """Build DistMoE routed experts while retaining the stock config template.

    Args:
        cfg: Stock routed-expert configuration used for shape, initialization,
            sharding, and checkpoint compatibility.
        backend: Optional expert execution policy. Defaults to the standard
            BF16 all-recompute VMM configuration.

    Returns:
        DistMoE config that defers fused initialization and sharding until
        build, after ``update_from_config`` has populated the stock template.
    """
    return DistMoeRoutedExperts.Config(
        inner_experts=cfg.inner_experts,
        token_dispatcher=cfg.token_dispatcher,
        backend=backend or DistMoeBackendConfig(),
    )
