# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the standalone DistMoE TorchTitan model backend."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import spmd_types as spmd
import torch
from dist_moe import BlockScaledFormat, DistMoeBlockScaledConfig, plan_dist_moe_memory
from torch.distributed.pipelining.schedules import _Action, _ComputationType

from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.quantization import MXFP8LinearConverter
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
from torchtitan.models.common.attention import VarlenAttention
from torchtitan.models.common.dist_moe import (
    _assign_microbatch_slots,
    _assign_stage_microbatch_slots,
    _DistMoeRuntime,
    _kernel_config,
    _pipeline_microbatch_slots,
    _pipeline_stage_microbatch_slots,
    _resolve_device_memory_budget,
    dist_moe_config,
    DistMoeBackendConfig,
    DistMoeRoutedExperts,
    setup_dist_moe,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts
from torchtitan.models.common.router_gate import RouterGateLinear
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.models.deepseek_v3 import model_registry
from torchtitan.models.deepseek_v3.config_registry import (
    _mxfp8_dense_converters,
    attention_batch_size,
    deepseek_v3_16b_dist_moe_bf16,
    deepseek_v3_16b_dist_moe_mxfp8,
    deepseek_v3_16b_dist_moe_mxfp8_spmd_mlperf,
    deepseek_v3_16b_minimal_async_ep,
    deepseek_v3_671b_dist_moe_bf16,
    deepseek_v3_671b_dist_moe_mxfp8,
    enable_mlperf_packing,
)
from torchtitan.protocols.sharding import ShardingConfig


_DIM = 64
_HIDDEN = 64
_EXPERTS = 4


def _cuda_build_supports_current_device() -> bool:
    """Return whether this Torch build can launch on the current CUDA device."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


def _stock_config() -> RoutedExperts.Config:
    """Return a small stock expert configuration for adapter tests."""
    layout = SpmdLayout({MeshAxisName.TP: spmd.S(1)})
    return RoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=_DIM,
            hidden_dim=_HIDDEN,
            num_experts=_EXPERTS,
            param_init={
                "w1_EFD": lambda tensor: tensor.fill_(1.0),
                "w2_EDF": lambda tensor: tensor.zero_(),
                "w3_EFD": lambda tensor: tensor.fill_(3.0),
            },
            sharding_config=ShardingConfig(
                state_shardings={
                    "w1_EFD": layout,
                    "w2_EDF": SpmdLayout({MeshAxisName.TP: spmd.S(2)}),
                    "w3_EFD": layout,
                }
            ),
        ),
        token_dispatcher=LocalTokenDispatcher.Config(
            num_experts=_EXPERTS,
            top_k=2,
        ),
        sharding_config=ShardingConfig(
            in_src_shardings={"x_BLD": layout},
            in_dst_shardings={"x_BLD": layout},
        ),
    )


def _build_experts() -> DistMoeRoutedExperts:
    """Build a CPU expert module without creating its CUDA runtime."""
    return dist_moe_config(_stock_config()).build()


def _build_mxfp8_experts() -> DistMoeRoutedExperts:
    """Build MXFP8 experts without materializing the CUDA runtime."""
    return dist_moe_config(
        _stock_config(),
        backend=DistMoeBackendConfig(
            blockscaled=DistMoeBlockScaledConfig(
                format=BlockScaledFormat.MXFP8_E4M3,
            )
        ),
    ).build()


def _optimizer(model: torch.nn.Module) -> OptimizersContainer:
    """Build a stateful CPU AdamW container around ``model``."""
    config = OptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.0},
            )
        ],
    )
    return config.build(model_parts=[model])


class DistMoeBackendTest(unittest.TestCase):
    """Validate config derivation and stock checkpoint compatibility."""

    def test_rejects_unimplemented_mxfp8_weight_gather(self) -> None:
        """The backend must not silently substitute BF16 parameter gather."""
        with self.assertRaisesRegex(
            ValueError,
            "MXFP8 parameter all-gather is not implemented",
        ):
            DistMoeBackendConfig(weight_gather="mxfp8")

    def test_fuses_initialization_and_shifts_sharding(self) -> None:
        """The inserted gate axis preserves each stock sharded dimension."""
        replacement = dist_moe_config(_stock_config())
        self.assertIsInstance(replacement, DistMoeRoutedExperts.Config)
        module = replacement.build()
        with torch.no_grad():
            module.init_states()

        self.assertTrue(torch.all(module.w13[:, 0] == 1.0))
        self.assertTrue(torch.all(module.w13[:, 1] == 3.0))
        self.assertFalse(hasattr(module, "inner_experts"))

        assert module._sharding_config is not None
        state = module._sharding_config.state_shardings
        self.assertEqual(set(state), {"w13", "w2_EDF"})
        w13_tp = state["w13"].axis_types[MeshAxisName.TP]
        self.assertIsInstance(w13_tp, spmd.Shard)
        self.assertEqual(w13_tp.dim, 2)

    def test_model_state_dict_uses_stock_keys(self) -> None:
        """Model save and load split and merge the non-interleaved weight."""
        source = _build_experts()
        with torch.no_grad():
            source.w13.copy_(torch.randn_like(source.w13))
            source.w2_EDF.copy_(torch.randn_like(source.w2_EDF))

        state = source.state_dict()
        self.assertEqual(
            set(state),
            {
                "inner_experts.w1_EFD",
                "inner_experts.w2_EDF",
                "inner_experts.w3_EFD",
            },
        )
        torch.testing.assert_close(state["inner_experts.w1_EFD"], source.w13[:, 0])
        torch.testing.assert_close(state["inner_experts.w3_EFD"], source.w13[:, 1])

        target = _build_experts()
        target.load_state_dict(state)
        torch.testing.assert_close(target.w13, source.w13)
        torch.testing.assert_close(target.w2_EDF, source.w2_EDF)

    def test_mxfp8_fsdp_wrapper_loads_stock_checkpoint(self) -> None:
        """Checkpoint loading writes through to the persistent BF16 storage."""
        source = _build_mxfp8_experts()
        target = _build_mxfp8_experts()
        with torch.no_grad():
            source.init_states()
            target.init_states()
            source.w13.normal_()
            source.w2_EDF.normal_()

        target.configure_fsdp()
        target.load_state_dict(source.state_dict())

        torch.testing.assert_close(target.w13._tensor[:, 0], source.w13[:, 0])
        torch.testing.assert_close(target.w13._tensor[:, 1], source.w13[:, 1])
        torch.testing.assert_close(target.w2_EDF._tensor, source.w2_EDF)

    def test_optimizer_state_dict_round_trips_stock_keys(self) -> None:
        """Full Adam states split and merge with duplicated scalar metadata."""
        source = torch.nn.ModuleDict({"experts": _build_experts()})
        source_optimizer = _optimizer(source)
        source_optimizer.state_dict()
        source_state = source_optimizer.optimizers[0].state[source["experts"].w13]
        expected = torch.arange(
            source["experts"].w13.numel(),
            dtype=source["experts"].w13.dtype,
        ).view_as(source["experts"].w13)
        source_state["exp_avg"].copy_(expected)

        saved = source_optimizer.state_dict()
        self.assertNotIn("state.experts.w13.exp_avg", saved)
        torch.testing.assert_close(
            saved["state.experts.inner_experts.w1_EFD.exp_avg"], expected[:, 0]
        )
        torch.testing.assert_close(
            saved["state.experts.inner_experts.w3_EFD.exp_avg"], expected[:, 1]
        )
        self.assertIn("param_groups.experts.inner_experts.w1_EFD.lr", saved)
        self.assertIn("param_groups.experts.inner_experts.w3_EFD.lr", saved)

        target = torch.nn.ModuleDict({"experts": _build_experts()})
        target_optimizer = _optimizer(target)
        target_optimizer.load_state_dict(saved)
        loaded = target_optimizer.optimizers[0].state[target["experts"].w13]
        torch.testing.assert_close(loaded["exp_avg"], expected)

    def test_kernel_config_is_derived_from_module_policy(self) -> None:
        """Runtime-only token and layer counts complete the annex config."""
        module = _build_experts()
        config = _kernel_config(module, num_tokens=128, num_moe_layers=6)

        self.assertEqual(config.max_num_tokens, 128)
        self.assertEqual(config.num_moe_layers, 6)
        self.assertEqual(config.hidden_dim, _DIM)
        self.assertEqual(config.intermediate_dim, _HIDDEN)
        self.assertEqual(config.top_k, 2)
        self.assertEqual(config.num_microbatch_stacks, 1)
        self.assertIsNotNone(config.vmm)
        assert config.vmm is not None
        self.assertEqual(config.vmm.host_scratch_imbalance_factor, 16.0)
        self.assertIsNone(config.blockscaled)

    def test_inplace_wgrad_option_reaches_annex_call(self) -> None:
        """The backend forwards one immutable eager execution policy."""
        module = dist_moe_config(
            _stock_config(),
            backend=DistMoeBackendConfig(inplace_wgrad_accum=True),
        ).build()
        module._runtime = Mock(context=object())
        x = torch.zeros(1, 2, _DIM)
        scores = torch.full((1, 2, 2), 0.5)
        expert_ids = torch.zeros(1, 2, 2, dtype=torch.int64)
        local_tokens = torch.zeros(_EXPERTS, dtype=torch.int64)

        with patch(
            "torchtitan.models.common.dist_moe.run_dist_moe",
            return_value=torch.zeros_like(x),
        ) as run:
            module(x, scores, expert_ids, local_tokens)

        options = run.call_args.kwargs["options"]
        self.assertIsNot(options, module._execution_options)
        self.assertTrue(options.inplace_wgrad_accum)
        assert options.wgrad_parameter_owners is not None
        self.assertIs(options.wgrad_parameter_owners[0], module.w13)
        self.assertIs(options.wgrad_parameter_owners[1], module.w2_EDF)

    def test_inplace_wgrad_rejects_outer_gradient_accumulation(self) -> None:
        """Separate SPMD backwards need an explicit FSDP lifetime contract."""
        module = dist_moe_config(
            _stock_config(),
            backend=DistMoeBackendConfig(inplace_wgrad_accum=True),
        ).build()
        batch_mesh = Mock()
        batch_mesh.size.return_value = 2
        parallel_dims = Mock(pp=1)
        parallel_dims.get_optional_mesh.return_value = batch_mesh
        config = SimpleNamespace(
            checkpoint=SimpleNamespace(create_seed_checkpoint=False),
            training=SimpleNamespace(
                num_tokens_per_microbatch_per_dp_rank=16,
                num_tokens_per_train_step=64,
                mixed_precision_param="bfloat16",
            ),
            parallelism=SimpleNamespace(num_pp_microbatches=1),
        )

        with self.assertRaisesRegex(ValueError, "outer gradient accumulation"):
            setup_dist_moe(
                config=config,
                model_parts=[module],
                parallel_dims=parallel_dims,
                device=torch.device("cpu"),
            )

    def test_blockscaled_policy_uses_activation_arena_runtime(self) -> None:
        """Block-scaled policy preserves the shared activation and VMM plan."""
        policy = DistMoeBlockScaledConfig(
            format=BlockScaledFormat.MXFP8_E4M3,
            fast_math=True,
        )
        module = dist_moe_config(
            _stock_config(),
            backend=DistMoeBackendConfig(blockscaled=policy),
        ).build()
        config = _kernel_config(module, num_tokens=128, num_moe_layers=6)

        self.assertIs(config.blockscaled, policy)
        self.assertIsNotNone(config.vmm)
        assert config.vmm is not None
        self.assertEqual(config.vmm.host_scratch_imbalance_factor, 16.0)
        self.assertIsNone(config.device_memory_budget_bytes)

    def test_mxfp8_fsdp_configuration_preserves_parameter_identity(self) -> None:
        """Prepared expert weights retain model and optimizer ownership keys."""
        module = dist_moe_config(
            _stock_config(),
            backend=DistMoeBackendConfig(
                blockscaled=DistMoeBlockScaledConfig(
                    format=BlockScaledFormat.MXFP8_E4M3,
                )
            ),
        ).build()
        w13 = module.w13
        w2 = module.w2_EDF

        module.configure_fsdp()

        self.assertIs(module.w13, w13)
        self.assertIs(module.w2_EDF, w2)
        self.assertEqual(set(module.parameters()), {w13, w2})

    def test_blockscaled_policy_rejects_bf16_kernel_schedule(self) -> None:
        """The BF16 grouped-GEMM schedule cannot configure MXFP8 kernels."""
        module = _build_experts()
        policy = DistMoeBlockScaledConfig()
        module._runtime_policy = replace(
            module._runtime_policy,
            blockscaled=policy,
            vmm_host_scratch_imbalance_factor=2.0,
        )
        config = _kernel_config(module, num_tokens=128, num_moe_layers=6)
        assert config.vmm is not None
        self.assertEqual(config.vmm.host_scratch_imbalance_factor, 2.0)

        module._runtime_policy = replace(
            module._runtime_policy,
            kernel_config="1cta_128x128x64",
        )
        with self.assertRaisesRegex(ValueError, "specific to BF16"):
            _kernel_config(module, num_tokens=128, num_moe_layers=6)

    def test_vmm_capacity_type_is_validated(self) -> None:
        """The integration rejects unrecognized host-scratch capacity values."""
        module = _build_experts()
        module._runtime_policy = replace(
            module._runtime_policy,
            vmm_host_scratch_imbalance_factor="invalid",
        )
        with self.assertRaisesRegex(TypeError, "must be 'auto'"):
            _kernel_config(module, num_tokens=128, num_moe_layers=6)

    def test_microbatch_slots_avoid_modulo_collisions(self) -> None:
        """Static interval coloring uses peak liveness without modulo aliases."""
        forward = _ComputationType.FORWARD
        backward = _ComputationType.FULL_BACKWARD
        actions = [
            _Action(0, forward, 0),
            _Action(0, forward, 2),
            _Action(0, backward, 2),
            _Action(0, forward, 1),
            _Action(0, backward, 1),
            _Action(0, backward, 0),
        ]

        plan = _assign_microbatch_slots(actions, num_microbatches=3)

        self.assertEqual(plan.num_slots, 2)
        self.assertEqual(plan.slot_by_microbatch, (0, 1, 1))
        self.assertEqual(0 % plan.num_slots, 2 % plan.num_slots)
        self.assertNotEqual(plan.slot_by_microbatch[0], plan.slot_by_microbatch[2])

    def test_stage_microbatch_slots_release_each_stage_independently(self) -> None:
        """Stage-local backward permits a slot to be reused before later stages."""
        forward = _ComputationType.FORWARD
        backward = _ComputationType.FULL_BACKWARD
        actions = [
            _Action(0, forward, 0),
            _Action(0, backward, 0),
            _Action(2, forward, 0),
            _Action(2, backward, 0),
        ]

        plan = _assign_stage_microbatch_slots(
            actions,
            num_microbatches=1,
            stage_indices=(0, 2),
        )

        self.assertEqual(plan.num_slots, 1)
        self.assertEqual(plan.slots_by_stage, ((0,), (0,)))

    def test_schedule_slots_are_rank_local_and_exact(self) -> None:
        """Schedule IR yields deterministic per-rank activation-slot plans."""
        for pp_rank in range(4):
            gpipe = _pipeline_microbatch_slots(
                schedule="GPipe",
                pp_degree=4,
                pp_rank=pp_rank,
                num_microbatches=4,
                num_stages_per_rank=1,
            )
            self.assertEqual(gpipe.num_slots, 4)

        plans = [
            _pipeline_microbatch_slots(
                schedule="Interleaved1F1B",
                pp_degree=2,
                pp_rank=pp_rank,
                num_microbatches=4,
                num_stages_per_rank=2,
            )
            for pp_rank in range(2)
        ]
        self.assertEqual(plans[0].num_slots, 4)
        self.assertEqual(plans[0].slot_by_microbatch, (0, 1, 2, 3))
        self.assertEqual(plans[1].num_slots, 3)
        self.assertEqual(plans[1].slot_by_microbatch, (0, 1, 2, 0))
        for pp_rank, plan in enumerate(plans):
            self.assertEqual(len(plan.slot_by_microbatch), 4)
            self.assertEqual(max(plan.slot_by_microbatch) + 1, plan.num_slots)
            self.assertEqual(
                plan,
                _pipeline_microbatch_slots(
                    schedule="Interleaved1F1B",
                    pp_degree=2,
                    pp_rank=pp_rank,
                    num_microbatches=4,
                    num_stages_per_rank=2,
                ),
            )

    def test_stage_microbatch_schedule_slots_match_target_topologies(self) -> None:
        """Stage-aware coloring matches the local and scale-test schedules."""
        cases = (
            (2, 2, 8, (5, 3)),
            (2, 4, 16, (9, 7)),
            (4, 2, 32, (11, 9, 7, 5)),
        )
        for pp_degree, stages_per_rank, microbatches, expected in cases:
            actual = []
            for pp_rank in range(pp_degree):
                stage_indices = tuple(
                    pp_rank + stage * pp_degree for stage in range(stages_per_rank)
                )
                plan = _pipeline_stage_microbatch_slots(
                    schedule="Interleaved1F1B",
                    pp_degree=pp_degree,
                    pp_rank=pp_rank,
                    num_microbatches=microbatches,
                    num_stages_per_rank=stages_per_rank,
                    stage_indices=stage_indices,
                )
                self.assertEqual(len(plan.slots_by_stage), stages_per_rank)
                self.assertTrue(
                    all(len(slots) == microbatches for slots in plan.slots_by_stage)
                )
                actual.append(plan.num_slots)
            self.assertEqual(tuple(actual), expected)

    def test_split_backward_schedule_is_rejected(self) -> None:
        """Slot release cannot precede an independently scheduled weight grad."""
        actions = [
            _Action(0, _ComputationType.FORWARD, 0),
            _Action(0, _ComputationType.BACKWARD_INPUT, 0),
            _Action(0, _ComputationType.BACKWARD_WEIGHT, 0),
        ]

        with self.assertRaisesRegex(ValueError, "whole-backward"):
            _assign_microbatch_slots(actions, num_microbatches=1)

    def test_pipeline_stack_policy_validates_schedule_requirement(self) -> None:
        """Auto uses the peak while explicit policies may only overprovision."""
        module = _build_experts()
        automatic = _kernel_config(
            module,
            num_tokens=128,
            num_moe_layers=6,
            num_microbatch_stacks=2,
        )
        self.assertEqual(automatic.num_microbatch_stacks, 2)

        module._runtime_policy = replace(
            module._runtime_policy,
            num_microbatch_stacks=3,
        )
        explicit = _kernel_config(
            module,
            num_tokens=128,
            num_moe_layers=6,
            num_microbatch_stacks=2,
        )
        self.assertEqual(explicit.num_microbatch_stacks, 3)

        module._runtime_policy = replace(
            module._runtime_policy,
            num_microbatch_stacks=1,
        )
        with self.assertRaisesRegex(ValueError, "required=2"):
            _kernel_config(
                module,
                num_tokens=128,
                num_moe_layers=6,
                num_microbatch_stacks=2,
            )

        with self.assertRaisesRegex(ValueError, "positive"):
            DistMoeBackendConfig(num_microbatch_stacks=0)
        with self.assertRaisesRegex(ValueError, "activation_slot_granularity"):
            DistMoeBackendConfig(activation_slot_granularity="layer")

    def test_runtime_translates_raw_microbatch_to_static_slot(self) -> None:
        """The pipeline callback writes a slot, not a modulo-derived raw ID."""
        context = Mock()
        runtime = _DistMoeRuntime(
            config=Mock(),
            group=Mock(),
            prefetch=None,
            slots_by_stage=((0, 1, 0),),
            moe_layers_by_stage=(4,),
            context=context,
        )

        runtime.select_microbatch(2)

        context.select_activation_slot.assert_called_once_with(0, 4)
        with self.assertRaisesRegex(IndexError, "unknown pipeline microbatch"):
            runtime.select_microbatch(3)

    @unittest.skipUnless(
        _cuda_build_supports_current_device(),
        "the Torch CUDA build must support the current device",
    )
    def test_runtime_selector_is_visible_to_cuda_graph_replay(self) -> None:
        """A stream-ordered slot write is visible to a captured graph."""
        selector = torch.zeros((), dtype=torch.int64, device="cuda")
        observed = torch.empty_like(selector)
        context = Mock()
        context.select_activation_slot.side_effect = (
            lambda slot, num_layers: selector.fill_(slot * 8 + num_layers)
        )
        runtime = _DistMoeRuntime(
            config=Mock(),
            group=Mock(),
            prefetch=None,
            slots_by_stage=((0, 1, 0),),
            moe_layers_by_stage=(3,),
            context=context,
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            observed.copy_(selector)

        runtime.select_microbatch(1)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(observed.item(), 11)

        runtime.select_microbatch(2)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(observed.item(), 3)

    def test_maximum_useful_budget_is_deferred_until_ep_is_known(self) -> None:
        """The symbolic full-save policy becomes a planner-derived byte count."""
        module = _build_experts()
        module._runtime_policy = replace(
            module._runtime_policy,
            device_memory_budget_bytes="maximum_useful",
            vmm_host_scratch_imbalance_factor=None,
        )

        config = _kernel_config(module, num_tokens=128, num_moe_layers=6)
        self.assertIsNone(config.device_memory_budget_bytes)
        resolved = _resolve_device_memory_budget(
            config,
            "maximum_useful",
            ep_size=1,
        )
        resolved_plan = plan_dist_moe_memory(resolved, ep_size=1)
        self.assertEqual(
            resolved_plan.device_memory_budget_bytes,
            resolved_plan.maximum_useful_device_budget_bytes,
        )
        self.assertEqual(resolved_plan.host_scratch_bytes, 0)

    def test_model_registry_selects_backend_without_runtime_inner_experts(self) -> None:
        """DeepSeek builds parent-owned DistMoE parameters and lifecycle hooks."""
        spec = model_registry("debugmodel", moe_backend="dist_moe")

        self.assertIsNotNone(spec.post_parallelize_fn)
        self.assertIsNotNone(spec.cleanup_fn)
        with torch.device("meta"):
            model = spec.model.build()
        routed_experts = [
            layer.moe.routed_experts
            for layer in model.layers.values()
            if layer.moe_enabled
        ]
        self.assertTrue(routed_experts)
        self.assertTrue(
            all(isinstance(module, DistMoeRoutedExperts) for module in routed_experts)
        )
        self.assertTrue(
            all(not hasattr(module, "inner_experts") for module in routed_experts)
        )

    def test_backend_rejects_a_second_communication_backend(self) -> None:
        """DistMoE owns dispatch and cannot compose with a token dispatcher."""
        with self.assertRaisesRegex(ValueError, "owns token communication"):
            model_registry(
                "debugmodel",
                moe_backend="dist_moe",
                moe_comm_backend="minimal_async_ep",
            )

    def test_standard_trainer_configs_use_varlen_cuda_graphs_without_compile(
        self,
    ) -> None:
        """Both precision policies share the capture-safe Trainer contract."""
        with (
            patch.object(MXFP8LinearConverter, "__init__", return_value=None),
            patch.object(
                MXFP8LinearConverter,
                "convert",
                side_effect=lambda config: config,
            ),
        ):
            configs = (
                deepseek_v3_16b_dist_moe_bf16(),
                deepseek_v3_16b_dist_moe_mxfp8(),
                deepseek_v3_671b_dist_moe_bf16(),
                deepseek_v3_671b_dist_moe_mxfp8(),
            )

        for config in configs:
            self.assertEqual(config.training.dtype, "float32")
            self.assertEqual(config.training.mixed_precision_param, "bfloat16")
            self.assertEqual(config.training.mixed_precision_reduce, "bfloat16")
            self.assertEqual(config.optimizer.implementation, "fused_opt_states_bf16")
            for param_group in config.optimizer.param_groups:
                self.assertEqual(param_group.optimizer_name, "AdamW")
            self.assertFalse(config.compile.enable)
            self.assertFalse(config.training.disable_cuda_graphs)
            self.assertIn(
                "torchtitan.overrides.fused_mla.fused_mla",
                config.override.imports,
            )
            config._validate_cuda_graphs()
            for layer in config.model_spec.model.layers:
                attention = layer.attention.inner_attention
                self.assertIsInstance(attention, VarlenAttention.Config)
                self.assertEqual(attention.max_num_documents, 512)
                if layer.moe is not None:
                    gate = layer.moe.router.gate
                    self.assertIsInstance(gate, RouterGateLinear.Config)
                    self.assertEqual(gate.forward.input_dtype, torch.bfloat16)
                    self.assertEqual(gate.forward.compute_mode, "bf16")
                    self.assertEqual(gate.forward.output_dtype, torch.float32)
                    self.assertEqual(gate.backward.input_dtype, torch.float32)
                    self.assertEqual(gate.backward.compute_mode, "tf32")
                    self.assertEqual(gate.backward.output_dtype, torch.float32)

    def test_non_dist_moe_router_remains_linear(self) -> None:
        """The explicit router policy is scoped to DistMoE recipes."""
        config = deepseek_v3_16b_minimal_async_ep()
        for layer in config.model_spec.model.layers:
            if layer.moe is not None:
                self.assertIs(type(layer.moe.router.gate), Linear.Config)

    def test_mlperf_packing_derives_varlen_capacity(self) -> None:
        """Unmasked rows are one document each, so capacity is the batch dim."""
        config = deepseek_v3_16b_dist_moe_mxfp8()
        enable_mlperf_packing(config)
        expected = attention_batch_size(config)
        self.assertEqual(expected, 4)
        self.assertFalse(config.dataloader.dataset.mask_document_boundaries)
        for _, attention, _, _ in config.model_spec.model.traverse(
            VarlenAttention.Config
        ):
            self.assertEqual(attention.max_num_documents, expected)

    def test_mlperf_packing_preserves_dataloader_prefetch(self) -> None:
        config = deepseek_v3_16b_dist_moe_mxfp8()
        original_prefetch = config.dataloader.num_prefetch_batches
        enable_mlperf_packing(config)
        self.assertEqual(config.dataloader.num_prefetch_batches, original_prefetch)

    def test_mxfp8_spmd_mlperf_config(self) -> None:
        config = deepseek_v3_16b_dist_moe_mxfp8_spmd_mlperf()

        self.assertEqual(
            config.training.num_tokens_per_microbatch_per_dp_rank,
            4 * 4096,
        )
        self.assertEqual(config.training.num_tokens_per_train_step, 512 * 4096)
        self.assertEqual(config.training.max_context_length, 4096)
        self.assertFalse(config.training.disable_cuda_graphs)
        self.assertIsNone(config.activation_checkpoint)
        self.assertFalse(config.compile.enable)

        parallelism = config.parallelism
        self.assertEqual(parallelism.data_parallel_replicate_degree, 1)
        self.assertEqual(parallelism.data_parallel_shard_degree, -1)
        self.assertEqual(parallelism.tensor_parallel_degree, 1)
        self.assertEqual(parallelism.context_parallel_degree, 1)
        self.assertEqual(parallelism.pipeline_parallel_degree, 1)
        self.assertEqual(parallelism.expert_parallel_degree, 8)
        self.assertTrue(parallelism.enable_sequence_parallel)
        self.assertEqual(parallelism.fsdp_reshard_after_forward, "never")
        parallel_dims = ParallelDims.from_config(parallelism, world_size=16)
        self.assertEqual(parallel_dims.dp_shard, 16)
        self.assertEqual(
            parallel_dims.dp_shard
            * parallel_dims.cp
            * parallel_dims.tp
            // parallel_dims.ep,
            2,
        )
        self.assertEqual(
            config.training.num_tokens_per_train_step
            // (
                config.training.num_tokens_per_microbatch_per_dp_rank
                * parallel_dims.dp_shard
            ),
            8,
        )

        self.assertFalse(config.dataloader.dataset.mask_document_boundaries)
        attentions = [
            attention
            for _, attention, _, _ in config.model_spec.model.traverse(
                VarlenAttention.Config
            )
        ]
        self.assertEqual(len(attentions), 27)
        for attention in attentions:
            self.assertEqual(attention.max_num_documents, 4)
            self.assertTrue(attention.single_document_rows)

        expert_configs = [
            experts
            for _, experts, _, _ in config.model_spec.model.traverse(
                DistMoeRoutedExperts.Config
            )
        ]
        self.assertEqual(len(expert_configs), 26)
        for experts in expert_configs:
            self.assertEqual(
                experts.backend.device_memory_budget_bytes,
                "maximum_useful",
            )
            self.assertIsNone(experts.backend.vmm_host_scratch_imbalance_factor)
            self.assertFalse(experts.backend.inplace_wgrad_accum)
            assert experts.backend.blockscaled is not None
            self.assertEqual(
                experts.backend.blockscaled.format,
                BlockScaledFormat.MXFP8_E4M3,
            )
            self.assertTrue(experts.backend.blockscaled.fast_math)
            self.assertEqual(experts.backend.blockscaled.pipeline, "staged")

        self.assertTrue(config.debug.moe_force_load_balance)
        self.assertEqual(config.training.dtype, "float32")
        self.assertEqual(config.training.mixed_precision_param, "bfloat16")
        self.assertEqual(config.training.mixed_precision_reduce, "bfloat16")
        self.assertEqual(config.optimizer.implementation, "fused_opt_states_bf16")
        self.assertIn(
            "torchtitan.overrides.fused_mla.fused_mla",
            config.override.imports,
        )
        self.assertEqual(
            config.override.imports.count(
                "torchtitan.overrides.fused_swiglu.fused_swiglu"
            ),
            1,
        )
        self.assertNotIn(
            "torchtitan.overrides.force_load_balanced_routing."
            "force_load_balanced_routing",
            config.override.imports,
        )
        config._validate_cuda_graphs()

    def test_masked_recipes_are_untouched_by_mlperf_packing(self) -> None:
        """Opting in must not change the recipes the A/B baselines used."""
        for config in (deepseek_v3_16b_dist_moe_mxfp8(),):
            self.assertTrue(config.dataloader.dataset.mask_document_boundaries)
            for _, attention, _, _ in config.model_spec.model.traverse(
                VarlenAttention.Config
            ):
                self.assertEqual(attention.max_num_documents, 512)

    def test_attention_batch_size_follows_pipeline_parallelism(self) -> None:
        """Without PP the dataloader yields the local batch directly."""
        spmd = deepseek_v3_16b_dist_moe_mxfp8()
        self.assertEqual(spmd.parallelism.pipeline_parallel_degree, 1)
        self.assertEqual(attention_batch_size(spmd), 4)

    def test_mxfp8_dist_moe_quantizes_lm_head(self) -> None:
        """The dense MXFP8 policy includes every requested non-routed GEMM."""
        (converter,) = _mxfp8_dense_converters()

        self.assertEqual(
            converter.fqns,
            ["attention", "shared_experts", "feed_forward", "lm_head"],
        )

    def test_pipeline_recipes_are_outside_spmd_scope(self) -> None:
        self.skipTest("Pipeline-parallel DistMoE is outside the SPMD perf stack")


if __name__ == "__main__":
    unittest.main()
