# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.models.common.moe import get_expert_parameter_owner, RoutedExperts
from torchtitan.models.common.router_gate import RouterGateLinear, RouterGemmConfig
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.protocols.module import Module


def _build_qwen3_moe_model(num_experts: int = 8) -> Qwen3Model:
    """Build a tiny Qwen3 MoE model with a configurable number of experts."""
    from torchtitan.models.common import CosSinRoPE, Embedding, Linear, RMSNorm

    # Use a tiny variant of the standard MoE debug config, overriding
    # num_experts to exercise the expert-sharding branches.
    from torchtitan.models.qwen3 import _build_qwen3_moe_layers

    dim = 256
    head_dim = 128
    n_layers = 4
    vocab_size = 2048

    config = Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=RMSNorm.Config(normalized_shape=dim),
        tok_embeddings=Embedding.Config(num_embeddings=vocab_size, embedding_dim=dim),
        lm_head=Linear.Config(in_features=dim, out_features=vocab_size),
        layers=_build_qwen3_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=num_experts,
            # top_k must not exceed num_experts (router selects top_k of them).
            top_k=min(8, num_experts),
            # This test only checks expert-param sharding (no forward), so the
            # attention backend is irrelevant; use the default flex backend
            attn_backend="flex",
            moe_comm_backend="standard",
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=4096,
                theta=1000000.0,
            ),
        ),
    )
    return Qwen3Model(config)


def _get_expert_shard_dim(model: Qwen3Model) -> int | None:
    """Return the shard dim used for expert params, or None if not sharded."""
    for layer in model.layers.values():
        if layer.moe_enabled:
            for param in layer.moe.routed_experts.inner_experts.parameters():
                if hasattr(param, "placements"):
                    for p in param.placements:
                        if isinstance(p, Shard):
                            return p.dim
    return None


class _RouterGateFSDPModel(nn.Module):
    """Parent execution unit that owns an explicit router gate."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = RouterGateLinear.Config(
            in_features=32,
            out_features=8,
            bias=True,
            forward=RouterGemmConfig(
                input_dtype=torch.bfloat16,
                compute_mode="bf16",
                output_dtype=torch.float32,
            ),
            backward=RouterGemmConfig(
                input_dtype=torch.float32,
                compute_mode="tf32",
                output_dtype=torch.float32,
            ),
        ).build()

    def forward(self, input_BLD: torch.Tensor) -> torch.Tensor:
        """Return router probabilities from the explicit gate.

        Args:
            input_BLD: BF16 token activations.

        Returns:
            FP32 router probabilities.
        """
        return self.gate(input_BLD).softmax(dim=-1)


class _SelfOwnedRoutedExperts(RoutedExperts):
    """Minimal routed experts whose parameters live on the parent module."""

    def __init__(self) -> None:
        Module.__init__(self)
        self.weight = nn.Parameter(torch.empty(2, 2))

    def expert_parameters_module(self) -> nn.Module:
        """Return the routed module because it directly owns its parameters."""
        return self


class _IncompleteOwnerRoutedExperts(_SelfOwnedRoutedExperts):
    """Routed experts with parameters split across two module owners."""

    def __init__(self) -> None:
        super().__init__()
        self.child = nn.Linear(2, 2, bias=False)

    def expert_parameters_module(self) -> nn.Module:
        """Return an incomplete owner to exercise contract validation."""
        return self.child


class TestExpertParameterOwner(unittest.TestCase):
    def test_accepts_self_owned_expert_parameters(self) -> None:
        routed_experts = _SelfOwnedRoutedExperts()

        self.assertIs(get_expert_parameter_owner(routed_experts), routed_experts)

    def test_rejects_owner_that_omits_routed_parameters(self) -> None:
        routed_experts = _IncompleteOwnerRoutedExperts()

        with self.assertRaisesRegex(ValueError, "own every parameter"):
            get_expert_parameter_owner(routed_experts)


class TestApplyFsdpMoESharding(DTensorTestBase):
    """Test apply_fsdp_to_decoder expert sharding behavior with ep_degree=1 and ep_degree>1."""

    @property
    def world_size(self):
        return 8

    @with_comms
    def test_no_ep_fsdp_gt_num_experts_shards_dim1(self):
        """ep_degree=1, fsdp_size(8) > num_experts(4) → Shard(1)."""
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_qwen3_moe_model(num_experts=4).to(self.device_type)

        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=1,
        )

        self.assertEqual(_get_expert_shard_dim(model), 1)

    @with_comms
    def test_no_ep_fsdp_le_num_experts_shards_dim0(self):
        """ep_degree=1, fsdp_size(8) <= num_experts(8) → Shard(0)."""
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_qwen3_moe_model(num_experts=8).to(self.device_type)

        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=1,
        )

        self.assertEqual(_get_expert_shard_dim(model), 0)

    @with_comms
    def test_with_ep_fsdp_gt_num_experts_shards_dim1(self):
        """ep_degree=2, efsdp*ep(8) > num_experts(4) → Shard(1)."""
        # edp_mesh: 2D mesh [efsdp=4, ep=2], dp_mesh: 1D mesh [8]
        edp_mesh = init_device_mesh(
            self.device_type, (4, 2), mesh_dim_names=("efsdp", "ep")
        )
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_qwen3_moe_model(num_experts=4).to(self.device_type)

        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=2,
            edp_mesh=edp_mesh,
        )

        self.assertEqual(_get_expert_shard_dim(model), 1)


class TestRouterGateLinearFSDP(DTensorTestBase):
    """Validate explicit router GEMMs under the production FSDP dtype policy."""

    @property
    def world_size(self):
        """Use the minimum distributed world size that exercises sharding."""
        return 2

    @with_comms
    def test_reshard_after_forward_policies(self):
        """Both RAF policies match an equivalent unsharded BF16 reference."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        for reshard_after_forward in (True, False):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                torch.manual_seed(1234)
                base = _RouterGateFSDPModel().to(self.device_type)
                reference = deepcopy(base).bfloat16()
                sharded = deepcopy(base)
                fully_shard(
                    sharded,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                    ),
                    reshard_after_forward=reshard_after_forward,
                )

                torch.manual_seed(5678)
                input_BLD = torch.randn(
                    4,
                    5,
                    32,
                    device=self.device_type,
                    dtype=torch.bfloat16,
                )
                reference_input_BLD = input_BLD.clone().requires_grad_(True)
                sharded_input_BLD = input_BLD.clone().requires_grad_(True)
                grad_output_BLE = torch.randn(
                    4,
                    5,
                    8,
                    device=self.device_type,
                    dtype=torch.float32,
                )

                reference_output_BLE = reference(reference_input_BLD)
                sharded_output_BLE = sharded(sharded_input_BLD)
                reference_output_BLE.backward(grad_output_BLE)
                sharded_output_BLE.backward(grad_output_BLE)

                torch.testing.assert_close(
                    sharded_output_BLE,
                    reference_output_BLE,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    sharded_input_BLD.grad,
                    reference_input_BLD.grad,
                    rtol=0,
                    atol=0,
                )
                for name, reference_param in reference.named_parameters():
                    sharded_param = sharded.get_parameter(name)
                    torch.testing.assert_close(
                        sharded_param.grad.full_tensor(),
                        reference_param.grad.float(),
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
