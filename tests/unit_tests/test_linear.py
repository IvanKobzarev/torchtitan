# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.checkpoint import checkpoint

from torchtitan.models.common.linear import Linear, ScaledBiasRowwiseLinear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.router_gate import (
    RouterGateLinear,
    RouterGateLinearConverter,
    RouterGemmConfig,
)
from torchtitan.protocols.module import Module


class TestLinear(unittest.TestCase):
    """Tests for the Linear class used in the codebase."""

    def test_config_build(self):
        """Linear.Config.build() creates a working linear."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        self.assertIsInstance(linear, Linear)
        self.assertIsInstance(linear, nn.Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))
        self.assertIsNone(linear.bias)

    def test_config_build_with_bias(self):
        """Linear.Config(bias=True).build() creates a linear with bias."""
        config = Linear.Config(in_features=32, out_features=16, bias=True)
        linear = config.build()
        self.assertIsNotNone(linear.bias)
        self.assertEqual(linear.bias.shape, torch.Size([16]))

    def test_config_build_without_fields_raises(self):
        """Linear.Config() raises TypeError when required features are not provided."""
        with self.assertRaises(TypeError):
            Linear.Config()

    def test_init_states(self):
        """init_states re-initializes the weight tensor."""
        config = Linear.Config(
            in_features=16,
            out_features=8,
            param_init={
                "weight": partial(nn.init.trunc_normal_, std=0.02),
                "bias": nn.init.zeros_,
            },
        )
        linear = config.build()

        with torch.no_grad():
            nn.init.zeros_(linear.weight)
            self.assertTrue(torch.all(linear.weight == 0))
            linear.init_states()
            self.assertFalse(torch.all(linear.weight == 0))

    def test_custom_init_std(self):
        """Linear respects custom mean and std."""
        config = Linear.Config(
            in_features=1000,
            out_features=500,
            param_init={
                "weight": partial(nn.init.normal_, mean=0.1, std=0.02),
                "bias": nn.init.zeros_,
            },
        )
        linear = config.build()

        torch.manual_seed(42)
        with torch.no_grad():
            linear.init_states()
        # With large amount of samples (1000 * 500) the sample statistics should
        # be close to the requested values. places=3 checks within 0.0005, which
        # is well within statistical tolerance for this sample size.
        self.assertAlmostEqual(linear.weight.mean().item(), 0.1, places=3)
        self.assertAlmostEqual(linear.weight.std().item(), 0.02, places=3)

    def test_forward(self):
        """Forward pass works through nn.Linear's implementation."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        x = torch.randn(2, 10, 32)
        out = linear(x)
        self.assertEqual(out.shape, torch.Size([2, 10, 16]))

    def test_shared_config_builds_independent_instances(self):
        """A single Linear.Config can build multiple independent linears."""
        cfg1 = Linear.Config(in_features=32, out_features=16)
        l1 = cfg1.build()
        cfg2 = Linear.Config(in_features=64, out_features=8)
        l2 = cfg2.build()
        self.assertIsNot(l1, l2)
        self.assertEqual(l1.weight.shape, torch.Size([16, 32]))
        self.assertEqual(l2.weight.shape, torch.Size([8, 64]))

    def test_isinstance_checks(self):
        """Linear is instance of nn.Linear, and Module."""
        config = Linear.Config(in_features=8, out_features=4)
        linear = config.build()
        self.assertIsInstance(linear, nn.Linear)
        self.assertIsInstance(linear, Module)

    def test_default_bias_false(self):
        """Linear.Config defaults to bias=False."""
        config = Linear.Config(in_features=4, out_features=4)
        self.assertFalse(config.bias)

    def test_direct_construction(self):
        """Linear can be constructed directly (Flux-style, non-Configurable parents)."""
        config = Linear.Config(in_features=32, out_features=16, bias=True)
        linear = Linear(config)
        self.assertIsInstance(linear, Linear)
        self.assertIsNotNone(linear.bias)

    def test_config_pre_specified_build(self):
        """Linear.Config with both fields pre-specified builds with no kwargs."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        self.assertIsInstance(linear, Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))

    def test_config_partial_pre_specified(self):
        """Linear.Config with fields specified at construction builds correctly."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        self.assertIsInstance(linear, Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))


class TestRouterGateLinear(unittest.TestCase):
    """Tests for explicit router-gate matrix multiplication policies."""

    @staticmethod
    def _build_megatron_gate() -> RouterGateLinear:
        """Build the explicit gate policy used by DeepSeek V3 DistMoE.

        Returns:
            Router gate with BF16 forward operands, FP32 logits, and TF32
            backward GEMMs.
        """
        return RouterGateLinear.Config(
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

    def test_converter_preserves_gate_config_and_is_idempotent(self):
        """Conversion changes only the gate type and explicit policies."""
        param_init = {"weight": nn.init.zeros_}
        router = TokenChoiceTopKRouter.Config(
            num_experts=8,
            gate=Linear.Config(
                in_features=32,
                out_features=8,
                bias=True,
                param_init=param_init,
            ),
        )
        converter = RouterGateLinearConverter.Config(
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

        self.assertIs(converter.convert(router), router)
        self.assertIsInstance(router.gate, RouterGateLinear.Config)
        gate = router.gate
        self.assertEqual(gate.in_features, 32)
        self.assertEqual(gate.out_features, 8)
        self.assertTrue(gate.bias)
        self.assertIs(gate.param_init, param_init)

        self.assertIs(converter.convert(router), router)
        self.assertIs(router.gate, gate)

        conflicting = RouterGateLinearConverter.Config(
            forward=RouterGemmConfig(
                input_dtype=torch.bfloat16,
                compute_mode="bf16",
                output_dtype=torch.bfloat16,
            ),
            backward=converter.config.backward,
        ).build()
        with self.assertRaisesRegex(ValueError, "conflicting explicit policy"):
            conflicting.convert(router)

    def test_config_is_json_serializable(self):
        """Trainer config snapshots preserve explicit router dtype names."""
        config = RouterGateLinear.Config(
            in_features=32,
            out_features=8,
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
        )

        serialized = config.to_dict()
        json.dumps(serialized)
        self.assertEqual(serialized["forward"]["input_dtype"], "bfloat16")
        self.assertEqual(serialized["backward"]["compute_mode"], "tf32")

    def test_converter_rejects_incompatible_gate(self):
        """Conversion fails instead of discarding another gate's semantics."""
        router = TokenChoiceTopKRouter.Config(
            num_experts=8,
            gate=ScaledBiasRowwiseLinear.Config(
                in_features=32,
                out_features=8,
            ),
        )
        converter = RouterGateLinearConverter.Config(
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

        with self.assertRaisesRegex(ValueError, "unsupported gate config"):
            converter.convert(router)

    def test_rejects_invalid_gemm_config(self):
        """Invalid compute and dtype policies fail while the module is built."""
        valid = RouterGemmConfig(
            input_dtype=torch.float32,
            compute_mode="tf32",
            output_dtype=torch.float32,
        )
        with self.assertRaisesRegex(ValueError, "forward.compute_mode"):
            RouterGateLinear.Config(
                in_features=32,
                out_features=8,
                forward=RouterGemmConfig(
                    input_dtype=torch.float32,
                    compute_mode="invalid",  # type: ignore[arg-type]
                    output_dtype=torch.float32,
                ),
                backward=valid,
            ).build()
        with self.assertRaisesRegex(ValueError, "requires input_dtype"):
            RouterGateLinear.Config(
                in_features=32,
                out_features=8,
                forward=RouterGemmConfig(
                    input_dtype=torch.float32,
                    compute_mode="bf16",
                    output_dtype=torch.float32,
                ),
                backward=valid,
            ).build()
        with self.assertRaisesRegex(ValueError, "does not support output_dtype"):
            RouterGateLinear.Config(
                in_features=32,
                out_features=8,
                forward=RouterGemmConfig(
                    input_dtype=torch.float32,
                    compute_mode="tf32",
                    output_dtype=torch.bfloat16,
                ),
                backward=valid,
            ).build()

    def test_state_dict_matches_linear(self):
        """The explicit gate preserves ordinary Linear checkpoint keys."""
        linear = Linear.Config(in_features=32, out_features=8, bias=True).build()
        gate = RouterGateLinear.Config(
            in_features=32,
            out_features=8,
            bias=True,
            forward=RouterGemmConfig(
                input_dtype=torch.bfloat16,
                compute_mode="bf16",
                output_dtype=torch.bfloat16,
            ),
            backward=RouterGemmConfig(
                input_dtype=torch.float32,
                compute_mode="tf32",
                output_dtype=torch.float32,
            ),
        ).build()

        gate.load_state_dict(linear.state_dict())
        self.assertEqual(set(gate.state_dict()), {"weight", "bias"})
        torch.testing.assert_close(gate.weight, linear.weight)
        assert gate.bias is not None and linear.bias is not None
        torch.testing.assert_close(gate.bias, linear.bias)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_forward_backward_compute_modes(self):
        """Forward and backward use their independently configured modes."""
        policies = [
            (
                RouterGemmConfig(
                    input_dtype=torch.bfloat16,
                    compute_mode="bf16",
                    output_dtype=torch.bfloat16,
                ),
                RouterGemmConfig(
                    input_dtype=torch.bfloat16,
                    compute_mode="bf16",
                    output_dtype=torch.bfloat16,
                ),
            ),
            (
                RouterGemmConfig(
                    input_dtype=torch.bfloat16,
                    compute_mode="bf16",
                    output_dtype=torch.float32,
                ),
                RouterGemmConfig(
                    input_dtype=torch.float32,
                    compute_mode="tf32",
                    output_dtype=torch.float32,
                ),
            ),
            (
                RouterGemmConfig(
                    input_dtype=torch.float32,
                    compute_mode="bf16x9",
                    output_dtype=torch.float32,
                ),
                RouterGemmConfig(
                    input_dtype=torch.float32,
                    compute_mode="bf16x9",
                    output_dtype=torch.float32,
                ),
            ),
        ]
        if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
            policies = policies[:2]
        if torch.cuda.get_device_capability()[0] < 10:
            policies = policies[:2]

        for forward, backward in policies:
            with self.subTest(
                forward=forward,
                backward=backward,
            ):
                gate = RouterGateLinear.Config(
                    in_features=16,
                    out_features=7,
                    bias=True,
                    forward=forward,
                    backward=backward,
                ).build()
                gate.cuda()
                input_BLD = torch.randn(
                    2, 3, 16, device="cuda", dtype=torch.float32, requires_grad=True
                )
                grad_output_BLE = torch.randn(
                    2, 3, 7, device="cuda", dtype=forward.output_dtype
                )

                output_BLE = gate(input_BLD)
                output_BLE.backward(grad_output_BLE)

                input_TD = input_BLD.detach().reshape(-1, 16).to(forward.input_dtype)
                weight_ED = gate.weight.detach().to(forward.input_dtype)
                expected_TE = torch._mm_with_compute_mode(
                    input_TD,
                    weight_ED.t(),
                    compute_mode=forward.compute_mode,
                    out_dtype=forward.output_dtype,
                )
                expected_grad_input_TD = (
                    torch._mm_with_compute_mode(
                        grad_output_BLE.reshape(-1, 7).to(backward.input_dtype),
                        weight_ED.to(backward.input_dtype),
                        compute_mode=backward.compute_mode,
                        out_dtype=backward.output_dtype,
                    )
                    .to(forward.input_dtype)
                    .to(input_BLD.dtype)
                )
                expected_grad_weight_ED = (
                    torch._mm_with_compute_mode(
                        grad_output_BLE.reshape(-1, 7).t().to(backward.input_dtype),
                        input_TD.to(backward.input_dtype),
                        compute_mode=backward.compute_mode,
                        out_dtype=backward.output_dtype,
                    )
                    .to(forward.input_dtype)
                    .to(gate.weight.dtype)
                )

                self.assertEqual(output_BLE.dtype, forward.output_dtype)
                torch.testing.assert_close(
                    output_BLE,
                    (expected_TE + gate.bias.detach().to(forward.output_dtype)).reshape(
                        2, 3, 7
                    ),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    input_BLD.grad,
                    expected_grad_input_TD.reshape(2, 3, 16),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    gate.weight.grad,
                    expected_grad_weight_ED,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    gate.bias.grad,
                    grad_output_BLE.sum(dim=(0, 1)).to(gate.bias.dtype),
                    rtol=0,
                    atol=0,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_token_choice_router_uses_explicit_gate(self):
        """The real token-choice router preserves FP32 scores and gradients."""
        router = TokenChoiceTopKRouter.Config(
            num_experts=8,
            top_k=2,
            gate=RouterGateLinear.Config(
                in_features=16,
                out_features=8,
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
            ),
        ).build()
        router.to(device="cuda", dtype=torch.bfloat16)
        input_BLD = torch.randn(
            2, 3, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        topk_scores_BLK, topk_expert_ids_BLK, scores_BLE = router(input_BLD)
        self.assertEqual(scores_BLE.dtype, torch.float32)
        self.assertEqual(topk_scores_BLK.dtype, torch.float32)
        self.assertEqual(topk_expert_ids_BLK.dtype, torch.int64)
        topk_scores_BLK.sum().backward()
        self.assertEqual(input_BLD.grad.dtype, torch.bfloat16)
        self.assertEqual(router.gate.weight.grad.dtype, torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_activation_checkpoint_matches_eager(self):
        """Full activation recomputation preserves router outputs and gradients."""
        torch.manual_seed(1)
        eager = self._build_megatron_gate().cuda().bfloat16()
        checkpointed = deepcopy(eager)
        torch.manual_seed(2)
        input_BLD = torch.randn(4, 5, 32, device="cuda", dtype=torch.bfloat16)
        grad_output_BLE = torch.randn(4, 5, 8, device="cuda", dtype=torch.float32)
        eager_input_BLD = input_BLD.clone().requires_grad_(True)
        checkpointed_input_BLD = input_BLD.clone().requires_grad_(True)

        eager_output_BLE = eager(eager_input_BLD)
        checkpointed_output_BLE = checkpoint(
            checkpointed,
            checkpointed_input_BLD,
            use_reentrant=False,
        )
        eager_output_BLE.backward(grad_output_BLE)
        checkpointed_output_BLE.backward(grad_output_BLE)

        torch.testing.assert_close(
            checkpointed_output_BLE, eager_output_BLE, rtol=0, atol=0
        )
        torch.testing.assert_close(
            checkpointed_input_BLD.grad, eager_input_BLD.grad, rtol=0, atol=0
        )
        for checkpointed_param, eager_param in zip(
            checkpointed.parameters(), eager.parameters(), strict=True
        ):
            torch.testing.assert_close(
                checkpointed_param.grad, eager_param.grad, rtol=0, atol=0
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_make_fx_and_aot_autograd(self):
        """Proxy tracing and AOTAutograd preserve the explicit GEMM boundary."""
        torch.manual_seed(3)
        eager = self._build_megatron_gate().cuda().bfloat16()
        compiled = torch.compile(
            deepcopy(eager),
            backend="aot_eager",
            fullgraph=True,
        )
        input_BLD = torch.randn(4, 5, 32, device="cuda", dtype=torch.bfloat16)
        grad_output_BLE = torch.randn(4, 5, 8, device="cuda", dtype=torch.float32)

        traced = make_fx(eager)(input_BLD)
        self.assertTrue(
            any(
                node.target is torch.ops.aten._mm_with_compute_mode.default
                for node in traced.graph.nodes
            )
        )
        torch.testing.assert_close(traced(input_BLD), eager(input_BLD), rtol=0, atol=0)

        eager_input_BLD = input_BLD.clone().requires_grad_(True)
        compiled_input_BLD = input_BLD.clone().requires_grad_(True)
        eager_output_BLE = eager(eager_input_BLD)
        compiled_output_BLE = compiled(compiled_input_BLD)
        eager_output_BLE.backward(grad_output_BLE)
        compiled_output_BLE.backward(grad_output_BLE)

        torch.testing.assert_close(
            compiled_output_BLE, eager_output_BLE, rtol=0, atol=0
        )
        torch.testing.assert_close(
            compiled_input_BLD.grad, eager_input_BLD.grad, rtol=0, atol=0
        )
        for compiled_param, eager_param in zip(
            compiled.parameters(), eager.parameters(), strict=True
        ):
            torch.testing.assert_close(
                compiled_param.grad, eager_param.grad, rtol=0, atol=0
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_full_cuda_graph_forward_backward_replay(self):
        """A full forward/backward CUDA graph replays with stable numerics."""
        torch.manual_seed(4)
        graphed = self._build_megatron_gate().cuda().bfloat16()
        eager = deepcopy(graphed)
        static_input_BLD = torch.randn(
            4, 5, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        base_input_BLD = static_input_BLD.detach().clone()
        grad_output_BLE = torch.randn(4, 5, 8, device="cuda", dtype=torch.float32)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                graphed.zero_grad(set_to_none=False)
                if static_input_BLD.grad is not None:
                    static_input_BLD.grad.zero_()
                graphed(static_input_BLD).backward(grad_output_BLE)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graphed.zero_grad(set_to_none=False)
            static_input_BLD.grad.zero_()
            static_output_BLE = graphed(static_input_BLD)
            static_output_BLE.backward(grad_output_BLE)

        for scale in (1.0, 2.0):
            input_BLD = base_input_BLD * scale
            with torch.no_grad():
                static_input_BLD.copy_(input_BLD)
            graph.replay()
            torch.cuda.synchronize()

            eager.zero_grad(set_to_none=True)
            eager_input_BLD = input_BLD.clone().requires_grad_(True)
            eager_output_BLE = eager(eager_input_BLD)
            eager_output_BLE.backward(grad_output_BLE)
            torch.testing.assert_close(
                static_output_BLE, eager_output_BLE, rtol=0, atol=0
            )
            torch.testing.assert_close(
                static_input_BLD.grad, eager_input_BLD.grad, rtol=0, atol=0
            )
            for graphed_param, eager_param in zip(
                graphed.parameters(), eager.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    graphed_param.grad, eager_param.grad, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
