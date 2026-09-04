# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Router gates with explicit GEMM numerical policies."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import torch
from torch.autograd.function import once_differentiable

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module


_RouterComputeMode = Literal["bf16", "fp32", "tf32", "bf16x9"]
_ROUTER_COMPUTE_MODES = ("bf16", "fp32", "tf32", "bf16x9")
_ROUTER_GEMM_DTYPES = {
    "bf16": (torch.bfloat16, (torch.bfloat16, torch.float32)),
    "fp32": (torch.float32, (torch.float32,)),
    "tf32": (torch.float32, (torch.float32,)),
    "bf16x9": (torch.float32, (torch.float32,)),
}


@dataclass(frozen=True, kw_only=True, slots=True)
class RouterGemmConfig:
    """Configure the numerical contract for one router GEMM phase.

    Args:
        input_dtype: Dtype to which both operands are explicitly converted.
        compute_mode: Explicit cuBLAS computation mode.
        output_dtype: Dtype written by the GEMM.
    """

    input_dtype: torch.dtype
    compute_mode: _RouterComputeMode
    output_dtype: torch.dtype

    def to_dict(self) -> dict[str, str]:
        """Serialize the numerical policy for trainer config snapshots.

        Returns:
            JSON-compatible input, compute, and output dtype names.
        """
        return {
            "input_dtype": str(self.input_dtype).removeprefix("torch."),
            "compute_mode": self.compute_mode,
            "output_dtype": str(self.output_dtype).removeprefix("torch."),
        }


def _validate_router_gemm_config(config: RouterGemmConfig, *, name: str) -> None:
    """Validate one router GEMM phase against the ATen operator contract.

    Args:
        config: Explicit input, compute, and output policy.
        name: Configuration field name used in validation errors.

    Raises:
        ValueError: If the compute mode or dtype combination is unsupported.
    """
    if config.compute_mode not in _ROUTER_COMPUTE_MODES:
        raise ValueError(
            f"Unsupported {name}.compute_mode={config.compute_mode!r}; expected "
            f"one of {sorted(_ROUTER_COMPUTE_MODES)}"
        )
    input_dtype, output_dtypes = _ROUTER_GEMM_DTYPES[config.compute_mode]
    if config.input_dtype != input_dtype:
        raise ValueError(
            f"{name}.compute_mode={config.compute_mode!r} requires "
            f"input_dtype={input_dtype}, got {config.input_dtype}"
        )
    if config.output_dtype not in output_dtypes:
        raise ValueError(
            f"{name}.compute_mode={config.compute_mode!r} does not support "
            f"output_dtype={config.output_dtype}; expected one of "
            f"{sorted(str(dtype) for dtype in output_dtypes)}"
        )


class _RouterGateLinearFunction(torch.autograd.Function):
    """Apply explicit forward and backward matrix multiplication policies."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        input_TD: torch.Tensor,
        weight_ED: torch.Tensor,
        forward_compute: _RouterComputeMode,
        forward_output_dtype: torch.dtype,
        backward_input_dtype: torch.dtype,
        backward_compute: _RouterComputeMode,
        backward_output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute router logits and save the exact forward operands.

        Args:
            ctx: Autograd context.
            input_TD: Flattened token activations.
            weight_ED: Router gate weight.
            forward_compute: Compute mode for the forward GEMM.
            forward_output_dtype: Output dtype for the forward GEMM.
            backward_input_dtype: Operand dtype for both backward GEMMs.
            backward_compute: Compute mode for both backward GEMMs.
            backward_output_dtype: Output dtype for both backward GEMMs.

        Returns:
            Router logits with shape ``(T, E)``.
        """
        ctx.save_for_backward(input_TD, weight_ED)
        ctx.backward_input_dtype = backward_input_dtype
        ctx.backward_compute = backward_compute
        ctx.backward_output_dtype = backward_output_dtype
        return torch._mm_with_compute_mode(  # pyrefly: ignore [missing-attribute]
            input_TD,
            weight_ED.t(),
            compute_mode=forward_compute,
            out_dtype=forward_output_dtype,
        )

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(
        ctx, grad_output_TE: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None, None,]:
        """Compute input and weight gradients with the configured policy.

        Args:
            ctx: Autograd context populated by ``forward``.
            grad_output_TE: Gradient of the router logits.

        Returns:
            Input and weight gradients followed by ``None`` for configuration
            arguments.
        """
        input_TD, weight_ED = ctx.saved_tensors
        input_dtype = ctx.backward_input_dtype
        compute_mode = ctx.backward_compute
        output_dtype = ctx.backward_output_dtype
        grad_output_TE = grad_output_TE.to(input_dtype)

        grad_input_TD = None
        if ctx.needs_input_grad[0]:
            grad_input_TD = (
                torch._mm_with_compute_mode(  # pyrefly: ignore [missing-attribute]
                    grad_output_TE,
                    weight_ED.to(input_dtype),
                    compute_mode=compute_mode,
                    out_dtype=output_dtype,
                ).to(input_TD.dtype)
            )

        grad_weight_ED = None
        if ctx.needs_input_grad[1]:
            grad_weight_ED = (
                torch._mm_with_compute_mode(  # pyrefly: ignore [missing-attribute]
                    grad_output_TE.t(),
                    input_TD.to(input_dtype),
                    compute_mode=compute_mode,
                    out_dtype=output_dtype,
                ).to(weight_ED.dtype)
            )

        return grad_input_TD, grad_weight_ED, None, None, None, None, None


class RouterGateLinear(Linear):
    """Linear router gate with explicit CUDA GEMM policies."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Configure the forward and backward router GEMMs.

        Args:
            forward: Numerical policy for router logits.
            backward: Numerical policy for input and weight gradients.
        """

        forward: RouterGemmConfig
        backward: RouterGemmConfig

    def __init__(self, config: Config):
        super().__init__(config)
        _validate_router_gemm_config(config.forward, name="forward")
        _validate_router_gemm_config(config.backward, name="backward")
        self.forward_config = config.forward
        self.backward_config = config.backward

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute router logits for arbitrary leading dimensions.

        Args:
            input: Input activations whose final dimension is
                ``in_features``.

        Returns:
            Router logits with the configured dtype and original leading
            dimensions.
        """
        input_shape = input.shape
        input_TD = input.reshape(-1, input_shape[-1])
        forward = self.forward_config
        backward = self.backward_config
        with torch.autocast(device_type=input.device.type, enabled=False):
            output_TE = _RouterGateLinearFunction.apply(
                input_TD.to(forward.input_dtype),
                self.weight.to(forward.input_dtype),
                forward.compute_mode,
                forward.output_dtype,
                backward.input_dtype,
                backward.compute_mode,
                backward.output_dtype,
            )
            if self.bias is not None:
                output_TE = output_TE + self.bias.to(forward.output_dtype)
        return output_TE.reshape(*input_shape[:-1], self.out_features)


class RouterGateLinearConverter(ModelConfigConverter):
    """Replace token-choice router gates with `RouterGateLinear`."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        """Configure the router gate policies applied to a model config.

        Args:
            forward: Numerical policy for router logits.
            backward: Numerical policy for input and weight gradients.
        """

        forward: RouterGemmConfig
        backward: RouterGemmConfig

    def __init__(self, config: Config):
        _validate_router_gemm_config(config.forward, name="forward")
        _validate_router_gemm_config(config.backward, name="backward")
        self.config = config

    def convert(self, model_config: Module.Config) -> Module.Config:
        """Convert each token-choice router's existing gate in place.

        Args:
            model_config: Root model configuration to traverse.

        Returns:
            The same root configuration with converted router gates.

        Raises:
            ValueError: If a router already uses a conflicting explicit policy
                or an unsupported gate subclass.
        """
        for fqn, router, _, _ in model_config.traverse(TokenChoiceTopKRouter.Config):
            assert isinstance(router, TokenChoiceTopKRouter.Config)
            gate = router.gate
            if isinstance(gate, RouterGateLinear.Config):
                if (
                    gate.forward != self.config.forward
                    or gate.backward != self.config.backward
                ):
                    raise ValueError(
                        f"Router {fqn!r} already has a conflicting explicit policy"
                    )
                continue
            if type(gate) is not Linear.Config:
                raise ValueError(
                    f"Router {fqn!r} uses unsupported gate config "
                    f"{type(gate).__qualname__}"
                )

            gate_fields = {
                field.name: getattr(gate, field.name) for field in fields(gate)
            }
            router.gate = RouterGateLinear.Config(
                **gate_fields,
                forward=self.config.forward,
                backward=self.config.backward,
            )
        return model_config


__all__ = [
    "RouterGateLinear",
    "RouterGateLinearConverter",
    "RouterGemmConfig",
]
