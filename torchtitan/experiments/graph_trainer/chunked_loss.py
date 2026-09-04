# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    disable_active_parametrization,
)


class ChunkedLossWrapperWithParamGrads(ChunkedLossWrapper):
    """Expose chunked lm_head gradients as explicit autograd outputs."""

    _weight_gradient_reduce_dtype: torch.dtype | None = None

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        pass

    def set_weight_gradient_reduce_dtype(self, dtype: torch.dtype) -> None:
        """Set the dtype used to accumulate and reduce the lm_head gradient."""
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                "Chunked loss weight-gradient reduction supports only "
                f"bfloat16 or float32, got {dtype}"
            )
        self._weight_gradient_reduce_dtype = dtype

    @property
    def weight_gradient_reduce_dtype(self) -> torch.dtype | None:
        """Return the configured lm_head gradient reduction dtype."""
        return self._weight_gradient_reduce_dtype

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute chunked loss with one SimpleFSDP gradient reduction.

        Args:
            pred: Decoder hidden states.
            labels: Target token IDs.
            global_valid_tokens: Optional global valid-token count.
            **loss_inputs: Additional per-token inputs for the inner loss.

        Returns:
            The differentiable loss and accumulated metrics.
        """
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedLossWrapper"
        params = tuple(lm_head.named_parameters())
        assert len(params) == 1 and params[0][0] == "weight", (
            "ChunkedLossWrapperWithParamGrads requires a bias-free lm_head "
            "with one parameter named 'weight'"
        )

        # SimpleFSDP represents the sharded parameter as a DTensor. Its
        # redistribution backward would otherwise reduce-scatter every chunk.
        if not pred.requires_grad or not isinstance(params[0][1], DTensor):
            return super().__call__(
                pred,
                labels,
                global_valid_tokens,
                **loss_inputs,
            )

        reduce_dtype = self._weight_gradient_reduce_dtype
        if reduce_dtype is None:
            raise RuntimeError(
                "GraphTrainer must configure the chunked loss weight-gradient "
                "reduction dtype before using SimpleFSDP"
            )
        local_lm_head = _LocalGradientAccumulationModule(
            lm_head,
            reduce_dtype=reduce_dtype,
        )
        self.lm_head = local_lm_head
        try:
            return super().__call__(
                pred,
                labels,
                global_valid_tokens,
                **loss_inputs,
            )
        finally:
            local_lm_head.close()
            self.lm_head = lm_head

    @staticmethod
    def _gradient_backprop(
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        total_loss: torch.Tensor,
        lm_head: nn.Module,
        fsdp_enabled: bool,
    ) -> torch.Tensor:
        """Attach the precomputed decoder and lm_head gradients to the loss."""
        if isinstance(lm_head, _LocalGradientAccumulationModule):
            weight, weight_grad = lm_head.finish()
        else:
            assert not fsdp_enabled, "GraphTrainer chunked loss requires SimpleFSDP"
            params = tuple(lm_head.named_parameters())
            assert len(params) == 1 and params[0][0] == "weight"
            weight = params[0][1]
            weight_grad = weight.grad
            assert weight_grad is not None
            weight.grad = None
        return _ChunkedLossWrapperWithParamGrads.apply(
            hidden_states,
            accumulated_grad,
            total_loss,
            weight,
            weight_grad,
        )


@dataclass(slots=True)
class _LocalWeight:
    """SimpleFSDP weight views needed for deferred gradient reduction."""

    sharded: DTensor
    unsharded: torch.Tensor
    local: torch.Tensor
    grad: torch.Tensor | None = None


class _LocalGradientAccumulationModule(nn.Module):
    """Run an lm_head with one unshard and one final gradient reduction."""

    def __init__(self, lm_head: nn.Module, *, reduce_dtype: torch.dtype) -> None:
        super().__init__()
        self.wrapped_lm_head = lm_head
        self._reduce_dtype = reduce_dtype
        sharded_weight = next(lm_head.parameters())
        assert isinstance(sharded_weight, DTensor)
        unsharded_weight = lm_head.weight  # pyrefly: ignore[missing-attribute]
        local_weight = unsharded_weight.detach().requires_grad_(
            sharded_weight.requires_grad
        )
        self._weight = _LocalWeight(
            sharded=sharded_weight,
            unsharded=unsharded_weight,
            local=local_weight,
        )
        self._hook: torch.utils.hooks.RemovableHandle | None = (
            local_weight.register_post_accumulate_grad_hook(self._accumulate_grad)
        )

    def _accumulate_grad(self, weight: torch.Tensor) -> None:
        """Move one chunk gradient into the reduction-dtype accumulator."""
        grad = weight.grad
        assert grad is not None
        if self._weight.grad is None:
            self._weight.grad = grad.to(self._reduce_dtype)
        else:
            self._weight.grad.add_(grad)
        weight.grad = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped lm_head without re-running its parametrization."""
        with disable_active_parametrization():
            return torch.func.functional_call(
                self.wrapped_lm_head,
                {"weight": self._weight.local},
                args,
                kwargs,
                strict=False,
            )

    def finish(self) -> tuple[DTensor, DTensor]:
        """Return the sharded weight and its single reduced gradient."""
        self.close()
        accumulated_grad = self._weight.grad
        assert accumulated_grad is not None
        sharded = self._weight.sharded
        unsharded = self._weight.unsharded
        if isinstance(unsharded, DTensor):
            num_dp_axes = sharded.device_mesh.ndim - unsharded.device_mesh.ndim
            non_dp_placements = unsharded.placements
        else:
            num_dp_axes = sharded.device_mesh.ndim
            non_dp_placements = ()
        assert num_dp_axes > 0

        local_grad = (
            accumulated_grad.to_local()
            if isinstance(accumulated_grad, DTensor)
            else accumulated_grad
        )
        partial_grad = DTensor.from_local(
            local_grad,
            device_mesh=sharded.device_mesh,
            placements=(Partial(reduce_op="sum"),) * num_dp_axes + non_dp_placements,
            run_check=False,
            shape=sharded.shape,
            stride=sharded.stride(),
        )
        reduced_grad = partial_grad.redistribute(placements=sharded.placements)
        return sharded, reduced_grad.to(sharded.dtype)

    def close(self) -> None:
        """Remove the temporary leaf-gradient hook if it is still installed."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


class _ChunkedLossWrapperWithParamGrads(torch.autograd.Function):
    """Expose precomputed decoder and lm_head gradients through autograd."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        accumulated_h_grad: torch.Tensor,
        total_loss: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_weight_grad: torch.Tensor,
    ) -> torch.Tensor:
        """Save precomputed gradients and return a detached loss value."""
        ctx.save_for_backward(accumulated_h_grad, lm_head_weight_grad)
        return total_loss.detach().clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore[bad-override]
        """Return scaled decoder and lm_head gradients."""
        accumulated_h_grad, weight_grad = ctx.saved_tensors
        return (
            accumulated_h_grad * grad_output,
            None,
            None,
            weight_grad * grad_output,
            None,
        )
