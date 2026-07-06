# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial, Replicate
from torch.distributed.tensor.experimental import local_map

from torchtitan.components.loss import ChunkedLossWrapper, GradAccumulator
from torchtitan.distributed.utils import get_spmd_backend


def _reduce_scatter_grad_like(local_grad: torch.Tensor, param: torch.Tensor):
    """Wrap an unsharded fp32 local grad in ``param``'s sharded DTensor layout."""
    if not isinstance(param, DTensor):
        return local_grad
    mesh = param.device_mesh
    mesh_axis_names = mesh.mesh_dim_names
    src_placements = [
        pl
        if (mesh_axis_names is not None and mesh_axis_names[i] == "tp")
        else Partial()
        for i, pl in enumerate(param.placements)
    ]
    return DTensor.from_local(local_grad, mesh, src_placements).redistribute(
        placements=param.placements
    )


class ChunkedLossWrapperWithParamGrads(ChunkedLossWrapper):
    """ChunkedLossWrapper variant for graph_trainer.

    It exposes lm_head parameter grads as explicit autograd outputs of the
    returned loss, and coalesces simple-FSDP lm_head collectives to one
    all-gather plus one reduce-scatter around the chunk loop.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        pass

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden_states = pred
        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedLossWrapper"

        if isinstance(hidden_states, DTensor):
            mesh = hidden_states.device_mesh
            mesh_axis_names = mesh.mesh_dim_names
            if mesh_axis_names is not None and "tp" in mesh_axis_names:
                tp_axis = mesh_axis_names.index("tp")
                placements = list(hidden_states.placements)
                if not isinstance(placements[tp_axis], Replicate):
                    placements[tp_axis] = Replicate()
                    hidden_states = hidden_states.redistribute(mesh, tuple(placements))

        requires_grad = hidden_states.requires_grad

        def _chunk_local(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
            seq_len = t.shape[1]
            torch._check(
                seq_len % num_chunks == 0,
                lambda: "ChunkedLossWrapper sequence length must be divisible by num_chunks",
            )
            chunk_len = seq_len // num_chunks
            return tuple(
                c.contiguous() for c in torch.split(t, [chunk_len] * num_chunks, dim=1)
            )

        def _chunk(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
            if not isinstance(t, DTensor):
                return _chunk_local(t)
            p = t.placements
            wrapped = local_map(
                _chunk_local,
                out_placements=(p,) * num_chunks,
                in_placements=(p,),
                device_mesh=t.device_mesh,
            )
            return wrapped(t)

        with spmd.local():
            h_chunks = [
                c.detach().requires_grad_(requires_grad) for c in _chunk(hidden_states)
            ]
            label_chunks = list(_chunk(labels))
            input_chunks = {
                key: _chunk(value) if isinstance(value, torch.Tensor) else value
                for key, value in loss_inputs.items()
            }

            raw_weight = lm_head._parameters["weight"]
            raw_bias = lm_head._parameters.get("bias")
            weight = lm_head.weight
            bias = lm_head.bias if raw_bias is not None else None

            total_loss = hidden_states.new_zeros((), dtype=torch.float32)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                for axis_name, dst in {
                    "dp": spmd.P,
                    "cp": spmd.P,
                    "tp": spmd.I,
                }.items():
                    total_loss = spmd.mutate_type(
                        total_loss, axis_name, src=spmd.R, dst=dst
                    )
            metrics: dict[str, torch.Tensor] = {}

            if not requires_grad:
                for i, (h_chunk, label_chunk) in enumerate(
                    zip(h_chunks, label_chunks)
                ):
                    logits = lm_head(h_chunk)
                    chunk_inputs = {
                        key: chunks[i] if isinstance(chunks, tuple) else chunks
                        for key, chunks in input_chunks.items()
                    }
                    chunk_loss, chunk_metrics = self.loss_fn(
                        logits, label_chunk, global_valid_tokens, **chunk_inputs
                    )
                    metrics = self._combine_chunk_metrics(metrics, chunk_metrics)
                    total_loss = total_loss + chunk_loss.detach()
                return total_loss, metrics

            w_leaf = weight.detach().requires_grad_(True)
            b_leaf = bias.detach().requires_grad_(True) if bias is not None else None

            grad_accumulator = GradAccumulator(
                hidden_states,
                num_chunks=num_chunks,
                dtype=torch.float32,
            )

            def _to_local(t: torch.Tensor) -> torch.Tensor:
                return t.to_local() if isinstance(t, DTensor) else t

            w_grad_buf = torch.zeros_like(_to_local(w_leaf), dtype=torch.float32)
            b_grad_buf = (
                torch.zeros_like(_to_local(b_leaf), dtype=torch.float32)
                if b_leaf is not None
                else None
            )

            for i, (h_chunk, label_chunk) in enumerate(zip(h_chunks, label_chunks)):
                logits = F.linear(h_chunk, w_leaf, b_leaf)
                chunk_inputs = {
                    key: chunks[i] if isinstance(chunks, tuple) else chunks
                    for key, chunks in input_chunks.items()
                }
                chunk_loss, chunk_metrics = self.loss_fn(
                    logits, label_chunk, global_valid_tokens, **chunk_inputs
                )
                metrics = self._combine_chunk_metrics(metrics, chunk_metrics)
                if get_spmd_backend() == "spmd_types":
                    spmd.assert_type(chunk_loss, {"dp": spmd.P, "cp": spmd.P})
                total_loss = total_loss + chunk_loss.detach()

                with spmd.no_typecheck():
                    inputs = [h_chunk, w_leaf] + (
                        [b_leaf] if b_leaf is not None else []
                    )
                    grads = torch.autograd.grad(chunk_loss, inputs)
                    grad_accumulator.add(grads[0])
                    w_grad_buf += _to_local(grads[1]).float()
                    if b_leaf is not None:
                        assert b_grad_buf is not None
                        b_grad_buf += _to_local(grads[2]).float()

            accumulated_grad = grad_accumulator.result().to(hidden_states.dtype)

        params: list[torch.Tensor] = [raw_weight]
        param_grads: list[torch.Tensor] = [
            _reduce_scatter_grad_like(w_grad_buf, raw_weight)
        ]
        if bias is not None:
            params.append(raw_bias)
            assert b_grad_buf is not None
            param_grads.append(_reduce_scatter_grad_like(b_grad_buf, raw_bias))

        with spmd.no_typecheck():
            loss = _ChunkedParamGradBridge.apply(
                hidden_states,
                accumulated_grad,
                total_loss,
                len(params),
                *params,
                *param_grads,
            )
        return loss, metrics


class _ChunkedParamGradBridge(torch.autograd.Function):
    """Return a detached loss whose backward emits precomputed grads."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        accumulated_h_grad: torch.Tensor,
        total_loss: torch.Tensor,
        num_params: int,
        *params_and_grads: torch.Tensor,
    ) -> torch.Tensor:
        param_grads = params_and_grads[num_params:]
        ctx.save_for_backward(accumulated_h_grad, *param_grads)
        ctx.num_params = num_params
        return total_loss.detach().clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore[bad-override]
        saved = ctx.saved_tensors
        h_grad = saved[0]
        param_grads = saved[1:]
        return (
            h_grad,
            None,
            None,
            None,
            *param_grads,
            *([None] * ctx.num_params),
        )
