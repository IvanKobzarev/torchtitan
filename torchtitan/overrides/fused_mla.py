# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Opt-in fused DeepSeek-V3 MLA Q/KV assembly.

Activate with::

    --override.imports torchtitan.overrides.fused_mla.fused_mla

Scope and limitations
---------------------
This override is specific to TorchTitan's DeepSeek-V3 ``Attention`` module and
its packed MLA Q/KV projection layout. It is not a generic RoPE fusion and does
not apply to non-MLA models such as Qwen3, Qwen3.5, or GPT-OSS.

The kernels implement TorchTitan ``ComplexRoPE`` and require its complex-valued
cache. They do not support ``CosSinRoPE`` or ``MRoPE``, whose real cos/sin cache
layouts and rotation conventions require a separate kernel path. A future MLA
model using either of those RoPE implementations cannot use this override
without such an adaptation.

Design provenance
-----------------
The core fusion strategy is borrowed from NVIDIA Megatron Core's fused MLA
design in ``megatron/core/fusions/fused_mla_yarn_rope_apply.py`` (Megatron
Core 0.17.0): apply RoPE to Q in place, directly assemble the expanded K while
applying K RoPE, and fuse KV-gradient packing with the shared K-position
gradient reduction. Credit belongs to the NVIDIA Megatron Core authors for
that design. This file is an independent TorchTitan adaptation and does not
import Megatron Core or TransformerEngine.

The implementation differs from Megatron Core in several important ways:

* It consumes TorchTitan's BSHD tensors and explicit per-example position IDs,
  rather than Megatron's SBHD/THD tensors and packed-sequence/CP indexing.
* It implements TorchTitan ``ComplexRoPE``'s adjacent-pair complex convention,
  rather than Megatron MLA's YaRN cos/sin output layout.
* V remains a zero-copy view of TorchTitan's packed KV projection; Megatron's
  fused KV path materializes a separate V output.
* It preserves TorchTitan eager's BF16/FP16 reduction-rounding boundary and
  wraps local results back into DTensors.
* Flattened offsets use 64-bit arithmetic because the traced 671B local tensors
  exceed 2**31 elements.
* Every Triton launch is exposed as a stable ``torch.library`` custom operator,
  so GraphTrainer's fake-tensor ``make_fx`` trace keeps the fused boundaries.

The override keeps the stock Attention parameters and state-dict layout.  It
only replaces the Q/KV layout boundary around ComplexRoPE:

* Q RoPE rotates the positional tail of the Q projection into a new tensor.
  The backward also rotates into fresh storage because autograd may share its
  incoming gradient with another consumer.
* K RoPE, head expansion, and final K materialization are one Triton kernel.
* V remains a view of the packed KV projection (no extra forward copy).
* KV backward packs dK-nope and dV while reducing/inverse-rotating dK-pos.

No Megatron-Core or TransformerEngine dependency is required.
"""

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor

from torchtitan.config import derive, override
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.rope import _maybe_check_max_pos, ComplexRoPE
from torchtitan.models.deepseek_v3.model import Attention

__all__ = [
    "FusedMLAAttention",
    "FusedMLAKernelConfig",
    "fused_mla",
    "fused_mla_q",
    "fused_mla_kv",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class FusedMLAKernelConfig:
    """Triton launch configuration for the three fused MLA kernels.

    The defaults were tuned on GB300 for DeepSeek-V3 with batch 16, sequence
    length 4096, and 128 attention heads.
    """

    q_block_h: int = 64
    q_num_warps: int = 4
    k_block_h: int = 16
    k_num_warps: int = 4
    kv_backward_block_h: int = 64
    kv_backward_num_warps: int = 4

    def __post_init__(self) -> None:
        """Validate values required by Triton launch configuration."""
        for name in ("q_block_h", "k_block_h", "kv_backward_block_h"):
            value = getattr(self, name)
            if value <= 0 or value & (value - 1):
                raise ValueError(f"{name} must be a positive power of two, got {value}")
        for name in ("q_num_warps", "k_num_warps", "kv_backward_num_warps"):
            value = getattr(self, name)
            if value not in (1, 2, 4, 8):
                raise ValueError(f"{name} must be one of 1, 2, 4, or 8, got {value}")


_DEFAULT_KERNEL_CONFIG = FusedMLAKernelConfig()

# Columns of q_nope carried per iteration when the RoPE kernel also passes the
# non-positional half through. The kernel holds a [BLOCK_H, BLOCK_NOPE] tile,
# so this trades register pressure against loop trips; 64 measured best across
# 32/64/128 at DeepSeek-V3 geometry.
_NOPE_BLOCK = 64


@triton.jit
def _fused_q_rope_kernel(
    q,
    q_out,
    rope_cache,
    positions,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_L: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    Q_STRIDE_D: tl.constexpr,
    QO_STRIDE_B: tl.constexpr,
    QO_STRIDE_L: tl.constexpr,
    QO_STRIDE_H: tl.constexpr,
    QO_STRIDE_D: tl.constexpr,
    CACHE_STRIDE_M: tl.constexpr,
    CACHE_STRIDE_P: tl.constexpr,
    CACHE_STRIDE_R: tl.constexpr,
    POS_STRIDE_B: tl.constexpr,
    POS_STRIDE_L: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    NUM_HEAD_BLOCKS: tl.constexpr,
    Q_NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    INVERSE: tl.constexpr,
    COPY_NOPE: tl.constexpr = False,
    BLOCK_NOPE: tl.constexpr = 64,
) -> None:
    # Production DeepSeek shapes exceed 2**31 elements, so every flattened
    # tensor index must be promoted before multiplying by a stride.
    program = tl.program_id(0).to(tl.int64)
    head_block = program % NUM_HEAD_BLOCKS
    token = program // NUM_HEAD_BLOCKS
    seq = token % SEQ_LEN
    batch = token // SEQ_LEN

    head = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    head_mask = head < N_HEADS

    if COPY_NOPE:
        # Carry the untouched q_nope half across in the same pass. Doing it
        # here instead of as a separate slice assignment turns a strided copy
        # of 128 of every 192 elements into part of one contiguous sweep; the
        # standalone copy measured 1.90 TB/s against roughly 5-6 achievable.
        # Chunked over the nope dimension so the register tile stays bounded.
        nope_in = batch * Q_STRIDE_B + seq * Q_STRIDE_L + head * Q_STRIDE_H
        nope_out = batch * QO_STRIDE_B + seq * QO_STRIDE_L + head * QO_STRIDE_H
        for start in tl.range(0, Q_NOPE_DIM, BLOCK_NOPE):
            col = start + tl.arange(0, BLOCK_NOPE)[None, :]
            nope_mask = head_mask & (col < Q_NOPE_DIM)
            tl.store(
                q_out + nope_out + col * QO_STRIDE_D,
                tl.load(q + nope_in + col * Q_STRIDE_D, mask=nope_mask, other=0.0),
                mask=nope_mask,
            )

    pair = tl.arange(0, BLOCK_ROPE // 2)[None, :]
    pair_mask = pair < ROPE_DIM // 2
    position = tl.load(positions + batch * POS_STRIDE_B + seq * POS_STRIDE_L)

    q_base = (
        batch * Q_STRIDE_B
        + seq * Q_STRIDE_L
        + head * Q_STRIDE_H
        + Q_NOPE_DIM * Q_STRIDE_D
    )
    qo_base = (
        batch * QO_STRIDE_B
        + seq * QO_STRIDE_L
        + head * QO_STRIDE_H
        + Q_NOPE_DIM * QO_STRIDE_D
    )
    # Load the head's rope slice as one contiguous BLOCK_ROPE-wide tile and
    # de-interleave in registers. Addressing the even and odd lanes as two
    # stride-2 gathers instead costs ~19x: it defeats vectorization, so each
    # 128-byte cache line is fetched by scalar 16-bit accesses.
    rope_lane = tl.arange(0, BLOCK_ROPE)[None, :]
    rope_mask = head_mask & (rope_lane < ROPE_DIM)
    q_pairs = tl.reshape(
        tl.load(q + q_base + rope_lane * Q_STRIDE_D, mask=rope_mask, other=0.0),
        (BLOCK_H, BLOCK_ROPE // 2, 2),
    )
    q_even, q_odd = tl.split(q_pairs)
    q_even = q_even.to(tl.float32)
    q_odd = q_odd.to(tl.float32)

    cache_base = position * CACHE_STRIDE_M + pair * CACHE_STRIDE_P
    cos = tl.load(
        rope_cache + cache_base,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        rope_cache + cache_base + CACHE_STRIDE_R,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)

    if INVERSE:
        out_even = q_even * cos + q_odd * sin
        out_odd = q_odd * cos - q_even * sin
    else:
        out_even = q_even * cos - q_odd * sin
        out_odd = q_even * sin + q_odd * cos

    tl.store(
        q_out + qo_base + rope_lane * QO_STRIDE_D,
        tl.reshape(tl.join(out_even, out_odd), (BLOCK_H, BLOCK_ROPE)),
        mask=rope_mask,
    )


@triton.jit
def _fused_k_rope_kernel(
    kv,
    k_pe,
    rope_cache,
    positions,
    k,
    KV_STRIDE_B: tl.constexpr,
    KV_STRIDE_L: tl.constexpr,
    KV_STRIDE_H: tl.constexpr,
    KV_STRIDE_D: tl.constexpr,
    KPE_STRIDE_B: tl.constexpr,
    KPE_STRIDE_L: tl.constexpr,
    KPE_STRIDE_D: tl.constexpr,
    CACHE_STRIDE_M: tl.constexpr,
    CACHE_STRIDE_P: tl.constexpr,
    CACHE_STRIDE_R: tl.constexpr,
    POS_STRIDE_B: tl.constexpr,
    POS_STRIDE_L: tl.constexpr,
    K_STRIDE_B: tl.constexpr,
    K_STRIDE_L: tl.constexpr,
    K_STRIDE_H: tl.constexpr,
    K_STRIDE_D: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    NUM_HEAD_BLOCKS: tl.constexpr,
    Q_NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
) -> None:
    program = tl.program_id(0).to(tl.int64)
    head_block = program % NUM_HEAD_BLOCKS
    token = program // NUM_HEAD_BLOCKS
    seq = token % SEQ_LEN
    batch = token // SEQ_LEN

    head = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    dim = tl.arange(0, BLOCK_D)[None, :]
    head_mask = head < N_HEADS
    nope_mask = head_mask & (dim < Q_NOPE_DIM)
    kv_base = batch * KV_STRIDE_B + seq * KV_STRIDE_L + head * KV_STRIDE_H
    k_base = batch * K_STRIDE_B + seq * K_STRIDE_L + head * K_STRIDE_H
    k_nope = tl.load(
        kv + kv_base + dim * KV_STRIDE_D,
        mask=nope_mask,
        other=0.0,
    )
    tl.store(k + k_base + dim * K_STRIDE_D, k_nope, mask=nope_mask)

    # k_pe is shared by every attention head. Rotate it once per head tile,
    # then broadcast the result while storing the tile instead of repeating the
    # FP32 complex multiply in a separate program for every head.
    pair = tl.arange(0, BLOCK_ROPE // 2)
    pair_mask = pair < ROPE_DIM // 2
    kpe_base = batch * KPE_STRIDE_B + seq * KPE_STRIDE_L
    # Contiguous load, de-interleaved in registers -- see _fused_q_rope_kernel.
    rope_lane_1d = tl.arange(0, BLOCK_ROPE)
    kpe_even, kpe_odd = tl.split(
        tl.reshape(
            tl.load(
                k_pe + kpe_base + rope_lane_1d * KPE_STRIDE_D,
                mask=rope_lane_1d < ROPE_DIM,
                other=0.0,
            ),
            (BLOCK_ROPE // 2, 2),
        )
    )
    kpe_even = kpe_even.to(tl.float32)
    kpe_odd = kpe_odd.to(tl.float32)

    position = tl.load(positions + batch * POS_STRIDE_B + seq * POS_STRIDE_L)
    cache_base = position * CACHE_STRIDE_M + pair * CACHE_STRIDE_P
    cos = tl.load(
        rope_cache + cache_base,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        rope_cache + cache_base + CACHE_STRIDE_R,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    out_even = kpe_even * cos - kpe_odd * sin
    out_odd = kpe_even * sin + kpe_odd * cos

    rope_base = k_base + Q_NOPE_DIM * K_STRIDE_D
    rope_out = tl.reshape(tl.join(out_even, out_odd), (BLOCK_ROPE,))
    tl.store(
        k + rope_base + rope_lane_1d[None, :] * K_STRIDE_D,
        tl.broadcast_to(rope_out[None, :], (BLOCK_H, BLOCK_ROPE)),
        mask=head_mask & (rope_lane_1d[None, :] < ROPE_DIM),
    )


@triton.jit
def _fused_kv_backward_kernel(
    grad_k,
    grad_v,
    rope_cache,
    positions,
    grad_kv,
    grad_k_pe,
    GK_STRIDE_B: tl.constexpr,
    GK_STRIDE_L: tl.constexpr,
    GK_STRIDE_H: tl.constexpr,
    GK_STRIDE_D: tl.constexpr,
    GV_STRIDE_B: tl.constexpr,
    GV_STRIDE_L: tl.constexpr,
    GV_STRIDE_H: tl.constexpr,
    GV_STRIDE_D: tl.constexpr,
    CACHE_STRIDE_M: tl.constexpr,
    CACHE_STRIDE_P: tl.constexpr,
    CACHE_STRIDE_R: tl.constexpr,
    POS_STRIDE_B: tl.constexpr,
    POS_STRIDE_L: tl.constexpr,
    GKV_STRIDE_B: tl.constexpr,
    GKV_STRIDE_L: tl.constexpr,
    GKV_STRIDE_H: tl.constexpr,
    GKV_STRIDE_D: tl.constexpr,
    GKPE_STRIDE_B: tl.constexpr,
    GKPE_STRIDE_L: tl.constexpr,
    GKPE_STRIDE_D: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    Q_NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    ROUND_BF16_SUM: tl.constexpr,
    ROUND_FP16_SUM: tl.constexpr,
) -> None:
    token = tl.program_id(0).to(tl.int64)
    seq = token % SEQ_LEN
    batch = token // SEQ_LEN

    dim = tl.arange(0, BLOCK_D)[None, :]
    grad_pos_even = tl.zeros((BLOCK_PAIRS,), dtype=tl.float32)
    grad_pos_odd = tl.zeros((BLOCK_PAIRS,), dtype=tl.float32)

    for head_start in tl.static_range(0, N_HEADS, BLOCK_H):
        head = head_start + tl.arange(0, BLOCK_H)[:, None]
        head_mask = head < N_HEADS

        gk_base = batch * GK_STRIDE_B + seq * GK_STRIDE_L + head * GK_STRIDE_H
        gv_base = batch * GV_STRIDE_B + seq * GV_STRIDE_L + head * GV_STRIDE_H
        gkv_base = batch * GKV_STRIDE_B + seq * GKV_STRIDE_L + head * GKV_STRIDE_H

        nope_mask = head_mask & (dim < Q_NOPE_DIM)
        grad_nope = tl.load(
            grad_k + gk_base + dim * GK_STRIDE_D,
            mask=nope_mask,
            other=0.0,
        )
        tl.store(
            grad_kv + gkv_base + dim * GKV_STRIDE_D,
            grad_nope,
            mask=nope_mask,
        )

        value_mask = head_mask & (dim < V_DIM)
        grad_value = tl.load(
            grad_v + gv_base + dim * GV_STRIDE_D,
            mask=value_mask,
            other=0.0,
        )
        tl.store(
            grad_kv + gkv_base + (Q_NOPE_DIM + dim) * GKV_STRIDE_D,
            grad_value,
            mask=value_mask,
        )

        # Contiguous load, de-interleaved in registers -- see
        # _fused_q_rope_kernel. Masked lanes read 0.0 and so do not perturb the
        # head reduction below.
        rope_lane = tl.arange(0, BLOCK_ROPE)[None, :]
        grad_even, grad_odd = tl.split(
            tl.reshape(
                tl.load(
                    grad_k + gk_base + (Q_NOPE_DIM + rope_lane) * GK_STRIDE_D,
                    mask=head_mask & (rope_lane < ROPE_DIM),
                    other=0.0,
                ),
                (BLOCK_H, BLOCK_ROPE // 2, 2),
            )
        )
        grad_even = grad_even.to(tl.float32)
        grad_odd = grad_odd.to(tl.float32)
        grad_pos_even += tl.sum(grad_even, axis=0)
        grad_pos_odd += tl.sum(grad_odd, axis=0)

    # Stock expand-backward materializes the head reduction in the input dtype
    # before ComplexRoPE backward upcasts it. Preserve that rounding boundary.
    if ROUND_BF16_SUM:
        grad_pos_even = grad_pos_even.to(tl.bfloat16).to(tl.float32)
        grad_pos_odd = grad_pos_odd.to(tl.bfloat16).to(tl.float32)
    if ROUND_FP16_SUM:
        grad_pos_even = grad_pos_even.to(tl.float16).to(tl.float32)
        grad_pos_odd = grad_pos_odd.to(tl.float16).to(tl.float32)

    pair_1d = tl.arange(0, BLOCK_PAIRS)
    pair_mask_1d = pair_1d < ROPE_DIM // 2
    position = tl.load(positions + batch * POS_STRIDE_B + seq * POS_STRIDE_L)
    cache_base = position * CACHE_STRIDE_M + pair_1d * CACHE_STRIDE_P
    cos = tl.load(
        rope_cache + cache_base,
        mask=pair_mask_1d,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        rope_cache + cache_base + CACHE_STRIDE_R,
        mask=pair_mask_1d,
        other=0.0,
    ).to(tl.float32)
    out_even = grad_pos_even * cos + grad_pos_odd * sin
    out_odd = grad_pos_odd * cos - grad_pos_even * sin

    gkpe_base = batch * GKPE_STRIDE_B + seq * GKPE_STRIDE_L
    rope_lane_1d = tl.arange(0, BLOCK_ROPE)
    tl.store(
        grad_k_pe + gkpe_base + rope_lane_1d * GKPE_STRIDE_D,
        tl.reshape(tl.join(out_even, out_odd), (BLOCK_ROPE,)),
        mask=rope_lane_1d < ROPE_DIM,
    )


def _q_rope_strides(q: torch.Tensor, q_head_dim: int) -> tuple[int, int]:
    """Return the per-head and per-element strides of a 3-D or 4-D query.

    Args:
        q: Query tensor shaped ``(B, L, H * D)`` or ``(B, L, H, D)``.
        q_head_dim: Per-head query dimension ``D``.

    Returns:
        The head stride and the last-dimension stride.
    """
    if q.ndim == 3:
        return q_head_dim * q.stride(2), q.stride(2)
    return q.stride(2), q.stride(3)


@torch.library.custom_op(
    "torchtitan::fused_mla_q_rope",
    mutates_args=(),
    device_types="cuda",
)
def _fused_mla_q_rope_out_op(
    q: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    inverse: bool,
    block_h: int,
    num_warps: int,
) -> torch.Tensor:
    """Rotate the RoPE half of the query into a freshly allocated tensor.

    The in-place variant needs ``ctx.mark_dirty`` in the autograd Function,
    which makes autograd record a ``CopySlices`` and materialise a full copy
    of the projection output in the backward pass. Writing to a new tensor
    removes that copy.

    The kernel carries the untouched ``q_nope`` half across itself
    (``COPY_NOPE``) rather than leaving it to a separate slice assignment. That
    assignment moved 128 of every 192 elements, so it was strided and ran at
    1.90 TB/s where the part sustains roughly 5-6; folding it into the rope
    sweep makes the traffic contiguous and halves the launches.
    """
    batch, seq_len = q.shape[:2]
    n_heads, q_head_dim = _query_geometry(
        q,
        q_nope_dim,
        2 * rope_cache_real.shape[-2],
    )
    rope_dim = q_head_dim - q_nope_dim
    out = torch.empty_like(q)

    q_stride_h, q_stride_d = _q_rope_strides(q, q_head_dim)
    out_stride_h, out_stride_d = _q_rope_strides(out, q_head_dim)
    block_h = min(block_h, triton.next_power_of_2(n_heads))
    num_head_blocks = triton.cdiv(n_heads, block_h)
    block_rope = triton.next_power_of_2(rope_dim)
    _fused_q_rope_kernel[(batch * seq_len * num_head_blocks,)](
        q,
        out,
        rope_cache_real,
        positions,
        Q_STRIDE_B=q.stride(0),
        Q_STRIDE_L=q.stride(1),
        Q_STRIDE_H=q_stride_h,
        Q_STRIDE_D=q_stride_d,
        QO_STRIDE_B=out.stride(0),
        QO_STRIDE_L=out.stride(1),
        QO_STRIDE_H=out_stride_h,
        QO_STRIDE_D=out_stride_d,
        CACHE_STRIDE_M=rope_cache_real.stride(0),
        CACHE_STRIDE_P=rope_cache_real.stride(1),
        CACHE_STRIDE_R=rope_cache_real.stride(2),
        POS_STRIDE_B=positions.stride(0),
        POS_STRIDE_L=positions.stride(1),
        SEQ_LEN=seq_len,
        N_HEADS=n_heads,
        NUM_HEAD_BLOCKS=num_head_blocks,
        Q_NOPE_DIM=q_nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_H=block_h,
        BLOCK_ROPE=block_rope,
        INVERSE=inverse,
        COPY_NOPE=True,
        BLOCK_NOPE=_NOPE_BLOCK,
        num_warps=num_warps,
    )
    return out


@_fused_mla_q_rope_out_op.register_fake
def _fused_mla_q_rope_out_op_fake(
    q: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    inverse: bool,
    block_h: int,
    num_warps: int,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "torchtitan::fused_mla_k_rope",
    mutates_args=(),
    device_types="cuda",
)
def _fused_mla_k_rope_op(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    block_h: int,
    num_warps: int,
) -> torch.Tensor:
    batch, seq_len, n_heads, _ = kv.shape
    rope_dim = k_pe.shape[-1]
    k = torch.empty(
        (batch, seq_len, n_heads, q_nope_dim + rope_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    block_h = min(block_h, triton.next_power_of_2(n_heads))
    num_head_blocks = triton.cdiv(n_heads, block_h)
    _fused_k_rope_kernel[(batch * seq_len * num_head_blocks,)](
        kv,
        k_pe,
        rope_cache_real,
        positions,
        k,
        KV_STRIDE_B=kv.stride(0),
        KV_STRIDE_L=kv.stride(1),
        KV_STRIDE_H=kv.stride(2),
        KV_STRIDE_D=kv.stride(3),
        KPE_STRIDE_B=k_pe.stride(0),
        KPE_STRIDE_L=k_pe.stride(1),
        KPE_STRIDE_D=k_pe.stride(2),
        CACHE_STRIDE_M=rope_cache_real.stride(0),
        CACHE_STRIDE_P=rope_cache_real.stride(1),
        CACHE_STRIDE_R=rope_cache_real.stride(2),
        POS_STRIDE_B=positions.stride(0),
        POS_STRIDE_L=positions.stride(1),
        K_STRIDE_B=k.stride(0),
        K_STRIDE_L=k.stride(1),
        K_STRIDE_H=k.stride(2),
        K_STRIDE_D=k.stride(3),
        SEQ_LEN=seq_len,
        N_HEADS=n_heads,
        NUM_HEAD_BLOCKS=num_head_blocks,
        Q_NOPE_DIM=q_nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_H=block_h,
        BLOCK_D=triton.next_power_of_2(q_nope_dim),
        BLOCK_PAIRS=triton.next_power_of_2(rope_dim // 2),
        BLOCK_ROPE=triton.next_power_of_2(rope_dim),
        num_warps=num_warps,
    )
    return k


@_fused_mla_k_rope_op.register_fake
def _fused_mla_k_rope_op_fake(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    block_h: int,
    num_warps: int,
) -> torch.Tensor:
    return torch.empty(
        (*kv.shape[:3], q_nope_dim + k_pe.shape[-1]),
        dtype=kv.dtype,
        device=kv.device,
    )


@torch.library.custom_op(
    "torchtitan::fused_mla_kv_backward",
    mutates_args=(),
    device_types="cuda",
)
def _fused_mla_kv_backward_op(
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    rope_dim: int,
    block_h: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, n_heads, _ = grad_k.shape
    v_dim = grad_v.shape[-1]
    grad_kv = torch.empty(
        (batch, seq_len, n_heads, q_nope_dim + v_dim),
        dtype=grad_k.dtype,
        device=grad_k.device,
    )
    grad_k_pe = torch.empty(
        (batch, seq_len, rope_dim),
        dtype=grad_k.dtype,
        device=grad_k.device,
    )
    block_h = min(block_h, triton.next_power_of_2(n_heads))
    _fused_kv_backward_kernel[(batch * seq_len,)](
        grad_k,
        grad_v,
        rope_cache_real,
        positions,
        grad_kv,
        grad_k_pe,
        GK_STRIDE_B=grad_k.stride(0),
        GK_STRIDE_L=grad_k.stride(1),
        GK_STRIDE_H=grad_k.stride(2),
        GK_STRIDE_D=grad_k.stride(3),
        GV_STRIDE_B=grad_v.stride(0),
        GV_STRIDE_L=grad_v.stride(1),
        GV_STRIDE_H=grad_v.stride(2),
        GV_STRIDE_D=grad_v.stride(3),
        CACHE_STRIDE_M=rope_cache_real.stride(0),
        CACHE_STRIDE_P=rope_cache_real.stride(1),
        CACHE_STRIDE_R=rope_cache_real.stride(2),
        POS_STRIDE_B=positions.stride(0),
        POS_STRIDE_L=positions.stride(1),
        GKV_STRIDE_B=grad_kv.stride(0),
        GKV_STRIDE_L=grad_kv.stride(1),
        GKV_STRIDE_H=grad_kv.stride(2),
        GKV_STRIDE_D=grad_kv.stride(3),
        GKPE_STRIDE_B=grad_k_pe.stride(0),
        GKPE_STRIDE_L=grad_k_pe.stride(1),
        GKPE_STRIDE_D=grad_k_pe.stride(2),
        SEQ_LEN=seq_len,
        N_HEADS=n_heads,
        Q_NOPE_DIM=q_nope_dim,
        ROPE_DIM=rope_dim,
        V_DIM=v_dim,
        BLOCK_H=block_h,
        BLOCK_D=triton.next_power_of_2(max(q_nope_dim, v_dim)),
        BLOCK_PAIRS=triton.next_power_of_2(rope_dim // 2),
        BLOCK_ROPE=triton.next_power_of_2(rope_dim),
        ROUND_BF16_SUM=grad_k.dtype == torch.bfloat16,
        ROUND_FP16_SUM=grad_k.dtype == torch.float16,
        num_warps=num_warps,
    )
    return grad_kv, grad_k_pe


@_fused_mla_kv_backward_op.register_fake
def _fused_mla_kv_backward_op_fake(
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    rope_dim: int,
    block_h: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(
            (*grad_k.shape[:3], q_nope_dim + grad_v.shape[-1]),
            dtype=grad_k.dtype,
            device=grad_k.device,
        ),
        torch.empty(
            (*grad_k.shape[:2], rope_dim),
            dtype=grad_k.dtype,
            device=grad_k.device,
        ),
    )


@spmd.register_autograd_function
class _FusedMLAQ(torch.autograd.Function):
    @staticmethod
    def typecheck_forward(
        q: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
        q_nope_dim: int,
        block_h: int,
        num_warps: int,
    ) -> torch.Tensor:
        q_type = (spmd.V, spmd.PartitionSpec(None, ("dp", "cp"), "tp", None))
        positions_type = (spmd.V, spmd.PartitionSpec(None, ("dp", "cp")))
        spmd.assert_type(q, *q_type)
        spmd.assert_type(rope_cache_real, spmd.R)
        spmd.assert_type(positions, *positions_type)
        output = _FusedMLAQ.apply(
            q,
            rope_cache_real,
            positions,
            q_nope_dim,
            block_h,
            num_warps,
        )
        spmd.assert_type(output, *q_type)
        return output

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
        q_nope_dim: int,
        block_h: int,
        num_warps: int,
    ) -> torch.Tensor:
        ctx.q_nope_dim = q_nope_dim
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        ctx.save_for_backward(rope_cache_real, positions)
        # Out-of-place: mutating the projection output here would need
        # ctx.mark_dirty, and autograd would then record a CopySlices whose
        # backward materialises a full copy of that tensor.
        return _fused_mla_q_rope_out_op(
            q,
            rope_cache_real,
            positions,
            q_nope_dim,
            False,
            block_h,
            num_warps,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_q: torch.Tensor):
        rope_cache_real, positions = ctx.saved_tensors
        # Autograd may share grad_q with sibling backward paths. Rotate into
        # fresh storage so this Function does not mutate another consumer's
        # gradient. The fused kernel copies q_nope and rotates q_rope together.
        grad_q_local = _fused_mla_q_rope_out_op(
            _to_local(grad_q),
            rope_cache_real,
            positions,
            ctx.q_nope_dim,
            True,
            ctx.block_h,
            ctx.num_warps,
        )
        return _from_local(grad_q_local, grad_q), None, None, None, None, None


@spmd.register_autograd_function
class _FusedMLAKV(torch.autograd.Function):
    @staticmethod
    def typecheck_forward(
        kv: torch.Tensor,
        k_pe: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
        q_nope_dim: int,
        k_block_h: int,
        k_num_warps: int,
        backward_block_h: int,
        backward_num_warps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_type = (spmd.V, spmd.PartitionSpec(None, ("dp", "cp"), "tp", None))
        k_pe_type = (spmd.V, spmd.PartitionSpec(None, ("dp", "cp"), None))
        positions_type = (spmd.V, spmd.PartitionSpec(None, ("dp", "cp")))
        spmd.assert_type(kv, *kv_type)
        spmd.assert_type(k_pe, *k_pe_type)
        spmd.assert_type(rope_cache_real, spmd.R)
        spmd.assert_type(positions, *positions_type)
        k, v = _FusedMLAKV.apply(
            kv,
            k_pe,
            rope_cache_real,
            positions,
            q_nope_dim,
            k_block_h,
            k_num_warps,
            backward_block_h,
            backward_num_warps,
        )
        spmd.assert_type(k, *kv_type)
        spmd.assert_type(v, *kv_type)
        return k, v

    @staticmethod
    def forward(
        ctx,
        kv: torch.Tensor,
        k_pe: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
        q_nope_dim: int,
        k_block_h: int,
        k_num_warps: int,
        backward_block_h: int,
        backward_num_warps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.q_nope_dim = q_nope_dim
        ctx.rope_dim = k_pe.shape[-1]
        ctx.backward_block_h = backward_block_h
        ctx.backward_num_warps = backward_num_warps
        ctx.save_for_backward(rope_cache_real, positions)
        k = _fused_mla_k_rope_op(
            kv,
            k_pe,
            rope_cache_real,
            positions,
            q_nope_dim,
            k_block_h,
            k_num_warps,
        )
        # Preserve the stock zero-copy V view. The custom backward combines its
        # gradient with dK-nope directly into the packed KV gradient.
        v = kv[..., q_nope_dim:]
        return k, v

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_k: torch.Tensor, grad_v: torch.Tensor):
        rope_cache_real, positions = ctx.saved_tensors
        grad_kv, grad_k_pe = _fused_mla_kv_backward_op(
            grad_k,
            grad_v,
            rope_cache_real,
            positions,
            ctx.q_nope_dim,
            ctx.rope_dim,
            ctx.backward_block_h,
            ctx.backward_num_warps,
        )
        return grad_kv, grad_k_pe, None, None, None, None, None, None, None


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _from_local(local: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
    if isinstance(spec, DTensor):
        return DTensor.from_local(
            local,
            spec.device_mesh,
            spec.placements,
            run_check=False,
        )
    return local


def _resolve_positions(
    positions: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if positions is not None:
        pos = _to_local(positions)
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        batch = reference.shape[0]
        if pos.shape[0] == 1 and batch != 1:
            pos = pos.expand(batch, -1)
        return pos.contiguous()
    batch, seq_len = reference.shape[:2]
    pos = torch.arange(seq_len, device=reference.device, dtype=torch.int32)
    return pos.unsqueeze(0).expand(batch, -1).contiguous()


def _query_geometry(
    q: torch.Tensor,
    q_nope_dim: int,
    rope_dim: int,
) -> tuple[int, int]:
    """Resolve logical MLA head geometry from flat or explicit layout.

    Args:
        q: Query tensor in ``[B, L, H * D]`` or ``[B, L, H, D]`` layout.
        q_nope_dim: Non-positional dimensions in each query head.
        rope_dim: Positional dimensions in each query head.

    Returns:
        Number of query heads and total query head dimension.

    Raises:
        ValueError: If the rank or logical head dimensions are invalid.
    """
    q_head_dim = q_nope_dim + rope_dim
    if q.ndim == 3:
        if q.shape[-1] % q_head_dim != 0:
            raise ValueError(
                "The flattened query projection must contain whole MLA heads"
            )
        return q.shape[-1] // q_head_dim, q_head_dim
    if q.ndim == 4:
        if q.shape[-1] != q_head_dim:
            raise ValueError(
                f"Expected query head dimension {q_head_dim}, got {q.shape[-1]}"
            )
        return q.shape[-2], q_head_dim
    raise ValueError(f"Expected a 3-D or 4-D query tensor, got {q.ndim}-D")


def fused_mla_q(
    q: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None,
    q_nope_dim: int,
    kernel_config: FusedMLAKernelConfig = _DEFAULT_KERNEL_CONFIG,
) -> torch.Tensor:
    """Apply ComplexRoPE to Q's positional tail.

    Args:
        q: Contiguous ``[B, L, H * D]`` projection output or logical
            ``[B, L, H, D]`` tensor.
        rope_cache: Complex-valued rotary cache.
        positions: Optional token positions for each batch row.
        q_nope_dim: Non-positional dimensions in each query head.
        kernel_config: Triton launch configuration.

    Returns:
        A new tensor holding the non-positional dimensions unchanged and the
        positional ones rotated. ``q`` is left untouched.
    """
    q_local = _to_local(q)
    cache_local = _to_local(rope_cache)
    rope_dim = 2 * cache_local.shape[-1]
    _query_geometry(q_local, q_nope_dim, rope_dim)
    positions_local = _resolve_positions(positions, q_local)
    cache_real = torch.view_as_real(cache_local).contiguous()
    return _FusedMLAQ.apply(
        q,
        cache_real,
        positions_local,
        q_nope_dim,
        kernel_config.q_block_h,
        kernel_config.q_num_warps,
    )


def fused_mla_kv(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None,
    q_nope_dim: int,
    kernel_config: FusedMLAKernelConfig = _DEFAULT_KERNEL_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize K and expose V with a fused custom backward."""
    kv_local = _to_local(kv)
    kpe_local = _to_local(k_pe)
    cache_local = _to_local(rope_cache)
    positions_local = _resolve_positions(positions, kv_local)
    cache_real = torch.view_as_real(cache_local).contiguous()
    k_local, v_local = _FusedMLAKV.apply(
        kv_local,
        kpe_local,
        cache_real,
        positions_local,
        q_nope_dim,
        kernel_config.k_block_h,
        kernel_config.k_num_warps,
        kernel_config.kv_backward_block_h,
        kernel_config.kv_backward_num_warps,
    )
    return _from_local(k_local, kv), _from_local(v_local, kv)


class FusedMLAAttention(Attention):
    """Stock DeepSeek-V3 attention with fused MLA tensor assembly."""

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        kernel_config: FusedMLAKernelConfig = field(
            default_factory=FusedMLAKernelConfig
        )

    def __init__(self, config: Config):
        super().__init__(config)
        self.kernel_config = config.kernel_config
        if not isinstance(self.rope, ComplexRoPE):
            raise TypeError(
                "FusedMLAAttention currently requires ComplexRoPE, got "
                f"{type(self.rope).__name__}."
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            return super().forward(x, attention_masks, positions)

        num_tokens = x.shape[0]
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        with spmd.local():
            q = q.view(num_tokens, -1, self.qk_head_dim)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    q,
                    spmd.V,
                    spmd.PartitionSpec(("dp", "cp"), "tp", None),
                )

        if positions is not None:
            _maybe_check_max_pos(
                positions,
                max_valid_pos=self.rope.cache.shape[0] - 1,
            )
        q = fused_mla_q(
            q.unsqueeze(0),
            self.rope.cache,
            positions,
            self.qk_nope_head_dim,
            self.kernel_config,
        ).squeeze(0)

        kv_down = self.wkv_a(x)
        kv_latent, k_pe = torch.split(
            kv_down,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )

        kv = self.wkv_b(self.kv_norm(kv_latent))
        with spmd.local():
            kv = kv.view(num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)
            k, v = fused_mla_kv(
                kv.unsqueeze(0),
                k_pe.unsqueeze(0),
                self.rope.cache,
                positions,
                self.qk_nope_head_dim,
                self.kernel_config,
            )
            k, v = k.squeeze(0), v.squeeze(0)
            if (
                get_spmd_backend() == "spmd_types"
                and spmd.is_type_checking()
                and not torch.compiler.is_compiling()
            ):
                for tensor in (k, v):
                    spmd.assert_type(
                        tensor,
                        spmd.V,
                        spmd.PartitionSpec(("dp", "cp"), "tp", None),
                    )

        output = self.inner_attention(
            q,
            k,
            v,
            attention_masks=attention_masks,
            scale=self.softmax_scale,
        ).contiguous()
        output = output.view(num_tokens, -1)
        return self.wo(output)


@override(
    target=Attention.Config,
    description="Fuse DeepSeek-V3 MLA Q/KV RoPE assembly with Triton kernels.",
)
def fused_mla(cfg: Attention.Config) -> FusedMLAAttention.Config:
    return derive(cfg, FusedMLAAttention.Config)
