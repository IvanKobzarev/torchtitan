# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import weakref
from dataclasses import dataclass, field, fields
from importlib.util import find_spec
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.distributed._prepared_weight import _FSDPPreparedWeight
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import swap_token_dispatcher

try:
    from torchao.prototype.mx_formats.kernels import (
        mxfp8_quantize_cuda,
        triton_to_mxfp8_32x32_swizzle_dim0_and_dim1,
        triton_to_mxfp8_32x32_swizzle_dim0_and_dim1_out,
    )
    from torchao.prototype.mx_formats.utils import to_blocked

    @dataclass(frozen=True, slots=True)
    class _MXFP8PreparedWeight:
        """Explicit MXFP8 weight operands for one unshard lifetime."""

        qdata: torch.Tensor
        fprop_scale: torch.Tensor
        dgrad_scale: torch.Tensor

    class _MXFP8FSDPWeight(_FSDPPreparedWeight):
        """BF16 FSDP parameter carrying unshard-lifetime MXFP8 operands."""

        def __init__(
            self,
            tensor: torch.Tensor,
            prepared: _MXFP8PreparedWeight | None = None,
            **logical_metadata,
        ) -> None:
            super().__init__(tensor, prepared, **logical_metadata)
            if prepared is not None:
                self._qdata = prepared.qdata
                self._fprop_scale = prepared.fprop_scale
                self._dgrad_scale = prepared.dgrad_scale

        def _new(
            self,
            tensor: torch.Tensor,
            prepared: _MXFP8PreparedWeight | None,
            **logical_metadata,
        ) -> _MXFP8FSDPWeight:
            """Construct an MXFP8 wrapper for one FSDP lifecycle state.

            Args:
                tensor: BF16 storage or prepared-state metadata anchor.
                prepared: Optional shared qdata and scale tensors.
                **logical_metadata: Logical tensor metadata overrides.

            Returns:
                MXFP8 FSDP weight wrapper.
            """
            return _MXFP8FSDPWeight(
                tensor,
                prepared,
                **logical_metadata,
            )

        def __tensor_flatten__(self):
            """Expose the storage owned by the current FSDP lifecycle.

            Returns:
                Inner tensor names and format metadata.
            """
            if self._tensor is not None:
                return ["_tensor"], ("sharded", self.dtype)
            return [
                "_qdata",
                "_fprop_scale",
                "_dgrad_scale",
            ], ("prepared", self.dtype)

        @staticmethod
        def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
            """Restore a sharded wrapper without prepared operands.

            Args:
                inner_tensors: Serialized high-precision storage.
                metadata: Format metadata returned by ``__tensor_flatten__``.
                outer_size: Serialized outer shape.
                outer_stride: Serialized outer stride.

            Returns:
                Sharded MXFP8 FSDP weight.
            """
            state, dtype = metadata
            if state == "sharded":
                return _MXFP8FSDPWeight(inner_tensors["_tensor"])
            prepared = _MXFP8PreparedWeight(
                qdata=inner_tensors["_qdata"],
                fprop_scale=inner_tensors["_fprop_scale"],
                dgrad_scale=inner_tensors["_dgrad_scale"],
            )
            return _MXFP8FSDPWeight(
                prepared.qdata,
                prepared,
                _logical_size=outer_size,
                _logical_stride=outer_stride,
                _logical_dtype=dtype,
                _logical_device=prepared.qdata.device,
            )

        def _prepare(
            self,
            weight: torch.Tensor,
            out: _MXFP8PreparedWeight | None = None,
        ) -> _MXFP8PreparedWeight:
            """Allocate or refill shared qdata and both scale layouts.

            Args:
                weight: Unsharded high-precision weight.
                out: Optional caller-owned qdata and scale outputs.

            Returns:
                Shared qdata, forward scales, and dgrad scales.
            """
            if out is None:
                (
                    qdata,
                    fprop_scale,
                    dgrad_scale,
                ) = triton_to_mxfp8_32x32_swizzle_dim0_and_dim1(weight)
                return _MXFP8PreparedWeight(qdata, fprop_scale, dgrad_scale)
            triton_to_mxfp8_32x32_swizzle_dim0_and_dim1_out(
                weight,
                out.qdata,
                out.fprop_scale,
                out.dgrad_scale,
            )
            return out

        def _prepared_tensors(
            self,
            prepared: _MXFP8PreparedWeight,
        ) -> tuple[torch.Tensor, ...]:
            """Return every independently allocated MXFP8 output tensor.

            Args:
                prepared: Shared qdata and both scale layouts.

            Returns:
                Tensors whose storage follows the FSDP unshard lifetime.
            """
            return (
                prepared.qdata,
                prepared.fprop_scale,
                prepared.dgrad_scale,
            )

        def prepared_operands(self) -> _MXFP8PreparedWeight | None:
            """Return explicit FPROP and DGRAD operands when materialized.

            Returns:
                Prepared weight state, or ``None`` while sharded.
            """
            return self.prepared_state()

    def _blocked_scale(scale: torch.Tensor) -> torch.Tensor:
        """Convert a logical E8M0 scale matrix to the cuBLAS blocked layout.

        Args:
            scale: Logical rowwise or columnwise E8M0 scale matrix.

        Returns:
            Flattened scale tensor in the layout consumed by ``_scaled_mm``.
        """
        return to_blocked(scale).view(torch.float8_e8m0fnu)

    def _scaled_mm(
        lhs_data: torch.Tensor,
        rhs_data: torch.Tensor,
        lhs_scale: torch.Tensor,
        rhs_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Run one BF16-output MXFP8 matrix multiplication.

        Args:
            lhs_data: Row-major left E4M3 operand.
            rhs_data: Row- or column-major right E4M3 operand.
            lhs_scale: Blocked left E8M0 scales.
            rhs_scale: Blocked right E8M0 scales.

        Returns:
            BF16 matrix product.
        """
        return torch._scaled_mm(
            lhs_data,
            rhs_data,
            lhs_scale,
            rhs_scale,
            out_dtype=torch.bfloat16,
        )

    def _local_parameter_grad(
        parameter_ref: weakref.ReferenceType[torch.Tensor] | None,
        expected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Resolve a compatible unsharded gradient owned by autograd.

        Args:
            parameter_ref: Weak reference to the original parameter owner.
            expected: Logical unsharded weight used by this backward.

        Returns:
            The parameter and its local gradient, or ``None`` before the first
            contribution.

        Raises:
            RuntimeError: If the parameter was released or its gradient does
                not match the unsharded WGRAD contract.
        """
        if parameter_ref is None:
            raise RuntimeError(
                "in-place MXFP8 WGRAD accumulation requires a parameter owner"
            )
        parameter = parameter_ref()
        if parameter is None:
            raise RuntimeError("MXFP8Linear parameter was released before backward")
        grad = parameter.grad
        if grad is None:
            return None
        local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
        if (
            local_grad.shape != expected.shape
            or local_grad.dtype != torch.bfloat16
            or local_grad.device != expected.device
            or not local_grad.is_contiguous()
        ):
            raise RuntimeError(
                "in-place MXFP8 WGRAD accumulation requires a contiguous "
                "unsharded BF16 parameter gradient"
            )
        return parameter, local_grad

    @torch._dynamo.allow_in_graph
    class _MXFP8LinearFunction(torch.autograd.Function):
        """Autograd composition over explicit MXFP8 qdata and scale tensors."""

        @staticmethod
        # pyrefly: ignore [bad-override]
        def forward(
            ctx,
            input_hp: torch.Tensor,
            weight_hp: torch.Tensor,
            weight_qdata: torch.Tensor,
            weight_fprop_scale: torch.Tensor,
            weight_dgrad_scale: torch.Tensor,
            parameter_ref: weakref.ReferenceType[torch.Tensor] | None,
            inplace_wgrad_accum: bool,
        ) -> torch.Tensor:
            """Quantize activations and run the forward scaled GEMM.

            Args:
                ctx: Autograd context used to retain WGRAD operands.
                input_hp: High-precision activation ending in ``in_features``.
                weight_hp: Logical high-precision parameter receiving WGRAD.
                weight_qdata: Shared 32x32 E4M3 weight payload.
                weight_fprop_scale: Blocked FPROP E8M0 scales.
                weight_dgrad_scale: Blocked DGRAD E8M0 scales.
                parameter_ref: Weak reference to the parameter owning WGRAD.
                inplace_wgrad_accum: Whether to fuse later WGRAD contributions
                    into the existing parameter gradient.

            Returns:
                BF16 output flattened to two dimensions.
            """
            input_2d = input_hp.reshape(-1, input_hp.shape[-1])
            save_wgrad_input = weight_hp.requires_grad
            (
                input_row,
                input_col,
                input_row_scale,
                input_col_scale,
            ) = mxfp8_quantize_cuda(
                input_2d,
                rowwise=True,
                colwise=save_wgrad_input,
                scaling_mode="rceil",
            )
            if not save_wgrad_input:
                input_col = input_hp.new_empty(0, dtype=torch.float8_e4m3fn)
                input_col_scale = input_hp.new_empty(
                    0,
                    dtype=torch.float8_e8m0fnu,
                )
            prepared_weight = isinstance(weight_hp, _MXFP8FSDPWeight)
            saved_qdata = (
                input_hp.new_empty(0, dtype=torch.float8_e4m3fn)
                if prepared_weight
                else weight_qdata
            )
            saved_dgrad_scale = (
                input_hp.new_empty(0, dtype=torch.float8_e8m0fnu)
                if prepared_weight
                else weight_dgrad_scale
            )
            ctx.input_shape = input_hp.shape
            ctx.prepared_weight = prepared_weight
            ctx.parameter_ref = parameter_ref
            ctx.inplace_wgrad_accum = inplace_wgrad_accum
            ctx.save_for_backward(
                input_col,
                input_col_scale,
                weight_hp,
                saved_qdata,
                saved_dgrad_scale,
            )
            return _scaled_mm(
                input_row,
                weight_qdata.t(),
                _blocked_scale(input_row_scale),
                weight_fprop_scale.flatten(),
            )

        @staticmethod
        # pyrefly: ignore [bad-override]
        def backward(ctx, grad_output_hp: torch.Tensor):
            """Compute DGRAD and WGRAD with explicit MXFP8 operands.

            Args:
                ctx: Autograd context populated by :meth:`forward`.
                grad_output_hp: High-precision output gradient.

            Returns:
                Gradients for the high-precision input and weight followed by
                ``None`` for prepared operands.
            """
            (
                input_col,
                input_col_scale,
                weight_hp,
                weight_qdata,
                weight_dgrad_scale,
            ) = ctx.saved_tensors
            if ctx.prepared_weight:
                if not isinstance(weight_hp, _MXFP8FSDPWeight):
                    raise RuntimeError("FSDP restored an incompatible MXFP8 weight")
                prepared = weight_hp.prepared_operands()
                if prepared is None:
                    raise RuntimeError("FSDP did not prepare MXFP8 weight for backward")
                weight_qdata = prepared.qdata
                weight_dgrad_scale = prepared.dgrad_scale
            grad_output_2d = grad_output_hp.contiguous().reshape(
                -1,
                grad_output_hp.shape[-1],
            )
            needs_dgrad, needs_wgrad = ctx.needs_input_grad[:2]
            grad_row, grad_col, grad_row_scale, grad_col_scale = mxfp8_quantize_cuda(
                grad_output_2d,
                rowwise=needs_dgrad,
                colwise=needs_wgrad,
                scaling_mode="rceil",
            )
            grad_input = None
            if needs_dgrad:
                grad_input = _scaled_mm(
                    grad_row,
                    weight_qdata,
                    _blocked_scale(grad_row_scale),
                    weight_dgrad_scale.flatten(),
                ).reshape(*ctx.input_shape)

            grad_weight = None
            if needs_wgrad:
                grad_col_t = grad_col.t()
                grad_scale = _blocked_scale(grad_col_scale)
                input_scale = _blocked_scale(input_col_scale)
                owner_and_grad = (
                    _local_parameter_grad(ctx.parameter_ref, weight_hp)
                    if ctx.inplace_wgrad_accum
                    else None
                )
                if owner_and_grad is None:
                    grad_weight = _scaled_mm(
                        grad_col_t,
                        input_col,
                        grad_scale,
                        input_scale,
                    )
                else:
                    parameter, previous = owner_and_grad
                    previous._scaled_addmm_(  # pyrefly: ignore [missing-attribute]
                        grad_col_t,
                        input_col,
                        grad_scale,
                        input_scale,
                    )
                    parameter.grad = None
                    grad_weight = previous
            return grad_input, grad_weight, None, None, None, None, None

    class MXFP8Linear(Linear):
        """TorchTitan MXFP8 linear with FSDP-aware prepared weight lifetime."""

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            """Drop-in replacement for Linear.Config that builds MXFP8Linear."""

            weight_gather: Literal["bf16", "mxfp8"] = "bf16"
            inplace_wgrad_accum: bool = False

        def __init__(self, config: Config):
            if config.weight_gather != "bf16":
                raise ValueError(
                    "MXFP8 parameter all-gather is not implemented; "
                    "weight_gather must be 'bf16'"
                )
            super().__init__(config)
            self.inplace_wgrad_accum = config.inplace_wgrad_accum

        def configure_fsdp(self) -> None:
            """Tie MXFP8 operands to each resolved FSDP unshard lifetime."""
            weight = self.weight
            if isinstance(weight, DTensor):
                local = weight.to_local()
                if isinstance(local, _MXFP8FSDPWeight):
                    return
                wrapped = _MXFP8FSDPWeight(local)
                weight = DTensor.from_local(
                    wrapped,
                    weight.device_mesh,
                    weight.placements,
                    run_check=False,
                    shape=weight.shape,
                    stride=weight.stride(),
                )
            elif not isinstance(weight, _MXFP8FSDPWeight):
                weight = _MXFP8FSDPWeight(weight)
            torch.utils.swap_tensors(
                self.weight,
                nn.Parameter(weight, requires_grad=self.weight.requires_grad),
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Run dynamic activation quantization and optional prepared weights.

            Args:
                input: High-precision activation ending in ``in_features``.

            Returns:
                High-precision linear output.
            """
            if self.inplace_wgrad_accum and torch.compiler.is_compiling():
                raise RuntimeError(
                    "in-place MXFP8 WGRAD accumulation is supported only in eager mode"
                )
            weight = (
                self.weight.to_local()
                if isinstance(self.weight, DTensor)
                else self.weight
            )
            prepared = (
                weight.prepared_operands()
                if isinstance(weight, _MXFP8FSDPWeight)
                else None
            )
            if prepared is None:
                with torch.no_grad():
                    (
                        qdata,
                        fprop_scale,
                        dgrad_scale,
                    ) = triton_to_mxfp8_32x32_swizzle_dim0_and_dim1(weight)
                prepared = _MXFP8PreparedWeight(
                    qdata,
                    fprop_scale,
                    dgrad_scale,
                )
            output = _MXFP8LinearFunction.apply(
                input,
                weight,
                prepared.qdata,
                prepared.fprop_scale,
                prepared.dgrad_scale,
                weakref.ref(self.weight) if self.inplace_wgrad_accum else None,
                self.inplace_wgrad_accum,
            )
            output = output.view(*input.shape[:-1], output.shape[-1])
            if self.bias is not None:
                bias = (
                    self.bias.to_local()
                    if isinstance(self.bias, DTensor)
                    else self.bias
                )
                output = output + bias.to(output.dtype)
            return output

except ImportError:
    MXFP8Linear = None


class MXFP8LinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with MXFP8Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply MXFP8 quantization to.
        Only Linear.Config entries whose FQN contains a match are converted.
        If empty, all Linear modules are converted.
        """
        weight_gather: Literal["bf16", "mxfp8"] = "bf16"
        """Parameter communication format; only BF16 is currently implemented."""
        inplace_wgrad_accum: bool = False
        """Fuse serialized eager WGRAD contributions into ``parameter.grad``."""

    def __init__(self, config: Config):
        self.config = config

        if MXFP8Linear is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )
        if self.config.weight_gather != "bf16":
            raise ValueError(
                "MXFP8 parameter all-gather is not implemented; "
                "weight_gather must be 'bf16'"
            )
        if self.config.inplace_wgrad_accum and self.config.model_compile_enabled:
            raise ValueError(
                "in-place MXFP8 WGRAD accumulation requires eager execution"
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config):
        assert MXFP8Linear is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = MXFP8Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                    weight_gather=self.config.weight_gather,
                    inplace_wgrad_accum=self.config.inplace_wgrad_accum,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to MXFP8Linear")
        return model_config


_mxfp8_experts_cache: dict[type, type] = {}


def _get_mxfp8_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an MXFP8-quantized subclass of *parent_cls*.

    Works for any experts module exposing the ``_grouped_mm`` seam (the common
    ``GroupedExperts`` and ``GptOssGroupedExperts``). The returned class has a
    proper ``_owner`` set by ``__init_subclass__``.

    The subclass overrides ``_grouped_mm`` to call torchao's
    ``_quantize_then_scaled_grouped_mm``.
    """
    if parent_cls in _mxfp8_experts_cache:
        return _mxfp8_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP8GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            recipe_name: str = "mxfp8_rceil"

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import (
                MXFP8TrainingOpConfig,
                MXFP8TrainingRecipe,
            )

            recipe = MXFP8TrainingRecipe(config.recipe_name)
            self._mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)

        def _grouped_mm(self, *, A, B_t, offs):
            from torchao.prototype.moe_training.utils import (
                _quantize_then_scaled_grouped_mm,
            )

            return _quantize_then_scaled_grouped_mm(
                A, B_t, config=self._mxfp8_op_config, offs=offs
            )

    MXFP8GroupedExperts.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8GroupedExperts.__qualname__ = f"MXFP8{parent_cls.__name__}"
    _mxfp8_experts_cache[parent_cls] = MXFP8GroupedExperts
    return MXFP8GroupedExperts


class MXFP8GroupedExpertsConverter(QuantizationConverter):
    """Apply MXFP8 quantization to MoE expert grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode
          when computing the e8m0 scale factors.
        """
        pad_multiple: int = 32
        """
        Pad per-expert token groups to this multiple for MXFP8 grouped GEMM alignment.
        The CuTeDSL quantization kernel on sm_100 requires multiples of 128.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config):
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            # ``parent`` is the RoutedExperts.Config owning inner_experts + dispatcher.
            swap_token_dispatcher(parent, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                recipe_name=self.config.recipe_name,
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Converted GroupedExperts to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm ops"
        )
        return model_config
