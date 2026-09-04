# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private FSDP lifecycle support for derived weight representations."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


_FSDP_SUBCLASS_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}

_FSDP_PREPARED_VIEW_OPS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
}

_FSDP_PREPARED_FACTORY_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.zeros_like.default,
}


class _PrepareFSDPWeight(torch.autograd.Function):
    """Preserve the SimpleFSDP gradient edge through weight preparation."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, weight: torch.Tensor, preparer: _FSDPPreparedWeight):
        """Prepare a weight without retaining its replicated storage.

        Args:
            ctx: Autograd context, unused because backward is an identity.
            weight: Replicated high-precision parameter from SimpleFSDP.
            preparer: Sharded wrapper defining the compute representation.

        Returns:
            Storage-free prepared parameter linked to ``weight`` for backward.
        """
        del ctx
        return preparer._prepare_unsharded_impl(weight)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_weight: torch.Tensor):
        """Return the logical parameter gradient to SimpleFSDP.

        Args:
            ctx: Autograd context, unused because forward saves no tensors.
            grad_weight: Gradient for the logical high-precision parameter.

        Returns:
            Gradient for the replicated input and no preparer gradient.
        """
        del ctx
        return grad_weight, None


class _FSDPPreparedWeight(torch.Tensor):
    """High-precision FSDP storage with unshard-lifetime compute operands."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, *args: Any, **kwargs: Any):
        """Create a transparent wrapper with explicit logical metadata.

        Args:
            tensor: High-precision storage or a prepared-state metadata anchor.
            *args: Subclass construction metadata.
            **kwargs: Optional logical tensor metadata overrides.

        Returns:
            Wrapper with the requested logical tensor metadata.
        """
        del args
        logical_size = kwargs.get("_logical_size", tensor.size())
        logical_stride = kwargs.get("_logical_stride", tensor.stride())
        logical_storage_offset = kwargs.get(
            "_logical_storage_offset",
            tensor.storage_offset(),
        )
        logical_dtype = kwargs.get("_logical_dtype", tensor.dtype)
        logical_device = kwargs.get("_logical_device", tensor.device)
        logical_requires_grad = kwargs.get(
            "_logical_requires_grad",
            tensor.requires_grad,
        )
        return torch.Tensor._make_wrapper_subclass(
            cls,
            logical_size,
            strides=logical_stride,
            storage_offset=logical_storage_offset,
            dtype=logical_dtype,
            layout=tensor.layout,
            device=logical_device,
            pin_memory=tensor.is_pinned(),
            requires_grad=logical_requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        prepared: Any = None,
        **logical_metadata: Any,
    ) -> None:
        del logical_metadata
        self._tensor = tensor if prepared is None else None
        self._prepared = prepared

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """Preserve wrapper state across FSDP storage operations.

        Args:
            func: Tensor operation dispatched on the wrapper.
            types: Participating tensor subclass types.
            args: Positional operation arguments.
            kwargs: Optional keyword operation arguments.

        Returns:
            Plain compute results or wrappers preserving FSDP lifecycle state.
        """
        del types
        template = None
        preserve_wrapper = func in _FSDP_SUBCLASS_OPS or func in (
            torch.ops.aten.detach.default,
            torch.ops.aten.alias.default,
        )

        def unwrap(tensor: _FSDPPreparedWeight) -> torch.Tensor:
            """Return inner storage while retaining one wrapper template."""
            nonlocal template
            if template is None:
                template = tensor
            elif preserve_wrapper and not tensor._same_metadata(template):
                raise RuntimeError("FSDP operation mixed prepared-weight metadata")
            if tensor._tensor is None:
                return torch.empty_strided(
                    tensor.size(),
                    tensor.stride(),
                    dtype=tensor.dtype,
                    device="meta",
                    requires_grad=tensor.requires_grad,
                )
            return tensor._tensor

        original_args = args
        original_kwargs = kwargs or {}
        args, kwargs = pytree.tree_map_only(
            cls,
            unwrap,
            (original_args, original_kwargs),
        )
        if template is not None and template._tensor is None:
            if func in _FSDP_PREPARED_FACTORY_OPS:
                kwargs["device"] = template.device
                return func(*args, **kwargs)
            if func not in _FSDP_PREPARED_VIEW_OPS:
                raise RuntimeError(
                    f"{func} attempted to read a storage-free FSDP prepared weight"
                )
            output = func(*args, **kwargs)
            wrapped = pytree.tree_map_only(
                torch.Tensor,
                template._rewrap_prepared_view,
                output,
            )
            return return_and_correct_aliasing(
                func,
                original_args,
                original_kwargs,
                wrapped,
            )
        output = func(*args, **kwargs)
        if not preserve_wrapper:
            return output
        assert template is not None
        return pytree.tree_map_only(torch.Tensor, template._rewrap, output)

    def _same_metadata(self, other: _FSDPPreparedWeight) -> bool:
        """Return whether two wrappers may participate in one tensor op.

        Args:
            other: Second prepared-weight wrapper.

        Returns:
            Whether both wrappers share type and prepared state.
        """
        return type(self) is type(other) and self._prepared is other._prepared

    def _new(
        self,
        tensor: torch.Tensor,
        prepared: Any,
        **logical_metadata: Any,
    ) -> _FSDPPreparedWeight:
        """Construct the backend-specific wrapper for one lifecycle state.

        Args:
            tensor: High-precision storage or prepared-state metadata anchor.
            prepared: Backend-specific prepared state, or ``None`` while sharded.
            **logical_metadata: Optional logical tensor metadata overrides.

        Returns:
            Backend-specific prepared-weight wrapper.
        """
        raise NotImplementedError

    def _rewrap(self, tensor: torch.Tensor):
        """Preserve the current state across an FSDP tensor operation."""
        return self._new(tensor, self._prepared)

    def _rewrap_prepared(self, tensor: torch.Tensor, prepared: Any):
        """Create a storage-free wrapper after weight preparation."""
        return self._new(tensor, prepared)

    def _rewrap_prepared_view(self, tensor: torch.Tensor):
        """Preserve prepared storage while applying logical view metadata."""
        assert self._prepared is not None
        anchor = self._prepared_tensors(self._prepared)[0]
        return self._new(
            anchor,
            self._prepared,
            _logical_size=tensor.size(),
            _logical_stride=tensor.stride(),
            _logical_storage_offset=tensor.storage_offset(),
            _logical_dtype=self.dtype,
            _logical_device=self.device,
            _logical_requires_grad=tensor.requires_grad,
        )

    def _prepare(self, weight: torch.Tensor, out: Any = None) -> Any:
        """Prepare backend-specific compute operands.

        Args:
            weight: Logical unsharded high-precision weight.
            out: Existing prepared storage to refill.

        Returns:
            Backend-specific prepared state.
        """
        raise NotImplementedError

    def _prepared_tensors(self, prepared: Any) -> tuple[torch.Tensor, ...]:
        """Return independently allocated prepared tensors.

        Args:
            prepared: Backend-specific prepared state.

        Returns:
            Tensors whose storage follows the FSDP unshard lifetime.
        """
        raise NotImplementedError

    def _prepare_unsharded_impl(self, weight: torch.Tensor):
        """Prepare a storage-free parameter without adding an autograd edge."""
        local_weight = weight._local_tensor if isinstance(weight, DTensor) else weight
        source = local_weight
        if isinstance(local_weight, _FSDPPreparedWeight):
            if local_weight._tensor is None:
                raise RuntimeError("Cannot prepare an already prepared FSDP weight")
            source = local_weight._tensor
        with torch.no_grad():
            prepared = self._prepare(source)
        prepared_local = self._rewrap_prepared(local_weight, prepared)
        if not isinstance(weight, DTensor):
            return prepared_local
        return DTensor.from_local(
            prepared_local,
            weight.device_mesh,
            weight.placements,
            run_check=False,
            shape=weight.shape,
            stride=weight.stride(),
        )

    def prepare_unsharded(self, weight: torch.Tensor):
        """Prepare a storage-free parameter from a replicated weight.

        Args:
            weight: Replicated high-precision parameter produced by SimpleFSDP.

        Returns:
            Backend-specific storage-free prepared weight.
        """
        return _PrepareFSDPWeight.apply(weight, self)

    def fsdp_pre_all_gather(
        self,
        mesh,
        outer_size,
        outer_stride,
        module,
        mp_policy,
    ):
        """Return a padded high-precision communication tensor.

        Args:
            mesh: FSDP shard mesh.
            outer_size: Logical unsharded parameter shape.
            outer_stride: Logical contiguous parameter stride.
            module: Parameter-owning module.
            mp_policy: FSDP mixed-precision policy.

        Returns:
            One padded all-gather input and the logical unsharded shape.
        """
        del module
        if self._tensor is None:
            raise RuntimeError("Cannot all-gather a prepared FSDP weight")
        padded_rows = math.ceil(outer_size[0] / mesh.size())
        padded_size = torch.Size((padded_rows, *outer_size[1:]))
        source = self._tensor
        if source.size() != padded_size:
            if (
                source.ndim == 0
                or source.size()[1:] != padded_size[1:]
                or source.size(0) > padded_size[0]
            ):
                raise RuntimeError(
                    "FSDP prepared-weight shard is incompatible with its padded shape"
                )
            required_bytes = math.prod(padded_size) * source.element_size()
            if (
                source.storage_offset() == 0
                and source.untyped_storage().nbytes() >= required_bytes
            ):
                source = torch.as_strided(
                    source,
                    padded_size,
                    outer_stride,
                    storage_offset=0,
                )
            else:
                padded = source.new_zeros(padded_size)
                padded.narrow(0, 0, source.size(0)).copy_(source)
                source = padded
        dtype = mp_policy.param_dtype or source.dtype
        return (source.to(dtype),), outer_size

    def fsdp_post_all_gather(
        self,
        all_gather_outputs,
        metadata,
        param_dtype,
        *,
        out=None,
    ):
        """Create or refill prepared state after FSDP all-gather.

        Args:
            all_gather_outputs: Padded unsharded high-precision weight tuple.
            metadata: Logical unsharded shape from :meth:`fsdp_pre_all_gather`.
            param_dtype: FSDP compute parameter dtype.
            out: Existing allocation-stable unsharded parameter.

        Returns:
            A prepared wrapper and owned tensors on first unshard, or ``None``
            after refilling the existing tensors.
        """
        del param_dtype
        (weight,) = all_gather_outputs
        logical_size = torch.Size(metadata)
        if weight.size() != logical_size:
            if (
                weight.ndim == 0
                or weight.size()[1:] != logical_size[1:]
                or weight.size(0) < logical_size[0]
            ):
                raise RuntimeError(
                    "FSDP all-gather output does not contain the logical parameter"
                )
            weight = weight.narrow(0, 0, logical_size[0])
        if out is None:
            with torch.no_grad():
                prepared = self._prepare(weight)
            wrapper = self._rewrap_prepared(weight, prepared)
            return wrapper, self._prepared_tensors(prepared), True

        target = out._local_tensor if isinstance(out, DTensor) else out
        if not isinstance(target, type(self)) or target._prepared is None:
            raise RuntimeError("FSDP output does not own prepared tensors")
        previous_tensors = target._prepared_tensors(target._prepared)
        with torch.no_grad():
            refilled = target._prepare(weight, out=target._prepared)
        refilled_tensors = target._prepared_tensors(refilled)
        if len(previous_tensors) != len(refilled_tensors) or any(
            previous is not current
            for previous, current in zip(
                previous_tensors,
                refilled_tensors,
                strict=True,
            )
        ):
            raise RuntimeError("FSDP prepared-weight refill replaced owned storage")
        target._prepared = refilled

    def prepared_state(self) -> Any:
        """Return backend-specific state for the current unshard lifetime."""
        return self._prepared
