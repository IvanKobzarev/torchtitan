# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

from torch import nn

from torchtitan.distributed.fsdp import enable_fsdp_symm_mem


class _FakeProcessGroup:
    def __init__(self, degree: int) -> None:
        self._degree = degree

    def size(self) -> int:
        return self._degree


class _FakeParamGroup:
    def __init__(self, degree: int) -> None:
        self._all_gather_process_group = _FakeProcessGroup(degree)
        self.symm_mem_backend: str | None = None

    def set_symm_mem(self, backend: str = "NCCL") -> None:
        self.symm_mem_backend = backend


class _FakeFSDPState:
    __slots__ = ("_fsdp_param_groups",)

    def __init__(self, param_groups: list[_FakeParamGroup]) -> None:
        self._fsdp_param_groups = param_groups


class _FakeFSDPModule(nn.Module):
    """Stands in for an FSDPModule owning groups at the given degrees."""

    def __init__(self, *degrees: int) -> None:
        super().__init__()
        self.param_groups = [_FakeParamGroup(degree) for degree in degrees]
        self.force_sum_reduction = False
        self.module_symm_mem_calls = 0

    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        self.force_sum_reduction = enable

    def set_symm_mem_for_comm(self, backend: str = "NCCL") -> None:
        self.module_symm_mem_calls += 1

    def _get_fsdp_state(self) -> _FakeFSDPState:
        return _FakeFSDPState(self.param_groups)


def _build_model(*modules: _FakeFSDPModule) -> nn.Module:
    model = nn.Module()
    for index, module in enumerate(modules):
        model.add_module(f"unit{index}", module)
    # A module FSDP never wrapped, to check it is left alone.
    model.add_module("unwrapped", nn.Linear(4, 4))
    return model


class TestFSDPSymmMemPolicy(unittest.TestCase):
    """Which FSDP parameter groups symmetric memory is applied to."""

    def setUp(self) -> None:
        patcher = mock.patch("torchtitan.distributed.fsdp.FSDPModule", _FakeFSDPModule)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_all_uses_the_module_level_setter(self) -> None:
        dense, experts = _FakeFSDPModule(256), _FakeFSDPModule(4)
        enable_fsdp_symm_mem(_build_model(dense, experts), "all")

        for module in (dense, experts):
            self.assertEqual(module.module_symm_mem_calls, 1)
            self.assertTrue(module.force_sum_reduction)
            for group in module.param_groups:
                self.assertIsNone(group.symm_mem_backend)

    def test_widest_skips_narrower_groups(self) -> None:
        dense, experts = _FakeFSDPModule(256), _FakeFSDPModule(4)
        enable_fsdp_symm_mem(_build_model(dense, experts), "widest")

        self.assertEqual(dense.param_groups[0].symm_mem_backend, "NCCL")
        self.assertIsNone(experts.param_groups[0].symm_mem_backend)
        # The module-level setter would have covered every group, so the
        # widest policy must not reach for it.
        self.assertEqual(dense.module_symm_mem_calls, 0)
        self.assertEqual(experts.module_symm_mem_calls, 0)

    def test_widest_still_forces_sum_reduction_everywhere(self) -> None:
        dense, experts = _FakeFSDPModule(256), _FakeFSDPModule(4)
        enable_fsdp_symm_mem(_build_model(dense, experts), "widest")

        # Gradient scaling must not depend on the policy.
        self.assertTrue(dense.force_sum_reduction)
        self.assertTrue(experts.force_sum_reduction)

    def test_widest_selects_per_group_not_per_module(self) -> None:
        mixed = _FakeFSDPModule(256, 4)
        enable_fsdp_symm_mem(_build_model(mixed), "widest")

        self.assertEqual(mixed.param_groups[0].symm_mem_backend, "NCCL")
        self.assertIsNone(mixed.param_groups[1].symm_mem_backend)

    def test_widest_covers_everything_at_a_single_degree(self) -> None:
        # Without expert parallelism every group shares one degree, so the two
        # policies must select the same set.
        first, second = _FakeFSDPModule(256), _FakeFSDPModule(256)
        enable_fsdp_symm_mem(_build_model(first, second), "widest")

        for module in (first, second):
            self.assertEqual(module.param_groups[0].symm_mem_backend, "NCCL")

    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            enable_fsdp_symm_mem(_build_model(_FakeFSDPModule(256)), "dense")


if __name__ == "__main__":
    unittest.main()
