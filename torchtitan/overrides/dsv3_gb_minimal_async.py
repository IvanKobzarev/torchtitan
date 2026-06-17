# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GB200/GB300 DeepSeek-V3 overrides for scale validation.

Opt-in only. Use with::

    --override.imports torchtitan.overrides.dsv3_gb_minimal_async

This keeps the stock 671B config names while switching the relevant component
configs to the GB scale target:

- FlexAttention uses the flex_flash backend.
- Standard EP token dispatch is replaced with MinimalAsyncEP.
"""

from torchtitan.config import derive, override
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    MinimalAsyncEPTokenDispatcher,
)


@override(
    "dsv3_gb_flex_flash",
    target=FlexAttention.Config,
    exact=True,
    description="Use the FlexAttention FLASH backend on GB200/GB300.",
)
def dsv3_gb_flex_flash(cfg: FlexAttention.Config) -> FlexAttention.Config:
    kernel_options = dict(cfg.kernel_options)
    kernel_options["BACKEND"] = "FLASH"
    return derive(
        cfg,
        FlexAttention.Config,
        block_size=(256, 128),
        kernel_options=kernel_options,
    )


@override(
    "dsv3_gb_minimal_async_ep",
    target=AllToAllTokenDispatcher.Config,
    exact=True,
    description="Use MinimalAsyncEP for DeepSeek-V3 GB scale runs.",
)
def dsv3_gb_minimal_async_ep(
    cfg: AllToAllTokenDispatcher.Config,
) -> MinimalAsyncEPTokenDispatcher.Config:
    return derive(cfg, MinimalAsyncEPTokenDispatcher.Config)
