# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GB200/GB300 DeepSeek-V3 MinimalAsyncEP override without flex FLASH.

Opt-in only. Use with::

    --override.imports torchtitan.overrides.dsv3_gb_minimal_async_no_flash

This preserves the configured attention backend while replacing standard EP
token dispatch with MinimalAsyncEP for GB scale validation.
"""

from torchtitan.config import derive, override
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    MinimalAsyncEPTokenDispatcher,
)


@override(
    "dsv3_gb_minimal_async_ep_no_flash",
    target=AllToAllTokenDispatcher.Config,
    exact=True,
    description=(
        "Use MinimalAsyncEP for DeepSeek-V3 GB scale runs without changing "
        "attention backend."
    ),
)
def dsv3_gb_minimal_async_ep_no_flash(
    cfg: AllToAllTokenDispatcher.Config,
) -> MinimalAsyncEPTokenDispatcher.Config:
    return derive(cfg, MinimalAsyncEPTokenDispatcher.Config)
