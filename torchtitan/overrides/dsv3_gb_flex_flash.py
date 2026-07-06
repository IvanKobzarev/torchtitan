# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GB200/GB300 DeepSeek-V3 FlexAttention FLASH override.

Use this when the run needs the flex FLASH backend but must keep the configured
MoE token dispatcher, for example MXFP8 runs that require the padding-capable
standard dispatcher instead of MinimalAsyncEP.
"""

from torchtitan.config import derive, override
from torchtitan.models.common.attention import FlexAttention


@override(
    "dsv3_gb_flex_flash_only",
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
        block_size=(256, 256),
        kernel_options=kernel_options,
    )
