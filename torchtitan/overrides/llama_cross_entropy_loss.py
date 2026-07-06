# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override ChunkedLossWrapper with plain CrossEntropyLoss.

Opt-in only. Use with::

    --override.imports torchtitan.overrides.llama_cross_entropy_loss

This is useful for minimizing precompile/cudagraph failures against a simple
loss while keeping the rest of the selected model config unchanged.
"""

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config import override


@override(
    "llama_cross_entropy_loss",
    target=ChunkedLossWrapper.Config,
    fqns=["loss"],
    description="Use plain CrossEntropyLoss instead of ChunkedLossWrapper.",
)
def llama_cross_entropy_loss(
    cfg: ChunkedLossWrapper.Config,
) -> CrossEntropyLoss.Config:
    del cfg
    return CrossEntropyLoss.Config()
