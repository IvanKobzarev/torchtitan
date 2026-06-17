# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest

import torch

from tests.utils import hash_gradient, hash_model


class TestHashUtils(unittest.TestCase):
    def test_hash_model_supports_bfloat16(self):
        model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]],
                    dtype=torch.bfloat16,
                )
            )

        before = hash_model(model)
        with torch.no_grad():
            model.weight[0, 0] = torch.tensor(5.0, dtype=torch.bfloat16)
        after = hash_model(model)

        self.assertNotEqual(before, after)
        self.assertIn("weight", json.loads(hash_model(model, per_tensor=True)))

    def test_hash_gradient_supports_bfloat16(self):
        model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        model.weight.grad = torch.ones_like(model.weight)

        hashes = json.loads(hash_gradient(model, per_tensor=True))

        self.assertIn("weight.grad", hashes)

    def test_hash_model_supports_bfloat16_scalar_buffer(self):
        model = torch.nn.Module()
        model.register_buffer("scale", torch.tensor(1.5, dtype=torch.bfloat16))

        hashes = json.loads(hash_model(model, per_tensor=True))

        self.assertIn("scale", hashes)

    def test_checkpoint_wrapped_module_name_is_normalized(self):
        model = torch.nn.Module()
        model.layer = torch.nn.Module()
        model.layer._checkpoint_wrapped_module = torch.nn.Linear(2, 2, bias=False)

        hashes = json.loads(hash_model(model, per_tensor=True))

        self.assertIn("layer.weight", hashes)
        self.assertNotIn("layer._checkpoint_wrapped_module.weight", hashes)


if __name__ == "__main__":
    unittest.main()
