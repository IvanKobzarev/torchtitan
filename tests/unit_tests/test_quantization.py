# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import weakref
from types import SimpleNamespace

import pytest
import spmd_types as spmd
import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.components.quantization import Float8Linear
from torchtitan.components.quantization.float8 import _get_float8_grouped_experts_cls
from torchtitan.components.quantization.mx import _get_mxfp8_grouped_experts_cls
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import ConfigManager
from torchtitan.models.common.decoder_sharding import colwise_config, rowwise_config
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts


def test_no_float8_by_default():
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    model_config = config.model_spec.model
    assert not has_quantization(model_config)
    # All Linear.Config instances should remain Linear.Config
    if Float8Linear is not None:
        for _fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
            assert not isinstance(lc, Float8Linear.Config)


def test_float8_applied_by_model_registry():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_float8_emulate_lora"]
    )
    model_config = config.model_spec.model
    assert has_quantization(model_config)
    # Some Linear.Config instances should be swapped to Float8Linear
    converted = [
        fqn
        for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if isinstance(lc, Float8Linear.Config)
    ]
    assert len(converted) > 0
    lora_converted = {
        fqn
        for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if hasattr(lc, "rank") and hasattr(lc, "alpha")
    }
    assert lora_converted == {
        f"layers.{layer}.attention.{projection}"
        for layer in range(6)
        for projection in ("qkv_linear.wqkv", "wo")
    }


@pytest.mark.parametrize(
    "module, recipe, expected_num_layers",
    [
        ("llama3", "llama3_debugmodel_nvfp4", 6),
        ("qwen3", "qwen3_debugmodel_nvfp4", 8),
    ],
)
def test_nvfp4_converter_targets_layers_not_lm_head(
    monkeypatch, module, recipe, expected_num_layers
):
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    # Exercise convert() targeting independent of GPU: bypass the sm100 gate
    # that NVFP4LinearConverter.__init__ enforces (hardware is irrelevant to the
    # config-tree transform under test).
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)

    config_manager = ConfigManager()
    config = config_manager.parse_args(["--module", module, "--config", recipe])
    model_config = config.model_spec.model
    assert has_quantization(model_config)

    converted, stock = [], []
    for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
        (converted if isinstance(lc, NVFP4Linear.Config) else stock).append(fqn)

    # Every in-layer linear is swapped; the lm_head stays stock (NVFP4 requires
    # each GEMM dim divisible by 128, which the vocab projection violates).
    assert converted and all("layers" in fqn for fqn in converted)
    assert {int(fqn.split(".")[1]) for fqn in converted} == set(
        range(expected_num_layers)
    )
    assert stock == ["lm_head"]


def test_nvfp4_bf16_tail_fqns():
    from torchtitan.components.quantization.nvfp4 import nvfp4_bf16_tail_fqns

    # 32 layers, 15% tail -> ceil(4.8)=5 bf16, convert layers 0..26.
    fqns = nvfp4_bf16_tail_fqns(32, 0.15)
    assert fqns == [f"layers.{i}." for i in range(27)]
    # Every fqn is trailing-dot anchored so "layers.2." matches layer 2 only,
    # not "layers.20".."layers.29" (the converter substring-matches).
    assert all(f.startswith("layers.") and f.endswith(".") for f in fqns)
    # Fraction 0 keeps nothing in bf16 -> every layer converted.
    assert nvfp4_bf16_tail_fqns(4, 0.0) == [
        "layers.0.",
        "layers.1.",
        "layers.2.",
        "layers.3.",
    ]
    # A fraction that rounds up to all layers leaves nothing to convert -> raise
    # (an empty fqns list would instead convert *all* Linears).
    with pytest.raises(ValueError, match="nothing to convert"):
        nvfp4_bf16_tail_fqns(4, 1.0)


@pytest.mark.parametrize(
    "module, recipe, expected_cutoff",
    [
        ("llama3", "llama3_debugmodel_first_85_pct_layers_nvfp4", 5),
        ("llama3", "llama3_8b_first_85_pct_layers_nvfp4", 27),
        ("qwen3", "qwen3_debugmodel_first_85_pct_layers_nvfp4", 6),
        ("qwen3", "qwen3_8b_first_85_pct_layers_nvfp4", 30),
    ],
)
def test_nvfp4_first_85_pct_layers_converts_only_leading_layers(
    monkeypatch, module, recipe, expected_cutoff
):
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    import math

    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)

    config = ConfigManager().parse_args(["--module", module, "--config", recipe])
    model_config = config.model_spec.model
    n_layers = len(model_config.layers)
    cutoff = n_layers - math.ceil(n_layers * 0.15)
    assert cutoff == expected_cutoff
    assert 0 < cutoff < n_layers  # a real split: some NVFP4, some bf16

    converted_layers, stock = set(), []
    for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
        if isinstance(lc, NVFP4Linear.Config):
            converted_layers.add(int(fqn.split(".")[1]))
        else:
            stock.append(fqn)

    # Only the leading layers are NVFP4; the bf16 tail + lm_head stay stock.
    assert converted_layers == set(range(cutoff))
    assert "lm_head" in stock
    assert all(
        not fqn.startswith("layers.") or int(fqn.split(".")[1]) >= cutoff
        for fqn in stock
    )


def _nvfp4_linear_cls():
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    return NVFP4Linear


@pytest.mark.parametrize("in_features, out_features", [(512, 300), (300, 512)])
def test_nvfp4_config_rejects_non_128_dims(in_features, out_features):
    # The model dims are known at config-build time, so a non-128 in/out_features
    # (e.g. the LM head) is rejected in Config.__post_init__ before any TP.
    NVFP4Linear = _nvfp4_linear_cls()
    with pytest.raises(ValueError, match="divisible by 128"):
        NVFP4Linear.Config(in_features=in_features, out_features=out_features)


@pytest.mark.parametrize(
    "sharding_config_factory, input_tp, input_grad_tp",
    [
        pytest.param(lambda: colwise_config(), spmd.R, spmd.P, id="colwise"),
        pytest.param(
            lambda: rowwise_config(output_sp=True),
            spmd.S(-1),
            spmd.S(-1),
            id="rowwise",
        ),
    ],
)
def test_nvfp4_build_configures_local_spmd_sharding(
    sharding_config_factory, input_tp, input_grad_tp
):
    # Config.build() folds the stock colwise/rowwise sharding into the local
    # SPMD region for the opaque NVFP4 GEMM.
    NVFP4Linear = _nvfp4_linear_cls()
    from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
    from torchtitan.models.common.decoder_sharding import dense_activation_placement

    module = NVFP4Linear.Config(
        in_features=512,
        out_features=1024,
        sharding_config=sharding_config_factory(),
    ).build()
    sc = module._sharding_config
    assert sc.local_map is not None
    input_layout = dense_activation_placement(tp=input_tp, cp=spmd.S(0))
    assert sc.in_src_shardings == {"x": input_layout}
    assert sc.in_dst_shardings == {"x": input_layout}
    assert sc.local_map.in_grad_placements == (
        dense_activation_placement(tp=input_grad_tp, cp=spmd.S(0)),
    )
    assert "weight" in sc.state_shardings
    assert sc.state_shardings["_sr_seed"] == SpmdLayout(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.CP: spmd.V,
            MeshAxisName.TP: spmd.V,
        }
    )


@pytest.mark.parametrize(
    "module, recipe",
    [
        ("llama3", "llama3_debugmodel_nvfp4"),
        ("llama3", "llama3_debugmodel_first_85_pct_layers_nvfp4"),
        ("llama3", "llama3_8b_first_85_pct_layers_nvfp4"),
        ("qwen3", "qwen3_debugmodel_nvfp4"),
        ("qwen3", "qwen3_debugmodel_first_85_pct_layers_nvfp4"),
        ("qwen3", "qwen3_8b_first_85_pct_layers_nvfp4"),
    ],
)
def test_nvfp4_recipes_default_to_spmd_types_and_allow_cli_override(
    monkeypatch, module, recipe
):
    _nvfp4_linear_cls()
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)
    base_args = ["--module", module, "--config", recipe]

    config = ConfigManager().parse_args(base_args)
    assert config.parallelism.spmd_backend == "spmd_types"

    overridden = ConfigManager().parse_args(
        [*base_args, "--parallelism.spmd_backend", "partial_dtensor"]
    )
    assert overridden.parallelism.spmd_backend == "partial_dtensor"


@pytest.mark.parametrize(
    "recipe",
    [
        "qwen3_debugmodel_nvfp4",
        "qwen3_debugmodel_first_85_pct_layers_nvfp4",
        "qwen3_8b_first_85_pct_layers_nvfp4",
    ],
)
def test_qwen3_recipes_resolve(monkeypatch, recipe):
    _nvfp4_linear_cls()
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)
    config = ConfigManager().parse_args(["--module", "qwen3", "--config", recipe])
    assert config.model_spec.name == "qwen3"
    if recipe == "qwen3_8b_first_85_pct_layers_nvfp4":
        assert isinstance(config.dataloader, GrainDataLoader.Config)
        packed_dataset = config.dataloader.dataset
        assert isinstance(packed_dataset, FirstFitPackingConfig)
        dataset = packed_dataset.dataset
        assert isinstance(dataset, SingleDatasetConfig)
        assert isinstance(dataset.source, HuggingFaceRandomAccessSource.Config)
        assert dataset.source.path == "openai/gsm8k"
        assert config.checkpoint.initial_load_in_hf
        assert config.compile.enable
        assert "model" in config.compile.components


def test_nvfp4_module_buffers_and_native_checkpoint():
    """Built module has the stock weight param plus the two NVFP4 runtime
    buffers, and both buffers are non-persistent -- the RHT vector is a fixed
    constant and the SR seed is per-rank -- so a native checkpoint carries only
    the stock weight."""
    NVFP4Linear = _nvfp4_linear_cls()
    from torchtitan.components.quantization.nvfp4 import _HARDCODED_SIGN_VECTOR

    module = NVFP4Linear.Config(in_features=512, out_features=1024).build()
    assert {name for name, _ in module.named_parameters()} == {"weight"}
    module.init_states()
    buffers = dict(module.named_buffers())
    assert set(buffers) == {"_sr_seed", "_rht_sign_vector"}
    assert buffers["_sr_seed"].dtype == torch.int64
    assert tuple(buffers["_rht_sign_vector"].shape) == (16,)
    # The RHT vector is the fixed v1-recipe constant, identical on every rank.
    assert tuple(int(v) for v in buffers["_rht_sign_vector"]) == _HARDCODED_SIGN_VECTOR
    # Both runtime buffers are non-persistent, so a native checkpoint carries
    # only the stock weight.
    assert set(module.state_dict()) == {"weight"}


def test_nvfp4_stock_checkpoint_loads_before_init_states():
    """A stock bf16 checkpoint (no NVFP4 buffers) loads; buffers stay unmaterialized
    until init_states creates them."""
    NVFP4Linear = _nvfp4_linear_cls()
    stock = Linear.Config(in_features=512, out_features=1024).build()
    nvfp4 = NVFP4Linear.Config(in_features=512, out_features=1024).build()

    nvfp4.load_state_dict(stock.state_dict(), strict=False)
    assert nvfp4._rht_sign_vector is None
    assert nvfp4._rht_sign_vector_tuple is None

    nvfp4.init_states()
    assert nvfp4._rht_sign_vector is not None
    assert nvfp4._rht_sign_vector_tuple is not None


def test_nvfp4_hf_export_strips_buffers(monkeypatch):
    """The HF export boundary contains only stock keys -- no NVFP4 runtime buffers."""
    NVFP4Linear = _nvfp4_linear_cls()
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

    config = ConfigManager().parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_nvfp4"]
    )
    model_config = config.model_spec.model
    model = model_config.build()
    model.init_states()
    assert isinstance(model.get_submodule("layers.0.feed_forward.w1"), NVFP4Linear)

    sd = model.state_dict()
    # Both NVFP4 runtime buffers are non-persistent, so neither the RHT vector
    # nor the per-rank SR seed appears in the native state dict.
    assert not any("_rht_sign_vector" in k for k in sd)
    assert not any("_sr_seed" in k for k in sd)

    hf_sd = Llama3StateDictAdapter(model_config, hf_assets_path=None).to_hf(sd)
    assert "model.layers.0.mlp.gate_proj.weight" in hf_sd
    assert not any("_rht_sign_vector" in k for k in hf_sd)


def test_quantized_grouped_experts():
    """Quantized GroupedExperts: _owner, subclass handling, extra config fields."""
    # Base case
    MXFP8GroupedExperts = _get_mxfp8_grouped_experts_cls(GroupedExperts)
    Float8GroupedExperts = _get_float8_grouped_experts_cls(GroupedExperts)

    assert MXFP8GroupedExperts.Config._owner is MXFP8GroupedExperts
    assert Float8GroupedExperts.Config._owner is Float8GroupedExperts

    # Subclass case (GptOssGroupedExperts has extra swiglu_limit field)
    mxfp8_cls = _get_mxfp8_grouped_experts_cls(GptOssGroupedExperts)
    float8_cls = _get_float8_grouped_experts_cls(GptOssGroupedExperts)

    assert mxfp8_cls.Config._owner is mxfp8_cls
    assert float8_cls.Config._owner is float8_cls
    assert issubclass(mxfp8_cls, GptOssGroupedExperts)
    assert issubclass(float8_cls, GptOssGroupedExperts)
    assert hasattr(mxfp8_cls.Config, "swiglu_limit")
    assert hasattr(float8_cls.Config, "swiglu_limit")


def test_mxfp8_fsdp_configuration_preserves_parameter_identity():
    """Prepared weights preserve parameter identity and checkpoint loading."""
    pytest.importorskip("torchao")
    from torchtitan.components.quantization.mx import MXFP8Linear

    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8 training prototype not available")
    module = MXFP8Linear.Config(in_features=64, out_features=64).build()
    weight = module.weight

    module.configure_fsdp()

    assert module.weight is weight
    assert weight in set(module.parameters())
    expected = torch.randn_like(module.weight)
    module.load_state_dict({"weight": expected})
    assert torch.equal(module.weight._tensor, expected)


def test_mxfp8_prepared_weight_tensor_flatten_round_trip():
    """Prepared-state serialization preserves logical metadata and operands."""
    pytest.importorskip("torchao")
    from torchtitan.components.quantization.mx import (
        _MXFP8FSDPWeight,
        _MXFP8PreparedWeight,
    )

    qdata = torch.empty(64, 64, dtype=torch.float8_e4m3fn)
    fprop_scale = torch.empty(4, dtype=torch.float8_e8m0fnu)
    dgrad_scale = torch.empty(4, dtype=torch.float8_e8m0fnu)
    prepared = _MXFP8PreparedWeight(qdata, fprop_scale, dgrad_scale)
    weight = _MXFP8FSDPWeight(
        qdata,
        prepared,
        _logical_size=(64, 64),
        _logical_stride=(64, 1),
        _logical_dtype=torch.bfloat16,
    )

    names, metadata = weight.__tensor_flatten__()
    restored = weight.__tensor_unflatten__(
        {name: getattr(weight, name) for name in names},
        metadata,
        weight.size(),
        weight.stride(),
    )

    assert restored.size() == weight.size()
    assert restored.stride() == weight.stride()
    assert restored.dtype == torch.bfloat16
    restored_prepared = restored.prepared_operands()
    assert restored_prepared is not None
    assert restored_prepared.qdata is qdata
    assert restored_prepared.fprop_scale is fprop_scale
    assert restored_prepared.dgrad_scale is dgrad_scale


def test_mxfp8_weight_gather_rejects_unimplemented_format():
    """Configuration must not silently treat MXFP8 parameter gather as BF16."""
    pytest.importorskip("torchao")
    from torchtitan.components.quantization.mx import MXFP8Linear

    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8 training prototype not available")
    with pytest.raises(ValueError, match="parameter all-gather is not implemented"):
        MXFP8Linear.Config(
            in_features=64,
            out_features=64,
            weight_gather="mxfp8",
        ).build()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 scaled GEMM requires SM100 or later",
)
def test_explicit_mxfp8_linear_matches_mxtensor_operands_bitwise():
    """Explicit qdata/scales must preserve the previous GEMM bytes."""
    from torchao.prototype.mx_formats.kernels import (
        mxfp8_quantize_cuda,
        triton_to_mxfp8_32x32_swizzle_dim0_and_dim1,
        triton_to_mxfp8_dim0,
    )
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    from torchtitan.components.quantization.mx import _MXFP8LinearFunction

    torch.manual_seed(0)
    input_hp = torch.randn(
        64,
        256,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    weight_hp = torch.randn(
        512,
        256,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    grad_output = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
    (
        weight_qdata,
        weight_fprop_scale,
        weight_dgrad_scale,
    ) = triton_to_mxfp8_32x32_swizzle_dim0_and_dim1(weight_hp.detach())

    output = _MXFP8LinearFunction.apply(
        input_hp,
        weight_hp,
        weight_qdata,
        weight_fprop_scale,
        weight_dgrad_scale,
        None,
        False,
    )
    output.backward(grad_output)

    input_row, input_col, input_row_scale, input_col_scale = mxfp8_quantize_cuda(
        input_hp.detach(),
        rowwise=True,
        colwise=True,
        scaling_mode="rceil",
    )
    grad_row, grad_row_scale = triton_to_mxfp8_dim0(
        grad_output,
        scaling_mode="rceil",
    )
    _, grad_col, _, grad_col_scale = mxfp8_quantize_cuda(
        grad_output,
        rowwise=False,
        colwise=True,
        scaling_mode="rceil",
    )
    common = (
        torch.float8_e4m3fn,
        32,
        torch.bfloat16,
        KernelPreference.AUTO,
        None,
    )
    input_fprop = MXTensor(input_row, input_row_scale, *common, False)
    input_wgrad = MXTensor(input_col.t(), input_col_scale, *common, False)
    grad_dgrad = MXTensor(grad_row, grad_row_scale, *common, False)
    grad_wgrad = MXTensor(grad_col.t(), grad_col_scale, *common, False)
    weight_fprop = MXTensor(
        weight_qdata,
        weight_fprop_scale.flatten(),
        *common,
        True,
    )
    weight_dgrad = MXTensor(
        weight_qdata.t().contiguous(),
        weight_dgrad_scale.flatten(),
        *common,
        True,
    )

    expected_output = torch.mm(input_fprop, weight_fprop.t())
    expected_dgrad = torch.mm(grad_dgrad, weight_dgrad.t())
    expected_wgrad = torch.mm(grad_wgrad, input_wgrad.t())
    assert torch.equal(output, expected_output)
    assert torch.equal(input_hp.grad, expected_dgrad)
    assert torch.equal(weight_hp.grad, expected_wgrad)


def test_explicit_mxfp8_linear_fake_forward_backward():
    """FakeTensor must trace the explicit FPROP, DGRAD, and WGRAD contract."""
    pytest.importorskip("torchao")
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torchao.prototype.mx_formats.kernels import (
        triton_to_mxfp8_32x32_swizzle_dim0_and_dim1,
    )
    from torchtitan.components.quantization.mx import _MXFP8LinearFunction

    with FakeTensorMode():
        input_hp = torch.empty(
            64,
            256,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        weight_hp = torch.empty(
            512,
            256,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        prepared = triton_to_mxfp8_32x32_swizzle_dim0_and_dim1(weight_hp.detach())
        output = _MXFP8LinearFunction.apply(
            input_hp,
            weight_hp,
            *prepared,
            None,
            False,
        )
        grad_input, grad_weight = torch.autograd.grad(
            output.sum(),
            (input_hp, weight_hp),
        )

    assert output.shape == (64, 512)
    assert grad_input.shape == input_hp.shape
    assert grad_weight.shape == weight_hp.shape


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 scaled GEMM requires SM100 or later",
)
def test_explicit_mxfp8_linear_accumulates_wgrad_in_parameter_storage():
    """Fused WGRAD accumulation reuses the autograd-owned gradient buffer."""
    from torchao.prototype.mx_formats.kernels import (
        triton_to_mxfp8_32x32_swizzle_dim0_and_dim1,
    )
    from torchtitan.components.quantization.mx import _MXFP8LinearFunction

    torch.manual_seed(0)
    weight = torch.nn.Parameter(
        torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    )
    prepared = triton_to_mxfp8_32x32_swizzle_dim0_and_dim1(weight.detach())
    inputs = [
        torch.randn(64, 256, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    output_grads = [
        torch.randn(64, 256, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]

    reference = torch.nn.Parameter(weight.detach().clone())
    for input_hp, grad_output in zip(inputs, output_grads, strict=True):
        output = _MXFP8LinearFunction.apply(
            input_hp,
            reference,
            *prepared,
            None,
            False,
        )
        output.backward(grad_output)

    pointers = []
    for input_hp, grad_output in zip(inputs, output_grads, strict=True):
        output = _MXFP8LinearFunction.apply(
            input_hp,
            weight,
            *prepared,
            weakref.ref(weight),
            True,
        )
        output.backward(grad_output)
        assert weight.grad is not None
        pointers.append(weight.grad.data_ptr())

    assert len(set(pointers)) == 1
    torch.testing.assert_close(weight.grad, reference.grad, rtol=1e-2, atol=2e-3)


def test_prepared_weight_uses_padded_shard_and_logical_gather_view():
    """Uneven FSDP shards communicate padding but never prepare padded rows."""
    from torchtitan.distributed._prepared_weight import _FSDPPreparedWeight

    class CopyPreparedWeight(_FSDPPreparedWeight):
        """Test carrier whose prepared representation is an owned clone."""

        def _new(self, tensor, prepared, **logical_metadata):
            """Construct a carrier with the requested lifecycle state."""
            return CopyPreparedWeight(tensor, prepared, **logical_metadata)

        def _prepare(self, weight, out=None):
            """Clone or refill the logical unsharded value."""
            if out is None:
                return weight.clone()
            out.copy_(weight)
            return out

        def _prepared_tensors(self, prepared):
            """Return the single owned prepared tensor."""
            return (prepared,)

    padded_storage = torch.zeros(3, 2, dtype=torch.bfloat16)
    padded_storage[:2].copy_(torch.arange(4).reshape(2, 2))
    wrapper = CopyPreparedWeight(padded_storage[:2])
    mesh = SimpleNamespace(size=lambda: 2)
    policy = SimpleNamespace(param_dtype=torch.bfloat16)

    (all_gather_input,), metadata = wrapper.fsdp_pre_all_gather(
        mesh,
        torch.Size((5, 2)),
        (2, 1),
        None,
        policy,
    )
    assert all_gather_input.shape == (3, 2)
    assert torch.equal(all_gather_input[-1], torch.zeros(2, dtype=torch.bfloat16))

    gathered = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    prepared_wrapper, prepared_tensors, release = wrapper.fsdp_post_all_gather(
        (gathered,),
        metadata,
        torch.bfloat16,
    )
    assert prepared_wrapper.shape == (5, 2)
    assert prepared_tensors[0].shape == (5, 2)
    assert torch.equal(prepared_tensors[0], gathered[:5])
    assert release

    prepared_storage = prepared_tensors[0]
    updated = gathered.add(10)
    assert (
        wrapper.fsdp_post_all_gather(
            (updated,),
            metadata,
            torch.bfloat16,
            out=prepared_wrapper,
        )
        is None
    )
    assert prepared_wrapper.prepared_state() is prepared_storage
    assert torch.equal(prepared_storage, updated[:5])


@pytest.mark.parametrize("parent_cls", [GroupedExperts, GptOssGroupedExperts])
def test_float8_grouped_experts_checkpoint_state_uses_plain_tensors(parent_cls):
    pytest.importorskip("torchao")
    stock = parent_cls.Config(dim=16, hidden_dim=32, num_experts=2).build()
    float8_cls = _get_float8_grouped_experts_cls(parent_cls)
    module = float8_cls.Config(dim=16, hidden_dim=32, num_experts=2).build()

    assert all(type(param) is torch.nn.Parameter for param in module.parameters())
    stock_state = stock.state_dict()
    float8_state = module.state_dict()
    assert float8_state.keys() == stock_state.keys()
    for key, value in float8_state.items():
        assert type(value) is torch.Tensor
        assert value.shape == stock_state[key].shape
        assert value.dtype == stock_state[key].dtype


@pytest.mark.filterwarnings("ignore:torch.distributed is disabled")
def test_float8_grouped_experts_dcp_round_trip_needs_no_safe_globals(tmp_path):
    pytest.importorskip("torchao")
    float8_cls = _get_float8_grouped_experts_cls(GroupedExperts)
    config = float8_cls.Config(dim=16, hidden_dim=32, num_experts=2)
    source = config.build()
    target = config.build()

    with torch.no_grad():
        for value, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(value)
        for parameter in target.parameters():
            parameter.zero_()

    saved_safe_globals = torch.serialization.get_safe_globals()
    try:
        torch.serialization.clear_safe_globals()
        dcp.save(source.state_dict(), checkpoint_id=tmp_path, no_dist=True)
        dcp.load(target.state_dict(), checkpoint_id=tmp_path, no_dist=True)
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(saved_safe_globals)

    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        torch.testing.assert_close(target_parameter, source_parameter)
