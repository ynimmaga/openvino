# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify the OV lm_head compiles from the checkpoint's int8 weight when it can.

The nncf_hooks loader stashes a per-row int8 lm_head (or the embedding it is
tied to) on `model._ov_nncf_int8_weights`; _install_ov_lm_head hands it to
build_ov_lm_head only if it still decompresses to the live lm_head weight,
and on GPU builds the OV lm_head by default. This checks the match, the stale
fallback, that the stash is dropped, the per-device default, and (when a GPU
is present) the compressed GPU build itself.

No model download needed: a minimal stand-in model carries the weights.
"""

import os

import pytest

os.environ.setdefault("VLLM_USE_LAYERNAME", "0")

import torch

from openvino.frontend.pytorch.torchdynamo.vllm import lm_head as lm_head_mod
from openvino.frontend.pytorch.torchdynamo.vllm import plugin
from openvino.frontend.pytorch.torchdynamo.vllm.model_loader import _dequantize


class _FakeLMHead:
    def __init__(self, weight):
        self.weight = weight
        self.cpu_linear = None


class _FakeModel:
    def __init__(self, weight, stash, device=None):
        self.lm_head = _FakeLMHead(weight)
        self._ov_nncf_int8_weights = stash
        if device is not None:
            self._ov_compile_options = {"device": device}


def _int8_checkpoint(V, H, dtype):
    """NNCF per-row int8 asym tensors and the dense weight the loader yields from them."""
    stored = torch.randint(0, 256, (V, H), dtype=torch.uint8)
    zero_point = torch.randint(100, 156, (V, 1), dtype=torch.uint8)
    scale = (torch.rand(V, 1) * 0.01 + 0.001).to(torch.float16)
    dense = _dequantize(stored, scale, zero_point, dtype)
    return (stored, scale, zero_point), torch.nn.Parameter(dense, requires_grad=False)


@pytest.fixture
def build_spy(monkeypatch):
    """Record build_ov_lm_head's arguments, still building the real thing."""
    calls = []
    real = lm_head_mod.build_ov_lm_head

    def spy(weight, nthreads=None, device="CPU", int8=None):
        calls.append({"device": device, "int8": int8})
        return real(weight, nthreads=nthreads, device=device, int8=int8)

    monkeypatch.setattr(lm_head_mod, "build_ov_lm_head", spy)
    return calls


@pytest.mark.precommit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matching_int8_is_used_and_dropped(dtype, build_spy, monkeypatch):
    """A stash that decompresses to lm_head.weight is compiled int8, then freed."""
    monkeypatch.setenv("OV_LM_HEAD", "1")
    torch.manual_seed(0)
    int8, weight = _int8_checkpoint(64, 32, dtype)
    model = _FakeModel(weight, {"model.embed_tokens.weight": int8})

    plugin._install_ov_lm_head(model)

    assert len(build_spy) == 1 and build_spy[0]["int8"] is not None, "int8 stash was not used"
    assert model._ov_nncf_int8_weights == {}, "int8 stash was not dropped after install"
    x = torch.randn(3, 32, dtype=dtype)
    out = model.lm_head.cpu_linear(x, model.lm_head.weight, None)
    ref = x.float() @ weight.float().T
    assert out.dtype == dtype
    assert torch.allclose(out.float(), ref, rtol=0.02, atol=0.05), "int8 lm_head output does not match the dense weight"


@pytest.mark.precommit
def test_stale_int8_falls_back_to_dense(build_spy, monkeypatch):
    """After an in-place weight update the stash no longer matches and must be ignored."""
    monkeypatch.setenv("OV_LM_HEAD", "1")
    torch.manual_seed(1)
    int8, weight = _int8_checkpoint(64, 32, torch.float16)
    new = torch.randn(64, 32, dtype=torch.float16)
    weight.data.copy_(new)
    model = _FakeModel(weight, {"model.embed_tokens.weight": int8})

    plugin._install_ov_lm_head(model)

    assert build_spy[0]["int8"] is None, "a stale int8 stash was compiled into lm_head"
    x = torch.randn(2, 32, dtype=torch.float16)
    out = model.lm_head.cpu_linear(x, model.lm_head.weight, None)
    assert torch.allclose(out.float(), x.float() @ new.float().T, rtol=0.02, atol=0.05)


@pytest.mark.precommit
def test_stash_dropped_on_onednn_path(monkeypatch):
    """With the OV lm_head off the stash is unread, and must not stay resident."""
    monkeypatch.setenv("OV_LM_HEAD", "0")
    int8, weight = _int8_checkpoint(64, 32, torch.float16)
    model = _FakeModel(weight, {"model.embed_tokens.weight": int8})

    plugin._install_ov_lm_head(model)

    assert model._ov_nncf_int8_weights == {}


@pytest.mark.precommit
@pytest.mark.parametrize("device, expect_ov", [("CPU", False), ("GPU", True), ("GPU.0", True)])
def test_default_depends_on_device(device, expect_ov, monkeypatch):
    """Unset OV_LM_HEAD: OV lm_head on GPU devices, oneDNN on CPU."""
    monkeypatch.delenv("OV_LM_HEAD", raising=False)
    calls = []
    monkeypatch.setattr(lm_head_mod, "build_ov_lm_head",
                        lambda weight, nthreads=None, device="CPU", int8=None:
                        calls.append(device) or (lambda x, w, b: x))
    int8, weight = _int8_checkpoint(64, 32, torch.float16)
    model = _FakeModel(weight, {"model.embed_tokens.weight": int8}, device=device)

    plugin._install_ov_lm_head(model)

    assert calls == ([device] if expect_ov else [])


def _has_gpu():
    try:
        import openvino as ov
        return any(d.startswith("GPU") for d in ov.Core().available_devices)
    except Exception:
        return False


@pytest.mark.skipif(not _has_gpu(), reason="no OpenVINO GPU device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gpu_lm_head_is_int8_compressed(dtype):
    """On GPU the int8 weight stays int8 in device memory and gives the dense result."""
    import openvino as ov

    torch.manual_seed(2)
    V, H = 4096, 512
    int8, weight = _int8_checkpoint(V, H, dtype)
    core = ov.Core()
    before = sum(core.get_property("GPU", "GPU_MEMORY_STATISTICS").values())
    fn = lm_head_mod.build_ov_lm_head(weight.data, device="GPU", int8=int8)
    used = sum(core.get_property("GPU", "GPU_MEMORY_STATISTICS").values()) - before
    assert fn is not None
    assert used < V * H * 2 * 0.75, f"lm_head holds {used} bytes on GPU: weight was not kept int8"

    x = torch.randn(2, H, dtype=dtype)
    out = fn(x, weight, None)
    ref = x.float() @ weight.float().T
    assert out.dtype == dtype
    assert torch.allclose(out.float(), ref, rtol=0.02, atol=0.1)
