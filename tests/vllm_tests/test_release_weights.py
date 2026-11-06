# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify the torch weights an OV GPU compile baked in are handed back to the OS.

_release_baked_weights madvises their pages away but keeps each tensor's
object, shape and address, so dynamo's guards and the structural cache still
match; the patched openvino_compile then refuses any new compile that would
read them. This checks what is released and what is kept, the compile guard,
the reload escape hatch and the per-device default.

No model or GPU needed: a minimal stand-in model carries the weights.
"""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("VLLM_USE_LAYERNAME", "0")

import torch

from openvino.frontend.pytorch.torchdynamo import execute
from openvino.frontend.pytorch.torchdynamo.vllm import plugin

# Above _KEY_CONTENT_NUMEL, so keyed by address and eligible for release.
_BIG = (1024, 1024)


class _Model(torch.nn.Module):
    def __init__(self, tied_lm_head=False):
        super().__init__()
        self.proj = torch.nn.Linear(*_BIG, bias=False)
        self.small = torch.nn.Linear(8, 8, bias=False)
        self.lm_head = torch.nn.Linear(*_BIG, bias=False)
        if tied_lm_head:
            self.lm_head.weight = self.proj.weight
        for p in self.parameters():
            torch.nn.init.constant_(p, 1.0)

    def forward(self, x):
        return self.lm_head(self.proj(x))


def _ptr(t):
    return t.untyped_storage().data_ptr()


def _released(t):
    """Zeroed but for the partial pages at either end, which madvise must leave."""
    nonzero_bytes = int(t.count_nonzero()) * t.element_size()
    return nonzero_bytes <= 2 * os.sysconf("SC_PAGE_SIZE")


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(plugin, "_baked_storages", set())
    monkeypatch.setattr(plugin, "_released_storages", set())
    monkeypatch.setattr(plugin, "_superseded_storages", set())


def _bake(model):
    plugin._baked_storages.update(_ptr(p) for p in model.parameters())


@pytest.mark.precommit
def test_release_zeroes_baked_weights_in_place():
    model = _Model()
    model._ov_lm_head_on_ov = True
    _bake(model)
    before = {name: (_ptr(p), p.shape) for name, p in model.named_parameters()}

    plugin._release_baked_weights(model)

    for name, p in model.named_parameters():
        assert (_ptr(p), p.shape) == before[name], f"{name} moved or changed shape"
    assert _released(model.proj.weight) and _released(model.lm_head.weight)
    # Keyed by content in the structural cache, so it must keep its values.
    assert model.small.weight.eq(1.0).all()
    assert plugin._released_storages == {_ptr(model.proj.weight), _ptr(model.lm_head.weight)}


@pytest.mark.precommit
def test_lm_head_on_torch_keeps_its_weight():
    model = _Model(tied_lm_head=True)
    model._ov_lm_head_on_ov = False
    _bake(model)

    plugin._release_baked_weights(model)

    # Tied: the embedding-side Parameter is the one lm_head reads, so both stay.
    assert model.lm_head.weight.eq(1.0).all()
    assert not plugin._released_storages


@pytest.mark.precommit
def test_unbaked_weights_stay():
    model = _Model()
    model._ov_lm_head_on_ov = True
    plugin._baked_storages.add(_ptr(model.proj.weight))

    plugin._release_baked_weights(model)

    assert _released(model.proj.weight)
    assert model.lm_head.weight.eq(1.0).all()


@pytest.mark.precommit
def test_compile_after_release_is_refused(monkeypatch):
    compiled = []
    monkeypatch.setattr(execute, "openvino_compile", lambda gm, *args, **kw: compiled.append(gm) or "ov")
    monkeypatch.setattr(execute, "_ov_plugin_compile_patched", False, raising=False)
    plugin._patch_openvino_compile()

    model = _Model()
    model._ov_lm_head_on_ov = True
    gm = torch.fx.symbolic_trace(model)
    x = torch.ones(1, _BIG[0])

    assert execute.openvino_compile(gm, x) == "ov"
    assert _ptr(model.proj.weight) in plugin._baked_storages

    plugin._release_baked_weights(model)
    with pytest.raises(RuntimeError, match="OV_RELEASE_TORCH_WEIGHTS=0"):
        execute.openvino_compile(gm, x)
    assert len(compiled) == 1

    # A full reload rewrote the weights, so compiling from them is fine again.
    plugin._forget_released(model)
    assert execute.openvino_compile(gm, x) == "ov"


def _config(device):
    return SimpleNamespace(compilation_config=SimpleNamespace(
        mode=SimpleNamespace(name="STOCK_TORCH_COMPILE"), backend="openvino",
        inductor_compile_config={"device": device}))


@pytest.mark.precommit
@pytest.mark.parametrize("device, env, expected", [
    ("GPU", None, True), ("GPU.0", None, True), ("GPU", "0", False), ("CPU", None, False), ("CPU", "1", False),
])
def test_release_default_per_device(monkeypatch, device, env, expected):
    if env is None:
        monkeypatch.delenv("OV_RELEASE_TORCH_WEIGHTS", raising=False)
    else:
        monkeypatch.setenv("OV_RELEASE_TORCH_WEIGHTS", env)
    assert plugin._release_enabled(_config(device)) is expected
