# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify the OV forward is compiled without dynamo guards on the model itself.

_compile_ov_forward compiles the model's bound forward, so `self` is a frame
input and dynamo would guard every weight and attribute it traced, re-checking
them all before each step. _drop_self_guards keeps only the guards on the
inputs and globals. This checks which guards are dropped, that a changed model
attribute no longer recompiles while a changed input rank still does, and the
OV_DYNAMO_FULL_GUARDS opt-out.

No model or GPU needed: a minimal stand-in model is compiled on OV CPU.
"""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("VLLM_USE_LAYERNAME", "0")

import torch

from openvino.frontend.pytorch.torchdynamo.vllm import plugin


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.scale = 2.0
        self._ov_eager_forward = self._forward

    def _forward(self, input_ids):
        return self.proj(input_ids) * self.scale


def _entry(name):
    return SimpleNamespace(name=name)


@pytest.mark.parametrize("name, kept", [
    ("self._modules['model']._modules['layers']", False),
    ("dict(type(self).__dict__)", False),
    ("self", False),
    ("input_ids", True),
    ("positions", True),
    ("G['torch']", True),
    ("type(G['torch'].nn)", True),
    ("selfish", True),
])
def test_drop_self_guards(name, kept):
    assert plugin._drop_self_guards([_entry(name)]) == [kept]


def _frames_compiled(model, calls):
    """Dynamo frames compiled while `calls(model)` runs on a freshly compiled forward."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    plugin._compile_ov_forward(model, {"device": "CPU"})
    calls(model)
    return torch._dynamo.utils.counters["frames"]["ok"]


def _change_attribute(model):
    x = torch.randn(4, 8)
    model.forward(x)
    model.scale = 3.0
    model.forward(x)


def _change_input_rank(model):
    model.forward(torch.randn(4, 8))
    model.forward(torch.randn(2, 4, 8))


@pytest.mark.parametrize("full_guards, expected", [("0", 1), ("1", 2)])
def test_model_attribute_change(monkeypatch, full_guards, expected):
    monkeypatch.setenv("OV_DYNAMO_FULL_GUARDS", full_guards)
    assert _frames_compiled(_Model(), _change_attribute) == expected


def test_input_rank_change_recompiles(monkeypatch):
    monkeypatch.setenv("OV_DYNAMO_FULL_GUARDS", "0")
    assert _frames_compiled(_Model(), _change_input_rank) == 2


def test_compile_options_left_unchanged(monkeypatch):
    monkeypatch.setenv("OV_DYNAMO_FULL_GUARDS", "0")
    options = {"device": "CPU"}
    torch._dynamo.reset()
    plugin._compile_ov_forward(_Model(), options)
    # _rebuild_ov_model reuses the stored options; the filter must not leak in.
    assert options == {"device": "CPU"}
