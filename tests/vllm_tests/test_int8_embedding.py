# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify the input embedding looks up the checkpoint's int8 table when it can.

The nncf_hooks loader stashes a per-row int8 embedding on
`model._ov_nncf_int8_weights`; _install_compressed_embeddings points the embedding at
it only if it still decompresses to the live weight, and on GPU by default.
This checks that the lookup is bit-identical to the dense one, the stale
fallback, the per-device default, and that the dense table it supersedes is
released while the lookup keeps working.

No model download needed: a minimal stand-in model carries the weights.
"""

import os

import pytest

os.environ.setdefault("VLLM_USE_LAYERNAME", "0")

import torch

from openvino.frontend.pytorch.torchdynamo.vllm import plugin
from openvino.frontend.pytorch.torchdynamo.vllm.model_loader import _dequantize
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod, VocabParallelEmbedding)

# Above _KEY_CONTENT_NUMEL, so the dense table is large enough to release.
_VOCAB, _HIDDEN = 4096, 64


class _Embedding(VocabParallelEmbedding):
    """VocabParallelEmbedding without the tensor-parallel setup its __init__ needs."""

    def __init__(self, weight):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.tp_size = 1
        self.quant_method = UnquantizedEmbeddingMethod()


class _Model(torch.nn.Module):
    def __init__(self, weight, stash, device="GPU"):
        super().__init__()
        self.embed_tokens = _Embedding(weight)
        self._ov_nncf_int8_weights = stash
        self._ov_compile_options = {"device": device}


def _int8_table(asym=True):
    torch.manual_seed(0)
    stored = torch.randint(0, 256, (_VOCAB, _HIDDEN), dtype=torch.uint8)
    scale = (torch.rand(_VOCAB, 1) * 0.01 + 1e-3).to(torch.float16)
    zero_point = torch.randint(0, 256, (_VOCAB, 1), dtype=torch.uint8) if asym else None
    return stored, scale, zero_point


def _model(dtype=torch.float16, asym=True, device="GPU", stale=False):
    stored, scale, zero_point = _int8_table(asym)
    dense = _dequantize(stored, scale, zero_point, dtype)
    if stale:
        dense = dense + 1
    stash = {"model.embed_tokens.weight": (stored, scale, zero_point)}
    return _Model(dense.clone(), stash, device), dense


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.delenv("OV_INT8_EMBEDDING", raising=False)
    monkeypatch.setattr(plugin, "_baked_storages", set())
    monkeypatch.setattr(plugin, "_released_storages", set())
    monkeypatch.setattr(plugin, "_superseded_storages", set())


def _ids():
    return torch.randint(0, _VOCAB, (7,))


@pytest.mark.precommit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("asym", [True, False])
def test_int8_lookup_is_bit_identical(dtype, asym):
    model, dense = _model(dtype, asym)
    plugin._install_compressed_embeddings(model)

    assert isinstance(model.embed_tokens.quant_method, plugin._CompressedEmbeddingMethod)
    ids = _ids()
    out = model.embed_tokens(ids)
    assert out.dtype == dtype
    assert torch.equal(out, torch.nn.functional.embedding(ids, dense))


@pytest.mark.precommit
def test_stale_table_keeps_dense_lookup():
    model, _ = _model(stale=True)
    plugin._install_compressed_embeddings(model)

    assert isinstance(model.embed_tokens.quant_method, UnquantizedEmbeddingMethod)
    assert model.embed_tokens._ov_q_weight is None
    assert not plugin._superseded_storages


@pytest.mark.precommit
def test_reinstall_after_stale_update_falls_back():
    model, _ = _model()
    plugin._install_compressed_embeddings(model)
    # A weight update changed the live table; the stash no longer describes it.
    model.embed_tokens.weight.data += 1
    plugin._install_compressed_embeddings(model)

    assert isinstance(model.embed_tokens.quant_method, UnquantizedEmbeddingMethod)
    assert not plugin._superseded_storages


@pytest.mark.precommit
@pytest.mark.parametrize("device, env, expected", [
    ("GPU", None, True), ("GPU", "0", False), ("CPU", None, False), ("CPU", "1", True),
])
def test_default_per_device(monkeypatch, device, env, expected):
    if env is not None:
        monkeypatch.setenv("OV_INT8_EMBEDDING", env)
    model, _ = _model(device=device)
    plugin._install_compressed_embeddings(model)
    assert isinstance(model.embed_tokens.quant_method, plugin._CompressedEmbeddingMethod) is expected


@pytest.mark.precommit
def test_dense_table_released_lookup_unaffected():
    model, dense = _model()
    plugin._install_compressed_embeddings(model)
    weight = model.embed_tokens.weight

    plugin._release_baked_weights(model)

    assert weight.untyped_storage().data_ptr() in plugin._released_storages
    # Released pages read as zeros, but the lookup never touches them.
    page_bytes = 2 * os.sysconf("SC_PAGE_SIZE")
    assert int(weight.count_nonzero()) * weight.element_size() <= page_bytes
    ids = _ids()
    assert torch.equal(model.embed_tokens(ids), torch.nn.functional.embedding(ids, dense))
