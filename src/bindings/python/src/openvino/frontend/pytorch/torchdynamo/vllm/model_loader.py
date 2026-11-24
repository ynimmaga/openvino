# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""vLLM model loader for NNCF hook-storage checkpoints (`load_format="nncf_hooks"`), safetensors or `.pt2`.

Translates NNCF's layout to quantization.py's "nncf" parameters at load time; config.json still needs a
`quantization_config`, since vLLM fixes each layer's parameters before reading any weight.
"""

import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from .quantization import _unpack_uint4

if TYPE_CHECKING:
    import torch
    from vllm.config import ModelConfig

logger = logging.getLogger(__name__)

LOAD_FORMAT = "nncf_hooks"
_HOOK = "__nncf_hooks.post_hooks.{0}__0.0.{1}"
_registered: type[Any] | None = None
# Per-row int8 weights kept in stored form: lm_head's own, and the embedding
# (looked up as int8, and the source of a tied lm_head).
_LM_HEAD_SOURCES = ("lm_head.weight", "embed_tokens.weight")
# Input embeddings kept packed when stored grouped int4 (_keep_int4_embeddings).
_INT4_EMBEDDING_SOURCES = ("embed_tokens.weight", "embed_tokens_per_layer.weight")


def _hook_key(weight_name: str, field: str) -> str:
    """`model.layers.0.mlp.up_proj.weight` -> its hook-storage key."""
    return _HOOK.format(weight_name.replace(".", ":"), field)


def _dequantize(stored: "torch.Tensor", scale: "torch.Tensor", zero_point: "torch.Tensor | None",
                out_dtype: "torch.dtype") -> "torch.Tensor":
    """Dense [out, in] weight from any NNCF mode, computed in the scale's dtype as NNCF's decompressors do."""
    out_features = scale.shape[0]
    if scale.dim() == 3:
        groups = scale.shape[1]
        weight = _unpack_uint4(stored).reshape(out_features, groups, -1)
        weight = weight.type(scale.dtype)
        if zero_point is None:
            # Symmetric nibbles carry a +8 bias to stay unsigned.
            weight = weight - 8
        else:
            zp = _unpack_uint4(zero_point).reshape(out_features, groups, 1)
            weight = weight - zp.type(scale.dtype)
    else:
        weight = stored.type(scale.dtype)
        if zero_point is not None:
            weight = weight - zero_point.type(scale.dtype)
    weight = weight * scale
    return weight.reshape(out_features, -1).type(out_dtype)


def _keep_int4_embeddings():
    """Keep int4 embeddings packed: only on OV runs, where the plugin can look them up."""
    import os

    if os.environ.get("OV_INT4_EMBEDDING", "1") == "0":
        return False
    try:
        from vllm.config import get_current_vllm_config_or_none
        from openvino.frontend.pytorch.torchdynamo.vllm.plugin import _ov_requested
    except Exception:
        return False
    return _ov_requested(get_current_vllm_config_or_none())


def _attach_int4_embeddings(model, pending):
    """Hand each packed int4 table to the embedding module of its shape.

    Returns the names of the dense weights this leaves unloaded. Matched by
    shape, not name: vLLM renames checkpoint weights (Gemma-4 drops
    `language_model.`). A tied lm_head shares the embedding's Parameter, so it
    needs nothing of its own.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead, VocabParallelEmbedding)

    unloaded = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, VocabParallelEmbedding) or isinstance(module, ParallelLMHead):
            continue
        vocab, hidden = module.weight.shape
        for name, (stored, scale, zero_point) in list(pending.items()):
            if scale.shape[0] != vocab or stored.numel() * 2 != vocab * hidden:
                continue
            if getattr(module, "tp_size", 1) > 1:
                raise ValueError(f"[OV plugin] {LOAD_FORMAT}: int4 embedding {name} "
                                 "needs tensor parallel size 1; set OV_INT4_EMBEDDING=0")
            groups = scale.shape[1]
            module._ov_nncf_int4 = (
                stored.reshape(vocab, hidden // 2), scale,
                None if zero_point is None else zero_point.reshape(vocab, groups // 2))
            unloaded.add(f"{module_name}.weight")
            del pending[name]
    if pending:
        raise ValueError(f"[OV plugin] {LOAD_FORMAT}: no embedding of the shape of "
                         f"int4 {sorted(pending)}; set OV_INT4_EMBEDDING=0")
    return unloaded


def _pt2_state_dict(path: str) -> "dict[str, torch.Tensor]":
    """Persistent parameters and buffers of a `torch.export.save` archive (exported with return_dict=False)."""
    import torch

    try:
        exported = torch.export.load(path)
    except Exception as exc:
        raise ValueError(
            f"[OV plugin] {LOAD_FORMAT}: cannot read {path}: {exc}. Export "
            "with kwargs return_dict=False so the output is a plain tuple") from exc
    return exported.state_dict


def _define_loader() -> type[Any]:
    import contextlib
    import glob
    import os

    from safetensors import safe_open
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader(LOAD_FORMAT)
    class NNCFHooksModelLoader(DefaultModelLoader):
        """DefaultModelLoader with NNCF's hook layout translated in transit."""

        @staticmethod
        def _to_vllm(model: "torch.nn.Module", module: str) -> str | None:
            """HF module name -> vLLM's unstacked one (q_proj stays q_proj), or None if the model drops it."""
            mapper = getattr(model, "hf_to_vllm_mapper", None)
            if mapper is None:
                return module
            mapped = mapper.get_unstacked_mapper().apply_list([module])
            return mapped[0] if mapped else None

        def _target_is_quantized(self, model: "torch.nn.Module", params: "dict[str, torch.nn.Parameter]",
                                 module: str) -> bool:
            """Does the vLLM layer this HF module loads into hold a qweight?"""
            vllm_module = self._to_vllm(model, module)
            if vllm_module is None:
                return False
            parent, _, leaf = vllm_module.rpartition(".")
            mapping = getattr(model, "packed_modules_mapping", {})
            for fused, sources in mapping.items():
                if leaf in sources:
                    leaf = fused
                    break
            return f"{parent}.{leaf}.qweight" in params

        def _kv_shared_zeros(self, model: "torch.nn.Module", params: "dict[str, torch.nn.Parameter]",
                             seen: set[str], out_dtype: "torch.dtype") -> "Iterator[tuple[str, torch.Tensor]]":
            """Zeros for the k/v/k_norm vLLM builds on a KV-shared layer (HF has none); computed, then discarded.

            Zeros rather than empty memory, so a missed KV-sharing substitution writes zeros, not NaN, to the cache.
            """
            import torch

            modules = dict(model.named_modules())
            for name in sorted(seen):
                if not name.endswith(".q_proj.weight"):
                    continue
                base = name.removesuffix(".q_proj.weight")
                wanted = [f"{base}.{p}.weight" for p in ("k_proj", "v_proj", "k_norm")]
                missing = [w for w in wanted if w not in seen]
                if not missing:
                    continue
                vllm_base = self._to_vllm(model, f"{base}.q_proj")
                attn = modules.get(vllm_base.rpartition(".")[0]) if vllm_base else None
                if not getattr(attn, "is_kv_shared_layer", False):
                    # Not ours to fill (e.g. MLA); a real gap still fails the missing-weights check.
                    continue
                assert attn is not None
                qkv = attn.qkv_proj
                if self._target_is_quantized(model, params, f"{base}.k_proj"):
                    raise ValueError(
                        f"[OV plugin] {LOAD_FORMAT}: {base}.qkv_proj was built "
                        "int4 but its k/v are absent; drop the KV-shared "
                        "k_proj/v_proj from quantized_modules")
                # Unsharded: the qkv weight_loader takes this rank's slice.
                kv_shape = (qkv.total_num_kv_heads * qkv.head_size, qkv.input_size)
                for weight_name in missing:
                    if ".k_norm." in weight_name:
                        yield weight_name, torch.zeros_like(attn.k_norm.weight)
                    else:
                        yield weight_name, torch.zeros(kv_shape, dtype=out_dtype)

        @staticmethod
        @contextlib.contextmanager
        def _checkpoint(model_dir: str) -> "Iterator[tuple[set[str], Callable[[str], torch.Tensor]]]":
            """`(keys, get_tensor)` over all safetensors (else pt2) files; a scale may sit in another shard."""
            paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
            if paths:
                with contextlib.ExitStack() as stack:
                    owner: dict[str, Any] = {}
                    for path in paths:
                        shard = stack.enter_context(safe_open(path, framework="pt"))
                        owner.update(dict.fromkeys(shard.keys(), shard))
                    yield set(owner), lambda key: owner[key].get_tensor(key)
                return
            state: "dict[str, torch.Tensor]" = {}
            for path in sorted(glob.glob(os.path.join(model_dir, "*.pt2"))):
                state.update(_pt2_state_dict(path))
            yield set(state), state.__getitem__

        def _weights(self, model: "torch.nn.Module",
                     model_config: "ModelConfig") -> "Iterator[tuple[str, torch.Tensor]]":
            params = dict(model.named_parameters())
            out_dtype = model_config.dtype
            quant_config = getattr(model_config.hf_config, "quantization_config", None) or {}
            bits = quant_config.get("bits", 4)
            model._ov_nncf_int8_weights = {}
            model._ov_nncf_int4_pending = {}
            keep_int4 = _keep_int4_embeddings()
            with self._checkpoint(model_config.model) as (keys, get_tensor):
                for name in sorted(keys):
                    if name.startswith("__nncf_hooks."):
                        # Consumed alongside the weight it belongs to.
                        continue
                    scale_key = _hook_key(name, "_scale")
                    if scale_key not in keys:
                        # Norms and anything else NNCF left alone.
                        yield name, get_tensor(name)
                        continue

                    stored = get_tensor(name)
                    scale = get_tensor(scale_key)
                    zp_key = _hook_key(name, "_zero_point")
                    zp = get_tensor(zp_key) if zp_key in keys else None
                    module = name.removesuffix(".weight")

                    if self._target_is_quantized(model, params, module):
                        # Already row-major over (out, in) or (out, in/2), so the reshapes are free.
                        out_features = scale.shape[0]
                        yield f"{module}.qweight", stored.reshape(out_features, -1)
                        yield f"{module}.scale", scale.reshape(out_features, -1)
                        if zp is not None:
                            # int4 zero points are packed; unpacked, they concatenate like `scale`.
                            zp = _unpack_uint4(zp) if bits == 4 else zp
                            yield f"{module}.zero_point", zp.reshape(out_features, -1)
                    elif (scale.dim() == 3 and keep_int4 and name.endswith(_INT4_EMBEDDING_SOURCES)
                          and (zp is None or scale.shape[1] % 2 == 0)):
                        # Left packed, and the dense weight unloaded; see
                        # _attach_int4_embeddings.
                        model._ov_nncf_int4_pending[name] = (stored, scale, zp)
                    else:
                        if scale.dim() == 2 and name.endswith(_LM_HEAD_SOURCES):
                            # Per-row int8 that lm_head or the embedding reads:
                            # kept so both can stay compressed. Read by
                            # _install_int8_embedding, dropped by _install_ov_lm_head.
                            model._ov_nncf_int8_weights[name] = (stored, scale, zp)
                        yield name, _dequantize(stored, scale, zp, out_dtype)
                yield from self._kv_shared_zeros(model, params, keys, out_dtype)

        def load_weights(self, model: "torch.nn.Module", model_config: "ModelConfig") -> None:
            loaded = model.load_weights(self._weights(model, model_config))
            unloaded = _attach_int4_embeddings(model, model.__dict__.pop("_ov_nncf_int4_pending", {}))
            # DefaultModelLoader skips this check for quantized models; here a
            # missed key means a layout this loader does not understand.
            missing = {n for n, _ in model.named_parameters()} - set(loaded or ()) - unloaded
            if missing:
                raise ValueError(
                    f"[OV plugin] {LOAD_FORMAT}: not initialized from "
                    f"checkpoint: {sorted(missing)}")

    return NNCFHooksModelLoader


def register() -> type[Any] | None:
    """Add `load_format="nncf_hooks"` to vLLM, once (vLLM warns on a second registration)."""
    global _registered
    if _registered is None:
        try:
            _registered = _define_loader()
        except Exception as _e:
            logger.debug("[OV plugin] model loader not registered: %s", _e)
            return None
        logger.debug("[OV plugin] registered load format '%s'", LOAD_FORMAT)
    return _registered
