# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""vLLM general plugin (`vllm.general_plugins`) wiring torch.compile(backend="openvino") into the CPU runner."""

import contextlib
import itertools
import logging
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
import re
if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# One id per compiled model for execute.py's caches; id()-based keys can be recycled after GC.
_next_model_id: Iterator[int] = itertools.count()


def _ov_active(self: Any) -> bool:
    """True if `self.vllm_config.compilation_config` requests the OV backend."""
    return _ov_requested(getattr(self, "vllm_config", None))


def _ov_requested(vllm_config: Any) -> bool:
    """True if `vllm_config.compilation_config` requests the OV backend."""
    comp_cfg = getattr(vllm_config, "compilation_config", None)
    if comp_cfg is None:
        return False
    try:
        mode = getattr(comp_cfg, "mode", None)
        mode_name = getattr(mode, "name", None) if mode is not None else None
        backend = getattr(comp_cfg, "backend", None)
    except Exception:
        return False
    return mode_name == "STOCK_TORCH_COMPILE" and backend == "openvino"


# OV backend options a user may set through compilation_config. They ride in
# inductor_compile_config: CompilationConfig rejects unknown fields, and it is
# the only free-form dict there. vLLM itself only forwards it to torch.compile
# for the inductor backend, so the plugin reads it here.
_USER_OPTION_KEYS = ("device", "config")


def _ov_user_options(vllm_config) -> dict:
    """The `_USER_OPTION_KEYS` entries of `compilation_config.inductor_compile_config`."""
    comp_cfg = getattr(vllm_config, "compilation_config", None)
    user = getattr(comp_cfg, "inductor_compile_config", None) or {}
    return {key: user[key] for key in _USER_OPTION_KEYS if key in user}


def _find_lm_head(model: "torch.nn.Module") -> "torch.nn.Module | None":
    """The model's lm_head; multimodal wrappers (e.g. Gemma 4) keep it on their language model."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "get_language_model"):
        lm_head = getattr(model.get_language_model(), "lm_head", None)
    return lm_head


def _int8_lm_head_source(model, weight):
    """The loader's stored int8 tensors that `weight` was decompressed from, or None.

    Matched on content, not name: a tied lm_head has no checkpoint entry of its
    own, and after a weight update the stored tensors no longer describe the
    live weight. Pops the stash either way; nothing else reads it.
    """
    import torch
    from openvino.frontend.pytorch.torchdynamo.vllm.model_loader import _dequantize

    stash = getattr(model, "_ov_nncf_int8_weights", None) or {}
    model._ov_nncf_int8_weights = {}
    return _match_int8(stash, weight)


def _match_int8(stash, weight):
    """The stashed (stored, scale, zero_point) that decompresses to `weight`, or None."""
    import torch
    from openvino.frontend.pytorch.torchdynamo.vllm.model_loader import _dequantize

    rows = torch.linspace(0, weight.shape[0] - 1, 16).long()
    for stored, scale, zero_point in stash.values():
        if tuple(stored.shape) != tuple(weight.shape):
            continue
        dense = _dequantize(stored[rows], scale[rows],
                            None if zero_point is None else zero_point[rows], weight.dtype)
        if torch.equal(dense, weight[rows]):
            return stored, scale, zero_point
    return None


class _CompressedEmbeddingMethod:
    """Embedding lookup from the checkpoint's int8 (per-row) or int4 (grouped) table.

    Gathers the stored rows, then dequantizes just those rows with
    _dequantize's arithmetic (in the scale's dtype, cast after), so the output
    is bit-identical to a dense lookup of the decompressed table. OV keeps the
    gathered table as a u8 constant: half the dense f16 memory for int8, a
    quarter for packed int4. Eager callers (a VLM's embed_input_ids) only ever
    touch the rows they look up.
    """

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def embedding(self, layer, input_):
        import torch
        import torch.nn.functional as F
        scale = F.embedding(input_, layer._ov_q_scale)
        rows = F.embedding(input_, layer._ov_q_weight)
        zero_point = layer._ov_q_zero_point
        if layer._ov_q_bits == 4:
            # Packed nibbles, low first, grouped: scale [V, groups], as in _dequantize.
            groups = scale.shape[-1]
            rows = torch.stack((rows & 15, rows >> 4), dim=-1).reshape(*input_.shape, groups, -1)
            rows = rows.type(scale.dtype)
            if zero_point is None:
                # Symmetric nibbles carry a +8 bias to stay unsigned.
                rows = rows - 8
            else:
                zp = F.embedding(input_, zero_point)
                zp = torch.stack((zp & 15, zp >> 4), dim=-1).reshape(*input_.shape, groups, 1)
                rows = rows - zp.type(scale.dtype)
            rows = (rows * scale.unsqueeze(-1)).reshape(*input_.shape, -1)
        else:
            rows = rows.type(scale.dtype)
            if zero_point is not None:
                rows = rows - F.embedding(input_, zero_point).type(scale.dtype)
            rows = rows * scale
        # Not layer.weight.dtype: reading the dense weight would pull it into the graph.
        return rows.type(layer._ov_q_out_dtype)


def _set_embedding_table(module, base, bits, stored=None, scale=None, zero_point=None):
    """Point `module` at a compressed table, or back at its dense weight if `bits` is None."""
    module.register_buffer("_ov_q_weight", stored, persistent=False)
    module.register_buffer("_ov_q_scale", scale, persistent=False)
    module.register_buffer("_ov_q_zero_point", zero_point, persistent=False)
    module._ov_q_bits = bits
    module._ov_q_out_dtype = module.weight.dtype
    module.quant_method = base if bits is None else _CompressedEmbeddingMethod(base)


def _install_compressed_embeddings(model) -> None:
    """Point each input embedding at its compressed checkpoint table.

    Runs before _install_ov_lm_head, which consumes the int8 stash.
    - int4: the loader left the table packed on `_ov_nncf_int4` and the dense
      weight unloaded, so it is always used, on every device. The unloaded
      weight is marked released: the compile guard then refuses any graph
      that would read it.
    - int8: used while the stashed table still matches the live weight, so a
      weight update brings the dense lookup back. On GPU by default
      (OV_INT8_EMBEDDING=0 opts out), CPU opts in with 1. The dense weight it
      replaces is marked superseded, for _release_baked_weights to hand back.
    """
    try:
        import torch
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead, VocabParallelEmbedding)
    except Exception as _e:
        logger.debug("[OV plugin] compressed embedding skipped: %s", _e)
        return
    if not isinstance(model, torch.nn.Module):
        return
    device = str((getattr(model, "_ov_compile_options", None) or {}).get("device", "CPU"))
    int8_enabled = os.environ.get("OV_INT8_EMBEDDING", "0" if device.startswith("CPU") else "1") != "0"
    stash = getattr(model, "_ov_nncf_int8_weights", None) or {}
    for module in model.modules():
        if not isinstance(module, VocabParallelEmbedding) or isinstance(module, ParallelLMHead) \
                or getattr(module, "tp_size", 1) > 1:
            continue
        base = getattr(module.quant_method, "base", module.quant_method)
        dense_ptr = module.weight.untyped_storage().data_ptr()
        int4 = getattr(module, "_ov_nncf_int4", None)
        if int4 is not None:
            stored, scale, zero_point = int4
            _set_embedding_table(module, base, 4, stored, scale.reshape(scale.shape[0], -1), zero_point)
            _released_storages.add(dense_ptr)
            logger.info("[OV plugin] embedding %s looks up its int4 checkpoint table",
                        tuple(module.weight.shape))
            continue
        int8 = _match_int8(stash, module.weight) if int8_enabled and not module.weight.is_meta else None
        if int8 is None:
            _set_embedding_table(module, base, None)
            _superseded_storages.discard(dense_ptr)
            continue
        _set_embedding_table(module, base, 8, *int8)
        _superseded_storages.add(dense_ptr)
        logger.info("[OV plugin] embedding %s looks up its int8 checkpoint table",
                    tuple(module.weight.shape))


def _int4_embedding_of(model, weight):
    """The packed int4 table of the embedding whose (tied) weight is `weight`, or None."""
    if not hasattr(model, "modules"):
        return None
    for module in model.modules():
        if getattr(module, "weight", None) is weight and getattr(module, "_ov_nncf_int4", None) is not None:
            return module._ov_nncf_int4
    return None

def _install_ov_lm_head(model: "torch.nn.Module") -> None:
    """(Re)build `lm_head.cpu_linear` from the current weight: oneDNN by default, OV with OV_LM_HEAD=1."""
    int4 = None
    fn = None
    try:
        import vllm._custom_ops as _ops
        from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
        from openvino.frontend.pytorch.torchdynamo.vllm.lm_head import build_ov_lm_head
        lm_head = _find_lm_head(model)
        if lm_head is not None and hasattr(lm_head, "weight") and not lm_head.weight.is_meta:
            device = str((getattr(model, "_ov_compile_options", None) or {}).get("device", "CPU"))
            int4 = _int4_embedding_of(model, lm_head.weight)
            on_ov = os.environ.get("OV_LM_HEAD", "0" if device.startswith("CPU") else "1") != "0"
            if int4 is not None and not on_ov:
                logger.warning("[OV plugin] lm_head is tied to an int4 embedding with no dense "
                               "weight, so it runs on OV despite OV_LM_HEAD=0")
            if on_ov or int4 is not None:
                try:
                    fn = build_ov_lm_head(
                        lm_head.weight.data, device=device, int4=int4,
                        int8=None if int4 is not None else _int8_lm_head_source(model, lm_head.weight.data))
                except Exception as _e:
                    logger.warning("[OV plugin] lm_head OV compile failed: %s", _e)
            # Unread on the oneDNN path; drop the ~V*H bytes it holds.
            model._ov_nncf_int8_weights = {}
            # Off OV, lm_head still reads its torch weight (_release_baked_weights).
            model._ov_lm_head_on_ov = fn is not None
            if fn is not None:
                lm_head.cpu_linear = fn
            elif int4 is not None:
                pass  # raised below, outside this best-effort block
            else:
                _saved = getattr(_ops, "_supports_onednn", True)
                _ops._supports_onednn = True
                try:
                    # Keep the weight: a later rebuild needs it.
                    dispatch_cpu_unquantized_gemm(lm_head, remove_weight=False)
                    logger.info("[OV plugin] lm_head re-dispatched with onednn enabled")
                except Exception as _e:
                    logger.warning("[OV plugin] lm_head onednn re-dispatch failed: %s", _e)
                finally:
                    _ops._supports_onednn = _saved
    except Exception as _e:
        logger.debug("[OV plugin] lm_head fast path unavailable: %s", _e)
    if int4 is not None and fn is None:
        raise RuntimeError(
            "OpenVINO vLLM backend: lm_head is tied to an int4 embedding, which has no dense "
            "weight, and could not be compiled int4 on OV; set OV_INT4_EMBEDDING=0 to load it dense")


_KV_UPDATE_BACKENDS: dict[type, type] = {}


def _kv_update_owned_backend(base: type) -> type:
    """`base` subclass with forward_includes_kv_cache_update=True, one per base (vLLM groups layers by class)."""
    cached = _KV_UPDATE_BACKENDS.get(base)
    if cached is None:
        from vllm.v1.attention.backend import subclass_attention_backend_with_overrides
        cached = subclass_attention_backend_with_overrides(
            name_prefix="OVKVUpdate",
            attention_backend_cls=base,
            overrides={"forward_includes_kv_cache_update": True},
        )
        _KV_UPDATE_BACKENDS[base] = cached
    return cached


def _unsupported_attention_features(module: "torch.nn.Module") -> list[str]:
    """Settings of one attention layer the OV PagedAttention translator would silently drop."""
    from vllm.v1.attention.backend import AttentionType
    impl = getattr(module, "impl", None)
    features = []
    if getattr(impl, "alibi_slopes", None) is not None:
        features.append("ALiBi slopes")
    if getattr(impl, "sinks", None) is not None:
        features.append("attention sinks")
    if getattr(impl, "logits_soft_cap", None):
        features.append("logits soft-capping")
    attn_type = getattr(impl, "attn_type", AttentionType.DECODER)
    if attn_type != AttentionType.DECODER:
        features.append(f"attention type {attn_type}")
    return features


def _own_kv_cache_update(model: "torch.nn.Module") -> int:
    """Let OV's PagedAttention own this model's KV-cache write, per layer (CPUAttentionBackend is process-global).

    Refuses a model with unsupported attention first: there is no falling back to vLLM's attention afterwards.
    """
    from vllm.v1.attention.backend import AttentionBackend
    layers = []
    for name, module in model.named_modules():
        # Some layers hold an AttentionBackendEnum here, not a backend class.
        backend = getattr(module, "attn_backend", None)
        if (isinstance(backend, type) and issubclass(backend, AttentionBackend)
                and not backend.forward_includes_kv_cache_update):
            layers.append((name, module, backend))
    unsupported = {name: features for name, module, _ in layers
                   if (features := _unsupported_attention_features(module))}
    if unsupported:
        first = next(iter(unsupported))
        raise NotImplementedError(
            f"The OpenVINO backend does not support {', '.join(unsupported[first])} in attention "
            f"(layer {first!r}, {len(unsupported)} layer(s) affected). Run this model without "
            f"the OpenVINO compilation backend.")
    for _, module, backend in layers:
        module.attn_backend = _kv_update_owned_backend(backend)
    return len(layers)


# Past the limit dynamo falls back to eager, which corrupts decode once OV owns the KV-cache write.
_DYNAMO_CACHE_SIZE_LIMIT_DEFAULT = 64


def _raise_dynamo_cache_limit() -> None:
    """Raise dynamo's cache_size_limit unless customized; OV_DYNAMO_FAIL_ON_CACHE_LIMIT=1 errors on exhaustion."""
    try:
        import torch._dynamo.config as _dynamo_cfg
    except Exception as _e:
        logger.debug("[OV plugin] dynamo cache_size_limit raise skipped: %s", _e)
        return
    limit = int(os.environ.get(
        "OV_DYNAMO_CACHE_SIZE_LIMIT", _DYNAMO_CACHE_SIZE_LIMIT_DEFAULT))
    if _dynamo_cfg.cache_size_limit == 8 and limit != 8:
        _dynamo_cfg.cache_size_limit = limit
        logger.debug("[OV plugin] dynamo cache_size_limit raised 8 -> %d", limit)
    if os.environ.get("OV_DYNAMO_FAIL_ON_CACHE_LIMIT", "0") != "0":
        _dynamo_cfg.fail_on_recompile_limit_hit = True
        logger.debug("[OV plugin] dynamo fail_on_recompile_limit_hit enabled")


# Forward inputs sized by the step's token count, and the index of that dim.
# positions is [3, num_tokens] under M-RoPE, hence -1.
_TOKEN_DIM_INPUTS = {"input_ids": 0, "positions": -1, "inputs_embeds": 0}


def _mark_token_dim_dynamic(compiled_forward, eager_forward):
    """Wrap `compiled_forward` so dynamo's first trace has a symbolic token count.

    vLLM's profile run makes the first call. Unmarked, dynamo specializes that
    trace to its exact size, so OV compiles a static model only the profile run
    uses, then a dynamic one on the first real request. Marked, the profile-run
    compile is the dynamic serving model, and the RSS vLLM sizes the KV cache
    from right after it already includes that model.
    """
    import inspect
    import torch
    param_names = list(inspect.signature(eager_forward).parameters)

    def forward(*args, **kwargs):
        for name, dim in _TOKEN_DIM_INPUTS.items():
            tensor = kwargs.get(name)
            if tensor is None and name in param_names and param_names.index(name) < len(args):
                tensor = args[param_names.index(name)]
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                torch._dynamo.maybe_mark_dynamic(tensor, dim % tensor.dim())
        return compiled_forward(*args, **kwargs)

    return forward


_SELF_ROOTED_GUARD = re.compile(r"(\w+\()*self\b")


def _drop_self_guards(guard_entries):
    """Dynamo `guard_filter_fn`: keep the guards on the forward's inputs only.

    The compiled frame is the model's bound forward, so `self` is a frame input
    and dynamo guards every weight and module attribute tracing touched
    (~20k checks on Gemma 4, re-run before each decode step). The weights are
    baked into the OV model as constants, so those guards can't change which
    graph runs. vLLM's own compile path skips all guards; this keeps the input
    guards so a change of input rank or dtype still recompiles.
    """
    # Names are source expressions rooted at `self`, possibly wrapped, e.g.
    # "self._modules['model']..." or "dict(type(self)...)".
    keep = [not _SELF_ROOTED_GUARD.match(entry.name) for entry in guard_entries]
    logger.debug("[OV plugin] dynamo guards: kept %d of %d", sum(keep), len(keep))
    return keep


def _compile_ov_forward(model: "torch.nn.Module", options: dict[str, Any]) -> None:
    import torch
    if os.environ.get("OV_DYNAMO_FULL_GUARDS", "0") == "0":
        options = dict(options, guard_filter_fn=_drop_self_guards)
    compiled = torch.compile(
        model._ov_eager_forward,
        backend="openvino",
        fullgraph=False,
        dynamic=None,
        options=options,
    )
    model.forward = _mark_token_dim_dynamic(compiled, model._ov_eager_forward)


def _rebuild_ov_model(runner: Any, model: "torch.nn.Module") -> None:
    """Recompile the OV forward and lm_head after an in-place weight update, which frozen constants don't see."""
    recompiled = getattr(model, "_ov_eager_forward", None) is not None
    if recompiled:
        import torch
        from openvino.frontend.pytorch.torchdynamo import execute
        execute.clear_caches()
        # Dynamo's cached frames hold the old OV graphs; the sampler recompiles on first use.
        torch._dynamo.reset()
        options = dict(model._ov_compile_options, model_id=next(_next_model_id))
        _compile_ov_forward(model, options)
    _install_compressed_embeddings(model)
    _install_ov_lm_head(model)
    if recompiled and hasattr(runner, "warming_up_model"):
        # As at load: the warmup compiles the dynamic serving model, so it is
        # built here rather than on the first request after the rebuild.
        runner.warming_up_model()


# Storage addresses of host tensors an OV compile consumed, and of those
# _release_baked_weights returned to the OS.
_baked_storages: set = set()
_released_storages: set = set()
# Dense weights nothing reads any more (an embedding now looking up int8 rows).
_superseded_storages: set = set()

_MADV_DONTNEED = 4


def _host_storages(gm, args) -> set:
    """Storage addresses of the host tensors an FX graph holds or is called with."""
    import torch
    tensors = list(gm.parameters()) + list(gm.buffers())
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            obj = gm
            for part in str(node.target).split("."):
                obj = getattr(obj, part, None)
            tensors.append(obj)
    tensors += list(args)
    return {t.untyped_storage().data_ptr() for t in tensors
            if isinstance(t, torch.Tensor) and t.device.type == "cpu" and not t.is_meta}


def _patch_openvino_compile():
    """Track the weights each OV compile bakes in, and refuse released ones.

    After _release_baked_weights the torch weights read as zeros, so a compile
    that reads them would bake zeros into the OV model and serve wrong tokens
    without any error. Dynamo re-traces that hit the structural cache never
    get here, so they are unaffected.
    """
    from openvino.frontend.pytorch.torchdynamo import execute
    if getattr(execute, "_ov_plugin_compile_patched", False):
        return
    _orig_compile = execute.openvino_compile

    def _patched_compile(gm, *args, **kwargs):
        storages = _host_storages(gm, args)
        if storages & _released_storages:
            raise RuntimeError(
                "OpenVINO vLLM backend: a new OV compile needs torch weights that were released "
                "after the GPU compile; set OV_RELEASE_TORCH_WEIGHTS=0 to keep them")
        compiled = _orig_compile(gm, *args, **kwargs)
        _baked_storages.update(storages)
        return compiled

    execute.openvino_compile = _patched_compile
    execute._ov_plugin_compile_patched = True


def _release_baked_weights(model) -> None:
    """Drop the host copy of every torch weight an OV GPU compile baked in.

    On GPU the OV model holds its own copy in device memory, so the torch
    copy is dead weight (~1 GiB for a 1B int4 model). Superseded weights (the
    dense table of an embedding that now looks up int8 rows) go too, already
    at load. madvise(MADV_DONTNEED)
    hands the pages back while every tensor keeps its object, shape and
    address: dynamo's guards still hold, and the structural cache, which keys
    large constants by address, still matches the re-trace of the first real
    request. Constants it keys by content (up to _KEY_CONTENT_NUMEL elements)
    stay, since zeroing them would change that key. Weights torch still reads
    (lm_head off OV, eager parts such as vision towers) were never baked in,
    so they stay too.
    """
    import ctypes
    import torch
    from openvino.frontend.pytorch.torchdynamo.execute import _KEY_CONTENT_NUMEL
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    page = os.sysconf("SC_PAGE_SIZE")
    keep = set()
    lm_head = _find_lm_head(model)
    if lm_head is not None and not getattr(model, "_ov_lm_head_on_ov", False) \
            and isinstance(getattr(lm_head, "weight", None), torch.Tensor):
        keep.add(lm_head.weight.untyped_storage().data_ptr())
    freed = 0
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if tensor.device.type != "cpu" or tensor.is_meta:
            continue
        storage = tensor.untyped_storage()
        ptr = storage.data_ptr()
        if (ptr not in _baked_storages and ptr not in _superseded_storages) \
                or ptr in keep or ptr in _released_storages \
                or tensor.numel() <= _KEY_CONTENT_NUMEL:
            continue
        # Whole pages only: the first and last may be shared with other allocations.
        start = -(-ptr // page) * page
        end = (ptr + storage.nbytes()) // page * page
        if end > start and libc.madvise(start, end - start, _MADV_DONTNEED) == 0:
            freed += end - start
            _released_storages.add(ptr)
    logger.info("[OV plugin] released %.2f GiB of torch weights the OV GPU model no longer reads", freed / 2**30)


def _forget_released(model) -> None:
    """A full weight reload rewrote every released tensor, so OV may read them again."""
    if not _released_storages:
        return
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        _released_storages.discard(tensor.untyped_storage().data_ptr())


def _release_enabled(vllm_config) -> bool:
    """On by default on OV GPU only: on CPU the OV model may share the torch weights' memory."""
    return _on_ov_gpu(vllm_config) and os.environ.get("OV_RELEASE_TORCH_WEIGHTS", "1") != "0"


def _patch_release_weights():
    """Release the baked torch weights once vLLM's warmup has compiled the serving model."""
    try:
        from vllm.v1.worker.cpu_worker import CPUWorker
    except Exception as _e:
        logger.debug("[OV plugin] weight release patch skipped: %s", _e)
        return
    if getattr(CPUWorker, "_ov_plugin_release_patched", False):
        return
    _orig_warm_up = CPUWorker.compile_or_warm_up_model

    def _patched_warm_up(self, *args, **kwargs):
        result = _orig_warm_up(self, *args, **kwargs)
        if _release_enabled(getattr(self, "vllm_config", None)):
            _release_baked_weights(self.model_runner.get_model())
        return result

    CPUWorker.compile_or_warm_up_model = _patched_warm_up
    CPUWorker._ov_plugin_release_patched = True


def _patch_cpu_model_runner() -> None:
    try:
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    except Exception:
        return

    if getattr(CPUModelRunner, "_ov_plugin_patched", False):
        return

    _orig_load_model = CPUModelRunner.load_model

    def patched_load_model(self: Any, load_dummy_weights: bool = False) -> None:
        # vLLM's FC layers read _supports_onednn at load; restore it for a later non-OV model.
        is_ov = _ov_active(self)
        if not is_ov:
            try:
                import vllm._custom_ops as _ops
                if getattr(_ops, "_ov_plugin_onednn_disabled", False):
                    _ops._supports_onednn = getattr(_ops, "_ov_plugin_onednn_saved", True)
                    _ops._ov_plugin_onednn_disabled = False
                    logger.debug("[OV plugin] _supports_onednn restored (backend != openvino)")
            except Exception as _e:
                logger.debug("[OV plugin] _supports_onednn restore skipped: %s", _e)
            _orig_load_model(self, load_dummy_weights)
            return

        try:
            import vllm._custom_ops as _ops
            if not getattr(_ops, "_ov_plugin_onednn_disabled", False):
                _ops._ov_plugin_onednn_saved = getattr(_ops, "_supports_onednn", True)
                _ops._supports_onednn = False
                _ops._ov_plugin_onednn_disabled = True
                logger.debug("[OV plugin] _supports_onednn forced False (backend=openvino)")
        except Exception as _e:
            logger.debug("[OV plugin] _supports_onednn flip skipped: %s", _e)

        _check_layername()
        _warn_if_unpinned()
        # vLLM picks each Linear's CPU kernel while loading weights.
        with _sgl_kernel_disabled():
            _orig_load_model(self, load_dummy_weights)

        options: dict[str, Any] = {"aot_autograd": True, "vllm": True, "model_id": next(_next_model_id)}
        # The dtype vLLM built the model with, for the frontend decoder.
        _model_dtype = getattr(getattr(self, "model_config", None), "dtype", None)
        if _model_dtype is not None:
            options["default_float_type"] = _model_dtype
        options.update(_ov_user_options(self.vllm_config))

        from openvino.frontend.pytorch.torchdynamo.vllm import owns_kv_cache_update
        if owns_kv_cache_update(options):
            try:
                _n = _own_kv_cache_update(self.model)
                logger.debug(
                    "[OV plugin] kv-cache update ownership moved to OV "
                    "PagedAttention for %d attention layer(s)", _n)
            except NotImplementedError:
                raise  # attention the OV rewrite can't express: refuse the model
            except Exception as _e:
                logger.debug(
                    "[OV plugin] kv_cache_update suppression skipped: %s", _e)

        # Here, not in register(), so eager/inductor workers keep vLLM's sampler.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import sampler as _vs
            _vs.install()
        except Exception as _e:
            logger.debug("[OV plugin] sampler install skipped: %s", _e)

        try:
            import openvino.torch  # registers the backend
        except Exception as e:
            logger.warning("OV plugin: failed to import openvino.torch: %s", e)
            return

        # Undo vLLM's 1-core pin before OV's thread pool is created.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import compile_hooks as _vh_aff
            _vh_aff.widen_affinity_if_needed()
        except Exception as _e:
            logger.debug("[OV plugin] affinity widen skipped: %s", _e)

        _raise_dynamo_cache_limit()
        _patch_openvino_compile()
        _warn_if_spinning_single_process()

        logger.info("[OV plugin] Compiling model with torch.compile backend=openvino")
        # Stops dynamo specializing num_tokens==1, so decode reuses prefill's compiled model.
        import torch.fx.experimental._config as _dynamo_cfg
        _dynamo_cfg.backed_size_oblivious = True
        # dynamic=None with only the token dim marked dynamic (see
        # _mark_token_dim_dynamic) -- cheaper than dynamic=True (~1.7x
        # steady-state cost), which symbolizes every dim.
        self.model._ov_eager_forward = self.model.forward
        self.model._ov_compile_options = options
        _compile_ov_forward(self.model, options)

        _install_compressed_embeddings(self.model)
        # lm_head runs outside the compiled forward(); disabling onednn above
        # left it on plain F.linear, so give it back a fast GEMM (see lm_head.py).
        _install_ov_lm_head(self.model)
        if _release_enabled(self.vllm_config):
            # Before the compile, so the dense embedding table is not in its peak either.
            _release_baked_weights(self.model)

    CPUModelRunner.load_model = patched_load_model
    CPUModelRunner._ov_plugin_patched = True
    logger.debug("[OV plugin] CPUModelRunner.load_model patched")


def _patch_reload_weights() -> None:
    """Rebuild the OV model and lm_head after CPUModelRunner.reload_weights."""
    try:
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    except Exception:
        return

    if getattr(CPUModelRunner, "_ov_plugin_reload_patched", False):
        return

    _orig_reload_weights = CPUModelRunner.reload_weights

    def patched_reload_weights(self: Any, *args: Any, **kwargs: Any) -> None:
        _orig_reload_weights(self, *args, **kwargs)
        if _ov_active(self):
            _forget_released(self.model)
            _rebuild_ov_model(self, self.model)
            if _release_enabled(getattr(self, "vllm_config", None)):
                _release_baked_weights(self.model)

    CPUModelRunner.reload_weights = patched_reload_weights
    CPUModelRunner._ov_plugin_reload_patched = True
    logger.debug("[OV plugin] CPUModelRunner.reload_weights patched")


def _patch_worker_update_weights() -> None:
    """Rebuild the OV model and lm_head at CPUWorker.finish_weight_update, once per RLHF sync, not per chunk."""
    try:
        from vllm.v1.worker.cpu_worker import CPUWorker
    except Exception:
        return

    if getattr(CPUWorker, "_ov_plugin_update_patched", False):
        return

    _orig_finish_weight_update = CPUWorker.finish_weight_update

    def patched_finish_weight_update(self: Any, *args: Any, **kwargs: Any) -> None:
        _orig_finish_weight_update(self, *args, **kwargs)
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None and _ov_active(model_runner):
            model = model_runner.get_model()
            if any(
                tensor.untyped_storage().data_ptr() in _released_storages
                for tensor in itertools.chain(model.parameters(), model.buffers())
                if tensor.device.type == "cpu" and not tensor.is_meta
            ):
                # A sync may cover only some weights, so the rest would rebuild from garbage.
                raise RuntimeError(
                    "OpenVINO vLLM backend: weight updates need the torch weights, which were "
                    "released after the GPU compile (OV_RELEASE_TORCH_WEIGHTS=0 keeps them) or, "
                    "for an int4 embedding, never loaded (OV_INT4_EMBEDDING=0 loads them)")
            _rebuild_ov_model(model_runner, model_runner.get_model())

    CPUWorker.finish_weight_update = patched_finish_weight_update
    CPUWorker._ov_plugin_update_patched = True
    logger.debug("[OV plugin] CPUWorker.finish_weight_update patched")


def _check_layername() -> None:
    """Warn if vLLM latched VLLM_USE_LAYERNAME=1 at import; too late to fix here."""
    try:
        from vllm.utils.torch_utils import _USE_LAYERNAME
    except Exception:
        return
    if not _USE_LAYERNAME:
        return
    logger.warning(
        "[OV plugin] VLLM_USE_LAYERNAME is enabled; the OV backend cannot "
        "compile graphs whose layer_name is a hoisted LayerName input and "
        "will fail with \"'LayerName' object has no attribute 'type'\". Call "
        "openvino.frontend.pytorch.torchdynamo.vllm.preset.set_pre_import_env() "
        "before importing vllm. Setting the env var now has no effect: vLLM "
        "reads it at import, before plugins load."
    )


def _apply_default_env() -> None:
    """Fill in unset env vars only; any user-set value takes priority."""
    defaults = {
        # Caches input bindings and output views across steps; read only by the OV backend.
        "OV_FAST_INFER": "1",
    }
    for key, value in defaults.items():
        if os.environ.get(key) is None:
            os.environ[key] = value
            logger.debug("[OV plugin] %s=%s (default)", key, value)


def _warn_if_unpinned() -> None:
    """Warn when the process is not pinned: threads on several NUMA nodes cut decode memory bandwidth."""
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return
    ncpus = os.cpu_count() or 0
    if ncpus and len(allowed) >= ncpus:
        logger.warning(
            "[OV plugin] process is not CPU-pinned (%d cores visible). "
            "For best decode throughput on multi-socket systems, launch with "
            "`taskset -c 0-N` or `numactl --cpunodebind=0 --membind=0` to pin "
            "to a single NUMA node.",
            len(allowed),
        )


# Only the KV pool; small bookkeeping tensors may rely on zeros.
_LAZY_KV_MIN_BYTES = 64 << 20


@contextlib.contextmanager
def _sgl_kernel_disabled() -> Iterator[None]:
    """Keep vLLM's CPU linears off the SGL kernel for one model load (OV_DISABLE_SGL_KERNEL=0 opts out).

    Its op is unsupported by the partitioner and would split the graph into one partition per layer.
    """
    if os.environ.get("OV_DISABLE_SGL_KERNEL", "1") == "0":
        logger.info("[OV plugin] SGL-kernel disable OFF (OV_DISABLE_SGL_KERNEL=0)")
        yield
        return
    try:
        from vllm.model_executor.layers import utils as _vllm_utils
        saved = _vllm_utils.check_cpu_sgl_kernel
    except Exception as _e:
        logger.debug("[OV plugin] disable-SGL patch skipped: %s", _e)
        yield
        return
    _vllm_utils.check_cpu_sgl_kernel = lambda *a, **kw: False
    logger.debug("[OV plugin] check_cpu_sgl_kernel forced to False for this model load")
    try:
        yield
    finally:
        _vllm_utils.check_cpu_sgl_kernel = saved


# KV-cache GiB for OV runs with no size set; vLLM's default fills node memory and OOMs on shared nodes.
_OV_DEFAULT_KVCACHE_GIB = 4

# Cap (GiB) on the OV GPU KV cache when the user set no size. OV allocates the
# whole pool up front, and gpu_memory_utilization alone would reserve half of an
# iGPU's (host) memory. 1 holds ~16 full 2k-token sequences of a 1B model.
_OV_DEFAULT_GPU_KVCACHE_GIB = 1


# The GPU PagedAttention kernel's block size (cldnn::paged_attention::block_size).
_OV_GPU_BLOCK_SIZE = 16


def _check_device_kv_config(vllm_config) -> None:
    """Fit vLLM's config to an OV-owned, device-resident KV cache, or refuse it.

    Sets the GPU kernel's block size. The KV size is left unset: it comes from
    device memory (see _patch_device_kv_cache), not the CPU default below.
    Features that read or write KV contents from the host can't reach it.
    """
    cache_config = vllm_config.cache_config
    if cache_config.block_size != _OV_GPU_BLOCK_SIZE:
        logger.info("[OV plugin] block_size %s -> %d for the OpenVINO GPU PagedAttention kernel",
                    cache_config.block_size, _OV_GPU_BLOCK_SIZE)
        cache_config.block_size = _OV_GPU_BLOCK_SIZE
    unsupported = []
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        unsupported.append("kv_transfer_config (KV connectors)")
    if getattr(cache_config, "kv_offloading_size", None):
        unsupported.append("kv_offloading_size")
    if getattr(vllm_config.model_config, "enable_sleep_mode", False):
        unsupported.append("enable_sleep_mode")
    if unsupported:
        raise ValueError(
            "OpenVINO vLLM backend: the GPU KV cache lives in device memory owned by OpenVINO, "
            "so these features, which access it from the host, are unsupported: " + ", ".join(unsupported))


def _patch_kv_cache_default() -> None:
    """Default kv_cache_memory_bytes for OV runs, where vLLM applies VLLM_CPU_KVCACHE_SPACE."""
    try:
        from vllm.platforms.cpu import CpuPlatform
    except Exception as _e:
        logger.debug("[OV plugin] KV-cache default skipped: %s", _e)
        return
    if getattr(CpuPlatform, "_ov_plugin_kv_default_patched", False):
        return
    _orig = CpuPlatform.check_and_update_config.__func__

    def _patched(cls: Any, vllm_config: Any) -> None:
        _orig(cls, vllm_config)
        cache_config = getattr(vllm_config, "cache_config", None)
        if cache_config is not None and _on_ov_gpu(vllm_config):
            _check_device_kv_config(vllm_config)
            return
        if (cache_config is None or cache_config.kv_cache_memory_bytes is not None
                or not _ov_requested(vllm_config)):
            return
        cache_config.kv_cache_memory_bytes = _OV_DEFAULT_KVCACHE_GIB << 30
        logger.info(
            "[OV plugin] KV cache size defaults to %d GiB for the OpenVINO backend; "
            "set VLLM_CPU_KVCACHE_SPACE or kv_cache_memory_bytes to change it",
            _OV_DEFAULT_KVCACHE_GIB)

    CpuPlatform.check_and_update_config = classmethod(_patched)
    CpuPlatform._ov_plugin_kv_default_patched = True

def _warn_if_spinning_single_process():
    """Warn when an in-process engine runs without OMP_WAIT_POLICY.

    With VLLM_ENABLE_V1_MULTIPROCESSING=0 no worker is spawned, so the patch
    above never applies, and libgomp has already read its environment by the
    time vLLM loads this plugin. Only the launching script can set it, before
    its first torch/openvino import (Llama-3.2-1B on an iGPU: 60 vs 79 tok/s).
    """
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") == "0" and "OMP_WAIT_POLICY" not in os.environ:
        logger.warning(
            "[OV plugin] VLLM_ENABLE_V1_MULTIPROCESSING=0 without OMP_WAIT_POLICY: torch's OpenMP "
            "threads will spin on the cores OV needs; set OMP_WAIT_POLICY=PASSIVE before importing torch")


def _ov_device(vllm_config) -> str:
    """The OV device requested through compilation_config (see _ov_user_options)."""
    return str(_ov_user_options(vllm_config).get("device", "CPU"))


def _on_ov_gpu(vllm_config) -> bool:
    return _ov_requested(vllm_config) and _ov_device(vllm_config).startswith("GPU")


def _serving_compiled_model():
    """The compiled OV model with PagedAttention inputs, or None.

    The warmup compiles it (the dynamic serving model) before vLLM sizes and
    allocates the KV cache.
    """
    from openvino.frontend.pytorch.torchdynamo import execute
    from openvino.frontend.pytorch.torchdynamo.vllm.runtime_hooks import has_pa_inputs
    candidates = [c for c, _req in execute.structural_cache.values()] + list(execute.compiled_cache.values())
    return next((c for c in candidates if has_pa_inputs(c)), None)


def _ov_kv_block_bytes(compiled) -> int:
    """Bytes one KV block takes across all layers, in the layout the device plugin chose.

    Read off the compiled key/value cache ports: the plugin picks the precision
    (i8 caches carry their scales inside the block) and the per-block shape.
    """
    import numpy as np
    from openvino.frontend.pytorch.torchdynamo.vllm.compile_hooks import _is_kv_cache_port
    total = 0
    for port in compiled.inputs:
        if _is_kv_cache_port(port):
            block_dims = [d.get_length() for d in list(port.get_partial_shape())[1:]]
            total += int(np.prod(block_dims)) * port.get_element_type().size
    return total


def _patch_device_kv_cache() -> None:
    """On an OV GPU device, let OV own the KV cache in device memory.

    vLLM still manages the blocks (tables, prefix caching, scheduling), but
    OV allocates their storage on the device (side_channel._device_kv_tensors).
    So vLLM's own KV buffers become shape-only placeholders on the meta device,
    and the KV size comes from device memory instead of host RSS.
    """
    try:
        import torch
        from vllm.v1.worker.cpu_worker import CPUWorker
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as _e:
        logger.debug("[OV plugin] device KV patch skipped: %s", _e)
        return

    if getattr(GPUModelRunner, "_ov_plugin_device_kv_patched", False):
        return

    _orig_alloc = GPUModelRunner._allocate_kv_cache_tensors

    def _patched_alloc(self, kv_cache_config, *args, **kwargs):
        if not _on_ov_gpu(getattr(self, "vllm_config", None)):
            return _orig_alloc(self, kv_cache_config, *args, **kwargs)
        real_zeros = torch.zeros
        # Gated DeltaNet / Mamba state stays in vLLM's host buffers (the side
        # channel gathers and scatters it around each infer), so any buffer a
        # state layer shares must be real; only attention-only buffers, which OV
        # holds on the device, become placeholders. One decision per
        # torch.zeros call, in _orig_alloc's order: packed tensors share one.
        from vllm.v1.kv_cache_interface import MambaSpec
        state_layers = {name for group in kv_cache_config.kv_cache_groups
                        if isinstance(group.kv_cache_spec, MambaSpec) for name in group.layer_names}
        tensors = kv_cache_config.kv_cache_tensors
        packed_real = any(not state_layers.isdisjoint(t.shared_by) for t in tensors if t.block_stride > 0)
        real = [not state_layers.isdisjoint(t.shared_by) for t in tensors if t.block_stride <= 0]
        if any(t.block_stride > 0 for t in tensors):
            first_packed = next(i for i, t in enumerate(tensors) if t.block_stride > 0)
            real.insert(sum(1 for t in tensors[:first_packed] if t.block_stride <= 0), packed_real)
        real = iter(real)
        host_bytes = 0

        def _meta_zeros(*a, **kw):
            nonlocal host_bytes
            size = a[0] if a else kw.get("size")
            if isinstance(size, int) and kw.get("dtype") == torch.int8:
                if next(real, False):
                    host_bytes += size
                    return real_zeros(*a, **kw)
                return torch.empty(size, dtype=torch.int8, device="meta")
            return real_zeros(*a, **kw)

        torch.zeros = _meta_zeros
        try:
            return _orig_alloc(self, kv_cache_config, *args, **kwargs)
        finally:
            torch.zeros = real_zeros
            if host_bytes:
                logger.info("[OV plugin] %.2f GiB of KV buffers kept on host for %d state layers",
                            host_bytes / 2**30, len(state_layers))

    _orig_available = CPUWorker.determine_available_memory

    def _patched_available(self):
        vllm_config = getattr(self, "vllm_config", None)
        if not _on_ov_gpu(vllm_config):
            return _orig_available(self)
        self.model_runner.warming_up_model()
        compiled = _serving_compiled_model()
        if compiled is None:
            raise RuntimeError("OpenVINO vLLM backend: the warmup compiled no PagedAttention model")
        ov_block_bytes = _ov_kv_block_bytes(compiled)

        from openvino.frontend.pytorch.torchdynamo.compile import get_core
        core = get_core()
        device = _ov_device(vllm_config)
        explicit = vllm_config.cache_config.kv_cache_memory_bytes
        if explicit is not None:
            budget = explicit
        else:
            # Device memory not yet used by the compiled model (weights, buffers).
            total = core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE")
            used = sum(core.get_property(device, "GPU_MEMORY_STATISTICS").values())
            budget = int(total * vllm_config.cache_config.gpu_memory_utilization) - used
            if str(core.get_property(device, "DEVICE_TYPE")).endswith("INTEGRATED"):
                # An iGPU's device memory is host memory: leave the host its share.
                import psutil
                budget = min(budget, int(psutil.virtual_memory().available
                                         * vllm_config.cache_config.gpu_memory_utilization))
            budget = min(budget, _OV_DEFAULT_GPU_KVCACHE_GIB << 30)
        num_blocks = max(budget, 0) // ov_block_bytes

        # vLLM computes num_blocks = available // page_size // num_layers from its
        # own (model-dtype) page size; report what yields exactly num_blocks.
        vllm_block_bytes = sum(spec.page_size_bytes for spec in self.model_runner.get_kv_cache_spec().values())
        logger.info("[OV plugin] KV cache on %s: %d blocks, %.2f GiB of device memory",
                    device, num_blocks, num_blocks * ov_block_bytes / 2**30)
        return num_blocks * vllm_block_bytes

    GPUModelRunner._allocate_kv_cache_tensors = _patched_alloc
    CPUWorker.determine_available_memory = _patched_available
    GPUModelRunner._ov_plugin_device_kv_patched = True


def _patch_omp_wait_policy() -> None:

    """Spawn OV workers with OMP_WAIT_POLICY=PASSIVE unless the user set it.
    Spinning libgomp threads steal the cores OV's TBB workers are pinned to. Set at spawn, since libgomp reads it at
    load, and scoped like vLLM's own OMP variables, so other engines keep the spin.
    """
    try:
        from vllm.utils.ompmultiprocessing import OMPProcessManager
    except Exception as _e:
        logger.debug("[OV plugin] OMP wait-policy patch skipped: %s", _e)
        return
    if getattr(OMPProcessManager, "_ov_plugin_omp_patched", False):
        return
    _orig_init = OMPProcessManager.__init__
    _orig_configure = OMPProcessManager.configure_omp_envs

    def _patched_init(self, config):
        _orig_init(self, config)
        self._ov_requested = _ov_requested(config)

    @contextlib.contextmanager
    def _patched_configure(self, rank, local_rank):
        with _orig_configure(self, rank, local_rank):
            if not getattr(self, "_ov_requested", False) or "OMP_WAIT_POLICY" in os.environ:
                yield
                return
            os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
            try:
                yield
            finally:
                os.environ.pop("OMP_WAIT_POLICY", None)

    OMPProcessManager.__init__ = _patched_init
    OMPProcessManager.configure_omp_envs = _patched_configure
    OMPProcessManager._ov_plugin_omp_patched = True

def _ov_device(vllm_config) -> str:
    """The OV device requested through compilation_config (see _ov_user_options)."""
    return str(_ov_user_options(vllm_config).get("device", "CPU"))


def _on_ov_gpu(vllm_config) -> bool:
    return _ov_requested(vllm_config) and _ov_device(vllm_config).startswith("GPU")


def _serving_compiled_model():
    """The compiled OV model with PagedAttention inputs, or None.

    The warmup compiles it (the dynamic serving model) before vLLM sizes and
    allocates the KV cache.
    """
    from openvino.frontend.pytorch.torchdynamo import execute
    from openvino.frontend.pytorch.torchdynamo.vllm.runtime_hooks import has_pa_inputs
    candidates = [c for c, _req in execute.structural_cache.values()] + list(execute.compiled_cache.values())
    return next((c for c in candidates if has_pa_inputs(c)), None)


def _ov_kv_block_bytes(compiled) -> int:
    """Bytes one KV block takes across all layers, in the layout the device plugin chose.

    Read off the compiled key/value cache ports: the plugin picks the precision
    (i8 caches carry their scales inside the block) and the per-block shape.
    """
    import numpy as np
    from openvino.frontend.pytorch.torchdynamo.vllm.compile_hooks import _is_kv_cache_port
    total = 0
    for port in compiled.inputs:
        if _is_kv_cache_port(port):
            block_dims = [d.get_length() for d in list(port.get_partial_shape())[1:]]
            total += int(np.prod(block_dims)) * port.get_element_type().size
    return total


def _patch_device_kv_cache():
    """On an OV GPU device, let OV own the KV cache in device memory.

    vLLM still manages the blocks (tables, prefix caching, scheduling), but
    OV allocates their storage on the device (side_channel._device_kv_tensors).
    So vLLM's own KV buffers become shape-only placeholders on the meta device,
    and the KV size comes from device memory instead of host RSS.
    """
    try:
        import torch
        from vllm.v1.worker.cpu_worker import CPUWorker
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as _e:
        logger.debug("[OV plugin] device KV patch skipped: %s", _e)
        return

    if getattr(GPUModelRunner, "_ov_plugin_device_kv_patched", False):
        return

    _orig_alloc = GPUModelRunner._allocate_kv_cache_tensors

    def _patched_alloc(self, kv_cache_config, *args, **kwargs):
        if not _on_ov_gpu(getattr(self, "vllm_config", None)):
            return _orig_alloc(self, kv_cache_config, *args, **kwargs)
        real_zeros = torch.zeros
        # Gated DeltaNet / Mamba state stays in vLLM's host buffers (the side
        # channel gathers and scatters it around each infer), so any buffer a
        # state layer shares must be real; only attention-only buffers, which OV
        # holds on the device, become placeholders. One decision per
        # torch.zeros call, in _orig_alloc's order: packed tensors share one.
        from vllm.v1.kv_cache_interface import MambaSpec
        state_layers = {name for group in kv_cache_config.kv_cache_groups
                        if isinstance(group.kv_cache_spec, MambaSpec) for name in group.layer_names}
        tensors = kv_cache_config.kv_cache_tensors
        packed_real = any(not state_layers.isdisjoint(t.shared_by) for t in tensors if t.block_stride > 0)
        real = [not state_layers.isdisjoint(t.shared_by) for t in tensors if t.block_stride <= 0]
        if any(t.block_stride > 0 for t in tensors):
            first_packed = next(i for i, t in enumerate(tensors) if t.block_stride > 0)
            real.insert(sum(1 for t in tensors[:first_packed] if t.block_stride <= 0), packed_real)
        real = iter(real)
        host_bytes = 0

        def _meta_zeros(*a, **kw):
            nonlocal host_bytes
            size = a[0] if a else kw.get("size")
            if isinstance(size, int) and kw.get("dtype") == torch.int8:
                if next(real, False):
                    host_bytes += size
                    return real_zeros(*a, **kw)
                return torch.empty(size, dtype=torch.int8, device="meta")
            return real_zeros(*a, **kw)

        torch.zeros = _meta_zeros
        try:
            return _orig_alloc(self, kv_cache_config, *args, **kwargs)
        finally:
            torch.zeros = real_zeros
            if host_bytes:
                logger.info("[OV plugin] %.2f GiB of KV buffers kept on host for %d state layers",
                            host_bytes / 2**30, len(state_layers))

    _orig_available = CPUWorker.determine_available_memory

    def _patched_available(self):
        vllm_config = getattr(self, "vllm_config", None)
        if not _on_ov_gpu(vllm_config):
            return _orig_available(self)
        self.model_runner.warming_up_model()
        compiled = _serving_compiled_model()
        if compiled is None:
            raise RuntimeError("OpenVINO vLLM backend: the warmup compiled no PagedAttention model")
        ov_block_bytes = _ov_kv_block_bytes(compiled)

        from openvino.frontend.pytorch.torchdynamo.compile import get_core
        core = get_core()
        device = _ov_device(vllm_config)
        explicit = vllm_config.cache_config.kv_cache_memory_bytes
        if explicit is not None:
            budget = explicit
        else:
            # Device memory not yet used by the compiled model (weights, buffers).
            total = core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE")
            used = sum(core.get_property(device, "GPU_MEMORY_STATISTICS").values())
            budget = int(total * vllm_config.cache_config.gpu_memory_utilization) - used
            if str(core.get_property(device, "DEVICE_TYPE")).endswith("INTEGRATED"):
                # An iGPU's device memory is host memory: leave the host its share.
                import psutil
                budget = min(budget, int(psutil.virtual_memory().available
                                         * vllm_config.cache_config.gpu_memory_utilization))
            budget = min(budget, _OV_DEFAULT_GPU_KVCACHE_GIB << 30)
        num_blocks = max(budget, 0) // ov_block_bytes

        # vLLM computes num_blocks = available // page_size // num_layers from its
        # own (model-dtype) page size; report what yields exactly num_blocks.
        vllm_block_bytes = sum(spec.page_size_bytes for spec in self.model_runner.get_kv_cache_spec().values())
        logger.info("[OV plugin] KV cache on %s: %d blocks, %.2f GiB of device memory",
                    device, num_blocks, num_blocks * ov_block_bytes / 2**30)
        return num_blocks * vllm_block_bytes

    GPUModelRunner._allocate_kv_cache_tensors = _patched_alloc
    CPUWorker.determine_available_memory = _patched_available
    GPUModelRunner._ov_plugin_device_kv_patched = True


def _patch_lazy_kv_cache() -> None:
    """Allocate the KV pool with torch.empty (OV_LAZY_KVCACHE=1): PA writes each block before reading it."""
    if os.environ.get("OV_LAZY_KVCACHE", "0") == "0":
        return
    try:
        import torch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as _e:
        logger.debug("[OV plugin] lazy KV patch skipped: %s", _e)
        return

    if getattr(GPUModelRunner, "_ov_plugin_lazy_kv_patched", False):
        return

    _orig_alloc = GPUModelRunner._allocate_kv_cache_tensors

    def _patched_alloc(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _ov_active(self):
            return _orig_alloc(self, *args, **kwargs)
        # Only for the duration of this call.
        real_zeros = torch.zeros

        def _maybe_lazy_zeros(*a: Any, **kw: Any) -> "torch.Tensor":
            size = a[0] if a else kw.get("size")
            nbytes = size if isinstance(size, int) else None
            if nbytes is not None and nbytes >= _LAZY_KV_MIN_BYTES:
                return torch.empty(*a, **kw)
            return real_zeros(*a, **kw)

        torch.zeros = _maybe_lazy_zeros
        try:
            return _orig_alloc(self, *args, **kwargs)
        finally:
            torch.zeros = real_zeros

    GPUModelRunner._allocate_kv_cache_tensors = _patched_alloc
    GPUModelRunner._ov_plugin_lazy_kv_patched = True
    logger.info("[OV plugin] KV cache pool allocated lazily (OV_LAZY_KVCACHE=1)")


def _register_quantization() -> None:
    """Register the NNCF quantization method and loader; unconditional, as vLLM reads them before load_model."""
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm import quantization as _q
        _q.register()
    except Exception as _e:
        logger.debug("[OV plugin] quantization registration skipped: %s", _e)
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm import model_loader as _ml
        _ml.register()
    except Exception as _e:
        logger.debug("[OV plugin] model loader registration skipped: %s", _e)


def register() -> None:
    """Entry point for `vllm.general_plugins`; runs in every process, so nothing here may affect non-OV runs."""
    _apply_default_env()
    _register_quantization()
    _patch_kv_cache_default()
    _patch_omp_wait_policy()
    _patch_cpu_model_runner()
    _patch_reload_weights()
    _patch_worker_update_weights()
    _patch_release_weights()
    _patch_device_kv_cache()
    _patch_lazy_kv_cache()
