# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""vLLM general plugin for the OpenVINO torch.compile backend.

Registered via the `vllm.general_plugins` entry point. Patches
`CPUModelRunner.load_model` to wire `torch.compile(backend="openvino")`, plus
`reload_weights`/`CPUWorker.finish_weight_update` to recompile the OV model
and lm_head after a post-load weight mutation. No-op unless the user requested
the OV backend.
"""

import contextlib
import itertools
import logging
import os

logger = logging.getLogger(__name__)

# One id per compiled model, threaded into execute.py's compile caches so a
# second model can never reuse the first's compiled graph (and its weights)
# via an id(gm)/id(self.model)-style key that Python can recycle after GC.
_next_model_id = itertools.count()


def _ov_active(self) -> bool:
    """True if `self.vllm_config.compilation_config` requests the OV backend.

    Shared by the model-runner and worker patches below -- both expose
    `self.vllm_config`.
    """
    return _ov_requested(getattr(self, "vllm_config", None))


def _ov_requested(vllm_config) -> bool:
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


def _find_lm_head(model):
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
    rows = torch.linspace(0, weight.shape[0] - 1, 16).long()
    for stored, scale, zero_point in stash.values():
        if tuple(stored.shape) != tuple(weight.shape):
            continue
        dense = _dequantize(stored[rows], scale[rows],
                            None if zero_point is None else zero_point[rows], weight.dtype)
        if torch.equal(dense, weight[rows]):
            return stored, scale, zero_point
    return None


def _install_ov_lm_head(model) -> None:
    """(Re)compile lm_head on OV and swap it onto `lm_head.cpu_linear`.

    Rerun on every weight mutation (reload_weights/update_weights below) since
    build_ov_lm_head bakes the weight into an OV constant. On CPU defaults to
    the oneDNN dispatch: the OV path roughly doubles main-graph per-step time
    (see lm_head.py); OV_LM_HEAD=1 opts in. On GPU defaults to OV on the same
    device, int8-compressed when the checkpoint stores it so; OV_LM_HEAD=0
    opts out.
    """
    try:
        import vllm._custom_ops as _ops
        from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
        from openvino.frontend.pytorch.torchdynamo.vllm.lm_head import build_ov_lm_head
        lm_head = _find_lm_head(model)
        if lm_head is not None and hasattr(lm_head, "weight") and not lm_head.weight.is_meta:
            fn = None
            device = str((getattr(model, "_ov_compile_options", None) or {}).get("device", "CPU"))
            if os.environ.get("OV_LM_HEAD", "0" if device.startswith("CPU") else "1") != "0":
                try:
                    fn = build_ov_lm_head(
                        lm_head.weight.data, device=device,
                        int8=_int8_lm_head_source(model, lm_head.weight.data))
                except Exception as _e:
                    logger.warning("[OV plugin] lm_head OV compile failed: %s", _e)
            # Unread on the oneDNN path; drop the ~V*H bytes it holds.
            model._ov_nncf_int8_weights = {}
            if fn is not None:
                lm_head.cpu_linear = fn
            else:
                _saved = getattr(_ops, "_supports_onednn", True)
                _ops._supports_onednn = True
                try:
                    # remove_weight=False: create_onednn_mm consumes the
                    # weight, and the OV path may still need it.
                    dispatch_cpu_unquantized_gemm(lm_head, remove_weight=False)
                    logger.info("[OV plugin] lm_head re-dispatched with onednn enabled")
                except Exception as _e:
                    logger.warning("[OV plugin] lm_head onednn re-dispatch failed: %s", _e)
                finally:
                    _ops._supports_onednn = _saved
    except Exception as _e:
        logger.debug("[OV plugin] lm_head fast path unavailable: %s", _e)


_KV_UPDATE_BACKENDS: dict = {}


def _kv_update_owned_backend(base):
    """Memoized `base` subclass reporting forward_includes_kv_cache_update=True.

    Memoized because the runner dedupes attention groups by backend class
    (`full_cls_name`, gpu_model_runner.py:6997) and also collects the classes
    themselves into a set: handing out a freshly built class per layer would
    split the one CPU_ATTN group into N.
    """
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


def _unsupported_attention_features(module) -> list:
    """Attention settings the OV PagedAttention rewrite does not carry over, for one layer.

    The translator (frontends/pytorch/src/op/paged_attention.cpp) passes empty
    ALiBi slopes and sinks, has no soft-cap input, and always builds causal
    decoder attention, so a layer using any of these would compute different
    attention without an error.
    """
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


def _own_kv_cache_update(model) -> int:
    """Point this model's attention layers at a kv-update-owning backend.

    OV's PagedAttention does the paged write itself, so vLLM's separate
    unified_kv_cache_update op is redundant and, having
    no OV translator, would be stranded in eager PyTorch by the partitioner.
    Scoped per layer, not on CPUAttentionBackend, which is process-global.

    Refuses, before touching any layer, a model with attention settings the
    rewrite can't express: once OV owns the KV-cache write there is no falling
    back to vLLM's attention for those layers.
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


# Past this many guard specializations, dynamo silently drops to eager --
# and _own_kv_cache_update above already disabled vLLM's fallback KV-cache
# write, so eager fallback corrupts decode. Raise the limit to make that
# far less likely to trigger.
_DYNAMO_CACHE_SIZE_LIMIT_DEFAULT = 64


def _raise_dynamo_cache_limit():
    """Raise torch._dynamo's cache_size_limit, unless already customized.

    OV_DYNAMO_CACHE_SIZE_LIMIT overrides the floor below.
    OV_DYNAMO_FAIL_ON_CACHE_LIMIT=1 hard-errors on exhaustion instead of
    falling back silently -- but kills the whole engine, not just the
    offending request, so it's opt-in, not default.
    """
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


def _compile_ov_forward(model, options) -> None:
    import torch
    compiled = torch.compile(
        model._ov_eager_forward,
        backend="openvino",
        fullgraph=False,
        dynamic=None,
        options=options,
    )
    model.forward = _mark_token_dim_dynamic(compiled, model._ov_eager_forward)


def _rebuild_ov_model(runner, model) -> None:
    """Recompile the OV forward and lm_head after the weights were updated in place.

    AOT freezing baked every weight into the OV model as a constant, and the
    caches behind it (compiled models, unpacked MoE experts) are keyed by
    storage address, which an in-place update keeps. Without a rebuild the
    model would keep serving the old weights.
    """
    recompiled = getattr(model, "_ov_eager_forward", None) is not None
    if recompiled:
        import torch
        from openvino.frontend.pytorch.torchdynamo import execute
        execute.clear_caches()
        # Dynamo's cached frames still hold the old OV graphs. This resets every
        # compiled function in the process; the sampler recompiles on first use.
        torch._dynamo.reset()
        options = dict(model._ov_compile_options, model_id=next(_next_model_id))
        _compile_ov_forward(model, options)
    _install_ov_lm_head(model)
    if recompiled and hasattr(runner, "warming_up_model"):
        # As at load: the warmup compiles the dynamic serving model, so it is
        # built here rather than on the first request after the rebuild.
        runner.warming_up_model()


def _patch_cpu_model_runner():
    try:
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    except Exception:
        return

    if getattr(CPUModelRunner, "_ov_plugin_patched", False):
        return

    _orig_load_model = CPUModelRunner.load_model

    def patched_load_model(self, load_dummy_weights: bool = False) -> None:
        # Flip _supports_onednn before load so vLLM's FC layers see it built.
        # else restores it for a later non-OV model in the same process.
        is_ov = _ov_active(self)
        if is_ov:
            try:
                import vllm._custom_ops as _ops
                if not getattr(_ops, "_ov_plugin_onednn_disabled", False):
                    _ops._ov_plugin_onednn_saved = getattr(_ops, "_supports_onednn", True)
                    _ops._supports_onednn = False
                    _ops._ov_plugin_onednn_disabled = True
                    logger.debug("[OV plugin] _supports_onednn forced False (backend=openvino)")
            except Exception as _e:
                logger.debug("[OV plugin] _supports_onednn flip skipped: %s", _e)
        else:
            try:
                import vllm._custom_ops as _ops
                if getattr(_ops, "_ov_plugin_onednn_disabled", False):
                    _ops._supports_onednn = getattr(_ops, "_ov_plugin_onednn_saved", True)
                    _ops._ov_plugin_onednn_disabled = False
                    logger.debug("[OV plugin] _supports_onednn restored (backend != openvino)")
            except Exception as _e:
                logger.debug("[OV plugin] _supports_onednn restore skipped: %s", _e)

        if is_ov:
            _check_layername()
            _warn_if_unpinned()
            # vLLM picks each Linear's CPU kernel while loading weights, so the
            # SGL kernel is turned off for exactly this model's load.
            with _sgl_kernel_disabled():
                _orig_load_model(self, load_dummy_weights)
        else:
            _orig_load_model(self, load_dummy_weights)
        if not is_ov:
            return

        # "vllm": True turns on every vLLM-required flag (see preset.py).
        # Precision keys are deliberately absent: derived per-model dtype.
        options = {"aot_autograd": True, "vllm": True, "model_id": next(_next_model_id)}
        # vLLM built the model with torch's default dtype set to model_config.dtype
        # (model_loader/base_loader.py); the frontend's decoder gets the same value.
        _model_dtype = getattr(getattr(self, "model_config", None), "dtype", None)
        if _model_dtype is not None:
            options["default_float_type"] = _model_dtype
        options.update(_ov_user_options(self.vllm_config))

        # Hand the KV-cache write to OV's PagedAttention for this model's
        # layers only; see _own_kv_cache_update. Only when OV runs the attention:
        # the partitioner then refuses any call site it can't rewrite.
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

        # Gated here rather than in register() so eager/inductor workers do not
        # get monkey-patched with the OV sampler path.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import sampler as _vs
            _vs.install()
        except Exception as _e:
            logger.debug("[OV plugin] sampler install skipped: %s", _e)

        import torch
        try:
            import openvino.torch  # noqa: F401  (registers backend)
        except Exception as e:
            logger.warning("OV plugin: failed to import openvino.torch: %s", e)
            return

        # Undo vLLM's 1-core pin before torch.compile so OV's pool inherits
        # the wide mask at creation.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import compile_hooks as _vh_aff
            _vh_aff.widen_affinity_if_needed(None)
        except Exception as _e:
            logger.debug("[OV plugin] affinity widen skipped: %s", _e)

        _raise_dynamo_cache_limit()

        logger.info("[OV plugin] Compiling model with torch.compile backend=openvino")
        # Stops dynamo specializing num_tokens==1, so decode reuses
        # prefill's compiled model instead of compiling its own copy.
        import torch.fx.experimental._config as _dynamo_cfg
        _dynamo_cfg.backed_size_oblivious = True
        # dynamic=None with only the token dim marked dynamic (see
        # _mark_token_dim_dynamic) -- cheaper than dynamic=True (~1.7x
        # steady-state cost), which symbolizes every dim.
        self.model._ov_eager_forward = self.model.forward
        self.model._ov_compile_options = options
        _compile_ov_forward(self.model, options)

        # lm_head runs outside the compiled forward(); disabling onednn above
        # left it on plain F.linear, so give it back a fast GEMM (see lm_head.py).
        _install_ov_lm_head(self.model)

    CPUModelRunner.load_model = patched_load_model
    CPUModelRunner._ov_plugin_patched = True
    logger.debug("[OV plugin] CPUModelRunner.load_model patched")


def _patch_reload_weights():
    """Rebuild the OV model and lm_head after CPUModelRunner.reload_weights.

    reload_weights copies the new weights into the existing Parameters, but the
    OV model and lm_head hold them as constants frozen at load time.
    """
    try:
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    except Exception:
        return

    if getattr(CPUModelRunner, "_ov_plugin_reload_patched", False):
        return

    _orig_reload_weights = CPUModelRunner.reload_weights

    def patched_reload_weights(self, *args, **kwargs):
        _orig_reload_weights(self, *args, **kwargs)
        if _ov_active(self):
            _rebuild_ov_model(self, self.model)

    CPUModelRunner.reload_weights = patched_reload_weights
    CPUModelRunner._ov_plugin_reload_patched = True
    logger.debug("[OV plugin] CPUModelRunner.reload_weights patched")


def _patch_worker_update_weights():
    """Rebuild the OV model and lm_head after an RLHF weight sync on CPUWorker.

    Lives on the worker, not the model runner, so it needs its own patch. The
    rebuild hooks finish_weight_update, which ends the session, rather than
    update_weights, which runs once per chunk.
    """
    try:
        from vllm.v1.worker.cpu_worker import CPUWorker
    except Exception:
        return

    if getattr(CPUWorker, "_ov_plugin_update_patched", False):
        return

    _orig_finish_weight_update = CPUWorker.finish_weight_update

    def patched_finish_weight_update(self, *args, **kwargs):
        _orig_finish_weight_update(self, *args, **kwargs)
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None and _ov_active(model_runner):
            _rebuild_ov_model(model_runner, model_runner.get_model())

    CPUWorker.finish_weight_update = patched_finish_weight_update
    CPUWorker._ov_plugin_update_patched = True
    logger.debug("[OV plugin] CPUWorker.finish_weight_update patched")


def _check_layername():
    """Warn if vLLM latched VLLM_USE_LAYERNAME=1 before this plugin loaded.

    Too late to fix here -- vllm.utils.torch_utils freezes it at import,
    before any entry point loads -- so just warn instead of an obscure
    AttributeError later in openvino_compile.
    """
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


def _apply_default_env():
    """Fill in unset env vars only; any user-set value takes priority."""
    defaults = {
        # execute.py fast path caching set_tensor bindings and output views
        # across steps. +5-15% greedy, no correctness regression. Read only
        # by the OV backend, so it's inert for other vLLM runs.
        "OV_FAST_INFER": "1",
    }
    for key, value in defaults.items():
        if os.environ.get(key) is None:
            os.environ[key] = value
            logger.debug("[OV plugin] %s=%s (default)", key, value)


def _warn_if_unpinned():
    """Warn when the process is not pinned to a subset of CPUs.

    Full-machine affinity lets threads land on multiple NUMA nodes, which
    halves decode memory bandwidth. Best-effort: silently skipped where
    sched_getaffinity is unavailable (macOS, Windows).
    """
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


# Only swap allocations at least this large: targets the KV pool while
# leaving small bookkeeping tensors (which may rely on zeros) alone.
_LAZY_KV_MIN_BYTES = 64 << 20


@contextlib.contextmanager
def _sgl_kernel_disabled():
    """Make vLLM CPU linear dispatch skip torch.ops._C.weight_packed_linear, for one model load.

    That op is an auto_functionalized_v2 HOP the OV Partitioner treats as
    unsupported, so it fragments the graph into one partition per layer
    (e.g. 71 partitions / 77 alive CompiledModels on Gemma-4-E2B). Forcing
    the oneDNN / aten.linear fallback keeps the whole model in a single
    OV partition. Restored afterwards, so other models in the process keep
    vLLM's choice. On by default; opt out with OV_DISABLE_SGL_KERNEL=0.
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


# KV-cache pool size (GiB) used when the OV backend is requested and the user
# set no size. vLLM's own default fills node memory, which OOM-kills on shared
# NUMA nodes. 4 covers 1-2B models at 2k ctx.
_OV_DEFAULT_KVCACHE_GIB = 4


# The GPU PagedAttention kernel's block size (cldnn::paged_attention::block_size).
_OV_GPU_BLOCK_SIZE = 16


def _check_device_kv_config(vllm_config):
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


def _patch_kv_cache_default():
    """Default the KV-cache size for OV runs only.

    Hooks CpuPlatform.check_and_update_config, which is where vLLM turns
    VLLM_CPU_KVCACHE_SPACE into cache_config.kv_cache_memory_bytes and the
    first point where the compilation config is known. Setting the same field
    keeps the behaviour of the env var (including its skip of the startup
    free-memory check) without changing non-OV runs.
    """
    try:
        from vllm.platforms.cpu import CpuPlatform
    except Exception as _e:
        logger.debug("[OV plugin] KV-cache default skipped: %s", _e)
        return
    if getattr(CpuPlatform, "_ov_plugin_kv_default_patched", False):
        return
    _orig = CpuPlatform.check_and_update_config.__func__

    def _patched(cls, vllm_config):
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


def _patch_omp_wait_policy():
    """Spawn OV workers with OMP_WAIT_POLICY=PASSIVE unless the user set it.

    vLLM pins one libgomp thread per core, and the CPU plugin pins its TBB
    workers to the same cores during infer. By default each libgomp worker
    spins for about 1 ms after every torch parallel region, which steals those
    cores from the next OV infer (TinyLlama decode 21.1 -> 18.3 ms/token,
    Qwen3-8B 102 -> 98 ms/token). Inductor runs only OMP and needs the spin, so
    this is limited to OV configs and to the worker spawn:
    OMPProcessManager.configure_omp_envs scopes vLLM's own OMP variables the
    same way, so the parent process and later non-OV engines are unaffected.
    libgomp reads the variable at load, hence the spawn and not load_model.
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

    def _patched_alloc(self, *args, **kwargs):
        if not _on_ov_gpu(getattr(self, "vllm_config", None)):
            return _orig_alloc(self, *args, **kwargs)
        real_zeros = torch.zeros

        def _meta_zeros(*a, **kw):
            size = a[0] if a else kw.get("size")
            if isinstance(size, int) and kw.get("dtype") == torch.int8:
                return torch.empty(size, dtype=torch.int8, device="meta")
            return real_zeros(*a, **kw)

        torch.zeros = _meta_zeros
        try:
            return _orig_alloc(self, *args, **kwargs)
        finally:
            torch.zeros = real_zeros

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


def _patch_lazy_kv_cache():
    """Allocate the KV pool lazily (OV_LAZY_KVCACHE=1).

    vLLM sizes the pool from VLLM_CPU_KVCACHE_SPACE and zero-fills it, which
    faults every page in at startup even though PagedAttention writes each
    block before reading it. torch.empty lets RSS track blocks actually used.
    """
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

    def _patched_alloc(self, *args, **kwargs):
        if not _ov_active(self):
            return _orig_alloc(self, *args, **kwargs)
        # Swap only for the duration of this call, so unrelated torch.zeros
        # users are untouched and upstream's own logic stays authoritative.
        real_zeros = torch.zeros

        def _maybe_lazy_zeros(*a, **kw):
            size = a[0] if a else kw.get("size")
            try:
                nbytes = size if isinstance(size, int) else None
            except Exception:
                nbytes = None
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


def _register_quantization():
    """Teach vLLM the weight-only integer methods in quantization.py.

    Unconditional, unlike the patches below: the registry is read when
    ModelConfig is built, long before load_model can tell whether the OV
    backend was requested, and registering a name changes nothing for a model
    that does not ask for it.
    """
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm import quantization as _q
        _q.register()
    except Exception as _e:
        logger.debug("[OV plugin] quantization registration skipped: %s", _e)
    # The loader that reads NNCF's hook-storage layout into those methods.
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm import model_loader as _ml
        _ml.register()
    except Exception as _e:
        logger.debug("[OV plugin] model loader registration skipped: %s", _e)


def register():
    """Entry point for `vllm.general_plugins`.

    vLLM loads plugins in every process, before the config exists, so nothing
    here may change behaviour for non-OV runs: OV-active detection is deferred
    to the patched config hook and load_model, which can read the real
    compilation_config.backend instead of guessing from env vars.
    """
    _apply_default_env()
    _register_quantization()
    _patch_kv_cache_default()
    _patch_omp_wait_policy()
    _patch_cpu_model_runner()
    _patch_reload_weights()
    _patch_worker_update_weights()
    _patch_device_kv_cache()
    _patch_lazy_kv_cache()
