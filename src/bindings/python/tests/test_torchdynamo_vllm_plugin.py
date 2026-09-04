# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path


def _load_plugin_module():
    plugin_path = (
        Path(__file__).resolve().parents[1]
        / "src/openvino/frontend/pytorch/torchdynamo/vllm/plugin.py"
    )
    spec = importlib.util.spec_from_file_location("test_vllm_plugin", plugin_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_module(monkeypatch, name, module):
    parts = name.split(".")
    for idx in range(1, len(parts)):
        package_name = ".".join(parts[:idx])
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, package_name, package)
        if idx > 1:
            parent_name = ".".join(parts[: idx - 1])
            setattr(sys.modules[parent_name], parts[idx - 1], sys.modules[package_name])
    monkeypatch.setitem(sys.modules, name, module)
    if len(parts) > 1:
        setattr(sys.modules[".".join(parts[:-1])], parts[-1], module)


def _make_cpu_model_runner(backend):
    class CPUModelRunner:
        def __init__(self):
            self.original_forward = lambda *args, **kwargs: (args, kwargs)
            self.model = types.SimpleNamespace(forward=self.original_forward, lm_head=None)
            mode = types.SimpleNamespace(name="STOCK_TORCH_COMPILE")
            self.vllm_config = types.SimpleNamespace(
                compilation_config=types.SimpleNamespace(mode=mode, backend=backend)
            )
            self.load_model_called = False

        def load_model(self, load_dummy_weights=False):
            self.load_model_called = True

    return CPUModelRunner


def test_vllm_plugin_passes_dynamic_none_only_on_openvino_path(monkeypatch):
    plugin = _load_plugin_module()
    cpu_model_runner_module = types.ModuleType("vllm.v1.worker.cpu_model_runner")
    cpu_model_runner_module.CPUModelRunner = _make_cpu_model_runner("openvino")
    _install_module(monkeypatch, "vllm.v1.worker.cpu_model_runner", cpu_model_runner_module)

    sampler_installed = {"called": False}
    sampler_module = types.ModuleType(
        "openvino.frontend.pytorch.torchdynamo.vllm.sampler"
    )
    sampler_module.install = lambda: sampler_installed.__setitem__("called", True)
    _install_module(
        monkeypatch,
        "openvino.frontend.pytorch.torchdynamo.vllm.sampler",
        sampler_module,
    )

    affinity_calls = {"called": False}
    hooks_module = types.ModuleType(
        "openvino.frontend.pytorch.torchdynamo.vllm.compile_hooks"
    )
    hooks_module.widen_affinity_if_needed = (
        lambda _arg: affinity_calls.__setitem__("called", True)
    )
    _install_module(
        monkeypatch,
        "openvino.frontend.pytorch.torchdynamo.vllm.compile_hooks",
        hooks_module,
    )

    custom_ops_module = types.ModuleType("vllm._custom_ops")
    custom_ops_module._supports_onednn = True
    _install_module(monkeypatch, "vllm._custom_ops", custom_ops_module)

    dispatch_module = types.ModuleType("vllm.model_executor.layers.utils")
    dispatch_module.dispatch_cpu_unquantized_gemm = lambda *_args, **_kwargs: None
    _install_module(monkeypatch, "vllm.model_executor.layers.utils", dispatch_module)

    _install_module(monkeypatch, "openvino.torch", types.ModuleType("openvino.torch"))

    compiled_forward = lambda *args, **kwargs: None
    compile_call = {}

    def fake_compile(forward, **kwargs):
        compile_call["forward"] = forward
        compile_call["kwargs"] = kwargs
        return compiled_forward

    torch_module = types.ModuleType("torch")
    torch_module.compile = fake_compile
    _install_module(monkeypatch, "torch", torch_module)

    plugin._patch_cpu_model_runner()
    runner = cpu_model_runner_module.CPUModelRunner()
    runner.load_model()

    assert runner.load_model_called is True
    assert sampler_installed["called"] is True
    assert affinity_calls["called"] is True
    assert runner.model.forward is compiled_forward
    assert compile_call["forward"] is runner.original_forward
    assert compile_call["kwargs"] == {
        "backend": "openvino",
        "fullgraph": False,
        "dynamic": None,
        "options": {"aot_autograd": True, "vllm": True},
    }


def test_vllm_plugin_leaves_non_openvino_paths_unchanged(monkeypatch):
    plugin = _load_plugin_module()
    cpu_model_runner_module = types.ModuleType("vllm.v1.worker.cpu_model_runner")
    cpu_model_runner_module.CPUModelRunner = _make_cpu_model_runner("inductor")
    _install_module(monkeypatch, "vllm.v1.worker.cpu_model_runner", cpu_model_runner_module)

    compile_calls = {"count": 0}

    def fake_compile(*args, **kwargs):
        compile_calls["count"] += 1
        return None

    torch_module = types.ModuleType("torch")
    torch_module.compile = fake_compile
    _install_module(monkeypatch, "torch", torch_module)

    plugin._patch_cpu_model_runner()
    runner = cpu_model_runner_module.CPUModelRunner()
    original_forward = runner.model.forward
    runner.load_model()

    assert runner.load_model_called is True
    assert compile_calls["count"] == 0
    assert runner.model.forward is original_forward
